#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from hmac import new
from turtle import pos, position
from typing import List, Optional, Tuple, Union, Dict
from httpx import get
from more_itertools import last
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config
from .modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
import random
from torch.nn import functional as F

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        budget_frame_len = kwargs.pop("budget_frame_len", 32)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes, budget_frame_len=budget_frame_len)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes

        # if input_ids.shape[1] == 1:
        #     inputs["position_ids"] = inputs["position_ids"][:, -1].unsqueeze(0) + 1
        return inputs

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None, budget_frame_len=32):
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features = self.encode_images(concat_images) # [frame, patch, dim] [32, 729 = 26^2, 3584]

            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = [] # [32, 196 = 14^2, 3584]
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                else:
                    image_features.append(image_feat)

            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        if mm_newline_position == "one_token":
                            # one-token
                            resolution = image_feature.shape[1]
                            image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    else:
                        raise NotImplementedError
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_position_ids = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_position_ids = []

            last_pos = 0
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_position_ids.append(torch.arange(last_pos, last_pos + cur_input_embeds_no_im[i].shape[0], dtype=position_ids.dtype, device=position_ids.device))
                last_pos += cur_input_embeds_no_im[i].shape[0]
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_image_position_ids = torch.arange(last_pos, last_pos + cur_image_features.shape[0], dtype=position_ids.dtype, device=position_ids.device)
                    last_pos += cur_image_features.shape[0]

                    selected_indices = None
                    best_metric = -100000
                    final_key_frame_len = 0
                    # for key_frame_len in range(1, budget_frame_len + 1):
                    key_frame_len_space = []
                    for ii in range(1, budget_frame_len // 2 + 1):
                        key_frame_len_space.append(ii)
                    for key_frame_len in key_frame_len_space:
                        selected_indices_now = self._select_tokens(cur_image_features, resolution, budget_frame_len, key_frame_len)
                        metric = self._cal_metric(cur_image_features, cur_input_embeds_no_im[i], cur_input_embeds_no_im[i + 1], selected_indices_now)
                        if metric > best_metric:
                            selected_indices = selected_indices_now
                            best_metric = metric
                            final_key_frame_len = key_frame_len
                    print(f"key_frame_len: {final_key_frame_len}, metric: {best_metric}")

                    selected_cur_image_features = cur_image_features[selected_indices]
                    selected_cur_image_postion_ids = cur_image_position_ids[selected_indices]

                    cur_new_input_embeds.append(selected_cur_image_features)
                    cur_new_labels.append(torch.full((selected_cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_position_ids.append(selected_cur_image_postion_ids)

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_position_ids = torch.cat(cur_new_position_ids)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_position_ids.append(cur_new_position_ids)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels, cur_new_position_ids) in enumerate(zip(new_input_embeds, new_labels, new_position_ids)):
            cur_len = cur_new_embed.shape[0]
            new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = cur_new_position_ids
                assert position_ids[i].shape[0] == cur_len

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def _select_tokens(self, video_embeds, resolution, budget_frame_len=32, key_frame_len=16, delta_allocate="proportion"):
        video_embeds_tmp = video_embeds[:-1].view(-1, resolution, video_embeds.shape[-1]) # [T, H*W, dim]

        budget_token_len = budget_frame_len * resolution
        video_grid_t = video_embeds_tmp.shape[0]

        if video_embeds_tmp.shape[0] <= budget_frame_len:
            return list(range(video_embeds_tmp.shape[0] * resolution))
        
        selected_indices = []
        # select key frames using uniform sampling [key_frame_len + 1]
        frame_indices = torch.linspace(0, video_grid_t, key_frame_len + 1, dtype=torch.long, device=video_embeds.device)

        # select delta tokens
        differences = frame_indices[1:] - frame_indices[:-1] - 1
        if delta_allocate == "proportion":
            # 1. based on proportion of intervals
            total_diff = torch.sum(differences)
            remain_budget = budget_token_len - key_frame_len * resolution
            delta_token_len = (remain_budget / total_diff * differences).int()
            remain_budget -= torch.sum(delta_token_len)
            if remain_budget > 0:
                sorted_indices = torch.argsort(-differences)
                for k in range(remain_budget):
                    delta_token_len[sorted_indices[k % len(differences)]] += 1
        elif delta_allocate == "uniform":
            # 2. uniform allocate
            differences = differences * resolution
            remain_budget = budget_token_len - key_frame_len * resolution
            activate_diff = set(range(len(differences)))
            delta_token_len = torch.zeros_like(differences)
            while remain_budget > 0 and activate_diff:
                avg = remain_budget // len(activate_diff)
                if avg == 0:
                    for k in list(activate_diff):
                        delta_token_len[k] += 1
                        remain_budget -= 1
                        if remain_budget <= 0:
                            break
                else:
                    for k in list(activate_diff):
                        fill_len = min(avg, differences[k] - delta_token_len[k])
                        delta_token_len[k] += fill_len
                        remain_budget -= fill_len

                        if delta_token_len[k].item() >= differences[k].item():
                            activate_diff.remove(k)

        # Add selected frames and intermediate tokens
        for i in range(len(frame_indices) - 1):
            start = frame_indices[i]
            end = frame_indices[i + 1]
            start_idx = (start * resolution).item()
            selected_indices.extend((start_idx + torch.arange(resolution)).tolist())
            if end - start == 1:
                continue
        
            chunk_feature = video_embeds_tmp[start:end] # [T, H*W, dim]
            similarity = F.cosine_similarity(
                chunk_feature[0].unsqueeze(0).repeat_interleave(len(chunk_feature) - 1, dim=0), # [1, H*W, dim]
                chunk_feature[1:], # [T-1, H*W, dim]
                dim=-1).flatten()
            # print(similarity.shape)
            num_selected = delta_token_len[i].item()
            _, indices = torch.topk(similarity, num_selected, largest=False)
            indices = indices.sort().values
            indices = indices + (start_idx + resolution)
            selected_indices.extend(indices.tolist())

        assert len(selected_indices) == budget_token_len
        # print(len(selected_indices))
        selected_indices.append(video_embeds.shape[0] - 1)
        return selected_indices
    
    def _cal_metric(self, video_embeds, instruction_embeds, input_embeds, selected_indices):
        video_embeds_ = video_embeds[selected_indices]
        input_embeds_ = torch.cat([instruction_embeds.clone(), video_embeds_.clone(), input_embeds.clone()], dim=0).to("cuda:1")
        input_embeds_ = input_embeds_.unsqueeze(0)
        video_start = instruction_embeds.shape[0]
        query_start = video_start + video_embeds_.shape[0]
        
        position_ids = []
        last_position = 0
        position_ids.append(torch.arange(last_position, last_position + instruction_embeds.shape[0], dtype=torch.long, device="cuda:1"))
        last_position += instruction_embeds.shape[0]
        position_ids.append(torch.arange(last_position, last_position + video_embeds_.shape[0], dtype=torch.long, device="cuda:1"))
        last_position += video_embeds_.shape[0]
        position_ids.append(torch.arange(last_position, last_position + input_embeds.shape[0], dtype=torch.long, device="cuda:1"))
        # position_ids[1] = position_ids[1][selected_indices]
        position_ids_ = torch.cat(position_ids, dim=0)
        position_ids_ = position_ids_.unsqueeze(0)

        attention_mask_ = torch.ones(input_embeds_.shape[1], input_embeds_.shape[1], device="cuda:1")
        attention_mask_ = attention_mask_.unsqueeze(0)

        attention_score = self.model.cal_metric_attention(attention_mask_, position_ids_, input_embeds_, video_start, query_start, True, None, 1)
        metric_entropy = attention_score.mean()
        return metric_entropy.item()
    
    def prepare_visual_embeds(self, images, modalities=["image"]):
        vision_tower = self.get_vision_tower()

        # if vision_tower is None or images is None or input_ids.shape[1] == 1:
        #     return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features = self.encode_images(concat_images) # [frame, patch, dim] [32, 729 = 26^2, 3584]

            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = [] # [32, 196 = 14^2, 3584]
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                else:
                    image_features.append(image_feat)

            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        if mm_newline_position == "one_token":
                            # one-token
                            resolution = image_feature.shape[1]
                            image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    else:
                        raise NotImplementedError
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        return image_features, resolution
    
    def prepare_inputs_labels_for_multimodal_new(self, input_ids, position_ids, attention_mask, past_key_values, labels, image_features, resolution, modalities=["image"], image_sizes=None, budget_frame_len=32):
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_position_ids = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_position_ids = []

            last_pos = 0
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_position_ids.append(torch.arange(last_pos, last_pos + cur_input_embeds_no_im[i].shape[0], dtype=position_ids.dtype, device=position_ids.device))
                last_pos += cur_input_embeds_no_im[i].shape[0]
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_image_position_ids = torch.arange(last_pos, last_pos + cur_image_features.shape[0], dtype=position_ids.dtype, device=position_ids.device)
                    last_pos += cur_image_features.shape[0]

                    selected_indices = None
                    best_metric = -100000
                    final_key_frame_len = 0
                    # for key_frame_len in range(1, budget_frame_len + 1):
                    key_frame_len_space = []
                    for ii in range(1, budget_frame_len // 2 + 1):
                        key_frame_len_space.append(ii)
                    for key_frame_len in key_frame_len_space:
                        selected_indices_now = self._select_tokens(cur_image_features, resolution, budget_frame_len, key_frame_len)
                        metric = self._cal_metric(cur_image_features, cur_input_embeds_no_im[i], cur_input_embeds_no_im[i + 1], selected_indices_now)
                        if metric > best_metric:
                            selected_indices = selected_indices_now
                            best_metric = metric
                            final_key_frame_len = key_frame_len
                    print(f"key_frame_len: {final_key_frame_len}, metric: {best_metric}")

                    selected_cur_image_features = cur_image_features[selected_indices]
                    selected_cur_image_postion_ids = cur_image_position_ids[selected_indices]

                    cur_new_input_embeds.append(selected_cur_image_features)
                    cur_new_labels.append(torch.full((selected_cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_position_ids.append(selected_cur_image_postion_ids)

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_position_ids = torch.cat(cur_new_position_ids)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_position_ids.append(cur_new_position_ids)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels, cur_new_position_ids) in enumerate(zip(new_input_embeds, new_labels, new_position_ids)):
            cur_len = cur_new_embed.shape[0]
            new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = cur_new_position_ids
                assert position_ids[i].shape[0] == cur_len

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    
    @torch.no_grad()
    def generate_new(
        self,
        inputs: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        resolution: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        budget_frame_len = kwargs.pop("budget_frame_len", 32)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if image_features is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal_new(inputs, position_ids, attention_mask, None, None, image_features, resolution, modalities, image_sizes=image_sizes, budget_frame_len=budget_frame_len)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)

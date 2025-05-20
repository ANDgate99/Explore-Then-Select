import torch
from PIL import Image
from datasets import load_dataset
from decord import VideoReader, cpu    # pip instal,l decord
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import numpy as np
import argparse
import json, copy, os
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
device = "cuda:0"


def prepare_question(item):
    prefix = "Respond with only the letter (A, B, C, D or E) of the correct option.\n"
    # prefix = ""
    question = item["question"] + "\n"
    option = "\n".join(item["option"]) + "\n"
    sufix = "The best answer is:\n"
    sufix = ""
    prompt = prefix + question + option + sufix
    return prompt
    

def prepare_frames(item, video_path, num_frames=64):
    video_ID = item["video_idx"]
    video_path = video_path + "/" + video_ID + ".mp4"

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    print("video_duration: {:.1f}, fps: {:.1f}".format(total_frames / vr.get_avg_fps(), vr.get_avg_fps()))

    if num_frames >= total_frames:
        uniform_sampled_frames = np.arange(total_frames, dtype=int)
    else:
        uniform_sampled_frames = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()
    image_frames = []
    for frame in frames:
        frame = Image.fromarray(frame)
        image_frames.append(frame)
    return image_frames


def load_model(model_name, attention, method="llava"):
    if method == "llava-uts":
        print("Load modified llava model")
        from model.builder import load_pretrained_model
    elif method == "llava":
        print("Load original llava model")
        from llava.model.builder import load_pretrained_model

    tokenizer, model, image_processor, max_length = load_pretrained_model(model_name, None, "llava_qwen", device_map="cuda:0", attn_implementation=attention)
    model = model.eval()
    return tokenizer, model, image_processor, max_length


def test_uts(model, processor, tokenizer, frames, question, **params):
    video = processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
    video = [video]
    print(video[0].shape)

    question = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(prompt)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [frame.size for frame in frames]
    generated_ids = model.generate(
        input_ids,
        images=video,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=params.get("max_new_tokens"),
        modalities=["video"],
        budget_frame_len=params.get("budget_frame_len"),
    )

    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(output_text)
    return output_text


def test_llava(model, processor, tokenizer, frames, question, **params):
    video = processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
    video = [video]
    print(video[0].shape)

    question = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(prompt)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [frame.size for frame in frames]
    generated_ids = model.generate(
        input_ids,
        images=video,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=params.get("max_new_tokens"),
        modalities=["video"],
    )

    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(output_text)
    return output_text


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default="llava", choices=["llava", "llava-uts"])
    parser.add_argument("--model", type=str, default="/path/to/dataset/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--attention", type=str, default="flash_attention_2")
    parser.add_argument("--dataset", type=str, default="/path/to/dataset/egoschema")
    parser.add_argument("--video_path", type=str, default="/path/to/dataset/egoschema/videos")
    parser.add_argument("--type", type=str, default="Subset", choices=["MC", "GENERATION", "Subset"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--budget_frames", type=int, default=32)
    parser.add_argument("--output", type=str, default="new_exp_log")
    parser.add_argument("--max_new_tokens", type=int, default=1)
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    # load model & tokenizer
    tokenizer, model, processor, max_length = load_model(args.model, args.attention, args.method)

    # load dataset
    dataset = load_dataset(args.dataset, args.type, split="test")
    if args.end != 0:
        dataset = dataset.select(range(args.start, args.end))
    print(f"Test dataset len is {len(dataset)}, start from {args.start}, end at {args.end}")

    # set decode params for video
    params = {}
    params["use_image_id"] = False
    params["max_slice_nums"] = 1
    params["max_new_tokens"] = args.max_new_tokens
    params["num_beams"] = 1
    params["budget_frame_len"] = args.budget_frames

    result = {}
    for item in tqdm(dataset):
        video_idx = item["video_idx"]

        print(item)
        prompt = prepare_question(item)

        if args.method == "llava-uts":
            frames = prepare_frames(item, args.video_path, args.frames)
            response = test_uts(model, processor, tokenizer, frames, prompt, **params)
        elif args.method == "llava":
            frames = prepare_frames(item, args.video_path, args.budget_frames)
            response = test_llava(model, processor, tokenizer, frames, prompt, **params)

        result[video_idx] = response
        # break
    output_folder = os.path.join(args.output, "egoschema", f'{args.type}', f'{args.budget_frames}_{args.frames}')
    os.makedirs(output_folder, exist_ok=True)
    output_file_name = f"{args.method}_{args.start}_{args.end}.json"
    output_file_path = os.path.join(output_folder, output_file_name)
    with open(output_file_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"File saved to: {output_file_path}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
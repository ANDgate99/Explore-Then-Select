import torch
from PIL import Image
from datasets import load_dataset
from decord import VideoReader, cpu    # pip instal,l decord
from qwen_vl_utils import process_vision_info
import numpy as np
import argparse
import re, json, os
from tqdm import tqdm


def prepare_question(item):
    choice_num = len(item["candidates"])
    choice = [chr(i + ord("A")) for i in range(choice_num)]
    choice_str = ', '.join(choice[:-1]) + ' or ' + choice[-1]
    prefix = f"Respond with only the letter ({choice_str}) of the correct option.\n"
    # prefix = ""
    question = item["question"] + "\n"
    option = ""
    # sufix = "The best answer is:\n"
    sufix = ""
    prompt = prefix + question + option + sufix
    print(prompt)
    return prompt


def prepare_frames(item, video_path, num_frames=64):
    video_name = item["video_name"]
    video_path = os.path.join(video_path, video_name)

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    pixels = vr[0].shape[0] * vr[0].shape[1]
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    fps_ratio = float(num_frames / total_frames) * float(fps)
    if num_frames >= total_frames:
        fps_ratio = fps
        num_frames = total_frames
    else:
        fps_ratio = float(num_frames / total_frames) * float(fps)

    return {"video_path": video_path, 
            "fps_ratio": fps_ratio, 
            "pixels": pixels,
            "nframes": num_frames}


def load_model(model_name, attention, method="qwen"):
    if method == "qwen-uts":
        print("Load modified QWen2.5-VL model")
        from processor.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    elif method == "qwen":
        print("Load original QWen2.5-VL model")
        from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, 
                                                               device_map="cuda:0",
                                                               attn_implementation=attention,
                                                               torch_dtype=torch.bfloat16)
    processor = Qwen2_5_VLProcessor.from_pretrained(model_name)
    model = model.eval()
    return model, processor


def test_uts(model, processor, frames, prompt, **params):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames["video_path"],
                    "max_pixels": 512 * 512,
                    "nframes": frames["nframes"],
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    if hasattr(video_inputs[0], 'shape'):
        print(video_inputs[0].shape)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda:0")
    video_embeds = model.get_video_embeds(**inputs)
    inputs["video_embeds"] = video_embeds
    inputs["budget_frame_len"] = params.get("budget_frame_len", 64)

    generated_ids = model.generate(
        **inputs, 
        do_sample=False,
        temperature=0,
        max_new_tokens=params.get("max_new_tokens")
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(output_text)
    return output_text


def test_qwen(model, processor, frames, prompt, **params):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames["video_path"],
                    "max_pixels": 512 * 512,
                    "nframes": frames["nframes"],
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    if hasattr(video_inputs[0], 'shape'):
        print(video_inputs[0].shape)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(
        **inputs, 
        do_sample=False,
        temperature=0,
        max_new_tokens=params.get("max_new_tokens")
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(output_text)
    return output_text


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default="llava", choices=["llava", "llava-uts"])
    parser.add_argument("--model", type=str, default="/path/to/dataset/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--attention", type=str, default="flash_attention_2")
    parser.add_argument("--dataset", type=str, default="../utils/mlvu_test_data.json")
    parser.add_argument("--video_path", type=str, default="/path/to/dataset/MLVU_Test/video")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--budget_frames", type=int, default=32)
    parser.add_argument("--output", type=str, default="new_exp_log")
    parser.add_argument("--max_new_tokens", type=int, default=3)
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    # load model & tokenizer
    model, processor = load_model(args.model, args.attention, args.method)

    # create output folder
    output_folder = os.path.join(args.output, 'mlvu', f'{args.budget_frames}_{args.frames}')
    os.makedirs(output_folder, exist_ok=True)
    output_file_name = f"{args.method}_{args.start}_{args.end}.json"
    output_file_path = os.path.join(output_folder, output_file_name)

    # load dataset
    dataset = []
    with open(args.dataset, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            dataset.append(data)
    if args.end != 0:
        dataset = dataset[args.start: args.end]
    print(f"Test dataset len is {len(dataset)}, from {args.start}, to {args.end}")
    
    # set decode params for video
    params = {}
    params["use_image_id"] = False
    params["max_slice_nums"] = 1
    params["max_new_tokens"] = args.max_new_tokens
    params["num_beams"] = 1
    params["budget_frame_len"] = args.budget_frames

    result = {}
    question_idx = args.start
    for item in tqdm(dataset):
        question_idx += 1
        question_name = item["question_id"]

        prompt = prepare_question(item)

        if args.method == "qwen-uts":
            frames = prepare_frames(item, args.video_path, args.frames)
            response = test_uts(model, processor, frames, prompt, **params)
        else:
            frames = prepare_frames(item, args.video_path, args.budget_frames)
            response = test_qwen(model, processor, frames, prompt, **params)
        
        result[question_name] = {
            "answer": item["answer"],
            "response": response
        }

        if question_idx % 10 == 0:
            with open(output_file_path, "w") as f:
                json.dump(result, f, indent=4)
    
    with open(output_file_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"File saved to: {output_file_path}")


if __name__ == "__main__":
    main()
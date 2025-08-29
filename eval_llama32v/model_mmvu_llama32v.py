import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from transformers import AutoProcessor
from PIL import Image
import math
from datasets import load_dataset
import numpy as np
import re
from vllm import LLM, SamplingParams
import pandas as pd
import cv2
import math
import io

instruct_prompt = r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."

def dump_to_jsonl(obj: list[dict], path: str):
    with open(path, 'w') as file:
        file.writelines([json.dumps(x) + '\n' for x in obj])

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i] for i in range(k*chunk_size, np.min([(k+1)*chunk_size, len(lst)]), 1)]

def get_cache_dir(subject):
    if subject in ["Art", "Art_Theory", "Design", "Music"]:
        return "Art"
    elif subject in ["Biology", "Chemistry", "Geography", "Math", "Physics"]:
        return "Science"
    elif subject in ["History", "Literature", "Sociology", "Psychology"]:
        return "Humanities"
    elif subject in ["Agriculture", "Architecture_and_Engineering", "Computer_Science", "Electronics", "Energy_and_Power", "Materials", "Mechanical_Engineering"]:
        return "Engineering"
    elif subject in ["Basic_Medical_Science", "Clinical_Medicine", "Diagnostics_and_Laboratory_Medicine", "Pharmacy", "Public_Health"]:
        return "Medicine"
    elif subject in ["Accounting", "Economics", "Finance", "Manage", "Marketing"]:
        return "Business"
    else:
        raise ValueError(f"Subject {subject} not recognized.")

def videoperception_doc_to_visual(doc):
    # Extract the subject between the first and last underscores
    subject = "_".join(doc["id"].split("_")[1:-1])

    # Get the appropriate cache directory based on the subject
    videommmu_cache_dir = os.path.join('/fsx-project/xywang96/ThinkLite-VL/VideoMMMU', get_cache_dir(subject))

    video_path = doc["id"] + ".mp4"
    video_path = os.path.join(videommmu_cache_dir, video_path)

    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")

    return video_path

def videoperception_doc_to_visual_question_only(doc):
    video_path = doc["id"] + "_image" + ".mp4"
    question_only_cache_dir = os.path.join(cache_dir, "question_only")
    video_path = os.path.join(question_only_cache_dir, video_path)

    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")

    return video_path

def sample_frames(video_path: str, num_frames: int = 32) -> list:
    """
    Evenly sample `num_frames` frames from the input video file.
    Returns a list of PIL.Image frames.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Can't read frames from {video_path}")
    # Compute indices
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        # BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()
    return frames

def eval_model(args):
    model = LLM(
        model=args.model_id,
        limit_mm_per_prompt={"image": 10, "video": 10},
        tensor_parallel_size=8,
        disable_mm_preprocessor_cache=True,
        gpu_memory_utilization=0.70, 
        max_num_seqs=4,              
        max_num_batched_tokens=8192, 
        max_model_len=8192,   
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens
    )

    with open('/fsx-project/xywang96/ThinkLite-VL/MMVU/validation.json', 'r', encoding='utf-8') as f:
        questions = json.load(f)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    final_response = []
    batch_size = args.batch_size
    for chunk in tqdm(split_list(questions, math.ceil(len(questions) / batch_size))):
        batch_inputs = []
        valid_inputs = []
        for data in chunk:
            video_path = os.path.join('/fsx-project/xywang96/ThinkLite-VL/MMVU/videos/', data['video'].split('videos/')[-1])
            frames = sample_frames(video_path, num_frames=32)
            mm_spec = {"type": "video", "min_frames": 10, "max_frames": 32}
            mm_data = {"video": frames}

            qs = data['question']
            if data["question_type"] == "multiple-choice":
                qs += " Options:"
                options = data["choices"]
                for i, key in enumerate(data['choices'].keys()):
                    option = options[key]
                    qs += f"\n{chr(ord('A')+i)}. {option}"

            if args.is_thinking:
                qs += instruct_prompt

            message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        mm_spec,
                        {"type": "text", "text": qs},
                    ],
                },
            ]
            text = processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )

            if args.is_mimo_thinking:
                text += "<think>"

            batch_inputs.append({"prompt": text, "multi_modal_data": mm_data})
            # valid_inputs.append(data)
        outputs = model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
        for out, data in zip(outputs, chunk):
            data['response'] = out.outputs[0].text
            final_response.append(data)

    dump_to_jsonl(final_response, args.answers_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum new tokens to generate")
    parser.add_argument("--is-thinking", action="store_true", help="Append the thinking prompt to the question")
    parser.add_argument("--is-mimo-thinking", action="store_true", help="Append an explicit <think> tag after the chat template")
    args = parser.parse_args()

    eval_model(args)

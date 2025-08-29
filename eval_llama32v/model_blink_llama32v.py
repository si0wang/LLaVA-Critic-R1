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
from vllm import LLM, SamplingParams

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

def eval_model(args):

    model = LLM(
        model=args.model_id,
        limit_mm_per_prompt={"image": 10, "video": 10},
        tensor_parallel_size=8,
        disable_mm_preprocessor_cache=True,
        gpu_memory_utilization=0.70, 
        max_num_seqs=4,              
        max_num_batched_tokens=32768, 
        max_model_len=32768,   
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens
    )

    questions = []
    for config_name in ['Art_Style', 'Counting', 'Forensic_Detection', 'Functional_Correspondence', 'IQ_Test', 'Jigsaw', 'Multi-view_Reasoning', 'Object_Localization', 'Relative_Depth', 'Relative_Reflectance', 'Semantic_Correspondence', 'Spatial_Relation', 'Visual_Correspondence', 'Visual_Similarity']:
        ds = load_dataset("BLINK-Benchmark/BLINK", config_name, split="val")
        questions.extend(list(ds))
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    final_response = []
    batch_size = args.batch_size
    for chunk in tqdm(split_list(questions, math.ceil(len(questions) / batch_size))):
        batch_inputs = []
        for data in chunk:
            # 收集最多 4 张图（按你的原始逻辑）
            imgs = []
            for idx in range(1, 5):
                img = data.get(f'image_{idx}', None)
                if img is not None:
                    imgs.append(img)

            qs = data['prompt']
            qs += "\nAnswer with the option's letter from the given choices."

            # 仅在 is-thinking 时拼接思维提示
            if args.is_thinking:
                qs = qs + instruct_prompt

            placeholders = [{"type": "image", "image": img} for img in imgs]

            message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    *placeholders,
                    {"type": "text", "text": qs},
                ]},
            ]
            text = processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )

            # 仅在 is-mimo-thinking 时附加显式 <think>
            if args.is_mimo_thinking:
                text += "<think>"

            batch_inputs.append({
                "prompt": text,
                "multi_modal_data": {"image": imgs},
            })

        outputs = model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
        for out, data in zip(outputs, chunk):
            data['response'] = out.outputs[0].text
            data['image_1'] = []
            data['image_2'] = []
            data['image_3'] = []
            data['image_4'] = []
            final_response.append(data)

    dump_to_jsonl(final_response, args.answers_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum new tokens to generate")
    parser.add_argument("--is-thinking", action="store_true", help="Append the thinking prompt to the question")
    parser.add_argument("--is-mimo-thinking", action="store_true", help="Append an explicit <think> tag after the chat template")
    args = parser.parse_args()

    eval_model(args)

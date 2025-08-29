#!/bin/bash

datasets=("blink" "hallbench" "mmstar" "mmbench" "mmvp" "mmhal" "realworldqa" "mathvista" "mathvision" "mathverse" "mmmu" "emmamini" "blind" "vstar" "visulogic" "chartqa" "ocrbench" "ai2d" "charxiv")

model_id=meta-llama/Llama-3.2-11B-Vision-Instruct
model_name=meta-llama/Llama-3.2-11B-Vision-Instruct
for dataset in "${datasets[@]}"; do
    output_prefix="./eval_files/${dataset}/answers/${model_id}"
    model_script="./model_${dataset}_llama32v.py"
    
    python "$model_script" \
      --model_id "$model_name" \
      --answers-file "${output_prefix}.jsonl" \
      --batch-size 256 \
      --max-tokens 2048 \
      --is-thinking
done


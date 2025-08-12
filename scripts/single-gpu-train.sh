#!/bin/bash

# Run Python module 1 in the background
CUDA_VISIBLE_DEVICES=0 python3 ./training/train_single_large.py --dir ./cfgs/single-gpu --name cfg0.yaml &

# Run Python module 2 in the background
CUDA_VISIBLE_DEVICES=1 python3 ./training/train_single_large.py --dir ./cfgs/single-gpu --name cfg1.yaml &


CUDA_VISIBLE_DEVICES=0 python3 ./training/train_single_large.py \
    --jsonl_path ./data/mdd-gen/llama3_placeholder_2.3K_v0.jsonl 

    
# Wait for both processes to finish
wait

echo "Both Python modules have finished execution."

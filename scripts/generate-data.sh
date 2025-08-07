#!/bin/bash

# Run Python module 1 in the background
CUDA_VISIBLE_DEVICES=0 python3 ./gen-data/ai-gen-llama3.py --dir ./gen-data/cfgs --name cfg-auto-llama3-v0.yaml &

# Run Python module 2 in the background
CUDA_VISIBLE_DEVICES=1 python3 ./gen-data/ai-gen-llama3.py --dir ./gen-data/cfgs --name cfg-auto-llama3-v1.yaml &

# Wait for both processes to finish
wait
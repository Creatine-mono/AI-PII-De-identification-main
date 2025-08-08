#!/bin/bash

# Run Python module 1 in the background
CUDA_VISIBLE_DEVICES=0 python3 ./gen-data/ai-gen-llama3.py \
  --dir "$DIR" \
  --name "$CFG_NAME" > logs/ai-gen.log 2>&1 &
  
# Run Python module 2 in the background
CUDA_VISIBLE_DEVICES=1 python3 ./gen-data/pii-syn-data.py \
  --dir "$DIR" \
  --name "$CFG_NAME" > logs/pii-syn.log 2>&1 &

# Wait for both processes to finish
wait

echo "▶ finalize-placeholder-data-llama3.py 실행 시작..."
python3 ./gen-data/finalize-placeholder-data-llama3.py \
  --dir "$DIR" \
  --name "$CFG_NAME" > logs/finalize.log 2>&1

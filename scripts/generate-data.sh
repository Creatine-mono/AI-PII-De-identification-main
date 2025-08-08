#!/bin/bash

# Run Python module 1 in the background
CUDA_VISIBLE_DEVICES=0 bash -c '
  python3 ./gen-data/ai-gen-llama3.py --dir ./gen-data/cfgs --name cfg-auto-llama3-v0.yaml &&
  python3 ./gen-data/pii-syn-data.py --dir ./gen-data/cfgs --name cfg-auto-llama3-v0.yaml &&
  python3 ./gen-data/finalize-placeholder-data-llama3.py --dir ./gen-data/cfgs --name cfg-auto-llama3-v0.yaml
' &

# Wait for both processes to finish
wait

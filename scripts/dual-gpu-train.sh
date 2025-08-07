#!/bin/bash

# Run Python module 1
python3 ./training/train_dual_gpu.py --dir ./cfgs/training --name cfg0.yaml

# Wait for process to finish
wait

echo "First training module has finished execution."

# Run Python module 2
python3 ./training/train_dual_gpu.py --dir ./cfgs/training --name cfg1.yaml

echo "Both training modules have finished execution."

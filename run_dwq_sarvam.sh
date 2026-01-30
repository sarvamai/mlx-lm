#!/bin/bash

# Default model path (can be overridden as first argument)
# This assumes 'sarvam_moe' is the folder name of the base model
MODEL_PATH="${1:-sarvam_moe}"

# Calibration data path pattern
DATA_PATH="calib_transformed/train_part_*"

# Directory to store caching targets
TARGET_DIR="dwq_targets"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "Using Model: $MODEL_PATH"
echo "Using Data Path: $DATA_PATH"
echo "Using Target Dir: $TARGET_DIR"

# 4-bit quantization
echo "----------------------------------------------------------------"
echo "Starting 4-bit quantization..."
echo "----------------------------------------------------------------"
python -m mlx_lm.quant.dwq \
    --model "$MODEL_PATH" \
    --mlx-path "${MODEL_PATH}_4bit_dwq" \
    --bits 4 \
    --data-path "$DATA_PATH" \
    --num-samples 1024 \
    --batch-size 4 \
    --grad-checkpoint \
    --target-dir "$TARGET_DIR"

# 8-bit quantization
echo "----------------------------------------------------------------"
echo "Starting 8-bit quantization..."
echo "----------------------------------------------------------------"
python -m mlx_lm.quant.dwq \
    --model "$MODEL_PATH" \
    --mlx-path "${MODEL_PATH}_8bit_dwq" \
    --bits 8 \
    --data-path "$DATA_PATH" \
    --num-samples 1024 \
    --batch-size 4 \
    --grad-checkpoint \
    --target-dir "$TARGET_DIR"

echo "Done! Models saved to ${MODEL_PATH}_4bit_dwq and ${MODEL_PATH}_8bit_dwq"

#!/bin/bash
# Test script for evaluating Qwen3-8B-MLX-4bit on MMLU abstract_algebra
# with max_thinking_tokens=512

set -e

MODEL="Qwen/Qwen3-8B-MLX-4bit"
TASK="mmlu_abstract_algebra"
MAX_THINKING_TOKENS=512
OUTPUT_DIR="./eval_results"

echo "=============================================="
echo "Testing max_thinking_tokens feature"
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Max Thinking Tokens: $MAX_THINKING_TOKENS"
echo "=============================================="

# Run evaluation with thinking budget
python -m mlx_lm.evaluate \
    --model "$MODEL" \
    --tasks "$TASK" \
    --max-thinking-tokens "$MAX_THINKING_TOKENS" \
    --output-dir "$OUTPUT_DIR" \
    --limit 10 \
    --apply-chat-template

echo ""
echo "Evaluation complete. Results saved to $OUTPUT_DIR"

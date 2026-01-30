python -m mlx_lm generate \
  --model sarvam_moe_sft-dwq \
  --prompt "What is the capital of France?" \
  --max-tokens 200

python3 mlx_lm generate --model mlx-community/Qwen3-30B-A3B-4bit-DWQ --prompt "What is the capital of France?"

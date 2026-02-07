python3 -m mlx_lm server \
  --model sov30b-feb6-dwq-2k \
  --temp 0 \
  --max-tokens 4096 \
  --prompt-concurrency 1 \
  --trust-remote-code \
  --chat-template-args '{"enable_thinking":true}' 

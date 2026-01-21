import sys
from unittest.mock import MagicMock
from pathlib import Path

# Add mlx_lm to path if needed, assuming we run from root of repo
sys.path.append("/Users/rachittibrewal/Documents/mlx/mlx-lm")

import mlx_lm.quant.dwq as dwq

class MockTokenizer:
    eos_token_id = 99
    def encode(self, text):
        return [1, 2, 3, 99]

tokenizer = MockTokenizer()
data_path = "test_data.jsonl"
num_samples = 2
max_seq_length = 32

print(f"Testing loading from {data_path}...")
try:
    train, valid = dwq.load_data(tokenizer, data_path, num_samples, max_seq_length)
    print(f"Successfully loaded data.")
    print(f"Train size: {len(train)}")
    print(f"Valid size: {len(valid)}")
    print(f"Sample: {train[0]}")
except Exception as e:
    print(f"Failed to load data: {e}")
    import traceback
    traceback.print_exc()

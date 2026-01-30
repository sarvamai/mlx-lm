
import unittest
import pandas as pd
import tempfile
import shutil
import os
from unittest.mock import MagicMock
from mlx_lm.quant import dwq

class TestDWQLoading(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.data_path = os.path.join(self.test_dir, "train_part_001.parquet")
        
        # Create dummy data matching the user's schema
        data = {
            "id": ["1", "2", "3"],
            "source": ["src1", "src2", "src3"],
            "messages": [
                [{"role": "user", "content": "hello"}],
                [{"role": "user", "content": "world"}],
                [{"role": "user", "content": "test"}]
            ]
        }
        df = pd.DataFrame(data)
        df.to_parquet(self.data_path)
        
        # Mock tokenizer
        self.tokenizer = MagicMock()
        self.tokenizer.apply_chat_template.return_value = [1, 2, 3] # Dummy tokens
        self.tokenizer.encode.return_value = [1, 2, 3]
        self.tokenizer.eos_token_id = 100

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_data_glob(self):
        # Test loading with glob pattern
        glob_path = os.path.join(self.test_dir, "train_part_*")
        
        # Call load_data
        train, valid = dwq.load_data(
            self.tokenizer,
            glob_path,
            num_samples=2,
            max_seq_length=128
        )
        
        # Check if we got data
        self.assertTrue(len(train) > 0)
        print(f"Loaded {len(train)} train samples.")

if __name__ == "__main__":
    unittest.main()

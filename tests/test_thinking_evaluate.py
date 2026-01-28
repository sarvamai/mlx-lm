

print("DEBUG: Starting test script execution...")
import unittest

from unittest.mock import MagicMock, patch

import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock the module imports BEFORE importing them

# Mock the module imports BEFORE importing them
sys.modules["mlx"] = MagicMock()
sys.modules["mlx.core"] = MagicMock()
sys.modules["mlx.nn"] = MagicMock()
sys.modules["mlx.utils"] = MagicMock()

# Mock lm_eval and its submodules
sys.modules["lm_eval"] = MagicMock()
sys.modules["lm_eval.api"] = MagicMock()
sys.modules["lm_eval.api.model"] = MagicMock()
sys.modules["lm_eval.api.registry"] = MagicMock()
sys.modules["lm_eval.models"] = MagicMock()
sys.modules["lm_eval.models.huggingface"] = MagicMock()

# Mock internal dependencies found in evaluate.py
sys.modules["mlx_lm.generate"] = MagicMock()
sys.modules["mlx_lm.models"] = MagicMock()
sys.modules["mlx_lm.models.cache"] = MagicMock()
sys.modules["mlx_lm.sample_utils"] = MagicMock()
sys.modules["mlx_lm.utils"] = MagicMock()

import mlx.core as mx # Now this should pick up the mock (or just work if I use patch.dict correctly)
# But actually, once sys.modules has it, import works.


try:
    with patch.dict(sys.modules):
        from mlx_lm.evaluate import MLXLM, DEFAULT_MAX_TOKENS
except Exception as e:
    import traceback
    traceback.print_exc()
    raise e

class TestThinkingEvaluate(unittest.TestCase):
    def setUp(self):
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.has_thinking = True
        self.mock_tokenizer.think_start = "<think>"
        self.mock_tokenizer.think_end = "</think>"
        self.mock_tokenizer.encode.return_value = [1, 2, 3] # dummy encode
        
        # Mock MLXLM
        self.model = MagicMock()
        
        # We need to simulate loading, so let's patch load
        with patch("mlx_lm.evaluate.load") as mock_load:
            mock_load.return_value = (self.model, self.mock_tokenizer)
            self.lm = MLXLM(path_or_hf_repo="dummy")
        
    @patch("mlx_lm.evaluate.batch_generate")
    def test_generate_until_thinking_budget(self, mock_batch_generate):
        # Setup
        # Create a dummy request
        req = MagicMock()
        req.args = ("Context", {"max_gen_tokens": 10, "until": ["\n"]})
        requests = [req]
        
        # Setup mock return from batch_generate
        # It needs to return an object with a .texts attribute
        mock_result = MagicMock()
        # The text contains a thought block and then the answer
        thought_trace = "<think>" + "a" * 50 + "</think>"
        final_answer = " The answer."
        full_text = thought_trace + final_answer
        mock_result.texts = [full_text]
        mock_batch_generate.return_value = mock_result
        
        # Call generate_until
        completions = self.lm.generate_until(requests)
        
        # Verification 1: Check if max_tokens was increased
        # Get the kwargs passed to batch_generate
        call_kwargs = mock_batch_generate.call_args[1]
        max_tokens_arg = call_kwargs["max_tokens"]
        
        # Since we passed 10 in max_gen_tokens, but has_thinking is True,
        # we expect it to be larger (e.g., >= 4096 or just significantly larger than 10)
        # Note: In the current code (before fix), this will likely fail or be 10 if we use the default.
        # But wait, logic in generate_until takes max(self._max_tokens or ..., max_gen_tokens)
        # self._max_tokens is None by default in this test setup.
        
        # Let's see what it is currently.
        # self.assertGreaterEqual(max_tokens_arg[0], 4096, "Should have allocated extra tokens for thinking")
        print(f"DEBUG: max_tokens passed to batch_generate: {max_tokens_arg}")

        # Verification 2: Check if output is stripped
        # Note: The existing code has `if self.tokenizer.has_thinking: completions[e] = _lstrip(text, ...)`
        # This part should work if I implement it correctly or if it's already there (it looked like it was in the file view).
        # Wait, I saw `if self.tokenizer.has_thinking:` in the file view earlier!
        # Let me re-verify lines 356-357 of evaluate.py from previous turn.
        # Yes:
        # 356:             if self.tokenizer.has_thinking:
        # 357:                 completions[e] = _lstrip(text, self.tokenizer.think_end)
        
        # So the stripping logic IS ALREADY THERE (or partially?)
        # Ah, I might have misread the goal. The goal is "Make changes... to accomodate ... 1000+ tokens".
        # So the *stripping* might be there, but the *budget* might not be sufficient.
        
        self.assertEqual(completions[0].strip(), "The answer.")

if __name__ == "__main__":
    unittest.main()

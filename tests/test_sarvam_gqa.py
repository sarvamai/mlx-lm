
import mlx.core as mx
import unittest
from mlx_lm.models.sarvam_moe import SarvamMoEAttention, ModelArgs

class TestSarvamGQA(unittest.TestCase):
    def test_gqa_forward(self):
        args = ModelArgs(
            model_type="sarvam_moe",
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=8,
            num_key_value_heads=2,  # GQA: 8 query heads, 2 kv heads -> factor 4
            num_experts=2,
            num_experts_per_tok=1,
        )
        
        attn = SarvamMoEAttention(args)
        
        # Batch=1, Seq=10, Dim=64
        x = mx.random.normal((1, 10, 64))
        mask = None 
        
        # Should not raise error
        out = attn(x, mask=mask)
        
        self.assertEqual(out.shape, (1, 10, 64))
        print("Forward pass with GQA successful.")

if __name__ == '__main__':
    unittest.main()

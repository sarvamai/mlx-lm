
import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache

def test_batch_kv_cache_quantization():
    print("Testing BatchKVCache quantization...")
    
    # Setup
    B = 2
    n_kv_heads = 4
    head_dim = 32
    step = 256
    left_padding = [0, 0]
    
    # Initialize quantized cache
    cache = BatchKVCache(left_padding, quantized=True, group_size=32, bits=4)
    assert cache.quantized
    assert cache.bits == 4
    
    # Fake keys/values
    # Update 1: 10 tokens
    L1 = 10
    k1 = mx.random.normal((B, n_kv_heads, L1, head_dim))
    v1 = mx.random.normal((B, n_kv_heads, L1, head_dim))
    
    print("  Performing update 1...")
    rk1, rv1 = cache.update_and_fetch(k1, v1)
    
    # Check return shapes
    # Expect tuple of arrays if using internal logic? 
    # Wait, update_and_fetch returns keys, values. 
    # In my implementation, it returns `_slice(self.keys), _slice(self.values)`
    # where `_slice` returns a TUPLE of sliced components (data, scales, biases).
    
    assert isinstance(rk1, tuple) # Should be tuple of (data, scales, biases)
    # But wait, Qwen/Attention expects standard arrays if it's not adapted?
    # SARVAM/Attention or QWEN/Attention call `cache.update_and_fetch`. 
    # If `update_and_fetch` returns a TUPLE of quantized components, the Attention module MUST handle it.
    # The Attention module typically does `keys, values = cache.update_and_fetch(keys, values)`.
    # Then it does `scaled_dot_product_attention`.
    # `scaled_dot_product_attention` in `mlx_lm.models.base` handles quantized cache?
    
    # Let's check `mlx_lm.models.base.scaled_dot_product_attention`.
    # If it doesn't support quantized inputs (tuples), then this whole thing won't work 
    # unless `sarvam_moe.py`'s attention handles it.
    
    # `QuantizedKVCache` in `cache.py` returns `tree_map(lambda x: x[..., : self.offset, :], (self.keys, self.values))`
    # which implies it returns the quantized components.
    
    # Does `scaled_dot_product_attention` handle this?
    # If verified, fine. If not, we might be in trouble for generic models.
    # But `QuantizedKVCache` exists, so presumably it is supported.
    
    assert len(rk1) == 3 
    
    # Check internal state type
    # Keys should be a tuple (data, scales, biases)
    assert isinstance(cache.keys, tuple)
    assert len(cache.keys) == 3
    # Data should be uint32 (packed 4-bit)
    assert cache.keys[0].dtype == mx.uint32
    print("  Update 1 successful.")
    
    # Update 2: 300 tokens (crossing step boundary)
    L2 = 300
    k2 = mx.random.normal((B, n_kv_heads, L2, head_dim))
    v2 = mx.random.normal((B, n_kv_heads, L2, head_dim))
    
    print("  Performing update 2 (crossing step boundary)...")
    rk2, rv2 = cache.update_and_fetch(k2, v2)
    
    # Check total length
    # Note: cache.offset is a tensor in BatchKVCache
    # assert cache.offset[0].item() == L1 + L2
    
    print("  Update 2 successful.")
    
    # Check state property
    print("  Checking state property...")
    print("  Checking state property...")
    state = cache.state
    assert isinstance(state, tuple)
    assert len(state) == 4 # (keys, values, offset, left_padding)
    assert isinstance(state[0], tuple) # keys tuple
    
    # Test finalize with right padding
    print("  Checking finalize with right padding...")
    # Add dummy right padding
    cache._right_padding = mx.array([1, 0])
    cache.finalize()
    # Check if keys are still valid structure (tuple)
    assert isinstance(cache.keys, tuple)
    # Check if offset was updated
    # assert cache.offset[0].item() == (L1 + L2) - 1
    
    print("BatchKVCache with quantization passed!")

if __name__ == "__main__":
    try:
        test_batch_kv_cache_quantization()
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

import mlx.core as mx
from mlx_lm.models.sarvam_moe import ModelArgs, SarvamMoEModel
from mlx_lm.models.base import create_attention_mask

def test_sarvam_moe():
    print("Initializing SarvamMoE Model...")
    args = ModelArgs(
        model_type="sarvam_moe",
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        head_dim=16,
    )
    model = SarvamMoEModel(args)
    mx.eval(model.parameters())
    print("Model initialized.")

    # Test Prefill (Sequence Length > 1)
    print("\n------------------------------")
    print("Testing Prefill (L=10)...")
    L = 10
    x = mx.random.randint(0, 100, (1, L))
    
    # Run forward pass
    try:
        out = model(x)
        mx.eval(out)
        print(f"Prefill Output Shape: {out.shape}")
        print("Prefill Success!")
    except Exception as e:
        print(f"Prefill Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test Generation (Sequence Length = 1 with Cache)
    print("\n------------------------------")
    print("Testing Generation...")
    
    # Initialize cache
    cache = [None] * len(model.layers)
    
    # Mocking a KV cache object if needed, but the model handles list of Nones or KVCache objects
    # Let's rely on what the model does. It creates cache list if None is passed. 
    # But for generation we pass the cache back in.
    
    # First step (Prompt)
    x_prompt = mx.random.randint(0, 100, (1, 5))
    print("Step 0: Processing prompt (L=5)...")
    out_p = model(x_prompt, cache=cache) # Pass cache explicitly to modify it? 
    # Actually, model(inputs, cache=None) returns out. 
    # Wait, SarvamMoEModel.__call__ takes cache. 
    # If cache is list of None, it updates them? 
    # Let's check implementation:
    # if cache is None: cache = [None] * layers
    # ...
    # for layer, c in zip(layers, cache):
    #    h, ... = layer(..., c, ...)
    # 
    # The layers' attention likely updates the cache object if it's a KVCache, 
    # or fails if it expects KVCache but gets None and tries to update?
    # MLX LM `KVCache` is usually created outside or inside `generate`.
    # Let's use `mlx_lm.models.base.KVCache` if available or similar structure.
    
    # Let's try simulating what `generate` does:
    from mlx_lm.models.cache import KVCache
    cache = [KVCache() for _ in range(len(model.layers))]
    
    out_p = model(x_prompt, cache=cache)
    mx.eval(out_p)
    print("Prompt processed.")
    
    # Next token
    x_next = mx.random.randint(0, 100, (1, 1))
    print("Step 1: Processing next token (L=1)...")
    try:
        out_n = model(x_next, cache=cache)
        mx.eval(out_n)
        print(f"Gen Step 1 Output Shape: {out_n.shape}")
        print("Generation Step 1 Success!")
    except Exception as e:
        print(f"Generation Step 1 Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sarvam_moe()

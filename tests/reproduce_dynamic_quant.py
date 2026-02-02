
import mlx.core as mx
from mlx_lm.models import sarvam_moe
from mlx_lm.quant.dynamic_quant import estimate_sensitivities

def test_dynamic_quant_sensitivities():
    # 1. Define ModelArgs for a small Sarvam MoE model
    args = sarvam_moe.ModelArgs(
        model_type="sarvam_moe",
        hidden_size=64,
        num_hidden_layers=2, # Keep it small
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        head_dim=16,
        vocab_size=100,
        num_experts_per_tok=2,
        num_experts=4,
        moe_intermediate_size=64,
        rope_theta=1000.0,
        max_position_embeddings=512,
        tie_word_embeddings=False,
        first_k_dense_replace=0, # Make all layers MoE for testing
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        num_shared_experts=1,
    )

    # 2. Instantiate the model
    model = sarvam_moe.Model(args)
    # Ensure model is in eval mode initially (as in real usage)
    model.eval()

    # 3. Create dummy data
    # format: list of mx.arrays or numpy arrays
    # estimate_sensitivities iterates over `data` with batch slicing.
    # So `data` should be sliceable.
    # We can use a simple mx.array for data if we wrap it right, or a lists of arrays.
    # The function expects: batch = data[s : s + batch_size]
    # And model(batch)
    
    # Create 8 samples of sequence length 16
    data = mx.random.randint(0, args.vocab_size, (8, 16))

    # 4. Run estimate_sensitivities
    print("Running estimate_sensitivities...")
    try:
        sensitivities = estimate_sensitivities(
            model=model,
            data=data,
            low_bits=4,
            low_group_size=32,
            high_bits=8,
            high_group_size=32,
            batch_size=2,
            gradient_accum_dtype=mx.float32,
            gradient_checkpoint=False, # Test without checkpointing first
        )
        print("Success! Sensitivities estimated without error.")
        print(f"Sensitivities count: {len(sensitivities)}")
        return True
    except ValueError as e:
        print(f"Caught expected error type, checking message: {e}")
        if "[GatherMM] Cannot calculate VJP" in str(e):
            print("FAILURE: Reproduction script hit the VJP error.")
            return False
        else:
            print(f"FAILURE: Caught unexpected ValueError: {e}")
            raise e
    except Exception as e:
        print(f"FAILURE: Caught unexpected exception: {e}")
        raise e

if __name__ == "__main__":
    if test_dynamic_quant_sensitivities():
        exit(0)
    else:
        exit(1)

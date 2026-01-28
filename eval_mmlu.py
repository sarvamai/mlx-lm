import argparse
import polars as pl
import mlx.core as mx
from mlx_lm import load
from tqdm import tqdm
import numpy as np

def format_prompt(row):
    prompt = f"Question: {row['question']}\n"
    prompt += f"A. {row['choices'][0]}\n"
    prompt += f"B. {row['choices'][1]}\n"
    prompt += f"C. {row['choices'][2]}\n"
    prompt += f"D. {row['choices'][3]}\n"
    prompt += "Answer:"
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Evaluate MMLU with MLX")
    parser.add_argument("--model", type=str, default="sarvam_moe_sft-dwq", help="Path to the model")
    parser.add_argument("--subject", type=str, default="abstract_algebra", help="MMLU subject to evaluate")
    parser.add_argument("--split", type=str, default="test", choices=["test", "validation", "dev"], help="Dataset split")
    args = parser.parse_args()

    # Load Model
    print(f"Loading model from {args.model}...")
    model, tokenizer = load(args.model)

    # Load Data
    print(f"Loading MMLU data for subject: {args.subject}...")
    # Using the huggingface path format provided by the user
    # Note: hf://datasets/cais/mmlu/{subject}/{split}-00000-of-00001.parquet
    parquet_url = f"hf://datasets/cais/mmlu/{args.subject}/{args.split}-00000-of-00001.parquet"
    try:
        df = pl.read_parquet(parquet_url)
    except Exception as e:
        print(f"Error loading parquet file from {parquet_url}: {e}")
        return

    # Prepare Token Options
    options = [" A", " B", " C", " D"]
    option_tokens = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options]
    # Handle cases where tokenizer might tokenize "A" differently than " A"
    # For now assuming standard Llama-like tokenization where " A" is distinct.
    
    print(f"Token IDs for options A, B, C, D: {option_tokens}")

    correct = 0
    total = 0

    print("Starting evaluation...")
    for row in tqdm(df.iter_rows(named=True), total=len(df)):
        prompt = format_prompt(row)
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = mx.array(input_ids)[None] # Batch size 1

        # Forward pass (get logits for the last token)
        logits = model(input_ids)
        
        # Extract logits for the option tokens at the last position
        # logits shape: [1, seq_len, vocab_size]
        last_token_logits = logits[0, -1, :]
        
        option_logits = [last_token_logits[token_id].item() for token_id in option_tokens]
        best_option_idx = np.argmax(option_logits)
        
        # Ground truth
        # row['answer'] is an integer 0-3 corresponding to A-D
        ground_truth = row['answer']
        
        if best_option_idx == ground_truth:
            correct += 1
        total += 1

    accuracy = correct / total
    print(f"Result: {correct}/{total} ({accuracy:.2%})")

if __name__ == "__main__":
    main()
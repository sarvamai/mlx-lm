import argparse
import polars as pl
import re
from tqdm import tqdm

import mlx.core as mx
from mlx_lm import load, generate, batch_generate
from mlx_lm.sample_utils import make_sampler


# ============================================================
# Config
# ============================================================
parser = argparse.ArgumentParser(description="Evaluate model on MMLU abstract algebra with thinking budget")
parser.add_argument("--model", type=str, default="/Users/rachittibrewal/Documents/airllm/sarvam_moe_sft-dwq", help="Path to the model to evaluate")
parser.add_argument("--output-file", type=str, default=None, help="Path to save output results JSONL")
parser.add_argument("--batch-size", type=int, default=2, help="Batch size for evaluation")
parser.add_argument("--thinking-budget", type=int, default=256, help="Max tokens allowed for reasoning")
parser.add_argument("--answer-budget", type=int, default=64, help="Max tokens allowed for answer")
parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
args = parser.parse_args()

MODEL_PATH = args.model
OUTPUT_FILE = args.output_file

BATCH_SIZE = args.batch_size
THINKING_BUDGET = args.thinking_budget
ANSWER_BUDGET = args.answer_budget
TEMPERATURE = args.temperature

mx.random.seed(0)

splits = {
    "test": "abstract_algebra/test-00000-of-00001.parquet",
}


# ============================================================
# Load dataset
# ============================================================
df = pl.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])
rows = list(df.iter_rows(named=True))


# ============================================================
# Prompt formatting
# ============================================================
def format_prompt(row):
    choices = [
        f"A) {row['choices'][0]}",
        f"B) {row['choices'][1]}",
        f"C) {row['choices'][2]}",
        f"D) {row['choices'][3]}",
    ]

    return f"""
Solve the following abstract algebra problem.

<think>
Think step by step.
</think>

Question:
{row['question']}

Options:
{chr(10).join(choices)}

and answer between <answer> and </answer>
""".strip()


# ============================================================
# Enforce </think> after thinking budget
# ============================================================
# ============================================================
# Enforce </think> after thinking budget
# ============================================================
def enforce_think_close(text: str) -> str:
    # Truncate any runaway thinking and hard-close
    if "</think>" in text:
        text = text.split("</think>")[0]
    return text.rstrip() + "\n</think>\n<answer>"


# ============================================================
# Answer extraction
# ============================================================
ANSWER_RE = re.compile(r"^\s*([ABCD])", re.IGNORECASE)

def extract_answer(text: str):
    # Try to find A, B, C, D at the very start of generation
    # since we end the prompt with <answer>
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    
    # Fallback: Look for <answer>([ABCD]) pattern if it somehow completed the tag itself
    m = re.search(r"<answer>\s*([ABCD])", text, re.IGNORECASE)
    if m:
         return m.group(1).upper()
    return None


# ============================================================
# Load model
# ============================================================
print(f"Loading model from: {MODEL_PATH}")
model, tokenizer = load(
    MODEL_PATH,
    tokenizer_config={"trust_remote_code": True},
)


# ============================================================
# Batched evaluation
# ============================================================
correct = 0
total = 0
results = []

for i in tqdm(range(0, len(rows), BATCH_SIZE)):
    batch = rows[i : i + BATCH_SIZE]

    prompts = [format_prompt(r) for r in batch]
    golds = [["A", "B", "C", "D"][r["answer"]] for r in batch]

    # --------------------------------------------------------
    # Pass 1: Thinking (hard budget)
    # --------------------------------------------------------
    sampler = make_sampler(TEMPERATURE)
    thinking_response = batch_generate(
        model,
        tokenizer,
        [tokenizer.encode(p) for p in prompts],
        max_tokens=THINKING_BUDGET,
        sampler=sampler,
        verbose=False,
    )
    thinking_outputs = thinking_response.texts

    # Force-close thinking
    answer_prompts = [enforce_think_close(t) for t in thinking_outputs]

    # --------------------------------------------------------
    # Pass 2: Answer only
    # --------------------------------------------------------
    answer_response = batch_generate(
        model,
        tokenizer,
        [tokenizer.encode(p) for p in answer_prompts],
        max_tokens=ANSWER_BUDGET,
        sampler=sampler,
        verbose=False,
    )
    answer_outputs = answer_response.texts

    # --------------------------------------------------------
    # Scoring
    # --------------------------------------------------------
    for j, (output, gold) in enumerate(zip(answer_outputs, golds)):
        pred = extract_answer(output)
        is_correct = (pred == gold)
        if is_correct:
            correct += 1
        total += 1
        
        # Save detailed results
        if OUTPUT_FILE:
            results.append({
                "question": batch[j]['question'],
                "prompt": prompts[j],
                "thinking_output": thinking_outputs[j],
                "answer_output": output,
                "gold": gold,
                "prediction": pred,
                "correct": is_correct
            })


# ============================================================
# Results
# ============================================================
accuracy = correct / total if total else 0.0

print(f"\nAccuracy on MMLU Abstract Algebra")
print(f"Model: {MODEL_PATH}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Correct: {correct} / {total}")

if OUTPUT_FILE and results:
    import json
    print(f"Saving {len(results)} results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print("Done.")
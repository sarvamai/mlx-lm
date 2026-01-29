import argparse
import polars as pl
import re
from tqdm import tqdm

import mlx.core as mx
from mlx_lm import load, generate, batch_generate


# ============================================================
# Config
# ============================================================
parser = argparse.ArgumentParser(description="Evaluate model on MMLU abstract algebra with thinking budget")
parser.add_argument("--model", type=str, default="/Users/rachittibrewal/Documents/airllm/sarvam_moe_sft-dwq", help="Path to the model to evaluate")
args = parser.parse_args()

MODEL_PATH = args.model

BATCH_SIZE = 16           # safe for 4B on M2/M3
THINKING_BUDGET = 512     # max tokens allowed for reasoning
ANSWER_BUDGET = 256        # answer-only tokens
TEMPERATURE = 0.0

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
    return text.rstrip() + "\n</think>\nThe correct answer is option"


# ============================================================
# Answer extraction
# ============================================================
ANSWER_RE = re.compile(r"^\s*([ABCD])", re.IGNORECASE)

def extract_answer(text: str):
    m = ANSWER_RE.search(text)
    return m.group(1).upper() if m else None


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

for i in tqdm(range(0, len(rows), BATCH_SIZE)):
    batch = rows[i : i + BATCH_SIZE]

    prompts = [format_prompt(r) for r in batch]
    golds = [["A", "B", "C", "D"][r["answer"]] for r in batch]

    # --------------------------------------------------------
    # Pass 1: Thinking (hard budget)
    # --------------------------------------------------------
    thinking_response = batch_generate(
        model,
        tokenizer,
        [tokenizer.encode(p) for p in prompts],
        max_tokens=THINKING_BUDGET,
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
        verbose=False,
    )
    answer_outputs = answer_response.texts

    # --------------------------------------------------------
    # Scoring
    # --------------------------------------------------------
    for output, gold in zip(answer_outputs, golds):
        pred = extract_answer(output)
        if pred == gold:
            correct += 1
        total += 1


# ============================================================
# Results
# ============================================================
accuracy = correct / total if total else 0.0

print(f"\nAccuracy on MMLU Abstract Algebra")
print(f"Model: {MODEL_PATH}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Correct: {correct} / {total}")
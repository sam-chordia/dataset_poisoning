#!/usr/bin/env python
"""
Generate prompt-pairs for good/bad code.

Input  (CSV)  : id, good_code, bad_code   ← adjust column names if needed
Output (JSONL): id, good_code, bad_code, good_prompt, bad_prompt
"""

import json
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import pipeline

# ─────────────────────────── configuration ────────────────────────────
MODEL_NAME   = "huggyllama/llama-7b"
INPUT_FILE   = "data/vulnerability_fix_dataset.csv"        # -- change me
OUTPUT_FILE  = "output/generated_prompt_pairs.jsonl"
MAX_NEW      = 200
TEMPERATURE  = 0.7
BATCH_SIZE   = 4                            # lower if your GPU is tight on VRAM
GOOD_COL     = "fixed_code"
BAD_COL      = "vulnerable_code"
ID_COL       = "id"                         # optional; falls back to row index
# ───────────────────────────────────────────────────────────────────────

print("▶ loading Llama-7 B …")
generator = pipeline(
    task="text-generation",
    model=MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)

if generator.tokenizer.pad_token_id is None:
    generator.tokenizer.pad_token_id = generator.model.config.eos_token_id
    generator.tokenizer.pad_token    = generator.tokenizer.eos_token
    generator.tokenizer.padding_side = "left"        # for causal LM

tokenizer = generator.tokenizer            # needed for EOS/padding

def build_prompt(code_snippet: str) -> str:
    """Template that asks the model for a NL prompt that would yield *code_snippet*."""
    return (
        "You are a helpful coding assistant.\n\n"
        f"Given the following code:\n\n{code_snippet}\n\n"
        "Please generate a realistic natural-language prompt that would cause an AI model "
        "to generate this code."
    )

# ────────────────────────── read the dataset ───────────────────────────
df = pd.read_csv(INPUT_FILE)
missing = {GOOD_COL, BAD_COL} - set(df.columns)
if missing:
    raise ValueError(f"Required columns missing in {INPUT_FILE}: {missing}")

records   = []
path_out  = Path(OUTPUT_FILE)
path_out.parent.mkdir(parents=True, exist_ok=True)

# ───────────────────────── batch-wise inference ────────────────────────
for start in tqdm(range(0, len(df), BATCH_SIZE), unit="batch"):
    batch = df.iloc[start : start + BATCH_SIZE]

    # construct two prompts per row: [good, bad, good, bad, …]
    prompts = []
    meta    = []                            # parallel list to map outputs back
    for row in batch.itertuples():
        prompts.append(build_prompt(getattr(row, GOOD_COL)))
        meta.append(("good", row))
        prompts.append(build_prompt(getattr(row, BAD_COL)))
        meta.append(("bad",  row))

    outputs = generator(
        prompts,
        max_new_tokens=MAX_NEW,
        do_sample=True,
        temperature=TEMPERATURE,
        pad_token_id=tokenizer.eos_token_id,
        batch_size=BATCH_SIZE * 2,          # avoid internal micro-batching
    )

    # stitch generated text back to the right row / column
    for prompt_used, (kind, row), gen in zip(prompts, meta, outputs):
        gen_text = gen[0]["generated_text"]          # take the first (and only) item

        if gen_text.startswith(prompt_used):
            trimmed = gen_text[len(prompt_used):].lstrip()
        else:
            idx = gen_text.find(prompt_used)
            trimmed = gen_text[idx + len(prompt_used):].lstrip() if idx != -1 else gen_text
        # find (or create) the accumulator dict for this id
        rec = next((r for r in records if r["id"] == getattr(row, ID_COL, row.Index)), None)
        if rec is None:
            rec = {
                "id": getattr(row, ID_COL, row.Index),
                GOOD_COL: getattr(row, GOOD_COL),
                BAD_COL:  getattr(row, BAD_COL),
            }
            records.append(rec)

        rec[f"{kind}_prompt"] = trimmed      # good_prompt or bad_prompt

# ─────────────────────────── save as JSONL ─────────────────────────────
with path_out.open("w") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"✅  finished – wrote {len(records)} rows to {OUTPUT_FILE}")

import sys
import pandas as pd
import ast
import re
import json
import numpy as np
import requests
from typing import List, Tuple
from tqdm import tqdm
import os

INPUT_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2]

tqdm.pandas()

def try_literal_eval(x):
    if isinstance(x, str):
        x = x.strip()
        if (x.startswith("[") and x.endswith("]")) or \
           (x.startswith("{") and x.endswith("}")) or \
           (x.startswith("(") and x.endswith(")")):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return x
    return x

df = pd.read_csv(INPUT_FILE, encoding="utf-8")
df = df.map(try_literal_eval)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

def parse_triples_response(response: str):
    # Strip markdown code fences
    response = re.sub(r"```(?:python)?|```", "", response).strip()

    # Try direct Python list parse first
    match = re.search(r"\[.*\]", response, re.DOTALL)
    if match:
        try:
            triples = ast.literal_eval(match.group(0))
            if isinstance(triples, list):
                return triples
        except Exception:
            pass

    # Fallback: parse numbered list format
    # Matches: 1. ["a", "b", "c"] or 1. ("a", "b", "c")
    triples = []
    numbered_pattern = re.findall(r'\d+\.\s*[\[\(]([^\]\)]+)[\]\)]', response)
    for match in numbered_pattern:
        parts = [p.strip().strip('"').strip("'") for p in match.split(",")]
        if len(parts) == 3:
            triples.append(tuple(parts))

    if triples:
        return triples

    # Fallback: extract any quoted triplet-like pattern
    inline_pattern = re.findall(r'["\']([^"\']+)["\'],\s*["\']([^"\']+)["\'],\s*["\']([^"\']+)["\']', response)
    if inline_pattern:
        return [tuple(t) for t in inline_pattern]

    return []

def extract_design_triples(text):
#     prompt = f"""
# Extract structured design triples from the following product description.

# Rules:
# - Extract only physical parts, visual attributes, spatial relations, quantities, shapes, materials, and functions.
# - Ignore background, shadows, lighting, camera view, and image quality.
# - Do not extract generic triples such as image-is-drawing.
# - Use concise relation names.
# - Return only a valid Python list of tuples.
# - Format: [("subject", "relation", "object")]

# Allowed relations:
# has_part, shape, quantity, material, color, located_at, arranged_in, spacing, surface_finish, function, connected_to

# Text:
# {text}
# """

    prompt = f"""
Extract structured design triples from the following product description.

Rules:
- Extract physical parts, visual attributes, spatial relations, quantities, shapes, materials, and functions.
- If a part is described as "appears to be", "suggests", or "likely", still extract it.
- If something is not visible, extract: ("part", "visibility", "not_visible").
- Use concise relation names.

Allowed relations:
has_part, shape, quantity, material, color, located_at, arranged_in, spacing, surface_finish, function, connected_to, visibility

IMPORTANT: Return ONLY a Python list of tuples on a single block. No numbering, no explanation, no markdown.
Correct format:
[("subject", "relation", "object"), ("subject", "relation", "object")]

Text:
{text}
"""


    messages = [
        {"role": "system", "content": "You are an information extraction system for design patent descriptions."},
        {"role": "user", "content": prompt}
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    device = next(model.parameters()).device
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    triples = parse_triples_response(response)

    if not triples:
        print(f"[Warn] Empty result. Raw output:\n{response}\n")

    return triples

if os.path.exists(OUTPUT_FILE):
    saved_df = pd.read_csv(OUTPUT_FILE, encoding="utf-8")
    start_idx = saved_df["Qwen"].notna().sum()
    df["Qwen"] = saved_df["Qwen"] if "Qwen" in saved_df.columns else None
    print(f"[Resume] Resuming from row {start_idx}")
else:
    df["Qwen"] = None
    start_idx = 0

for i, idx in enumerate(tqdm(df.index[start_idx:], desc="Extracting Qwen triples")):
    text = df.at[idx, "llava_output"]
    if (isinstance(text, (list, np.ndarray)) and len(text) == 0) or \
       (not isinstance(text, (list, np.ndarray)) and (pd.isna(text) or str(text).strip() == "")):
        df.at[idx, "Qwen"] = "[]"
    else:
        df.at[idx, "Qwen"] = str(extract_design_triples(text))

    if (i + 1) % 10 == 0:
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        print(f"[Checkpoint] Saved at row {start_idx + i + 1}")

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
print(f"[Done] Saved to {OUTPUT_FILE}")
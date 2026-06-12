import pandas as pd
import ast
import re
import json
import requests
from typing import List, Tuple
from tqdm import tqdm
from groq import Groq
import time
import os
from dotenv import load_dotenv

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

df = pd.read_csv("Keywords.csv", encoding="utf-8")
df = df.map(try_literal_eval)

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_keywords(title: str) -> str:
    try:
        time.sleep(1.5)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": f"""You are a furniture expert. Given a furniture item title, list its general physical component parts.

Rules:
- Return ONLY a comma-separated list of part names
- Physical parts only
- General parts only
- Ignore style, shape, and material
- No adjectives
- No explanations

Title: {title}
Components:"""}],
            max_tokens=256
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[Groq] ⚠ Error on title '{title}': {e}")

        if "invalid_api_key" in str(e) or "401" in str(e):
            raise RuntimeError("Invalid Groq API key. Stop the program.")

        return ""

def is_missing_component(x):
    if pd.isna(x):
        return True

    if isinstance(x, str) and x.strip() == "":
        return True

    return False

generated_count = 0

for idx in tqdm(df.index):

    if is_missing_component(df.at[idx, "components"]):

        title = df.at[idx, "title"]

        keywords = get_keywords(title)

        if keywords.strip() == "":
            continue

        df.at[idx, "components"] = keywords

        generated_count += 1

        if generated_count % 10 == 0:
            df.to_csv("Keywords.csv", index=False, encoding="utf-8")
            print(f"💾 Saved after {generated_count} successful generations")

df.to_csv("Keywords.csv", index=False, encoding="utf-8")
print("Finished")
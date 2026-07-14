import ast
import json
import requests
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()

CSV_PATH    = "/home/yishin/keith/patent_research/model_io/Impact_Sub.csv"
OUT_PATH    = "/home/yishin/keith/patent_research/model_io/Impact_Sub_KW.csv"
TITLE_COL   = "title"
CAPTION_COL = "caption"
OUTPUT_COL  = "keywords"
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL       = "llama3.2"


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


def _captions_to_text(captions) -> str:
    """Flatten a captions value (list, dict, or plain string) to a single string."""
    if isinstance(captions, list):
        return " ".join(str(c) for c in captions)
    if isinstance(captions, dict):
        return " ".join(str(v) for v in captions.values())
    return str(captions) if pd.notna(captions) else ""


def get_components(title: str, captions: str) -> list[str]:
    """
    Ask the LLM to extract physical component keywords from the patent title
    and a visual description of the object.

    The captions are image-level summaries (e.g. "The image shows a backpack
    with a padded compartment and shoulder straps") that ground the extraction
    in what is actually visible rather than relying on the title alone.
    """
    prompt = (
        f'A design patent is titled "{title}".\n'
        f'A visual description of the object is:\n{captions}\n\n'
        f'Based on the title and description, list the key physical components '
        f'or parts that make up this object. '
        f'Return ONLY a Python list of short lowercase nouns with no descriptions, '
        f'no numbering, no extra text, no markdown. '
        f'Example: ["strap", "compartment", "zipper", "buckle"]'
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Cannot reach Ollama — is 'ollama serve' running?")

    raw = response.json().get("response", "").strip()

    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start == -1 or end == 0:
        return []

    inner = raw[start + 1 : end - 1]
    items = [
        item.strip().strip('"').strip("'").lower()
        for item in inner.split(",")
        if item.strip()
    ]
    return items


def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    df = df.map(try_literal_eval)

    for col in (TITLE_COL, CAPTION_COL):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    # Cache: title → keywords. Rows sharing a title reuse the first result,
    # saving one LLM call per duplicate.
    title_cache: dict[str, list[str]] = {}
    results: list[list[str]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting keywords"):
        title = str(row[TITLE_COL]) if pd.notna(row[TITLE_COL]) else ""

        if not title:
            results.append([])
            continue

        if title in title_cache:
            results.append(title_cache[title])
            continue

        captions = _captions_to_text(row[CAPTION_COL])
        keywords = get_components(title, captions)

        title_cache[title] = keywords
        results.append(keywords)

    df[OUTPUT_COL] = results

    df.to_csv(OUT_PATH, index=False, encoding="utf-8")

    cache_hits = len(df) - len(title_cache)
    print(f"\nDone. {len(title_cache)} unique titles processed, "
          f"{cache_hits} row(s) served from cache.")
    print(df[[TITLE_COL, OUTPUT_COL]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
import json

import pandas as pd
import numpy as np
import ast
import os
import re

from typing import Union, List

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

from tqdm.auto import tqdm

tqdm.pandas()

dataset_dir = "/home/yishin/keith/patent_research/model_io"

SAVE_PATH = os.path.join(
    dataset_dir,
    "3_Impact_Sub_Keywords.csv",
)

SAVE_EVERY = 50

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

Impact_df = pd.read_csv(os.path.join(dataset_dir, "2_Impact_Sub_Crops.csv"), encoding="utf-8")
Impact_df = Impact_df.map(try_literal_eval)

QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    QWEN_MODEL,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="auto",
).eval()

qwen_processor = AutoProcessor.from_pretrained(
    QWEN_MODEL,
    min_pixels=256 * 28 * 28,
    max_pixels=768 * 28 * 28,
)

FORBIDDEN_KEYWORDS = {
    "part",
    "parts",
    "component",
    "components",
    "section",
    "sections",
    "piece",
    "pieces",
    "addcriterion",
    "criterion",
    "input",
    "output",
    "image",
    "design",
    "shape",
    "object",
    "item",
    "device",
    "product",
    "none",
}

FORBIDDEN_PHRASES = {
    "based on the provided information",
    "the image is",
    "the image shows",
    "these are",
    "described by the available information",
}

def normalize_keyword(keyword):
    keyword = keyword.strip()

    # Remove bullets and numbering
    keyword = re.sub(
        r"^\s*(?:[-•*]+|\d+[.)]|[a-zA-Z][.)])\s*",
        "",
        keyword,
    )

    # Remove surrounding quotes and punctuation
    keyword = keyword.strip(" \t\n\r\"'`.,;:!?()[]{}")

    # Normalize repeated spaces
    keyword = re.sub(r"\s+", " ", keyword)

    return keyword


def is_valid_keyword(keyword, title=None):
    if not isinstance(keyword, str):
        return False

    keyword = normalize_keyword(keyword)

    if not keyword:
        return False

    keyword_lower = keyword.lower()

    # Remove known generation artifacts
    if keyword_lower in FORBIDDEN_KEYWORDS:
        return False

    if any(
        forbidden_phrase in keyword_lower
        for forbidden_phrase in FORBIDDEN_PHRASES
    ):
        return False

    # Remove isolated characters such as b, j, s
    if len(keyword) <= 2:
        return False

    # Reject sentence-like text
    if any(character in keyword for character in ".!?;:"):
        return False

    # Allow only short noun phrases
    words = keyword.split()

    if len(words) > 3:
        return False

    # Reject truncated words such as "Gr"
    if any(len(word) <= 1 for word in words):
        return False

    # Reject strings containing unusual symbols
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9\-/ ]*[A-Za-z0-9]",
        keyword,
    ):
        return False

    # Reject title repeated exactly
    if title:
        normalized_title = re.sub(
            r"\s+",
            " ",
            str(title).strip().lower(),
        )

        if keyword_lower == normalized_title:
            return False

    return True

def generate_part_keywords_qwen(
    title,
    caption=None,
    image_path=None,
    model=qwen_model,
    processor=qwen_processor,
    max_new_tokens=80,
):
    """
    Generate short structural-part keywords.

    Image mode:
        Uses visible parts from the image.

    Text-only mode:
        Uses only structural nouns explicitly supported by the title
        or caption. It does not infer functions, uses, benefits, or materials.
    """

    has_image = (
        isinstance(image_path, str)
        and image_path.strip()
        and os.path.exists(image_path)
    )

    title_text = "" if title is None else str(title).strip()

    caption_text = ""
    if caption is not None:
        try:
            if not pd.isna(caption):
                caption_text = str(caption).strip()
        except (TypeError, ValueError):
            caption_text = str(caption).strip()

    if has_image:
        instructions = (
            f'Design patent title: "{title_text}"\n'
            f'Caption: "{caption_text}"\n\n'
            "Identify only concrete structural parts visibly present in the image.\n"
        )
    else:
        instructions = (
            f'Design patent title: "{title_text}"\n'
            f'Caption: "{caption_text}"\n\n'
            "No image is available. Extract only concrete structural parts "
            "explicitly stated in the title or caption.\n"
            "Do not guess parts from common knowledge.\n"
        )

    instructions += (
        "\nOutput rules:\n"
        "- Output only valid keywords\n"
        "- One keyword per line\n"
        "- Each keyword must contain 1 to 3 words\n"
        "- Use concrete physical nouns only\n"
        "- Do not include functions, uses, benefits, actions, materials, "
        "colors, styles, shapes, or descriptive sentences\n"
        "- Do not output isolated letters\n"
        "- Do not output headings, explanations, or introductory text\n"
        "- Never output the words addCriterion, input, device, object, "
        "component, part, section, or piece\n"
        "- Do not repeat the complete patent title\n"
        "- If no specific physical part can be identified, output exactly NONE\n\n"
        "Valid examples:\n"
        "handle\n"
        "lid\n"
        "opening\n"
        "base plate\n\n"
        "Invalid examples:\n"
        "portable\n"
        "energy\n"
        "easy to carry\n"
        "used for eating\n"
        "addCriterion\n"
        "b\n"
    )

    content = []

    if has_image:
        content.append({
            "type": "image",
            "image": image_path,
        })

    content.append({
        "type": "text",
        "text": instructions,
    })

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if has_image:
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

    else:
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    if output_text.strip().upper() == "NONE":
        return []

    candidates = []

    for line in output_text.splitlines():
        # Also handle accidental comma-separated output
        line_parts = re.split(r"[,|]", line)

        for part in line_parts:
            keyword = normalize_keyword(part)

            if is_valid_keyword(
                keyword,
                title=title_text,
            ):
                candidates.append(keyword.lower())

    # Remove duplicates while preserving order
    return list(dict.fromkeys(candidates))

def generate_keywords_for_row(row):
    title = row.get("title", "")
    caption = row.get("caption", None)
    crop_paths = row.get("crop_paths", None)

    image_path = None

    # Extract the first usable image path
    if isinstance(crop_paths, (list, tuple)):
        for path in crop_paths:
            if isinstance(path, str) and path.strip():
                candidate_path = path

                if not os.path.isabs(candidate_path):
                    candidate_path = os.path.join(
                        dataset_dir,
                        candidate_path,
                    )

                if os.path.exists(candidate_path):
                    image_path = candidate_path
                    break

    elif isinstance(crop_paths, str) and crop_paths.strip():
        candidate_path = crop_paths

        if not os.path.isabs(candidate_path):
            candidate_path = os.path.join(
                dataset_dir,
                candidate_path,
            )

        if os.path.exists(candidate_path):
            image_path = candidate_path

    try:
        return generate_part_keywords_qwen(
            image_path=image_path,
            title=str(title),
            caption=caption,
        )

    except Exception as error:
        print(
            f"Keyword generation failed for title={title!r}: {error}"
        )
        return []


if "keywords" not in Impact_df.columns:
    Impact_df["keywords"] = None

for position, index in enumerate(
    tqdm(Impact_df.index, desc="Generating keywords"),
    start=1,
):
    row = Impact_df.loc[index]

    try:
        Impact_df.at[index, "keywords"] = generate_keywords_for_row(row)

    except Exception as error:
        print(
            f"\nKeyword generation failed at row {index}, "
            f"title={row.get('title', '')!r}: {error}"
        )
        Impact_df.at[index, "keywords"] = []

    # Save after every 50 processed rows
    if position % SAVE_EVERY == 0:
        Impact_df.to_csv(
            SAVE_PATH,
            index=False,
            encoding="utf-8",
        )

        print(
            f"\nCheckpoint saved after {position} rows "
            f"to: {SAVE_PATH}"
        )


# Final save for any remaining rows
Impact_df.to_csv(
    SAVE_PATH,
    index=False,
    encoding="utf-8",
)

print(f"\nFinal results saved to: {SAVE_PATH}")
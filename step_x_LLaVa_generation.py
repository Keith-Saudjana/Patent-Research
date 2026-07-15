import os
import ast
import sys
import torch
import pandas as pd
from PIL import Image
import copy
from tqdm.auto import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

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

# --- Parse keyword_available argument ---
if len(sys.argv) < 2:
    print("Usage: python script.py <keyword_available: true|false>")
    sys.exit(1)

arg = sys.argv[1].strip().lower()
if arg in ("true", "1", "yes"):
    keyword_available = True
elif arg in ("false", "0", "no"):
    keyword_available = False
else:
    print(f"Invalid value for keyword_available: '{sys.argv[1]}'. Use true or false.")
    sys.exit(1)

# --- Load model ---
pretrained = "lmms-lab/llama3-llava-next-8b"
model_name = "llava_llama3"
device = "cuda:0"

tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name,
    device_map=device,
    attn_implementation="eager"
)
model.tie_weights()
model.eval()

# --- Inference functions ---
def llava_describe_patent_with_components(
    model, tokenizer, image_path, title, keywords,
    image_processor, device="cuda", conv_template="llava_llama_3", max_new_tokens=512,
):
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",")]
    if not keywords:
        return ""

    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_img.to(dtype=torch.float16, device=device) for _img in image_tensor]
    image_sizes = [image.size]

    keyword_text = ", ".join(keywords)

    question = (
        DEFAULT_IMAGE_TOKEN
        + f"""
You are analyzing a design patent image of a {title}.

Step 1: Look carefully at the image and describe all visible parts as a technical drawing.
Be specific about shapes, contours, positions, proportions, orientation, symmetry, surface features,
openings, edges, joints, and any visible connection between parts. Only describe what can be seen
in this specific image.

Step 2: Describe the spatial arrangement of the object.
Explain where each major part is located relative to the others, such as upper/lower, left/right,
front/back, central/peripheral, inner/outer, above/below, attached to, surrounding, inserted into,
projecting from, aligned with, parallel to, or connected to another part. Include the overall layout
and whether the design appears symmetrical, elongated, compact, layered, hollow, enclosed, or modular.

Step 3: Describe the likely functional role of each visible component based only on its visual form
and placement. For example, explain whether a part appears to function as a handle, support, base,
cover, connector, container, opening, hinge, frame, control area, gripping area, display area, or
decorative surface. If the function is uncertain, state it cautiously as "appears to serve as" rather
than inventing unsupported information.

Step 4: You must describe every one of these components: {keyword_text}
- If you already described it in the earlier steps, use that observation.
- If you missed it, look again carefully before describing it.
- For each component, include its visible appearance, spatial position, relationship to nearby parts,
and likely functional role if inferable from the image.
- Do not write generic definitions. Every description must be grounded in what is visible in this
specific patent image.

Write your entire response as a single continuous paragraph. Naturally incorporate each component's
name, visual description, spatial relationship, and likely function into the flowing text without
bullet points, lists, or headers.
"""
    )

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)

    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        modalities=["image"] * input_ids.shape[0],
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.batch_decode(cont, skip_special_tokens=True)[0]


def llava_describe_patent(
    model, tokenizer, image_path, title,
    image_processor, device="cuda", conv_template="llava_llama_3", max_new_tokens=512,
):
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_img.to(dtype=torch.float16, device=device) for _img in image_tensor]
    image_sizes = [image.size]

    question = (
        DEFAULT_IMAGE_TOKEN
        + f"""
You are analyzing a design patent image of a {title}.

Look carefully at the image and write a detailed technical description of the object as a whole.
Describe its overall shape, structure, visible components, proportions, orientation, contours,
surface details, openings, edges, joints, and any notable visual characteristics.

Also describe the spatial arrangement of the object and its parts. Explain where the main regions
or components are located relative to one another, such as upper/lower, left/right, front/back,
central/peripheral, inner/outer, above/below, attached to, surrounding, inserted into, projecting from,
aligned with, parallel to, or connected to another part. Mention whether the design appears symmetrical,
elongated, compact, layered, hollow, enclosed, modular, flat, curved, or raised when this is visible.

Describe the likely functional role of visible components based only on their shape and placement.
For example, explain whether a part appears to serve as a handle, support, base, cover, connector,
container, opening, hinge, frame, control area, gripping area, display area, decorative surface,
or structural element. If the function is uncertain, use cautious wording such as "appears to serve as"
or "may function as" rather than inventing unsupported information.

Write your response as a single continuous paragraph grounded only in what you can actually see in this image.
Do not use bullet points, lists, or headers.
"""
    )

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)

    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        modalities=["image"] * input_ids.shape[0],
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.batch_decode(cont, skip_special_tokens=True)[0]


# --- Load data ---
df = pd.read_csv("model_io/Impact_2022_Sub_KW.csv", encoding="utf-8")
df = df.map(try_literal_eval)

# --- Run inference ---
if keyword_available:
    df["llava_output"] = df.progress_apply(
        lambda row: llava_describe_patent_with_components(
            model=model,
            tokenizer=tokenizer,
            image_path=row["file_names"][0],
            title=row["title"],
            keywords=row["components"],
            image_processor=image_processor,
            device=device,
        ),
        axis=1,
    )
    output_path = "model_io/keyword_method/Impact_2022_Sub_LLaVa_KW.csv"
else:
    df["llava_output"] = df.progress_apply(
        lambda row: llava_describe_patent(
            model=model,
            tokenizer=tokenizer,
            image_path=row["file_names"][0],
            title=row["title"],
            image_processor=image_processor,
            device=device,
        ),
        axis=1,
    )
    output_path = "model_io/no_keyword_method/Impact_2022_Sub_LLaVa_NoKW.csv"

df.to_csv(output_path, index=False, encoding="utf-8")
print(f"Saved to {output_path}")
import sys
import ast
import os
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

INPUT_CSV  = "/home/yishin/keith/patent_research/model_io/Impact_Sub_KW.csv"
IMAGE_COL  = "image_paths"
GRID_SIZE  = int(sys.argv[3]) if len(sys.argv) == 4 else 3

BASE_DIR   = os.path.dirname(os.path.abspath(INPUT_CSV))
OUTPUT_DIR = os.path.join(BASE_DIR, "impact_dataset_grid")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Highlight settings
DIM_OPACITY  = 180          # 0 = fully transparent, 255 = fully opaque black overlay
BORDER_COLOR = (255, 80, 80)  # red border around the highlighted cell
BORDER_WIDTH = 3

print(f"[Config] Input CSV  : {INPUT_CSV}")
print(f"[Config] Image col  : {IMAGE_COL}")
print(f"[Config] Grid size  : {GRID_SIZE}x{GRID_SIZE}")
print(f"[Config] Output dir : {OUTPUT_DIR}")

# --- Helpers ---
def try_literal_eval(x):
    if isinstance(x, str):
        x = x.strip()
        if x.startswith("[") or x.startswith("{") or x.startswith("("):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                pass
    return x

def to_path_list(val):
    if isinstance(val, list):
        return [str(p) for p in val]
    if isinstance(val, str):
        return [val]
    return []

def get_cell_box(img_w, img_h, n, target_row, target_col):
    """Return (left, upper, right, lower) pixel box for a grid cell."""
    cell_w = img_w // n
    cell_h = img_h // n
    left   = target_col * cell_w
    upper  = target_row * cell_h
    right  = left + cell_w
    lower  = upper + cell_h
    return left, upper, right, lower

def make_highlighted_image(img: Image.Image, n: int, target_row: int, target_col: int) -> Image.Image:
    """
    Return a copy of img where every cell EXCEPT (target_row, target_col)
    is dimmed with a semi-transparent black overlay, and the target cell
    has a colored border drawn around it.
    """
    w, h   = img.size
    result = img.copy().convert("RGBA")

    # --- Dim all non-target cells ---
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    for row in range(n):
        for col in range(n):
            if row == target_row and col == target_col:
                continue
            box = get_cell_box(w, h, n, row, col)
            draw.rectangle(box, fill=(0, 0, 0, DIM_OPACITY))

    result = Image.alpha_composite(result, overlay)

    # --- Draw border around the highlighted cell ---
    draw_result = ImageDraw.Draw(result)
    tbox = get_cell_box(w, h, n, target_row, target_col)
    for offset in range(BORDER_WIDTH):
        draw_result.rectangle(
            [tbox[0] + offset, tbox[1] + offset,
             tbox[2] - offset, tbox[3] - offset],
            outline=BORDER_COLOR + (255,)
        )

    return result.convert("RGB")

# --- Load data ---
df = pd.read_csv(INPUT_CSV, encoding="utf-8")
df[IMAGE_COL] = df[IMAGE_COL].apply(try_literal_eval)

print(f"Loaded {len(df)} rows.\n")

# --- Process ---
errors = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Rows"):
    paths = to_path_list(row[IMAGE_COL])
    if not paths:
        print(f"  [Warning] Row {idx}: no image paths found, skipping.")
        continue

    row_dir = os.path.join(OUTPUT_DIR, f"row_{idx:05d}")
    os.makedirs(row_dir, exist_ok=True)

    for img_path in paths:
        if not os.path.isfile(img_path):
            print(f"  [Warning] Row {idx}: file not found → {img_path}")
            errors.append({"row": idx, "path": img_path, "error": "file not found"})
            continue

        try:
            img  = Image.open(img_path).convert("RGB")
            stem = os.path.splitext(os.path.basename(img_path))[0]

            # Generate one highlighted image per cell
            for target_row in range(GRID_SIZE):
                for target_col in range(GRID_SIZE):
                    highlighted = make_highlighted_image(img, GRID_SIZE, target_row, target_col)
                    out_name    = f"{stem}_grid_r{target_row}c{target_col}.jpg"
                    out_path    = os.path.join(row_dir, out_name)
                    highlighted.save(out_path, format="JPEG", quality=95)

        except Exception as e:
            print(f"  [Error] Row {idx}, {img_path}: {e}")
            errors.append({"row": idx, "path": img_path, "error": str(e)})

# --- Summary ---
print(f"\n[Done] Highlighted grid images saved to: {OUTPUT_DIR}")
print(f"       {GRID_SIZE}x{GRID_SIZE} = {GRID_SIZE**2} images per source image")
print(f"       Rows processed : {len(df)}")
print(f"       Errors         : {len(errors)}")

if errors:
    error_log = os.path.join(OUTPUT_DIR, "errors.csv")
    pd.DataFrame(errors).to_csv(error_log, index=False)
    print(f"       Error log      : {error_log}")
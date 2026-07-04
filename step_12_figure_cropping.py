"""
step_12_figure_cropping.py
==========================
Streamlined pipeline for matching, cropping, and saving the best figures
from design-patent images, applied across a DataFrame.

Usage
-----
    python step_12_figure_cropping.py

Output
------
    figure_crops/
        {row_index}_{title}/
            FIG_1.jpg
            FIG_2.jpg
            ...

    Impact_Sub_Crops.csv   ← original df + 'crop_paths' column
"""

from __future__ import annotations

import ast
import math
import os
import re
import cv2
import numpy as np
import pytesseract
import pandas as pd

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CSV_IN      = "/home/yishin/keith/patent_research/model_io/Impact_Sub.csv"
CSV_OUT     = "/home/yishin/keith/patent_research/model_io/Impact_Sub_Crops.csv"
OUTPUT_ROOT = "figure_crops"
WORKERS     = 4          # parallel rows processed simultaneously; tune to CPU count

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FIG_LABEL = re.compile(
    r'\bF[I1lL|]G[S]?[.\s_\-·,]?\s*(\d{1,3}\s*[A-Za-z]?)\b',
    re.IGNORECASE,
)
_CONFIGS = ["--oem 3 --psm 11", "--oem 3 --psm 6", "--oem 3 --psm 3"]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


def _sanitize(name: str) -> str:
    """Strip filesystem-unsafe characters."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip()


def _fig_sort_key(lbl: str) -> int:
    m = re.search(r'(\d+)', lbl)
    return int(m.group(1)) if m else 0

# ---------------------------------------------------------------------------
# OCR preprocessing
# ---------------------------------------------------------------------------

def preprocess_for_ocr(image: Image.Image, min_height: int, dilate: bool = True) -> Image.Image:
    image = image.convert("L")
    image = ImageOps.autocontrast(image)

    w, h = image.size
    scale = max(1.0, min_height / h)
    if scale > 1.0:
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    arr = np.array(image)
    arr = cv2.fastNlMeansDenoising(arr, h=15)
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if dilate:
        kernel = np.ones((2, 2), np.uint8)
        arr = cv2.dilate(arr, kernel, iterations=1)

    arr = cv2.copyMakeBorder(arr, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    return Image.fromarray(arr)

# ---------------------------------------------------------------------------
# Rotation helper
# ---------------------------------------------------------------------------

def rotate_cv2(img: np.ndarray, angle: int) -> np.ndarray:
    """Rotate an OpenCV array counter-clockwise by angle degrees."""
    if angle == 0:
        return img
    pil = Image.fromarray(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil = pil.rotate(angle, expand=True)
    arr = np.array(pil)
    return arr if img.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

# ---------------------------------------------------------------------------
# FIG label extraction
# ---------------------------------------------------------------------------

def extract_fig_labels(
    img_path: str,
    min_height: int = 800,
) -> tuple[list[tuple[str, int]], int]:
    """
    Returns (matched_labels, total_fig_count).

    Tries rotations [0, 90, 270] and multiple PSM configs, short-circuiting
    as soon as any combination yields matches.
    """
    pil_img = Image.open(img_path)

    for rotation in [0, 90, 270]:
        rotated   = pil_img.rotate(rotation, expand=True) if rotation else pil_img
        processed = preprocess_for_ocr(rotated, min_height=min_height)

        for cfg in _CONFIGS:
            raw_text   = pytesseract.image_to_string(processed, config=cfg)
            normalized = " ".join(raw_text.split())

            all_keys_seen: dict[str, tuple[str, int]] = {}
            for m in _FIG_LABEL.finditer(normalized):
                num = re.sub(r'\s+', '', m.group(1)).upper()
                key = f"FIG. {num}"
                if key not in all_keys_seen:
                    all_keys_seen[key] = (key, rotation)

            if all_keys_seen:
                return list(all_keys_seen.values()), len(all_keys_seen)

    return [], 0

# ---------------------------------------------------------------------------
# OCR matching: image paths → fig dict
# ---------------------------------------------------------------------------

def match_figs_by_ocr(
    image_paths,
    best_fig_desc,
    verbose: bool = False,
) -> dict:
    """
    Returns dict mapping FIG key → {'path', 'rotation', 'fig_count'}.
    """
    if isinstance(image_paths, str):
        image_paths = ast.literal_eval(image_paths)
    if isinstance(best_fig_desc, str):
        best_fig_desc = ast.literal_eval(best_fig_desc)

    target_keys = set(best_fig_desc.keys())
    result: dict = {}

    for path in image_paths:
        if not Path(path).exists():
            if verbose:
                print(f"  [SKIP] missing: {path}")
            continue

        try:
            found_labels, total_fig_count = extract_fig_labels(path)
        except Exception as e:
            if verbose:
                print(f"  [ERROR] {path}: {e}")
            continue

        for (label, rotation) in found_labels:
            if label in target_keys and label not in result:
                result[label] = {
                    "path":      path,
                    "rotation":  rotation,
                    "fig_count": total_fig_count,
                }
                if verbose:
                    print(f"  [MATCH] {label} → {Path(path).name} "
                          f"(rotation={rotation}°, page_figs={total_fig_count})")

        if result.keys() == target_keys:
            break

    return result

# ---------------------------------------------------------------------------
# Bounding box detection + merging
# ---------------------------------------------------------------------------

def detect_and_merge_boxes(
    img_bgr:   np.ndarray,
    fig_count: int,
    min_area:  int  = 500,
    padding:   int  = 25,
    merge_gap: int  = 35,
    invert:    bool = True,
) -> list[tuple[int, int, int, int]]:
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    binary  = cv2.threshold(blurred, 0, 255, thresh_type + cv2.THRESH_OTSU)[1]
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (merge_gap, merge_gap))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = img_bgr.shape[:2]
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        x1 = max(x - padding, 0);  y1 = max(y - padding, 0)
        x2 = min(x + w + padding, w_img);  y2 = min(y + h + padding, h_img)
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    boxes.sort(key=lambda b: (b[1], b[0]))

    if len(boxes) > fig_count:
        boxes = merge_figure_boxes(boxes, box_count=fig_count)
        boxes.sort(key=lambda b: (b[1], b[0]))

    return boxes


def merge_figure_boxes(
    boxes: list[tuple[int, int, int, int]],
    box_count: int,
    mode: str = "edge",
) -> list[tuple[int, int, int, int]]:
    if box_count <= 0:
        raise ValueError("box_count must be a positive integer.")
    if len(boxes) <= box_count:
        return list(boxes)

    def _edge_dist(a, b):
        dx = max(0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
        dy = max(0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
        return math.hypot(dx, dy)

    def _center_dist(a, b):
        return math.hypot(
            (a[0] + a[2] / 2) - (b[0] + b[2] / 2),
            (a[1] + a[3] / 2) - (b[1] + b[3] / 2),
        )

    def _union(a, b):
        x  = min(a[0], b[0]);  y  = min(a[1], b[1])
        x2 = max(a[0] + a[2], b[0] + b[2])
        y2 = max(a[1] + a[3], b[1] + b[3])
        return (x, y, x2 - x, y2 - y)

    dist_fn = _edge_dist if mode == "edge" else _center_dist
    pool    = list(boxes)

    while len(pool) > box_count:
        best_i, best_j, best_d = 0, 1, float("inf")
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                d = dist_fn(pool[i], pool[j])
                if d < best_d:
                    best_d, best_i, best_j = d, i, j
        merged = _union(pool[best_i], pool[best_j])
        pool   = [b for k, b in enumerate(pool) if k not in (best_i, best_j)]
        pool.append(merged)

    pool.sort(key=lambda b: (b[1], b[0]))
    return pool

# ---------------------------------------------------------------------------
# Per-crop OCR label detection
# ---------------------------------------------------------------------------

def _detect_fig_num_in_crop(crop_bgr: np.ndarray, min_height: int = 800) -> Optional[str]:
    pil       = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    processed = preprocess_for_ocr(pil, min_height=min_height)

    for cfg in ["--oem 3 --psm 11", "--oem 3 --psm 6", "--oem 3 --psm 3"]:
        text = pytesseract.image_to_string(processed, config=cfg)
        m    = _FIG_LABEL.search(" ".join(text.split()))
        if m:
            num = re.sub(r'\s+', '', m.group(1)).upper()
            return f"FIG. {num}"
    return None

# ---------------------------------------------------------------------------
# Crop assignment + saving
# ---------------------------------------------------------------------------

def _save_crop(crop_bgr: np.ndarray, out_dir: Path, fig_label: str) -> Path:
    filename = _sanitize(fig_label.replace(". ", "_").replace(" ", "_")) + ".jpg"
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), crop_bgr)
    return out_path


def assign_crops_to_figs(
    fig_dict:    dict,
    title:       str,
    row_index:   int,
    output_root: str  = OUTPUT_ROOT,
    min_area:    int  = 500,
    padding:     int  = 25,
    merge_gap:   int  = 35,
    invert:      bool = True,
    min_height:  int  = 800,
    verbose:     bool = False,
) -> dict:
    """
    Detect, crop, and save each figure in fig_dict.

    Crops are saved to:
        {output_root}/{row_index}_{title}/FIG_N.jpg

    Returns the fig_dict with 'box', 'crop', and 'saved_path' added per entry.
    """
    folder_name = f"{row_index}_{_sanitize(title)}"
    out_dir     = Path(output_root) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"  → saving to: {out_dir}")

    result = {
        fig: {**meta, "box": None, "crop": None, "saved_path": None}
        for fig, meta in fig_dict.items()
    }

    # Group FIGs that share the same source page.
    page_groups: dict[tuple, list[str]] = defaultdict(list)
    for fig_label, meta in fig_dict.items():
        page_key = (meta["path"], meta["rotation"], meta["fig_count"])
        page_groups[page_key].append(fig_label)

    for (path, rotation, fig_count), fig_labels in page_groups.items():

        if not Path(path).exists():
            if verbose:
                print(f"    [SKIP] missing: {path}")
            continue

        raw = cv2.imread(path)
        if raw is None:
            if verbose:
                print(f"    [ERROR] could not read: {path}")
            continue

        img = rotate_cv2(raw, rotation)

        # --- Fast path: single figure on page ---
        if fig_count == 1:
            h_img, w_img  = img.shape[:2]
            fig_label     = fig_labels[0]
            saved         = _save_crop(img, out_dir, fig_label)

            result[fig_label]["box"]        = (0, 0, w_img, h_img)
            result[fig_label]["crop"]       = img
            result[fig_label]["saved_path"] = str(saved)

            if verbose:
                print(f"    [1-fig page] {fig_label} → {saved.name}")
            continue

        # --- Multi-figure page: detect → merge → identify ---
        boxes = detect_and_merge_boxes(img, fig_count, min_area, padding, merge_gap, invert)
        crops = [img[y: y + h, x: x + w] for x, y, w, h in boxes]

        if verbose:
            print(f"    [{Path(path).name}] rotation={rotation}° "
                  f"target={fig_count} boxes={len(boxes)}")

        # OCR each crop to identify which FIG it contains.
        ocr_map: dict[str, int] = {}
        for i, crop in enumerate(crops):
            label = _detect_fig_num_in_crop(crop, min_height=min_height)
            if label and label not in ocr_map:
                ocr_map[label] = i
                if verbose:
                    print(f"      crop[{i}] → OCR: {label}")

        # Positional fallback for any unidentified crops.
        unmatched = [l for l in sorted(fig_labels, key=_fig_sort_key) if l not in ocr_map]
        unused    = [i for i in range(len(crops)) if i not in ocr_map.values()]

        for label, idx in zip(unmatched, unused):
            ocr_map[label] = idx
            if verbose:
                print(f"      crop[{idx}] → positional fallback: {label}")

        for fig_label in fig_labels:
            if fig_label not in ocr_map:
                if verbose:
                    print(f"      [UNMATCHED] {fig_label}")
                continue

            idx         = ocr_map[fig_label]
            x, y, w, h = boxes[idx]
            crop        = crops[idx]
            saved       = _save_crop(crop, out_dir, fig_label)

            result[fig_label]["box"]        = (x, y, w, h)
            result[fig_label]["crop"]       = crop
            result[fig_label]["saved_path"] = str(saved)

            if verbose:
                print(f"      {fig_label} → {saved.name}")

    return result

# ---------------------------------------------------------------------------
# Per-row worker (runs in a subprocess for parallelism)
# ---------------------------------------------------------------------------

def _process_row(args: tuple) -> tuple[int, list[str]]:
    """
    Worker function executed in a separate process.

    Parameters
    ----------
    args : (row_index, title, image_paths, best_fig_desc, output_root)

    Returns
    -------
    (row_index, saved_paths)
        saved_paths is a flat list of all crop file paths for this row,
        or an empty list if nothing was matched/cropped.
    """
    row_index, title, image_paths, best_fig_desc, output_root = args

    try:
        fig_dict = match_figs_by_ocr(image_paths, best_fig_desc)

        if not fig_dict:
            return row_index, []

        result = assign_crops_to_figs(
            fig_dict    = fig_dict,
            title       = title,
            row_index   = row_index,
            output_root = output_root,
        )

        saved = [
            data["saved_path"]
            for data in result.values()
            if data["saved_path"] is not None
        ]
        return row_index, saved

    except Exception as e:
        print(f"  [ROW {row_index} ERROR] {e}")
        return row_index, []

# ---------------------------------------------------------------------------
# DataFrame pipeline
# ---------------------------------------------------------------------------

def process_dataframe(
    df:          pd.DataFrame,
    output_root: str = OUTPUT_ROOT,
    workers:     int = WORKERS,
    title_col:   str = "title",
    paths_col:   str = "image_paths",
    fig_col:     str = "best_fig_desc",
) -> pd.DataFrame:
    """
    Apply the full figure-cropping pipeline to every row of df in parallel.

    Adds a 'crop_paths' column containing the list of saved crop file paths
    for each row.

    Parameters
    ----------
    df : pd.DataFrame
    output_root : str
        Root folder for figure_crops subfolders.
    workers : int
        Number of parallel processes.  Set to 1 to disable parallelism
        (useful for debugging).
    """
    # Build the argument list for each row.
    args_list = [
        (
            idx,
            str(row[title_col]),
            row[paths_col] if isinstance(row[paths_col], list) else try_literal_eval(row[paths_col]),
            row[fig_col]   if isinstance(row[fig_col],   dict) else try_literal_eval(row[fig_col]),
            output_root,
        )
        for idx, row in df.iterrows()
    ]

    crop_paths: dict[int, list[str]] = {idx: [] for idx in df.index}

    if workers == 1:
        # Single-process path — easier to debug.
        for args in tqdm(args_list, desc="Cropping figures"):
            row_index, saved = _process_row(args)
            crop_paths[row_index] = saved
    else:
        # Parallel path — each row runs in its own subprocess.
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_row, args): args[0] for args in args_list}

            with tqdm(total=len(futures), desc="Cropping figures") as pbar:
                for future in as_completed(futures):
                    row_index, saved = future.result()
                    crop_paths[row_index] = saved
                    pbar.update(1)
                    pbar.set_postfix(row=row_index, crops=len(saved))

    df = df.copy()
    df["crop_paths"] = pd.Series(crop_paths)
    return df

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading {CSV_IN} ...")
    df = pd.read_csv(CSV_IN, encoding="utf-8").map(try_literal_eval)
    print(f"  {len(df)} rows loaded.\n")

    df = process_dataframe(df, output_root=OUTPUT_ROOT, workers=WORKERS)

    df.to_csv(CSV_OUT, index=False, encoding="utf-8")
    print(f"\nSaved → {CSV_OUT}")

    total_crops = df["crop_paths"].apply(len).sum()
    print(f"Total crops saved: {total_crops}")
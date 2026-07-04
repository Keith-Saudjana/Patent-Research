import json

import pandas as pd
import numpy as np
import cv2
import ast
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import pipeline

from typing import Union, List

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

dataset_dir = "/home/yishin/keith/patent_research/model_io"
Impact_df = pd.read_csv(os.path.join(dataset_dir, "3_Impact_Sub_Complete.csv"), encoding="utf-8")
Impact_df = Impact_df.map(try_literal_eval)

mask_generator = pipeline(
    "mask-generation",
    model="facebook/sam-vit-huge",
    device="cuda"
)

def get_sam_cells(image_path, points_per_side=32, min_mask_area=2000, max_mask_area_pct=0.7):
    img = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    h, w = image_rgb.shape[:2]
    total_area = h * w

    outputs = mask_generator(pil_image, points_per_side=points_per_side)

    grid_cells = []
    for i, mask in enumerate(outputs["masks"]):
        mask_np = np.array(mask)
        area = mask_np.sum()

        if area < min_mask_area:
            continue
        if area > total_area * max_mask_area_pct: 
            continue

        ys, xs = np.where(mask_np > 0)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        grid_cells.append({
            "id": len(grid_cells),
            "row": 0,
            "col": len(grid_cells),
            "bbox": (x1, y1, x2, y2),
            "mask": mask_np,
            "confidence_score": 1.0
        })

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 150)

    return image_rgb, grid_cells, edges

def deduplicate_masks(grid_cells, iou_threshold=0.7):
    def compute_iou(a, b):
        ax1, ay1, ax2, ay2 = a["bbox"]
        bx1, by1, bx2, by2 = b["bbox"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)

    kept = []
    for cell in sorted(grid_cells, key=lambda c: c["mask"].sum(), reverse=True):
        if not any(compute_iou(cell, k) > iou_threshold for k in kept):
            kept.append(cell)
    return kept

def draw_grid_cells(image_rgb, grid_cells):
    img = Image.fromarray(image_rgb)
    output = img.copy()
    draw = ImageDraw.Draw(output)

    for cell in grid_cells:
        x1, y1, x2, y2 = cell["bbox"]

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=(255, 0, 0),
            width=3
        )

    return output

def get_fig_id(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]

def make_json_safe_region(cell):
    """
    Keep bbox and metadata, but do not store full mask array in the DataFrame.
    Store mask area instead.
    """
    return {
        "id": int(cell["id"]),
        "bbox": tuple(map(int, cell["bbox"])),
        "area": int(cell["mask"].sum()) if "mask" in cell else None
    }

def generate_sam_regions_for_crop_paths(crop_paths):
    fig_regions = {}

    for img_path in crop_paths:
        fig_id = get_fig_id(img_path)

        image_rgb, grid_cells, edges = get_sam_cells(img_path)

        grid_cells = deduplicate_masks(grid_cells)

        fig_regions[fig_id] = [
            make_json_safe_region(cell)
            for cell in grid_cells
        ]

    return fig_regions

tqdm.pandas()

Impact_df["bbox"] = Impact_df["crop_paths"].progress_apply(
    generate_sam_regions_for_crop_paths
)

save_path = os.path.join(dataset_dir, "4_Impact_Sub_Regions.csv")

Impact_df.to_csv(
    save_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"Saved to: {save_path}")
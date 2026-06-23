"""
figure_cropping.py
==================
Pipeline for cropping individual figures from design-patent images,
driven by a pandas DataFrame.

Main entry point
----------------
    results = process_dataframe(df)

This iterates over every row, reads image paths from the ``image_paths``
column, filters by OCR-detected FIG labels, crops individual figures, and
saves them under::

    image_crops/
        {title}/
            {source_image_stem}_01.jpg
            {source_image_stem}_02.jpg
            ...

Return value
------------
    {
        "Patent Title A": {
            "USD001-D00001": {
                "fig_count":   7,
                "boxes":       [(x, y, w, h), ...],
                "saved_crops": ["image_crops/Patent Title A/USD001-D00001_01.jpg", ...],
            },
            ...
        },
        ...
    }

Optional post-processing
------------------------
    merged_boxes = merge_figure_boxes(boxes, box_count=3)

Visualisation helpers (require matplotlib)
------------------------------------------
    show_images_vertical(image_paths)
    show_boxes_on_images(results)          # expects crop_patent_figures() output
    visualize_merge(image_path, boxes, box_count=3)
"""

from __future__ import annotations

import ast
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import pandas as pd
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COLORS: List[Tuple[int, int, int]] = [
    (230,  90,  50),
    ( 50, 170,  90),
    ( 80, 120, 220),
    (190,  60, 160),
    (200, 160,  30),
    ( 40, 180, 180),
    (220,  60, 100),
    (100,  80, 220),
]

_FIG_PATTERN = re.compile(
    r"\bF[I1L]G(?:[\.\s]*\d+)?\b",
    re.IGNORECASE,
)

_FIG_NUM_PATTERN = re.compile(
    r"\bF[I1L]G[\.\s]*(\d+)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _sanitize_folder_name(name: str) -> str:
    """Replace filesystem-unsafe characters with underscores."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip()


def _preprocess_for_ocr(
    image: Image.Image,
    min_height: int,
    threshold: int = 180,
    dilate: bool = True,
) -> Image.Image:
    """
    Preprocess patent figure images for Tesseract OCR.
    Useful for thin lines, weak labels, and vertical FIG labels.
    """
    image = image.convert("L")
    image = ImageOps.autocontrast(image)

    w, h = image.size
    scale = max(1.0, min_height / h)

    if scale > 1.0:
        image = image.resize(
            (int(w * scale), int(h * scale)),
            Image.Resampling.LANCZOS,
        )

    arr = np.array(image)
    arr = np.where(arr < threshold, 0, 255).astype(np.uint8)

    if dilate:
        kernel = np.ones((2, 2), np.uint8)
        arr = cv2.dilate(arr, kernel, iterations=1)

    return Image.fromarray(arr)


def count_fig_labels(
    image_path: str,
    min_height: int,
    psm_modes: Tuple[int, ...] = (6, 11, 3, 12, 5),
    rotation_angles: Tuple[int, ...] = (0, 90, 180, 270),
) -> Tuple[int, int]:
    """
    Detect FIG labels using Tesseract.

    Returns
    -------
    best_count : int
        Highest number of FIG-like labels detected.

    best_rotation : int
        Rotation angle where the best count was detected.
    """
    try:
        with Image.open(image_path) as img:
            source = img.convert("RGB").copy()
    except Exception as err:
        tqdm.write(f"[OCR Error] {image_path}: {err}")
        return 0, 0

    best_count = 0
    best_rotation = 0

    tess_config_base = (
        "-c tessedit_char_whitelist=FIGfig.0123456789 "
        "-c preserve_interword_spaces=1"
    )

    for angle in rotation_angles:
        rotated = source.rotate(angle, expand=True) if angle != 0 else source

        w, h = rotated.size

        candidates = [
            rotated,
            rotated.crop((0, 0, int(w * 0.25), h)),
            rotated.crop((0, int(h * 0.75), w, h)),
        ]

        for candidate in candidates:
            processed = _preprocess_for_ocr(
                candidate,
                min_height=min_height,
                threshold=190,
                dilate=True,
            )

            for psm in psm_modes:
                try:
                    text = pytesseract.image_to_string(
                        processed,
                        config=f"--psm {psm} {tess_config_base}",
                    )
                except Exception as err:
                    tqdm.write(
                        f"[Tesseract Error] {image_path} angle={angle} psm={psm}: {err}"
                    )
                    continue

                normalized = " ".join(text.split())
                count = len(_FIG_PATTERN.findall(normalized))

                if count > best_count:
                    best_count = count
                    best_rotation = angle

    return best_count, best_rotation


def _unique_output_path(output_dir: str, stem: str) -> str:
    """Return a non-colliding .jpg path inside output_dir."""
    candidate = os.path.join(output_dir, f"{stem}.jpg")
    if not os.path.exists(candidate):
        return candidate
    counter = 1
    while True:
        candidate = os.path.join(output_dir, f"{stem}_{counter}.jpg")
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def _filter_images_with_fig(
    image_paths: List[str],
    min_height: int,
    psm_modes: Tuple[int, ...],
    rotation_angles: Tuple[int, ...],
    show_progress: bool,
) -> Dict[str, Dict]:
    """
    Detect FIG labels and keep rotated images in memory instead of saving
    temporary JPEGs.

    Returns
    -------
    retained:
        {
            original_image_path: {
                "image_rgb": np.ndarray,
                "fig_count": int,
                "rotation": int,
                "original_path": str,
            }
        }
    """
    retained: Dict[str, Dict] = {}

    iterator = tqdm(
        image_paths,
        desc="  Fig label detection",
        disable=not show_progress,
    )

    for image_path in iterator:
        image_path = str(image_path)

        if not os.path.isfile(image_path):
            tqdm.write(f"  [Warning] File not found: {image_path}")
            continue

        count, rotation = count_fig_labels(
            image_path=image_path,
            min_height=min_height,
            psm_modes=psm_modes,
            rotation_angles=rotation_angles,
        )

        if count == 0:
            continue

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")

                if rotation != 0:
                    image = image.rotate(rotation, expand=True)

                image_rgb = np.array(image)

            retained[image_path] = {
                "image_rgb": image_rgb,
                "fig_count": count,
                "rotation": rotation,
                "original_path": image_path,
            }

        except Exception as err:
            tqdm.write(f"  [Image Load Error] {image_path}: {err}")

    return retained


def _crop_by_whitespace(
    retained_images: Dict[str, Dict],
    min_area: int,
    padding: int,
    merge_gap: int,
    invert: bool,
    show_progress: bool,
) -> Dict[str, Dict]:
    """
    Detect and crop figures directly from in-memory rotated RGB images.
    """
    results: Dict[str, Dict] = {}

    iterator = tqdm(
        retained_images.items(),
        desc="  Whitespace division",
        disable=not show_progress,
    )

    for image_path, meta in iterator:
        img_rgb = meta["image_rgb"]

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

        binary = cv2.threshold(
            blurred,
            0,
            255,
            thresh_type + cv2.THRESH_OTSU,
        )[1]

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (merge_gap, merge_gap),
        )

        dilated = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        h_img, w_img = img_rgb.shape[:2]
        boxes: List[Tuple[int, int, int, int]] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w * h < min_area:
                continue

            x1 = max(x - padding, 0)
            y1 = max(y - padding, 0)
            x2 = min(x + w + padding, w_img)
            y2 = min(y + h + padding, h_img)

            boxes.append((x1, y1, x2 - x1, y2 - y1))

        boxes.sort(key=lambda b: (b[1], b[0]))

        crops = [
            img_rgb[y:y + h, x:x + w]
            for x, y, w, h in boxes
        ]

        results[image_path] = {
            "image_rgb": img_rgb,
            "fig_count": meta["fig_count"],
            "rotation": meta["rotation"],
            "original_path": meta["original_path"],
            "boxes": boxes,
            "crops": crops,
        }

    return results

def _apply_box_merge(
    image_rgb: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    fig_count: int,
) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
    """
    Merge boxes down to fig_count and re-extract crops directly from
    the in-memory rotated RGB image.
    """
    if fig_count <= 0 or len(boxes) <= fig_count:
        return boxes, []

    merged_boxes = merge_figure_boxes(
        boxes,
        box_count=fig_count,
    )

    crops = [
        image_rgb[y:y + h, x:x + w]
        for x, y, w, h in merged_boxes
    ]

    return merged_boxes, crops


def detect_fig_label(
    crop_rgb: np.ndarray,
    min_height: int = 1000,
    psm_modes: Tuple[int, ...] = (6, 11, 3, 12, 5),
) -> Optional[str]:
    """
    Detect FIG number from one cropped image using Tesseract.

    Returns
    -------
    str or None
        Example: "Fig_1", "Fig_2", or None.
    """
    pil_img = Image.fromarray(crop_rgb)

    tess_config_base = (
        "-c tessedit_char_whitelist=FIGfig.0123456789 "
        "-c preserve_interword_spaces=1"
    )

    for angle in (0, 90, 180, 270):
        rotated = pil_img.rotate(angle, expand=True) if angle != 0 else pil_img

        processed = _preprocess_for_ocr(
            rotated,
            min_height=min_height,
            threshold=190,
            dilate=True,
        )

        for psm in psm_modes:
            try:
                text = pytesseract.image_to_string(
                    processed,
                    config=f"--psm {psm} {tess_config_base}",
                )
            except Exception:
                continue

            normalized = " ".join(text.split())
            match = _FIG_NUM_PATTERN.search(normalized)

            if match:
                return f"Fig_{match.group(1)}"

    return None


def _save_crops(
    crop_arrays: List[np.ndarray],
    source_stem: str,
    output_dir: str,
    jpeg_quality: int,
    detect_labels: bool = True,
) -> List[str]:
    """
    Save a list of RGB crop arrays to output_dir as JPEGs.

    When ``detect_labels=True``, OCR is run on each crop to detect its FIG
    label; the file is saved as ``Fig_1.jpg``, ``Fig_2.jpg``, etc.
    If detection fails or a label collision occurs, the fallback name
    ``{source_stem}_{idx:02d}.jpg`` is used instead.

    Returns the list of saved file paths.
    """
    saved:      List[str] = []
    used_names: set       = set()

    for idx, crop in enumerate(crop_arrays, start=1):
        label = detect_fig_label(crop) if detect_labels else None
        filename = f"{label}.jpg" if (label and label not in used_names) \
                   else f"{source_stem}_{idx:02d}.jpg"
        used_names.add(filename)

        out_path = os.path.join(output_dir, filename)
        Image.fromarray(crop).save(out_path, format="JPEG", quality=jpeg_quality)
        saved.append(out_path)

    return saved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_dataframe(
    df,
    output_root: str = "image_crops",
    title_col: str = "title",
    image_paths_col: str = "image_paths",
    # --- OCR filtering ---
    min_height: int = 2000,
    psm_modes: Tuple[int, ...] = (6, 11, 3, 12),
    rotation_angles: Tuple[int, ...] = (0, 90, 180, 270),
    jpeg_quality: int = 95,
    # --- Whitespace cropping ---
    min_area: int = 500,
    padding: int = 25,
    merge_gap: int = 35,
    invert: bool = True,
    # --- General ---
    merge_to_fig_count: bool = True,
    detect_labels: bool = True,
    show_progress: bool = True,
) -> Dict[str, Dict]:
    """
    Run the full figure-cropping pipeline over every row of a DataFrame.

    For each row the function:

    1. Reads image paths from ``image_paths_col`` (accepts both a Python list
       and a stringified list as stored by pandas after CSV round-tripping).
    2. Filters images to those containing at least one FIG-like label (OCR).
    3. Crops individual figures from each retained image using whitespace
       separation.
    4. Saves crops as ``{source_stem}_01.jpg``, ``{source_stem}_02.jpg``, …
       inside ``{output_root}/{title}/``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Must contain ``title_col`` and ``image_paths_col``.

    output_root : str
        Root directory for all output. Defaults to ``"image_crops"``.

    title_col : str
        Column whose value is used as the per-row subfolder name.

    image_paths_col : str
        Column containing a list of image file paths for each row.

    min_height : int
        Minimum pixel height for OCR upscaling.

    psm_modes : tuple[int, ...]
        Tesseract page-segmentation modes to try. The highest FIG count
        across modes is used (counts are *not* summed).

    jpeg_quality : int
        JPEG quality (1–95) for saved crop files.

    min_area : int
        Minimum bounding-box area (px²) to keep a detected region.

    padding : int
        Extra pixels added on every side of each bounding box.

    merge_gap : int
        Dilation kernel size used to merge nearby ink contours before
        finding bounding boxes. Increase to merge split figures; decrease
        to separate merged ones.

    invert : bool
        True for dark ink on a white background (typical patent sketches).

    merge_to_fig_count : bool
        If True (default), merge the whitespace-detected bounding boxes down
        to the number of FIG labels found by OCR before saving. This corrects
        cases where whitespace detection over-splits a single figure into
        multiple boxes. Set to False to keep the raw detected boxes.

    detect_labels : bool
        If True (default), OCR is run on each crop after merging to detect
        its FIG label (e.g. "FIG. 3"). Crops are then saved as ``Fig_3.jpg``
        instead of a numbered fallback. When detection fails or produces a
        duplicate, the fallback name ``{source_stem}_{idx:02d}.jpg`` is used.

    show_progress : bool
        Show a top-level tqdm progress bar over rows.

    Returns
    -------
    dict
        Nested dict keyed first by title, then by source image stem::

            {
                "Patent Title A": {
                    "USD001-D00001": {
                        "fig_count":   7,
                        "boxes":       [(x, y, w, h), ...],
                        "saved_crops": [
                            "image_crops/Patent Title A/USD001-D00001_01.jpg",
                            "image_crops/Patent Title A/USD001-D00001_02.jpg",
                        ],
                    },
                },
            }

    Examples
    --------
    >>> results = process_dataframe(df)
    >>> for title, images in results.items():
    ...     total = sum(len(v["saved_crops"]) for v in images.values())
    ...     print(f"{title}: {total} crops saved")
    """
    results_by_title: Dict[str, Dict] = {}
    paths_by_index:   Dict[int, List[str]] = {}

    rows = list(df.iterrows())

    # Count total images up-front so the image bar has a known total.
    def _path_list(val):
        return val if isinstance(val, list) else (try_literal_eval(val) or [])

    total_images = sum(len(_path_list(row[image_paths_col])) for _, row in rows)

    row_bar = tqdm(total=len(rows),    desc="Patents", unit="patent",
                   position=0, leave=True,  dynamic_ncols=True)

    for idx, row in rows:
        title       = str(row[title_col])
        image_paths = row[image_paths_col]

        paths_by_index[idx] = []

        row_bar.set_postfix(patent=title[:45], refresh=False)

        if not isinstance(image_paths, list):
            image_paths = try_literal_eval(image_paths)
        if not image_paths:
            tqdm.write(f"[Skip] No image paths for: {title!r}")
            results_by_title[title] = {}
            row_bar.update(1)
            continue

        image_paths = [str(p) for p in image_paths]

        safe_title = _sanitize_folder_name(title)
        title_dir  = os.path.join(output_root, f"{idx}_{safe_title}")
        os.makedirs(title_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_filtered:

            # Step 1 — FIG label detection, rotation detection, and in-memory rotation
            retained_images = _filter_images_with_fig(
                image_paths=image_paths,
                min_height=min_height,
                psm_modes=psm_modes,
                rotation_angles=rotation_angles,
                show_progress=False,
            )

            if not retained_images:
                tqdm.write(f"  [Skip] No FIG images found for: {title!r}")
                results_by_title[title] = {}
                row_bar.update(1)
                continue

            # Step 2 — Whitespace crop directly from memory
            crop_results = _crop_by_whitespace(
                retained_images=retained_images,
                min_area=min_area,
                padding=padding,
                merge_gap=merge_gap,
                invert=invert,
                show_progress=False,
            )

            # Step 3 — Optionally merge boxes
            if merge_to_fig_count:
                for image_path, data in crop_results.items():
                    fig_count = data["fig_count"]

                    if fig_count > 0 and len(data["boxes"]) > fig_count:
                        merged_boxes, merged_crops = _apply_box_merge(
                            image_rgb=data["image_rgb"],
                            boxes=data["boxes"],
                            fig_count=fig_count,
                        )

                        if merged_crops:
                            data["boxes"] = merged_boxes
                            data["crops"] = merged_crops

            # Step 4 — Save final crops and collect metadata
            row_summary: Dict[str, Dict] = {}

            for image_path, data in crop_results.items():
                base_stem = os.path.splitext(os.path.basename(image_path))[0]
                fig_count = data["fig_count"]

                saved_paths = _save_crops(
                    crop_arrays=data["crops"],
                    source_stem=base_stem,
                    output_dir=title_dir,
                    jpeg_quality=jpeg_quality,
                    detect_labels=detect_labels,
                )

                row_summary[base_stem] = {
                    "fig_count": fig_count,
                    "rotation": data.get("rotation", 0),
                    "original_path": data.get("original_path"),
                    "boxes": data["boxes"],
                    "saved_crops": saved_paths,
                }

                tqdm.write(f"  ✓ {base_stem}: {len(saved_paths)} crop(s)")

        results_by_title[title] = row_summary

        # Flatten all saved crop paths for this row into the column value.
        paths_by_index[idx] = [
            p for v in row_summary.values() for p in v["saved_crops"]
        ]

        row_bar.update(1)

    row_bar.close()

    df["cropped_paths"] = pd.Series(paths_by_index)

    return results_by_title


def merge_figure_boxes(
    boxes: List[Tuple[int, int, int, int]],
    box_count: int,
    mode: str = "edge",
) -> List[Tuple[int, int, int, int]]:
    """
    Greedily merge bounding boxes until exactly *box_count* remain.

    At each step the two closest boxes are merged into their union rectangle.

    Parameters
    ----------
    boxes : list[(x, y, w, h)]
        Bounding boxes to merge.

    box_count : int
        Target number of boxes after merging.

    mode : {"edge", "center"}
        Distance metric for selecting the closest pair.

        * ``"edge"``   — gap between nearest edges (0 when boxes overlap).
        * ``"center"`` — Euclidean distance between centroids.

    Returns
    -------
    list[(x, y, w, h)]
        Merged bounding boxes (order is not guaranteed).
    """
    if box_count <= 0:
        raise ValueError("box_count must be a positive integer.")
    if len(boxes) <= box_count:
        return list(boxes)

    def _edge_dist(a: tuple, b: tuple) -> float:
        dx = max(0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
        dy = max(0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
        return math.hypot(dx, dy)

    def _center_dist(a: tuple, b: tuple) -> float:
        return math.hypot(
            (a[0] + a[2] / 2) - (b[0] + b[2] / 2),
            (a[1] + a[3] / 2) - (b[1] + b[3] / 2),
        )

    def _union(a: tuple, b: tuple) -> Tuple[int, int, int, int]:
        x  = min(a[0], b[0])
        y  = min(a[1], b[1])
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

    return pool


# ---------------------------------------------------------------------------
# Visualisation utilities (optional — require matplotlib)
# ---------------------------------------------------------------------------

def show_images_vertical(image_paths: List[str], width: int = 8) -> None:
    """Display a list of images stacked vertically."""
    import matplotlib.pyplot as plt

    if not image_paths:
        print("No image paths provided.")
        return

    fig, axes = plt.subplots(
        nrows=len(image_paths), ncols=1,
        figsize=(width, width * len(image_paths)),
    )
    if len(image_paths) == 1:
        axes = [axes]

    for ax, path in zip(axes, image_paths):
        try:
            with Image.open(path) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(path, fontsize=9)
        except Exception as err:
            ax.text(0.5, 0.5, f"Could not open:\n{path}\n\n{err}", ha="center", va="center")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_boxes_on_images(
    results: Dict[str, Dict],
    figsize: Tuple[int, int] = (10, 10),
    box_color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 3,
) -> None:
    """
    Overlay detected bounding boxes on each source image.

    Accepts the per-image dict returned by ``_crop_by_whitespace``, which
    contains ``"boxes"`` and an image path as the key.
    """
    import matplotlib.pyplot as plt

    for image_path, data in results.items():
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Could not read: {image_path}")
            continue

        overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i, (x, y, w, h) in enumerate(data["boxes"], start=1):
            cv2.rectangle(overlay, (x, y), (x + w, y + h), box_color, thickness)
            cv2.putText(
                overlay, str(i), (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2, cv2.LINE_AA,
            )

        plt.figure(figsize=figsize)
        plt.imshow(overlay)
        plt.title(Path(image_path).name)
        plt.axis("off")
        plt.show()


def visualize_merge(
    image_path: str,
    boxes: List[Tuple[int, int, int, int]],
    box_count: int,
    mode: str = "edge",
    show_original: bool = True,
) -> List[Tuple[int, int, int, int]]:
    """
    Merge boxes to ``box_count`` and display the result overlaid on the image.

    Original boxes are shown as thin blue outlines; merged boxes as coloured
    filled rectangles with numbered labels.

    Returns the merged box list.
    """
    import matplotlib.pyplot as plt

    merged = merge_figure_boxes(boxes, box_count, mode)

    img  = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    lw   = max(2, img.width // 300)
    fs   = max(12, img.width // 60)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
    except OSError:
        font = ImageFont.load_default()

    if show_original:
        for x, y, w, h in boxes:
            draw.rectangle(
                [x, y, x + w, y + h],
                outline=(70, 140, 230, 140),
                width=max(1, lw - 1),
            )

    for idx, (x, y, w, h) in enumerate(merged):
        r, g, b = _COLORS[idx % len(_COLORS)]
        draw.rectangle([x, y, x + w, y + h], fill=(r, g, b, 45), outline=(r, g, b), width=lw * 2)
        label = str(idx + 1)
        tw    = draw.textlength(label, font=font)
        tx, ty = x + 6, y + 4
        draw.rectangle([tx - 2, ty - 2, tx + tw + 4, ty + fs + 4], fill=(r, g, b, 210))
        draw.text((tx, ty), label, fill="white", font=font)

    plt.figure(figsize=(14, 10))
    plt.imshow(img.convert("RGB"))
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return merged


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def try_literal_eval(value):
    """
    Safely parse a stringified Python literal (list, dict, tuple).
    Returns the original value unchanged if parsing fails or is not applicable.
    Useful when image path lists have been serialised to CSV and read back as strings.
    """
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in ("[", "{", "(") and stripped[-1] in ("]", "}", ")"):
            try:
                return ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                pass
    return value

df = pd.read_csv("/home/yishin/keith/patent_research/model_io/Impact_Sub_KW.csv").map(try_literal_eval)

results = process_dataframe(
    df,
    output_root="image_crops",
    rotation_angles=(0, 90, 270),
    min_height=2000,
    psm_modes=(6, 11, 3, 12),
)

df.to_csv(
    "/home/yishin/keith/patent_research/model_io/Impact_Sub_KW_IMG.csv",
    index=False,
)
import json

import pandas as pd
import numpy as np
import ast
import os
import re
import math

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from typing import Union, List
from pathlib import Path
from PIL import Image
import networkx as nx

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

from tqdm.auto import tqdm

tqdm.pandas()

dataset_dir = "/home/yishin/keith/patent_research/model_io"

SAVE_PATH = os.path.join(
    dataset_dir,
    "4_Impact_Sub_GDino.csv",
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

Impact_df = pd.read_csv(os.path.join(dataset_dir, "3_Impact_Sub_Keywords.csv"), encoding="utf-8")
Impact_df = Impact_df.map(try_literal_eval)

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Grounding DINO Model
GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-base"

gdino_processor = AutoProcessor.from_pretrained(
    GROUNDING_DINO_MODEL
)

gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    GROUNDING_DINO_MODEL
).to(DEVICE)

gdino_model.eval()

def show_grounding_dino_results(
    results,
    cols=2,
    figsize_per_image=8,
):
    """
    Display Grounding DINO detections over each original image.
    """

    items = list(results.items())

    if not items:
        print("No results to display.")
        return

    rows = math.ceil(len(items) / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            cols * figsize_per_image,
            rows * figsize_per_image
        )
    )

    axes = np.atleast_1d(axes).flatten()

    for ax, (fig_name, figure_result) in zip(axes, items):
        image = Image.open(
            figure_result["image_path"]
        ).convert("RGB")

        ax.imshow(image)

        for detection in figure_result["detections"]:
            x1, y1, x2, y2 = detection["bbox"]

            ax.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    linewidth=2,
                )
            )

            label = (
                f"{detection['id']}: "
                f"{detection['label']} "
                f"{detection['score']:.2f}"
            )

            ax.text(
                x1,
                max(0, y1 - 4),
                label,
                fontsize=8,
                va="bottom",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                },
            )

        ax.set_title(
            f"{fig_name} — "
            f"{len(figure_result['detections'])} detections"
        )
        ax.axis("off")

    for ax in axes[len(items):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
def detect_keywords_with_grounding_dino(
    image_paths,
    keywords,
    box_threshold=0.25,
    text_threshold=0.20,
    cols=2,
    figsize_per_image=8,
    prompt_template=None,
    show=True,
):
    if not image_paths:
        raise ValueError("image_paths cannot be empty.")

    if not keywords:
        raise ValueError("keywords cannot be empty.")

    keywords = [
        str(keyword).strip()
        for keyword in keywords
        if str(keyword).strip()
    ]

    if prompt_template is None:
        labels = keywords
    else:
        labels = [
            prompt_template.format(keyword)
            for keyword in keywords
        ]

    # Grounding DINO expects one period-separated prompt string
    text_prompt = ". ".join(labels) + "."

    all_results = {}

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        fig_name = Path(image_path).stem

        inputs = gdino_processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.inference_mode():
            outputs = gdino_model(**inputs)

        target_sizes = torch.tensor(
            [[image.height, image.width]]
        )

        processed = gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]

        boxes = processed["boxes"].detach().cpu().numpy()
        scores = processed["scores"].detach().cpu().numpy()

        # Depending on transformers version, this may be
        # "text_labels" rather than "labels"
        detected_labels = processed.get(
            "text_labels",
            processed.get("labels", [])
        )

        detections = []

        for detection_id, (box, score, label) in enumerate(
            zip(boxes, scores, detected_labels)
        ):
            x1, y1, x2, y2 = box.tolist()

            detections.append({
                "id": detection_id,
                "bbox": (
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                ),
                "label": str(label),
                "score": float(score),
            })

        all_results[fig_name] = {
            "image_path": str(image_path),
            "prompt": text_prompt,
            "detections": detections,
        }

    if show:
        show_grounding_dino_results(
            all_results,
            cols=cols,
            figsize_per_image=figsize_per_image,
        )

    return all_results

# gdino_results = detect_keywords_with_grounding_dino(
#     image_paths=crops,
#     keywords=chair_keywords,
#     box_threshold=0.20,
#     text_threshold=0.18,
#     cols=2,
# )


# Create a Patent Graph
def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_contains(
    outer_bbox,
    inner_bbox,
    tolerance=0,
    require_strict=True,
    minimum_area_ratio=1.05,
):
    """
    outer_bbox must be at least minimum_area_ratio times larger
    than inner_bbox to qualify as its parent.
    """
    ox1, oy1, ox2, oy2 = outer_bbox
    ix1, iy1, ix2, iy2 = inner_bbox

    contained = (
        ix1 >= ox1 - tolerance
        and iy1 >= oy1 - tolerance
        and ix2 <= ox2 + tolerance
        and iy2 <= oy2 + tolerance
    )

    if not contained:
        return False

    outer_area = bbox_area(outer_bbox)
    inner_area = bbox_area(inner_bbox)

    if inner_area == 0:
        return False

    if require_strict and outer_area <= inner_area:
        return False

    area_ratio = outer_area / inner_area

    return area_ratio >= minimum_area_ratio

def bbox_iou(box_a, box_b):
    """
    Calculate intersection-over-union for two boxes:
    (x1, y1, x2, y2)
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1,
    )
    intersection_height = max(
        0,
        intersection_y2 - intersection_y1,
    )

    intersection_area = (
        intersection_width * intersection_height
    )

    area_a = bbox_area(box_a)
    area_b = bbox_area(box_b)

    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area

def normalize_detection_label(label):
    label = str(label).lower()
    label = label.replace("##", "")
    label = re.sub(r"\s+", " ", label).strip()

    words = label.split()
    cleaned_words = []

    for word in words:
        if not cleaned_words or cleaned_words[-1] != word:
            cleaned_words.append(word)

    return " ".join(cleaned_words)


def extract_known_keywords(
    detection_label,
    known_keywords,
):
    """
    Match a GDINO detection label against the keywords already
    present in the figure prompt.
    """
    label = normalize_detection_label(
        detection_label
    )

    matched = []

    # Check longer keywords first.
    sorted_keywords = sorted(
        known_keywords,
        key=len,
        reverse=True,
    )

    for keyword in sorted_keywords:
        normalized_keyword = normalize_detection_label(
            keyword
        )

        if normalized_keyword in label:
            matched.append(normalized_keyword)

    return list(dict.fromkeys(matched))

def get_detection_keywords(
    detection,
    known_keywords,
):
    raw_label = detection.get("label", "")

    matched_keywords = extract_known_keywords(
        detection_label=raw_label,
        known_keywords=known_keywords,
    )

    if matched_keywords:
        return set(matched_keywords)

    normalized = normalize_detection_label(raw_label)

    if normalized:
        return {normalized}

    return set()
def remove_duplicate_detections(
    detections,
    known_keywords,
    iou_threshold=0.85,
    require_keyword_overlap=True,
):
    """
    Remove highly overlapping duplicate detections.

    The highest-confidence box is kept first.

    Parameters
    ----------
    iou_threshold:
        Boxes with IoU greater than this value are considered
        duplicates.

    require_keyword_overlap:
        If True, overlapping boxes are removed only when their
        keyword sets overlap.
    """
    sorted_detections = sorted(
        detections,
        key=lambda detection: float(
            detection.get("score", 0.0)
        ),
        reverse=True,
    )

    kept_detections = []

    for candidate in sorted_detections:
        candidate_bbox = tuple(candidate["bbox"])

        candidate_keywords = get_detection_keywords(
            candidate,
            known_keywords,
        )

        is_duplicate = False

        for kept in kept_detections:
            kept_bbox = tuple(kept["bbox"])

            overlap = bbox_iou(
                candidate_bbox,
                kept_bbox,
            )

            if overlap < iou_threshold:
                continue

            if require_keyword_overlap:
                kept_keywords = get_detection_keywords(
                    kept,
                    known_keywords,
                )

                keyword_overlap = bool(
                    candidate_keywords & kept_keywords
                )

                if not keyword_overlap:
                    continue

            is_duplicate = True
            break

        if not is_duplicate:
            kept_detections.append(candidate)

    return kept_detections

def bbox_contains_with_ratio(
    outer_bbox,
    inner_bbox,
    tolerance=2,
    minimum_area_ratio=1.10,
):
    """
    Return True when inner_bbox is contained by outer_bbox.

    minimum_area_ratio prevents nearly identical boxes from
    producing unnecessary parent-child chains.
    """
    ox1, oy1, ox2, oy2 = outer_bbox
    ix1, iy1, ix2, iy2 = inner_bbox

    contained = (
        ix1 >= ox1 - tolerance
        and iy1 >= oy1 - tolerance
        and ix2 <= ox2 + tolerance
        and iy2 <= oy2 + tolerance
    )

    if not contained:
        return False

    outer_area = bbox_area(outer_bbox)
    inner_area = bbox_area(inner_bbox)

    if inner_area <= 0 or outer_area <= inner_area:
        return False

    return outer_area / inner_area >= minimum_area_ratio


def find_bbox_parents_with_ratio(
    detections,
    tolerance=2,
    minimum_area_ratio=1.10,
):
    """
    For each bounding box, select the smallest valid containing
    bounding box as its direct parent.
    """
    parent_map = {}

    for child in detections:
        child_id = child["id"]
        child_bbox = tuple(child["bbox"])

        possible_parents = []

        for candidate in detections:
            if candidate["id"] == child_id:
                continue

            candidate_bbox = tuple(candidate["bbox"])

            if bbox_contains_with_ratio(
                outer_bbox=candidate_bbox,
                inner_bbox=child_bbox,
                tolerance=tolerance,
                minimum_area_ratio=minimum_area_ratio,
            ):
                possible_parents.append(candidate)

        if not possible_parents:
            parent_map[child_id] = None
            continue

        direct_parent = min(
            possible_parents,
            key=lambda item: bbox_area(item["bbox"]),
        )

        parent_map[child_id] = direct_parent["id"]

    return parent_map
def find_bbox_parents(detections, tolerance=0):
    """
    Determine the direct parent of each bounding box.

    Returns:
        {
            child_detection_id: parent_detection_id | None
        }
    """
    valid_detections = [
        detection
        for detection in detections
        if detection.get("bbox") is not None
        and len(detection["bbox"]) == 4
    ]

    parent_map = {}

    for child in valid_detections:
        child_id = child["id"]
        child_bbox = tuple(child["bbox"])

        possible_parents = []

        for candidate_parent in valid_detections:
            parent_id = candidate_parent["id"]

            if parent_id == child_id:
                continue

            parent_bbox = tuple(candidate_parent["bbox"])

            if bbox_contains(
                outer_bbox=parent_bbox,
                inner_bbox=child_bbox,
                tolerance=tolerance,
                require_strict=True,
                minimum_area_ratio=1.1,
            ):
                possible_parents.append(candidate_parent)

        if not possible_parents:
            parent_map[child_id] = None
            continue

        # The nearest parent is the smallest box that contains the child.
        direct_parent = min(
            possible_parents,
            key=lambda detection: bbox_area(tuple(detection["bbox"])),
        )

        parent_map[child_id] = direct_parent["id"]

    return parent_map

def build_patent_detection_graph(
    patent_title: str,
    gdino_output: dict,
    score_threshold: float = 0.20,
    containment_tolerance: int = 0,
):
    """
    Graph structure:

        patent title
            ↓
        figure
            ↓
        top-level bounding box
            ↓
        nested bounding box
            ↓
        keyword

    Bounding boxes share keyword nodes across figures.
    Bounding-box nodes also store a cropped PIL image (crop_image)
    so the graph can be drawn with real crops instead of circles.
    """
    graph = nx.DiGraph()

    root_id = f"title::{patent_title}"

    # Needed to filter out keywords that are just the title
    # (e.g. GDINO detecting "chair" on a patent titled "Chair").
    normalized_title = normalize_detection_label(patent_title)

    graph.add_node(
        root_id,
        node_type="title",
        display_label=patent_title,
    )

    for figure_name, figure_data in gdino_output.items():
        figure_id = f"figure::{figure_name}"
        image_path = figure_data.get("image_path")

        graph.add_node(
            figure_id,
            node_type="figure",
            display_label=figure_name,
            image_path=image_path,
        )

        graph.add_edge(
            root_id,
            figure_id,
            edge_type="contains_figure",
        )

        # Load the source image once per figure so we can crop bboxes.
        source_image = None
        image_width = image_height = None

        if image_path and Path(image_path).exists():
            source_image = Image.open(image_path).convert("RGB")
            image_width, image_height = source_image.size

        # Keep only detections above the threshold.
        detections = [
            detection.copy()
            for detection in figure_data.get("detections", [])
            if float(detection.get("score", 0.0))
            >= score_threshold
        ]

        # Clip bboxes to image bounds and drop degenerate boxes,
        # same as the single-figure version does.
        if source_image is not None:
            valid_detections = []

            for detection in detections:
                if "bbox" not in detection:
                    continue

                x1, y1, x2, y2 = map(int, detection["bbox"])

                x1 = max(0, min(x1, image_width))
                y1 = max(0, min(y1, image_height))
                x2 = max(0, min(x2, image_width))
                y2 = max(0, min(y2, image_height))

                if x2 <= x1 or y2 <= y1:
                    continue

                detection["bbox"] = (x1, y1, x2, y2)
                valid_detections.append(detection)

            detections = valid_detections

        known_keywords = [
            keyword.strip().lower()
            for keyword in figure_data.get(
                "prompt",
                "",
            ).split(".")
            if keyword.strip()
        ]

        # Never treat the title itself as a valid keyword.
        # known_keywords = [
        #     keyword
        #     for keyword in known_keywords
        #     if normalize_detection_label(keyword) != normalized_title
        # ]

        detections = remove_duplicate_detections(
            detections=detections,
            known_keywords=known_keywords,
            iou_threshold=0.80,
            require_keyword_overlap=False,
        )

        # -------------------------------------------------
        # 1. Add every bounding-box node first
        # -------------------------------------------------
        for detection in detections:
            detection_id = detection["id"]
            bbox = tuple(detection["bbox"])
            score = float(detection.get("score", 0.0))
            raw_label = detection.get("label", "")

            bbox_id = (
                f"bbox::{figure_name}::{detection_id}"
            )

            crop_image = None
            if source_image is not None:
                crop_image = source_image.crop(bbox)

            graph.add_node(
                bbox_id,
                node_type="bbox",
                display_label=(
                    f"Box {detection_id}\n"
                    f"{score:.2f}\n"
                    f"{bbox}"
                ),
                figure=figure_name,
                detection_id=detection_id,
                bbox=bbox,
                bbox_area=bbox_area(bbox),
                score=score,
                raw_label=raw_label,
                crop_image=crop_image,
            )

        # -------------------------------------------------
        # 2. Calculate containment hierarchy
        # -------------------------------------------------
        parent_map = find_bbox_parents(
            detections=detections,
            tolerance=containment_tolerance,
        )

        for detection in detections:
            detection_id = detection["id"]
            bbox_id = (
                f"bbox::{figure_name}::{detection_id}"
            )

            parent_detection_id = parent_map[detection_id]

            if parent_detection_id is None:
                # No containing box: direct child of figure.
                graph.add_edge(
                    figure_id,
                    bbox_id,
                    edge_type="contains_top_level_bbox",
                )
            else:
                parent_bbox_id = (
                    f"bbox::{figure_name}::"
                    f"{parent_detection_id}"
                )

                graph.add_edge(
                    parent_bbox_id,
                    bbox_id,
                    edge_type="contains_bbox",
                )

        # -------------------------------------------------
        # 3. Connect boxes to shared keyword nodes
        # -------------------------------------------------
        # Connect bounding boxes to keywords.
        for detection in detections:
            detection_id = detection["id"]
            raw_label = str(detection.get("label", ""))
            score = float(detection.get("score", 0.0))

            bbox_node = f"bbox::{figure_name}::{detection_id}"

            matched_keywords = extract_known_keywords(
                detection_label=raw_label,
                known_keywords=known_keywords,
            )

            # Use the cleaned raw label only when no known keyword matches.
            if not matched_keywords:
                cleaned_label = normalize_detection_label(raw_label)

                if (
                    cleaned_label
                    and cleaned_label != normalized_title
                ):
                    matched_keywords = [cleaned_label]

            for keyword in matched_keywords:
                keyword_node = f"keyword::{keyword}"

                if keyword_node not in graph:
                    graph.add_node(
                        keyword_node,
                        node_type="keyword",
                        display_label=keyword,
                    )

                graph.add_edge(
                    bbox_node,
                    keyword_node,
                    edge_type="classified_as",
                    score=score,
                )

    return graph

def graph_to_serializable_json(graph):
    """
    Convert a networkx DiGraph into a JSON-safe node-link dict.
    Drops PIL crop_image objects (not serializable) but keeps
    bbox + image_path on bbox nodes, so crops can be regenerated
    later if needed.
    """
    export_graph = graph.copy()

    for _, data in export_graph.nodes(data=True):
        data.pop("crop_image", None)

    node_link = nx.node_link_data(export_graph, edges="edges")
    return json.dumps(node_link)

def process_row(row):
    title = row["title"]
    crop_paths = row["crop_paths"]
    keywords = row["keywords"]

    if not crop_paths or not keywords:
        return None

    if isinstance(crop_paths, str):
        crop_paths = [crop_paths]
    if isinstance(keywords, str):
        keywords = [keywords]

    gdino_results = detect_keywords_with_grounding_dino(
        image_paths=crop_paths,
        keywords=keywords,
        box_threshold=0.25,
        text_threshold=0.20,
        show=False,
    )

    graph = build_patent_detection_graph(
        patent_title=title,
        gdino_output=gdino_results,
        score_threshold=0.20,
        containment_tolerance=20,
    )

    return graph_to_serializable_json(graph)

if os.path.exists(SAVE_PATH):
    progress_df = pd.read_csv(SAVE_PATH, encoding="utf-8")
    progress_df = progress_df.map(try_literal_eval)
    print(f"Resuming from checkpoint: {len(progress_df)} rows loaded.")
else:
    progress_df = Impact_df.copy()
    progress_df["detection_graph_json"] = None
    progress_df["graph_status"] = "pending"

for idx in tqdm(progress_df.index, desc="Building detection graphs"):
    if progress_df.at[idx, "graph_status"] == "done":
        continue

    row = progress_df.loc[idx]

    try:
        graph_json = process_row(row)

        if graph_json is None:
            progress_df.at[idx, "graph_status"] = "skipped_no_data"
        else:
            progress_df.at[idx, "detection_graph_json"] = graph_json
            progress_df.at[idx, "graph_status"] = "done"

    except Exception as e:
        progress_df.at[idx, "graph_status"] = f"error: {e}"
        print(f"Row {idx} failed: {e}")

    if (idx + 1) % SAVE_EVERY == 0:
        progress_df.to_csv(SAVE_PATH, index=False, encoding="utf-8")
        print(f"Checkpoint saved at row {idx + 1}")

progress_df.to_csv(SAVE_PATH, index=False, encoding="utf-8")
print("Done. Final checkpoint saved.")
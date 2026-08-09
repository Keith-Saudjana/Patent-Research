"""
Graph-only prompt-based classification with DesignCLIP PatentCLIP.

This script replaces the whole-patent image input with the semantic
region graph stored in `detection_graph_json`.

Architecture
------------
1. Freeze the released PatentCLIP model.
2. Precompute PatentCLIP features for every graph node:
   - bbox nodes    -> PatentCLIP image feature of the region crop
   - figure nodes  -> PatentCLIP image feature of the full figure
   - keyword nodes -> PatentCLIP text feature of the keyword
   - title nodes   -> PatentCLIP text feature of the patent title
3. Run relation-specific message passing over:
   - contains_figure
   - contains_top_level_bbox
   - contains_bbox
   - classified_as
   Reverse relations are added automatically.
4. Attention-pool all nodes into one graph representation.
5. Project the graph representation into PatentCLIP's joint embedding space.
6. Compare it with PatentCLIP descriptor-prompt embeddings:
       "a patent drawing of {descriptor}."
7. Train with cross-entropy over graph-to-prompt similarity scores.

Important
---------
- No whole-image branch is used.
- No separate linear classification head is used.
- PatentCLIP is frozen. Only the graph encoder and graph projection train.
- The graph contains both visual and textual node features. It is therefore
  graph-only at the sample level, but not strictly vision-only.
- The code assumes the node-link graph schema used by the existing project:
  title, figure, bbox, and keyword nodes with typed edges.

Dependencies
------------
pip install open_clip_torch networkx pandas scikit-learn pillow tqdm
"""

import ast
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATASET_DIR = "/home/yishin/keith/patent_research/model_io"
CHECKPOINT_DIR = "/home/yishin/keith/patent_research/checkpoints"

INPUT_CSV_PATH = os.path.join(
    DATASET_DIR,
    "5_Impact_Sub_Final.csv",
)
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "patentclip_graph_prompt_classifier.pth",
)

LABEL_COLUMN = "full_class_desc"
GRAPH_COLUMN = "detection_graph_json"

PATENTCLIP_MODEL_ID = "hf-hub:hhshomee2/PatentCLIP_Vit_B"
PROMPT_TEMPLATE = "a patent drawing of {}."

BASE_RELATIONS = [
    "contains_figure",
    "contains_top_level_bbox",
    "contains_bbox",
    "classified_as",
]
ALL_RELATIONS = BASE_RELATIONS + [
    f"{relation}_rev"
    for relation in BASE_RELATIONS
]

NODE_TYPE_TO_INDEX = {
    "title": 0,
    "figure": 1,
    "bbox": 2,
    "keyword": 3,
}

HIDDEN_DIM = 128
NUM_GNN_LAYERS = 2
DROPOUT = 0.30

EMBED_BATCH_SIZE = 64
BATCH_SIZE = 16
NUM_WORKERS = 0

MAX_EPOCHS = 100
PATIENCE = 8
MIN_DELTA = 0.001
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
MAX_GRAD_NORM = 1.0

USE_CLASS_WEIGHTING = False
CLASS_WEIGHT_BETA = 1.2

RANDOM_STATE = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# General data helpers
# ----------------------------------------------------------------------
def try_literal_eval(value):
    """Parse string representations of Python containers when possible."""
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    is_container = (
        (stripped.startswith("[") and stripped.endswith("]"))
        or (stripped.startswith("{") and stripped.endswith("}"))
        or (stripped.startswith("(") and stripped.endswith(")"))
    )

    if not is_container:
        return value

    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return value


def has_value(value) -> bool:
    if value is None:
        return False

    if isinstance(value, str):
        return value.strip().lower() not in {
            "",
            "none",
            "nan",
            "null",
            "{}",
            "[]",
        }

    if isinstance(value, (dict, list, tuple, set)):
        return len(value) > 0

    try:
        return not pd.isna(value)
    except (TypeError, ValueError):
        return True


def resolve_existing_path(path_value) -> Optional[str]:
    """
    Resolve image paths stored either as absolute paths or relative to
    DATASET_DIR. Return None when no existing file can be found.
    """
    if not has_value(path_value):
        return None

    raw_path = os.path.expanduser(str(path_value).strip())

    candidates = [
        raw_path,
        os.path.join(DATASET_DIR, raw_path),
    ]

    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isfile(candidate):
            return candidate

    return None


def stratified_train_val_test_split(
    dataframe: pd.DataFrame,
    label_column: str,
    train_size: float = 0.60,
    validation_size: float = 0.20,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(
        train_size + validation_size + test_size,
        1.0,
    ):
        raise ValueError(
            "train_size, validation_size, and test_size must sum to 1."
        )

    split_dataframe = dataframe.dropna(
        subset=[label_column]
    ).copy()

    training_dataframe, temporary_dataframe = train_test_split(
        split_dataframe,
        test_size=validation_size + test_size,
        stratify=split_dataframe[label_column],
        random_state=random_state,
    )

    relative_test_size = (
        test_size / (validation_size + test_size)
    )

    validation_dataframe, test_dataframe = train_test_split(
        temporary_dataframe,
        test_size=relative_test_size,
        stratify=temporary_dataframe[label_column],
        random_state=random_state,
    )

    return (
        training_dataframe.reset_index(drop=True),
        validation_dataframe.reset_index(drop=True),
        test_dataframe.reset_index(drop=True),
    )


def load_and_split_data():
    dataframe = pd.read_csv(
        INPUT_CSV_PATH,
        encoding="utf-8",
    )
    dataframe = dataframe.map(try_literal_eval)

    required_columns = [
        LABEL_COLUMN,
        GRAPH_COLUMN,
        "keywords",
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing required CSV columns: "
            + ", ".join(missing_columns)
        )

    mask = dataframe[
        ["keywords", GRAPH_COLUMN]
    ].apply(
        lambda row: all(
            has_value(value)
            for value in row
        ),
        axis=1,
    )

    filtered_dataframe = (
        dataframe[mask]
        .copy()
        .reset_index(drop=True)
    )

    class_counts = filtered_dataframe[
        LABEL_COLUMN
    ].value_counts()

    valid_classes = class_counts[
        class_counts > 5
    ].index

    filtered_dataframe = (
        filtered_dataframe[
            filtered_dataframe[LABEL_COLUMN].isin(
                valid_classes
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    # Preserve one globally unique identifier across all three splits.
    # This avoids collisions after each split is reset to index 0.
    filtered_dataframe["_row_id"] = np.arange(
        len(filtered_dataframe)
    )

    print(
        "Rows after graph and rare-class filtering: "
        f"{len(filtered_dataframe)}"
    )
    print(
        "Descriptor classes: "
        f"{filtered_dataframe[LABEL_COLUMN].nunique()}"
    )

    return stratified_train_val_test_split(
        filtered_dataframe,
        label_column=LABEL_COLUMN,
    )


# ----------------------------------------------------------------------
# Graph parsing
# ----------------------------------------------------------------------
def load_graph(graph_value) -> nx.Graph:
    """
    Load a NetworkX node-link graph from a JSON string or dictionary.
    Supports both the newer `edges` key and older `links` key.
    """
    if isinstance(graph_value, nx.Graph):
        return graph_value.copy()

    if isinstance(graph_value, str):
        node_link_data = json.loads(graph_value)
    elif isinstance(graph_value, dict):
        node_link_data = graph_value
    else:
        raise TypeError(
            "detection_graph_json must be a JSON string or dictionary."
        )

    edge_key = (
        "edges"
        if "edges" in node_link_data
        else "links"
    )

    try:
        return nx.node_link_graph(
            node_link_data,
            edges=edge_key,
        )
    except TypeError:
        # Compatibility with older NetworkX releases.
        return nx.node_link_graph(
            node_link_data,
            link=edge_key,
        )


def node_text(node_data: dict) -> str:
    """Find the most likely textual label for title or keyword nodes."""
    for key in (
        "display_label",
        "label",
        "text",
        "name",
        "title",
        "keyword",
    ):
        value = node_data.get(key)
        if has_value(value):
            return str(value).strip()

    return ""


def figure_image_path_for_bbox(
    graph: nx.Graph,
    bbox_node_data: dict,
) -> Optional[str]:
    """
    Locate the figure image associated with a bbox node.

    The existing graph schema stores a figure name on the bbox node and
    stores the actual image_path on the corresponding figure node.
    """
    figure_name = bbox_node_data.get("figure")

    if has_value(figure_name):
        figure_node_id = f"figure::{figure_name}"

        if figure_node_id in graph.nodes:
            image_path = graph.nodes[
                figure_node_id
            ].get("image_path")

            resolved = resolve_existing_path(image_path)
            if resolved is not None:
                return resolved

    # Defensive fallback: find a connected figure node.
    for neighbor_id in graph.neighbors(
        bbox_node_data.get("_node_id", "")
    ):
        neighbor_data = graph.nodes[neighbor_id]

        if neighbor_data.get("node_type") == "figure":
            resolved = resolve_existing_path(
                neighbor_data.get("image_path")
            )
            if resolved is not None:
                return resolved

    return None


def valid_bbox_tuple(value) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(value, (list, tuple)):
        return None

    if len(value) != 4:
        return None

    try:
        x1, y1, x2, y2 = [
            int(round(float(number)))
            for number in value
        ]
    except (TypeError, ValueError):
        return None

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


# ----------------------------------------------------------------------
# Frozen PatentCLIP feature extraction
# ----------------------------------------------------------------------
def normalize_features(
    features: torch.Tensor,
) -> torch.Tensor:
    return features / features.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)


@torch.no_grad()
def embed_text_values(
    texts: Sequence[str],
    patentclip_model: nn.Module,
    tokenizer,
) -> Dict[str, torch.Tensor]:
    cache: Dict[str, torch.Tensor] = {}

    unique_texts = list(dict.fromkeys(texts))

    for start in tqdm(
        range(0, len(unique_texts), EMBED_BATCH_SIZE),
        desc="Encoding graph text nodes",
    ):
        batch_texts = unique_texts[
            start : start + EMBED_BATCH_SIZE
        ]

        if not batch_texts:
            continue

        tokens = tokenizer(batch_texts).to(DEVICE)
        features = patentclip_model.encode_text(
            tokens
        ).float()
        features = normalize_features(features).cpu()

        for text, feature in zip(
            batch_texts,
            features,
        ):
            cache[text] = feature

    return cache


@torch.no_grad()
def embed_figure_paths(
    image_paths: Sequence[str],
    patentclip_model: nn.Module,
    image_transform,
) -> Dict[str, torch.Tensor]:
    cache: Dict[str, torch.Tensor] = {}
    unique_paths = list(dict.fromkeys(image_paths))

    for start in tqdm(
        range(0, len(unique_paths), EMBED_BATCH_SIZE),
        desc="Encoding graph figure nodes",
    ):
        batch_paths = unique_paths[
            start : start + EMBED_BATCH_SIZE
        ]

        valid_paths: List[str] = []
        image_tensors: List[torch.Tensor] = []

        for image_path in batch_paths:
            try:
                with Image.open(image_path) as image:
                    image_tensor = image_transform(
                        image.convert("RGB")
                    )
                valid_paths.append(image_path)
                image_tensors.append(image_tensor)
            except Exception:
                continue

        if not image_tensors:
            continue

        image_batch = torch.stack(
            image_tensors
        ).to(DEVICE)

        features = patentclip_model.encode_image(
            image_batch
        ).float()
        features = normalize_features(features).cpu()

        for image_path, feature in zip(
            valid_paths,
            features,
        ):
            cache[image_path] = feature

    return cache


@torch.no_grad()
def embed_bbox_requests(
    crop_requests: Sequence[
        Tuple[str, Tuple[int, int, int, int]]
    ],
    patentclip_model: nn.Module,
    image_transform,
) -> Dict[
    Tuple[str, Tuple[int, int, int, int]],
    torch.Tensor,
]:
    cache = {}
    unique_requests = list(
        dict.fromkeys(crop_requests)
    )

    for start in tqdm(
        range(0, len(unique_requests), EMBED_BATCH_SIZE),
        desc="Encoding graph bbox nodes",
    ):
        batch_requests = unique_requests[
            start : start + EMBED_BATCH_SIZE
        ]

        valid_requests = []
        crop_tensors = []

        for image_path, bbox in batch_requests:
            try:
                with Image.open(image_path) as image:
                    crop = image.convert("RGB").crop(
                        bbox
                    )
                    crop_tensor = image_transform(crop)

                valid_requests.append(
                    (image_path, bbox)
                )
                crop_tensors.append(crop_tensor)
            except Exception:
                continue

        if not crop_tensors:
            continue

        crop_batch = torch.stack(
            crop_tensors
        ).to(DEVICE)

        features = patentclip_model.encode_image(
            crop_batch
        ).float()
        features = normalize_features(features).cpu()

        for request, feature in zip(
            valid_requests,
            features,
        ):
            cache[request] = feature

    return cache


def collect_graph_requests(
    dataframe: pd.DataFrame,
):
    """
    Parse every graph once and collect the unique image and text requests
    needed to build its PatentCLIP node features.
    """
    graphs_by_row_id: Dict[int, nx.Graph] = {}

    figure_paths: List[str] = []
    crop_requests: List[
        Tuple[str, Tuple[int, int, int, int]]
    ] = []
    text_values: List[str] = []

    invalid_graph_count = 0

    for _, row in dataframe.iterrows():
        row_id = int(row["_row_id"])

        try:
            graph = load_graph(
                row[GRAPH_COLUMN]
            )
        except Exception:
            invalid_graph_count += 1
            continue

        # Store the current node ID in the node attributes so fallback
        # path resolution can inspect graph neighbors if necessary.
        for node_id in graph.nodes:
            graph.nodes[node_id][
                "_node_id"
            ] = node_id

        graphs_by_row_id[row_id] = graph

        for _, node_data in graph.nodes(
            data=True
        ):
            node_type = node_data.get(
                "node_type"
            )

            if node_type == "figure":
                image_path = resolve_existing_path(
                    node_data.get("image_path")
                )

                if image_path is not None:
                    figure_paths.append(image_path)

            elif node_type == "bbox":
                image_path = (
                    figure_image_path_for_bbox(
                        graph,
                        node_data,
                    )
                )
                bbox = valid_bbox_tuple(
                    node_data.get("bbox")
                )

                if (
                    image_path is not None
                    and bbox is not None
                ):
                    crop_requests.append(
                        (image_path, bbox)
                    )

            elif node_type in {
                "title",
                "keyword",
            }:
                text = node_text(node_data)

                if text:
                    text_values.append(text)

    print(
        f"Parsed graphs: {len(graphs_by_row_id)}"
    )
    print(
        f"Invalid graphs skipped: {invalid_graph_count}"
    )
    print(
        "Unique figure images: "
        f"{len(set(figure_paths))}"
    )
    print(
        "Unique bbox crops: "
        f"{len(set(crop_requests))}"
    )
    print(
        "Unique title/keyword texts: "
        f"{len(set(text_values))}"
    )

    return (
        graphs_by_row_id,
        figure_paths,
        crop_requests,
        text_values,
    )


# ----------------------------------------------------------------------
# Graph tensor construction
# ----------------------------------------------------------------------
def build_graph_tensors(
    graph: nx.Graph,
    figure_cache: Dict[str, torch.Tensor],
    crop_cache: Dict[
        Tuple[str, Tuple[int, int, int, int]],
        torch.Tensor,
    ],
    text_cache: Dict[str, torch.Tensor],
    embedding_dimension: int,
):
    node_ids = list(graph.nodes())

    if not node_ids:
        return None

    node_id_to_index = {
        node_id: index
        for index, node_id in enumerate(
            node_ids
        )
    }

    node_features = torch.zeros(
        len(node_ids),
        embedding_dimension,
        dtype=torch.float32,
    )
    node_type_ids = torch.zeros(
        len(node_ids),
        dtype=torch.long,
    )

    populated_nodes = 0

    for node_id, node_data in graph.nodes(
        data=True
    ):
        node_index = node_id_to_index[node_id]
        node_type = node_data.get("node_type")

        node_type_ids[node_index] = (
            NODE_TYPE_TO_INDEX.get(
                node_type,
                0,
            )
        )

        feature = None

        if node_type == "figure":
            image_path = resolve_existing_path(
                node_data.get("image_path")
            )

            if image_path in figure_cache:
                feature = figure_cache[image_path]

        elif node_type == "bbox":
            image_path = (
                figure_image_path_for_bbox(
                    graph,
                    node_data,
                )
            )
            bbox = valid_bbox_tuple(
                node_data.get("bbox")
            )

            request = (
                (image_path, bbox)
                if image_path is not None
                and bbox is not None
                else None
            )

            if request in crop_cache:
                feature = crop_cache[request]

        elif node_type in {
            "title",
            "keyword",
        }:
            text = node_text(node_data)

            if text in text_cache:
                feature = text_cache[text]

        if feature is not None:
            node_features[node_index] = feature
            populated_nodes += 1

    # A graph containing no usable PatentCLIP node features cannot be
    # meaningfully classified.
    if populated_nodes == 0:
        return None

    edge_lists = {
        relation: ([], [])
        for relation in ALL_RELATIONS
    }

    for source, destination, edge_data in graph.edges(
        data=True
    ):
        relation = edge_data.get("edge_type")

        if relation not in BASE_RELATIONS:
            continue

        source_index = node_id_to_index[source]
        destination_index = (
            node_id_to_index[destination]
        )

        edge_lists[relation][0].append(
            source_index
        )
        edge_lists[relation][1].append(
            destination_index
        )

        reverse_relation = (
            f"{relation}_rev"
        )
        edge_lists[
            reverse_relation
        ][0].append(destination_index)
        edge_lists[
            reverse_relation
        ][1].append(source_index)

    edge_index_dictionary = {}

    for relation in ALL_RELATIONS:
        source_list, destination_list = (
            edge_lists[relation]
        )

        edge_index_dictionary[relation] = (
            torch.tensor(
                source_list,
                dtype=torch.long,
            ),
            torch.tensor(
                destination_list,
                dtype=torch.long,
            ),
        )

    return (
        node_features,
        node_type_ids,
        edge_index_dictionary,
    )


class PatentGraphDataset(Dataset):
    """
    Every item consists only of a graph and its class label.

    PatentCLIP node features are already precomputed, so no image loading
    or text tokenization occurs inside the training loop.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        graphs_by_row_id: Dict[int, nx.Graph],
        label_to_index: Dict[str, int],
        figure_cache: Dict[str, torch.Tensor],
        crop_cache,
        text_cache: Dict[str, torch.Tensor],
        embedding_dimension: int,
    ):
        self.items = []
        skipped = 0

        for _, row in dataframe.iterrows():
            row_id = int(row["_row_id"])
            label = row[LABEL_COLUMN]
            graph = graphs_by_row_id.get(row_id)

            if (
                graph is None
                or label not in label_to_index
            ):
                skipped += 1
                continue

            graph_tensors = build_graph_tensors(
                graph=graph,
                figure_cache=figure_cache,
                crop_cache=crop_cache,
                text_cache=text_cache,
                embedding_dimension=(
                    embedding_dimension
                ),
            )

            if graph_tensors is None:
                skipped += 1
                continue

            (
                node_features,
                node_type_ids,
                edge_index_dictionary,
            ) = graph_tensors

            self.items.append(
                (
                    node_features,
                    node_type_ids,
                    edge_index_dictionary,
                    label_to_index[label],
                )
            )

        self.labels = [
            item[3]
            for item in self.items
        ]

        print(
            f"Built {len(self.items)} graphs; "
            f"skipped {skipped} unusable rows."
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate_graph_batch(batch):
    all_node_features = []
    all_node_type_ids = []
    graph_membership = []
    labels = []

    batched_edges = {
        relation: ([], [])
        for relation in ALL_RELATIONS
    }

    node_offset = 0

    for graph_index, (
        node_features,
        node_type_ids,
        edge_index_dictionary,
        label,
    ) in enumerate(batch):
        number_of_nodes = node_features.size(0)

        all_node_features.append(node_features)
        all_node_type_ids.append(node_type_ids)
        graph_membership.append(
            torch.full(
                (number_of_nodes,),
                graph_index,
                dtype=torch.long,
            )
        )
        labels.append(label)

        for relation, (
            source_indices,
            destination_indices,
        ) in edge_index_dictionary.items():
            if source_indices.numel() == 0:
                continue

            batched_edges[relation][0].append(
                source_indices + node_offset
            )
            batched_edges[relation][1].append(
                destination_indices + node_offset
            )

        node_offset += number_of_nodes

    node_feature_batch = torch.cat(
        all_node_features,
        dim=0,
    )
    node_type_batch = torch.cat(
        all_node_type_ids,
        dim=0,
    )
    graph_membership_batch = torch.cat(
        graph_membership,
        dim=0,
    )
    label_batch = torch.tensor(
        labels,
        dtype=torch.long,
    )

    edge_index_batch = {}

    for relation, (
        source_batches,
        destination_batches,
    ) in batched_edges.items():
        if source_batches:
            edge_index_batch[relation] = (
                torch.cat(source_batches),
                torch.cat(destination_batches),
            )
        else:
            edge_index_batch[relation] = (
                torch.empty(
                    0,
                    dtype=torch.long,
                ),
                torch.empty(
                    0,
                    dtype=torch.long,
                ),
            )

    return (
        node_feature_batch,
        node_type_batch,
        edge_index_batch,
        graph_membership_batch,
        label_batch,
        len(batch),
    )


# ----------------------------------------------------------------------
# Relational graph encoder
# ----------------------------------------------------------------------
class RelationalGNNLayer(nn.Module):
    def __init__(
        self,
        dimension: int,
        relation_names: Sequence[str],
        dropout: float,
    ):
        super().__init__()

        self.self_projection = nn.Linear(
            dimension,
            dimension,
        )
        self.relation_projections = nn.ModuleDict(
            {
                relation: nn.Linear(
                    dimension,
                    dimension,
                )
                for relation in relation_names
            }
        )
        self.normalization = nn.LayerNorm(
            dimension
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index_dictionary,
    ) -> torch.Tensor:
        number_of_nodes = node_features.size(0)

        output = self.self_projection(
            node_features
        )

        for relation, (
            source_indices,
            destination_indices,
        ) in edge_index_dictionary.items():
            if source_indices.numel() == 0:
                continue

            messages = (
                self.relation_projections[
                    relation
                ](
                    node_features[
                        source_indices
                    ]
                )
            )

            aggregated_messages = torch.zeros(
                number_of_nodes,
                messages.size(-1),
                dtype=messages.dtype,
                device=messages.device,
            )
            aggregated_messages.index_add_(
                0,
                destination_indices,
                messages,
            )

            destination_degree = torch.zeros(
                number_of_nodes,
                dtype=messages.dtype,
                device=messages.device,
            )
            destination_degree.index_add_(
                0,
                destination_indices,
                torch.ones(
                    destination_indices.size(0),
                    dtype=messages.dtype,
                    device=messages.device,
                ),
            )
            destination_degree = (
                destination_degree
                .clamp_min(1.0)
                .unsqueeze(-1)
            )

            output = (
                output
                + aggregated_messages
                / destination_degree
            )

        # Residual connection keeps the original PatentCLIP node signal.
        output = self.normalization(
            output + node_features
        )
        output = F.gelu(output)
        return self.dropout(output)


class AttentionGraphPooling(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.attention_score = nn.Linear(
            dimension,
            1,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        graph_membership: torch.Tensor,
        number_of_graphs: int,
    ) -> torch.Tensor:
        raw_scores = self.attention_score(
            node_features
        ).squeeze(-1)

        pooled_features = torch.zeros(
            number_of_graphs,
            node_features.size(-1),
            dtype=node_features.dtype,
            device=node_features.device,
        )

        for graph_index in range(
            number_of_graphs
        ):
            mask = (
                graph_membership
                == graph_index
            )

            if not torch.any(mask):
                continue

            weights = F.softmax(
                raw_scores[mask],
                dim=0,
            )

            pooled_features[graph_index] = (
                weights.unsqueeze(-1)
                * node_features[mask]
            ).sum(dim=0)

        return pooled_features


class PatentCLIPGraphPromptClassifier(nn.Module):
    """
    Trainable graph encoder with a frozen PatentCLIP text classifier.

    The final logits are scaled cosine similarities between:
      - the learned graph embedding;
      - each frozen PatentCLIP descriptor-prompt embedding.
    """

    def __init__(
        self,
        clip_embedding_dimension: int,
        hidden_dimension: int,
        class_text_features: torch.Tensor,
        patentclip_logit_scale: torch.Tensor,
        number_of_node_types: int,
        number_of_layers: int,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            clip_embedding_dimension,
            hidden_dimension,
        )
        self.node_type_embedding = nn.Embedding(
            number_of_node_types,
            hidden_dimension,
        )

        self.gnn_layers = nn.ModuleList(
            [
                RelationalGNNLayer(
                    dimension=hidden_dimension,
                    relation_names=ALL_RELATIONS,
                    dropout=DROPOUT,
                )
                for _ in range(
                    number_of_layers
                )
            ]
        )

        self.graph_pooling = (
            AttentionGraphPooling(
                hidden_dimension
            )
        )
        self.graph_projection = nn.Linear(
            hidden_dimension,
            clip_embedding_dimension,
        )

        # These are frozen components of the PatentCLIP classifier.
        self.register_buffer(
            "class_text_features",
            normalize_features(
                class_text_features.float()
            ),
        )
        self.register_buffer(
            "logit_scale",
            patentclip_logit_scale
            .detach()
            .float()
            .clamp(max=100.0),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        node_type_ids: torch.Tensor,
        edge_index_dictionary,
        graph_membership: torch.Tensor,
        number_of_graphs: int,
    ):
        hidden_features = (
            self.input_projection(
                node_features
            )
            + self.node_type_embedding(
                node_type_ids
            )
        )
        hidden_features = F.gelu(
            hidden_features
        )

        for layer in self.gnn_layers:
            hidden_features = layer(
                hidden_features,
                edge_index_dictionary,
            )

        graph_features = self.graph_pooling(
            hidden_features,
            graph_membership,
            number_of_graphs,
        )
        graph_features = self.graph_projection(
            graph_features
        )
        graph_features = normalize_features(
            graph_features
        )

        logits = (
            self.logit_scale
            * graph_features
            @ self.class_text_features.T
        )

        return logits, graph_features


# ----------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------
def move_graph_batch_to_device(batch):
    (
        node_features,
        node_type_ids,
        edge_index_dictionary,
        graph_membership,
        labels,
        number_of_graphs,
    ) = batch

    node_features = node_features.to(
        DEVICE,
        non_blocking=True,
    )
    node_type_ids = node_type_ids.to(
        DEVICE,
        non_blocking=True,
    )
    graph_membership = graph_membership.to(
        DEVICE,
        non_blocking=True,
    )
    labels = labels.to(
        DEVICE,
        non_blocking=True,
    )

    edge_index_dictionary = {
        relation: (
            source_indices.to(
                DEVICE,
                non_blocking=True,
            ),
            destination_indices.to(
                DEVICE,
                non_blocking=True,
            ),
        )
        for relation, (
            source_indices,
            destination_indices,
        ) in edge_index_dictionary.items()
    }

    return (
        node_features,
        node_type_ids,
        edge_index_dictionary,
        graph_membership,
        labels,
        number_of_graphs,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    scaler: GradScaler,
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc="Training graph encoder",
        leave=False,
    )

    for step, batch in enumerate(
        progress,
        start=1,
    ):
        (
            node_features,
            node_type_ids,
            edge_index_dictionary,
            graph_membership,
            labels,
            number_of_graphs,
        ) = move_graph_batch_to_device(batch)

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(enabled=USE_AMP):
            logits, _ = model(
                node_features=node_features,
                node_type_ids=node_type_ids,
                edge_index_dictionary=(
                    edge_index_dictionary
                ),
                graph_membership=(
                    graph_membership
                ),
                number_of_graphs=(
                    number_of_graphs
                ),
            )
            loss = criterion(
                logits,
                labels,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM,
        )

        scaler.step(optimizer)
        scaler.update()

        predictions = logits.argmax(dim=1)

        running_loss += (
            loss.item()
            * labels.size(0)
        )
        correct += (
            predictions == labels
        ).sum().item()
        total += labels.size(0)

        progress.set_postfix(
            loss=(
                f"{running_loss / max(total, 1):.4f}"
            ),
            accuracy=(
                f"{100.0 * correct / max(total, 1):.2f}%"
            ),
        )

    return (
        running_loss / max(total, 1),
        correct / max(total, 1),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions: List[int] = []
    all_labels: List[int] = []

    for batch in tqdm(
        loader,
        desc="Evaluating",
        leave=False,
    ):
        (
            node_features,
            node_type_ids,
            edge_index_dictionary,
            graph_membership,
            labels,
            number_of_graphs,
        ) = move_graph_batch_to_device(batch)

        with autocast(enabled=USE_AMP):
            logits, _ = model(
                node_features=node_features,
                node_type_ids=node_type_ids,
                edge_index_dictionary=(
                    edge_index_dictionary
                ),
                graph_membership=(
                    graph_membership
                ),
                number_of_graphs=(
                    number_of_graphs
                ),
            )
            loss = criterion(
                logits,
                labels,
            )

        predictions = logits.argmax(dim=1)

        running_loss += (
            loss.item()
            * labels.size(0)
        )
        correct += (
            predictions == labels
        ).sum().item()
        total += labels.size(0)

        all_predictions.extend(
            predictions.cpu().tolist()
        )
        all_labels.extend(
            labels.cpu().tolist()
        )

    if total == 0:
        raise RuntimeError(
            "No graphs were evaluated."
        )

    accuracy = correct / total
    average_loss = running_loss / total
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return (
        accuracy,
        average_loss,
        macro_f1,
    )


@torch.no_grad()
def prediction_distribution_diagnostic(
    model: nn.Module,
    loader: DataLoader,
    index_to_label: Dict[int, str],
):
    model.eval()

    predicted_counts = Counter()
    true_counts = Counter()

    for batch in loader:
        (
            node_features,
            node_type_ids,
            edge_index_dictionary,
            graph_membership,
            labels,
            number_of_graphs,
        ) = move_graph_batch_to_device(batch)

        logits, _ = model(
            node_features=node_features,
            node_type_ids=node_type_ids,
            edge_index_dictionary=(
                edge_index_dictionary
            ),
            graph_membership=(
                graph_membership
            ),
            number_of_graphs=(
                number_of_graphs
            ),
        )

        predictions = logits.argmax(
            dim=1
        ).cpu().tolist()

        predicted_counts.update(
            predictions
        )
        true_counts.update(
            labels.cpu().tolist()
        )

    print(
        "\nPrediction distribution diagnostic"
    )
    print(
        "Classes predicted at least once: "
        f"{len(predicted_counts)} / "
        f"{len(true_counts)}"
    )

    print("Five most frequently predicted classes:")

    for class_index, count in (
        predicted_counts.most_common(5)
    ):
        print(
            "  "
            f"{index_to_label.get(class_index, class_index)}: "
            f"{count}"
        )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    set_seed(RANDOM_STATE)
    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    (
        training_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = load_and_split_data()

    print(
        "Rows -- "
        f"train: {len(training_dataframe)}  "
        f"validation: {len(validation_dataframe)}  "
        f"test: {len(test_dataframe)}"
    )

    class_names = sorted(
        training_dataframe[LABEL_COLUMN]
        .dropna()
        .unique()
        .tolist()
    )
    label_to_index = {
        label: index
        for index, label in enumerate(
            class_names
        )
    }
    index_to_label = {
        index: label
        for label, index in label_to_index.items()
    }

    print(
        f"Candidate descriptor classes: "
        f"{len(class_names)}"
    )

    # --------------------------------------------------------------
    # Frozen PatentCLIP initialization
    # --------------------------------------------------------------
    print(
        f"Loading PatentCLIP: "
        f"{PATENTCLIP_MODEL_ID}"
    )

    (
        patentclip_model,
        _,
        patentclip_preprocess,
    ) = open_clip.create_model_and_transforms(
        PATENTCLIP_MODEL_ID,
        device=DEVICE,
    )
    tokenizer = open_clip.get_tokenizer(
        PATENTCLIP_MODEL_ID
    )

    patentclip_model.eval()

    for parameter in (
        patentclip_model.parameters()
    ):
        parameter.requires_grad = False

    if hasattr(
        patentclip_model,
        "text_projection",
    ):
        clip_embedding_dimension = int(
            patentclip_model
            .text_projection
            .shape[-1]
        )
    else:
        clip_embedding_dimension = int(
            patentclip_model
            .visual
            .output_dim
        )

    prompts = [
        PROMPT_TEMPLATE.format(
            class_name
        )
        for class_name in class_names
    ]

    with torch.no_grad():
        class_tokens = tokenizer(
            prompts
        ).to(DEVICE)
        class_text_features = (
            patentclip_model
            .encode_text(class_tokens)
            .float()
        )
        class_text_features = (
            normalize_features(
                class_text_features
            )
            .cpu()
        )
        patentclip_logit_scale = (
            patentclip_model
            .logit_scale
            .exp()
            .detach()
            .float()
            .cpu()
            .clamp(max=100.0)
        )

    print(
        f"PatentCLIP embedding dimension: "
        f"{clip_embedding_dimension}"
    )
    print(
        f"Prompt template: {PROMPT_TEMPLATE}"
    )

    # --------------------------------------------------------------
    # One-time graph node embedding
    # --------------------------------------------------------------
    all_rows = pd.concat(
        [
            training_dataframe,
            validation_dataframe,
            test_dataframe,
        ],
        ignore_index=True,
    )

    (
        graphs_by_row_id,
        figure_paths,
        crop_requests,
        text_values,
    ) = collect_graph_requests(all_rows)

    figure_cache = embed_figure_paths(
        image_paths=figure_paths,
        patentclip_model=(
            patentclip_model
        ),
        image_transform=(
            patentclip_preprocess
        ),
    )
    crop_cache = embed_bbox_requests(
        crop_requests=crop_requests,
        patentclip_model=(
            patentclip_model
        ),
        image_transform=(
            patentclip_preprocess
        ),
    )
    text_cache = embed_text_values(
        texts=text_values,
        patentclip_model=(
            patentclip_model
        ),
        tokenizer=tokenizer,
    )

    # PatentCLIP is no longer needed on GPU after all node and class
    # features have been computed.
    del patentclip_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------------
    # Graph datasets
    # --------------------------------------------------------------
    print("\nBuilding graph datasets")

    training_dataset = PatentGraphDataset(
        dataframe=training_dataframe,
        graphs_by_row_id=graphs_by_row_id,
        label_to_index=label_to_index,
        figure_cache=figure_cache,
        crop_cache=crop_cache,
        text_cache=text_cache,
        embedding_dimension=(
            clip_embedding_dimension
        ),
    )
    validation_dataset = PatentGraphDataset(
        dataframe=validation_dataframe,
        graphs_by_row_id=graphs_by_row_id,
        label_to_index=label_to_index,
        figure_cache=figure_cache,
        crop_cache=crop_cache,
        text_cache=text_cache,
        embedding_dimension=(
            clip_embedding_dimension
        ),
    )
    test_dataset = PatentGraphDataset(
        dataframe=test_dataframe,
        graphs_by_row_id=graphs_by_row_id,
        label_to_index=label_to_index,
        figure_cache=figure_cache,
        crop_cache=crop_cache,
        text_cache=text_cache,
        embedding_dimension=(
            clip_embedding_dimension
        ),
    )

    print(
        "Usable graphs -- "
        f"train: {len(training_dataset)}  "
        f"validation: {len(validation_dataset)}  "
        f"test: {len(test_dataset)}"
    )

    if len(training_dataset) == 0:
        raise RuntimeError(
            "No usable training graphs were built."
        )

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_graph_batch,
        pin_memory=DEVICE == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_graph_batch,
        pin_memory=DEVICE == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_graph_batch,
        pin_memory=DEVICE == "cuda",
    )

    # --------------------------------------------------------------
    # Loss
    # --------------------------------------------------------------
    if USE_CLASS_WEIGHTING:
        label_counts = np.bincount(
            training_dataset.labels,
            minlength=len(class_names),
        )

        class_weights = torch.ones(
            len(class_names),
            dtype=torch.float32,
        )

        for class_index, frequency in enumerate(
            label_counts
        ):
            class_weights[class_index] = (
                1.0
                / max(
                    int(frequency),
                    1,
                )
                ** CLASS_WEIGHT_BETA
            )

        # Normalize so the mean active class weight is approximately 1.
        active_mask = label_counts > 0
        class_weights[active_mask] /= (
            class_weights[active_mask].mean()
        )

        class_weights = class_weights.to(
            DEVICE
        )

        print(
            "Class weighting enabled "
            f"(beta={CLASS_WEIGHT_BETA})."
        )
    else:
        class_weights = None
        print(
            "Class weighting disabled."
        )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------------
    # Graph-to-prompt model
    # --------------------------------------------------------------
    model = PatentCLIPGraphPromptClassifier(
        clip_embedding_dimension=(
            clip_embedding_dimension
        ),
        hidden_dimension=HIDDEN_DIM,
        class_text_features=(
            class_text_features
        ),
        patentclip_logit_scale=(
            patentclip_logit_scale
        ),
        number_of_node_types=len(
            NODE_TYPE_TO_INDEX
        ),
        number_of_layers=(
            NUM_GNN_LAYERS
        ),
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )
    )
    scaler = GradScaler(
        enabled=USE_AMP
    )

    best_validation_accuracy = -1.0
    epochs_without_improvement = 0

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------
    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        (
            training_loss,
            training_accuracy,
        ) = train_one_epoch(
            model=model,
            loader=training_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
        )

        (
            validation_accuracy,
            validation_loss,
            validation_macro_f1,
        ) = evaluate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
        )

        scheduler.step(
            validation_accuracy
        )

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"train_loss={training_loss:.4f} | "
            f"train_acc={training_accuracy * 100:.2f}% | "
            f"val_loss={validation_loss:.4f} | "
            f"val_acc={validation_accuracy * 100:.2f}% | "
            f"val_macro_f1={validation_macro_f1:.4f} | "
            f"lr={current_learning_rate:.2e}"
        )

        if (
            validation_accuracy
            > best_validation_accuracy
            + MIN_DELTA
        ):
            best_validation_accuracy = (
                validation_accuracy
            )
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "label_to_index": (
                        label_to_index
                    ),
                    "index_to_label": (
                        index_to_label
                    ),
                    "class_names": class_names,
                    "label_column": LABEL_COLUMN,
                    "prompt_template": (
                        PROMPT_TEMPLATE
                    ),
                    "patentclip_model_id": (
                        PATENTCLIP_MODEL_ID
                    ),
                    "clip_embedding_dimension": (
                        clip_embedding_dimension
                    ),
                    "hidden_dimension": (
                        HIDDEN_DIM
                    ),
                    "number_of_gnn_layers": (
                        NUM_GNN_LAYERS
                    ),
                    "relations": ALL_RELATIONS,
                    "node_type_to_index": (
                        NODE_TYPE_TO_INDEX
                    ),
                    "best_validation_accuracy": (
                        best_validation_accuracy
                    ),
                },
                CHECKPOINT_PATH,
            )

            print(
                "  -> saved best checkpoint "
                f"(val_acc="
                f"{best_validation_accuracy * 100:.2f}%)"
            )
        else:
            epochs_without_improvement += 1

            print(
                "  -> no validation improvement "
                f"({epochs_without_improvement}/"
                f"{PATIENCE})"
            )

            if (
                epochs_without_improvement
                >= PATIENCE
            ):
                print(
                    f"Early stopping after "
                    f"epoch {epoch}."
                )
                break

    # --------------------------------------------------------------
    # Best-checkpoint test evaluation
    # --------------------------------------------------------------
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    if (
        checkpoint.get(
            "patentclip_model_id"
        )
        != PATENTCLIP_MODEL_ID
    ):
        raise RuntimeError(
            "Checkpoint PatentCLIP model mismatch."
        )

    if (
        checkpoint.get("class_names")
        != class_names
    ):
        raise RuntimeError(
            "Checkpoint class order does not "
            "match the current dataset."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    (
        test_accuracy,
        test_loss,
        test_macro_f1,
    ) = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
    )

    print(
        "\nFinal PatentCLIP graph-to-prompt results"
    )
    print(
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )
    print(
        f"Test macro-F1: "
        f"{test_macro_f1:.4f}"
    )
    print(
        f"Test loss: "
        f"{test_loss:.4f}"
    )
    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy * 100:.2f}%"
    )
    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    prediction_distribution_diagnostic(
        model=model,
        loader=test_loader,
        index_to_label=index_to_label,
    )


if __name__ == "__main__":
    main()
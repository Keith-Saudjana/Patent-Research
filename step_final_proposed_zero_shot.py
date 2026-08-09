"""
Zero-shot graph-based PatentCLIP classification for the IMPACT dataset.

This script restructures the image-only zero-shot baseline so that each
patent is represented by its semantic graph in `detection_graph_json`.

Zero-shot requirement
---------------------
No IMPACT labels are used to train model parameters. Therefore, this
script does NOT use a trainable GNN. A randomly initialized GNN would
destroy PatentCLIP's embedding space, while a trained GNN would make the
experiment supervised rather than zero-shot.

Instead, the script uses:
1. Frozen PatentCLIP image/text encoders for graph-node features.
2. Parameter-free relation-aware message passing.
3. Parameter-free type-balanced graph pooling.
4. Frozen descriptor-prompt embeddings.
5. Cosine similarity for classification.

Graph node features
-------------------
- figure nodes  -> PatentCLIP image embeddings
- bbox nodes    -> PatentCLIP embeddings of cropped regions
- title nodes   -> PatentCLIP text embeddings
- keyword nodes -> PatentCLIP text embeddings

Graph relations
---------------
- contains_figure
- contains_top_level_bbox
- contains_bbox
- classified_as

Reverse relations are added automatically so information can propagate
in both directions.

This is graph-only at the patent-sample level: the classifier receives a
semantic graph rather than one representative patent image. However, the
graph remains multimodal because its nodes contain image- and text-derived
PatentCLIP features.

Dependencies
------------
pip install open_clip_torch networkx pandas scikit-learn pillow tqdm
"""

import ast
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATASET_DIR = "/home/yishin/keith/patent_research/model_io"
INPUT_CSV_PATH = os.path.join(
    DATASET_DIR,
    "5_Impact_Sub_Final.csv",
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

# Parameter-free graph propagation.
NUM_PROPAGATION_LAYERS = 2
NEIGHBOR_WEIGHT = 0.50

# Type-balanced pooling first averages nodes within each node type and
# then averages the available node-type representations. This prevents
# a graph with many bbox nodes from overwhelming title/keyword nodes.
TYPE_BALANCED_POOLING = True

EMBED_BATCH_SIZE = 64
EVALUATION_BATCH_SIZE = 32
NUM_WORKERS = 0
RANDOM_STATE = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
# Data helpers
# ----------------------------------------------------------------------
def try_literal_eval(value):
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


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)


def resolve_existing_path(path_value) -> Optional[str]:
    if not has_value(path_value):
        return None

    raw_path = os.path.expanduser(
        str(path_value).strip()
    )

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
):
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


def load_and_split():
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
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # Retained for direct comparison with the other IMPACT scripts.
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

    # Keep a unique identifier after split indices are reset.
    filtered_dataframe["_row_id"] = np.arange(
        len(filtered_dataframe)
    )

    print(
        "Rows after graph + rare-class filtering: "
        f"{len(filtered_dataframe)}"
    )
    print(
        "Classes: "
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
    if isinstance(graph_value, nx.Graph):
        return graph_value.copy()

    if isinstance(graph_value, str):
        graph_data = json.loads(graph_value)
    elif isinstance(graph_value, dict):
        graph_data = graph_value
    else:
        raise TypeError(
            "detection_graph_json must be a JSON string or dictionary."
        )

    edge_key = (
        "edges"
        if "edges" in graph_data
        else "links"
    )

    try:
        return nx.node_link_graph(
            graph_data,
            edges=edge_key,
        )
    except TypeError:
        return nx.node_link_graph(
            graph_data,
            link=edge_key,
        )


def node_text(node_data: dict) -> str:
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


def valid_bbox_tuple(
    value,
) -> Optional[Tuple[int, int, int, int]]:
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


def figure_image_path_for_bbox(
    graph: nx.Graph,
    bbox_node_id,
    bbox_node_data: dict,
) -> Optional[str]:
    figure_name = bbox_node_data.get("figure")

    if has_value(figure_name):
        figure_node_id = f"figure::{figure_name}"

        if figure_node_id in graph.nodes:
            resolved_path = resolve_existing_path(
                graph.nodes[
                    figure_node_id
                ].get("image_path")
            )

            if resolved_path is not None:
                return resolved_path

    # Fallback: inspect connected figure nodes.
    for neighbor_id in graph.neighbors(
        bbox_node_id
    ):
        neighbor_data = graph.nodes[
            neighbor_id
        ]

        if neighbor_data.get(
            "node_type"
        ) == "figure":
            resolved_path = resolve_existing_path(
                neighbor_data.get("image_path")
            )

            if resolved_path is not None:
                return resolved_path

    return None


def collect_graph_requests(
    dataframe: pd.DataFrame,
):
    graphs_by_row_id: Dict[int, nx.Graph] = {}
    figure_paths: List[str] = []
    crop_requests: List[
        Tuple[str, Tuple[int, int, int, int]]
    ] = []
    text_values: List[str] = []

    invalid_graphs = 0

    for _, row in dataframe.iterrows():
        row_id = int(row["_row_id"])

        try:
            graph = load_graph(
                row[GRAPH_COLUMN]
            )
        except Exception:
            invalid_graphs += 1
            continue

        graphs_by_row_id[row_id] = graph

        for node_id, node_data in graph.nodes(
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
                image_path = figure_image_path_for_bbox(
                    graph,
                    node_id,
                    node_data,
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

    print(f"Parsed graphs: {len(graphs_by_row_id)}")
    print(f"Invalid graphs skipped: {invalid_graphs}")
    print(
        f"Unique figure images: "
        f"{len(set(figure_paths))}"
    )
    print(
        f"Unique bbox crops: "
        f"{len(set(crop_requests))}"
    )
    print(
        f"Unique text nodes: "
        f"{len(set(text_values))}"
    )

    return (
        graphs_by_row_id,
        figure_paths,
        crop_requests,
        text_values,
    )


# ----------------------------------------------------------------------
# Frozen PatentCLIP feature extraction
# ----------------------------------------------------------------------
@torch.no_grad()
def embed_text_values(
    texts: Sequence[str],
    model,
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

        tokens = tokenizer(
            batch_texts
        ).to(DEVICE)

        features = model.encode_text(
            tokens
        ).float()
        features = normalize_features(
            features
        ).cpu()

        for text, feature in zip(
            batch_texts,
            features,
        ):
            cache[text] = feature

    return cache


@torch.no_grad()
def embed_figure_paths(
    image_paths: Sequence[str],
    model,
    preprocess,
) -> Dict[str, torch.Tensor]:
    cache: Dict[str, torch.Tensor] = {}
    unique_paths = list(
        dict.fromkeys(image_paths)
    )

    for start in tqdm(
        range(0, len(unique_paths), EMBED_BATCH_SIZE),
        desc="Encoding figure nodes",
    ):
        batch_paths = unique_paths[
            start : start + EMBED_BATCH_SIZE
        ]

        valid_paths: List[str] = []
        image_tensors: List[torch.Tensor] = []

        for image_path in batch_paths:
            try:
                with Image.open(
                    image_path
                ) as image:
                    tensor = preprocess(
                        image.convert("RGB")
                    )

                valid_paths.append(image_path)
                image_tensors.append(tensor)
            except Exception:
                continue

        if not image_tensors:
            continue

        image_batch = torch.stack(
            image_tensors
        ).to(DEVICE)

        features = model.encode_image(
            image_batch
        ).float()
        features = normalize_features(
            features
        ).cpu()

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
    model,
    preprocess,
):
    cache = {}
    unique_requests = list(
        dict.fromkeys(crop_requests)
    )

    for start in tqdm(
        range(0, len(unique_requests), EMBED_BATCH_SIZE),
        desc="Encoding bbox nodes",
    ):
        batch_requests = unique_requests[
            start : start + EMBED_BATCH_SIZE
        ]

        valid_requests = []
        crop_tensors = []

        for image_path, bbox in batch_requests:
            try:
                with Image.open(
                    image_path
                ) as image:
                    crop = (
                        image.convert("RGB")
                        .crop(bbox)
                    )
                    tensor = preprocess(crop)

                valid_requests.append(
                    (image_path, bbox)
                )
                crop_tensors.append(tensor)
            except Exception:
                continue

        if not crop_tensors:
            continue

        crop_batch = torch.stack(
            crop_tensors
        ).to(DEVICE)

        features = model.encode_image(
            crop_batch
        ).float()
        features = normalize_features(
            features
        ).cpu()

        for request, feature in zip(
            valid_requests,
            features,
        ):
            cache[request] = feature

    return cache


# ----------------------------------------------------------------------
# Graph tensor construction
# ----------------------------------------------------------------------
def build_graph_tensors(
    graph: nx.Graph,
    figure_cache: Dict[str, torch.Tensor],
    crop_cache,
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
    valid_node_mask = torch.zeros(
        len(node_ids),
        dtype=torch.bool,
    )

    for node_id, node_data in graph.nodes(
        data=True
    ):
        index = node_id_to_index[node_id]
        node_type = node_data.get(
            "node_type"
        )

        node_type_ids[index] = (
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
                feature = figure_cache[
                    image_path
                ]

        elif node_type == "bbox":
            image_path = figure_image_path_for_bbox(
                graph,
                node_id,
                node_data,
            )
            bbox = valid_bbox_tuple(
                node_data.get("bbox")
            )

            request = (
                (image_path, bbox)
                if (
                    image_path is not None
                    and bbox is not None
                )
                else None
            )

            if request in crop_cache:
                feature = crop_cache[
                    request
                ]

        elif node_type in {
            "title",
            "keyword",
        }:
            text = node_text(node_data)

            if text in text_cache:
                feature = text_cache[text]

        if feature is not None:
            node_features[index] = feature
            valid_node_mask[index] = True

    if not torch.any(valid_node_mask):
        return None

    edge_lists = {
        relation: ([], [])
        for relation in ALL_RELATIONS
    }

    for source, destination, edge_data in graph.edges(
        data=True
    ):
        relation = edge_data.get(
            "edge_type"
        )

        if relation not in BASE_RELATIONS:
            continue

        source_index = node_id_to_index[
            source
        ]
        destination_index = node_id_to_index[
            destination
        ]

        # Only keep edges whose endpoints both have usable features.
        if not (
            valid_node_mask[source_index]
            and valid_node_mask[destination_index]
        ):
            continue

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
        sources, destinations = (
            edge_lists[relation]
        )

        edge_index_dictionary[relation] = (
            torch.tensor(
                sources,
                dtype=torch.long,
            ),
            torch.tensor(
                destinations,
                dtype=torch.long,
            ),
        )

    return (
        node_features,
        node_type_ids,
        valid_node_mask,
        edge_index_dictionary,
    )


class PatentGraphDataset(Dataset):
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
            graph = graphs_by_row_id.get(
                row_id
            )
            label = row[LABEL_COLUMN]

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
                embedding_dimension=embedding_dimension,
            )

            if graph_tensors is None:
                skipped += 1
                continue

            (
                node_features,
                node_type_ids,
                valid_node_mask,
                edge_index_dictionary,
            ) = graph_tensors

            self.items.append(
                (
                    node_features,
                    node_type_ids,
                    valid_node_mask,
                    edge_index_dictionary,
                    label_to_index[label],
                )
            )

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
    all_valid_masks = []
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
        valid_node_mask,
        edge_index_dictionary,
        label,
    ) in enumerate(batch):
        number_of_nodes = node_features.size(
            0
        )

        all_node_features.append(
            node_features
        )
        all_node_type_ids.append(
            node_type_ids
        )
        all_valid_masks.append(
            valid_node_mask
        )
        graph_membership.append(
            torch.full(
                (number_of_nodes,),
                graph_index,
                dtype=torch.long,
            )
        )
        labels.append(label)

        for relation, (
            sources,
            destinations,
        ) in edge_index_dictionary.items():
            if sources.numel() == 0:
                continue

            batched_edges[relation][0].append(
                sources + node_offset
            )
            batched_edges[relation][1].append(
                destinations + node_offset
            )

        node_offset += number_of_nodes

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
        torch.cat(
            all_node_features,
            dim=0,
        ),
        torch.cat(
            all_node_type_ids,
            dim=0,
        ),
        torch.cat(
            all_valid_masks,
            dim=0,
        ),
        edge_index_batch,
        torch.cat(
            graph_membership,
            dim=0,
        ),
        torch.tensor(
            labels,
            dtype=torch.long,
        ),
        len(batch),
    )


# ----------------------------------------------------------------------
# Parameter-free zero-shot graph encoder
# ----------------------------------------------------------------------
@torch.no_grad()
def relation_aware_propagation(
    node_features: torch.Tensor,
    valid_node_mask: torch.Tensor,
    edge_index_dictionary,
    number_of_layers: int,
    neighbor_weight: float,
) -> torch.Tensor:
    """
    Parameter-free relation-aware message passing.

    For each relation:
      - average source-node features arriving at each destination;
      - treat the relation mean as one message channel.

    Across relations:
      - average all relation channels available at a node.

    Final update:
      h_new = normalize(
          (1 - neighbor_weight) * h
          + neighbor_weight * neighbor_mean
      )

    No weights are learned.
    """
    hidden = normalize_features(
        node_features
    )

    # Ensure missing nodes remain zero.
    hidden = hidden * valid_node_mask.unsqueeze(
        -1
    )

    number_of_nodes = hidden.size(0)

    for _ in range(number_of_layers):
        relation_sum = torch.zeros_like(
            hidden
        )
        relation_count = torch.zeros(
            number_of_nodes,
            dtype=hidden.dtype,
            device=hidden.device,
        )

        for sources, destinations in (
            edge_index_dictionary.values()
        ):
            if sources.numel() == 0:
                continue

            messages = hidden[sources]

            destination_sum = torch.zeros_like(
                hidden
            )
            destination_sum.index_add_(
                0,
                destinations,
                messages,
            )

            destination_degree = torch.zeros(
                number_of_nodes,
                dtype=hidden.dtype,
                device=hidden.device,
            )
            destination_degree.index_add_(
                0,
                destinations,
                torch.ones(
                    destinations.size(0),
                    dtype=hidden.dtype,
                    device=hidden.device,
                ),
            )

            has_relation_message = (
                destination_degree > 0
            )

            relation_mean = (
                destination_sum
                / destination_degree
                .clamp_min(1.0)
                .unsqueeze(-1)
            )

            relation_sum += relation_mean
            relation_count += (
                has_relation_message.float()
            )

        has_neighbor = relation_count > 0

        neighbor_mean = (
            relation_sum
            / relation_count
            .clamp_min(1.0)
            .unsqueeze(-1)
        )

        updated = hidden.clone()
        updated[has_neighbor] = (
            (1.0 - neighbor_weight)
            * hidden[has_neighbor]
            + neighbor_weight
            * neighbor_mean[has_neighbor]
        )

        hidden = normalize_features(updated)
        hidden = (
            hidden
            * valid_node_mask.unsqueeze(-1)
        )

    return hidden


@torch.no_grad()
def pool_graph_embeddings(
    node_features: torch.Tensor,
    node_type_ids: torch.Tensor,
    valid_node_mask: torch.Tensor,
    graph_membership: torch.Tensor,
    number_of_graphs: int,
) -> torch.Tensor:
    graph_embeddings = []

    for graph_index in range(
        number_of_graphs
    ):
        graph_mask = (
            graph_membership == graph_index
        ) & valid_node_mask

        if not torch.any(graph_mask):
            raise RuntimeError(
                "A graph batch contains no valid nodes."
            )

        graph_node_features = (
            node_features[graph_mask]
        )
        graph_node_types = (
            node_type_ids[graph_mask]
        )

        if TYPE_BALANCED_POOLING:
            type_embeddings = []

            for node_type_index in (
                NODE_TYPE_TO_INDEX.values()
            ):
                type_mask = (
                    graph_node_types
                    == node_type_index
                )

                if torch.any(type_mask):
                    type_embedding = (
                        graph_node_features[
                            type_mask
                        ].mean(dim=0)
                    )
                    type_embeddings.append(
                        normalize_features(
                            type_embedding.unsqueeze(0)
                        ).squeeze(0)
                    )

            graph_embedding = torch.stack(
                type_embeddings,
                dim=0,
            ).mean(dim=0)
        else:
            graph_embedding = (
                graph_node_features.mean(
                    dim=0
                )
            )

        graph_embeddings.append(
            normalize_features(
                graph_embedding.unsqueeze(0)
            ).squeeze(0)
        )

    return torch.stack(
        graph_embeddings,
        dim=0,
    )


# ----------------------------------------------------------------------
# Prompt encoding and evaluation
# ----------------------------------------------------------------------
@torch.no_grad()
def encode_class_prompts(
    model,
    tokenizer,
    class_names: Sequence[str],
):
    prompts = [
        PROMPT_TEMPLATE.format(
            str(class_name).strip()
        )
        for class_name in class_names
    ]

    tokens = tokenizer(
        prompts
    ).to(DEVICE)

    text_features = model.encode_text(
        tokens
    ).float()
    text_features = normalize_features(
        text_features
    )

    return text_features, prompts


@torch.no_grad()
def evaluate_zero_shot_graphs(
    loader: DataLoader,
    class_text_features: torch.Tensor,
    logit_scale: torch.Tensor,
):
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    all_predictions = []
    all_labels = []

    for batch in tqdm(
        loader,
        desc="Zero-shot graph evaluation",
    ):
        (
            node_features,
            node_type_ids,
            valid_node_mask,
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
        valid_node_mask = valid_node_mask.to(
            DEVICE,
            non_blocking=True,
        )
        graph_membership = (
            graph_membership.to(
                DEVICE,
                non_blocking=True,
            )
        )
        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        edge_index_dictionary = {
            relation: (
                sources.to(
                    DEVICE,
                    non_blocking=True,
                ),
                destinations.to(
                    DEVICE,
                    non_blocking=True,
                ),
            )
            for relation, (
                sources,
                destinations,
            ) in edge_index_dictionary.items()
        }

        propagated_nodes = (
            relation_aware_propagation(
                node_features=node_features,
                valid_node_mask=valid_node_mask,
                edge_index_dictionary=(
                    edge_index_dictionary
                ),
                number_of_layers=(
                    NUM_PROPAGATION_LAYERS
                ),
                neighbor_weight=(
                    NEIGHBOR_WEIGHT
                ),
            )
        )

        graph_features = pool_graph_embeddings(
            node_features=propagated_nodes,
            node_type_ids=node_type_ids,
            valid_node_mask=valid_node_mask,
            graph_membership=(
                graph_membership
            ),
            number_of_graphs=(
                number_of_graphs
            ),
        )

        logits = (
            logit_scale
            * graph_features
            @ class_text_features.T
        )

        top5_predictions = logits.topk(
            k=min(
                5,
                logits.shape[1],
            ),
            dim=1,
        ).indices
        predictions = top5_predictions[:, 0]

        correct_top1 += (
            predictions == labels
        ).sum().item()

        correct_top5 += (
            top5_predictions
            == labels.unsqueeze(1)
        ).any(dim=1).sum().item()

        total += labels.size(0)

        all_predictions.extend(
            predictions.cpu().tolist()
        )
        all_labels.extend(
            labels.cpu().tolist()
        )

    if total == 0:
        raise RuntimeError(
            "No usable graphs were evaluated."
        )

    top1_accuracy = (
        correct_top1 / total
    )
    top5_accuracy = (
        correct_top5 / total
    )
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return (
        top1_accuracy,
        top5_accuracy,
        macro_f1,
        total,
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    set_seed(RANDOM_STATE)

    train_dataframe, _, test_dataframe = (
        load_and_split()
    )

    print(
        "Rows -- "
        f"train: {len(train_dataframe)}  "
        f"test: {len(test_dataframe)}"
    )

    # Training rows are used only to define the candidate class list.
    # No model parameters are learned from them.
    class_names = sorted(
        train_dataframe[LABEL_COLUMN]
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

    print(
        "Candidate descriptor classes: "
        f"{len(class_names)}"
    )

    print(
        f"Loading PatentCLIP: "
        f"{PATENTCLIP_MODEL_ID}"
    )

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            PATENTCLIP_MODEL_ID,
            device=DEVICE,
        )
    )
    tokenizer = open_clip.get_tokenizer(
        PATENTCLIP_MODEL_ID
    )
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    if hasattr(
        model,
        "text_projection",
    ):
        embedding_dimension = int(
            model.text_projection.shape[-1]
        )
    else:
        embedding_dimension = int(
            model.visual.output_dim
        )

    class_text_features, prompts = (
        encode_class_prompts(
            model,
            tokenizer,
            class_names,
        )
    )
    logit_scale = (
        model.logit_scale
        .exp()
        .float()
        .clamp(max=100.0)
    )

    print(
        f"PatentCLIP embedding dimension: "
        f"{embedding_dimension}"
    )
    print(
        "Example class prompt: "
        f"{prompts[0] if prompts else 'N/A'}"
    )

    # Only the test graphs must be embedded for zero-shot evaluation.
    (
        graphs_by_row_id,
        figure_paths,
        crop_requests,
        text_values,
    ) = collect_graph_requests(
        test_dataframe
    )

    figure_cache = embed_figure_paths(
        image_paths=figure_paths,
        model=model,
        preprocess=preprocess,
    )
    crop_cache = embed_bbox_requests(
        crop_requests=crop_requests,
        model=model,
        preprocess=preprocess,
    )
    text_cache = embed_text_values(
        texts=text_values,
        model=model,
        tokenizer=tokenizer,
    )

    test_dataset = PatentGraphDataset(
        dataframe=test_dataframe,
        graphs_by_row_id=graphs_by_row_id,
        label_to_index=label_to_index,
        figure_cache=figure_cache,
        crop_cache=crop_cache,
        text_cache=text_cache,
        embedding_dimension=(
            embedding_dimension
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_graph_batch,
        pin_memory=DEVICE == "cuda",
    )

    print(
        f"Usable test graphs: "
        f"{len(test_dataset)}"
    )
    print(
        "Graph propagation: "
        f"{NUM_PROPAGATION_LAYERS} layers, "
        f"neighbor weight={NEIGHBOR_WEIGHT}"
    )
    print(
        "Graph pooling: "
        + (
            "type-balanced mean pooling"
            if TYPE_BALANCED_POOLING
            else "ordinary node mean pooling"
        )
    )

    (
        top1_accuracy,
        top5_accuracy,
        macro_f1,
        total,
    ) = evaluate_zero_shot_graphs(
        loader=test_loader,
        class_text_features=(
            class_text_features
        ),
        logit_scale=logit_scale,
    )

    print(
        "\nFinal zero-shot PatentCLIP graph results"
    )
    print(f"Evaluated graphs: {total}")
    print(
        f"Top-1 accuracy: "
        f"{top1_accuracy * 100:.2f}%"
    )
    print(
        f"Top-5 accuracy: "
        f"{top5_accuracy * 100:.2f}%"
    )
    print(
        f"Macro-F1: {macro_f1:.4f}"
    )


if __name__ == "__main__":
    main()
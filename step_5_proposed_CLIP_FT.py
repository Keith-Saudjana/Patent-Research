import sys
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import open_clip
import os

from PIL import Image
from tqdm import tqdm
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# --- Args ---
if len(sys.argv) != 3:
    print("Usage: python script.py <input_csv> <output_dir>")
    sys.exit(1)

INPUT_FILE = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[Config] Input : {INPUT_FILE}")
print(f"[Config] Output: {OUTPUT_DIR}")

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Helpers ---
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

# --- Data ---
df = pd.read_csv(INPUT_FILE, encoding="utf-8")
df = df.map(try_literal_eval)

df_clean = df[["file_names", "Qwen", "main_class"]].dropna()
df_clean["image_path"] = df_clean["file_names"].apply(lambda x: x[0] if isinstance(x, list) else x)
df_clean["triples"] = df_clean["Qwen"]
df_clean["label"] = df_clean["main_class"]
df_clean = df_clean[["image_path", "triples", "label"]].copy()

def normalize_triple(t):
    if not isinstance(t, (list, tuple)):
        return None
    if len(t) == 3:
        return tuple(t)
    if len(t) == 2:
        subject, relation = t
        if relation in ["not_visible", "not mentioned", "not_mentioned"]:
            return (subject, "visibility", relation)
        return (subject, "relation", relation)
    if len(t) == 4:
        subject, relation_type, predicate, obj = t
        return (subject, predicate, obj)
    return None

def clean_triples(triples):
    if not isinstance(triples, list):
        return []
    return [n for t in triples if (n := normalize_triple(t)) is not None]

df_clean["triples"] = df_clean["triples"].apply(clean_triples)

min_count = 2
label_counts = df_clean["label"].value_counts()
valid_labels = label_counts[label_counts >= min_count].index
df_clean = df_clean[df_clean["label"].isin(valid_labels)]

# Remap labels to contiguous 0-based indices for the classifier head
unique_labels = sorted(df_clean["label"].unique())
label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
idx_to_label  = {i: l for l, i in label_to_idx.items()}
df_clean["label_idx"] = df_clean["label"].map(label_to_idx)
num_classes = len(unique_labels)

print(f"Removed {(label_counts < min_count).sum()} singleton classes")
print(f"Remaining samples: {len(df_clean)}, classes: {num_classes}")

train_df, temp_df = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean["label"])
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

# --- Locarno ---
LOCARNO_CATEGORIES = {
    1: "Foodstuffs", 2: "Articles of clothing and haberdashery",
    3: "Travel goods, cases, parasols and personal belongings", 4: "Brushware",
    5: "Textile piecegoods, artificial and natural sheet material", 6: "Furnishing",
    7: "Household goods", 8: "Tools and hardware",
    9: "Packages and containers for the transport or handling of goods",
    10: "Clocks and watches and other measuring instruments",
    11: "Articles of adornment", 12: "Means of transport or hoisting",
    13: "Equipment for production, distribution or transformation of electricity",
    14: "Recording, communication or information retrieval equipment",
    15: "Machines, not elsewhere specified",
    16: "Photographic, cinematographic and optical equipment",
    17: "Musical instruments", 18: "Printing and office machinery",
    19: "Stationery and office equipment", 20: "Sales and advertising equipment",
    21: "Games, toys, tents and sports goods",
    22: "Arms, pyrotechnic articles, articles for hunting and fishing",
    23: "Fluid distribution equipment", 24: "Medical and laboratory equipment",
    25: "Building units and construction elements", 26: "Lighting apparatus",
    27: "Tobacco products and smokers' supplies",
    28: "Pharmaceutical and cosmetic products", 29: "Devices for handling of fire",
    30: "Articles for the care and handling of animals",
    31: "Machines and appliances for preparing food or drink",
    32: "Graphic symbols and logos, surface patterns, ornamentation",
}
ID_TO_CATEGORY = {k: f"Class {k:02d}: {v}" for k, v in LOCARNO_CATEGORIES.items()}

def triples_to_text(triples: list) -> Optional[str]:
    if not triples:
        return None
    parts = []
    for triple in triples:
        try:
            s, p, o = triple
            s = " ".join(s) if isinstance(s, list) else str(s)
            p = " ".join(p) if isinstance(p, list) else str(p)
            o = " ".join(o) if isinstance(o, list) else str(o)
            parts.append(f"{s} {p.replace('_', ' ')} {o}")
        except (ValueError, TypeError):
            continue
    return ". ".join(parts) if parts else None

# --- Load PatentCLIP (frozen) ---
print("Loading PatentCLIP...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'hf-hub:patentclip/PatentCLIP_Vit_B', device=device
)
clip_tokenizer = open_clip.get_tokenizer('hf-hub:patentclip/PatentCLIP_Vit_B')
clip_model.eval()

# Freeze all CLIP parameters — only the head will be trained
for param in clip_model.parameters():
    param.requires_grad = False

# Infer CLIP embedding dimension
with torch.no_grad():
    dummy = preprocess(Image.new("RGB", (224, 224))).unsqueeze(0).to(device)
    embed_dim = clip_model.encode_image(dummy).shape[-1]

print(f"CLIP embedding dim: {embed_dim}, num classes: {num_classes}")

# --- Classification head ---
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim, num_classes):
        super().__init__()
        self.clip = clip_model
        # Fuses image + optional text embedding, then classifies
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, images, triple_tokens=None):
        with torch.no_grad():
            image_emb = self.clip.encode_image(images)
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

            if triple_tokens is not None:
                text_emb = self.clip.encode_text(triple_tokens)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                fused = (image_emb + text_emb) / 2
                fused = fused / fused.norm(dim=-1, keepdim=True)
            else:
                fused = image_emb

        return self.head(fused.float())

model = CLIPClassifier(clip_model, embed_dim, num_classes).to(device)

# --- Dataset ---
class PatentDataset(Dataset):
    def __init__(self, df, preprocess, tokenizer):
        self.samples   = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.tokenizer  = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row      = self.samples.iloc[idx]
        triples  = [tuple(t) for t in row["triples"]] if isinstance(row["triples"], list) else []
        label    = int(row["label_idx"])

        image  = self.preprocess(Image.open(row["image_path"]).convert("RGB"))

        triple_text = triples_to_text(triples)
        if triple_text:
            tokens = self.tokenizer([triple_text]).squeeze(0)
        else:
            tokens = None

        return image, tokens, label

def collate_fn(batch):
    images, tokens_list, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    # Pad token sequences; use zeros for missing triples
    if any(t is not None for t in tokens_list):
        token_dim = next(t for t in tokens_list if t is not None).shape[0]
        tokens = torch.stack([
            t if t is not None else torch.zeros(token_dim, dtype=torch.long)
            for t in tokens_list
        ])
    else:
        tokens = None
    return images, tokens, labels

# --- Evaluate ---
def evaluate(model, df, split_name):
    model.eval()
    dataset = PatentDataset(df, preprocess, clip_tokenizer)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False,
                         num_workers=2, collate_fn=collate_fn)

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, tokens, labels in tqdm(loader, desc=split_name):
            images = images.to(device)
            tokens = tokens.to(device) if tokens is not None else None

            logits = model(images, tokens)
            preds  = logits.argmax(dim=-1).cpu().tolist()
            y_pred.extend([idx_to_label[p] for p in preds])
            y_true.extend([idx_to_label[l.item()] for l in labels])

    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")

    return {"accuracy": acc, "precision": precision, "recall": recall}

# --- Train ---
def train(
    model,
    train_df, val_df, test_df,
    output_dir,
    num_epochs=10,
    batch_size=32,
    lr=1e-3,
):
    train_dataset = PatentDataset(train_df, preprocess, clip_tokenizer)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=2, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc  = 0.0
    save_path = os.path.join(output_dir, "best_model.pt")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for images, tokens, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            images = images.to(device)
            tokens = tokens.to(device) if tokens is not None else None
            labels = labels.to(device)

            logits = model(images, tokens)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"\n[Epoch {epoch}] Loss: {avg_loss:.4f}")

        print(f"[Epoch {epoch}] Validation:")
        val_results = evaluate(model, val_df, "Validation")

        scheduler.step()

        if val_results["accuracy"] > best_acc:
            best_acc = val_results["accuracy"]
            torch.save(model.state_dict(), save_path)
            print(f"  [Saved] Best model → {save_path} (acc={best_acc:.4f})")

    print("\n[Final] Loading best model for test evaluation...")
    model.load_state_dict(torch.load(save_path))

    print("[Final] Test:")
    evaluate(model, test_df, "Test")

# --- Run ---
train(
    model=model,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    output_dir=OUTPUT_DIR,
    num_epochs=10,
    batch_size=32,
    lr=1e-3,
)
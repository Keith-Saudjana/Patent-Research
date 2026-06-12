import sys
import pandas as pd
import ast
import torch
import numpy as np
from typing import Optional
from PIL import Image
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

from torch.utils.data import Dataset
from transformers import (
    LlavaNextProcessor, LlavaNextForConditionalGeneration,
    TrainingArguments, Trainer, TrainerCallback, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from tqdm import tqdm

INPUT_FILE = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

print(f"[Config] Input : {INPUT_FILE}")
print(f"[Config] Output: {OUTPUT_DIR}")

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
    return [normalized for t in triples if (normalized := normalize_triple(t)) is not None]

df_clean["triples"] = df_clean["triples"].apply(clean_triples)

min_count = 2
label_counts = df_clean["label"].value_counts()
valid_labels = label_counts[label_counts >= min_count].index
df_clean = df_clean[df_clean["label"].isin(valid_labels)]

print(f"Removed {(label_counts < min_count).sum()} singleton classes")
print(f"Remaining samples: {len(df_clean)}, classes: {df_clean['label'].nunique()}")

train_df, temp_df = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean["label"])
val_df, test_df   = train_test_split(temp_df,  test_size=0.5, random_state=42)

print(f"Train: {len(train_df)}, classes: {train_df['label'].nunique()}")
print(f"Validation: {len(val_df)}, classes: {val_df['label'].nunique()}")
print(f"Test: {len(test_df)}, classes: {test_df['label'].nunique()}")

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
CATEGORY_TO_ID = {v: k for k, v in ID_TO_CATEGORY.items()}

def format_triples(triples: list[tuple[str, str, str]]) -> str:
    if not triples:
        return "No triples available."
    return "\n".join(f"  - {s} | {p} | {o}" for s, p, o in triples)

def build_prompt(triples: list[tuple[str, str, str]], for_training: bool = False, label: Optional[int] = None) -> str:
    categories_block = "\n".join(f"  {v}" for v in ID_TO_CATEGORY.values())

    if triples:
        triples_section = (
            f"Below are semantic triples extracted from a design patent description:\n"
            f"{format_triples(triples)}\n\n"
            f"The image above is the visual representation of the same patent.\n\n"
            f"Using BOTH the image and the triples, classify this patent into one of the following Locarno categories:\n"
        )
    else:
        triples_section = (
            f"No semantic triples are available for this patent.\n\n"
            f"The image above is the visual representation of the patent.\n\n"
            f"Using the image alone, classify this patent into one of the following Locarno categories:\n"
        )

    prompt = (
        f"<image>\n"
        f"You are a patent classification expert specializing in the Locarno International Classification for Industrial Designs.\n\n"
        f"{triples_section}"
        f"{categories_block}\n\n"
        f"Respond with ONLY the exact category string (e.g., 'Class 07: Household goods').\n"
        f"Answer:"
    )

    if for_training and label is not None:
        prompt += f" {ID_TO_CATEGORY[label]}"

    return prompt

# --- Dataset ---
@dataclass
class PatentSample:
    image_path: str
    triples: list[tuple[str, str, str]]
    label: int

class LocarnoPatentDataset(Dataset):
    def __init__(self, df, processor, max_length=1024, is_train=True):
        self.processor = processor
        self.max_length = max_length
        self.is_train = is_train
        self.samples = [
            PatentSample(
                image_path=row["image_path"],
                triples=[tuple(t) for t in row["triples"]],
                label=int(row["label"]),
            )
            for _, row in df.iterrows()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        
        triples = sample.triples if sample.triples else []

        prompt = build_prompt(triples, for_training=self.is_train, label=sample.label)

        encoding = self.processor(
            text=prompt, images=image,
            return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_length,
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}

        if self.is_train:
            labels = item["input_ids"].clone()
            answer_token_ids = self.processor.tokenizer.encode("Answer:", add_special_tokens=False)
            input_ids = item["input_ids"].tolist()
            answer_start = self._find_subsequence(input_ids, answer_token_ids)
            if answer_start != -1:
                labels[:answer_start + len(answer_token_ids)] = -100
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            item["labels"] = labels

        return item

    @staticmethod
    def _find_subsequence(seq, subseq):
        for i in range(len(seq) - len(subseq) + 1):
            if seq[i:i + len(subseq)] == subseq:
                return i
        return -1

class CleanLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        clean = {}
        if "loss" in logs:
            clean["loss"] = round(logs["loss"], 4)
        if "eval_loss" in logs:
            clean["eval_loss"] = round(logs["eval_loss"], 4)
        if clean:
            print(f"[Log] Epoch {round(state.epoch, 2)} | " + " | ".join(f"{k}: {v}" for k, v in clean.items()))

# --- Per-epoch generation-based eval callback ---
class GenerationEvalCallback(TrainerCallback):
    def __init__(self, eval_df, processor, max_length=3072, max_new_tokens=20, device="cuda"):
        self.eval_df = eval_df.reset_index(drop=True)
        self.processor = processor
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.device = device

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch_label = state.epoch if isinstance(state.epoch, str) else int(state.epoch)
        print(f"\n[Eval] Epoch {epoch_label}")
        model.eval()

        y_true, y_pred = [], []

        for _, row in tqdm(self.eval_df.iterrows(), total=len(self.eval_df), desc="Gen-Eval"):
            image = Image.open(row["image_path"]).convert("RGB")
            triples = [tuple(t) for t in row["triples"]] if isinstance(row["triples"], list) else []
            label_id = int(row["label"])

            prompt = build_prompt(triples, for_training=False)
            encoding = self.processor(
                text=prompt, images=image,
                return_tensors="pt", truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **encoding,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            input_len = encoding["input_ids"].shape[1]
            generated = self.processor.tokenizer.decode(
                output_ids[0][input_len:], skip_special_tokens=True
            ).strip()

            predicted_id = CATEGORY_TO_ID.get(generated, -1)
            if predicted_id == -1:
                for cat_str, cat_id in CATEGORY_TO_ID.items():
                    if cat_str.lower() in generated.lower():
                        predicted_id = cat_id
                        break

            y_true.append(label_id)
            y_pred.append(predicted_id)

        acc       = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall    = recall_score(y_true, y_pred, average="weighted", zero_division=0)

        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")

        model.train()

# --- Model ---
def load_model_and_processor(model_id="llava-hf/llava-v1.6-vicuna-7b-hf"):
    processor = LlavaNextProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, device_map="auto",
        quantization_config=bnb_config, attn_implementation="eager",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    return model, processor

def apply_lora(model, r=16, lora_alpha=32, lora_dropout=0.05):
    lora_config = LoraConfig(
        r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

# --- Train ---
def train_lora_llava(
    train_df, val_df, test_df,
    output_dir=OUTPUT_DIR,
    model_id="llava-hf/llava-v1.6-vicuna-7b-hf",
    num_epochs=3, batch_size=1, grad_accum=8,
    lr=2e-4, max_length=3072, lora_r=8,
):
    print("Loading model...")
    model, processor = load_model_and_processor(model_id)

    print("Applying LoRA...")
    model = apply_lora(model, r=lora_r)

    print("Preparing datasets...")
    train_dataset = LocarnoPatentDataset(train_df, processor, max_length=max_length, is_train=True)
    val_dataset   = LocarnoPatentDataset(val_df,   processor, max_length=max_length, is_train=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        fp16=True, bf16=False,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        save_total_limit=2,
    )

    # Attach the per-epoch generation eval on the validation set
    gen_eval_callback = GenerationEvalCallback(
        eval_df=val_df,
        processor=processor,
        max_length=max_length,
        device="cuda",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[CleanLogCallback(), gen_eval_callback],
    )

    print("Starting LoRA fine-tuning...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    # --- Final test set evaluation ---
    print("\n[Eval] Running final evaluation on test set...")
    gen_eval_callback.eval_df = test_df.reset_index(drop=True)
    gen_eval_callback.on_epoch_end(args=training_args, state=type("S", (), {"epoch": "final"})(), control=None, model=model)

train_lora_llava(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    model_id="llava-hf/llava-v1.6-vicuna-7b-hf",
    output_dir=OUTPUT_DIR,
    num_epochs=3,
    batch_size=1,
    grad_accum=8,
    lr=2e-4,
    lora_r=8,
    max_length=3072,
)
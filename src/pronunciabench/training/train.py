"""ByT5 fine-tuning pipeline for G2P."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)


@dataclass
class TrainingConfig:
    seed: int = 42
    model_name: str = "google/byt5-small"
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 100
    max_length: int = 128
    eval_interval: int = 100
    save_interval: int = 200
    early_stopping_patience: int = 2
    output_dir: str = "artifacts/checkpoints"


@dataclass
class TrainResult:
    config: TrainingConfig
    final_loss: float
    eval_loss: float
    best_epoch: int
    checkpoint_path: str
    logs: list[dict] = field(default_factory=list)


class G2PDataset(Dataset):
    """Dataset for G2P training."""

    def __init__(self, examples: list[dict], tokenizer, max_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        text = ex.get("text", "")
        pronunciation = ex.get("pronunciation", "")
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        target_encoding = self.tokenizer(
            pronunciation, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
        }


def load_dataset_from_jsonl(path: str) -> list[dict]:
    """Load training data from JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def train_model(
    train_path: str,
    eval_path: str | None = None,
    config: TrainingConfig | None = None,
    mlflow_tracking_uri: str | None = None,
) -> TrainResult:
    """Train a ByT5 model on G2P data."""
    if config is None:
        config = TrainingConfig()

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)

    train_examples = load_dataset_from_jsonl(train_path)
    train_dataset = G2PDataset(train_examples, tokenizer, config.max_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    eval_examples = load_dataset_from_jsonl(eval_path) if eval_path else []
    eval_dataset = G2PDataset(eval_examples, tokenizer, config.max_length) if eval_examples else None
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False) if eval_dataset else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps,
    )

    if mlflow_tracking_uri:
        import mlflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.start_run()
        mlflow.log_params({"model_name": config.model_name, "epochs": config.epochs,
                           "batch_size": config.batch_size, "learning_rate": config.learning_rate,
                           "seed": config.seed})

    best_eval_loss = float("inf")
    patience_counter = 0
    logs: list[dict] = []
    avg_train_loss = 0.0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            if mlflow_tracking_uri and step % config.eval_interval == 0:
                import mlflow
                mlflow.log_metric("train/loss", total_loss / (step + 1), epoch * len(train_loader) + step)

        avg_train_loss = total_loss / len(train_loader)
        log_entry = {"epoch": epoch + 1, "train_loss": round(avg_train_loss, 4)}
        logs.append(log_entry)

        if eval_loader:
            model.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for batch in eval_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    eval_loss += out.loss.item()
            avg_eval_loss = eval_loss / len(eval_loader)
            log_entry["eval_loss"] = round(avg_eval_loss, 4)
            if mlflow_tracking_uri:
                import mlflow
                mlflow.log_metric("eval/loss", avg_eval_loss, epoch)
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                patience_counter = 0
                os.makedirs(config.output_dir, exist_ok=True)
                ckpt = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch+1}")
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                log_entry["best_checkpoint"] = ckpt
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    log_entry["early_stop"] = True
                    logs.append(log_entry)
                    break
        else:
            log_entry["eval_loss"] = None

        if mlflow_tracking_uri:
            import mlflow
            mlflow.log_metric("train/avg_loss", avg_train_loss, epoch)
            mlflow.log_metric("lr", scheduler.get_last_lr()[0], epoch)

    result = TrainResult(config=config, final_loss=avg_train_loss, eval_loss=best_eval_loss,
                         best_epoch=epoch + 1,
                         checkpoint_path=os.path.join(config.output_dir, f"checkpoint_epoch_{epoch+1}"),
                         logs=logs)
    if mlflow_tracking_uri:
        import mlflow
        mlflow.end_run()
    return result
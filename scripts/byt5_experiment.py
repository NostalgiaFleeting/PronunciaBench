"""Low-disk ByT5 CMUdict training and finalization pipeline."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import inspect
import io
import json
import os
import random
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "google/byt5-small"
MODEL_REVISION = "6f07f879d308b7b762708b50c83d41b27e329992"
EXPERIMENT_ID = "byt5-cmudict-001"
SEED = 42
PHONE_SEPARATOR = " "
DEFAULT_EXPERIMENT_DIR = Path("experiments") / EXPERIMENT_ID
EXPERIMENT_PROTOCOL_PATH = Path("reports/experiment_protocol.json")
TEST_MANIFEST_PATH = Path("reports/test_manifest.json")
EXPECTED_PROTOCOL_SHA256 = "b1cc34ec06832e22b15976066954d8be21dfaa427230482f7bae0d63e10ccbcc"
EXPECTED_TEST_MANIFEST_SHA256 = (
    "a5058b9b75e450bb62f40d9ae3ab22f477b7b90516f7f3e4b0a9c252d18c43f9"
)
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
FINALIZATION_STAGE_COUNT = 8
NO_REPEAT_NGRAM_SIZE = 3

DEFAULT_PER_DEVICE_BATCH = 1
DEFAULT_GRADIENT_ACCUMULATION = 32
DEFAULT_GRADIENT_CHECKPOINTING = True


def print_gpu_diagnostics() -> dict[str, Any]:
    """Print GPU diagnostics and return hardware facts."""
    info: dict[str, Any] = {
        "gpu_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "gpu_name": "CPU",
        "vram_gb": 0.0,
        "compute_capability": None,
        "bf16_supported": False,
        "fp16_supported": False,
    }
    print("\n=== Hardware Diagnostics ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda or 'not available'}")

    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": properties.name,
                "vram_gb": round(properties.total_memory / 1024**3, 2),
                "compute_capability": f"{properties.major}.{properties.minor}",
                "bf16_supported": properties.major >= 8,
                "fp16_supported": True,
            }
        )
        free_memory = torch.cuda.mem_get_info()[0]
        used_memory = properties.total_memory - free_memory
        print(f"GPU: {properties.name}")
        print(f"VRAM: {info['vram_gb']} GB")
        print(f"Compute Capability: ({properties.major}, {properties.minor})")
        print(f"BF16 supported: {info['bf16_supported']}")
        print(f"FP16 supported: {info['fp16_supported']}")
        print(
            f"Memory: {used_memory / 1024**3:.2f} GB used / "
            f"{info['vram_gb']} GB total"
        )
    else:
        print("No GPU available - running on CPU (will be very slow)")
    print("=" * 40)
    return info


def load_split(path: Path) -> list[dict[str, str]]:
    """Load one JSONL split."""
    data = []
    with path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            stripped = line.strip()
            if stripped:
                data.append(json.loads(stripped))
    return data


def load_splits(
    data_dir: str | Path,
    names: tuple[str, ...],
) -> dict[str, list[dict[str, str]]]:
    """Load only explicitly requested splits."""
    split_dir = Path(data_dir)
    splits: dict[str, list[dict[str, str]]] = {}
    for name in names:
        path = split_dir / f"{name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset split not found: {path}")
        splits[name] = load_split(path)
        print(f"Loaded {name}: {len(splits[name])} entries from {path}")
    return splits


def load_all_splits(data_dir: str = "data/experiment") -> dict[str, list[dict[str, str]]]:
    """Load all splits for backwards-compatible manual diagnostics."""
    return load_splits(data_dir, ("train", "validation", "calibration", "test"))


def is_canary_run(args: argparse.Namespace) -> bool:
    """Return whether training is explicitly bounded by update steps."""
    return args.max_steps is not None


def training_split_names(args: argparse.Namespace) -> tuple[str, ...]:
    """Return pre-training split names, excluding frozen test data."""
    if is_canary_run(args) and not args.finalization_smoke:
        return ("train",)
    return ("train", "validation")


def load_training_splits(args: argparse.Namespace) -> dict[str, list[dict[str, str]]]:
    """Load only splits required before training."""
    splits = load_splits(args.data_dir, training_split_names(args))
    if args.max_samples:
        splits["train"] = splits["train"][: args.max_samples]
        if "validation" in splits and not args.finalization_smoke:
            splits["validation"] = splits["validation"][: args.max_samples]
    return splits


def select_finalization_data(
    args: argparse.Namespace,
    validation_data: list[dict[str, str]],
) -> tuple[str | None, list[dict[str, str]], bool]:
    """Choose finalization data and open test only for an unbounded full run."""
    if args.finalization_smoke:
        if len(validation_data) < args.finalization_smoke:
            raise ValueError(
                f"--finalization-smoke requested {args.finalization_smoke} validation "
                f"samples, but only {len(validation_data)} are available"
            )
        print("FINALIZATION SMOKE")
        print("Evaluation split: validation")
        print("Frozen test opened: NO")
        return "validation", validation_data[: args.finalization_smoke], False
    if is_canary_run(args):
        print("Canary finalization: no evaluation requested")
        print("Frozen test opened: NO")
        return None, [], False

    test_data = load_splits(args.data_dir, ("test",))["test"]
    verify_frozen_test_data(test_data)
    if args.max_samples:
        test_data = test_data[: args.max_samples]
    print("Evaluation split: test (frozen)")
    print("Frozen test opened: YES")
    return "test", test_data, True


class CMUdictDataset(torch.utils.data.Dataset):
    """Dataset for ByT5 G2P training on CMUdict ARPAbet."""

    def __init__(
        self,
        data: list[dict[str, str]],
        tokenizer: Any,
        max_input_length: int = 64,
        max_target_length: int = 128,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.data[index]
        word = item["word"].lower()
        target = item["pronunciation"]
        inputs = self.tokenizer(
            word,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            target,
            max_length=self.max_target_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": targets["input_ids"].squeeze(0),
            "word": word,
            "target": target,
        }


def compute_per_with_stress(reference: str, hypothesis: str) -> float:
    """Compute phone error rate with stress digits preserved."""
    reference_phones = reference.split()
    hypothesis_phones = hypothesis.split()
    if not reference_phones:
        return 0.0 if not hypothesis_phones else 1.0
    rows, columns = len(reference_phones), len(hypothesis_phones)
    distances = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(rows + 1):
        distances[row][0] = row
    for column in range(columns + 1):
        distances[0][column] = column
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            substitution_cost = reference_phones[row - 1] != hypothesis_phones[column - 1]
            distances[row][column] = min(
                distances[row - 1][column] + 1,
                distances[row][column - 1] + 1,
                distances[row - 1][column - 1] + substitution_cost,
            )
    return distances[rows][columns] / rows


def compute_stress_stripped(phones: str) -> str:
    """Remove stress digits from ARPAbet phones."""
    return PHONE_SEPARATOR.join(phone.rstrip("012") for phone in phones.split() if phone.rstrip("012"))


def compute_per_without_stress(reference: str, hypothesis: str) -> float:
    """Compute phone error rate with stress digits stripped."""
    return compute_per_with_stress(
        compute_stress_stripped(reference),
        compute_stress_stripped(hypothesis),
    )


def compute_exact_match(reference: str, hypothesis: str) -> bool:
    """Return whether phone tokens match exactly."""
    return reference.split() == hypothesis.split()


def bootstrap_ci(
    per_values: list[float],
    n_trials: int = 1000,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Compute a deterministic bootstrap 95% interval for mean PER."""
    if not per_values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    count = len(per_values)
    means = []
    for _ in range(n_trials):
        sample = [per_values[rng.randint(0, count - 1)] for _ in range(count)]
        means.append(sum(sample) / count)
    means.sort()
    return (
        sum(means) / n_trials,
        means[int(0.025 * n_trials)],
        means[int(0.975 * n_trials)],
    )


def compute_oov_metrics(
    predictions: list[dict[str, Any]],
    train_data: list[dict[str, str]],
) -> dict[str, Any]:
    """Compute OOV metrics - test words whose root never appeared in training."""
    train_words = {item["word"].lower() for item in train_data}
    oov_predictions = [p for p in predictions if p["word"].lower() not in train_words]
    if not oov_predictions:
        return {
            "oov_words": 0,
            "oov_total": len(predictions),
            "oov_per_with_stress": None,
            "oov_per_without_stress": None,
            "oov_exact_match": None,
            "note": "All test words present in training (no OOV evaluation possible)",
        }
    oov_per_with_stress = np.mean([p["per_with_stress"] for p in oov_predictions])
    oov_per_without_stress = np.mean([p["per_without_stress"] for p in oov_predictions])
    oov_exact_match = np.mean([p["exact_match"] for p in oov_predictions])
    return {
        "oov_words": len(oov_predictions),
        "oov_per_with_stress": round(float(oov_per_with_stress), 4),
        "oov_per_without_stress": round(float(oov_per_without_stress), 4),
        "oov_exact_match": round(float(oov_exact_match), 4),
        "note": f"{len(oov_predictions)} OOV words out of {len(predictions)} test words",
    }


def generate_error_analysis(
    predictions: list[dict[str, Any]],
    n_worst: int = 20,
) -> list[dict[str, Any]]:
    """Build an error table for the worst predictions."""
    errors = []
    for prediction in sorted(
        predictions,
        key=lambda item: item["per_with_stress"],
        reverse=True,
    )[:n_worst]:
        reference_phones = prediction["reference"].split()
        hypothesis_phones = prediction["prediction"].split()
        rows, columns = len(reference_phones), len(hypothesis_phones)
        distances = [[0] * (columns + 1) for _ in range(rows + 1)]
        for row in range(rows + 1):
            distances[row][0] = row
        for column in range(columns + 1):
            distances[0][column] = column
        for row in range(1, rows + 1):
            for column in range(1, columns + 1):
                substitution_cost = reference_phones[row - 1] != hypothesis_phones[column - 1]
                distances[row][column] = min(
                    distances[row - 1][column] + 1,
                    distances[row][column - 1] + 1,
                    distances[row - 1][column - 1] + substitution_cost,
                )
        errors.append(
            {
                "word": prediction["word"],
                "reference_phones": prediction["reference"],
                "predicted_phones": prediction["prediction"],
                "per_with_stress": round(prediction["per_with_stress"], 4),
                "per_without_stress": round(prediction["per_without_stress"], 4),
                "exact_match": prediction["exact_match"],
                "word_length": len(prediction["word"]),
                "ref_phone_count": len(reference_phones),
                "hyp_phone_count": len(hypothesis_phones),
                "edit_distance": distances[rows][columns],
            }
        )
    return errors


def evaluate_reliability_layer(
    predictions: list[dict[str, Any]],
    calibration_data: list[dict[str, str]],
) -> dict[str, Any]:
    """Evaluate the existing length-based reliability diagnostic."""
    del calibration_data  # Kept in the interface for protocol-compatible callers.
    results: list[dict[str, Any]] = []
    for prediction in predictions:
        reference_length = len(prediction["reference"].split())
        hypothesis_length = len(prediction["prediction"].split())
        confidence = (
            1.0 - abs(reference_length - hypothesis_length) / max(reference_length, hypothesis_length)
            if reference_length > 0
            else 0.0
        )
        confidence = max(0.0, min(1.0, confidence))
        results.append(
            {
                "word": prediction["word"],
                "per": prediction["per_with_stress"],
                "confidence": round(confidence, 4),
                "exact_match": prediction["exact_match"],
            }
        )
    buckets: dict[str, dict[str, int | float]] = {}
    n_buckets = 5
    for row in results:
        bucket_index = min(n_buckets - 1, int(row["confidence"] * n_buckets))
        key = f"confidence_{bucket_index}_{bucket_index + 1}"
        if key not in buckets:
            buckets[key] = {"count": 0, "total_per": 0.0, "exact_matches": 0}
        buckets[key]["count"] += 1
        buckets[key]["total_per"] += row["per"]
        if row["exact_match"]:
            buckets[key]["exact_matches"] += 1
    bucket_results = []
    for key in sorted(buckets.keys()):
        bucket = buckets[key]
        bucket_results.append(
            {
                "bucket": key,
                "sample_count": bucket["count"],
                "mean_per": round(bucket["total_per"] / bucket["count"], 4),
                "exact_match_rate": round(
                    bucket["exact_matches"] / bucket["count"], 4
                ),
            }
        )
    is_monotonic = False
    if len(bucket_results) >= 2:
        per_values = [bucket["mean_per"] for bucket in bucket_results]
        is_monotonic = all(
            per_values[index] >= per_values[index + 1]
            for index in range(len(per_values) - 1)
        )
    abstention_results = []
    for threshold in (0.3, 0.5, 0.7):
        selected = [row for row in results if row["confidence"] >= threshold]
        if selected:
            coverage = len(selected) / len(results)
            selective_per = sum(row["per"] for row in selected) / len(selected)
            abstention_results.append(
                {
                    "threshold": threshold,
                    "coverage": round(coverage, 4),
                    "selective_per": round(selective_per, 4),
                    "abstention_rate": round(1.0 - coverage, 4),
                }
            )
    return {
        "bucket_analysis": bucket_results,
        "monotonicity_preserved": is_monotonic,
        "abstention_results": abstention_results,
        "total_samples": len(results),
    }


def configure_precision(args: argparse.Namespace, gpu_info: dict[str, Any]) -> tuple[bool, bool]:
    """Resolve the requested AMP backend without changing it after training starts."""
    gpu_available = gpu_info["gpu_available"]
    if args.amp_backend == "bf16" and not gpu_info["bf16_supported"]:
        raise RuntimeError("BF16 was requested but is unsupported by this GPU")
    if args.amp_backend == "fp16" and not gpu_available:
        raise RuntimeError("FP16 was requested but CUDA is unavailable")
    if args.amp_backend == "fp32":
        return False, False
    if args.amp_backend == "bf16":
        return True, False
    if args.amp_backend == "fp16":
        return False, True

    compute_capability = gpu_info.get("compute_capability") or ""
    if compute_capability.startswith("6."):
        print("Pascal GPU detected: auto precision resolves to fp32")
        return False, False
    if gpu_info["bf16_supported"]:
        return True, False
    return False, gpu_available


def _training_argument_parameters() -> set[str]:
    return set(inspect.signature(TrainingArguments.__init__).parameters)


def build_training_argument_kwargs(
    args: argparse.Namespace,
    output_path: Path,
    use_bf16: bool,
    use_fp16: bool,
) -> dict[str, Any]:
    """Build version-compatible Trainer arguments for a run."""
    parameters = _training_argument_parameters()
    eval_key = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    canary = is_canary_run(args)

    kwargs: dict[str, Any] = {
        "output_dir": str(output_path),
        "num_train_epochs": 1 if canary else args.epochs,
        "max_steps": args.max_steps if canary else -1,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "logging_steps": args.logging_steps,
        "save_total_limit": 1 if canary else 2,
        "load_best_model_at_end": not canary,
        "metric_for_best_model": "per",
        "greater_is_better": False,
        "report_to": "none",
        "seed": args.seed,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "gradient_checkpointing": args.gradient_checkpointing,
        "dataloader_num_workers": 0,
        "save_only_model": False,
    }

    if canary:
        kwargs.update(
            {
                eval_key: "no",
                "save_strategy": "steps",
                "save_steps": args.max_steps,
            }
        )
    elif args.low_disk:
        kwargs.update({eval_key: "epoch", "save_strategy": "epoch"})
    else:
        kwargs.update(
            {
                eval_key: "steps",
                "save_strategy": "steps",
                "save_steps": args.save_steps,
                "eval_steps": args.save_steps,
            }
        )

    if "save_safetensors" in parameters:
        kwargs["save_safetensors"] = True
    for optional_name in ("dataloader_num_workers", "save_only_model"):
        if optional_name not in parameters:
            kwargs.pop(optional_name, None)
    return kwargs


def build_compute_metrics(tokenizer: Any) -> Callable[[Any], dict[str, float]]:
    """Build the validation metric callback used for best-checkpoint selection."""

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        raw_predictions = eval_prediction.predictions
        label_ids = eval_prediction.label_ids
        token_ids = raw_predictions[0] if isinstance(raw_predictions, tuple) else raw_predictions
        if token_ids.ndim == 3:
            token_ids = np.argmax(token_ids, axis=-1)
        prediction_texts = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        masked_labels = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        label_texts = tokenizer.batch_decode(masked_labels, skip_special_tokens=True)
        per_values = []
        stressless_values = []
        exact_matches = []
        for reference, hypothesis in zip(label_texts, prediction_texts, strict=True):
            reference = reference.strip().upper()
            hypothesis = hypothesis.strip().upper()
            per_values.append(compute_per_with_stress(reference, hypothesis))
            stressless_values.append(compute_per_without_stress(reference, hypothesis))
            exact_matches.append(compute_exact_match(reference, hypothesis))
        return {
            "per": float(np.mean(per_values)),
            "per_without_stress": float(np.mean(stressless_values)),
            "exact_match": float(np.mean(exact_matches)),
        }

    return compute_metrics


def preprocess_logits_for_metrics(logits: Any, _labels: Any) -> Any:
    """Reduce evaluation logits to token IDs before Trainer accumulates them."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def create_trainer(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    train_dataset: CMUdictDataset,
    validation_dataset: CMUdictDataset | None,
    output_path: Path,
    use_bf16: bool,
    use_fp16: bool,
) -> tuple[Trainer, dict[str, Any]]:
    """Create a Trainer with pinned optimizer behavior and low-memory metrics."""
    argument_kwargs = build_training_argument_kwargs(
        args,
        output_path,
        use_bf16,
        use_fp16,
    )
    training_args = TrainingArguments(**argument_kwargs)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "compute_metrics": build_compute_metrics(tokenizer),
        "data_collator": DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
        "preprocess_logits_for_metrics": preprocess_logits_for_metrics,
    }
    if args.optimizer == "adafactor":
        from transformers import Adafactor

        trainer_kwargs["optimizer_cls_and_kwargs"] = (
            Adafactor,
            {
                "lr": args.learning_rate,
                "relative_step": False,
                "scale_parameter": False,
                "warmup_init": False,
            },
        )
    return Trainer(**trainer_kwargs), argument_kwargs


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_experiment_root(value: str | Path) -> Path:
    """Resolve an experiment root without requiring it to exist yet."""
    return Path(value).expanduser().resolve(strict=False)


def _nearest_existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(f"Experiment volume is unavailable: {path.anchor}")
        candidate = candidate.parent
    return candidate


def inspect_output_disk(experiment_root: Path, min_free_gb: float) -> dict[str, Any]:
    """Measure and enforce free space on the selected output volume."""
    if min_free_gb < 0:
        raise ValueError("--min-free-gb cannot be negative")
    existing = _nearest_existing_ancestor(experiment_root)
    usage = shutil.disk_usage(existing)
    free_gb = usage.free / 1024**3
    volume = experiment_root.anchor or existing.anchor
    print("\n=== Output Volume Preflight ===")
    print(f"Experiment root: {experiment_root}")
    print(f"Experiment volume: {volume}")
    print(f"Free disk: {free_gb:.3f} GiB")
    print(f"Configured minimum: {min_free_gb:.3f} GiB")
    if free_gb < min_free_gb:
        raise RuntimeError(
            "ERROR: insufficient free disk space for configured safety threshold.\n"
            "Training has NOT started."
        )
    return {
        "free_disk_gb_at_start": free_gb,
        "experiment_volume": volume,
        "min_free_gb_policy": min_free_gb,
    }


def _require_contained_path(path: Path, root: Path, description: str) -> Path:
    resolved_path = path.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    if resolved_path == resolved_root or not _path_is_within(resolved_path, resolved_root):
        raise ValueError(f"Unsafe {description}: {resolved_path} is not below {resolved_root}")
    return resolved_path


def prepare_run_directory(args: argparse.Namespace, experiment_root: Path) -> tuple[Path, Path | None]:
    """Create a run directory or validate a checkpoint resume target."""
    experiment_root.mkdir(parents=True, exist_ok=True)
    if args.resume_from_checkpoint:
        checkpoint = Path(args.resume_from_checkpoint).resolve(strict=True)
        if not checkpoint.is_dir() or not CHECKPOINT_PATTERN.fullmatch(checkpoint.name):
            raise ValueError(f"Invalid Trainer checkpoint: {checkpoint}")
        run_dir = _require_contained_path(checkpoint.parent, experiment_root, "resume run")
        print(f"Resuming run directory: {run_dir}")
        print(f"Resume checkpoint: {checkpoint}")
        return run_dir, checkpoint

    prefix = "canary" if is_canary_run(args) else "run"
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = experiment_root / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir, None


def atomic_write_text(path: Path, content: str) -> None:
    """Write a small artifact through a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as file_handle:
            file_handle.write(content)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically serialize JSON."""
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


ERROR_ANALYSIS_FIELDS = (
    "word",
    "reference_phones",
    "predicted_phones",
    "per_with_stress",
    "per_without_stress",
    "exact_match",
    "word_length",
    "ref_phone_count",
    "hyp_phone_count",
    "edit_distance",
)


def atomic_write_error_analysis(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write the error-analysis CSV."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=ERROR_ANALYSIS_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_protocol_references(strict: bool) -> dict[str, Any]:
    """Record immutable protocol files and optionally enforce their known hashes."""
    references: dict[str, Any] = {}
    expected_hashes = {
        "experiment_protocol": (EXPERIMENT_PROTOCOL_PATH, EXPECTED_PROTOCOL_SHA256),
        "test_manifest": (TEST_MANIFEST_PATH, EXPECTED_TEST_MANIFEST_SHA256),
    }
    for name, (path, expected_hash) in expected_hashes.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required protocol file not found: {path}")
        actual_hash = sha256_file(path)
        if strict and actual_hash.lower() != expected_hash:
            raise RuntimeError(
                f"Strict protocol hash mismatch for {path}: expected {expected_hash}, "
                f"found {actual_hash}"
            )
        references[name] = {
            "path": str(path),
            "sha256": actual_hash,
            "expected_sha256": expected_hash,
            "matches_frozen_reference": actual_hash.lower() == expected_hash,
        }
    manifest = json.loads(TEST_MANIFEST_PATH.read_text(encoding="utf-8"))
    references["test_manifest"]["canonical_test_content_sha256"] = manifest.get(
        "test_sha256"
    )
    return references


def compute_canonical_split_sha256(
    rows: list[dict[str, str]],
) -> tuple[str, int, int]:
    """Hash rows using the importer's canonical tab-separated representation."""
    canonical_rows = sorted(
        rows,
        key=lambda item: (item["word"].lower(), item.get("label", item["word"])),
    )
    content = "\n".join(
        "\t".join(
            (
                item["word"],
                item["pronunciation"],
                item.get("label", item["word"]),
            )
        )
        for item in canonical_rows
    )
    unique_words = len({item["word"].lower() for item in canonical_rows})
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return digest, unique_words, len(canonical_rows)


def verify_frozen_test_data(test_data: list[dict[str, str]]) -> dict[str, Any]:
    """Verify authorized frozen-test data against its canonical manifest."""
    manifest = json.loads(TEST_MANIFEST_PATH.read_text(encoding="utf-8"))
    actual_hash, unique_words, variants = compute_canonical_split_sha256(test_data)
    checks = {
        "canonical_test_content_sha256": actual_hash,
        "expected_canonical_test_content_sha256": manifest["test_sha256"],
        "n_lexical_items": unique_words,
        "expected_n_lexical_items": manifest["n_lexical_items"],
        "n_pronunciation_variants": variants,
        "expected_n_pronunciation_variants": manifest["n_pronunciation_variants"],
    }
    if (
        actual_hash != manifest["test_sha256"]
        or unique_words != manifest["n_lexical_items"]
        or variants != manifest["n_pronunciation_variants"]
    ):
        raise RuntimeError(f"Frozen test split does not match its manifest: {checks}")
    return checks


def get_git_sha() -> str:
    """Return the current Git revision without failing finalization."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip()


def resolve_best_checkpoint(trainer_state: Any, run_dir: Path) -> Path:
    """Resolve Trainer's best checkpoint or the latest saved canary checkpoint."""
    reported = getattr(trainer_state, "best_model_checkpoint", None)
    if reported:
        candidate = Path(reported)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidate = candidate.resolve(strict=False)
        _require_contained_path(candidate, run_dir, "best checkpoint")
        if candidate.is_dir():
            return candidate

    candidates = []
    for path in run_dir.iterdir():
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            candidates.append((int(match.group(1)), path.resolve()))
    if not candidates:
        raise RuntimeError(f"No checkpoint-* directory found in {run_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def load_checkpoint_model(checkpoint: Path, device: torch.device) -> Any:
    """Reload a selected checkpoint locally and place it on the evaluation device."""
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(checkpoint),
        local_files_only=True,
        use_safetensors=True,
    )
    return model.to(device)


def configure_inference_model(model: Any) -> None:
    """Disable training-only behavior before deterministic generation."""
    model.eval()
    disable_checkpointing = getattr(model, "gradient_checkpointing_disable", None)
    if callable(disable_checkpointing):
        disable_checkpointing()
    model.config.use_cache = True


def generate_predictions(
    model: Any,
    tokenizer: Any,
    dataset: CMUdictDataset,
    device: torch.device,
    max_new_tokens: int = 128,
    eval_batch_size: int = 1,
    progress_label: str = "generation",
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Generate deterministic predictions and measure throughput and peak VRAM."""
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be at least 1")
    configure_inference_model(model)
    total = len(dataset)
    predictions: list[dict[str, Any]] = []
    progress_every = 16 if total <= 256 else max(100, total // 20)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, total, eval_batch_size):
            stop = min(start + eval_batch_size, total)
            items = [dataset[index] for index in range(start, stop)]
            input_ids = torch.stack([item["input_ids"] for item in items]).to(device)
            attention_mask = torch.stack([item["attention_mask"] for item in items]).to(device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for item, text in zip(items, texts, strict=True):
                normalized = " ".join(text.upper().split())
                predictions.append(
                    {
                        "word": item["word"],
                        "reference": item["target"],
                        "prediction": normalized,
                        "per_with_stress": compute_per_with_stress(
                            item["target"], normalized
                        ),
                        "per_without_stress": compute_per_without_stress(
                            item["target"], normalized
                        ),
                        "exact_match": compute_exact_match(item["target"], normalized),
                    }
                )
            if stop % progress_every == 0 or stop == total:
                print(f"{progress_label}: {stop}/{total}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_gpu_memory_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    else:
        peak_gpu_memory_gb = 0.0
    elapsed = time.perf_counter() - started
    samples_per_second = total / elapsed if elapsed else 0.0
    seconds_per_sample = elapsed / total if total else 0.0
    stats: dict[str, float | int] = {
        "samples": total,
        "elapsed_seconds": elapsed,
        "samples_per_second": samples_per_second,
        "seconds_per_sample": seconds_per_sample,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
        "eval_batch_size": eval_batch_size,
    }
    print(
        f"{progress_label}: {total} samples in {elapsed:.2f}s "
        f"({samples_per_second:.3f} samples/s, {seconds_per_sample:.3f} s/sample); "
        f"peak GPU memory {peak_gpu_memory_gb:.3f} GiB"
    )
    return predictions, stats


def summarize_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the frozen evaluation metrics from generated predictions."""
    if not predictions:
        return {
            "per_with_stress": None,
            "per_without_stress": None,
            "exact_match": None,
            "n_samples": 0,
            "bootstrap_ci_95": None,
        }
    per_values = [prediction["per_with_stress"] for prediction in predictions]
    confidence_interval = bootstrap_ci(per_values)
    return {
        "per_with_stress": float(np.mean(per_values)),
        "per_without_stress": float(
            np.mean([prediction["per_without_stress"] for prediction in predictions])
        ),
        "exact_match": float(
            np.mean([prediction["exact_match"] for prediction in predictions])
        ),
        "n_samples": len(predictions),
        "bootstrap_ci_95": [confidence_interval[1], confidence_interval[2]],
    }


def add_frozen_test_duration_estimate(
    throughput: dict[str, float | int],
) -> dict[str, float | int | str]:
    """Project frozen-test generation time from validation throughput only."""
    manifest = json.loads(TEST_MANIFEST_PATH.read_text(encoding="utf-8"))
    test_samples = int(manifest["n_pronunciation_variants"])
    samples_per_second = float(throughput["samples_per_second"])
    estimated_seconds = test_samples / samples_per_second if samples_per_second > 0 else 0.0
    estimate = {
        **throughput,
        "estimated_frozen_test_samples": test_samples,
        "estimated_frozen_test_seconds": estimated_seconds,
        "estimated_frozen_test_hours": estimated_seconds / 3600,
        "estimate_note": (
            "Projection from validation-smoke throughput; actual frozen-test duration is not guaranteed"
        ),
    }
    print(
        "Estimated frozen-test generation duration: "
        f"{estimated_seconds:.1f}s ({estimated_seconds / 3600:.2f}h) for "
        f"{test_samples} samples; estimate only"
    )
    return estimate


def export_best_model_once(model: Any, tokenizer: Any, run_dir: Path) -> Path:
    """Export one finalized model tree without overwriting an existing export."""
    destination = run_dir / "best_model"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing best-model export: {destination}")
    temporary = run_dir / f".best_model.tmp-{os.getpid()}-{time.time_ns()}"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        save_parameters = inspect.signature(model.save_pretrained).parameters
        save_kwargs = {"safe_serialization": True} if "safe_serialization" in save_parameters else {}
        model.save_pretrained(str(temporary), **save_kwargs)
        tokenizer.save_pretrained(str(temporary))
        weight_files = list(temporary.glob("*.safetensors"))
        if not weight_files or not (temporary / "config.json").is_file():
            raise RuntimeError("Best-model export verification failed")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def verify_best_model_export(path: Path) -> None:
    """Verify a finalized model export before checkpoint compaction."""
    if not path.is_dir() or not (path / "config.json").is_file():
        raise RuntimeError(f"Best-model export is incomplete: {path}")
    if not list(path.glob("*.safetensors")):
        raise RuntimeError(f"Best-model safetensors are missing: {path}")


def _tree_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            total += child.stat().st_size
    return total


def checkpoint_cleanup_targets(run_dir: Path) -> list[Path]:
    """Return direct-child Trainer checkpoints eligible for cleanup."""
    return sorted(
        path
        for path in run_dir.iterdir()
        if path.is_dir() and CHECKPOINT_PATTERN.fullmatch(path.name)
    )


def validate_cleanup_targets(targets: list[Path], run_dir: Path) -> list[Path]:
    """Canonicalize cleanup targets and prove direct containment in the run."""
    resolved_run = run_dir.resolve(strict=True)
    if resolved_run == Path(resolved_run.anchor):
        raise ValueError(f"Refusing cleanup against a drive root: {resolved_run}")
    validated = []
    for target in targets:
        resolved = target.resolve(strict=True)
        if resolved.parent != resolved_run or not CHECKPOINT_PATTERN.fullmatch(resolved.name):
            raise ValueError(f"Unsafe cleanup target outside active run: {resolved}")
        validated.append(resolved)
    return validated


def remove_checkpoint_targets(targets: list[Path], run_dir: Path) -> int:
    """Delete validated checkpoint directories and verify their removal."""
    validated = validate_cleanup_targets(targets, run_dir)
    sizes = {target: _tree_size(target) for target in validated}
    print("Files to be removed:")
    if not validated:
        print("  (none)")
    for target in validated:
        print(f"  {target} ({sizes[target] / 1024**2:.2f} MiB)")
    reclaimable = sum(sizes.values())
    print(f"Estimated bytes reclaimed: {reclaimable}")
    for target in validated:
        shutil.rmtree(target)
    remaining = [str(target) for target in validated if target.exists()]
    if remaining:
        raise RuntimeError(f"Checkpoint cleanup verification failed: {remaining}")
    return reclaimable


def run_post_success_cleanup(
    success: bool,
    args: argparse.Namespace,
    run_dir: Path,
    best_model_dir: Path | None,
) -> int:
    """Apply optional cleanup only after every finalization gate succeeded."""
    if not success:
        print("Finalization did not succeed; resumable checkpoints were preserved.")
        return 0
    if args.cleanup_canary:
        if not is_canary_run(args):
            raise ValueError("--cleanup-canary can only remove checkpoints from a canary run")
        return remove_checkpoint_targets(checkpoint_cleanup_targets(run_dir), run_dir)
    if args.compact_after_success:
        if is_canary_run(args):
            raise ValueError("Use --cleanup-canary, not --compact-after-success, for canaries")
        if best_model_dir is None:
            raise RuntimeError("Compaction requires a verified best-model export")
        verify_best_model_export(best_model_dir)
        for required_name in (
            "results.json",
            "provenance.json",
            "error_analysis.csv",
            "training_log.json",
            "protocol_references.json",
        ):
            if not (run_dir / required_name).is_file():
                raise RuntimeError(f"Compaction blocked; artifact is missing: {required_name}")
        return remove_checkpoint_targets(checkpoint_cleanup_targets(run_dir), run_dir)
    return 0


def record_cleanup_provenance(
    run_dir: Path,
    mode: str,
    reclaimed_bytes: int,
) -> None:
    """Atomically append verified post-success cleanup facts to provenance."""
    provenance_path = run_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    storage = provenance.setdefault("storage", {})
    storage["post_success_cleanup"] = {
        "mode": mode,
        "reclaimed_bytes": reclaimed_bytes,
        "checkpoint_count_after": len(checkpoint_cleanup_targets(run_dir)),
        "completed_at": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(provenance_path, provenance)


class FinalizationTracker:
    """Emit bounded stage progress with elapsed time and failure context."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.timings: dict[str, float] = {}

    def record_training_complete(self, elapsed: float) -> None:
        key = "1_training_complete"
        self.timings[key] = elapsed
        print(f"[FINALIZE 1/{FINALIZATION_STAGE_COUNT}] Training complete ({elapsed:.2f}s)")

    def run(self, index: int, name: str, action: Callable[[], Any]) -> Any:
        key = f"{index}_{name.lower().replace(' ', '_')}"
        started = time.perf_counter()
        print(f"[FINALIZE {index}/{FINALIZATION_STAGE_COUNT}] {name}...")
        try:
            result = action()
        except Exception as exception:
            elapsed = time.perf_counter() - started
            self.timings[key] = elapsed
            print(
                f"[FINALIZE {index}/{FINALIZATION_STAGE_COUNT}] {name} FAILED "
                f"after {elapsed:.2f}s: {type(exception).__name__}: {exception}"
            )
            raise
        elapsed = time.perf_counter() - started
        self.timings[key] = elapsed
        print(
            f"[FINALIZE {index}/{FINALIZATION_STAGE_COUNT}] {name} complete "
            f"({elapsed:.2f}s)"
        )
        return result

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started


def _validate_strict_operational_config(args: argparse.Namespace) -> None:
    expected = {
        "epochs": 3,
        "learning_rate": 1e-4,
        "batch_size": 1,
        "gradient_accumulation": 32,
        "optimizer": "adafactor",
        "amp_backend": "fp32",
        "seed": SEED,
        "max_new_tokens": 128,
    }
    mismatches = [
        f"{name}={getattr(args, name)!r} (expected {value!r})"
        for name, value in expected.items()
        if getattr(args, name) != value
    ]
    if not args.gradient_checkpointing:
        mismatches.append("gradient_checkpointing=False (expected True)")
    if not args.low_disk:
        mismatches.append("low_disk=False (expected True)")
    if not is_canary_run(args) and args.max_samples:
        mismatches.append("max_samples is not allowed for a strict full run")
    if mismatches:
        raise ValueError("Strict configuration mismatch:\n- " + "\n- ".join(mismatches))


def validate_cli_args(args: argparse.Namespace) -> None:
    """Validate mode combinations before touching model or dataset state."""
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1")
    if args.max_samples is not None and not is_canary_run(args):
        raise ValueError("--max-samples requires --max-steps to preserve frozen-test isolation")
    if args.finalization_smoke is not None:
        if args.finalization_smoke < 1:
            raise ValueError("--finalization-smoke must be at least 1")
        if not is_canary_run(args):
            raise ValueError("--finalization-smoke requires --max-steps")
    if args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be at least 1")
    if args.cleanup_canary and not is_canary_run(args):
        raise ValueError("--cleanup-canary requires --max-steps")
    if args.compact_after_success and is_canary_run(args):
        raise ValueError("--compact-after-success is only valid for full runs")
    if args.cleanup_canary and args.compact_after_success:
        raise ValueError("Cleanup modes are mutually exclusive")
    if args.strict_config:
        _validate_strict_operational_config(args)


def _release_training_objects(trainer: Trainer) -> None:
    """Detach Trainer-owned references before the caller releases the model."""
    trainer.optimizer = None
    trainer.lr_scheduler = None
    trainer.model = None


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Train and execute the complete low-disk finalization gate."""
    validate_cli_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    experiment_root = resolve_experiment_root(args.experiment_dir)
    disk_at_start = inspect_output_disk(experiment_root, args.min_free_gb)
    protocol_references = build_protocol_references(strict=args.strict_config)
    run_dir, resume_checkpoint = prepare_run_directory(args, experiment_root)
    print(f"Active run directory: {run_dir}")

    gpu_info = print_gpu_diagnostics()
    use_bf16, use_fp16 = configure_precision(args, gpu_info)
    precision = "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32")
    print(f"Mixed precision: {precision}")

    print("\n=== Loading Pre-Training Data ===")
    splits = load_training_splits(args)
    train_data = splits["train"]
    validation_data = splits.get("validation", [])
    print(f"Train: {len(train_data)} | Validation loaded: {len(validation_data)}")
    print("Frozen test opened before training: NO")

    print(f"\n=== Loading Base Model: {MODEL_ID} ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_safetensors=True,
    )
    device = torch.device("cuda" if gpu_info["gpu_available"] else "cpu")
    model = model.to(device)
    print(f"Pinned model revision: {MODEL_REVISION}")

    train_dataset = CMUdictDataset(
        train_data,
        tokenizer,
        args.max_input_length,
        args.max_target_length,
    )
    validation_dataset = None
    if validation_data:
        validation_dataset = CMUdictDataset(
            validation_data,
            tokenizer,
            args.max_input_length,
            args.max_target_length,
        )
    trainer, training_argument_kwargs = create_trainer(
        args,
        model,
        tokenizer,
        train_dataset,
        validation_dataset,
        run_dir,
        use_bf16,
        use_fp16,
    )
    print("\n=== Training ===")
    print(
        f"Epochs: {args.epochs}, LR: {args.learning_rate}, batch: {args.batch_size}, "
        f"gradient accumulation: {args.gradient_accumulation}"
    )
    print(
        f"Save strategy: {training_argument_kwargs['save_strategy']}, "
        f"save total limit: {training_argument_kwargs['save_total_limit']}, "
        f"save only model: {training_argument_kwargs.get('save_only_model', False)}"
    )

    training_started = time.perf_counter()
    try:
        if resume_checkpoint:
            train_result = trainer.train(resume_from_checkpoint=str(resume_checkpoint))
        else:
            train_result = trainer.train()
    except torch.cuda.OutOfMemoryError as exception:
        if args.strict_config:
            raise RuntimeError(
                "OOM in strict-config mode; the frozen configuration was not changed"
            ) from exception
        raise RuntimeError(
            "Training ran out of GPU memory. Retry with an explicit smaller batch or fp16; "
            "no automatic scientific-configuration fallback was applied."
        ) from exception
    training_time = time.perf_counter() - training_started
    trainer_state = trainer.state
    training_log = list(trainer_state.log_history)
    training_loss = float(train_result.training_loss)

    tracker = FinalizationTracker()
    tracker.record_training_complete(training_time)
    best_checkpoint = tracker.run(
        2,
        "Resolving best checkpoint",
        lambda: resolve_best_checkpoint(trainer_state, run_dir),
    )

    _release_training_objects(trainer)
    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reloaded_model = tracker.run(
        3,
        "Loading best checkpoint",
        lambda: load_checkpoint_model(best_checkpoint, device),
    )
    tracker.run(
        4,
        "Configuring inference mode",
        lambda: configure_inference_model(reloaded_model),
    )

    evaluation_state: dict[str, Any] = {}

    def run_evaluation() -> None:
        evaluation_split, evaluation_data, frozen_test_opened = select_finalization_data(
            args,
            validation_data,
        )
        evaluation_state.update(
            {
                "split": evaluation_split,
                "data": evaluation_data,
                "frozen_test_opened": frozen_test_opened,
                "predictions": [],
                "throughput": {
                    "samples": 0,
                    "elapsed_seconds": 0.0,
                    "samples_per_second": 0.0,
                    "seconds_per_sample": 0.0,
                    "peak_gpu_memory_gb": 0.0,
                    "eval_batch_size": args.eval_batch_size,
                },
            }
        )
        if evaluation_split is None:
            return
        dataset = CMUdictDataset(
            evaluation_data,
            tokenizer,
            args.max_input_length,
            args.max_target_length,
        )
        predictions, throughput = generate_predictions(
            reloaded_model,
            tokenizer,
            dataset,
            device,
            max_new_tokens=args.max_new_tokens,
            eval_batch_size=args.eval_batch_size,
            progress_label=f"{evaluation_split} generation",
        )
        if args.finalization_smoke:
            throughput = add_frozen_test_duration_estimate(throughput)
        evaluation_state["predictions"] = predictions
        evaluation_state["throughput"] = throughput

    tracker.run(5, "Running evaluation", run_evaluation)

    metric_state: dict[str, Any] = {}

    def compute_final_metrics() -> None:
        predictions = evaluation_state["predictions"]
        metrics = summarize_predictions(predictions)
        errors = generate_error_analysis(predictions)
        calibration_data: list[dict[str, str]] = []
        if evaluation_state["split"] == "test":
            calibration_data = load_splits(args.data_dir, ("calibration",))["calibration"]
            oov_metrics = compute_oov_metrics(predictions, train_data)
            reliability = evaluate_reliability_layer(predictions, calibration_data)
        else:
            oov_metrics = {
                "oov_words": None,
                "note": "OOV metrics are reported only for the authorized frozen-test evaluation",
            }
            reliability = {
                "bucket_analysis": [],
                "monotonicity_preserved": None,
                "abstention_results": [],
                "total_samples": 0,
                "note": "Reliability evaluation is not part of a canary",
            }
        metric_state.update(
            {
                "metrics": metrics,
                "error_analysis": errors,
                "oov_metrics": oov_metrics,
                "reliability": reliability,
                "calibration_data": calibration_data,
            }
        )
        if predictions:
            print(f"PER with stress: {metrics['per_with_stress']:.6f}")
            print(f"PER without stress: {metrics['per_without_stress']:.6f}")
            print(f"Exact match: {metrics['exact_match']:.6f}")

    tracker.run(6, "Computing metrics", compute_final_metrics)

    artifact_state: dict[str, Any] = {"best_model_dir": None}

    def write_artifacts() -> None:
        best_model_dir = None
        if not is_canary_run(args):
            best_model_dir = export_best_model_once(reloaded_model, tokenizer, run_dir)
            verify_best_model_export(best_model_dir)
        artifact_state["best_model_dir"] = best_model_dir

        end_usage = shutil.disk_usage(_nearest_existing_ancestor(experiment_root))
        evaluation_metrics = metric_state["metrics"]
        status = (
            "CANARY_RUN"
            if is_canary_run(args)
            else ("COMPLETED" if gpu_info["gpu_available"] else "REAL_GPU_RUN_PENDING")
        )
        generation_config = {
            "do_sample": False,
            "num_beams": 1,
            "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
            "max_new_tokens": args.max_new_tokens,
            "eval_batch_size": args.eval_batch_size,
        }
        frozen_test_verification = (
            verify_frozen_test_data(evaluation_state["data"])
            if evaluation_state["split"] == "test"
            else None
        )
        if evaluation_state["split"] == "test":
            test_results = {
                **evaluation_metrics,
                "oov_metrics": metric_state["oov_metrics"],
                "canary_mode": False,
                "opened": True,
            }
        else:
            test_results = {
                "per_with_stress": 0.0,
                "per_without_stress": 0.0,
                "exact_match": 0.0,
                "n_samples": 0,
                "bootstrap_ci_95": [0.0, 0.0],
                "oov_metrics": metric_state["oov_metrics"],
                "canary_mode": is_canary_run(args),
                "opened": False,
            }
        results = {
            "experiment_id": EXPERIMENT_ID,
            "status": status,
            "evaluation_split": evaluation_state["split"],
            "frozen_test_opened": evaluation_state["frozen_test_opened"],
            "evaluation_results": evaluation_metrics,
            "validation_results": (
                evaluation_metrics if evaluation_state["split"] == "validation" else None
            ),
            "test_results": test_results,
            "oov_metrics": metric_state["oov_metrics"],
            "reliability": metric_state["reliability"],
            "generation_throughput": evaluation_state["throughput"],
            "frozen_test_verification": frozen_test_verification,
            "training": {
                "training_loss": training_loss,
                "wall_time_seconds": training_time,
                "max_steps": args.max_steps,
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }
        provenance = {
            "experiment_id": EXPERIMENT_ID,
            "status": status,
            "run": {
                "directory": str(run_dir),
                "canary": is_canary_run(args),
                "finalization_smoke_samples": args.finalization_smoke,
                "resume_from_checkpoint": (
                    str(resume_checkpoint) if resume_checkpoint else None
                ),
            },
            "hardware": {
                **gpu_info,
                "mixed_precision_used": precision,
            },
            "storage": {
                **disk_at_start,
                "free_disk_gb_after_artifacts": end_usage.free / 1024**3,
                "experiment_root": str(experiment_root),
                "low_disk": args.low_disk,
                "cleanup_canary_requested": args.cleanup_canary,
                "compact_after_success_requested": args.compact_after_success,
            },
            "dataset": {
                "train_size": len(train_data),
                "validation_size_loaded": len(validation_data),
                "calibration_size_loaded": len(metric_state["calibration_data"]),
                "evaluation_split": evaluation_state["split"],
                "evaluation_size": len(evaluation_state["data"]),
                "frozen_test_opened": evaluation_state["frozen_test_opened"],
                "seed": args.seed,
            },
            "training": {
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "effective_batch_size": args.batch_size * args.gradient_accumulation,
                "gradient_checkpointing": args.gradient_checkpointing,
                "optimizer": args.optimizer,
                "mixed_precision": precision,
                "wall_time_seconds": training_time,
                "max_steps": args.max_steps,
                "trainer_arguments": training_argument_kwargs,
            },
            "model": {
                "base_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "selected_checkpoint": str(best_checkpoint),
                "best_model_export": str(best_model_dir) if best_model_dir else None,
                "best_model_export_count": 0 if best_model_dir is None else 1,
            },
            "generation": generation_config,
            "generation_throughput": evaluation_state["throughput"],
            "protocol_references": protocol_references,
            "finalization": {
                "complete": False,
                "stage_elapsed_seconds": dict(tracker.timings),
                "elapsed_seconds_before_artifact_verification": tracker.elapsed,
            },
            "git_sha": get_git_sha(),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        artifact_state.update({"results": results, "provenance": provenance})
        atomic_write_error_analysis(
            run_dir / "error_analysis.csv",
            metric_state["error_analysis"],
        )
        atomic_write_json(run_dir / "training_log.json", training_log)
        atomic_write_json(run_dir / "protocol_references.json", protocol_references)
        atomic_write_json(run_dir / "results.json", results)
        atomic_write_json(run_dir / "provenance.json", provenance)

    tracker.run(7, "Writing artifacts", write_artifacts)

    def verify_finalization() -> None:
        required = (
            "results.json",
            "provenance.json",
            "error_analysis.csv",
            "training_log.json",
            "protocol_references.json",
        )
        missing = [name for name in required if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Finalization artifacts missing: {missing}")
        json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        provenance = artifact_state["provenance"]
        provenance["finalization"] = {
            "complete": True,
            "stage_elapsed_seconds": dict(tracker.timings),
            "elapsed_seconds_before_completion_write": tracker.elapsed,
        }
        atomic_write_json(run_dir / "provenance.json", provenance)

    tracker.run(8, "Finalization", verify_finalization)
    finalization_time = tracker.elapsed
    best_model_dir = artifact_state["best_model_dir"]
    reclaimed_bytes = run_post_success_cleanup(
        True,
        args,
        run_dir,
        best_model_dir,
    )
    if args.cleanup_canary or args.compact_after_success:
        cleanup_mode = "cleanup_canary" if args.cleanup_canary else "compact_after_success"
        record_cleanup_provenance(run_dir, cleanup_mode, reclaimed_bytes)
    print("\n=== Run Summary ===")
    print(f"Status: {artifact_state['results']['status']}")
    print(f"Run directory: {run_dir}")
    print(f"Training time: {training_time:.2f}s")
    print(f"Finalization time: {finalization_time:.2f}s")
    print(f"Frozen test opened: {'YES' if evaluation_state['frozen_test_opened'] else 'NO'}")
    print(f"Checkpoint cleanup reclaimed bytes: {reclaimed_bytes}")
    return {
        "run_dir": run_dir,
        "best_checkpoint": best_checkpoint,
        "best_model_dir": best_model_dir,
        "training_time_seconds": training_time,
        "finalization_time_seconds": finalization_time,
        "metrics": metric_state["metrics"],
        "throughput": evaluation_state["throughput"],
        "frozen_test_opened": evaluation_state["frozen_test_opened"],
        "reclaimed_bytes": reclaimed_bytes,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="ByT5 CMUdict G2P experiment")
    parser.add_argument("--data-dir", default="data/experiment")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_PER_DEVICE_BATCH)
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION,
    )
    parser.add_argument("--max-input-length", type=int, default=64)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--finalization-smoke", type=int, metavar="N", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--min-free-gb", type=float, default=0.0)
    parser.add_argument("--low-disk", action="store_true")
    parser.add_argument("--cleanup-canary", action="store_true")
    parser.add_argument("--compact-after-success", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=DEFAULT_GRADIENT_CHECKPOINTING,
        dest="gradient_checkpointing",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_false",
        dest="gradient_checkpointing",
    )
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "adafactor"),
        default="adafactor",
    )
    parser.add_argument(
        "--amp-backend",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
    )
    parser.add_argument(
        "--strict-config",
        action="store_true",
        help="Abort instead of changing the frozen operational configuration",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_cli_args(args)
    except ValueError as exception:
        parser.error(str(exception))
    if is_canary_run(args):
        print(f"Running canary mode: {args.max_steps} training step(s)")
    if args.finalization_smoke:
        print(
            f"Finalization smoke will evaluate exactly {args.finalization_smoke} "
            "validation samples"
        )
    run_experiment(args)


if __name__ == "__main__":
    main()

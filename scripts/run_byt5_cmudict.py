"""ByT5 CMUdict G2P experiment pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
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
EXPERIMENT_ID = "byt5-cmudict-001"
SEED = 42
PHONE_SEPARATOR = " "
OUTPUT_DIR = Path("experiments") / EXPERIMENT_ID

# Default config optimized for low-VRAM GPUs (e.g. GTX 1070 8GB)
DEFAULT_PER_DEVICE_BATCH = 1
DEFAULT_GRADIENT_ACCUMULATION = 32
DEFAULT_GRADIENT_CHECKPOINTING = True


def print_gpu_diagnostics() -> dict[str, Any]:
    """Print GPU diagnostics and return hardware info dict."""
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
        prop = torch.cuda.get_device_properties(0)
        info["gpu_name"] = prop.name
        info["vram_gb"] = round(prop.total_memory / 1024**3, 2)
        info["compute_capability"] = f"{prop.major}.{prop.minor}"
        info["bf16_supported"] = prop.major >= 8
        info["fp16_supported"] = True  # All CUDA GPUs support fp16

        print(f"GPU: {prop.name}")
        print(f"VRAM: {info['vram_gb']} GB")
        print(f"Compute Capability: ({prop.major}, {prop.minor})")
        print(f"BF16 supported: {info['bf16_supported']}")
        print(f"FP16 supported: {info['fp16_supported']}")

        # Show free/total memory estimate
        free_mem = torch.cuda.mem_get_info()[0]
        used_mem = prop.total_memory - free_mem
        print(f"Memory: {round(used_mem / 1024**3, 2)} GB used / {info['vram_gb']} GB total")
    else:
        print("No GPU available — running on CPU (will be very slow)")
    print(f"{'='*40}\n")
    return info


def load_split(path: Path) -> list[dict[str, str]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_all_splits(data_dir: str = "data/experiment") -> dict[str, list[dict[str, str]]]:
    split_dir = Path(data_dir)
    splits = {}
    for name in ["train", "validation", "calibration", "test"]:
        path = split_dir / f"{name}.jsonl"
        if path.exists():
            splits[name] = load_split(path)
            print(f"Loaded {name}: {len(splits[name])} entries from {path}")
        else:
            print(f"WARNING: {path} not found")
            splits[name] = []
    return splits


class CMUdictDataset(torch.utils.data.Dataset):
    """Dataset for ByT5 G2P training on CMUdict ARPAbet."""

    def __init__(
        self,
        data: list[dict[str, str]],
        tokenizer: Any,
        max_input_length: int = 64,
        max_target_length: int = 128,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
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


def compute_per_with_stress(ref: str, hyp: str) -> float:
    """Compute PER with stress digits preserved."""
    ref_phones = ref.split()
    hyp_phones = hyp.split()
    if not ref_phones:
        return 0.0 if not hyp_phones else 1.0
    m, n = len(ref_phones), len(hyp_phones)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_phones[i - 1] == hyp_phones[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n] / m


def compute_stress_stripped(phones: str) -> str:
    """Remove stress digits from ARPAbet phones."""
    result = []
    for phone in phones.split():
        stripped = phone.rstrip("012")
        if stripped:
            result.append(stripped)
    return PHONE_SEPARATOR.join(result)


def compute_per_without_stress(ref: str, hyp: str) -> float:
    """Compute PER with stress digits stripped."""
    ref_stripped = compute_stress_stripped(ref)
    hyp_stripped = compute_stress_stripped(hyp)
    return compute_per_with_stress(ref_stripped, hyp_stripped)


def compute_exact_match(ref: str, hyp: str) -> bool:
    """Check if prediction exactly matches reference."""
    return ref.split() == hyp.split()


def bootstrap_ci(per_values: list[float], n_trials: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Compute bootstrap 95% CI for mean PER."""
    rng = random.Random(seed)
    n = len(per_values)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot_means = []
    for _ in range(n_trials):
        sample = [per_values[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lower = boot_means[int(0.025 * n_trials)]
    upper = boot_means[int(0.975 * n_trials)]
    mean = sum(boot_means) / n_trials
    return mean, lower, upper


def generate_predictions(
    model: AutoModelForSeq2SeqLM,
    tokenizer: Any,
    dataset: CMUdictDataset,
    device: torch.device,
    max_new_tokens: int = 128,
) -> list[dict[str, Any]]:
    """Generate predictions for a dataset."""
    model.eval()
    predictions: list[dict[str, Any]] = []
    start_time = time.time()
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            attention_mask = item["attention_mask"].unsqueeze(0).to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            pred_tokens = outputs[0]
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            pred_text = " ".join(pred_text.upper().split())
            predictions.append({
                "word": item["word"],
                "reference": item["target"],
                "prediction": pred_text,
                "per_with_stress": compute_per_with_stress(item["target"], pred_text),
                "per_without_stress": compute_per_without_stress(item["target"], pred_text),
                "exact_match": compute_exact_match(item["target"], pred_text),
            })
            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{len(dataset)} predictions")
    wall_time = time.time() - start_time
    print(f"  Generated {len(predictions)} predictions in {wall_time:.1f}s")
    return predictions


def compute_oov_metrics(
    predictions: list[dict[str, Any]],
    train_data: list[dict[str, str]],
) -> dict[str, Any]:
    """Compute OOV metrics — test words whose root never appeared in training."""
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
    oov_per_ws = np.mean([p["per_with_stress"] for p in oov_predictions])
    oov_per_wos = np.mean([p["per_without_stress"] for p in oov_predictions])
    oov_em = np.mean([p["exact_match"] for p in oov_predictions])
    return {
        "oov_words": len(oov_predictions),
        "oov_per_with_stress": round(float(oov_per_ws), 4),
        "oov_per_without_stress": round(float(oov_per_wos), 4),
        "oov_exact_match": round(float(oov_em), 4),
        "note": f"{len(oov_predictions)} OOV words out of {len(predictions)} test words",
    }


def generate_error_analysis(predictions: list[dict[str, Any]], n_worst: int = 20) -> list[dict[str, Any]]:
    """Generate error analysis for worst predictions."""
    sorted_preds = sorted(predictions, key=lambda p: p["per_with_stress"], reverse=True)
    errors: list[dict[str, Any]] = []
    for pred in sorted_preds[:n_worst]:
        ref_phones = pred["reference"].split()
        hyp_phones = pred["prediction"].split()
        m, n = len(ref_phones), len(hyp_phones)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_phones[i - 1] == hyp_phones[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        errors.append({
            "word": pred["word"],
            "reference_phones": pred["reference"],
            "predicted_phones": pred["prediction"],
            "per_with_stress": round(pred["per_with_stress"], 4),
            "per_without_stress": round(pred["per_without_stress"], 4),
            "exact_match": pred["exact_match"],
            "word_length": len(pred["word"]),
            "ref_phone_count": len(ref_phones),
            "hyp_phone_count": len(hyp_phones),
            "edit_distance": dp[m][n],
        })
    return errors


def evaluate_reliability_layer(
    predictions: list[dict[str, Any]],
    calibration_data: list[dict[str, str]],
) -> dict[str, Any]:
    """Evaluate reliability/abstention layer against real errors."""
    results: list[dict[str, Any]] = []
    for pred in predictions:
        ref_len = len(pred["reference"].split())
        hyp_len = len(pred["prediction"].split())
        confidence = 1.0 - abs(ref_len - hyp_len) / max(ref_len, hyp_len) if ref_len > 0 else 0.0
        confidence = max(0.0, min(1.0, confidence))
        results.append({
            "word": pred["word"],
            "per": pred["per_with_stress"],
            "confidence": round(confidence, 4),
            "exact_match": pred["exact_match"],
        })
    buckets: dict[str, dict[str, int | float]] = {}
    n_buckets = 5
    for r in results:
        bucket_idx = min(n_buckets - 1, int(r["confidence"] * n_buckets))
        key = f"confidence_{bucket_idx}_{bucket_idx + 1}"
        if key not in buckets:
            buckets[key] = {"count": 0, "total_per": 0.0, "exact_matches": 0}
        buckets[key]["count"] += 1
        buckets[key]["total_per"] += r["per"]
        if r["exact_match"]:
            buckets[key]["exact_matches"] += 1
    bucket_results = []
    for key in sorted(buckets.keys()):
        b = buckets[key]
        bucket_results.append({
            "bucket": key,
            "sample_count": b["count"],
            "mean_per": round(b["total_per"] / b["count"], 4) if b["count"] > 0 else 0.0,
            "exact_match_rate": round(b["exact_matches"] / b["count"], 4) if b["count"] > 0 else 0.0,
        })
    is_monotonic = False
    if len(bucket_results) >= 2:
        per_values = [b["mean_per"] for b in bucket_results]
        is_monotonic = all(per_values[i] >= per_values[i + 1] for i in range(len(per_values) - 1))
    thresholds = [0.3, 0.5, 0.7]
    abstention_results = []
    for threshold in thresholds:
        selected = [r for r in results if r["confidence"] >= threshold]
        if selected:
            coverage = len(selected) / len(results)
            selective_per = sum(r["per"] for r in selected) / len(selected)
            abstention_results.append({
                "threshold": threshold,
                "coverage": round(coverage, 4),
                "selective_per": round(selective_per, 4),
                "abstention_rate": round(1.0 - coverage, 4),
            })
    return {
        "bucket_analysis": bucket_results,
        "monotonicity_preserved": is_monotonic,
        "abstention_results": abstention_results,
        "total_samples": len(results),
    }


def _train_with_oom_fallback(
    trainer: Trainer,
    model: AutoModelForSeq2SeqLM,
    args: argparse.Namespace,
    use_bf16: bool,
    use_fp16: bool,
) -> Any:
    """Retry training with progressively smaller batch sizes and optimizer changes on OOM."""
    fallback_batches = [b // 2 for b in [args.batch_size] if b // 2 >= 1]
    # Also try adafactor with current batch if adamw failed
    fallback_optimizers = ["adafactor"] if args.optimizer == "adamw" else []

    config_attempts = []
    for bs in fallback_batches:
        for opt in ["adamw", *fallback_optimizers]:
            for mp_name, fp16_on, bf16_on in [("fp16", True, False), ("bf16", False, True), ("fp32", False, False)]:
                config_attempts.append((bs, opt, mp_name, fp16_on, bf16_on))

    for batch_size, optimizer_name, amp_name, fp16, bf16 in config_attempts:
        print(f"\n  Retry: batch={batch_size}, optimizer={optimizer_name}, amp={amp_name}")
        try:
            new_args = TrainingArguments(
                output_dir=trainer.args.output_dir,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=args.learning_rate,
                weight_decay=0.01,
                warmup_steps=100,
                logging_steps=args.logging_steps,
                save_steps=args.save_steps,
                eval_strategy="steps",
                eval_steps=args.save_steps,
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="per",
                greater_is_better=False,
                report_to="none",
                seed=args.seed,
                bf16=bf16,
                fp16=fp16,
                gradient_accumulation_steps=args.gradient_accumulation,
                gradient_checkpointing=args.gradient_checkpointing,
                max_steps=args.max_steps,
            )
            optimizer_kwargs = {}
            if optimizer_name == "adafactor":
                from transformers import Adafactor
                optimizer_kwargs = {"optimizer_cls": Adafactor}
            new_trainer = Trainer(
                model=model, args=new_args, train_dataset=trainer.train_dataset,
                eval_dataset=trainer.eval_dataset, compute_metrics=trainer.compute_metrics,
                data_collator=trainer.data_collator, **optimizer_kwargs,
            )
            result = new_trainer.train()
            print(f"  Success with batch={batch_size}, optimizer={optimizer_name}, amp={amp_name}")
            return result
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM again: {e}")
            continue

    raise RuntimeError("All OOM fallback attempts failed. Consider using --optimizer adafactor or reducing batch size.")


def run_experiment(args: argparse.Namespace) -> None:
    """Run the full ByT5 G2P experiment."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Print hardware diagnostics
    gpu_info = print_gpu_diagnostics()
    gpu_available = gpu_info["gpu_available"]
    gpu_name = gpu_info["gpu_name"]

    # Determine mixed precision backend based on GPU capability
    use_bf16 = gpu_available and gpu_info["bf16_supported"] and (args.amp_backend == "auto" or args.amp_backend == "bf16")
    use_fp16 = gpu_available and (
        (args.amp_backend == "auto" and not use_bf16) or args.amp_backend == "fp16"
    )
    # For Pascal (CC 6.x), bf16 is not supported — only use fp16 if explicitly requested or auto
    cc = gpu_info["compute_capability"]
    if cc and cc.startswith("6.") and args.amp_backend == "auto":
        use_bf16 = False
        use_fp16 = False  # Start with fp32; fall back to fp16 only if OOM
        print("Pascal GPU detected (CC 6.x): starting with fp32, will try fp16 if OOM")
    elif cc and cc.startswith("7.") and args.amp_backend == "auto":
        use_fp16 = True  # Turing+ can use fp16 safely

    print(f"Mixed precision: bf16={use_bf16}, fp16={use_fp16}")

    print("\n=== Loading Data ===")
    splits = load_all_splits(args.data_dir)
    if args.max_samples:
        for name in splits:
            splits[name] = splits[name][:args.max_samples]
    train_data, val_data, cal_data, test_data = (
        splits["train"], splits["validation"], splits["calibration"], splits["test"],
    )
    print(f"\nTrain: {len(train_data)} | Val: {len(val_data)} | Cal: {len(cal_data)} | Test: {len(test_data)}")

    print(f"\n=== Loading Model: {MODEL_ID} ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    device = torch.device("cuda" if gpu_available else "cpu")
    model = model.to(device)
    try:
        import subprocess
        model_revision = subprocess.run(
            ["git", "ls-remote", "https://huggingface.co/google/byt5-small", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.split()[0][:12]
    except Exception:
        model_revision = "unknown"
    print(f"Model revision: {model_revision}")

    train_dataset = CMUdictDataset(train_data, tokenizer, args.max_input_length, args.max_target_length)
    val_dataset = CMUdictDataset(val_data, tokenizer, args.max_input_length, args.max_target_length)
    test_dataset = CMUdictDataset(test_data, tokenizer, args.max_input_length, args.max_target_length)
    CMUdictDataset(cal_data, tokenizer, args.max_input_length, args.max_target_length)

    output_path = OUTPUT_DIR / f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    output_path.mkdir(parents=True, exist_ok=True)

    # When max_steps is set, use step-based training instead of epoch-based
    train_epochs_override = 1 if args.max_steps else args.epochs

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=train_epochs_override,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="per",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    def compute_metrics(eval_pred):
        # eval_pred is EvalPrediction(predictions=..., label_ids=...)
        # predictions may be logits tuple or ndarray
        raw_preds = eval_pred.predictions
        label_ids = eval_pred.label_ids
        logits = raw_preds[0] if isinstance(raw_preds, tuple) else raw_preds
        pred_tokens = np.argmax(logits, axis=-1)
        pred_texts = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
        masked_labels = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        label_texts = tokenizer.batch_decode(masked_labels, skip_special_tokens=True)
        per_values, per_wo_stress, exact_matches = [], [], []
        for ref, hyp in zip(label_texts, pred_texts, strict=False):
            ref, hyp = ref.strip().upper(), hyp.strip().upper()
            per_values.append(compute_per_with_stress(ref, hyp))
            per_wo_stress.append(compute_per_without_stress(ref, hyp))
            exact_matches.append(1.0 if ref.split() == hyp.split() else 0.0)
        return {
            "per": float(np.mean(per_values)),
            "per_without_stress": float(np.mean(per_wo_stress)),
            "exact_match": float(np.mean(exact_matches)),
        }

    trainer_kwargs: dict[str, Any] = dict(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, compute_metrics=compute_metrics,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )
    if args.optimizer == "adafactor":
        from transformers import Adafactor
        trainer_kwargs["optimizer_cls"] = Adafactor
    trainer = Trainer(**trainer_kwargs)
    print("\n=== Training ===")
    print(f"Epochs: {args.epochs}, LR: {args.learning_rate}, Batch: {args.batch_size}")
    print(f"Grad accum: {args.gradient_accumulation}, Checkpointing: {args.gradient_checkpointing}")
    print(f"Mixed precision: bf16={use_bf16}, fp16={use_fp16}")

    train_start = time.time()
    try:
        train_result = trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        print(f"\nOOM at batch_size={args.batch_size}: {e}")
        print("Falling back to smaller batch sizes...")
        train_result = _train_with_oom_fallback(trainer, model, args, use_bf16, use_fp16)
    train_time = time.time() - train_start
    print(f"Training completed in {train_time:.1f}s, loss: {train_result.training_loss:.4f}")

    best_checkpoint = output_path / "best_model"
    best_checkpoint.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_checkpoint))
    tokenizer.save_pretrained(str(best_checkpoint))
    print(f"Best model saved to {best_checkpoint}")

    print("\n=== Validation Evaluation ===")
    val_preds = generate_predictions(model, tokenizer, val_dataset, device)
    val_per_ws = np.mean([p["per_with_stress"] for p in val_preds])
    val_per_wos = np.mean([p["per_without_stress"] for p in val_preds])
    val_em = np.mean([p["exact_match"] for p in val_preds])
    val_ci = bootstrap_ci([p["per_with_stress"] for p in val_preds])
    print(f"VAL PER (w/ stress): {val_per_ws:.4f} [{val_ci[1]:.4f}, {val_ci[2]:.4f}]")
    print(f"VAL PER (w/o stress): {val_per_wos:.4f}")
    print(f"VAL Exact Match: {val_em:.4f}")

    print("\n=== Test Evaluation (Frozen) ===")
    if args.max_steps:
        print("SKIP — canary mode (--max-steps), frozen test must not be opened")
        test_preds = []
        test_per_ws, test_per_wos, test_em = 0.0, 0.0, 0.0
        test_ci = (0.0, 0.0, 0.0)
        oov_metrics = {"oov_words": 0, "oov_total": 0, "oov_per_with_stress": None,
                        "oov_per_without_stress": None, "oov_exact_match": None,
                        "note": "canary run — test not evaluated"}
        error_analysis = []
        rel_results = {"bucket_analysis": [], "monotonicity_preserved": None,
                        "abstention_results": [], "total_samples": 0}
    else:
        test_preds = generate_predictions(model, tokenizer, test_dataset, device)
        test_per_ws = np.mean([p["per_with_stress"] for p in test_preds])
        test_per_wos = np.mean([p["per_without_stress"] for p in test_preds])
        test_em = np.mean([p["exact_match"] for p in test_preds])
        test_ci = bootstrap_ci([p["per_with_stress"] for p in test_preds])

        oov_metrics = compute_oov_metrics(test_preds, train_data)
        print(f"\nOOV Words: {oov_metrics['oov_words']}")
        if oov_metrics["oov_per_with_stress"] is not None:
            print(f"OOV PER (w/ stress): {oov_metrics['oov_per_with_stress']:.4f}")
            print(f"OOV PER (w/o stress): {oov_metrics['oov_per_without_stress']:.4f}")
            print(f"OOV Exact Match: {oov_metrics['oov_exact_match']:.4f}")
        else:
            print(oov_metrics["note"])

        error_analysis = generate_error_analysis(test_preds, n_worst=20)
        error_path = output_path / "error_analysis.csv"
        with error_path.open("w", encoding="utf-8") as f:
            f.write("word,reference_phones,predicted_phones,per_with_stress,per_without_stress,exact_match,word_length,ref_phone_count,hyp_phone_count,edit_distance\n")
            for e in error_analysis:
                f.write(f"{e['word']},{e['reference_phones']},{e['predicted_phones']},"
                        f"{e['per_with_stress']},{e['per_without_stress']},{e['exact_match']},"
                        f"{e['word_length']},{e['ref_phone_count']},{e['hyp_phone_count']},{e['edit_distance']}\n")
        print(f"Error analysis saved to {error_path}")

        print("\n=== Reliability Layer Evaluation ===")
        rel_results = evaluate_reliability_layer(test_preds, cal_data)
        print(f"Monotonicity preserved: {rel_results['monotonicity_preserved']}")
        for b in rel_results["bucket_analysis"]:
            print(f"  {b['bucket']}: n={b['sample_count']}, mean_per={b['mean_per']:.4f}, em={b['exact_match_rate']:.4f}")
        for a in rel_results["abstention_results"]:
            print(f"  Threshold {a['threshold']}: coverage={a['coverage']:.4f}, selective_per={a['selective_per']:.4f}, abstention={a['abstention_rate']:.4f}")

    checkpoint_hash = ""
    ckpt_path = best_checkpoint / "pytorch_model.bin"
    if ckpt_path.exists():
        h = hashlib.sha256()
        with ckpt_path.open("rb") as fh:
            while True:
                chunk = fh.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        checkpoint_hash = h.hexdigest()[:16]

    try:
        import subprocess
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()[:8]
    except Exception:
        git_sha = "unknown"

    test_manifest_path = Path("reports/test_manifest.json")
    if test_manifest_path.exists():
        with test_manifest_path.open("r", encoding="utf-8") as fh:
            tm = json.load(fh)
            tm.get("test_sha256", "")[:16]

    results = {
        "experiment_id": EXPERIMENT_ID,
        "status": "CANARY_RUN" if args.max_steps else ("REAL_GPU_RUN_PENDING" if not gpu_available else "COMPLETED"),
        "hardware": {
            "gpu_available": gpu_available, "gpu_name": gpu_name,
            "pytorch_version": torch.__version__,
            "cuda_version": gpu_info.get("cuda_version", "N/A"),
            "vram_gb": gpu_info.get("vram_gb", 0.0),
            "compute_capability": gpu_info.get("compute_capability"),
            "bf16_supported": gpu_info.get("bf16_supported", False),
            "mixed_precision_used": "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"),
        },
        "dataset": {"train_size": len(train_data), "validation_size": len(val_data),
                     "calibration_size": len(cal_data), "test_size": len(test_data), "seed": args.seed},
        "training": {"epochs": args.epochs, "learning_rate": args.learning_rate,
                      "batch_size": args.batch_size, "gradient_accumulation": args.gradient_accumulation,
                      "gradient_checkpointing": args.gradient_checkpointing,
                      "optimizer": args.optimizer,
                      "mixed_precision": "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"),
                      "wall_time_seconds": train_time,
                      "max_steps": args.max_steps},
        "model": {"base_id": MODEL_ID, "fine_tuned_id": f"byt5-cmudict-{EXPERIMENT_ID}",
                   "revision": model_revision, "checkpoint_hash": checkpoint_hash},
        "test_results": {
            "per_with_stress": float(test_per_ws),
            "per_without_stress": float(test_per_wos),
            "exact_match": float(test_em),
            "n_samples": len(test_preds) if test_preds else 0,
            "bootstrap_ci_95": [float(test_ci[1]), float(test_ci[2])] if test_preds else [0.0, 0.0],
            "oov_metrics": oov_metrics,
            "canary_mode": bool(args.max_steps),
        },
        "reliability": rel_results,
        "git_sha": git_sha,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    results_path = output_path / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {results_path}")

    final_path = OUTPUT_DIR / EXPERIMENT_ID
    if final_path.exists():
        shutil.rmtree(final_path)
    shutil.copytree(output_path, final_path)
    print(f"Results copied to {final_path}")

    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Status: {results['status']}")
    print(f"Hardware: {gpu_name}")
    print(f"Dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    print(f"Test PER (w/ stress): {test_per_ws:.4f}")
    print(f"Test PER (w/o stress): {test_per_wos:.4f}")
    print(f"Test Exact Match: {test_em:.4f}")
    oov_per = oov_metrics.get('oov_per_with_stress')
    print(f"OOV PER (w/ stress): {oov_per if oov_per is not None else 'N/A'}")
    print(f"Bootstrap 95% CI: [{test_ci[1]:.4f}, {test_ci[2]:.4f}]")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ByT5 CMUdict G2P Experiment")
    parser.add_argument("--data-dir", default="data/experiment")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=64)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit each split to N samples (for smoke testing)")
    parser.add_argument("--max-steps", type=int, default=None, help="Run at most N training steps (for canary/validation runs)")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True, dest="gradient_checkpointing", help="Enable gradient checkpointing to save VRAM (default: on)")
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--optimizer", choices=["adamw", "adafactor"], default="adamw", help="Optimizer (adafactor saves VRAM)")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Eval batch size (default: same as train)")
    parser.add_argument("--logging-steps", type=int, default=100, help="Logging interval in steps")
    parser.add_argument("--amp-backend", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                        help="Mixed precision backend (auto: bf16 on Ampere+, fp16 on Turing+, fp32 on Pascal)")
    args = parser.parse_args()
    if args.max_samples:
        print(f"Running smoke test with max {args.max_samples} samples per split")
    if args.max_steps:
        print(f"Running canary mode: max {args.max_steps} steps (no frozen test evaluation)")
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    run_experiment(args)


if __name__ == "__main__":
    main()

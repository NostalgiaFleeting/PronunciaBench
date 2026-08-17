"""Dataset loading and management utilities."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pronunciabench.data.models import PronunciationExample, VerificationStatus


@dataclass
class DatasetStats:
    n_examples: int
    n_verified: int
    n_unverified: int
    n_exploratory: int
    locales: dict[str, int]
    splits: dict[str, int]
    source_stats: dict[str, int]
    unique_graphemes: int
    unique_phonemes: int


def load_jsonl(path: str) -> list[PronunciationExample]:
    """Load examples from a JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                examples.append(PronunciationExample(**data))
    return examples


def compute_stats(examples: list[PronunciationExample]) -> DatasetStats:
    """Compute dataset statistics."""
    locales = Counter(e.locale for e in examples if e.locale)
    splits = Counter(e.split for e in examples if e.split)
    sources = Counter(e.source for e in examples)
    verified = sum(1 for e in examples if e.verification_status == VerificationStatus.VERIFIED)
    unverified = sum(1 for e in examples if e.verification_status == VerificationStatus.UNVERIFIED)
    exploratory = sum(1 for e in examples if e.verification_status == VerificationStatus.EXPLORATORY)

    graphemes = set()
    phonemes = set()
    for ex in examples:
        graphemes.update(ex.text.lower())
        phonemes.update(ex.pronunciation.lower())

    return DatasetStats(
        n_examples=len(examples),
        n_verified=verified,
        n_unverified=unverified,
        n_exploratory=exploratory,
        locales=dict(locales),
        splits=dict(splits),
        source_stats=dict(sources),
        unique_graphemes=len(graphemes),
        unique_phonemes=len(phonemes),
    )


def split_dataset(
    examples: list[PronunciationExample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[PronunciationExample], list[PronunciationExample], list[PronunciationExample]]:
    """Split dataset into train/val/test."""
    import random
    rng = random.Random(seed)
    shuffled = list(examples)  # Copy to avoid mutating the input
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:])


def generate_report(stats: DatasetStats) -> str:
    """Generate human-readable dataset report."""
    lines = [
        f"Total examples: {stats.n_examples}",
        f"  Verified:     {stats.n_verified}",
        f"  Unverified:   {stats.n_unverified}",
        f"  Exploratory:  {stats.n_exploratory}",
        f"Unique graphemes: {stats.unique_graphemes}",
        f"Unique phonemes:  {stats.unique_phonemes}",
        f"Locale distribution: {dict(stats.locales)}",
        f"Split distribution:  {dict(stats.splits)}",
        f"Source distribution: {dict(stats.source_stats)}",
    ]
    return "\n".join(lines)
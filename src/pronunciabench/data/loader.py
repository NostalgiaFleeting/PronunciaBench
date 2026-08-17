"""Dataset loading and management utilities."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field

from pronunciabench.data.models import PronunciationExample, VerificationStatus


@dataclass
class LeakageReport:
    exact_text_overlaps: list = field(default_factory=list)
    normalized_text_overlaps: list = field(default_factory=list)
    duplicate_pronunciations: list = field(default_factory=list)
    cross_split_pairs: list = field(default_factory=list)

    @property
    def has_leakage(self) -> bool:
        return bool(self.cross_split_pairs)

    def to_dict(self) -> dict:
        return {
            "exact_text_overlaps": len(self.exact_text_overlaps),
            "normalized_text_overlaps": len(self.normalized_text_overlaps),
            "duplicate_pronunciations": len(self.duplicate_pronunciations),
            "cross_split_pairs": len(self.cross_split_pairs),
            "has_leakage": self.has_leakage,
            "cross_split_details": self.cross_split_pairs[:20],
        }


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
    phoneme_systems: dict[str, int] = field(default_factory=dict)
    dataset_hash: str = ""
    duplicate_count: int = 0
    leakage: LeakageReport = field(default_factory=LeakageReport)


def load_jsonl(path: str) -> list[PronunciationExample]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                examples.append(PronunciationExample(**data))
    return examples


def compute_dataset_hash(examples: list[PronunciationExample]) -> str:
    lines = []
    for e in sorted(examples, key=lambda x: x.text):
        lines.append(f"{e.text.lower()}\t{e.pronunciation.lower()}\t{e.locale or ''}")
    content = "\n".join(lines)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def audit_leakage(examples: list[PronunciationExample]) -> LeakageReport:
    by_split: dict[str, list] = {}
    for ex in examples:
        split = ex.split or "unsplit"
        by_split.setdefault(split, []).append(ex)

    report = LeakageReport()
    split_names = list(by_split.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            texts1 = {ex.text.lower().strip() for ex in by_split[s1]}
            texts2 = {ex.text.lower().strip() for ex in by_split[s2]}
            overlaps = texts1 & texts2
            for text in overlaps:
                report.exact_text_overlaps.append((text, s1, s2))
                for ex1 in by_split[s1]:
                    if ex1.text.lower().strip() == text:
                        for ex2 in by_split[s2]:
                            if ex2.text.lower().strip() == text:
                                report.cross_split_pairs.append((ex1.text, ex2.text, s1, s2))

    pron_map: dict = {}
    for ex in examples:
        key = (ex.pronunciation.lower().strip(), ex.locale or "")
        pron_map.setdefault(key, []).append(ex)
    for _key, exs in pron_map.items():
        if len(exs) > 1:
            report.duplicate_pronunciations.append((exs[0].text, exs[1].text))
    return report


def compute_stats(examples: list[PronunciationExample]) -> DatasetStats:
    locales = Counter(e.locale for e in examples if e.locale)
    splits = Counter(e.split for e in examples if e.split)
    sources = Counter(e.source for e in examples)
    verified = sum(1 for e in examples if e.verification_status == VerificationStatus.VERIFIED)
    unverified = sum(1 for e in examples if e.verification_status == VerificationStatus.UNVERIFIED)
    exploratory = sum(1 for e in examples if e.verification_status == VerificationStatus.EXPLORATORY)
    phoneme_sys = Counter(e.phoneme_system.value for e in examples if e.phoneme_system)
    graphemes = set()
    phonemes = set()
    for ex in examples:
        graphemes.update(ex.text.lower())
        phonemes.update(ex.pronunciation.lower())
    dataset_hash = compute_dataset_hash(examples)
    leakage = audit_leakage(examples)
    return DatasetStats(
        n_examples=len(examples), n_verified=verified, n_unverified=unverified,
        n_exploratory=exploratory, locales=dict(locales), splits=dict(splits),
        source_stats=dict(sources), unique_graphemes=len(graphemes),
        unique_phonemes=len(phonemes), phoneme_systems=dict(phoneme_sys),
        dataset_hash=dataset_hash, duplicate_count=len(leakage.duplicate_pronunciations),
        leakage=leakage,
    )


def split_dataset(
    examples: list[PronunciationExample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:])


def generate_report(stats: DatasetStats) -> str:
    lines = [
        f"Total examples: {stats.n_examples}",
        f"  Verified:     {stats.n_verified}",
        f"  Unverified:   {stats.n_unverified}",
        f"  Exploratory:  {stats.n_exploratory}",
        f"Unique graphemes: {stats.unique_graphemes}",
        f"Unique phonemes:  {stats.unique_phonemes}",
        f"Phoneme systems:  {dict(stats.phoneme_systems)}",
        f"Locale distribution: {dict(stats.locales)}",
        f"Split distribution:  {dict(stats.splits)}",
        f"Source distribution: {dict(stats.source_stats)}",
        f"Dataset hash: {stats.dataset_hash}",
        f"Duplicate pronunciations: {stats.duplicate_count}",
        f"Cross-split overlaps: {len(stats.leakage.cross_split_pairs)}",
        f"Leakage detected: {stats.leakage.has_leakage}",
    ]
    if stats.leakage.cross_split_pairs:
        lines.append("Cross-split duplicates:")
        for t1, t2, s1, s2 in stats.leakage.cross_split_pairs[:10]:
            lines.append(f"  '{t1}' in [{s1}] and '{t2}' in [{s2}]")
    return "\n".join(lines)

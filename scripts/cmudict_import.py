"""CMUdict importer for ByT5 G2P experiment."""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"


@dataclass
class CMUdictEntry:
    """A single pronunciation entry from CMUdict."""

    word: str
    pronunciation: str
    label: str
    stress_preserved: bool = True


def parse_cmudict_line(line: str) -> CMUdictEntry | None:
    """Parse a single CMUdict line."""
    line = line.strip()
    if not line or line.startswith(";;;"):
        return None
    parts = line.split()
    if len(parts) < 2:
        return None
    label = parts[0]
    phones = " ".join(parts[1:])
    if not re.match(r"^[A-Z0-9\s]+$", phones):
        return None
    word_match = re.match(r"^([A-Za-z]+)", label)
    if not word_match:
        return None
    word = word_match.group(1).upper()
    return CMUdictEntry(word=word, pronunciation=phones, label=label, stress_preserved=True)


def fetch_cmudict(force_download: bool = False, cache_dir: str = "data/cmudict") -> list[CMUdictEntry]:
    """Fetch and parse CMUdict from GitHub."""
    cache_path = Path(cache_dir) / "cmudict.dict"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not force_download and cache_path.exists():
        raw_text = cache_path.read_text(encoding="utf-8")
    else:
        print(f"Downloading CMUdict from {CMUDICT_URL}...")
        r = requests.get(CMUDICT_URL, timeout=60)
        r.raise_for_status()
        raw_text = r.text
        cache_path.write_text(raw_text, encoding="utf-8")
        print(f"Cached to {cache_path}")
    entries: list[CMUdictEntry] = []
    for line in raw_text.splitlines():
        entry = parse_cmudict_line(line)
        if entry is not None:
            entries.append(entry)
    print(f"Parsed {len(entries)} entries from CMUdict")
    return entries


def compute_sha256(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_lexical_root(word: str) -> str:
    """Get the lexical root for grouping (lowercased word)."""
    return word.lower()


def split_dataset(
    entries: list[CMUdictEntry],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    cal_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> dict[str, list[CMUdictEntry]]:
    """Split CMUdict entries into train/val/cal/test with lexical grouping."""
    rng = random.Random(seed)
    groups: dict[str, list[CMUdictEntry]] = defaultdict(list)
    for entry in entries:
        root = get_lexical_root(entry.word)
        groups[root].append(entry)
    sorted_roots = sorted(groups.keys())
    rng.shuffle(sorted_roots)
    n_groups = len(sorted_roots)
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)
    n_cal = int(n_groups * cal_ratio)
    splits: dict[str, list[CMUdictEntry]] = {"train": [], "validation": [], "calibration": [], "test": []}
    idx = 0
    for root in sorted_roots[:n_train]:
        splits["train"].extend(groups[root])
    idx = n_train
    for root in sorted_roots[idx : idx + n_val]:
        splits["validation"].extend(groups[root])
    idx += n_val
    for root in sorted_roots[idx : idx + n_cal]:
        splits["calibration"].extend(groups[root])
    idx += n_cal
    for root in sorted_roots[idx:]:
        splits["test"].extend(groups[root])
    for split_name in splits:
        splits[split_name] = rng.sample(splits[split_name], k=len(splits[split_name]))
    return splits


def audit_leakage(splits: dict[str, list[CMUdictEntry]]) -> dict[str, object]:
    """Audit for leakage between splits."""
    word_sets: dict[str, set[str]] = {}
    for split_name, entries in splits.items():
        word_sets[split_name] = {e.word.lower() for e in entries}
    leakage: dict[str, dict[str, object]] = {}
    split_names = list(splits.keys())
    for i, name_a in enumerate(split_names):
        for name_b in split_names[i + 1 :]:
            overlap = word_sets[name_a] & word_sets[name_b]
            leakage[f"{name_a}_vs_{name_b}"] = {
                "overlap_count": len(overlap),
                "overlapping_words": sorted(overlap)[:10],
            }
    total = sum(v["overlap_count"] for v in leakage.values())
    return {
        "leakage_detected": total > 0,
        "total_overlapping_words": total,
        "pairwise": leakage,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "unique_words_per_split": {k: len(word_sets[k]) for k in word_sets},
    }


def main() -> None:
    """Main entry point for CMUdict import and splitting."""
    import argparse

    parser = argparse.ArgumentParser(description="Import CMUdict and create splits")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--cache-dir", default="data/cmudict")
    parser.add_argument("--output-dir", default="data/experiment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    entries = fetch_cmudict(force_download=args.force_download, cache_dir=args.cache_dir)
    if not entries:
        print("ERROR: No entries parsed from CMUdict", file=sys.stderr)
        sys.exit(1)

    try:
        import subprocess
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()[:8]
    except Exception:
        git_sha = "unknown"

    splits = split_dataset(entries, seed=args.seed)
    audit = audit_leakage(splits)
    print(f"\nLeakage audit: {audit['total_overlapping_words']} overlapping words")
    for split_name, count in audit["split_sizes"].items():
        print(f"  {split_name}: {count} entries, {audit['unique_words_per_split'][split_name]} unique words")

    if audit["leakage_detected"]:
        print("WARNING: Leakage detected between splits!", file=sys.stderr)
        for pair, info in audit["pairwise"].items():
            if info["overlap_count"] > 0:
                print(f"  {pair}: {info['overlap_count']} overlapping words")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "splits": {},
        "audit": audit,
    }

    for split_name, split_entries in splits.items():
        sorted_e = sorted(split_entries, key=lambda e: (e.word.lower(), e.label))
        content = "\n".join(f"{e.word}\t{e.pronunciation}\t{e.label}" for e in sorted_e)
        sha = compute_sha256(content)
        manifest["splits"][split_name] = {
            "sha256": sha,
            "n_entries": len(split_entries),
            "n_unique_words": len(set(e.word.lower() for e in split_entries)),
            "first_word": sorted_e[0].word if sorted_e else None,
            "last_word": sorted_e[-1].word if sorted_e else None,
        }
        jsonl_path = output_dir / f"{split_name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for entry in sorted_e:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        print(f"Saved {split_name}: {len(split_entries)} entries -> {jsonl_path}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    audit_path = Path("reports/cmudict_data_audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved audit report -> {audit_path}")

    test_manifest = {
        "dataset_revision": "cmudict-master",
        "split_seed": args.seed,
        "test_sha256": manifest["splits"]["test"]["sha256"],
        "n_lexical_items": manifest["splits"]["test"]["n_unique_words"],
        "n_pronunciation_variants": manifest["splits"]["test"]["n_entries"],
        "timestamp": manifest["generated_at"],
        "git_sha": git_sha,
    }
    test_path = Path("reports/test_manifest.json")
    test_path.write_text(json.dumps(test_manifest, indent=2), encoding="utf-8")
    print(f"Saved test manifest -> {test_path}")
    print("\nDone. Experiment data ready in", args.output_dir)


if __name__ == "__main__":
    main()



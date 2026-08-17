"""Tests for data loading and management."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from pronunciabench.data import load_jsonl, compute_stats, generate_report, split_dataset
from pronunciabench.data.models import PronunciationExample, VerificationStatus


class TestLoadJsonl:
    def test_load_samples(self, tmp_path: Path):
        data = [
            {"text": "Smith", "pronunciation": "/smɪθ/", "locale": "en-US", "source": "CMU"},
            {"text": "Garcia", "pronunciation": "/ɡarsia/", "locale": "es-ES", "source": "WikiPron"},
        ]
        f = tmp_path / "test.jsonl"
        with open(f, "w", encoding="utf-8") as fh:
            for ex in data:
                fh.write(json.dumps(ex) + "\n")
        examples = load_jsonl(str(f))
        assert len(examples) == 2
        assert examples[0].text == "Smith"
        assert examples[1].locale == "es-ES"

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        examples = load_jsonl(str(f))
        assert len(examples) == 0


class TestComputeStats:
    def test_basic_stats(self):
        examples = [
            PronunciationExample(text="Smith", pronunciation="/smɪθ/", locale="en-US",
                                 source="CMU", verification_status=VerificationStatus.VERIFIED),
            PronunciationExample(text="Garcia", pronunciation="/ɡarsia/", locale="es-ES",
                                 source="WikiPron", verification_status=VerificationStatus.UNVERIFIED),
        ]
        stats = compute_stats(examples)
        assert stats.n_examples == 2
        assert stats.n_verified == 1
        assert stats.n_unverified == 1
        assert "en-US" in stats.locales
        assert "es-ES" in stats.locales

    def test_report_output(self):
        examples = [
            PronunciationExample(text="Test", pronunciation="/tɛst/", locale="en-US",
                                 source="CMU", verification_status=VerificationStatus.VERIFIED),
        ]
        stats = compute_stats(examples)
        report = generate_report(stats)
        assert "Total examples: 1" in report
        assert "en-US" in report


class TestSplitDataset:
    def test_split_ratios(self):
        examples = [
            PronunciationExample(text=f"Name{i}", pronunciation=f"/na{i}me/", locale="en-US",
                                 source="test", verification_status=VerificationStatus.VERIFIED)
            for i in range(100)
        ]
        train, val, test = split_dataset(examples, train_ratio=0.8, val_ratio=0.1, seed=42)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10
        assert len(train) + len(val) + len(test) == len(examples)

    def test_deterministic(self):
        examples = [
            PronunciationExample(text=f"Name{i}", pronunciation=f"/na{i}me/", locale="en-US",
                                 source="test", verification_status=VerificationStatus.VERIFIED)
            for i in range(50)
        ]
        t1, v1, te1 = split_dataset(examples, seed=42)
        t2, v2, te2 = split_dataset(examples, seed=42)
        assert [e.text for e in t1] == [e.text for e in t2]
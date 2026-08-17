"""Tests for dataset leakage detection."""

from __future__ import annotations

from pronunciabench.data import audit_leakage, compute_stats, load_jsonl
from pronunciabench.data.models import PronunciationExample, VerificationStatus


class TestLeakageDetection:
    def test_no_leakage_clean_splits(self):
        train = [
            PronunciationExample(text="Smith", pronunciation="/smɪθ/", locale="en-US",
                                 source="test", split="train",
                                 verification_status=VerificationStatus.VERIFIED),
            PronunciationExample(text="Jones", pronunciation="/dʒoʊnz/", locale="en-US",
                                 source="test", split="train",
                                 verification_status=VerificationStatus.VERIFIED),
        ]
        test = [
            PronunciationExample(text="Brown", pronunciation="/bɹaʊn/", locale="en-US",
                                 source="test", split="test",
                                 verification_status=VerificationStatus.VERIFIED),
        ]
        all_examples = train + test
        report = audit_leakage(all_examples)
        assert not report.has_leakage
        assert len(report.cross_split_pairs) == 0

    def test_detects_exact_overlap(self):
        train = [
            PronunciationExample(text="Dubois", pronunciation="/dybwa/", locale="fr-FR",
                                 source="test", split="train",
                                 verification_status=VerificationStatus.VERIFIED),
        ]
        test = [
            PronunciationExample(text="Dubois", pronunciation="/dybwa/", locale="fr-FR",
                                 source="test", split="test",
                                 verification_status=VerificationStatus.VERIFIED),
        ]
        report = audit_leakage(train + test)
        assert report.has_leakage
        assert len(report.cross_split_pairs) == 1
        assert report.cross_split_pairs[0][0] == "Dubois"
        assert report.cross_split_pairs[0][2] == "train"
        assert report.cross_split_pairs[0][3] == "test"

    def test_detects_duplicate_pronunciation(self):
        examples = [
            PronunciationExample(text="Smith", pronunciation="/smɪθ/", locale="en-US",
                                 source="test", split="train",
                                 verification_status=VerificationStatus.VERIFIED),
            PronunciationExample(text="smith", pronunciation="/smɪθ/", locale="en-US",
                                 source="test", split="train",
                                 verification_status=VerificationStatus.VERIFIED),
        ]
        report = audit_leakage(examples)
        assert len(report.duplicate_pronunciations) >= 1

    def test_real_dataset_no_leakage(self):
        """Verify our sample datasets have no cross-split leakage."""
        test_exs = load_jsonl("data/samples/test.jsonl")
        train_exs = load_jsonl("data/samples/train.jsonl")
        all_exs = test_exs + train_exs
        stats = compute_stats(all_exs)
        assert not stats.leakage.has_leakage, \
            f"Leakage detected: {stats.leakage.to_dict()}"
        assert stats.duplicate_count == 0

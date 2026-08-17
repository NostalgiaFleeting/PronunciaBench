"""Tests for reliability scoring and conformal abstention."""

from __future__ import annotations

import pytest

from pronunciabench.data.models import AbstentionDecision
from pronunciabench.reliability.scorer import ConformalAbstainer, ReliabilityScorer


class TestReliabilityScorer:
    def test_single_prediction_high_confidence(self):
        scorer = ReliabilityScorer()
        result = scorer.score(["/smɪθ/"], "en-US")
        assert result.confidence > 0.5
        assert result.decision == AbstentionDecision.ACCEPT

    def test_disagreement_low_confidence(self):
        scorer = ReliabilityScorer(abstention_threshold=0.9)
        result = scorer.score(["/smɪθ/", "/XXXX/", "/YYYY/"], "en-US")
        assert result.confidence < 0.9
        assert result.decision == AbstentionDecision.ABSTAIN

    def test_agreement_high_confidence(self):
        scorer = ReliabilityScorer()
        result = scorer.score(["/smɪθ/", "/smɪθ/", "/smɪθ/"], "en-US")
        assert result.confidence > 0.5
        assert result.components["model_agreement"] == 1.0

    def test_locale_support(self):
        scorer = ReliabilityScorer()
        result_well = scorer.score(["/smɪθ/"], "en-US")
        result_poor = scorer.score(["/smɪθ/"], "xx-XX")
        assert result_well.components["locale_support"] >= result_poor.components["locale_support"]

    def test_components_explainable(self):
        scorer = ReliabilityScorer()
        result = scorer.score(["/smɪθ/"], "en-US")
        assert "model_agreement" in result.components
        assert "sequence_confidence" in result.components
        assert "locale_support" in result.components
        assert "prediction_stability" in result.components
        assert "oov_signal" in result.components
        assert isinstance(result.reason, str) and len(result.reason) > 0

    def test_fallback_detection(self):
        scorer = ReliabilityScorer()
        result = scorer.score(["/[unknown]/"], "en-US")
        assert result.components["oov_signal"] < 1.0


class TestConformalAbstainer:
    def test_calibration(self):
        abstainer = ConformalAbstainer(alpha=0.05)
        abstainer.calibrate([0.1, 0.2, 0.3, 0.4, 0.5])
        assert abstainer._q_hat is not None
        assert 0.3 <= abstainer._q_hat <= 0.5

    def test_accept_low_score(self):
        abstainer = ConformalAbstainer(alpha=0.05)
        abstainer.calibrate([0.1, 0.2, 0.3, 0.4, 0.5])
        decision = abstainer.predict(0.15)
        assert decision == AbstentionDecision.ACCEPT

    def test_abstain_high_score(self):
        abstainer = ConformalAbstainer(alpha=0.05)
        abstainer.calibrate([0.1, 0.2, 0.3, 0.4, 0.5])
        decision = abstainer.predict(0.6)
        assert decision == AbstentionDecision.ABSTAIN

    def test_no_calibration_accepts(self):
        abstainer = ConformalAbstainer(alpha=0.05)
        decision = abstainer.predict(0.9)
        assert decision == AbstentionDecision.ACCEPT

    def test_empty_calibration(self):
        abstainer = ConformalAbstainer(alpha=0.05)
        abstainer.calibrate([])
        assert abstainer._q_hat == 1.0

    def test_evaluate_coverage(self):
        abstainer = ConformalAbstainer(alpha=0.1)
        abstainer.calibrate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        labels = [True, True, False, True, False]
        result = abstainer.evaluate_coverage(scores, labels)
        assert "coverage" in result
        assert "abstention_rate" in result
        assert "selective_accuracy" in result
        assert result["coverage"] + result["abstention_rate"] == 1.0
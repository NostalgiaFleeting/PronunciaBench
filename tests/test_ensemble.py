"""Tests for ensemble consensus."""

from __future__ import annotations

from pronunciabench.data.models import PronunciationPrediction
from pronunciabench.ensemble.consensus import ConsensusEngine


class TestConsensusEngine:
    def test_single_model(self):
        preds = [PronunciationPrediction(model_name="A", prediction="/smɪθ/")]
        engine = ConsensusEngine(preds)
        result = engine.compute_consensus()
        assert result.consensus_pronunciation == "/smɪθ/"
        assert result.agreement_score == 1.0
        assert result.n_models == 1

    def test_identical_predictions(self):
        preds = [
            PronunciationPrediction(model_name="A", prediction="/smɪθ/"),
            PronunciationPrediction(model_name="B", prediction="/smɪθ/"),
        ]
        engine = ConsensusEngine(preds)
        result = engine.compute_consensus()
        assert result.agreement_score == 1.0
        assert result.outlier_models == []

    def test_disagreement(self):
        preds = [
            PronunciationPrediction(model_name="A", prediction="/smɪθ/"),
            PronunciationPrediction(model_name="B", prediction="/XXXX/"),
        ]
        engine = ConsensusEngine(preds)
        result = engine.compute_consensus()
        assert result.agreement_score < 1.0
        assert result.n_models == 2
        assert "model_distances" in dir(result)

    def test_three_models(self):
        preds = [
            PronunciationPrediction(model_name="A", prediction="/smɪθ/"),
            PronunciationPrediction(model_name="B", prediction="/smɪθ/"),
            PronunciationPrediction(model_name="C", prediction="/smlθ/"),
        ]
        engine = ConsensusEngine(preds)
        result = engine.compute_consensus()
        assert result.n_models == 3
        assert result.consensus_pronunciation != ""

    def test_empty(self):
        engine = ConsensusEngine([])
        result = engine.compute_consensus()
        assert result.consensus_pronunciation == ""
        assert result.agreement_score == 0.0
        assert result.n_models == 0

    def test_report_generation(self):
        preds = [
            PronunciationPrediction(model_name="A", prediction="/smɪθ/"),
            PronunciationPrediction(model_name="B", prediction="/smɪθ/"),
        ]
        engine = ConsensusEngine(preds)
        report = engine.generate_report()
        assert "consensus" in report.lower() or "confidence" in report.lower()

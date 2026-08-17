"""Tests for evaluation metrics."""

from __future__ import annotations

from pronunciabench.data.models import PronunciationPrediction
from pronunciabench.evaluation.metrics import (
    Evaluator,
    ModelComparison,
    ModelMetrics,
    paired_bootstrap_comparison,
)


class TestModelMetrics:
    def test_to_dict(self):
        m = ModelMetrics(model_name="test", per=0.15, cer=0.20, exact_match_accuracy=0.7)
        d = m.to_dict()
        assert d["model"] == "test"
        assert d["phoneme_error_rate"] == 0.15

    def test_to_markdown_table(self):
        m = ModelMetrics(model_name="espeak", per=0.184)
        table = m.to_markdown_table()
        assert "espeak" in table
        assert "0.184" in table


class TestModelComparison:
    def test_model_a_wins(self):
        c = ModelComparison(
            model_a="A", model_b="B", per_a=0.1, per_b=0.2,
            delta_per=-0.1, relative_improvement=0.5,
            bootstrap_ci_lower=-0.15, bootstrap_ci_upper=-0.05,
        )
        assert c.verdict == "A wins"

    def test_model_b_wins(self):
        c = ModelComparison(
            model_a="A", model_b="B", per_a=0.2, per_b=0.1,
            delta_per=0.1, relative_improvement=-0.5,
            bootstrap_ci_lower=0.05, bootstrap_ci_upper=0.15,
        )
        assert c.verdict == "B wins"

    def test_tie(self):
        c = ModelComparison(
            model_a="A", model_b="B", per_a=0.15, per_b=0.15,
            delta_per=0.0, relative_improvement=0.0,
            bootstrap_ci_lower=-0.05, bootstrap_ci_upper=0.05,
        )
        assert c.verdict == "tie"

    def test_to_dict(self):
        c = ModelComparison(
            model_a="A", model_b="B", per_a=0.1, per_b=0.2,
            delta_per=-0.1, relative_improvement=0.5,
            bootstrap_ci_lower=-0.15, bootstrap_ci_upper=-0.05,
        )
        d = c.to_dict()
        assert d["per_a"] == 0.1
        assert d["model_a"] == "A"


class TestEvaluator:
    def test_compute_metrics(self):
        preds = [
            PronunciationPrediction(model_name="test", prediction="/smɪθ/", locale="en-US"),
            PronunciationPrediction(model_name="test", prediction="/ɡɑɹsi.a/", locale="es-ES"),
        ]
        refs = ["/smɪθ/", "/ɡɑɹsi.a/"]
        ev = Evaluator(preds, refs)
        metrics = ev.compute_metrics()
        assert metrics.model_name == "test"
        assert metrics.per == 0.0
        assert metrics.exact_match_accuracy == 1.0

    def test_with_errors(self):
        preds = [
            PronunciationPrediction(model_name="test", prediction="/smɪθ/", locale="en-US"),
            PronunciationPrediction(model_name="test", prediction="/XXXX/", locale="en-US"),
        ]
        refs = ["/smɪθ/", "/ɡɑɹsi.a/"]
        ev = Evaluator(preds, refs)
        metrics = ev.compute_metrics()
        assert metrics.per > 0.0

    def test_locale_filter(self):
        preds = [
            PronunciationPrediction(model_name="test", prediction="/smɪθ/", locale="en-US"),
            PronunciationPrediction(model_name="test", prediction="/XXX/", locale="fr-FR"),
        ]
        refs = ["/smɪth/", "/XXX/"]
        ev = Evaluator(preds, refs, locale_filter="en-US")
        metrics = ev.compute_metrics()
        assert metrics.n_samples == 1


class TestPairedBootstrap:
    def test_identical_models(self):
        per_a = [0.1] * 10
        per_b = [0.1] * 10
        mean_delta, ci_lo, ci_hi, wins, ties, losses = paired_bootstrap_comparison(per_a, per_b)
        assert abs(mean_delta) < 0.01
        assert ci_lo <= 0 <= ci_hi

    def test_better_model_a(self):
        per_a = [0.1] * 20
        per_b = [0.2] * 20
        mean_delta, ci_lo, ci_hi, wins, ties, losses = paired_bootstrap_comparison(per_a, per_b)
        assert mean_delta > 0  # B is worse
        assert ci_lo > 0  # CI does not cross zero

    def test_empty_lists(self):
        result = paired_bootstrap_comparison([], [])
        assert result == (0.0, 0.0, 0.0, 0, 0, 0)

    def test_different_lengths(self):
        result = paired_bootstrap_comparison([0.1], [0.1, 0.2])
        assert result == (0.0, 0.0, 0.0, 0, 0, 0)

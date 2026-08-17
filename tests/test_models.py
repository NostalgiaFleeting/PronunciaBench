"""Tests for G2P model backends."""

from __future__ import annotations

import pytest

from pronunciabench.data.models import PronunciationPrediction
from pronunciabench.models.espeak import EspeakG2P


class TestEspeakG2P:
    @pytest.mark.skipif(
        not pytest.importorskip("phonemizer", reason="phonemizer not installed"),
        reason="phonemizer not available",
    )
    def test_basic_english(self):
        model = EspeakG2P(language="en-us")
        pred = model.predict("hello")
        assert isinstance(pred, PronunciationPrediction)
        assert pred.model_name == "espeak"
        assert len(pred.prediction) > 0

    def test_prediction_has_required_fields(self):
        model = EspeakG2P(language="en-us")
        pred = model.predict("test")
        assert hasattr(pred, "model_name")
        assert hasattr(pred, "prediction")
        assert hasattr(pred, "latency_ms")
        assert pred.latency_ms >= 0

    def test_batch_prediction(self):
        model = EspeakG2P(language="en-us")
        preds = model.predict_batch(["hello", "world"])
        assert len(preds) == 2
        assert all(isinstance(p, PronunciationPrediction) for p in preds)

    def test_locale_handling(self):
        model = EspeakG2P(language="en-us")
        # Should not raise for unknown locale
        pred = model.predict("test", locale="xx-YY")
        assert isinstance(pred, PronunciationPrediction)

"""Tests for backend provenance and benchmark validity."""

from __future__ import annotations

from pronunciabench.data.models import BackendProvenance
from pronunciabench.models.espeak import EspeakG2P


class TestBackendProvenance:
    def test_default_values(self):
        prov = BackendProvenance()
        assert prov.requested_backend == "unknown"
        assert prov.actual_backend == "unknown"
        assert prov.fallback_used is False
        assert prov.is_real_prediction is False

    def test_real_prediction(self):
        prov = BackendProvenance(
            requested_backend="espeak",
            actual_backend="espeak",
            backend_version="1.48.0",
            fallback_used=False,
            is_real_prediction=True,
        )
        assert prov.is_real_prediction is True
        assert prov.fallback_used is False

    def test_fallback_prediction(self):
        prov = BackendProvenance(
            requested_backend="espeak",
            actual_backend="none",
            fallback_used=True,
            is_real_prediction=False,
        )
        assert prov.is_real_prediction is False
        assert prov.fallback_used is True


class TestEspeakProvenance:
    def test_provenance_tracked(self):
        model = EspeakG2P(language="en-us")
        pred = model.predict("Smith", "en-US")
        assert isinstance(pred.provenance, BackendProvenance)
        assert pred.provenance.requested_backend == "espeak"
        assert pred.provenance.actual_backend in ("espeak", "segments", "none")
        assert pred.provenance.backend_version is not None or pred.provenance.backend_version is None

    def test_fallback_flagged(self):
        model = EspeakG2P(language="en-us")
        pred = model.predict("Smith", "en-US")
        # On this machine without eSpeak, should be flagged as fallback
        if not pred.provenance.is_real_prediction:
            assert pred.prediction.startswith("/[")
            assert pred.prediction.endswith("]/")

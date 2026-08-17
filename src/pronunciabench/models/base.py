"""Base G2P model with shared utilities."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from pronunciabench.data.models import G2PModel, PronunciationPrediction


class BaseG2PModel(ABC, G2PModel):
    """Abstract base class for all G2P backends."""

    model_name: str = "base_g2p"

    @abstractmethod
    def _predict_impl(self, text: str, locale: str | None) -> str:
        """Subclasses implement actual pronunciation prediction."""
        ...

    def predict(self, text: str, locale: str | None = None) -> PronunciationPrediction:
        """Run prediction and measure latency."""
        start = time.perf_counter()
        prediction = self._predict_impl(text, locale)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return PronunciationPrediction(
            model_name=self.model_name,
            prediction=prediction,
            locale=locale,
            latency_ms=round(elapsed_ms, 2),
        )

    def predict_batch(
        self, texts: list[str], locale: str | None = None
    ) -> list[PronunciationPrediction]:
        """Run batch prediction."""
        return [self.predict(text, locale) for text in texts]
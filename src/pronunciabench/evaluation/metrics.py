"""Evaluation metrics for G2P systems."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pronunciabench.data.models import PronunciationPrediction
from pronunciabench.data.normalize import (
    character_error_rate,
    exact_match_accuracy,
    normalize_ipa,
    phoneme_error_rate,
)


@dataclass
class ModelMetrics:
    """Aggregate metrics for a single model."""

    model_name: str
    per: float = 0.0
    cer: float = 0.0
    exact_match_accuracy: float = 0.0
    coverage: float = 1.0
    mean_latency_ms: float = 0.0
    abstention_rate: float = 0.0
    n_samples: int = 0
    n_verified: int = 0
    breakdown: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "phoneme_error_rate": round(self.per, 4),
            "character_error_rate": round(self.cer, 4),
            "exact_match_accuracy": round(self.exact_match_accuracy, 4),
            "coverage": round(self.coverage, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "abstention_rate": round(self.abstention_rate, 4),
            "n_samples": self.n_samples,
            "n_verified": self.n_verified,
            "breakdown": self.breakdown,
        }

    def to_markdown_table(self) -> str:
        per_str = f"{self.per:.4f}" if self.per else "N/A"
        em_str = f"{self.exact_match_accuracy:.4f}" if self.exact_match_accuracy else "N/A"
        cov_str = f"{self.coverage:.4f}" if self.coverage else "N/A"
        lat_str = f"{self.mean_latency_ms:.1f} ms" if self.mean_latency_ms else "N/A"
        return f"| **{self.model_name}** | {per_str} | {em_str} | {cov_str} | {lat_str} |"


@dataclass
class ModelComparison:
    """Paired comparison between two models."""

    model_a: str
    model_b: str
    per_a: float
    per_b: float
    delta_per: float
    relative_improvement: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    n_trials: int = 1000
    wins: int = 0
    ties: int = 0
    losses: int = 0

    @property
    def verdict(self) -> str:
        if self.delta_per < -0.01:
            return f"{self.model_a} wins"
        elif self.delta_per > 0.01:
            return f"{self.model_b} wins"
        return "tie"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "per_a": round(self.per_a, 4),
            "per_b": round(self.per_b, 4),
            "delta_per": round(self.delta_per, 4),
            "relative_improvement_pct": round(self.relative_improvement * 100, 1),
            "bootstrap_95_ci": [
                round(self.bootstrap_ci_lower, 4),
                round(self.bootstrap_ci_upper, 4),
            ],
            "verdict": self.verdict,
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
        }


class Evaluator:
    """Evaluate G2P predictions against reference pronunciations."""

    def __init__(
        self,
        predictions: list[PronunciationPrediction],
        references: list[str],
        locale_filter: str | None = None,
    ):
        self.predictions = predictions
        self.references = references
        self.locale_filter = locale_filter

    def compute_metrics(self) -> ModelMetrics:
        """Compute aggregate metrics for all predictions."""
        filtered: list[tuple[PronunciationPrediction, str]] = []
        for pred, ref in zip(self.predictions, self.references):
            if self.locale_filter and pred.locale != self.locale_filter:
                continue
            filtered.append((pred, ref))

        if not filtered:
            return ModelMetrics(model_name="filtered")

        pers, cers, latencies = [], [], []
        exact_matches = 0

        for pred, ref in filtered:
            per = phoneme_error_rate(ref, pred.prediction)
            cer = character_error_rate(ref, pred.prediction)
            pers.append(per)
            cers.append(cer)
            latencies.append(pred.latency_ms)
            if normalize_ipa(ref) == normalize_ipa(pred.prediction):
                exact_matches += 1

        n = len(filtered)
        return ModelMetrics(
            model_name=self.predictions[0].model_name if self.predictions else "unknown",
            per=np.mean(pers) if pers else 0.0,
            cer=np.mean(cers) if cers else 0.0,
            exact_match_accuracy=exact_matches / n if n else 0.0,
            coverage=1.0,
            mean_latency_ms=np.mean(latencies) if latencies else 0.0,
            abstention_rate=0.0,
            n_samples=n,
        )

    def compute_breakdown_by_locale(self) -> dict[str, ModelMetrics]:
        """Break down metrics by locale."""
        by_locale: dict[str, list[tuple[PronunciationPrediction, str]]] = defaultdict(list)
        for pred, ref in zip(self.predictions, self.references):
            loc = pred.locale or "unknown"
            by_locale[loc].append((pred, ref))

        result = {}
        for loc, items in by_locale.items():
            sub_eval = Evaluator(
                [p for p, _ in items],
                [r for _, r in items],
            )
            result[loc] = sub_eval.compute_metrics()
        return result


def paired_bootstrap_comparison(
    per_a: list[float],
    per_b: list[float],
    n_trials: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float, int, int, int]:
    """Paired bootstrap test for model comparison.

    Returns (mean_delta, ci_lower, ci_upper, wins, ties, losses).
    Positive delta means B is worse than A.
    """
    if len(per_a) != len(per_b) or len(per_a) == 0:
        return 0.0, 0.0, 0.0, 0, 0, 0

    rng = np.random.default_rng(seed)
    n = len(per_a)
    deltas = [b - a for a, b in zip(per_a, per_b)]

    wins, ties, losses = 0, 0, 0
    boot_deltas: list[float] = []

    for _ in range(n_trials):
        sample_idx = rng.integers(0, n, size=n)
        boot_delta = np.mean([deltas[i] for i in sample_idx])
        boot_deltas.append(boot_delta)

        if boot_delta < -0.001:
            wins += 1
        elif boot_delta > 0.001:
            losses += 1
        else:
            ties += 1

    boot_deltas.sort()
    ci_lower = boot_deltas[int(0.025 * n_trials)]
    ci_upper = boot_deltas[int(0.975 * n_trials)]
    mean_delta = np.mean(boot_deltas)

    return mean_delta, ci_lower, ci_upper, wins, ties, losses
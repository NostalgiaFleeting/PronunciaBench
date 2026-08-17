"""Multi-model consensus and ensemble layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pronunciabench.data.models import PronunciationPrediction
from pronunciabench.data.normalize import (
    extract_phonemes,
    normalize_ipa,
    phoneme_edit_distance,
)


@dataclass
class ConsensusResult:
    """Result of multi-model consensus computation."""

    consensus_pronunciation: str
    agreement_score: float
    model_distances: dict[str, dict[str, float]]
    outlier_models: list[str]
    confidence: float
    n_models: int


class ConsensusEngine:
    """Aggregate predictions from multiple G2P models."""

    def __init__(
        self,
        predictions: list[PronunciationPrediction],
    ):
        self.predictions = predictions
        self.norm_predictions: list[str] = [
            normalize_ipa(p.prediction) for p in predictions
        ]

    def compute_consensus(self) -> ConsensusResult:
        """Compute consensus pronunciation from multiple models."""
        if not self.predictions:
            return ConsensusResult(
                consensus_pronunciation="",
                agreement_score=0.0,
                model_distances={},
                outlier_models=[],
                confidence=0.0,
                n_models=0,
            )

        if len(self.predictions) == 1:
            return ConsensusResult(
                consensus_pronunciation=self.norm_predictions[0],
                agreement_score=1.0,
                model_distances={},
                outlier_models=[],
                confidence=self.predictions[0].confidence or 0.7,
                n_models=1,
            )

        # Compute pairwise phoneme distances
        n = len(self.norm_predictions)
        distances: dict[str, dict[str, float]] = {
            p.model_name: {} for p in self.predictions
        }
        all_pairwise_dists: list[float] = []

        for i in range(n):
            for j in range(i + 1, n):
                phones_i = extract_phonemes(self.norm_predictions[i])
                phones_j = extract_phonemes(self.norm_predictions[j])
                _, _, _, dist = phoneme_edit_distance(phones_i, phones_j)
                max_len = max(len(phones_i), len(phones_j), 1)
                norm_dist = dist / max_len
                all_pairwise_dists.append(norm_dist)
                distances[self.predictions[i].model_name][
                    self.predictions[j].model_name
                ] = round(norm_dist, 4)
                distances[self.predictions[j].model_name][
                    self.predictions[i].model_name
                ] = round(norm_dist, 4)

        # Agreement score: 1 - mean pairwise distance
        agreement = max(0.0, 1.0 - np.mean(all_pairwise_dists)) if all_pairwise_dists else 1.0

        # Find consensus candidate (median/central prediction)
        consensus = self._find_consensus_candidate()

        # Identify outlier models
        outlier_models = self._find_outliers(distances, agreement)

        # Confidence based on agreement and model confidences
        model_confidences = [
            p.confidence or 0.5 for p in self.predictions
        ]
        base_confidence = np.mean(model_confidences)
        confidence = agreement * 0.6 + base_confidence * 0.4

        return ConsensusResult(
            consensus_pronunciation=consensus,
            agreement_score=round(agreement, 4),
            model_distances=distances,
            outlier_models=outlier_models,
            confidence=round(confidence, 4),
            n_models=n,
        )

    def _find_consensus_candidate(self) -> str:
        """Select the most central prediction as consensus."""
        if len(self.norm_predictions) == 1:
            return self.norm_predictions[0]

        # Find prediction with minimum average distance to others
        n = len(self.norm_predictions)
        dists_to_all: list[list[float]] = []

        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0.0)
                    continue
                phones_i = extract_phonemes(self.norm_predictions[i])
                phones_j = extract_phonemes(self.norm_predictions[j])
                _, _, _, dist = phoneme_edit_distance(phones_i, phones_j)
                max_len = max(len(phones_i), len(phones_j), 1)
                row.append(dist / max_len)
            dists_to_all.append(row)

        avg_dists = [np.mean(row) for row in dists_to_all]
        best_idx = int(np.argmin(avg_dists))
        return self.norm_predictions[best_idx]

    def _find_outliers(
        self,
        distances: dict[str, dict[str, float]],
        agreement: float,
    ) -> list[str]:
        """Identify models with unusually high distance from consensus."""
        if agreement > 0.9:
            return []

        outliers = []
        threshold = 1.0 - agreement
        for model in self.norm_predictions:
            # Use model name from predictions
            pass

        for pred in self.predictions:
            name = pred.model_name
            if name not in distances:
                continue
            dists = distances[name]
            if not dists:
                continue
            avg_dist = np.mean(list(dists.values()))
            if avg_dist > threshold * 0.5:
                outliers.append(name)
        return outliers

    def generate_report(self) -> str:
        """Generate human-readable consensus report."""
        result = self.compute_consensus()
        lines = [f"Model disagreement: {'LOW' if result.agreement_score > 0.8 else 'MEDIUM' if result.agreement_score > 0.5 else 'HIGH'}"]
        lines.append(f"Consensus confidence: {result.confidence:.2f}")
        lines.append("")

        for model_a, dists in result.model_distances.items():
            for model_b, dist in sorted(dists.items()):
                if model_a < model_b:
                    lines.append(f"{model_a} ↔ {model_b} distance: {dist:.2f}")

        if result.outlier_models:
            lines.append("")
            lines.append(f"Outlier models: {', '.join(result.outlier_models)}")

        return "\n".join(lines)
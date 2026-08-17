

"""Confidence estimation and reliability scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pronunciabench.data.models import AbstentionDecision, ReliabilityResult
from pronunciabench.data.normalize import extract_phonemes, normalize_ipa


@dataclass
class ReliabilityScorer:
    """Estimates confidence and drives abstention decisions."""

    abstention_threshold: float = 0.5
    calibration_scores: list[float] | None = None
    calibration_method: str = "conformal"

    def score(
        self,
        predictions: list[str],
        locale: str | None,
        reference: str | None = None,
        model_confidences: list[float] | None = None,
    ) -> ReliabilityResult:
        components: dict[str, float] = {}
        norm_preds = [normalize_ipa(p) for p in predictions]
        agreement = self._compute_agreement(norm_preds)
        components["model_agreement"] = agreement
        seq_conf = self._compute_sequence_confidence(predictions, model_confidences)
        components["sequence_confidence"] = seq_conf
        locale_support = self._compute_locale_support(locale)
        components["locale_support"] = locale_support
        stability = self._compute_stability(norm_preds)
        components["prediction_stability"] = stability
        oov_signal = self._compute_oov_signal(predictions)
        components["oov_signal"] = oov_signal

        weights = {"model_agreement": 0.30, "sequence_confidence": 0.25,
                   "locale_support": 0.15, "prediction_stability": 0.20, "oov_signal": 0.10}
        total = sum(weights.values())
        confidence = sum(components[k] * weights[k] / total for k in weights)

        if self.calibration_scores is not None and reference is not None:
            confidence = self._apply_conformal_calibration(confidence, reference, predictions)

        decision = AbstentionDecision.ABSTAIN if confidence < self.abstention_threshold else AbstentionDecision.ACCEPT
        reason = self._generate_reason(confidence, decision, components, predictions)
        return ReliabilityResult(confidence=round(confidence, 4), components=components,
                                 decision=decision, reason=reason)

    def _compute_agreement(self, norm_preds: list[str]) -> float:
        if len(norm_preds) <= 1:
            return 1.0
        if len(set(norm_preds)) == 1:
            return 1.0
        from pronunciabench.data.normalize import phoneme_edit_distance
        phones = [extract_phonemes(p) for p in norm_preds]
        distances = []
        for i in range(len(phones)):
            for j in range(i + 1, len(phones)):
                _, _, _, dist = phoneme_edit_distance(phones[i], phones[j])
                max_len = max(len(phones[i]), len(phones[j]), 1)
                distances.append(dist / max_len)
        return max(0.0, 1.0 - np.mean(distances)) if distances else 1.0

    def _compute_sequence_confidence(self, predictions: list[str],
                                     model_confidences: list[float] | None) -> float:
        if model_confidences and all(c is not None for c in model_confidences):
            return float(np.mean(model_confidences))
        if len(predictions) <= 1:
            return 0.7
        lengths = [len(p) for p in predictions]
        return 0.8 if len(set(lengths)) == 1 else 0.5

    def _compute_locale_support(self, locale: str | None) -> float:
        if not locale:
            return 0.5
        well_supported = {"en-US", "en-GB", "de-DE", "fr-FR", "es-ES", "pt-BR",
                          "ru-RU", "ja-JP", "ko-KR", "zh-CN", "zh-TW", "ar-SA",
                          "hi-IN", "th-TH", "tr-TR", "nl-NL", "pl-PL", "sv-SE"}
        if locale in well_supported:
            return 1.0
        parts = locale.split("-")
        if len(parts) == 2 and len(parts[0]) == 2 and len(parts[1]) == 2:
            return 0.7
        return 0.3

    def _compute_stability(self, norm_preds: list[str]) -> float:
        if len(norm_preds) <= 1:
            return 1.0
        matches = sum(1 for i in range(len(norm_preds))
                      for j in range(i + 1, len(norm_preds)) if norm_preds[i] == norm_preds[j])
        total = len(norm_preds) * (len(norm_preds) - 1) // 2
        return matches / total if total > 0 else 1.0

    def _compute_oov_signal(self, predictions: list[str]) -> float:
        import re
        fallback = sum(1 for p in predictions if re.search(r"/\[.*\]/", p))
        return 1.0 - (fallback / len(predictions)) if predictions else 1.0

    def _apply_conformal_calibration(self, raw: float, reference: str,
                                     predictions: list[str]) -> float:
        if not self.calibration_scores:
            return raw
        ref_phones = extract_phonemes(normalize_ipa(reference))
        pred_phones = extract_phonemes(normalize_ipa(predictions[0]))
        from pronunciabench.data.normalize import phoneme_edit_distance
        _, _, _, dist = phoneme_edit_distance(ref_phones, pred_phones)
        perc = dist / max(len(ref_phones), 1)
        return max(0.0, min(1.0, 1.0 - perc))

    def _generate_reason(self, confidence: float, decision: AbstentionDecision,
                         components: dict[str, float], predictions: list[str]) -> str:
        parts = []
        agreement = components.get("model_agreement", 0)
        label = "disagreement" if agreement < 0.5 else "Moderate agreement" if agreement < 0.8 else "Strong agreement"
        parts.append(f"{label} (agreement={agreement:.2f})")
        if components.get("oov_signal", 1.0) < 0.5:
            parts.append("Possible OOV detected")
        parts.append("below calibrated risk threshold" if decision == AbstentionDecision.ABSTAIN
                     else "within acceptable reliability region")
        return "; ".join(parts)


class ConformalAbstainer:
    """Conformal prediction-based abstention system."""

    def __init__(self, alpha: float = 0.05, calibration_set: list[float] | None = None):
        self.alpha = alpha
        self.calibration_set = calibration_set or []
        self._q_hat: float | None = None

    def calibrate(self, nonconformity_scores: list[float]) -> None:
        """Compute conformal quantile from calibration set."""
        if not nonconformity_scores:
            self._q_hat = 1.0
            return
        scores = sorted(nonconformity_scores)
        n = len(scores)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        q_idx = min(int(np.floor(q_level * n)), n - 1)
        self._q_hat = scores[q_idx]

    def predict(self, nonconformity_score: float) -> AbstentionDecision:
        """Abstain if nonconformity exceeds calibrated threshold."""
        if self._q_hat is None:
            return AbstentionDecision.ACCEPT
        return (
            AbstentionDecision.ABSTAIN
            if nonconformity_score > self._q_hat
            else AbstentionDecision.ACCEPT
        )

    def evaluate_coverage(
        self, scores: list[float], true_labels: list[bool]
    ) -> dict[str, float]:
        """Evaluate selective prediction performance."""
        decisions = [self.predict(s) for s in scores]
        n = len(scores)
        accepted = [i for i, d in enumerate(decisions) if d == AbstentionDecision.ACCEPT]
        accepted_labels = [true_labels[i] for i in accepted]

        coverage = len(accepted) / n if n else 0.0
        abstention_rate = 1.0 - coverage
        selective_accuracy = (
            sum(accepted_labels) / len(accepted_labels)
            if accepted_labels else 0.0
        )
        overall_accuracy = sum(true_labels) / n if n else 0.0

        return {
            "coverage": round(coverage, 4),
            "abstention_rate": round(abstention_rate, 4),
            "selective_accuracy": round(selective_accuracy, 4),
            "overall_accuracy": round(overall_accuracy, 4),
            "n_accepted": len(accepted),
            "n_total": n,
        }
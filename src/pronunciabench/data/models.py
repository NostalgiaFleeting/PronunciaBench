"""Core data models for pronunciation examples and predictions."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator


class VerificationStatus(str, Enum):
    """Whether ground-truth pronunciation has been human-verified."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    EXPLORATORY = "exploratory"


class PhonemeSystem(str, Enum):
    """Phoneme representation system used in pronunciation labels."""

    IPA = "ipa"
    ARPABET = "arpabet"
    UNKNOWN = "unknown"


class AbstentionDecision(str, Enum):
    ACCEPT = "accept"
    ABSTAIN = "abstain"


class PronunciationExample(BaseModel):
    """A single pronunciation example with provenance."""

    text: str
    pronunciation: str
    locale: str | None = None
    source: str
    split: str | None = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    phoneme_system: PhonemeSystem = PhonemeSystem.UNKNOWN
    source_url: str | None = None
    source_license: str | None = None
    source_version: str | None = None
    notes: str = ""

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()

    @field_validator("pronunciation")
    @classmethod
    def strip_pronunciation(cls, v: str) -> str:
        return v.strip()


class BackendProvenance(BaseModel):
    """Tracks which backend actually produced a prediction."""

    requested_backend: str = "unknown"
    actual_backend: str = "unknown"
    backend_version: str | None = None
    fallback_used: bool = False
    is_real_prediction: bool = False


class PronunciationPrediction(BaseModel):
    """Output from a single G2P backend."""

    model_name: str
    model_version: str = "1.0.0"
    locale: str | None = None
    prediction: str
    confidence: float | None = None
    latency_ms: float = 0.0
    provenance: BackendProvenance = Field(default_factory=BackendProvenance)

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float | None) -> float | None:
        if v is None:
            return None
        return max(0.0, min(1.0, v))


class ReliabilityResult(BaseModel):
    """Confidence estimation result with explainable components."""

    confidence: float
    components: dict[str, float]
    decision: AbstentionDecision
    reason: str
    is_calibrated_probability: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


@runtime_checkable
class G2PModel(Protocol):
    """Interface that all G2P backends must implement."""

    model_name: str

    def predict(
        self,
        text: str,
        locale: str | None = None,
    ) -> PronunciationPrediction:
        ...

    def predict_batch(
        self,
        texts: list[str],
        locale: str | None = None,
    ) -> list[PronunciationPrediction]:
        ...

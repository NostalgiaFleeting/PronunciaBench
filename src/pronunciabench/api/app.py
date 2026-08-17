"""FastAPI application for PronunciaBench."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pronunciabench.data.models import PronunciationPrediction
from pronunciabench.ensemble.consensus import ConsensusEngine
from pronunciabench.models.espeak import EspeakG2P
from pronunciabench.reliability.scorer import ReliabilityScorer


class PronounceRequest(BaseModel):
    text: str
    locale: str | None = None
    models: list[str] = Field(default_factory=lambda: ["espeak"])


class PronounceResponse(BaseModel):
    text: str
    locale: str | None
    predictions: list[dict]
    consensus: str
    agreement_score: float
    confidence: float
    decision: str
    reason: str
    latency_ms: float


app = FastAPI(title="PronunciaBench API", version="0.1.0")

_backends: dict[str, object] = {}


def _get_backend(name: str, locale: str | None) -> object:
    if name == "espeak" or name == "espeak-ng":
        if "espeak" not in _backends:
            _backends["espeak"] = EspeakG2P(language=locale)
        return _backends["espeak"]
    raise HTTPException(status_code=400, detail=f"Unknown model: {name}")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "pronunciabench"}


@app.get("/v1/models")
def list_models() -> dict:
    return {"models": ["espeak"], "version": "0.1.0"}


@app.post("/v1/pronounce", response_model=PronounceResponse)
def pronounce(req: PronounceRequest) -> PronounceResponse:
    backends = {}
    for m in req.models:
        backends[m] = _get_backend(m, req.locale)

    predictions = []
    for backend in backends.values():
        if hasattr(backend, "predict"):
            predictions.append(backend.predict(req.text, req.locale))

    if not predictions:
        raise HTTPException(status_code=500, detail="No predictions generated")

    consensus_engine = ConsensusEngine(predictions)
    consensus = consensus_engine.compute_consensus()

    scorer = ReliabilityScorer()
    reliability = scorer.score(
        [p.prediction for p in predictions], req.locale,
        model_confidences=[p.confidence for p in predictions],
    )

    total_latency = sum(p.latency_ms for p in predictions)
    return PronounceResponse(
        text=req.text, locale=req.locale,
        predictions=[{"model": p.model_name, "ipa": p.prediction, "latency_ms": p.latency_ms}
                     for p in predictions],
        consensus=consensus.consensus_pronunciation,
        agreement_score=consensus.agreement_score,
        confidence=reliability.confidence,
        decision=reliability.decision.value,
        reason=reliability.reason,
        latency_ms=round(total_latency, 2),
    )


@app.post("/v1/evaluate")
def evaluate(predictions: list[dict], references: list[str]) -> JSONResponse:
    from pronunciabench.evaluation.metrics import Evaluator
    preds = [PronunciationPrediction(model_name=p["model"], prediction=p["ipa"]) for p in predictions]
    evaluator = Evaluator(preds, references)
    metrics = evaluator.compute_metrics()
    return JSONResponse(content=metrics.to_dict())

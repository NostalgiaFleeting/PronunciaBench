"""Command-line interface for PronunciaBench."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from pronunciabench.data.models import PronunciationPrediction
from pronunciabench.models.espeak import EspeakG2P
from pronunciabench.ensemble.consensus import ConsensusEngine
from pronunciabench.reliability.scorer import ReliabilityScorer
from pronunciabench.data.normalize import phoneme_error_rate, normalize_ipa


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """PronunciaBench — Multilingual G2P reliability framework."""
    pass


@main.command()
@click.argument("text")
@click.option("--locale", "-l", default=None, help="Locale code (e.g., vi-VN, en-US)")
@click.option("--models", "-m", default="espeak", help="Comma-separated model names")
@click.option("--output", "-o", default=None, help="Output file (JSON)")
def pronounce(text: str, locale: str | None, models: str, output: str | None) -> None:
    """Pronounce a name using configured G2P backends."""
    model_list = [m.strip() for m in models.split(",")]
    backends: dict[str, object] = {}
    if "espeak" in model_list:
        backends["espeak"] = EspeakG2P(language=locale)
    if not backends:
        backends["espeak"] = EspeakG2P(language=locale)

    predictions = []
    for name, backend in backends.items():
        if hasattr(backend, "predict"):
            predictions.append(backend.predict(text, locale))

    if not predictions:
        click.echo("No predictions generated.", err=True)
        sys.exit(1)

    consensus_engine = ConsensusEngine(predictions)
    consensus = consensus_engine.compute_consensus()
    scorer = ReliabilityScorer()
    reliability = scorer.score(
        [p.prediction for p in predictions], locale,
        model_confidences=[p.confidence for p in predictions],
    )

    result = {
        "text": text, "locale": locale,
        "predictions": [{"model": p.model_name, "ipa": p.prediction, "latency_ms": p.latency_ms}
                        for p in predictions],
        "consensus": consensus.consensus_pronunciation,
        "agreement_score": consensus.agreement_score,
        "confidence": reliability.confidence,
        "decision": reliability.decision.value,
        "reason": reliability.reason,
        "components": reliability.components,
    }
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        click.echo(f"Result saved to {output}")


@main.command()
@click.option("--dataset", "-d", required=True, help="Path to JSONL dataset")
@click.option("--models", "-m", default="espeak", help="Comma-separated model names")
@click.option("--locale", "-l", default=None, help="Filter by locale")
@click.option("--output", "-o", default=None, help="Output JSON report")
@click.option("--require-real-backends", is_flag=True, default=False,
              help="Fail if any backend fell back to placeholder predictions")
def benchmark(dataset: str, models: str, locale: str | None, output: str | None,
              require_real_backends: bool) -> None:
    """Run benchmark evaluation on a dataset."""
    data_path = Path(dataset)
    if not data_path.exists():
        click.echo(f"Dataset not found: {dataset}", err=True)
        sys.exit(1)

    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if locale:
        examples = [e for e in examples if e.get("locale") == locale]
    if not examples:
        click.echo("No matching examples found.", err=True)
        sys.exit(1)

    import numpy as np
    backends: dict[str, object] = {"espeak": EspeakG2P()}

    reports = {}
    all_has_fallback = False
    for model_name, backend in backends.items():
        pers, predictions = [], []
        for ex in examples:
            text = ex.get("text", ex.get("name", ""))
            ref = ex.get("pronunciation", ex.get("reference_ipa", ""))
            loc = ex.get("locale")
            if hasattr(backend, "predict"):
                pred = backend.predict(text, loc)
                predictions.append(pred)
                per = phoneme_error_rate(ref, pred.prediction) if ref else 0.0
                pers.append(per)
                if not pred.provenance.is_real_prediction:
                    all_has_fallback = True
        if pers:
            reports[model_name] = {
                "model": model_name, "phoneme_error_rate": round(float(np.mean(pers)), 4),
                "n_samples": len(pers),
                "mean_latency_ms": round(float(np.mean([p.latency_ms for p in predictions])), 2),
                "any_fallback": any(not p.provenance.is_real_prediction for p in predictions),
            }

    # Consensus benchmark
    all_predictions, all_references = [], []
    for ex in examples:
        text = ex.get("text", ex.get("name", ""))
        ref = ex.get("pronunciation", ex.get("reference_ipa", ""))
        loc = ex.get("locale")
        preds = [b.predict(text, loc) for b in backends.values() if hasattr(b, "predict")]
        if preds:
            all_predictions.extend(preds)
            all_references.append(ref)

    benchmark_valid = not all_has_fallback
    if all_predictions:
        ce = ConsensusEngine(all_predictions)
        cs = ce.compute_consensus()
        per = phoneme_error_rate(all_references[0], cs.consensus_pronunciation) if all_references else 0.0
        reports["consensus"] = {"model": "consensus", "phoneme_error_rate": round(per, 4),
                                "agreement_score": cs.agreement_score, "n_samples": len(all_predictions)}

    scorer = ReliabilityScorer()
    rel = scorer.score([p.prediction for p in all_predictions], locale)
    reports["reliability"] = {"confidence": rel.confidence, "decision": rel.decision.value,
                              "components": rel.components}

    result = {
        "dataset": str(data_path), "n_examples": len(examples),
        "locale_filter": locale, "benchmark_valid": benchmark_valid,
        "fallback_detected": all_has_fallback, "reports": reports,
    }
    if require_real_backends and all_has_fallback:
        click.echo("ERROR: Benchmark contains placeholder predictions. "
                   "Set --require-real-backends=false to allow fallback results.", err=True)
        sys.exit(2)
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        click.echo(f"Report saved to {output}")
    else:
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))


@main.command()
@click.option("--model-a", "-a", required=True, help="First model name")
@click.option("--model-b", "-b", required=True, help="Second model name")
@click.option("--dataset", "-d", required=True, help="Path to JSONL dataset")
def compare(model_a: str, model_b: str, dataset: str) -> None:
    """Compare two models using paired bootstrap test."""
    import numpy as np
    from pronunciabench.evaluation.metrics import paired_bootstrap_comparison

    data_path = Path(dataset)
    if not data_path.exists():
        click.echo(f"Dataset not found: {dataset}", err=True)
        sys.exit(1)

    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    backends = {"espeak": EspeakG2P()}
    per_a, per_b = [], []
    for ex in examples:
        text = ex.get("text", ex.get("name", ""))
        ref = ex.get("pronunciation", ex.get("reference_ipa", ""))
        if not ref:
            continue
        for name, backend in backends.items():
            if hasattr(backend, "predict"):
                pred = backend.predict(text, ex.get("locale"))
                per = phoneme_error_rate(ref, pred.prediction)
                if name == model_a or model_a == "espeak":
                    per_a.append(per)
                if name == model_b or model_b == "espeak":
                    per_b.append(per)

    if len(per_a) < 2 or len(per_b) < 2:
        click.echo("Insufficient data for comparison.", err=True)
        sys.exit(1)

    mean_delta, ci_lower, ci_upper, wins, ties, losses = paired_bootstrap_comparison(per_a, per_b)
    per_a_mean, per_b_mean = np.mean(per_a), np.mean(per_b)
    rel_imp = (per_a_mean - per_b_mean) / per_a_mean if per_a_mean > 0 else 0.0
    verdict = "model_a wins" if mean_delta < -0.01 else "model_b wins" if mean_delta > 0.01 else "tie"

    result = {"model_a": model_a, "model_b": model_b,
              "per_a": round(per_a_mean, 4), "per_b": round(per_b_mean, 4),
              "delta_per": round(mean_delta, 4),
              "relative_improvement_pct": round(rel_imp * 100, 1),
              "bootstrap_95_ci": [round(ci_lower, 4), round(ci_upper, 4)],
              "wins": wins, "ties": ties, "losses": losses, "verdict": verdict}
    click.echo(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
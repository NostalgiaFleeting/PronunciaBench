"""Gradio dashboard for PronunciaBench."""

from __future__ import annotations

import json

import gradio as gr
import numpy as np

from pronunciabench.data.models import PronunciationPrediction
from pronunciabench.data.normalize import phoneme_error_rate
from pronunciabench.ensemble.consensus import ConsensusEngine
from pronunciabench.models.espeak import EspeakG2P
from pronunciabench.reliability.scorer import ReliabilityScorer


def pronounce_name(text: str, locale: str, risk_tolerance: float) -> dict:
    """Run pronunciation pipeline and return results."""
    if not text.strip():
        return {"error": "Please enter a name"}

    model = EspeakG2P(language=locale or None)
    pred = model.predict(text, locale or None)

    consensus_engine = ConsensusEngine([pred])
    consensus = consensus_engine.compute_consensus()

    scorer = ReliabilityScorer(abstention_threshold=risk_tolerance)
    reliability = scorer.score([pred.prediction], locale)

    return {
        "text": text,
        "locale": locale or "unspecified",
        "ipa": pred.prediction,
        "consensus": consensus.consensus_pronunciation,
        "confidence": round(reliability.confidence, 4),
        "decision": reliability.decision.value,
        "reason": reliability.reason,
        "latency_ms": round(pred.latency_ms, 2),
        "components": reliability.components,
    }


def benchmark_results(dataset_path: str) -> str:
    """Run benchmark and return formatted report."""
    try:
        from pronunciabench.data import load_jsonl, compute_stats, generate_report
        from pronunciabench.evaluation.metrics import Evaluator
        examples = load_jsonl(dataset_path)
        stats = compute_stats(examples)
        report = generate_report(stats)

        if examples:
            model = EspeakG2P()
            preds = []
            refs = []
            for ex in examples:
                p = model.predict(ex.text, ex.locale)
                preds.append(p)
                refs.append(ex.pronunciation)

            evaluator = Evaluator(preds, refs)
            metrics = evaluator.compute_metrics()
            report += f"\n\n--- Benchmark Results ---\n"
            report += metrics.to_markdown_table()

        return report
    except Exception as e:
        return f"Error: {str(e)}"


def create_dashboard() -> gr.Blocks:
    """Create the Gradio dashboard."""
    with gr.Blocks(title="PronunciaBench", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# PronunciaBench")
        gr.Markdown("Multilingual G2P reliability lab — measure accuracy *and* uncertainty.")

        with gr.Tab("Pronunciation Lab"):
            with gr.Row():
                with gr.Column():
                    name_input = gr.Textbox(label="Name", placeholder="e.g., Nguyen, Schwarzenegger")
                    locale_input = gr.Textbox(label="Locale", placeholder="e.g., vi-VN, en-US")
                    risk_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                        label="Risk Tolerance (abstention threshold)",
                    )
                    pronounce_btn = gr.Button("Pronounce", variant="primary")
                with gr.Column():
                    ipa_output = gr.Textbox(label="IPA Prediction", interactive=False)
                    consensus_output = gr.Textbox(label="Consensus", interactive=False)
                    confidence_output = gr.Number(label="Confidence", interactive=False)
                    decision_output = gr.Textbox(label="Decision", interactive=False)
                    reason_output = gr.Textbox(label="Reasoning", interactive=False)
                    latency_output = gr.Number(label="Latency (ms)", interactive=False)

            pronounce_btn.click(
                pronounce_name,
                inputs=[name_input, locale_input, risk_slider],
                outputs=[ipa_output, consensus_output, confidence_output,
                         decision_output, reason_output, latency_output],
                fn=lambda text, locale, risk: pronounce_name(text, locale, risk),
                _js="""(text, locale, risk) => {
                    const r = pronounce_name(text, locale, risk);
                    return [r.ipa, r.consensus, r.confidence, r.decision, r.reason, r.latency_ms];
                }""",
            )

        with gr.Tab("Benchmark"):
            dataset_input = gr.Textbox(label="Dataset path (JSONL)")
            benchmark_btn = gr.Button("Run Benchmark")
            benchmark_output = gr.Textbox(label="Results", lines=20, interactive=False)
            benchmark_btn.click(benchmark_results, inputs=[dataset_input], outputs=[benchmark_output])

        with gr.Tab("Reliability"):
            gr.Markdown("### Risk-Coverage Analysis")
            gr.Markdown("Use the CLI to generate risk-coverage curves:")
            gr.Code(
                language="bash",
                code='pronunciabench benchmark --dataset data/samples/test.jsonl --output reports/benchmark.json',
            )

    return demo


if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch(server_name="0.0.0.0", server_port=7860)
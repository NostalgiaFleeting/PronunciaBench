# PronunciaBench

**A G2P reliability framework that measures when a pronunciation model should abstain.**

Pronunciation systems often return confident predictions even for names, languages, or spellings they handle poorly. PronunciaBench addresses this by combining multiple G2P backends with calibrated confidence estimation and conformal abstention.

## Architecture

```
Input Name ──► Normalization ──► Multilingual G2P Backends
                                      ├── Espeak baseline (rule-based)
                                      ├── ByT5 base checkpoint (for fine-tuning)
                                      └── Fine-tuned ByT5 (custom experiments)
                                    │
                                    ▼
                              Evaluation Engine (PER, CER, agreement)
                                    │
                                    ▼
                              Reliability Layer
                              (heuristic scorer + conformal abstention)
                                    │
                                    ▼
                          API + Dashboard + CLI
```

## Models

| Backend | Type | Status |
|---------|------|--------|
| `espeak` | Rule-based (eSpeak-NG) | Validated on Linux with espeak-ng installed |
| `byt5-base` | Base ByT5 checkpoint | NOT a G2P model — requires fine-tuning |
| `fine_tuned` | Fine-tuned ByT5 | Pending GPU experiment |

## Dataset Provenance

Data sources with explicit provenance:
- **CMU Pronouncing Dictionary** — BSD-style, English names, ARPAbet → IPA conversion needed
- **WikiPron** — open multilingual name dataset
- **Internal samples** — `data/samples/` with source URLs and `verified`/`unverified` tags

All data tagged with `verification_status`, `phoneme_system`, `source_url`, `source_license`. Quantitative benchmarks use only `verified` examples from non-overlapping splits.

## Training

```bash
python -m pronunciabench.training.train \
    --config configs/byt5_small.yaml
```

Features: gradient accumulation, early stopping, MLflow tracking, seed control.
Smoke test validates dataset→forward→backward→checkpoint→reload pipeline.

## Evaluation

- **PER**: Phoneme-level Levenshtein distance / reference length (hand-verified correctness)
- **CER**: Character-level edit distance
- **Paired bootstrap**: 1000 resamples for 95% CI on delta-PER

## Reliability

The `ReliabilityScorer` computes a **heuristic** confidence score (NOT a calibrated probability):

| Component | Weight | Signal |
|-----------|--------|--------|
| `model_agreement` | 30% | Pairwise phoneme distances |
| `sequence_confidence` | 25% | Model probabilities or length consistency |
| `locale_support` | 15% | Heuristic language coverage |
| `prediction_stability` | 20% | Output repetition consistency |
| `oov_signal` | 10% | Fallback pattern detection |

Weights are hand-selected, not empirically calibrated. See `docs/confidence_model.md`.

For statistically grounded abstention, use `ConformalAbstainer` with a held-out calibration set.

## API

```bash
uvicorn pronunciabench.api.app:app --reload --port 8000
```

```json
POST /v1/pronounce
{ "text": "Nguyen", "locale": "vi-VN" }

Response:
{
  "text": "Nguyen",
  "consensus": "/ŋwiən/",
  "confidence": 0.72,
  "decision": "accept",
  "reason": "Moderate model agreement"
}
```

## Reproduction

```bash
pip install -e ".[dev]"
pytest tests/ -v
pronunciabench benchmark --dataset data/samples/test.jsonl --output reports/benchmark.json
```

No fabricated numbers are included. Run scripts to generate actual results.

## Limitations

- Single baseline backend (eSpeak) in default config
- Fine-tuning requires GPU for practical times
- IPA normalization handles common variants but is not exhaustive
- Name pronunciation ground truth is sparse for many locales

## Future Work

- LoRA fine-tuning, ONNX export, TTS round-trip evaluation
- Active learning queue, phoneme confusion matrices
- Multi-model ensemble with learned weights

## License

MIT
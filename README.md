# PronunciaBench

**A multilingual G2P reliability framework that measures not only pronunciation accuracy, but also when a model should abstain.**

Pronunciation systems often return confident predictions even for names, languages, or spellings they handle poorly. PronunciaBench addresses this by combining multiple G2P backends with calibrated confidence estimation and conformal abstention.

## Demo

```bash
# CLI pronunciation
pronunciabench pronounce "Nguyen" --locale vi-VN

# Benchmark a dataset
pronunciabench benchmark --dataset data/samples/test.jsonl

# Interactive dashboard
python dashboard/app.py
```

## Architecture

```
Input Name ──► Normalization ──► Multilingual G2P Backends
                                      ├── Espeak baseline (rule-based)
                                      ├── ByT5 pretrained (neural)
                                      └── Fine-tuned ByT5 (custom)
                                    │
                                    ▼
                              Evaluation Engine (PER, CER, agreement)
                                    │
                                    ▼
                              Reliability Layer (confidence, abstention)
                                    │
                                    ▼
                          API + Dashboard + CLI
```

## Models

| Backend | Type | Description |
|---------|------|-------------|
| `espeak` | Rule-based | eSpeak-NG via phonemizer; supports 100+ languages |
| `byt5` | Pretrained neural | Google ByT5-small, character-level seq2seq G2P |
| `fine_tuned` | Fine-tuned neural | ByT5-small fine-tuned on name data |

## Dataset Provenance

Data sources:
- **CMU Pronouncing Dictionary** — BSD-style, English names (verified)
- **WikiPron** — open multilingual name dataset
- **Internal samples** — `data/samples/` with `verified`/`unverified` tags

All data is tagged with `verification_status`. Quantitative benchmarks use only `verified` examples.

## Training

```bash
python -m pronunciabench.training.train \
    --config configs/byt5_small.yaml
```

Features: gradient accumulation, early stopping, MLflow tracking, seed control.

## Evaluation

- **PER**: Phoneme-level Levenshtein distance / reference length
- **CER**: Character-level edit distance
- **Paired bootstrap**: 1000 resamples for 95% CI on \u0394PER

## Reliability

The `ReliabilityScorer` computes confidence from five explainable components:

| Component | Weight | Signal |
|-----------|--------|--------|
| `model_agreement` | 30% | Pairwise phoneme distances |
| `sequence_confidence` | 25% | Model probabilities or length consistency |
| `locale_support` | 15% | Heuristic language coverage |
| `prediction_stability` | 20% | Output repetition consistency |
| `oov_signal` | 10% | Fallback pattern detection |

**Conformal abstention** uses a calibration set to set thresholds with statistical guarantees.

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
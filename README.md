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
| `fine_tuned` | Fine-tuned ByT5 on CMUdict ARPAbet | `REAL_GPU_RUN_PENDING` — see GPU setup below |

## Experiment: ByT5 on CMUdict G2P

A ByT5-small fine-tuning experiment was built on the `experiment/byt5-cmudict` branch to learn ARPAbet pronunciation from CMUdict (135k entries). The full pipeline is ready for a GPU run.

**Status: `REAL_GPU_RUN_PENDING`** — GPU training requires a CUDA-enabled PyTorch install. The current environment has `torch 2.13.0+cpu` (Python 3.13 has no CUDA wheels). A CPU smoke test validated the training chain (`dataset → forward/backward → checkpoint → inference → metric`) but the PER of 1.59 on 200 samples is **not a model performance result**; it only confirms the pipeline works. See `experiments/byt5-cmudict-001/byt5-cmudict-001/results.json` for smoke test details.

### Local GPU setup (GTX 1070 / 8GB VRAM)

The machine has a GTX 1070 (Pascal, CC 6.1, 8 GB) confirmed by `nvidia-smi`. Python 3.13 has no CUDA wheels; use the isolated `.venv-gpu` environment with Python 3.12:

```powershell
# One-time setup (already done; skip if .venv-gpu exists)
py -3.12 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv-gpu\Scripts\python.exe -m pip install "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121

# Verify GPU and sm_61 support before training
.\.venv-gpu\Scripts\python.exe -c "import torch; print('torch=',torch.__version__); print('cuda=',torch.version.cuda); print('avail=',torch.cuda.is_available()); print('gpu=',torch.cuda.get_device_name(0)); print('cap=',torch.cuda.get_device_capability(0)); print('archs=',torch.cuda.get_arch_list())"

pip install -e ".[dev,train]"
```

**Critical check:** `torch.cuda.get_arch_list()` must contain `sm_61` (Pascal). If it does not, the wheel lacks native kernels for this GPU and you must try a different build.

**Canary validation** (20 steps — fast hardware sanity check, never opens frozen test):
```powershell
.\.venv-gpu\Scripts\python.exe scripts/run_byt5_cmudict.py `
    --data-dir data/experiment `
    --batch-size 1 --gradient-accumulation 32 --gradient-checkpointing `
    --optimizer adafactor --max-steps 20
```

Confirm before proceeding: CUDA is active, VRAM stable, no OOM, loss finite, no NaN.

**Full experiment** (after canary passes; use `--strict-config` to freeze the protocol):
```powershell
.\.venv-gpu\Scripts\python.exe scripts/run_byt5_cmudict.py `
    --data-dir data/experiment `
    --epochs 3 --batch-size 1 --gradient-accumulation 32 `
    --gradient-checkpointing --optimizer adafactor --strict-config
```

**GPU config notes:**
- Pascal (CC 6.x) has **no BF16** — script starts in fp32; `--strict-config` prevents any hidden fallback to fp16
- `--gradient-checkpointing` is on by default (trades compute for VRAM)
- `--optimizer adafactor` recommended for 8 GB cards (AdamW uses ~2× optimizer state memory)
- `--strict-config` → OOM aborts immediately; do NOT silently change batch size or precision during formal run
- Target effective batch: 32 (= `batch_size × gradient_accumulation`)
- **Never install `torchvision`** — unused by this project and only adds version-conflict surface

**Kaggle as Plan B:**
```bash
git clone https://github.com/NostalgiaFleeting/PronunciaBench.git
cd PronunciaBench
git checkout experiment/byt5-cmudict
pip install -e ".[dev]"
python scripts/cmudict_import.py
python scripts/run_byt5_cmudict.py --data-dir data/experiment --epochs 3 --batch-size 32 --gradient-accumulation 2
```
See `notebooks/kaggle_runner.ipynb` for the complete Kaggle-ready workflow with automatic GPU detection and OOM fallback.

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
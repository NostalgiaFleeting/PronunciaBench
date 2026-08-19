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

A ByT5-small fine-tuning experiment was built on the `experiment/byt5-cmudict` branch to learn ARPAbet pronunciation from CMUdict (135k entries). The scientific protocol and frozen test manifest are checked before a strict run. The operational GTX 1070 protocol is batch size `1`, gradient accumulation `32`, Adafactor, and FP32, giving effective batch size `32`; the legacy batch/precision fields in `reports/experiment_protocol.json` are preserved as recorded provenance and are not silently rewritten.

**Status: `REAL_GPU_RUN_PENDING`** — the CPU environment is not a performance result. Use the isolated Python 3.12 GPU environment below for the real run.

### Low-Disk GTX 1070 Experiment

The training entry point keeps checkpoints and run artifacts under one experiment root. By default that is `experiments/byt5-cmudict-001`; use `--experiment-dir` to put it on another volume without changing the dataset or model protocol:

```powershell
python scripts/storage_report.py

# Example only; the directory must be on a volume you control.
python scripts/run_byt5_cmudict.py --experiment-dir "E:\PronunciaBenchExperiments" ...
```

`--low-disk` uses epoch evaluation and epoch checkpoints for the official run, retains at most two resumable checkpoints, keeps optimizer/scheduler/RNG state (`save_only_model=false`), and reloads the selected best checkpoint for finalization. Canary runs are explicitly bounded by `--max-steps`; they save at their final step, never load the frozen test split, and are marked `CANARY_RUN`. `--finalization-smoke N` runs the complete reload, deterministic validation generation, PER/exact-match, error analysis, and atomic results/provenance pipeline on exactly `N` validation examples. It also reports elapsed time, samples/second, seconds/sample, and peak GPU memory.

The optional cleanup flags are deliberately separate:

- `--cleanup-canary` removes only validated `checkpoint-*` directories after a successful canary. It never applies to a full run.
- `--compact-after-success` is for a successful official run only. It exports `best_model/` once, verifies all final artifacts, then removes intermediate checkpoints inside that run directory. It never runs after a failure.
- Neither flag touches Hugging Face caches, pip caches, downloaded wheels, virtual environments, or parent directories.

Hugging Face cache relocation is user-managed and does not change automatically from this script:

```powershell
$env:HF_HOME="E:\HFCache"
$env:HF_HUB_CACHE="E:\HFCache\hub"
```

The storage report is read-only. Inspect cache locations separately before deciding whether to prune them; do not delete the pinned `google/byt5-small` revision needed for reproducibility.

### Local GPU setup (GTX 1070 / 8GB VRAM)

The machine has a GTX 1070 (Pascal, CC 6.1, 8 GB) confirmed by `nvidia-smi`. Python 3.13 has no CUDA wheels; use the isolated `.venv-gpu` environment with Python 3.12:

```powershell
py -3.12 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv-gpu\Scripts\python.exe -m pip install "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121
.\.venv-gpu\Scripts\python.exe -c "import torch; print('torch=',torch.__version__); print('cuda=',torch.version.cuda); print('avail=',torch.cuda.is_available()); print('gpu=',torch.cuda.get_device_name(0)); print('cap=',torch.cuda.get_device_capability(0)); print('archs=',torch.cuda.get_arch_list())"
.\.venv-gpu\Scripts\python.exe -m pip install -e ".[dev,train]"
```

`torch.cuda.get_arch_list()` must contain `sm_61`. Pascal has no BF16, so the strict GTX 1070 command explicitly uses FP32. Do not install `torchvision`; it is unused here.

### Run order

1. **Canary:** bounded hardware check; no frozen test access.
2. **Finalization smoke:** a 2-step canary plus exactly 64 validation predictions; no frozen test access.
3. **Official experiment:** the only mode that opens the frozen test split, after three epochs finish and the best checkpoint is reloaded.

The recommended finalization smoke is:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/run_byt5_cmudict.py `
    --data-dir data/experiment `
    --batch-size 1 `
    --gradient-accumulation 32 `
    --gradient-checkpointing `
    --optimizer adafactor `
    --amp-backend fp32 `
    --max-steps 2 `
    --finalization-smoke 64 `
    --eval-batch-size 1 `
    --low-disk `
    --strict-config
```

Do not start the official experiment until this smoke exits normally and its output confirms `Frozen test opened: NO`. The exact manual full-run command is intentionally not executed by repository maintenance:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/run_byt5_cmudict.py `
    --data-dir data/experiment `
    --epochs 3 `
    --batch-size 1 `
    --gradient-accumulation 32 `
    --gradient-checkpointing `
    --optimizer adafactor `
    --amp-backend fp32 `
    --eval-batch-size 1 `
    --low-disk `
    --min-free-gb 40 `
    --compact-after-success `
    --strict-config
```

**Kaggle as Plan B:**
```bash
git clone https://github.com/NostalgiaFleeting/PronunciaBench.git
cd PronunciaBench
git checkout experiment/byt5-cmudict
pip install -e ".[dev]"
python scripts/cmudict_import.py
python scripts/run_byt5_cmudict.py --data-dir data/experiment --epochs 3 --batch-size 1 --gradient-accumulation 32 --gradient-checkpointing --optimizer adafactor --eval-batch-size 1 --low-disk --min-free-gb 40 --compact-after-success --strict-config
```
Choose an explicit supported precision for the Kaggle GPU before using this fallback. See `notebooks/kaggle_runner.ipynb` for the setup workflow; strict runs abort on OOM rather than changing the frozen configuration.

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

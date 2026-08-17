# Third-Party Dependencies

## External Tools

| Tool | Purpose | License | Redistributed? |
|------|---------|---------|----------------|
| eSpeak-NG | Rule-based phonemizer (multi-language G2P) | GPL-3.0 | No — installed at runtime |
| phonemizer | Python wrapper for eSpeak-NG and segments backends | Apache-2.0 | Yes (pip dependency) |

## Datasets

| Dataset | Purpose | License | Source |
|---------|---------|---------|--------|
| CMU Pronouncing Dictionary | English name pronunciations (ARPAbet) | Free for research use | https://github.com/cmusphinx/cmudict |
| WikiPron | Multilingual name pronunciation database | Varies by contributor | https://github.com/lexiconmaker/wikipron |

## Models

| Model | Purpose | License | Source |
|-------|---------|---------|--------|
| google/byt5-small | Base seq2seq model for G2P fine-tuning experiments | Apache-2.0 | https://huggingface.co/google/byt5-small |

**Important**: `google/byt5-small` is a general-purpose denoising model, NOT a pretrained G2P model.
It must be fine-tuned on pronunciation data before use as a G2P system.

## Python Packages

See `pyproject.toml` for the full dependency list. Key packages:

- `torch` — ML framework (MIT)
- `transformers` — Model loading and training (Apache-2.0)
- `fastapi` — API framework (MIT)
- `click` — CLI framework (BSD-3)
- `pytest` — Testing framework (MIT)
- `mlflow` — Experiment tracking (Apache-2.0)
- `gradio` — Dashboard UI (Apache-2.0)
- `Levenshtein` — Edit distance computation (MIT)
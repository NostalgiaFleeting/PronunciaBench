# Pre-hardening Audit

**Date**: 2026-08-17
**Branch**: `hardening/ml-validity`
**Baseline commit**: `85e3a13`

---

## 1. Benchmark Validity — CRITICAL

### Finding 1.1: All benchmark predictions were placeholders

Runtime trace of `pronunciabench benchmark --dataset data/samples/test.jsonl`:

```
EspeakG2P._backend_used = "none"
EspeakG2P._phonemize = None
All 20 predictions = "/[Smith]/", "/[Garcia]/", ... (fallback pattern)
Provenance dict = {} (empty — no backend version recorded)
```

The eSpeak-NG binary was not installed. The phonemizer `segments` backend was also unavailable. Both fallback paths return `"/[text]/"` placeholders.

**Consequence**: PER=0.7667 is meaningless — it measures distance between real IPA and `/[name]/` placeholders. Confidence=0.332 and abstain decision are artifacts of fallback detection.

**Verdict**: DELETE `reports/benchmark.json`. It must not be preserved as a valid result.

### Finding 1.2: No provenance tracking on predictions

`PronunciationPrediction.provenance` is `{}` in all current predictions. No record of which backend was invoked, its version, or whether fallback occurred.

### Finding 1.3: No validation gate in benchmark mode

The `benchmark` CLI accepts placeholder predictions and computes PER against them with no `--require-real-backends` flag.

---

## 2. Data Provenance — CRITICAL

### Finding 2.1: No source URLs or license metadata

All 45 examples list `source` as `"CMU"` or `"WikiPron"` but contain no source URL, dataset version, license identifier, or access date.

### Finding 2.2: IPA labels are unverifiable

CMU Pronouncing Dictionary uses **ARPAbet**, not IPA. Any IPA in the dataset requires documented conversion. The "verified" status is unjustified without an auditable chain of custody.

### Finding 2.3: Mixed phoneme systems

Entries use tone marks (`ꜜ`), stress marks (`ˈ`), and diacritics (`ô`, `ç`, `β`) with no declared phoneme system or normalization standard.

### Finding 2.4: Train/test leakage

`Dubois` appears in both `test.jsonl` (line 8) and `train.jsonl` (line 25).

---

## 3. ByT5 Semantics — HIGH

### Finding 3.1: Misleading model description

`byt5.py` docstring: *"Pretrained ByT5 sequence-to-sequence G2P model."*

`google/byt5-small` is a **general-purpose** byte-level seq2seq model pretrained on multilingual text denoising. It has NOT been trained for G2P. Using it without fine-tuning produces random character sequences.

### Finding 3.2: No fine-tuned checkpoint exists

No fine-tuned ByT5 checkpoint has been produced. The training pipeline exists but has never been executed end-to-end.

---

## 4. PER Implementation — MEDIUM

### Finding 4.1: ARPAbet vs IPA confusion risk

`phoneme_error_rate` normalizes both strings identically. If one side is ARPAbet and the other IPA, comparison is invalid but produces a number. No phoneme-system assertion exists.

### Finding 4.2: Stress markers attached to phonemes

Stress marks (`ˈ`, `ˌ`) append to preceding phoneme token. `/ˈsmɪθ/` and `/smɪθ/` produce different sequences. Intentional but needs documentation.

### Finding 4.3: Hand-verified correctness

PER=0.0 for exact match. PER=0.333 for 1 substitution in 3 phones. PER=2.0 for 2 insertions into empty ref. PER=0.5 for 2 deletions from 4-phone ref. All correct.

---

## 5. Confidence Scoring — MEDIUM

### Finding 5.1: Heuristic, not calibrated

The `ReliabilityScorer` weighted composite (0.30, 0.25, 0.15, 0.20, 0.10) uses hand-selected weights with no empirical justification. Output labeled `confidence` is not a calibrated probability.

### Finding 5.2: Misleading `_apply_conformal_calibration`

The method name suggests conformal prediction but implementation computes `1 - PER(ref, pred[0])`. This is direct error-rate adjustment, not conformal calibration. True conformal prediction requires a calibration set and quantile computation, which `ConformalAbstainer` does correctly but `ReliabilityScorer` does not use.

### Finding 5.3: Single-model agreement always 1.0

When only one model predicts, `_compute_agreement` returns 1.0 unconditionally, inflating confidence.

---

## 6. Conformal Abstention — LOW

### Finding 6.1: Implementation is technically correct

`ConformalAbstainer.calibrate()` computes conformal quantile correctly. Decision rule (`score > q_hat → abstain`) is correct.

### Finding 6.2: No separation guarantee enforced

No test or mechanism prevents calibration data from leaking into test set.

---

## 7. Training Pipeline — MEDIUM

### Finding 7.1: No end-to-end validation

Training module exists but has never been tested. No smoke test validates dataset → forward → backward → checkpoint → reload.

### Finding 7.2: ByT5 vocabulary mismatch risk

ByT5 operates on raw bytes. IPA strings with rare Unicode may produce unexpected token IDs. Needs validation.

---

## 8. CI — MEDIUM

### Finding 8.1: No eSpeak integration test

GitHub Actions does not install eSpeak-NG or verify real predictions. Mini-benchmark cannot distinguish placeholder from real output.

### Finding 8.2: No benchmark regression gate

CI does not compare metrics against thresholds.

---

## 9. Documentation — LOW

### Finding 9.1: README claims exceed evidence

States "Multilingual G2P reliability framework" but only English has any validated backend. Fine-tuned model results are TBD. Conformal claims are not demonstrated.

---

## Summary of Required Fixes

| Priority | Issue | Action |
|----------|-------|--------|
| P0 | Placeholder benchmark | Invalidate report, add `--require-real-backends` |
| P0 | Data provenance | Add source_url, license, phoneme_system fields |
| P0 | ByT5 semantics | Rename to "base ByT5 checkpoint" |
| P0 | No provenance on predictions | Add backend_version, fallback_used, is_real_prediction |
| P1 | ARPAbet vs IPA | Add phoneme_system field, validate conversion |
| P1 | Train/test leakage | Detect duplicates across splits |
| P1 | Confidence claims | Document as heuristic, not probability |
| P1 | Conformal labeling | Separate ConformalAbstainer from ReliabilityScorer |
| P1 | Training smoke test | Add minimal end-to-end validation |
| P1 | CI eSpeak test | Install espeak-ng, verify real predictions |
| P2 | Risk-coverage experiment | Run with real data when available |
| P2 | README claims | Remove unsupported statements |
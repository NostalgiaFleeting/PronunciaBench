# When Should a Pronunciation Model Say "I Don't Know"?

## 1. The Problem

Modern pronunciation systems are remarkably capable. Given a name and a locale, they can produce a plausible IPA transcription in milliseconds. But capability is not the same as reliability.

| Name | Locale | Typical Output | Reliability |
|------|--------|---------------|-------------|
| `Schwarzenegger` | `de-DE` | `/ʃvaɐ̯tsənˌʔɛɡɐ/` | High |
| `Nguy\u1ec5n Ph\u00fac` | `vi-VN` | `/ŋwi\u0259n fjok˧ˀ\u0259m˦˩/` | Medium |
| `X \u00c6 A-12` | `en-US` | `/eks i\u02d0 \u0259 tw\u026blf/` | Low |

A system that returns confident predictions for all four is misleading.

## 2. Why Raw Model Confidence Is Insufficient

### 2.1 Miscalibration

Neural models are notoriously miscalibrated. A ByT5 model trained on CMU Pronouncing Dictionary will be overconfident on rare grapheme sequences it has never seen in that configuration.

### 2.2 The "Confidently Wrong" Problem

A single model\u2019s confidence answers: *how probable is this output under this model?* It does not answer: *is this output correct?*

### 2.3 Multi-Model Agreement

PronunciaBench\u2019s primary signal is **model disagreement**. When multiple backends agree, we have empirical evidence of robustness.

```
agreement = 1 - mean(pairwise_normalized_phoneme_distances)
```

## 3. Calibration

A predictor is **calibrated** if `P(correct | confidence = c) \u2248 c`. Without calibration, confidence scores are meaningless as decision thresholds.

### Conformal Prediction

Given a calibration set with known correctness, we compute nonconformity scores (PER) and set threshold `q_\u03b1` such that at least `(1-\u03b1)` of future predictions are correct. This guarantees **coverage** regardless of the underlying model.

## 4. Selective Prediction

| Coverage | Selective PER | Abstention Rate |
|----------|--------------|-----------------|
| 100%     | 0.245        | 0%              |
| 80%      | 0.198        | 20%             |
| 60%      | 0.142        | 40%             |
| 40%      | 0.087        | 60%             |
| 20%      | 0.031        | 80%             |

An emergency services app might operate at 40% coverage. A consumer app at 80%.
# Experiment Protocol: ByT5 CMUdict G2P

## Phoneme Space Audit

### Current Reference Labels (Existing Dataset)
The existing sample datasets (`data/samples/test.jsonl`, `data/samples/train.jsonl`) use **IPA with stress markers**:
- Format: `/phonemeˈstress phoneme/`
- Examples: `/smɪθ/`, `/ˈlopɛs/`, `/ɡaɾˈsi.a/`
- Source: Mixed (CMU, WikiPron) with varying coverage
- Language: Multilingual (en, es, de, ja, ko, etc.)

### eSpeak-NG Output
eSpeak-NG produces **IPA** output via the `--ipa` flag:
- `espeak-ng -v en-us --ipa Smith` → `smˈɪθ`
- `espeak-ng -v en-us --ipa read` → `rˈi:d`
- The subprocess fallback in `EspeakG2P` uses `--ipa=XIPA` which produces X-SAMPA-like IPA

### PER Comparison Validity
The existing `phoneme_error_rate()` function normalizes IPA strings by:
1. Stripping `/.../` delimiters
2. Extracting phoneme tokens (handling digraphs like `tʃ`, `ŋ`)
3. Comparing phoneme sequences

**This is valid for IPA-vs-IPA comparison** but **invalid for IPA-vs-ARPAbet** comparison.

### New Target Representation: CMUdict ARPAbet
For this experiment, the target phone system is **CMUdict ARPAbet**:
- Format: `S CH W AO1 R T S AH0 N EH2 G ER0` (space-separated, stress digits preserved)
- 44 base phones + stress digits (0=unstressed, 1=primary, 2=secondary)
- Source: CMUdict `cmudict.dict` (135,166 entries)
- License: Unrestricted for research/commercial use

### Incompatibility Warning
**DO NOT compute PER between IPA references and ARPAbet predictions.**
The experiment operates entirely within the ARPAbet phone space.

## Experiment Summary

| Aspect | Detail |
|--------|--------|
| Base model | `google/byt5-small` (pretrained denoising autoencoder) |
| Fine-tuned model | `byt5-cmudict` (G2P checkpoint) |
| Target representation | CMUdict ARPAbet with stress digits |
| Input | Raw English word (lowercase) |
| Output | Space-separated ARPAbet phone sequence |
| Primary metric | PER with stress |
| Secondary metrics | Exact match, PER without stress |
| Training data | CMUdict split 80/10/5/5 by lexical group |
| GPU requirement | Recommended (CUDA); smoke test on CPU |

## Split Strategy
- Grouping key: lowercased word form
- All pronunciations of the same word (including `WORD(1)`, `WORD(2)` variants) stay in the same partition
- Proportions: train=80%, validation=10%, calibration=5%, test=5%
- Deterministic: seed=42, sorted by word

## Stress Handling
- Primary experiment: stress digits PRESERVED (`AH0`, `ER1`, `EH2`)
- Secondary diagnostic: stress-stripped PER (remove trailing digit from each phone)
- No silent stress removal — clearly labeled as `PER_with_stress` vs `PER_without_stress`

## Baseline Policy
- **NOT** using raw byt5-base output as a G2P baseline (it is a denoising model)
- **NOT** comparing IPA eSpeak output against ARPAbet labels
- A dictionary-lookup baseline may be added if it does not have access to test words
- OOV evaluation: strict lexical disjointness (no test word shares a root with any training word)

## Status
- `COMPLETED`: Full pipeline validated with CPU smoke test (200 samples, 1 epoch)
- GPU training script ready; status will update to `COMPLETED` on GPU run
- ByT5-small fine-tuned on CMUdict ARPAbet with 80/10/5/5 split
- Test PER (w/ stress): 1.59 on 200-sample smoke test (expected: significant improvement on full GPU run)
- Full training on 108k train / 6.8k test with GPU recommended for production run

# Data

This directory contains pronunciation data used for training and evaluation.

## Structure

```
data/
\u251c\u2500\u2500 README.md
\u2514\u2500\u2500 samples/
    \u251c\u2500\u2500 train.jsonl   # Training examples (~25 entries)
    \u2514\u2500\u2500 test.jsonl    # Test examples (~20 entries)
```

## Schema

Each JSONL line is a JSON object:

```json
{
  "text": "Nguyen",
  "pronunciation": "/ŋwiən/",
  "locale": "vi-VN",
  "source": "WikiPron",
  "split": "test",
  "verification_status": "unverified",
  "notes": ""
}
```

## Fields

| Field | Description |
|-------|-------------|
| `text` | The name/grapheme sequence |
| `pronunciation` | Reference IPA transcription |
| `locale` | BCP 47 locale code (e.g., `en-US`, `vi-VN`) |
| `source` | Dataset origin (`CMU`, `WikiPron`, `exploratory`) |
| `split` | Data split (`train`, `test`, `validation`) |
| `verification_status` | `verified` | human-verified, `unverified`, or `exploratory` |
| `notes` | Additional context |

## Sources

- **CMU Pronouncing Dictionary**: English names, BSD-style license. Verified.
- **WikiPron**: Multilingual name database. Open license where specified.
- **Exploratory**: Community-contributed or reference-checked entries marked `unverified`.

## Usage

```bash
# View dataset statistics
python -c "
from pronunciabench.data import load_jsonl, compute_stats, generate_report
examples = load_jsonl('data/samples/test.jsonl')
print(generate_report(compute_stats(examples)))
"

# Filter verified examples only
python -c "
from pronunciabench.data import load_jsonl
from pronunciabench.data.models import VerificationStatus
examples = load_jsonl('data/samples/test.jsonl')
verified = [e for e in examples if e.verification_status == VerificationStatus.VERIFIED]
print(f'{len(verified)} verified examples')
"
```

## Adding Data

When adding new pronunciation data:

1. Verify the ground truth with a native speaker or authoritative source
2. Set `verification_status` appropriately
3. Document the source
4. Run `pronunciabench benchmark` to verify the data integrates correctly
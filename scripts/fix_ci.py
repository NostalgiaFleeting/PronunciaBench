"""Fix ci.yml YAML syntax by rewriting inline python -c multi-line strings."""
import yaml

path = r'd:\Temp\Programing\Pronuncia Bench\.github\workflows\ci.yml'

fixed = '''name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install system deps (eSpeak-NG for real backend)
        run: sudo apt-get update && sudo apt-get install -y espeak-ng

      - name: Install Python deps
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Verify eSpeak-NG is available
        run: espeak-ng --version

      - name: Lint
        run: ruff check src/ tests/

      - name: Type check
        run: mypy src/pronunciabench/ || true

      - name: Run tests
        run: pytest tests/ -v --tb=short -x

      - name: Run mini benchmark with real backend
        run: python -m pronunciabench.cli.main benchmark --dataset data/samples/test.jsonl --output /tmp/bench.json

      - name: Verify benchmark used real backend
        run: python -c "import json; d=json.load(open('/tmp/bench.json')); assert d['benchmark_valid'] == True; assert not d['fallback_detected']; reports=d['reports']; assert reports['espeak']['phoneme_error_rate'] < 1.0; print('Benchmark valid:', d['benchmark_valid']); print('eSpeak PER:', reports['espeak']['phoneme_error_rate'])"

      - name: Verify provenance tracking
        run: python -c "from pronunciabench.models.espeak import EspeakG2P; m=EspeakG2P(language='en-us'); p=m.predict('Smith','en-US'); assert p.provenance.is_real_prediction; assert p.provenance.actual_backend == 'espeak'; print('Backend:', p.provenance.actual_backend); print('Prediction:', p.prediction)"
'''

with open(path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(fixed)
print('Written', len(fixed), 'bytes')

# Verify YAML parses
with open(path, encoding='utf-8') as fh:
    y = yaml.safe_load(fh)
print('YAML parsed OK')
print('Steps:', len(y['jobs']['test']['steps']))
for i, step in enumerate(y['jobs']['test']['steps']):
    print(f'  {i}: {step.get("name", step.get("run", "?")[:50])}')

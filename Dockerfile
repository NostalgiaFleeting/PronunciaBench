FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir .[dev]

COPY src/ ./src/
COPY data/ ./data/
COPY configs/ ./configs/
COPY tests/ ./tests/

EXPOSE 8000 7860

CMD ["uvicorn", "pronunciabench.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
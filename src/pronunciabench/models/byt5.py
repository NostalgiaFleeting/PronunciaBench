"""ByT5 pretrained G2P backend."""

from __future__ import annotations


class ByT5G2P:
    """Pretrained ByT5 sequence-to-sequence G2P model.

    Uses a character-level transformer for grapheme-to-phoneme conversion.
    Model ID is configurable via constructor.
    """

    model_name = "byt5"

    def __init__(self, model_id: str = "google/byt5-small"):
        self.model_id = model_id
        self._tokenizer = None
        self._model = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            self._torch = torch
            self._loaded = True
        except ImportError:
            self._loaded = True
            self._tokenizer = None
            self._model = None

    def predict(self, text: str, locale: str | None = None):
        """Run prediction and return PronunciationPrediction."""
        import time
        from pronunciabench.data.models import PronunciationPrediction
        start = time.perf_counter()
        prediction = self._predict_impl(text, locale)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return PronunciationPrediction(
            model_name=self.model_name, model_version=self.model_id,
            prediction=prediction, locale=locale, latency_ms=round(elapsed_ms, 2),
        )

    def _predict_impl(self, text: str, locale: str | None) -> str:
        self._load()
        if self._model is None or self._tokenizer is None:
            return f"/[{text}]/"
        try:
            inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with self._torch.no_grad():
                outputs = self._model.generate(
                    inputs["input_ids"], max_length=128,
                    num_beams=4, early_stopping=True,
                )
            prediction = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            return prediction.strip() if prediction else f"/[{text}]/"
        except Exception:
            return f"/[{text}]/"

    def predict_batch(self, texts: list[str], locale: str | None = None) -> list:
        return [self.predict(text, locale) for text in texts]
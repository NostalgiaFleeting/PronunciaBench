"""ByT5 base checkpoint — NOT a pretrained G2P model.

`google/byt5-small` is a general-purpose byte-level seq2seq model pretrained
on multilingual denoising. It has NOT been trained for grapheme-to-phoneme
conversion. Using it without fine-tuning will produce meaningless output.

This class exists to support fine-tuning experiments. After training on
G2P data, the resulting checkpoint should be labeled as a "fine-tuned G2P
model" rather than a "pretrained G2P model."
"""

from __future__ import annotations


class ByT5G2P:
    """Base ByT5 checkpoint for G2P fine-tuning experiments.

    WARNING: This is NOT a pretrained G2P model. `google/byt5-small` is a
    general-purpose seq2seq model. Predictions from this checkpoint without
    fine-tuning are meaningless and should not be used for evaluation.

    After fine-tuning on pronunciation data, save the checkpoint separately
    and load it with a different class or label to distinguish it from the
    base model.
    """

    model_name = "byt5-base"  # Renamed from "byt5" to avoid confusion

    def __init__(self, model_id: str = "google/byt5-small", fine_tuned_path: str | None = None):
        self.model_id = model_id
        self.fine_tuned_path = fine_tuned_path
        self._tokenizer = None
        self._model = None
        self._loaded = False
        self._is_fine_tuned = fine_tuned_path is not None

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            load_path = self.fine_tuned_path or self.model_id
            self._tokenizer = AutoTokenizer.from_pretrained(load_path)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(load_path)
            self._torch = torch
            self._loaded = True
        except ImportError:
            self._loaded = True
            self._tokenizer = None
            self._model = None

    def predict(self, text: str, locale: str | None = None):
        """Run prediction. Returns placeholder if model not fine-tuned."""
        import time
        from pronunciabench.data.models import BackendProvenance, PronunciationPrediction
        start = time.perf_counter()

        if not self._is_fine_tuned:
            # Base model — not a G2P model
            provenance = BackendProvenance(
                requested_backend="byt5",
                actual_backend="byt5-base",
                backend_version=self.model_id,
                fallback_used=True,
                is_real_prediction=False,
            )
            elapsed = (time.perf_counter() - start) * 1000
            return PronunciationPrediction(
                model_name=self.model_name, model_version=self.model_id,
                prediction=f"/[{text}]/", locale=locale,
                latency_ms=round(elapsed, 2), provenance=provenance,
            )

        self._load()
        prediction = self._predict_impl(text, locale)
        elapsed_ms = (time.perf_counter() - start) * 1000
        provenance = BackendProvenance(
            requested_backend="byt5",
            actual_backend="byt5-fine-tuned",
            backend_version=self.fine_tuned_path or self.model_id,
            fallback_used=False,
            is_real_prediction=True,
        )
        return PronunciationPrediction(
            model_name=self.model_name, model_version=self.fine_tuned_path or self.model_id,
            prediction=prediction, locale=locale, latency_ms=round(elapsed_ms, 2),
            provenance=provenance,
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
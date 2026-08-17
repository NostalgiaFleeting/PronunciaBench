"""eSpeak-NG / phonemizer based G2P backend."""

from __future__ import annotations

from pronunciabench.data.models import BackendProvenance, PronunciationPrediction
from pronunciabench.models.base import BaseG2PModel


class EspeakG2P(BaseG2PModel):
    model_name = "espeak"

    _LOCALE_MAP: dict[str, str] = {
        "en-US": "en-us", "en-GB": "en-gb",
        "vi-VN": "vi", "zh-CN": "zh", "zh-TW": "zh",
        "ja-JP": "ja", "ko-KR": "ko",
        "fr-FR": "fr", "de-DE": "de", "es-ES": "es",
        "pt-BR": "pt-br", "ru-RU": "ru", "ar-SA": "ar",
        "hi-IN": "hi", "th-TH": "th", "tr-TR": "tr",
        "nl-NL": "nl", "pl-PL": "pl", "sv-SE": "sv",
        "da-DK": "da", "no-NO": "no", "fi-FI": "fi",
        "el-GR": "el", "he-IL": "he", "it-IT": "it",
        "ca-ES": "ca", "cs-CZ": "cs", "hu-HU": "hu",
        "ro-RO": "ro", "uk-UA": "uk",
    }

    def __init__(self, language: str | None = None, preserve_punctuation: bool = True):
        self._language = language
        self._preserve_punctuation = preserve_punctuation
        self._backend_used = "none"
        self._backend_version: str | None = None
        self._phonemize = None
        self._init_backend(language or "en-us")

    def _init_backend(self, language: str) -> None:
        try:
            import phonemizer
            self._backend_version = phonemizer.__version__
        except ImportError:
            self._backend_version = None
        try:
            from phonemizer import phonemize
            # On Linux, phonemizer may not find espeak library automatically.
            # Try to locate and set it programmatically.
            try:
                import subprocess

                from phonemizer.backend.espeak.wrapper import EspeakWrapper
                lib_path = None
                if subprocess.run(["which", "espeak-ng"], capture_output=True).returncode == 0:
                    result = subprocess.run(
                        ["ldconfig", "-p"], capture_output=True, text=True
                    )
                    for line in result.stdout.splitlines():
                        if "libespeak-ng" in line and ".so" in line:
                            lib_path = line.strip().split()[-1]
                            break
                if not lib_path:
                    lib_path = "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1"
                if EspeakWrapper.set_library:
                    EspeakWrapper.set_library(lib_path)
            except Exception:
                pass
            self._phonemize = phonemize
            self._backend_used = "espeak"
            try:
                import subprocess
                result = subprocess.run(["espeak-ng", "--version"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    self._backend_version = result.stdout.strip().split()[0]
            except Exception:
                pass
            return
        except (ImportError, RuntimeError):
            pass
        try:
            from phonemizer import phonemize
            self._phonemize = phonemize
            self._backend_used = "segments"
            return
        except ImportError:
            pass
        self._phonemize = None
        self._backend_used = "none"

    def _predict_impl(self, text: str, locale: str | None) -> tuple[str, BackendProvenance]:
        fallback = f"/[{text}]/"
        if self._phonemize is None:
            provenance = BackendProvenance(
                requested_backend="espeak", actual_backend="none",
                backend_version=self._backend_version, fallback_used=True, is_real_prediction=False,
            )
            return fallback, provenance
        lang = locale or self._language or "en-us"
        phonemize_lang = self._LOCALE_MAP.get(lang, lang.split("-")[0])
        try:
            if self._backend_used == "espeak":
                result = self._phonemize(
                    [text], language=phonemize_lang, backend="espeak",
                    preserve_punctuation=self._preserve_punctuation, with_stress=True,
                )
            else:
                result = self._phonemize(
                    [text], language=phonemize_lang, backend="segments",
                    preserve_punctuation=False, with_stress=False,
                )
            prediction = result.strip() if result else fallback
            is_real = prediction != fallback
            provenance = BackendProvenance(
                requested_backend="espeak", actual_backend=self._backend_used,
                backend_version=self._backend_version, fallback_used=not is_real, is_real_prediction=is_real,
            )
            return prediction, provenance
        except Exception:
            provenance = BackendProvenance(
                requested_backend="espeak", actual_backend=self._backend_used,
                backend_version=self._backend_version, fallback_used=True, is_real_prediction=False,
            )
            return fallback, provenance

    def predict(self, text: str, locale: str | None = None) -> PronunciationPrediction:
        import time
        start = time.perf_counter()
        prediction, provenance = self._predict_impl(text, locale)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return PronunciationPrediction(
            model_name=self.model_name, model_version=self._backend_used,
            prediction=prediction, locale=locale,
            latency_ms=round(elapsed_ms, 2), provenance=provenance,
        )

    def predict_batch(self, texts: list[str], locale: str | None = None) -> list:
        return [self.predict(text, locale) for text in texts]

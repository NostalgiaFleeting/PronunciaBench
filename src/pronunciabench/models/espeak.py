"""eSpeak-NG / phonemizer based G2P backend."""

from __future__ import annotations

import warnings

from pronunciabench.models.base import BaseG2PModel


class EspeakG2P(BaseG2PModel):
    """Rule-based G2P using eSpeak-NG via the phonemizer package.

    Falls back to the `segments` backend if eSpeak-NG is not installed.
    """

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
        self._backend_used = "unknown"
        self._init_backend(language or "en-us")

    def _init_backend(self, language: str) -> None:
        """Initialize the phonemizer backend, with fallbacks."""
        # Try eSpeak first
        try:
            from phonemizer.backend import ESpeakBackend
            from phonemizer import phonemize
            self._phonemize = phonemize
            self._backend_used = "espeak"
            return
        except (ImportError, RuntimeError):
            pass

        # Fallback to segments backend
        try:
            from phonemizer.backend import SegmentsBackend
            from phonemizer import phonemize
            self._phonemize = phonemize
            self._backend_used = "segments"
            return
        except ImportError:
            pass

        # Ultimate fallback
        self._phonemize = None
        self._backend_used = "none"

    def _predict_impl(self, text: str, locale: str | None) -> str:
        if self._phonemize is None:
            return f"/[{text}]/"

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
            return result.strip() if result else f"/[{text}]/"
        except Exception:
            return f"/[{text}]/"
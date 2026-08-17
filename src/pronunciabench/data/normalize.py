"""IPA and grapheme normalization utilities."""

from __future__ import annotations

import re
from functools import lru_cache


# IPA stress markers (primary ˈ, secondary ˌ)
_STRESS_MARKS = set("ˈˌ")

# Characters that are pure diacritics/stress markers, not phonemes
_DIACRITIC_ONLY = re.compile(r"[\u02CCA-\u02D5\u0300-\u036F]")

# Normalization rules for common IPA variants
_VARIANT_MAP: dict[str, str] = {
    # Alternative IPA symbols
    "Ɪ": "i",  # Latin capital I
    "Ʞ": "ɰ",  # Latin small turned m
    "Ⱥ": "ɐ",  # Latin small letter turned a
    "ƛ": "tɬ",  # ejective lateral
    # Tone/suprasegmental normalization
    "˥": "˧˥",  # high tone approximation
    "˨": "˩˨",  # low tone approximation
}


@lru_cache(maxsize=None)
def normalize_ipa(ipa: str) -> str:
    """Normalize an IPA string for comparison.

    Strips diacritics-only characters, applies variant normalization,
    and lowercases for consistent comparison.
    """
    if not ipa:
        return ""

    # Apply variant substitutions
    result = ipa
    for old, new in _VARIANT_MAP.items():
        result = result.replace(old, new)

    # Remove standalone diacritic-only characters (stress markers stay)
    cleaned: list[str] = []
    i = 0
    while i < len(result):
        ch = result[i]
        # Skip combining diacritics (U+0300–U+036F) unless they're stress markers
        if "\u0300" <= ch <= "\u036F" and ch not in _STRESS_MARKS:
            if cleaned and cleaned[-1] in _STRESS_MARKS:
                pass  # skip this combining mark
            else:
                i += 1
                continue
        cleaned.append(ch)
        i += 1

    normalized = "".join(cleaned)

    # Collapse multiple spaces
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized.lower()


@lru_cache(maxsize=None)
def extract_phonemes(ipa: str) -> list[str]:
    """Split an IPA string into individual phoneme tokens."""
    if not ipa:
        return []

    ipa = normalize_ipa(ipa)

    DIGRAPHS = {
        "tʃ", "dʒ", "ŋ", "ɲ", "ʃ", "ʒ", "ʂ", "ʐ",
        "ts", "dz", "tɬ", "dɮ", "tf", "df",
        "ɡʷ", "kʷ", "xʷ", "ɣʷ",
        "aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ",
        "ɪə", "eə", "ʊə",
        "ɐɪ", "ɐʊ",
        "ɡʲ", "kʲ", "xʲ",
    }

    phonemes: list[str] = []
    i = 0
    while i < len(ipa):
        matched = False
        for length in (3, 2):
            if i + length <= len(ipa):
                sub = ipa[i : i + length]
                if sub in DIGRAPHS:
                    phonemes.append(sub)
                    i += length
                    matched = True
                    break
            if matched:
                break

        if not matched:
            ch = ipa[i]
            if ch == " ":
                i += 1
                continue
            if ch in _STRESS_MARKS and phonemes:
                phonemes[-1] += ch
                i += 1
                continue
            if "\u0300" <= ch <= "\u036F":
                i += 1
                continue
            phonemes.append(ch)
            i += 1

    return phonemes


def phoneme_edit_distance(
    ref_phonemes: list[str], hyp_phonemes: list[str]
) -> tuple[int, int, int, int]:
    """Compute phoneme-level edit distance (Levenshtein).

    Returns (substitutions, deletions, insertions, total_distance).
    """
    m, n = len(ref_phonemes), len(hyp_phonemes)
    # dp[i][j] = (subs, dels, ins, dist)
    dp: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, j) for j in range(n + 1)] for _ in range(m + 1)
    ]
    for i in range(m + 1):
        dp[i][0] = (0, i, 0, i)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_phonemes[i - 1] == hyp_phonemes[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = (dp[i - 1][j - 1][0] + 1,
                       dp[i - 1][j - 1][1],
                       dp[i - 1][j - 1][2],
                       dp[i - 1][j - 1][3] + 1)
                dels = (dp[i - 1][j][0],
                        dp[i - 1][j][1] + 1,
                        dp[i - 1][j][2],
                        dp[i - 1][j][3] + 1)
                ins = (dp[i][j - 1][0],
                       dp[i][j - 1][1],
                       dp[i][j - 1][2] + 1,
                       dp[i][j - 1][3] + 1)
                dp[i][j] = min(sub, dels, ins, key=lambda x: x[3])

    return dp[m][n]


def phoneme_error_rate(
    ref: str, hyp: str, normalize: bool = True
) -> float:
    """Compute Phoneme Error Rate (PER)."""
    if normalize:
        ref = normalize_ipa(ref)
        hyp = normalize_ipa(hyp)

    ref_phones = extract_phonemes(ref)
    hyp_phones = extract_phonemes(hyp)

    if not ref_phones:
        return 0.0 if not hyp_phones else 1.0

    _, _, _, distance = phoneme_edit_distance(ref_phones, hyp_phones)
    return distance / len(ref_phones)


def character_error_rate(ref: str, hyp: str) -> float:
    """Compute character-level error rate."""
    if not ref:
        return 0.0 if not hyp else 1.0
    import Levenshtein
    distance = Levenshtein.distance(ref, hyp)
    return distance / len(ref)


def exact_match_accuracy(predictions: list[tuple[str, str]]) -> float:
    """Compute exact match accuracy for (ref, hyp) pairs."""
    if not predictions:
        return 0.0
    matches = sum(
        1 for ref, hyp in predictions if normalize_ipa(ref) == normalize_ipa(hyp)
    )
    return matches / len(predictions)
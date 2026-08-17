"""Tests for phoneme normalization utilities."""

from __future__ import annotations

import pytest

from pronunciabench.data.normalize import (
    character_error_rate,
    exact_match_accuracy,
    extract_phonemes,
    normalize_ipa,
    phoneme_edit_distance,
    phoneme_error_rate,
)


class TestNormalizeIPA:
    def test_empty_string(self):
        assert normalize_ipa("") == ""

    def test_basic_ipa(self):
        assert normalize_ipa("/smɪθ/") == "/smɪθ/"

    def test_stress_markers_preserved(self):
        result = normalize_ipa("/ˈmʊlɐ/")
        assert "ˈ" in result

    def test_diacritics_stripped(self):
        # Combining diacritics should be removed
        result = normalize_ipa("a\u0300b")  # a + combining grave
        assert "\u0300" not in result

    def test_case_insensitive(self):
        assert normalize_ipa("ABC") == "abc"

    def test_whitespace_collapsed(self):
        assert normalize_ipa("a  b   c") == "a b c"


class TestExtractPhonemes:
    def test_single_phonemes(self):
        assert extract_phonemes("ab") == ["a", "b"]

    def test_digraphs(self):
        phones = extract_phonemes("tʃ dʒ ŋ")
        assert "tʃ" in phones
        assert "dʒ" in phones
        assert "ŋ" in phones

    def test_empty_string(self):
        assert extract_phonemes("") == []

    def test_stress_attached(self):
        phones = extract_phonemes("mˈʊlɐ")
        assert any("ˈ" in p for p in phones)

    def test_complex_ipa(self):
        phones = extract_phonemes("/smɪθ/")
        assert "s" in phones
        assert "m" in phones
        assert "ɪ" in phones
        assert "θ" in phones


class TestPhonemeEditDistance:
    def test_identical(self):
        subs, dels, ins, dist = phoneme_edit_distance(["a", "b"], ["a", "b"])
        assert dist == 0
        assert subs == 0 and dels == 0 and ins == 0

    def test_substitution(self):
        _, _, _, dist = phoneme_edit_distance(["a", "b"], ["a", "c"])
        assert dist == 1

    def test_insertion(self):
        _, _, _, dist = phoneme_edit_distance(["a"], ["a", "b"])
        assert dist == 1

    def test_deletion(self):
        _, _, _, dist = phoneme_edit_distance(["a", "b"], ["a"])
        assert dist == 1

    def test_empty_ref(self):
        _, _, _, dist = phoneme_edit_distance([], ["a", "b"])
        assert dist == 2


class TestPhonemeErrorRate:
    def test_perfect_match(self):
        assert phoneme_error_rate("/smɪθ/", "/smɪθ/") == 0.0

    def test_completely_different(self):
        per = phoneme_error_rate("a", "b")
        assert per > 0.0

    def test_empty_reference(self):
        assert phoneme_error_rate("", "") == 0.0
        assert phoneme_error_rate("", "a") == 1.0

    def test_normalization_applied(self):
        # After stripping slashes, phoneme sequences should match
        per = phoneme_error_rate("smɪθ", "smɪθ")
        assert per == 0.0


class TestCharacterErrorRate:
    def test_identical(self):
        assert character_error_rate("abc", "abc") == 0.0

    def test_single_change(self):
        cer = character_error_rate("abc", "abd")
        assert cer == 1 / 3

    def test_empty(self):
        assert character_error_rate("", "") == 0.0
        assert character_error_rate("", "a") == 1.0


class TestExactMatchAccuracy:
    def test_all_match(self):
        pairs = [("/smɪθ/", "/smɪθ/"), ("/ɡɑɹsi.a/", "/ɡɑɹsi.a/")]
        assert exact_match_accuracy(pairs) == 1.0

    def test_none_match(self):
        pairs = [("/smɪθ/", "/XXXX/"), ("/abc/", "/def/")]
        assert exact_match_accuracy(pairs) == 0.0

    def test_empty(self):
        assert exact_match_accuracy([]) == 0.0

    def test_partial_match(self):
        pairs = [("/smɪθ/", "/smɪθ/"), ("/abc/", "/def/")]
        assert exact_match_accuracy(pairs) == 0.5
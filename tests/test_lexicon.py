"""Tests for the shared tokenizer and noise filter."""

from scripts.lexicon import is_noise, is_phrase_noise, tokenize


def test_tokenize_lowercases_and_splits_on_punctuation() -> None:
    assert tokenize("DNA-binding ATPase; cytoplasm [GO:0005737]") == [
        "dna",
        "binding",
        "atpase",
        "cytoplasm",
        "go",
        "0005737",
    ]


def test_tokenize_keeps_internal_apostrophes() -> None:
    assert tokenize("5'-end and 3' end") == ["5", "end", "and", "3", "end"]


def test_single_characters_are_noise() -> None:
    # "C atoms", "L-alanine", "glycoprotein L", subunit "S5" -> "s5" is
    # handled by the two-character rule, the isolated letters by length.
    for token in ("a", "c", "d", "l", "n", "o", "s", "x", "z"):
        assert is_noise(token) is True


def test_numbers_and_roman_numerals_are_noise() -> None:
    assert is_noise("0000255") is True
    assert is_noise("ii") is True
    assert is_noise("iii") is True
    assert is_noise("iv") is True


def test_notation_fragments_are_noise() -> None:
    for token in ("n6", "c1", "9z", "2c", "hj", "nf", "er", "s5", "l1"):
        assert is_noise(token) is True


def test_short_whitelist_tokens_are_kept() -> None:
    for token in ("fe", "zn", "ca", "atp", "e3", "h3", "b12", "16s", "70s"):
        assert is_noise(token) is False


def test_prose_and_fragments_are_noise() -> None:
    for token in ("via", "one", "two", "non", "pre", "de", "end", "key", "go"):
        assert is_noise(token) is True


def test_real_keywords_are_kept() -> None:
    for token in (
        "binding",
        "catalyzes",
        "cytoplasm",
        "kinase",
        "membrane",
        "translation",
        "ubiquitin",
    ):
        assert is_noise(token) is False


def test_phrase_filter_keeps_generic_biology_words() -> None:
    # Inside a concept these words carry the distinction; dropping them
    # would turn "identical protein binding" into "identical_binding".
    assert is_phrase_noise("protein") is False
    assert is_phrase_noise("activity") is False
    assert is_phrase_noise("subunit") is False
    assert is_phrase_noise("large") is False


def test_phrase_filter_still_drops_structural_debris() -> None:
    for token in ("of", "the", "via", "one", "c", "n6", "ii"):
        assert is_phrase_noise(token) is True

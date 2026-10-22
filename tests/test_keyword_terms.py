"""Tests for the keyword-terms helpers (GO parsing + keep filter)."""

from protein_database.keywords import (
    build_phrase_index,
    find_phrases,
    keep_term,
    parse_go_cell,
    phrase_key,
)


def test_parse_go_cell_single() -> None:
    assert parse_go_cell("ATP binding [GO:0005524]") == [
        ("ATP binding", "GO:0005524")
    ]


def test_parse_go_cell_multiple() -> None:
    assert parse_go_cell("cytoplasm [GO:0005737]; ATP binding [GO:0005524]") == [
        ("cytoplasm", "GO:0005737"),
        ("ATP binding", "GO:0005524"),
    ]


def test_parse_go_cell_empty_produces_nothing() -> None:
    assert parse_go_cell(None) == []
    assert parse_go_cell("") == []
    assert parse_go_cell("   ") == []


def test_parse_go_cell_skips_non_matching_segments() -> None:
    assert parse_go_cell("ATP binding [GO:0005524]; junk without id") == [
        ("ATP binding", "GO:0005524")
    ]


def test_keep_boundaries_are_inclusive() -> None:
    assert keep_term(5, 100) is True
    assert keep_term(95, 100) is True
    assert keep_term(4, 100) is False
    assert keep_term(96, 100) is False
    assert keep_term(50, 100) is True


def test_keep_term_rejects_empty_table() -> None:
    assert keep_term(0, 0) is False


def test_build_phrase_index_skips_single_token_terms() -> None:
    index = build_phrase_index(
        [("plasma", "membrane"), ("cytoplasm",), ("atp", "binding")]
    )
    assert set(index) == {"plasma", "atp"}
    assert index["plasma"] == (("plasma", "membrane"),)


def test_build_phrase_index_prefers_longest_match_first() -> None:
    index = build_phrase_index(
        [("atp", "binding"), ("atp", "binding", "cassette")]
    )
    assert index["atp"][0] == ("atp", "binding", "cassette")


def test_find_phrases_matches_contiguous_tokens() -> None:
    index = build_phrase_index(
        [("plasma", "membrane"), ("atp", "binding"), ("structural", "constituent")]
    )
    found = find_phrases(
        ["the", "plasma", "membrane", "contains", "atp", "binding", "sites"],
        index,
    )
    assert found == {"plasma_membrane", "atp_binding"}


def test_find_phrases_reports_overlapping_phrases() -> None:
    index = build_phrase_index(
        [("atp", "binding"), ("atp", "binding", "cassette")]
    )
    assert find_phrases(["atp", "binding", "cassette"], index) == {
        "atp_binding",
        "atp_binding_cassette",
    }


def test_find_phrases_requires_contiguity_and_order() -> None:
    index = build_phrase_index([("plasma", "membrane")])
    assert find_phrases(["membrane", "plasma"], index) == set()
    assert find_phrases(["plasma", "and", "membrane"], index) == set()


def test_phrase_key_joins_with_underscore() -> None:
    assert phrase_key(["structural", "constituent", "ribosome"]) == (
        "structural_constituent_ribosome"
    )

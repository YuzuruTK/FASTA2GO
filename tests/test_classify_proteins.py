"""Tests for the protein classification logic in scripts/classify_proteins.py.

These tests exercise the pure scoring/assignment functions against a small
synthetic label dictionary, so no database is required.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.classify_proteins import (
    assigned_classes,
    load_labels,
    score_description,
    top_class,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SYNTHETIC_LABELS = [
    # word, family, class, weight, strength, reason
    ("catalyzes", "catalysis", "enzyme", 1.0, "strong", "catalysis"),
    ("kinase", "ec_class", "enzyme", 1.0, "strong", "EC family"),
    ("binding", "general_binding", "enzyme", 0.2, "weak", "generic"),
    ("transporter", "transport", "transporter", 1.0, "strong", "transport"),
    ("membrane", "transport", "transporter", 0.3, "weak", "membrane"),
    ("receptor", "receptor", "receptor_signaling", 1.0, "strong", "receptor"),
    ("phosphorylation", "reaction_chemistry", "enzyme", 0.6, "medium", "rxn"),
    ("phosphorylation", "phosphorylation", "receptor_signaling", 0.5,
     "medium", "signaling"),
]


@pytest.fixture()
def labels(tmp_path: Path) -> dict[str, list[tuple[str, float]]]:
    path = tmp_path / "functional_labels.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["word", "family", "class", "weight", "strength", "reason"]
        )
        writer.writerows(SYNTHETIC_LABELS)
    return load_labels(path)


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------


class TestLoadLabels:
    def test_loads_words_with_class_weights(self, labels):
        assert labels["kinase"] == [("enzyme", 1.0)]

    def test_multi_class_words_have_multiple_entries(self, labels):
        assert sorted(labels["phosphorylation"]) == [
            ("enzyme", 0.6),
            ("receptor_signaling", 0.5),
        ]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_labels(tmp_path / "nope.csv")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def test_single_strong_word_reaches_threshold(self, labels):
        scores, matched = score_description(
            ["this", "protein", "catalyzes", "the", "reaction"], labels
        )
        assert scores["enzyme"] == pytest.approx(1.0)
        assert matched == ["catalyzes"]

    def test_weights_accumulate_per_class(self, labels):
        scores, _ = score_description(["kinase", "binding"], labels)
        assert scores["enzyme"] == pytest.approx(1.2)

    def test_multi_class_word_scores_both_classes(self, labels):
        scores, _ = score_description(["phosphorylation"], labels)
        assert scores["enzyme"] == pytest.approx(0.6)
        assert scores["receptor_signaling"] == pytest.approx(0.5)

    def test_repeated_word_counts_once(self, labels):
        scores, _ = score_description(["kinase", "kinase", "kinase"], labels)
        assert scores["enzyme"] == pytest.approx(1.0)

    def test_no_match_gives_empty_scores(self, labels):
        scores, matched = score_description(["totally", "unrelated"], labels)
        assert scores == {}
        assert matched == []


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------


class TestAssignment:
    def test_weak_word_alone_does_not_classify(self, labels):
        """A lone 'membrane' (0.3) must not classify at threshold 1.0."""
        scores, _ = score_description(["membrane"], labels)
        assert assigned_classes(scores, threshold=1.0) == []

    def test_weak_word_classifies_at_lower_threshold(self, labels):
        scores, _ = score_description(["membrane"], labels)
        assert assigned_classes(scores, threshold=0.3) == ["transporter"]

    def test_multi_label_assignment(self, labels):
        scores, _ = score_description(["kinase", "receptor"], labels)
        assert assigned_classes(scores, threshold=1.0) == [
            "enzyme",
            "receptor_signaling",
        ]

    def test_threshold_is_inclusive(self, labels):
        scores, _ = score_description(["catalyzes"], labels)
        assert assigned_classes(scores, threshold=1.0) == ["enzyme"]

    def test_top_class_picks_highest_score(self, labels):
        scores, _ = score_description(["kinase", "binding"], labels)
        assert top_class(scores) == "enzyme"

    def test_top_class_tie_breaks_alphabetically(self, labels):
        scores = {"enzyme": 1.0, "transporter": 1.0}
        assert top_class(scores) == "enzyme"

    def test_top_class_empty_scores(self, labels):
        assert top_class({}) is None
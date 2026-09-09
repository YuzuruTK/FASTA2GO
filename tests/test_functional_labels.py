"""Tests for the functional-keyword classification rules.

These tests exercise the pure rule-matching logic of
``scripts/build_functional_labels.py`` against a small synthetic
vocabulary, so no database or generated CSV is required.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.build_functional_labels import (
    CLASSES,
    build_discard_rows,
    build_label_rows,
    load_vocabulary,
    sort_label_rows,
    strength_for,
    validate,
    write_labels_csv,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SYNTHETIC_VOCAB = [
    # strong functional words
    "catalyzes", "kinase", "transporter", "transmembrane", "polymerase",
    "helicase", "receptor", "signaling", "structural", "motor",
    "antimicrobial",
    # ambiguous words
    "atp", "binding", "membrane",
    # discard candidates
    "cell", "probably", "alanine", "l",
    # word with no rules at all
    "zzz_unknown_word",
]


@pytest.fixture()
def vocab(tmp_path: Path) -> dict[str, int]:
    """Write a synthetic vocabulary CSV and load it."""
    path = tmp_path / "word_frequencies.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "word", "proteins", "total"])
        for rank, word in enumerate(SYNTHETIC_VOCAB, 1):
            writer.writerow([rank, word, 100 - rank, 200 - rank])
    return load_vocabulary(path)


@pytest.fixture()
def label_rows(vocab: dict[str, int]) -> list[dict]:
    rows, _ = build_label_rows(vocab)
    return sort_label_rows(rows)


def labels_for(rows: list[dict], word: str) -> list[dict]:
    """Return all label rows for a given word."""
    return [row for row in rows if row["word"] == word]


def discarded_words(vocab: dict[str, int]) -> set[str]:
    return {row["word"] for row in build_discard_rows(vocab, set())}


# ---------------------------------------------------------------------------
# Classification rules
# ---------------------------------------------------------------------------


class TestStrongClassifications:
    def test_catalyzes_is_strong_enzyme(self, label_rows):
        rows = labels_for(label_rows, "catalyzes")
        assert any(
            r["class"] == "enzyme" and r["strength"] == "strong" for r in rows
        )

    def test_kinase_is_strong_enzyme(self, label_rows):
        rows = labels_for(label_rows, "kinase")
        assert any(
            r["class"] == "enzyme" and r["strength"] == "strong" for r in rows
        )

    def test_transporter_is_strong_transporter(self, label_rows):
        rows = labels_for(label_rows, "transporter")
        assert any(
            r["class"] == "transporter" and r["strength"] == "strong"
            for r in rows
        )

    def test_transmembrane_is_weaker_transporter(self, label_rows):
        """transmembrane indicates membrane association, not transport."""
        transmembrane = [
            float(r["weight"])
            for r in labels_for(label_rows, "transmembrane")
            if r["class"] == "transporter"
        ]
        transporter = [
            float(r["weight"])
            for r in labels_for(label_rows, "transporter")
            if r["class"] == "transporter"
        ]
        assert transmembrane
        assert max(transmembrane) < max(transporter)

    def test_polymerase_is_strong_dna_rna_binding(self, label_rows):
        rows = labels_for(label_rows, "polymerase")
        assert any(
            r["class"] == "dna_rna_binding" and r["strength"] == "strong"
            for r in rows
        )

    def test_helicase_is_strong_dna_rna_binding(self, label_rows):
        rows = labels_for(label_rows, "helicase")
        assert any(
            r["class"] == "dna_rna_binding" and r["strength"] == "strong"
            for r in rows
        )

    def test_receptor_is_strong_receptor_signaling(self, label_rows):
        rows = labels_for(label_rows, "receptor")
        assert any(
            r["class"] == "receptor_signaling" and r["strength"] == "strong"
            for r in rows
        )

    def test_signaling_maps_to_receptor_signaling(self, label_rows):
        rows = labels_for(label_rows, "signaling")
        assert any(r["class"] == "receptor_signaling" for r in rows)

    def test_structural_maps_to_structural(self, label_rows):
        rows = labels_for(label_rows, "structural")
        assert any(r["class"] == "structural" for r in rows)

    def test_motor_maps_to_motor(self, label_rows):
        rows = labels_for(label_rows, "motor")
        assert any(r["class"] == "motor" for r in rows)

    def test_antimicrobial_maps_to_defense(self, label_rows):
        rows = labels_for(label_rows, "antimicrobial")
        assert any(r["class"] == "defense" for r in rows)


class TestAmbiguousWords:
    def test_atp_is_not_definitive_enzyme(self, label_rows):
        """ATP must never be a weight-1.0 enzyme indicator."""
        enzyme_weights = [
            float(r["weight"])
            for r in labels_for(label_rows, "atp")
            if r["class"] == "enzyme"
        ]
        assert enzyme_weights
        assert max(enzyme_weights) <= 0.3

    def test_binding_is_not_a_definitive_class(self, label_rows):
        """Generic binding must only carry low weights."""
        rows = labels_for(label_rows, "binding")
        assert rows
        assert all(float(r["weight"]) <= 0.2 for r in rows)
        assert all(r["strength"] == "weak" for r in rows)

    def test_membrane_is_weak_transporter_evidence(self, label_rows):
        rows = [
            r
            for r in labels_for(label_rows, "membrane")
            if r["class"] == "transporter"
        ]
        assert rows
        assert all(float(r["weight"]) <= 0.4 for r in rows)


class TestDiscards:
    @pytest.mark.parametrize("word", ["cell", "probably", "alanine"])
    def test_generic_words_are_discarded(self, vocab, word):
        assert word in discarded_words(vocab)

    def test_single_letters_are_discarded(self, vocab):
        assert "l" in discarded_words(vocab)

    def test_discarded_words_are_not_labeled(self, vocab):
        rows, labeled = build_label_rows(vocab)
        discards = {r["word"] for r in build_discard_rows(vocab, labeled)}
        assert not (discards & labeled)


class TestVocabularyFiltering:
    def test_absent_words_are_not_emitted(self, label_rows):
        """Words not present in the input vocabulary get no output."""
        emitted = {r["word"] for r in label_rows}
        assert "ferritin" not in emitted
        assert "zzz_unknown_word" not in emitted

    def test_only_vocabulary_words_are_emitted(self, label_rows, vocab):
        assert all(r["word"] in vocab for r in label_rows)


class TestOutputInvariants:
    def test_no_duplicate_rows(self, label_rows):
        keys = [(r["word"], r["family"], r["class"]) for r in label_rows]
        assert len(keys) == len(set(keys))

    def test_all_classes_are_known(self, label_rows):
        assert all(r["class"] in CLASSES for r in label_rows)

    def test_weights_in_range(self, label_rows):
        for row in label_rows:
            assert 0.0 < float(row["weight"]) <= 1.0

    def test_strength_matches_weight(self, label_rows):
        for row in label_rows:
            assert row["strength"] == strength_for(float(row["weight"]))

    def test_sort_order(self, label_rows):
        keys = [
            (r["class"], -float(r["weight"]), r["family"], r["word"])
            for r in label_rows
        ]
        assert keys == sorted(keys)

    def test_validate_passes_on_generated_rows(self, label_rows):
        assert validate(label_rows, vocab_size=1000) == []

    def test_written_csv_has_expected_header(self, tmp_path, label_rows):
        path = tmp_path / "functional_labels.csv"
        write_labels_csv(label_rows, path)
        with path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        assert header == ["word", "family", "class", "weight", "strength",
                          "reason"]
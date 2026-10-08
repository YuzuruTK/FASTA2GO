"""Tests for the amino acid composition function."""

import pytest

from protein_database.features import composition
from protein_database.schema import AMINO_ACIDS, COMPOSITION_COLUMNS


def test_columns_cover_all_amino_acids() -> None:
    assert len(AMINO_ACIDS) == 20
    assert len(COMPOSITION_COLUMNS) == 20
    assert set(AMINO_ACIDS) == set("ACDEFGHIKLMNPQRSTVWY")


def test_composition_all_alanine() -> None:
    result = composition("AAAAAAAAAA")
    assert result["alanine_percentage"] == pytest.approx(100.0)
    assert result["arginine_percentage"] == pytest.approx(0.0)
    assert result["valine_percentage"] == pytest.approx(0.0)


def test_composition_mixed() -> None:
    # 5 A, 3 C, 2 G in a 10-residue sequence
    result = composition("AAAAACCCGG")
    assert result["alanine_percentage"] == pytest.approx(50.0)
    assert result["cysteine_percentage"] == pytest.approx(30.0)
    assert result["glycine_percentage"] == pytest.approx(20.0)


def test_composition_sums_to_100() -> None:
    result = composition("ACDEFGHIKLMNPQRSTVWY")
    assert sum(result.values()) == pytest.approx(100.0)


def test_composition_handles_lowercase() -> None:
    assert composition("aaaa")["alanine_percentage"] == pytest.approx(100.0)


def test_composition_empty_raises() -> None:
    with pytest.raises(ValueError):
        composition("")


def test_composition_no_standard_aa_raises() -> None:
    with pytest.raises(ValueError):
        composition("BZX")

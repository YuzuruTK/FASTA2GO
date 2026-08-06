"""Tests for the amino acid composition module."""

import pytest

from protein_database.features.composition import (
    AMINO_ACID_FIELD_MAP,
    compute_amino_acid_composition,
    count_amino_acids,
)
from protein_database.features.sequence_features import STANDARD_AMINO_ACIDS


def test_field_map_covers_all_amino_acids() -> None:
    """Test that the field map covers all 20 standard amino acids."""
    assert len(AMINO_ACID_FIELD_MAP) == 20
    assert set(AMINO_ACID_FIELD_MAP.keys()) == set(STANDARD_AMINO_ACIDS)


def test_composition_all_alanine() -> None:
    """Test composition when sequence is all alanine."""
    composition = compute_amino_acid_composition("AAAAAAAAAA")
    assert composition.alanine_percentage == pytest.approx(100.0)
    assert composition.arginine_percentage == pytest.approx(0.0)
    assert composition.valine_percentage == pytest.approx(0.0)


def test_composition_mixed() -> None:
    """Test composition with a known sequence."""
    # 5 A, 3 C, 2 G in a 10-residue sequence
    sequence = "AAAAACCCGG"
    composition = compute_amino_acid_composition(sequence)
    assert composition.alanine_percentage == pytest.approx(50.0)
    assert composition.cysteine_percentage == pytest.approx(30.0)
    assert composition.glycine_percentage == pytest.approx(20.0)


def test_composition_sums_to_100() -> None:
    """Test that all composition percentages sum to 100."""
    # Hemoglobin subunit alpha (141 aa) - well-known protein
    sequence = (
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF"
        "DLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRV"
        "DPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
    )
    composition = compute_amino_acid_composition(sequence)
    total = sum(
        getattr(composition, field) for field in AMINO_ACID_FIELD_MAP.values()
    )
    assert abs(total - 100.0) < 0.5  # Allow for rounding


def test_count_amino_acids() -> None:
    """Test the amino acid counting function."""
    counts = count_amino_acids("ACDEFGHIKLMNPQRSTVWY")
    assert len(counts) == 20
    assert all(v == 1 for v in counts.values())

    counts = count_amino_acids("AAAA")
    assert counts["A"] == 4
    assert counts["C"] == 0


def test_composition_handles_lowercase() -> None:
    """Test that composition normalizes lowercase input."""
    composition = compute_amino_acid_composition("aaaa")
    assert composition.alanine_percentage == pytest.approx(100.0)


def test_composition_empty_raises() -> None:
    """Test that empty sequence raises ValueError."""
    with pytest.raises(ValueError):
        compute_amino_acid_composition("")


def test_composition_no_standard_aa_raises() -> None:
    """Test that sequence with no standard amino acids raises ValueError."""
    with pytest.raises(ValueError):
        compute_amino_acid_composition("BZX")
"""Tests for the sequence features module."""

import pytest

from protein_database.features.sequence_features import (
    HYDROPHOBIC_RESIDUES,
    NEGATIVE_RESIDUES,
    POLAR_RESIDUES,
    POSITIVE_RESIDUES,
    STANDARD_AMINO_ACIDS,
    compute_sequence_features,
)


# Hemoglobin subunit alpha (141 aa) - well-known protein
HBA_SEQUENCE = (
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF"
    "DLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRV"
    "DPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
)


def test_standard_amino_acids() -> None:
    """Test that all 20 standard amino acids are present."""
    assert len(STANDARD_AMINO_ACIDS) == 20
    assert set(STANDARD_AMINO_ACIDS) == set("ACDEFGHIKLMNPQRSTVWY")


def test_chemical_classifications() -> None:
    """Test that chemical classifications are non-overlapping and complete."""
    # Hydrophobic: A, V, I, L, M, F, W, Y
    assert HYDROPHOBIC_RESIDUES == set("AVILMFWY")
    # Polar: S, T, N, Q, C
    assert POLAR_RESIDUES == set("STNCQ")
    # Positive: K, R, H
    assert POSITIVE_RESIDUES == set("KRH")
    # Negative: D, E
    assert NEGATIVE_RESIDUES == set("DE")


def test_compute_sequence_features_hba() -> None:
    """Test feature computation for hemoglobin alpha."""
    features = compute_sequence_features(HBA_SEQUENCE)

    # Molecular weight should be reasonable for a 141 aa protein
    assert 15000 < features.molecular_weight < 16000

    # Isoelectric point should be in a reasonable range
    assert 7.0 < features.isoelectric_point < 10.0

    # GRAVY should be near zero for hemoglobin alpha (slightly amphipathic)
    assert -0.2 < features.gravy_index < 0.2

    # Instability index should be a positive number
    assert features.instability_index > 0

    # Aliphatic index should be positive
    assert features.aliphatic_index > 0

    # Aromaticity should be between 0 and 1
    assert 0 < features.aromaticity < 1

    # Percentages should sum to ~100 for the main groups.
    # Note: Glycine (G) and Proline (P) are not in any chemical class,
    # so the sum is slightly less than 100.
    total = (
        features.hydrophobic_percentage
        + features.polar_percentage
        + features.charged_percentage
    )
    assert 80.0 < total < 100.0

    # Shannon entropy should be between 0 and log2(20) ≈ 4.32
    assert 0 < features.sequence_entropy <= 4.32

    # Amino acid diversity should be between 0 and 1
    assert 0 < features.amino_acid_diversity <= 1.0


def test_compute_sequence_features_single_amino_acid() -> None:
    """Test feature computation for a homopolymer sequence."""
    features = compute_sequence_features("AAAAAAAAAA")

    # All alanine: hydrophobic 100%, polar 0%, charged 0%
    assert features.hydrophobic_percentage == pytest.approx(100.0)
    assert features.polar_percentage == pytest.approx(0.0)
    assert features.charged_percentage == pytest.approx(0.0)

    # Shannon entropy should be 0 for a uniform sequence
    assert features.sequence_entropy == pytest.approx(0.0)

    # Diversity should be 1/20
    assert features.amino_acid_diversity == pytest.approx(1 / 20)


def test_compute_sequence_features_empty_raises() -> None:
    """Test that empty sequence raises ValueError."""
    with pytest.raises(ValueError):
        compute_sequence_features("")


def test_compute_sequence_features_no_standard_aa_raises() -> None:
    """Test that sequence with no standard amino acids raises ValueError."""
    with pytest.raises(ValueError):
        compute_sequence_features("XXXXX")
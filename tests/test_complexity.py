"""Tests for the sequence complexity module."""

import math

import pytest

from protein_database.features.complexity import (
    amino_acid_diversity,
    compute_complexity,
    count_repeated_amino_acids,
    find_low_complexity_regions,
    max_homopolymer_length,
    shannon_entropy,
)


def test_shannon_entropy_uniform() -> None:
    """Test that a uniform sequence has zero entropy."""
    assert shannon_entropy("AAAAAAAAAA") == pytest.approx(0.0)


def test_shannon_entropy_maximum() -> None:
    """Test that a maximally diverse sequence has high entropy."""
    # All 20 amino acids, each appearing once
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    entropy = shannon_entropy(sequence)
    # Max entropy for 20 equally likely symbols is log2(20)
    assert entropy == pytest.approx(math.log2(20))


def test_shannon_entropy_empty() -> None:
    """Test that empty sequence has zero entropy."""
    assert shannon_entropy("") == 0.0


def test_amino_acid_diversity() -> None:
    """Test amino acid diversity calculation."""
    assert amino_acid_diversity("ACDEFGHIKLMNPQRSTVWY") == pytest.approx(1.0)
    assert amino_acid_diversity("AAAAAAAAAA") == pytest.approx(1 / 20)
    assert amino_acid_diversity("") == 0.0


def test_count_repeated_amino_acids() -> None:
    """Test counting repeated amino acids."""
    # A appears 3 times, C appears 2 times, G appears 1 time
    assert count_repeated_amino_acids("AAACCG") == 2
    # All unique
    assert count_repeated_amino_acids("ACDEFG") == 0
    # All same
    assert count_repeated_amino_acids("AAAA") == 1


def test_find_low_complexity_regions() -> None:
    """Test low-complexity region detection."""
    # One run of 4+ identical residues
    assert find_low_complexity_regions("AAAACDEFG") == 1
    # Two runs of 4+ identical residues
    assert find_low_complexity_regions("AAAACCCC") == 2
    # No runs of 4+
    assert find_low_complexity_regions("ACDEFGHIK") == 0
    # Short sequence
    assert find_low_complexity_regions("AAA") == 0


def test_max_homopolymer_length() -> None:
    """Test finding the longest homopolymer run."""
    assert max_homopolymer_length("AAAACCCC") == 4
    assert max_homopolymer_length("ACDEFG") == 1
    assert max_homopolymer_length("") == 0
    assert max_homopolymer_length("AAAAA") == 5


def test_compute_complexity() -> None:
    """Test the full complexity computation."""
    result = compute_complexity("AAAACDEFG")
    assert result.shannon_entropy > 0
    assert result.amino_acid_diversity > 0
    assert result.repeated_amino_acid_count >= 1
    assert result.low_complexity_region_count == 1
    assert result.max_homopolymer_length == 4
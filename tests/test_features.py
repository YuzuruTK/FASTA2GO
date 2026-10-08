"""Tests for the sequence features module."""

import pytest

from protein_database.features import (
    HYDROPHOBIC_RESIDUES,
    NEGATIVE_RESIDUES,
    POLAR_RESIDUES,
    POSITIVE_RESIDUES,
    STANDARD_AMINO_ACIDS,
    sequence_features,
    shannon_entropy,
)

HBA_SEQUENCE = (
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF"
    "DLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRV"
    "DPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
)


def test_standard_amino_acids() -> None:
    assert sorted(STANDARD_AMINO_ACIDS) == sorted("ACDEFGHIKLMNPQRSTVWY")


def test_chemical_classifications() -> None:
    assert HYDROPHOBIC_RESIDUES == set("AVILMFWY")
    assert POLAR_RESIDUES == set("STNCQ")
    assert POSITIVE_RESIDUES == set("KRH")
    assert NEGATIVE_RESIDUES == set("DE")


def test_shannon_entropy() -> None:
    assert shannon_entropy("AAAAAAAAAA") == pytest.approx(0.0)
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("ACDEFGHIKLMNPQRSTVWY" * 100) == pytest.approx(4.32, abs=0.01)


def test_sequence_features_hba() -> None:
    features = sequence_features(HBA_SEQUENCE)
    assert 15000 < features["molecular_weight"] < 16000
    assert 7.0 < features["isoelectric_point"] < 10.0
    assert -0.2 < features["gravy_index"] < 0.2
    assert features["instability_index"] > 0
    assert features["aliphatic_index"] > 0
    assert 0 < features["aromaticity"] < 1
    total = (
        features["hydrophobic_percentage"]
        + features["polar_percentage"]
        + features["charged_percentage"]
    )
    assert 80.0 < total < 100.0
    assert 0 < features["sequence_entropy"] <= 4.32
    assert 0 < features["amino_acid_diversity"] <= 1.0


def test_sequence_features_single_amino_acid() -> None:
    features = sequence_features("AAAAAAAAAA")
    assert features["hydrophobic_percentage"] == pytest.approx(100.0)
    assert features["polar_percentage"] == pytest.approx(0.0)
    assert features["charged_percentage"] == pytest.approx(0.0)
    assert features["sequence_entropy"] == pytest.approx(0.0)
    assert features["amino_acid_diversity"] == pytest.approx(1 / 20)


def test_sequence_features_empty_raises() -> None:
    with pytest.raises(ValueError):
        sequence_features("")


def test_sequence_features_no_standard_aa_raises() -> None:
    with pytest.raises(ValueError):
        sequence_features("XXXXX")

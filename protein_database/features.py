"""Physicochemical features and amino-acid composition from a sequence.

Both functions return plain dicts keyed by their schema column names, so the
column list lives only in ``schema.py``.
"""

import math
from collections import Counter

from Bio.SeqUtils import ProtParam
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint

from protein_database.schema import AMINO_ACIDS

#: Standard amino acid one-letter codes.
STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_STANDARD = set(STANDARD_AMINO_ACIDS)

#: Chemical classifications of amino acids.
HYDROPHOBIC_RESIDUES = set("AVILMFWY")
POLAR_RESIDUES = set("STNCQ")
POSITIVE_RESIDUES = set("KRH")
NEGATIVE_RESIDUES = set("DE")
SULFUR_RESIDUES = set("CM")
AROMATIC_RESIDUES = set("FYW")


def _clean(sequence: str) -> str:
    """Uppercase a sequence and drop non-standard residues (X, *, -, ...)."""
    return "".join(c for c in sequence.upper() if c in _STANDARD)


def shannon_entropy(sequence: str) -> float:
    """Shannon entropy of a sequence in bits (max ``log2(20)``)."""
    if not sequence:
        return 0.0
    length = len(sequence)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in Counter(sequence).values()
    )


def sequence_features(sequence: str) -> dict[str, float]:
    """Compute physicochemical features, keyed by schema column name.

    Args:
        sequence: Amino acid sequence.

    Returns:
        Mapping of feature column name -> value.

    Raises:
        ValueError: If the sequence contains no standard amino acids.
    """
    cleaned = _clean(sequence)
    if not cleaned:
        raise ValueError("Sequence contains no standard amino acids.")

    analysis = ProtParam.ProteinAnalysis(cleaned)
    counts = analysis.count_amino_acids()
    length = len(cleaned)
    percent = analysis.amino_acids_percent  # already 0-100

    def share(residues: set[str]) -> float:
        return sum(percent.get(aa, 0.0) for aa in residues)

    return {
        "molecular_weight": analysis.molecular_weight(),
        "isoelectric_point": IsoelectricPoint(cleaned).pi(),
        "gravy_index": analysis.gravy(),
        "instability_index": analysis.instability_index(),
        "aliphatic_index": (
            counts.get("A", 0)
            + 2.9 * counts.get("V", 0)
            + 3.9 * (counts.get("I", 0) + counts.get("L", 0))
        ) / length * 100.0,
        "aromaticity": analysis.aromaticity(),
        "charge_positive_percentage": share(POSITIVE_RESIDUES),
        "charge_negative_percentage": share(NEGATIVE_RESIDUES),
        "hydrophobic_percentage": share(HYDROPHOBIC_RESIDUES),
        "polar_percentage": share(POLAR_RESIDUES),
        "charged_percentage": share(POSITIVE_RESIDUES | NEGATIVE_RESIDUES),
        "sulfur_percentage": share(SULFUR_RESIDUES),
        "aromatic_percentage": share(AROMATIC_RESIDUES),
        "sequence_entropy": shannon_entropy(cleaned),
        "amino_acid_diversity": len(set(cleaned)) / len(_STANDARD),
    }


def composition(sequence: str) -> dict[str, float]:
    """Compute percentage of each amino acid, keyed by schema column name.

    Args:
        sequence: Amino acid sequence.

    Returns:
        Mapping of ``<amino_acid>_percentage`` column -> value.

    Raises:
        ValueError: If the sequence contains no standard amino acids.
    """
    counts = Counter(_clean(sequence))
    total = sum(counts.values())
    if not total:
        raise ValueError("Sequence contains no standard amino acids.")
    return {
        f"{AMINO_ACIDS[aa]}_percentage": 100.0 * counts.get(aa, 0) / total
        for aa in AMINO_ACIDS
    }

"""Sequence-derived characteristics.

Every denominator is explicit:

* AAC  -> percentage over the canonical residues (``canonical_length``), sums to 100.
* DPC  -> percentage over the pairs of canonically adjacent residues
          (``canonical_pairs``), sums to 100, and is undefined (NULL) when the
          sequence has no such pair.

Residues outside the 20 canonical amino acids are removed before the AAC and the
physicochemical formulas, because none of them is defined for those residues. The
DPC is the exception: it counts pairs on the original sequence, so a non-canonical
residue breaks adjacency instead of creating a new pair (``ACXDE`` has ``AC`` and
``DE``, never ``CD``). The number of removed residues and the DPC denominator are
stored next to the features, so the denominator of every stored percentage is
auditable.
"""

from __future__ import annotations

import math
from collections import Counter

from Bio.SeqUtils import ProtParam
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint

from fasta2go.fasta import CANONICAL_RESIDUES

#: Bump when a formula or a column definition changes; the pipeline then
#: recomputes every stored row instead of mixing definitions.
FEATURE_VERSION = 2

#: One-letter code -> column prefix of the composition table.
AMINO_ACIDS = {
    "A": "alanine",
    "R": "arginine",
    "N": "asparagine",
    "D": "aspartic_acid",
    "C": "cysteine",
    "Q": "glutamine",
    "E": "glutamic_acid",
    "G": "glycine",
    "H": "histidine",
    "I": "isoleucine",
    "L": "leucine",
    "K": "lysine",
    "M": "methionine",
    "F": "phenylalanine",
    "P": "proline",
    "S": "serine",
    "T": "threonine",
    "W": "tryptophan",
    "Y": "tyrosine",
    "V": "valine",
}

AAC_COLUMNS = [f"{name}_pct" for name in AMINO_ACIDS.values()]
DPC_COLUMNS = [f"{first}_{second}" for first in AMINO_ACIDS for second in AMINO_ACIDS]
FEATURE_COLUMNS = [
    "feature_version",
    "canonical_length",
    "noncanonical_residues",
    "canonical_pairs",
    "molecular_weight",
    "isoelectric_point",
    "gravy",
    "instability_index",
    "aliphatic_index",
    "aromaticity",
    "sequence_entropy",
]


def clean_sequence(sequence: str) -> str:
    """Return the canonical residues of a sequence, in order."""
    return "".join(char for char in sequence.upper() if char in CANONICAL_RESIDUES)


def amino_acid_composition(cleaned: str) -> dict[str, float | None]:
    """Percentage of each canonical amino acid, keyed by AAC column name."""
    total = len(cleaned)
    if not total:
        return dict.fromkeys(AAC_COLUMNS)
    counts = Counter(cleaned)
    return {
        f"{AMINO_ACIDS[residue]}_pct": 100.0 * counts[residue] / total
        for residue in AMINO_ACIDS
    }


def canonical_pairs(sequence: str) -> int:
    """Number of adjacent pairs in the original sequence with both residues canonical.

    This is the denominator of the DPC: a non-canonical residue breaks adjacency
    instead of disappearing and gluing its neighbours.
    """
    return sum(
        1
        for first, second in zip(sequence, sequence[1:])
        if first in CANONICAL_RESIDUES and second in CANONICAL_RESIDUES
    )


def dipeptide_composition(sequence: str) -> dict[str, float | None]:
    """Percentage of each adjacent residue pair, keyed by DPC column name.

    Pairs are counted on the original sequence and only when both residues are
    canonical, so ``ACXDE`` contributes ``AC`` and ``DE`` but never ``CD``; the
    denominator is :func:`canonical_pairs`. Every value is NULL when no such pair
    exists, because the percentage is undefined rather than zero.
    """
    counts = Counter(
        first + second
        for first, second in zip(sequence, sequence[1:])
        if first in CANONICAL_RESIDUES and second in CANONICAL_RESIDUES
    )
    total = sum(counts.values())
    if not total:
        return dict.fromkeys(DPC_COLUMNS)
    return {
        f"{first}_{second}": 100.0 * counts[first + second] / total
        for first in AMINO_ACIDS
        for second in AMINO_ACIDS
    }


def sequence_features(sequence: str) -> dict[str, float | int | None]:
    """Physicochemical properties of a sequence, keyed by column name.

    Raises:
        ValueError: If the sequence has no canonical residue, in which case no
            formula below is defined.
    """
    cleaned = clean_sequence(sequence)
    if not cleaned:
        raise ValueError("sequence has no canonical amino acid")

    analysis = ProtParam.ProteinAnalysis(cleaned)
    counts = analysis.count_amino_acids()
    length = len(cleaned)

    return {
        "feature_version": FEATURE_VERSION,
        "canonical_length": length,
        "noncanonical_residues": len(sequence) - length,
        "canonical_pairs": canonical_pairs(sequence),
        "molecular_weight": analysis.molecular_weight(),
        "isoelectric_point": IsoelectricPoint(cleaned).pi(),
        "gravy": analysis.gravy(),
        "instability_index": analysis.instability_index(),
        # Ikai (1980): 100 * (A + 2.9 V + 3.9 (I + L)) / N
        "aliphatic_index": 100.0
        * (
            counts["A"]
            + 2.9 * counts["V"]
            + 3.9 * (counts["I"] + counts["L"])
        )
        / length,
        "aromaticity": analysis.aromaticity(),
        "sequence_entropy": shannon_entropy(cleaned),
    }


def shannon_entropy(cleaned: str) -> float:
    """Shannon entropy of the residue composition, in bits (max ``log2(20)``)."""
    length = len(cleaned)
    if not length:
        return 0.0
    return -sum(
        (count / length) * math.log2(count / length)
        for count in Counter(cleaned).values()
    )

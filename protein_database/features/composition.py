"""Amino acid composition calculations for protein sequences."""

from dataclasses import dataclass

from protein_database.features.sequence_features import STANDARD_AMINO_ACIDS


@dataclass
class AminoAcidComposition:
    """Amino acid percentage composition for a protein sequence.

    Each field represents the percentage abundance of that amino acid.
    Field names map to database column names in the amino_acid_composition table.
    """

    alanine_percentage: float = 0.0
    arginine_percentage: float = 0.0
    asparagine_percentage: float = 0.0
    aspartic_acid_percentage: float = 0.0
    cysteine_percentage: float = 0.0
    glutamine_percentage: float = 0.0
    glutamic_acid_percentage: float = 0.0
    glycine_percentage: float = 0.0
    histidine_percentage: float = 0.0
    isoleucine_percentage: float = 0.0
    leucine_percentage: float = 0.0
    lysine_percentage: float = 0.0
    methionine_percentage: float = 0.0
    phenylalanine_percentage: float = 0.0
    proline_percentage: float = 0.0
    serine_percentage: float = 0.0
    threonine_percentage: float = 0.0
    tryptophan_percentage: float = 0.0
    tyrosine_percentage: float = 0.0
    valine_percentage: float = 0.0


# Mapping from amino acid one-letter code to the dataclass field name
AMINO_ACID_FIELD_MAP: dict[str, str] = {
    "A": "alanine_percentage",
    "R": "arginine_percentage",
    "N": "asparagine_percentage",
    "D": "aspartic_acid_percentage",
    "C": "cysteine_percentage",
    "Q": "glutamine_percentage",
    "E": "glutamic_acid_percentage",
    "G": "glycine_percentage",
    "H": "histidine_percentage",
    "I": "isoleucine_percentage",
    "L": "leucine_percentage",
    "K": "lysine_percentage",
    "M": "methionine_percentage",
    "F": "phenylalanine_percentage",
    "P": "proline_percentage",
    "S": "serine_percentage",
    "T": "threonine_percentage",
    "W": "tryptophan_percentage",
    "Y": "tyrosine_percentage",
    "V": "valine_percentage",
}


def compute_amino_acid_composition(sequence: str) -> AminoAcidComposition:
    """Compute the percentage composition of each amino acid in a sequence.

    Args:
        sequence: Amino acid sequence.

    Returns:
        AminoAcidComposition dataclass with percentage for each amino acid.

    Raises:
        ValueError: If the sequence is empty or has no standard amino acids.
    """
    # Normalize and filter to standard amino acids
    cleaned = "".join(
        c for c in sequence.upper() if c in STANDARD_AMINO_ACIDS
    )

    if not cleaned:
        raise ValueError("Sequence contains no standard amino acids.")

    length = len(cleaned)
    counts = {aa: 0 for aa in STANDARD_AMINO_ACIDS}

    for c in cleaned:
        if c in counts:
            counts[c] += 1

    composition = AminoAcidComposition()
    for aa, field_name in AMINO_ACID_FIELD_MAP.items():
        percentage = (counts[aa] / length) * 100.0
        setattr(composition, field_name, percentage)

    return composition


def count_amino_acids(sequence: str) -> dict[str, int]:
    """Count each amino acid occurrence in a sequence.

    Args:
        sequence: Amino acid sequence.

    Returns:
        Dict mapping amino acid one-letter code to its count.
    """
    counts = {aa: 0 for aa in STANDARD_AMINO_ACIDS}
    for c in sequence.upper():
        if c in counts:
            counts[c] += 1
    return counts
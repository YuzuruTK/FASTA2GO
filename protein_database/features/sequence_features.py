"""Basic physicochemical features computed from protein sequences."""

from dataclasses import dataclass

from Bio.SeqUtils import ProtParam
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint

# Standard amino acid one-letter codes
STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Chemical classifications of amino acids
HYDROPHOBIC_RESIDUES: set[str] = set("AVILMFWY")
POLAR_RESIDUES: set[str] = set("STNCQ")
POSITIVE_RESIDUES: set[str] = set("KRH")
NEGATIVE_RESIDUES: set[str] = set("DE")
SULFUR_RESIDUES: set[str] = set("CM")
AROMATIC_RESIDUES: set[str] = set("FYW")


@dataclass
class SequenceFeatures:
    """Computed physicochemical features for a protein sequence."""

    molecular_weight: float
    isoelectric_point: float
    gravy_index: float
    instability_index: float
    aliphatic_index: float
    aromaticity: float
    charge_positive_percentage: float
    charge_negative_percentage: float
    hydrophobic_percentage: float
    polar_percentage: float
    charged_percentage: float
    sulfur_percentage: float
    aromatic_percentage: float
    sequence_entropy: float
    amino_acid_diversity: float


def compute_sequence_features(sequence: str) -> SequenceFeatures:
    """Compute basic physicochemical features for a protein sequence.

    Uses Biopython's ProtParam to calculate molecular weight, isoelectric
    point, GRAVY index, instability index, aliphatic index, and aromaticity.
    Chemical classification percentages are computed from residue counts.

    Args:
        sequence: Amino acid sequence.

    Returns:
        SequenceFeatures dataclass with all computed values.

    Raises:
        ValueError: If the sequence is empty.
    """
    if not sequence:
        raise ValueError("Cannot compute features for an empty sequence.")

    # Normalize sequence to uppercase for consistent analysis
    seq_upper = sequence.upper()

    # Remove non-standard characters (e.g., X, *, -)
    cleaned = "".join(c for c in seq_upper if c in STANDARD_AMINO_ACIDS)

    if not cleaned:
        raise ValueError("Sequence contains no standard amino acids.")

    analysis = ProtParam.ProteinAnalysis(cleaned)
    pp = analysis

    # Basic properties via Biopython
    molecular_weight = pp.molecular_weight()
    iso_point = IsoelectricPoint(cleaned)
    isoelectric_point = iso_point.pi()
    gravy_index = pp.gravy()
    instability_index = pp.instability_index()

    length = len(cleaned)

    # Aliphatic index - not in Biopython's ProteinAnalysis directly.
    # Formula: AI = (Ala + 2.9*Val + 3.9*(Ile + Leu)) / length * 100
    aa_counts = pp.count_amino_acids()
    aliphatic_index = (
        aa_counts.get("A", 0)
        + 2.9 * aa_counts.get("V", 0)
        + 3.9 * (aa_counts.get("I", 0) + aa_counts.get("L", 0))
    ) / length * 100.0

    aromaticity = pp.aromaticity()

    # Amino acid composition percentages
    compositions = pp.amino_acids_percent

    def residue_percentage(residue_set: set[str]) -> float:
        """Compute total percentage of residues in the given set.

        Note: Biopython's `amino_acids_percent` property already returns
        values in the 0-100 range, so no further scaling is applied.
        """
        return sum(compositions.get(aa, 0.0) for aa in residue_set)

    charge_positive_percentage = residue_percentage(POSITIVE_RESIDUES)
    charge_negative_percentage = residue_percentage(NEGATIVE_RESIDUES)
    hydrophobic_percentage = residue_percentage(HYDROPHOBIC_RESIDUES)
    polar_percentage = residue_percentage(POLAR_RESIDUES)
    charged_percentage = charge_positive_percentage + charge_negative_percentage
    sulfur_percentage = residue_percentage(SULFUR_RESIDUES)
    aromatic_percentage = residue_percentage(AROMATIC_RESIDUES)

    # Sequence complexity: amino acid diversity
    unique_amino_acids = {aa for aa in cleaned if aa in STANDARD_AMINO_ACIDS}
    amino_acid_diversity = len(unique_amino_acids) / len(STANDARD_AMINO_ACIDS)

    # Shannon entropy is computed in the complexity module.
    # We'll compute it here for convenience to keep the dataclass complete.
    from protein_database.features.complexity import shannon_entropy

    sequence_entropy = shannon_entropy(cleaned)

    return SequenceFeatures(
        molecular_weight=molecular_weight,
        isoelectric_point=isoelectric_point,
        gravy_index=gravy_index,
        instability_index=instability_index,
        aliphatic_index=aliphatic_index,
        aromaticity=aromaticity,
        charge_positive_percentage=charge_positive_percentage,
        charge_negative_percentage=charge_negative_percentage,
        hydrophobic_percentage=hydrophobic_percentage,
        polar_percentage=polar_percentage,
        charged_percentage=charged_percentage,
        sulfur_percentage=sulfur_percentage,
        aromatic_percentage=aromatic_percentage,
        sequence_entropy=sequence_entropy,
        amino_acid_diversity=amino_acid_diversity,
    )
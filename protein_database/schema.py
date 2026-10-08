"""Single source of truth for the database schema.

Column lists are data. The DDL and the ``analysis_data`` view are derived
from them, so a feature/column is defined exactly once.
"""

INPUT_COLUMNS = ["fasta_id", "source_file"]

PROTEIN_COLUMNS = [
    "fasta_id",
    "source_file",
    "description",
    "sequence",
    "sequence_length",
    "created_at",
]

FEATURE_COLUMNS = [
    "molecular_weight",
    "isoelectric_point",
    "gravy_index",
    "instability_index",
    "aliphatic_index",
    "aromaticity",
    "charge_positive_percentage",
    "charge_negative_percentage",
    "hydrophobic_percentage",
    "polar_percentage",
    "charged_percentage",
    "sulfur_percentage",
    "aromatic_percentage",
    "sequence_entropy",
    "amino_acid_diversity",
]

#: One-letter code -> full amino acid name (used for composition columns).
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

COMPOSITION_COLUMNS = [f"{name}_percentage" for name in AMINO_ACIDS.values()]

ANNOTATION_COLUMNS = [
    "fasta_id",
    "source_file",
    "uniprot_accession",
    "uniprot_id",
    "protein_name",
    "gene_name",
    "function",
    "subcellular_location",
    "protein_existence",
    "organism",
    "keywords",
    "status",
    "retrieved_at",
]

#: Annotation columns that live only in the view (renamed status).
VIEW_ANNOTATION_COLUMNS = ANNOTATION_COLUMNS[2:-2]

KEY = "(fasta_id, source_file)"


def _child_table(name: str, columns: list[str]) -> str:
    """Build DDL for a feature table keyed by the protein composite key."""
    doubles = ", ".join(f"{column} DOUBLE" for column in columns)
    return (
        f"CREATE TABLE IF NOT EXISTS {name} ("
        f"fasta_id VARCHAR NOT NULL, source_file VARCHAR NOT NULL, {doubles}, "
        f"PRIMARY KEY {KEY}, "
        f"FOREIGN KEY {KEY} REFERENCES proteins {KEY})"
    )


SCHEMA_SQL = {
    "proteins": (
        "CREATE TABLE IF NOT EXISTS proteins ("
        "fasta_id VARCHAR NOT NULL, source_file VARCHAR NOT NULL, "
        "description VARCHAR, sequence VARCHAR NOT NULL, "
        "sequence_length INTEGER NOT NULL, created_at TIMESTAMP NOT NULL, "
        f"PRIMARY KEY {KEY})"
    ),
    "sequence_features": _child_table("sequence_features", FEATURE_COLUMNS),
    "amino_acid_composition": _child_table(
        "amino_acid_composition", COMPOSITION_COLUMNS
    ),
    "uniprot_annotations": (
        "CREATE TABLE IF NOT EXISTS uniprot_annotations ("
        "fasta_id VARCHAR NOT NULL, source_file VARCHAR NOT NULL, "
        "uniprot_accession VARCHAR, uniprot_id VARCHAR, protein_name VARCHAR, "
        "gene_name VARCHAR, function VARCHAR, subcellular_location VARCHAR, "
        "protein_existence VARCHAR, organism VARCHAR, keywords VARCHAR, "
        "status VARCHAR NOT NULL, retrieved_at TIMESTAMP NOT NULL, "
        f"PRIMARY KEY {KEY}, FOREIGN KEY {KEY} REFERENCES proteins {KEY})"
    ),
}

#: Table creation order (parents first); drop in reverse.
TABLE_NAMES = list(SCHEMA_SQL)


def _select(alias: str, columns: list[str]) -> str:
    return ", ".join(f"{alias}.{column}" for column in columns)


ANALYSIS_VIEW_SQL = f"""CREATE OR REPLACE VIEW analysis_data AS
SELECT
    {_select("p", PROTEIN_COLUMNS)},
    {_select("sf", FEATURE_COLUMNS)},
    {_select("aac", COMPOSITION_COLUMNS)},
    {_select("ua", VIEW_ANNOTATION_COLUMNS)},
    ua.status AS annotation_status,
    ua.retrieved_at
FROM proteins p
LEFT JOIN sequence_features sf USING (fasta_id, source_file)
LEFT JOIN amino_acid_composition aac USING (fasta_id, source_file)
LEFT JOIN uniprot_annotations ua USING (fasta_id, source_file)
"""

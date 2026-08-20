"""Schema definitions and dataclasses for the protein feature database.

This module defines the DuckDB schema and lightweight dataclasses used to
carry computed feature data. It does not use an ORM; persistence is handled
natively by the DuckDB Python API in the pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utcnow() -> str:
    """Return the current UTC timestamp formatted as an ISO string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Protein:
    """Raw FASTA information for a protein."""

    fasta_id: str
    source_file: str
    sequence: str
    description: str | None = None
    created_at: str = field(default_factory=_utcnow)

    @property
    def sequence_length(self) -> int:
        """Number of amino acids in the sequence."""
        return len(self.sequence)


@dataclass
class SequenceFeatures:
    """Calculated physicochemical properties of a protein sequence."""

    fasta_id: str
    source_file: str
    molecular_weight: float | None = None
    isoelectric_point: float | None = None
    gravy_index: float | None = None
    instability_index: float | None = None
    aliphatic_index: float | None = None
    aromaticity: float | None = None
    charge_positive_percentage: float | None = None
    charge_negative_percentage: float | None = None
    hydrophobic_percentage: float | None = None
    polar_percentage: float | None = None
    charged_percentage: float | None = None
    sulfur_percentage: float | None = None
    aromatic_percentage: float | None = None
    sequence_entropy: float | None = None
    amino_acid_diversity: float | None = None


@dataclass
class AminoAcidComposition:
    """Amino acid frequency percentages for a protein sequence."""

    fasta_id: str
    source_file: str
    alanine_percentage: float | None = None
    arginine_percentage: float | None = None
    asparagine_percentage: float | None = None
    aspartic_acid_percentage: float | None = None
    cysteine_percentage: float | None = None
    glutamine_percentage: float | None = None
    glutamic_acid_percentage: float | None = None
    glycine_percentage: float | None = None
    histidine_percentage: float | None = None
    isoleucine_percentage: float | None = None
    leucine_percentage: float | None = None
    lysine_percentage: float | None = None
    methionine_percentage: float | None = None
    phenylalanine_percentage: float | None = None
    proline_percentage: float | None = None
    serine_percentage: float | None = None
    threonine_percentage: float | None = None
    tryptophan_percentage: float | None = None
    tyrosine_percentage: float | None = None
    valine_percentage: float | None = None


@dataclass
class ValidationFeatures:
    """Future validation metrics. All fields initially None."""

    fasta_id: str
    source_file: str
    sequence_quality_score: float | None = None
    stability_score: float | None = None
    solubility_score: float | None = None
    structure_score: float | None = None
    overall_score: float | None = None
    classification: str | None = None


@dataclass
class UniProtAnnotation:
    """UniProt annotation data for a single local protein row."""

    fasta_id: str
    source_file: str
    uniprot_accession: str | None = None
    uniprot_id: str | None = None
    protein_name: str | None = None
    gene_name: str | None = None
    function: str | None = None
    subcellular_location: str | None = None
    protein_existence: str | None = None
    organism: str | None = None
    keywords: str | None = None
    status: str = "fetched"
    retrieved_at: str = field(default_factory=_utcnow)


#: Names of all tables in the database, in dependency order (children first).
TABLE_NAMES = [
    "uniprot_accession_queue",
    "uniprot_annotations",
    "sequence_features",
    "amino_acid_composition",
    "validation_features",
    "proteins",
]

#: DuckDB DDL statements used to create the normalized schema.
SCHEMA_SQL = {
    "proteins": """
        CREATE TABLE IF NOT EXISTS proteins (
            fasta_id        VARCHAR NOT NULL,
            source_file     VARCHAR NOT NULL,
            description     VARCHAR,
            sequence        VARCHAR NOT NULL,
            sequence_length INTEGER NOT NULL,
            created_at      TIMESTAMP NOT NULL,
            PRIMARY KEY (fasta_id, source_file)
        )
    """,
    "sequence_features": """
        CREATE TABLE IF NOT EXISTS sequence_features (
            fasta_id                   VARCHAR NOT NULL,
            source_file                VARCHAR NOT NULL,
            molecular_weight           DOUBLE,
            isoelectric_point          DOUBLE,
            gravy_index                DOUBLE,
            instability_index          DOUBLE,
            aliphatic_index            DOUBLE,
            aromaticity                DOUBLE,
            charge_positive_percentage DOUBLE,
            charge_negative_percentage DOUBLE,
            hydrophobic_percentage     DOUBLE,
            polar_percentage           DOUBLE,
            charged_percentage         DOUBLE,
            sulfur_percentage          DOUBLE,
            aromatic_percentage        DOUBLE,
            sequence_entropy           DOUBLE,
            amino_acid_diversity       DOUBLE,
            PRIMARY KEY (fasta_id, source_file),
            FOREIGN KEY (fasta_id, source_file)
                REFERENCES proteins (fasta_id, source_file)
        )
    """,
    "amino_acid_composition": """
        CREATE TABLE IF NOT EXISTS amino_acid_composition (
            fasta_id                VARCHAR NOT NULL,
            source_file             VARCHAR NOT NULL,
            alanine_percentage      DOUBLE,
            arginine_percentage     DOUBLE,
            asparagine_percentage   DOUBLE,
            aspartic_acid_percentage DOUBLE,
            cysteine_percentage     DOUBLE,
            glutamine_percentage    DOUBLE,
            glutamic_acid_percentage DOUBLE,
            glycine_percentage      DOUBLE,
            histidine_percentage    DOUBLE,
            isoleucine_percentage   DOUBLE,
            leucine_percentage      DOUBLE,
            lysine_percentage       DOUBLE,
            methionine_percentage   DOUBLE,
            phenylalanine_percentage DOUBLE,
            proline_percentage      DOUBLE,
            serine_percentage       DOUBLE,
            threonine_percentage    DOUBLE,
            tryptophan_percentage   DOUBLE,
            tyrosine_percentage     DOUBLE,
            valine_percentage       DOUBLE,
            PRIMARY KEY (fasta_id, source_file),
            FOREIGN KEY (fasta_id, source_file)
                REFERENCES proteins (fasta_id, source_file)
        )
    """,
    "validation_features": """
        CREATE TABLE IF NOT EXISTS validation_features (
            fasta_id               VARCHAR NOT NULL,
            source_file            VARCHAR NOT NULL,
            sequence_quality_score DOUBLE,
            stability_score        DOUBLE,
            solubility_score       DOUBLE,
            structure_score        DOUBLE,
            overall_score          DOUBLE,
            classification         VARCHAR,
            PRIMARY KEY (fasta_id, source_file),
            FOREIGN KEY (fasta_id, source_file)
                REFERENCES proteins (fasta_id, source_file)
        )
    """,
    "uniprot_annotations": """
        CREATE TABLE IF NOT EXISTS uniprot_annotations (
            fasta_id             VARCHAR NOT NULL,
            source_file          VARCHAR NOT NULL,
            uniprot_accession    VARCHAR,
            uniprot_id           VARCHAR,
            protein_name         VARCHAR,
            gene_name            VARCHAR,
            function             VARCHAR,
            subcellular_location VARCHAR,
            protein_existence    VARCHAR,
            organism             VARCHAR,
            keywords             VARCHAR,
            status               VARCHAR NOT NULL,
            retrieved_at         TIMESTAMP NOT NULL,
            PRIMARY KEY (fasta_id, source_file),
            FOREIGN KEY (fasta_id, source_file)
                REFERENCES proteins (fasta_id, source_file)
        )
    """,
    "uniprot_accession_queue": """
        CREATE TABLE IF NOT EXISTS uniprot_accession_queue (
            accession    VARCHAR PRIMARY KEY,
            status       VARCHAR NOT NULL DEFAULT 'pending',
            retry_count  INTEGER NOT NULL DEFAULT 0,
            last_attempt TIMESTAMP,
            created_at   TIMESTAMP NOT NULL DEFAULT now(),
            completed_at TIMESTAMP
        )
    """,
}

#: DuckDB DDL for the denormalized analysis view joining all per-protein tables.
ANALYSIS_VIEW_SQL = """
    CREATE OR REPLACE VIEW analysis_data AS
    SELECT
        p.fasta_id,
        p.source_file,
        p.description,
        p.sequence,
        p.sequence_length,
        p.created_at,
        sf.molecular_weight,
        sf.isoelectric_point,
        sf.gravy_index,
        sf.instability_index,
        sf.aliphatic_index,
        sf.aromaticity,
        sf.charge_positive_percentage,
        sf.charge_negative_percentage,
        sf.hydrophobic_percentage,
        sf.polar_percentage,
        sf.charged_percentage,
        sf.sulfur_percentage,
        sf.aromatic_percentage,
        sf.sequence_entropy,
        sf.amino_acid_diversity,
        aac.alanine_percentage,
        aac.arginine_percentage,
        aac.asparagine_percentage,
        aac.aspartic_acid_percentage,
        aac.cysteine_percentage,
        aac.glutamine_percentage,
        aac.glutamic_acid_percentage,
        aac.glycine_percentage,
        aac.histidine_percentage,
        aac.isoleucine_percentage,
        aac.leucine_percentage,
        aac.lysine_percentage,
        aac.methionine_percentage,
        aac.phenylalanine_percentage,
        aac.proline_percentage,
        aac.serine_percentage,
        aac.threonine_percentage,
        aac.tryptophan_percentage,
        aac.tyrosine_percentage,
        aac.valine_percentage,
        vf.sequence_quality_score,
        vf.stability_score,
        vf.solubility_score,
        vf.structure_score,
        vf.overall_score,
        vf.classification,
        ua.uniprot_accession,
        ua.uniprot_id,
        ua.protein_name,
        ua.gene_name,
        ua.function,
        ua.subcellular_location,
        ua.protein_existence,
        ua.organism,
        ua.keywords,
        ua.status AS annotation_status,
        ua.retrieved_at
    FROM proteins p
    LEFT JOIN sequence_features sf
        ON p.fasta_id = sf.fasta_id AND p.source_file = sf.source_file
    LEFT JOIN amino_acid_composition aac
        ON p.fasta_id = aac.fasta_id AND p.source_file = aac.source_file
    LEFT JOIN validation_features vf
        ON p.fasta_id = vf.fasta_id AND p.source_file = vf.source_file
    LEFT JOIN uniprot_annotations ua
        ON p.fasta_id = ua.fasta_id AND p.source_file = ua.source_file
"""

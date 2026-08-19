"""Pipeline orchestrator: FASTA parse -> feature extraction -> DuckDB storage."""

from pathlib import Path

import duckdb
from tqdm import tqdm

from protein_database.database.connection import connect, drop_tables, init_database
from protein_database.database.models import (
    AminoAcidComposition as CompositionModel,
    Protein as ProteinModel,
    SequenceFeatures as SequenceFeaturesModel,
)
from protein_database.features.composition import compute_amino_acid_composition
from protein_database.features.sequence_features import compute_sequence_features
from protein_database.parser.fasta_parser import FastaRecord, count_fasta_entries, parse_fasta


def process_fasta_file(
    fasta_path: str | Path,
    db_path: str | Path = "proteins.duckdb",
    batch_size: int = 1000,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Process a FASTA file and populate a DuckDB database with features.

    This function:
    1. Opens the FASTA file and counts the total number of entries.
    2. Parses each protein record.
    3. Computes sequence features and amino acid composition.
    4. Inserts everything into DuckDB in batches via executemany, skipping
       records that already exist in the database (idempotent).
    5. Returns a summary dict with processing statistics.

    Args:
        fasta_path: Path to the input FASTA file.
        db_path: Path to the DuckDB database file.
        batch_size: Number of records to batch-insert to the database.
        limit: Optional maximum number of records to process (for testing).
        overwrite: If True, drop and recreate the database before processing.

    Returns:
        Dict with keys: 'processed', 'skipped', 'total_entries',
        'elapsed_seconds', 'database_path'.
    """
    import time

    start_time = time.time()
    fasta_path = Path(fasta_path)
    source_file = fasta_path.name

    # Count total entries for the progress bar
    print(f"Counting FASTA entries in {fasta_path}...")
    total_entries = count_fasta_entries(fasta_path)
    print(f"Found {total_entries:,} entries.")

    if limit is not None:
        total_entries = min(total_entries, limit)

    # Initialize database
    conn = connect(db_path)

    if overwrite:
        drop_tables(conn)
        print("Overwrite mode: dropped existing database tables.")

    init_database(conn)

    processed = 0
    skipped = 0
    errors = 0

    # Pre-load existing fasta_ids for this source file to skip duplicates
    existing_ids = set(
        row[0]
        for row in conn.execute(
            "SELECT fasta_id FROM proteins WHERE source_file = ?", [source_file]
        ).fetchall()
    )
    if existing_ids:
        print(f"Found {len(existing_ids):,} existing records for '{source_file}'. Skipping duplicates.")

    batch_proteins: list[list] = []
    batch_features: list[list] = []
    batch_compositions: list[list] = []

    progress_bar = tqdm(parse_fasta(fasta_path), total=total_entries, desc="Processing", unit="protein")
    for record in progress_bar:
        # Check limit before processing each record
        if limit is not None and processed + skipped >= limit:
            break

        try:
            # Skip records that already exist in the database
            if record.fasta_id in existing_ids:
                skipped += 1
                progress_bar.set_postfix({"skipped": skipped})
                continue

            protein, features, composition = _extract_and_build_models(record, source_file)
            batch_proteins.append(_protein_row(protein))
            batch_features.append(_features_row(features))
            batch_compositions.append(_composition_row(composition))
            processed += 1

            if len(batch_proteins) >= batch_size:
                _flush_batches(conn, batch_proteins, batch_features, batch_compositions)
                batch_proteins = []
                batch_features = []
                batch_compositions = []

        except Exception:
            errors += 1
            progress_bar.set_postfix({"errors": errors})

    # Flush any remaining batch
    if batch_proteins:
        _flush_batches(conn, batch_proteins, batch_features, batch_compositions)

    conn.close()
    elapsed = time.time() - start_time
    progress_bar.close()

    return {
        "processed": processed,
        "skipped": skipped,
        "total_entries": total_entries,
        "errors": errors,
        "elapsed_seconds": elapsed,
        "database_path": str(db_path),
    }


def _extract_and_build_models(
    record: FastaRecord, source_file: str
) -> tuple[ProteinModel, SequenceFeaturesModel, CompositionModel]:
    """Compute features for a record and build model instances.

    Args:
        record: Parsed FASTA record.
        source_file: Source FASTA filename.

    Returns:
        Tuple of (Protein, SequenceFeatures, AminoAcidComposition) models.
    """
    # Compute features
    features = compute_sequence_features(record.sequence)
    composition = compute_amino_acid_composition(record.sequence)

    # Build protein model
    protein = ProteinModel(
        fasta_id=record.fasta_id,
        description=record.description,
        sequence=record.sequence,
        source_file=source_file,
    )

    # Build sequence features model
    seq_feature = SequenceFeaturesModel(
        fasta_id=record.fasta_id,
        source_file=source_file,
        molecular_weight=features.molecular_weight,
        isoelectric_point=features.isoelectric_point,
        gravy_index=features.gravy_index,
        instability_index=features.instability_index,
        aliphatic_index=features.aliphatic_index,
        aromaticity=features.aromaticity,
        charge_positive_percentage=features.charge_positive_percentage,
        charge_negative_percentage=features.charge_negative_percentage,
        hydrophobic_percentage=features.hydrophobic_percentage,
        polar_percentage=features.polar_percentage,
        charged_percentage=features.charged_percentage,
        sulfur_percentage=features.sulfur_percentage,
        aromatic_percentage=features.aromatic_percentage,
        sequence_entropy=features.sequence_entropy,
        amino_acid_diversity=features.amino_acid_diversity,
    )

    # Build amino acid composition model
    composition_model = CompositionModel(
        fasta_id=record.fasta_id,
        source_file=source_file,
        alanine_percentage=composition.alanine_percentage,
        arginine_percentage=composition.arginine_percentage,
        asparagine_percentage=composition.asparagine_percentage,
        aspartic_acid_percentage=composition.aspartic_acid_percentage,
        cysteine_percentage=composition.cysteine_percentage,
        glutamine_percentage=composition.glutamine_percentage,
        glutamic_acid_percentage=composition.glutamic_acid_percentage,
        glycine_percentage=composition.glycine_percentage,
        histidine_percentage=composition.histidine_percentage,
        isoleucine_percentage=composition.isoleucine_percentage,
        leucine_percentage=composition.leucine_percentage,
        lysine_percentage=composition.lysine_percentage,
        methionine_percentage=composition.methionine_percentage,
        phenylalanine_percentage=composition.phenylalanine_percentage,
        proline_percentage=composition.proline_percentage,
        serine_percentage=composition.serine_percentage,
        threonine_percentage=composition.threonine_percentage,
        tryptophan_percentage=composition.tryptophan_percentage,
        tyrosine_percentage=composition.tyrosine_percentage,
        valine_percentage=composition.valine_percentage,
    )

    return protein, seq_feature, composition_model


_PROTEIN_COLUMNS = [
    "fasta_id",
    "source_file",
    "description",
    "sequence",
    "sequence_length",
    "created_at",
]


def _protein_row(protein: ProteinModel) -> list:
    """Convert a Protein model to a row of values for insertion."""
    return [
        protein.fasta_id,
        protein.source_file,
        protein.description,
        protein.sequence,
        protein.sequence_length,
        protein.created_at,
    ]


_FEATURES_COLUMNS = [
    "fasta_id",
    "source_file",
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


def _features_row(features: SequenceFeaturesModel) -> list:
    """Convert a SequenceFeatures model to a row of values for insertion."""
    return [
        features.fasta_id,
        features.source_file,
        features.molecular_weight,
        features.isoelectric_point,
        features.gravy_index,
        features.instability_index,
        features.aliphatic_index,
        features.aromaticity,
        features.charge_positive_percentage,
        features.charge_negative_percentage,
        features.hydrophobic_percentage,
        features.polar_percentage,
        features.charged_percentage,
        features.sulfur_percentage,
        features.aromatic_percentage,
        features.sequence_entropy,
        features.amino_acid_diversity,
    ]


_COMPOSITION_COLUMNS = [
    "fasta_id",
    "source_file",
    "alanine_percentage",
    "arginine_percentage",
    "asparagine_percentage",
    "aspartic_acid_percentage",
    "cysteine_percentage",
    "glutamine_percentage",
    "glutamic_acid_percentage",
    "glycine_percentage",
    "histidine_percentage",
    "isoleucine_percentage",
    "leucine_percentage",
    "lysine_percentage",
    "methionine_percentage",
    "phenylalanine_percentage",
    "proline_percentage",
    "serine_percentage",
    "threonine_percentage",
    "tryptophan_percentage",
    "tyrosine_percentage",
    "valine_percentage",
]


def _composition_row(composition: CompositionModel) -> list:
    """Convert an AminoAcidComposition model to a row of values for insertion."""
    return [
        composition.fasta_id,
        composition.source_file,
        composition.alanine_percentage,
        composition.arginine_percentage,
        composition.asparagine_percentage,
        composition.aspartic_acid_percentage,
        composition.cysteine_percentage,
        composition.glutamine_percentage,
        composition.glutamic_acid_percentage,
        composition.glycine_percentage,
        composition.histidine_percentage,
        composition.isoleucine_percentage,
        composition.leucine_percentage,
        composition.lysine_percentage,
        composition.methionine_percentage,
        composition.phenylalanine_percentage,
        composition.proline_percentage,
        composition.serine_percentage,
        composition.threonine_percentage,
        composition.tryptophan_percentage,
        composition.tyrosine_percentage,
        composition.valine_percentage,
    ]


def _insert_many(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    columns: list[str],
    rows: list[list],
) -> None:
    """Bulk-insert rows into a table using executemany.

    Args:
        conn: DuckDB connection.
        table: Target table name.
        columns: Column names to insert into.
        rows: List of value rows.
    """
    if not rows:
        return
    placeholder = ", ".join(["?"] * len(columns))
    col_str = ", ".join(columns)
    conn.executemany(
        f"INSERT INTO {table} ({col_str}) VALUES ({placeholder})",
        rows,
    )


def _flush_batches(
    conn: duckdb.DuckDBPyConnection,
    batch_proteins: list[list],
    batch_features: list[list],
    batch_compositions: list[list],
) -> None:
    """Insert all batched rows into their respective tables.

    Args:
        conn: DuckDB connection.
        batch_proteins: Rows for the proteins table.
        batch_features: Rows for the sequence_features table.
        batch_compositions: Rows for the amino_acid_composition table.
    """
    _insert_many(conn, "proteins", _PROTEIN_COLUMNS, batch_proteins)
    _insert_many(conn, "sequence_features", _FEATURES_COLUMNS, batch_features)
    _insert_many(conn, "amino_acid_composition", _COMPOSITION_COLUMNS, batch_compositions)
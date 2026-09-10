"""Pipeline orchestrator: FASTA parse -> feature extraction -> DuckDB storage."""

import csv
from pathlib import Path
from uuid import uuid4

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
    csv_cache_dir: str | Path = "cache/csv_bulk",
) -> dict:
    """Process a FASTA file and populate a DuckDB database with features.

    This function:
    1. Opens the FASTA file and counts the total number of entries.
    2. Parses each protein record.
    3. Computes sequence features and amino acid composition.
    4. Inserts everything into DuckDB in batches by staging CSVs in the
       cache directory and loading them with DuckDB's native CSV import,
       skipping records that already exist in the database (idempotent).
    5. Returns a summary dict with processing statistics.

    Args:
        fasta_path: Path to the input FASTA file.
        db_path: Path to the DuckDB database file.
        batch_size: Number of records to batch-insert to the database.
        limit: Optional maximum number of records to process (for testing).
        overwrite: If True, drop and recreate the database before processing.
        csv_cache_dir: Directory used to stage batch CSVs before the native
            DuckDB import. Files are removed after a successful import.

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

    cache_dir = Path(csv_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

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
                _flush_batches(
                    conn, batch_proteins, batch_features, batch_compositions, cache_dir
                )
                batch_proteins = []
                batch_features = []
                batch_compositions = []

        except Exception:
            errors += 1
            progress_bar.set_postfix({"errors": errors})

    # Flush any remaining batch
    if batch_proteins:
        _flush_batches(conn, batch_proteins, batch_features, batch_compositions, cache_dir)

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


#: Default directory used to stage batch CSVs before the native DuckDB import.
_CSV_CACHE_DIR = Path("cache/csv_bulk")


def _table_column_types(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    columns: list[str],
) -> dict[str, str]:
    """Return the DuckDB type of each requested column in the target table.

    Types are read from ``information_schema.columns`` so the CSV import stays
    correct even if the schema evolves.

    Args:
        conn: DuckDB connection.
        table: Target table name.
        columns: Column names to look up.

    Returns:
        Mapping of column name -> DuckDB type name, in ``columns`` order.

    Raises:
        ValueError: If any requested column is missing from the table.
    """
    rows = conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = ?",
        [table],
    ).fetchall()
    type_by_name = {name: dtype for name, dtype in rows}
    missing = [col for col in columns if col not in type_by_name]
    if missing:
        raise ValueError(f"Columns {missing} not found in table '{table}'.")
    return {col: type_by_name[col] for col in columns}


def _insert_many(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    columns: list[str],
    rows: list[list],
    cache_dir: str | Path | None = None,
) -> None:
    """Bulk-insert rows into a table by staging a CSV and importing it natively.

    Instead of ``executemany`` (row-by-row, slow in DuckDB), the rows are
    written to a CSV file in the cache directory with the stdlib ``csv``
    module and then loaded with DuckDB's native ``read_csv`` reader into the
    target table in a single statement. ``None`` values are written as empty
    (unquoted) CSV fields, which DuckDB reads back as SQL NULL.

    The staging CSV is deleted after a successful import; on failure it is
    left in the cache directory for debugging and the exception is re-raised.

    Args:
        conn: DuckDB connection.
        table: Target table name.
        columns: Column names to insert into (same order as rows).
        rows: List of value rows.
        cache_dir: Directory used to stage the CSV files. Defaults to
            ``cache/csv_bulk`` relative to the working directory.

    Raises:
        ValueError: If the requested columns are missing from the table.
    """
    if not rows:
        return

    cache_path = Path(cache_dir) if cache_dir else _CSV_CACHE_DIR
    cache_path.mkdir(parents=True, exist_ok=True)
    csv_path = cache_path / f"{table}_{uuid4().hex}.csv"

    # Stage the rows as a CSV (header + data). None becomes an empty unquoted
    # field, which DuckDB's CSV reader converts to NULL on import.
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        writer.writerows(rows)

    col_list = ", ".join(columns)
    column_types = _table_column_types(conn, table, columns)
    types_spec = ", ".join(
        f"'{name}': '{dtype}'" for name, dtype in column_types.items()
    )
    escaped_path = str(csv_path).replace("'", "''")

    try:
        conn.execute(
            f"INSERT INTO {table} ({col_list}) "
            f"SELECT {col_list} FROM read_csv('{escaped_path}', header=true, "
            f"columns={{{types_spec}}})"
        )
    except Exception:
        # Keep the staged CSV for debugging; the caller handles the error.
        raise

    # The import succeeded; remove the staged CSV. A cleanup failure is
    # non-fatal (rows are already committed), so only warn about it.
    try:
        csv_path.unlink()
    except OSError as exc:  # pragma: no cover - cleanup-only path
        print(f"Warning: could not remove staged CSV {csv_path}: {exc}")


def _flush_batches(
    conn: duckdb.DuckDBPyConnection,
    batch_proteins: list[list],
    batch_features: list[list],
    batch_compositions: list[list],
    cache_dir: str | Path | None = None,
) -> None:
    """Insert all batched rows into their respective tables.

    Args:
        conn: DuckDB connection.
        batch_proteins: Rows for the proteins table.
        batch_features: Rows for the sequence_features table.
        batch_compositions: Rows for the amino_acid_composition table.
        cache_dir: Directory used to stage the batch CSVs (see _insert_many).
    """
    _insert_many(conn, "proteins", _PROTEIN_COLUMNS, batch_proteins, cache_dir)
    _insert_many(conn, "sequence_features", _FEATURES_COLUMNS, batch_features, cache_dir)
    _insert_many(conn, "amino_acid_composition", _COMPOSITION_COLUMNS, batch_compositions, cache_dir)
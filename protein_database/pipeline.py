"""Pipeline orchestrator: FASTA parse -> feature extraction -> SQLite storage."""

from pathlib import Path

from sqlalchemy.orm import Session
from tqdm import tqdm

from protein_database.database.connection import create_database_engine, create_session_factory, init_database
from protein_database.database.models import (
    AminoAcidComposition as CompositionModel,
    Protein as ProteinModel,
    SequenceFeature as SequenceFeatureModel,
)
from protein_database.features.composition import compute_amino_acid_composition
from protein_database.features.sequence_features import compute_sequence_features
from protein_database.parser.fasta_parser import FastaRecord, count_fasta_entries, parse_fasta


def process_fasta_file(
    fasta_path: str | Path,
    db_path: str | Path = "proteins.db",
    batch_size: int = 1000,
    limit: int | None = None,
) -> dict:
    """Process a FASTA file and populate the SQLite database with features.

    This function:
    1. Opens the FASTA file and counts the total number of entries.
    2. Parses each protein record.
    3. Computes sequence features and amino acid composition.
    4. Inserts everything into SQLite in batches.
    5. Returns a summary dict with processing statistics.

    Args:
        fasta_path: Path to the input FASTA file.
        db_path: Path to the SQLite database file.
        batch_size: Number of records to batch-commit to the database.
        limit: Optional maximum number of records to process (for testing).

    Returns:
        Dict with keys: 'processed', 'total_entries', 'elapsed_seconds',
        'database_path'.
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
    engine = create_database_engine(db_path)
    init_database(engine)
    session_factory = create_session_factory(engine)

    processed = 0
    errors = 0

    with session_factory() as session:
        batch: list[tuple[ProteinModel, SequenceFeatureModel, CompositionModel]] = []

        progress_bar = tqdm(parse_fasta(fasta_path), total=total_entries, desc="Processing", unit="protein")
        for record in progress_bar:
            try:
                protein, features, composition = _extract_and_build_models(
                    record, source_file
                )
                batch.append((protein, features, composition))
                processed += 1

                if len(batch) >= batch_size:
                    _flush_batch(session, batch)
                    batch = []

            except Exception as exc:
                errors += 1
                progress_bar.set_postfix({"errors": errors})

            if limit is not None and processed >= limit:
                break

        # Flush any remaining batch
        if batch:
            _flush_batch(session, batch)

    elapsed = time.time() - start_time
    progress_bar.close()

    return {
        "processed": processed,
        "total_entries": total_entries,
        "errors": errors,
        "elapsed_seconds": elapsed,
        "database_path": str(db_path),
    }


def _extract_and_build_models(
    record: FastaRecord, source_file: str
) -> tuple[ProteinModel, SequenceFeatureModel, CompositionModel]:
    """Compute features for a record and build ORM model instances.

    Args:
        record: Parsed FASTA record.
        source_file: Source FASTA filename.

    Returns:
        Tuple of (Protein, SequenceFeature, AminoAcidComposition) models.
    """
    # Compute features
    features = compute_sequence_features(record.sequence)
    composition = compute_amino_acid_composition(record.sequence)

    # Build protein model
    protein = ProteinModel(
        fasta_id=record.fasta_id,
        description=record.description,
        sequence=record.sequence,
        sequence_length=record.sequence_length,
        source_file=source_file,
    )

    # Build sequence features model
    seq_feature = SequenceFeatureModel(
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


def _flush_batch(
    session: Session,
    batch: list[tuple[ProteinModel, SequenceFeatureModel, CompositionModel]],
) -> None:
    """Add a batch of models to the session and commit.

    Args:
        session: SQLAlchemy Session.
        batch: List of (Protein, SequenceFeature, Composition) tuples.
    """
    for protein, seq_feature, composition in batch:
        session.add(protein)
        session.flush()  # Get the protein.id

        seq_feature.protein_id = protein.id
        composition.protein_id = protein.id

        session.add(seq_feature)
        session.add(composition)

    session.commit()
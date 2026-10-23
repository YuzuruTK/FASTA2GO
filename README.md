# Protein Sequence Feature Database

A Python application that parses protein FASTA files, extracts computational features from amino acid sequences, and stores the results in a DuckDB database for future analysis.

## Overview

This project creates a dataset of protein sequence features that can be used for validation, filtering, clustering, and machine learning analysis. The system does **not** generate proteins — it only analyzes existing protein sequences.

## Features

- **Streaming FASTA parser** — handles very large FASTA files (e.g., 275 MB Swiss-Prot) without loading everything into memory
- **Physicochemical feature extraction** using Biopython:
  - Molecular weight
  - Theoretical isoelectric point (pI)
  - GRAVY hydropathy index
  - Instability index
  - Aliphatic index
  - Aromaticity
- **Amino acid composition** — percentage of all 20 standard amino acids
- **Chemical classification** — hydrophobic, polar, charged, sulfur, and aromatic residue percentages
- **Sequence complexity** — Shannon entropy and amino acid diversity
- **DuckDB storage** — native, columnar, analytics-optimized
- **Bulk insert pipeline** — CSV-staged native DuckDB import (fast `read_csv` batch loading)
- **UniProt annotation enrichment** — one streaming download of the reviewed proteome, ingested natively by DuckDB (Function, protein/gene names, subcellular location, organism, keywords)
- **Keyword classification** — scores protein function text against a curated label dictionary
- **CLI with progress bar** — shows percentage, ETA, and processing rate

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd tcc

# Install dependencies with uv
uv sync
```

## Usage

### Process a FASTA file

```bash
# Basic usage (skips proteins already in the database)
uv run python main.py --input uniprot_sprot.fasta

# Custom database path
uv run python main.py --input uniprot_sprot.fasta --db my_proteins.duckdb

# Process only the first 1000 proteins (for testing)
uv run python main.py --input uniprot_sprot.fasta --limit 1000

# Custom batch size
uv run python main.py --input uniprot_sprot.fasta --batch-size 500

# Force re-processing from scratch (drops the database first)
uv run python main.py --input uniprot_sprot.fasta --overwrite
```

### CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input` | `-i` | Path to input FASTA file | *(required)* |
| `--db` | `-d` | Path to output DuckDB database | `proteins.duckdb` |
| `--batch-size` | `-b` | Records per database batch insert | `1000` |
| `--limit` | `-l` | Max proteins to process (testing) | `None` (all) |
| `--overwrite` | | Drop and recreate the database before processing | `False` |
| `--csv-cache-dir` | | Directory to stage batch CSVs before the native import | `cache/csv_bulk` |

### Query the database

```bash
# Show database summary
uv run python scripts/query_example.py --summary

# Show the longest proteins
uv run python scripts/query_example.py --longest --top 20

# Show the most recently added proteins with features
uv run python scripts/query_example.py --top 10
```

### Enrich with UniProt annotations

The reviewed proteome is downloaded once as a gzipped TSV and ingested
natively by DuckDB — no per-accession requests.

```bash
# Download the snapshot (cached) and annotate every matching protein
uv run python scripts/annotate_proteins.py --db proteins.duckdb

# Offline replay of local TSV files only (no network at all)
uv run python scripts/annotate_proteins.py --db proteins.duckdb --source cache

# Force a fresh snapshot download even when one is cached
uv run python scripts/annotate_proteins.py --db proteins.duckdb --refresh-snapshot

# Custom snapshot/cache directory
uv run python scripts/annotate_proteins.py --db proteins.duckdb --cache-dir cache/uniprot
```

| Option | Description | Default |
|--------|-------------|---------|
| `--db` | Path to the DuckDB database | `proteins.duckdb` |
| `--cache-dir` | Directory for the snapshot and any local TSVs | `cache/uniprot` |
| `--source` | `bulk` (download + ingest) or `cache` (local TSVs only) | `bulk` |
| `--refresh-snapshot` | Re-download the snapshot even when cached | `False` |

### Classify protein functions

```bash
# Score every function text against the curated label dictionary
uv run python scripts/classify_proteins.py

# Lower the score threshold and export per-protein assignments
uv run python scripts/classify_proteins.py --threshold 0.5 \
    --export-assignments exports/protein_classes.csv \
    --export-unclassified exports/unclassified.csv
```

### Analyse function word frequencies

```bash
# Top words across a sample of the function column
uv run python scripts/word_count_test.py --limit 1000 --top 30

# Export the top-N most frequent words for label building
uv run python scripts/word_count_test.py --export-top 1000 \
    --export-top-out data/word_frequencies.csv
```

### Correlate database features

`correlations.py` is the single home for every correlation in this project:
term co-occurrence, amino-acid composition vs terms, sequence features vs
terms, and sequence features vs themselves.

```bash
# Build every matrix and render every heatmap
uv run python scripts/correlations.py

# One family at a time
uv run python scripts/correlations.py terms       --top 50
uv run python scripts/correlations.py amino-acids --kind phrases
uv run python scripts/correlations.py features    # z-scored features (default)
uv run python scripts/correlations.py features --no-scale
uv run python scripts/correlations.py plot --dpi 300    # render only

# Use another database
uv run python scripts/correlations.py --db my_proteins.duckdb
```

`features` writes `exports/correlations/features_vs_words.csv` and
`features_vs_phrases.csv`: the 15 numeric `sequence_features` columns
(z-scored, so Da, percentages, entropy and 0..1 properties sit on one common
axis) correlated against the top-50 word and phrase presence vectors.
Pearson is affine-invariant, so `--no-scale` returns the same coefficients
from the raw values; scaling only makes the feature magnitudes comparable.

Every matrix lands in `exports/correlations/` and its heatmap in
`exports/correlations/plots/` (including
`sequence_features_correlation.png`). The same analysis is shown inline in
`notebooks/correlations.ipynb`.

### Run tests

```bash
uv run pytest
```

## Database Schema

DuckDB is used natively (no ORM). Feature tables reference proteins via a
composite foreign key on `(fasta_id, source_file)`.

### Table: `proteins`

Stores original FASTA information. Primary key is `(fasta_id, source_file)`.

| Column | Type | Description |
|--------|------|-------------|
| `fasta_id` | VARCHAR | Original FASTA identifier (PK) |
| `source_file` | VARCHAR | Original FASTA filename (PK) |
| `description` | VARCHAR | FASTA description/header |
| `sequence` | VARCHAR | Amino acid sequence |
| `sequence_length` | INTEGER | Number of amino acids |
| `created_at` | TIMESTAMP | Processing timestamp |

### Table: `sequence_features`

Stores calculated physicochemical properties.

| Column | Type | Description |
|--------|------|-------------|
| `fasta_id` | VARCHAR | PK/FK → `proteins.fasta_id` |
| `source_file` | VARCHAR | PK/FK → `proteins.source_file` |
| `molecular_weight` | DOUBLE | Molecular weight (Da) |
| `isoelectric_point` | DOUBLE | Theoretical pI |
| `gravy_index` | DOUBLE | GRAVY hydropathy score |
| `instability_index` | DOUBLE | Instability index |
| `aliphatic_index` | DOUBLE | Aliphatic index |
| `aromaticity` | DOUBLE | Aromaticity |
| `charge_positive_percentage` | DOUBLE | % K, R, H |
| `charge_negative_percentage` | DOUBLE | % D, E |
| `hydrophobic_percentage` | DOUBLE | % A, V, I, L, M, F, W, Y |
| `polar_percentage` | DOUBLE | % S, T, N, Q, C |
| `charged_percentage` | DOUBLE | % K, R, H, D, E |
| `sulfur_percentage` | DOUBLE | % C, M |
| `aromatic_percentage` | DOUBLE | % F, W, Y |
| `sequence_entropy` | DOUBLE | Shannon entropy (bits) |
| `amino_acid_diversity` | DOUBLE | Fraction of 20 AAs present |

### Table: `amino_acid_composition`

Stores amino acid frequency percentages (one column per amino acid).

| Column | Description |
|--------|-------------|
| `fasta_id` | PK/FK → `proteins.fasta_id` |
| `source_file` | PK/FK → `proteins.source_file` |
| `alanine_percentage` | % Alanine |
| `arginine_percentage` | % Arginine |
| ... | ... (20 columns total) |
| `valine_percentage` | % Valine |

### Table: `uniprot_annotations`

Stores UniProt annotations ingested from the reviewed-proteome snapshot, one
row per local protein. Primary key is `(fasta_id, source_file)`.

| Column | Type | Description |
|--------|------|-------------|
| `fasta_id` | VARCHAR | PK/FK → `proteins.fasta_id` |
| `source_file` | VARCHAR | PK/FK → `proteins.source_file` |
| `uniprot_accession` | VARCHAR | UniProt accession |
| `uniprot_id` | VARCHAR | UniProt entry name/ID |
| `protein_name` | VARCHAR | Protein names |
| `gene_name` | VARCHAR | Gene names |
| `function` | VARCHAR | Function annotation (may be NULL) |
| `subcellular_location` | VARCHAR | Subcellular location |
| `protein_existence` | VARCHAR | Protein existence |
| `organism` | VARCHAR | Organism scientific name |
| `keywords` | VARCHAR | Keywords (semicolon-joined) |
| `status` | VARCHAR | `fetched` or `missing_function` |
| `retrieved_at` | TIMESTAMP | Retrieval timestamp |

## Project Structure

```
FASTA2GO/
├── main.py                         # CLI entry point (FASTA → features → DuckDB)
├── pyproject.toml                  # UV project configuration
├── README.md                       # This file
├── protein_database/
│   ├── __init__.py
│   ├── schema.py                   # Column lists, DDL and analysis view (single source)
│   ├── fasta.py                    # Streaming FASTA parser
│   ├── features.py                 # Physicochemical features + amino-acid composition
│   ├── db.py                       # DuckDB connect / init / bulk insert
│   ├── pipeline.py                 # Orchestrates parse → extract → store
│   └── uniprot.py                  # Snapshot download + native DuckDB ingest
├── scripts/
│   ├── lexicon.py                  # Shared word lists + tokenizer
│   ├── build_functional_labels.py  # Build the curated label dictionary
│   ├── classify_proteins.py        # Score proteins against the label dictionary
│   ├── word_count_test.py          # Word-frequency analysis / export
│   ├── verify_completeness.py      # Data completeness report
│   ├── query_example.py            # Example DuckDB queries
│   └── annotate_proteins.py        # UniProt enrichment CLI
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_parser.py
│   ├── test_features.py
│   ├── test_composition.py
│   ├── test_pipeline.py
│   ├── test_uniprot.py
│   ├── test_functional_labels.py
│   └── test_classify_proteins.py
└── uv.lock                         # Locked dependencies
```

## Idempotent Processing

Running `main.py` multiple times on the same FASTA file is **safe** — the pipeline
automatically skips proteins that already exist in the database (identified by
`fasta_id` + `source_file`). This prevents duplicate records.

- **First run**: processes all proteins
- **Second run**: skips all previously processed proteins (reports `skipped` count)
- **Partial run** (e.g., `--limit 1000` then full run): only processes the remaining proteins
- **`--overwrite` flag**: drops and recreates the database before processing, forcing a full re-run

## Processing Pipeline

```
FASTA file
    |
    v
FASTA Parser (streaming)
    |
    v
Protein Feature Extractor
    |
    v
DuckDB Database
```

## Future Compatibility

The database schema uses composite `(fasta_id, source_file)` foreign keys across all feature tables, allowing future additions without breaking existing data:

- UniProt metadata
- BLAST similarity results
- Pfam domain annotations
- AlphaFold structure predictions
- Protein embeddings
- Machine learning classification models

## License

This project is for academic/research purposes.
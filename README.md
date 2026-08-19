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
- **Sequence complexity** — Shannon entropy, amino acid diversity, low-complexity region detection
- **DuckDB storage** — native, columnar, analytics-optimized
- **Bulk insert pipeline** — fast `executemany` batch loading
- **UniProt annotation enrichment** — bulk-retrieves Function, protein/gene names, subcellular location, organism, keywords, and more from the UniProtKB REST API, with resumable batches, raw-response caching, and retry/backoff
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

```bash
# Enrich all proteins (batched, resumable)
uv run python scripts/annotate_proteins.py --db proteins.duckdb

# Custom batch size and cache directory
uv run python scripts/annotate_proteins.py --db proteins.duckdb --batch-size 200 --cache-dir cache/uniprot

# Process only a limited number of accessions (for testing)
uv run python scripts/annotate_proteins.py --db proteins.duckdb --limit 5000
```

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

Stores UniProt annotations fetched from the UniProtKB REST API, one row per
local protein. Primary key is `(fasta_id, source_file)`.

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

### Table: `uniprot_accession_queue`

Tracks the deduplicated UniProt accessions to fetch and their processing
state (used for resumable batch retrieval).

| Column | Type | Description |
|--------|------|-------------|
| `accession` | VARCHAR | PK, UniProt accession |
| `status` | VARCHAR | `pending`, `completed`, `missing_function`, `not_found`, or `failed` |
| `retry_count` | INTEGER | Number of failed attempts |
| `last_attempt` | TIMESTAMP | Last processing attempt |
| `created_at` | TIMESTAMP | Queue insertion time |
| `completed_at` | TIMESTAMP | When the accession finished processing |

## Project Structure

```
tcc/
├── main.py                      # CLI entry point
├── pyproject.toml               # UV project configuration
├── README.md                    # This file
├── protein_database/
│   ├── __init__.py
│   ├── pipeline.py              # Orchestrates parse → extract → store
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py            # Dataclasses + DuckDB schema DDL
│   │   └── connection.py        # DuckDB connection management
│   ├── parser/
│   │   ├── __init__.py
│   │   └── fasta_parser.py      # Streaming FASTA parser
│   ├── features/
│   │   ├── __init__.py
│   │   ├── sequence_features.py # MW, pI, GRAVY, instability, etc.
│   │   ├── composition.py       # AA percentages + chemical classes
│   │   └── complexity.py        # Shannon entropy, diversity, low-complexity
│   └── uniprot/
│       ├── __init__.py
│       ├── accession.py         # FASTA header → UniProt accession
│       ├── batching.py          # Chunk accessions into queries
│       ├── client.py            # UniProtKB REST client + retry/backoff
│       ├── cache.py             # Raw response cache
│       ├── parser.py            # TSV response parsing
│       └── annotate.py          # Enrichment orchestrator + resume
├── scripts/
│   ├── query_example.py         # Example DuckDB query script
│   └── annotate_proteins.py     # UniProt enrichment CLI
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_parser.py
│   ├── test_features.py
│   ├── test_composition.py
│   ├── test_complexity.py
│   ├── test_pipeline.py
│   ├── test_uniprot_accession.py
│   ├── test_uniprot_batching.py
│   ├── test_uniprot_parser.py
│   ├── test_uniprot_client.py
│   └── test_uniprot_annotate.py
└── uv.lock                      # Locked dependencies
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
# Protein Sequence Feature Database

A Python application that parses protein FASTA files, extracts computational features from amino acid sequences, and stores the results in a SQLite database for future analysis.

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
- **SQLite storage** via SQLAlchemy ORM
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
# Basic usage
uv run python main.py --input uniprot_sprot.fasta

# Custom database path
uv run python main.py --input uniprot_sprot.fasta --db my_proteins.db

# Process only the first 1000 proteins (for testing)
uv run python main.py --input uniprot_sprot.fasta --limit 1000

# Custom batch size
uv run python main.py --input uniprot_sprot.fasta --batch-size 500
```

### CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input` | `-i` | Path to input FASTA file | *(required)* |
| `--db` | `-d` | Path to output SQLite database | `proteins.db` |
| `--batch-size` | `-b` | Records per database commit | `1000` |
| `--limit` | `-l` | Max proteins to process (testing) | `None` (all) |

### Query the database

```bash
# Show database summary
uv run python scripts/query_example.py --summary

# Show the longest proteins
uv run python scripts/query_example.py --longest --top 20

# Show the most recently added proteins with features
uv run python scripts/query_example.py --top 10
```

### Run tests

```bash
uv run pytest
```

## Database Schema

### Table: `proteins`

Stores original FASTA information.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-incrementing ID |
| `fasta_id` | TEXT | Original FASTA identifier |
| `description` | TEXT | FASTA description/header |
| `sequence` | TEXT | Amino acid sequence |
| `sequence_length` | INTEGER | Number of amino acids |
| `source_file` | TEXT | Original FASTA filename |
| `created_at` | DATETIME | Processing timestamp |

### Table: `sequence_features`

Stores calculated physicochemical properties.

| Column | Type | Description |
|--------|------|-------------|
| `protein_id` | INTEGER FK | References `proteins.id` |
| `molecular_weight` | FLOAT | Molecular weight (Da) |
| `isoelectric_point` | FLOAT | Theoretical pI |
| `gravy_index` | FLOAT | GRAVY hydropathy score |
| `instability_index` | FLOAT | Instability index |
| `aliphatic_index` | FLOAT | Aliphatic index |
| `aromaticity` | FLOAT | Aromaticity |
| `charge_positive_percentage` | FLOAT | % K, R, H |
| `charge_negative_percentage` | FLOAT | % D, E |
| `hydrophobic_percentage` | FLOAT | % A, V, I, L, M, F, W, Y |
| `polar_percentage` | FLOAT | % S, T, N, Q, C |
| `charged_percentage` | FLOAT | % K, R, H, D, E |
| `sulfur_percentage` | FLOAT | % C, M |
| `aromatic_percentage` | FLOAT | % F, W, Y |
| `sequence_entropy` | FLOAT | Shannon entropy (bits) |
| `amino_acid_diversity` | FLOAT | Fraction of 20 AAs present |

### Table: `amino_acid_composition`

Stores amino acid frequency percentages (one column per amino acid).

| Column | Description |
|--------|-------------|
| `protein_id` | References `proteins.id` |
| `alanine_percentage` | % Alanine |
| `arginine_percentage` | % Arginine |
| ... | ... (20 columns total) |
| `valine_percentage` | % Valine |

### Table: `validation_features`

Stores future validation metrics. All fields initially `NULL`.

| Column | Type | Description |
|--------|------|-------------|
| `protein_id` | INTEGER FK | References `proteins.id` |
| `sequence_quality_score` | FLOAT | Future use |
| `stability_score` | FLOAT | Future use |
| `solubility_score` | FLOAT | Future use |
| `structure_score` | FLOAT | Future use |
| `overall_score` | FLOAT | Future use |
| `classification` | TEXT | Future use |

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
│   │   ├── models.py            # SQLAlchemy ORM models
│   │   └── connection.py        # DB engine/session management
│   ├── parser/
│   │   ├── __init__.py
│   │   └── fasta_parser.py      # Streaming FASTA parser
│   └── features/
│       ├── __init__.py
│       ├── sequence_features.py # MW, pI, GRAVY, instability, etc.
│       ├── composition.py       # AA percentages + chemical classes
│       └── complexity.py        # Shannon entropy, diversity, low-complexity
├── scripts/
│   └── query_example.py         # Example database query script
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_parser.py
│   ├── test_features.py
│   ├── test_composition.py
│   └── test_complexity.py
└── uv.lock                      # Locked dependencies
```

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
SQLite Database
```

## Future Compatibility

The database schema uses `protein_id` foreign keys across all feature tables, allowing future additions without breaking existing data:

- UniProt metadata
- BLAST similarity results
- Pfam domain annotations
- AlphaFold structure predictions
- Protein embeddings
- Machine learning classification models

## License

This project is for academic/research purposes.
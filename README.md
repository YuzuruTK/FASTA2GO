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
- **UniProt annotation enrichment** — one streaming download of the reviewed proteome ingested natively by DuckDB, plus an incremental `--source api` mode for missing/stale accessions (Function, protein/gene names, subcellular location, organism, keywords)
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

Three modes, all writing through the same staging table and `INSERT … ON
CONFLICT` upsert:

- **bulk** (default) — one streaming request for the whole reviewed proteome
  (gzipped TSV, cached), ingested natively by DuckDB.
- **cache** — replay local TSV files only, no network.
- **api** — fetch just the accessions that are missing or stale from the
  `/uniprotkb/accessions` endpoint (1000 per request, retries on 429/5xx).

```bash
# Download the snapshot (cached) and annotate every matching protein
uv run python scripts/annotate_proteins.py --db proteins.duckdb

# Offline replay of local TSV files only (no network at all)
uv run python scripts/annotate_proteins.py --db proteins.duckdb --source cache

# Force a fresh snapshot download even when one is cached
uv run python scripts/annotate_proteins.py --db proteins.duckdb --refresh-snapshot

# Incremental: only proteins with no annotation row yet
uv run python scripts/annotate_proteins.py --db proteins.duckdb --source api --only-missing

# Refresh annotations older than 7 days (missing rows always qualify)
uv run python scripts/annotate_proteins.py --db proteins.duckdb --source api --older-than-days 7

# Custom snapshot/cache directory
uv run python scripts/annotate_proteins.py --db proteins.duckdb --cache-dir cache/uniprot
```

| Option | Description | Default |
|--------|-------------|---------|
| `--db` | Path to the DuckDB database | `proteins.duckdb` |
| `--cache-dir` | Directory for the snapshot, staging files and local TSVs | `cache/uniprot` |
| `--source` | `bulk` (download + ingest), `cache` (local TSVs only) or `api` (missing/stale only) | `bulk` |
| `--refresh-snapshot` | Re-download the snapshot even when cached | `False` |
| `--only-missing` | API mode: only proteins without an annotation row | `False` |
| `--older-than-days` | API mode: also refresh annotations older than N days | `None` |
| `--limit` | API mode: cap the number of local proteins selected | `None` |
| `--chunk-size` | API mode: accessions per request (max 1000) | `1000` |
| `--max-retries` / `--base-delay` | API mode: retry count and initial backoff | `5` / `1.0` |

Requests are sequential: one batch of up to 1000 accessions is one HTTP call,
so concurrency was removed. A network error, an unexpected HTTP status or an
empty body is reported and never written as a missing annotation: the proteins
keep their current annotation, the command exits non-zero and the same command
retries them. Accessions UniProt answers for but does not return are stored
with `status = 'not_found', so `--only-missing` skips them;
`--older-than-days` retries them later.


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

### Build the GO Slim Molecular Function labels

```bash
# Download the pinned ontology + slim (fails if a SHA-256 does not match)
uv run python scripts/build_go_slim_mapping.py --fetch

# Map an auditable sample; the full protein base is not built yet
uv run python scripts/build_go_slim_mapping.py --sample-size 400
```

Writes the frozen vocabulary to `data/go_slim/vocabulary_mf.csv`, the sample
labels to `exports/go_slim/sample/protein_labels.csv` and an auditable
`exports/go_slim/sample/manifest.json` (sources, hashes, rules, counts,
coverage, limitations). Re-running with the same sources is byte-identical.

### Run the experiment

Everything that selects, tunes or ranks runs on the development rows; the
feature subsets are built there and then scored with every model on the same
folds.

```bash
# inspect the data and the split plan without fitting anything
uv run python scripts/run_experiment.py --limit 50 --dry-run

# compare the four strategies with one model per family
uv run python scripts/run_experiment.py --out exports/experiments/sample

# score one chosen configuration on the external test block, once
uv run python scripts/run_experiment.py --out exports/experiments/sample \
    --evaluate-test --test-strategy all_features --test-model logistic_ovr
```

See [Experiment flow](#experiment-flow-project-extension) for the steps it runs
and for what the external-test guarantee actually covers.

### Run tests

```bash
uv run pytest
```

## GO Slim labels (project extension)

The classification target is a **Molecular Function** label space derived from
the Gene Ontology, not a free-text keyword score.

- **Vocabulary**: the official generic GO slim (`goslim_generic`) **plus one
  documented extension, `GO:0005488 binding`**, giving 41 MF terms. It is
  therefore **not** the unmodified official slim, and every artifact says so
  (`is_unmodified_official_slim: false`).
- **Why the extension**: `goslim_generic` omits `binding` while keeping its
  sibling `catalytic activity`. Without it, 296 annotation instances in a
  400-protein sample reached no slim ancestor (ATP binding alone in 77/400
  proteins) and 22/400 proteins had no label at all; with it, 6 instances and
  0/400.
- **Root excluded**: `GO:0003674 molecular_function` is an ancestor of every MF
  term, so with all-ancestor propagation it would become a constant positive
  label. It is excluded by rule, never by a training/evaluation mask.
- **Propagation**: every applicable ancestor through `is_a` and `part_of` (not
  only the nearest), the term itself included, restricted to the 41-term
  vocabulary. `alt_id` is resolved to its primary id, obsolete and unknown terms
  are excluded and counted, and `(protein, label)` pairs are deduplicated.
- **Aspect guard**: the traversal is restricted to the vocabulary aspect. A
  BP/CC node is never expanded, so a cross-aspect path can neither *become* a
  label nor *lead back* to a Molecular Function label. This is structural, not a
  property of the current file: the pinned `go-basic.obo` happens to contain
  **zero cross-namespace edges** among its `is_a`/`part_of` links (audited over
  all 38,092 live terms), but another release or the unfiltered `go.obo` could
  contain them, and the result must not depend on which file is in use.
- **Obsolete terms, replacements and alt_ids**: an annotation to an obsolete
  term is counted as `obsolete` and produces no label; its `replaced_by` is
  **not** followed (silently substituting a term would invent a label the
  curator never assigned). An `alt_id` is resolved to its primary term before
  traversal, and an `alt_id` of an obsolete term is obsolete too. An obsolete
  term terminates the walk: it is neither traversed nor a label.
- **Provenance**: `data/go_slim/sources.json` pins the release and the SHA-256
  of both `.obo` files. The 30 MB `go-basic.obo` is **not** versioned in git; it
  lives in `cache/go_slim/` and any hash mismatch is a hard error. The slim file
  is checked against the `subset: goslim_generic` signature inside the ontology
  (140 terms each, symmetric difference empty).
- **Accounting**: annotations are partitioned into `mapped`,
  `no_vocabulary_ancestor`, `obsolete` and `unknown`, and the counts must add up
  to the annotation total; a protein with no vocabulary label is reported, never
  silently dropped or treated as all-negative.
- **Limitations**: only MF annotations present in the UniProt snapshot are used
  (absence of an annotation is not evidence of absence of the function); the
  sample is a deterministic hash-based slice, **not** a population estimate; the
  propagation rules above are the only relations considered.

## Model comparison (project extension)

`protein_database/ml/model_comparison.py` scores several model families on
**fixed** feature subsets under one evaluation policy. It is infrastructure
only: no experiment on the real base has been run, and the external test block
has never been read.

### Supported models

| `model_id` | Pipeline | Scaling | Status |
| --- | --- | --- | --- |
| `logistic_ovr` (`LogisticConfig`) | `StandardScaler` → `OneVsRestClassifier(LogisticRegression)` | yes | default, frozen reference |
| `random_forest_ovr` (`RandomForestConfig`) | `OneVsRestClassifier(RandomForestClassifier)` | none | supported |
| `xgboost_ovr` (`XGBoostConfig`) | `OneVsRestClassifier(XGBClassifier)` | none | **optional dependency, not installed** |

- All three wrap the classifier in `OneVsRestClassifier`, so the target encoding
  (one binary task per label) is identical and only the model varies.
- `LogisticConfig` and `build_estimator()` keep exactly the previous defaults
  (`C=1.0`, `liblinear`, `max_iter=1000`, `tol=1e-4`, `random_state=42`); the
  logistic pipeline is unchanged.
- `RandomForestConfig` exposes `n_estimators`, `max_depth`,
  `min_samples_split`, `min_samples_leaf`, `max_features`, `class_weight`,
  `n_jobs` and `random_state`. No scaler is inserted (trees are invariant to
  monotone rescaling). `n_jobs` parallelises labels only; each forest stays
  sequential, and a fixed `random_state` makes a run reproducible.
- `XGBoostConfig` **requires `xgboost` to be installed separately**
  (`uv add xgboost`); it is not a project dependency and is absent from
  `pyproject.toml`/`uv.lock`. Requesting it without the package raises
  `MissingDependencyError` instead of silently substituting another model.
- `describe_estimator(config)` records the effective configuration, including
  `model_id`, the pipeline type, whether a scaler is used, the 0.5 decision
  threshold and, for XGBoost, whether the dependency was found.

### Evaluation conventions

`compare_models(dataset, plan, subsets=..., models=...)` reuses the selection
evaluator, so every number follows the same rules:

- the validation folds of the shared `SplitPlan` (`fold_0`, `fold_1`, ...); the
  test block is never indexed;
- the same label masks (`fit`: ≥2 positives and ≥2 negatives; `eval`: ≥1
  positive) intersected per fold, with excluded labels recorded per fold;
- the same macro-F1 implementation and the same fixed 0.5 threshold;
- any transformation fitted inside the fold's training block only;
- macro-F1 per fold, plus mean and population standard deviation over folds;
- two separate cost counters: `n_estimator_fits` (pipelines fitted, one per
  fold) and `n_binary_fits` (binary models trained inside OvR, per label and
  fold), so the two are never conflated. Wall-clock time and memory are **not**
  measured: including them would make the report non-reproducible;
- failures recorded, not hidden: a missing dependency or a crash becomes a row
  with `error` (or is re-raised with `on_error="raise"`), and non-convergence is
  reported through `converged` plus `non_converged()`.

Feature subsets are supplied by the caller. The experiment flow builds them
with `strategies`/`feature_selection` and passes `{strategy: features}` to
`compare_models`, so the subset stays fixed while the model varies and no
per-model feature search is done.

The report (`ModelComparison.as_dict()` / `.to_json()`) is a table, not a
ranking: it has no "best"/"winner" notion, because no significance test exists
yet. Differences between models on synthetic data only prove that the
infrastructure works; they say **nothing** about which family predicts GO terms
better on real proteins, and no superiority claim may be made from them.

### Provisional parameters (pending scientific consolidation)

The following are placeholders chosen for reproducibility, not tuned values:
`C=1.0` and `max_iter=1000` (logistic), `n_estimators=200`, `max_depth=None`,
`max_features="sqrt"` (forest), `n_estimators=200`, `max_depth=6`,
`learning_rate=0.3` (xgboost), `class_weight=None` as the default of every
family, the 0.5 threshold, and the simple mean over folds. Class weighting is
available in the model configurations, but it is not chosen by default and it
is **not** evaluated yet. Supervised filtering (`SelectKBest`, chi²) is not
implemented.

## Experiment flow (project extension)

`protein_database/ml/experiments.py` is the single flow: build the feature
subsets on the development rows, then score every (strategy, model) pair on the
same folds. No real run has been performed; every number produced so far comes
from synthetic data in the test suite.

### What runs, and in which order

1. `development_block(dataset, plan)` copies the development rows and re-indexes
   the folds onto the copy. Selection and comparison only ever see that copy.
2. Each requested strategy produces one subset: `all_features`,
   `variance_prefilter` (variance filter fitted on each fold training block and
   intersected over folds), `forward`, `backward`.
3. Every subset is scored with every model through
   `model_comparison.compare_models`, so folds, label masks, estimator
   parameters and the macro-F1 implementation are identical across the table.
4. The table is printed and, with `--out`, written as `results.json` plus
   `results.csv`.

Defaults: one configuration per family (`LogisticConfig`,
`RandomForestConfig`, `XGBoostConfig`), all four strategies, `max_features=10`,
`min_features=2` for backward elimination, seed 42. Every parameter stays
configurable through `ExperimentSettings` and the CLI flags.

### The external test block

- `development_block` physically removes the test rows before any selection, so
  the selection and comparison code cannot read them: there is nothing to guard
  and nothing to intercept.
- `evaluate_external(dataset, plan, features, model)` is the only function that
  indexes `plan.test`, and `run_experiment` never calls it. It fits on the
  development rows and scores the test block once.
- The CLI evaluates it only with `--evaluate-test --test-strategy ...
  --test-model ...`: the configuration is chosen on development numbers and the
  script never picks one automatically.
- Honest limit: `validate_dataset` checks shapes, finiteness and duplicate keys
  on the full matrix **before** the split, so it does inspect test values; it
  never uses them to choose, rank or tune anything.

### Result reuse

`results.json` stores the settings, the sources and a SHA-256 of the effective
`X` and `y` values plus the settings. A later run reuses a stored strategy only
when that fingerprint matches, so changing features, labels, sample or
parameters starts a new run instead of mixing rows.

### Grouping

Groups come from `dataset.load_sequence_groups`, that is `md5(sequence)` in the
database. **Only identical sequences are grouped**: homologous proteins whose
sequences differ are not clustered together, and no alignment or clustering is
performed. `validate_split_plan` re-checks that no fold shares a sample or a
group with its training block and that the external test block shares neither
with any fold.

### Running it

```bash
# inspect the data and the plan without fitting anything
uv run python scripts/run_experiment.py --limit 50 --dry-run

# compare strategies and models on the development rows
uv run python scripts/run_experiment.py --out exports/experiments/sample

# score one chosen configuration on the external test block, once
uv run python scripts/run_experiment.py --out exports/experiments/sample \
    --evaluate-test --test-strategy all_features --test-model logistic_ovr
```

The table is a comparison, not a ranking: no significance test exists, so no
best-model claim is produced. XGBoost rows are reported as errors while the
optional package is absent; the model is never replaced by another estimator.

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
| `status` | VARCHAR | `fetched`, `missing_function` or `not_found` |
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
│   ├── uniprot.py                  # Snapshot download + native DuckDB ingest
│   └── ml/
│       ├── evaluation.py           # Label masks + multilabel metrics
│       ├── dataset.py              # Feature/label loading, alignment, validation
│       ├── splits.py               # Grouped split plan + leakage guards
│       ├── prefilter.py            # Per-fold variance filter + correlation report
│       ├── models.py               # Model configs (logistic, forest, optional xgboost)
│       ├── feature_selection.py    # Forward selection / backward elimination
│       ├── strategies.py           # Strategy names (all, variance, forward, backward)
│       ├── model_comparison.py     # Model x feature-subset comparison report
│       ├── experiments.py          # Development-only experiment flow + results
│       └── go_slim.py              # GO Slim MF vocabulary, propagation, mapping
├── scripts/
│   ├── lexicon.py                  # Shared word lists + tokenizer
│   ├── build_functional_labels.py  # Build the curated label dictionary
│   ├── classify_proteins.py        # Score proteins against the label dictionary
│   ├── word_count_test.py          # Word-frequency analysis / export
│   ├── verify_completeness.py      # Data completeness report
│   ├── query_example.py            # Example DuckDB queries
│   ├── annotate_proteins.py        # UniProt enrichment CLI
│   ├── run_experiment.py           # Feature-selection x model comparison CLI
│   └── build_go_slim_mapping.py    # GO Slim MF labels (auditable sample)
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
├── data/go_slim/                   # Pinned GO sources + frozen 41-term vocabulary
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
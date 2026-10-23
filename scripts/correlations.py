#!/usr/bin/env python3
"""Every correlation used in the FASTA2GO keyword research, in one file.

The matrices are all built from the same two ingredients: a numeric block
(amino-acid percentages or sequence features) and a binary block (presence of
the top keyword terms), streamed out of DuckDB so the full table never has to
be materialized.

Matrices produced
-----------------
* **Phi coefficient** -- Pearson correlation of two binary presence vectors.
  ``+1`` = always together, ``-1`` = mutually exclusive, ``0`` = independent.
* **Jaccard** -- ``|A n B| / |A u B|``, pure overlap regardless of prevalence.
* **Amino acids vs terms** -- point-biserial Pearson ``r`` between each
  amino-acid percentage and term presence.
* **Sequence features vs terms** -- the same point-biserial ``r`` for the 15
  numeric ``sequence_features`` columns. The features are standardized
  (z-scored) first, so properties reported in wildly different units
  (molecular weight in Da, aromaticity in 0..1, percentages in 0..100) are
  comparable. Pearson is affine-invariant, so z-scoring leaves every
  coefficient unchanged; ``--no-scale`` reproduces the raw matrix exactly.
* **Sequence features vs sequence features** -- Pearson ``r`` across the
  numeric columns of ``sequence_features``.

Prerequisites: run ``scripts/build_keyword_terms.py`` first so
``protein_keyword_terms`` exists in ``proteins.duckdb``.

Usage:
    uv run python scripts/correlations.py                 # everything
    uv run python scripts/correlations.py terms --top 50
    uv run python scripts/correlations.py amino-acids --kind phrases
    uv run python scripts/correlations.py features --no-scale
    uv run python scripts/correlations.py plot --dpi 300
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterator

import matplotlib

if "IPython" not in sys.modules:  # keep notebook backends untouched
    matplotlib.use("Agg")  # headless rendering

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from protein_database.db import connect
from protein_database.schema import AMINO_ACIDS, COMPOSITION_COLUMNS, FEATURE_COLUMNS

DEFAULT_DB = PROJECT_ROOT / "proteins.duckdb"
DEFAULT_INPUT = PROJECT_ROOT / "exports" / "correlations"

TABLE = "protein_keyword_terms"
COLUMNS = {"words": "words", "phrases": "phrases"}

FEATURE_TABLE = "sequence_features"
FEATURE_MATRIX = "sequence_features_correlation"
FEATURE_ROW_LABEL = "feature"

KINDS = ["both", "words", "phrases"]


# ---------------------------------------------------------------------------
# 1. Term x term math (DB-independent, unit-tested)
# ---------------------------------------------------------------------------


def cooccurrence_matrix(presence: np.ndarray) -> np.ndarray:
    """Pairwise counts of proteins where both terms are present."""
    matrix = presence.astype(np.int32)
    return matrix.T @ matrix


def phi_matrix(presence: np.ndarray) -> np.ndarray:
    """Phi coefficient matrix from a boolean (proteins x terms) array.

    phi = (n11*N - n1*n2) / sqrt(n1*(N-n1)*n2*(N-n2))

    Terms that are always or never present have zero variance, so phi is
    undefined; those entries are reported as 0. The diagonal is 1.
    """
    n = presence.shape[0]
    co = cooccurrence_matrix(presence).astype(np.float64)
    df = presence.sum(axis=0).astype(np.float64)
    expected = np.outer(df, df)
    denominator = np.sqrt(
        np.outer(df, df) * np.outer(n - df, n - df)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = (co * n - expected) / denominator
    phi[~np.isfinite(phi)] = 0.0
    np.fill_diagonal(phi, 1.0)
    return phi


def jaccard_matrix(presence: np.ndarray) -> np.ndarray:
    """Jaccard similarity matrix from a boolean (proteins x terms) array."""
    co = cooccurrence_matrix(presence).astype(np.float64)
    df = presence.sum(axis=0).astype(np.float64)
    union = df[:, None] + df[None, :] - co
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = co / union
    jaccard[~np.isfinite(jaccard)] = 0.0
    return jaccard


# ---------------------------------------------------------------------------
# 2. Numeric x binary math (DB-independent, unit-tested)
# ---------------------------------------------------------------------------


def correlation_matrix_chunks(
    chunks: Iterator[tuple[np.ndarray, np.ndarray]] | list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Pearson correlation between columns of x and columns of y.

    ``chunks`` yields ``(x, y)`` row blocks with a fixed column layout, so
    the full table never has to be materialized. Returns a
    ``(x_columns, y_columns)`` matrix; constant columns correlate as 0.
    """
    n = 0
    sx = sy = sx2 = sy2 = None
    sxy = None

    for x, y in chunks:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if sx is None:
            sx = np.zeros(x.shape[1])
            sx2 = np.zeros(x.shape[1])
            sy = np.zeros(y.shape[1])
            sy2 = np.zeros(y.shape[1])
            sxy = np.zeros((x.shape[1], y.shape[1]))
        n += x.shape[0]
        sx += x.sum(axis=0)
        sx2 += (x * x).sum(axis=0)
        sy += y.sum(axis=0)
        sy2 += (y * y).sum(axis=0)
        sxy += x.T @ y

    numerator = n * sxy - np.outer(sx, sy)
    denominator = np.sqrt(
        np.outer(n * sx2 - sx * sx, n * sy2 - sy * sy)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = numerator / denominator
    correlation[~np.isfinite(correlation)] = 0.0
    return np.clip(correlation, -1.0, 1.0)


def correlation_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convenience wrapper for an in-memory (non-chunked) correlation."""
    return correlation_matrix_chunks([(x, y)])


# ---------------------------------------------------------------------------
# 3. Data access (streamed from DuckDB)
# ---------------------------------------------------------------------------


def top_terms(conn, kind: str, limit: int) -> list[str]:
    """Return the ``limit`` most frequent terms of one kind, deterministic."""
    column = COLUMNS[kind]
    rows = conn.execute(
        "SELECT term, count(*) AS df FROM ("
        f"SELECT unnest(string_split({column}, ' ')) AS term FROM {TABLE}"
        ") WHERE term <> '' "
        "GROUP BY term ORDER BY df DESC, term LIMIT ?",
        [limit],
    ).fetchall()
    return [term for term, _ in rows]


def presence_matrix(conn, kind: str, terms: list[str]) -> np.ndarray:
    """Return a boolean (proteins x terms) presence array."""
    column = COLUMNS[kind]
    select = ", ".join(
        f"list_contains(string_split(COALESCE({column}, ''), ' '), ?) AS t{i}"
        for i in range(len(terms))
    )
    columns = conn.execute(f"SELECT {select} FROM {TABLE}", terms).fetchnumpy()
    return np.column_stack([columns[f"t{i}"] for i in range(len(terms))])


def _presence_sql(term_count: int, column: str) -> str:
    return ", ".join(
        f"list_contains(string_split(COALESCE({column}, ''), ' '), ?) AS t{i}"
        for i in range(term_count)
    )


def _require_terms(terms: list[str]) -> None:
    if not terms:
        raise ValueError("At least one term is required.")


def amino_acid_chunks(
    conn, kind: str, terms: list[str], batch_size: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (amino-acid percentages, term presence) row blocks."""
    _require_terms(terms)
    column = COLUMNS[kind]
    amino_acids = ", ".join(f"aac.{name}" for name in COMPOSITION_COLUMNS)
    sql = (
        f"SELECT {amino_acids}, {_presence_sql(len(terms), f'k.{column}')} "
        f"FROM {TABLE} k "
        "JOIN amino_acid_composition aac USING (fasta_id, source_file)"
    )
    cursor = conn.execute(sql, terms)
    split = len(COMPOSITION_COLUMNS)
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        block = np.array(rows, dtype=np.float64)
        yield block[:, :split], block[:, split:]


def feature_chunks(conn, batch_size: int) -> Iterator[np.ndarray]:
    """Yield raw ``sequence_features`` rows as numeric batches."""
    columns = ", ".join(FEATURE_COLUMNS)
    cursor = conn.execute(f"SELECT {columns} FROM {FEATURE_TABLE}")
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            return
        yield np.asarray(rows, dtype=np.float64)


def feature_stats(conn) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and population standard deviation of the features.

    Zero-variance columns get ``std = 1.0`` so the z-score stays finite
    (and such a column correlates as 0, like any constant column).
    """
    _require_feature_rows(conn)
    aggregates = ", ".join(f"avg({column})" for column in FEATURE_COLUMNS)
    aggregates += ", " + ", ".join(
        f"stddev_pop({column})" for column in FEATURE_COLUMNS
    )
    values = conn.execute(f"SELECT {aggregates} FROM {FEATURE_TABLE}").fetchone()
    split = len(FEATURE_COLUMNS)
    mean = np.nan_to_num(
        np.asarray(values[:split], dtype=np.float64), nan=0.0
    )
    std = np.asarray(values[split:], dtype=np.float64)
    std = np.where(np.isfinite(std) & (std > 0.0), std, 1.0)
    return mean, std


def _require_feature_rows(conn) -> None:
    empty = conn.execute(f"SELECT count(*) FROM {FEATURE_TABLE}").fetchone()[0] == 0
    if empty:
        raise ValueError(f"Table '{FEATURE_TABLE}' contains no rows.")


def feature_term_chunks(
    conn,
    kind: str,
    terms: list[str],
    batch_size: int = 25_000,
    stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (features, term presence) row blocks, row-aligned by protein.

    When ``stats`` is given the feature block is standardized in SQL
    (``(value - mean) / std``), which keeps the streaming one-pass shape:
    the mean/std are computed once up front by :func:`feature_stats`.
    """
    _require_terms(terms)
    column = COLUMNS[kind]
    parameters: list[float | str] = []

    if stats is None:
        features = ", ".join(f"sf.{name}" for name in FEATURE_COLUMNS)
    else:
        mean, std = stats
        features = ", ".join(
            f"(sf.{name} - ?) / ?" for name in FEATURE_COLUMNS
        )
        for average, deviation in zip(mean, std):
            parameters.extend([float(average), float(deviation)])

    parameters.extend(terms)
    sql = (
        f"SELECT {features}, {_presence_sql(len(terms), f'k.{column}')} "
        f"FROM {TABLE} k "
        f"JOIN {FEATURE_TABLE} sf USING (fasta_id, source_file)"
    )
    cursor = conn.execute(sql, parameters)
    split = len(FEATURE_COLUMNS)
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        block = np.array(rows, dtype=np.float64)
        yield block[:, :split], block[:, split:]


# ---------------------------------------------------------------------------
# 4. Matrix builders
# ---------------------------------------------------------------------------


def term_matrix(conn, kind: str, terms: list[str], metric: str = "phi") -> np.ndarray:
    """Term x term matrix (``phi`` or ``jaccard``) for the given terms."""
    presence = presence_matrix(conn, kind, terms)
    if metric == "phi":
        return phi_matrix(presence)
    if metric == "jaccard":
        return jaccard_matrix(presence)
    raise ValueError(f"Unknown metric: {metric!r}")


def amino_acid_vs_terms(
    conn, kind: str, terms: list[str], batch_size: int = 25_000
) -> np.ndarray:
    """Point-biserial matrix: amino-acid percentages x term presence."""
    return correlation_matrix_chunks(
        amino_acid_chunks(conn, kind, terms, batch_size)
    )


def feature_vs_terms(
    conn,
    kind: str,
    terms: list[str],
    batch_size: int = 25_000,
    scaled: bool = True,
) -> np.ndarray:
    """Point-biserial matrix: sequence features x term presence.

    ``scaled=True`` standardizes every feature column (z-score) before the
    correlation. Pearson is affine-invariant, so the coefficients match the
    raw ones exactly; scaling only puts the feature magnitudes on one axis.
    """
    _require_feature_rows(conn)
    stats = feature_stats(conn) if scaled else None
    return correlation_matrix_chunks(
        feature_term_chunks(conn, kind, terms, batch_size, stats)
    )


def sequence_feature_matrix(conn, batch_size: int = 25_000) -> np.ndarray:
    """Pearson correlation matrix across the numeric sequence features."""
    _require_feature_rows(conn)
    return correlation_matrix_chunks(
        (chunk, chunk) for chunk in feature_chunks(conn, batch_size)
    )


def refresh_feature_matrix(
    db_path: Path, matrix_path: Path, batch_size: int = 25_000
) -> Path:
    """Recompute the sequence-feature matrix from DuckDB and save it."""
    conn = connect(db_path)
    try:
        values = sequence_feature_matrix(conn, batch_size)
    finally:
        conn.close()
    return write_matrix_csv(
        matrix_path,
        FEATURE_COLUMNS,
        FEATURE_COLUMNS,
        values,
        row_label=FEATURE_ROW_LABEL,
    )


# ---------------------------------------------------------------------------
# 5. Reporting (CSV writers and rankings)
# ---------------------------------------------------------------------------


def write_matrix_csv(
    path: Path,
    row_labels: list[str],
    column_labels: list[str],
    values: np.ndarray,
    row_label: str = "amino_acid",
) -> Path:
    """Write a labelled matrix CSV that :func:`load_matrix` can read back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([row_label, *column_labels])
        for label, row in zip(row_labels, values):
            writer.writerow([label, *(f"{value:+.4f}" for value in row)])
    return path


def load_matrix(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Read a labelled matrix CSV into (row labels, column labels, values)."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")
    header = rows[0]
    row_labels = [row[0] for row in rows[1:]]
    column_labels = header[1:]
    values = np.array([[float(cell) for cell in row[1:]] for row in rows[1:]])
    if values.shape != (len(row_labels), len(column_labels)):
        raise ValueError(f"{path} is not a rectangular matrix")
    return row_labels, column_labels, values


def strongest(
    row_labels: list[str], column_labels: list[str], values: np.ndarray,
    limit: int = 10,
) -> list[tuple[str, str, float]]:
    """The ``limit`` strongest absolute associations across the whole matrix."""
    pairs = [
        (row_labels[i], column_labels[j], float(values[i, j]))
        for i in range(len(row_labels))
        for j in range(len(column_labels))
    ]
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    return pairs[:limit]


def top_pairs(
    terms: list[str], values: np.ndarray, positive: int = 10, negative: int = 5
) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
    """Return the strongest positive and negative off-diagonal pairs."""
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            pairs.append((terms[i], terms[j], float(values[i, j])))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[:positive], pairs[::-1][:negative]


# ---------------------------------------------------------------------------
# 6. Rendering
# ---------------------------------------------------------------------------


def is_symmetric(values: np.ndarray) -> bool:
    """True for a square matrix that is symmetric within float tolerance."""
    return (
        values.ndim == 2
        and values.shape[0] == values.shape[1]
        and np.allclose(values, values.T)
    )


def scale_for(name: str, values: np.ndarray) -> dict:
    """Choose colormap/limits for a matrix based on its file name."""
    if "jaccard" in name:
        return {
            "cmap": "rocket",
            "center": None,
            "vmin": 0.0,
            "vmax": float(values.max()) if values.size else 1.0,
        }
    limit = max(float(np.abs(values).max()), 1e-3) if values.size else 1.0
    return {
        "cmap": "vlag",
        "center": 0.0,
        "vmin": -limit,
        "vmax": limit,
    }


def title_for(name: str) -> str:
    """Human-readable title from a matrix file stem."""
    if name == FEATURE_MATRIX:
        return "Sequence features — Pearson correlation"
    if name.startswith("features_vs_"):
        stem = name[:-4] if name.endswith("_raw") else name
        target = "phrases" if stem.endswith("phrases") else "words"
        scaling = "raw" if name.endswith("_raw") else "z-scored"
        return f"Sequence features vs {target} — Pearson r ({scaling})"
    kind = "Phrase" if name.startswith("phrases") else "Word"
    if name.endswith("_phi"):
        return f"{kind} co-occurrence — phi correlation"
    if name.endswith("_jaccard"):
        return f"{kind} co-occurrence — Jaccard overlap"
    if name.startswith("amino_acids_vs_"):
        target = "phrases" if name.endswith("phrases") else "words"
        return f"Amino-acid composition vs {target} — Pearson r"
    return name.replace("_vs_", " vs ").replace("_", " ").title()


def plot_matrix(path: Path, out_dir: Path, dpi: int) -> Path:
    """Render one matrix CSV as a seaborn heatmap PNG and return its path."""
    row_labels, column_labels, values = load_matrix(path)
    style = scale_for(path.stem, values)
    symmetric = is_symmetric(values)

    rows, columns = values.shape
    width = max(6.0, columns * 0.34 + 3.0)
    height = max(5.0, rows * 0.32 + 2.5)
    font = 7 if max(rows, columns) > 25 else 9

    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        values,
        ax=ax,
        mask=np.triu(np.ones_like(values, dtype=bool), k=0) if symmetric else None,
        xticklabels=column_labels,
        yticklabels=row_labels,
        linewidths=0.2 if rows * columns <= 1200 else 0.0,
        linecolor="white",
        square=False,
        cbar_kws={
            "shrink": 0.6,
            "label": "Jaccard overlap" if "jaccard" in path.stem else "correlation",
        },
        **style,
    )

    ax.set_title(title_for(path.stem), fontsize=13, pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=90, labelsize=font)
    ax.tick_params(axis="y", labelrotation=0, labelsize=font)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_all(
    input_dir: Path,
    plots_dir: Path,
    dpi: int = 200,
    db_path: Path | None = None,
    no_db: bool = False,
) -> None:
    """Render every matrix CSV in ``input_dir`` as a heatmap in ``plots_dir``."""
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory not found: {input_dir}")

    matrix_path = input_dir / f"{FEATURE_MATRIX}.csv"
    if not no_db and db_path is not None:
        if not Path(db_path).exists():
            print(
                f"Warning: database not found: {db_path}; rendering "
                "existing matrices only.",
                file=sys.stderr,
            )
        else:
            try:
                print(
                    f"{Path(db_path).name} -> "
                    f"{refresh_feature_matrix(Path(db_path), matrix_path)}"
                )
            except Exception as exc:  # locked by another client, etc.
                reason = str(exc).splitlines()[0]
                print(
                    f"Warning: could not compute the sequence-feature "
                    f"matrix ({reason}); rendering existing matrices only.",
                    file=sys.stderr,
                )

    matrices = sorted(input_dir.glob("*.csv"))
    if not matrices:
        raise FileNotFoundError(f"no matrix CSVs in {input_dir}")

    for path in matrices:
        print(f"{path.name} -> {plot_matrix(path, plots_dir, dpi)}")


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------


def _kinds(kind: str) -> list[str]:
    return ["words", "phrases"] if kind == "both" else [kind]


def run_terms(conn, args) -> None:
    """Write the term x term phi and Jaccard matrices."""
    for kind in _kinds(args.kind):
        terms = top_terms(conn, kind, args.top)
        presence = presence_matrix(conn, kind, terms)
        phi = phi_matrix(presence)

        phi_path = write_matrix_csv(
            args.out_dir / f"{kind}_phi.csv", terms, terms, phi, row_label="term"
        )
        jaccard_path = write_matrix_csv(
            args.out_dir / f"{kind}_jaccard.csv",
            terms,
            terms,
            jaccard_matrix(presence),
            row_label="term",
        )
        print(f"\n{kind}: {len(terms)} terms")
        print(f"  matrix -> {phi_path}")
        print(f"  matrix -> {jaccard_path}")
        best, worst = top_pairs(terms, phi)
        print(f"  strongest positive phi ({kind}):")
        for a, b, value in best:
            print(f"    {value:+.3f}  {a}  <->  {b}")
        print(f"  strongest negative phi ({kind}):")
        for a, b, value in worst:
            print(f"    {value:+.3f}  {a}  <->  {b}")


def run_amino_acids(conn, args) -> None:
    """Write the amino-acid composition x term matrices."""
    amino_acids = list(AMINO_ACIDS.values())
    for kind in _kinds(args.kind):
        terms = top_terms(conn, kind, args.top)
        values = amino_acid_vs_terms(conn, kind, terms, args.batch_size)
        path = write_matrix_csv(
            args.out_dir / f"amino_acids_vs_{kind}.csv",
            amino_acids,
            terms,
            values,
        )
        print(f"\n{kind}: {len(amino_acids)} amino acids x {len(terms)} terms")
        print(f"  matrix -> {path}")
        print(f"  strongest absolute correlations ({kind}):")
        for amino_acid, term, value in strongest(amino_acids, terms, values):
            print(f"    {value:+.3f}  {amino_acid:<14} <->  {term}")


def run_features(conn, args) -> None:
    """Write the (standardized) sequence-feature x term matrices."""
    scaling = "raw" if args.no_scale else "z-scored"
    suffix = "_raw" if args.no_scale else ""
    for kind in _kinds(args.kind):
        terms = top_terms(conn, kind, args.top)
        values = feature_vs_terms(
            conn, kind, terms, args.batch_size, scaled=not args.no_scale
        )
        path = write_matrix_csv(
            args.out_dir / f"features_vs_{kind}{suffix}.csv",
            FEATURE_COLUMNS,
            terms,
            values,
            row_label=FEATURE_ROW_LABEL,
        )
        print(
            f"\n{kind}: {len(FEATURE_COLUMNS)} features x {len(terms)} terms "
            f"({scaling})"
        )
        print(f"  matrix -> {path}")
        print(f"  strongest absolute correlations ({kind}):")
        for feature, term, value in strongest(FEATURE_COLUMNS, terms, values):
            print(f"    {value:+.3f}  {feature:<28} <->  {term}")


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB,
        help=f"Path to the DuckDB database (default: {DEFAULT_DB}).",
    )
    parser.add_argument(
        "--top", type=int, default=50,
        help="Number of terms per kind (default: 50).",
    )
    parser.add_argument(
        "--kind", choices=KINDS, default="both",
        help="Which term kind to correlate (default: both).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_INPUT,
        help=f"Directory for the matrices (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=25_000,
        help="Rows per streamed batch (default: 25000).",
    )


def _add_plot(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT,
        help=f"Directory with matrix CSVs (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="PNG output directory (default: <input-dir>/plots).",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Output resolution (default: 200).",
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB,
        help="DuckDB used for the sequence-feature matrix.",
    )
    parser.add_argument(
        "--no-db", action="store_true",
        help="Skip the DuckDB refresh and only render existing CSVs.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlation matrices for keywords, composition and features."
    )
    sub = parser.add_subparsers(dest="command")

    terms = sub.add_parser("terms", help="Term x term phi and Jaccard.")
    _add_shared(terms)

    amino_acids = sub.add_parser(
        "amino-acids", help="Amino-acid composition vs terms."
    )
    _add_shared(amino_acids)

    features = sub.add_parser(
        "features", help="Sequence features vs terms (z-scored by default)."
    )
    _add_shared(features)
    features.add_argument(
        "--no-scale", action="store_true",
        help="Correlate the raw features instead of the z-scored ones.",
    )

    plot = sub.add_parser("plot", help="Render every matrix CSV as a heatmap.")
    _add_plot(plot)

    everything = sub.add_parser("all", help="Build every matrix and render it.")
    _add_shared(everything)
    everything.add_argument(
        "--no-scale", action="store_true",
        help="Correlate the raw features instead of the z-scored ones.",
    )
    everything.add_argument("--dpi", type=int, default=200)
    everything.add_argument("--no-db", action="store_true")

    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        argv = ["all", *argv]
    args = parser.parse_args(argv)
    command = args.command or "all"

    if command == "plot":
        sns.set_theme(style="white", context="notebook")
        try:
            render_all(
                args.input_dir,
                args.out_dir or args.input_dir / "plots",
                args.dpi,
                args.db,
                args.no_db,
            )
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    if not args.db.exists():
        print(f"Error: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    conn = connect(args.db)
    try:
        if command in {"terms", "all"}:
            run_terms(conn, args)
        if command in {"amino-acids", "all"}:
            run_amino_acids(conn, args)
        if command in {"features", "all"}:
            run_features(conn, args)
    finally:
        conn.close()

    if command == "all":
        sns.set_theme(style="white", context="notebook")
        try:
            render_all(
                args.out_dir, args.out_dir / "plots", args.dpi, args.db, args.no_db
            )
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()

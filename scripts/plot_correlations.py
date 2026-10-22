#!/usr/bin/env python3
"""Render the correlation matrices in ``exports/correlations`` as heatmaps.

Reads every ``*.csv`` matrix written by ``scripts/term_correlations.py`` and
``scripts/amino_acid_correlations.py`` and saves a seaborn heatmap PNG for
each one.

* ``*_phi.csv`` and ``amino_acids_vs_*.csv`` are signed correlations, drawn
  on a diverging scale centred at zero and symmetric around it.
* ``*_jaccard.csv`` is a 0..1 overlap, drawn on a sequential scale.
* Term x term matrices are symmetric, so the upper triangle is masked to
  avoid showing every pair twice.

Usage:
    uv run python scripts/plot_correlations.py
    uv run python scripts/plot_correlations.py --dpi 300
    uv run python scripts/plot_correlations.py --input-dir exports/corr
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT = PROJECT_ROOT / "exports" / "correlations"


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


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
    kind = "Phrase" if name.startswith("phrases") else "Word"
    if name.endswith("_phi"):
        return f"{kind} co-occurrence — phi correlation"
    if name.endswith("_jaccard"):
        return f"{kind} co-occurrence — Jaccard overlap"
    if name.startswith("amino_acids_vs_"):
        target = "phrases" if name.endswith("phrases") else "words"
        return f"Amino-acid composition vs {target} — Pearson r"
    return name.replace("_vs_", " vs ").replace("_", " ").title()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_matrix(path: Path, out_dir: Path, dpi: int) -> Path:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render correlation matrices as seaborn heatmaps."
    )
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
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: input directory not found: {args.input_dir}",
              file=sys.stderr)
        sys.exit(1)

    matrices = sorted(args.input_dir.glob("*.csv"))
    if not matrices:
        print(f"Error: no matrix CSVs in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or args.input_dir / "plots"
    sns.set_theme(style="white", context="notebook")
    for path in matrices:
        print(f"{path.name} -> {plot_matrix(path, out_dir, args.dpi)}")


if __name__ == "__main__":
    main()

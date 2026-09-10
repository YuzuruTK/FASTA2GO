"""CSV staging helpers for fast native DuckDB bulk ingestion.

DuckDB's row-by-row ``executemany`` path is slow. These helpers stage rows
as CSV files in a cache directory so callers can load them with DuckDB's
native ``read_csv`` reader in a single statement.
"""

from __future__ import annotations

import csv
from pathlib import Path
from uuid import uuid4

import duckdb

#: Default directory used to stage batch CSVs before native DuckDB imports.
CSV_CACHE_DIR = Path("cache/csv_bulk")


def table_column_types(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    columns: list[str],
) -> dict[str, str]:
    """Return the DuckDB type of each requested column in the target table.

    Types are read from ``information_schema.columns`` so imports stay
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


def stage_rows_to_csv(
    cache_dir: str | Path | None,
    table: str,
    columns: list[str],
    rows: list[list],
) -> Path:
    """Write a header row plus data rows to a uniquely-named CSV file.

    ``None`` values become empty (unquoted) CSV fields, which DuckDB's CSV
    reader converts to SQL NULL on import.

    Args:
        cache_dir: Directory used to stage the CSV files. Defaults to
            ``cache/csv_bulk`` relative to the working directory.
        table: Table name used as the CSV filename prefix.
        columns: Column names written as the header row.
        rows: List of value rows.

    Returns:
        Path of the staged CSV file.
    """
    cache_path = Path(cache_dir) if cache_dir else CSV_CACHE_DIR
    cache_path.mkdir(parents=True, exist_ok=True)
    csv_path = cache_path / f"{table}_{uuid4().hex}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        writer.writerows(rows)
    return csv_path


def remove_staged_csv(csv_path: str | Path) -> None:
    """Delete a staged CSV file.

    A cleanup failure is non-fatal (the rows are already committed), so only
    a warning is printed instead of raising.
    """
    try:
        Path(csv_path).unlink()
    except OSError as exc:  # pragma: no cover - cleanup-only path
        print(f"Warning: could not remove staged CSV {csv_path}: {exc}")

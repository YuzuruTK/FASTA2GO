"""DuckDB persistence: connection, schema initialisation, bulk insert."""

import csv
from pathlib import Path
from uuid import uuid4

import duckdb

from protein_database.schema import ANALYSIS_VIEW_SQL, SCHEMA_SQL, TABLE_NAMES

#: Default directory used to stage batch CSVs before native DuckDB imports.
CSV_CACHE_DIR = Path("cache/csv_bulk")


def connect(db_path: str | Path = "proteins.duckdb") -> duckdb.DuckDBPyConnection:
    """Connect to a DuckDB database file (parent dir created if needed)."""
    if str(db_path) == ":memory:":
        return duckdb.connect(":memory:")
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def init_database(conn: duckdb.DuckDBPyConnection, overwrite: bool = False) -> None:
    """Create all tables and the analysis view (idempotent)."""
    if overwrite:
        for table in reversed(TABLE_NAMES):  # children before proteins
            conn.execute(f"DROP TABLE IF EXISTS {table}")
    for ddl in SCHEMA_SQL.values():
        conn.execute(ddl)
    conn.execute(ANALYSIS_VIEW_SQL)


def _column_types(
    conn: duckdb.DuckDBPyConnection, table: str, columns: list[str]
) -> dict[str, str]:
    """Return the DuckDB type of each requested column, in ``columns`` order."""
    rows = conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = ?",
        [table],
    ).fetchall()
    types = dict(rows)
    missing = [column for column in columns if column not in types]
    if missing:
        raise ValueError(f"Columns {missing} not found in table '{table}'.")
    return {column: types[column] for column in columns}


def _stage_csv(
    cache_dir: str | Path | None,
    table: str,
    columns: list[str],
    rows: list[list],
) -> Path:
    """Write rows to a staging CSV and return its path.

    ``None`` values become empty fields, which DuckDB reads back as NULL.
    """
    cache = Path(cache_dir) if cache_dir else CSV_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    csv_path = cache / f"{table}_{uuid4().hex}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    return csv_path


def _import_select(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    columns: list[str],
    csv_path: Path,
    suffix: str = "",
) -> None:
    """Import a staged CSV natively, appending an optional SQL suffix."""
    column_list = ", ".join(columns)
    types_spec = ", ".join(
        f"'{name}': '{dtype}'"
        for name, dtype in _column_types(conn, table, columns).items()
    )
    escaped_path = str(csv_path).replace("'", "''")
    conn.execute(
        f"INSERT INTO {table} ({column_list}) "
        f"SELECT {column_list} FROM read_csv('{escaped_path}', header=true, "
        f"columns={{{types_spec}}}){suffix}"
    )


def bulk_insert(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    columns: list[str],
    rows: list[list],
    cache_dir: str | Path | None = None,
) -> None:
    """Insert rows by staging a CSV and importing it with DuckDB's reader.

    ``None`` values are written as empty CSV fields, which DuckDB reads back
    as SQL NULL. The staging CSV is removed on success and kept on failure
    (for debugging), re-raising the error either way.
    """
    if not rows:
        return
    csv_path = _stage_csv(cache_dir, table, columns, rows)
    try:
        _import_select(conn, table, columns, csv_path)
    except Exception:
        raise  # keep the staged CSV for debugging
    csv_path.unlink(missing_ok=True)


def bulk_upsert(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    columns: list[str],
    rows: list[list],
    conflict_columns: list[str],
    update_columns: list[str],
    cache_dir: str | Path | None = None,
) -> None:
    """Insert-or-update rows via a CSV-staged native import.

    Mirrors :func:`bulk_insert`, adding ``ON CONFLICT (… ) DO UPDATE SET``
    for ``update_columns``. The staging CSV is removed on success and kept
    on failure (for debugging), re-raising the error either way.
    """
    if not rows:
        return
    conflict = ", ".join(conflict_columns)
    updates = ", ".join(f"{column} = excluded.{column}" for column in update_columns)
    suffix = f" ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
    csv_path = _stage_csv(cache_dir, table, columns, rows)
    try:
        _import_select(conn, table, columns, csv_path, suffix)
    except Exception:
        raise  # keep the staged CSV for debugging
    csv_path.unlink(missing_ok=True)

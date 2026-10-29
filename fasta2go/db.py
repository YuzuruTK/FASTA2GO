"""DuckDB access: connection, schema bootstrap and bulk writes.

Bulk writes go through a staged CSV read by DuckDB's native CSV reader, which is
far faster than one statement per row. The SQL itself lives in ``sql/``.
"""

from __future__ import annotations

import csv
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_SQL = PROJECT_ROOT / "sql" / "schema.sql"
VIEWS_SQL = PROJECT_ROOT / "sql" / "views.sql"
STAGING_DIR = PROJECT_ROOT / "cache" / "staging"


def connect(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """Open (and create if needed) a DuckDB database file."""
    if str(db_path) == ":memory:":
        return duckdb.connect(":memory:")
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the tables and the views; idempotent."""
    conn.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
    conn.execute(VIEWS_SQL.read_text(encoding="utf-8"))


@contextmanager
def transaction(conn: duckdb.DuckDBPyConnection) -> Iterator[None]:
    """Run a block in one transaction, rolling back on any error."""
    conn.execute("BEGIN TRANSACTION")
    try:
        yield
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def write_rows(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    key: Sequence[str] = (),
    update: Sequence[str] = (),
) -> int:
    """Insert rows through a staged CSV and return how many were sent.

    ``csv.writer`` keeps text containing commas, semicolons or line breaks
    intact, and ``None`` becomes an empty field, which DuckDB reads as NULL.
    With ``key``, conflicting rows are updated (``update``) or ignored.
    """
    if not rows:
        return 0

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    path = STAGING_DIR / f"{table}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)

    column_list = ", ".join(columns)
    escaped = str(path).replace("'", "''")
    sql = (
        f"INSERT INTO {table} ({column_list}) "
        f"SELECT {column_list} FROM read_csv('{escaped}', header = true, "
        "all_varchar = true)"
    )
    if key:
        conflict = f" ON CONFLICT ({', '.join(key)}) DO "
        if update:
            conflict += "UPDATE SET " + ", ".join(
                f"{column} = excluded.{column}" for column in update
            )
        else:
            conflict += "NOTHING"
        sql += conflict

    conn.execute(sql)
    path.unlink(missing_ok=True)
    return len(rows)

"""DuckDB connection management for the protein feature database."""

from pathlib import Path
from typing import Any

import duckdb

from protein_database.database.models import ANALYSIS_VIEW_SQL, SCHEMA_SQL, TABLE_NAMES


def connect(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """Connect to a DuckDB database file.

    The parent directory is created if it does not exist.

    Args:
        db_path: Path to the DuckDB database file. Use ':memory:' for an
            in-memory database.

    Returns:
        A DuckDB connection object.
    """
    if db_path != ":memory:":
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(path))
    return duckdb.connect(":memory:")


def init_database(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all tables in the database if they don't exist.

    Also creates the denormalized ``analysis_data`` view that joins all
    per-protein tables for analysis purposes.

    Args:
        conn: DuckDB connection.
    """
    for ddl in SCHEMA_SQL.values():
        conn.execute(ddl)
    conn.execute(ANALYSIS_VIEW_SQL)


def drop_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Drop all tables in the database if they exist.

    Args:
        conn: DuckDB connection.
    """
    for table in TABLE_NAMES:
        conn.execute(f"DROP TABLE IF EXISTS {table}")


def table_exists(conn: duckdb.DuckDBPyConnection, table: str) -> bool:
    """Check whether a table exists in the database.

    Args:
        conn: DuckDB connection.
        table: Name of the table to check.

    Returns:
        True if the table exists, False otherwise.
    """
    result = conn.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_name = ?",
        [table],
    ).fetchone()
    return result is not None and result[0] > 0


def query(conn: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None):
    """Run a query and return the raw result set.

    Args:
        conn: DuckDB connection.
        sql: SQL statement to execute.
        params: Optional list of parameters.

    Returns:
        The DuckDB result.
    """
    return conn.execute(sql, params)


def bulk_insert(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    columns: list[str],
    rows: list[list],
) -> None:
    """Bulk-insert rows into a table using DuckDB executemany.

    All rows are inserted in a single prepared-statement batch, avoiding
    row-by-row database writes.

    Args:
        conn: DuckDB connection.
        table: Target table name.
        columns: Column names to insert into (same order as rows).
        rows: List of value rows.

    Raises:
        ValueError: If rows is empty.
    """
    if not rows:
        raise ValueError("No rows to insert.")
    placeholder = ", ".join(["?"] * len(columns))
    col_str = ", ".join(columns)
    conn.executemany(
        f"INSERT INTO {table} ({col_str}) VALUES ({placeholder})",
        rows,
    )


def bulk_upsert(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    columns: list[str],
    rows: list[list],
    on_conflict_columns: list[str],
    update_columns: list[str],
) -> None:
    """Upsert rows into a table using DuckDB ON CONFLICT.

    Args:
        conn: DuckDB connection.
        table: Target table name.
        columns: Column names to insert into (same order as rows).
        rows: List of value rows.
        on_conflict_columns: Columns that form the conflict target (PK).
        update_columns: Columns to update on conflict (excluding PK).

    Raises:
        ValueError: If rows is empty.
    """
    if not rows:
        raise ValueError("No rows to upsert.")
    placeholder = ", ".join(["?"] * len(columns))
    col_str = ", ".join(columns)
    conflict_str = ", ".join(on_conflict_columns)
    update_str = ", ".join(f"{c} = excluded.{c}" for c in update_columns)
    conn.executemany(
        f"INSERT INTO {table} ({col_str}) VALUES ({placeholder}) "
        f"ON CONFLICT ({conflict_str}) DO UPDATE SET {update_str}",
        rows,
    )

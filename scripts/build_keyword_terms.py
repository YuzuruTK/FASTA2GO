#!/usr/bin/env python3
"""Build the per-protein keyword-terms table and term-frequency report.

Reads each protein's function text plus the Gene Ontology columns from the
UniProt snapshot, stores one space-separated string per protein in
``protein_keyword_terms`` (drop + rebuild, never upsert), then counts
document frequency per term and writes ``data/term_frequencies.csv`` with a
keep flag for terms within [min-occ, --max-occ]. Words and phrases have
separate floors (``--min-occ-words`` and ``--min-occ-phrases``): a phrase is
a specific concept and is naturally rarer, so it needs a lower floor to stay
useful for grouping.

``words`` holds single tokens; ``phrases`` holds multi-token concepts only.
Phrase detection is vocabulary-driven: every multi-token GO term in the
snapshot (``plasma membrane``, ``ATP binding``, ``structural constituent of
ribosome``, ...) becomes a phrase, and the same vocabulary is matched
against free-text function annotations so proteins without GO columns still
get the concept. Single-token GO terms stay in ``words`` where they belong.

Usage:
    uv run python scripts/build_keyword_terms.py --db proteins.duckdb
    uv run python scripts/build_keyword_terms.py --db proteins.duckdb \
        --min-occ-phrases 0.001 --min-occ-words 0.01 --max-occ 0.95
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from protein_database import db
from protein_database.keywords import (
    build_phrase_index,
    find_phrases,
    keep_term,
    parse_go_cell,
    phrase_key,
)
from protein_database.schema import SCHEMA_SQL
from protein_database.uniprot import GO_HEADERS, download_snapshot
from scripts.lexicon import is_noise, is_phrase_noise, tokenize

DEFAULT_DB = PROJECT_ROOT / "proteins.duckdb"
DEFAULT_OUT = PROJECT_ROOT / "data" / "term_frequencies.csv"

TABLE = "protein_keyword_terms"
TABLE_COLUMNS = ["fasta_id", "source_file", "words", "phrases", "go_ids"]

#: Snapshot header -> GO column role.
GO_COLUMNS = {
    "Gene Ontology (biological process)": "go_p",
    "Gene Ontology (cellular component)": "go_c",
    "Gene Ontology (molecular function)": "go_f",
}


def _assert_go_headers(conn, snapshot: Path) -> None:
    """Fail loudly when the snapshot has no GO columns (stale cache)."""
    headers = {
        column[0]
        for column in conn.execute(
            "SELECT * FROM read_csv(?, delim='\\t', header=true) LIMIT 0",
            [str(snapshot)],
        ).description
    }
    missing = GO_HEADERS - headers
    if missing:
        raise SystemExit(
            f"Snapshot {snapshot} has no GO columns ({sorted(missing)}). "
            "Re-run: uv run python scripts/annotate_proteins.py "
            "--db <db> --source bulk --refresh-snapshot"
        )


def _build_staging(conn, snapshot: Path) -> None:
    """Ingest the snapshot's Entry + GO columns into a temp table."""
    conn.execute(
        "CREATE OR REPLACE TEMP TABLE go_staging AS "
        f'SELECT "Entry", {", ".join(f"\"{h}\"" for h in GO_HEADERS)} '
        f"FROM read_csv(?, delim='\\t', header=true, union_by_name=true) "
        'WHERE "Entry" IS NOT NULL',
        [str(snapshot)],
    )


def _go_terms(conn) -> list[str]:
    """Return every distinct GO term string in the staged snapshot."""
    segments = " UNION ALL ".join(
        f'SELECT unnest(string_split("{header}", \';\')) AS segment FROM go_staging'
        for header in GO_HEADERS
    )
    rows = conn.execute(
        f"SELECT DISTINCT segment FROM ({segments}) WHERE segment IS NOT NULL"
    ).fetchall()
    return [term for (segment,) in rows for term, _ in parse_go_cell(segment)]


def _concept_tokens(text: str) -> list[str]:
    """Tokenize and drop only structural debris (for phrase matching)."""
    return [token for token in tokenize(text) if not is_phrase_noise(token)]


def _row_terms(
    function: str | None,
    go_cells: list[str | None],
    phrase_index: dict[str, tuple[tuple[str, ...], ...]],
) -> tuple[str, str, str]:
    """Return (words, phrases, go_ids) strings for one protein.

    ``words`` collects every clean token from the function text and the GO
    terms. ``phrases`` keeps multi-token concepts only: GO terms of two or
    more tokens, plus the same phrases found in the free-text function
    annotation. Phrase tokens keep generic biology words so the concept
    stays faithful ("identical protein binding").
    """
    raw_tokens = tokenize(function or "")
    words: set[str] = {t for t in raw_tokens if not is_noise(t)}
    phrases: set[str] = find_phrases(
        [t for t in raw_tokens if not is_phrase_noise(t)], phrase_index
    )
    go_ids: set[str] = set()

    for cell in go_cells:
        for term, go_id in parse_go_cell(cell):
            term_tokens = tokenize(term)
            words.update(t for t in term_tokens if not is_noise(t))
            concept = [t for t in term_tokens if not is_phrase_noise(t)]
            if len(concept) >= 2:
                phrases.add(phrase_key(concept))
            go_ids.add(go_id)

    return (
        " ".join(sorted(words)),
        " ".join(sorted(phrases)),
        " ".join(sorted(go_ids)),
    )


def _term_rows(conn) -> list[list]:
    """Return one [fasta_id, source_file, words, phrases, go_ids] row per protein."""
    phrase_index = build_phrase_index([tuple(_concept_tokens(t)) for t in _go_terms(conn)])
    go_select = ", ".join(f's."{header}"' for header in GO_HEADERS)
    rows = conn.execute(
        "SELECT p.fasta_id, p.source_file, ua.function, " + go_select + " "
        "FROM proteins p "
        "LEFT JOIN uniprot_annotations ua USING (fasta_id, source_file) "
        "JOIN go_staging s "
        "ON split_part(p.fasta_id, '|', 1) IN ('sp', 'tr') "
        'AND upper(split_part(p.fasta_id, \'|\', 2)) = s."Entry"'
    ).fetchall()

    return [
        [fasta_id, source_file, *_row_terms(function, go_cells, phrase_index)]
        for fasta_id, source_file, function, *go_cells in rows
    ]


def _rebuild_table(conn, terms: list[list]) -> int:
    """Drop + recreate the table and insert all rows. Returns row count."""
    conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
    conn.execute(SCHEMA_SQL[TABLE])
    db.bulk_insert(conn, TABLE, TABLE_COLUMNS, terms)
    return conn.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0]


def _write_frequencies(
    conn,
    out_path: Path,
    min_occ_words: float,
    min_occ_phrases: float,
    max_occ: float,
) -> dict[str, int]:
    """Count document frequency per kind/term and write the CSV.

    Words and phrases use different floors: a phrase is a specific concept,
    so it is naturally rarer and a lower floor keeps it usable for grouping,
    while words need a higher floor to stay clear of prose filler. Returns
    the kept count per kind.
    """
    n = conn.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0]
    df_rows = conn.execute(
        "SELECT kind, term, count(*) AS df FROM ("
        "SELECT fasta_id, source_file, 'word' AS kind, "
        "unnest(string_split(words, ' ')) AS term "
        f"FROM {TABLE} "
        "UNION ALL "
        "SELECT fasta_id, source_file, 'phrase', "
        "unnest(string_split(phrases, ' ')) "
        f"FROM {TABLE}) "
        "WHERE term <> '' "
        "GROUP BY kind, term "
        "ORDER BY kind, df DESC, term"
    ).fetchall()

    min_occ = {"word": min_occ_words, "phrase": min_occ_phrases}
    kept = {"word": 0, "phrase": 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "term", "df", "occ_pct", "keep"])
        for kind, term, df in df_rows:
            keep = keep_term(df, n, min_occ[kind], max_occ)
            kept[kind] += keep
            writer.writerow([kind, term, df, f"{df / n * 100:.2f}", str(keep).lower()])
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build protein_keyword_terms and term_frequencies.csv."
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB,
        help=f"Path to the DuckDB database (default: {DEFAULT_DB}).",
    )
    parser.add_argument(
        "--snapshot", type=Path, default=None,
        help="Snapshot TSV path (default: resolve via download_snapshot cache).",
    )
    parser.add_argument(
        "--min-occ-phrases", type=float, default=0.001,
        help="Minimum df/N to keep a phrase (default: 0.001 = 0.1%%).",
    )
    parser.add_argument(
        "--min-occ-words", type=float, default=0.01,
        help="Minimum df/N to keep a word (default: 0.01 = 1%%).",
    )
    parser.add_argument(
        "--max-occ", type=float, default=0.95,
        help="Maximum df/N to keep a term, either kind (default: 0.95).",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"Term-frequency CSV path (default: {DEFAULT_OUT}).",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    snapshot = args.snapshot or download_snapshot()
    if not snapshot.exists():
        print(f"Error: Snapshot not found: {snapshot}", file=sys.stderr)
        sys.exit(1)

    conn = db.connect(args.db)
    try:
        db.init_database(conn)
        _assert_go_headers(conn, snapshot)

        print("Staging GO columns from snapshot...")
        _build_staging(conn, snapshot)

        print("Building keyword terms...")
        terms = _term_rows(conn)
        written = _rebuild_table(conn, terms)
        empty = sum(1 for row in terms if not row[2])
        print(f"Wrote {written:,} rows to {TABLE} ({empty:,} with empty words).")

        print("Counting term frequencies...")
        kept = _write_frequencies(
            conn, args.out, args.min_occ_words, args.min_occ_phrases, args.max_occ
        )
        print(
            f"Kept {kept['phrase']:,} phrases (>={args.min_occ_phrases:.1%}) "
            f"and {kept['word']:,} words (>={args.min_occ_words:.1%}) "
            f"-> {args.out}"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()

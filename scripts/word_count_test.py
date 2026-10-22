#!/usr/bin/env python3
"""Word-frequency analysis and category coverage for the ``function`` column.

Four modes of operation:

1. Word frequency (default): sample rows from the ``analysis_data`` view,
   tokenize the ``function`` texts, and report which words appear most
   often across distinct proteins.
2. Export by word (``--export-word WORD``): export all rows whose
   ``function`` text contains the given word to a CSV file.
3. Coverage check (``--coverage-file FILE``): given a category dictionary
   file (one word per line), report how many rows match at least one
   dictionary word and how many rows are left unmatched.
4. Export top words (``--export-top N``): full scan of all rows with a
   non-null ``function``, writing the top-N most frequent words to a CSV
   with columns ``rank,word,proteins,total``. Deterministic: ties are
   broken alphabetically by word.

Usage:
    uv run python scripts/word_count_test.py [--limit 1000] [--top 30]
    uv run python scripts/word_count_test.py --export-word kinase \
        --export-out kinase_proteins.csv
    uv run python scripts/word_count_test.py --coverage-file category_words.txt \
        --export-unmatched unmatched.csv
    uv run python scripts/word_count_test.py \
        --exclude-file scripts/word_filter.txt \
        --export-top 1000 --export-top-out data/word_frequencies.csv
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

# Add project root to path so we can find the protein_database package
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from protein_database.db import connect
from scripts.lexicon import is_noise, load_word_file, tokenize


def fetch_function_rows(conn, limit: int | None) -> list[tuple]:
    """Fetch (fasta_id, source_file, function) rows with non-null function."""
    sql = (
        "SELECT fasta_id, source_file, function FROM analysis_data "
        "WHERE function IS NOT NULL"
    )
    params: list = []
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


def export_rows_to_csv(rows: list[tuple], columns: list[str], out_path: str | Path) -> None:
    """Write rows to a CSV file, creating parent directories as needed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def run_frequency_mode(conn, args) -> None:
    """Count word frequencies in the 'function' column."""
    rows = fetch_function_rows(conn, args.limit)

    total_counter: Counter[str] = Counter()
    doc_counter: Counter[str] = Counter()

    for _, _, function_text in rows:
        words = tokenize(function_text)
        total_counter.update(words)
        # Count each word at most once per protein (document frequency).
        doc_counter.update(set(words))

    if not args.no_stopwords:
        extra = load_word_file(args.exclude_file) if args.exclude_file else set()
        total_counter = Counter(
            {w: c for w, c in total_counter.items() if not is_noise(w) and w not in extra}
        )
        doc_counter = Counter(
            {w: c for w, c in doc_counter.items() if not is_noise(w) and w not in extra}
        )

    total_words = sum(total_counter.values())
    unique_words = len(total_counter)

    print("=" * 60)
    print("  Word Frequency Test — 'function' column")
    print("=" * 60)
    print(f"  Rows sampled        : {len(rows):,}")
    print(f"  Total words         : {total_words:,}")
    print(f"  Unique words        : {unique_words:,}")
    print(f"  Stopwords filtered  : {'no' if args.no_stopwords else 'yes'}")
    print("=" * 60)

    print(f"\nTop {args.top} words by number of proteins containing them:")
    print(f"  {'Rank':>4}  {'Word':<20} {'Proteins':>10} {'Total':>10}")
    print("  " + "-" * 50)
    for rank, (word, doc_count) in enumerate(doc_counter.most_common(args.top), 1):
        print(f"  {rank:>4}  {word:<20} {doc_count:>10,} {total_counter[word]:>10,}")


def compute_word_counters(rows: list[tuple], no_stopwords: bool,
                          exclude_file: str | None) -> tuple[Counter, Counter]:
    """Tokenize rows and return (total_counter, doc_counter).

    Applies ``is_noise`` (stopwords, numbers, fragments, short-token debris)
    plus any extra exclusion file, unless ``no_stopwords`` is set.
    """
    total_counter: Counter[str] = Counter()
    doc_counter: Counter[str] = Counter()

    for _, _, function_text in rows:
        words = tokenize(function_text)
        total_counter.update(words)
        # Count each word at most once per protein (document frequency).
        doc_counter.update(set(words))

    if not no_stopwords:
        extra = load_word_file(exclude_file) if exclude_file else set()
        total_counter = Counter(
            {w: c for w, c in total_counter.items() if not is_noise(w) and w not in extra}
        )
        doc_counter = Counter(
            {w: c for w, c in doc_counter.items() if not is_noise(w) and w not in extra}
        )

    return total_counter, doc_counter


def run_export_top_mode(conn, args) -> None:
    """Export the top-N most frequent words to a deterministic CSV.

    Performs a FULL scan of all rows with a non-null function (ignores
    --limit) so results are reproducible. Words are ranked by document
    frequency (number of distinct proteins containing them); ties are
    broken alphabetically by word.
    """
    rows = fetch_function_rows(conn, limit=None)
    total_counter, doc_counter = compute_word_counters(
        rows, args.no_stopwords, args.exclude_file
    )

    ranked = sorted(doc_counter.items(), key=lambda item: (-item[1], item[0]))
    top = ranked[:args.export_top]

    out_path = Path(args.export_top_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "word", "proteins", "total"])
        for rank, (word, doc_count) in enumerate(top, 1):
            writer.writerow([rank, word, doc_count, total_counter[word]])

    print("=" * 60)
    print("  Export Top Words — 'function' column")
    print("=" * 60)
    print(f"  Rows scanned        : {len(rows):,}")
    print(f"  Unique words        : {len(doc_counter):,}")
    print(f"  Words exported      : {len(top):,}")
    print(f"  Output file         : {out_path}")
    print("=" * 60)


def run_export_mode(conn, args) -> None:
    """Export all analysis_data rows whose function contains a word."""
    word = args.export_word.lower()
    result = conn.execute(
        """
        SELECT *
        FROM analysis_data
        WHERE function IS NOT NULL
          AND regexp_matches(lower(function), ?)
        """,
        [rf"\b{re.escape(word)}\b"],
    )
    columns = [desc[0] for desc in result.description]
    rows = result.fetchall()

    out_path = args.export_out or f"exports/{word}_proteins.csv"
    export_rows_to_csv(rows, columns, out_path)

    print(f"Exported {len(rows):,} rows containing '{word}' to: {out_path}")
    print(f"Columns: {', '.join(columns)}")


def run_coverage_mode(conn, args) -> None:
    """Check how many rows match at least one word of a category dictionary."""
    dictionary = load_word_file(args.coverage_file)
    if not dictionary:
        print("Error: coverage file contains no words.", file=sys.stderr)
        sys.exit(1)

    # Coverage checks always scan ALL rows with a non-null function.
    rows = fetch_function_rows(conn, limit=None)

    word_counter: Counter[str] = Counter()
    matched = 0
    unmatched_rows: list[tuple] = []

    for row in rows:
        fasta_id, source_file, function_text = row
        hits = dictionary.intersection(tokenize(function_text))
        if hits:
            matched += 1
            word_counter.update(hits)
        else:
            unmatched_rows.append(row)

    total = len(rows)
    unmatched = total - matched
    coverage = (matched / total * 100) if total else 0.0

    print("=" * 60)
    print("  Category Coverage Check — 'function' column")
    print("=" * 60)
    print(f"  Dictionary words    : {len(dictionary):,}")
    print(f"  Rows with function  : {total:,}")
    print(f"  Rows matched        : {matched:,} ({coverage:.1f}%)")
    print(f"  Rows unmatched      : {unmatched:,} ({100 - coverage:.1f}%)")
    print("=" * 60)

    print(f"\nMatches per dictionary word:")
    print(f"  {'Word':<20} {'Proteins':>10}")
    print("  " + "-" * 32)
    for word, count in word_counter.most_common():
        print(f"  {word:<20} {count:>10,}")
    unused = dictionary - set(word_counter)
    if unused:
        print(f"\n  Dictionary words that matched no row: "
              f"{', '.join(sorted(unused))}")

    if args.export_unmatched and unmatched_rows:
        export_rows_to_csv(
            unmatched_rows,
            ["fasta_id", "source_file", "function"],
            args.export_unmatched,
        )
        print(f"\nExported {len(unmatched_rows):,} unmatched rows to: "
              f"{args.export_unmatched}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze word frequencies in the 'function' column."
    )
    parser.add_argument(
        "--db",
        type=str,
        default="proteins.duckdb",
        help="Path to the DuckDB database (default: proteins.duckdb).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of rows to sample in frequency mode (default: 1000).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of top words to display (default: 30).",
    )
    parser.add_argument(
        "--no-stopwords",
        action="store_true",
        help="Do not filter out stopwords/noise/generic words.",
    )
    parser.add_argument(
        "--exclude-file",
        type=str,
        default=None,
        help="Text file with extra words to exclude (one per line).",
    )
    parser.add_argument(
        "--export-word",
        type=str,
        default=None,
        help="Export all rows whose function contains this word to CSV.",
    )
    parser.add_argument(
        "--export-out",
        type=str,
        default=None,
        help="Output CSV path for --export-word "
             "(default: exports/<word>_proteins.csv).",
    )
    parser.add_argument(
        "--coverage-file",
        type=str,
        default=None,
        help="Category dictionary file (one word per line). Reports how "
             "many rows match at least one dictionary word.",
    )
    parser.add_argument(
        "--export-unmatched",
        type=str,
        default=None,
        help="Export rows matching no dictionary word to this CSV path.",
    )
    parser.add_argument(
        "--export-top",
        type=int,
        default=None,
        metavar="N",
        help="Full-scan the 'function' column and export the top-N most "
             "frequent words to --export-top-out as rank,word,proteins,total.",
    )
    parser.add_argument(
        "--export-top-out",
        type=str,
        default="data/word_frequencies.csv",
        help="Output CSV path for --export-top "
             "(default: data/word_frequencies.csv).",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = connect(db_path)
    try:
        if args.export_top:
            run_export_top_mode(conn, args)
        elif args.coverage_file:
            run_coverage_mode(conn, args)
        elif args.export_word:
            run_export_mode(conn, args)
        else:
            run_frequency_mode(conn, args)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
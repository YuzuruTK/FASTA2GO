#!/usr/bin/env python3
"""Word-frequency analysis and category coverage for the ``function`` column.

Three modes of operation:

1. Word frequency (default): sample rows from the ``analysis_data`` view,
   tokenize the ``function`` texts, and report which words appear most
   often across distinct proteins.
2. Export by word (``--export-word WORD``): export all rows whose
   ``function`` text contains the given word to a CSV file.
3. Coverage check (``--coverage-file FILE``): given a category dictionary
   file (one word per line), report how many rows match at least one
   dictionary word and how many rows are left unmatched.

Usage:
    uv run python scripts/word_count_test.py [--limit 1000] [--top 30]
    uv run python scripts/word_count_test.py --export-word kinase \
        --export-out kinase_proteins.csv
    uv run python scripts/word_count_test.py --coverage-file category_words.txt \
        --export-unmatched unmatched.csv
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

from protein_database.database.connection import connect

#: Common English stopwords that carry no categorization signal.
STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be
    because been before being below between both but by can cannot could
    did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it
    its itself just me more most my myself no nor not now of off on once
    only or other our ours ourselves out over own same she should so some
    such than that the their theirs them themselves then there these they
    this those through to too under until up very was we were what when
    where which while who whom why will with you your yours yourself
    yourselves may might must shall upon within whose
    """.split()
)

#: UniProt annotation metadata tokens (evidence codes, cross-references)
#: that carry no categorization signal.
NOISE_WORDS = frozenset(
    """
    eco pubmed uniprotkb hamap rule mf cc evidence interpro
    prov predicted inferred similarity
    """.split()
)

#: Generic filler words that appear in many descriptions but do not help
#: categorize proteins. Curated exclusion list.
GENERIC_WORDS = frozenset(
    """
    function functions functional protein proteins gene genes product
    products factor factors component components element elements
    molecule molecules entity entities involved involving associated
    associated_with participates participating contributes contributing
    plays play role roles required necessary important critical essential
    acts acting serves serving mediates mediating facilitates
    facilitating helps helping supports supporting putative probable
    predicted possible potential hypothetical unknown uncharacterized
    conserved similar homolog homologous ortholog paralog family member
    contains containing encoding encoded related like type class group
    part form forms forming formation formed complex subunit chain region
    domain domains motif motifs site sites residue residues alpha beta
    gamma delta large small major minor high low long short specific
    general common unique different various multiple single
    thereby thus activity process processes
    """.split()
)

#: Token pattern: lowercase alphanumeric words (keeps internal apostrophes).
WORD_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase a text and split it into alphanumeric word tokens."""
    return [token.strip("'") for token in WORD_RE.findall(text.lower()) if token.strip("'")]


def load_word_file(path: str | Path) -> set[str]:
    """Load words from a text file (one per line, '#' starts a comment)."""
    words: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().lower()
        if line:
            words.add(line)
    return words


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
        excluded = STOPWORDS | NOISE_WORDS | GENERIC_WORDS
        if args.exclude_file:
            excluded |= load_word_file(args.exclude_file)
        # Also drop pure-numeric tokens (GO evidence codes like 0000250).
        total_counter = Counter(
            {w: c for w, c in total_counter.items() if w not in excluded and not w.isdigit()}
        )
        doc_counter = Counter(
            {w: c for w, c in doc_counter.items() if w not in excluded and not w.isdigit()}
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
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = connect(db_path)
    try:
        if args.coverage_file:
            run_coverage_mode(conn, args)
        elif args.export_word:
            run_export_mode(conn, args)
        else:
            run_frequency_mode(conn, args)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
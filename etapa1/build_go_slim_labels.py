#!/usr/bin/env python3
"""Build the 41-label Molecular Function labels of the database.

The vocabulary is the generic GO slim (``goslim_generic``, Molecular Function
only) plus one documented extension, ``GO:0005488 binding``; the aspect root
``GO:0003674`` is excluded by rule. Every source is pinned in
``data/go_slim/sources.json`` and verified by SHA-256 before use, so the labels
can only be produced from the recorded GO release.

    uv run python etapa1/build_go_slim_labels.py

There are no arguments: the pinned sources are downloaded when missing (and
always verified), and the database path comes from ``etapa1/fasta2go.config``.
The database is opened read-only and is never modified. The output is one row
per protein, including the proteins without any Molecular Function annotation.
"""

from __future__ import annotations

import configparser
import json
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fasta2go import db  # noqa: E402
from fasta2go.go_slim import (  # noqa: E402
    ASPECT_ROOT,
    MF_NAMESPACE,
    GoSlimError,
    build_ancestor_map,
    build_vocabulary,
    map_proteins,
    parse_obo,
    read_term_ids,
    read_vocabulary_ids,
    sha256_file,
    verify_integrity,
    write_labels_csv,
)

CONFIG_FILE = Path(__file__).resolve().parent / "fasta2go.config"
SOURCES_FILE = PROJECT_ROOT / "data" / "go_slim" / "sources.json"
VOCABULARY_FILE = PROJECT_ROOT / "data" / "go_slim" / "vocabulary_mf.csv"
EXAMPLES_FILE = PROJECT_ROOT / "data" / "go_slim" / "expected_examples.json"
CACHE_DIR = PROJECT_ROOT / "cache" / "go_slim"
CONFIG = configparser.ConfigParser()
CONFIG.read(CONFIG_FILE, encoding="utf-8")
DATABASE = PROJECT_ROOT / CONFIG.get(
    "extraction", "database", fallback="db/fasta2go.duckdb"
)
OUTPUT = PROJECT_ROOT / "exports" / "go_slim" / "protein_labels.csv"

#: ``data/go_slim/vocabulary_mf.csv`` is the published label space; it must match
#: the current ``goslim_generic`` slim term for term, so a new GO release is
#: noticed instead of silently relabelling.
EXPECTED_TERMS = 41


def load_sources() -> dict:
    """Read the pinned sources and the propagation rules."""
    return json.loads(SOURCES_FILE.read_text(encoding="utf-8"))


def source_path(cache: Path, sources: dict, key: str) -> Path:
    """Local path of a pinned source file."""
    return cache / Path(sources[key]["url"]).name


def fetch(url: str, destination: Path, expected_sha256: str) -> str:
    """Download ``url`` and verify it against the pinned digest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "fasta2go"})
    with urllib.request.urlopen(request, timeout=300) as response:
        destination.write_bytes(response.read())
    return verify_integrity(destination, expected_sha256)



def read_annotations(db_path: Path) -> list[tuple[str, list[str]]]:
    """Read ``(fasta_id, [go_id])`` for every protein, Molecular Function only.

    A protein without any Molecular Function annotation is kept with an empty
    list, so the export covers the whole database instead of only the annotated
    part.
    """
    connection = db.connect(db_path, read_only=True)
    try:
        duplicates = connection.execute(
            "SELECT count(*) FROM (SELECT fasta_id FROM proteins "
            "GROUP BY fasta_id HAVING count(*) > 1)"
        ).fetchone()[0]
        if duplicates:
            raise GoSlimError(
                f"{duplicates} fasta_id value(s) appear in more than one source "
                "file; the export is keyed by fasta_id and would merge them."
            )
        rows = connection.execute(
            """
            SELECT p.fasta_id,
                   list(g.go_id ORDER BY g.go_id)
                       FILTER (WHERE g.go_id IS NOT NULL) AS go_ids
            FROM proteins p
            LEFT JOIN protein_go_annotations g
                   ON g.source_file = p.source_file
                  AND g.fasta_id = p.fasta_id
                  AND g.aspect = ?
            GROUP BY p.fasta_id
            ORDER BY p.fasta_id
            """,
            [MF_NAMESPACE],
        ).fetchall()
    finally:
        connection.close()
    return [(fasta_id, go_ids or []) for fasta_id, go_ids in rows]


def check_examples(ontology, vocabulary, ancestor_map) -> int:
    """Regression-check the documented propagation cases.

    The examples are raw GO terms, so they do not depend on the database and
    stay valid for any snapshot.

    Raises:
        SystemExit: If an example disagrees with its recorded expectation.
    """
    payload = json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))
    for example in payload["examples"]:
        mappings, _ = map_proteins(
            [(example["protein_id"], example["raw_terms"])],
            ontology,
            vocabulary,
            ancestor_map,
        )
        expected = sorted(example["expected_labels"])
        if list(mappings[0].labels) != expected:
            raise SystemExit(
                f"{example['protein_id']}: expected {expected}, got "
                f"{list(mappings[0].labels)} ({example['demonstrates']})"
            )
    return len(payload["examples"])


def build_labels(db_path: Path, out_path: Path, sources: dict, cache: Path) -> None:
    """Verify the pinned sources, map the database and write the labels."""
    started = time.time()
    digests = {
        key: verify_integrity(source_path(cache, sources, key), sources[key]["sha256"])
        for key in ("ontology", "slim")
    }

    ontology = parse_obo(source_path(cache, sources, "ontology"))
    slim_ids = read_term_ids(source_path(cache, sources, "slim"))
    vocabulary = build_vocabulary(ontology, slim_ids)

    published = read_vocabulary_ids(VOCABULARY_FILE)
    if vocabulary.term_set != published or len(vocabulary) != EXPECTED_TERMS:
        raise SystemExit(
            "The pinned slim no longer matches data/go_slim/vocabulary_mf.csv: "
            f"{len(published)} published labels, {len(vocabulary)} derived "
            f"(added {sorted(vocabulary.term_set - published)}, "
            f"removed {sorted(published - vocabulary.term_set)}). "
            "Review the new GO release before relabelling."
        )
    if ASPECT_ROOT in vocabulary.term_set:
        raise SystemExit(f"The aspect root {ASPECT_ROOT} must never be a label.")

    ancestor_map = build_ancestor_map(ontology, vocabulary)
    examples = check_examples(ontology, vocabulary, ancestor_map)
    mappings, summary = map_proteins(
        read_annotations(db_path), ontology, vocabulary, ancestor_map
    )
    write_labels_csv(out_path, mappings)

    statuses = summary.as_dict()["statuses"]
    print(f"Ontology     {ontology.release}  sha256 {digests['ontology'][:16]}...")
    print(
        f"Slim         {sources['slim']['data_version']}  "
        f"sha256 {digests['slim'][:16]}..."
    )
    print(
        f"Vocabulary   {len(vocabulary)} labels "
        f"({len(vocabulary.removed)} excluded by rule)"
    )
    print(f"Examples     {examples} propagation case(s) reproduced")
    print(
        f"Proteins     {summary.proteins:,}, "
        f"{summary.proteins_with_labels:,} with at least one label "
        f"(max {summary.max_labels} per protein)"
    )
    print(
        f"Annotations  {summary.annotations:,}: "
        + ", ".join(f"{name}={count:,}" for name, count in statuses.items())
    )
    print(f"Labels used  {summary.labels_used}/{summary.vocabulary_terms}")
    print(
        f"Written      {out_path} (sha256 {sha256_file(out_path)[:16]}...) "
        f"in {round(time.time() - started, 1)}s"
    )


def ensure_sources(sources: dict, cache: Path) -> None:
    """Download what is missing and verify every pinned source."""
    cache.mkdir(parents=True, exist_ok=True)
    for key in ("ontology", "slim"):
        path = source_path(cache, sources, key)
        if not path.exists():
            fetch(sources[key]["url"], path, sources[key]["sha256"])
            print(f"fetched {key:<8} -> {path}")
        verify_integrity(path, sources[key]["sha256"])


def main() -> int:
    """Verify the pinned sources and write the labels of every protein."""
    sources = load_sources()
    try:
        ensure_sources(sources, CACHE_DIR)
        build_labels(DATABASE, OUTPUT, sources, CACHE_DIR)
    except (GoSlimError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


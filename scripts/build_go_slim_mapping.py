#!/usr/bin/env python3
"""Build the GO Slim MF label mapping for an auditable sample.

Reads only pinned sources, verifies their SHA-256, applies the documented
propagation rules and writes an auditable manifest plus per-protein labels.
The full base is not processed here.

Usage:
    uv run python scripts/build_go_slim_mapping.py --fetch          # download pinned sources
    uv run python scripts/build_go_slim_mapping.py --sample-size 400
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from protein_database.ml.go_slim import (  # noqa: E402
    ASPECT_ROOT,
    MF_NAMESPACE,
    STATUS_MAPPED,
    STATUS_NO_ANCESTOR,
    STATUS_OBSOLETE,
    STATUS_UNKNOWN,
    build_ancestor_map,
    build_vocabulary,
    check_slim_equivalence,
    classify_annotation,
    map_proteins,
    parse_obo,
    read_term_ids,
    sha256_file,
    verify_integrity,
    write_json,
    write_labels_csv,
    write_vocabulary_csv,
)

SOURCES_FILE = PROJECT_ROOT / "data" / "go_slim" / "sources.json"
EXAMPLES_FILE = PROJECT_ROOT / "data" / "go_slim" / "expected_examples.json"
ONTOLOGY_PATH = PROJECT_ROOT / "cache" / "go_slim" / "go-basic.obo"
SLIM_PATH = PROJECT_ROOT / "cache" / "go_slim" / "goslim_generic.obo"
SNAPSHOT_PATH = PROJECT_ROOT / "cache" / "uniprot" / "bulk" / "uniprot_sprot.tsv.gz"
OUT_DIR = PROJECT_ROOT / "exports" / "go_slim" / "sample"
MF_COLUMN = "Gene Ontology (molecular function)"
GO_ID = re.compile(r"GO:\d{7}")
ACCESSION = re.compile(r"^[A-Z0-9]{6,10}$")
USER_AGENT = "FASTA2GO/0.1 (TCC; GO Slim mapping)"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    """Path as reported in the manifest: relative to the repo when possible."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(Path(path))


def peak_rss_mb() -> float | None:
    """Peak resident set size of this process in MiB (``None`` off POSIX)."""
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return None
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def collect_status_examples(
    rows: list[tuple[str, list[str]]],
    ontology,
    vocabulary,
    ancestor_map,
    *,
    limit: int = 3,
) -> dict[str, list[dict]]:
    """Real (protein, term) examples for every status that is not ``mapped``."""
    examples: dict[str, list[dict]] = {}
    for protein_id, raw_ids in rows:
        for raw_id in sorted(set(raw_ids)):
            outcome = classify_annotation(
                raw_id, ontology, ancestor_map, vocabulary
            )
            if outcome.status == STATUS_MAPPED:
                continue
            bucket = examples.setdefault(outcome.status, [])
            if len(bucket) >= limit:
                continue
            term = ontology.terms.get(outcome.primary_id or "")
            bucket.append(
                {
                    "protein_id": protein_id,
                    "raw_id": raw_id,
                    "primary_id": outcome.primary_id,
                    "name": term.name if term else "",
                }
            )
    return examples


def fetch(url: str, destination: Path, expected_sha256: str) -> str:
    """Download ``url`` to ``destination`` and verify its hash.

    Raises:
        GoSlimError: If the downloaded bytes do not match the pin.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return verify_integrity(destination, expected_sha256)


def read_annotations(
    snapshot: Path,
    *,
    accessions: list[str] | None = None,
    limit: int | None = None,
    modulus: int = 97,
) -> list[tuple[str, list[str]]]:
    """Read (accession, MF GO ids) from the UniProt snapshot.

    With ``accessions`` the rows are chosen explicitly; otherwise a
    deterministic hash-based sample is taken (``limit`` rows, ordered by
    accession). The sample is *not* a random sample of the population.
    """
    conditions = [f'"{MF_COLUMN}" IS NOT NULL']
    if accessions:
        for accession in accessions:
            if not ACCESSION.match(accession):
                raise ValueError(f"Unsafe accession: {accession!r}")
        quoted = ", ".join(f"'{a}'" for a in accessions)
        conditions.append(f'"Entry" IN ({quoted})')
    else:
        conditions.append(f"hash(\"Entry\") % {modulus} = 0")
    sql = (
        f'SELECT "Entry", "{MF_COLUMN}" '
        "FROM read_csv(?, delim='\\t', header=true, union_by_name=true) "
        f"WHERE {' AND '.join(conditions)} ORDER BY \"Entry\""
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    connection = duckdb.connect(":memory:")
    try:
        rows = connection.execute(sql, [str(snapshot)]).fetchall()
    finally:
        connection.close()
    return [(entry, GO_ID.findall(cell)) for entry, cell in rows]


def check_examples(
    path: Path, snapshot: Path, ontology, vocabulary, ancestor_map
) -> list[dict]:
    """Verify the documented propagation examples against the real snapshot.

    Raises:
        SystemExit: If any example disagrees with its recorded expectation.
    """
    spec = load_json(path)
    results: list[dict] = []
    for case in spec["examples"]:
        protein_id = case["protein_id"]
        rows = read_annotations(snapshot, accessions=[protein_id])
        if not rows:
            results.append(
                {"protein_id": protein_id, "matched": False, "error": "not in snapshot"}
            )
            continue
        _, raw = rows[0]
        mappings, _ = map_proteins(
            [(protein_id, raw)], ontology, vocabulary, ancestor_map
        )
        mapping = mappings[0]
        expected = sorted(case["expected_labels"])
        matched = list(mapping.labels) == expected and list(mapping.raw_terms) == sorted(
            case["raw_terms"]
        )
        results.append(
            {
                "protein_id": protein_id,
                "matched": matched,
                "expected_labels": expected,
                "observed_labels": list(mapping.labels),
                "observed_terms": list(mapping.raw_terms),
                "demonstrates": case.get("demonstrates", ""),
            }
        )

    failed = [item for item in results if not item["matched"]]
    if failed:
        raise SystemExit(f"Example verification failed: {failed}")
    return results


def build_manifest(
    *,
    sources: dict,
    ontology,
    slim_release: str,
    ontology_digest: str,
    slim_digest: str,
    equivalence: dict,
    vocabulary,
    vocabulary_csv: Path,
    summary,
    sample: dict,
    examples: list[dict] | None,
    label_distribution: dict[int, int] | None = None,
    snapshot: dict | None = None,
    status_examples: dict | None = None,
) -> dict:
    """Assemble the auditable manifest: sources, rules, counts and limitations."""
    origin_counts: dict[str, int] = {}
    for term_id in vocabulary.terms:
        origin = vocabulary.origin.get(term_id, "unknown")
        origin_counts[origin] = origin_counts.get(origin, 0) + 1

    return {
        "schema": "fasta2go/go-slim-mapping/1",
        "mode": "sample",
        "not_a_population_estimate": True,
        "sources": {
            "ontology": {
                "name": sources["ontology"]["name"],
                "url": sources["ontology"]["url"],
                "data_version": ontology.release,
                "sha256": ontology_digest,
                "bytes": sources["ontology"]["bytes"],
                "versioned_in_git": False,
                "cache_path": sources["ontology"]["cache_path"],
            },
            "slim": {
                "name": sources["slim"]["name"],
                "subset": sources["slim"]["subset"],
                "url": sources["slim"]["url"],
                "data_version": slim_release,
                "sha256": slim_digest,
                "bytes": sources["slim"]["bytes"],
                "versioned_in_git": False,
            },
        },
        "integrity": {
            "verified_before_processing": True,
            "failure_mode": "hard error on hash mismatch (no silent fallback)",
        },
        "slim_equivalence": equivalence,
        "vocabulary": {
            "name": vocabulary.name,
            "terms": len(vocabulary),
            "origin_counts": origin_counts,
            "subset": vocabulary.subset,
            "extension": list(sources["vocabulary"]["extension"]),
            "extension_justification": sources["vocabulary"]["extension_justification"],
            "excluded": list(vocabulary.excluded),
            "excluded_justification": sources["vocabulary"]["excluded_justification"],
            "removed_by_exclusion": list(vocabulary.removed),
            "skipped_non_mf_terms": len(vocabulary.skipped_non_mf),
            "is_unmodified_official_slim": False,
            "csv": {
                "path": _relative(vocabulary_csv),
                "sha256": sha256_file(vocabulary_csv),
            },
        },
        "rules": {
            "aspect": MF_NAMESPACE,
            "relations": ["is_a", "part_of"],
            "propagation": "all_ancestors (transitive closure, term itself included)",
            "traversal": "is_a + part_of, expanded only within the vocabulary aspect",
            "aspect_guard": "a BP/CC node is never traversed, so a cross-aspect path cannot yield a Molecular Function label",
            "root_excluded": ASPECT_ROOT,
            "alt_id": "resolved to the primary id before traversal",
            "obsolete_terms": "excluded and counted as 'obsolete'; their labels are never used",
            "replaced_by": "not followed: an obsolete annotation stays 'obsolete' and does not silently become its replacement",
            "unknown_terms": "not in the ontology and not an alt_id; counted as 'unknown'",
            "dedup": "set union per protein (protein, label) pairs are unique",
            "status_partition": [
                "mapped",
                "no_vocabulary_ancestor",
                "obsolete",
                "unknown",
            ],
        },
        "inputs": {
            "snapshot": snapshot or {},
        },
        "sample": sample,
        "counts": summary.as_dict(),
        "coverage": {
            "labels_total": summary.vocabulary_terms,
            "labels_used": summary.labels_used,
            "per_label": dict(summary.label_counts),
            "labels_per_protein": (
                {str(k): v for k, v in sorted(label_distribution.items())}
                if label_distribution
                else {}
            ),
        },
        "examples": examples or [],
        "status_examples": status_examples or {},
        "limitations": [
            "The sample is selected by hash(Entry) % modulus == 0 ordered by "
            "accession; it is not a random sample and must not be read as a "
            "population estimate.",
            "Only Molecular Function annotations present in the UniProt snapshot "
            "are used; absence of an annotation is not evidence of absence of the "
            "function.",
            "Propagation follows is_a and part_of only.",
            "The vocabulary is the official generic GO slim plus the documented "
            "extension GO:0005488; it is not the unmodified slim.",
            "Annotations with no vocabulary ancestor are counted, never mapped.",
            "The full protein base was not processed in this run.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Verify sources, map the sample and write the auditable artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=SOURCES_FILE)
    parser.add_argument("--go-basic", type=Path, default=ONTOLOGY_PATH)
    parser.add_argument("--slim", type=Path, default=SLIM_PATH)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--vocabulary-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "go_slim" / "vocabulary_mf.csv",
        help="Versioned artifact describing the frozen vocabulary.",
    )
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--modulus", type=int, default=97)
    parser.add_argument(
        "--fetch", action="store_true", help="Download the pinned sources first."
    )
    parser.add_argument("--skip-examples", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    timings: dict[str, float] = {}
    last = [started]

    def tick(name: str) -> None:
        now = time.perf_counter()
        timings[name] = round(now - last[0], 3)
        last[0] = now

    sources = load_json(args.sources)

    if args.fetch:
        for key, path in (("ontology", args.go_basic), ("slim", args.slim)):
            after = fetch(sources[key]["url"], path, sources[key]["sha256"])
            print(f"  fetched {key:<8} -> {path} (sha256 {after[:16]}...)")

    ontology_digest = verify_integrity(args.go_basic, sources["ontology"]["sha256"])
    slim_digest = verify_integrity(args.slim, sources["slim"]["sha256"])
    snapshot_digest = sha256_file(args.snapshot)
    tick("integrity")

    ontology = parse_obo(args.go_basic)
    slim = parse_obo(args.slim)
    tick("parse_obo")
    slim_ids = frozenset(
        term_id for term_id, term in slim.terms.items() if not term.obsolete
    )

    equivalence = check_slim_equivalence(
        ontology, slim_ids, sources["slim"]["subset"]
    )
    if not equivalence["equivalent"]:
        raise SystemExit(
            "Slim file and ontology subset signature disagree; refusing to build a "
            f"vocabulary from stale sources: {equivalence}"
        )

    vocabulary = build_vocabulary(
        ontology,
        slim_ids,
        extension=sources["vocabulary"]["extension"],
        excluded=sources["vocabulary"]["excluded"],
        subset=sources["slim"]["subset"],
    )
    expected_terms = sources["vocabulary"]["expected_terms"]
    if len(vocabulary) != expected_terms:
        raise SystemExit(
            f"Vocabulary has {len(vocabulary)} terms, sources pin {expected_terms}. "
            "Review the release before updating the pin."
        )

    print("=" * 78)
    print("  GO Slim MF mapping - auditable sample")
    print("=" * 78)
    print(f"  ontology   : {ontology.release}  sha256 {ontology_digest[:16]}...")
    print(f"  slim       : {slim.release}  sha256 {slim_digest[:16]}...")
    print(
        f"  equivalence: {equivalence['declared_in_slim_file']} declared vs "
        f"{equivalence['tagged_in_ontology']} tagged -> "
        f"{'equivalent' if equivalence['equivalent'] else 'MISMATCH'}"
    )
    print(f"  vocabulary : {vocabulary.name} ({len(vocabulary)} terms)")

    ancestor_map = build_ancestor_map(ontology, vocabulary)
    tick("ancestor_map")
    rows = read_annotations(
        args.snapshot, limit=args.sample_size, modulus=args.modulus
    )
    tick("read_snapshot")
    mappings, summary = map_proteins(rows, ontology, vocabulary, ancestor_map)
    tick("map")
    status_examples = collect_status_examples(
        rows, ontology, vocabulary, ancestor_map
    )

    examples = None
    if not args.skip_examples and EXAMPLES_FILE.exists():
        examples = check_examples(
            EXAMPLES_FILE, args.snapshot, ontology, vocabulary, ancestor_map
        )
    tick("examples")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels_csv = write_labels_csv(args.out_dir / "protein_labels.csv", mappings)
    vocabulary_csv = write_vocabulary_csv(
        args.vocabulary_csv, vocabulary, ontology, ancestor_map
    )
    tick("write_csv")

    manifest = build_manifest(
        sources=sources,
        ontology=ontology,
        slim_release=slim.release,
        ontology_digest=ontology_digest,
        slim_digest=slim_digest,
        equivalence=equivalence,
        vocabulary=vocabulary,
        vocabulary_csv=vocabulary_csv,
        summary=summary,
        sample={
            "limit": args.sample_size,
            "modulus": args.modulus,
            "proteins": len(mappings),
            "selection": "hash(Entry) % modulus == 0, ordered by Entry",
        },
        examples=examples,
        label_distribution=dict(
            Counter(len(mapping.labels) for mapping in mappings)
        ),
        snapshot={
            "path": _relative(args.snapshot),
            "sha256": snapshot_digest,
            "bytes": args.snapshot.stat().st_size,
            "column": MF_COLUMN,
        },
        status_examples=status_examples,
    )
    manifest["outputs"] = {
        "protein_labels_csv": {
            "path": _relative(labels_csv),
            "sha256": sha256_file(labels_csv),
            "rows": len(mappings),
        }
    }
    manifest_path = write_json(args.out_dir / "manifest.json", manifest)
    tick("write_manifest")

    run_stats = {
        "schema": "fasta2go/go-slim-run-stats/1",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_rss_mb": peak_rss_mb(),
        "phases_seconds": timings,
        "python": platform.python_version(),
        "proteins": len(mappings),
        "note": (
            "Timing and memory are machine-dependent and deliberately kept out of "
            "manifest.json, which stays byte-deterministic."
        ),
    }
    stats_path = write_json(args.out_dir / "run_stats.json", run_stats)

    counts = summary.as_dict()
    print("-" * 78)
    print(
        f"  proteins {counts['proteins']:,} | annotations {counts['annotations']:,} "
        f"| without labels {counts['proteins_without_labels']:,}"
    )
    for status, value in counts["statuses"].items():
        print(f"    {status:<24} {value:>7,}")
    print(
        f"  labels used {counts['labels_used']}/{counts['vocabulary_terms']} "
        f"| max per protein {counts['max_labels_per_protein']}"
    )
    if examples:
        print("-" * 78)
        for item in examples:
            print(
                f"  example {item['protein_id']}: "
                f"{'OK' if item['matched'] else 'MISMATCH'} "
                f"({len(item['observed_labels'])} labels)"
            )
    print("-" * 78)
    print(f"  vocabulary : {vocabulary_csv}")
    print(f"  labels     : {labels_csv}")
    print(f"  manifest   : {manifest_path}")
    print(f"  stats      : {stats_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

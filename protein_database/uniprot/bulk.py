"""Bulk acquisition of UniProt annotations via the streaming REST API.

Fetching ~575k accessions through the per-batch search endpoint needs
thousands of rate-limited requests (100 accessions per query). The
``uniprotkb/stream`` endpoint delivers the *entire reviewed proteome* in a
single gzipped TSV response using exactly the fields parsed by
``protein_database.uniprot.parser``, so one download replaces the whole
batched retrieval.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import requests

from protein_database.uniprot.client import STREAM_URL, UNIPROT_FIELDS, UniProtClient

#: Query selecting all of SwissProt (reviewed UniProtKB).
REVIEWED_QUERY = "reviewed:true"

#: Filename of the bulk snapshot under the bulk cache directory.
SNAPSHOT_FILENAME = "uniprot_sprot.tsv.gz"

#: Name of the SHA-1 manifest file written next to the snapshot.
SNAPSHOT_MANIFEST = "snapshot.sha1"


def split_query(terms: list[str], chunk_size: int = 100) -> list[str]:
    """Split an accession list into API-sized OR chunks (documented behavior).

    The UniProtKB search/stream query language rejects queries with more
    than 100 OR conditions, so long accession lists must be fetched as
    sequential chunk downloads.
    """
    return [
        " OR ".join(f"accession:{acc}" for acc in terms[i : i + chunk_size])
        for i in range(0, len(terms), chunk_size)
    ]


def _snapshot_fingerprint(queries: list[str]) -> str:
    return hashlib.sha1("\n".join(queries).encode()).hexdigest()


def download_chunk(
    client: UniProtClient,
    query: str,
    dest: Path,
    fingerprint: str,
) -> None:
    """Download one stream response chunk to ``dest``.

    The body is streamed to ``dest`` with a ``.part`` suffix during the
    transfer, then renamed into place only after the stream completes, so
    an interrupted run can never be mistaken for a complete chunk. The
    SHA-1 manifest entry for the chunk is written alongside it.

    Args:
        client: Configured UniProtClient (retry/backoff).
        query: UniProtKB query for this chunk.
        dest: Final destination path (``<hex>.tsv.gz``).
        fingerprint: SHA-1 fingerprint for the whole snapshot download.
    """
    part = dest.with_suffix(dest.suffix + ".part")
    client.fetch_stream(query, part)
    part.rename(dest)
    manifest = dest.parent / f"{fingerprint}.sha1"
    existing = manifest.read_text(encoding="utf-8") if manifest.exists() else ""
    manifest.write_text(f"{existing}{dest.name}\n", encoding="utf-8")


def download_snapshot(
    accessions: list[str] | None = None,
    dest_dir: str | Path = "cache/uniprot/bulk",
    refresh: bool = False,
    client: UniProtClient | None = None,
) -> list[Path]:
    """Download the bulk TSV snapshot as one chunk per 100-accession query.

    An explicit ``accessions`` list (or, when None, the whole reviewed
    proteome) is split into <=100 OR-condition chunks; each chunk is a
    single streaming response saved as a gzipped TSV. Reuses an existing
    snapshot (verified against its manifest) unless ``refresh`` is True.

    Args:
        accessions: Explicit accessions to fetch; None fetches everything
            reviewed (SwissProt). Kept for reproducibility and debugging.
        dest_dir: Directory holding the snapshot chunks and manifest.
        refresh: If True, re-download even when a snapshot exists.
        client: Optional configured UniProtClient (retry/backoff).

    Returns:
        Sorted list of snapshot chunk paths.
    """
    out_dir = Path(dest_dir)
    queries = (
        split_query(sorted(accessions)) if accessions else [REVIEWED_QUERY]
    )
    fingerprint = _snapshot_fingerprint(queries)
    manifest = out_dir / f"{fingerprint}.sha1"
    if manifest.exists() and not refresh:
        chunks = [
            out_dir / name
            for name in manifest.read_text(encoding="utf-8").splitlines()
            if name
        ]
        if chunks and all(p.exists() for p in chunks):
            return sorted(chunks)

    client = client or UniProtClient()
    out_dir.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        manifest.unlink()
    results: list[Path] = []
    for i, query in enumerate(queries):
        digest = hashlib.sha1(query.encode()).hexdigest()
        dest = out_dir / f"{digest}.tsv.gz"
        download_chunk(client, query, dest, fingerprint)
        results.append(dest)
    return sorted(results)

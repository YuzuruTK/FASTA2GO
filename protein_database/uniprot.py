"""UniProt enrichment: bulk snapshot or incremental API, one native ingest.

Bulk mode (default): a single request to the UniProtKB ``stream`` endpoint
delivers the whole reviewed proteome as gzipped TSV. DuckDB's native
``read_csv`` ingests the file(s) into a temp staging table, and one set-based
``INSERT ... ON CONFLICT`` fans the annotations out to every matching local
protein row.

API mode: only the accessions that are missing or stale are requested from
the UniProtKB ``/accessions`` endpoint (1000 accessions per request), and the
responses go through the exact same staging table and upsert statement.
"""

import glob as glob_module
import gzip
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import duckdb
import requests

from protein_database import db, pipeline
from protein_database.progress import track, track_bytes
from protein_database.schema import ANNOTATION_COLUMNS

STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
ACCESSIONS_URL = "https://rest.uniprot.org/uniprotkb/accessions"
REVIEWED_QUERY = "reviewed:true"
SNAPSHOT_NAME = "uniprot_sprot.tsv.gz"

#: UniProt allows at most this many accessions per ``/accessions`` request
#: (verified: 2000 returns HTTP 400 "Only '1000' accessions are allowed").
MAX_ACCESSIONS_PER_REQUEST = 1000

#: HTTP statuses worth retrying: rate limiting plus transient server errors.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

#: Accession shape accepted by UniProtKB (optional isoform suffix).
ACCESSION_PATTERN = re.compile(r"^[A-Z0-9]{6,10}(-\d+)?$")

#: Status recorded for an accession UniProt does not return (removed/invalid).
STATUS_NOT_FOUND = "not_found"

#: Verified UniProtKB return-field names (must match the header, in order).
UNIPROT_FIELDS = [
    "accession",
    "id",
    "protein_name",
    "gene_names",
    "cc_function",
    "cc_subcellular_location",
    "protein_existence",
    "organism_name",
    "keyword",
    "go_p",
    "go_c",
    "go_f",
    "go_id",
]

#: UniProt TSV headers for the Gene Ontology columns (they are delivered by
#: the snapshot but not mapped into ``uniprot_annotations``).
GO_HEADERS = {
    "Gene Ontology (biological process)",
    "Gene Ontology (cellular component)",
    "Gene Ontology (molecular function)",
    "Gene Ontology IDs",
}

#: UniProt TSV header -> database annotation column.
TSV_MAP = {
    "Entry": "uniprot_accession",
    "Entry Name": "uniprot_id",
    "Protein names": "protein_name",
    "Gene Names": "gene_name",
    "Function [CC]": "function",
    "Subcellular location [CC]": "subcellular_location",
    "Protein existence": "protein_existence",
    "Organism": "organism",
    "Keywords": "keywords",
}

STAGING_TABLE = "uniprot_staging"


class AccessionFormatError(Exception):
    """UniProt rejected the accession list with HTTP 400."""


def _quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def download_snapshot(
    bulk_dir: str | Path = "cache/uniprot/bulk", refresh: bool = False
) -> Path:
    """Download the reviewed-proteome TSV once and reuse it thereafter.

    The body is streamed to a ``.part`` file and renamed only after the
    transfer completes, so an interrupted run is never mistaken for a
    complete snapshot.
    """
    dest = Path(bulk_dir) / SNAPSHOT_NAME
    if dest.exists() and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with requests.get(
        STREAM_URL,
        params={
            "query": REVIEWED_QUERY,
            "format": "tsv",
            "fields": ",".join(UNIPROT_FIELDS),
            "compressed": "true",
        },
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()
        track_bytes(response, part, desc="Downloading snapshot")
    part.rename(dest)
    return dest


def _read_header(path: str | Path) -> list[str]:
    """Return the tab-separated header line of a (optionally gzipped) TSV."""
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8", newline="") as handle:
        line = handle.readline()
    return [column.strip() for column in line.rstrip("\r\n").split("\t")]


def _check_headers(files: list[str]) -> None:
    """Fail loudly when a source file cannot supply every mapped column.

    ``read_csv`` is told to sniff the columns (``all_varchar``) so that old
    and new snapshot vintages intersect cleanly, which makes a missing
    column a silent NULL instead of an error. This guard turns that into a
    clear message.
    """
    expected = set(TSV_MAP)
    for path in files:
        missing = expected - set(_read_header(path))
        if missing:
            raise ValueError(
                f"UniProt TSV {path!r} is missing columns {sorted(missing)}. "
                "Re-download the snapshot with --refresh-snapshot."
            )


def _staging_sql(sources: str) -> str:
    """SQL that ingests TSV source(s) into the deduped staging table.

    Column types are sniffed (``all_varchar``) instead of declared: the
    snapshot carries the four GO columns that ``TSV_MAP`` deliberately does
    not map, and legacy per-batch TSVs carry only the mapped nine. Declaring
    either set explicitly makes DuckDB reject the other.

    The staging table is created explicit (not ``OR REPLACE``) so the caller
    controls when the previous chunk is dropped.
    """
    selected = ", ".join(f'"{header}"' for header in TSV_MAP)
    aggregated = ", ".join(
        f'arg_min("{header}", _src) AS "{header}"'
        for header in TSV_MAP
        if header != "Entry"
    )
    return (
        f'SELECT "Entry", {aggregated} FROM ('
        f"SELECT filename AS _src, {selected} FROM read_csv({sources}, "
        f"delim='\\t', header=true, union_by_name=true, all_varchar=true, "
        f"filename=true)"
        f') WHERE "Entry" IS NOT NULL GROUP BY "Entry"'
    )


def _build_staging_table(conn: duckdb.DuckDBPyConnection, files: list[str]) -> int:
    """Ingest TSV file(s) into a deduped temp table (earlier files win)."""
    _check_headers(files)
    sources = "[" + ", ".join(_quote(path) for path in files) + "]"
    conn.execute(
        f"CREATE OR REPLACE TEMP TABLE {STAGING_TABLE} AS {_staging_sql(sources)}"
    )
    return conn.execute(f"SELECT count(*) FROM {STAGING_TABLE}").fetchone()[0]


def _drop_staging_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {STAGING_TABLE}")


def _staged_join_count(conn: duckdb.DuckDBPyConnection, predicate: str = "") -> int:
    """Count local protein rows matching the staged accessions."""
    sql = (
        f"SELECT count(*) FROM proteins p JOIN {STAGING_TABLE} s "
        f"ON split_part(p.fasta_id, '|', 1) IN ('sp', 'tr') "
        'AND upper(split_part(p.fasta_id, \'|\', 2)) = s."Entry"'
    )
    if predicate:
        sql += f" WHERE {predicate}"
    return conn.execute(sql).fetchone()[0]


def _utc_naive(value: datetime | None = None) -> datetime:
    """Return a naive UTC timestamp (DuckDB ``TIMESTAMP`` has no timezone)."""
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _upsert_annotations(
    conn: duckdb.DuckDBPyConnection,
    retrieved_at: datetime | None = None,
) -> tuple[int, int]:
    """Fan staged rows out to local proteins; return (written, missing_fn)."""
    retrieved_at = _utc_naive(retrieved_at)
    written = _staged_join_count(conn)
    missing = _staged_join_count(
        conn, 's."Function [CC]" IS NULL OR trim(s."Function [CC]") = \'\''
    )
    columns = ", ".join(ANNOTATION_COLUMNS)
    updates = ", ".join(
        f"{column} = excluded.{column}"
        for column in ANNOTATION_COLUMNS
        if column not in ("fasta_id", "source_file")
    )
    select = (
        "p.fasta_id, p.source_file, "
        's."Entry", s."Entry Name", s."Protein names", s."Gene Names", '
        's."Function [CC]", s."Subcellular location [CC]", '
        's."Protein existence", s."Organism", s."Keywords", '
        "CASE WHEN s.\"Function [CC]\" IS NULL "
        "OR trim(s.\"Function [CC]\") = '' "
        "THEN 'missing_function' ELSE 'fetched' END, ?"
    )
    conn.execute(
        f"INSERT INTO uniprot_annotations ({columns}) SELECT {select} "
        f"FROM proteins p JOIN {STAGING_TABLE} s "
        f"ON split_part(p.fasta_id, '|', 1) IN ('sp', 'tr') "
        f"AND upper(split_part(p.fasta_id, '|', 2)) = s.\"Entry\" "
        f"ON CONFLICT (fasta_id, source_file) DO UPDATE SET {updates}",
        [retrieved_at],
    )
    return written, missing


def _function_coverage(conn: duckdb.DuckDBPyConnection) -> float:
    """Percentage of local proteins that carry a non-empty function text."""
    total = conn.execute("SELECT count(*) FROM proteins").fetchone()[0]
    if not total:
        return 0.0
    with_function = conn.execute(
        "SELECT count(*) FROM uniprot_annotations "
        "WHERE function IS NOT NULL AND function <> ''"
    ).fetchone()[0]
    return with_function / total * 100.0


def select_stale_rows(
    conn: duckdb.DuckDBPyConnection,
    *,
    only_missing: bool = False,
    older_than_days: int | None = None,
    limit: int | None = None,
) -> list[tuple[str, str, str]]:
    """Return ``(fasta_id, source_file, accession)`` rows needing annotations.

    Without filters every parsable local protein is returned (a full API
    refresh); ``only_missing`` keeps rows with no annotation row at all, and
    ``older_than_days`` keeps rows whose ``retrieved_at`` is older than the
    cutoff (missing rows always qualify).
    """
    conditions = ["split_part(p.fasta_id, '|', 1) IN ('sp', 'tr')"]
    params: list = []
    if only_missing:
        conditions.append("a.fasta_id IS NULL")
    if older_than_days is not None:
        conditions.append(
            "(a.fasta_id IS NULL OR a.retrieved_at "
            "< timezone('UTC', now()) - CAST(? AS INTEGER) * INTERVAL 1 DAY)"
        )
        params.append(older_than_days)
    sql = (
        "SELECT p.fasta_id, p.source_file, "
        "upper(split_part(p.fasta_id, '|', 2)) "
        "FROM proteins p "
        "LEFT JOIN uniprot_annotations a USING (fasta_id, source_file) "
        f"WHERE {' AND '.join(conditions)} "
        "ORDER BY 3, 1, 2"
    )
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


def _backoff(
    attempt: int, base_delay: float, max_delay: float, retry_after: str | None
) -> float:
    """Exponential backoff in seconds, honouring ``Retry-After`` when present."""
    delay = min(base_delay * (2**attempt), max_delay)
    if retry_after:
        try:
            delay = min(float(retry_after), max_delay)
        except ValueError:
            pass
    return delay


def _request_tsv(
    accessions: list[str],
    session: requests.Session,
    *,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> str:
    """GET one accession batch from UniProtKB, retrying transient failures."""
    params = {
        "accessions": ",".join(accessions),
        "format": "tsv",
        "fields": ",".join(UNIPROT_FIELDS),
    }
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        retry_after: str | None = None
        try:
            response = session.get(ACCESSIONS_URL, params=params, timeout=120)
        except requests.RequestException as exc:
            last_error = exc
        else:
            if response.status_code == 200:
                return response.text
            if response.status_code == 400:
                raise AccessionFormatError(response.text.strip()[:300])
            if response.status_code not in RETRYABLE_STATUSES:
                response.raise_for_status()
                raise requests.HTTPError(  # pragma: no cover - defensive
                    f"Unexpected HTTP {response.status_code}: {response.text[:200]}"
                )
            retry_after = response.headers.get("Retry-After")
            last_error = requests.HTTPError(
                f"HTTP {response.status_code}: {response.text[:200]}"
            )
        if attempt == max_retries:
            break
        time.sleep(_backoff(attempt, base_delay, max_delay, retry_after))
    raise last_error or requests.HTTPError("UniProt request failed")


def _fetch_bisecting(
    accessions: list[str],
    session: requests.Session,
    *,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> tuple[str | None, list[str], list[str]]:
    """Fetch a batch, halving it when UniProt rejects the accession list.

    A single malformed accession makes ``/accessions`` answer HTTP 400 for
    the whole batch, so the batch is split recursively until the offender is
    isolated. Returns ``(header, row_lines, rejected_accessions)``.
    """
    try:
        text = _request_tsv(
            accessions,
            session,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )
    except AccessionFormatError:
        if len(accessions) == 1:
            return None, [], accessions
        middle = len(accessions) // 2
        left = _fetch_bisecting(
            accessions[:middle],
            session,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )
        right = _fetch_bisecting(
            accessions[middle:],
            session,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )
        return left[0] or right[0], left[1] + right[1], left[2] + right[2]
    lines = text.splitlines()
    if not lines:
        return None, [], []
    return lines[0], [line for line in lines[1:] if line.strip()], []


def fetch_tsv(
    accessions: list[str],
    *,
    session: requests.Session | None = None,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> tuple[str | None, list[str], list[str]]:
    """Fetch one API batch; return ``(header, rows, rejected accessions)``.

    Accessions that cannot be valid UniProtKB accessions are rejected before
    the request, so a single odd FASTA identifier never costs a round trip.
    """
    valid = [value for value in accessions if ACCESSION_PATTERN.match(value)]
    rejected = [value for value in accessions if not ACCESSION_PATTERN.match(value)]
    if not valid:
        return None, [], rejected
    header, rows, invalid = _fetch_bisecting(
        valid,
        session or requests.Session(),
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
    )
    return header, rows, rejected + invalid


def _stage_api_chunk(
    conn: duckdb.DuckDBPyConnection,
    header: str | None,
    rows: list[str],
    cache_dir: str | Path,
) -> int:
    """Stage one API response; return the number of distinct accessions."""
    if not rows or header is None:
        _drop_staging_table(conn)
        return 0
    api_cache = Path(cache_dir) / "api"
    api_cache.mkdir(parents=True, exist_ok=True)
    path = api_cache / f"chunk_{uuid4().hex}.tsv"
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    try:
        return _build_staging_table(conn, [str(path)])
    finally:
        path.unlink(missing_ok=True)


def _mark_not_found(
    conn: duckdb.DuckDBPyConnection,
    keys: list[tuple[str, str]],
    retrieved_at: datetime,
    cache_dir: str | Path,
) -> int:
    """Record ``status='not_found'`` rows so they are not requested forever."""
    if not keys:
        return 0
    retrieved_at = _utc_naive(retrieved_at)
    blank = [None] * (len(ANNOTATION_COLUMNS) - 4)
    rows = [
        [fasta_id, source_file, *blank, STATUS_NOT_FOUND, retrieved_at]
        for fasta_id, source_file in keys
    ]
    db.bulk_upsert(
        conn,
        "uniprot_annotations",
        ANNOTATION_COLUMNS,
        rows,
        ["fasta_id", "source_file"],
        [
            column
            for column in ANNOTATION_COLUMNS
            if column not in ("fasta_id", "source_file")
        ],
        cache_dir,
    )
    return len(rows)


def _process_api_chunk(
    conn: duckdb.DuckDBPyConnection,
    chunk: list[str],
    result: tuple[str | None, list[str], list[str]],
    *,
    keys_by_accession: dict[str, list[tuple[str, str]]],
    retrieved_at: datetime,
    cache_dir: str | Path,
    stats: dict[str, int],
) -> None:
    """Stage, upsert and book-keep one fetched chunk."""
    header, rows, rejected = result
    staged = _stage_api_chunk(conn, header, rows, cache_dir)
    try:
        if staged:
            written, missing = _upsert_annotations(conn, retrieved_at)
            found = {
                row[0].upper()
                for row in conn.execute(f'SELECT "Entry" FROM {STAGING_TABLE}').fetchall()
            }
        else:
            written = missing = 0
            found = set()
    finally:
        _drop_staging_table(conn)

    absent = (set(chunk) - found) | set(rejected)
    labels = [
        key
        for accession in sorted(absent)
        for key in keys_by_accession.get(accession, [])
    ]
    stats["annotated"] += written
    stats["missing_function"] += missing
    stats["not_found"] += _mark_not_found(conn, labels, retrieved_at, cache_dir)


def _annotate_via_api(
    conn: duckdb.DuckDBPyConnection,
    *,
    cache_dir: str | Path,
    only_missing: bool,
    older_than_days: int | None,
    limit: int | None,
    chunk_size: int,
    workers: int,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> dict:
    """Fetch only missing/stale accessions and upsert them chunk by chunk."""
    total = conn.execute("SELECT count(*) FROM proteins").fetchone()[0]
    selection = select_stale_rows(
        conn,
        only_missing=only_missing,
        older_than_days=older_than_days,
        limit=limit,
    )
    if not selection:
        print("Nothing to annotate: no local accession is missing or stale.")
        return {
            "total_local_proteins": total,
            "annotated": 0,
            "missing_function": 0,
            "not_found": 0,
            "function_coverage_pct": _function_coverage(conn),
            "files": 0,
            "mode": "api",
            "accessions": 0,
            "http_requests": 0,
            "failed": 0,
        }

    keys_by_accession: dict[str, list[tuple[str, str]]] = {}
    for fasta_id, source_file, accession in selection:
        keys_by_accession.setdefault(accession, []).append((fasta_id, source_file))
    accessions = sorted(keys_by_accession)
    chunks = [
        accessions[index : index + chunk_size]
        for index in range(0, len(accessions), chunk_size)
    ]
    print(
        f"{len(accessions):,} accessions need annotations "
        f"({len(chunks):,} request(s) of up to {chunk_size:,})."
    )

    retrieved_at = datetime.now(timezone.utc)
    stats = {"annotated": 0, "missing_function": 0, "not_found": 0, "failed": 0}
    requests_made = 0
    thread_local = threading.local()

    def _fetch(chunk: list[str]) -> tuple[str | None, list[str], list[str]]:
        session = getattr(thread_local, "session", None)
        if session is None:
            session = requests.Session()  # requests.Session is not thread-safe
            thread_local.session = session
        return fetch_tsv(
            chunk,
            session=session,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )

    def _run(chunk: list[str], future) -> None:
        nonlocal requests_made
        requests_made += 1
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001 - keep going, retry next run
            stats["failed"] += len(chunk)
            print(f"Warning: {len(chunk)} accessions failed: {exc}")
            return
        _process_api_chunk(
            conn,
            chunk,
            result,
            keys_by_accession=keys_by_accession,
            retrieved_at=retrieved_at,
            cache_dir=cache_dir,
            stats=stats,
        )

    window: deque = deque()
    in_flight = max(1, workers) * 2
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for chunk in track(chunks, desc="Annotating (api)", unit="batch"):
            window.append((chunk, pool.submit(_fetch, chunk)))
            if len(window) >= in_flight:
                pending_chunk, future = window.popleft()
                _run(pending_chunk, future)
        while window:
            pending_chunk, future = window.popleft()
            _run(pending_chunk, future)

    return {
        "total_local_proteins": total,
        "annotated": stats["annotated"],
        "missing_function": stats["missing_function"],
        "not_found": stats["not_found"],
        "function_coverage_pct": _function_coverage(conn),
        "files": 0,
        "mode": "api",
        "accessions": len(accessions),
        "http_requests": requests_made,
        "failed": stats["failed"],
    }


def annotate_database(
    db_path: str | Path = "proteins.duckdb",
    cache_dir: str | Path = "cache/uniprot",
    source: str = "bulk",
    refresh_snapshot: bool = False,
    *,
    only_missing: bool = False,
    older_than_days: int | None = None,
    limit: int | None = None,
    chunk_size: int = MAX_ACCESSIONS_PER_REQUEST,
    workers: int = 4,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> dict:
    """Enrich the protein database with UniProt annotations.

    Args:
        db_path: Path to the DuckDB database.
        cache_dir: Directory for the snapshot, staging files and any legacy
            ``*.tsv`` files.
        source: ``'bulk'`` downloads the snapshot (plus any legacy cache),
            ``'cache'`` ingests only local ``*.tsv``/``*.tsv.gz`` files, and
            ``'api'`` fetches just the missing/stale accessions.
        refresh_snapshot: Re-download the bulk snapshot even when cached.
        only_missing: API mode only - select proteins without an annotation.
        older_than_days: API mode only - also refresh annotations older than
            N days (missing rows always qualify).
        limit: API mode only - maximum number of local proteins to select.
        chunk_size: Accessions per API request (1..1000).
        workers: Concurrent HTTP fetches (database writes stay single-threaded).
        max_retries: Retries per request for 429/5xx responses.
        base_delay: Initial backoff delay in seconds.
        max_delay: Maximum backoff delay in seconds.

    Returns:
        Dict with run statistics.
    """
    if source not in ("bulk", "cache", "api"):
        raise ValueError(
            f"Unknown source: {source!r} (expected 'bulk', 'cache' or 'api')."
        )
    if not 1 <= chunk_size <= MAX_ACCESSIONS_PER_REQUEST:
        raise ValueError(
            f"chunk_size must be between 1 and {MAX_ACCESSIONS_PER_REQUEST}."
        )
    if source != "api" and (only_missing or older_than_days is not None or limit):
        raise ValueError("only_missing/older_than_days/limit require source='api'.")

    conn = db.connect(db_path)
    try:
        db.init_database(conn)

        if source == "api":
            return _annotate_via_api(
                conn,
                cache_dir=cache_dir,
                only_missing=only_missing,
                older_than_days=older_than_days,
                limit=limit,
                chunk_size=chunk_size,
                workers=workers,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
            )

        cache = Path(cache_dir)
        if source == "bulk":
            files = [str(download_snapshot(cache / "bulk", refresh_snapshot))]
            # Merge any legacy per-batch TSVs alongside the snapshot.
            files += sorted(glob_module.glob(str(cache / "*.tsv")))
        else:
            # Offline replay of every TSV/TSV.gz under the cache.
            files = sorted(
                glob_module.glob(str(cache / "**" / "*.tsv*"), recursive=True)
            )
        if not files:
            raise ValueError("No UniProt TSV sources found.")

        print("Backfilling sequence features and amino-acid composition...")
        report = pipeline.backfill_features(conn)
        print(f"Backfilled {report['written']:,} proteins ({report['errors']} errors).")

        staged = _build_staging_table(conn, files)
        print(f"Staged {staged:,} distinct accessions from {len(files)} file(s).")

        retrieved_at = datetime.now(timezone.utc)
        written, missing_function = _upsert_annotations(conn, retrieved_at)
        print(
            f"Upserted {written:,} annotation rows "
            f"({missing_function:,} missing function)."
        )

        not_found_labels = conn.execute(
            f"SELECT p.fasta_id, p.source_file FROM proteins p "
            f"WHERE split_part(p.fasta_id, '|', 1) IN ('sp', 'tr') "
            f"AND upper(split_part(p.fasta_id, '|', 2)) NOT IN "
            f'(SELECT "Entry" FROM {STAGING_TABLE})'
        ).fetchall()
        recorded = _mark_not_found(conn, not_found_labels, retrieved_at, cache_dir)
        if recorded:
            print(f"Recorded {recorded:,} proteins as not found in UniProt.")

        total = conn.execute("SELECT count(*) FROM proteins").fetchone()[0]
        return {
            "total_local_proteins": total,
            "annotated": written,
            "missing_function": missing_function,
            "not_found": recorded,
            "function_coverage_pct": _function_coverage(conn),
            "files": len(files),
            "mode": source,
            "accessions": staged,
            "http_requests": 1,
            "failed": 0,
        }
    finally:
        conn.close()


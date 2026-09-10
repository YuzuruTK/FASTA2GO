"""UniProtKB REST API client with retry/backoff and rate-limit handling."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests

from protein_database.uniprot.batching import MAX_OR_CONDITIONS, build_accession_query

logger = logging.getLogger(__name__)

SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

#: Verified UniProtKB return-field names (TSV).
UNIPROT_FIELDS = ",".join(
    [
        "accession",
        "id",
        "protein_name",
        "gene_names",
        "cc_function",
        "cc_subcellular_location",
        "protein_existence",
        "organism_name",
        "keyword",
    ]
)

#: Bulk streaming endpoint: delivers full query result sets in one response.
STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"

#: HTTP statuses treated as transient (retryable with backoff).
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass
class HttpResponse:
    """Minimal HTTP response wrapper for testability."""

    status_code: int
    text: str
    headers: dict[str, str] | None = None

    def get_header(self, name: str) -> str | None:
        if not self.headers:
            return None
        for k, v in self.headers.items():
            if k.lower() == name.lower():
                return v
        return None


class UniProtClient:
    """Client for the UniProtKB REST search endpoint.

    Args:
        batch_size: Maximum accessions per request. Capped at the UniProt
            API's hard limit of 100 OR conditions (default 100).
        max_retries: Maximum retries for transient failures (default 5).
        base_delay: Initial backoff delay in seconds (default 1.0).
        max_delay: Maximum backoff delay in seconds (default 30.0).
        session: Optional requests.Session for connection reuse.
    """

    def __init__(
        self,
        batch_size: int = MAX_OR_CONDITIONS,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.batch_size = min(batch_size, MAX_OR_CONDITIONS)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.session = session or requests.Session()

    def fetch_batch(self, accessions: list[str]) -> HttpResponse:
        """Fetch annotations for a batch of accessions via TSV.

        Args:
            accessions: List of UniProt accessions.

        Returns:
            HttpResponse with the TSV text.

        Raises:
            requests.RequestException: On non-transient HTTP errors after
                retries are exhausted.
        """
        query = build_accession_query(accessions)
        params = {
            "query": query,
            "format": "tsv",
            "fields": UNIPROT_FIELDS,
            "size": str(len(accessions)),
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(SEARCH_URL, params=params, timeout=60)
            except requests.RequestException as exc:
                last_error = exc
                logger.warning("Request error (attempt %d/%d): %s", attempt + 1, self.max_retries + 1, exc)
                delay = self._backoff_delay(attempt, retry_after=None)
                time.sleep(delay)
                continue

            if resp.status_code == 200:
                return HttpResponse(
                    status_code=resp.status_code,
                    text=resp.text,
                    headers=dict(resp.headers or {}),
                )

            if resp.status_code in RETRYABLE_STATUSES:
                retry_after = resp.get_header("Retry-After")
                delay = self._backoff_delay(attempt, retry_after)
                logger.warning(
                    "Retryable status %d (attempt %d/%d): retrying in %.1fs",
                    resp.status_code,
                    attempt + 1,
                    self.max_retries + 1,
                    delay,
                )
                time.sleep(delay)
                last_error = requests.HTTPError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}"
                )
                continue

            # Non-retryable (e.g. 400 invalid fields)
            raise requests.HTTPError(
                f"Non-retryable HTTP {resp.status_code}: {resp.text[:300]}"
            )

        raise requests.RequestException(
            f"Exhausted {self.max_retries} retries. Last error: {last_error}"
        )

    def fetch_stream(
        self,
        query: str,
        dest: "Path | str",
        chunk_size: int = 1 << 20,
    ) -> "Path":
        """Download a bulk TSV snapshot via the UniProt streaming endpoint.

        The full result set for ``query`` is streamed to ``dest`` in chunks.
        A ``.part`` suffix is used while downloading, so an interrupted run
        can never be mistaken for a complete snapshot; callers should rename
        the returned path to its final name after verifying the download.

        Args:
            query: UniProtKB query (e.g. ``reviewed:true``).
            dest: Destination path for the (possibly gzipped) stream.
            chunk_size: Bytes per streamed chunk.

        Returns:
            The destination path.

        Raises:
            requests.RequestException: On transient errors after retries are
                exhausted, or non-retryable HTTP statuses.
        """
        from pathlib import Path as _Path

        dest_path = _Path(dest)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(
                    STREAM_URL,
                    params={
                        "query": query,
                        "format": "tsv",
                        "fields": UNIPROT_FIELDS,
                        "compressed": "true",
                    },
                    stream=True,
                    timeout=120,
                )
            except requests.RequestException as exc:
                last_error = exc
                logger.warning(
                    "Stream error (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                time.sleep(self._backoff_delay(attempt, None))
                continue

            if resp.status_code == 200:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with open(dest_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            fh.write(chunk)
                return dest_path

            if resp.status_code in RETRYABLE_STATUSES:
                retry_after = resp.get_header("Retry-After")
                delay = self._backoff_delay(attempt, retry_after)
                logger.warning(
                    "Retryable stream status %d (attempt %d/%d): retrying in %.1fs",
                    resp.status_code,
                    attempt + 1,
                    self.max_retries + 1,
                    delay,
                )
                time.sleep(delay)
                body = getattr(resp, "text", "") or ""
                last_error = requests.HTTPError(
                    f"HTTP {resp.status_code}: {body[:200]}"
                )
                continue

            raise requests.HTTPError(
                f"Non-retryable stream HTTP {resp.status_code}"
            )

        raise requests.RequestException(
            f"Exhausted {self.max_retries} retries. Last error: {last_error}"
        )

    def _backoff_delay(self, attempt: int, retry_after: str | None) -> float:
        """Compute the backoff delay for an attempt.

        Honors the Retry-After header when present, otherwise exponential
        backoff clamped to [base_delay, max_delay].

        Args:
            attempt: Zero-based attempt index.
            retry_after: Optional Retry-After header value (seconds).

        Returns:
            Delay in seconds.
        """
        if retry_after is not None:
            try:
                return min(float(retry_after), self.max_delay)
            except ValueError:
                pass
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)
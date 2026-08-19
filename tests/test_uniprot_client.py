"""Tests for UniProt client retry/backoff behavior."""

import pytest

from protein_database.uniprot.batching import MAX_OR_CONDITIONS
from protein_database.uniprot.client import (
    RETRYABLE_STATUSES,
    UniProtClient,
    HttpResponse,
)


class FakeSession:
    """A fake requests.Session returning scripted responses."""

    def __init__(self, responses: list[HttpResponse]):
        self._responses = list(responses)
        self.calls = 0
        self.last_params = None

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        self.last_params = params
        if not self._responses:
            raise AssertionError("No more scripted responses")
        return self._responses.pop(0)


def test_retryable_status_backoff(monkeypatch) -> None:
    """A 429 followed by 200 retries with backoff then succeeds."""
    session = FakeSession(
        [
            HttpResponse(429, "rate limited", {"Retry-After": "5"}),
            HttpResponse(200, "data"),
        ]
    )
    client = UniProtClient(session=session, max_retries=2, base_delay=1.0, max_delay=30.0)
    client._backoff_delay = lambda attempt, retry_after: 0  # no sleep in test

    resp = client.fetch_batch(["P69905"])
    assert resp.status_code == 200
    assert session.calls == 2


def test_exhausts_retries_then_raises(monkeypatch) -> None:
    """Transient failures exhaust retries and raise."""
    session = FakeSession(
        [
            HttpResponse(503, "unavailable"),
            HttpResponse(503, "unavailable"),
            HttpResponse(503, "unavailable"),
        ]
    )
    client = UniProtClient(session=session, max_retries=2, base_delay=1.0, max_delay=30.0)
    client._backoff_delay = lambda attempt, retry_after: 0

    with pytest.raises(Exception):
        client.fetch_batch(["P69905"])
    assert session.calls == 3  # max_retries + 1


def test_non_retryable_raises_immediately() -> None:
    """A 400 (non-retryable) raises without retrying."""
    session = FakeSession([HttpResponse(400, "bad fields")])
    client = UniProtClient(session=session, max_retries=5, base_delay=1.0, max_delay=30.0)
    client._backoff_delay = lambda attempt, retry_after: 0

    with pytest.raises(Exception):
        client.fetch_batch(["P69905"])
    assert session.calls == 1


def test_backoff_honors_retry_after() -> None:
    """Retry-After header caps the delay."""
    client = UniProtClient()
    assert client._backoff_delay(0, "7") == 7.0
    assert client._backoff_delay(0, "500") == 30.0  # capped at max_delay


def test_backoff_exponential() -> None:
    """Exponential backoff without Retry-After."""
    client = UniProtClient(base_delay=1.0, max_delay=100.0)
    assert client._backoff_delay(0, None) == 1.0
    assert client._backoff_delay(1, None) == 2.0
    assert client._backoff_delay(2, None) == 4.0


def test_retryable_statuses_set() -> None:
    assert 429 in RETRYABLE_STATUSES
    assert 500 in RETRYABLE_STATUSES
    assert 503 in RETRYABLE_STATUSES


def test_batch_size_capped_at_100() -> None:
    """batch_size is capped at the UniProt 100-OR limit."""
    client = UniProtClient(batch_size=500)
    assert client.batch_size == MAX_OR_CONDITIONS
    client2 = UniProtClient(batch_size=50)
    assert client2.batch_size == 50


def test_builds_query_params() -> None:
    """The query params use the verified fields list."""
    session = FakeSession([HttpResponse(200, "data")])
    client = UniProtClient(session=session, max_retries=0, base_delay=1.0, max_delay=30.0)
    client._backoff_delay = lambda attempt, retry_after: 0
    client.fetch_batch(["P69905", "P68871"])
    params = session.last_params
    assert params["format"] == "tsv"
    assert params["size"] == "2"
    assert "accession:P69905" in params["query"]
    assert "protein_name" in params["fields"]
    assert "cc_function" in params["fields"]
    assert "cc_subcellular_location" in params["fields"]
    assert "keyword" in params["fields"]

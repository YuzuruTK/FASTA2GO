"""Tests for UniProt batching helpers."""

import pytest

from protein_database.uniprot.batching import build_accession_query, chunk_list


def test_chunk_list_basic() -> None:
    assert list(chunk_list(["a", "b", "c", "d", "e"], 2)) == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]


def test_chunk_list_exact() -> None:
    assert list(chunk_list(["a", "b"], 2)) == [["a", "b"]]


def test_chunk_list_empty() -> None:
    assert list(chunk_list([], 2)) == []


def test_chunk_list_invalid_batch() -> None:
    with pytest.raises(ValueError):
        list(chunk_list(["a"], 0))


def test_build_accession_query() -> None:
    q = build_accession_query(["P69905", "P68871"])
    assert q == "accession:P69905 OR accession:P68871"


def test_build_accession_query_empty() -> None:
    with pytest.raises(ValueError):
        build_accession_query([])
"""Batching helpers for UniProt accession retrieval."""

from typing import Iterator

#: UniProtKB imposes a hard limit of 100 OR conditions per query.
MAX_OR_CONDITIONS = 100


def chunk_list(items: list[str], batch_size: int) -> Iterator[list[str]]:
    """Yield fixed-size chunks from a list.

    Args:
        items: List of items to chunk.
        batch_size: Maximum number of items per chunk. Must be > 0.

    Yields:
        Successive chunks of at most ``batch_size`` items.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def build_accession_query(accessions: list[str]) -> str:
    """Build a UniProtKB search query for multiple accessions.

    The UniProtKB API rejects queries with more than ``MAX_OR_CONDITIONS``
    (100) OR terms, so this raises if the input exceeds that limit.

    Args:
        accessions: List of UniProt accessions.

    Returns:
        A query string of the form ``accession:A OR accession:B OR ...``.

    Raises:
        ValueError: If accessions is empty or has more than
            ``MAX_OR_CONDITIONS`` entries.
    """
    if not accessions:
        raise ValueError("accessions must not be empty")
    if len(accessions) > MAX_OR_CONDITIONS:
        raise ValueError(
            f"Too many accessions for a single UniProt query: "
            f"{len(accessions)} > {MAX_OR_CONDITIONS} max"
        )
    return " OR ".join(f"accession:{acc}" for acc in accessions)

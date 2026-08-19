"""Raw UniProt batch response caching for reproducibility and resume."""

from pathlib import Path


class BatchCache:
    """Cache for raw UniProt TSV responses, keyed by batch id.

    Args:
        cache_dir: Directory to store cached responses.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def batch_id(self, accessions: list[str]) -> str:
        """Derive a deterministic batch id from a sorted accession list.

        Args:
            accessions: List of accessions in the batch.

        Returns:
            A hash-based batch id string.
        """
        import hashlib

        key = ",".join(sorted(accessions))
        return hashlib.sha1(key.encode()).hexdigest()[:16]

    def path_for(self, batch_id: str) -> Path:
        """Return the cache file path for a batch id."""
        return self.cache_dir / f"{batch_id}.tsv"

    def save(self, batch_id: str, text: str) -> Path:
        """Write raw response text to the cache.

        Args:
            batch_id: Batch identifier.
            text: Raw TSV response text.

        Returns:
            The path of the written cache file.
        """
        path = self.path_for(batch_id)
        path.write_text(text, encoding="utf-8")
        return path

    def load(self, batch_id: str) -> str | None:
        """Load a cached raw response if present.

        Args:
            batch_id: Batch identifier.

        Returns:
            The cached TSV text, or None if not cached.
        """
        path = self.path_for(batch_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def exists(self, batch_id: str) -> bool:
        """Check whether a batch response is cached."""
        return self.path_for(batch_id).exists()

    def list_batches(self) -> list[str]:
        """List all cached batch ids.

        Returns:
            Sorted list of cached batch ids (without file extension).
        """
        return sorted(p.stem for p in self.cache_dir.glob("*.tsv"))
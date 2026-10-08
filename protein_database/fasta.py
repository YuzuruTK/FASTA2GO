"""Streaming FASTA file parser (constant memory)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class FastaRecord:
    """A single protein record parsed from a FASTA file."""

    fasta_id: str
    description: str
    sequence: str

    @property
    def sequence_length(self) -> int:
        """Length of the amino acid sequence."""
        return len(self.sequence)


def parse_fasta(file_path: str | Path) -> Iterator[FastaRecord]:
    """Parse a FASTA file lazily, yielding one record at a time.

    Args:
        file_path: Path to the FASTA file.

    Yields:
        FastaRecord for each protein entry in the FASTA file.
    """
    fasta_id: str | None = None
    description = ""
    chunks: list[str] = []

    with open(file_path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if fasta_id is not None:
                    yield FastaRecord(fasta_id, description, "".join(chunks))
                parts = line[1:].strip().split(None, 1)
                fasta_id = parts[0] if parts else ""
                description = parts[1] if len(parts) > 1 else ""
                chunks = []
            else:
                chunks.append(line)

    if fasta_id is not None:
        yield FastaRecord(fasta_id, description, "".join(chunks))

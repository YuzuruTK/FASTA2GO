"""Streaming FASTA file parser for protein sequences."""

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


def count_fasta_entries(file_path: str | Path) -> int:
    """Count the number of FASTA entries (header lines starting with '>').

    This performs a fast pre-scan of the file so a progress bar can be
    created with an accurate total.

    Args:
        file_path: Path to the FASTA file.

    Returns:
        Number of FASTA entries in the file.
    """
    count = 0
    with open(file_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(">"):
                count += 1
    return count


def parse_fasta(file_path: str | Path) -> Iterator[FastaRecord]:
    """Parse a FASTA file lazily, yielding one FastaRecord at a time.

    This is a streaming parser that reads the file line by line, so it
    can handle very large FASTA files without loading them into memory.

    Args:
        file_path: Path to the FASTA file.

    Yields:
        FastaRecord for each protein entry in the FASTA file.
    """
    path = Path(file_path)

    current_id: str | None = None
    current_description: str | None = None
    sequence_lines: list[str] = []

    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                # Yield the previous record if one exists
                if current_id is not None:
                    yield FastaRecord(
                        fasta_id=current_id,
                        description=current_description or "",
                        sequence="".join(sequence_lines),
                    )

                # Parse the header
                header = line[1:].strip()
                parts = header.split(None, 1)
                current_id = parts[0] if parts else ""
                current_description = parts[1] if len(parts) > 1 else ""
                sequence_lines = []

            else:
                # Sequence line
                sequence_lines.append(line)

    # Yield the last record
    if current_id is not None:
        yield FastaRecord(
            fasta_id=current_id,
            description=current_description or "",
            sequence="".join(sequence_lines),
        )
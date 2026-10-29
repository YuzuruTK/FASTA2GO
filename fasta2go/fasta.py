"""Streaming FASTA reader that reports what is wrong with each record.

The reader never repairs data silently: every deviation is returned as an
issue code so the pipeline can store it in ``ingestion_issues``.
"""

from __future__ import annotations

import gzip
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

#: The 20 amino acids a denominator may contain. Anything else is ambiguous
#: (X, B, Z), a rare residue (U, O) or not a residue at all (*, -).
CANONICAL_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")

#: UniProt-style identifier: ``sp|P69905|HBA_HUMAN``, ``tr|A0A023|NAME``,
#: optionally an isoform suffix (``P69905-2``).
UNIPROT_ID = re.compile(r"^(sp|tr)\|([A-Z0-9]{6,10}(?:-\d+)?)\|(\S+)$")

Issue = tuple[str, str]


@dataclass(frozen=True)
class FastaRecord:
    """One FASTA entry, exactly as read."""

    line_number: int
    header: str
    fasta_id: str
    uniprot_accession: str | None
    sequence: str
    issues: list[Issue]

    @property
    def is_valid(self) -> bool:
        """Whether the record can be stored (a header id and a non-empty sequence)."""
        return bool(self.fasta_id) and bool(self.sequence)

    @property
    def sequence_length(self) -> int:
        return len(self.sequence)


def open_fasta(path: str | Path) -> TextIO:
    """Open a plain or gzipped FASTA file as text."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def read_fasta(path: str | Path) -> Iterator[FastaRecord]:
    """Yield one :class:`FastaRecord` per entry, in file order, in constant memory.

    Raises:
        ValueError: If sequence lines appear before the first header, i.e. the
            file is not FASTA at all.
    """
    header: str | None = None
    header_line = 0
    chunks: list[str] = []

    with open_fasta(path) as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield _build_record(header_line, header, chunks)
                header, header_line, chunks = line[1:].strip(), line_number, []
                continue
            if header is None:
                raise ValueError(
                    f"{path}: sequence data before the first header "
                    f"(line {line_number})"
                )
            chunks.append(line.upper())

    if header is not None:
        yield _build_record(header_line, header, chunks)


def _build_record(line_number: int, header: str, chunks: list[str]) -> FastaRecord:
    sequence = "".join(chunks)
    tokens = header.split()
    fasta_id = tokens[0] if tokens else ""
    issues: list[Issue] = []
    accession = None

    if not fasta_id:
        issues.append(("malformed_header", header[:200]))
    else:
        match = UNIPROT_ID.match(fasta_id)
        if match:
            accession = match.group(2)
        else:
            issues.append(("header_without_accession", fasta_id))

    if not sequence:
        issues.append(("empty_sequence", ""))
    else:
        issues.extend(_residue_issues(sequence))

    return FastaRecord(line_number, header, fasta_id, accession, sequence, issues)


def _residue_issues(sequence: str) -> list[Issue]:
    """Report residues outside the canonical set, with their counts."""
    counts = Counter(char for char in sequence if char not in CANONICAL_RESIDUES)
    if not counts:
        return []
    detail = ", ".join(f"{char}={counts[char]}" for char in sorted(counts))
    return [("noncanonical_residues", detail)]

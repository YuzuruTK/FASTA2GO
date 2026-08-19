"""UniProt accession extraction and deduplication."""

from dataclasses import dataclass


@dataclass
class AccessionMap:
    """Maps UniProt accessions to the local protein rows that contain them."""

    accession: str
    fasta_ids: list[tuple[str, str]]  # list of (fasta_id, source_file)


def extract_accession(fasta_id: str) -> str | None:
    """Extract the UniProt accession from a FASTA identifier.

    Supports the UniProt precedence format: ``sp|ACCESSION|ENTRY_NAME`` or
    ``tr|ACCESSION|ENTRY_NAME``. The accession is the middle field.

    Args:
        fasta_id: The FASTA identifier (e.g. ``sp|P69905|HBA_HUMAN``).

    Returns:
        The UniProt accession if the identifier matches the UniProt format,
        otherwise None.
    """
    parts = fasta_id.split("|")
    if len(parts) == 3 and (parts[0] in ("sp", "tr")):
        accession = parts[1].strip()
        if accession:
            return accession.upper()
    return None


def extract_accessions(fasta_ids: list[str]) -> list[str]:
    """Extract and deduplicate UniProt accessions from FASTA identifiers.

    Args:
        fasta_ids: List of FASTA identifiers.

    Returns:
        Sorted list of unique UniProt accessions.
    """
    seen: set[str] = set()
    for fid in fasta_ids:
        acc = extract_accession(fid)
        if acc:
            seen.add(acc)
    return sorted(seen)


def group_accessions_by_local_rows(
    protein_rows: list[tuple[str, str, str]],
) -> dict[str, list[tuple[str, str]]]:
    """Group local protein rows by their UniProt accession.

    One accession may appear in multiple local ``(fasta_id, source_file)``
    rows; this builds the mapping so a single UniProt retrieval can be
    fanned out to all matching local rows.

    Args:
        protein_rows: List of ``(fasta_id, source_file, ?)`` tuples.

    Returns:
        Dict mapping accession -> list of (fasta_id, source_file).
    """
    mapping: dict[str, list[tuple[str, str]]] = {}
    for row in protein_rows:
        fasta_id, source_file, _ = row
        acc = extract_accession(fasta_id)
        if acc:
            mapping.setdefault(acc, []).append((fasta_id, source_file))
    return mapping
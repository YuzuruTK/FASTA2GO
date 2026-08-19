"""Parsing of UniProtKB TSV responses into annotation rows."""

from __future__ import annotations

import csv
import io
from typing import Any

#: Expected TSV column order (matches UNIPROT_FIELDS order).
EXPECTED_HEADERS = ["Entry", "Entry Name", "Protein names", "Gene Names",
                    "Function [CC]", "Subcellular location [CC]",
                    "Protein existence", "Organism", "Keywords"]

STATUS_FETCHED = "fetched"
STATUS_MISSING_FUNCTION = "missing_function"


def _x(value: Any) -> str | None:
    """Return the value if non-empty, else None."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


class TsvParseError(Exception):
    """Raised when a TSV response is malformed or incomplete."""


def parse_tsv(text: str, requested_accessions: list[str]) -> list[dict[str, str | None]]:
    """Parse a UniProtKB TSV response into annotation row dicts.

    Detects not-found accessions: any accession requested but not present in
    the response is flagged. Empty Function values are kept (they map to
    ``missing_function`` status downstream).

    Args:
        text: Raw TSV response text.
        requested_accessions: Accessions that were requested in this batch.

    Returns:
        List of dicts, one per returned entry, with keys:
        uniprot_accession, uniprot_id, protein_name, gene_name, function,
        subcellular_location, protein_existence, organism, keywords, status.

    Raises:
        TsvParseError: If the response is empty, missing the header, or has
            the wrong number of columns.
    """
    if not text or not text.strip():
        raise TsvParseError("Empty TSV response.")

    reader = csv.reader(io.StringIO(text), delimiter="\t")
    rows = list(reader)
    if not rows:
        raise TsvParseError("Empty TSV response.")

    header = [h.strip() for h in rows[0]]
    if header != EXPECTED_HEADERS:
        raise TsvParseError(f"Unexpected TSV header: {header}")

    results: list[dict[str, str | None]] = []
    returned_accessions: set[str] = set()
    for row in rows[1:]:
        if len(row) != len(EXPECTED_HEADERS):
            raise TsvParseError(
                f"Row has {len(row)} columns, expected {len(EXPECTED_HEADERS)}: {row[:2]}"
            )
        accession = (row[0].strip() if row[0] else "").upper()
        if not accession:
            continue
        returned_accessions.add(accession)

        function = _x(row[4])
        status = STATUS_FETCHED if function else STATUS_MISSING_FUNCTION
        results.append({
            "uniprot_accession": accession,
            "uniprot_id": _x(row[1]),
            "protein_name": _x(row[2]),
            "gene_name": _x(row[3]),
            "function": function,
            "subcellular_location": _x(row[5]),
            "protein_existence": _x(row[6]),
            "organism": _x(row[7]),
            "keywords": _x(row[8]),
            "status": status,
        })

    return results


def missing_accessions(requested: list[str], returned: list[dict[str, str | None]]) -> list[str]:
    """Return accessions that were requested but not present in the response.

    Args:
        requested: Accessions submitted for the batch.
        returned: Parsed annotation rows.

    Returns:
        Sorted list of accessions not returned by the API.
    """
    returned_set = {r["uniprot_accession"] for r in returned}
    return sorted(set(requested) - returned_set)
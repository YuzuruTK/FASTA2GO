"""UniProtKB REST enrichment: batched accession lookup and JSON parsing.

The ``/uniprotkb/accessions`` endpoint accepts a list of accessions and returns
only those UniProt knows. A single malformed accession makes the whole batch
fail with HTTP 400, so accessions are validated before the request.

Only source annotations are parsed. Ancestor propagation belongs to a later
stage and is deliberately absent here.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import requests

API_URL = "https://rest.uniprot.org/uniprotkb/accessions"

#: Only the fields this stage stores; the complete entry is several times larger.
RETURN_FIELDS = ",".join(
    [
        "accession",
        "id",
        "protein_name",
        "gene_names",
        "organism_name",
        "organism_id",
        "length",
        "go",
        "keyword",
        "cc_function",
        "cc_subcellular_location",
        "protein_existence",
    ]
)

#: Shape UniProt accepts for an accession (optional isoform suffix).
ACCESSION_PATTERN = re.compile(r"^[A-Z0-9]{6,10}(?:-\d+)?$")

#: GO aspect letter -> namespace used in the database.
ASPECTS = {
    "P": "biological_process",
    "C": "cellular_component",
    "F": "molecular_function",
}

#: Statuses worth retrying: rate limiting and transient server errors.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

#: Stored when the source reports a GO term without an evidence code.
UNSPECIFIED = "unspecified"


class UniProtError(RuntimeError):
    """A request failed.

    Deliberately different from a successful response without the accession:
    a failure must never be recorded as "UniProt does not know this protein".
    """


@dataclass(frozen=True)
class GoAnnotation:
    """One GO annotation as reported by UniProt."""

    go_id: str
    aspect: str
    go_name: str | None
    evidence_code: str
    evidence_source: str
    provenance: str | None


@dataclass(frozen=True)
class Annotation:
    """UniProtKB metadata of one entry."""

    accession: str
    uniprot_id: str | None = None
    protein_name: str | None = None
    gene_names: str | None = None
    organism: str | None = None
    taxon_id: int | None = None
    protein_existence: str | None = None
    secondary_accessions: str | None = None
    keywords: str | None = None
    function_description: str | None = None
    subcellular_location: str | None = None
    source_length: int | None = None
    go: tuple[GoAnnotation, ...] = ()


def fetch(
    accessions: Sequence[str],
    *,
    session: requests.Session | None = None,
    attempts: int = 3,
    timeout: float = 120.0,
    pause: float = 2.0,
) -> dict[str, Annotation]:
    """Return ``{accession: Annotation}`` for the accessions UniProt returned.

    Retries rate limiting and transient server errors with exponential backoff;
    any other failure raises :class:`UniProtError` so the caller records nothing.

    Raises:
        UniProtError: If the request fails.
    """
    valid = [value for value in accessions if ACCESSION_PATTERN.match(value)]
    if not valid:
        return {}

    client = session or requests.Session()
    params = {
        "accessions": ",".join(valid),
        "format": "json",
        "fields": RETURN_FIELDS,
    }
    last_error = "unknown error"
    delay = pause

    for attempt in range(1, attempts + 1):
        try:
            response = client.get(API_URL, params=params, timeout=timeout)
        except requests.RequestException as error:
            last_error = f"network error: {error}"
        else:
            if response.status_code == 200:
                payload = response.json()
                return {
                    annotation.accession: annotation
                    for annotation in map(parse_entry, payload.get("results", []))
                }
            if response.status_code not in RETRYABLE_STATUSES:
                raise UniProtError(
                    f"HTTP {response.status_code} from UniProt: "
                    f"{response.text[:300]}"
                )
            last_error = f"HTTP {response.status_code}"
            delay = _retry_after(response) or delay
        if attempt < attempts:
            time.sleep(delay)
            delay *= 2

    raise UniProtError(
        f"UniProt request for {len(valid)} accessions failed after "
        f"{attempts} attempts ({last_error})"
    )


def parse_entry(entry: dict[str, Any]) -> Annotation:
    """Convert one UniProtKB JSON entry into an :class:`Annotation`."""
    organism = entry.get("organism") or {}
    return Annotation(
        accession=entry["primaryAccession"].upper(),
        uniprot_id=entry.get("uniProtkbId"),
        protein_name=_protein_name(entry.get("proteinDescription") or {}),
        gene_names=_join(
            gene.get("geneName", {}).get("value") for gene in entry.get("genes", [])
        ),
        organism=organism.get("scientificName"),
        taxon_id=organism.get("taxonId"),
        protein_existence=entry.get("proteinExistence"),
        secondary_accessions=_join(entry.get("secondaryAccessions")),
        keywords=_join(keyword.get("name") for keyword in entry.get("keywords", [])),
        function_description=_comment_texts(entry, "FUNCTION"),
        subcellular_location=_subcellular_locations(entry),
        source_length=(entry.get("sequence") or {}).get("length"),
        go=parse_go_annotations(entry.get("uniProtKBCrossReferences", [])),
    )


def parse_go_annotations(xrefs: Sequence[dict[str, Any]]) -> tuple[GoAnnotation, ...]:
    """Extract GO cross references, one entry per ``(term, evidence)`` pair.

    UniProt reports the aspect and term name in ``GoTerm`` (``"C:cytosol"``),
    the evidence code and who assigned it in ``GoEvidenceType``
    (``"TAS:Reactome"``) and, when available, the literature reference in
    ``evidences``. The same term may be annotated with several evidence codes,
    so pairs are kept apart and their provenance merged.
    """
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}

    for xref in xrefs:
        if xref.get("database") != "GO":
            continue
        go_id = xref.get("id")
        properties = {
            prop.get("key"): prop.get("value") for prop in xref.get("properties", [])
        }
        term = properties.get("GoTerm") or ""
        aspect = ASPECTS.get(term[:1])
        if not go_id or aspect is None:
            continue

        code, _, source = (properties.get("GoEvidenceType") or "").partition(":")
        evidence = (code or UNSPECIFIED, source or UNSPECIFIED)
        entry = merged.setdefault(
            (go_id, *evidence),
            {"aspect": aspect, "go_name": term[2:].strip() or None, "provenance": []},
        )
        for reference in xref.get("evidences", []):
            entry["provenance"].append(
                f"{reference.get('source')}:{reference.get('id')}"
            )

    return tuple(
        GoAnnotation(
            go_id=go_id,
            aspect=value["aspect"],
            go_name=value["go_name"],
            evidence_code=code,
            evidence_source=source,
            provenance="; ".join(dict.fromkeys(value["provenance"])) or None,
        )
        for (go_id, code, source), value in merged.items()
    )


def _protein_name(description: dict[str, Any]) -> str | None:
    recommended = (description.get("recommendedName") or {}).get("fullName") or {}
    if recommended.get("value"):
        return recommended["value"]
    for submission in description.get("submissionNames", []):
        name = (submission.get("fullName") or {}).get("value")
        if name:
            return name
    return None


def _comment_texts(entry: dict[str, Any], comment_type: str) -> str | None:
    return _join(
        text.get("value")
        for comment in entry.get("comments", [])
        if comment.get("commentType") == comment_type
        for text in comment.get("texts", [])
    )


def _subcellular_locations(entry: dict[str, Any]) -> str | None:
    return _join(
        location.get("location", {}).get("value")
        for comment in entry.get("comments", [])
        if comment.get("commentType") == "SUBCELLULAR LOCATION"
        for location in comment.get("subcellularLocations", [])
    )


def _join(values: Iterable[Any] | None) -> str | None:
    """Join non-empty values with ``; ``, preserving order and dropping repeats."""
    cleaned = [
        str(value).strip()
        for value in (values or [])
        if value is not None and str(value).strip()
    ]
    unique = dict.fromkeys(cleaned)
    return "; ".join(unique) if unique else None


def _retry_after(response: requests.Response) -> float | None:
    try:
        return float(response.headers.get("Retry-After", ""))
    except ValueError:
        return None

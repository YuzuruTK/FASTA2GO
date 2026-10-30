"""GO Slim Molecular Function vocabulary, propagation and annotation mapping.

The project vocabulary is the official **generic GO slim** (``goslim_generic``)
plus one documented extension, ``GO:0005488 binding`` - it is *not* the
unmodified slim. Rules, all enforced here and checked again by
``etapa1/build_go_slim_labels.py``:

* aspect: ``molecular_function`` only;
* relations traversed: ``is_a`` and ``part_of``;
* propagation: **all** applicable ancestors, not just the nearest one;
* the aspect root ``GO:0003674`` is excluded by rule (never by a label mask);
* ``alt_id`` is resolved to its primary id, obsolete terms are rejected;
* ``(protein, label)`` pairs are deduplicated (union of sets).

Files are read only; nothing is downloaded here.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

MF_NAMESPACE = "molecular_function"
ASPECT_ROOT = "GO:0003674"
RELATIONS = ("is_a", "part_of")
GENERIC_SUBSET = "goslim_generic"
PROJECT_EXTENSION = ("GO:0005488",)  # binding

STATUS_MAPPED = "mapped"
STATUS_NO_ANCESTOR = "no_vocabulary_ancestor"
STATUS_OBSOLETE = "obsolete"
STATUS_UNKNOWN = "unknown"
STATUSES = (STATUS_MAPPED, STATUS_NO_ANCESTOR, STATUS_OBSOLETE, STATUS_UNKNOWN)

LABEL_COLUMNS = (
    "protein_id",
    "n_labels",
    "labels",
    "n_annotations",
    "n_mapped",
    "n_no_vocabulary_ancestor",
    "n_obsolete",
    "n_unknown",
)


class GoSlimError(ValueError):
    """Invalid ontology, vocabulary or integrity failure."""


@dataclass(frozen=True)
class GoTerm:
    """One non-obsolete OBO term, limited to what propagation needs."""

    id: str
    name: str
    namespace: str
    parents: tuple[str, ...] = ()
    subsets: tuple[str, ...] = ()
    alt_ids: tuple[str, ...] = ()
    obsolete: bool = False


@dataclass(frozen=True)
class GoOntology:
    """Parsed ontology plus the alias index used to resolve ``alt_id``."""

    release: str
    terms: Mapping[str, GoTerm]
    alt_to_primary: Mapping[str, str] = field(default_factory=dict)

    def resolve(self, raw_id: str) -> str:
        """Return the primary id for a raw id, resolving a secondary ``alt_id``."""
        return self.alt_to_primary.get(raw_id, raw_id)


@dataclass(frozen=True)
class Vocabulary:
    """The frozen label space: ordered ids plus how each one was admitted."""

    name: str
    terms: tuple[str, ...]
    origin: Mapping[str, str]
    release: str
    subset: str = GENERIC_SUBSET
    excluded: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    skipped_non_mf: tuple[str, ...] = ()

    @property
    def term_set(self) -> frozenset[str]:
        return frozenset(self.terms)

    def __len__(self) -> int:
        return len(self.terms)


def sha256_file(path: str | Path) -> str:
    """Return the full SHA-256 of a file, reading it in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_integrity(path: str | Path, expected_sha256: str) -> str:
    """Fail closed when a source file does not match its pinned hash.

    Args:
        path: File to verify.
        expected_sha256: Lowercase hex digest pinned in ``sources.json``.

    Returns:
        The verified digest.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        GoSlimError: If the digest differs (message names the expected release).
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    actual = sha256_file(source)
    if actual != expected_sha256.lower():
        raise GoSlimError(
            f"SHA-256 mismatch for {source}: expected {expected_sha256}, got {actual}. "
            "Re-download the pinned release; do not process an unverified ontology."
        )
    return actual


def _flush_term(current: dict | None, terms: dict[str, GoTerm]) -> None:
    """Freeze the stanza being parsed (only once all its lines were read)."""
    if not current or not current["id"] or current["id"] in terms:
        return
    terms[current["id"]] = GoTerm(
        id=current["id"],
        name=current["name"],
        namespace=current["namespace"],
        parents=tuple(current["parents"]),
        subsets=tuple(current["subsets"]),
        alt_ids=tuple(current["alt_ids"]),
        obsolete=current["obsolete"],
    )


def parse_obo(path: str | Path) -> GoOntology:
    """Parse an OBO file into terms, parents and the ``alt_id`` index.

    Only ``[Term]`` stanzas are kept. ``is_a`` and ``part_of`` (relationship
    form) are the only relations recorded, matching the project rule. A stanza
    is frozen only when it ends, because ``parents``, ``subset``, ``alt_id`` and
    ``is_obsolete`` all appear *after* ``id`` inside the stanza.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        GoSlimError: If the file declares no ``data-version`` or no terms.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"OBO file not found: {source}")

    release = ""
    terms: dict[str, GoTerm] = {}
    current: dict | None = None
    in_term = False

    with source.open("rt", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.strip():
                if in_term:
                    _flush_term(current, terms)
                current = None
                in_term = False
                continue
            if line.startswith("["):
                if in_term:
                    _flush_term(current, terms)
                in_term = line.strip() == "[Term]"
                current = (
                    {
                        "id": "",
                        "name": "",
                        "namespace": "",
                        "parents": [],
                        "subsets": [],
                        "alt_ids": [],
                        "obsolete": False,
                    }
                    if in_term
                    else None
                )
                continue
            if not in_term or current is None:
                if not release and line.startswith("data-version:"):
                    release = line.split(":", 1)[1].strip()
                continue

            key, _, value = line.partition(": ")
            key, value = key.strip(), value.strip()
            if key == "id":
                current["id"] = value
            elif key == "name":
                current["name"] = value
            elif key == "namespace":
                current["namespace"] = value
            elif key == "is_a":
                current["parents"].append(value.split("!", 1)[0].strip())
            elif key == "relationship":
                parts = value.split()
                if len(parts) >= 2 and parts[0] in RELATIONS:
                    current["parents"].append(parts[1])
            elif key == "subset":
                current["subsets"].append(value)
            elif key == "alt_id":
                current["alt_ids"].append(value)
            elif key == "is_obsolete":
                current["obsolete"] = value.lower() == "true"

    if in_term:
        _flush_term(current, terms)

    if not release:
        raise GoSlimError(f"{source} declares no 'data-version'; refusing to guess it.")
    if not terms:
        raise GoSlimError(f"{source} contains no [Term] stanzas.")

    alt_to_primary: dict[str, str] = {}
    for term in terms.values():
        for alt in term.alt_ids:
            alt_to_primary.setdefault(alt, term.id)
    return GoOntology(release=release, terms=terms, alt_to_primary=alt_to_primary)


def read_term_ids(path: str | Path) -> frozenset[str]:
    """Return the ids declared in a slim/OBO file (obsolete terms excluded)."""
    return frozenset(
        term_id for term_id, term in parse_obo(path).terms.items() if not term.obsolete
    )


def build_vocabulary(
    ontology: GoOntology,
    slim_ids: Iterable[str],
    *,
    extension: Sequence[str] = PROJECT_EXTENSION,
    excluded: Sequence[str] = (ASPECT_ROOT,),
    subset: str = GENERIC_SUBSET,
) -> Vocabulary:
    """Build the project vocabulary: slim (MF only) + documented extension.

    Terms that are declared in the slim but are not Molecular Function are
    dropped and recorded. ``excluded`` ids are removed unconditionally, even
    when the slim declares them.

    Raises:
        GoSlimError: If a declared or extended id is absent from the ontology,
            if an extension term is not Molecular Function, or if the resulting
            vocabulary is empty.
    """
    declared = sorted(set(slim_ids))
    missing = [term_id for term_id in declared if term_id not in ontology.terms]
    if missing:
        raise GoSlimError(
            f"{len(missing)} slim term(s) absent from the ontology: {missing[:5]}"
        )

    non_mf = [term_id for term_id in declared if not _is_mf(ontology, term_id)]
    candidates = [term_id for term_id in declared if _is_mf(ontology, term_id)]

    for extra in extension:
        term = ontology.terms.get(extra)
        if term is None:
            raise GoSlimError(f"Project extension {extra} is absent from the ontology.")
        if term.namespace != MF_NAMESPACE:
            raise GoSlimError(
                f"Project extension {extra} has namespace {term.namespace!r}, "
                f"not {MF_NAMESPACE!r}."
            )
        candidates.append(extra)

    origin = {term_id: subset for term_id in candidates}
    for extra in extension:
        origin[extra] = "project_extension"

    excluded_ids = tuple(sorted(set(excluded)))
    kept = sorted(term_id for term_id in set(candidates) if term_id not in excluded_ids)
    removed = tuple(sorted(set(candidates) & set(excluded_ids)))
    if not kept:
        raise GoSlimError("The resulting vocabulary is empty.")

    return Vocabulary(
        name=(
            f"{subset}+{'+'.join(sorted(extension))}"
            if extension
            else subset
        ),
        terms=tuple(kept),
        origin=origin,
        release=ontology.release,
        subset=subset,
        excluded=excluded_ids,
        removed=removed,
        skipped_non_mf=tuple(sorted(non_mf)),
    )


def _is_mf(ontology: GoOntology, term_id: str) -> bool:
    term = ontology.terms.get(term_id)
    return term is not None and term.namespace == MF_NAMESPACE and not term.obsolete


def build_ancestor_map(
    ontology: GoOntology,
    vocabulary: Vocabulary,
    *,
    restrict_to_aspect: bool = True,
) -> dict[str, frozenset[str]]:
    """Map every non-obsolete term to its vocabulary ancestors (itself included).

    Walks all paths through ``is_a`` and ``part_of``. Memoised over the DAG:
    ``memo[term] = frozenset()`` before recursing doubles as a cycle guard.

    ``restrict_to_aspect`` (default) expands a term only when it belongs to the
    vocabulary aspect (Molecular Function). A node of another aspect is never
    traversed, so a cross-aspect path can neither *become* a label nor *lead*
    to a Molecular Function label - the guarantee is structural, not a
    property of the current ``go-basic.obo`` (which today happens to contain no
    cross-aspect edge at all). Pass ``False`` only to reproduce the pre-audit
    behaviour when comparing both rules.
    """
    targets = vocabulary.term_set
    memo: dict[str, frozenset[str]] = {}

    def resolve(term_id: str) -> frozenset[str]:
        if term_id in memo:
            return memo[term_id]
        term = ontology.terms.get(term_id)
        if term is None or term.obsolete:
            memo[term_id] = frozenset()
            return memo[term_id]
        if restrict_to_aspect and term.namespace != MF_NAMESPACE:
            memo[term_id] = frozenset()
            return memo[term_id]
        memo[term_id] = frozenset()
        found = {term_id} if term_id in targets else set()
        for parent in term.parents:
            found |= resolve(parent)
        memo[term_id] = frozenset(found)
        return memo[term_id]

    for term_id, term in ontology.terms.items():
        if not term.obsolete:
            resolve(term_id)
    return memo


@dataclass(frozen=True)
class AnnotationOutcome:
    """Result of mapping one raw GO id onto the vocabulary."""

    raw_id: str
    primary_id: str | None
    status: str
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProteinMapping:
    """Deduplicated labels of one protein plus its annotation accounting."""

    protein_id: str
    labels: tuple[str, ...]
    statuses: tuple[str, ...]
    raw_terms: tuple[str, ...]

    def counts(self) -> dict[str, int]:
        counter = Counter(self.statuses)
        return {status: counter.get(status, 0) for status in STATUSES}


def classify_annotation(
    raw_id: str,
    ontology: GoOntology,
    ancestor_map: Mapping[str, frozenset[str]],
    vocabulary: Vocabulary,
) -> AnnotationOutcome:
    """Classify one raw GO id into exactly one of the four disjoint statuses.

    Order matters and makes the partition exclusive: unknown (not in the
    ontology, not an ``alt_id``) -> obsolete -> no vocabulary ancestor -> mapped.
    """
    primary = ontology.resolve(raw_id)
    term = ontology.terms.get(primary)
    if term is None:
        return AnnotationOutcome(raw_id=raw_id, primary_id=None, status=STATUS_UNKNOWN)
    if term.obsolete:
        return AnnotationOutcome(
            raw_id=raw_id, primary_id=primary, status=STATUS_OBSOLETE
        )
    labels = tuple(sorted(ancestor_map.get(primary, frozenset()) & vocabulary.term_set))
    if not labels:
        return AnnotationOutcome(
            raw_id=raw_id, primary_id=primary, status=STATUS_NO_ANCESTOR
        )
    return AnnotationOutcome(
        raw_id=raw_id, primary_id=primary, status=STATUS_MAPPED, labels=labels
    )


@dataclass(frozen=True)
class MappingSummary:
    """Auditable counters for a mapping run (the four statuses partition it)."""

    proteins: int
    annotations: int
    mapped: int
    no_vocabulary_ancestor: int
    obsolete: int
    unknown: int
    proteins_without_labels: int
    proteins_with_labels: int
    max_labels: int
    vocabulary_terms: int
    labels_used: int
    label_counts: Mapping[str, int]

    def as_dict(self) -> dict:
        return {
            "proteins": self.proteins,
            "annotations": self.annotations,
            "statuses": {
                STATUS_MAPPED: self.mapped,
                STATUS_NO_ANCESTOR: self.no_vocabulary_ancestor,
                STATUS_OBSOLETE: self.obsolete,
                STATUS_UNKNOWN: self.unknown,
            },
            "proteins_without_labels": self.proteins_without_labels,
            "proteins_with_labels": self.proteins_with_labels,
            "max_labels_per_protein": self.max_labels,
            "vocabulary_terms": self.vocabulary_terms,
            "labels_used": self.labels_used,
            "label_counts": dict(self.label_counts),
        }


def summarize(
    mappings: Sequence[ProteinMapping], vocabulary: Vocabulary
) -> MappingSummary:
    """Aggregate mappings into the audit counters, checking the partition.

    Raises:
        GoSlimError: If the status counts do not add up to the annotation total
            (which would mean the categories overlap).
    """
    counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    for mapping in mappings:
        counter.update(mapping.statuses)
        label_counter.update(mapping.labels)

    annotations = sum(counter.values())
    status_total = sum(counter.get(status, 0) for status in STATUSES)
    if status_total != annotations:
        raise GoSlimError(
            "Annotation statuses are not a partition: "
            f"{status_total} classified vs {annotations} annotations."
        )

    counts = {term_id: label_counter.get(term_id, 0) for term_id in vocabulary.terms}
    return MappingSummary(
        proteins=len(mappings),
        annotations=annotations,
        mapped=counter.get(STATUS_MAPPED, 0),
        no_vocabulary_ancestor=counter.get(STATUS_NO_ANCESTOR, 0),
        obsolete=counter.get(STATUS_OBSOLETE, 0),
        unknown=counter.get(STATUS_UNKNOWN, 0),
        proteins_without_labels=sum(1 for mapping in mappings if not mapping.labels),
        proteins_with_labels=sum(1 for mapping in mappings if mapping.labels),
        max_labels=max((len(mapping.labels) for mapping in mappings), default=0),
        vocabulary_terms=len(vocabulary),
        labels_used=sum(1 for count in counts.values() if count),
        label_counts=counts,
    )


def map_proteins(
    rows: Iterable[tuple[str, Sequence[str]]],
    ontology: GoOntology,
    vocabulary: Vocabulary,
    ancestor_map: Mapping[str, frozenset[str]],
) -> tuple[list[ProteinMapping], MappingSummary]:
    """Map ``(protein_id, raw_go_ids)`` rows; duplicates are collapsed per protein."""
    mappings: list[ProteinMapping] = []
    for protein_id, raw_ids in rows:
        unique = tuple(sorted(set(raw_ids)))
        outcomes = [
            classify_annotation(raw_id, ontology, ancestor_map, vocabulary)
            for raw_id in unique
        ]
        labels = tuple(sorted({label for outcome in outcomes for label in outcome.labels}))
        mappings.append(
            ProteinMapping(
                protein_id=protein_id,
                labels=labels,
                statuses=tuple(outcome.status for outcome in outcomes),
                raw_terms=unique,
            )
        )
    mappings.sort(key=lambda mapping: mapping.protein_id)
    return mappings, summarize(mappings, vocabulary)


def write_labels_csv(path: str | Path, mappings: Sequence[ProteinMapping]) -> Path:
    """Write one row per protein with its deduplicated labels and status counts."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for mapping in mappings:
            counts = mapping.counts()
            writer.writerow(
                {
                    "protein_id": mapping.protein_id,
                    "n_labels": len(mapping.labels),
                    "labels": ";".join(mapping.labels),
                    "n_annotations": len(mapping.raw_terms),
                    "n_mapped": counts[STATUS_MAPPED],
                    "n_no_vocabulary_ancestor": counts[STATUS_NO_ANCESTOR],
                    "n_obsolete": counts[STATUS_OBSOLETE],
                    "n_unknown": counts[STATUS_UNKNOWN],
                }
            )
    return destination


def read_vocabulary_ids(path: str | Path) -> frozenset[str]:
    """Return the ``go_id`` column of the pinned ``vocabulary_mf.csv``."""
    with Path(path).open("rt", encoding="utf-8", newline="") as handle:
        return frozenset(row["go_id"] for row in csv.DictReader(handle))

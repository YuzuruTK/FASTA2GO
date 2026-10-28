"""Align sequence features and GO Slim labels into a validated dataset.

The two sides come from different places and use different identifiers:

* features live in DuckDB, keyed by ``(fasta_id, source_file)`` with
  ``fasta_id`` shaped ``sp|P69905|HBA_HUMAN``;
* labels come from ``exports/go_slim/.../protein_labels.csv``, keyed by the
  UniProt **accession** (``A0A086F3E3``).

Alignment therefore happens on the accession extracted with the project's
existing convention (``split_part(fasta_id, '|', 2)``), never on row position.
The composite key is kept in :class:`ProteinKey` so every row stays traceable.

Nothing here reads the full base: ``load_features`` requires an explicit list of
accessions, and the label loader reads a CSV.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np

from protein_database.schema import COMPOSITION_COLUMNS, FEATURE_COLUMNS

#: Accession shape accepted by UniProtKB (optional isoform suffix).
ACCESSION_PATTERN = re.compile(r"^[A-Z0-9]{6,10}(-\d+)?$")

LABEL_REQUIRED_COLUMNS = ("protein_id", "labels")


class DatasetError(ValueError):
    """Invalid, ambiguous or incomplete dataset input."""


def feature_columns() -> tuple[str, ...]:
    """The 35 canonical numeric features, in schema order."""
    return tuple(FEATURE_COLUMNS) + tuple(COMPOSITION_COLUMNS)


def accession_of(fasta_id: str) -> str:
    """Extract the UniProt accession from a FASTA identifier.

    Uses the same convention as ``uniprot.py`` and ``build_keyword_terms.py``:
    only ``sp|``/``tr|`` entries qualify, and the accession is upper-cased.
    Returns ``""`` when the identifier does not follow that shape.
    """
    parts = fasta_id.split("|")
    if len(parts) < 3 or parts[0].lower() not in {"sp", "tr"}:
        return ""
    return parts[1].upper()


@dataclass(frozen=True, order=True)
class ProteinKey:
    """Stable identity of one protein row."""

    accession: str
    fasta_id: str
    source_file: str


@dataclass(frozen=True)
class FeatureTable:
    """Numeric block loaded from DuckDB, with its keys and column names."""

    keys: tuple[ProteinKey, ...]
    X: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class LabelTable:
    """Label block read from the GO Slim artifact, preserving empty proteins."""

    labels: Mapping[str, tuple[str, ...]]
    label_names: tuple[str, ...]
    proteins_without_labels: int
    unknown_labels_dropped: tuple[str, ...] = ()


def _clean_accessions(accessions: Sequence[str]) -> tuple[str, ...]:
    """Normalise, validate and sort accessions; refuse an empty request."""
    cleaned: list[str] = []
    bad: list[str] = []
    for raw in accessions:
        value = str(raw).strip().upper()
        if not ACCESSION_PATTERN.match(value):
            bad.append(raw)
        else:
            cleaned.append(value)
    if bad:
        raise DatasetError(
            f"{len(bad)} accession(s) do not look like UniProt accessions: {bad[:5]}"
        )
    if not cleaned:
        raise DatasetError("No accessions given; refusing to scan the whole table.")
    return tuple(sorted(set(cleaned)))


def load_features(
    conn: duckdb.DuckDBPyConnection,
    accessions: Sequence[str],
    *,
    columns: Sequence[str] | None = None,
) -> FeatureTable:
    """Load the numeric block for explicitly listed accessions.

    Requires both feature tables (``sequence_features`` and
    ``amino_acid_composition``), which are 1:1 with ``proteins`` in this schema.

    Raises:
        DatasetError: On invalid/empty accessions, no matching rows, an
            accession resolving to more than one protein, or non-finite values.
    """
    names = tuple(columns) if columns else feature_columns()
    allowed = set(feature_columns())
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise DatasetError(
            f"{len(unknown)} unknown feature column(s): {unknown}; allowed are "
            f"{len(allowed)} schema features."
        )

    wanted = _clean_accessions(accessions)
    selection = ", ".join(
        f"f.{name}" if name in FEATURE_COLUMNS else f"a.{name}" for name in names
    )
    placeholders = ", ".join("?" for _ in wanted)
    sql = (
        f"SELECT p.fasta_id, p.source_file, {selection} "
        "FROM proteins p "
        "JOIN sequence_features f USING (fasta_id, source_file) "
        "JOIN amino_acid_composition a USING (fasta_id, source_file) "
        "WHERE split_part(p.fasta_id, '|', 1) IN ('sp', 'tr') "
        f"AND upper(split_part(p.fasta_id, '|', 2)) IN ({placeholders}) "
        "ORDER BY 1, 2"
    )
    rows = conn.execute(sql, list(wanted)).fetchall()
    if not rows:
        raise DatasetError(
            "No feature rows matched the requested accessions; check that the "
            "proteins table contains them and that features were backfilled."
        )

    keys = tuple(
        ProteinKey(accession_of(fasta_id), fasta_id, source_file)
        for fasta_id, source_file, *_ in rows
    )
    counts = Counter(key.accession for key in keys)
    ambiguous = sorted(acc for acc, count in counts.items() if count > 1)
    if ambiguous:
        raise DatasetError(
            f"{len(ambiguous)} accession(s) map to more than one protein row "
            f"(e.g. {ambiguous[:5]}); alignment by accession would be ambiguous."
        )

    matrix = np.array([row[2:] for row in rows], dtype=np.float64)
    if matrix.shape != (len(rows), len(names)):
        raise DatasetError(
            f"Feature matrix has shape {matrix.shape}; expected "
            f"({len(rows)}, {len(names)})."
        )
    if not np.isfinite(matrix).all():
        bad_columns = [names[i] for i in np.where(~np.isfinite(matrix).all(axis=0))[0]]
        raise DatasetError(f"Non-finite values in feature column(s): {bad_columns}.")
    return FeatureTable(keys=keys, X=matrix, feature_names=names)


def load_labels_csv(
    path: str | Path,
    label_names: Sequence[str],
    *,
    protein_column: str = LABEL_REQUIRED_COLUMNS[0],
    label_column: str = LABEL_REQUIRED_COLUMNS[1],
    separator: str = ";",
) -> LabelTable:
    """Read ``protein_labels.csv`` into ``accession -> labels``.

    Proteins whose ``labels`` cell is empty are kept with an empty tuple and
    counted, never dropped. A label outside ``label_names`` is an error: the
    vocabulary is frozen and must not grow by accident.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        DatasetError: On missing columns, duplicate protein ids or unknown labels.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Label file not found: {source}")
    ordered = tuple(label_names)
    if not ordered:
        raise DatasetError("label_names is empty.")
    index = {name: position for position, name in enumerate(ordered)}
    if len(index) != len(ordered):
        raise DatasetError("label_names contains duplicates.")

    labels: dict[str, tuple[str, ...]] = {}
    empty = 0
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [
            column
            for column in (protein_column, label_column)
            if column not in fieldnames
        ]
        if missing:
            raise DatasetError(
                f"{source} is missing column(s) {missing}; found {fieldnames}."
            )
        for row in reader:
            protein_id = (row.get(protein_column) or "").strip().upper()
            if not protein_id:
                raise DatasetError(f"{source} has a row without a protein id.")
            if protein_id in labels:
                raise DatasetError(
                    f"Duplicate protein id {protein_id!r} in {source}."
                )
            cell = (row.get(label_column) or "").strip()
            names = tuple(
                sorted(
                    {
                        token.strip()
                        for token in cell.split(separator)
                        if token.strip()
                    }
                )
            )
            unknown = [name for name in names if name not in index]
            if unknown:
                raise DatasetError(
                    f"Protein {protein_id!r} carries label(s) outside the frozen "
                    f"vocabulary: {unknown}"
                )
            labels[protein_id] = names
            empty += 0 if names else 1

    if not labels:
        raise DatasetError(f"{source} contains no protein rows.")
    return LabelTable(
        labels=labels,
        label_names=ordered,
        proteins_without_labels=empty,
    )


@dataclass(frozen=True)
class DatasetReport:
    """What alignment did: matches, leftovers on each side, empty proteins."""

    matched: int
    unmatched_features: tuple[str, ...]
    unmatched_labels: tuple[str, ...]
    proteins_without_labels: tuple[str, ...]
    label_counts: Mapping[str, int]
    positive_pairs: int

    def as_dict(self) -> dict:
        return {
            "matched": self.matched,
            "unmatched_features": list(self.unmatched_features),
            "unmatched_labels": list(self.unmatched_labels),
            "proteins_without_labels": len(self.proteins_without_labels),
            "proteins_without_labels_ids": list(self.proteins_without_labels),
            "positive_pairs": self.positive_pairs,
            "label_counts": dict(self.label_counts),
        }


@dataclass(frozen=True)
class ProteinDataset:
    """Aligned matrix pair: one row per protein, one column per feature/label.

    ``keys[i]``, ``X[i]`` and ``y[i]`` always describe the same protein; rows are
    ordered by accession, and ``label_names`` keeps the frozen vocabulary order -
    labels absent from the sample stay as all-zero columns.
    """

    keys: tuple[ProteinKey, ...]
    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]
    report: DatasetReport

    @property
    def n_samples(self) -> int:
        return len(self.keys)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_labels(self) -> int:
        return len(self.label_names)

    def label_frequencies(self) -> dict[str, int]:
        """Positives per label, including labels with zero positives."""
        totals = self.y.sum(axis=0)
        return {
            name: int(totals[position])
            for position, name in enumerate(self.label_names)
        }


def align(features: FeatureTable, labels: LabelTable) -> ProteinDataset:
    """Join the two blocks on the UniProt accession (never on row position).

    Proteins present on only one side are reported, not silently dropped.
    Matched proteins without labels are kept as all-zero rows and counted.

    Raises:
        DatasetError: If there is no common protein.
    """
    feature_accessions = {key.accession for key in features.keys}
    label_accessions = set(labels.labels)
    matched = sorted(feature_accessions & label_accessions)
    if not matched:
        raise DatasetError(
            "No protein is present in both the feature block and the label "
            f"file ({len(feature_accessions)} features, {len(label_accessions)} "
            "labels)."
        )

    feature_index = {
        key.accession: position for position, key in enumerate(features.keys)
    }
    label_index = {
        name: position for position, name in enumerate(labels.label_names)
    }

    keys = tuple(features.keys[feature_index[acc]] for acc in matched)
    X = features.X[[feature_index[acc] for acc in matched], :].copy()

    y = np.zeros((len(matched), len(labels.label_names)), dtype=bool)
    positive_pairs = 0
    empty: list[str] = []
    for row, accession in enumerate(matched):
        names = labels.labels[accession]
        if not names:
            empty.append(accession)
        for name in names:
            y[row, label_index[name]] = True
            positive_pairs += 1

    report = DatasetReport(
        matched=len(matched),
        unmatched_features=tuple(sorted(feature_accessions - label_accessions)),
        unmatched_labels=tuple(sorted(label_accessions - feature_accessions)),
        proteins_without_labels=tuple(empty),
        label_counts={
            name: int(y[:, position].sum())
            for position, name in enumerate(labels.label_names)
        },
        positive_pairs=positive_pairs,
    )
    dataset = ProteinDataset(
        keys=keys,
        X=X,
        y=y,
        feature_names=features.feature_names,
        label_names=labels.label_names,
        report=report,
    )
    validate_dataset(dataset)
    return dataset


def validate_feature_matrix(
    X: object, feature_names: Sequence[str], *, name: str = "X"
) -> np.ndarray:
    """Validate a numeric feature block and return it as ``float64``.

    Shared by :func:`validate_dataset` and the pre-filtering layer, so both
    reject the same inputs with the same messages.

    Raises:
        DatasetError: On a non-2-D, empty, non-numeric or non-finite matrix, or
            on a name list that is empty, duplicated or of the wrong length.
    """
    matrix = np.asarray(X)
    if matrix.ndim != 2:
        raise DatasetError(f"{name} must be 2-D (n_samples, n_features).")
    if matrix.shape[0] == 0:
        raise DatasetError(f"{name} has no rows.")
    if matrix.shape[1] == 0:
        raise DatasetError(f"{name} has no feature columns.")
    names = tuple(feature_names)
    if not names:
        raise DatasetError("feature_names is empty.")
    if len(set(names)) != len(names):
        raise DatasetError("feature_names contains duplicates.")
    if matrix.shape[1] != len(names):
        raise DatasetError(
            f"{name} has {matrix.shape[1]} columns but {len(names)} feature names."
        )
    try:
        matrix = matrix.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"{name} must be numeric: {exc}") from exc
    if not np.isfinite(matrix).all():
        raise DatasetError(f"{name} contains non-finite values.")
    return matrix


def validate_dataset(dataset: ProteinDataset) -> None:
    """Check the invariants every downstream step relies on.

    Raises:
        DatasetError: On shape mismatch, non-finite features, non-boolean
            labels, duplicate keys or duplicate/empty column names.
    """
    if dataset.X.ndim != 2 or dataset.y.ndim != 2:
        raise DatasetError("X and y must both be 2-D.")
    if dataset.X.shape[0] != dataset.y.shape[0]:
        raise DatasetError(
            f"X has {dataset.X.shape[0]} rows but y has {dataset.y.shape[0]}."
        )
    if dataset.X.shape[0] != len(dataset.keys):
        raise DatasetError(
            f"{len(dataset.keys)} keys for {dataset.X.shape[0]} rows."
        )
    validate_feature_matrix(dataset.X, dataset.feature_names)
    if dataset.y.shape[1] != len(dataset.label_names):
        raise DatasetError(
            f"y has {dataset.y.shape[1]} columns but "
            f"{len(dataset.label_names)} label names."
        )
    if dataset.y.dtype != np.bool_:
        raise DatasetError(f"y must be boolean; got {dataset.y.dtype}.")
    if not dataset.label_names:
        raise DatasetError("label_names is empty.")
    if len(set(dataset.label_names)) != len(dataset.label_names):
        raise DatasetError("label_names contains duplicates.")
    accessions = [key.accession for key in dataset.keys]
    if len(set(accessions)) != len(accessions):
        raise DatasetError("Dataset keys contain duplicate accessions.")

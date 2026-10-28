"""Tests for dataset alignment between sequence features and GO Slim labels."""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

import duckdb
import numpy as np
import pytest

from protein_database.db import connect, init_database
from protein_database.ml.dataset import (
    DatasetError,
    DatasetReport,
    ProteinDataset,
    ProteinKey,
    accession_of,
    align,
    feature_columns,
    load_features,
    load_labels_csv,
    validate_dataset,
)
from protein_database.schema import COMPOSITION_COLUMNS, FEATURE_COLUMNS

LABELS = ("GO:0003824", "GO:0005488", "GO:0003677")


def _fill(conn: duckdb.DuckDBPyConnection, rows: list[tuple[str, str]], base: float) -> None:
    """Insert proteins + features + composition with accession-derived values.

    Values depend on the accession (not the insertion position), so two
    databases filled in different orders must produce identical matrices.
    """
    for accession, fasta_id in rows:
        offset = float(accession[-1])
        conn.execute(
            "INSERT INTO proteins VALUES (?, 'test.fasta', ?, 'MAAAA', 5, "
            "CURRENT_TIMESTAMP)",
            [fasta_id, f"protein {accession}"],
        )
        features = [
            base + offset + 0.01 * i for i in range(len(FEATURE_COLUMNS))
        ]
        conn.execute(
            "INSERT INTO sequence_features (fasta_id, source_file, "
            + ", ".join(FEATURE_COLUMNS)
            + ") VALUES (?, 'test.fasta', "
            + ", ".join("?" for _ in FEATURE_COLUMNS)
            + ")",
            [fasta_id, *features],
        )
        composition = [
            base + offset + 0.5 + 0.01 * i for i in range(len(COMPOSITION_COLUMNS))
        ]
        conn.execute(
            "INSERT INTO amino_acid_composition (fasta_id, source_file, "
            + ", ".join(COMPOSITION_COLUMNS)
            + ") VALUES (?, 'test.fasta', "
            + ", ".join("?" for _ in COMPOSITION_COLUMNS)
            + ")",
            [fasta_id, *composition],
        )


@pytest.fixture()
def conn():
    connection = connect(":memory:")
    init_database(connection)
    yield connection
    connection.close()


def _labels_csv(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    path = tmp_path / "protein_labels.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "protein_id",
                "n_labels",
                "labels",
                "n_annotations",
                "n_mapped",
                "n_no_vocabulary_ancestor",
                "n_obsolete",
                "n_unknown",
            ]
        )
        for protein_id, labels in rows:
            writer.writerow([protein_id, len(labels.split(";")) if labels else 0, labels, 1, 1, 0, 0, 0])
    return path


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------


def test_feature_columns_follow_the_schema_order() -> None:
    names = feature_columns()
    assert len(names) == 35
    assert names == tuple(FEATURE_COLUMNS) + tuple(COMPOSITION_COLUMNS)


def test_accession_of_parses_sp_and_tr_entries() -> None:
    assert accession_of("sp|P69905|HBA_HUMAN") == "P69905"
    assert accession_of("tr|A0A123|A0A123_HUMAN") == "A0A123"
    assert accession_of("sp|a0a086f3e3|X") == "A0A086F3E3"


def test_accession_of_rejects_malformed_identifiers() -> None:
    assert accession_of("P69905") == ""
    assert accession_of("foo|P69905|bar") == ""
    assert accession_of("") == ""


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------


def test_load_features_returns_one_row_per_accession(conn) -> None:
    _fill(conn, [("P00001", "sp|P00001|A"), ("P00002", "sp|P00002|B")], base=1.0)
    table = load_features(conn, ["P00002", "P00001"])

    assert table.X.shape == (2, 35)
    assert table.feature_names == feature_columns()
    assert [key.accession for key in table.keys] == ["P00001", "P00002"]
    assert table.keys[0].fasta_id == "sp|P00001|A"
    assert table.keys[0].source_file == "test.fasta"
    assert table.X[0, 0] < table.X[1, 0]


def test_load_features_accepts_a_column_subset(conn) -> None:
    _fill(conn, [("P00001", "sp|P00001|A")], base=1.0)
    table = load_features(conn, ["P00001"], columns=["molecular_weight"])
    assert table.X.shape == (1, 1)
    assert table.feature_names == ("molecular_weight",)


def test_load_features_requires_explicit_accessions(conn) -> None:
    with pytest.raises(DatasetError, match="No accessions"):
        load_features(conn, [])


def test_load_features_rejects_malformed_accessions(conn) -> None:
    with pytest.raises(DatasetError, match="look like UniProt"):
        load_features(conn, ["not an accession"])


def test_load_features_rejects_unknown_columns(conn) -> None:
    with pytest.raises(DatasetError, match="unknown feature column"):
        load_features(conn, ["P00001"], columns=["molecular_weight", "made_up"])


def test_load_features_raises_when_nothing_matches(conn) -> None:
    with pytest.raises(DatasetError, match="No feature rows"):
        load_features(conn, ["P99999"])


def test_load_features_rejects_ambiguous_accession(conn) -> None:
    _fill(conn, [("P00001", "sp|P00001|A"), ("P00001", "sp|P00001|B")], base=1.0)
    with pytest.raises(DatasetError, match="ambiguous"):
        load_features(conn, ["P00001"])


def test_load_features_rejects_non_finite_values(conn) -> None:
    _fill(conn, [("P00001", "sp|P00001|A")], base=1.0)
    conn.execute(
        "UPDATE sequence_features SET molecular_weight = 'Infinity'::DOUBLE "
        "WHERE fasta_id = 'sp|P00001|A'"
    )
    with pytest.raises(DatasetError, match="Non-finite"):
        load_features(conn, ["P00001"])


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------


def test_load_labels_csv_parses_semicolon_lists(tmp_path: Path) -> None:
    path = _labels_csv(
        tmp_path,
        [("P00001", "GO:0003824;GO:0005488"), ("P00002", "GO:0003677")],
    )
    table = load_labels_csv(path, LABELS)

    assert table.label_names == LABELS
    assert table.labels["P00001"] == ("GO:0003824", "GO:0005488")
    assert table.labels["P00002"] == ("GO:0003677",)
    assert table.proteins_without_labels == 0


def test_load_labels_csv_keeps_proteins_without_labels(tmp_path: Path) -> None:
    path = _labels_csv(tmp_path, [("P00001", ""), ("P00002", "GO:0003677")])
    table = load_labels_csv(path, LABELS)

    assert table.labels["P00001"] == ()
    assert table.proteins_without_labels == 1
    assert len(table.labels) == 2


def test_load_labels_csv_rejects_labels_outside_the_vocabulary(
    tmp_path: Path,
) -> None:
    path = _labels_csv(tmp_path, [("P00001", "GO:9999999")])
    with pytest.raises(DatasetError, match="outside the frozen vocabulary"):
        load_labels_csv(path, LABELS)


def test_load_labels_csv_rejects_duplicate_proteins(tmp_path: Path) -> None:
    path = _labels_csv(tmp_path, [("P00001", "GO:0003677"), ("P00001", "GO:0003824")])
    with pytest.raises(DatasetError, match="Duplicate protein id"):
        load_labels_csv(path, LABELS)


def test_load_labels_csv_rejects_a_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "broken.csv"
    path.write_text("protein_id,other\nP00001,x\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="missing column"):
        load_labels_csv(path, LABELS)


def test_load_labels_csv_requires_an_existing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_labels_csv(tmp_path / "absent.csv", LABELS)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def _blocks(conn, tmp_path: Path, label_rows, entries=None):
    entries = entries or [("P00001", "sp|P00001|A"), ("P00002", "sp|P00002|B")]
    _fill(conn, entries, base=1.0)
    features = load_features(conn, [accession for accession, _ in entries])
    labels = load_labels_csv(_labels_csv(tmp_path, label_rows), LABELS)
    return features, labels


def test_align_matches_by_accession_not_by_position(conn, tmp_path: Path) -> None:
    features, labels = _blocks(
        conn,
        tmp_path,
        [("P00002", "GO:0003677"), ("P00001", "GO:0003824")],
    )
    dataset = align(features, labels)

    assert [key.accession for key in dataset.keys] == ["P00001", "P00002"]
    assert dataset.X[0, 0] < dataset.X[1, 0]
    assert dataset.y[0].tolist() == [True, False, False]
    assert dataset.y[1].tolist() == [False, False, True]


def test_align_keeps_the_frozen_label_order_including_absent_labels(
    conn, tmp_path: Path
) -> None:
    features, labels = _blocks(conn, tmp_path, [("P00001", "GO:0003677")])
    dataset = align(features, labels)

    assert dataset.label_names == LABELS
    assert dataset.n_labels == 3
    assert dataset.label_frequencies() == {
        "GO:0003824": 0,
        "GO:0005488": 0,
        "GO:0003677": 1,
    }


def test_align_reports_unmatched_proteins_on_both_sides(conn, tmp_path: Path) -> None:
    features, labels = _blocks(
        conn,
        tmp_path,
        [("P00001", "GO:0003677"), ("P00003", "GO:0003824")],
    )
    dataset = align(features, labels)

    assert dataset.report.matched == 1
    assert dataset.report.unmatched_features == ("P00002",)
    assert dataset.report.unmatched_labels == ("P00003",)


def test_align_preserves_and_counts_proteins_without_labels(conn, tmp_path: Path) -> None:
    features, labels = _blocks(
        conn, tmp_path, [("P00001", ""), ("P00002", "GO:0003824")]
    )
    dataset = align(features, labels)

    assert dataset.report.matched == 2
    assert dataset.report.proteins_without_labels == ("P00001",)
    assert not dataset.y[0].any()
    assert dataset.y[1].tolist() == [True, False, False]


def test_align_fails_when_the_sides_do_not_intersect(conn, tmp_path: Path) -> None:
    features, labels = _blocks(conn, tmp_path, [("P00099", "GO:0003824")])
    with pytest.raises(DatasetError, match="No protein is present in both"):
        align(features, labels)


def test_align_is_deterministic_across_insertion_order(tmp_path: Path) -> None:
    entries = [("P00001", "sp|P00001|A"), ("P00002", "sp|P00002|B")]
    label_rows = [("P00001", "GO:0003824"), ("P00002", "GO:0003677")]

    first_conn = connect(":memory:")
    init_database(first_conn)
    _fill(first_conn, entries, base=1.0)
    first = align(
        load_features(first_conn, ["P00001", "P00002"]),
        load_labels_csv(_labels_csv(tmp_path, label_rows), LABELS),
    )
    first_conn.close()

    second_conn = connect(":memory:")
    init_database(second_conn)
    _fill(second_conn, list(reversed(entries)), base=1.0)
    second = align(
        load_features(second_conn, ["P00002", "P00001"]),
        load_labels_csv(_labels_csv(tmp_path, list(reversed(label_rows))), LABELS),
    )
    second_conn.close()

    assert first.keys == second.keys
    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.y, second.y)
    assert first.report.as_dict() == second.report.as_dict()


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def _valid_dataset() -> ProteinDataset:
    return ProteinDataset(
        keys=(ProteinKey("P00001", "sp|P00001|A", "test.fasta"),),
        X=np.zeros((1, 1)),
        y=np.zeros((1, 1), dtype=bool),
        feature_names=("molecular_weight",),
        label_names=("GO:0003824",),
        report=DatasetReport(1, (), (), (), {"GO:0003824": 0}, 0),
    )


def test_validate_dataset_accepts_a_consistent_dataset() -> None:
    validate_dataset(_valid_dataset())


def test_validate_dataset_rejects_row_mismatch() -> None:
    with pytest.raises(DatasetError, match="rows"):
        validate_dataset(dataclasses.replace(_valid_dataset(), X=np.zeros((2, 1))))


def test_validate_dataset_rejects_non_boolean_labels() -> None:
    with pytest.raises(DatasetError, match="boolean"):
        validate_dataset(dataclasses.replace(_valid_dataset(), y=np.zeros((1, 1))))


def test_validate_dataset_rejects_non_finite_features() -> None:
    with pytest.raises(DatasetError, match="non-finite"):
        validate_dataset(
            dataclasses.replace(_valid_dataset(), X=np.array([[np.inf]]))
        )


def test_validate_dataset_rejects_duplicate_keys() -> None:
    key = ProteinKey("P00001", "sp|P00001|A", "test.fasta")
    with pytest.raises(DatasetError, match="duplicate accessions"):
        validate_dataset(
            dataclasses.replace(
                _valid_dataset(),
                keys=(key, key),
                X=np.zeros((2, 1)),
                y=np.zeros((2, 1), dtype=bool),
            )
        )


def test_validate_dataset_rejects_duplicate_column_names() -> None:
    with pytest.raises(DatasetError, match="feature_names contains duplicates"):
        validate_dataset(
            dataclasses.replace(
                _valid_dataset(), feature_names=("f", "f"), X=np.zeros((1, 2))
            )
        )

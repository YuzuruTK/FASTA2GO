"""Behaviour tests for the extraction stage.

Twelve tests covering the real risks: FASTA reading, the denominators of the
composition features (AAC over canonical residues, DPC over adjacent canonical
pairs), idempotent persistence, one transaction per feature batch, resuming, and
the correspondence between proteins and UniProt/GO annotations. No network: the
UniProt client is exercised against a response captured from the real API
(tests/data).
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import pytest
import requests

from fasta2go import db, features, pipeline, uniprot
from fasta2go.fasta import read_fasta

DATA = Path(__file__).parent / "data"
SAMPLE = DATA / "sample.fasta"
HBA = (
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF"
    "DLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRV"
    "DPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
)
CANONICAL = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    """An empty in-memory database with the project schema.

    ``db.write_rows`` stages each table through one shared file per table, so the
    staging directory is pointed at ``tmp_path``: a pipeline running in another
    terminal can then never interfere with the tests.
    """
    monkeypatch.setattr(db, "STAGING_DIR", tmp_path / "staging")
    connection = db.connect(":memory:")
    db.init_schema(connection)
    yield connection
    connection.close()


class FakeResponse:
    status_code = 200
    text = ""
    headers: dict[str, str] = {}

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def json(self) -> dict:
        return self.payload


class FakeSession:
    """Serves the captured UniProt response instead of calling the API."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def get(self, url: str, params: dict | None = None, timeout: float | None = None):
        self.calls.append(params or {})
        return FakeResponse(self.payload)


class ExplodingSession:
    """Fails the test if any request is attempted."""

    def get(self, *args, **kwargs):
        raise AssertionError("no HTTP request expected")


def captured_payload() -> dict:
    return json.loads((DATA / "uniprot_response.json").read_text(encoding="utf-8"))


# 1. FASTA reading -----------------------------------------------------------

def test_parser_reads_records_and_reports_every_problem() -> None:
    records = list(read_fasta(SAMPLE))

    assert len(records) == 7
    first = records[0]
    assert first.fasta_id == "sp|P69905|HBA_HUMAN"
    assert first.uniprot_accession == "P69905"
    assert first.sequence == HBA  # three lines joined
    assert first.issues == []

    assert records[2].fasta_id == first.fasta_id  # repeated identifier

    generic = records[3]
    assert generic.fasta_id == "garbage"
    assert generic.uniprot_accession is None
    assert [code for code, _ in generic.issues] == ["header_without_accession"]

    empty = records[4]
    assert empty.is_valid is False
    assert [code for code, _ in empty.issues] == ["empty_sequence"]

    malformed = records[5]
    assert malformed.fasta_id == ""
    assert malformed.is_valid is False
    assert [code for code, _ in malformed.issues] == ["malformed_header"]

    ambiguous = records[6]
    assert ambiguous.sequence_length == 26
    assert ("noncanonical_residues", "*=1, B=1, X=1, Z=1") in ambiguous.issues


def test_parser_reads_gzip_and_rejects_missing_header(tmp_path: Path) -> None:
    content = ">sp|P69905|HBA_HUMAN Hemoglobin\nACDEFGHIK\n"
    compressed = tmp_path / "one.fasta.gz"
    with gzip.open(compressed, "wt", encoding="utf-8") as handle:
        handle.write(content)
    assert [record.fasta_id for record in read_fasta(compressed)] == [
        "sp|P69905|HBA_HUMAN"
    ]

    broken = tmp_path / "broken.fasta"
    broken.write_text("ACDEFGHIK\n>sp|P69905|HBA_HUMAN\nACD\n", encoding="utf-8")
    with pytest.raises(ValueError, match="before the first header"):
        list(read_fasta(broken))


# 2. Sequence characteristics ------------------------------------------------

def test_composition_uses_canonical_residues_as_denominator() -> None:
    composition = features.amino_acid_composition("ACDE")
    assert composition["alanine_pct"] == 25.0
    assert sum(composition.values()) == pytest.approx(100.0)

    # X, Z, B and * are not residues of the denominator; cleaning keeps MKVL.
    assert features.clean_sequence("MKXVLZB*") == "MKVL"

    pairs = features.dipeptide_composition("ACDE")
    assert len(pairs) == 400
    assert pairs["A_C"] == pytest.approx(100 / 3)
    assert pairs["C_D"] == pytest.approx(100 / 3)
    assert pairs["D_E"] == pytest.approx(100 / 3)
    assert sum(pairs.values()) == pytest.approx(100.0)
    assert pairs["A_A"] == 0.0

    assert features.dipeptide_composition("A")["A_A"] is None  # no pair exists


def test_dipeptide_keeps_the_adjacency_of_the_original_sequence() -> None:
    # X breaks adjacency: AC and DE exist, CD does not.
    pairs = features.dipeptide_composition("ACXDE")
    assert len(pairs) == 400
    assert pairs["A_C"] == pytest.approx(50.0)
    assert pairs["D_E"] == pytest.approx(50.0)
    assert pairs["C_D"] == 0.0
    assert sum(pairs.values()) == pytest.approx(100.0)
    assert features.canonical_pairs("ACXDE") == 2

    # Two non-canonical residues block every pair: nothing is defined.
    blocked = features.dipeptide_composition("AXXA")
    assert all(value is None for value in blocked.values())
    assert features.canonical_pairs("AXXA") == 0

    # AAC has its own denominator and ignores the adjacency rule.
    composition = features.amino_acid_composition(features.clean_sequence("AXXA"))
    assert composition["alanine_pct"] == pytest.approx(100.0)
    assert features.canonical_pairs("ACDE") == 3


def test_physicochemical_features_are_correct_and_deterministic() -> None:
    values = features.sequence_features(HBA)
    assert values["canonical_length"] == 142
    assert values["noncanonical_residues"] == 0
    assert values["molecular_weight"] == pytest.approx(15257.36, abs=0.01)
    assert 8.5 < values["isoelectric_point"] < 9.0
    assert 0.0 < values["gravy"] < 0.1
    assert values["instability_index"] > 0
    assert 60 < values["aliphatic_index"] < 120
    assert 0 < values["aromaticity"] < 1
    assert 0 < values["sequence_entropy"] <= math.log2(len(CANONICAL))
    assert features.sequence_features(HBA) == values

    with pytest.raises(ValueError, match="no canonical amino acid"):
        features.sequence_features("XXX")


# 3. Persistence -------------------------------------------------------------

def test_ingestion_and_features_are_idempotent(conn) -> None:
    first = pipeline.ingest(conn, SAMPLE)
    assert first["inserted"] == 4
    assert first["rejected"] == 2
    assert first["duplicates"] == 1

    second = pipeline.ingest(conn, SAMPLE)
    assert second["inserted"] == 0
    assert second["already_present"] == 4

    assert pipeline.compute_features(conn)["computed"] == 4
    assert pipeline.compute_features(conn)["computed"] == 0

    counts = conn.execute(
        "SELECT (SELECT count(*) FROM proteins), "
        "(SELECT count(*) FROM sequence_features), "
        "(SELECT count(*) FROM amino_acid_composition), "
        "(SELECT count(*) FROM dipeptide_composition)"
    ).fetchone()
    assert counts == (4, 4, 4, 4)

    issues = dict(
        conn.execute("SELECT issue, count(*) FROM ingestion_issues GROUP BY 1").fetchall()
    )
    assert issues == {
        "duplicate_id": 1,
        "empty_sequence": 1,
        "header_without_accession": 1,
        "malformed_header": 1,
        "noncanonical_residues": 1,
    }

    # An outdated feature row is recomputed instead of being kept or duplicated.
    conn.execute("UPDATE sequence_features SET feature_version = 0")
    assert pipeline.compute_features(conn)["computed"] == 4
    assert conn.execute("SELECT count(*) FROM sequence_features").fetchone()[0] == 4


def test_proteins_without_canonical_residues_keep_one_row_per_table(
    conn, tmp_path: Path
) -> None:
    fasta = tmp_path / "ambiguous.fasta"
    fasta.write_text(
        ">sp|P99999|ALL_AMBIGUOUS Nothing canonical here\nXXXXZ\n", encoding="utf-8"
    )

    pipeline.ingest(conn, fasta)
    counts = pipeline.compute_features(conn)

    assert counts["computed"] == 0
    assert counts["skipped"] == 1
    assert conn.execute(
        "SELECT (SELECT count(*) FROM sequence_features), "
        "(SELECT count(*) FROM amino_acid_composition), "
        "(SELECT count(*) FROM dipeptide_composition)"
    ).fetchone() == (1, 1, 1)

    row = conn.execute(
        "SELECT feature_version, canonical_length, noncanonical_residues, "
        "canonical_pairs, molecular_weight FROM sequence_features"
    ).fetchone()
    assert row == (features.FEATURE_VERSION, 0, 5, 0, None)
    assert conn.execute(
        "SELECT count(*) FROM amino_acid_composition WHERE alanine_pct IS NOT NULL"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT count(*) FROM dipeptide_composition WHERE A_A IS NOT NULL"
    ).fetchone()[0] == 0


def test_failed_batch_write_rolls_back_the_three_tables(conn, monkeypatch) -> None:
    pipeline.ingest(conn, SAMPLE)
    state = {"fail": True}
    real_write_rows = db.write_rows

    def flaky_write_rows(target, table, columns, rows, *, key=(), update=()):
        if table == "dipeptide_composition" and state["fail"]:
            raise RuntimeError("simulated failure")
        return real_write_rows(target, table, columns, rows, key=key, update=update)

    monkeypatch.setattr(db, "write_rows", flaky_write_rows)
    with pytest.raises(RuntimeError, match="simulated failure"):
        pipeline.compute_features(conn)

    # Nothing from the batch survived: features and AAC were written before the
    # failing DPC statement and must have been rolled back with it.
    assert conn.execute(
        "SELECT (SELECT count(*) FROM sequence_features), "
        "(SELECT count(*) FROM amino_acid_composition), "
        "(SELECT count(*) FROM dipeptide_composition)"
    ).fetchone() == (0, 0, 0)

    state["fail"] = False
    assert pipeline.compute_features(conn)["computed"] == 4


def test_enrichment_is_idempotent_resumable_and_marks_missing_entries(conn) -> None:
    pipeline.ingest(conn, SAMPLE)
    pipeline.compute_features(conn)
    session = FakeSession(captured_payload())

    first = pipeline.enrich(conn, session=session)
    # Q8N158 is not part of the captured response: it must be recorded as missing,
    # never annotated with another protein's data.
    assert (first["queried"], first["matched"], first["not_found"]) == (3, 2, 1)

    rows = dict(
        conn.execute(
            "SELECT fasta_id, status FROM uniprot_annotations"
        ).fetchall()
    )
    assert rows["sp|P69905|HBA_HUMAN"] == "matched"
    assert rows["sp|Q8N158|GPC2_HUMAN"] == "not_found"

    # Nothing pending: no request at all.
    second = pipeline.enrich(conn, session=session)
    assert second["queried"] == 0
    assert len(session.calls) == 1

    go_before = conn.execute("SELECT count(*) FROM protein_go_annotations").fetchone()[0]

    # Simulated interruption: one protein lost its annotation, only it is retried.
    conn.execute("DELETE FROM uniprot_annotations WHERE fasta_id = 'sp|P69905|HBA_HUMAN'")
    conn.execute("DELETE FROM protein_go_annotations WHERE fasta_id = 'sp|P69905|HBA_HUMAN'")
    third = pipeline.enrich(conn, session=session)
    assert third["queried"] == 1
    assert conn.execute(
        "SELECT count(*) FROM protein_go_annotations WHERE fasta_id = 'sp|P69905|HBA_HUMAN'"
    ).fetchone()[0] > 0
    assert conn.execute("SELECT count(*) FROM protein_go_annotations").fetchone()[0] == go_before


# 4. UniProt client ----------------------------------------------------------

def test_uniprot_entry_parsing_keeps_go_aspect_evidence_and_provenance() -> None:
    payload = captured_payload()
    annotation = uniprot.parse_entry(
        next(e for e in payload["results"] if e["primaryAccession"] == "P69905")
    )

    assert annotation.uniprot_id == "HBA_HUMAN"
    assert annotation.protein_name == "Hemoglobin subunit alpha"
    assert annotation.organism == "Homo sapiens"
    assert annotation.taxon_id == 9606
    assert annotation.gene_names == "HBA1; HBA2"
    assert annotation.source_length == 142
    assert annotation.function_description.startswith("Involved in oxygen transport")
    assert "Heme" in annotation.keywords

    found = {(go.go_id, go.evidence_code): go for go in annotation.go}
    assert found[("GO:0005576", "IDA")].aspect == "cellular_component"
    assert found[("GO:0005576", "IDA")].evidence_source == "UniProtKB"
    assert found[("GO:0005576", "IDA")].provenance == "PubMed:21805676"

    # Most UniProt GO annotations carry no literature reference at all.
    assert found[("GO:0005829", "TAS")].provenance is None
    assert found[("GO:0005829", "TAS")].evidence_source == "Reactome"

    assert found[("GO:0005344", "IBA")].aspect == "molecular_function"
    assert found[("GO:0005344", "IBA")].go_name == "oxygen carrier activity"
    assert {go.aspect for go in annotation.go} == {
        "molecular_function",
        "biological_process",
        "cellular_component",
    }


def test_failed_requests_are_never_recorded_as_missing(conn, monkeypatch) -> None:
    assert uniprot.fetch(["not-an-accession"], session=ExplodingSession()) == {}

    class OfflineSession:
        def get(self, *args, **kwargs):
            raise requests.ConnectionError("offline")

    with pytest.raises(uniprot.UniProtError, match="network error"):
        uniprot.fetch(["P69905"], session=OfflineSession(), attempts=1, pause=0)

    pipeline.ingest(conn, SAMPLE)

    def offline(*args, **kwargs):
        raise uniprot.UniProtError("offline")

    monkeypatch.setattr(uniprot, "fetch", offline)
    with pytest.raises(uniprot.UniProtError):
        pipeline.enrich(conn, session=OfflineSession())

    # A failed request is not a missing annotation: nothing is written.
    assert conn.execute("SELECT count(*) FROM uniprot_annotations").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM protein_go_annotations").fetchone()[0] == 0
    assert pipeline.database_summary(conn)["unqueried"] == 3


def test_protein_features_view_persists_consistent_denominators(conn) -> None:
    pipeline.ingest(conn, SAMPLE)
    pipeline.compute_features(conn)

    row = conn.execute(
        "SELECT canonical_length, hydrophobic_pct + polar_pct + charged_pct, "
        "aromatic_pct FROM protein_features WHERE fasta_id = 'sp|P69905|HBA_HUMAN'"
    ).fetchone()
    assert row[0] == 142
    # The classes overlap by design (charged residues are also polar/aromatic),
    # so the sum is checked against the stored composition, not against 100.
    assert 80 < row[1] < 100
    assert 0 < row[2] < 15

    # The DPC denominator is stored next to the features: 142 residues, no
    # non-canonical residue, so 141 adjacent canonical pairs.
    assert conn.execute(
        "SELECT canonical_pairs FROM sequence_features "
        "WHERE fasta_id = 'sp|P69905|HBA_HUMAN'"
    ).fetchone()[0] == 141

    summary = pipeline.database_summary(conn)
    assert summary["proteins"] == 4
    assert summary["with_features"] == 4
    assert summary["orphans"] == 0
    assert summary["feature_versions"] == "2"

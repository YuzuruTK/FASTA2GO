"""Behaviour tests for the extraction stage.

Thirteen tests covering the real risks: FASTA reading, the denominators of the
composition features (AAC over canonical residues, DPC over adjacent canonical
pairs), the integrated extraction (each batch is read, analysed, queried on
UniProt and stored in one transaction before the next one starts), the protein
limit, the refusal to mix with an existing database, and the correspondence
between proteins and UniProt/GO annotations. No network: the UniProt client is
exercised against a response captured from the real API (tests/data).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import duckdb
import pytest
import requests

import main
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


# 3. Integrated extraction ---------------------------------------------------

def test_extract_stores_each_batch_with_features_go_and_issues(conn) -> None:
    session = FakeSession(captured_payload())
    counts = pipeline.extract(conn, SAMPLE, batch_size=2, session=session)

    # Each batch is stored before the next one starts: four valid proteins, two
    # per batch, so two independent UniProt requests.
    assert (counts["batches"], counts["inserted"], counts["records"]) == (2, 4, 7)
    assert (counts["rejected"], counts["duplicates"]) == (2, 1)
    assert counts["computed"] == 4
    assert counts["requests"] == 2
    # Q8N158 is not part of the captured response: it must be recorded as
    # missing, never annotated with another protein's data.
    assert (counts["queried"], counts["matched"], counts["not_found"]) == (3, 2, 1)

    tables = (
        "proteins",
        "sequence_features",
        "amino_acid_composition",
        "dipeptide_composition",
    )
    counts_sql = "SELECT " + ", ".join(
        f"(SELECT count(*) FROM {table})" for table in tables
    )
    assert conn.execute(counts_sql).fetchone() == (4, 4, 4, 4)

    assert dict(
        conn.execute("SELECT fasta_id, status FROM uniprot_annotations").fetchall()
    ) == {
        "sp|P69905|HBA_HUMAN": "matched",
        "sp|Q6GZX4|001R_FRG3G": "matched",
        "sp|Q8N158|GPC2_HUMAN": "not_found",
    }
    assert conn.execute(
        "SELECT count(*) FROM protein_go_annotations "
        "WHERE fasta_id = 'sp|Q8N158|GPC2_HUMAN'"
    ).fetchone()[0] == 0
    hba_terms = {
        row[0]
        for row in conn.execute(
            "SELECT go_id FROM protein_go_annotations "
            "WHERE fasta_id = 'sp|P69905|HBA_HUMAN'"
        ).fetchall()
    }
    assert "GO:0005576" in hba_terms

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

    summary = pipeline.database_summary(conn)
    assert summary["proteins"] == 4
    assert summary["with_features"] == 4
    assert summary["with_composition"] == 4
    assert summary["with_dipeptides"] == 4
    assert summary["unqueried"] == 0
    assert summary["orphans"] == 0


def test_extract_respects_the_protein_limit(conn) -> None:
    counts = pipeline.extract(
        conn, SAMPLE, batch_size=10, limit=2, session=FakeSession(captured_payload())
    )

    assert counts["inserted"] == 2
    assert counts["records"] == 2  # reading stops as soon as the limit is reached
    assert conn.execute(
        "SELECT (SELECT count(*) FROM proteins), "
        "(SELECT count(*) FROM sequence_features), "
        "(SELECT count(*) FROM dipeptide_composition)"
    ).fetchone() == (2, 2, 2)
    assert conn.execute("SELECT count(*) FROM protein_go_annotations").fetchone()[0] > 0


def test_proteins_without_canonical_residues_keep_one_row_per_table(
    conn, tmp_path: Path
) -> None:
    fasta = tmp_path / "ambiguous.fasta"
    fasta.write_text(
        ">sp|P99999|ALL_AMBIGUOUS Nothing canonical here\nXXXXZ\n", encoding="utf-8"
    )

    counts = pipeline.extract(conn, fasta, with_uniprot=False)

    assert counts["computed"] == 0
    assert counts["skipped"] == 1
    assert counts["inserted"] == 1
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


def test_failed_batch_write_rolls_back_every_table(conn, monkeypatch) -> None:
    state = {"fail": True}
    real_write_rows = db.write_rows

    def flaky_write_rows(target, table, columns, rows, *, key=(), update=()):
        if table == "protein_go_annotations" and state["fail"]:
            raise RuntimeError("simulated failure")
        return real_write_rows(target, table, columns, rows, key=key, update=update)

    monkeypatch.setattr(db, "write_rows", flaky_write_rows)
    with pytest.raises(RuntimeError, match="simulated failure"):
        pipeline.extract(conn, SAMPLE, session=FakeSession(captured_payload()))

    # One transaction per batch: nothing of the failed batch survived in any table.
    assert conn.execute(
        "SELECT (SELECT count(*) FROM proteins), "
        "(SELECT count(*) FROM sequence_features), "
        "(SELECT count(*) FROM amino_acid_composition), "
        "(SELECT count(*) FROM dipeptide_composition), "
        "(SELECT count(*) FROM uniprot_annotations), "
        "(SELECT count(*) FROM protein_go_annotations), "
        "(SELECT count(*) FROM ingestion_issues)"
    ).fetchone() == (0, 0, 0, 0, 0, 0, 0)

    state["fail"] = False
    assert pipeline.extract(
        conn, SAMPLE, session=FakeSession(captured_payload())
    )["inserted"] == 4


def test_failed_uniprot_request_stores_nothing(conn, monkeypatch) -> None:
    class OfflineSession:
        def get(self, *args, **kwargs):
            raise requests.ConnectionError("offline")

    # The client reports a failure instead of an empty answer...
    with pytest.raises(uniprot.UniProtError, match="network error"):
        uniprot.fetch(["P69905"], session=OfflineSession(), attempts=1, pause=0)
    # ...and never sends a request for malformed accessions.
    assert uniprot.fetch(["not-an-accession"], session=ExplodingSession()) == {}

    def offline(*args, **kwargs):
        raise uniprot.UniProtError("offline")

    monkeypatch.setattr(uniprot, "fetch", offline)
    with pytest.raises(uniprot.UniProtError):
        pipeline.extract(conn, SAMPLE, session=OfflineSession())

    # A failed request is not a missing annotation: the batch is not written.
    assert conn.execute(
        "SELECT (SELECT count(*) FROM proteins), "
        "(SELECT count(*) FROM sequence_features), "
        "(SELECT count(*) FROM uniprot_annotations), "
        "(SELECT count(*) FROM protein_go_annotations)"
    ).fetchone() == (0, 0, 0, 0)


def test_protein_features_view_persists_consistent_denominators(conn) -> None:
    pipeline.extract(conn, SAMPLE, with_uniprot=False)

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


# 5. Command line ------------------------------------------------------------

def test_cli_replaces_an_existing_database_only_with_recreate(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(db, "STAGING_DIR", tmp_path / "staging")
    target = tmp_path / "fasta2go.duckdb"

    def arguments(**overrides) -> argparse.Namespace:
        settings = dict(
            fasta=str(SAMPLE),
            db=str(target),
            limit=2,
            feature_batch_size=2,
            batch_size=200,
            no_uniprot=True,
            recreate=False,
        )
        settings.update(overrides)
        return argparse.Namespace(**settings)

    main.run(arguments())
    with duckdb.connect(str(target), read_only=True) as handle:
        assert handle.execute("SELECT count(*) FROM proteins").fetchone()[0] == 2

    # An existing database is never replaced silently.
    with pytest.raises(SystemExit, match="already exists"):
        main.run(arguments())

    main.run(arguments(recreate=True))
    with duckdb.connect(str(target), read_only=True) as handle:
        assert handle.execute("SELECT count(*) FROM proteins").fetchone()[0] == 2
        assert handle.execute("SELECT count(*) FROM sequence_features").fetchone()[0] == 2

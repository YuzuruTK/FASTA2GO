"""Tests for UniProt TSV parsing."""

import pytest

from protein_database.uniprot.parser import (
    STATUS_FETCHED,
    STATUS_MISSING_FUNCTION,
    TsvParseError,
    missing_accessions,
    parse_tsv,
)

HEADER = (
    "Entry\tEntry Name\tProtein names\tGene Names\tFunction [CC]\t"
    "Subcellular location [CC]\tProtein existence\tOrganism\tKeywords"
)

ROW_WITH_FUNCTION = (
    "P69905\tHBA_HUMAN\tHemoglobin subunit alpha\tHBA1; HBA2\t"
    "FUNCTION: Involved in oxygen transport.\t\tEvidence at protein level\t"
    "Homo sapiens (Human)\t3D-structure;Acetylation"
)

ROW_NO_FUNCTION = (
    "A0A0C5B5G6\tMOTSC_HUMAN\tMitochondrial-derived peptide\tMOTS-c\t\t"
    "\tPredicted\tHomo sapiens (Human)\t"
)


def _tsv(*rows: str) -> str:
    return "\n".join([HEADER, *rows]) + "\n"


def test_parse_tsv_with_function() -> None:
    parsed = parse_tsv(_tsv(ROW_WITH_FUNCTION), ["P69905"])
    assert len(parsed) == 1
    ann = parsed[0]
    assert ann["uniprot_accession"] == "P69905"
    assert ann["uniprot_id"] == "HBA_HUMAN"
    assert ann["protein_name"] == "Hemoglobin subunit alpha"
    assert ann["gene_name"] == "HBA1; HBA2"
    assert ann["function"].startswith("FUNCTION: Involved in oxygen transport")
    assert ann["protein_existence"] == "Evidence at protein level"
    assert ann["organism"] == "Homo sapiens (Human)"
    assert ann["keywords"] == "3D-structure;Acetylation"
    assert ann["status"] == STATUS_FETCHED


def test_parse_tsv_missing_function() -> None:
    parsed = parse_tsv(_tsv(ROW_NO_FUNCTION), ["A0A0C5B5G6"])
    assert len(parsed) == 1
    ann = parsed[0]
    assert ann["uniprot_accession"] == "A0A0C5B5G6"
    assert ann["function"] is None
    assert ann["status"] == STATUS_MISSING_FUNCTION
    # Other annotations are still preserved.
    assert ann["organism"] == "Homo sapiens (Human)"


def test_parse_tsv_empty_raises() -> None:
    with pytest.raises(TsvParseError):
        parse_tsv("", ["P69905"])


def test_parse_tsv_wrong_header_raises() -> None:
    with pytest.raises(TsvParseError):
        parse_tsv("Bad\tHeader\nP69905\tx\n", ["P69905"])


def test_parse_tsv_truncated_row_raises() -> None:
    # A row with fewer columns than expected indicates truncation.
    truncated = HEADER + "\nP69905\tHBA_HUMAN\n"
    with pytest.raises(TsvParseError):
        parse_tsv(truncated, ["P69905"])


def test_missing_accessions() -> None:
    parsed = parse_tsv(_tsv(ROW_WITH_FUNCTION), ["P69905", "P68871", "Q6GZX4"])
    missing = missing_accessions(["P69905", "P68871", "Q6GZX4"], parsed)
    assert missing == ["P68871", "Q6GZX4"]


def test_missing_accessions_empty_when_all_found() -> None:
    parsed = parse_tsv(_tsv(ROW_WITH_FUNCTION), ["P69905"])
    assert missing_accessions(["P69905"], parsed) == []
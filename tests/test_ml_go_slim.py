"""Tests for the GO Slim MF mapping (vocabulary, propagation, accounting)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from protein_database.ml.go_slim import (
    ASPECT_ROOT,
    LABEL_COLUMNS,
    MF_NAMESPACE,
    STATUS_MAPPED,
    STATUS_NO_ANCESTOR,
    STATUS_OBSOLETE,
    STATUS_UNKNOWN,
    VOCABULARY_COLUMNS,
    GoSlimError,
    build_ancestor_map,
    build_vocabulary,
    check_slim_equivalence,
    classify_annotation,
    map_proteins,
    parse_obo,
    read_term_ids,
    sha256_file,
    summarize,
    verify_integrity,
    write_json,
    write_labels_csv,
    write_vocabulary_csv,
)

#: Two-level MF hierarchy plus a BP part_of parent, a CC term, an obsolete term
#: and one alt_id. Large enough to exercise every rule, small enough to read.
OBO_TEXT = """\
format-version: 1.2
data-version: test/release-1

[Term]
id: GO:0003674
name: molecular_function
namespace: molecular_function
subset: goslim_generic

[Term]
id: GO:0003824
name: catalytic activity
namespace: molecular_function
subset: goslim_generic
is_a: GO:0003674 ! molecular_function

[Term]
id: GO:0016787
name: hydrolase activity
namespace: molecular_function
subset: goslim_generic
is_a: GO:0003824 ! catalytic activity

[Term]
id: GO:0016740
name: transferase activity
namespace: molecular_function
subset: goslim_generic
is_a: GO:0003824 ! catalytic activity

[Term]
id: GO:0016301
name: kinase activity
namespace: molecular_function
is_a: GO:0016740 ! transferase activity

[Term]
id: GO:0004672
name: protein kinase activity
namespace: molecular_function
subset: goslim_generic
is_a: GO:0016301 ! kinase activity
relationship: part_of GO:0006468 ! protein phosphorylation

[Term]
id: GO:0004553
name: hydrolase acting on glycosyl bonds
namespace: molecular_function
subset: goslim_generic
relationship: part_of GO:0016787 ! hydrolase activity
relationship: regulates GO:0008150 ! biological_process

[Term]
id: GO:0005488
name: binding
namespace: molecular_function
is_a: GO:0003674 ! molecular_function

[Term]
id: GO:0005524
name: ATP binding
namespace: molecular_function
is_a: GO:0005488 ! binding

[Term]
id: GO:0005634
name: nucleus
namespace: cellular_component
subset: goslim_generic

[Term]
id: GO:0006468
name: protein phosphorylation
namespace: biological_process
subset: goslim_generic

[Term]
id: GO:0007000
name: obsolete ancestor activity
namespace: molecular_function
is_obsolete: true

[Term]
id: GO:0007001
name: activity under an obsolete parent
namespace: molecular_function
is_a: GO:0007000 ! obsolete ancestor activity
is_a: GO:0016787 ! hydrolase activity

[Term]
id: GO:0008888
name: renamed hydrolase activity
namespace: molecular_function
alt_id: GO:0007777
is_a: GO:0016787 ! hydrolase activity

[Term]
id: GO:0009999
name: obsolete activity
namespace: molecular_function
alt_id: GO:0009998
is_obsolete: true
replaced_by: GO:0016787
"""


#: Artificial graph where an MF term crosses into BP and the BP node points back
#: to another MF term of the vocabulary. No such path exists in go-basic.obo
#: (audited: 0 cross-namespace edges), so the guard must be structural.
CROSS_ASPECT_OBO = """\
format-version: 1.2
data-version: test/cross-aspect

[Term]
id: GO:0003674
name: molecular_function
namespace: molecular_function
subset: goslim_generic

[Term]
id: GO:0003824
name: catalytic activity
namespace: molecular_function
subset: goslim_generic
is_a: GO:0003674 ! molecular_function

[Term]
id: GO:0004672
name: protein kinase activity
namespace: molecular_function
subset: goslim_generic
is_a: GO:0003824 ! catalytic activity
relationship: part_of GO:0006468 ! protein phosphorylation

[Term]
id: GO:0006468
name: protein phosphorylation
namespace: biological_process
is_a: GO:0008000 ! invented MF reached through a BP node

[Term]
id: GO:0008000
name: invented MF reached through a BP node
namespace: molecular_function
subset: goslim_generic

[Term]
id: GO:0005488
name: binding
namespace: molecular_function
is_a: GO:0003674 ! molecular_function
"""


@pytest.fixture()
def cross_ontology(tmp_path: Path):
    path = tmp_path / "cross_aspect.obo"
    path.write_text(CROSS_ASPECT_OBO, encoding="utf-8")
    ontology = parse_obo(path)
    declared = {
        term_id
        for term_id, term in ontology.terms.items()
        if "goslim_generic" in term.subsets
    }
    vocabulary = build_vocabulary(
        ontology, declared, extension=("GO:0005488",)
    )
    return ontology, vocabulary


@pytest.fixture()
def obo_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.obo"
    path.write_text(OBO_TEXT, encoding="utf-8")
    return path


@pytest.fixture()
def ontology(obo_path: Path):
    return parse_obo(obo_path)


@pytest.fixture()
def vocabulary(ontology):
    return build_vocabulary(
        ontology, read_term_ids_from(ontology), extension=("GO:0005488",)
    )


def read_term_ids_from(ontology) -> set[str]:
    """Slim membership as declared by the fixture (``subset: goslim_generic``)."""
    return {
        term.id for term in ontology.terms.values() if "goslim_generic" in term.subsets
    }


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parse_reads_release_and_terms(ontology) -> None:
    assert ontology.release == "test/release-1"
    assert "GO:0003824" in ontology.terms
    assert "GO:0006468" in ontology.terms


def test_parse_keeps_fields_declared_after_id(ontology) -> None:
    """Regression: a stanza frozen too early loses parents, subset and alt_id."""
    term = ontology.terms["GO:0016787"]
    assert term.parents == ("GO:0003824",)
    assert "goslim_generic" in term.subsets
    assert ontology.terms["GO:0008888"].alt_ids == ("GO:0007777",)
    assert ontology.terms["GO:0009999"].obsolete is True


def test_parse_only_is_a_and_part_of_are_parents(ontology) -> None:
    parents = ontology.terms["GO:0004553"].parents
    assert parents == ("GO:0016787",)  # part_of kept, regulates dropped


def test_parse_builds_alt_id_index(ontology) -> None:
    assert ontology.resolve("GO:0007777") == "GO:0008888"
    assert ontology.resolve("GO:0003824") == "GO:0003824"


def test_parse_rejects_missing_release(tmp_path: Path) -> None:
    path = tmp_path / "no_release.obo"
    path.write_text("[Term]\nid: GO:0000001\nname: x\n", encoding="utf-8")
    with pytest.raises(GoSlimError, match="data-version"):
        parse_obo(path)


def test_parse_rejects_file_without_terms(tmp_path: Path) -> None:
    path = tmp_path / "empty.obo"
    path.write_text("format-version: 1.2\ndata-version: test/x\n", encoding="utf-8")
    with pytest.raises(GoSlimError, match="no \\[Term\\]"):
        parse_obo(path)


def test_parse_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_obo(tmp_path / "absent.obo")


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "payload.bin"
    path.write_bytes(b"fasta2go")
    assert sha256_file(path) == hashlib.sha256(b"fasta2go").hexdigest()


def test_verify_integrity_accepts_a_matching_hash(obo_path: Path) -> None:
    assert verify_integrity(obo_path, sha256_file(obo_path)) == sha256_file(obo_path)


def test_verify_integrity_fails_closed_on_mismatch(obo_path: Path) -> None:
    with pytest.raises(GoSlimError, match="mismatch"):
        verify_integrity(obo_path, "0" * 64)


def test_verify_integrity_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_integrity(tmp_path / "absent.obo", "0" * 64)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def test_vocabulary_keeps_only_molecular_function(vocabulary) -> None:
    # GO:0006468 (BP) and GO:0005634 (CC) are both declared in the slim file.
    assert vocabulary.skipped_non_mf == ("GO:0005634", "GO:0006468")
    assert "GO:0005634" not in vocabulary.terms
    assert "GO:0006468" not in vocabulary.terms


def test_vocabulary_excludes_the_aspect_root(vocabulary) -> None:
    assert ASPECT_ROOT not in vocabulary.terms
    assert vocabulary.removed == (ASPECT_ROOT,)


def test_vocabulary_records_extension_origin(vocabulary) -> None:
    assert "GO:0005488" in vocabulary.terms
    assert vocabulary.origin["GO:0005488"] == "project_extension"
    assert vocabulary.origin["GO:0003824"] == "goslim_generic"
    assert len(vocabulary) == 6


def test_vocabulary_name_names_the_extension(vocabulary) -> None:
    assert vocabulary.name == "goslim_generic+GO:0005488"


def test_vocabulary_rejects_unknown_term(ontology) -> None:
    with pytest.raises(GoSlimError, match="absent from the ontology"):
        build_vocabulary(ontology, ["GO:0000001"])


def test_vocabulary_rejects_non_mf_extension(ontology) -> None:
    with pytest.raises(GoSlimError, match="namespace"):
        build_vocabulary(
            ontology, read_term_ids_from(ontology), extension=("GO:0005634",)
        )


def test_vocabulary_rejects_an_empty_result(ontology) -> None:
    with pytest.raises(GoSlimError, match="empty"):
        build_vocabulary(ontology, [], extension=())


def test_slim_equivalence_detects_a_mismatch(ontology) -> None:
    ids = read_term_ids_from(ontology)
    assert check_slim_equivalence(ontology, ids)["equivalent"] is True

    report = check_slim_equivalence(ontology, ids - {"GO:0003824"})
    assert report["equivalent"] is False
    assert report["only_in_ontology"] == ["GO:0003824"]
    assert report["only_in_slim_file"] == []


# ---------------------------------------------------------------------------
# Propagation and classification
# ---------------------------------------------------------------------------


def test_ancestors_include_every_level(ontology, vocabulary) -> None:
    ancestors = build_ancestor_map(ontology, vocabulary)
    assert ancestors["GO:0004672"] == frozenset(
        {"GO:0003824", "GO:0016740", "GO:0004672"}
    )


def test_part_of_edges_are_traversed(ontology, vocabulary) -> None:
    ancestors = build_ancestor_map(ontology, vocabulary)
    assert ancestors["GO:0004553"] == frozenset(
        {"GO:0003824", "GO:0016787", "GO:0004553"}
    )


def test_extension_receives_its_descendants(ontology, vocabulary) -> None:
    ancestors = build_ancestor_map(ontology, vocabulary)
    assert ancestors["GO:0005524"] == frozenset({"GO:0005488"})


def test_classify_mapped_term_returns_every_label(ontology, vocabulary) -> None:
    ancestors = build_ancestor_map(ontology, vocabulary)
    outcome = classify_annotation("GO:0004672", ontology, ancestors, vocabulary)
    assert outcome.status == STATUS_MAPPED
    assert outcome.labels == ("GO:0003824", "GO:0004672", "GO:0016740")


def test_classify_resolves_alt_id_before_traversal(ontology, vocabulary) -> None:
    ancestors = build_ancestor_map(ontology, vocabulary)
    outcome = classify_annotation("GO:0007777", ontology, ancestors, vocabulary)
    assert outcome.primary_id == "GO:0008888"
    assert outcome.status == STATUS_MAPPED
    assert outcome.labels == ("GO:0003824", "GO:0016787")


def test_classify_obsolete_term(ontology, vocabulary) -> None:
    ancestors = build_ancestor_map(ontology, vocabulary)
    outcome = classify_annotation("GO:0009999", ontology, ancestors, vocabulary)
    assert outcome.status == STATUS_OBSOLETE
    assert outcome.labels == ()


def test_classify_unknown_term(ontology, vocabulary) -> None:
    ancestors = build_ancestor_map(ontology, vocabulary)
    outcome = classify_annotation("GO:9999999", ontology, ancestors, vocabulary)
    assert outcome.status == STATUS_UNKNOWN
    assert outcome.primary_id is None


def test_classify_root_annotation_has_no_vocabulary_ancestor(
    ontology, vocabulary
) -> None:
    ancestors = build_ancestor_map(ontology, vocabulary)
    outcome = classify_annotation(ASPECT_ROOT, ontology, ancestors, vocabulary)
    assert outcome.status == STATUS_NO_ANCESTOR


def test_classify_other_aspect_has_no_vocabulary_ancestor(ontology, vocabulary) -> None:
    ancestors = build_ancestor_map(ontology, vocabulary)
    outcome = classify_annotation("GO:0005634", ontology, ancestors, vocabulary)
    assert outcome.status == STATUS_NO_ANCESTOR


# ---------------------------------------------------------------------------
# Propagation audit: namespace guard, obsolete terms, substitutions, alt_ids
# ---------------------------------------------------------------------------


def test_every_label_is_a_vocabulary_member_and_mf(ontology, vocabulary) -> None:
    """The namespace guarantee comes from the vocabulary, not from the traversal."""
    ancestors = build_ancestor_map(ontology, vocabulary)
    for term_id, term in ontology.terms.items():
        if term.obsolete:
            continue
        outcome = classify_annotation(term_id, ontology, ancestors, vocabulary)
        assert set(outcome.labels) <= vocabulary.term_set
        for label in outcome.labels:
            assert ontology.terms[label].namespace == MF_NAMESPACE
            assert label != ASPECT_ROOT


def test_ancestor_map_never_contains_non_vocabulary_terms(
    ontology, vocabulary
) -> None:
    ancestors = build_ancestor_map(ontology, vocabulary)
    for labels in ancestors.values():
        assert labels <= vocabulary.term_set
        assert ASPECT_ROOT not in labels


def test_bp_parent_is_traversed_but_never_labelled(ontology, vocabulary) -> None:
    """protein kinase activity is part_of a BP term that is itself in the slim."""
    ancestors = build_ancestor_map(ontology, vocabulary)
    assert "GO:0006468" in ontology.terms["GO:0004672"].parents  # traversed
    assert "GO:0006468" not in vocabulary.terms  # but skipped: BP
    outcome = classify_annotation("GO:0004672", ontology, ancestors, vocabulary)
    assert "GO:0006468" not in outcome.labels
    assert outcome.labels == ("GO:0003824", "GO:0004672", "GO:0016740")


def test_obsolete_ancestor_contributes_no_label(ontology, vocabulary) -> None:
    ancestors = build_ancestor_map(ontology, vocabulary)
    outcome = classify_annotation("GO:0007001", ontology, ancestors, vocabulary)
    assert "GO:0007000" not in outcome.labels
    assert outcome.labels == ("GO:0003824", "GO:0016787")


def test_obsolete_annotation_is_not_silently_replaced(ontology, vocabulary) -> None:
    """``replaced_by`` exists in GO, but following it would invent a label."""
    ancestors = build_ancestor_map(ontology, vocabulary)
    outcome = classify_annotation("GO:0009999", ontology, ancestors, vocabulary)
    assert outcome.status == STATUS_OBSOLETE
    assert outcome.labels == ()
    assert "GO:0016787" in vocabulary.terms  # the replacement is a valid label...
    # ...but it must not be applied to the obsolete annotation.


def test_alt_id_of_an_obsolete_term_is_obsolete(ontology, vocabulary) -> None:
    ancestors = build_ancestor_map(ontology, vocabulary)
    outcome = classify_annotation("GO:0009998", ontology, ancestors, vocabulary)
    assert outcome.status == STATUS_OBSOLETE
    assert outcome.primary_id == "GO:0009999"


def test_alt_id_and_primary_produce_identical_labels(ontology, vocabulary) -> None:
    ancestors = build_ancestor_map(ontology, vocabulary)
    primary = classify_annotation("GO:0008888", ontology, ancestors, vocabulary)
    alias = classify_annotation("GO:0007777", ontology, ancestors, vocabulary)
    assert primary.labels == alias.labels
    assert primary.primary_id == alias.primary_id == "GO:0008888"


# ---------------------------------------------------------------------------
# Cross-aspect paths (MF -> BP -> MF)
# ---------------------------------------------------------------------------


def test_cross_aspect_path_does_not_add_an_mf_label(cross_ontology) -> None:
    """The BP node in the middle must not conduct a label back into MF."""
    ontology, vocabulary = cross_ontology
    ancestors = build_ancestor_map(ontology, vocabulary)

    assert ancestors["GO:0004672"] == frozenset({"GO:0003824", "GO:0004672"})
    assert "GO:0008000" not in ancestors["GO:0004672"]


def test_without_the_guard_the_cross_aspect_path_would_leak(cross_ontology) -> None:
    """Documents why the guard exists; flips the default only here."""
    ontology, vocabulary = cross_ontology
    unguarded = build_ancestor_map(ontology, vocabulary, restrict_to_aspect=False)

    assert "GO:0008000" in unguarded["GO:0004672"]


def test_annotation_to_another_aspect_cannot_yield_mf_labels(cross_ontology) -> None:
    ontology, vocabulary = cross_ontology
    guarded = build_ancestor_map(ontology, vocabulary)

    outcome = classify_annotation("GO:0006468", ontology, guarded, vocabulary)
    assert outcome.status == STATUS_NO_ANCESTOR
    assert outcome.labels == ()

    unguarded = build_ancestor_map(ontology, vocabulary, restrict_to_aspect=False)
    leaked = classify_annotation("GO:0006468", ontology, unguarded, vocabulary)
    assert leaked.labels == ("GO:0008000",)


def test_guard_keeps_intra_aspect_ancestry(cross_ontology) -> None:
    ontology, vocabulary = cross_ontology
    ancestors = build_ancestor_map(ontology, vocabulary)
    assert ancestors["GO:0003824"] == frozenset({"GO:0003824"})
    assert ancestors["GO:0005488"] == frozenset({"GO:0005488"})
    assert ancestors["GO:0004672"] == frozenset({"GO:0003824", "GO:0004672"})


# ---------------------------------------------------------------------------
# Protein mapping
# ---------------------------------------------------------------------------


def _mapped(ontology, vocabulary, rows):
    ancestors = build_ancestor_map(ontology, vocabulary)
    return map_proteins(rows, ontology, vocabulary, ancestors)


def test_map_dedups_repeated_terms_and_labels(ontology, vocabulary) -> None:
    mappings, summary = _mapped(
        ontology,
        vocabulary,
        [("P1", ["GO:0016787", "GO:0016787", "GO:0004553"])],
    )
    mapping = mappings[0]
    assert mapping.raw_terms == ("GO:0004553", "GO:0016787")
    assert mapping.labels == ("GO:0003824", "GO:0004553", "GO:0016787")
    assert summary.annotations == 2


def test_map_partitions_every_annotation(ontology, vocabulary) -> None:
    mappings, summary = _mapped(
        ontology,
        vocabulary,
        [
            ("P1", ["GO:0016787", "GO:0004553"]),
            ("P2", ["GO:0005524", "GO:0009999", "GO:9999999", ASPECT_ROOT]),
            ("P3", [ASPECT_ROOT]),
        ],
    )
    assert [mapping.protein_id for mapping in mappings] == ["P1", "P2", "P3"]
    assert summary.annotations == 7
    assert summary.mapped == 3
    assert summary.obsolete == 1
    assert summary.unknown == 1
    assert summary.no_vocabulary_ancestor == 2
    assert (
        summary.mapped
        + summary.obsolete
        + summary.unknown
        + summary.no_vocabulary_ancestor
        == summary.annotations
    )


def test_map_counts_proteins_without_labels(ontology, vocabulary) -> None:
    _, summary = _mapped(ontology, vocabulary, [("P1", [ASPECT_ROOT]), ("P2", ["GO:0005524"])])
    assert summary.proteins_without_labels == 1
    assert summary.proteins_with_labels == 1


def test_summary_reports_all_vocabulary_labels(ontology, vocabulary) -> None:
    _, summary = _mapped(ontology, vocabulary, [("P1", ["GO:0005524"])])
    assert summary.vocabulary_terms == 6
    assert set(summary.label_counts) == set(vocabulary.terms)
    assert summary.label_counts["GO:0005488"] == 1
    assert summary.label_counts["GO:0003824"] == 0
    assert summary.labels_used == 1


def test_protein_records_its_statuses(ontology, vocabulary) -> None:
    mappings, _ = _mapped(ontology, vocabulary, [("P1", ["GO:0009999", "GO:9999999"])])
    assert mappings[0].statuses == (STATUS_OBSOLETE, STATUS_UNKNOWN)
    assert mappings[0].counts() == {
        STATUS_MAPPED: 0,
        STATUS_NO_ANCESTOR: 0,
        STATUS_OBSOLETE: 1,
        STATUS_UNKNOWN: 1,
    }


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def test_labels_csv_has_stable_columns(tmp_path: Path, ontology, vocabulary) -> None:
    mappings, _ = _mapped(ontology, vocabulary, [("P1", ["GO:0004553"])])
    path = write_labels_csv(tmp_path / "labels.csv", mappings)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == list(LABEL_COLUMNS)
    assert lines[1] == (
        "P1,3,GO:0003824;GO:0004553;GO:0016787,1,1,0,0,0"
    )


def test_vocabulary_csv_marks_the_extension(
    tmp_path: Path, ontology, vocabulary
) -> None:
    ancestors = build_ancestor_map(ontology, vocabulary)
    path = write_vocabulary_csv(
        tmp_path / "vocab.csv", vocabulary, ontology, ancestors
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == list(VOCABULARY_COLUMNS)
    rows = {line.split(",")[0]: line.split(",") for line in lines[1:]}
    assert rows["GO:0005488"][2] == "project_extension"
    assert rows["GO:0005488"][3] == "false"
    assert rows["GO:0016787"][2] == "goslim_generic"
    assert rows["GO:0016787"][3] == "true"
    assert rows["GO:0016787"][4] == "GO:0003824"


def test_json_output_is_deterministic(tmp_path: Path) -> None:
    first = write_json(tmp_path / "a.json", {"b": 1, "a": [2, 3]})
    second = write_json(tmp_path / "b.json", {"a": [2, 3], "b": 1})
    assert first.read_bytes() == second.read_bytes()


def test_mapping_output_is_deterministic(
    tmp_path: Path, ontology, vocabulary
) -> None:
    rows = [("P2", ["GO:0005524"]), ("P1", ["GO:0004553"])]
    first, _ = _mapped(ontology, vocabulary, rows)
    second, _ = _mapped(ontology, vocabulary, list(reversed(rows)))
    a = write_labels_csv(tmp_path / "a.csv", first).read_bytes()
    b = write_labels_csv(tmp_path / "b.csv", second).read_bytes()
    assert a == b


def test_summarize_is_reproducible(ontology, vocabulary) -> None:
    mappings, first = _mapped(ontology, vocabulary, [("P1", ["GO:0004553"])])
    second = summarize(mappings, vocabulary)
    assert first.as_dict() == second.as_dict()


# ---------------------------------------------------------------------------
# Integration against the pinned sources (skipped without the local cache)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
CACHE_DIR = REPO_ROOT / "cache" / "go_slim"
REAL_ONTOLOGY = CACHE_DIR / "go-basic.obo"
REAL_SLIM = CACHE_DIR / "goslim_generic.obo"
REAL_SNAPSHOT = (
    REPO_ROOT / "cache" / "uniprot" / "bulk" / "uniprot_sprot.tsv.gz"
)
SOURCES_JSON = REPO_ROOT / "data" / "go_slim" / "sources.json"
EXAMPLES_JSON = REPO_ROOT / "data" / "go_slim" / "expected_examples.json"

pinned_sources_available = pytest.mark.skipif(
    not (REAL_ONTOLOGY.exists() and REAL_SLIM.exists() and REAL_SNAPSHOT.exists()),
    reason="pinned GO sources or UniProt snapshot not cached",
)


@pytest.fixture(scope="module")
def real_sources():
    sources = json.loads(SOURCES_JSON.read_text(encoding="utf-8"))
    verify_integrity(REAL_ONTOLOGY, sources["ontology"]["sha256"])
    verify_integrity(REAL_SLIM, sources["slim"]["sha256"])
    ontology = parse_obo(REAL_ONTOLOGY)
    vocabulary = build_vocabulary(
        ontology,
        read_term_ids(REAL_SLIM),
        extension=tuple(sources["vocabulary"]["extension"]),
        excluded=tuple(sources["vocabulary"]["excluded"]),
        subset=sources["slim"]["subset"],
    )
    return sources, ontology, vocabulary, build_ancestor_map(ontology, vocabulary)


@pinned_sources_available
def test_pinned_sources_build_the_forty_one_term_vocabulary(real_sources) -> None:
    sources, ontology, vocabulary, _ = real_sources
    assert len(vocabulary) == sources["vocabulary"]["expected_terms"] == 41
    assert ASPECT_ROOT not in vocabulary.terms
    assert vocabulary.origin["GO:0005488"] == "project_extension"
    assert "GO:0016491" in vocabulary.terms  # oxidoreductase from the official slim

    report = check_slim_equivalence(
        ontology, read_term_ids(REAL_SLIM), sources["slim"]["subset"]
    )
    assert report["equivalent"] is True
    assert report["declared_in_slim_file"] == report["tagged_in_ontology"] == 140


@pinned_sources_available
def test_documented_examples_reproduce_their_labels(real_sources) -> None:
    from scripts.build_go_slim_mapping import check_examples

    _, ontology, vocabulary, ancestors = real_sources
    results = check_examples(
        EXAMPLES_JSON, REAL_SNAPSHOT, ontology, vocabulary, ancestors
    )
    assert len(results) == 3
    assert all(item["matched"] for item in results)


@pinned_sources_available
def test_real_guard_is_a_no_op_on_the_pinned_release(real_sources) -> None:
    """On go-basic.obo the aspect guard changes no label for any term."""
    _, ontology, vocabulary, _ = real_sources
    guarded = build_ancestor_map(ontology, vocabulary)
    unguarded = build_ancestor_map(ontology, vocabulary, restrict_to_aspect=False)
    assert guarded == unguarded

    cross = 0
    for term in ontology.terms.values():
        if term.obsolete:
            continue
        for parent in term.parents:
            other = ontology.terms.get(parent)
            if other is not None and not other.obsolete:
                if other.namespace != term.namespace:
                    cross += 1
    assert cross == 0


@pinned_sources_available
def test_real_labels_are_always_mf_vocabulary_members(real_sources) -> None:
    from scripts.build_go_slim_mapping import read_annotations

    _, ontology, vocabulary, ancestors = real_sources
    rows = read_annotations(REAL_SNAPSHOT, limit=100)
    mappings, summary = map_proteins(rows, ontology, vocabulary, ancestors)

    assert len(mappings) == 100
    for mapping in mappings:
        assert set(mapping.labels) <= vocabulary.term_set
        for label in mapping.labels:
            assert ontology.terms[label].namespace == MF_NAMESPACE
    assert (
        summary.mapped
        + summary.no_vocabulary_ancestor
        + summary.obsolete
        + summary.unknown
        == summary.annotations
    )

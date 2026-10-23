"""Tests for the consolidated correlation module and its matrices."""

from pathlib import Path

import numpy as np
import pytest

from protein_database.db import connect, init_database
from protein_database.schema import COMPOSITION_COLUMNS, FEATURE_COLUMNS
from scripts.correlations import (
    FEATURE_MATRIX,
    amino_acid_chunks,
    cooccurrence_matrix,
    correlation_matrix,
    correlation_matrix_chunks,
    feature_stats,
    feature_term_chunks,
    feature_vs_terms,
    is_symmetric,
    jaccard_matrix,
    load_matrix,
    phi_matrix,
    plot_matrix,
    refresh_feature_matrix,
    scale_for,
    sequence_feature_matrix,
    term_matrix,
    title_for,
    top_pairs,
    write_matrix_csv,
)

#: 6 proteins x 4 terms; the last term is present in every protein.
PRESENCE = np.array(
    [
        [1, 1, 0, 1],
        [1, 1, 0, 1],
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ],
    dtype=bool,
)


# ---------------------------------------------------------------------------
# Helpers: tiny in-memory databases
# ---------------------------------------------------------------------------


def _write_matrix(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _keyword_db(
    rows: list[tuple[float, ...]],
    words: list[str],
    phrases: list[str] | None = None,
    composition: list[float] | None = None,
):
    """In-memory DB with one protein, feature row and term row per entry."""
    conn = connect(":memory:")
    init_database(conn)
    features = ", ".join(FEATURE_COLUMNS)
    placeholders = ", ".join("?" for _ in FEATURE_COLUMNS)
    phrase_values = phrases or [""] * len(rows)
    for index, values in enumerate(rows):
        fasta_id = f"p{index}"
        conn.execute(
            "INSERT INTO proteins VALUES (?, 'test', ?, 'AAAA', 4, "
            "CURRENT_TIMESTAMP)",
            [fasta_id, f"protein {index}"],
        )
        conn.execute(
            f"INSERT INTO sequence_features (fasta_id, source_file, {features}) "
            f"VALUES (?, 'test', {placeholders})",
            [fasta_id, *values],
        )
        conn.execute(
            "INSERT INTO protein_keyword_terms VALUES (?, 'test', ?, ?, NULL)",
            [fasta_id, words[index], phrase_values[index]],
        )
        if composition is not None:
            columns = ", ".join(COMPOSITION_COLUMNS)
            slots = ", ".join("?" for _ in COMPOSITION_COLUMNS)
            conn.execute(
                f"INSERT INTO amino_acid_composition (fasta_id, source_file, "
                f"{columns}) VALUES (?, 'test', {slots})",
                [fasta_id, *([composition[index]] * len(COMPOSITION_COLUMNS))],
            )
    return conn


def _feature_rows(count: int) -> list[tuple[float, ...]]:
    return [
        tuple([float(index)] * len(FEATURE_COLUMNS))
        for index in range(1, count + 1)
    ]


# ---------------------------------------------------------------------------
# 1. Term x term math
# ---------------------------------------------------------------------------


def test_cooccurrence_counts_pairwise_presence() -> None:
    co = cooccurrence_matrix(PRESENCE)
    assert co[0, 0] == 3  # A in 3 proteins
    assert co[0, 1] == 2  # A and B together in 2
    assert co[0, 2] == 0  # A and C never together
    assert co[0, 3] == 3  # D is everywhere A is
    assert np.array_equal(co, co.T)


def test_phi_matches_hand_computed_values() -> None:
    phi = phi_matrix(PRESENCE)
    assert phi[0, 1] == np.float64(6 / np.sqrt(72))
    assert phi[0, 2] == np.float64(-6 / np.sqrt(72))
    assert phi[1, 2] == np.float64(-0.5)
    assert phi[2, 1] == phi[1, 2]


def test_phi_diagonal_is_one_and_symmetric() -> None:
    phi = phi_matrix(PRESENCE)
    assert np.allclose(np.diag(phi), 1.0)
    assert np.allclose(phi, phi.T)


def test_phi_is_zero_for_zero_variance_terms() -> None:
    # Term 3 is present in every protein: phi is undefined, reported as 0.
    phi = phi_matrix(PRESENCE)
    assert phi[0, 3] == 0.0
    assert phi[3, 3] == 1.0  # diagonal is always 1


def test_jaccard_matches_hand_computed_values() -> None:
    jaccard = jaccard_matrix(PRESENCE)
    assert jaccard[0, 1] == np.float64(2 / 3)
    assert jaccard[0, 2] == 0.0
    assert jaccard[0, 3] == np.float64(3 / 6)
    assert np.allclose(jaccard, jaccard.T)


def test_jaccard_diagonal_is_one() -> None:
    jaccard = jaccard_matrix(PRESENCE)
    assert np.allclose(np.diag(jaccard), 1.0)


def test_jaccard_is_zero_when_both_terms_absent() -> None:
    empty = np.zeros((4, 2), dtype=bool)
    jaccard = jaccard_matrix(empty)
    assert np.allclose(jaccard, 0.0)


def test_top_pairs_orders_positive_and_negative() -> None:
    terms = ["a", "b", "c"]
    values = np.array(
        [
            [1.0, 0.6, -0.4],
            [0.6, 1.0, 0.1],
            [-0.4, 0.1, 1.0],
        ]
    )
    best, worst = top_pairs(terms, values, positive=2, negative=1)
    assert best[0] == ("a", "b", 0.6)
    assert worst[0] == ("a", "c", -0.4)


# ---------------------------------------------------------------------------
# 2. Numeric x binary math
# ---------------------------------------------------------------------------


def test_perfect_positive_and_negative_correlation() -> None:
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([[3.0], [5.0], [7.0], [9.0]])  # y = 2x + 1
    assert np.isclose(correlation_matrix(x, y)[0, 0], 1.0)

    assert np.isclose(correlation_matrix(x, -y)[0, 0], -1.0)


def test_matches_numpy_corrcoef() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(200, 4))
    y = rng.integers(0, 2, size=(200, 3)).astype(float)
    expected = np.corrcoef(x.T, y.T)[:4, 4:]
    assert np.allclose(correlation_matrix(x, y), expected, atol=1e-12)


def test_chunked_accumulation_equals_single_pass() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(101, 5))
    y = rng.integers(0, 2, size=(101, 6)).astype(float)
    chunks = [(x[:30], y[:30]), (x[30:70], y[30:70]), (x[70:], y[70:])]
    assert np.allclose(
        correlation_matrix_chunks(chunks), correlation_matrix(x, y)
    )


def test_constant_column_correlates_as_zero() -> None:
    x = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])
    y = np.array([[0.0], [0.0], [1.0], [1.0]])
    r = correlation_matrix(x, y)
    assert np.isfinite(r).all()
    assert r[0, 0] > 0.5
    assert r[1, 0] == 0.0  # constant amino-acid column


def test_binary_side_behaves_like_point_biserial() -> None:
    # Two proteins with the term, two without; x values 0,1,10,11.
    x = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array([[1.0], [1.0], [0.0], [0.0]])
    r = correlation_matrix(x, y)[0, 0]
    assert np.isclose(r, -0.9950371902099892)


# ---------------------------------------------------------------------------
# 3./4. Data access and matrix builders
# ---------------------------------------------------------------------------


def test_feature_stats_mean_and_population_std() -> None:
    conn = _keyword_db(_feature_rows(3), ["a", "b", "c"])
    mean, std = feature_stats(conn)
    conn.close()

    assert np.allclose(mean, 2.0)
    assert np.allclose(std, np.sqrt(2.0 / 3.0))


def test_feature_stats_zero_variance_gets_unit_std() -> None:
    conn = _keyword_db([tuple([4.0] * len(FEATURE_COLUMNS))] * 3, ["a", "b", "c"])
    mean, std = feature_stats(conn)
    conn.close()

    assert np.allclose(mean, 4.0)
    assert np.allclose(std, 1.0)


def test_feature_term_chunks_align_features_and_presence() -> None:
    conn = _keyword_db(
        _feature_rows(4),
        ["kinase atp", "kinase", "atp", ""],
    )
    blocks = list(feature_term_chunks(conn, "words", ["kinase", "atp"], 2))
    conn.close()

    assert len(blocks) == 2  # streamed in two batches
    features = np.vstack([block[0] for block in blocks])
    presence = np.vstack([block[1] for block in blocks])

    assert features.shape == (4, len(FEATURE_COLUMNS))
    assert np.allclose(features[:, 0], [1.0, 2.0, 3.0, 4.0])
    assert presence.tolist() == [[1, 1], [1, 0], [0, 1], [0, 0]]


def test_amino_acid_chunks_align_percentages_and_presence() -> None:
    conn = _keyword_db(
        _feature_rows(3),
        ["kinase", "", "kinase atp"],
        composition=[1.0, 2.0, 3.0],
    )
    blocks = list(amino_acid_chunks(conn, "words", ["kinase", "atp"], 10))
    conn.close()

    amino_acids, presence = blocks[0]
    assert amino_acids.shape == (3, len(COMPOSITION_COLUMNS))
    assert np.allclose(amino_acids[:, 0], [1.0, 2.0, 3.0])
    assert presence.tolist() == [[1, 0], [0, 0], [1, 1]]


def test_feature_vs_terms_shape_and_batching() -> None:
    conn = _keyword_db(
        _feature_rows(5),
        ["kinase atp", "kinase", "atp", "kinase atp", ""],
    )
    chunked = feature_vs_terms(conn, "words", ["kinase", "atp"], batch_size=2)
    single = feature_vs_terms(conn, "words", ["kinase", "atp"], batch_size=1000)
    conn.close()

    assert chunked.shape == (len(FEATURE_COLUMNS), 2)
    assert np.allclose(chunked, single)


def test_feature_vs_terms_scaling_is_pearson_invariant() -> None:
    conn = _keyword_db(
        _feature_rows(5),
        ["kinase atp", "kinase", "atp", "kinase atp", ""],
    )
    scaled = feature_vs_terms(conn, "words", ["kinase", "atp"], scaled=True)
    raw = feature_vs_terms(conn, "words", ["kinase", "atp"], scaled=False)
    conn.close()

    # z-scoring is affine, so the coefficients are unchanged.
    assert np.allclose(scaled, raw, atol=1e-12)


def test_feature_vs_terms_rejects_empty_table() -> None:
    conn = connect(":memory:")
    init_database(conn)
    with pytest.raises(ValueError, match="contains no rows"):
        feature_vs_terms(conn, "words", ["kinase"])
    conn.close()


def test_feature_vs_terms_requires_terms() -> None:
    conn = _keyword_db(_feature_rows(2), ["kinase", "atp"])
    with pytest.raises(ValueError, match="At least one term"):
        feature_vs_terms(conn, "words", [])
    conn.close()


def test_term_matrix_selects_phi_or_jaccard() -> None:
    conn = _keyword_db(_feature_rows(4), ["a b", "a", "b", ""])
    phi = term_matrix(conn, "words", ["a", "b"], "phi")
    jaccard = term_matrix(conn, "words", ["a", "b"], "jaccard")
    with pytest.raises(ValueError, match="Unknown metric"):
        term_matrix(conn, "words", ["a"], "cosine")
    conn.close()

    assert np.allclose(np.diag(phi), 1.0)
    assert np.allclose(phi, phi.T)
    assert np.allclose(np.diag(jaccard), 1.0)


# ---------------------------------------------------------------------------
# 5./6. Reporting and rendering
# ---------------------------------------------------------------------------


def test_load_matrix_parses_labels_and_values(tmp_path: Path) -> None:
    path = _write_matrix(
        tmp_path / "sample.csv",
        "term,a,b\n"
        "a,1.0,0.5\n"
        "b,0.5,1.0\n",
    )
    rows, columns, values = load_matrix(path)
    assert rows == ["a", "b"]
    assert columns == ["a", "b"]
    assert np.array_equal(values, np.array([[1.0, 0.5], [0.5, 1.0]]))


def test_is_symmetric_true_for_correlation_matrix() -> None:
    values = np.array([[1.0, -0.2], [-0.2, 1.0]])
    assert is_symmetric(values) is True


def test_is_symmetric_false_for_rectangular_or_asymmetric() -> None:
    assert is_symmetric(np.zeros((2, 3))) is False
    assert is_symmetric(np.array([[1.0, 0.2], [0.1, 1.0]])) is False


def test_scale_for_signed_correlation_is_centred_and_symmetric() -> None:
    values = np.array([[1.0, -0.4], [-0.4, 1.0]])
    style = scale_for("words_phi", values)
    assert style["center"] == 0.0
    assert style["vmin"] == -style["vmax"]


def test_scale_for_jaccard_is_zero_to_max() -> None:
    values = np.array([[1.0, 0.3], [0.3, 1.0]])
    style = scale_for("words_jaccard", values)
    assert style["vmin"] == 0.0
    assert style["vmax"] == 1.0


def test_plot_matrix_writes_png(tmp_path: Path) -> None:
    path = _write_matrix(
        tmp_path / "words_phi.csv",
        "term,a,b,c\n"
        "a,1.0,0.5,-0.2\n"
        "b,0.5,1.0,0.1\n"
        "c,-0.2,0.1,1.0\n",
    )
    out = plot_matrix(path, tmp_path / "plots", dpi=72)
    assert out.exists()
    assert out.stat().st_size > 0


def test_sequence_feature_matrix_is_a_correlation_matrix() -> None:
    conn = _keyword_db(
        [tuple([1.0] * 15), tuple([2.0] * 15), tuple([3.0] * 15)],
        ["a", "b", "c"],
    )
    values = sequence_feature_matrix(conn)
    conn.close()

    assert values.shape == (15, 15)
    assert np.allclose(values, 1.0)


def test_sequence_feature_matrix_uses_streamed_batches() -> None:
    conn = _keyword_db(_feature_rows(5), ["a", "b", "c", "d", "e"])
    chunked = sequence_feature_matrix(conn, batch_size=2)
    single = sequence_feature_matrix(conn, batch_size=1000)
    conn.close()

    assert np.allclose(chunked, single)


def test_sequence_feature_matrix_rejects_empty_table() -> None:
    conn = connect(":memory:")
    init_database(conn)
    with pytest.raises(ValueError, match="contains no rows"):
        sequence_feature_matrix(conn)
    conn.close()


def test_write_matrix_csv_round_trips_through_load_matrix(tmp_path: Path) -> None:
    rows = ["a", "b"]
    values = np.array([[1.0, -0.25], [-0.25, 1.0]])
    path = write_matrix_csv(tmp_path / "matrix.csv", rows, rows, values)

    loaded_rows, loaded_columns, loaded_values = load_matrix(path)
    assert loaded_rows == rows
    assert loaded_columns == rows
    assert np.allclose(loaded_values, values)


def test_write_matrix_csv_uses_custom_row_label(tmp_path: Path) -> None:
    values = np.zeros((2, 2))
    path = write_matrix_csv(
        tmp_path / "features.csv", ["a", "b"], ["c", "d"], values,
        row_label="feature",
    )
    assert path.read_text(encoding="utf-8").splitlines()[0].startswith("feature,")


def test_title_for_sequence_feature_matrix() -> None:
    assert title_for(FEATURE_MATRIX) == "Sequence features — Pearson correlation"


def test_title_for_feature_term_matrices() -> None:
    assert (
        title_for("features_vs_words")
        == "Sequence features vs words — Pearson r (z-scored)"
    )
    assert (
        title_for("features_vs_phrases_raw")
        == "Sequence features vs phrases — Pearson r (raw)"
    )


def test_refresh_feature_matrix_writes_labelled_csv(tmp_path: Path) -> None:
    db_path = tmp_path / "proteins.duckdb"
    conn = connect(db_path)
    init_database(conn)
    for index, value in enumerate([1.0, 2.0, 3.0]):
        conn.execute(
            "INSERT INTO proteins VALUES (?, 'test', ?, 'AAAA', 4, "
            "CURRENT_TIMESTAMP)",
            [f"p{index}", f"protein {index}"],
        )
        conn.execute(
            "INSERT INTO sequence_features (fasta_id, source_file, "
            f"{', '.join(FEATURE_COLUMNS)}) VALUES (?, 'test', "
            f"{', '.join(['?'] * len(FEATURE_COLUMNS))})",
            [f"p{index}", *([value] * len(FEATURE_COLUMNS))],
        )
    conn.close()

    matrix_path = tmp_path / "sequence_features_correlation.csv"
    assert refresh_feature_matrix(db_path, matrix_path) == matrix_path

    rows, columns, values = load_matrix(matrix_path)
    assert rows == FEATURE_COLUMNS
    assert columns == FEATURE_COLUMNS
    assert np.allclose(values, 1.0)


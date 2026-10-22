"""Tests for the amino-acid vs term correlation math."""

import numpy as np

from scripts.amino_acid_correlations import (
    correlation_matrix,
    correlation_matrix_chunks,
)


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

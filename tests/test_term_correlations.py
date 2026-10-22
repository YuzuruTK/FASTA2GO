"""Tests for the term correlation math."""

import numpy as np

from scripts.term_correlations import (
    cooccurrence_matrix,
    jaccard_matrix,
    phi_matrix,
    top_pairs,
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

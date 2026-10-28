"""Tests for the group-aware split guards (Spike S0 findings F1/F2/F4)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from protein_database.ml.splits import (
    assert_group_disjoint,
    reject_non_group_cv,
    validate_group_splits,
)

#: 24 samples, 6 groups of 4 consecutive rows.
GROUPS = np.repeat(np.arange(6), 4)


def test_assert_group_disjoint_accepts_disjoint_sets() -> None:
    assert_group_disjoint(np.array([1, 2]), np.array([3, 4]))


def test_assert_group_disjoint_accepts_an_empty_side() -> None:
    assert_group_disjoint(np.array([1]), np.array([], dtype=int))


def test_assert_group_disjoint_rejects_a_shared_group() -> None:
    with pytest.raises(ValueError, match="both sides"):
        assert_group_disjoint(np.array([1, 2]), np.array([2, 3]))


def test_assert_group_disjoint_rejects_two_dimensional_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        assert_group_disjoint(np.ones((2, 2)), np.array([3]))


@pytest.mark.parametrize("cv", [3, 0, None, "predefined"])
def test_reject_non_group_cv_explains_the_multilabel_trap(cv) -> None:
    with pytest.raises(ValueError, match="KFold"):
        reject_non_group_cv(cv)


def test_validate_group_splits_accepts_group_shuffle_split() -> None:
    X = np.zeros((GROUPS.size, 2))
    splits = list(
        GroupShuffleSplit(n_splits=2, test_size=0.25, random_state=0).split(
            X, None, GROUPS
        )
    )
    validated = validate_group_splits(splits, GROUPS, n_samples=GROUPS.size)
    assert len(validated) == 2
    for train, test in validated:
        assert np.intersect1d(GROUPS[train], GROUPS[test]).size == 0


def test_validate_group_splits_rejects_a_shared_group() -> None:
    # indices 0 and 1 both belong to group 0
    with pytest.raises(ValueError, match="group"):
        validate_group_splits([(np.array([0, 4, 8]), np.array([1, 12]))], GROUPS)


def test_validate_group_splits_rejects_overlapping_indices() -> None:
    with pytest.raises(ValueError, match="sample index"):
        validate_group_splits([(np.array([0, 1]), np.array([1, 8]))], GROUPS)


def test_validate_group_splits_rejects_a_cross_validator_object() -> None:
    with pytest.raises(ValueError, match="iterable of"):
        validate_group_splits(GroupKFold(n_splits=2), GROUPS)


def test_validate_group_splits_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_group_splits([], GROUPS)


def test_validate_group_splits_rejects_an_empty_side() -> None:
    with pytest.raises(ValueError, match="empty side"):
        validate_group_splits([(np.array([0, 1]), np.array([], dtype=int))], GROUPS)


def test_validate_group_splits_rejects_out_of_range_indices() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        validate_group_splits(
            [(np.array([0, 1]), np.array([99]))], GROUPS, n_samples=GROUPS.size
        )


def test_validate_group_splits_rejects_negative_indices() -> None:
    with pytest.raises(ValueError, match="negative"):
        validate_group_splits([(np.array([-1, 0]), np.array([8]))], GROUPS)

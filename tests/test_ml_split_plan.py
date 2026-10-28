"""Tests for the grouped split plan (external test + 3 inner folds)."""

from __future__ import annotations

import numpy as np
import pytest

from protein_database.ml.evaluation import (
    evaluable_label_mask,
    label_coverage_report,
)
from protein_database.ml.splits import (
    Fold,
    SplitPlan,
    make_group_split_plan,
    split_coverage,
    validate_split_plan,
)

#: 12 groups of 3 samples: enough for a 20% external block plus 3 inner folds.
GROUPS = np.repeat(np.arange(12), 3)
LABELS = ("GO:0003824", "GO:0005488", "GO:0003677")


def _plan(**kwargs) -> SplitPlan:
    return make_group_split_plan(GROUPS, **kwargs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_plan_has_three_folds() -> None:
    plan = _plan()
    assert plan.n_folds == 3
    assert [fold.index for fold in plan.folds] == [0, 1, 2]


def test_test_block_is_disjoint_from_every_fold() -> None:
    plan = _plan()
    for fold in plan.folds:
        assert np.intersect1d(plan.test, fold.train).size == 0
        assert np.intersect1d(plan.test, fold.validation).size == 0
        assert np.intersect1d(GROUPS[plan.test], GROUPS[fold.validation]).size == 0


def test_dev_covers_every_non_test_sample_exactly_once() -> None:
    plan = _plan()
    assert np.array_equal(plan.dev, np.setdiff1d(np.arange(GROUPS.size), plan.test))
    stacked = np.concatenate([fold.validation for fold in plan.folds])
    assert np.unique(stacked).size == stacked.size


def test_plan_is_reproducible_for_the_same_seed() -> None:
    first, second = _plan(seed=7), _plan(seed=7)
    assert np.array_equal(first.test, second.test)
    for left, right in zip(first.folds, second.folds):
        assert np.array_equal(left.train, right.train)
        assert np.array_equal(left.validation, right.validation)


def test_different_seeds_produce_a_different_test_block() -> None:
    first, second = _plan(seed=1), _plan(seed=2)
    assert not np.array_equal(first.test, second.test)


def test_validation_shares_no_group_with_training() -> None:
    plan = _plan()
    for fold in plan.folds:
        assert (
            np.intersect1d(GROUPS[fold.train], GROUPS[fold.validation]).size == 0
        )


def test_plan_rejects_too_few_groups() -> None:
    with pytest.raises(ValueError, match="at least 4 groups"):
        make_group_split_plan(np.repeat(np.arange(2), 4), n_folds=3)


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 2.0])
def test_plan_rejects_invalid_test_size(test_size: float) -> None:
    with pytest.raises(ValueError, match="test_size"):
        make_group_split_plan(GROUPS, test_size=test_size)


def test_plan_rejects_non_positive_folds() -> None:
    with pytest.raises(ValueError, match="n_folds"):
        make_group_split_plan(GROUPS, n_folds=0)


def test_plan_rejects_two_dimensional_groups() -> None:
    with pytest.raises(ValueError, match="1-D"):
        make_group_split_plan(np.zeros((4, 2)))


# ---------------------------------------------------------------------------
# Validation of a hand-built (bad) plan
# ---------------------------------------------------------------------------


def _manual_plan(test, folds, **kwargs) -> SplitPlan:
    return SplitPlan(
        test=np.asarray(test, dtype=int),
        folds=tuple(folds),
        seed=kwargs.get("seed", 0),
        test_size=kwargs.get("test_size", 0.2),
        n_groups_test=1,
        n_groups_dev=1,
    )


def test_validate_accepts_a_consistent_manual_plan() -> None:
    # test = group 0; dev (groups 1-11) split in two disjoint halves
    plan = _manual_plan(
        [0, 1, 2],
        [
            Fold(0, np.arange(18, 36), np.arange(3, 18)),
            Fold(1, np.arange(3, 18), np.arange(18, 36)),
        ],
    )
    validate_split_plan(plan, GROUPS)


def test_validate_rejects_a_fold_that_touches_the_test_block() -> None:
    plan = _manual_plan(
        [0, 1, 2],
        [Fold(0, np.array([9, 10, 11]), np.array([0, 3, 4, 5, 6, 7, 8]))],
    )
    with pytest.raises(ValueError, match="overlaps the test block"):
        validate_split_plan(plan, GROUPS)


def test_validate_rejects_a_shared_group_between_train_and_validation() -> None:
    # index 3 and 4 both belong to group 1
    plan = _manual_plan(
        [0, 1, 2],
        [Fold(0, np.array([3, 6, 7, 8, 9, 10, 11]), np.array([4, 5]))],
    )
    with pytest.raises(ValueError, match="both sides"):
        validate_split_plan(plan, GROUPS)


def test_validate_rejects_a_training_group_shared_with_the_test_block() -> None:
    # index 0 belongs to group 0, which is also the test block
    plan = _manual_plan(
        [0, 1, 2],
        [Fold(0, np.array([0, 3, 4, 5, 6, 7, 8, 9, 10, 11]), np.arange(12, 36))],
    )
    with pytest.raises(ValueError, match="train vs test"):
        validate_split_plan(plan, GROUPS)


def test_validate_rejects_duplicate_indices() -> None:
    plan = _manual_plan(
        [0, 1, 2],
        [Fold(0, np.array([3, 3, 6, 7, 8, 9, 10, 11]), np.array([4, 5]))],
    )
    with pytest.raises(ValueError, match="duplicate indices"):
        validate_split_plan(plan, GROUPS)


def test_validate_rejects_out_of_range_indices() -> None:
    plan = _manual_plan([0, 1, 2], [Fold(0, np.array([3, 4, 5]), np.array([999]))])
    with pytest.raises(ValueError, match="outside"):
        validate_split_plan(plan, GROUPS)


def test_validate_rejects_incomplete_development_coverage() -> None:
    # index 17 is in neither side, so the dev block is not fully covered
    plan = _manual_plan(
        [0, 1, 2],
        [Fold(0, np.arange(18, 36), np.arange(3, 17))],
    )
    with pytest.raises(ValueError, match="cover every development sample"):
        validate_split_plan(plan, GROUPS)


def test_validate_rejects_a_sample_validated_twice() -> None:
    plan = _manual_plan(
        [0, 1, 2],
        [
            Fold(0, np.array([3, 4, 5, 6, 7, 8]), np.array([9, 10, 11])),
            Fold(1, np.array([3, 4, 5, 6, 7, 8]), np.array([9, 10, 11])),
        ],
    )
    with pytest.raises(ValueError, match="more than one fold"):
        validate_split_plan(plan, GROUPS)


# ---------------------------------------------------------------------------
# Coverage reporting
# ---------------------------------------------------------------------------


def _coverage_matrix() -> np.ndarray:
    """36 samples x 3 labels; the third label only exists in groups 0-5."""
    y = np.zeros((GROUPS.size, 3), dtype=bool)
    y[:, 0] = True  # every protein
    y[::3, 1] = True  # one per group
    y[GROUPS < 6, 2] = True  # only in the first six groups
    return y


def test_split_coverage_reports_one_row_per_block() -> None:
    plan = _plan(seed=3)
    rows = split_coverage(_coverage_matrix(), plan, LABELS)

    assert len(rows) == 1 + 2 * plan.n_folds
    assert [row["split"] for row in rows][0] == "test"
    assert {row["role"] for row in rows} == {"test", "train", "validation"}


def test_split_coverage_counts_positives_and_proteins() -> None:
    plan = _plan(seed=3)
    rows = split_coverage(_coverage_matrix(), plan, LABELS)
    total_proteins = sum(row["proteins"] for row in rows if row["role"] == "test")
    assert total_proteins == plan.test.size
    for row in rows:
        assert row["positives"] == sum(row["label_positives"].values())
        assert row["labels_total"] == len(LABELS)


def test_split_coverage_lists_labels_below_the_support_floor() -> None:
    plan = _plan(seed=3)
    rows = split_coverage(_coverage_matrix(), plan, LABELS, min_positives=5)
    for row in rows:
        below = row["labels_below_min"]
        assert isinstance(below, list)
        assert all(name in LABELS for name in below)
        assert row["labels_with_support"] == len(LABELS) - len(below)


def test_split_coverage_counts_proteins_without_labels() -> None:
    y = _coverage_matrix()
    y[0, :] = False
    plan = _plan(seed=3)
    rows = split_coverage(y, plan, LABELS)
    total_without = sum(row["proteins_without_labels"] for row in rows)
    assert total_without >= 1


def test_split_coverage_rejects_non_boolean_matrix() -> None:
    plan = _plan(seed=3)
    with pytest.raises(ValueError, match="boolean"):
        split_coverage(np.zeros((GROUPS.size, 3)), plan, LABELS)


def test_split_coverage_rejects_label_name_mismatch() -> None:
    plan = _plan(seed=3)
    with pytest.raises(ValueError, match="label names"):
        split_coverage(_coverage_matrix(), plan, LABELS[:2])


def test_split_coverage_feeds_the_approved_label_masks() -> None:
    """A label absent from a validation block must be excluded, not scored as 0."""
    y = _coverage_matrix()
    plan = _plan(seed=3)
    rows = {(row["split"], row["role"]): row for row in split_coverage(y, plan, LABELS)}

    fold = plan.folds[0]
    mask = evaluable_label_mask(
        y[fold.train],
        y[fold.validation],
        min_positives_fit=2,
        min_negatives_fit=2,
        min_positives_eval=1,
    )
    report = label_coverage_report(
        y[fold.train], y[fold.validation], LABELS
    )

    # every label kept by the mask must be reported as supported by both blocks
    for position, name in enumerate(LABELS):
        if mask[position]:
            assert rows[(f"{fold.name}/validation", "validation")]["label_positives"][name] >= 1
            assert rows[(f"{fold.name}/train", "train")]["label_positives"][name] >= 2
            assert report[position]["kept"] is True
        else:
            assert report[position]["reason"] != ""

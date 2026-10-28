"""Guards that keep group-aware evaluation honest.

Spike S0 (findings F1 and F4), measured on scikit-learn 1.9.1:

* ``check_cv(cv=3, y, classifier=True)`` resolves to **``KFold``** for a
  multilabel target - not ``StratifiedKFold`` - so an integer ``cv`` silently
  validates on randomly split folds;
* ``SequentialFeatureSelector`` has no ``groups`` parameter, so a splitter that
  needs groups fails with "The 'groups' parameter should not be None";
* ``StratifiedKFold`` and ``StratifiedGroupKFold`` reject multilabel targets.

Consequence for this project: evaluation must always receive splits that were
built from sequence groups beforehand. :func:`make_group_split_plan` builds them
and :func:`validate_split_plan` re-checks them: no fold shares a sample or a
group with its training block, the external test block shares neither with any
fold, and every development sample is validated exactly once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np




def assert_group_disjoint(
    train_groups: object, eval_groups: object, *, name: str = "split"
) -> None:
    """Raise when any group appears on both sides of a split.

    Raises:
        ValueError: If the two group arrays are not 1-D or share a group.
    """
    train = np.asarray(train_groups)
    evaluation = np.asarray(eval_groups)
    if train.ndim != 1 or evaluation.ndim != 1:
        raise ValueError(
            "train_groups and eval_groups must be 1-D; got shapes "
            f"{train.shape} and {evaluation.shape}."
        )
    shared = np.intersect1d(train, evaluation)
    if shared.size:
        preview = shared[:5].tolist()
        raise ValueError(
            f"{name} has {shared.size} group(s) on both sides (e.g. {preview}); "
            "homologous or duplicate sequences would leak between train and "
            "evaluation."
        )




@dataclass(frozen=True)
class Fold:
    """One inner fold: positional indices for fitting and for validation."""

    index: int
    train: np.ndarray
    validation: np.ndarray

    @property
    def name(self) -> str:
        return f"fold_{self.index}"


@dataclass(frozen=True)
class SplitPlan:
    """External test block plus grouped inner folds, all built before modelling.

    Positions are indices into the aligned dataset and therefore into ``y``;
    the same plan must be reused for every candidate so comparisons share folds.
    """

    test: np.ndarray
    folds: tuple[Fold, ...]
    seed: int
    test_size: float
    n_groups_test: int
    n_groups_dev: int

    @property
    def dev(self) -> np.ndarray:
        """Every development position (train + validation of all folds)."""
        blocks = [fold.validation for fold in self.folds]
        if not blocks:
            return np.array([], dtype=int)
        return np.sort(np.concatenate(blocks))

    @property
    def n_folds(self) -> int:
        return len(self.folds)


def _indices_by_group(groups: np.ndarray, selected: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(groups, selected))


def make_group_split_plan(
    groups: object,
    *,
    n_folds: int = 3,
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitPlan:
    """Build an external group hold-out plus ``n_folds`` inner grouped folds.

    Groups (not rows) are divided: first into test and development, then the
    development groups are distributed round-robin into ``n_folds`` validation
    blocks. The split is a pure function of ``groups``, ``n_folds``,
    ``test_size`` and ``seed``.

    Args:
        groups: One group label per sample (e.g. ``md5(sequence)``).
        n_folds: Inner folds; the approved protocol starts with 3.
        test_size: Fraction of *groups* held out for the final evaluation.
        seed: Seed for the group permutations.

    Returns:
        A :class:`SplitPlan` already validated by :func:`validate_split_plan`.

    Raises:
        ValueError: On non-1-D groups, too few groups, or an impossible
            test/fold partition.
    """
    group_array = np.asarray(groups)
    if group_array.ndim != 1:
        raise ValueError(f"groups must be 1-D; got shape {group_array.shape}.")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1); got {test_size!r}.")
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1; got {n_folds!r}.")

    unique = np.unique(group_array)
    if unique.size < n_folds + 1:
        raise ValueError(
            f"{unique.size} group(s) cannot produce {n_folds} inner folds plus an "
            "external test block; at least "
            f"{n_folds + 1} groups are required."
        )

    n_test = int(round(test_size * unique.size))
    n_test = max(1, min(n_test, unique.size - n_folds))

    rng = np.random.default_rng(seed)
    order = rng.permutation(unique.size)
    test_groups = unique[order[:n_test]]
    dev_groups = unique[order[n_test:]]

    test = _indices_by_group(group_array, test_groups)

    dev_order = dev_groups[rng.permutation(dev_groups.size)]
    buckets: list[list] = [[] for _ in range(n_folds)]
    for position, group in enumerate(dev_order):
        buckets[position % n_folds].append(group)

    folds: list[Fold] = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            raise ValueError(
                f"Validation bucket {index} is empty; {dev_groups.size} development "
                f"group(s) cannot fill {n_folds} folds."
            )
        validation_groups = np.array(bucket)
        validation = _indices_by_group(group_array, validation_groups)
        train_groups = np.setdiff1d(dev_groups, validation_groups)
        if train_groups.size == 0:
            raise ValueError(
                f"Fold {index} has no training group left; reduce n_folds."
            )
        train = _indices_by_group(group_array, train_groups)
        folds.append(Fold(index=index, train=train, validation=validation))

    plan = SplitPlan(
        test=test,
        folds=tuple(folds),
        seed=seed,
        test_size=test_size,
        n_groups_test=int(test_groups.size),
        n_groups_dev=int(dev_groups.size),
    )
    validate_split_plan(plan, group_array)
    return plan


def validate_split_plan(
    plan: SplitPlan,
    groups: object,
    *,
    n_samples: int | None = None,
    n_folds: int | None = None,
) -> None:
    """Re-check every invariant of a plan against the data it will split.

    Verifies: valid, unique and in-range indices; no shared sample index and no
    shared group between train and validation of a fold; the test block shares
    neither index nor group with any fold; and every development position is
    validated exactly once.

    Raises:
        ValueError: On any violation, naming the offending block.
    """
    group_array = np.asarray(groups)
    if group_array.ndim != 1:
        raise ValueError(f"groups must be 1-D; got shape {group_array.shape}.")
    if n_samples is None:
        n_samples = group_array.size
    if plan.test.size == 0:
        raise ValueError("The test block is empty.")
    if n_folds is not None and plan.n_folds != n_folds:
        raise ValueError(f"plan has {plan.n_folds} folds, expected {n_folds}.")

    def check_block(name: str, indices: np.ndarray) -> None:
        if indices.size == 0:
            raise ValueError(f"{name} is empty.")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{name} contains duplicate indices.")
        if indices.min() < 0 or indices.max() >= n_samples:
            raise ValueError(
                f"{name} has indices outside [0, {n_samples}); got "
                f"[{indices.min()}, {indices.max()}]."
            )

    check_block("test", plan.test)
    validated: list[np.ndarray] = []
    for fold in plan.folds:
        check_block(f"{fold.name}/train", fold.train)
        check_block(f"{fold.name}/validation", fold.validation)
        shared = np.intersect1d(fold.train, fold.validation)
        if shared.size:
            raise ValueError(
                f"{fold.name} shares {shared.size} sample index(es) between train "
                "and validation."
            )
        assert_group_disjoint(
            group_array[fold.train],
            group_array[fold.validation],
            name=f"{fold.name}",
        )
        overlap_test = np.intersect1d(fold.validation, plan.test)
        if overlap_test.size:
            raise ValueError(
                f"{fold.name} validation overlaps the test block in "
                f"{overlap_test.size} sample(s); the external test must stay "
                "untouched during selection."
            )
        assert_group_disjoint(
            group_array[fold.validation], group_array[plan.test], name=f"{fold.name} vs test"
        )
        assert_group_disjoint(
            group_array[fold.train],
            group_array[plan.test],
            name=f"{fold.name} train vs test",
        )
        validated.append(fold.validation)

    stacked = np.sort(np.concatenate(validated))
    if np.unique(stacked).size != stacked.size:
        raise ValueError("A development sample is validated in more than one fold.")
    expected = np.setdiff1d(np.arange(n_samples), plan.test)
    if not np.array_equal(stacked, expected):
        raise ValueError(
            "Inner folds do not cover every development sample exactly once."
        )


def split_coverage(
    y: object,
    plan: SplitPlan,
    label_names: Sequence[str],
    *,
    min_positives: int = 1,
    include_test: bool = True,
) -> list[dict]:
    """Per-block label coverage: positives, support and missing labels.

    Reports one row for the external test block (unless ``include_test`` is
    false) and two per fold (``train``/``validation``). Labels with fewer than
    ``min_positives`` positives are listed by id; nothing is dropped or silently
    zeroed.

    ``include_test=False`` never indexes the test rows, so a caller that is not
    allowed to look at the external test block can still audit the folds.
    """
    matrix = np.asarray(y)
    if matrix.ndim != 2 or matrix.dtype != np.bool_:
        raise ValueError(
            f"y must be a 2-D boolean matrix; got shape {matrix.shape} and dtype "
            f"{matrix.dtype}."
        )
    if matrix.shape[1] != len(label_names):
        raise ValueError(
            f"y has {matrix.shape[1]} columns but {len(label_names)} label names."
        )
    names = tuple(label_names)

    rows: list[dict] = []

    def describe(name: str, role: str, indices: np.ndarray) -> None:
        block = matrix[indices]
        totals = block.sum(axis=0)
        below = [
            names[position]
            for position in range(len(names))
            if totals[position] < min_positives
        ]
        rows.append(
            {
                "split": name,
                "role": role,
                "proteins": int(indices.size),
                "positives": int(totals.sum()),
                "labels_with_support": int((totals >= min_positives).sum()),
                "labels_total": len(names),
                "labels_below_min": below,
                "proteins_without_labels": int((block.sum(axis=1) == 0).sum()),
                "label_positives": {
                    names[position]: int(totals[position])
                    for position in range(len(names))
                },
            }
        )

    if include_test:
        describe("test", "test", plan.test)
    for fold in plan.folds:
        describe(f"{fold.name}/train", "train", fold.train)
        describe(f"{fold.name}/validation", "validation", fold.validation)
    return rows

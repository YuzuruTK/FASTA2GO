"""A loud guard that makes reading the external test block impossible.

Corrupting ``X[test]`` and checking that results do not change is a *post hoc*
check: it only notices an access if the corrupted values happen to matter. This
module provides the stronger instrument the protocol asks for: a dataset whose
``X``/``y`` raise :class:`TestBlockAccessError` on **any** row selection that
intersects ``plan.test``, so an accidental read fails immediately, in the test
that causes it.

How it works
------------
``guard_dataset`` returns a normal :class:`ProteinDataset` (so every existing
validator and evaluator keeps working) whose ``X`` and ``y`` are numpy views
wrapping a :class:`_GuardedRows` subclass. That subclass checks the row
selector of every ``__getitem__`` before delegating, then returns a *plain*
``ndarray``, so the guard never propagates into derived arrays and cannot cause
false positives on later, already-authorized slices.

The check is on row selectors only. A selector that keeps every row (e.g.
``X[:, columns]``) intersects the test block and therefore raises - reading the
test rows at all is the leak, even if the caller then drops them.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from protein_database.ml.dataset import ProteinDataset
from protein_database.ml.splits import SplitPlan


class TestBlockAccessError(RuntimeError):
    """Raised when something tries to read the external test rows."""


class _GuardedRows(np.ndarray):
    """An array that refuses to return any row listed in ``_forbidden``."""

    def __new__(cls, array: object, forbidden: object) -> "_GuardedRows":
        obj = np.asarray(array).view(cls)
        obj._forbidden = frozenset(int(index) for index in np.asarray(forbidden).ravel())
        return obj

    def __array_finalize__(self, obj: object) -> None:
        # Derived arrays are plain slices of authorized rows: drop the guard so
        # a normal ``X[train][:, columns]`` never trips over stale indices.
        self._forbidden = frozenset()

    def __getitem__(self, key: object) -> object:
        forbidden = getattr(self, "_forbidden", frozenset())
        if forbidden:
            selected = np.arange(self.shape[0])[key]
            hits = sorted(set(np.atleast_1d(selected).tolist()) & forbidden)
            if hits:
                raise TestBlockAccessError(
                    "Attempt to read external test row(s) "
                    f"{hits[:5]}{'...' if len(hits) > 5 else ''}. The test block "
                    "is not part of this phase: only validation folds may be read."
                )
        base = self.view(np.ndarray)
        return np.asarray(np.ndarray.__getitem__(base, key))


def guard_rows(array: object, forbidden: object) -> np.ndarray:
    """Wrap ``array`` so any row selection touching ``forbidden`` raises."""
    return _GuardedRows(array, forbidden)


def guard_dataset(dataset: ProteinDataset, plan: SplitPlan) -> ProteinDataset:
    """Return the same dataset with ``X``/``y`` guarded against ``plan.test``."""
    return replace(
        dataset,
        X=guard_rows(dataset.X, plan.test),
        y=guard_rows(dataset.y, plan.test),
    )


def forbid_every_row(dataset: ProteinDataset) -> ProteinDataset:
    """Guard every row of ``X``/``y``: any read at all raises.

    Used to prove that planning (or any other pre-flight step) does not touch
    the data matrix, without needing a real access to go unnoticed.
    """
    forbidden = np.arange(len(dataset.keys))
    return replace(
        dataset,
        X=guard_rows(dataset.X, forbidden),
        y=guard_rows(dataset.y, forbidden),
    )

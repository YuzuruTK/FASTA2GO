"""Unsupervised pre-filtering of sequence features, fitted on training data only.

The feature space today is the 35 columns of :func:`protein_database.ml.dataset.feature_columns`
(15 physicochemical + 20 amino-acid percentages); the schema has no dipeptide
composition, so no filter here assumes hundreds of columns.

Two unsupervised filters, both learned **only** from the block passed to
``fit``:

* :class:`VarianceFilter` - drops columns that are constant (or below a
  configured variance) in the fitting block;
* :class:`CorrelationReport` - measures redundancy between numeric features and
  lists the pairs above a threshold. It **never drops a column**: removing one
  side of a correlated pair needs a tie-break policy that has not been approved.

Leakage rule: :func:`fit_variance_filter` receives the fitting block and nothing
else, and the feature-selection layer calls it with ``X[fold.train]`` only, so
validation rows never influence the decision. ``transform`` applies the stored
decision to new rows.

Future supervised filters (not implemented here)
------------------------------------------------
``SelectKBest``, chi-square or mutual information are deliberately absent. To be
added later they must: (1) accept a multilabel target and score per label or via
a documented reduction; (2) be fitted inside each fold on ``X[train]``/``y[train]``
with the label mask of that fold (``evaluation.fit_label_mask``); (3) never see
the external test block; (4) return kept names in the original order, like the
filters here; (5) log the score per feature and the parameters in the manifest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from protein_database.ml.dataset import validate_feature_matrix

DEFAULT_VARIANCE_THRESHOLD = 0.0
DEFAULT_CORRELATION_THRESHOLD = 0.9
SUPPORTED_METHODS = ("pearson",)


@dataclass(frozen=True)
class CorrelatedPair:
    """Two features whose absolute correlation reached the threshold."""

    first: str
    second: str
    correlation: float

    def as_dict(self) -> dict:
        return {
            "first": self.first,
            "second": self.second,
            "correlation": self.correlation,
        }


@dataclass(frozen=True)
class VarianceFilter:
    """Fitted variance filter: which columns survive, and why.

    ``kept`` and ``removed`` preserve the original column order. An empty
    ``kept`` is allowed (every column was constant); downstream code must check
    ``len(filter.kept)`` before asking for a model.
    """

    feature_names: tuple[str, ...]
    kept: tuple[str, ...]
    removed: tuple[str, ...]
    variances: Mapping[str, float]
    threshold: float

    @property
    def kept_indices(self) -> tuple[int, ...]:
        """Positions of the kept columns in the original matrix."""
        positions = {name: index for index, name in enumerate(self.feature_names)}
        return tuple(positions[name] for name in self.kept)

    def transform(self, X: object) -> np.ndarray:
        """Select the kept columns from ``X`` without refitting.

        Raises:
            DatasetError: If ``X`` is not a valid block with the same number of
                columns the filter was fitted on.
        """
        matrix = validate_feature_matrix(X, self.feature_names)
        return matrix[:, list(self.kept_indices)]


def fit_variance_filter(
    X: object,
    feature_names: Sequence[str],
    *,
    threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> VarianceFilter:
    """Fit a variance filter on ``X`` (population variance, ``ddof=0``).

    A column is kept when its variance is **strictly greater** than
    ``threshold``; ``threshold=0`` therefore removes only constant columns.

    Raises:
        DatasetError: If ``X`` or ``feature_names`` is invalid.
        ValueError: If ``threshold`` is negative or not finite.
    """
    matrix = validate_feature_matrix(X, feature_names)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(
            f"threshold must be a finite value >= 0; got {threshold!r}."
        )
    names = tuple(feature_names)
    variances = matrix.var(axis=0)
    keep_mask = variances > threshold
    kept = tuple(name for name, keep in zip(names, keep_mask) if keep)
    removed = tuple(name for name, keep in zip(names, keep_mask) if not keep)
    return VarianceFilter(
        feature_names=names,
        kept=kept,
        removed=removed,
        variances={
            name: float(variances[position]) for position, name in enumerate(names)
        },
        threshold=float(threshold),
    )


@dataclass(frozen=True)
class CorrelationReport:
    """Pearson correlation between surviving features, plus redundant pairs.

    ``matrix`` is symmetric with a unit diagonal. A constant column has an
    undefined correlation; its off-diagonal entries are set to ``0.0`` and the
    column is listed in ``constant_columns`` so it is never mistaken for an
    independent feature. Pairs involving constant columns are not reported.
    """

    feature_names: tuple[str, ...]
    matrix: np.ndarray
    threshold: float
    method: str
    pairs: tuple[CorrelatedPair, ...]
    constant_columns: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "threshold": self.threshold,
            "feature_names": list(self.feature_names),
            "constant_columns": list(self.constant_columns),
            "pairs": [pair.as_dict() for pair in self.pairs],
            "matrix": self.matrix.tolist(),
        }


def correlation_report(
    X: object,
    feature_names: Sequence[str],
    *,
    threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    method: str = "pearson",
) -> CorrelationReport:
    """Measure feature redundancy without removing anything.

    Args:
        X: Fitting block (never a validation or test block).
        feature_names: One name per column, original order.
        threshold: Report pairs with ``|r| >= threshold``; must be in ``(0, 1]``.
        method: Only ``"pearson"`` is implemented.

    Raises:
        DatasetError: If ``X`` or ``feature_names`` is invalid.
        ValueError: If the method is unsupported or the threshold is out of range.
    """
    matrix = validate_feature_matrix(X, feature_names)
    names = tuple(feature_names)
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported correlation method {method!r}; supported: "
            f"{SUPPORTED_METHODS}."
        )
    if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError(
            f"threshold must be in (0, 1]; got {threshold!r}."
        )

    variances = matrix.var(axis=0)
    constant = tuple(
        name for name, variance in zip(names, variances) if variance <= 0.0
    )
    if matrix.shape[1] == 1:
        correlations = np.ones((1, 1), dtype=np.float64)
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            correlations = np.corrcoef(matrix, rowvar=False)
        correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
    correlations = np.clip(correlations, -1.0, 1.0)
    np.fill_diagonal(correlations, 1.0)

    constant_mask = np.array(
        [variance <= 0.0 for variance in variances], dtype=bool
    )
    pairs: list[CorrelatedPair] = []
    for first in range(len(names)):
        if constant_mask[first]:
            continue
        for second in range(first + 1, len(names)):
            if constant_mask[second]:
                continue
            value = float(correlations[first, second])
            if abs(value) >= threshold:
                pairs.append(
                    CorrelatedPair(
                        first=names[first],
                        second=names[second],
                        correlation=value,
                    )
                )
    pairs.sort(key=lambda pair: (-abs(pair.correlation), pair.first, pair.second))
    return CorrelationReport(
        feature_names=names,
        matrix=correlations,
        threshold=float(threshold),
        method=method,
        pairs=tuple(pairs),
        constant_columns=constant,
    )

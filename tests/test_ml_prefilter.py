"""Tests for the unsupervised pre-filtering layer."""

from __future__ import annotations

import json

import numpy as np
import pytest

from protein_database.ml.dataset import DatasetError
from protein_database.ml.prefilter import (
    VarianceFilter,
    correlation_report,
    fit_variance_filter,
)

NAMES = ("f_const", "f_var", "f_dup", "f_neg")


def _matrix(rows: int = 6) -> np.ndarray:
    """Constant column, a varying column, an exact copy and its negation."""
    values = np.arange(rows, dtype=float)
    return np.column_stack(
        [np.full(rows, 5.0), values, values.copy(), -values]
    )


# ---------------------------------------------------------------------------
# Variance filter
# ---------------------------------------------------------------------------


def test_removes_only_constant_columns() -> None:
    fitted = fit_variance_filter(_matrix(), NAMES)
    assert fitted.removed == ("f_const",)
    assert fitted.kept == ("f_var", "f_dup", "f_neg")
    assert fitted.threshold == 0.0


def test_keeps_the_original_order_when_a_constant_sits_in_the_middle() -> None:
    names = ("a", "b", "c")
    matrix = np.column_stack(
        [np.arange(5.0), np.full(5, 1.0), np.arange(5.0) ** 2]
    )
    fitted = fit_variance_filter(matrix, names)
    assert fitted.kept == ("a", "c")
    assert fitted.removed == ("b",)


def test_threshold_is_strictly_greater_than() -> None:
    matrix = _matrix()
    variance = float(np.var(matrix[:, 1]))

    # a column sitting exactly on the threshold is removed
    at_threshold = fit_variance_filter(matrix, NAMES, threshold=variance)
    assert "f_var" not in at_threshold.kept

    # just below the threshold it survives
    below = fit_variance_filter(matrix, NAMES, threshold=variance - 1e-12)
    assert "f_var" in below.kept


def test_single_feature_is_accepted() -> None:
    fitted = fit_variance_filter(np.arange(4.0).reshape(-1, 1), ("only",))
    assert fitted.kept == ("only",)
    assert fitted.transform(np.arange(3.0).reshape(-1, 1)).shape == (3, 1)


def test_all_constant_is_allowed_and_yields_zero_columns() -> None:
    matrix = np.ones((4, 2))
    fitted = fit_variance_filter(matrix, ("a", "b"))
    assert fitted.kept == ()
    assert fitted.removed == ("a", "b")
    assert fitted.transform(matrix).shape == (4, 0)


def test_rejects_zero_columns() -> None:
    with pytest.raises(DatasetError, match="no feature columns"):
        fit_variance_filter(np.zeros((3, 0)), ())


def test_rejects_zero_rows() -> None:
    with pytest.raises(DatasetError, match="no rows"):
        fit_variance_filter(np.zeros((0, 2)), ("a", "b"))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_values(bad: float) -> None:
    matrix = _matrix()
    matrix[2, 1] = bad
    with pytest.raises(DatasetError, match="non-finite"):
        fit_variance_filter(matrix, NAMES)


def test_rejects_non_numeric_values() -> None:
    matrix = np.array([["a", "b"], ["c", "d"]], dtype=object)
    with pytest.raises(DatasetError, match="numeric"):
        fit_variance_filter(matrix, ("x", "y"))


def test_rejects_name_mismatch() -> None:
    with pytest.raises(DatasetError, match="feature names"):
        fit_variance_filter(_matrix(), NAMES[:2])


def test_rejects_duplicate_names() -> None:
    with pytest.raises(DatasetError, match="duplicates"):
        fit_variance_filter(_matrix(), ("a", "a", "b", "c"))


def test_rejects_empty_names() -> None:
    with pytest.raises(DatasetError, match="feature_names is empty"):
        fit_variance_filter(np.zeros((3, 1)), ())


@pytest.mark.parametrize("threshold", [-1.0, float("nan"), float("inf")])
def test_rejects_invalid_thresholds(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        fit_variance_filter(_matrix(), NAMES, threshold=threshold)


def test_transform_uses_the_fitted_decision_without_refitting() -> None:
    fitted = fit_variance_filter(_matrix(6), NAMES)

    # in validation the "constant" column varies; a refit would keep it
    validation = _matrix(4)
    validation[:, 0] = [1.0, 2.0, 3.0, 4.0]
    transformed = fitted.transform(validation)

    assert transformed.shape == (4, 3)
    assert np.allclose(transformed[:, 0], np.arange(4.0))
    assert "f_const" not in fitted.kept


def test_transform_rejects_a_different_column_count() -> None:
    fitted = fit_variance_filter(_matrix(6), NAMES)
    with pytest.raises(DatasetError, match="feature names"):
        fitted.transform(np.zeros((3, 2)))


# ---------------------------------------------------------------------------
# Correlation report
# ---------------------------------------------------------------------------


def _correlation(threshold: float = 0.9, rows: int = 6):
    report = correlation_report(_matrix(rows), NAMES, threshold=threshold)
    return report


def test_correlation_matrix_is_symmetric_with_unit_diagonal() -> None:
    report = _correlation()
    assert report.matrix.shape == (4, 4)
    assert np.allclose(report.matrix, report.matrix.T)
    assert np.allclose(np.diag(report.matrix), 1.0)
    assert np.all(np.abs(report.matrix) <= 1.0)


def test_correlation_pairs_capture_positive_and_negative_redundancy() -> None:
    report = _correlation(threshold=0.99)
    pairs = {(pair.first, pair.second): pair.correlation for pair in report.pairs}

    assert pairs[("f_var", "f_dup")] == pytest.approx(1.0)
    assert pairs[("f_var", "f_neg")] == pytest.approx(-1.0)
    assert pairs[("f_dup", "f_neg")] == pytest.approx(-1.0)
    assert len(pairs) == 3


def test_correlation_pairs_are_ordered_by_absolute_strength() -> None:
    report = _correlation(threshold=0.99)
    strengths = [abs(pair.correlation) for pair in report.pairs]
    assert strengths == sorted(strengths, reverse=True)


def test_correlation_threshold_is_configurable() -> None:
    matrix = np.column_stack(
        [np.arange(6.0), np.array([0.0, 1.0, 2.0, 3.0, 4.0, 10.0])]
    )
    names = ("a", "b")
    low = correlation_report(matrix, names, threshold=0.5)
    high = correlation_report(matrix, names, threshold=0.99)

    assert len(low.pairs) == 1
    assert high.pairs == ()
    assert abs(low.pairs[0].correlation) == pytest.approx(
        abs(float(np.corrcoef(matrix[:, 0], matrix[:, 1])[0, 1]))
    )


def test_correlation_constant_column_is_reported_and_never_paired() -> None:
    report = _correlation()
    assert report.constant_columns == ("f_const",)
    for pair in report.pairs:
        assert "f_const" not in (pair.first, pair.second)
    position = report.feature_names.index("f_const")
    assert np.allclose(
        np.delete(report.matrix[position], position), 0.0
    )


def test_correlation_single_feature_returns_a_one_by_one_matrix() -> None:
    report = correlation_report(np.arange(5.0).reshape(-1, 1), ("only",))
    assert report.matrix.shape == (1, 1)
    assert report.matrix[0, 0] == pytest.approx(1.0)
    assert report.pairs == ()


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5, float("nan")])
def test_correlation_rejects_invalid_thresholds(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        correlation_report(_matrix(), NAMES, threshold=threshold)


def test_correlation_rejects_unsupported_methods() -> None:
    with pytest.raises(ValueError, match="Unsupported correlation method"):
        correlation_report(_matrix(), NAMES, method="spearman")


def test_correlation_report_serialises_for_audit() -> None:
    payload = _correlation().as_dict()
    assert payload["method"] == "pearson"
    assert payload["threshold"] == 0.9
    assert payload["constant_columns"] == ["f_const"]
    assert len(payload["matrix"]) == 4
    json.dumps(payload)  # must be JSON-serialisable

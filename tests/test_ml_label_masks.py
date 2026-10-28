"""Tests for label masks, coverage reporting and multilabel metrics.

The masks exist because a label with a single class is *silent* in the OvR
pipeline (Spike S0, F6). These tests pin both directions: labels that cannot be
fitted are excluded, labels that cannot be scored are excluded, and the metric
does not silently absorb either case.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from protein_database.ml.evaluation import (
    REASON_EVAL_POSITIVES,
    REASON_FIT_NEGATIVES,
    REASON_FIT_POSITIVES,
    as_binary_matrix,
    eval_label_mask,
    evaluable_label_mask,
    fit_label_mask,
    label_coverage_report,
    multilabel_metrics,
    per_label_report,
)

LABELS = ["GO:0000001", "GO:0000002", "GO:0000003"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_as_binary_matrix_converts_zero_one_integers() -> None:
    matrix = as_binary_matrix(np.array([[1, 0], [0, 1]]))
    assert matrix.dtype == bool
    assert matrix.tolist() == [[True, False], [False, True]]


def test_as_binary_matrix_rejects_non_binary_values() -> None:
    with pytest.raises(ValueError, match="only 0/1"):
        as_binary_matrix(np.array([[0, 2]]))


def test_as_binary_matrix_rejects_one_dimensional_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        as_binary_matrix(np.array([1, 0, 1]))


def test_as_binary_matrix_rejects_empty_label_set() -> None:
    with pytest.raises(ValueError, match="no label columns"):
        as_binary_matrix(np.zeros((4, 0), dtype=bool))


def test_as_binary_matrix_rejects_nan() -> None:
    with pytest.raises(ValueError, match="0/1"):
        as_binary_matrix(np.array([[1.0, np.nan]]))


# ---------------------------------------------------------------------------
# Mask rules
# ---------------------------------------------------------------------------


def _fit_matrix() -> np.ndarray:
    """label0: 2 pos / 8 neg; label1: 1 pos; label2: 9 pos / 1 neg."""
    matrix = np.zeros((10, 3), dtype=bool)
    matrix[:2, 0] = True
    matrix[0, 1] = True
    matrix[:9, 2] = True
    return matrix


def test_fit_mask_requires_positives_and_negatives() -> None:
    mask = fit_label_mask(_fit_matrix())
    assert mask.tolist() == [True, False, False]


def test_fit_mask_thresholds_are_configurable() -> None:
    matrix = _fit_matrix()
    assert fit_label_mask(matrix, min_positives=1, min_negatives=1).tolist() == [
        True,
        True,
        True,
    ]


def test_eval_mask_requires_a_positive() -> None:
    y_eval = np.zeros((5, 3), dtype=bool)
    y_eval[0, 0] = True
    y_eval[0, 1] = True
    assert eval_label_mask(y_eval).tolist() == [True, True, False]


def test_evaluable_mask_is_the_intersection() -> None:
    y_fit = _fit_matrix()
    y_eval = np.ones((4, 3), dtype=bool)
    mask = evaluable_label_mask(y_fit, y_eval)
    assert mask.tolist() == fit_label_mask(y_fit).tolist()


def test_evaluable_mask_rejects_mismatched_label_counts() -> None:
    with pytest.raises(ValueError, match="same label count"):
        evaluable_label_mask(np.zeros((4, 2), dtype=bool), np.zeros((4, 3), dtype=bool))


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------


def test_coverage_report_assigns_each_reason() -> None:
    y_fit = _fit_matrix()
    y_eval = np.zeros((6, 3), dtype=bool)
    y_eval[0, 0] = True
    y_eval[0, 1] = True  # label1 can be scored, but not fitted
    y_eval[0, 2] = True
    y_eval[1, 2] = True

    rows = {row["label"]: row for row in label_coverage_report(y_fit, y_eval, LABELS)}

    assert rows[LABELS[0]]["kept"] is True
    assert rows[LABELS[0]]["reason"] == ""
    assert rows[LABELS[1]]["reason"] == REASON_FIT_POSITIVES
    assert rows[LABELS[2]]["reason"] == REASON_FIT_NEGATIVES
    assert rows[LABELS[1]]["positives_fit"] == 1
    assert rows[LABELS[2]]["negatives_fit"] == 1


def test_coverage_report_flags_label_absent_from_evaluation() -> None:
    y_fit = np.zeros((10, 1), dtype=bool)
    y_fit[:5, 0] = True
    y_eval = np.zeros((4, 1), dtype=bool)
    rows = label_coverage_report(y_fit, y_eval, ["GO:0000001"])
    assert rows[0]["kept"] is False
    assert rows[0]["reason"] == REASON_EVAL_POSITIVES
    assert rows[0]["positives_eval"] == 0


def test_coverage_report_rejects_label_name_mismatch() -> None:
    with pytest.raises(ValueError, match="label_names"):
        label_coverage_report(_fit_matrix(), _fit_matrix(), LABELS[:2])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

TRUE = np.array([[1, 0], [1, 0], [0, 1], [0, 0]], dtype=bool)
PRED = np.array([[1, 0], [0, 0], [0, 1], [0, 0]], dtype=bool)


def test_macro_f1_matches_hand_computed_values() -> None:
    metrics = multilabel_metrics(TRUE, PRED)
    assert metrics["macro_f1"] == pytest.approx(5 / 6)
    assert metrics["macro_precision"] == pytest.approx(1.0)
    assert metrics["macro_recall"] == pytest.approx(0.75)
    assert metrics["n_labels_evaluated"] == 2
    assert metrics["n_labels_total"] == 2
    assert metrics["n_samples"] == 4


def test_micro_f1_matches_hand_computed_value() -> None:
    assert multilabel_metrics(TRUE, PRED)["micro_f1"] == pytest.approx(0.8)


def test_metrics_ignore_labels_without_positives_in_evaluation() -> None:
    """A label absent from evaluation must not be scored as zero."""
    truth = np.zeros((4, 3), dtype=bool)
    truth[:2, 0] = True
    truth[2, 1] = True
    predicted = np.zeros_like(truth)
    predicted[0, 0] = True
    predicted[2, 1] = True

    mask = eval_label_mask(truth)
    unmasked = multilabel_metrics(truth, predicted)
    masked = multilabel_metrics(truth, predicted, mask=mask)

    assert unmasked["n_labels_evaluated"] == 3
    assert masked["n_labels_evaluated"] == 2
    assert masked["macro_f1"] > unmasked["macro_f1"]


def test_metrics_reject_an_all_masked_out_label_set() -> None:
    with pytest.raises(ValueError, match="No evaluable labels"):
        multilabel_metrics(TRUE, PRED, mask=np.array([False, False]))


def test_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        multilabel_metrics(TRUE, PRED[:3])


def test_per_label_report_carries_support_and_term() -> None:
    rows = per_label_report(
        TRUE,
        PRED,
        ["GO:0000001", "GO:0000002"],
        terms={"GO:0000001": "catalytic activity"},
    )
    assert [row["label"] for row in rows] == ["GO:0000001", "GO:0000002"]
    assert rows[0]["term"] == "catalytic activity"
    assert rows[0]["support"] == 2
    assert rows[0]["f1"] == pytest.approx(2 / 3)
    assert rows[1]["f1"] == pytest.approx(1.0)


def test_masks_do_not_mutate_the_inputs() -> None:
    y_fit = _fit_matrix()
    y_eval = np.ones((4, 3), dtype=bool)
    before_fit, before_eval = y_fit.copy(), y_eval.copy()
    evaluable_label_mask(y_fit, y_eval)
    label_coverage_report(y_fit, y_eval, LABELS)
    assert np.array_equal(y_fit, before_fit)
    assert np.array_equal(y_eval, before_eval)


def test_mask_recovers_metric_poisoned_by_constant_label() -> None:
    """Integration with F6: masking removes the silently-zero label."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = np.column_stack([X[:, k] > 0.0 for k in range(3)])

    train, evaluation = slice(0, 60), slice(60, 80)
    y_train = y[train].copy()
    y_eval = y[evaluation].copy()
    y_train[:, 0] = False  # label 0 cannot be fitted...
    y_eval[0, 0] = True  # ...but it can be scored

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = OneVsRestClassifier(
            LogisticRegression(max_iter=500, solver="liblinear")
        ).fit(X[train], y_train)
    predicted = model.predict(X[evaluation])

    mask = evaluable_label_mask(y_train, y_eval)
    assert mask.tolist() == [False, True, True]

    poisoned = multilabel_metrics(y_eval, predicted)["macro_f1"]
    recovered = multilabel_metrics(y_eval, predicted, mask=mask)["macro_f1"]
    assert recovered > poisoned


# ---------------------------------------------------------------------------
# The metrics module must not re-invent masks silently
# ---------------------------------------------------------------------------


def test_per_label_report_rejects_label_name_mismatch() -> None:
    with pytest.raises(ValueError, match="label_names"):
        per_label_report(TRUE, PRED, ["only-one"])

"""Label masks and multilabel metrics shared by selection, validation and test.

Why masks exist (Spike S0, finding F6): ``OneVsRestClassifier`` does **not**
raise when a label has a single class in the training split. It warns
("Label ... is present in all training examples") and returns a constant-false
predictor, so that label scores F1 = 0 and macro-F1 dropped by 0.23 in a
four-label spike. Masking keeps the selection metric comparable across
candidate feature sets and records why each label was dropped.

Three distinct filters, applied at three different moments:

* ``fit_label_mask``    -> labels a binary classifier can actually be fitted on;
* ``eval_label_mask``   -> labels that can be scored on an evaluation split;
* ``evaluable_label_mask`` -> the intersection used per candidate evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

DEFAULT_MIN_POSITIVES_FIT = 2
DEFAULT_MIN_NEGATIVES_FIT = 2
DEFAULT_MIN_POSITIVES_EVAL = 1

REASON_FIT_POSITIVES = "fit_too_few_positives"
REASON_FIT_NEGATIVES = "fit_too_few_negatives"
REASON_EVAL_POSITIVES = "eval_no_positive"


def as_binary_matrix(y: object, name: str = "y") -> np.ndarray:
    """Return ``y`` as a 2-D boolean matrix of shape ``(n_samples, n_labels)``.

    Accepts booleans or 0/1 numerics. Rejects 1-D input (a single-label target
    silently accepts a multilabel one), empty label sets, non 0/1 values and
    non-numeric dtypes.

    Raises:
        ValueError: If ``y`` is not a valid multilabel indicator matrix.
    """
    matrix = np.asarray(y)
    if matrix.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D (n_samples, n_labels); got shape {matrix.shape}."
        )
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} has no label columns.")
    if matrix.dtype != np.bool_:
        if not np.issubdtype(matrix.dtype, np.number):
            raise ValueError(
                f"{name} must be boolean or 0/1 numeric; got dtype {matrix.dtype}."
            )
        if not np.isin(matrix, (0, 1)).all():
            raise ValueError(f"{name} must contain only 0/1 values.")
        matrix = matrix.astype(bool)
    return matrix


def _check_count(name: str, value: int, minimum: int) -> int:
    if not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}.")
    return int(value)


def fit_label_mask(
    y_fit: object,
    *,
    min_positives: int = DEFAULT_MIN_POSITIVES_FIT,
    min_negatives: int = DEFAULT_MIN_NEGATIVES_FIT,
) -> np.ndarray:
    """Labels that can be fitted: enough positives **and** enough negatives.

    A label below either floor makes the OvR binary task degenerate; per the
    spike that is silent (constant predictor), so it must be excluded here.

    Returns:
        Boolean array of shape ``(n_labels,)``.
    """
    matrix = as_binary_matrix(y_fit, "y_fit")
    min_pos = _check_count("min_positives", min_positives, 1)
    min_neg = _check_count("min_negatives", min_negatives, 1)
    positives = matrix.sum(axis=0)
    negatives = matrix.shape[0] - positives
    return (positives >= min_pos) & (negatives >= min_neg)


def eval_label_mask(
    y_eval: object, *, min_positives: int = DEFAULT_MIN_POSITIVES_EVAL
) -> np.ndarray:
    """Labels that can be scored: at least ``min_positives`` positives.

    Labels without positives are undefined (or, with ``zero_division=0``,
    score a misleading zero), so they are dropped from the average instead of
    being penalised.
    """
    matrix = as_binary_matrix(y_eval, "y_eval")
    min_pos = _check_count("min_positives", min_positives, 1)
    return matrix.sum(axis=0) >= min_pos


def evaluable_label_mask(
    y_fit: object,
    y_eval: object,
    *,
    min_positives_fit: int = DEFAULT_MIN_POSITIVES_FIT,
    min_negatives_fit: int = DEFAULT_MIN_NEGATIVES_FIT,
    min_positives_eval: int = DEFAULT_MIN_POSITIVES_EVAL,
) -> np.ndarray:
    """Intersection of the fit and evaluation masks (used per candidate).

    Raises:
        ValueError: If ``y_fit`` and ``y_eval`` disagree on the number of labels.
    """
    fit = as_binary_matrix(y_fit, "y_fit")
    evaluation = as_binary_matrix(y_eval, "y_eval")
    if fit.shape[1] != evaluation.shape[1]:
        raise ValueError(
            "y_fit and y_eval must share the same label count; got "
            f"{fit.shape[1]} and {evaluation.shape[1]}."
        )
    return fit_label_mask(
        fit, min_positives=min_positives_fit, min_negatives=min_negatives_fit
    ) & eval_label_mask(evaluation, min_positives=min_positives_eval)


def label_coverage_report(
    y_fit: object,
    y_eval: object,
    label_names: Sequence[str],
    *,
    min_positives_fit: int = DEFAULT_MIN_POSITIVES_FIT,
    min_negatives_fit: int = DEFAULT_MIN_NEGATIVES_FIT,
    min_positives_eval: int = DEFAULT_MIN_POSITIVES_EVAL,
) -> list[dict]:
    """One audit row per label: counts, kept flag and drop reason.

    This is the in-memory form of ``labels_dropped.csv`` / ``split_coverage.csv``
    (written by a later step). Reason precedence follows the order the filters
    are applied: fit positives, fit negatives, evaluation positives.
    """
    fit = as_binary_matrix(y_fit, "y_fit")
    evaluation = as_binary_matrix(y_eval, "y_eval")
    if fit.shape[1] != evaluation.shape[1]:
        raise ValueError(
            "y_fit and y_eval must share the same label count; got "
            f"{fit.shape[1]} and {evaluation.shape[1]}."
        )
    if len(label_names) != fit.shape[1]:
        raise ValueError(
            f"label_names has {len(label_names)} entries but y has {fit.shape[1]} labels."
        )

    fit_positives = fit.sum(axis=0)
    fit_negatives = fit.shape[0] - fit_positives
    eval_positives = evaluation.sum(axis=0)

    rows: list[dict] = []
    for index, name in enumerate(label_names):
        if fit_positives[index] < min_positives_fit:
            reason = REASON_FIT_POSITIVES
        elif fit_negatives[index] < min_negatives_fit:
            reason = REASON_FIT_NEGATIVES
        elif eval_positives[index] < min_positives_eval:
            reason = REASON_EVAL_POSITIVES
        else:
            reason = ""
        rows.append(
            {
                "label": name,
                "kept": not reason,
                "reason": reason,
                "positives_fit": int(fit_positives[index]),
                "negatives_fit": int(fit_negatives[index]),
                "positives_eval": int(eval_positives[index]),
            }
        )
    return rows


def _masked_columns(y_true: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Return the effective column mask, rejecting an empty evaluation set."""
    if mask is None:
        effective = np.ones(y_true.shape[1], dtype=bool)
    else:
        effective = np.asarray(mask)
        if effective.dtype != np.bool_:
            raise ValueError(f"mask must be boolean; got dtype {effective.dtype}.")
        if effective.shape != (y_true.shape[1],):
            raise ValueError(
                f"mask must have shape ({y_true.shape[1]},); got {effective.shape}."
            )
    if not effective.any():
        raise ValueError(
            "No evaluable labels: every label was masked out. Refusing to report "
            "metrics computed on zero labels."
        )
    return effective


def multilabel_metrics(
    y_true: object,
    y_pred: object,
    *,
    mask: np.ndarray | None = None,
    zero_division: int = 0,
) -> dict[str, float | int]:
    """Macro/micro F1 and macro precision/recall over the masked labels.

    The mask must come from :func:`evaluable_label_mask` (or the evaluation
    mask for the final test split). ``zero_division=0`` makes a label with
    support but no positive prediction score 0 instead of NaN.

    Raises:
        ValueError: On shape mismatch, non-binary input, or an all-false mask.
    """
    truth = as_binary_matrix(y_true, "y_true")
    predicted = as_binary_matrix(y_pred, "y_pred")
    if truth.shape != predicted.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; got {truth.shape} and "
            f"{predicted.shape}."
        )
    effective = _masked_columns(truth, mask)
    true_block = truth[:, effective]
    pred_block = predicted[:, effective]

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_block,
        pred_block,
        average="macro",
        zero_division=zero_division,
    )
    return {
        "macro_f1": float(f1),
        "micro_f1": float(
            f1_score(
                true_block,
                pred_block,
                average="micro",
                zero_division=zero_division,
            )
        ),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "n_labels_evaluated": int(effective.sum()),
        "n_labels_total": int(truth.shape[1]),
        "n_samples": int(truth.shape[0]),
    }


def per_label_report(
    y_true: object,
    y_pred: object,
    label_names: Sequence[str],
    *,
    mask: np.ndarray | None = None,
    terms: dict[str, str] | None = None,
    zero_division: int = 0,
) -> list[dict]:
    """Per-label precision, recall, F1 and support for the masked labels."""
    truth = as_binary_matrix(y_true, "y_true")
    predicted = as_binary_matrix(y_pred, "y_pred")
    if truth.shape != predicted.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; got {truth.shape} and "
            f"{predicted.shape}."
        )
    if len(label_names) != truth.shape[1]:
        raise ValueError(
            f"label_names has {len(label_names)} entries but y has "
            f"{truth.shape[1]} labels."
        )
    effective = _masked_columns(truth, mask)

    precision, recall, f1, support = precision_recall_fscore_support(
        truth[:, effective],
        predicted[:, effective],
        average=None,
        zero_division=zero_division,
    )
    kept = [i for i, keep in enumerate(effective) if keep]
    rows: list[dict] = []
    for position, column in enumerate(kept):
        name = label_names[column]
        rows.append(
            {
                "label": name,
                "term": (terms or {}).get(name, ""),
                "precision": float(precision[position]),
                "recall": float(recall[position]),
                "f1": float(f1[position]),
                "support": int(support[position]),
            }
        )
    return rows

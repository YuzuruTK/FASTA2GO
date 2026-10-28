"""Tests for the forward-selection audit trail (recording, not fitting).

Rules under test (plan R1-R6): the reference score is the current selected
set's score, only the best candidate's gain decides the stop, a rejected
candidate never enters ``selected``, the first round is always accepted, no
ordering is imposed on candidates, and accepted scores never decrease.
"""

from __future__ import annotations

import pytest

from protein_database.ml.feature_selection import (
    HISTORY_COLUMNS,
    STOP_BUDGET,
    STOP_MAX_FEATURES,
    STOP_MIN_IMPROVEMENT,
    STOP_NO_CANDIDATES,
    SelectionRecorder,
)


def test_first_round_has_no_reference_and_is_accepted() -> None:
    recorder = SelectionRecorder(min_improvement=0.01)
    round_ = recorder.record("a", 0.50, n_candidates_evaluated=5)
    assert round_.round == 1
    assert round_.reference_score is None
    assert round_.gain is None
    assert round_.accepted is True
    assert round_.stop_reason is None
    assert recorder.selected == ("a",)


def test_gain_is_measured_against_the_current_selected_set() -> None:
    recorder = SelectionRecorder(min_improvement=0.0)
    recorder.record("a", 0.40, n_candidates_evaluated=4)
    second = recorder.record("b", 0.55, n_candidates_evaluated=3)
    assert second.reference_score == pytest.approx(0.40)
    assert second.gain == pytest.approx(0.15)


def test_rejected_round_is_recorded_but_not_selected() -> None:
    recorder = SelectionRecorder(min_improvement=0.05)
    recorder.record("a", 0.60, n_candidates_evaluated=4)
    rejected = recorder.record("b", 0.61, n_candidates_evaluated=3)
    assert rejected.accepted is False
    assert rejected.stop_reason == STOP_MIN_IMPROVEMENT
    assert recorder.selected == ("a",)
    assert len(recorder.finalize().history) == 2


def test_rejected_round_may_score_below_the_accepted_set() -> None:
    """R5: candidate scores are not required to be monotonic."""
    recorder = SelectionRecorder(min_improvement=0.01)
    recorder.record("a", 0.60, n_candidates_evaluated=4)
    rejected = recorder.record("b", 0.10, n_candidates_evaluated=3)
    assert rejected.accepted is False
    assert rejected.gain == pytest.approx(-0.5)


def test_no_round_can_be_recorded_after_a_stop() -> None:
    recorder = SelectionRecorder(min_improvement=0.05)
    recorder.record("a", 0.60, n_candidates_evaluated=4)
    recorder.record("b", 0.60, n_candidates_evaluated=3)
    with pytest.raises(RuntimeError, match="after stop reason"):
        recorder.record("c", 0.90, n_candidates_evaluated=2)


def test_candidate_cannot_be_selected_twice() -> None:
    recorder = SelectionRecorder(min_improvement=0.0)
    recorder.record("a", 0.5, n_candidates_evaluated=3)
    with pytest.raises(ValueError, match="already selected"):
        recorder.record("a", 0.6, n_candidates_evaluated=2)


def test_zero_min_improvement_accepts_a_tie() -> None:
    recorder = SelectionRecorder(min_improvement=0.0)
    recorder.record("a", 0.5, n_candidates_evaluated=3)
    tied = recorder.record(
        "b", 0.5, n_candidates_evaluated=2, tied_candidates=("b", "c")
    )
    assert tied.accepted is True
    assert tied.gain == pytest.approx(0.0)


def test_negative_min_improvement_is_rejected() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        SelectionRecorder(min_improvement=-0.1)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_scores_are_rejected(score: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        SelectionRecorder().record("a", score, n_candidates_evaluated=1)


def test_tied_candidates_must_contain_the_winner() -> None:
    with pytest.raises(ValueError, match="one of the tied"):
        SelectionRecorder().record(
            "a", 0.5, n_candidates_evaluated=2, tied_candidates=("b",)
        )


def test_candidate_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="n_candidates_evaluated"):
        SelectionRecorder().record("a", 0.5, n_candidates_evaluated=0)


def test_negative_fit_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_model_fits"):
        SelectionRecorder().record(
            "a", 0.5, n_candidates_evaluated=1, n_model_fits=-1
        )


def test_history_rows_follow_stable_columns() -> None:
    recorder = SelectionRecorder(min_improvement=0.0)
    recorder.record(
        "a",
        0.5,
        n_candidates_evaluated=3,
        n_model_fits=9,
        tied_candidates=("a", "b"),
    )
    rows = recorder.finalize(stopped_reason=STOP_MAX_FEATURES).as_rows()
    assert [list(row) for row in rows] == [list(HISTORY_COLUMNS)]
    assert rows[0]["tied_candidates"] == "a;b"


def test_result_totals_sum_the_rounds() -> None:
    recorder = SelectionRecorder(min_improvement=0.0)
    recorder.record("a", 0.5, n_candidates_evaluated=4, n_model_fits=40)
    recorder.record("b", 0.6, n_candidates_evaluated=3, n_model_fits=30)
    result = recorder.finalize(stopped_reason=STOP_MAX_FEATURES)
    assert result.n_evaluations == 7
    assert result.n_model_fits == 70
    assert result.best_score == pytest.approx(0.6)
    assert result.scoring == "macro_f1"
    assert result.stopped_reason == STOP_MAX_FEATURES


def test_accepted_scores_are_non_decreasing_and_keep_order() -> None:
    """R6 plus the audit of selection order."""
    recorder = SelectionRecorder(min_improvement=0.01)
    for name, score in (("c", 0.30), ("a", 0.35), ("b", 0.40)):
        recorder.record(name, score, n_candidates_evaluated=2)
    result = recorder.finalize(stopped_reason=STOP_MAX_FEATURES)
    accepted = [round_.score for round_ in result.history if round_.accepted]
    assert accepted == sorted(accepted)
    assert result.selected == ("c", "a", "b")
    assert [round_.round for round_ in result.history] == [1, 2, 3]


def test_finalize_without_rounds_reports_no_candidates() -> None:
    result = SelectionRecorder().finalize()
    assert result.stopped_reason == STOP_NO_CANDIDATES
    assert result.selected == ()
    assert result.history == ()
    assert result.best_score is None


def test_stop_records_a_budget_reason() -> None:
    recorder = SelectionRecorder()
    recorder.record("a", 0.5, n_candidates_evaluated=3)
    recorder.stop(STOP_BUDGET)
    assert recorder.finalize().stopped_reason == STOP_BUDGET


def test_stop_cannot_be_applied_twice() -> None:
    recorder = SelectionRecorder()
    recorder.stop(STOP_BUDGET)
    with pytest.raises(RuntimeError, match="already stopped"):
        recorder.stop(STOP_BUDGET)


def test_contradictory_stop_reason_is_rejected() -> None:
    recorder = SelectionRecorder(min_improvement=0.1)
    recorder.record("a", 0.5, n_candidates_evaluated=3)
    recorder.record("b", 0.5, n_candidates_evaluated=2)
    with pytest.raises(ValueError, match="contradicts"):
        recorder.finalize(stopped_reason=STOP_MAX_FEATURES)

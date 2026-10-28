"""Tests for the run budget and its ledger (pure bookkeeping, no training)."""

from __future__ import annotations

import pytest

from protein_database.ml.experiment_budget import (
    BudgetLedger,
    CostEstimate,
    RunBudget,
    total_cost,
)


class FakeClock:
    """A monotonic clock the test controls."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _evaluation_cost(folds: int = 3, labels: int = 2) -> CostEstimate:
    return CostEstimate(
        tasks=1,
        fold_evaluations=folds,
        estimator_fits=folds,
        binary_fits=folds * labels,
        exact=False,
        notes=("upper bound",),
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        {"max_tasks": 0},
        {"max_tasks": -1},
        {"max_tasks": 1.5},
        {"max_fold_evaluations": True},
        {"max_estimator_fits": 0},
        {"max_selection_candidate_evaluations": 0},
        {"max_seconds": 0},
        {"max_seconds": -2.0},
        {"max_seconds": float("nan")},
        {"max_seconds": "60"},
    ],
)
def test_budget_rejects_invalid_limits(override: dict) -> None:
    with pytest.raises(ValueError):
        RunBudget(**override)


def test_unlimited_budget_and_honest_metadata() -> None:
    budget = RunBudget()
    assert budget.unlimited is True
    payload = budget.as_dict()
    assert payload["time_limit_mode"] == "cooperative, checked before each task"
    assert payload["memory_limit"] is None
    assert "not budgeted" in " ".join(payload["notes"])
    assert RunBudget(max_tasks=2).unlimited is False


# ---------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------


def test_cost_estimates_add_up_and_merge_notes() -> None:
    first = _evaluation_cost()
    second = CostEstimate(tasks=1, selection_candidate_evaluations=6, notes=("search",))
    total = first.plus(second)
    assert total.tasks == 2
    assert total.fold_evaluations == 3
    assert total.selection_candidate_evaluations == 6
    assert total.binary_fits == 6
    assert total.exact is False
    assert set(total.notes) == {"upper bound", "search"}
    assert total.as_dict()["exact"] is False


def test_total_cost_of_nothing_is_zero() -> None:
    total = total_cost([])
    assert total.tasks == 0
    assert total.fold_evaluations == 0
    assert total.estimator_fits == 0
    assert total.selection_candidate_evaluations == 0


# ---------------------------------------------------------------------------
# Affordability and charging
# ---------------------------------------------------------------------------


def test_can_afford_accepts_a_fitting_cost() -> None:
    ledger = BudgetLedger(budget=RunBudget(max_tasks=2, max_estimator_fits=6))
    affordable, reason = ledger.can_afford(_evaluation_cost())
    assert affordable is True
    assert reason is None


@pytest.mark.parametrize(
    "budget, reason_prefix, cost",
    [
        (RunBudget(max_tasks=1), "max_tasks", None),
        (RunBudget(max_fold_evaluations=1), "max_fold_evaluations", None),
        (RunBudget(max_estimator_fits=1), "max_estimator_fits", None),
        (
            RunBudget(max_selection_candidate_evaluations=1),
            "max_selection_candidate_evaluations",
            CostEstimate(tasks=1, selection_candidate_evaluations=2),
        ),
    ],
)
def test_can_afford_reports_the_blocking_limit(budget, reason_prefix, cost) -> None:
    ledger = BudgetLedger(budget=budget)
    ledger.charge(tasks=1, fold_evaluations=3, estimator_fits=3, binary_fits=6)
    affordable, reason = ledger.can_afford(cost or _evaluation_cost())
    assert affordable is False
    assert reason.startswith(reason_prefix)
    assert "used" in reason and "estimated" in reason


def test_spending_reduces_the_remaining_budget() -> None:
    ledger = BudgetLedger(budget=RunBudget(max_tasks=2, max_fold_evaluations=4))
    ledger.charge(tasks=1, fold_evaluations=3)
    affordable, reason = ledger.can_afford(_evaluation_cost())
    assert affordable is False
    assert reason.startswith("max_fold_evaluations")
    assert ledger.remaining()["max_tasks"] == 1.0
    assert ledger.remaining()["max_fold_evaluations"] == 1.0


def test_time_limit_is_cooperative_and_checked_before_a_task() -> None:
    clock = FakeClock()
    ledger = BudgetLedger(budget=RunBudget(max_seconds=5), clock=clock)
    assert ledger.can_afford(_evaluation_cost())[0] is True
    clock.advance(5)
    affordable, reason = ledger.can_afford(_evaluation_cost())
    assert affordable is False
    assert reason.startswith("max_seconds")
    assert "cooperative" in reason


def test_charging_rejects_negative_values() -> None:
    ledger = BudgetLedger()
    with pytest.raises(ValueError):
        ledger.charge(tasks=-1)
    with pytest.raises(ValueError):
        ledger.charge(fold_evaluations=1.5)  # type: ignore[arg-type]


def test_charge_observed_reads_a_mapping() -> None:
    ledger = BudgetLedger()
    ledger.charge_observed(
        {
            "tasks": 1,
            "fold_evaluations": 3,
            "estimator_fits": 3,
            "binary_fits": 6,
            "selection_candidate_evaluations": 0,
        }
    )
    consumed = ledger.consumed()
    assert consumed["tasks_started"] == 1
    assert consumed["fold_evaluations"] == 3
    assert consumed["binary_fits"] == 6
    assert consumed["selection_candidate_evaluations"] == 0
    assert consumed["elapsed_seconds"] >= 0.0


def test_snapshot_separates_limits_from_consumption() -> None:
    clock = FakeClock()
    ledger = BudgetLedger(budget=RunBudget(max_tasks=3), clock=clock)
    ledger.charge(tasks=2)
    clock.advance(1.5)
    snapshot = ledger.snapshot()
    assert snapshot["limits"]["max_tasks"] == 3
    assert snapshot["consumed"]["tasks_started"] == 2
    assert snapshot["remaining"]["max_tasks"] == 1.0
    assert snapshot["consumed"]["elapsed_seconds"] == pytest.approx(1.5)
    assert snapshot["remaining"]["max_seconds"] is None

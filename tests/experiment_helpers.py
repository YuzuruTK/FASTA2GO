"""Shared synthetic fixtures for the Phase 9 orchestration tests.

Small on purpose: 72 rows, 4 features (one constant), 3 labels (one with a
single positive). No real data is involved and no test in this module reads a
FASTA file, a database or the GO cache.
"""

from __future__ import annotations

import numpy as np

from protein_database.ml.dataset import DatasetReport, ProteinDataset, ProteinKey
from protein_database.ml.experiment_config import (
    FailurePolicy,
    ExperimentConfig,
    SelectionSpec,
    default_experiment_config,
    model_spec,
)
from protein_database.ml.experiment_plan import build_execution_plan
from protein_database.ml.feature_selection import ForwardSelectionConfig
from protein_database.ml.models import LogisticConfig, RandomForestConfig
from protein_database.ml.splits import SplitPlan, make_group_split_plan

GROUPS = np.repeat(np.arange(12), 6)
FEATURES = ("f_a", "n1", "n2", "constant")
LABELS = ("GO:0003824", "GO:0005488", "GO:RARE")


def make_dataset(seed: int = 3) -> ProteinDataset:
    """Two learnable labels, one label with a single positive, one constant column."""
    rng = np.random.default_rng(seed)
    n = GROUPS.size
    signal = rng.normal(size=n)
    noise = rng.normal(size=(n, 2))
    rare = np.zeros(n, dtype=bool)
    rare[0] = True
    return ProteinDataset(
        keys=tuple(
            ProteinKey(f"P{index:05d}", f"sp|P{index:05d}|X", "test.fasta")
            for index in range(n)
        ),
        X=np.column_stack([signal, noise[:, 0], noise[:, 1], np.ones(n)]),
        y=np.column_stack([signal > 0, signal > 0.5, rare]),
        feature_names=FEATURES,
        label_names=LABELS,
        report=DatasetReport(n, (), (), (), {}, 0),
    )


def make_plan(seed: int = 3) -> SplitPlan:
    return make_group_split_plan(GROUPS, seed=seed)


def small_models() -> tuple:
    """Two fast configurations of two different families."""
    return (
        model_spec("logit", LogisticConfig(max_iter=500)),
        model_spec("forest", RandomForestConfig(n_estimators=10)),
    )


def small_selection() -> tuple[SelectionSpec, ...]:
    """Forward/backward with a low ceiling so the tests stay quick."""
    return (
        SelectionSpec(
            strategy="forward", config=ForwardSelectionConfig(max_features=2)
        ),
        SelectionSpec(
            strategy="backward",
            config=ForwardSelectionConfig(max_features=2),
            min_features=2,
        ),
    )


def small_config(**overrides) -> ExperimentConfig:
    """A 10-task experiment: 2 selection tasks + 2 models x 4 strategies."""
    settings = {
        "models": small_models(),
        "selection": small_selection(),
        "label": "synthetic",
    }
    settings.update(overrides)
    return default_experiment_config(**settings)


def tiny_config(**overrides) -> ExperimentConfig:
    """A 1-task experiment (one model, one strategy) for budget/state tests."""
    settings = {
        "models": (model_spec("logit", LogisticConfig(max_iter=500)),),
        "selection": (),
        "strategies": ("all_features",),
        "label": "tiny",
    }
    settings.update(overrides)
    return default_experiment_config(**settings)


def plan_for(config: ExperimentConfig, *, dataset=None, plan=None, **kwargs):
    return build_execution_plan(
        config,
        dataset if dataset is not None else make_dataset(),
        plan if plan is not None else make_plan(),
        **kwargs,
    )


__all__ = [
    "FEATURES",
    "GROUPS",
    "LABELS",
    "FailurePolicy",
    "make_dataset",
    "make_plan",
    "plan_for",
    "small_config",
    "small_models",
    "small_selection",
    "tiny_config",
]

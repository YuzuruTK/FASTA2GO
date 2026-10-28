"""Shared synthetic fixtures for the experiment tests.

Small on purpose: 72 rows, 4 features (one constant), 3 labels (one with a
single positive). No test here reads a FASTA file, a database or the GO cache.
"""

from __future__ import annotations

import numpy as np

from protein_database.ml.dataset import DatasetReport, ProteinDataset, ProteinKey
from protein_database.ml.experiments import ExperimentSettings
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
    return make_group_split_plan(GROUPS, n_folds=3, seed=seed)


def small_models() -> tuple:
    """Two fast configurations of two different families."""
    return (
        LogisticConfig(max_iter=500),
        RandomForestConfig(n_estimators=10),
    )


def small_settings(**overrides) -> ExperimentSettings:
    """Four strategies and two fast models, capped so the tests stay quick."""
    settings = {
        "models": small_models(),
        "forward": ForwardSelectionConfig(max_features=2),
        "backward": ForwardSelectionConfig(max_features=2),
        "backward_min_features": 2,
    }
    settings.update(overrides)
    return ExperimentSettings(**settings)


__all__ = [
    "FEATURES",
    "GROUPS",
    "LABELS",
    "make_dataset",
    "make_plan",
    "small_models",
    "small_settings",
]

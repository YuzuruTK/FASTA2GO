"""Explicit, serialisable configuration of one experiment.

This module only *describes* an experiment: which model configurations, which
feature-selection configurations and which strategies will be compared. It
never touches data, never trains and never imports an optional dependency at
module import time.

Identity rules (all deterministic, no timestamps, no randomness)
---------------------------------------------------------------
* ``ModelSpec.spec_id``   - ``"<family>/<12 hex>"`` derived from the family and
  the effective estimator parameters (defaults included). Two catalog entries
  with identical parameters therefore share one id and are deduplicated; two
  entries of the same family with different parameters get different ids, which
  is what makes a within-family comparison possible.
* ``SelectionSpec.spec_id`` - same idea for forward/backward selection.
* ``ExperimentConfig.config_id`` - digest of the whole effective configuration
  (catalog order included), excluding creation metadata.
* ``ExecutionPlan.plan_id`` - built in :mod:`protein_database.ml.experiment_plan`
  from the protocol, the configuration, the dataset identity, the folds and the
  ordered task ids.

The external test block is never part of any identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace

from protein_database.ml.feature_selection import ForwardSelectionConfig
from protein_database.ml.models import (
    XGBOOST_MODULE,
    LogisticConfig,
    RandomForestConfig,
    XGBoostConfig,
    describe_estimator,
)
from protein_database.ml.strategies import (
    STRATEGIES,
    STRATEGY_ALL,
    STRATEGY_BACKWARD,
    STRATEGY_FORWARD,
    STRATEGY_VARIANCE,
)

PROTOCOL_VERSION = "phase9.v1"
"""Version of the experiment protocol; changing it invalidates stored results."""

FAMILY_LOGISTIC = "logistic"
FAMILY_RANDOM_FOREST = "random_forest"
FAMILY_XGBOOST = "xgboost"

SELECTION_FORWARD = "forward"
SELECTION_BACKWARD = "backward"
SELECTION_STRATEGIES = (SELECTION_FORWARD, SELECTION_BACKWARD)

ON_ERROR_CONTINUE = "continue"
ON_ERROR_ABORT = "abort"
ON_ERROR_MODES = (ON_ERROR_CONTINUE, ON_ERROR_ABORT)

_SEARCHABLE_STRATEGIES = (STRATEGY_FORWARD, STRATEGY_BACKWARD)
_PLAIN_STRATEGIES = (STRATEGY_ALL, STRATEGY_VARIANCE)


class ExperimentConfigError(ValueError):
    """The experiment configuration is invalid; raised before any training."""


class IdentifierCollisionError(RuntimeError):
    """Two different payloads produced the same identifier."""


def canonical_json(payload: object) -> str:
    """Deterministic JSON text: sorted keys, no insignificant whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(payload: object, *, length: int = 16) -> str:
    """Short SHA-256 of the canonical JSON of ``payload``."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:length]


def family_of(config: object) -> str:
    """Map an estimator configuration object to its family name."""
    if isinstance(config, LogisticConfig):
        return FAMILY_LOGISTIC
    if isinstance(config, RandomForestConfig):
        return FAMILY_RANDOM_FOREST
    if isinstance(config, XGBoostConfig):
        return FAMILY_XGBOOST
    raise TypeError(f"Unsupported estimator configuration: {type(config).__name__}.")


def spec_parameters(config: object) -> dict:
    """Effective estimator parameters, without environment-dependent fields.

    ``describe_estimator`` reports whether the optional ``xgboost`` package was
    found; that is environment state, not part of a configuration identity, so
    it is removed here.
    """
    payload = describe_estimator(config)
    payload.pop("dependency_available", None)
    return payload


@dataclass(frozen=True)
class ModelSpec:
    """One model configuration (family + parameters), with a content id."""

    label: str
    family: str
    config: object
    spec_id: str = ""

    def __post_init__(self) -> None:
        expected = f"{self.family}/{digest(spec_parameters(self.config), length=12)}"
        if not self.spec_id:
            object.__setattr__(self, "spec_id", expected)
        elif self.spec_id != expected:
            raise IdentifierCollisionError(
                f"spec_id {self.spec_id!r} does not match its content ({expected!r})."
            )

    @property
    def model_id(self) -> str:
        """Family-level identifier (``logistic_ovr``, ``random_forest_ovr``, ...)."""
        return self.config.model_id

    @property
    def requires_dependency(self) -> str | None:
        return XGBOOST_MODULE if self.family == FAMILY_XGBOOST else None

    def payload(self) -> dict:
        return {
            "label": self.label,
            "family": self.family,
            "model_id": self.model_id,
            "parameters": spec_parameters(self.config),
        }

    def as_dict(self) -> dict:
        return {"spec_id": self.spec_id, **self.payload()}


def model_spec(label: str, config: object) -> ModelSpec:
    """Build a :class:`ModelSpec` whose id is derived from its content."""
    if not isinstance(label, str) or not label:
        raise ExperimentConfigError(f"label must be a non-empty string; got {label!r}.")
    return ModelSpec(label=label, family=family_of(config), config=config)


def selection_parameters(config: ForwardSelectionConfig, *, min_features: int) -> dict:
    """Effective selection parameters for the identity payload."""
    return {
        "max_features": config.max_features,
        "min_improvement": config.min_improvement,
        "max_evaluations": config.max_evaluations,
        "variance_threshold": config.variance_threshold,
        "min_positives_fit": config.min_positives_fit,
        "min_negatives_fit": config.min_negatives_fit,
        "min_positives_eval": config.min_positives_eval,
        "estimator": spec_parameters(config.estimator),
        "min_features": min_features,
    }


@dataclass(frozen=True)
class SelectionSpec:
    """One forward-selection or backward-elimination configuration.

    ``min_features`` only affects backward elimination, so it is part of the
    identity payload *only* for that strategy: two forward configurations that
    differ only in ``min_features`` would behave identically and must not be
    reported as distinct.
    """

    strategy: str
    config: ForwardSelectionConfig = field(default_factory=ForwardSelectionConfig)
    min_features: int = 1
    spec_id: str = ""

    def __post_init__(self) -> None:
        if self.strategy not in SELECTION_STRATEGIES:
            raise ExperimentConfigError(
                f"Selection strategy must be one of {SELECTION_STRATEGIES}; "
                f"got {self.strategy!r}."
            )
        if not isinstance(self.config, ForwardSelectionConfig):
            raise ExperimentConfigError(
                "Selection config must be a ForwardSelectionConfig; got "
                f"{type(self.config).__name__}."
            )
        if not isinstance(self.min_features, int) or self.min_features < 1:
            raise ExperimentConfigError(
                f"min_features must be an integer >= 1; got {self.min_features!r}."
            )
        expected = f"{self.strategy}/{digest(self.parameters(), length=12)}"
        if not self.spec_id:
            object.__setattr__(self, "spec_id", expected)
        elif self.spec_id != expected:
            raise IdentifierCollisionError(
                f"spec_id {self.spec_id!r} does not match its content ({expected!r})."
            )

    def parameters(self) -> dict:
        payload = selection_parameters(
            self.config,
            min_features=self.min_features if self.strategy == SELECTION_BACKWARD else 1,
        )
        return {"strategy": self.strategy, **payload}

    def as_dict(self) -> dict:
        return {"spec_id": self.spec_id, **self.parameters()}


@dataclass(frozen=True)
class FailurePolicy:
    """What the runner does when a task raises.

    A failed *selection* always leaves its dependent evaluations
    ``skipped_dependency_failure``: no substitute subset is invented, whatever
    this policy says. ``on_task_error="abort"`` additionally stops scheduling new
    tasks; already planned tasks keep the ``planned`` state instead of being
    reported as executed or cancelled.
    """

    on_task_error: str = ON_ERROR_CONTINUE

    def __post_init__(self) -> None:
        if self.on_task_error not in ON_ERROR_MODES:
            raise ExperimentConfigError(
                f"on_task_error must be one of {ON_ERROR_MODES}; got "
                f"{self.on_task_error!r}."
            )

    def as_dict(self) -> dict:
        return {"on_task_error": self.on_task_error}


@dataclass(frozen=True)
class ExperimentConfig:
    """Which model configurations, selections and strategies will be compared.

    The order of ``models``, ``selection`` and ``strategies`` is the reference
    order used everywhere (plan order, report order). No set or dict iteration
    participates, so two equal configurations always produce equal plans.
    """

    models: tuple[ModelSpec, ...]
    selection: tuple[SelectionSpec, ...] = ()
    strategies: tuple[str, ...] = STRATEGIES
    evaluation: ForwardSelectionConfig = field(default_factory=ForwardSelectionConfig)
    seed: int = 42
    protocol_version: str = PROTOCOL_VERSION
    failure_policy: FailurePolicy = field(default_factory=FailurePolicy)
    label: str = "untitled"
    deduplicated: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ExperimentConfigError("label must be a non-empty string.")
        if not isinstance(self.protocol_version, str) or not self.protocol_version:
            raise ExperimentConfigError("protocol_version must be a non-empty string.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ExperimentConfigError(f"seed must be an integer; got {self.seed!r}.")
        if not isinstance(self.evaluation, ForwardSelectionConfig):
            raise ExperimentConfigError(
                "evaluation must be a ForwardSelectionConfig; got "
                f"{type(self.evaluation).__name__}."
            )
        if not isinstance(self.failure_policy, FailurePolicy):
            raise ExperimentConfigError(
                "failure_policy must be a FailurePolicy; got "
                f"{type(self.failure_policy).__name__}."
            )
        self._validate_strategies()
        self._validate_models()
        self._validate_selection()

    def _validate_strategies(self) -> None:
        strategies = tuple(self.strategies)
        if not strategies:
            raise ExperimentConfigError("strategies is empty: nothing to compare.")
        unknown = [name for name in strategies if name not in STRATEGIES]
        if unknown:
            raise ExperimentConfigError(
                f"Unknown strategy/strategies {unknown}; expected a subset of "
                f"{STRATEGIES}."
            )
        if len(set(strategies)) != len(strategies):
            raise ExperimentConfigError(f"strategies repeats an entry: {strategies}.")
        for name in strategies:
            if name in _SEARCHABLE_STRATEGIES and not any(
                spec.strategy == name for spec in self.selection
            ):
                raise ExperimentConfigError(
                    f"Strategy {name!r} needs a matching SelectionSpec."
                )
        for spec in self.selection:
            if spec.strategy not in strategies:
                raise ExperimentConfigError(
                    f"SelectionSpec for {spec.strategy!r} is present but the "
                    "strategy is disabled; remove one of them."
                )

    def _validate_models(self) -> None:
        models = tuple(self.models)
        if not models:
            raise ExperimentConfigError("models is empty: nothing to compare.")
        seen: dict[str, str] = {}
        duplicates: list[str] = []
        kept: list[ModelSpec] = []
        for spec in models:
            if not isinstance(spec, ModelSpec):
                raise ExperimentConfigError(
                    f"models must contain ModelSpec objects; got {type(spec).__name__}."
                )
            payload = canonical_json(
                {"family": spec.family, "parameters": spec_parameters(spec.config)}
            )
            previous = seen.get(spec.spec_id)
            if previous is None:
                seen[spec.spec_id] = payload
                kept.append(spec)
            elif previous != payload:
                raise IdentifierCollisionError(
                    f"Two different configurations share spec_id {spec.spec_id!r}."
                )
            else:
                # identical content (labels may differ): one task, first label wins
                duplicates.append(spec.label)
        if len({spec.label for spec in kept}) != len(kept):
            raise ExperimentConfigError(
                "Two model specs share a label; labels must be unique."
            )
        object.__setattr__(self, "models", tuple(kept))
        object.__setattr__(self, "deduplicated", tuple(duplicates))

    def _validate_selection(self) -> None:
        by_strategy: dict[str, SelectionSpec] = {}
        for spec in self.selection:
            if not isinstance(spec, SelectionSpec):
                raise ExperimentConfigError(
                    "selection must contain SelectionSpec objects; got "
                    f"{type(spec).__name__}."
                )
            if spec.strategy in by_strategy:
                raise ExperimentConfigError(
                    f"Two selection configurations for {spec.strategy!r}: this "
                    "phase runs one selection per strategy, otherwise the "
                    "strategy would map to more than one subset."
                )
            by_strategy[spec.strategy] = spec
        object.__setattr__(
            self,
            "selection",
            tuple(
                by_strategy[name]
                for name in SELECTION_STRATEGIES
                if name in by_strategy
            ),
        )

    # -- identity -----------------------------------------------------------

    def selection_spec(self, strategy: str) -> SelectionSpec | None:
        for spec in self.selection:
            if spec.strategy == strategy:
                return spec
        return None

    def effective(self) -> dict:
        """Full serialisable configuration, defaults included."""
        return {
            "protocol_version": self.protocol_version,
            "label": self.label,
            "seed": self.seed,
            "strategies": list(self.strategies),
            "evaluation": selection_parameters(self.evaluation, min_features=1),
            "models": [spec.as_dict() for spec in self.models],
            "selection": [spec.as_dict() for spec in self.selection],
            "failure_policy": self.failure_policy.as_dict(),
            "deduplicated_labels": list(self.deduplicated),
        }

    @property
    def config_id(self) -> str:
        return digest(self.effective())

    def as_dict(self) -> dict:
        return {"config_id": self.config_id, **self.effective()}


# ---------------------------------------------------------------------------
# Moderate catalog of candidate configurations
# ---------------------------------------------------------------------------
#
# A candidate list, never a claim that these are the best hyperparameters. Each
# entry changes one justified parameter relative to its family baseline, so the
# comparison stays interpretable; no grid search happens in this phase.
#
# Expected size: ``len(models) x len(strategies) + len(selection)``. With the
# defaults below that is ``7 x 4 + 2 = 30`` planned tasks (the XGBoost ones are
# planned as ``skipped_unavailable_dependency`` while the package is absent).

CATALOG_LOGISTIC_BASE = ("logistic_base", LogisticConfig())
CATALOG_LOGISTIC_BALANCED = (
    "logistic_balanced",
    LogisticConfig(class_weight="balanced"),
)
CATALOG_LOGISTIC_L2 = ("logistic_C0.1", LogisticConfig(C=0.1))

CATALOG_FOREST_BASE = ("forest_base", RandomForestConfig(n_estimators=200))
CATALOG_FOREST_BALANCED = (
    "forest_balanced",
    RandomForestConfig(n_estimators=200, class_weight="balanced"),
)
CATALOG_FOREST_SHALLOW = (
    "forest_shallow",
    RandomForestConfig(n_estimators=200, max_depth=8),
)

CATALOG_XGBOOST_BASE = ("xgboost_base", XGBoostConfig())

DEFAULT_CATALOG = (
    CATALOG_LOGISTIC_BASE,
    CATALOG_LOGISTIC_BALANCED,
    CATALOG_LOGISTIC_L2,
    CATALOG_FOREST_BASE,
    CATALOG_FOREST_BALANCED,
    CATALOG_FOREST_SHALLOW,
    CATALOG_XGBOOST_BASE,
)

DEFAULT_FORWARD_SELECTION = SelectionSpec(
    strategy=SELECTION_FORWARD,
    config=ForwardSelectionConfig(max_features=10),
)

DEFAULT_BACKWARD_SELECTION = SelectionSpec(
    strategy=SELECTION_BACKWARD,
    config=ForwardSelectionConfig(max_features=10),
    min_features=2,
)

DEFAULT_SELECTION = (DEFAULT_FORWARD_SELECTION, DEFAULT_BACKWARD_SELECTION)


def default_catalog() -> tuple[ModelSpec, ...]:
    """The moderate catalog: 3 logistic + 3 forests + 1 optional XGBoost."""
    return tuple(model_spec(label, config) for label, config in DEFAULT_CATALOG)


def default_selection_specs() -> tuple[SelectionSpec, ...]:
    """One forward and one backward configuration (provisional parameters)."""
    return (
        SelectionSpec(
            strategy=SELECTION_FORWARD,
            config=ForwardSelectionConfig(max_features=10),
        ),
        SelectionSpec(
            strategy=SELECTION_BACKWARD,
            config=ForwardSelectionConfig(max_features=10),
            min_features=2,
        ),
    )


def default_experiment_config(
    *,
    models: tuple[ModelSpec, ...] | None = None,
    selection: tuple[SelectionSpec, ...] | None = None,
    strategies: tuple[str, ...] = STRATEGIES,
    evaluation: ForwardSelectionConfig | None = None,
    seed: int = 42,
    label: str = "phase9-default",
    failure_policy: FailurePolicy | None = None,
) -> ExperimentConfig:
    """Assemble the default experiment, accepting overrides for tests and future runs."""
    return ExperimentConfig(
        models=default_catalog() if models is None else tuple(models),
        selection=default_selection_specs() if selection is None else tuple(selection),
        strategies=tuple(strategies),
        evaluation=ForwardSelectionConfig() if evaluation is None else evaluation,
        seed=seed,
        label=label,
        failure_policy=failure_policy or FailurePolicy(),
    )


def describe_catalog(config: ExperimentConfig) -> dict:
    """Catalog size and the arithmetic that produced it (no execution)."""
    n_models = len(config.models)
    n_strategies = len(config.strategies)
    n_selection = len(config.selection)
    return {
        "models": [spec.as_dict() for spec in config.models],
        "selection": [spec.as_dict() for spec in config.selection],
        "strategies": list(config.strategies),
        "n_model_specs": n_models,
        "n_selection_specs": n_selection,
        "n_evaluation_tasks": n_models * n_strategies,
        "n_selection_tasks": n_selection,
        "n_tasks": n_models * n_strategies + n_selection,
        "formula": "len(models) * len(strategies) + len(selection)",
        "deduplicated_labels": list(config.deduplicated),
    }

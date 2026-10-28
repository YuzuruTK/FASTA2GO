"""Tests for the experiment configuration and catalog (synthetic, no training)."""

from __future__ import annotations

import json

import pytest

from protein_database.ml.experiment_config import (
    PROTOCOL_VERSION,
    ExperimentConfig,
    ExperimentConfigError,
    FailurePolicy,
    IdentifierCollisionError,
    ModelSpec,
    SelectionSpec,
    canonical_json,
    default_catalog,
    default_experiment_config,
    default_selection_specs,
    describe_catalog,
    digest,
    model_spec,
    spec_parameters,
)
from protein_database.ml.feature_selection import ForwardSelectionConfig
from protein_database.ml.models import LogisticConfig, RandomForestConfig, XGBoostConfig
from tests.experiment_helpers import small_config, tiny_config


# ---------------------------------------------------------------------------
# Deterministic serialisation and identifiers
# ---------------------------------------------------------------------------


def test_canonical_json_ignores_key_order() -> None:
    first = {"b": 1, "a": [2, {"d": 4, "c": 3}]}
    second = {"a": [2, {"c": 3, "d": 4}], "b": 1}
    assert canonical_json(first) == canonical_json(second)
    assert digest(first) == digest(second)


def test_digest_depends_on_content_and_length() -> None:
    assert digest({"a": 1}) != digest({"a": 2})
    assert len(digest({"a": 1})) == 16
    assert len(digest({"a": 1}, length=8)) == 8


def test_spec_id_is_content_derived_and_label_independent() -> None:
    first = model_spec("one", LogisticConfig())
    second = model_spec("two", LogisticConfig())
    assert first.spec_id == second.spec_id
    assert first.label != second.label
    assert first.spec_id.startswith("logistic/")


def test_distinct_configurations_of_one_family_get_distinct_ids() -> None:
    base = model_spec("base", LogisticConfig())
    balanced = model_spec("balanced", LogisticConfig(class_weight="balanced"))
    regularized = model_spec("l2", LogisticConfig(C=0.1))
    assert len({base.spec_id, balanced.spec_id, regularized.spec_id}) == 3
    assert {base.model_id, balanced.model_id} == {"logistic_ovr"}


def test_spec_parameters_exclude_environment_dependent_fields() -> None:
    payload = spec_parameters(XGBoostConfig())
    assert "dependency_available" not in payload
    assert payload["model_id"] == "xgboost_ovr"


def test_model_spec_rejects_a_forged_identifier() -> None:
    with pytest.raises(IdentifierCollisionError):
        ModelSpec(
            label="forged",
            family="logistic",
            config=LogisticConfig(),
            spec_id="logistic/deadbeef",
        )


def test_model_spec_rejects_unknown_configuration_objects() -> None:
    with pytest.raises(TypeError):
        model_spec("bad", "not a config")


# ---------------------------------------------------------------------------
# Deduplication and identity of the whole configuration
# ---------------------------------------------------------------------------


def test_identical_configurations_are_deduplicated_and_recorded() -> None:
    config = ExperimentConfig(
        models=(
            model_spec("first", LogisticConfig()),
            model_spec("duplicate", LogisticConfig()),
        ),
        strategies=("all_features",),
    )
    assert len(config.models) == 1
    assert config.models[0].label == "first"
    assert config.deduplicated == ("duplicate",)
    assert config.effective()["deduplicated_labels"] == ["duplicate"]


def test_config_id_is_stable_and_changes_with_content() -> None:
    first = small_config()
    second = small_config()
    assert first.config_id == second.config_id
    assert small_config(seed=7).config_id != first.config_id
    assert tiny_config().config_id != first.config_id


def test_effective_configuration_records_defaults() -> None:
    payload = tiny_config().effective()
    assert payload["protocol_version"] == PROTOCOL_VERSION
    assert payload["seed"] == 42
    assert payload["models"][0]["parameters"]["class_weight"] is None
    assert payload["models"][0]["parameters"]["solver"] == "liblinear"
    assert payload["evaluation"]["estimator"]["model_id"] == "logistic_ovr"
    assert payload["failure_policy"] == {"on_task_error": "continue"}
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload


def test_config_serialisation_is_order_independent() -> None:
    models = (
        model_spec("a", LogisticConfig()),
        model_spec("b", RandomForestConfig(n_estimators=5)),
    )
    forward = ExperimentConfig(models=models, strategies=("all_features",))
    reversed_models = ExperimentConfig(
        models=tuple(reversed(models)), strategies=("all_features",)
    )
    # the declared order is part of the identity, so it must matter
    assert forward.config_id != reversed_models.config_id
    assert [spec.label for spec in reversed_models.models] == ["b", "a"]


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_default_catalog_is_moderate_and_documented() -> None:
    catalog = default_catalog()
    labels = [spec.label for spec in catalog]
    families = [spec.family for spec in catalog]
    assert labels == [
        "logistic_base",
        "logistic_balanced",
        "logistic_C0.1",
        "forest_base",
        "forest_balanced",
        "forest_shallow",
        "xgboost_base",
    ]
    assert families.count("logistic") == 3
    assert families.count("random_forest") == 3
    assert families.count("xgboost") == 1
    assert len({spec.spec_id for spec in catalog}) == len(catalog)


def test_catalog_has_balanced_and_unbalanced_variants() -> None:
    catalog = {spec.label: spec for spec in default_catalog()}
    assert catalog["logistic_base"].config.class_weight is None
    assert catalog["logistic_balanced"].config.class_weight == "balanced"
    assert catalog["forest_base"].config.class_weight is None
    assert catalog["forest_balanced"].config.class_weight == "balanced"
    # XGBoost exposes no balancing parameter in this project, so none is invented
    assert not hasattr(catalog["xgboost_base"].config, "scale_pos_weight")


def test_catalog_size_follows_the_documented_formula() -> None:
    config = default_experiment_config()
    described = describe_catalog(config)
    assert described["n_model_specs"] == 7
    assert described["n_selection_specs"] == 2
    assert described["n_evaluation_tasks"] == 28
    assert described["n_tasks"] == 30
    assert described["formula"] == "len(models) * len(strategies) + len(selection)"
    assert len(config.selection) == 2


def test_default_selection_specs_are_distinct_and_identified() -> None:
    forward, backward = default_selection_specs()
    assert forward.strategy == "forward"
    assert backward.strategy == "backward"
    assert forward.spec_id.startswith("forward/")
    assert backward.spec_id.startswith("backward/")
    assert forward.spec_id != backward.spec_id


def test_min_features_only_matters_for_backward_identity() -> None:
    first = SelectionSpec(strategy="forward", min_features=1)
    second = SelectionSpec(strategy="forward", min_features=5)
    assert first.spec_id == second.spec_id
    third = SelectionSpec(strategy="backward", min_features=1)
    fourth = SelectionSpec(strategy="backward", min_features=5)
    assert third.spec_id != fourth.spec_id


def test_selection_spec_rejects_invalid_input() -> None:
    with pytest.raises(ExperimentConfigError):
        SelectionSpec(strategy="sideways")
    with pytest.raises(ExperimentConfigError):
        SelectionSpec(strategy="forward", config="not a config")  # type: ignore[arg-type]
    with pytest.raises(ExperimentConfigError):
        SelectionSpec(strategy="backward", min_features=0)
    with pytest.raises(IdentifierCollisionError):
        SelectionSpec(strategy="forward", spec_id="forward/deadbeef")


# ---------------------------------------------------------------------------
# Validation before any training
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        {"models": ()},
        {"strategies": ()},
        {"strategies": ("all_features", "nonsense")},
        {"strategies": ("all_features", "all_features")},
        {"label": ""},
        {"protocol_version": ""},
        {"seed": "42"},
        {"seed": True},
        {"evaluation": "not a config"},
        {"failure_policy": "not a policy"},
    ],
)
def test_config_rejects_invalid_values(override: dict) -> None:
    settings = {"models": tiny_config().models, "strategies": ("all_features",)}
    settings.update(override)
    with pytest.raises((ExperimentConfigError, ValueError, TypeError)):
        ExperimentConfig(**settings)


def test_forward_strategy_requires_a_matching_selection_spec() -> None:
    with pytest.raises(ExperimentConfigError, match="needs a matching SelectionSpec"):
        ExperimentConfig(models=tiny_config().models, strategies=("forward",))


def test_selection_spec_for_a_disabled_strategy_is_rejected() -> None:
    with pytest.raises(ExperimentConfigError, match="disabled"):
        ExperimentConfig(
            models=tiny_config().models,
            selection=small_config().selection,
            strategies=("all_features",),
        )


def test_two_selection_configurations_per_strategy_are_rejected() -> None:
    with pytest.raises(ExperimentConfigError, match="Two selection configurations"):
        ExperimentConfig(
            models=tiny_config().models,
            selection=(
                SelectionSpec(strategy="forward"),
                SelectionSpec(
                    strategy="forward", config=ForwardSelectionConfig(max_features=3)
                ),
            ),
            strategies=("forward",),
        )


def test_duplicate_labels_and_bad_model_entries_are_rejected() -> None:
    with pytest.raises(ExperimentConfigError, match="unique"):
        ExperimentConfig(
            models=(
                model_spec("same", LogisticConfig()),
                model_spec("same", LogisticConfig(C=0.5)),
            ),
            strategies=("all_features",),
        )
    with pytest.raises(ExperimentConfigError, match="ModelSpec"):
        ExperimentConfig(models=("logistic",), strategies=("all_features",))


def test_failure_policy_rejects_unknown_modes() -> None:
    assert FailurePolicy().on_task_error == "continue"
    assert FailurePolicy(on_task_error="abort").as_dict() == {"on_task_error": "abort"}
    with pytest.raises(ExperimentConfigError):
        FailurePolicy(on_task_error="explode")

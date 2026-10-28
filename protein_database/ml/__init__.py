"""Machine-learning building blocks for FASTA2GO.

Only auditable contracts live here: label masks and multilabel metrics
(``evaluation``), dataset loading/alignment (``dataset``), grouped split plans
(``splits``), the variance prefilter (``prefilter``), the model configurations
(``models``), Forward Selection and Backward Elimination
(``feature_selection``), the strategy comparison (``strategies``), the
model-by-subset comparison (``model_comparison``) and the GO Slim label space
(``go_slim``).

Experiment orchestration is split on purpose: ``experiment_config`` (what would
run), ``experiment_plan`` (the enumerated tasks and their cost estimates),
``experiment_budget`` (limits and observed costs), ``experiment_guard`` (a
test-block access guard), ``experiment_runner`` (plan-then-execute with states
and an incremental store) and ``experiment_report`` (manifest and report). None
of them touches the database, the real base or the external test block.
"""

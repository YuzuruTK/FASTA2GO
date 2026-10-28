"""Machine-learning building blocks for FASTA2GO.

Label masks and multilabel metrics (``evaluation``), dataset loading, alignment
and sequence groups (``dataset``), grouped split plans (``splits``), the
variance prefilter and the correlation report (``prefilter``), the model
configurations (``models``), Forward Selection and Backward Elimination
(``feature_selection``), the strategy names (``strategies``), the
model-by-subset comparison (``model_comparison``), the GO Slim label space
(``go_slim``) and the experiment flow (``experiments``).

``experiments`` is the entry point: it slices the dataset to the development
rows, builds one subset per strategy, scores every (strategy, model) pair on
the same folds and writes the results. ``experiments.evaluate_external`` is the
only function that indexes the external test block, and ``run_experiment``
never calls it.
"""

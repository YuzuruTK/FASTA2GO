"""The four feature-selection strategies compared by the experiment flow.

1. ``all_features`` - every column, no selection and no variance filtering;
2. ``variance_prefilter`` - the columns kept by the variance filter of every
   inner fold, without any supervised search;
3. ``forward`` - :func:`protein_database.ml.feature_selection.run_forward_selection`;
4. ``backward`` - :func:`protein_database.ml.feature_selection.run_backward_elimination`.

This module only names them. Building the subsets and scoring them lives in
:mod:`protein_database.ml.experiments` (one subset per strategy, then every
model on the same folds), so there is a single path from a strategy name to a
result row.
"""

STRATEGY_ALL = "all_features"
STRATEGY_VARIANCE = "variance_prefilter"
STRATEGY_FORWARD = "forward"
STRATEGY_BACKWARD = "backward"

STRATEGIES = (STRATEGY_ALL, STRATEGY_VARIANCE, STRATEGY_FORWARD, STRATEGY_BACKWARD)

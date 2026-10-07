"""The non-deep baseline: handcrafted features plus gradient boosting.

Given to you complete. Its job is to establish the score a network has to beat
to have earned its complexity, and because it trains in seconds it is also the
right instrument for cheap experiments: which features matter, what a wrong
split costs. The instructor's leaderboard baseline is a model of this kind.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from src import config


def fit_baseline(X: pd.DataFrame, y: np.ndarray, seed: int = config.SEED) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=1.0, random_state=seed,
    )
    return model.fit(X, y)


def predict_baseline(model: HistGradientBoostingRegressor, X: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(X), 0.0, 1.0)


def importance(model: HistGradientBoostingRegressor, X: pd.DataFrame, y: np.ndarray,
               seed: int = config.SEED) -> pd.Series:
    """Permutation importance: how much RMSE rises when one column is shuffled.

    Measured on data the model did not train on, so it reflects what the
    model *uses*, not what it could have used.
    """
    r = permutation_importance(model, X, y, scoring="neg_root_mean_squared_error",
                               n_repeats=10, random_state=seed)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)

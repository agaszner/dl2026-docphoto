"""Score predictions and break the error down.

RMSE is the competition metric, but a single number will not tell you what to
fix. Slice the error by document family, language, font, font size and
photographer to see where your model actually fails, and leave `eval_ok = 0`
rows out of every number you report.

Two runs that differ by a few thousandths may not differ at all. Resample by
page, not by image, when you estimate the uncertainty of a comparison:
photographs of the same page are not independent samples. That is next week.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def rmse(y, pred) -> float:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def by_slice(df: pd.DataFrame, pred: np.ndarray, column: str) -> pd.DataFrame:
    """RMSE per value of `column`, with the count and the mean target and prediction."""
    d = pd.DataFrame({column: df[column].to_numpy(), "y": df[config.TARGET].to_numpy(), "p": pred})
    rows = []
    for key, g in d.groupby(column):
        rows.append({column: key, "n": len(g), "rmse": rmse(g["y"], g["p"]),
                     "mean_target": g["y"].mean(), "mean_pred": g["p"].mean()})
    return pd.DataFrame(rows).set_index(column).sort_values("rmse", ascending=False)


def worst(df: pd.DataFrame, pred: np.ndarray, k: int = 5) -> pd.DataFrame:
    """The k rows with the largest absolute error, with the columns worth looking at."""
    d = df.copy()
    d["pred"] = pred
    d["error"] = d["pred"] - d[config.TARGET]
    cols = ["image_id", config.TARGET, "pred", "error", "family", "font", "size_pt", "photographer"]
    return d.reindex(d["error"].abs().sort_values(ascending=False).index)[cols].head(k)

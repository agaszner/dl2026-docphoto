"""The one test worth having on day one.

The same page appears in dozens of photographs and every layout exists in a
Hungarian and an English version. A split that puts one unit on both sides
makes every score you report optimistic: for a ResNet-18 on this data, 0.113
by random folds against 0.148 on the held-out test, and nothing about the
training run looks wrong when it happens.
"""
from __future__ import annotations

import pandas as pd

from src.data import splits


def test_imports() -> None:
    """The project is importable and the paths resolve. Fails if PYTHONPATH is wrong."""
    from src import config

    assert config.ROOT.exists()
    assert config.TARGET == "usability"


def toy_table() -> pd.DataFrame:
    """The columns of train.csv that matter for splitting: 40 rows, 5 strict units of 8."""
    return pd.DataFrame({
        "image_id": [f"ph_{i:08x}" for i in range(40)],
        "group_strict": [i // 8 for i in range(40)],
        "group": [i // 4 for i in range(40)],
        "page_id": [i // 2 for i in range(40)],
        "photographer": [f"P{i // 4:02d}" for i in range(40)],
        "usability": [0.5] * 40,
    })


def test_no_unit_appears_in_both_splits() -> None:
    """No strict unit may be in training and validation at the same time."""
    df = toy_table()
    train, val = splits.build_splits(df, val_fraction=0.2)   # the default column is group_strict

    overlap = set(df.loc[train, "group_strict"]) & set(df.loc[val, "group_strict"])
    assert not overlap, f"units on both sides of the split: {sorted(overlap)}"
    assert len(train) + len(val) == len(df), "the split lost or duplicated rows"
    assert len(val) >= 0.2 * len(df), "the validation side is smaller than val_fraction"
    assert len(set(train) & set(val)) == 0, "a row is on both sides"


def test_split_is_reproducible_and_seeded() -> None:
    """The same seed gives the same split; the row order of the table does not change which units are held out."""
    df = toy_table()
    _, val_a = splits.build_splits(df, val_fraction=0.2, seed=3)
    _, val_b = splits.build_splits(df, val_fraction=0.2, seed=3)
    assert list(val_a) == list(val_b)

    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    _, val_c = splits.build_splits(shuffled, val_fraction=0.2, seed=3)
    assert set(shuffled.loc[val_c, "group_strict"]) == set(df.loc[val_a, "group_strict"]), \
        "which units are held out must not depend on the row order: sort the units before shuffling"

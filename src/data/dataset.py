"""Loading photographs and their labels.

This module is given to you complete. Read it once.

    python -m src.data.dataset                # builds data/work/train_768 and the 64 px pixel table

Three things live here:

* `load_table` reads `train.csv` and adds the path of every photograph;
* `build_cache` resizes every photograph once to a 768 px long side, which is
  what the handcrafted features are computed on;
* `pixel_table` turns every photograph into one row of grey levels, the
  input of the first network.

Does not belong here: anything that decides which rows are training and which
are validation. That lives in splits.py, and keeping it separate is what stops
a split leaking by accident.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src import config

RAW = config.ROOT / "data" / "raw"
WORK = config.ROOT / "data" / "work"

# These describe the page and the photographer. Use them to split and to slice
# the error, never as model input: a test image comes with none of them.
META_COLUMNS = ["page_id", "layout_id", "family", "lang", "font", "size_pt", "photographer"]


def load_table(path: Path = RAW / "train.csv") -> pd.DataFrame:
    """The label table with one extra column, `path`, pointing at the photograph."""
    df = pd.read_csv(path)
    df["path"] = [str(RAW / "train" / f"{i}.jpg") for i in df["image_id"]]
    return df


# ----------------------------------------------------------------- the cache

def cache_dir(size: int) -> Path:
    return WORK / f"train_{size}"


def _resize_one(src: str, dst: Path, size: int) -> None:
    im = Image.open(src)
    # A JPEG can be decoded at 1/2, 1/4 or 1/8 scale straight from its DCT
    # coefficients. draft() picks the smallest scale that is still at least
    # the requested size, which makes this loop several times faster than
    # decoding 2048 px and resizing afterwards.
    im.draft("RGB", (size, size))
    im = im.convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)  # long side -> size, aspect kept
    im.save(dst, quality=92)


def build_cache(df: pd.DataFrame, size: int, workers: int = 4) -> Path:
    """Resize every photograph once so that its long side is `size` px.

    Idempotent: images already in the cache are skipped, so it is safe to call
    at the top of every script.
    """
    out = cache_dir(size)
    out.mkdir(parents=True, exist_ok=True)
    todo = [(p, out / f"{i}.jpg") for i, p in zip(df["image_id"], df["path"], strict=True)
            if not (out / f"{i}.jpg").exists()]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda t: _resize_one(t[0], t[1], size), todo))
    return out


# ---------------------------------------------------------------- the pixels

PAD_GREY = 0.5


def pixels_of_image(path: str | Path, size: int) -> np.ndarray:
    """One photograph as a flat vector of size * size grey levels in [0, 1].

    The long side is scaled to `size` with antialiasing, the short side is
    padded with mid-grey to make a square, and the square is flattened row by
    row. Nothing else: no rectification, no cropping to the page, no
    orientation fix. Whatever the network is to learn, it learns from this.
    """
    im = Image.open(path)
    im.draft("L", (size, size))
    im = im.convert("L")
    im.thumbnail((size, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float32) / 255.0
    out = np.full((size, size), PAD_GREY, dtype=np.float32)
    top, left = (size - a.shape[0]) // 2, (size - a.shape[1]) // 2
    out[top:top + a.shape[0], left:left + a.shape[1]] = a
    return out.reshape(-1)


def pixel_table(df: pd.DataFrame, size: int = 64, workers: int = 4) -> np.ndarray:
    """All photographs as one float32 array of shape (len(df), size * size), rows in the order of `df`.

    Computed once and stored in data/work/pixels_<size>.npz; later calls load
    it in milliseconds. Reads the 768 px cache when it exists, the originals
    otherwise.
    """
    store = WORK / f"pixels_{size}.npz"
    ids = df["image_id"].to_numpy().astype(str)
    if store.exists():
        z = np.load(store)
        if len(z["ids"]) == len(ids) and (z["ids"] == ids).all():
            return z["x"]
    source = cache_dir(768)
    paths = [source / f"{i}.jpg" if (source / f"{i}.jpg").exists() else p
             for i, p in zip(df["image_id"], df["path"], strict=True)]
    with ThreadPoolExecutor(workers) as ex:
        x = np.stack(list(ex.map(lambda p: pixels_of_image(p, size), paths)))
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez(store, ids=ids, x=x)
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the caches under data/work/")
    ap.add_argument("--pixels", type=int, action="append", help="pixel table size; repeatable (default 64)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    df = load_table()
    out = build_cache(df, 768, a.workers)
    print(f"{out}: {len(list(out.glob('*.jpg')))} images")
    for size in a.pixels or [64]:
        print(f"pixel table {size} px: {pixel_table(df, size, a.workers).shape}")


if __name__ == "__main__":
    main()

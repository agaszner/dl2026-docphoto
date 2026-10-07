"""Classical image-quality features for the baseline model.

Seven features for the baseline, given to you. Five are whole-frame, two are tile-wise with a
non-mean summary, which is the part that needs thought: the *minimum*
sharpness over tiles sees a smear across one column that the mean sharpness
averages away, and the *maximum* dark fraction over tiles sees a shadow across
one corner. Designing better ones is the homework; the docstrings say what
each of these sees and, by omission, what none of them does.

Features are computed on the 768 px cache, so every image is at the same
scale and a sharpness number means the same thing on every row.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

from src.data.dataset import cache_dir

FEATURE_SIZE = 768
GRID = 4


def laplacian_variance(gray: np.ndarray) -> float:
    """Sharpness. Blur removes high frequencies, and the Laplacian's variance drops with them."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dark_fraction(gray: np.ndarray, threshold: int = 64) -> float:
    """Exposure. The share of pixels darker than `threshold`."""
    return float((gray < threshold).mean())


# ---------------------------------------------------------------- more

def tenengrad(gray: np.ndarray) -> float:
    """Sharpness measured on gradients rather than on the second derivative; less noise-sensitive."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def contrast_p95_p5(gray: np.ndarray) -> float:
    """Dynamic range between paper and ink, robust to the few saturated pixels."""
    p5, p95 = np.percentile(gray, [5, 95])
    return float(p95 - p5)


def bright_fraction(gray: np.ndarray, threshold: int = 245) -> float:
    """Saturated pixels: a reflection or a blown-out page corner."""
    return float((gray > threshold).mean())


def _tiles(gray: np.ndarray, n: int = GRID):
    h, w = gray.shape
    for i in range(n):
        for j in range(n):
            yield gray[i * h // n:(i + 1) * h // n, j * w // n:(j + 1) * w // n]


def tile_lapvar_min(gray: np.ndarray) -> float:
    """The blurriest tile. A smear over one column drags this down and leaves the mean alone.

    The outermost ring of tiles is mostly background and its sharpness says
    nothing about the text, so the minimum is taken over the inner tiles only.
    """
    vals = np.array([cv2.Laplacian(t, cv2.CV_64F).var() for t in _tiles(gray)]).reshape(GRID, GRID)
    return float(vals[1:-1, 1:-1].min())


def tile_dark_max(gray: np.ndarray, threshold: int = 64) -> float:
    """The darkest tile. A shadow across one corner shows here, not in the whole-frame fraction."""
    return float(max((t < threshold).mean() for t in _tiles(gray)))


FEATURES = {
    "lap_var": laplacian_variance,
    "dark_frac": dark_fraction,
    "tenengrad": tenengrad,
    "contrast_p95_p5": contrast_p95_p5,
    "bright_frac": bright_fraction,
    "tile_lapvar_min": tile_lapvar_min,
    "tile_dark_max": tile_dark_max,
}


def extract(image_id: str, size: int = FEATURE_SIZE) -> dict:
    gray = cv2.imread(str(cache_dir(size) / f"{image_id}.jpg"), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"{image_id} not in the {size} px cache")
    return {name: fn(gray) for name, fn in FEATURES.items()}


def feature_table(df: pd.DataFrame, size: int = FEATURE_SIZE, workers: int = 4) -> pd.DataFrame:
    """One row of features per row of `df`, same index."""
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(lambda i: extract(i, size), df["image_id"]))
    return pd.DataFrame(rows, index=df.index)


def features_of_image(path: str, size: int = FEATURE_SIZE) -> dict:
    """The same features for an arbitrary image file, for prediction on photographs not in the cache.

    Resizes so the long side is `size`, exactly as the cache does, so a number
    computed here means the same thing as one computed from the cache.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape
    f = size / max(h, w)
    gray = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
    return {name: fn(gray) for name, fn in FEATURES.items()}

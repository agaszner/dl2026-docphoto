"""Turn a trained checkpoint into a Kaggle submission file.

    python -m src.predict --config configs/cnn.yaml \
                          --checkpoint models/cnn_best.pt \
                          --out submissions/cnn_best.csv

This file is a skeleton. The parts that define the *submission contract* are
written out, because getting them wrong wastes a submission. The parts that are
your model are marked TODO; for the lab 3 MLP they are: rebuild the model from
the checkpoint (it stores the config, the input size and the standardiser),
turn one image into pixels with `pixels_of_image`, standardise them with the
stored mean and scale, and run the model. This is homework, since the competition is not open
yet, and it is the first thing to test when it opens.

THE CONTRACT
------------
Output a CSV with exactly two columns and one row per test image:

    image_id,usability
    ph_00659482,0.8213
    ph_0112e28f,0.1044

* `image_id` is the test image filename **without its extension**, exactly as it
  appears in the competition's `sample_submission.csv`. Do not invent your own
  ids, do not reorder rows, and do not add an index column.
* `usability` is a float in [0, 1]. Clip it. A prediction of 1.3 is not a bolder
  guess, it is an error the metric will charge you for.
* Every id in `sample_submission.csv` must appear exactly once. A missing row is
  a rejected submission, not a zero.

THE RULE THAT SHAPES THIS FILE
------------------------------
Your prediction for an image must depend on **that image and your trained
weights only**. Not on other test images, not on any network call, not on an
OCR engine.

That is why the loop below runs one image at a time and why nothing is shared
between iterations. Batching for speed is fine, since a batch of independent
images still gives each its own answer. What is not fine is anything that pools
information across the test set. See section 2 of the project description.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_model(checkpoint: Path, cfg: dict):
    """Rebuild the architecture from the config and load the saved weights.

    Put the model in eval mode and disable gradients. Return something callable
    that maps one preprocessed image to one float.
    """
    raise NotImplementedError("TODO: build the model and load the checkpoint")


def preprocess(image_path: Path, cfg: dict):
    """Read one image and turn it into model input.

    This must match training exactly. A resize or a normalisation that differs
    between training and prediction is the single most common reason a good
    validation score turns into a bad leaderboard score.
    """
    raise NotImplementedError("TODO: load and preprocess a single image")


def predict_one(model, image_path: Path, cfg: dict) -> float:
    """One image in, one number out."""
    raise NotImplementedError("TODO: run the model, return a float in [0, 1]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a Kaggle submission file")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="the weights that will produce this submission")
    ap.add_argument("--test-dir", default="data/raw/test",
                    help="directory of test images")
    ap.add_argument("--sample", default="data/raw/sample_submission.csv",
                    help="the competition's sample submission; defines ids and row order")
    ap.add_argument("--out", required=True, help="where to write the submission CSV")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)

    # The sample submission is the source of truth for which ids exist and in
    # what order. Build the output from it rather than from a directory listing,
    # and the two can never disagree.
    sample = pd.read_csv(a.sample)
    if list(sample.columns) != ["image_id", "usability"]:
        raise SystemExit(f"unexpected sample columns: {list(sample.columns)}")

    model = load_model(Path(a.checkpoint), cfg)
    test_dir = Path(a.test_dir)

    predictions = []
    for image_id in sample["image_id"]:
        image_path = test_dir / f"{image_id}.jpg"
        if not image_path.exists():
            raise SystemExit(f"missing test image: {image_path}")
        predictions.append(predict_one(model, image_path, cfg))

    out = pd.DataFrame({"image_id": sample["image_id"], "usability": predictions})
    out["usability"] = out["usability"].clip(0.0, 1.0)

    # Checks worth running before every upload. Each of these has cost somebody
    # a submission slot at some point.
    assert len(out) == len(sample), "row count does not match the sample submission"
    assert out["image_id"].is_unique, "duplicate image_id"
    assert out["usability"].notna().all(), "NaN in predictions"
    assert out["usability"].between(0, 1).all(), "prediction outside [0, 1]"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} rows to {a.out}")
    print(out["usability"].describe())
    print("\nRemember: the checkpoint that produced this file is part of your "
          "submission. Keep it, and record which run it came from.")


if __name__ == "__main__":
    main()

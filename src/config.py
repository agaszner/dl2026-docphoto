"""The few settings every script shares.

Paths to data and hyperparameters belong in the YAML configs, so that an
experiment is described by one file rather than by whatever was typed on the
command line at the time. This module holds only what never changes.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The target is continuous on [0, 1]; the competition metric is RMSE.
TARGET = "usability"
SEED = 42

# The validation split has its own seed. It is part of how you measure, not of
# what you train: rerunning a model with another SEED to see its run-to-run
# variance must leave the validation photographs where they are. With the
# procedure in data/splits.py this value holds out strict unit 4: receipts and
# tables, about a fifth of the rows, both families still present in training.
SPLIT_SEED = 3

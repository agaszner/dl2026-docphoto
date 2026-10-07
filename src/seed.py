"""Make a run reproducible.

Call this once at the start of every script. An experiment you cannot repeat is
an anecdote, and you will want to compare runs all semester.

Fixing the seeds removes the randomness you control: initialisation, shuffling,
dropout. Some GPU kernels remain non-deterministic unless you also set
`torch.use_deterministic_algorithms(True)`, which costs speed; two runs with
the same seed therefore agree closely but not bit for bit.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

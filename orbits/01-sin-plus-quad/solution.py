"""Closed-form fit for the toy symbolic-regression benchmark.

Derived by eye from research/eval/train_data.csv:
  - Even part of y, (y(x)+y(-x))/2, tracks 0.1 * x^2 (endpoints ≈ 1.6 at |x|=4).
  - Odd part, (y(x)-y(-x))/2, tracks sin(x) (amplitude 1, one oscillation on [-4,4]).
No fitting loop, no sklearn, no scipy.optimize — pure symbolic guess.
"""

from __future__ import annotations

import numpy as np


def f(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sin(x) + 0.1 * x**2

# Toy Symbolic Regression — Closed-Form by Inspection

## Problem Statement
`research/eval/train_data.csv` contains 40 noisy `(x, y)` points on `x ∈ [-4, 4]`
with noise σ=0.03. Propose a closed-form `f(x)` that best fits the data.

Write the symbolic expression as Python in `solution.py`, exporting
`f(x: np.ndarray) -> np.ndarray`. Constraints:

- NO sklearn, NO fitting loops, NO scipy.optimize.
- Tune coefficients by inspection (eyeball), not `curve_fit`.
- Evaluator is pre-provided at `research/eval/evaluator.py` — DO NOT rebuild it.

## Solution Interface
`solution.py` must define `f(x: np.ndarray) -> np.ndarray`. The evaluator calls
`f(x_test)` on a clean held-out grid of 400 points on `[-4, 4]`.

## Success Metric
MSE on held-out test set (minimize). Target: MSE < 0.01.

## Budget
Max 2 orbits.

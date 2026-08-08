# Unlearning

Minimal subspace-based machine unlearning demo.

It recovers forget / retain subspaces with column-pivoted QR, removes the
retain components from the forget basis, then subtracts the remaining
directions from layer weights.

## Run

```bash
python scripts/demo_unlearning.py
```

Requires: `numpy`, `scipy`.

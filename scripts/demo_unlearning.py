#!/usr/bin/env python3
"""Tiny synthetic demo of subspace-based machine unlearning.

We craft forget / retain task matrices so that:
  - forget contains a unique direction u_f
  - retain contains a unique direction u_r
  - both share a common direction u_shared

The update should suppress the forget-only direction while largely
keeping the shared / retain directions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unlearning import select_forget_layers, subspace_unlearn_update


def main() -> None:
    rng = np.random.default_rng(0)
    d_out, d_in = 64, 32

    def unit(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)

    u_f = unit(rng.normal(size=d_out))
    u_r = unit(rng.normal(size=d_out))
    u_shared = unit(rng.normal(size=d_out))
    v = unit(rng.normal(size=d_in))

    T_f = 3.0 * np.outer(u_f, v) + 1.5 * np.outer(u_shared, v)
    T_r = 3.0 * np.outer(u_r, v) + 1.5 * np.outer(u_shared, v)
    W = rng.normal(scale=0.05, size=(d_out, d_in)) + T_f

    W_new = subspace_unlearn_update(W, T_f=T_f, T_r=T_r, rank=2)
    delta = W - W_new

    def proj_energy(mat: np.ndarray, u: np.ndarray) -> float:
        return float(np.linalg.norm(u @ mat))

    print("Subspace unlearning synthetic demo")
    print(f"  ||delta||_F           = {np.linalg.norm(delta):.4f}")
    print(
        "  forget-only energy    : "
        f"before={proj_energy(W, u_f):.4f}  after={proj_energy(W_new, u_f):.4f}"
    )
    print(
        "  retain-only energy    : "
        f"before={proj_energy(W, u_r):.4f}  after={proj_energy(W_new, u_r):.4f}"
    )
    print(
        "  shared energy         : "
        f"before={proj_energy(W, u_shared):.4f}  after={proj_energy(W_new, u_shared):.4f}"
    )

    layers = {
        "layer_high": T_f,
        "layer_mid": 0.3 * T_f,
        "layer_low": 0.05 * T_f,
    }
    selected = select_forget_layers(layers)
    print(f"  layer-localized pick  : {selected}")


if __name__ == "__main__":
    main()

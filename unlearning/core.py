"""Minimal subspace-based machine unlearning.

Core steps:
  1) Column-pivoted QR on forget / retain task matrices
  2) Remove retain-subspace components from the forget basis
  3) Update weights: W' = W - T_f_tilde
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import qr


def _cpqr_basis(
    T: np.ndarray, rank: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-pivoted QR: T Pi = Q R. Return leading Q, R, Pi."""
    Q, R, piv = qr(T, mode="economic", pivoting=True)
    k = T.shape[1] if rank is None else min(rank, T.shape[1], Q.shape[1])
    Qk = Q[:, :k]
    Rk = R[:k, :]
    Pi = np.eye(T.shape[1])[:, piv]
    return Qk, Rk, Pi


def subspace_unlearn_update(
    W: np.ndarray,
    T_f: np.ndarray,
    T_r: np.ndarray,
    rank: Optional[int] = None,
) -> np.ndarray:
    """Apply one-layer subspace unlearning update.

    Args:
        W: layer weights, shape (d_out, d_in)
        T_f: forget task matrix, same shape as W
        T_r: retain task matrix, same shape as W
        rank: optional low-rank truncation for CPQR

    Returns:
        Updated weights W' = W - T_f_tilde
    """
    if W.shape != T_f.shape or W.shape != T_r.shape:
        raise ValueError("W, T_f, T_r must share the same shape")

    Q_f, R_f, Pi_f = _cpqr_basis(T_f, rank=rank)
    Q_r, _, _ = _cpqr_basis(T_r, rank=rank)

    # Keep forget directions outside the retain subspace.
    Q_f_tilde = Q_f - Q_r @ (Q_r.T @ Q_f)
    T_f_tilde = Q_f_tilde @ R_f @ Pi_f.T
    return W - T_f_tilde


def select_forget_layers(task_matrices: Dict[str, np.ndarray]) -> List[str]:
    """Select layers with above-average forget-task energy."""
    energies = {name: float(np.linalg.norm(T, ord="fro")) for name, T in task_matrices.items()}
    total = sum(energies.values()) + 1e-12
    avg = 1.0 / max(len(energies), 1)
    return [name for name, e in energies.items() if (e / total) > avg]

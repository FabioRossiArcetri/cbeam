"""Noll-ordered Zernike polynomials sampled at the lantern's output cores.

Projecting the per-core signals onto this basis is a change of
representation, not a reduction: with 19 cores the sampled basis has rank 19
at most, so any number of modes beyond that spans the same space (and a
degree-d polynomial in the modal coefficients spans the same functions as a
degree-d polynomial in the raw signals).
"""
from __future__ import annotations

from math import factorial

import numpy as np


def noll_to_nm(j: int):
    """Noll index j >= 1 -> radial order n and signed azimuthal frequency m."""
    n = int(np.floor((-1 + np.sqrt(8 * (j - 1) + 1)) / 2))
    p = j - (n * (n + 1)) // 2 - 1
    m = 2 * ((p + 1) // 2) if n % 2 == 0 else 2 * (p // 2) + 1
    return n, (m if j % 2 == 0 else -m)


def radial(n: int, m: int, r: np.ndarray) -> np.ndarray:
    m = abs(m)
    R = np.zeros_like(r, dtype=float)
    for k in range((n - m) // 2 + 1):
        R += ((-1) ** k * factorial(n - k)
              / (factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k))
              ) * r ** (n - 2 * k)
    return R


def zernike(j: int, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """RMS-normalised Noll Zernike Z_j on the unit disk."""
    n, m = noll_to_nm(j)
    norm = np.sqrt(2 * (n + 1)) if m != 0 else np.sqrt(n + 1)
    return norm * radial(n, m, r) * (np.cos(m * theta) if m >= 0 else np.sin(-m * theta))


def zernike_basis(core_xy, n_modes: int, fill: float = 0.8, drop_piston: bool = False):
    """Sample Z_1..Z_n_modes (Z_2.. if drop_piston) at the cores.

    core_xy : (n_cores, 2) core positions, any length unit; centred and scaled
        so the outermost core sits at radius ``fill`` of the unit disk.

    Returns ``(B, names)``: B is (n_cores, n_modes) with B[i, k] = Z_k at core
    i, and names are "Z<j>".
    """
    xy = np.asarray(core_xy, float)
    xy = xy - xy.mean(axis=0)
    xy = xy / np.linalg.norm(xy, axis=1).max() * fill
    r, theta = np.linalg.norm(xy, axis=1), np.arctan2(xy[:, 1], xy[:, 0])
    js = np.arange(2 if drop_piston else 1, n_modes + 1)
    B = np.stack([zernike(j, r, theta) for j in js], axis=1)
    return B, [f"Z{j}" for j in js]


def modal_projector(B: np.ndarray, rcond: float = 1e-3) -> np.ndarray:
    """Minimum-norm least-squares projector (n_modes, n_cores): coeffs = P @ s."""
    return np.linalg.pinv(B, rcond=rcond)

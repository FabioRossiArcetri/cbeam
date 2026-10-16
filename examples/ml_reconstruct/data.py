"""Datasets of (lantern signals, aberration coefficients) and their scaling."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LanternDataset:
    """Simulated lantern measurements and the aberrations that produced them.

    signals : (n, n_cores) at one wavelength, or (n, n_wl, n_cores) spectral
    coeff   : (n, n_modes) DM mode coefficients [nm]
    n_train : the first n_train samples are the training split, the rest
              validation (the samples are i.i.d., so this split is random)
    core_xy : (n_cores, 2) physical position [um] of the core behind each
              signal column, at the lantern output
    wavelengths_nm : (n_wl,) wavelengths of a spectral dataset's axis 1
    meta    : free-form description (units, how the signals were measured)
    """
    signals: np.ndarray
    coeff: np.ndarray
    n_train: int
    core_xy: np.ndarray
    wavelengths_nm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    meta: dict = field(default_factory=dict)

    @property
    def train(self):
        return self.signals[:self.n_train], self.coeff[:self.n_train]

    @property
    def val(self):
        return self.signals[self.n_train:], self.coeff[self.n_train:]

    def save(self, path: str) -> None:
        np.savez(path, signals=self.signals, coeff=self.coeff, n_train=self.n_train,
                 core_xy=self.core_xy, wavelengths_nm=self.wavelengths_nm,
                 meta_keys=np.asarray(list(self.meta), dtype=str),
                 meta_values=np.asarray([str(v) for v in self.meta.values()], dtype=str))

    @classmethod
    def load(cls, path: str) -> "LanternDataset":
        with np.load(path) as f:
            meta = dict(zip(f["meta_keys"].tolist(), f["meta_values"].tolist()))
            return cls(signals=f["signals"], coeff=f["coeff"], n_train=int(f["n_train"]),
                       core_xy=f["core_xy"], wavelengths_nm=f["wavelengths_nm"], meta=meta)


class Standardizer:
    """Per-column zero mean / unit variance, with statistics from the data
    passed to ``fit`` only (the training split -- the validation split must
    not leak into the scaling)."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def fit(self, A):
        A = np.asarray(A, float)
        self.mean = A.mean(axis=0)
        self.std = A.std(axis=0) + self.eps
        return self

    def transform(self, A):
        return (np.asarray(A, float) - self.mean) / self.std

    def inverse(self, Z):
        return np.asarray(Z) * self.std + self.mean


def subsample_rows(n: int, share: float, seed: int = 0) -> np.ndarray:
    """Sorted random subset of round(share * n) of range(n), without
    replacement; all rows if share >= 1."""
    n_want = max(1, int(round(share * n)))
    if n_want >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=n_want, replace=False))


def rmse_nm(pred, true) -> np.ndarray:
    """Per-column RMSE of predictions vs truth, both in nm."""
    return np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2, axis=0))

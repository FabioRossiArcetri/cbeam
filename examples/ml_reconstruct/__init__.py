"""ml_reconstruct -- wavefront (DM-mode) reconstruction from photonic-lantern
signals, used by the trainMLP*.ipynb / PL19_EML.ipynb notebooks.

    data      LanternDataset (signals + coefficients + core positions), Standardizer
    simulate  dataset generation with batch_pipeline / multi_wvl
              (imported on demand: it needs the full cbeam stack)
    mlp       JAX MLP, Adam training, save / load
    sparse    polynomial features, dense ridge and OMP (term-budget) fits
    zernike   Noll Zernike basis sampled at the output cores
    unwrap    per-wavelength wrapped-phase models + multi-wavelength unwrap
    cache     fit-or-load ModelCache
"""
from __future__ import annotations

from .data import LanternDataset, Standardizer, subsample_rows, rmse_nm
from .cache import ModelCache
from .mlp import init_mlp_params, mlp_forward, huber_loss, train_mlp, save_mlp, load_mlp
from .sparse import (
    n_poly_terms, poly_features, omp_fit, omp_predict, fit_omp_poly, predict_omp_poly,
    n_terms, fit_dense_poly_ridge, predict_dense_poly_ridge,
)
from .zernike import noll_to_nm, zernike, zernike_basis, modal_projector
from .unwrap import (
    wrap, phasor_targets, phasor_decode, per_wavelength_features, fit_wrapped_models,
    unwrap, unwrap_all,
)

__all__ = [
    "LanternDataset", "Standardizer", "subsample_rows", "rmse_nm", "ModelCache",
    "init_mlp_params", "mlp_forward", "huber_loss", "train_mlp", "save_mlp", "load_mlp",
    "n_poly_terms", "poly_features", "omp_fit", "omp_predict", "fit_omp_poly",
    "predict_omp_poly", "n_terms", "fit_dense_poly_ridge", "predict_dense_poly_ridge",
    "noll_to_nm", "zernike", "zernike_basis", "modal_projector",
    "wrap", "phasor_targets", "phasor_decode", "per_wavelength_features",
    "fit_wrapped_models", "unwrap", "unwrap_all",
]

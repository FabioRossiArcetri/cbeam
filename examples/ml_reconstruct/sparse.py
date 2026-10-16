"""Polynomial-feature regressions: dense ridge and sparse (OMP) fits.

The sparse model is Orthogonal Matching Pursuit with a hard term budget K,
run on a candidate pool pre-screened by marginal correlation with the
target -- the explicit, explainable alternative to the MLP.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore", message="Orthogonal matching pursuit ended prematurely")


def n_poly_terms(n_inputs: int, degree: int) -> int:
    """Number of monomials of degree 1..degree in n_inputs variables."""
    from math import comb
    return comb(n_inputs + degree, degree) - 1


def poly_features(F_tr, F_va, degree: int, copy: bool = True):
    """Standardised degree-``degree`` polynomial features (no bias column).

    Returns ``(Z_tr, Z_va, scaler, poly)``. ``copy=False`` standardises in
    place, halving the peak memory for large expansions.
    """
    pf = PolynomialFeatures(degree, include_bias=False)
    P_tr = pf.fit_transform(F_tr)
    sc = StandardScaler(copy=copy).fit(P_tr)
    return sc.transform(P_tr), sc.transform(pf.transform(F_va)), sc, pf


def omp_fit(Z, y, K: int, screen: int):
    """OMP with at most K terms, chosen from the ``screen`` columns of Z most
    correlated with y. Returns ``(omp, sel)``; predict with
    ``omp.predict(Z_new[:, sel])``."""
    yc = y - y.mean()
    sel = np.argsort(-np.abs(Z.T @ yc))[:min(screen, Z.shape[1])]
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=min(K, len(sel) - 1)).fit(Z[:, sel], y)
    return omp, sel


def omp_predict(Z, fit):
    omp, sel = fit
    return omp.predict(Z[:, sel])


def fit_omp_poly(F_tr, Y_tr, degree: int, K: int, screen: int):
    """Polynomial features + one OMP per target column.
    Returns ``(poly, scaler, [(omp, sel), ...])``."""
    pf = PolynomialFeatures(degree, include_bias=False)
    Z = pf.fit_transform(F_tr)
    sc = StandardScaler(copy=False).fit(Z)
    Z = sc.transform(Z)
    Y_tr = np.asarray(Y_tr).reshape(len(Z), -1)
    return pf, sc, [omp_fit(Z, Y_tr[:, j], K, screen) for j in range(Y_tr.shape[1])]


def predict_omp_poly(model, F):
    pf, sc, fits = model
    Z = sc.transform(pf.transform(F))
    return np.stack([omp_predict(Z, f) for f in fits], axis=1)


def n_terms(fit) -> int:
    """Number of non-zero terms of one (omp, sel) fit."""
    return int(np.count_nonzero(fit[0].coef_))


def fit_dense_poly_ridge(F_tr, Y_tr, degree: int, alphas=np.logspace(-2, 5, 15)):
    """Every degree-``degree`` monomial, standardised, and a ridge regression
    with the penalty chosen per target column by efficient leave-one-out
    (one factorisation for all targets). Returns ``(poly, scaler, ridge)``."""
    pf = PolynomialFeatures(degree, include_bias=False)
    Z = pf.fit_transform(F_tr)
    sc = StandardScaler(copy=False).fit(Z)
    Z = sc.transform(Z)
    Y_tr = np.asarray(Y_tr).reshape(len(Z), -1)
    return pf, sc, RidgeCV(alphas=alphas, alpha_per_target=True).fit(Z, Y_tr)


def predict_dense_poly_ridge(model, F):
    pf, sc, ridge = model
    return ridge.predict(sc.transform(pf.transform(F))).reshape(len(F), -1)

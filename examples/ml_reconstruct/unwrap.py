"""Large pistons from wrapped per-wavelength estimates.

At one wavelength the lantern signals depend on a piston only modulo the
wavelength (the pupil phase is 2*pi*p/lambda), so a per-wavelength model can
recover only the *wrapped* piston. It regresses the smooth phasor
(sin, cos)(2*pi*p/T) rather than the sawtooth wrap(p, T), and the estimates
from many wavelengths are then fused into the full-range piston by
``unwrap`` (a circular maximum-likelihood search followed by a robust
refinement) -- the multi-wavelength analogue of synthetic-wavelength
interferometry.
"""
from __future__ import annotations

import time

import numpy as np

from .sparse import fit_omp_poly, predict_omp_poly


def wrap(x, T):
    """Map x into [-T/2, T/2), modulo T (T scalar or broadcastable)."""
    return x - T * np.round(x / T)


def phasor_targets(P, T):
    """(n, M) pistons -> (n, 2M) targets [sin, cos](2*pi*P/T)."""
    ph = 2 * np.pi * P / T
    return np.concatenate([np.sin(ph), np.cos(ph)], axis=1)


def phasor_decode(S, T):
    """(n, 2M) predicted phasors -> (wrapped piston in [-T/2, T/2), confidence),
    the confidence being the phasor length (~1 when the model is sure)."""
    M = S.shape[1] // 2
    s, c = S[:, :M], S[:, M:]
    return np.arctan2(s, c) * T / (2 * np.pi), np.hypot(s, c)


def per_wavelength_features(X3, w, fib_use, nb_wl=0):
    """(n, n_wl, n_fib) signals -> (n, d) inputs of the model at wavelength
    index w: fibres fib_use at wavelengths w-nb_wl .. w+nb_wl (clipped)."""
    lo, hi = max(0, w - nb_wl), min(X3.shape[1], w + nb_wl + 1)
    return X3[:, lo:hi][:, :, fib_use].reshape(len(X3), -1)


def fit_wrapped_models(X3_tr, X3_va, P_tr, P_va, periods, wl_use, fib_use, *,
                       degree, K, screen, cache, nb_wl=0, rows=None, tag=""):
    """One degree-``degree`` OMP polynomial model per wavelength in wl_use,
    regressing the phasor of the piston wrapped at ``periods[w]``.

    X3_* : (n, n_wl, n_fib) standardised signals; P_* : (n, M) pistons [nm].
    rows : optional subset of training rows to fit on.

    Returns a dict: ``ghat_va`` / ``conf_va`` (n_va, M, n_use) wrapped
    estimates and confidences, ``sig_tr`` / ``sig_va`` (n_use, M) wrapped RMSE,
    ``periods`` (n_use,), ``wl_use``, ``seconds``.
    """
    wl_use = np.asarray(wl_use, int)
    rows = np.arange(len(X3_tr)) if rows is None else np.asarray(rows)
    M, n_va = P_tr.shape[1], len(X3_va)
    ghat_va = np.zeros((n_va, M, len(wl_use)))
    conf_va = np.zeros_like(ghat_va)
    sig_tr = np.zeros((len(wl_use), M))
    sig_va = np.zeros((len(wl_use), M))
    t0 = time.time()
    for i, w in enumerate(wl_use):
        T = periods[w]
        F_tr = per_wavelength_features(X3_tr, w, fib_use, nb_wl)[rows]
        F_va = per_wavelength_features(X3_va, w, fib_use, nb_wl)
        model = cache(f"{tag}-w{w}", lambda F_tr=F_tr, T=T: fit_omp_poly(
            F_tr, phasor_targets(P_tr[rows], T), degree, K, screen))
        g_va, c_va = phasor_decode(predict_omp_poly(model, F_va), T)
        g_tr, _ = phasor_decode(predict_omp_poly(model, F_tr), T)
        ghat_va[:, :, i], conf_va[:, :, i] = g_va, c_va
        sig_va[i] = np.sqrt(np.mean(wrap(g_va - wrap(P_va, T), T) ** 2, axis=0))
        sig_tr[i] = np.sqrt(np.mean(wrap(g_tr - wrap(P_tr[rows], T), T) ** 2, axis=0))
    return dict(ghat_va=ghat_va, conf_va=conf_va, sig_tr=sig_tr, sig_va=sig_va,
                periods=np.asarray(periods)[wl_use], wl_use=wl_use,
                seconds=time.time() - t0)


def unwrap(ghat, conf, lam, sig, pspan, conf_min=0.15, huber_c=2.0, n_irls=12, chunk=4000):
    """Fuse wrapped estimates of ONE piston into its full-range value.

    ghat, conf : (S, W) wrapped estimates / confidences at W wrap periods lam (W,)
    sig        : (W,) per-period wrapped RMSE (sets each period's weight)
    pspan      : the piston is searched in [-pspan, pspan]

    Cost J(p) = -sum_w W_sw cos(phi_sw - 2*pi*p/lam_w) (von Mises / circular
    ML), W_sw = conf_sw * (lam_w / (2*pi*sig_w))^2 with low-confidence
    estimates gated out; minimised on a coarse grid over the whole range, then
    on a fine grid around the coarse minimum, then polished by IRLS/Huber
    with per-wavelength integer fringe fixing.

    Returns ``(p, q)``: the estimates (S,) and an agreement score in [0, 1]
    (the fraction of used wavelengths consistent with p within lam/4).
    """
    S, Wn = ghat.shape
    phi = 2 * np.pi * ghat / lam[None, :]
    kappa = (lam / (2 * np.pi * np.maximum(sig, 1e-6))) ** 2
    base = np.where(conf >= conf_min, conf, 0.0) * kappa[None, :]
    base = np.where(base.sum(1, keepdims=True) > 0, base, kappa[None, :])  # all gated -> ungate
    wn = (2 * np.pi / lam)[None, :, None]
    mp = float(lam.mean())

    def vm_cost(cand, phi_c, base_c):                       # cand (c, G) -> (c, G)
        return -np.einsum("cw,cwg->cg", base_c,
                          np.cos(phi_c[:, :, None] - wn * cand[:, None, :]))

    # stage 1: coarse grid over the whole range
    step = max(lam.min() / 4, 2 * pspan / 6000)
    g1 = np.arange(-pspan, pspan + step, step)
    p0 = np.empty(S)
    for a in range(0, S, chunk):
        phi_c, base_c = phi[a:a + chunk], base[a:a + chunk]
        cost = vm_cost(np.broadcast_to(g1, (len(phi_c), g1.size)), phi_c, base_c)
        p0[a:a + chunk] = g1[cost.argmin(1)]

    # stage 2: fine grid, +-2.5 mean periods around the coarse minimum
    win, sf = 2.5 * mp, max(lam.min() / 30, mp / 400)
    rel = np.arange(-win, win + sf, sf)
    p1 = np.empty(S)
    for a in range(0, S, chunk):
        phi_c, base_c = phi[a:a + chunk], base[a:a + chunk]
        cand = np.clip(p0[a:a + chunk, None] + rel[None, :], -pspan, pspan)
        p1[a:a + chunk] = cand[np.arange(len(cand)), vm_cost(cand, phi_c, base_c).argmin(1)]

    # stage 3: IRLS / Huber weighted mean with integer fringe fixing
    p = p1
    for _ in range(n_irls):
        c = ghat + np.round((p[:, None] - ghat) / lam[None, :]) * lam[None, :]
        r = c - p[:, None]
        s = 1.4826 * np.median(np.abs(r), axis=1, keepdims=True) + 1e-9
        wt = base * np.minimum(1.0, huber_c * s / np.maximum(np.abs(r), 1e-9))
        den = wt.sum(1)
        p_new = np.where(den > 0, (wt * c).sum(1) / np.maximum(den, 1e-12), p)
        if np.max(np.abs(p_new - p)) < 1e-6:
            p = p_new
            break
        p = p_new

    # stage 4: per-sample agreement score
    c = ghat + np.round((p[:, None] - ghat) / lam[None, :]) * lam[None, :]
    used = base > 0
    q = (np.abs(c - p[:, None]) < lam[None, :] / 4) & used
    q = q.sum(1) / np.maximum(used.sum(1), 1)
    return np.clip(p, -1.5 * pspan, 1.5 * pspan), q


def unwrap_all(fit: dict, pspan: float, **kwargs):
    """unwrap() every piston of a fit_wrapped_models() result.
    Returns ``(P_hat, Q)``, both (n_va, M)."""
    M = fit["ghat_va"].shape[1]
    out = [unwrap(fit["ghat_va"][:, k, :], fit["conf_va"][:, k, :], fit["periods"],
                  fit["sig_tr"][:, k], pspan, **kwargs) for k in range(M)]
    return np.stack([o[0] for o in out], axis=1), np.stack([o[1] for o in out], axis=1)

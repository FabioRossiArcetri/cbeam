"""Unit tests for ``examples/ml_reconstruct`` on small synthetic problems.

Skipped when jax, optax or scikit-learn are not installed (they are optional
dependencies of the ML examples only).
"""

import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")
pytest.importorskip("sklearn")

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import ml_reconstruct as ml  # noqa: E402
from cbeam.waveguide import hex_ring_positions  # noqa: E402


# --------------------------------------------------------------------------- #
# zernike
# --------------------------------------------------------------------------- #
def test_noll_indices():
    expected = {1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0), 5: (2, -2), 6: (2, 2),
                7: (3, -1), 8: (3, 1), 11: (4, 0)}
    for j, nm in expected.items():
        assert ml.noll_to_nm(j) == nm


def test_zernike_basis_at_19_cores():
    xy = np.asarray(hex_ring_positions(3, 3.6))
    B, names = ml.zernike_basis(xy, 36)
    assert B.shape == (19, 36) and names[0] == "Z1"
    assert np.linalg.matrix_rank(B) == 19          # 19 samples: rank 19 at most
    np.testing.assert_allclose(B[:, 0], 1.0)       # piston
    # rank 19 = n_cores: projecting any core signal and expanding it back is lossless
    s = np.random.default_rng(0).normal(size=19)
    np.testing.assert_allclose(B @ (ml.modal_projector(B) @ s), s, atol=1e-8)


# --------------------------------------------------------------------------- #
# sparse polynomial fits
# --------------------------------------------------------------------------- #
def test_omp_recovers_sparse_polynomial():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(2000, 4))
    y = 2 * F[:, 0] - 3 * F[:, 1] * F[:, 2] + 0.5 * F[:, 3] ** 3
    model = ml.fit_omp_poly(F[:1500], y[:1500], degree=3, K=3, screen=34)
    assert ml.n_terms(model[2][0]) == 3
    np.testing.assert_allclose(ml.predict_omp_poly(model, F[1500:])[:, 0], y[1500:], atol=1e-8)
    assert ml.n_poly_terms(4, 3) == model[0].n_output_features_ == 34


def test_dense_poly_ridge():
    rng = np.random.default_rng(1)
    F = rng.normal(size=(1500, 3))
    Y = np.stack([F[:, 0] * F[:, 1], F[:, 2] ** 2 - F[:, 0]], axis=1)
    model = ml.fit_dense_poly_ridge(F[:1000], Y[:1000], degree=2, alphas=[1e-6, 1e-3])
    np.testing.assert_allclose(ml.predict_dense_poly_ridge(model, F[1000:]), Y[1000:], atol=1e-3)


# --------------------------------------------------------------------------- #
# unwrap
# --------------------------------------------------------------------------- #
def test_phasor_round_trip():
    P = np.array([[-150.0, 0.0, 380.0]])
    g, conf = ml.phasor_decode(ml.phasor_targets(P, 800.0), 800.0)
    np.testing.assert_allclose(g, ml.wrap(P, 800.0), atol=1e-9)
    np.testing.assert_allclose(conf, 1.0)


def test_unwrap_recovers_large_pistons():
    rng = np.random.default_rng(2)
    lam = np.linspace(700, 900, 11)
    p = rng.uniform(-5000, 5000, size=500)
    sig = np.full(len(lam), 5.0)
    ghat = ml.wrap(p[:, None] + rng.normal(scale=sig, size=(len(p), len(lam))), lam)
    p_hat, q = ml.unwrap(ghat, np.ones_like(ghat), lam, sig, pspan=5000.0)
    assert np.mean(np.abs(p_hat - p) > lam.min() / 2) == 0      # no wrong fringe
    assert ml.rmse_nm(p_hat, p) < 3.0                            # ~5 nm / sqrt(11)
    assert q.min() > 0.9


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def test_dataset_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    ds = ml.LanternDataset(signals=rng.random((10, 3, 19)), coeff=rng.normal(size=(10, 5)),
                           n_train=8, core_xy=rng.random((19, 2)),
                           wavelengths_nm=np.array([700.0, 800.0, 900.0]), meta={"measurement": "x"})
    path = str(tmp_path / "ds.npz")
    ds.save(path)
    back = ml.LanternDataset.load(path)
    for name in ("signals", "coeff", "core_xy", "wavelengths_nm"):
        np.testing.assert_array_equal(getattr(back, name), getattr(ds, name))
    assert back.n_train == 8 and back.meta == {"measurement": "x"}
    assert len(back.train[0]) == 8 and len(back.val[0]) == 2


def test_standardizer_uses_training_statistics_only():
    tr, va = np.array([[0.0], [2.0]]), np.array([[100.0]])
    s = ml.Standardizer(eps=0.0).fit(tr)
    np.testing.assert_allclose(s.transform(va), [[99.0]])
    np.testing.assert_allclose(s.inverse(s.transform(va)), va)


def test_subsample_rows():
    a, b = ml.subsample_rows(100, 0.2, seed=0), ml.subsample_rows(100, 0.2, seed=0)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 20 and np.all(np.diff(a) > 0)
    np.testing.assert_array_equal(ml.subsample_rows(5, 1.0), np.arange(5))


# --------------------------------------------------------------------------- #
# mlp
# --------------------------------------------------------------------------- #
def test_train_mlp_and_save_load(tmp_path):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(600, 6)).astype(np.float32)
    Y = np.stack([np.sin(X[:, 0]) + X[:, 1] * X[:, 2], X[:, 3] ** 2], axis=1).astype(np.float32)
    params, hist = ml.train_mlp([6, 32, 32, 2], X[:500], Y[:500], X[500:], Y[500:],
                                epochs=60, batch_size=50, learning_rate=3e-3, log_every=0)
    assert len(hist["val"]) == 60 and 1 <= hist["best_epoch"] <= 60
    assert min(hist["val"]) < 0.2 * hist["val"][0]
    best = float(ml.huber_loss(Y[500:], ml.mlp_forward(params, X[500:])))
    assert best == pytest.approx(min(hist["val"]), rel=1e-4)    # the best epoch's weights

    path = str(tmp_path / "mlp.npz")
    ml.save_mlp(path, params, [6, 32, 32, 2], best_epoch=hist["best_epoch"])
    back, layers, extra = ml.load_mlp(path)
    assert layers == [6, 32, 32, 2] and int(extra["best_epoch"]) == hist["best_epoch"]
    np.testing.assert_allclose(ml.mlp_forward(back, X[500:]), ml.mlp_forward(params, X[500:]))

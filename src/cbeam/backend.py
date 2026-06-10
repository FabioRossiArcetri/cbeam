import os

backend_choice = os.environ.get("CBEAM_BACKEND", "numpy").lower()
using_jax = backend_choice == "jax"

if using_jax:
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as xp
    print("Using JAX backend.")
    try:
        import diffrax
        print("Using diffrax for ODE integration.")
    except ImportError:
        raise ImportError("Please install diffrax: pip install diffrax")
else:
    import numpy as xp
    import scipy
    import scipy.integrate
    import scipy.interpolate


def get_xp():
    return xp


def get_backend():
    return "jax" if using_jax else "numpy"


def get_jax_device():
    """Return the JAX device selected by CBEAM_JAX_DEVICE_INDEX (default 0).

    Falls back gracefully when the requested index does not exist (e.g. no GPU
    or only one GPU is available).
    """
    if not using_jax:
        return None
    import jax
    idx = int(os.environ.get("CBEAM_JAX_DEVICE_INDEX", "0"))
    devices = jax.devices()
    if idx >= len(devices):
        idx = 0
    return devices[idx]


# ---------------------------------------------------------------------------
# solve_ivp shim
# ---------------------------------------------------------------------------

def solve_ivp(fun, t_span, y0, **kwargs):
    """Unified solve_ivp wrapper.

    For the numpy backend this is a thin pass-through to scipy.integrate.solve_ivp.
    For the JAX backend it wraps diffrax.diffeqsolve.

    Memory note: by default only the final state is saved (SaveAt(t1=True)).
    Pass ``t_eval`` to request specific save-points (still no full-step buffer).
    The old ``SaveAt(t1=True, steps=True)`` was the primary cause of OOM errors
    because it allocated a buffer for every adaptive micro-step.
    """
    if using_jax:
        import diffrax

        t0, t1 = float(t_span[0]), float(t_span[1])
        dt0 = kwargs.get("dt0", abs(t1 - t0) * 0.01)

        diffrax_fun = lambda t, y, args=None: fun(t, y)
        term = diffrax.ODETerm(diffrax_fun)

        t_eval = kwargs.get("t_eval", None)
        if t_eval is not None:
            saveat = diffrax.SaveAt(ts=xp.asarray(t_eval, dtype=xp.float64))
        else:
            # Save only the endpoint — avoids allocating a per-step buffer.
            saveat = diffrax.SaveAt(t1=True)
            # saveat=diffrax.SaveAt(steps=True) # for fair comparison with numpy path


        rtol = kwargs.get("rtol", 1e-12)
        atol = kwargs.get("atol", 1e-10)
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

        out = diffrax.diffeqsolve(
            term,
            diffrax.Dopri5(),
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=200_000,
        )
        return out
    else:
        return scipy.integrate.solve_ivp(fun, t_span, y0, **kwargs)


# ---------------------------------------------------------------------------
# _JAXCubicSpline
# ---------------------------------------------------------------------------

class _JAXCubicSpline:
    """
    Minimal natural cubic spline for JAX, matching the scipy.interpolate.CubicSpline
    interface needed by the propagator: evaluation, .derivative(), .antiderivative().

    Supports N-d output arrays via vectorisation over trailing axes.
    ``y`` shape: (n_points, *trailing_dims)
    """

    def __init__(self, xs, ys, axis=0, **kwargs):
        ys = xp.moveaxis(xp.asarray(ys, dtype=xp.float64), axis, 0)
        xs = xp.asarray(xs, dtype=xp.float64)
        self.xs = xs
        self.ys = ys
        self.axis = 0
        self._coeffs = self._fit(xs, ys)

    @staticmethod
    def _fit(xs, ys):
        n = xs.shape[0]
        h  = xp.diff(xs)
        dy = xp.diff(ys, axis=0)

        shape_tail = ys.shape[1:]

        import numpy as np
        h_np  = np.array(h)
        dy_np = np.array(dy)

        m_np = np.zeros((n,) + shape_tail)

        if n > 2:
            c_prime = np.zeros(n)
            d_prime = np.zeros((n,) + shape_tail)

            denom      = 2.0 * (h_np[0] + h_np[1])
            c_prime[1] = h_np[1] / denom
            rhs        = 3.0 * (dy_np[1] / h_np[1] - dy_np[0] / h_np[0])
            d_prime[1] = rhs / denom

            for i in range(2, n - 1):
                denom      = 2.0 * (h_np[i-1] + h_np[i]) - h_np[i-1] * c_prime[i-1]
                c_prime[i] = h_np[i] / denom
                rhs        = 3.0 * (dy_np[i] / h_np[i] - dy_np[i-1] / h_np[i-1])
                d_prime[i] = (rhs - h_np[i-1] * d_prime[i-1]) / denom

            m_np[n-2] = d_prime[n-2]
            for i in range(n-3, 0, -1):
                m_np[i] = d_prime[i] - c_prime[i] * m_np[i+1]

        m = xp.array(m_np)

        sl = tuple([slice(None)] + [None] * len(shape_tail))
        a  = ys[:-1]
        b  = dy / h[sl] - h[sl] * (2 * m[:-1] + m[1:]) / 3
        c  = m[:-1]
        d  = (m[1:] - m[:-1]) / (3 * h[sl])
        return a, b, c, d

    def _eval(self, z, coeffs):
        a, b, c, d = coeffs
        xs = self.xs
        z  = xp.clip(z, xs[0], xs[-1])
        idx = xp.searchsorted(xs, z, side="right") - 1
        idx = xp.clip(idx, 0, xs.shape[0] - 2)
        dx  = z - xs[idx]
        return a[idx] + dx * (b[idx] + dx * (c[idx] + dx * d[idx]))

    def __call__(self, z):
        return self._eval(z, self._coeffs)

    def derivative(self):
        a, b, c, d = self._coeffs
        deriv_coeffs = (b, 2 * c, 3 * d, xp.zeros_like(d))
        parent = self

        class _Deriv:
            def __call__(self_, z):
                b2, c2, d2, _ = deriv_coeffs
                xs  = parent.xs
                z_  = xp.clip(z, xs[0], xs[-1])
                idx = xp.searchsorted(xs, z_, side="right") - 1
                idx = xp.clip(idx, 0, xs.shape[0] - 2)
                dx  = z_ - xs[idx]
                return b2[idx] + dx * (c2[idx] + dx * d2[idx])

        return _Deriv()

    def antiderivative(self):
        a, b, c, d = self._coeffs
        xs         = self.xs
        h          = xp.diff(xs)
        shape_tail = a.shape[1:]
        sl         = tuple([slice(None)] + [None] * len(shape_tail))

        seg_integrals = (
            a * h[sl]
            + b * h[sl] ** 2 / 2
            + c * h[sl] ** 3 / 3
            + d * h[sl] ** 4 / 4
        )
        cumulative = xp.concatenate(
            [xp.zeros((1,) + shape_tail), xp.cumsum(seg_integrals, axis=0)], axis=0
        )
        parent = self

        class _Antideriv:
            def __call__(self_, z):
                a2, b2, c2, d2 = parent._coeffs
                xs2 = parent.xs
                z_  = xp.clip(z, xs2[0], xs2[-1])
                idx = xp.searchsorted(xs2, z_, side="right") - 1
                idx = xp.clip(idx, 0, xs2.shape[0] - 2)
                dx  = z_ - xs2[idx]
                seg = cumulative[idx]
                return seg + dx * (a2[idx] + dx * (b2[idx] / 2 + dx * (c2[idx] / 3 + dx * d2[idx] / 4)))

        return _Antideriv()


# ---------------------------------------------------------------------------
# Public interpolation helpers
# ---------------------------------------------------------------------------

def myCubicSpline(x, y, axis=0, **kwargs):
    if using_jax:
        return _JAXCubicSpline(x, y, axis=axis)
    else:
        return scipy.interpolate.CubicSpline(x, y, axis=axis, **kwargs)


def UnivariateSpline(x, y, **kwargs):
    if using_jax:
        return _JAXCubicSpline(x, y, axis=0)
    else:
        return scipy.interpolate.UnivariateSpline(x, y, **kwargs)


def interp1d(x, y, kind="linear", axis=0, **kwargs):
    if using_jax:
        if kind in ("cubic", 3):
            return _JAXCubicSpline(x, y, axis=axis)
        x = xp.asarray(x, dtype=xp.float64)
        y = xp.asarray(y, dtype=xp.float64)

        def interp_func(x_new):
            x_new = xp.asarray(x_new, dtype=xp.float64)
            idx   = xp.searchsorted(x, x_new, side="right") - 1
            idx   = xp.clip(idx, 0, x.shape[0] - 2)
            x0, x1 = x[idx], x[idx + 1]
            y0 = y[idx]
            y1 = y[idx + 1]
            t  = (x_new - x0) / (x1 - x0)
            return y0 + t[..., *([None] * (y.ndim - 1))] * (y1 - y0)

        return interp_func
    else:
        return scipy.interpolate.interp1d(x, y, kind=kind, axis=axis, **kwargs)
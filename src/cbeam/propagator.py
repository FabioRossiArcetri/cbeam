from __future__ import annotations

import copy
import os
import time
from bisect import bisect_left
from typing import Union

import matplotlib.pyplot as plt
import numpy as _np          # always plain numpy — used for matplotlib helpers
from matplotlib.colors import ListedColormap
from matplotlib.tri import Triangulation
from matplotlib.widgets import Slider

from .backend import (
    get_xp, get_backend,
    solve_ivp, interp1d, myCubicSpline, UnivariateSpline,
)

from wavesolve.fe_solver import solve_waveguide, get_eff_index, construct_B, plot_scalar_mode
from cbeam.waveguide import load_meshio_mesh, Waveguide, plot_mesh
from cbeam import FEval

from scipy.interpolate import make_interp_spline  # noqa: F401 (kept for external callers)
import numpy as np

# ---------------------------------------------------------------------------
# Module-level colourmap — always built with plain numpy; must not hold any
# JAX-device tensor so it is safe to construct at import time.
# ---------------------------------------------------------------------------
_normcmap_data = _np.zeros([256, 4])
_normcmap_data[:, 3] = _np.linspace(0, 1, 256)[::-1]
normcmap = ListedColormap(_normcmap_data)


# ---------------------------------------------------------------------------
# Free-standing plot helpers
# ---------------------------------------------------------------------------

def plot_cfield(field, mesh, fig=None, ax=None, show_mesh=False,
                res=1., xlim=None, ylim=None):
    xp   = get_xp()
    show = False
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        show = True
    xm   = xp.max(mesh.points[:, 0])
    ym   = xp.max(mesh.points[:, 1])
    xlim = (-xm, xm) if xlim is None else xlim
    ylim = (-ym, ym) if ylim is None else ylim
    xa   = xp.arange(*xlim, res, dtype=xp.float64)
    ya   = xp.arange(*ylim, res, dtype=xp.float64)
    if not hasattr(mesh, "tree"):
        FEval.sort_mesh(mesh)
    fgrid  = xp.array(FEval.evaluate_grid(xa, ya, field, mesh.tree)).T
    alphas = xp.abs(fgrid)
    alphas /= xp.max(alphas)
    ax.set_facecolor("k")
    im = ax.imshow(xp.angle(fgrid), extent=(*xlim, *ylim), cmap="hsv",
                   vmin=-xp.pi, vmax=xp.pi, origin="lower")
    ax.imshow(alphas, extent=(*xlim, *ylim), cmap=normcmap,
              interpolation="bicubic", origin="lower")
    if show_mesh:
        plot_mesh(mesh, plot_points=False, ax=ax, alpha=0.1, verbose=False)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if fig is not None:
        fig.colorbar(im, ax=ax, label="phase")
    if show:
        plt.show()
    return im


def plot_field(field, mesh, ax=None, show_mesh=False):
    """Plot a real-valued finite element field on the mesh."""
    plot_scalar_mode(mesh, field, show_mesh, ax)


# ---------------------------------------------------------------------------
# Propagator
# ---------------------------------------------------------------------------

class Propagator:
    """Class for coupled-mode propagation of tapered waveguides."""

    # -- propagation params --------------------------------------------------
    solver = "RK45"
    #: bool: whether to include the WKB-like correction term
    WKB    = True

    # -- z-stepping params ---------------------------------------------------
    z_acc       = 0.
    fixed_zstep = None
    min_zstep      = 0.625
    min_zstep_neff = 10.
    max_zstep      = float('inf')   # plain Python float — no backend dependency
    init_zstep     = 10.

    # -- misc ----------------------------------------------------------------
    degen_crit          = 1e-5
    degen_groups        = []
    skipped_modes       = []
    allow_swaps         = True
    cmat_correction_mode = "from_interp"

    # =========================================================================

    def __init__(self, wl, wvg: Union[None, Waveguide] = None,
                 Nmax=None, save_dir=None):
        self.backend = get_backend()
        self.xp      = get_xp()
        self.wvg     = wvg
        self.wl      = wl
        self.Nmax    = Nmax
        self.k       = 2 * self.xp.pi / wl

        self.cmats      = None
        self.neffs      = None
        self.vs         = None
        self.mesh       = None
        self.zs         = None

        self.cmats_funcs          = None
        self.neffs_funcs          = None
        self.get_v                = None
        self.points0              = None
        self.channel_basis_matrix = None
        
        self._channel_basis_z     = None
        self._channel_basis_tol   = 1e-9
        
        self.meshpoints           = None

        self.save_dir = './data' if save_dir is None else save_dir
        self.check_and_make_folders()

    # =========================================================================
    # Main public functions
    # =========================================================================

    def solve_at(self, z=0, mesh=None):
        """Solve for waveguide modes at a given z value."""
        mesh     = self.make_mesh_at_z(z) if mesh is None else mesh
        IOR_dict = self.wvg.assign_IOR()
        w, v, N  = solve_waveguide(mesh, self.wl, IOR_dict, sparse=True, Nmax=self.Nmax)
        self.mesh = mesh
        return get_eff_index(self.wl, w), v

    def characterize(self, zi=None, zf=None, mesh=None, tag='', save=False):
        """Compute modes and coupling coefficients and set up for propagation."""

        # z-invariant waveguide shortcut
        if self.wvg.z_invariant:
            ps          = "_" + tag if tag is not None else ""
            meshwriteto = (self.save_dir + "/meshes/mesh" + ps) if save else None
            self.mesh   = self.generate_mesh(writeto=meshwriteto) if mesh is None else mesh
            print("mesh has ", len(self.mesh.points), " points")
            neff, v = self.solve_at(0)
            zs      = self.xp.array([0.])
            neffs   = self.xp.array([neff])
            vs      = self.xp.array([v])
            self.zs, self.neffs, self.vs = zs, neffs, vs
            if save:
                self.save(zs=zs, neffs=neffs, vs=vs, tag=tag)
            self.make_interp_funcs_zinv()
            return zs, neffs, vs, None

        start_time = time.time()
        self.compute_modes(zi, zf, mesh, tag, save)
        self.compute_cmats(save=save, tag=tag)
        print("time elapsed: ", time.time() - start_time)
        self.make_interp_funcs(self.zs)
        return self.zs, self.neffs, self.vs, self.cmats

    # alias
    prop_setup = characterize

    def apply_phase(self, u, z, zi=None):
        """Apply the e^{i beta_j z} phase to mode amplitudes."""
        if zi is None:
            zi = self.zs[0]
        # u may arrive as a python list; ``list * jax_array`` raises (numpy would
        # broadcast).  Coerce first so both backends behave the same.
        u = self.xp.asarray(u)
        phase = self.xp.exp(
            1.j * self.k * self.xp.array(self.get_int_neff(z) - self.get_int_neff(zi))
        )
        return u * phase

    # -------------------------------------------------------------------------
    # propagate
    # -------------------------------------------------------------------------

    def propagate(self, u0, zi=None, zf=None):
        assert self.zs is not None, \
            "no propagation data detected — run characterize() or load() first"
        if zi is None:
            zi = float(self.zs[0])
        if zf is None:
            zf = float(self.zs[-1])

        if len(self.zs) == 1:
            return self.zs, self.xp.array([u0]), self.apply_phase(u0, zf, zi)
        if zi > zf:
            return self.backpropagate(u0, zi, zf)

        # ========================= JAX CODEPATH ==============================
        if self.backend == "jax":
            import jax
            import jax.numpy as jnp
            import diffrax

            neffs_spline   = self._prop_neffs_spline
            int_neffs_func = self._prop_int_neffs_func
            dif_neffs_func = self._prop_dif_neffs_func
            M_spline       = self._prop_M_spline
            wl             = self.wl
            wkb_flag       = bool(self.WKB)

            def dz_dt(z, u, args=None):
                phases    = (2 * jnp.pi / wl *
                             (int_neffs_func(z) - int_neffs_func(zi))) % (2 * jnp.pi)
                phase_mat = jnp.exp(1j * (phases[None, :] - phases[:, None]))
                M_z       = M_spline(z)
                neffs     = neffs_spline(z)
                ddz       = -1. / neffs * jnp.einsum(
                    'ij,...j->...i', phase_mat * M_z, u * neffs)
                if wkb_flag:
                    ddz += -0.5 * dif_neffs_func(z) / neffs * u
                return ddz

            @jax.jit
            def _run(u0_val):
                sol = diffrax.diffeqsolve(
                    diffrax.ODETerm(dz_dt),
                    diffrax.Dopri5(),
                    t0=zi,
                    t1=zf,
                    dt0=(zf - zi) * 0.01,
                    y0=jnp.asarray(u0_val, dtype=jnp.complex128),
                    # SaveAt(t1=True) keeps only the final state → minimal memory
                    saveat=diffrax.SaveAt(t1=True),
                    #saveat=diffrax.SaveAt(steps=True), # for fair comparison with numpy path
                    stepsize_controller=diffrax.PIDController(rtol=1e-12, atol=1e-10),
                    max_steps=200_000,
                )
                return sol.ys, sol.ts

            u0_jnp         = jnp.asarray(u0, dtype=jnp.complex128)
            us_grid, z_grid = _run(u0_jnp)
            uf = self.apply_phase(us_grid[-1], float(z_grid[-1]), zi)
            return z_grid, us_grid, uf

        # ========================= NUMPY/SCIPY CODEPATH ======================
        else:
            u0         = self.xp.array(u0, dtype=self.xp.complex128)
            orig_shape = u0.shape
            nmodes     = orig_shape[-1]
            u0_flat    = u0.reshape(-1, nmodes) if len(orig_shape) > 1 else u0

            k             = 2 * self.xp.pi / self.wl
            int_neffs_zi  = self.get_int_neff(zi)

            def deriv(z, u_in):
                neffs = self.get_neff(z)
                cmat  = self.get_cmat(z)
                p     = self.xp.exp(1.j * k * (self.get_int_neff(z) - int_neffs_zi))
                if len(orig_shape) > 1:
                    u_curr  = u_in.reshape(-1, nmodes)
                    v       = u_curr * neffs * p
                    mat_vec = np.einsum('ij,...j->...i', cmat, v)
                    ddz     = -(np.conj(p) / neffs) * mat_vec
                    if self.WKB:
                        ddz += self.WKB_cor(z) * u_curr
                    return ddz.flatten()
                else:
                    v       = u_in * neffs * p
                    mat_vec = self.xp.dot(cmat, v)
                    ddz     = -(self.xp.conj(p) / neffs) * mat_vec
                    if self.WKB:
                        ddz += self.WKB_cor(z) * u_in
                    return ddz

            y0_input = u0_flat.flatten() if len(orig_shape) > 1 else u0_flat
            sol = solve_ivp(deriv, (zi, zf), y0_input,
                            method=self.solver, rtol=1e-12, atol=1e-10, 
                            first_step=abs(zf-zi)*0.01, ) # removing t_eval=[zf] means to compute and return all internal steps

            num_steps = len(sol.t)
            if len(orig_shape) > 1:
                us_grid = sol.y.T.reshape(num_steps, *orig_shape[:-1], nmodes)
                uf_flat = sol.y[:, -1].reshape(*orig_shape[:-1], nmodes)
            else:
                us_grid = sol.y.T
                uf_flat = sol.y[:, -1]

            uf = self.apply_phase(uf_flat, sol.t[-1], zi)
            return sol.t, us_grid, uf

    # -------------------------------------------------------------------------
    # backpropagate
    # -------------------------------------------------------------------------

    def backpropagate(self, uf, zi=None, zf=None):
        """Propagate backwards: given the field at zi, recover the field at zf < zi."""
        assert self.zs is not None, \
            "no propagation data detected — run characterize() or load() first"
        if zi is None:
            zi = float(self.zs[-1])
        if zf is None:
            zf = float(self.zs[0])

        if len(self.zs) == 1:
            return self.zs, self.xp.array([uf]), self.apply_phase(uf, zf, zi)

        # ========================= JAX CODEPATH ==============================
        if self.backend == "jax":
            import jax
            import jax.numpy as jnp
            import diffrax

            # Reuse exactly the same splines as the forward path.
            neffs_spline   = self._prop_neffs_spline
            int_neffs_func = self._prop_int_neffs_func
            dif_neffs_func = self._prop_dif_neffs_func
            M_spline       = self._prop_M_spline
            wl             = self.wl
            wkb_flag       = bool(self.WKB)

            def dz_dt(z, u, args=None):
                phases    = (2 * jnp.pi / wl *
                             (int_neffs_func(z) - int_neffs_func(zi))) % (2 * jnp.pi)
                phase_mat = jnp.exp(1j * (phases[None, :] - phases[:, None]))
                M_z       = M_spline(z)
                neffs     = neffs_spline(z)
                ddz       = -1. / neffs * jnp.einsum(
                    'ij,...j->...i', phase_mat * M_z, u * neffs)
                if wkb_flag:
                    ddz += -0.5 * dif_neffs_func(z) / neffs * u
                return ddz

            @jax.jit
            def _run_backward(uf_val):
                # diffrax integrates backward when t0 > t1 and dt0 < 0
                sol = diffrax.diffeqsolve(
                    diffrax.ODETerm(dz_dt),
                    diffrax.Dopri5(),
                    t0=zi,
                    t1=zf,
                    dt0=(zf - zi) * 0.01,   # negative, since zf < zi
                    y0=jnp.asarray(uf_val, dtype=jnp.complex128),
                    saveat=diffrax.SaveAt(t1=True),
                    # saveat=diffrax.SaveAt(steps=True), # for fair comparison with numpy path
                    stepsize_controller=diffrax.PIDController(rtol=1e-12, atol=1e-10),
                    max_steps=200_000,
                )
                return sol.ys, sol.ts

            uf_jnp          = jnp.asarray(uf, dtype=jnp.complex128)
            us_grid, z_grid = _run_backward(uf_jnp)
            ui = self.apply_phase(us_grid[-1], float(z_grid[-1]), zi)
            return z_grid, us_grid, ui

        # ========================= NUMPY/SCIPY CODEPATH ======================
        else:
            u0         = self.xp.array(uf, dtype=self.xp.complex128)
            orig_shape = u0.shape
            nmodes     = orig_shape[-1]
            u0_flat    = u0.reshape(-1, nmodes) if len(orig_shape) > 1 else u0

            def deriv(z, u_in):
                zp        = self.zs[-1] - z
                neffs     = self.get_neff(zp)
                phases    = (self.k * (self.get_int_neff(zp) - self.get_int_neff(zf))) % (2 * self.xp.pi)
                cmat      = self.get_cmat(zp)
                phase_mat = self.xp.exp(1.j * (phases[None, :] - phases[:, None]))
                if len(orig_shape) > 1:
                    u_curr = u_in.reshape(-1, nmodes)
                    ddz    = -1. / neffs * np.einsum(
                        'ij,...j->...i', phase_mat * cmat, u_curr * neffs)
                    if self.WKB:
                        ddz += self.WKB_cor(zp) * u_curr
                    return -ddz.flatten()
                else:
                    ddz = -1. / neffs * self.xp.dot(phase_mat * cmat, u_in * neffs)
                    if self.WKB:
                        ddz += self.WKB_cor(zp) * u_in
                    return -ddz

            y0_input = u0_flat.flatten() if len(orig_shape) > 1 else u0_flat
            sol = solve_ivp(deriv,
                            (self.zs[-1] - zf, self.zs[-1] - zi),
                            y0_input, 
                            method=self.solver, rtol=1e-12, atol=1e-10,
                            first_step=abs(zf-zi)*0.01, )  # removing t_eval=[zf] means to compute and return all internal steps

            num_steps = len(sol.t)
            if len(orig_shape) > 1:
                us_grid = sol.y.T.reshape(num_steps, *orig_shape[:-1], nmodes)
                uf_flat = sol.y[:, -1].reshape(*orig_shape[:-1], nmodes)
            else:
                us_grid = sol.y.T
                uf_flat = sol.y[:, -1]

            ui = self.apply_phase(uf_flat, zi, zf)
            return sol.t, us_grid, ui

    # =========================================================================
    # Setup / characterisation helpers
    # =========================================================================

    def compute_neffs(self, zi=0, zf=None, mesh=None, tag='', save=False):
        zf = self.wvg.z_ex if zf is None else zf
        start_time  = time.time()
        neffs, zs, vs = [], [], []
        self.wvg.update(0)
        ps          = "_" + tag if tag is not None else ""
        meshwriteto = (self.save_dir + "/meshes/mesh" + ps) if save else None
        if mesh is None:
            mesh = self.generate_mesh(meshwriteto)
        self.mesh = mesh
        print("mesh has ", len(self.mesh.points), " points")
        zstep0      = self.init_zstep if self.fixed_zstep is None else self.fixed_zstep
        min_zstep   = self.min_zstep_neff
        neff_interp = None
        IOR_dict    = self.wvg.assign_IOR()
        z = zi
        print("computing effective indices ...")

        if zi == zf:
            w, v, N = solve_waveguide(mesh, self.wl, IOR_dict, sparse=True, Nmax=self.Nmax)
            return z, get_eff_index(self.wl, w), v

        while True:
            _mesh    = self.wvg.transform_mesh(mesh, 0, z)
            w, v, N  = solve_waveguide(_mesh, self.wl, IOR_dict, sparse=True, Nmax=self.Nmax)
            neff     = get_eff_index(self.wl, w)
            if len(neffs) > 0:
                if len(neffs) == 1:
                    self.track_modes(vs[-1], v, neffs[-1], neff)
                elif 1 < len(vs) < 4:
                    _N          = len(neffs)
                    neff_interp = interp1d(zs[-_N:], self.xp.array(neffs)[-_N:, :],
                                          kind=_N-1, axis=0, fill_value="extrapolate")
                    self.track_modes(vs[-1], v, neff_interp(z), neff)
                else:
                    neff_interp = myCubicSpline(zs[-4:], neffs[-4:], axis=0)
                    self.track_modes(vs[-1], v, neff_interp(z), neff)

            if len(neffs) < 4 or self.fixed_zstep:
                neffs.append(neff); zs.append(z); vs.append(v)
                print("\rcurrent z: {0} / {1} ; current zstep: {2}        ".format(
                    z, zf, zstep0), end='', flush=True)
                if z == zf:
                    break
                z = min(zf, z + zstep0)
                continue

            refac = self._ref_fac_n(neff_interp(z), neff, neffs[-1])
            if refac >= 0 or zstep0 == min_zstep:
                neffs.append(neff); zs.append(z); vs.append(v)
                print("\rcurrent z: {0} / {1} ; current zstep: {2}        ".format(
                    z, zf, zstep0), end='', flush=True)
                if z == zf:
                    break
                if refac == 1:
                    zstep0 *= 2
                z = min(zf, z + zstep0)
            else:
                print("\rcurrent z: {0} / {1}; tol. not met, reducing step        ".format(
                    z, zf), end='', flush=True)
                z      = zs[-1]
                zstep0 = max(zstep0 / 2, min_zstep)
                z     += zstep0

        neffs = self.xp.array(neffs)
        vs    = self.xp.array(vs)
        zs    = self.xp.array(zs)
        neff_funcs = []
        for i in range(self.Nmax):
            try:
                neff_funcs.append(UnivariateSpline(zs, neffs[:, i], s=0))
            except NotImplementedError:
                neff_funcs.append(interp1d(zs, neffs[:, i], kind='linear'))
        self.neffs_funcs = neff_funcs
        self.neffs = neffs
        self.zs    = zs
        self.vs    = vs
        if save:
            self.save(zs, None, neffs, vs, tag=tag)
        self.make_interp_funcs(zs, make_cmat=False)
        print("time elapsed: ", time.time() - start_time)
        return zs, neffs

    def compute_modes(self, zi=None, zf=None, mesh=None, tag='', save=False):
        zi = 0 if zi is None else zi
        hit_min_zstep = False
        if zf is None:
            assert self.wvg.z_ex is not None, \
                "loaded waveguide has no set length (attribute z_ex); pass in a value for zf."
            zf = self.wvg.z_ex

        fixed_step = self.fixed_zstep
        min_zstep  = self.min_zstep
        max_zstep  = self.max_zstep
        zstep0     = self.init_zstep if fixed_step is None else fixed_step

        self.wvg.update(0)
        neffs, vs, vs_dec, zs = [], [], [], []
        meshpoints = []
        vinterp    = None

        ps          = "_" + tag if tag is not None else ""
        meshwriteto = (self.save_dir + "/meshes/mesh" + ps) if save else None
        mesh        = self.generate_mesh(writeto=meshwriteto) if mesh is None else mesh
        print("mesh has ", len(mesh.points), " points")
        _mesh    = copy.deepcopy(mesh)
        IOR_dict = self.wvg.assign_IOR()
        z = zi
        print("computing modes ...")

        while True:
            self.wvg.transform_mesh(mesh, 0, z, _mesh)
            if len(vs) == 0 and self.vs is not None and self.neffs is not None:
                # Bootstrap from load_init_conds().  Force a *writable* host numpy
                # copy: the mode-tracking bookkeeping below (track_modes /
                # correct_degeneracy / make_sign_consistent / avg_degen_neff) does
                # in-place mutation and list indexing, which a JAX array rejects
                # outright and a bare np.asarray() of one leaves read-only.
                neff = np.array(self.neffs[0])
                v    = np.array(self.vs[0])
            else:
                w, v, N = solve_waveguide(_mesh, self.wl, IOR_dict, sparse=True, Nmax=self.Nmax)
                neff    = get_eff_index(self.wl, w)

            if len(neffs) > 0:
                if len(neffs) == 1:
                    self.track_modes(vs[-1], v, neffs[-1], neff)
                elif 1 < len(vs) < 4:
                    _N          = len(neffs)
                    neff_interp = interp1d(zs[-_N:], self.xp.array(neffs)[-_N:, :],
                                          kind=_N-1, axis=0, fill_value="extrapolate")
                    self.track_modes(vs[-1], v, neff_interp(z), neff)
                else:
                    neff_interp = myCubicSpline(zs[-4:], neffs[-4:], axis=0,
                                               extrapolate=True, bc_type='natural')
                    self.track_modes(vs[-1], v, neff_interp(z), neff)

            for gr in self.degen_groups:
                self.avg_degen_neff(gr, neff)
            v = self._zero_skipped(v)

            vdec = self.decimate(v)
            if len(vs) < 4 or fixed_step:
                neffs.append(neff); zs.append(z); vs.append(v); vs_dec.append(vdec)
                if self.cmat_correction_mode == "from_interp" and not self.wvg.linear:
                    meshpoints.append(self.xp.copy(_mesh.points[:, :2]))
                print("\rcurrent z: {0} / {1} ; current zstep: {2}        ".format(
                    z, zf, zstep0), end='', flush=True)
                if z == zf:
                    break
                z = min(zf, z + zstep0)
                continue
            else:
                vinterp = myCubicSpline(self.xp.array(zs[-4:]), vs_dec[-4:], axis=0)
                refac   = self._ref_fac_v(vdec, vs_dec[-1], vinterp(z))
                if refac >= 0 or zstep0 == min_zstep:
                    if not hit_min_zstep and zstep0 == min_zstep:
                        hit_min_zstep = True
                    neffs.append(neff); zs.append(z); vs.append(v); vs_dec.append(vdec)
                    if self.cmat_correction_mode == "from_interp" and not self.wvg.linear:
                        meshpoints.append(self.xp.copy(_mesh.points[:, :2]))
                    print("\rcurrent z: {0} / {1} ; current zstep: {2}        ".format(
                        z, zf, zstep0), end='', flush=True)
                    if z == zf:
                        break
                    if refac == 1:
                        zstep0 = min(max_zstep, zstep0 * 2)
                    z = min(zf, z + zstep0)
                else:
                    print("\rcurrent z: {0} / {1}; tol. not met, reducing step        ".format(
                        z, zf), end='', flush=True)
                    z      = zs[-1]
                    zstep0 = max(zstep0 / 2, min_zstep)
                    z      = min(zf, z + zstep0)

        neffs = self.xp.array(neffs)
        vs    = self.xp.array(vs)
        zs    = self.xp.array(zs)
        self.neffs = neffs
        self.zs    = zs
        self.vs    = vs
        self.mesh  = mesh
        if self.wvg.linear:
            meshpoints.append(mesh.points[:, :2])
            meshpoints.append(_mesh.points[:, :2])
        self.meshpoints = self.xp.array(meshpoints)
        if save:
            self.save(zs, None, neffs, vs, self.meshpoints, tag=tag)
        self.make_interp_funcs(zs, True, True, True)
        if hit_min_zstep:
            print("\nwarning: hit minimum z step when computing modes")
        return zs, neffs, vs

    def load_init_conds(self, init_prop: Propagator, z=None):
        self.vs    = []
        self.neffs = []
        if z is None:
            self.vs.append(init_prop.vs[-1])
            self.neffs.append(init_prop.neffs[-1])
        else:
            self.vs.append(init_prop.get_v(z))
            self.neffs.append(init_prop.get_neff(z))

    def compute_cmats(self, zs=None, vs=None, mesh=None, tag='', save=False):
        zs   = self.zs   if zs   is None else zs
        vs   = self.vs   if vs   is None else vs
        mesh = self.mesh if mesh is None else mesh
        _mesh = copy.deepcopy(mesh)
        if _mesh.points.shape[1] == 3:
            _mesh.points = _mesh.points[:, :2]
        zi, zf = zs[0], zs[-1]
        print("\ncomputing coupling matrix ...")
        vi = myCubicSpline(zs, vs, axis=0)

        _linear = (len(self.meshpoints) == 2)
        dmeshdz = None
        if self.cmat_correction_mode == "from_interp":
            if _linear:
                slope   = (self.meshpoints[1] - self.meshpoints[0]) / (zs[-1] - zs[0])
                points  = lambda z: slope * (z - zs[0]) + self.meshpoints[0]
                dmeshdz = lambda z: slope
            else:
                points  = myCubicSpline(zs, self.meshpoints, axis=0)
                dmeshdz = points.derivative()

        dvdz    = vi.derivative()
        cmats   = []
        points0 = mesh.points.T[:2, :]
        for i, z in enumerate(zs):
            print("\rcurrent z: {0} / {1}        ".format(z, zf), end='', flush=True)
            if self.cmat_correction_mode == "from_interp":
                if _linear:
                    _mesh.points[:] = points(z)
                else:
                    _mesh.points[:] = self.meshpoints[i]
                dxydz = dmeshdz(z)
            else:
                self.wvg.transform_mesh(mesh, 0, z, _mesh)
                dxydz = self.xp.array(
                    self.wvg.deriv_transform(points0[0], points0[1], 0, z)).T

            B     = construct_B(_mesh, True)
            dvdxy = FEval.transverse_gradient(vs[i], _mesh.cells[1].data, _mesh.points)
            cor   = self.xp.sum(dxydz[None, :, :] * dvdxy, axis=2)
            cmat  = self.inner_product(dvdz(z) - cor, vs[i], B)
            cmats.append(cmat)

        cmats      = self.xp.array(cmats)
        self.cmats = cmats
        self.make_interp_funcs(zs, make_cmat=True, make_neff=True, make_v=True)
        if save:
            self.save(cmats=cmats, tag=tag)
        return cmats

    # =========================================================================
    # Interpolation construction  —  single source of truth
    # =========================================================================

    def make_interp_funcs(self, zs=None, make_neff=True, make_cmat=True, make_v=True):
        """Build all interpolation / spline objects for both backends.

        This is the *single* place where splines are constructed.  It fully
        replaces the old ``_prepare_jax_splines`` duplication.
        """
        if zs is None:
            zs = self.xp.copy(self.zs)

        # =====================================================================
        # JAX BACKEND
        # =====================================================================
        if self.backend == "jax":
            import jax.numpy as jnp

            zs_jnp = jnp.asarray(zs, dtype=jnp.float64)

            # -----------------------------------------------------------------
            # Effective indices
            # -----------------------------------------------------------------
            if make_neff and self.neffs is not None:
                neff_data    = jnp.asarray(self.neffs[:, :self.Nmax], dtype=jnp.float64)
                neff_spline  = myCubicSpline(zs_jnp, neff_data, axis=0)

                self._prop_neffs_spline    = neff_spline
                self._prop_int_neffs_func  = neff_spline.antiderivative()
                self._prop_dif_neffs_func  = neff_spline.derivative()
                self._dif_neffs_spline_jax = self._prop_dif_neffs_func  # alias

                # Legacy per-mode wrappers
                _anti = self._prop_int_neffs_func
                _dif  = self._prop_dif_neffs_func
                self.neffs_int_funcs = [
                    (lambda i: (lambda z: _anti(z)[i]))(i) for i in range(self.Nmax)
                ]
                self.neffs_dif_funcs = [
                    (lambda i: (lambda z: _dif(z)[i]))(i)  for i in range(self.Nmax)
                ]
                # These are not used in the JAX path but keep attribute consistent
                self.neffs_funcs  = None
                self.neffs_spline = None

            # -----------------------------------------------------------------
            # Coupling matrices
            # -----------------------------------------------------------------
            if make_cmat and self.cmats is not None:
                cmats_data = jnp.asarray(self.cmats, dtype=jnp.float64)

                # Skew-symmetric form used inside the ODE
                cmats_skew       = 0.5 * (jnp.swapaxes(cmats_data, 1, 2) - cmats_data)
                self._prop_M_spline    = myCubicSpline(zs_jnp, cmats_skew, axis=0)

                # Full-matrix form used by get_cmat()
                self._cmat_spline_jax  = myCubicSpline(zs_jnp, cmats_data, axis=0)

                self.cmats_funcs  = None
                self.cmats_spline = None

            # -----------------------------------------------------------------
            # Eigenmodes  —  keep this interpolant on the HOST.
            #
            # get_v is only used for host-side field reconstruction / basis
            # change (to_channel_basis, make_field, compute_change_of_basis),
            # never inside the ODE.  Its coefficient tensor is
            # (4, n_z, Nmodes, Npoints) complex128 — several GB for a photonic
            # lantern — so putting it on the GPU (with front + back propagators
            # resident at once in the multi-wavelength pipeline) is what drives
            # the CUDA OOM.  A plain scipy CubicSpline on numpy is the same
            # not-a-knot interpolant and costs zero device memory.
            if make_v and self.vs is not None:
                import scipy.interpolate as _si
                _vs_host = np.asarray(self.vs)
                _zs_host = np.asarray(zs)
                try:
                    self.get_v = _si.CubicSpline(_zs_host, _vs_host, axis=0)
                except (ValueError, TypeError):
                    # older scipy: no direct complex support -> fit re/im apart
                    _csr = _si.CubicSpline(_zs_host, _vs_host.real, axis=0)
                    _csi = _si.CubicSpline(_zs_host, _vs_host.imag, axis=0)
                    self.get_v = lambda z, _r=_csr, _i=_csi: _r(z) + 1j * _i(z)

            self._splines_ready = True

        # =====================================================================
        # NUMPY BACKEND
        # =====================================================================
        else:
            if make_cmat and self.cmats is not None:
                def _make_c(i, j):
                    assert i < j
                    return UnivariateSpline(
                        zs, 0.5 * (self.cmats[:, i, j] - self.cmats[:, j, i]),
                        ext=0, s=0)
                cmat_funcs = []
                for j in range(1, self.Nmax):
                    for i in range(j):
                        cmat_funcs.append(_make_c(i, j))
                self.cmats_funcs  = cmat_funcs
                self.cmats_spline = None

            if make_neff and self.neffs is not None:
                neff_funcs = [
                    UnivariateSpline(zs, self.neffs[:, i], s=0)
                    for i in range(self.Nmax)
                ]
                self.neffs_funcs     = neff_funcs
                self.neffs_spline    = None
                self.neffs_int_funcs = [f.antiderivative() for f in neff_funcs]
                self.neffs_dif_funcs = [f.derivative()     for f in neff_funcs]

            if make_v and self.vs is not None:
                self.get_v = myCubicSpline(zs, self.vs, axis=0)

    def _prepare_jax_splines(self):
        """Ensure JAX splines are initialised.

        This is now a lightweight guard: ``make_interp_funcs`` is the single
        source of truth.  If it has already been called (the normal case), this
        is a no-op.  It is kept so that existing call-sites in ``load()`` and
        ``characterize()`` continue to work without change.
        """
        if self.backend != "jax":
            return
        if getattr(self, "_splines_ready", False):
            return
        if self.zs is not None and len(self.zs) > 1:
            self.make_interp_funcs(
                self.zs,
                make_cmat=(self.cmats is not None),
                make_neff=(self.neffs is not None),
                make_v=(self.vs is not None),
            )

    def make_interp_funcs_zinv(self):
        """Create interpolation callables for z-invariant waveguides."""
        _v_zinv        = self.vs[0]
        _neff_zinv     = self.neffs[0]
        _neff_dif_zinv = self.xp.zeros_like(self.neffs[0])
        _cmat_zinv     = self.xp.zeros((self.Nmax, self.Nmax))

        self.get_v        = lambda z: _v_zinv
        self.get_neff     = lambda z: _neff_zinv
        self.get_int_neff = lambda z: z * _neff_zinv
        self.get_dif_neff = lambda z: _neff_dif_zinv
        self.get_cmat     = lambda z: _cmat_zinv

    # =========================================================================
    # Accessor functions
    # =========================================================================

    def get_cmat(self, z):
        """Return the coupling matrix at z."""
        if self.backend == "jax":
            # _cmat_spline_jax holds the full NxN matrix directly.
            return self._cmat_spline_jax(z)
        else:
            # Reconstruct the anti-symmetric NxN matrix from packed coefficients.
            raw    = self.xp.array([c(z) for c in self.cmats_funcs])
            nmodes = int((1 + (1 + 8 * raw.shape[0]) ** 0.5) // 2)
            j_idx, i_idx = np.tril_indices(nmodes, k=-1)
            matrix = np.zeros((nmodes, nmodes), dtype=np.complex128)
            matrix[i_idx, j_idx] = -raw
            matrix[j_idx, i_idx] =  raw
            return matrix

    def get_neff(self, z):
        """Return effective indices at z."""
        if self.backend == "jax":
            return self._prop_neffs_spline(z)
        else:
            return self.xp.array([f(z) for f in self.neffs_funcs])

    def get_int_neff(self, z):
        """Return the cumulative integral of effective indices at z."""
        if self.backend == "jax":
            return self._prop_int_neffs_func(z)
        else:
            return self.xp.array([f(z) for f in self.neffs_int_funcs])

    def get_dif_neff(self, z):
        """Return the derivative of effective indices at z."""
        if self.backend == "jax":
            return self._prop_dif_neffs_func(z)
        else:
            return self.xp.array([f(z) for f in self.neffs_dif_funcs])

    def WKB_cor(self, z):
        dbeta_dz = self.k * self.get_dif_neff(z)
        return -0.5 * dbeta_dz / (self.k * self.get_neff(z))

    # =========================================================================
    # I/O utility
    # =========================================================================

    def check_and_make_folders(self):
        for sq in ['', 'eigenmodes', 'eigenvalues', 'cplcoeffs',
                   'zvals', 'meshes', 'meshpoints']:
            d = os.path.join(self.save_dir, sq)
            if not os.path.exists(d):
                os.makedirs(d)

    def save(self, zs=None, cmats=None, neffs=None, vs=None,
             meshpoints=None, tag=""):
        ps = "" if tag == "" else "_" + tag
        if vs is not None:
            vs = self.xp.array(vs)
            self.xp.save(self.save_dir + '/eigenmodes/eigenmodes' + ps, vs)
        if cmats is not None:
            self.xp.save(self.save_dir + '/cplcoeffs/cplcoeffs' + ps, cmats)
        if neffs is not None:
            self.xp.save(self.save_dir + '/eigenvalues/eigenvalues' + ps, neffs)
        if zs is not None:
            self.xp.save(self.save_dir + '/zvals/zvals' + ps, zs)
        if meshpoints is not None and len(meshpoints) > 0:
            self.xp.save(self.save_dir + '/meshpoints/meshpoints' + ps, meshpoints)

    def load(self, tag=""):
        """Load saved propagation data from files identified by *tag*."""
        ps = "" if tag == "" else "_" + tag
        self.neffs = self.xp.load(self.save_dir + '/eigenvalues/eigenvalues' + ps + '.npy')
        self.vs    = self.xp.load(self.save_dir + '/eigenmodes/eigenmodes'   + ps + '.npy')
        self.zs    = self.xp.load(self.save_dir + '/zvals/zvals'             + ps + '.npy')
        if self.Nmax is None:
            self.Nmax = len(self.neffs[0])
        self._channel_basis_z     = None
        self.channel_basis_matrix = None
        self._splines_ready       = False   # force rebuild after load

        try:
            self.cmats = self.xp.load(self.save_dir + '/cplcoeffs/cplcoeffs' + ps + '.npy')
            if len(self.zs) > 1:
                self.make_interp_funcs(self.zs, make_cmat=True,
                                       make_neff=True, make_v=True)
            else:
                self.make_interp_funcs_zinv()
        except Exception:
            print("no coupling matrix file found … skipping")

        try:
            self.meshpoints = self.xp.load(
                self.save_dir + '/meshpoints/meshpoints' + ps + '.npy')
        except Exception:
            print("no mesh points found … skipping")

        self.mesh    = load_meshio_mesh(self.save_dir + '/meshes/mesh' + ps)
        self.points0 = self.xp.copy(self.mesh.points)

        if self.Nmax is None:
            self.Nmax = self.neffs.shape[1]

        # Rebuild interpolants if they were not built above (e.g. no cmats file)
        if len(self.zs) > 1:
            self.make_interp_funcs(self.zs, make_cmat=False,
                                   make_neff=True, make_v=True)
        else:
            self.make_interp_funcs_zinv()

    # =========================================================================
    # Misc maths helpers
    # =========================================================================

    def make_sign_consistent(self, v, _v):
        flip_mask = (self.xp.sum(self.xp.abs(v - _v), axis=1) >
                     self.xp.sum(self.xp.abs(v + _v), axis=1))
        _v[flip_mask] *= -1
        return flip_mask

    def inner_product(self, v1, v2, B):
        return B.dot(v1.T).T.dot(v2.T)

    def _zero_skipped(self, resids):
        """Zero the rows of `resids` at `self.skipped_modes` without in-place
        mutation, so it also works on immutable JAX arrays.  Multiplying by an
        exact 0/1 mask is bit-identical to ``resids[skipped] = 0`` for finite
        input."""
        if not len(self.skipped_modes):
            return resids
        keep = np.ones(resids.shape[0])
        keep[list(self.skipped_modes)] = 0.0
        return resids * keep.reshape((-1,) + (1,) * (resids.ndim - 1))

    def _ref_fac_v(self, v, vlast, vi):
        resids = self._zero_skipped(v - vi)
        err = self.xp.sqrt(self.xp.mean(self.xp.power(resids, 2)))
        tol = (max(self.xp.sqrt(self.xp.mean(self.xp.power(v - vlast, 2))) / 100., 1e-7)
               * self.xp.power(10., -float(self.z_acc)))
        if err < 0.1 * tol:
            return 1
        elif err > tol:
            return -1
        return 0

    def _ref_fac_n(self, ninterp, n, nlast):
        resids = self._zero_skipped(ninterp - n)
        err   = self.xp.sqrt(self.xp.mean(self.xp.power(resids, 2)))
        nsort = sorted(nlast, reverse=True)
        tol   = (max((nsort[0] - nsort[1]) / 100., 1e-9)
                 * self.xp.power(10., -float(self.z_acc)))
        if 0.1 * tol < err < tol:
            return 0
        elif err < 0.1 * tol:
            return 1
        return -1

    # =========================================================================
    # Mesh generation
    # =========================================================================

    def generate_mesh(self, writeto=None):
        return self.wvg.make_mesh(writeto=writeto)

    def make_mesh_at_z(self, z):
        mesh = self.generate_mesh() if self.mesh is None else self.mesh
        if z == 0:
            return copy.deepcopy(mesh)
        return self.wvg.transform_mesh(mesh, 0, z)

    # =========================================================================
    # Field construction helpers
    # =========================================================================

    def compute_change_of_basis(self, newbasis, z=None, u=None):
        if z is None:
            z = self.zs[-1]
        m   = self.make_mesh_at_z(z)
        B   = construct_B(m, sparse=True)
        oldbasis = self.get_v(z)
        cob = self.inner_product(newbasis, oldbasis, B)
        self.channel_basis_matrix = cob
        self._channel_basis_z = float(z)
        if u is not None:
            return cob, self.xp.dot(cob, u)
        return cob

    def _compute_isolated_basis_at_z(self, z):
        m       = self.make_mesh_at_z(z)
        self.wvg.assign_IOR()
        wvg_dim = len(self.wvg.prim3Dgroups[-1])
        npts    = m.points.shape[0]
        # Built row-by-row from solve_waveguide (host) output -> plain numpy so
        # the item assignment works on the JAX backend too.  Consumed host-side
        # by compute_change_of_basis / inner_product.
        _v      = np.zeros((wvg_dim, npts), dtype=np.complex128)
        for i in range(wvg_dim):
            _dict = self.wvg.isolate(i)
            try:
                _wi, _vi, _Ni = solve_waveguide(m, self.wl, _dict, sparse=True, Nmax=1)
            except Exception as exc:
                raise RuntimeError(
                    f"isolated-basis solve failed for channel {i} at z={z}"
                ) from exc
            _vi = np.ravel(np.asarray(_vi))
            if _vi.shape[0] != npts:
                raise RuntimeError(
                    f"isolated-basis shape mismatch at z={z}: "
                    f"got {_vi.shape[0]} points, expected {npts}"
                )
            if float(np.real(np.sum(_vi))) < 0.0:
                _vi = -_vi
            _v[i, :] = _vi
        return _v

    def compute_isolated_basis(self, z=None):
        if z is None:
            z = self.zs[-1]
        try:
            return self._compute_isolated_basis_at_z(z)
        except Exception as exc_primary:
            z_fallback = self.zs[-1] if self.zs is not None else z
            if abs(float(z_fallback) - float(z)) <= self._channel_basis_tol:
                raise
            try:
                print(
                    f"isolated-basis failed at z={z}; retrying at z={z_fallback} "
                    f"(output end)"
                )
                return self._compute_isolated_basis_at_z(z_fallback)
            except Exception as exc_fallback:
                raise RuntimeError(
                    f"isolated-basis failed at z={z} and fallback z={z_fallback}"
                ) from exc_fallback

    def to_channel_basis(self, uf, z=None):
        if z is None:
            z = self.zs[-1]
        need_rebuild = (
            self.channel_basis_matrix is None
            or self._channel_basis_z is None
            or abs(float(z) - float(self._channel_basis_z)) > self._channel_basis_tol
        )
        if need_rebuild:
            _v = self.compute_isolated_basis(z)
            self.compute_change_of_basis(_v, z)
        return self.xp.dot(self.channel_basis_matrix, uf)

    def make_field(self, mode_amps, z=None, plot=False, apply_phase=True):
        assert self.zs is not None, \
            "no propagation data detected — run characterize() or load() first"
        zinv = len(self.zs) == 1
        assert z is not None or zinv, \
            "z can only be left as None if the waveguide is z-invariant"
        z  = 0 if z is None and zinv else z
        zi = self.zs[0] if self.zs is not None else 0.
        u  = self.xp.array(mode_amps, dtype=self.xp.complex128)
        basis = self.get_v(z)
        if apply_phase:
            uf    = self.apply_phase(u, z, zi)
            field = self.xp.sum(uf[:, None] * basis, axis=0)
        else:
            field = self.xp.sum(u[:, None] * basis, axis=0)
        if plot:
            self.plot_cfield(field, z, show_mesh=True)
        return field

    def make_mode_vector(self, field, z=None, mesh=None):
        if z is None:
            z = 0. if self.zs is None else self.zs[0]
        mesh  = self.make_mesh_at_z(z) if mesh is None else mesh
        B     = construct_B(self.mesh, sparse=True)
        basis = self.get_v(z)
        amps  = [self.inner_product(basis[i], field, B)
                 for i in range(basis.shape[0])]
        return self.xp.array(amps)

    def decimate(self, arr, outsize=10, axis=1):
        split_arrs = self.xp.array_split(arr, outsize, axis=axis)
        return self.xp.array([self.xp.mean(a, axis=axis) for a in split_arrs]).T

    def swap_modes(self, w, _w, _v):
        sidxs   = self.xp.argsort(w)[::-1]
        indices = self.xp.argsort(sidxs)
        return _v[indices], _w[indices]

    def track_modes(self, v, _v, w, _w):
        if w is not None and self.allow_swaps:
            _v[:], _w[:] = self.swap_modes(w, _w, _v)
        for gr in self.degen_groups:
            self.correct_degeneracy(gr, v, _v)
        self.make_sign_consistent(v, _v)

    def correct_degeneracy(self, group, v, _v, q=None):
        if q is None:
            coeff_mat = self.xp.dot(v[group, :], _v[group, :].T)
            u, s, vh  = self.xp.linalg.svd(coeff_mat)
            q         = self.xp.dot(vh.T, u.T)
        _vq        = self.xp.dot(_v[group, :].T, q)
        _v[group, :] = _vq[:, :].T
        return v, _v, q

    def avg_degen_neff(self, group, neffs):
        neffs[group] = self.xp.mean(neffs[group])[None]
        return neffs

    def compute_transfer_matrix(self, channel_basis=True, zi=None, zf=None):
        N   = self.Nmax
        mat = self.xp.zeros((N, N), dtype=self.xp.complex128)
        u0  = self.xp.zeros(N)
        for j in range(N):
            print("\rpropagating mode {0}".format(j), end='', flush=True)
            if j in self.skipped_modes:
                continue
            u0[:] = 0.
            u0[j] = 1.
            zs, us, uf = self.propagate(u0, zi, zf)
            out = self.to_channel_basis(uf, z=zf) if channel_basis else uf
            M   = len(out)
            mat[:M, j] = out
        return mat

    # =========================================================================
    # Plotting
    # =========================================================================

    def plot_wavefront(self, zs, us, zi=0, fig=None, ax=None):
        plot = False
        if ax is None or fig is None:
            fig, ax = plt.subplots(1, 1)
            fig.subplots_adjust(bottom=0.25)
            plot = True
        mesh = copy.deepcopy(self.mesh)
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        x0, y0, w, h = ax.get_position().bounds
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

        def update(z):
            ax.clear()
            ix   = bisect_left(zs, z)
            f    = self.make_field(us[ix], z)
            self.wvg.transform_mesh(self.mesh, 0, z, mesh)
            x    = mesh.points[:, 0]
            y    = mesh.points[:, 1]
            tri  = Triangulation(x, y, self.mesh.cells[1].data[:, :3])
            alphas = self.xp.abs(f)
            alphas /= self.xp.max(alphas)
            im = ax.tripcolor(tri, self.xp.angle(f), cmap='hsv',
                              vmin=-self.xp.pi, vmax=self.xp.pi,
                              shading="gouraud", alpha=alphas)
            fig.canvas.draw_idle()
            return im,

        slider = None
        im, = update(0)
        if len(zs) > 1:
            axsl   = fig.add_axes([x0, y0 - 0.25, w, 0.1])
            slider = Slider(ax=axsl, label=r'$z$', valmin=zs[0],
                            valmax=zs[-1], valinit=zs[0])
            slider.on_changed(update)
            if zi != 0:
                slider.set_val(zi)
        if fig is not None:
            fig.colorbar(im, cmap='hsv', ax=ax,
                         ticks=self.xp.linspace(-self.xp.pi, self.xp.pi, 5, endpoint=True),
                         label="phase")
        if plot:
            plt.show()
        return slider

    def plot_neffs(self):
        neffs = self.neffs.T
        for i in range(self.Nmax):
            plt.plot(self.zs, neffs[i], label="mode " + str(i))
        for z in self.zs:
            plt.axvline(x=z, alpha=0.05, color='k', zorder=-100)
        plt.xlabel(r"$z$")
        plt.ylabel("effective index")
        plt.legend(loc='best', bbox_to_anchor=(1.04, 1.))
        plt.tight_layout()
        plt.show()

    def plot_neff_diffs(self, yscale="log"):
        assert yscale in ["log", "lin"], "yscale not recognized"
        neffs = self.neffs.T
        for i in range(1, self.Nmax):
            if yscale == "log":
                plt.semilogy(self.zs, neffs[0] - neffs[i], label="mode " + str(i))
            else:
                plt.plot(self.zs, neffs[0] - neffs[i], label="mode " + str(i))
        for z in self.zs:
            plt.axvline(x=z, alpha=0.05, color='k', zorder=-100)
        plt.xlabel(r"$z$")
        plt.ylabel("difference in index from mode 0")
        plt.legend(loc='best', bbox_to_anchor=(1.04, 1.))
        plt.tight_layout()
        plt.show()

    def plot_field(self, field, z=None, mesh=None, ax=None, show_mesh=False):
        mesh = self.make_mesh_at_z(z) if mesh is None else mesh
        plot_field(field, mesh, ax, show_mesh)

    def plot_coupling_coeffs(self, legend=True):
        fig, ax = plt.subplots()
        colors      = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
                       '#984ea3', '#999999', '#e41a1c', '#dede00']
        line_styles = ['solid', 'dashed', 'dotted', 'dashdot', (5, (10, 3))]
        for j in range(self.Nmax):
            for i in range(j):
                ax.plot(self.zs, self.cmats[:, i, j],
                        label=str(i)+str(j),
                        ls=line_styles[i % 5], c=colors[j % 9])
        for z in self.zs:
            ax.axvline(x=z, alpha=0.05, color='k', zorder=-100)
        if legend:
            ax.legend(bbox_to_anchor=(1.04, 1.))
        ax.set_title("coupling coefficient matrix")
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$\kappa_{ij}$")
        plt.tight_layout()
        plt.show()

    def plot_mode_powers(self, zs, us):
        for i in range(us.shape[1]):
            plt.plot(zs, self.xp.power(self.xp.abs(us[:, i]), 2),
                     label="mode " + str(i))
        plt.xlabel(r'$z$ (um)')
        plt.ylabel("power")
        plt.legend(loc='best', bbox_to_anchor=(1.04, 1))
        plt.tight_layout()
        plt.show()

    def plot_cfield(self, field, z=None, mesh=None, fig=None, ax=None,
                   show_mesh=False, res=1., xlim=None, ylim=None):
        zinv = ((self.zs is not None and len(self.zs) == 1) or
                (self.wvg is not None and self.wvg.z_invariant))
        assert z is not None or mesh is not None or zinv, \
            "one of `z` or `mesh` needs to be passed for plotting"
        if zinv:
            mesh = self.make_mesh_at_z(0) if mesh is None else mesh
        else:
            mesh = self.make_mesh_at_z(z) if mesh is None else mesh
        plot_cfield(field, mesh, fig, ax, show_mesh, res, xlim, ylim)

    def plot_waveguide_mode(self, i, zi=0, fig=None, ax=None):
        plot = False
        if ax is None or fig is None:
            fig, ax = plt.subplots(1, 1)
            fig.subplots_adjust(bottom=0.25)
            plot = True
        mesh = copy.deepcopy(self.mesh)
        ax.set_aspect('equal')
        x0, y0, w, h = ax.get_position().bounds
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

        def update(z):
            ax.clear()
            v   = self.get_v(z)[i]
            self.wvg.transform_mesh(self.mesh, 0, z, mesh)
            x   = mesh.points[:, 0]
            y   = mesh.points[:, 1]
            tri = Triangulation(x, y, self.mesh.cells[1].data[:, :3])
            ax.tripcolor(tri, v, shading='gouraud')
            fig.canvas.draw_idle()

        slider = None
        update(0)
        if len(self.zs) > 1:
            axsl   = fig.add_axes([x0, y0 - 0.25, w, 0.1])
            slider = Slider(ax=axsl, label=r'$z$', valmin=self.zs[0],
                            valmax=self.zs[-1], valinit=self.zs[0])
            slider.on_changed(update)
            if zi != 0:
                slider.set_val(zi)
        if plot:
            plt.show()
        return slider


# ---------------------------------------------------------------------------
# ChainPropagator
# ---------------------------------------------------------------------------

class ChainPropagator(Propagator):
    """A series of Propagators connected end-to-end."""

    def __init__(self, propagators: list):
        self.propagators = propagators
        self.z_breaks    = [propagators[0].zs[0]]
        for p in propagators:
            self.z_breaks.append(p.zs[-1])

        p0 = propagators[0]
        self.wl      = p0.wl
        self.wvg     = p0.wvg
        self.Nmax    = p0.Nmax
        self.skipped_modes = p0.skipped_modes
        self.mesh    = p0.mesh
        self.xp      = get_xp()
        self.backend = get_backend()
        self.zs      = self.xp.concatenate([p.zs for p in propagators])

    def get_v(self, z):
        return self.get_prop(z).get_v(z)

    def get_prop(self, z):
        idx = max(0, bisect_left(self.z_breaks, z) - 1)
        return self.propagators[min(idx, len(self.propagators) - 1)]

    def propagate(self, u0, zi=None, zf=None):
        if zi is None:
            zi = self.propagators[0].zs[0]
        if zf is None:
            zf = self.propagators[-1].zs[-1]

        xp      = self.xp
        u       = xp.array(u0)
        all_zs  = []
        all_us  = []
        z       = zi
        first   = True

        while z < zf - 1e-10:
            p           = self.get_prop(z + 1e-8)
            segment_end = min(p.zs[-1], zf)
            if segment_end - z < 1e-10:
                z = segment_end
                continue
            zs, us, u = p.propagate(u, z, segment_end)
            if float(zs[-1]) <= z:
                raise RuntimeError(
                    f"Propagate did not advance: current z {z} zs[-1] {zs[-1]}")
            if first:
                all_zs.append(zs); all_us.append(us)
                first = False
            else:
                all_zs.append(zs[1:]); all_us.append(us[1:])
            z = float(zs[-1])

        zs_full = xp.concatenate(all_zs)
        us_full = xp.concatenate(all_us)
        return zs_full, us_full, u

    def to_channel_basis(self, uf, z=None):
        if z is None:
            z = self.propagators[-1].zs[-1]
        return self.get_prop(z).to_channel_basis(uf, z)

    def make_field(self, mode_amps, z, plot=False, apply_phase=True):
        return self.get_prop(z).make_field(mode_amps, z, plot, apply_phase)

    
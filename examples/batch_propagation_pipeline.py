# =====================================================================
# BATCH PROPAGATION PIPELINE
# =====================================================================
# Supports both JAX (GPU) and NumPy (CPU) backends.
#
# Device selection:          CBEAM_JAX_DEVICE_INDEX  (env-var, default 0)
#
# Field-generation back-end:
#   Even when the global backend is JAX the field-generation phase
#   (OPD → pupil → FFT → mesh projection) is very memory-intensive and
#   can saturate the GPU.  By default it therefore runs on the CPU with
#   plain NumPy/SciPy.  Pass  field_gen_on_gpu=True  to
#   BatchPropagationPipeline.__init__() to move it onto the GPU, or
#   override per call with the  use_gpu  keyword of
#   generate_batch_modal_coefficients().
#
# Chunked field generation:
#   generate_batch_modal_coefficients() processes the batch in chunks of
#   field_gen_chunk_size fields at a time.  Intermediate arrays (OPD,
#   pupil, FFT grid) are created and released for each chunk before the
#   next one starts.  This limits peak memory to O(chunk_size) regardless
#   of the total batch size and independently of which backend is used.
#   The default (field_gen_chunk_size=64) is conservative; increase it if
#   memory headroom allows.
#
# Chunked propagation:
#   propagate_batch() applies the same strategy with chunk_size=32 by
#   default (JAX) or the full batch (numpy).
# =====================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from cbeam.backend import get_backend, get_jax_device
from cbeam.waveguide import PhotonicLantern, get_19port_positions
from cbeam.propagator import Propagator, ChainPropagator

backend = get_backend()
default_chunk_size = 10000
default_gen_chunk_size = 1000

# -- optional JAX imports ------------------------------------------------
if backend == 'jax':
    import jax
    import jax.numpy as jnp
    _jax_device = get_jax_device()
else:
    jnp         = None
    _jax_device = None


# =====================================================================
# LAYER 1: CONFIGURATION & CONSTANTS
# =====================================================================

L1 = 50000
L2 = 50000

DEFAULT_WAVELENGTH_UM   = 0.8
DEFAULT_WAVELENGTH_NM   = 1550.0
DEFAULT_GRID_RESOLUTION = 400
DEFAULT_SUBPIXEL_N      = 7


def _jax_fft_batch(E_pupil_batch, pad_width):
    """JIT-compiled zero-pad + FFT for a batch of pupil fields (JAX only)."""
    E_padded = jnp.pad(
        E_pupil_batch,
        ((0, 0), (pad_width, pad_width), (pad_width, pad_width)),
        mode='constant',
        constant_values=0,
    )
    return jnp.fft.fftshift(
        jnp.fft.fft2(
            jnp.fft.ifftshift(E_padded, axes=(1, 2)),
            axes=(1, 2),
        ),
        axes=(1, 2),
    )


if backend == 'jax':
    _apply_pupil_jax_core = jax.jit(_jax_fft_batch, static_argnames=['pad_width'])


def get_simulation_parameters():
    """Return a dict of all setup constants and derived parameters."""
    params = {
        "wl":            DEFAULT_WAVELENGTH_UM,
        "wavelength_nm": DEFAULT_WAVELENGTH_NM,
        "taper_factor":  12.,
        "rclad":         9.0,
        "rjack":         27,
        "z_ex":          L1 + L2,
        "nclad":         1.444,
        "pad_factor":    4,
        "core_res":      16,
        "clad_res":      60,
        "jack_res":      30,
        "ifunc_file":    '/raid2/gcarla/git/ANDES/andes/PASSATA_scripts/data/ifunc/'
                         'ANDES_400pix_all_modes.fits',
    }
    params["rcore"]  = 1.8 / params["taper_factor"]
    params["ncore"]  = params["nclad"] + 8.8e-3
    params["njack"]  = params["nclad"] - 5.5e-3
    params["rcores"] = [params["rcore"]] * 19
    params["ncores"] = [params["ncore"]] * 19
    return params


# =====================================================================
# LAYER 2: WAVEGUIDE INFRASTRUCTURE
# =====================================================================

def build_and_characterize_lantern(p):
    """Set up the PhotonicLantern and return a ChainPropagator."""
    core_pos = get_19port_positions(p["rclad"] / 2.5)
    PL19 = PhotonicLantern(
        core_pos, p["rcores"], p["rclad"], p["rjack"],
        p["ncores"], p["nclad"], p["njack"], p["z_ex"],
        p["taper_factor"], p["core_res"], p["clad_res"], p["jack_res"],
    )

    prop1 = Propagator(p["wl"], PL19, 20)
    prop1.degen_groups  = [[1,2],[3,4],[6,7],[8,9],[10,11],[12,13],[15,16]]
    prop1.skipped_modes = [18]
    prop1.load("19port_0800_front")

    prop2 = Propagator(p["wl"], PL19, 20)
    prop2.skipped_modes = [18]
    prop2.degen_groups  = [[i for i in range(20) if i != 18]]
    prop2.load_init_conds(prop1)
    prop2.load("19port_0800_back")

    return ChainPropagator([prop1, prop2])


def get_waveguide_properties(prop12, mesh_z=0):
    """Extract waveguide modal properties at a given z position."""
    p_segment  = prop12.get_prop(mesh_z)
    mesh_obj   = p_segment.mesh
    mesh_areas = p_segment.wvg.assign_IOR()
    modes      = p_segment.vs

    if len(modes.shape) == 3:
        z_idx        = np.argmin(np.abs(p_segment.zs - mesh_z))
        active_modes = modes[z_idx]
    else:
        active_modes = modes

    n_mesh_points = mesh_obj.points.shape[0]
    areas         = getattr(p_segment, 'mesh_areas', np.ones(n_mesh_points))

    if active_modes.shape[0] == n_mesh_points:
        active_modes = active_modes.T

    n_modes   = active_modes.shape[0]
    points_2d = np.stack((mesh_obj.points[:, 1], mesh_obj.points[:, 0]), axis=-1)

    return {
        'mesh':          mesh_obj,
        'mesh_areas':    areas,
        'points':        points_2d,
        'modes':         active_modes,
        'n_modes':       n_modes,
        'n_mesh_points': n_mesh_points,
    }


# =====================================================================
# LAYER 3: FIELD GENERATION & PROJECTION
# =====================================================================

class ModalProjector:
    """Project spatial field profiles onto the modal basis with normalisation."""

    def __init__(self, wvg_props, xp=np):
        self.xp = xp
        modes   = xp.asarray(wvg_props["modes"])
        areas   = xp.asarray(wvg_props["mesh_areas"])
        self.projection_matrix = modes.conj() * areas

    def project_batch(self, E_batch):
        u0_batch = E_batch @ self.projection_matrix.T
        norms    = self.xp.linalg.norm(u0_batch, axis=1, keepdims=True)
        if self.xp is np:
            norms[norms == 0] = 1.0
        else:
            norms = self.xp.where(norms == 0, 1.0, norms)
        return u0_batch / norms


class IncidentFieldGenerator:
    """Generate incident field profiles with configurable aberrations.

    Parameters
    ----------
    p :      simulation parameter dict
    ifunc :  influence-function object
    xp :     array module to use — pass ``numpy`` (default) to keep all
             intermediate arrays on the CPU, or ``jax.numpy`` to keep them
             on the GPU.
    """

    def __init__(self, p, ifunc, xp=np):
        self.p     = p
        self.ifunc = ifunc
        self.xp    = xp

        self.mask_np   = ifunc.mask_inf_func.get() > 0
        self.grid_size = self.mask_np.shape[0]
        self.num_modes = len(ifunc.influence_function)
        self.ifunc_matrix = xp.stack(
            [f.get() for f in ifunc.influence_function], axis=0)

    def precompute_interpolation_weights(self, mesh_points):
        """Precompute static bilinear grid mappings from the padded FFT grid
        to the FE mesh.  Call once after construction."""
        padded_size = self.grid_size * self.p["pad_factor"]
        half_size   = padded_size // 2

        iy0 = self.xp.floor(mesh_points[:, 1] + half_size).astype(self.xp.int32)
        iy1 = iy0 + 1
        ix0 = self.xp.floor(mesh_points[:, 0] + half_size).astype(self.xp.int32)
        ix1 = ix0 + 1

        wy1 = (mesh_points[:, 1] + half_size) - iy0
        wy0 = 1.0 - wy1
        wx1 = (mesh_points[:, 0] + half_size) - ix0
        wx0 = 1.0 - wx1

        valid = ((iy0 >= 0) & (iy1 < padded_size) &
                 (ix0 >= 0) & (ix1 < padded_size))

        iy0 = self.xp.clip(iy0, 0, padded_size - 1)
        iy1 = self.xp.clip(iy1, 0, padded_size - 1)
        ix0 = self.xp.clip(ix0, 0, padded_size - 1)
        ix1 = self.xp.clip(ix1, 0, padded_size - 1)

        self.iy0 = self.xp.asarray(iy0)
        self.iy1 = self.xp.asarray(iy1)
        self.ix0 = self.xp.asarray(ix0)
        self.ix1 = self.xp.asarray(ix1)
        self.w00 = self.xp.asarray((wy0 * wx0 * valid)[:, None])
        self.w01 = self.xp.asarray((wy0 * wx1 * valid)[:, None])
        self.w10 = self.xp.asarray((wy1 * wx0 * valid)[:, None])
        self.w11 = self.xp.asarray((wy1 * wx1 * valid)[:, None])

    def generate_field_profiles_batch(self, coeff_batch):
        """Generate complex electric field profiles from aberration coefficients."""
        opd_flat_batch   = coeff_batch @ self.ifunc_matrix
        phase_flat_batch = opd_flat_batch * (2 * self.xp.pi / self.p["wavelength_nm"])
        E_flat_batch     = self.xp.exp(1j * phase_flat_batch)

        n_fields      = coeff_batch.shape[0]
        E_pupil_batch = self.xp.zeros(
            (n_fields, self.grid_size, self.grid_size), dtype=self.xp.complex128)
        if self.xp is np:
            E_pupil_batch[:, self.mask_np] = E_flat_batch
        else:
            E_pupil_batch = E_pupil_batch.at[:, self.mask_np].set(E_flat_batch)
        return E_pupil_batch

    def apply_pupil_to_lantern_jax(self, E_pupil_batch):
        """Pad and FFT-transform using JAX (runs on the selected device)."""
        pad_width         = (self.grid_size * self.p["pad_factor"] - self.grid_size) // 2
        E_pupil_batch_dev = jax.device_put(E_pupil_batch, _jax_device)
        return _apply_pupil_jax_core(E_pupil_batch_dev, pad_width)

    def apply_pupil_to_lantern(self, E_pupil_batch):
        """Pad and FFT-transform using SciPy (CPU, multithreaded)."""
        import scipy.fft as sp_fft
        N           = self.grid_size
        pad_width   = (N * self.p["pad_factor"] - N) // 2
        padded_size = N + 2 * pad_width

        E_padded = np.zeros(
            (E_pupil_batch.shape[0], padded_size, padded_size),
            dtype=np.complex128)
        # np.asarray() makes this safe for both numpy and jax array inputs.
        E_padded[:, pad_width:pad_width+N, pad_width:pad_width+N] = \
            np.asarray(E_pupil_batch)

        return sp_fft.fftshift(
            sp_fft.fft2(
                sp_fft.ifftshift(E_padded, axes=(1, 2)),
                axes=(1, 2),
                workers=-1,
            ),
            axes=(1, 2),
        )

    def resample_to_mesh(self, E_lantern_batch):
        """Bilinear interpolation from the padded FFT grid onto the FE mesh."""
        val00 = E_lantern_batch[:, self.iy0, self.ix0]
        val01 = E_lantern_batch[:, self.iy0, self.ix1]
        val10 = E_lantern_batch[:, self.iy1, self.ix0]
        val11 = E_lantern_batch[:, self.iy1, self.ix1]
        return (val00 * self.w00.T + val01 * self.w01.T +
                val10 * self.w10.T + val11 * self.w11.T)


# =====================================================================
# LAYER 4: BATCH PROPAGATION ENGINE
# =====================================================================

class BatchPropagationPipeline:
    """
    Orchestrates batch propagation of modal fields through a photonic lantern.

    Parameters
    ----------
    prop12 :               ChainPropagator
    p :                    simulation parameter dict
    ifunc :                influence-function object
    field_gen_on_gpu :     bool, optional (default False)
        When True *and* the JAX backend is active, field generation runs on
        the GPU.  When False (default), field generation always uses
        NumPy/SciPy on the CPU, which avoids GPU memory pressure during the
        OPD → pupil → FFT → projection phase.
        Can be overridden per call via the ``use_gpu`` keyword of
        ``generate_batch_modal_coefficients()``.
    field_gen_chunk_size : int or None, optional (default 64)
        Number of fields processed in each chunk of
        ``generate_batch_modal_coefficients()``.  The large intermediate
        arrays (OPD batch, padded FFT grid, mesh-sampled fields) are
        allocated, used, and released chunk by chunk, capping peak memory
        to O(chunk_size) independently of the total batch size and
        independently of which backend is used.  Set to None to disable
        chunking (process the whole batch at once).
        Can be overridden per call via the ``chunk_size`` keyword.
    """

    def __init__(self, prop12, p, ifunc,
                 field_gen_on_gpu=False,
                 field_gen_chunk_size=default_gen_chunk_size):
        self.prop12 = prop12
        self.p      = p

        # The *propagation* xp follows the global backend.
        self.xp = jnp if backend == 'jax' else np

        # Default chunk size for generate_batch_modal_coefficients().
        # None means "no chunking" (process the whole batch at once).
        self.field_gen_chunk_size = field_gen_chunk_size

        # -----------------------------------------------------------------
        # Decide the default array module for field generation.
        # -----------------------------------------------------------------
        self._field_gen_on_gpu_default = (backend == 'jax') and field_gen_on_gpu

        # Always build the numpy generators (cheap, host memory only).
        self._field_gen_np   = IncidentFieldGenerator(p, ifunc, xp=np)
        self._modal_proj_np  = None   # built after weight precomputation below

        # Build JAX generators only if explicitly requested.
        self._field_gen_jax  = None
        self._modal_proj_jax = None
        self._ifunc_ref      = ifunc  # kept for lazy JAX init

        wvg_props_input = get_waveguide_properties(prop12, mesh_z=0)
        self.wvg_props_input  = wvg_props_input
        self.wvg_props_output = get_waveguide_properties(prop12, mesh_z=p["z_ex"])

        self._field_gen_np.precompute_interpolation_weights(
            wvg_props_input['points'])
        self._modal_proj_np = ModalProjector(wvg_props_input, xp=np)

        if backend == 'jax' and field_gen_on_gpu:
            self._build_jax_field_gen()

        # Delaunay cache for output interpolation.
        self._delaunay_cache = {}
        self._delaunay_cache[DEFAULT_GRID_RESOLUTION] = \
            self._precompute_delaunay_grid(DEFAULT_GRID_RESOLUTION)

    # ------------------------------------------------------------------
    def _build_jax_field_gen(self):
        """Lazily initialise the JAX field-generation objects."""
        if self._field_gen_jax is not None:
            return
        self._field_gen_jax = IncidentFieldGenerator(
            self.p, self._ifunc_ref, xp=jnp)
        self._field_gen_jax.precompute_interpolation_weights(
            jnp.asarray(self.wvg_props_input['points'],
                        dtype=jnp.float64, device=_jax_device))
        self._modal_proj_jax = ModalProjector(self.wvg_props_input, xp=jnp)

    # ------------------------------------------------------------------
    def enable_gpu_field_gen(self):
        """
        Lazily initialise the JAX field-generation objects after construction.

        Call this if the pipeline was built with ``field_gen_on_gpu=False``
        but you later want to use ``use_gpu=True`` in some calls.
        """
        if backend != 'jax':
            raise RuntimeError(
                "enable_gpu_field_gen() requires the JAX backend.")
        self._build_jax_field_gen()
        print("JAX field-generation objects initialised on device:", _jax_device)

    # ------------------------------------------------------------------
    def _precompute_delaunay_grid(self, grid_resolution):
        """Precompute Delaunay triangulation and barycentric weights."""
        from scipy.spatial import Delaunay

        mesh_pts = self.wvg_props_output['mesh'].points
        x_min, x_max = mesh_pts[:, 0].min(), mesh_pts[:, 0].max()
        y_min, y_max = mesh_pts[:, 1].min(), mesh_pts[:, 1].max()

        plot_x = np.linspace(x_min, x_max, grid_resolution)
        plot_y = np.linspace(y_min, y_max, grid_resolution)
        X_plot, Y_plot = np.meshgrid(plot_x, plot_y)
        grid_pts = np.column_stack((X_plot.ravel(), Y_plot.ravel()))

        tri        = Delaunay(mesh_pts[:, :2])
        v_simplex  = tri.find_simplex(grid_pts)
        valid_grid = v_simplex >= 0

        v0 = np.zeros(grid_pts.shape[0], dtype=np.int32)
        v1 = np.zeros(grid_pts.shape[0], dtype=np.int32)
        v2 = np.zeros(grid_pts.shape[0], dtype=np.int32)
        w0 = np.zeros(grid_pts.shape[0], dtype=np.float64)
        w1 = np.zeros(grid_pts.shape[0], dtype=np.float64)
        w2 = np.zeros(grid_pts.shape[0], dtype=np.float64)

        if np.any(valid_grid):
            valid_simplices = v_simplex[valid_grid]
            v0[valid_grid]  = tri.simplices[valid_simplices, 0]
            v1[valid_grid]  = tri.simplices[valid_simplices, 1]
            v2[valid_grid]  = tri.simplices[valid_simplices, 2]

            transform = tri.transform[valid_simplices]
            r3        = tri.simplices[valid_simplices, 2].astype(np.intp)
            delta     = grid_pts[valid_grid] - tri.points[r3]
            bary      = np.einsum('nij,nj->ni', transform[:, :2], delta)
            w0[valid_grid] = bary[:, 0]
            w1[valid_grid] = bary[:, 1]
            w2[valid_grid] = 1.0 - bary[:, 0] - bary[:, 1]

        return X_plot, Y_plot, v0, v1, v2, w0, w1, w2, valid_grid

    # ------------------------------------------------------------------
    def interpolate_output_to_grid(self, E_output_batch,
                                   grid_resolution=DEFAULT_GRID_RESOLUTION):
        """Interpolate output spatial fields to a regular grid (always CPU)."""
        if grid_resolution not in self._delaunay_cache:
            self._delaunay_cache[grid_resolution] = \
                self._precompute_delaunay_grid(grid_resolution)
        X_plot, Y_plot, v0, v1, v2, w0, w1, w2, valid_grid = \
            self._delaunay_cache[grid_resolution]

        E_np            = np.asarray(E_output_batch)
        intensity_batch = np.abs(E_np) ** 2

        flat_interpolated = (
            intensity_batch[:, v0] * w0 +
            intensity_batch[:, v1] * w1 +
            intensity_batch[:, v2] * w2
        )
        flat_interpolated[:, ~valid_grid] = 0.0

        n_fields    = E_np.shape[0]
        uf_2d_batch = flat_interpolated.reshape(
            n_fields, grid_resolution, grid_resolution)
        return uf_2d_batch, X_plot, Y_plot

    # ------------------------------------------------------------------
    def _process_chunk(self, coeff_chunk, fg, proj, use_gpu):
        """Run the field-generation pipeline on a single coefficient chunk.

        Returns a plain numpy array of shape (chunk_size, n_modes).
        All intermediate arrays are local to this call and are released
        when it returns.
        """
        E_pupil  = fg.generate_field_profiles_batch(coeff_chunk)
        if use_gpu:
            E_lantern = fg.apply_pupil_to_lantern_jax(E_pupil)
        else:
            E_lantern = fg.apply_pupil_to_lantern(E_pupil)
        E_mesh   = fg.resample_to_mesh(E_lantern)
        u0_chunk = proj.project_batch(E_mesh)
        return np.asarray(u0_chunk)

    # ------------------------------------------------------------------
    def generate_batch_modal_coefficients(self, aberration_coeff_batch,
                                          use_gpu=None,
                                          chunk_size=default_gen_chunk_size):
        """
        Generate batch modal coefficients from aberration configurations.

        The input batch is split into chunks of *chunk_size* fields.  For
        each chunk the full pipeline (OPD → pupil → FFT → mesh resample →
        modal projection) is executed and its result accumulated in a plain
        numpy array on the host before the next chunk is started.  This
        limits peak memory to O(chunk_size) **regardless of the total batch
        size and regardless of which backend is used**.

        Parameters
        ----------
        aberration_coeff_batch : array-like, shape (n_configs, n_active_modes)
            Each row is one aberration configuration.  Zeros are appended
            automatically if n_active_modes < n_modes_total.
        use_gpu : bool or None, optional
            Override the instance-level ``field_gen_on_gpu`` setting.

            - ``None``  (default) — use the instance default.
            - ``False`` — force CPU/NumPy for this call.
            - ``True``  — force GPU/JAX for this call (requires JAX backend
              and initialised JAX generators).
        chunk_size : int or None, optional
            Number of fields per processing chunk.  Overrides the instance-
            level ``field_gen_chunk_size`` for this call only.

            - ``None``  — use the instance default (``field_gen_chunk_size``).
            - Positive int — use that chunk size.
            - ``0`` or negative — disable chunking (process whole batch).

        Returns
        -------
        u0_batch : np.ndarray, shape (n_configs, n_modes), complex128
            Normalised modal coefficients, always a plain numpy array.
        """
        # ------------------------------------------------------------------
        # Resolve backend choice.
        # ------------------------------------------------------------------
        if use_gpu is None:
            _use_gpu = self._field_gen_on_gpu_default
        else:
            _use_gpu = bool(use_gpu)

        if _use_gpu and backend != 'jax':
            raise RuntimeError(
                "use_gpu=True requires the JAX backend "
                "(set CBEAM_BACKEND=jax before importing cbeam).")
        if _use_gpu and self._field_gen_jax is None:
            raise RuntimeError(
                "JAX field generators have not been initialised.  "
                "Either construct BatchPropagationPipeline with "
                "field_gen_on_gpu=True, or call "
                "pipeline.enable_gpu_field_gen() first.")

        fg   = self._field_gen_jax  if _use_gpu else self._field_gen_np
        proj = self._modal_proj_jax if _use_gpu else self._modal_proj_np

        # ------------------------------------------------------------------
        # Resolve chunk size.
        # ------------------------------------------------------------------
        if chunk_size is None:
            _chunk_size = self.field_gen_chunk_size
        elif chunk_size <= 0:
            _chunk_size = None   # explicit disable
        else:
            _chunk_size = int(chunk_size)

        # ------------------------------------------------------------------
        # Prepare the full coefficient array (host numpy first, then
        # optionally moved to device inside _process_chunk).
        # ------------------------------------------------------------------
        coeff_np = np.asarray(aberration_coeff_batch, dtype=np.float64)
        if coeff_np.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {coeff_np.shape}")

        n_configs, n_active_modes = coeff_np.shape
        n_modes_total = fg.num_modes

        if n_active_modes < n_modes_total:
            padded = np.zeros((n_configs, n_modes_total), dtype=np.float64)
            padded[:, :n_active_modes] = coeff_np
            coeff_np = padded

        # ------------------------------------------------------------------
        # Process in chunks.
        # ------------------------------------------------------------------
        if _chunk_size is None or _chunk_size >= n_configs:
            # Single pass — no chunking overhead.
            coeff_chunk = self._to_device(coeff_np, _use_gpu)
            return self._process_chunk(coeff_chunk, fg, proj, _use_gpu)

        print(f"  Generating modal coefficients for {n_configs} fields "
              f"(chunk_size={_chunk_size}, "
              f"device={'GPU' if _use_gpu else 'CPU'}) ...", flush=True)

        u0_chunks = []
        for start in range(0, n_configs, _chunk_size):
            end         = min(start + _chunk_size, n_configs)
            coeff_chunk = self._to_device(coeff_np[start:end], _use_gpu)
            u0_chunk    = self._process_chunk(coeff_chunk, fg, proj, _use_gpu)
            u0_chunks.append(u0_chunk)
            print(f"    field-gen chunk {start}–{end-1} done.", flush=True)

        print("  Field generation complete.")
        return np.concatenate(u0_chunks, axis=0)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_device(coeff_np, use_gpu):
        """Move a numpy coefficient array to the appropriate device."""
        if use_gpu:
            return jnp.asarray(coeff_np, dtype=jnp.float64, device=_jax_device)
        return coeff_np   # already numpy, no copy needed

    # ------------------------------------------------------------------

    def propagate_batch_single(self, u0_batch, chunk_size=None):
        """
        Propagate a batch of modal coefficient vectors using a propagator that
        handles only one field at a time.

        This version never returns the full trajectory (us_batch = None) because
        individual fields may produce different z-grids, making stacking impossible.
        If full trajectories are needed, propagate each field separately.

        Parameters
        ----------
        u0_batch   : np.ndarray, shape (n_fields, n_modes)
        chunk_size : int or None

        Returns
        -------
        uf_batch  : np.ndarray (n_fields, n_modes)
        zs        : array of z values (from the first field; all fields share the same z grid)
        us_batch  : None (always)
        """
        n_fields = u0_batch.shape[0]

        if chunk_size is None:
            chunk_size = default_chunk_size if backend == 'jax' else n_fields

        print(f"  Propagating {n_fields} fields (single‑field mode, chunk_size={chunk_size}) ...", flush=True)

        uf_chunks   = []
        zs_first    = None
        multi_chunk = (n_fields > chunk_size)

        for start in range(0, n_fields, chunk_size):
            end = min(start + chunk_size, n_fields)
            chunk_uf = []
            for idx in range(start, end):
                field = u0_batch[idx]

                if backend == 'jax':
                    field = jnp.asarray(field, dtype=jnp.complex128, device=_jax_device)

                zs, us, uf = self.prop12.propagate(field)

                if zs_first is None:
                    zs_first = np.asarray(zs)

                chunk_uf.append(np.asarray(uf))

            uf_chunks.append(np.stack(chunk_uf, axis=0))   # (chunk_size, n_modes)
            print(f"    propagation chunk {start}–{end-1} done.", flush=True)

        uf_batch = np.concatenate(uf_chunks, axis=0)       # (n_fields, n_modes)

        # Never return trajectory because z-grids may differ between fields
        print("  (us_batch not returned because individual fields may have different z grids)")
        return uf_batch, zs_first, None
    def propagate_batch(self, u0_batch, chunk_size=None):
        """
        Propagate a batch of modal coefficient vectors through the waveguide.

        For the JAX backend the batch is split into chunks of *chunk_size*
        fields.  Each chunk is propagated, its output is immediately moved
        to host memory, and only then is the next chunk started, keeping
        peak GPU memory O(chunk_size) rather than O(n_batch).

        Parameters
        ----------
        u0_batch   : np.ndarray, shape (n_fields, n_modes)
        chunk_size : int or None
            If None a safe default is chosen:
              - JAX backend:   32  (tune upward if GPU has headroom)
              - numpy backend: full batch (no chunking needed)

        Returns
        -------
        uf_batch  : np.ndarray (n_fields, n_modes)
        zs        : array of z values (from the last chunk; all identical)
        us_batch  : np.ndarray (n_fields, n_z, n_modes) or None
                    Full trajectory — only populated for single-chunk runs
                    to avoid excessive memory use.
        """
        n_fields = u0_batch.shape[0]

        if chunk_size is None:
            chunk_size = default_chunk_size if backend == 'jax' else n_fields

        print(f"  Propagating {n_fields} fields "
              f"(chunk_size={chunk_size}) ...", flush=True)

        uf_chunks   = []
        us_chunks   = []
        zs_last     = None
        multi_chunk = (n_fields > chunk_size)

        for start in range(0, n_fields, chunk_size):
            end   = min(start + chunk_size, n_fields)
            chunk = u0_batch[start:end]

            if backend == 'jax':
                chunk = jnp.asarray(chunk, dtype=jnp.complex128,
                                    device=_jax_device)

            zs, us_grid, uf = self.prop12.propagate(chunk)

            uf_chunks.append(np.asarray(uf))
            zs_last = np.asarray(zs)

            if not multi_chunk:
                us_chunks.append(np.asarray(us_grid))

            print(f"    propagation chunk {start}–{end-1} done.", flush=True)

        uf_batch = np.concatenate(uf_chunks, axis=0)

        if multi_chunk:
            us_batch = None
            print("  (us_batch not returned for multi-chunk runs to save memory)")
        else:
            us_grid_full = np.concatenate(us_chunks, axis=0)
            us_batch = (np.transpose(us_grid_full, (1, 0, 2))
                        if us_grid_full.ndim == 3 else us_grid_full)

        print("  Batch propagation complete.")
        return uf_batch, zs_last, us_batch

    # ------------------------------------------------------------------
    def reconstruct_batch_output_fields(self, uf_batch):
        """Reconstruct spatial output fields from modal coefficients (CPU)."""
        modes_out = self.wvg_props_output['modes']
        return np.asarray(uf_batch) @ modes_out


# =====================================================================
# LAYER 5: VISUALISATION & ANALYSIS
# =====================================================================

def visualize_batch_output(uf_2d_batch, X_plot, Y_plot, titles=None):
    """Visualise a batch of output intensity maps."""
    n_fields = uf_2d_batch.shape[0]
    n_cols   = min(3, n_fields)
    n_rows   = (n_fields + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_fields == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n_fields):
        ax     = axes[i]
        im_log = np.log(np.abs(uf_2d_batch[i]) + 1e-10)
        im = ax.imshow(
            im_log, cmap='inferno',
            extent=[X_plot.min(), X_plot.max(),
                    Y_plot.min(), Y_plot.max()],
            origin='lower',
        )
        ax.set_title(titles[i] if titles else f"Field {i}")
        ax.set_xlabel('x (μm)')
        ax.set_ylabel('y (μm)')
        plt.colorbar(im, ax=ax, label='log(Intensity)')

    for i in range(n_fields, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def visualize_batch_hex_grid_signals(
    pipeline,
    E_output_batch,
    ideal_grid_positions,
    calibrate_subpixel_centers,
    collect_subpixel_signals,
    map_evaluated_to_ideal_geometry,
    display_hex_grid_plots,
    titles=None,
    grid_resolution=DEFAULT_GRID_RESOLUTION,
):
    """Execute centroid sub-pixel tracking and display hex grid configurations."""
    mesh_final            = pipeline.wvg_props_output['mesh']
    waveguide_modes_final = pipeline.wvg_props_output['modes']

    core_centers, _, _, dx, dy = calibrate_subpixel_centers(
        waveguide_modes_final, mesh_final)
    ideal_permutation = map_evaluated_to_ideal_geometry(
        core_centers, ideal_grid_positions)

    n_fields = E_output_batch.shape[0]
    print("\n" + "=" * 60)
    print("GEOMETRIC HEX SIGNAL EXTRACTION PROCESSING")
    print("=" * 60)

    uf_2d_batch, X_plot_out, Y_plot_out = pipeline.interpolate_output_to_grid(
        E_output_batch, grid_resolution)
    x0_out = X_plot_out[0, 0]
    y0_out = Y_plot_out[0, 0]

    for i in range(n_fields):
        field_title = titles[i] if titles else f"Field {i}"
        print(f"Processing core integration tracks for: {field_title}...")
        output_signals = collect_subpixel_signals(
            uf_2d_batch[i], x0_out, y0_out, dx, dy,
            core_centers, n=DEFAULT_SUBPIXEL_N)
        standardized_signals = np.zeros(19)
        standardized_signals[ideal_permutation] = output_signals
        print(f"Displaying Core Matrix for: {field_title}")
        display_hex_grid_plots(ideal_grid_positions, standardized_signals)


def batch_statistics(uf_batch, labels=None):
    """Compute and display statistics of batch propagation outputs."""
    n_fields = uf_batch.shape[0]
    print("\n" + "=" * 60)
    print("BATCH PROPAGATION STATISTICS")
    print("=" * 60)
    for i in range(n_fields):
        label       = labels[i] if labels else f"Field {i}"
        mode_powers = np.abs(uf_batch[i]) ** 2
        total_power = np.sum(mode_powers)
        max_mode    = np.argmax(mode_powers)
        top_3_power = np.sum(np.sort(mode_powers)[-3:])
        print(f"\n{label}:")
        print(f"  Total Power:          {total_power:.4f}")
        print(f"  Max Mode (idx):       {max_mode}")
        print(f"  Max Mode Power:       {mode_powers[max_mode]:.4f}")
        print(f"  Power in Top 3 Modes: {top_3_power:.4f}")


# =====================================================================
# LAYER 6: MAIN EXECUTION PIPELINE
# =====================================================================

def main_batch_propagation():
    """Main execution function for the batch propagation pipeline."""
    print("=" * 60)
    print("PHOTONIC LANTERN BATCH PROPAGATION PIPELINE")
    print("=" * 60)

    p = get_simulation_parameters()
    print(f"\n[1/5] Configuration loaded.")
    print(f"      Wavelength: {p['wl']} μm  |  Lantern length: {p['z_ex']} μm")
    print(f"      Global backend: {backend}")

    print(f"\n[2/5] Loading influence function...")
    import specula
    specula.init(0)
    from specula.data_objects.ifunc import IFunc
    ifunc = IFunc.restore(p["ifunc_file"])
    print(f"      Modes available: {len(ifunc.influence_function)}")

    print(f"\n[3/5] Building waveguide propagator...")
    prop12 = build_and_characterize_lantern(p)
    print(f"      Modes: 20, Segments: 2")

    print(f"\n[4/5] Initialising batch pipeline ...")
    # field_gen_on_gpu=False  → field generation uses NumPy/SciPy (CPU)
    # field_gen_chunk_size=64 → process 64 fields at a time during field gen
    pipeline = BatchPropagationPipeline(
        prop12, p, ifunc,
        field_gen_on_gpu=False,
        field_gen_chunk_size=64,
    )

    mode_indices  = [0, 3, 5]
    amplitudes_nm = [0.0, 100.0, 300.0, 600.0]
    aberration_configs = [
        {'mode_idx': m, 'amplitude_nm': a}
        for m in mode_indices for a in amplitudes_nm
    ]
    print(f"      Batch size: {len(aberration_configs)} fields")
    print(f"      Modes: {mode_indices}, Amplitudes (nm): {amplitudes_nm}")

    print(f"\n[5/5] Generating and propagating batch...")
    n_total      = pipeline._field_gen_np.num_modes
    coeff_matrix = np.zeros((len(aberration_configs), n_total), dtype=np.float64)
    for k, cfg in enumerate(aberration_configs):
        coeff_matrix[k, cfg['mode_idx']] = cfg['amplitude_nm']

    # chunk_size=None → use the instance default (field_gen_chunk_size=64).
    # Pass an explicit int to override, e.g. chunk_size=32 for tighter memory.
    u0_batch = pipeline.generate_batch_modal_coefficients(
        coeff_matrix, use_gpu=False)
    print(f"      Input field batch shape: {u0_batch.shape}")

    # chunk_size=8 is conservative for propagation; increase if GPU allows.
    uf_batch, zs, us_batch = pipeline.propagate_batch(u0_batch, chunk_size=8)
    print(f"      Output modal coefficient batch shape: {uf_batch.shape}")

    E_output_batch = pipeline.reconstruct_batch_output_fields(uf_batch)
    print(f"      Output spatial field batch shape: {E_output_batch.shape}")

    uf_2d_batch, X_plot, Y_plot = pipeline.interpolate_output_to_grid(E_output_batch)
    print(f"      Output grid batch shape: {uf_2d_batch.shape}")

    titles = [f"Mode {c['mode_idx']}, {c['amplitude_nm']:.0f} nm"
              for c in aberration_configs]

    print(f"\n[Results] Visualising batch output...")
    visualize_batch_output(uf_2d_batch, X_plot, Y_plot, titles)
    batch_statistics(uf_batch, titles)

    print("\n" + "=" * 60)
    print("BATCH PROPAGATION COMPLETE")
    print("=" * 60)

    return {
        'u0_batch':       u0_batch,
        'uf_batch':       uf_batch,
        'E_output_batch': E_output_batch,
        'uf_2d_batch':    uf_2d_batch,
        'zs':             zs,
        'us_batch':       us_batch,
        'configs':        aberration_configs,
        'pipeline':       pipeline,
    }


def create_random_aberration_configs(n, m, minv, maxv):
    """
    Generate *n* random aberration configurations each with *m* active modes.

    Args:
        n    (int):   Number of configurations.
        m    (int):   Number of active modes per config (modes 0 … m-1).
        minv (float): Minimum amplitude (nm).
        maxv (float): Maximum amplitude (nm).

    Returns:
        np.ndarray of shape (n, m), dtype float64.
    """
    return np.random.uniform(minv, maxv, (n, m)).astype(np.float64)

def create_ramp_aberration_configs(modes, n_steps, minv, maxv):
    """
    Generate aberration configurations that ramp each mode individually.

    For every mode index in *modes* a sequence of *n_steps* configurations is
    produced in which that mode's amplitude is varied linearly from *minv* to
    *maxv* while all other modes are kept at zero.  The resulting array has
    ``len(modes) * n_steps`` rows.

    Args:
        modes   (list[int]): mode indices to ramp, e.g. [0, 1, 2].
        n_steps (int):       number of amplitude steps per mode (≥ 2).
        minv    (float):     starting amplitude value (nm).
        maxv    (float):     ending amplitude value (nm).

    Returns:
        coeff_matrix : np.ndarray, shape (len(modes) * n_steps, max(modes)+1),
                       dtype float64.
                       Each row is one configuration; the column index equals
                       the mode index.
        labels       : list[str] of length len(modes) * n_steps, one human-
                       readable label per configuration, e.g.
                       "mode 2 | 150.0 nm".

    Example:
        # Ramp modes 0, 3, 5 over 10 steps from 0 to 500 nm
        coeff, labels = create_ramp_aberration_configs([0, 3, 5], 10, 0, 500)
        u0_batch = pipeline.generate_batch_modal_coefficients(coeff)
    """
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2 to define a ramp.")
    if len(modes) == 0:
        raise ValueError("modes list must not be empty.")

    amplitudes  = np.linspace(minv, maxv, n_steps)   # shape (n_steps,)
    n_cols      = max(modes) + 1
    n_configs   = len(modes) * n_steps

    coeff_matrix = np.zeros((n_configs, n_cols), dtype=np.float64)
    labels       = []

    for row, (mode_idx, amp) in enumerate(
            (m, a) for m in modes for a in amplitudes):
        coeff_matrix[row, mode_idx] = amp
        labels.append(f"mode {mode_idx} | {amp:.1f} nm")

    return coeff_matrix, labels

if __name__ == "__main__":
    results = main_batch_propagation()

    
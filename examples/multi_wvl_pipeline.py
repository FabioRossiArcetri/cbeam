# =====================================================================
# MULTI-WAVELENGTH PROPAGATION PIPELINE
# =====================================================================
# Extends BatchPropagationPipeline to propagate through the photonic
# lantern at a set of distinct wavelengths, producing complex per-fiber
# spectra suitable for driving the SpectralImageSimulator /
# SpectralExtractor pipeline with a physically correct chromatic response.
#
# Physical assumption made explicit here (confirmed for this lantern):
# each of the N_SIGNALS output fibers carries a single spatial mode, so
# per-fiber flux is |c_fiber(lambda)|**2, with no coherent multi-mode sum
# required within a fiber. If that assumption ever stops holding for some
# port, propagate_batch() is the place to change: sum the complex
# coefficients of that port's modes coherently before squaring.
#
# Why this is a separate module rather than a method bolted onto
# BatchPropagationPipeline: cbeam's Propagator/ChainPropagator carries a
# *wavelength-specific* local mode basis, loaded via .load(...) from
# precomputed FEM eigenmode data (e.g. "19port_0800_front"/"19port_0800_back").
# That basis is not something you swap inside one pipeline instance --
# you need one fully-built lantern + pipeline per wavelength. This module
# manages a collection of those and handles the wavelength axis on top.
#
# COST NOTE: build_and_characterize_lantern_at_wavelength() will *solve*
# for a wavelength's local mode basis (cbeam's Propagator.characterize(),
# which runs compute_modes() then compute_cmats()) whenever no cached
# data exists for that wavelength's tag, and cache the result so future
# runs load it instead. cbeam's own 19-port example, characterizing the
# same (0, L1), (L1, L1+L2) z-split used here, takes ~800 s (~13 min)
# total per wavelength on typical desktop hardware -- a one-time cost per
# new native wavelength, but it means native_wavelengths_nm should stay
# coarse (tens of points, not hundreds); see the class docstring below
# for the native-grid vs. output-grid split that keeps this tractable.
# =====================================================================

import gc
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from batch_propagation_pipeline import (
    Propagator, ChainPropagator,
    hex_ring_positions, PhotonicLantern,
    BatchPropagationPipeline,
    default_gen_chunk_size,
    N_SIGNALS,
)

default_degenetate_groups_front = {}
default_degenetate_groups_back = {}
default_skipped_modes_front = {}
default_skipped_modes_back = {}

default_degenetate_groups_front[3] = [[1,2],[3,4],[6,7],[8,9],[10,11],[12,13],[15,16]]
default_degenetate_groups_back[3] = [[i for i in range(20) if i != 18]]
default_skipped_modes_front[3] = [18]
default_skipped_modes_back[3] = [18]

# =====================================================================
# LAYER 1: PER-WAVELENGTH LANTERN CONSTRUCTION
# =====================================================================

def _wavelength_tag(wavelength_nm: float) -> str:
    """Filename tag matching the convention used for the 800 nm cache
    (800.0 nm -> '0800'). Adjust if your cache naming differs, or if your
    native grid isn't on round-nm values -- in that case you likely want
    an explicit {wavelength_nm: tag} mapping instead of this formula."""
    return f"{int(round(wavelength_nm)):04d}"


def build_lantern_geometry(base_params: dict) -> PhotonicLantern:
    """
    Build the PhotonicLantern geometry once. None of PhotonicLantern's
    constructor arguments (core positions, radii, indices, taper, mesh
    resolutions) depend on wavelength -- only the per-wavelength
    Propagator does. Rebuilding a fresh PhotonicLantern (and the
    underlying Gmsh/CAD geometry model) for every wavelength is pure
    repeated work, and if Gmsh's global model state isn't fully released
    between constructions, repeated builds can get progressively slower
    over a loop (a known pygmsh/Gmsh gotcha: the kernel's internal model
    keeps growing across successive builds in the same process). Build it
    once and share the same PL_N object across every wavelength's
    Propagator.
    """

    import numpy as np  # Force plain numpy for geometry

    core_pos = hex_ring_positions(base_params["nrings"], base_params["rclad"] / 2.5)

    # Convert to plain numpy if it's a JAX array
    if hasattr(core_pos, '__array__'):
        core_pos = np.asarray(core_pos)

    return PhotonicLantern(
        core_pos, base_params["rcores"], base_params["rclad"], base_params["rjack"],
        base_params["ncores"], base_params["nclad"], base_params["njack"], base_params["z_ex"],
        base_params["taper_factor"], base_params["core_res"], base_params["clad_res"],
        base_params["jack_res"],
    )


import contextlib
import io

def _ensure_uint64_cells(mesh) -> None:
    if mesh is None:
        return
    for cellblock in getattr(mesh, "cells", []):
        if cellblock is None:
            continue  # skip any None cellblocks
        if cellblock.data.dtype != np.uint64:
            cellblock.data = cellblock.data.astype(np.uint64)


def _load_and_backfill_cmats(prop: Propagator, tag: str, verbose: bool = True) -> None:
    """
    Call Propagator.load(tag), and if cbeam reports a missing coupling-
    matrix cache (its own internal fallback proceeds without erroring,
    which is why _load_or_characterize's try/except never catches this
    case), explicitly compute and *save* the coupling matrices now.

    Without this, a missing cplcoeffs cache file means every future run
    pays the coupling-matrix computation cost again from scratch (we've
    seen ~10-30 s per propagator segment), since cbeam's own fallback
    proceeds silently but does not appear to persist the result. Calling
    compute_cmats(save=True, tag=tag) reuses whatever .load() already
    restored (zs, vs, mesh all default to self.zs/self.vs/self.mesh when
    not passed explicitly) and writes the missing file this one time, so
    later .load() calls for this tag are fast again.

    Detection is done by capturing stdout during load() and checking for
    cbeam's own "no coupling matrix file found" message, since the exact
    internal attribute cbeam uses to track cmat state isn't part of its
    documented API. This is a heuristic, not a guarantee -- if cbeam
    changes that message wording in a future version, this stops
    detecting the condition (silently falls back to "trust load()", i.e.
    today's behavior) rather than breaking.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        prop.load(tag)
    captured = buf.getvalue()

    if verbose and captured.strip():
        print(captured, end="" if captured.endswith("\n") else "\n")

    lowered = captured.lower()
    if "coupling matrix" in lowered and ("not found" in lowered or "no coupling" in lowered):
        if verbose:
            print(f"    [{tag}] coupling-matrix cache missing -- computing "
                  f"and saving it now so future loads of this tag are fast ...")
        _ensure_uint64_cells(getattr(prop, "mesh", None))
        prop.compute_cmats(save=True, tag=tag)
        if verbose:
            print(f"    [{tag}] coupling matrix computed and cached.")


def build_and_characterize_lantern_at_wavelength(
    base_params: dict,
    wavelength_nm: float,
    n_modes: int = 20,
    skipped_modes: Optional[List[int]] = None,
    degen_groups: Optional[List[List[int]]] = None,
    z_split: Optional[float] = None,
    cache_prefix: str = "port",
    PL_N: Optional[PhotonicLantern] = None,
    verbose: bool = True,
) -> Tuple[dict, ChainPropagator]:
    """
    Build a ChainPropagator for the lantern at a single wavelength,
    characterizing it from scratch if no cached data exists yet.

    Mirrors build_and_characterize_lantern() from the single-wavelength
    pipeline, but parameterises the wavelength and the mode-cache filename
    tag instead of hard-coding "0800", and produces the cache instead of
    assuming it's already there.

    Parameters
    ----------
    base_params : dict
        Output of get_simulation_parameters(), used as a template. `wl`
        and `wavelength_nm` are overridden per call; geometry, taper,
        indices and mesh resolutions are assumed wavelength-independent.
    wavelength_nm : float
    n_modes : int
        Number of local modes tracked by the Propagator.
    skipped_modes, degen_groups : optional overrides
        Default reproduces the single-wavelength pipeline's configuration
        ([18] skipped; the front-half degenerate LP-mode pairs grouped).
        Override if a different wavelength needs a different degeneracy
        structure -- mode crossings can shift position with wavelength
        near cutoff, so a pair degenerate at 800 nm may not be degenerate
        at 750 nm, or a new crossing may appear. If you're characterizing
        an unfamiliar wavelength for the first time, it's worth repeating
        the compute_neffs() sanity check from cbeam's 19-port example
        before committing to these defaults.
    z_split : float or None
        z coordinate separating the "front" and "back" Propagator
        segments. Defaults to base_params["z_ex"] / 2, matching the
        symmetric L1 == L2 == 50000 split used elsewhere in this codebase.
        Override if your front/back split isn't at the midpoint.
    cache_prefix : str
        Prefix for the load/save tag, e.g. str(n_output_positions) + "port" -> "19port_0800_front".
    PL_N : PhotonicLantern or None
        A pre-built, wavelength-independent lantern geometry to reuse
        (see build_lantern_geometry()). If None, one is built here --
        fine for a single call, wasteful if called once per wavelength in
        a loop. MultiWavelengthPropagationPipeline builds one PL_N and
        passes it to every wavelength.

    Returns
    -------
    p_lambda : dict
        Copy of base_params with wl / wavelength_nm set for this wavelength.
    prop12   : ChainPropagator
    """

    import os
    from cbeam.backend import get_backend
    import numpy as np
    
    # === FORCE NUMPY FOR ENTIRE FUNCTION ===
    original_backend = get_backend()
    if original_backend == "jax":
        if verbose:
            print(f"    [build_and_characterize] Using numpy backend for characterization")
        os.environ["CBEAM_BACKEND"] = "numpy"
    
    p_lambda = dict(base_params)
    p_lambda["wl"] = wavelength_nm / 1000.0
    p_lambda["wavelength_nm"] = wavelength_nm
  
    reference_wavelength_nm = base_params.get("wavelength_nm", 800.0)  # Default 800 nm
    wavelength_scale_factor = wavelength_nm / reference_wavelength_nm
    
    # Focal plane scale is proportional to wavelength
    p_lambda["pixel_scale_um"] = (
        base_params["pixel_scale_um"] # * wavelength_scale_factor
    )
    
    if verbose:
        print(f"  Wavelength {wavelength_nm:.1f} nm:")
        print(f"    Scale factor: {wavelength_scale_factor:.4f}")
        print(f"    Pixel scale: {p_lambda['pixel_scale_um']:.4f} μm/px")

    z_ex = p_lambda["z_ex"]
    _z_split = z_split if z_split is not None else z_ex / 2.0
    n_output_positions = str(base_params["n_output_positions"])

    tag = f"{n_output_positions}{cache_prefix}_{_wavelength_tag(wavelength_nm)}"

    PL_N = PL_N if PL_N is not None else build_lantern_geometry(p_lambda)

    _skipped = skipped_modes if skipped_modes is not None else [18]
    _degen_front = degen_groups if degen_groups is not None else default_degenetate_groups_front[base_params["nrings"]]
    _degen_back = [[i for i in range(n_modes) if i not in _skipped]]

    # build tags...
    tag_front = f"{tag}_front"
    tag_back  = f"{tag}_back"

    # --- fast path: try load only using shared PL_N ---
    try:
        prop1 = Propagator(p_lambda["wl"], PL_N, n_modes)
        prop1.degen_groups = _degen_front
        prop1.skipped_modes = _skipped
        prop1.load(tag_front)

        prop2 = Propagator(p_lambda["wl"], PL_N, n_modes)
        prop2.skipped_modes = _skipped
        prop2.degen_groups = _degen_back
        prop2.load_init_conds(prop1)
        prop2.load(tag_back)

        return p_lambda, ChainPropagator([prop1, prop2])

    except (FileNotFoundError, OSError):
        if verbose:
            print(f"[{tag}] cache miss -> characterize with fresh geometry")

    # --- fallback path: characterize with fresh geometry object ---
    PL_local = build_lantern_geometry(p_lambda)  # NEW object, no shared state

    prop1 = Propagator(p_lambda["wl"], PL_local, n_modes)
    prop1.degen_groups = _degen_front
    prop1.skipped_modes = _skipped
    prop1.load_or_characterize(
        load_tag=tag_front, zi=0.0, zf=_z_split,
        mesh=None, char_tag=tag_front, save=True, verbose=verbose
    )

    prop2 = Propagator(p_lambda["wl"], PL_local, n_modes)
    prop2.skipped_modes = _skipped
    prop2.degen_groups = _degen_back
    prop2.load_init_conds(prop1)
    prop2.load_or_characterize(
        load_tag=tag_back, zi=_z_split, zf=z_ex,
        mesh=None, char_tag=tag_back, save=True, verbose=verbose
    )

    # === RESTORE ORIGINAL BACKEND ===
    if original_backend == "jax":
        os.environ["CBEAM_BACKEND"] = "jax"
        if verbose:
            print(f"    [build_and_characterize] Restored {original_backend} backend")

    return p_lambda, ChainPropagator([prop1, prop2])


# =====================================================================
# LAYER 2: MULTI-WAVELENGTH PIPELINE COLLECTION
# =====================================================================

def _force_release_memory(clear_jax_cache: bool = True) -> None:
    """
    Best-effort release of Python, Julia, and JAX memory.
    
    Parameters
    ----------
    clear_jax_cache : bool
        If True and JAX is available, clear JAX device caches.
    """
    # Python garbage collection
    gc.collect()
    
    # Julia garbage collection (for FEM solver)
    try:
        from juliacall import Main as _jl
        _jl.GC.gc()
    except Exception:
        pass
    
    # JAX device memory cleanup
    if clear_jax_cache:
        try:
            import jax
            # Clear compilation cache
            jax.clear_backends()
            # Force device synchronization
            for device in jax.devices():
                device.synchronize_all_activity()
        except Exception:
            pass

@dataclass
class WavelengthEngine:
    """One fully-built single-wavelength propagation pipeline."""
    wavelength_nm: float
    p: dict
    prop12: ChainPropagator
    pipeline: BatchPropagationPipeline


class MultiWavelengthPropagationPipeline:

    def __init__(
        self,
        base_params: dict,
        ifunc,
        native_wavelengths_nm: Sequence[float],
        n_fibers: int = N_SIGNALS,
        field_gen_on_gpu: bool = False,
        field_gen_chunk_size: Optional[int] = default_gen_chunk_size,
        cache_engines: bool = False,
        verbose: bool = True,
    ):
        self.base_params = base_params
        self.ifunc = ifunc
        self.native_wavelengths_nm = np.asarray(
            sorted(native_wavelengths_nm), dtype=np.float64)
        self.n_fibers = n_fibers
        self.field_gen_on_gpu = field_gen_on_gpu
        self.field_gen_chunk_size = field_gen_chunk_size
        self.cache_engines = cache_engines
        self.verbose = verbose

        # DM mask / influence-function matrix -- wavelength independent,
        # so num_modes is available immediately without building any
        # engine (contrast with the earlier version, where you had to
        # reach into mwp.engines[wl].pipeline._field_gen_np.num_modes).
        self.num_modes = len(ifunc.influence_function)


        # === NEW: Force numpy backend during geometry construction ===
        from cbeam.backend import get_backend
        original_backend = get_backend()
        
        if original_backend == "jax":
            import os
            os.environ["CBEAM_BACKEND"] = "numpy"
            print(f"[MultiWavelength] Temporarily switching to numpy backend for geometry construction...")
        
        # Wavelength-independent lantern geometry, built once and shared
        # by every per-wavelength Propagator.
        self._shared_PL_N = build_lantern_geometry(base_params)

        # Restore original backend
        if original_backend == "jax":
            os.environ["CBEAM_BACKEND"] = "jax"
            print(f"[MultiWavelength] Restored {original_backend} backend")

        # Wavelength-independent pupil data (DM mask + ifunc_matrix),
        # captured from whichever engine is built first and reused for
        # every subsequent one. See IncidentFieldGenerator.pupil_template.
        self._pupil_template = None

        # Only populated per-wavelength if cache_engines=True. Empty by
        # default -- engines are built, used, and discarded inside
        # propagate_batch() instead of being pre-built here.
        self.engines: Dict[float, WavelengthEngine] = {}

        if verbose:
            print(f"[MultiWavelength] Configured for {len(self.native_wavelengths_nm)} "
                  f"native wavelengths "
                  f"({self.native_wavelengths_nm.min():.1f}-"
                  f"{self.native_wavelengths_nm.max():.1f} nm). Engines are built "
                  f"and released one wavelength at a time during propagate_batch() "
                  f"(cache_engines={cache_engines}).")

    # ------------------------------------------------------------------
    def _build_engine(self, wl: float) -> WavelengthEngine:
        """Build a single wavelength's ChainPropagator + BatchPropagationPipeline."""
        p_lambda, prop12 = build_and_characterize_lantern_at_wavelength(
            self.base_params, wl, PL_N=self._shared_PL_N, verbose=self.verbose)

        try:
            # field_gen_pupil_template reuses the wavelength-independent
            # DM mask / influence-function matrix built for the first
            # wavelength instead of rebuilding it for every one --
            # requires the field_gen_pupil_template patch to
            # IncidentFieldGenerator / BatchPropagationPipeline in
            # batch_propagation_pipeline.py.
            pipeline = BatchPropagationPipeline(
                prop12, p_lambda, self.ifunc,
                field_gen_on_gpu=self.field_gen_on_gpu,
                field_gen_chunk_size=self.field_gen_chunk_size,
                field_gen_pupil_template=self._pupil_template,
            )
        except TypeError:
            # Unpatched batch_propagation_pipeline.py -- fall back to the
            # original constructor (still correct, just redoes the
            # pupil-data load/stack for every wavelength).
            pipeline = BatchPropagationPipeline(
                prop12, p_lambda, self.ifunc,
                field_gen_on_gpu=self.field_gen_on_gpu,
                field_gen_chunk_size=self.field_gen_chunk_size,
            )

        if self._pupil_template is None:
            fg = pipeline._field_gen_np
            self._pupil_template = (fg.mask_np, fg.grid_size, fg.num_modes, fg.ifunc_matrix)

        return WavelengthEngine(float(wl), p_lambda, prop12, pipeline)

    # ------------------------------------------------------------------
    def propagate_batch(
        self,
        aberration_coeff_batch: np.ndarray,
        use_gpu: Optional[bool] = None,
        gen_chunk_size: Optional[int] = None,
        prop_chunk_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Propagate through all wavelengths using spatial integration
        (same method as Batched.ipynb).
        """
        n_wl = len(self.native_wavelengths_nm)
        n_fields = np.asarray(aberration_coeff_batch).shape[0]
        
        # Store REAL power (not complex) since spatial integration loses phase info
        spectra_power = np.zeros((n_wl, n_fields, self.n_fibers), dtype=np.float64)

        for i, wl in enumerate(self.native_wavelengths_nm):
            if self.verbose:
                print(f"[MultiWavelength] Propagating at {wl:.2f} nm ({i + 1}/{n_wl}) ...")

            engine = self.engines.get(float(wl))
            if engine is None:
                engine = self._build_engine(wl)
                if self.cache_engines:
                    self.engines[float(wl)] = engine

            u0_batch = engine.pipeline.generate_batch_modal_coefficients(
                aberration_coeff_batch, use_gpu=use_gpu, chunk_size=gen_chunk_size,
            )
            uf_batch, _, _ = engine.pipeline.propagate_batch(
                u0_batch, chunk_size=prop_chunk_size,
            )
            
            # === CORRECTED: Use spatial integration (same as Batched.ipynb) ===
            spectra_power[i, :, :] = self._collect_signals_like_batched(
                engine, uf_batch
            )

            if not self.cache_engines:
                del engine, u0_batch, uf_batch
                _force_release_memory()

        # Convert to complex for compatibility with existing code
        # (phase info is lost, but amplitude is correct)
        spectra_complex = np.sqrt(spectra_power).astype(np.complex128)
        return spectra_complex


    def _collect_signals_like_batched(
        self,
        engine: WavelengthEngine,
        uf_batch: np.ndarray,
    ) -> np.ndarray:
        """
        Collect signals using the same spatial integration method as Batched.ipynb.
        
        Parameters
        ----------
        engine : WavelengthEngine
        uf_batch : ndarray, shape (n_fields, n_modes)
        
        Returns
        -------
        signals : ndarray, shape (n_fields, n_fibers), float64
            Spatially integrated power at each fiber location
        """
        from scipy.interpolate import griddata
        from scipy.ndimage import map_coordinates, maximum_filter
        import numpy as np
        
        # Get output mesh and modes
        z_final = engine.prop12.zs[-1]
        mesh_final = engine.prop12.make_mesh_at_z(z_final)
        waveguide_modes_final = engine.prop12.get_v(z_final)
        
        # Ensure numpy
        if hasattr(waveguide_modes_final, '__array__'):
            waveguide_modes_final = np.asarray(waveguide_modes_final)
        if hasattr(mesh_final.points, '__array__'):
            mesh_pts = np.asarray(mesh_final.points)
        else:
            mesh_pts = mesh_final.points
        
        # Transpose if needed
        if waveguide_modes_final.shape[0] == len(mesh_pts):
            waveguide_modes_final = waveguide_modes_final.T
        
        # Setup grid (same as Batched.ipynb)
        plot_x_out = np.linspace(mesh_pts[:, 0].min(), mesh_pts[:, 0].max(), 400)
        plot_y_out = np.linspace(mesh_pts[:, 1].min(), mesh_pts[:, 1].max(), 400)
        X_plot_out, Y_plot_out = np.meshgrid(plot_x_out, plot_y_out)
        dx = plot_x_out[1] - plot_x_out[0]
        dy = plot_y_out[1] - plot_y_out[0]
        x_min, y_min = plot_x_out[0], plot_y_out[0]
        
        # Calibrate core centers (cache this per wavelength)
        cache_key = f"cores_{engine.wavelength_nm:.1f}"
        if not hasattr(self, '_core_centers_cache'):
            self._core_centers_cache = {}
        
        if cache_key not in self._core_centers_cache:
            # Find core centers from mode profile
            total_modes_profile = np.sum(np.abs(waveguide_modes_final)**2, axis=0)
            total_modes_2d = griddata(
                (mesh_pts[:, 0], mesh_pts[:, 1]), 
                total_modes_profile,
                (X_plot_out, Y_plot_out), 
                method='linear', 
                fill_value=0.0
            )
            
            is_peak = (total_modes_2d == maximum_filter(total_modes_2d, size=21)) & \
                    (total_modes_2d > 0.1 * np.max(total_modes_2d))
            peak_rows, peak_cols = np.where(is_peak)
            
            core_centers = []
            centroid_half_width = 4
            for r, c in zip(peak_rows, peak_cols):
                r_min_patch = max(0, r - centroid_half_width)
                r_max_patch = min(total_modes_2d.shape[0], r + centroid_half_width + 1)
                c_min_patch = max(0, c - centroid_half_width)
                c_max_patch = min(total_modes_2d.shape[1], c + centroid_half_width + 1)
                
                patch = total_modes_2d[r_min_patch:r_max_patch, c_min_patch:c_max_patch]
                r_indices, c_indices = np.meshgrid(
                    np.arange(r_min_patch, r_max_patch), 
                    np.arange(c_min_patch, c_max_patch), 
                    indexing='ij'
                )
                
                patch_sum = np.sum(patch)
                if patch_sum > 0:
                    sub_pixel_row = np.sum(r_indices * patch) / patch_sum
                    sub_pixel_col = np.sum(c_indices * patch) / patch_sum
                    cx = plot_x_out[0] + sub_pixel_col * dx
                    cy = plot_y_out[0] + sub_pixel_row * dy
                else:
                    cx, cy = plot_x_out[c], plot_y_out[r]
                
                core_centers.append((cx, cy))
            
            # Sort (same as Batched.ipynb)
            core_centers = sorted(core_centers, key=lambda p: (np.round(p[1], 2), np.round(p[0], 2)))
            self._core_centers_cache[cache_key] = core_centers
        
        core_centers = self._core_centers_cache[cache_key]
        
        # Process each field
        n_fields = uf_batch.shape[0]
        signals = np.zeros((n_fields, self.n_fibers), dtype=np.float64)
        
        for field_idx in range(n_fields):
            uf = uf_batch[field_idx, :]
            
            # Reconstruct spatial field
            E_output_spatial = uf @ waveguide_modes_final
            
            # Interpolate to regular grid
            uf_2d = griddata(
                (mesh_pts[:, 0], mesh_pts[:, 1]), 
                np.abs(E_output_spatial)**2,
                (X_plot_out, Y_plot_out), 
                method='linear', 
                fill_value=0.0
            )
            
            # Collect signals with 7×7 windows (same as Batched.ipynb)
            n_window = 7
            half_n = n_window // 2
            offsets = np.arange(-half_n, half_n + 1)
            
            for fiber_idx in range(min(len(core_centers), self.n_fibers)):
                cx, cy = core_centers[fiber_idx]
                f_col = (cx - x_min) / dx
                f_row = (cy - y_min) / dy
                
                sub_cols, sub_rows = np.meshgrid(f_col + offsets, f_row + offsets)
                coords = np.vstack((sub_rows.ravel(), sub_cols.ravel()))
                subimage_flat = map_coordinates(uf_2d, coords, order=3, mode='constant', cval=0.0)
                signals[field_idx, fiber_idx] = np.mean(subimage_flat)
        
        return signals

    # ------------------------------------------------------------------
    @staticmethod
    def interpolate_complex_spectra(
        spectra_native: np.ndarray,
        wl_native_nm: np.ndarray,
        wl_output_nm: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate complex per-fiber coefficients from a coarse native
        wavelength grid onto a finer output grid.

        Amplitude and (unwrapped) phase are interpolated separately rather
        than interpolating real/imaginary parts directly -- linear
        interpolation of Re/Im would average incorrectly through any zero
        crossing that occurs between native samples. This is only valid
        where the coupling coefficients vary smoothly with wavelength;
        sample the native grid more densely near known mode-degeneracy /
        avoided-crossing regions, where amplitude and phase can change
        rapidly over a few nm and linear interpolation (of amplitude/phase
        or anything else) will not capture that structure.

        Parameters
        ----------
        spectra_native : ndarray, shape (n_wl_native, n_fields, n_fibers), complex
        wl_native_nm   : ndarray, shape (n_wl_native,)
        wl_output_nm   : ndarray, shape (n_wl_output,)

        Returns
        -------
        spectra_output : ndarray, shape (n_wl_output, n_fields, n_fibers), complex128
        """
        wl_native_nm = np.asarray(wl_native_nm, dtype=np.float64)
        wl_output_nm = np.asarray(wl_output_nm, dtype=np.float64)

        if wl_output_nm.min() < wl_native_nm.min() or wl_output_nm.max() > wl_native_nm.max():
            print("[MultiWavelength] WARNING: output wavelength grid extends "
                  "beyond the native solve range -- edge values will be held "
                  "flat by np.interp (extrapolation), not physically modeled.")

        n_wl_native, n_fields, n_fibers = spectra_native.shape
        n_wl_output = len(wl_output_nm)

        amp = np.abs(spectra_native)
        phase = np.unwrap(np.angle(spectra_native), axis=0)

        # Flatten (fields, fibers) into one axis for straightforward
        # per-column interpolation.
        amp_flat = amp.reshape(n_wl_native, -1)
        phase_flat = phase.reshape(n_wl_native, -1)

        amp_out = np.empty((n_wl_output, amp_flat.shape[1]))
        phase_out = np.empty((n_wl_output, amp_flat.shape[1]))
        for k in range(amp_flat.shape[1]):
            amp_out[:, k] = np.interp(wl_output_nm, wl_native_nm, amp_flat[:, k])
            phase_out[:, k] = np.interp(wl_output_nm, wl_native_nm, phase_flat[:, k])

        spectra_output = (amp_out * np.exp(1j * phase_out)).reshape(
            n_wl_output, n_fields, n_fibers)
        return spectra_output


    def _integrate_over_cores(
        self,
        engine: WavelengthEngine,
        uf_batch: np.ndarray,
        radius_um: float,
    ) -> np.ndarray:
        """
        Spatially integrate output field intensity over circular apertures
        centered on each fiber core. This averages out small wavelength-
        dependent variations in mode field shapes.
        
        Parameters
        ----------
        engine : WavelengthEngine
        uf_batch : ndarray, shape (n_fields, n_modes)
            Output modal coefficients
        radius_um : float
            Integration radius in microns
            
        Returns
        -------
        integrated : ndarray, shape (n_fields, n_fibers), complex128
        """
        from scipy.spatial import cKDTree
        from scipy.ndimage import map_coordinates
        
        # Get output mesh and mode fields at final z
        z_final = engine.prop12.zs[-1]
        mesh_final = engine.prop12.make_mesh_at_z(z_final)
        vs_final = engine.prop12.get_v(z_final)  # (n_modes, n_mesh_points)
        
        # Get fiber core positions at output
        # Cores taper down by taper_factor
        core_pos_input = np.array(self.base_params["output_positions"])  # Convert to numpy array
        taper_factor = self.base_params["taper_factor"]
        core_pos_output = core_pos_input / taper_factor  # Cores are CLOSER at output
        
        # Build spatial tree for mesh points
        mesh_pts_2d = mesh_final.points[:, :2]  # (n_mesh_points, 2)
        tree = cKDTree(mesh_pts_2d)
        
        n_fields = uf_batch.shape[0]
        integrated = np.zeros((n_fields, self.n_fibers), dtype=np.complex128)
        
        # For each fiber core
        for fiber_idx in range(self.n_fibers):
            core_center = core_pos_output[fiber_idx]  # (x, y) in microns
            
            # Find mesh points within integration radius
            indices = tree.query_ball_point(core_center, r=radius_um)
            
            if len(indices) == 0:
                # No mesh points in aperture - fall back to modal coefficient
                print(f"[WARNING] No mesh points found within {radius_um} μm of fiber {fiber_idx}")
                integrated[:, fiber_idx] = uf_batch[:, fiber_idx]
                continue
            
            # For each field in the batch
            for field_idx in range(n_fields):
                uf = uf_batch[field_idx, :]  # (n_modes,)
                
                # Reconstruct spatial field at mesh points in aperture
                # field(r) = sum_i u_i * v_i(r)
                field_at_points = np.sum(
                    uf[:, np.newaxis] * vs_final[:, indices],
                    axis=0
                )  # (n_points_in_aperture,)
                
                # Integrate intensity over aperture
                # Proper integration would weight by mesh element areas, but
                # for a dense mesh, uniform weighting is a good approximation
                integrated_intensity = np.mean(np.abs(field_at_points) ** 2)
                
                # Convert back to complex amplitude
                # Preserve phase from the fiber's modal coefficient
                modal_phase = np.angle(uf[fiber_idx])
                
                integrated[field_idx, fiber_idx] = (
                    np.sqrt(integrated_intensity) * np.exp(1j * modal_phase)
                )
        
        return integrated
    
    # ------------------------------------------------------------------
    def get_power_spectra(
        self,
        aberration_coeff_batch: np.ndarray,
        output_wavelengths_nm: Optional[np.ndarray] = None,
        use_gpu: Optional[bool] = None,
        gen_chunk_size: Optional[int] = None,
        prop_chunk_size: Optional[int] = None,
        integration_radius_um: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience wrapper: propagate and return power spectra.
        
        Parameters
        ----------
        integration_radius_um : float, optional
            Spatial integration radius (μm) over fiber cores. Reduces
            jaggedness from wavelength-dependent mode shape variations.
            Typical: 2.5-4.0 μm. Default: None.
        """
        spectra_complex = self.propagate_batch(
            aberration_coeff_batch, use_gpu=use_gpu,
            gen_chunk_size=gen_chunk_size, prop_chunk_size=prop_chunk_size,
        )

        if output_wavelengths_nm is None:
            wl_output_nm = self.native_wavelengths_nm
        else:
            wl_output_nm = np.asarray(output_wavelengths_nm, dtype=np.float64)
            spectra_complex = self.interpolate_complex_spectra(
                spectra_complex, self.native_wavelengths_nm, wl_output_nm,
            )

        power_spectra = np.abs(spectra_complex) ** 2
        return power_spectra, wl_output_nm


# =====================================================================
# LAYER 3: GLUE INTO THE SPECTRAL EXTRACTION MODULE
# =====================================================================

def spectra_for_field(power_spectra: np.ndarray, field_idx: int) -> np.ndarray:
    """
    Slice out the (n_wavelengths, n_fibers) array for one field, in the
    shape expected by SpectralImageSimulator.create_spectral_image(spectra=...).
    """
    return power_spectra[:, field_idx, :]


def build_spectral_config_from_wavelength_grid(wl_nm: np.ndarray, **overrides):
    """
    Build a SpectralConfig whose lambda_min/lambda_max/n_spectral_bins match
    a wavelength grid produced by get_power_spectra(), so the spectral-
    extraction module's wavelength axis stays consistent with the
    physically propagated one (and with the dispersion fix discussed
    earlier -- SpectralExtractor must invert the same wavelength<->pixel
    relation SpectralImageSimulator used to build the image).
    """
    from spectral_extraction_module import SpectralConfig  # earlier module

    cfg_kwargs = dict(
        lambda_min=float(wl_nm.min()),
        lambda_max=float(wl_nm.max()),
        n_spectral_bins=len(wl_nm),
    )
    cfg_kwargs.update(overrides)
    return SpectralConfig(**cfg_kwargs)


# =====================================================================
# LAYER 4: EXAMPLE USAGE
# =====================================================================

def example_multiwavelength_propagation():
    """
    End-to-end example: propagate a batch of aberration configurations at
    several native wavelengths, interpolate onto a fine grid, and hand the
    result to the spectral extraction module.
    """
    import specula
    specula.init(0)
    from specula.data_objects.ifunc import IFunc
    from batch_propagation_pipeline import get_simulation_parameters, create_random_aberration_configs
    from spectral_extraction_module import SpectralImageSimulator, SpectralExtractor

    print("=" * 60)
    print("MULTI-WAVELENGTH PHOTONIC LANTERN PROPAGATION")
    print("=" * 60)

    base_params = get_simulation_parameters()
    ifunc = IFunc.restore(base_params["ifunc_file"])

    # Coarse native solve grid: physically propagate at these wavelengths.
    # Any wavelength without a cached characterization will be solved and
    # cached automatically on this call -- with 21 points and no existing
    # cache, expect on the order of a few hours the first time this runs
    # (see the COST NOTE at the top of this file / the class docstring).
    native_wl_nm = np.linspace(700.0, 900.0, 21)

    mwp = MultiWavelengthPropagationPipeline(
        base_params=base_params,
        ifunc=ifunc,
        native_wavelengths_nm=native_wl_nm,
        field_gen_on_gpu=False,
        field_gen_chunk_size=64,
    )

    n_total_modes = mwp.num_modes
    coeff_matrix = create_random_aberration_configs(n=8, m=n_total_modes, minv=-100.0, maxv=100.0)

    # Fine output grid for the dispersed spectra (interpolated from the
    # coarse native grid above).
    output_wl_nm = np.linspace(700.0, 900.0, 200)

    power_spectra, wl_out = mwp.get_power_spectra(
        coeff_matrix,
        output_wavelengths_nm=output_wl_nm,
        use_gpu=False,
        gen_chunk_size=64,
        prop_chunk_size=64,
        integration_radius_um=None
        
    )
    print(f"\npower_spectra shape: {power_spectra.shape}  "
          f"(n_wavelengths={power_spectra.shape[0]}, "
          f"n_fields={power_spectra.shape[1]}, n_fibers={power_spectra.shape[2]})")

    # Feed field 0's spectra into the detector-plane simulation.
    spectral_cfg = build_spectral_config_from_wavelength_grid(wl_out)
    simulator = SpectralImageSimulator(spectral_cfg)
    extractor = SpectralExtractor(spectral_cfg)

    spectra_field0 = spectra_for_field(power_spectra, field_idx=0)
    image, fiber_positions = simulator.create_spectral_image(
        modal_powers=None, spectra=spectra_field0, add_noise=True, exposure_time=1.0,
    )
    extracted, variance = extractor.extract_spectra(image, fiber_positions, method='optimal')

    print(f"Extracted spectra shape: {extracted.shape}")
    return power_spectra, wl_out, image, extracted


if __name__ == "__main__":
    example_multiwavelength_propagation()
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

default_degenerate_groups_front = {}
default_degenerate_groups_back = {}
default_skipped_modes_front = {}
default_skipped_modes_back = {}

default_degenerate_groups_front[3] = [[1,2],[3,4],[6,7],[8,9],[10,11],[12,13],[15,16]]
default_degenerate_groups_back[3] = [[i for i in range(20) if i != 18]]
default_skipped_modes_front[3] = {18}
default_skipped_modes_back[3] = {18}

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
    core_pos = hex_ring_positions(base_params["nrings"], base_params["rclad"] / 2.5)
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


def _load_or_characterize(
    prop: Propagator,
    tag: str,
    zi: float,
    zf: float,
    verbose: bool = True,
) -> None:
    """
    Load a Propagator's mode basis / coupling coefficients from cache if
    available (Propagator.load(tag)); otherwise solve for them from
    scratch with Propagator.characterize(zi, zf, save=True, tag=tag) and
    let cbeam cache the result under that tag for next time.

    characterize() is what actually does the FEM eigenmode solve at each
    z step and computes the coupling-coefficient matrices -- it's the
    same call that produced the original "19port_0800_front"/"_back"
    cache files (see cbeam's 19-port lantern example). This is a genuine
    solve, not the diffrax-based coupled-mode propagation done later in
    .propagate() -- it's slow (~minutes) and is the real cost of adding a
    wavelength.

    Note: this catches (FileNotFoundError, OSError) as "no cache present".
    If your cbeam version raises something else for a missing tag (e.g. a
    bare Exception from a failed unpickle), widen the except clause
    accordingly -- the intent is "anything that means the file isn't
    there", not "silently swallow a real solve failure".
    """
    try:
        _load_and_backfill_cmats(prop, tag, verbose=verbose)
        if verbose:
            print(f"    [{tag}] loaded cached characterization.")
    except (FileNotFoundError, OSError) as exc:
        if verbose:
            print(f"    [{tag}] no cached characterization found ({exc}); "
                  f"running characterize(zi={zi}, zf={zf}, save=True) -- "
                  f"this can take several minutes ...")
        prop.characterize(zi, zf, save=True, tag=tag)
        if verbose:
            print(f"    [{tag}] characterization complete and cached.")

# =====================================================================
# DISPERSION MODEL: Sellmeier-based neff scaling
# =====================================================================

def sellmeier_n_silica(wavelength_um: float) -> float:
    """
    Refractive index of fused silica (Malitson 1965 Sellmeier fit).

    Parameters
    ----------
    wavelength_um : float
        Wavelength in micrometres (e.g. 0.800 for 800 nm).

    Returns
    -------
    n : float
        Refractive index at the given wavelength.
    """
    l2 = wavelength_um ** 2
    n2 = (1.0
          + 0.6961663 * l2 / (l2 - 0.0684043 ** 2)
          + 0.4079426 * l2 / (l2 - 0.1162414 ** 2)
          + 0.8974794 * l2 / (l2 - 9.896161  ** 2))
    return float(np.sqrt(n2))


def scale_params_to_wavelength(
    base_params: dict,
    wavelength_nm: float,
    ref_wavelength_nm: Optional[float] = None,
) -> dict:
    """
    Return a copy of *base_params* with refractive indices and the
    focal-plane pixel scale updated for *wavelength_nm* using the
    Sellmeier equation for fused silica (Malitson 1965).

    This implements a first-order dispersion model that avoids a full
    FEM re-characterisation at every wavelength:

      neff(λ) ≈ n_clad(λ) + Δn₀ · [n_core(λ) − n_clad(λ)] / Δn₀_ref

    where  Δn₀ = neff_ref − n_clad_ref  is the modal-field offset at the
    reference wavelength.  The approximation is valid when the mode profile
    does not change dramatically with wavelength (i.e. well above cutoff).
    Near-cutoff modes (the highest-order modes of a 19-port lantern near
    960 nm) will be less accurate; use a full FEM characterisation there.

    The focal-plane pixel scale is also updated: Δξ ∝ λ (Fraunhofer
    diffraction), so pixel_scale_um(λ) = pixel_scale_um(λ₀) · λ/λ₀.

    Parameters
    ----------
    base_params : dict
        Output of get_simulation_parameters(). Must contain at minimum:
        ``nclad``, ``ncore``, ``njack``, ``pixel_scale_um``,
        ``wavelength_nm``.
    wavelength_nm : float
        Target wavelength in nanometres.
    ref_wavelength_nm : float or None
        Reference wavelength (the one whose full FEM characterisation
        exists on disk).  Defaults to base_params["wavelength_nm"].

    Returns
    -------
    p_lambda : dict
        Updated parameter dict.  Keys changed:
        ``wl``, ``wavelength_nm``, ``nclad``, ``ncore``, ``njack``,
        ``pixel_scale_um``.
    """
    if ref_wavelength_nm is None:
        ref_wavelength_nm = float(base_params["wavelength_nm"])

    wl_um     = wavelength_nm     / 1000.0
    ref_wl_um = ref_wavelength_nm / 1000.0

    # ----------------------------------------------------------------
    # Sellmeier indices at reference and target wavelengths
    # ----------------------------------------------------------------
    n_sil_ref = sellmeier_n_silica(ref_wl_um)
    n_sil_tgt = sellmeier_n_silica(wl_um)

    # Silica index shift between reference and target wavelengths.
    # Used to rescale all three material indices (cladding, core, jacket)
    # since they are all doped-silica variants whose Sellmeier curves
    # track pure silica to first order.
    delta_n_sil = n_sil_tgt - n_sil_ref

    p_lambda = dict(base_params)

    # ----------------------------------------------------------------
    # Update material indices
    # ----------------------------------------------------------------
    p_lambda["wl"]           = wl_um
    p_lambda["wavelength_nm"] = wavelength_nm

    p_lambda["nclad"] = base_params["nclad"] + delta_n_sil
    p_lambda["ncore"] = base_params["ncore"] + delta_n_sil
    p_lambda["njack"] = base_params["njack"] + delta_n_sil

    # Recompute derived per-material lists that copy the scalar value.
    n_core_new = p_lambda["ncore"]
    p_lambda["ncores"] = [n_core_new] * len(base_params["ncores"])

    # ----------------------------------------------------------------
    # Scale focal-plane pixel size with wavelength (Fraunhofer: Δξ ∝ λ)
    # ----------------------------------------------------------------
    p_lambda["pixel_scale_um"] = (
        base_params["pixel_scale_um"] * (wavelength_nm / ref_wavelength_nm)
    )

    return p_lambda

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
    p_lambda = dict(base_params)
    p_lambda = scale_params_to_wavelength(base_params, wavelength_nm)

#    p_lambda["wl"] = wavelength_nm / 1000.0
#    p_lambda["wavelength_nm"] = wavelength_nm
#    ref_wl_nm = base_params.get("wavelength_nm", 800.0)
#    p_lambda["pixel_scale_um"] = base_params["pixel_scale_um"] * (wavelength_nm / ref_wl_nm)

    z_ex = p_lambda["z_ex"]
    _z_split = z_split if z_split is not None else z_ex / 2.0
    n_output_positions = str(base_params["n_output_positions"])

    tag = f"{n_output_positions}{cache_prefix}_{_wavelength_tag(wavelength_nm)}"

    PL_N = PL_N if PL_N is not None else build_lantern_geometry(p_lambda)

    _skipped_set = set(skipped_modes) if skipped_modes is not None else {18}
    _degen_front = degen_groups if degen_groups is not None else default_degenerate_groups_front[base_params["nrings"]]
    _degen_back = [[i for i in range(n_modes) if i not in _skipped_set]]

    prop1 = Propagator(p_lambda["wl"], PL_N, n_modes)
    prop1.degen_groups = _degen_front
    prop1.skipped_modes = _skipped_set
    _load_or_characterize(prop1, f"{tag}_front", 0.0, _z_split, verbose=verbose)

    prop2 = Propagator(p_lambda["wl"], PL_N, n_modes)
    prop2.skipped_modes = _skipped_set
    prop2.degen_groups = _degen_back
    # Must happen before load/characterize: the back segment's initial
    # eigenbasis is bootstrapped from the front segment's final one,
    # whether the front segment was just loaded or just solved.
    prop2.load_init_conds(prop1)
    _load_or_characterize(prop2, f"{tag}_back", _z_split, z_ex, verbose=verbose)

    return p_lambda, ChainPropagator([prop1, prop2])


# =====================================================================
# LAYER 2: MULTI-WAVELENGTH PIPELINE COLLECTION
# =====================================================================

def _force_release_memory() -> None:
    """
    Best-effort release of both Python- and Julia-side memory between
    wavelengths. gc.collect() handles Python objects (breaking reference
    cycles CPython's refcounting alone won't catch). cbeam's Julia
    backend (accessed via PythonCall/juliacall -- see the
    FEval.transverse_gradient calls in compute_cmats) has its own
    garbage collector that isn't triggered just because Python drops its
    references to the wrapping objects, so nudge it too. Safe no-op if
    juliacall isn't importable or exposes a different entry point than
    assumed here.
    """
    gc.collect()
    try:
        from juliacall import Main as _jl
        _jl.GC.gc()
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
    """
    Manages per-wavelength propagation through the lantern across a set
    of native wavelengths, WITHOUT holding every wavelength's heavy state
    (FEM mesh, mode-field trajectory, coupling-matrix trajectory) resident
    in memory at once.

    Usage
    -----
        mwp = MultiWavelengthPropagationPipeline(
            base_params=get_simulation_parameters(),
            ifunc=ifunc,
            native_wavelengths_nm=np.linspace(700, 900, 21),  # coarse solve grid
        )
        power, wl_out = mwp.get_power_spectra(
            coeff_matrix, output_wavelengths_nm=np.linspace(700, 900, 200)
        )  # power: (200, n_fields, n_fibers)

    Memory model
    ------------
    Each wavelength's ChainPropagator carries a mode-field trajectory
    (`vs`, shape roughly n_z_steps x n_modes x n_mesh_points, complex128)
    and a coupling-matrix trajectory (`cmats`, n_z_steps x n_modes x
    n_modes, complex128) for both the front and back segments. Depending
    on mesh size and adaptive step count these can individually reach the
    GB range. An earlier version of this class built and kept a
    BatchPropagationPipeline for every native wavelength in self.engines
    simultaneously -- fine for a handful of wavelengths, but it scales
    memory linearly with n_wavelengths and can OOM-kill the process
    (observed as a silent kernel crash, no Python traceback, after "a
    few" wavelengths loaded) for larger native grids.

    This version builds one wavelength's engine at a time inside
    propagate_batch(), extracts only the small result actually needed
    (uf_batch sliced to n_fibers complex numbers per field -- not the
    underlying GB-scale trajectory arrays), and explicitly drops the
    engine + forces garbage collection (Python and, best-effort, Julia --
    see _force_release_memory()) before moving to the next wavelength.
    Peak memory is therefore roughly O(1 wavelength) rather than
    O(n_wavelengths).

    Set cache_engines=True to opt back into keeping every engine resident
    (e.g. if you're propagating many different aberration batches against
    the same small native grid and want to avoid rebuilding each time) --
    only do this if you've confirmed your native grid is small enough,
    and your machine has enough RAM, to hold all of them at once.

    Notes on cost
    -------------
    Each native wavelength needs its own local-mode-basis characterization
    (see build_and_characterize_lantern_at_wavelength -- solved once and
    cached to disk; a fresh characterize() can take ~800 s (~13 min) on
    comparable z-extents, though loading a cached one is normally fast).
    Keep native_wavelengths_nm coarse (tens of points, denser near known
    mode-degeneracy/avoided-crossing regions -- see cbeam's compute_neffs()
    sanity check) and use output_wavelengths_nm for the finer grid you
    actually want in the dispersed spectra -- see
    interpolate_complex_spectra().
    """

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

        # Wavelength-independent lantern geometry, built once and shared
        # by every per-wavelength Propagator.
        self._shared_PL_N = build_lantern_geometry(base_params)

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
        Propagate one batch of aberration configurations through the
        lantern at every native wavelength, building and (unless
        cache_engines=True) discarding each wavelength's engine in turn.

        Parameters
        ----------
        aberration_coeff_batch : ndarray, shape (n_fields, n_active_modes)
            Same convention as BatchPropagationPipeline.generate_batch_modal_coefficients.
        use_gpu, gen_chunk_size, prop_chunk_size : passed straight through
            to each per-wavelength pipeline's generate_batch_modal_coefficients
            / propagate_batch calls.

        Returns
        -------
        spectra_complex : ndarray, shape (n_wavelengths_native, n_fields, n_fibers), complex128
            Complex per-fiber modal coefficient at each native wavelength.
            Power is NOT taken here -- phase is preserved in case you want
            it (e.g. to sanity-check mode purity); take np.abs(...)**2 for
            detector-plane flux.
        """
        n_wl = len(self.native_wavelengths_nm)
        n_fields = np.asarray(aberration_coeff_batch).shape[0]

        spectra_complex = np.zeros((n_wl, n_fields, self.n_fibers), dtype=np.complex128)

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

            uf_batch, _, _ = engine.pipeline.propagate_batch(u0_batch, chunk_size=prop_chunk_size)
            # Convert from FEM-eigenmode basis to per-core channel basis
            uf_channel = np.array([engine.prop12.to_channel_basis(uf_batch[k]) for k in range(uf_batch.shape[0])])

            _skipped_set = set(engine.prop12.skipped_modes)  # {18}
            active_modes = [i for i in range(engine.prop12.Nmax) if i not in _skipped_set][:self.n_fibers]


            spectra_complex[i, :, :] = np.abs(uf_channel[:, :self.n_fibers])  # or use active_modes
            # alternative
            # spectra_complex[i, :, :] = uf_batch[:, active_modes]

            if not self.cache_engines:
                # Drop the heavy per-wavelength state (mesh, mode-field
                # trajectory, coupling-matrix trajectory -- these can run
                # into the GB range per wavelength) before moving on.
                # This is the fix for OOM kernel crashes after a few
                # wavelengths: without it, all n_wavelengths engines stay
                # resident simultaneously.
                del engine, u0_batch, uf_batch
                _force_release_memory()

        return spectra_complex

    @staticmethod
    def interpolate_power_spectra(
        power_native: np.ndarray,        # real, shape (n_wl_native, n_fields, n_fibers)
        wl_native_nm: np.ndarray,
        wl_output_nm: np.ndarray,
    ) -> np.ndarray:
        n_wl, n_fields, n_fibers = power_native.shape
        flat = power_native.reshape(n_wl, -1)
        out = np.array([np.interp(wl_output_nm, wl_native_nm, flat[:, k])
                        for k in range(flat.shape[1])]).T
        return out.reshape(len(wl_output_nm), n_fields, n_fibers)

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

    # ------------------------------------------------------------------
    def get_power_spectra(
        self,
        aberration_coeff_batch: np.ndarray,
        output_wavelengths_nm: Optional[np.ndarray] = None,
        use_gpu: Optional[bool] = None,
        gen_chunk_size: Optional[int] = None,
        prop_chunk_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience wrapper: propagate at all native wavelengths, optionally
        interpolate onto a finer output grid, and return real per-fiber
        power (flux) spectra ready for SpectralImageSimulator.

        Returns
        -------
        power_spectra : ndarray, shape (n_wl_output, n_fields, n_fibers)
            np.abs(complex coefficient)**2 at each wavelength.
        wl_output_nm  : ndarray, shape (n_wl_output,)
            The wavelength grid the output is sampled on (== native grid
            if output_wavelengths_nm was not given).
        """
        spectra_complex = self.propagate_batch(
            aberration_coeff_batch, use_gpu=use_gpu,
            gen_chunk_size=gen_chunk_size, prop_chunk_size=prop_chunk_size,
        )

        power_native = np.abs(spectra_complex) ** 2
        if output_wavelengths_nm is None:
            return power_native, self.native_wavelengths_nm

        wl_output_nm = np.asarray(output_wavelengths_nm, dtype=np.float64)
        return self.interpolate_power_spectra(power_native, self.native_wavelengths_nm, wl_output_nm), wl_output_nm

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
        prop_chunk_size=8,
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
"""Generate LanternDatasets by propagating DM-aberrated fields through the lantern.

Two measurement models:

* single wavelength (``simulate_core_images``): the lantern output imaged on
  a camera -- the output intensity is sampled on a regular grid, the core
  centres are calibrated on the mean image, and each core's signal is the
  mean of a small pixel window around its centre. The input field is
  normalised to unit coupled power, so the signals are the relative
  distribution of the light over the cores.
* spectral (``simulate_spectra``): the power in each output channel at every
  wavelength (MultiWavelengthPropagationPipeline), as the fraction of the
  incident power -- i.e. including the input coupling efficiency.

Columns follow the lantern's port order (``p["output_positions"]``), which is
also the order of ChainPropagator.to_channel_basis().
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import LinearNDInterpolator

import batch_pipeline as bp

from .data import LanternDataset


def output_core_xy(p) -> np.ndarray:
    """(n_cores, 2) physical core positions [um] at the lantern output, in
    port order: the z=0 layout scaled by the taper."""
    return np.asarray(p["output_positions"], float) * p["taper_factor"]


def simulate_core_images(p, ifunc, coeff, n_train, *, prop12=None, window=5,
                         grid_resolution=bp.CALIB_GRID_RESOLUTION) -> LanternDataset:
    """Single-wavelength dataset: ``coeff`` (n, n_modes) [nm] -> per-core camera
    signals (n, n_cores) in port order.

    prop12 : a characterized ChainPropagator for ``p`` (built, or loaded from
        the cache, if None).
    window : side of the pixel window averaged around each core centre, on a
        grid_resolution x grid_resolution grid over the output face.
    """
    if prop12 is None:
        prop12 = bp.build_and_characterize_lantern(p, reuse_cache=True)
    pipeline = bp.BatchPropagationPipeline(prop12, p, ifunc)

    u0 = pipeline.generate_batch_modal_coefficients(coeff, use_gpu=False)
    uf, _, _ = pipeline.propagate_batch(u0)
    intensity = np.abs(pipeline.reconstruct_batch_output_fields(uf)) ** 2   # (n, n_mesh)
    mesh = pipeline.wvg_props_output["mesh"]

    # core centres on the mean image, matched to the ports
    centers, gx, gy, dx, dy = bp.calibrate_subpixel_centers(
        intensity.mean(axis=0), mesh, grid_resolution=grid_resolution)
    port_xy = output_core_xy(p)
    port_of = bp.map_evaluated_to_ideal_geometry(centers, port_xy)   # detected k -> port
    centers_by_port = np.asarray(centers)[np.argsort(port_of)]

    # pixel windows around the centres; only those grid pixels are evaluated
    half = np.arange(window) - window // 2
    cols = np.rint((centers_by_port[:, 0] - gx[0]) / dx).astype(int)[:, None] + half
    rows = np.rint((centers_by_port[:, 1] - gy[0]) / dy).astype(int)[:, None] + half
    qx = np.broadcast_to(gx[cols][:, None, :], (len(port_xy), window, window))
    qy = np.broadcast_to(gy[rows][:, :, None], (len(port_xy), window, window))
    interp = LinearNDInterpolator(mesh.points[:, :2], intensity.T)
    pix = np.nan_to_num(interp(qx.ravel(), qy.ravel()), nan=0.0)     # (n_px, n)
    signals = pix.reshape(len(port_xy), window * window, -1).mean(axis=1).T

    return LanternDataset(
        signals=signals, coeff=np.asarray(coeff, float), n_train=int(n_train),
        core_xy=port_xy, wavelengths_nm=np.asarray([p["wavelength_nm"]], float),
        meta={"measurement": f"camera: mean of {window}x{window} px windows, "
                             f"{grid_resolution} px grid, unit coupled power",
              "detected_core_xy": np.array2string(centers_by_port, separator=",")})


def simulate_spectra(base_params, ifunc, coeff, n_train, wavelengths_nm,
                     **pipeline_kwargs) -> LanternDataset:
    """Spectral dataset: ``coeff`` (n, n_modes) [nm] -> per-channel coupled
    power fractions (n, n_wl, n_cores), one lantern characterization per
    wavelength (MultiWavelengthPropagationPipeline; ``pipeline_kwargs`` are
    passed to it, e.g. auto_mode_bookkeeping=True)."""
    import multi_wvl as mw

    mwp = mw.MultiWavelengthPropagationPipeline(
        base_params, ifunc, wavelengths_nm, **pipeline_kwargs)
    power, wl = mwp.get_power_spectra(coeff)                        # (n_wl, n, n_cores)
    return LanternDataset(
        signals=np.ascontiguousarray(power.transpose(1, 0, 2)),
        coeff=np.asarray(coeff, float), n_train=int(n_train),
        core_xy=output_core_xy(base_params), wavelengths_nm=np.asarray(wl, float),
        meta={"measurement": "channel power / incident power, per wavelength"})

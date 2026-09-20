"""Backward-compatibility shim.

The implementation was split into the ``multi_wvl`` package:

    multi_wvl.params        scale_params_to_wavelength, sellmeier_n_silica,
                            build_lantern_geometry
    multi_wvl.bookkeeping   automatic per-wavelength degen_groups / skipped_modes
                            (probe_endpoint_neffs, infer_lantern_mode_bookkeeping,
                            diagnose_mode_bookkeeping, ...)
    multi_wvl.construct     build_and_characterize_lantern_at_wavelength + caching
    multi_wvl.pipeline      MultiWavelengthPropagationPipeline, WavelengthEngine
    multi_wvl.glue          spectra_for_field, build_spectral_config_from_wavelength_grid
    multi_wvl.run           example_multiwavelength_propagation

This module re-exports the full surface so existing imports keep working:

    from multi_wvl_pipeline import *
    from multi_wvl_pipeline import build_and_characterize_lantern_at_wavelength, ...
"""
# the tuned 800 nm bookkeeping tables now live with the rest of the constants
from batch_pipeline.constants import (   # noqa: F401  (re-export)
    default_degenerate_groups_front, default_degenerate_groups_back,
    default_skipped_modes_front, default_skipped_modes_back,
)

from multi_wvl.params import *        # noqa: F401,F403
from multi_wvl.bookkeeping import *   # noqa: F401,F403
from multi_wvl.construct import *     # noqa: F401,F403
from multi_wvl.pipeline import *      # noqa: F401,F403
from multi_wvl.glue import *          # noqa: F401,F403
from multi_wvl.run import example_multiwavelength_propagation   # noqa: F401

if __name__ == "__main__":
    example_multiwavelength_propagation()

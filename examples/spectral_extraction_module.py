"""Backward-compatibility shim.

The implementation was split into the ``spectral`` package:

    spectral.config       SpectralConfig
    spectral.dispersion   DispersionModel  (shared wavelength <-> pixel mapping)
    spectral.simulate     SpectralImageSimulator                 (no matplotlib)
    spectral.extract      SpectralExtractor                      (no matplotlib)
    spectral.integrate    add_spectral_extraction_to_pipeline
    spectral.plots        visualize_spectral_image / _extracted_spectra / _all_spectra_grid
    spectral.run          example_spectral_extraction

This module re-exports the full surface so existing imports keep working:

    from spectral_extraction_module import SpectralImageSimulator, SpectralExtractor, ...
"""
from spectral.config import SpectralConfig                       # noqa: F401
from spectral.dispersion import DispersionModel                  # noqa: F401
from spectral.simulate import SpectralImageSimulator             # noqa: F401
from spectral.extract import SpectralExtractor                   # noqa: F401
from spectral.integrate import add_spectral_extraction_to_pipeline  # noqa: F401
from spectral.plots import (                                     # noqa: F401
    visualize_spectral_image,
    visualize_extracted_spectra,
    visualize_all_spectra_grid,
)
from spectral.run import example_spectral_extraction             # noqa: F401


if __name__ == "__main__":
    image, extracted, truth = example_spectral_extraction()
    print("\nSpectral extraction module ready for integration.")
    print("To use with your pipeline:")
    print("  from spectral_extraction_module import add_spectral_extraction_to_pipeline")
    print("  pipeline = add_spectral_extraction_to_pipeline(pipeline, config)")

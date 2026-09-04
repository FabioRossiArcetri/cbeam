# Split out of the former single-file spectral_extraction_module.py.
"""Attach generate_spectral_image / extract_fiber_spectra onto a pipeline instance."""
from __future__ import annotations
import numpy as np

from .config import SpectralConfig
from .simulate import SpectralImageSimulator
from .extract import SpectralExtractor


def add_spectral_extraction_to_pipeline(pipeline, config: SpectralConfig):
    """
    Bolt a spectral simulator + extractor onto an existing pipeline object.

    Adds ``pipeline.generate_spectral_image(uf_batch, ...)`` and
    ``pipeline.extract_fiber_spectra(images, fiber_positions, ...)`` plus the
    ``spectral_config`` / ``spectral_simulator`` / ``spectral_extractor``
    attributes.
    """
    simulator = SpectralImageSimulator(config)
    extractor = SpectralExtractor(config)

    def generate_spectral_image(uf_batch, add_noise=True, exposure_time=1.0):
        """2D spectral image per field from modal coefficients (first 19 modes
        taken as the 19 fibers)."""
        modal_powers = np.abs(uf_batch) ** 2          # (n_fields, n_modes)
        n_fields = modal_powers.shape[0]
        n_fibers = min(19, modal_powers.shape[1])

        images, fiber_positions_list = [], []
        for i in range(n_fields):
            image, positions = simulator.create_spectral_image(
                modal_powers[i, :n_fibers],
                add_noise=add_noise,
                exposure_time=exposure_time,
            )
            images.append(image)
            fiber_positions_list.append(positions)

        return np.array(images), fiber_positions_list[0]

    def extract_fiber_spectra(spectral_images, fiber_positions, method="optimal"):
        """1D spectra + variance per field from the 2D spectral images."""
        n_fields = len(spectral_images)
        n_fibers = len(fiber_positions)
        n_wavelengths = config.n_spectral_bins

        all_spectra = np.zeros((n_fields, n_wavelengths, n_fibers))
        all_variance = np.zeros((n_fields, n_wavelengths, n_fibers))
        for i in range(n_fields):
            spectra, variance = extractor.extract_spectra(
                spectral_images[i], fiber_positions, method=method)
            all_spectra[i] = spectra
            all_variance[i] = variance

        return all_spectra, all_variance

    pipeline.generate_spectral_image = generate_spectral_image
    pipeline.extract_fiber_spectra = extract_fiber_spectra
    pipeline.spectral_config = config
    pipeline.spectral_simulator = simulator
    pipeline.spectral_extractor = extractor
    return pipeline

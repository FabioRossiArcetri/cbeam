# Examples

Run these from inside `examples/` (they import `batch_pipeline` from here and
write their characterization cache to `examples/data/`, which is git-ignored).

## Batched propagation

`batch_pipeline/` propagates whole batches of input fields through a
characterized 19-port photonic lantern in a single `propagate()` call, on the
numpy or the jax backend (`CBEAM_BACKEND`).

| | |
|---|---|
| `Batched.ipynb` | timing of batched vs. one-field-at-a-time `propagate()`, numpy and jax |
| `BatchedPiston.ipynb` | the full pipeline: DM-aberrated input pupils -> batched propagation -> output fields -> per-core signals |
| `python -m batch_pipeline.run` | the same pipeline as a script |
| `ifunc/` | the DM influence function the pipeline uses (see its README) |

Besides cbeam's own requirements:

- **jax + diffrax** for the jax backend (`pip install -e ".[jax]"`);
- **astropy** to read the influence-function FITS file;
- **specula** is optional. When it is installed the influence function is
  loaded through it, otherwise through plain numpy; the results are
  bit-identical either way.

The first run characterizes the lantern (several minutes per wavelength);
later runs load the cached characterization from `examples/data/`.

## Multi-wavelength propagation and spectrograph

`multi_wvl/` propagates a batch of aberrated fields at every wavelength of a
native grid -- one lantern characterization per wavelength, built and released
one at a time -- and returns each output fiber's spectrum as the fraction of
the incident power reaching it. `spectral/` disperses those spectra onto a
simulated spectrograph detector and extracts them back (optimal extraction).

| | |
|---|---|
| `multi_wavelength.ipynb` | 41 wavelengths (700-900 nm) x a petal-piston ramp: output spectra, input coupling vs. wavelength, detector stage |
| `multi_wavelength_vis.py` | the notebook's plotting helpers |
| `python -m multi_wvl.run` | a shorter end-to-end run as a script |

Each wavelength needs its own characterization: minutes per wavelength the
first time, cached in `examples/data/` afterwards (41 wavelengths take a few
hours to characterize).

## Wavefront reconstruction (machine learning)

`ml_reconstruct/` reconstructs DM-mode coefficients (here the petal pistons)
from simulated lantern signals: dataset generation, a JAX MLP, sparse and
dense polynomial fits, Zernike features and a multi-wavelength phase unwrap.

| | |
|---|---|
| `trainMLP.ipynb` | one wavelength, pistons within +-lambda/2: MLP vs. linear, k-NN, polynomial (dense / OMP), Zernike features, optional symbolic regression |
| `trainMLP_multiwavelength.ipynb` | +-5 um pistons from the 41-wavelength spectral response: single models on the whole spectrum |
| `PL19_EML.ipynb` | the same data: one model per wavelength on the wrapped piston, fused by a multi-wavelength unwrap |

The datasets and fitted models are written to `examples/data/ml/`. Generating
the single-wavelength dataset takes ~15-40 minutes; the spectral one (50000
samples x 41 wavelengths) several hours, best on a GPU with the jax backend.

These notebooks also need **jax + optax** and **scikit-learn**; the optional
symbolic-regression cell needs **PySR** (and Julia).

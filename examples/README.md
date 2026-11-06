# Examples

Run these from inside `examples/` (they import `batch_pipeline` from here and
write their characterization cache to `examples/data/`, which is git-ignored).

## Batched propagation

`batch_pipeline/` propagates whole batches of input fields through a
characterized photonic lantern in a single `propagate()` call, on the
numpy or the jax backend (`CBEAM_BACKEND`).

**Lantern presets.** The lantern design is a named preset in
`batch_pipeline/constants.py` (`LANTERNS`, with its mode-bookkeeping tables),
chosen with the environment variable `CBEAM_LANTERN` or
`get_simulation_parameters(lantern=...)`:

| preset | ports | design wavelength / band | |
|---|---|---|---|
| `pl19_800` (default) | 19 | 800 nm / 700-900 nm | the reference design (caches `19port_XXXX`) |
| `pl7_1200` | 7 | 1200 nm / 1100-1300 nm | Ge-doped cores, F-doped capillary, multimode NA 0.170 |
| `pl19_1200` | 19 | 1200 nm / 1100-1300 nm | the fibres of `pl7_1200`, multimode end sized to the reference V |

`get_simulation_parameters(3, 0.8)` still means `pl19_800`. The PSF is sized
relative to the multimode end the same way for every preset (fill = 0.30 x
V_ref / V). `../characterize.py --lantern <preset>` builds the 41
per-wavelength characterizations of the band, and `../run_characterize.sh`
runs them in parallel within a memory budget (half the host's RAM by
default). The notebooks below run for the lantern in `CBEAM_LANTERN`;
executed copies per lantern are in `runs/<preset>/`.

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
| `multi_wavelength.ipynb` | 41 wavelengths across the lantern's band x a petal-piston ramp: output spectra, input coupling vs. wavelength, detector stage |
| `multi_wavelength_vis.py` | the notebook's plotting helpers |
| `python -m multi_wvl.run` | a shorter end-to-end run as a script |

Each wavelength needs its own characterization: minutes per wavelength the
first time, cached in `examples/data/` afterwards (`../characterize.py`).

## Wavefront reconstruction (machine learning)

`ml_reconstruct/` reconstructs DM-mode coefficients (here the petal pistons)
from simulated lantern signals: dataset generation, a JAX MLP, sparse and
dense polynomial fits, Zernike features and a multi-wavelength phase unwrap.

| | |
|---|---|
| `trainMLP.ipynb` | one wavelength, pistons within +-lambda/2: MLP vs. linear, k-NN, polynomial (dense / OMP), Zernike features, optional symbolic regression |
| `trainMLP_multiwavelength.ipynb` | +-8 lambda pistons from the 41-wavelength spectral response: single models on the whole spectrum |
| `PL_EML.ipynb` | the same data: one model per wavelength on the wrapped piston, fused by a multi-wavelength unwrap |

The datasets and fitted models are written to `examples/data/ml/` (`pl19_800`)
or `examples/data/ml/<preset>/`; per-preset settings are in
`ml_reconstruct/presets.py`. Generating
the single-wavelength dataset takes ~15 minutes; the spectral one (50000
samples x 41 wavelengths) several hours on one core, so split it by wavelength
and chunk over many cores (see `PLREC7/gen_datasets.py`).

These notebooks also need **jax + optax** and **scikit-learn**; the optional
symbolic-regression cell needs **PySR** (and Julia).

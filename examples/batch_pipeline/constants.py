"""Backend selection, module constants and the tuned mode-bookkeeping tables."""
from __future__ import annotations
import os
from cbeam.backend import get_backend, get_jax_device
from cbeam.waveguide import hex_ring_positions

backend = get_backend()
default_chunk_size = 10000
default_gen_chunk_size = 64

# -- optional JAX imports ------------------------------------------------
if backend == 'jax':
    import jax
    import jax.numpy as jnp
    _jax_device = get_jax_device()
else:
    jax = None
    jnp = None
    _jax_device = None

L1 = 50000
L2 = 50000

DEFAULT_GRID_RESOLUTION = 400
CALIB_GRID_RESOLUTION   = 400    # grid the mode profile is resampled onto for core-centre calibration
DEFAULT_SUBPIXEL_N      = 5      # side of the NxN sub-pixel sampling window

# --------------------------------------------------------------------------- #
# Lantern presets
# --------------------------------------------------------------------------- #
# Every lantern design the examples know, by name.  get_simulation_parameters()
# builds the full parameter dict from one of these; the mode-bookkeeping tables
# below are keyed by the same names.  Radii are at the waist (multimode end,
# z = 0) in um; the single-mode end is taper_factor times larger.
#
#   pl19_800   the original 19-port lantern at 800 nm (the reference design;
#              its characterization caches are tagged "19port_XXXX").
#   pl7_1200   7 ports (1 + 6 hexagon) at 1200 nm: Ge-doped SMF-like cores
#              (+5e-3), F-doped capillary (-1e-2), multimode-end NA 0.170;
#              rclad 6.4 um keeps the LP31 pair guided up to 1300 nm.
#   pl19_1200  19 ports at 1200 nm in the same fibre technology as pl7_1200,
#              with rclad sized to the reference V = 8.90 (only the port
#              count differs from pl7_1200).
#
# ring_div: the outer core ring sits at rclad * (nrings - 1) / ring_div, i.e.
# the core pitch is rclad / ring_div -- half a pitch inside the cladding edge.
LANTERNS = {
    "pl19_800": dict(
        nrings=3, wavelength_um=0.8, band_nm=(700.0, 900.0), cache_tag="19port",
        taper_factor=12., rclad=9.0, rjack=27, rcore=1.8 / 12., ring_div=2.5,
        nclad=1.444, dn_core=8.8e-3, dn_jack=-5.5e-3, auto_bookkeeping=False),
    "pl7_1200": dict(
        nrings=2, wavelength_um=1.2, band_nm=(1100.0, 1300.0), cache_tag="pl7_1200",
        taper_factor=15., rclad=6.4, rjack=19.2, rcore=0.22, ring_div=1.5,
        nclad=1.4481, dn_core=5.0e-3, dn_jack=-1.0e-2, auto_bookkeeping=False),
    "pl19_1200": dict(
        nrings=3, wavelength_um=1.2, band_nm=(1100.0, 1300.0), cache_tag="pl19_1200",
        taper_factor=15., rclad=10.0, rjack=30.0, rcore=0.22, ring_div=2.5,
        nclad=1.4481, dn_core=5.0e-3, dn_jack=-1.0e-2,
        # near 1300 nm the weakest modes merge at the jacket cutoff: derive
        # the bookkeeping per wavelength when characterizing the band
        auto_bookkeeping=True),
}

# The lantern used when none is named: CBEAM_LANTERN, else the reference.
DEFAULT_LANTERN = os.environ.get("CBEAM_LANTERN", "pl19_800")
if DEFAULT_LANTERN not in LANTERNS:
    raise ValueError(f"CBEAM_LANTERN={DEFAULT_LANTERN!r}: known lanterns are {sorted(LANTERNS)}")

DEFAULT_WAVELENGTH_UM = LANTERNS[DEFAULT_LANTERN]["wavelength_um"]
N_RINGS   = LANTERNS[DEFAULT_LANTERN]["nrings"]
N_SIGNALS = len(hex_ring_positions(N_RINGS, 1.0))   # output cores of the default lantern

# --------------------------------------------------------------------------- #
# Mode bookkeeping, per lantern (at its design wavelength)
# --------------------------------------------------------------------------- #
# Tracked modes = output cores + 1. Front (multimode end): degenerate groups
# in n_eff order; back (single-mode end): the uncoupled core modes form one
# degenerate block of all kept modes.
default_degenerate_groups_front = {}
default_degenerate_groups_back = {}
default_skipped_modes_front = {}
default_skipped_modes_back = {}

# pl19_800: the tuned tables of the original pipeline.
default_degenerate_groups_front["pl19_800"] = [[1,2],[3,4],[6,7],[8,9],[10,11],[12,13],[15,16]]
default_degenerate_groups_back["pl19_800"] = [[i for i in range(20) if i != 18]]
default_skipped_modes_front["pl19_800"] = {18}
default_skipped_modes_back["pl19_800"] = {18}

# pl7_1200: LP01 | LP11 pair | LP21 pair | LP02 | LP31 split in two by the
# hexagon (not degenerate); the lower LP31 partner (mode 7) is skipped.
# Checked with multi_wvl.infer_lantern_mode_bookkeeping over 1100-1300 nm.
default_degenerate_groups_front["pl7_1200"] = [[1, 2], [3, 4]]
default_degenerate_groups_back["pl7_1200"]  = [[i for i in range(8) if i != 7]]
default_skipped_modes_front["pl7_1200"]     = {7}
default_skipped_modes_back["pl7_1200"]      = {7}

# pl19_1200: infer_lantern_mode_bookkeeping at 1100 and 1200 nm gives the
# pl19_800 front pairs with mode 19 skipped; near 1300 nm modes 17-19 merge,
# so the band is characterized with auto_bookkeeping (see LANTERNS).
default_degenerate_groups_front["pl19_1200"] = [[1,2],[3,4],[6,7],[8,9],[10,11],[12,13],[15,16]]
default_degenerate_groups_back["pl19_1200"]  = [[i for i in range(19)]]
default_skipped_modes_front["pl19_1200"]     = {19}
default_skipped_modes_back["pl19_1200"]      = {19}

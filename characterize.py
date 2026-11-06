#!/usr/bin/env python
"""Characterize a lantern preset at its 41 native wavelengths (band in 5 nm steps).

    python characterize.py --lantern pl7_1200            # all 41, one after the other
    python characterize.py --lantern pl7_1200 0 5 40     # only these wavelength indices
    python characterize.py --lantern pl7_1200 --list     # index -> wavelength

Presets: batch_pipeline.LANTERNS (pl19_800, pl7_1200, pl19_1200); without
--lantern, CBEAM_LANTERN or pl19_800. Every wavelength is independent, so on
a many-core host run one process per index, e.g.

    seq 0 40 | xargs -P 20 -I{} python characterize.py --lantern pl19_1200 {}

with OMP_NUM_THREADS=1, and as many processes as the memory budget allows
(see run_characterize.sh). Caches land in ./examples/data/, tagged by
batch_pipeline.characterization_tag (e.g. pl7_1200_1150_front/back); a
wavelength that is already cached is skipped. The mode bookkeeping is the
preset's table, or inferred per wavelength for presets with auto_bookkeeping.
"""
import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))       # the cbeam checkout
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "examples"))
os.chdir(os.path.join(ROOT, "examples"))                 # caches go to ./data

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--lantern", default=None)
ap.add_argument("--list", action="store_true")
ap.add_argument("indices", nargs="*", type=int)
args = ap.parse_args()

from batch_pipeline import get_simulation_parameters, characterization_tag  # noqa: E402
from multi_wvl import build_lantern_geometry, build_and_characterize_lantern_at_wavelength  # noqa: E402

base = get_simulation_parameters(lantern=args.lantern)
WAVELENGTHS_NM = np.linspace(*base["band_nm"], 41)

if args.list:
    for i, wl in enumerate(WAVELENGTHS_NM):
        print(i, wl)
    raise SystemExit

# cbeam's Propagator creates these on construction, but not race-free: with
# many processes started at once they collide, so make them up front.
for sub in ("cplcoeffs", "eigenmodes", "eigenvalues", "meshes", "meshpoints", "zvals"):
    os.makedirs(os.path.join("data", sub), exist_ok=True)

idx = args.indices or range(len(WAVELENGTHS_NM))
PL_N = build_lantern_geometry(base)                      # built once, reused

t0 = time.time()
for i in idx:
    wl = WAVELENGTHS_NM[i]
    t = time.time()
    log = io.StringIO()
    with contextlib.redirect_stdout(log):                # cbeam's per-step progress
        build_and_characterize_lantern_at_wavelength(base, wl, PL_N=PL_N, verbose=True)
    # keep the bookkeeping decisions (the n_eff table when inferred) next to the caches
    os.makedirs(os.path.join("data", "bookkeeping"), exist_ok=True)
    tag = characterization_tag(base, wl)
    with open(os.path.join("data", "bookkeeping", f"{tag}.log"), "w") as f:
        f.write("\n".join(ln for ln in log.getvalue().splitlines() if "current z" not in ln))
    print(f"[{i:2d}] {tag}  {time.time()-t:5.0f} s", flush=True)
print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)
try:                                                     # peak memory, for the parallel runner
    import resource
    print(f"PEAK_GB {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20:.2f}", flush=True)
except ImportError:                                      # not on Windows
    pass

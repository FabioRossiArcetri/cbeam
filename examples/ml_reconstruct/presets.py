"""Per-lantern settings of the reconstruction notebooks.

The lantern is named by CBEAM_LANTERN (see batch_pipeline.LANTERNS; default
pl19_800). Everything a notebook would otherwise hard-code for one lantern --
the single-wavelength piston range, the polynomial budgets, the Zernike basis
size, where its datasets and models live -- comes from here, so the same
notebook runs for every lantern.
"""
from __future__ import annotations

import os

ML = {
    # piston ranges: +-lambda/2 at one wavelength, +-8 lambda for the dispersed datasets
    # the reference: 19 cores at 800 nm
    "pl19_800": dict(single_span_nm=400.0, spectral_span_nm=6400.0, n_zern=36, zern_d2=18,
                     omp_ks=[100, 200, 300, 400, 500, 600, 750, 900, 1050, 1150, 1282],
                     omp_record=(500, 750, 1282), k_table=500, ks_mix=[100, 200, 300, 400, 500],
                     omp_k_wrap=1000),
    # 7 cores at 1200 nm; 119 degree-3 monomials of 7 signals in all
    "pl7_1200": dict(single_span_nm=600.0, spectral_span_nm=9600.0, n_zern=10, zern_d2=10,
                     omp_ks=[5, 10, 20, 30, 40, 60, 80, 100, 119],
                     omp_record=(40, 119), k_table=40, ks_mix=[5, 10, 20, 30, 40],
                     omp_k_wrap=119),
    # 19 cores at 1200 nm: the 19-core budgets
    "pl19_1200": dict(single_span_nm=600.0, spectral_span_nm=9600.0, n_zern=36, zern_d2=18,
                      omp_ks=[100, 200, 300, 400, 500, 600, 750, 900, 1050, 1150, 1282],
                      omp_record=(500, 750, 1282), k_table=500, ks_mix=[100, 200, 300, 400, 500],
                      omp_k_wrap=1000),
}


def lantern() -> str:
    """The lantern the notebooks run for (CBEAM_LANTERN, default pl19_800)."""
    name = os.environ.get("CBEAM_LANTERN", "pl19_800")
    if name not in ML:
        raise ValueError(f"CBEAM_LANTERN={name!r}: known lanterns are {sorted(ML)}")
    return name


def settings(name: str | None = None) -> dict:
    """ML settings of a lantern, plus its data folder ``data_dir``: data/ml for
    the reference lantern (its original location), data/ml/<lantern> else."""
    name = name or lantern()
    s = dict(ML[name], lantern=name)
    s["data_dir"] = "data/ml" if name == "pl19_800" else f"data/ml/{name}"
    return s

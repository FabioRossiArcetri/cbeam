"""Fit-or-load cache for fitted models (pickled under a directory)."""
from __future__ import annotations

import os
import pickle
from typing import Callable


class ModelCache:
    """``cache(name, build)`` returns the object pickled as ``<directory>/<name>.pkl``
    if it exists (and ``retrain`` is False), otherwise calls ``build()``, saves
    the result and returns it.

    The name should encode everything the fitted object depends on (data,
    features, hyper-parameters): a stale cache entry is loaded silently.
    Delete the directory after regenerating the dataset.
    """

    def __init__(self, directory: str, retrain: bool = False, verbose: bool = True):
        self.directory = directory
        self.retrain = retrain
        self.verbose = verbose

    def path(self, name: str) -> str:
        safe = name.replace("/", "_").replace(" ", "_")
        return os.path.join(self.directory, safe + ".pkl")

    def __call__(self, name: str, build: Callable):
        path = self.path(name)
        if not self.retrain and os.path.exists(path):
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if self.verbose:
                print(f"[cache] loaded {name}")
            return obj
        obj = build()
        os.makedirs(self.directory, exist_ok=True)
        tmp = path + ".tmp"            # write, then rename: an interrupted save
        with open(tmp, "wb") as f:     # never leaves a truncated file behind
            pickle.dump(obj, f)
        os.replace(tmp, path)
        if self.verbose:
            print(f"[cache] fitted and saved {name}")
        return obj

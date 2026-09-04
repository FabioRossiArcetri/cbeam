"""Backward-compatibility shim.

The implementation was split into the ``batch_pipeline`` package:

    batch_pipeline.constants   backend flag, module constants, mode-bookkeeping tables
    batch_pipeline.config      get_simulation_parameters, aberration-config generators
    batch_pipeline.engine      waveguide setup, IncidentFieldGenerator, ModalProjector,
                               BatchPropagationPipeline           (no matplotlib)
    batch_pipeline.analysis    peak finding, sub-pixel sampling, Procrustes  (no matplotlib)
    batch_pipeline.plots       every visualize_/display_/diagnose_ function + batch_statistics
    batch_pipeline.run         main_batch_propagation

This module re-exports the full surface so existing imports keep working:

    from batch_propagation_pipeline import *
    from batch_propagation_pipeline import BatchPropagationPipeline, get_simulation_parameters, ...
"""
import warnings

# The pre-split single-file module suppressed these process-wide as an import
# side effect.  The batch_pipeline.* package deliberately does NOT (a library
# module shouldn't mutate global warning state), but this shim exists so that
# old ``from batch_propagation_pipeline import *`` scripts and notebooks keep
# behaving exactly as before -- so restore the suppression here.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from cbeam.waveguide import PhotonicLantern, hex_ring_positions   # noqa: F401,E402  (re-export)
from cbeam.propagator import Propagator, ChainPropagator          # noqa: F401,E402  (re-export)

from batch_pipeline.constants import *   # noqa: F401,F403,E402
from batch_pipeline.config import *      # noqa: F401,F403,E402
from batch_pipeline.engine import *      # noqa: F401,F403,E402
from batch_pipeline.analysis import *    # noqa: F401,F403,E402
from batch_pipeline.plots import *       # noqa: F401,F403,E402
from batch_pipeline.run import main_batch_propagation   # noqa: F401,E402

if __name__ == "__main__":
    main_batch_propagation()

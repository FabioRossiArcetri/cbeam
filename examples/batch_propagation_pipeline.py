# =====================================================================
# BATCH PROPAGATION PIPELINE - FIXED VERSION
# =====================================================================
# This version fixes the PSF centering issues identified in the analysis.
#
# Key fixes:
# 1. Correct FFT grid centering for even-sized arrays
# 2. Fixed axis mapping in mesh coordinate interpolation
# 3. Added validation and diagnostic improvements
# 4. Proper half-pixel offset handling
# =====================================================================

from __future__ import annotations

import os
from unittest.mock import DEFAULT
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from cbeam.backend import get_backend, get_jax_device
from cbeam.waveguide import PhotonicLantern, hex_ring_positions
from cbeam.propagator import Propagator, ChainPropagator

from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment

from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import maximum_filter, map_coordinates
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import RegularPolygon

from wavesolve.fe_solver import construct_B

backend = get_backend()
default_chunk_size = 10000
default_gen_chunk_size = 1000

# -- optional JAX imports ------------------------------------------------
if backend == 'jax':
    import jax
    import jax.numpy as jnp
    _jax_device = get_jax_device()
else:
    jnp         = None
    _jax_device = None


# =====================================================================
# LAYER 1: CONFIGURATION & CONSTANTS
# =====================================================================

L1 = 50000
L2 = 50000

DEFAULT_WAVELENGTH_UM   = 0.8
DEFAULT_WAVELENGTH_NM   = 800.0
DEFAULT_GRID_RESOLUTION = 400
DEFAULT_SUBPIXEL_N      = 5
N_SIGNALS = 19
N_RINGS = 3

default_degenerate_groups_front = {}
default_degenerate_groups_back = {}
default_skipped_modes_front = {}
default_skipped_modes_back = {}

default_degenerate_groups_front[3] = [[1,2],[3,4],[6,7],[8,9],[10,11],[12,13],[15,16]]
default_degenerate_groups_back[3] = [[i for i in range(20) if i != 18]]
default_skipped_modes_front[3] = {18}
default_skipped_modes_back[3] = {18}


def diagnose_input_psf(pipeline):
    """
    Enhanced diagnostic tool for the input PSF with fixes for centering issues.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from scipy.special import j1

    print("\n=== RUNNING INPUT PSF CENTERING & SCALE DIAGNOSTIC (FIXED VERSION) ===")

    # 1. Generate unaberrated field (zero coefficients)
    fg_np = pipeline._field_gen_np
    n_total_modes = fg_np.num_modes
    zero_coeffs = np.zeros((1, n_total_modes))

    # 2. Check mask centering BEFORE field generation
    mask = fg_np.mask_np
    mask_center_y = np.mean(np.where(mask > 0)[0])
    mask_center_x = np.mean(np.where(mask > 0)[1])
    expected_center = (mask.shape[0] - 1) / 2.0  # Correct for indexing from 0
    
    print(f"\n[MASK CENTERING CHECK]")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask centroid: ({mask_center_y:.2f}, {mask_center_x:.2f})")
    print(f"  Expected center: ({expected_center:.2f}, {expected_center:.2f})")
    print(f"  Offset: ({mask_center_y - expected_center:.3f}, {mask_center_x - expected_center:.3f}) pixels")
    
    if abs(mask_center_y - expected_center) > 0.1 or abs(mask_center_x - expected_center) > 0.1:
        print("  ⚠️  WARNING: Mask is off-center! This will cause PSF misalignment.")
        print("  Recommendation: Re-center the mask before loading or adjust in IFunc object.")

    # 3. Extract field at key stages
    Ef_input = fg_np.generate_field_profiles_batch(zero_coeffs)
    Ef_focal_grid = fg_np.apply_ef_to_lantern(Ef_input)[0]
    Ef_focal_mesh = fg_np.resample_to_mesh(Ef_focal_grid[None, ...])[0]

    intensity_grid = np.abs(Ef_focal_grid) ** 2
    intensity_mesh = np.abs(Ef_focal_mesh) ** 2
    
    # 4. Check FFT grid centering
    padded_size = fg_np.grid_size * pipeline.p["pad_factor"]
    
    # FIXED: Use proper FFT center calculation
    if padded_size % 2 == 0:
        # For even grids, FFT center is at index padded_size // 2
        fft_center_idx = padded_size // 2
    else:
        # For odd grids, FFT center is at index (padded_size - 1) // 2
        fft_center_idx = (padded_size - 1) // 2
    
    peak_idx = np.unravel_index(np.argmax(intensity_grid), intensity_grid.shape)
    offset_px = (peak_idx[0] - fft_center_idx, peak_idx[1] - fft_center_idx)
    
    print(f"\n[FFT GRID CHECK]")
    print(f"  FFT grid shape: {intensity_grid.shape}")
    print(f"  FFT center index: ({fft_center_idx}, {fft_center_idx})")
    print(f"  PSF peak index: {peak_idx}")
    print(f"  Centering offset: {offset_px} pixels")
    
    if abs(offset_px[0]) > 0.5 or abs(offset_px[1]) > 0.5:
        print(f"  ⚠️  WARNING: PSF peak is off-center by > 0.5 pixels")
        print(f"  This indicates a centering problem in the FFT or mask.")

    # 5. Centroid calculation (subpixel accuracy)
    total_intensity = intensity_grid.sum()
    y_centroid = np.sum(np.indices(intensity_grid.shape)[0] * intensity_grid) / total_intensity
    x_centroid = np.sum(np.indices(intensity_grid.shape)[1] * intensity_grid) / total_intensity
    centroid_offset = (y_centroid - fft_center_idx, x_centroid - fft_center_idx)
    
    print(f"  PSF centroid (subpixel): ({y_centroid:.3f}, {x_centroid:.3f})")
    print(f"  Centroid offset: ({centroid_offset[0]:.3f}, {centroid_offset[1]:.3f}) pixels")

    # 6. Check mesh interpolation
    mesh_pts = pipeline.wvg_props_input['mesh'].points
    x_min, x_max = mesh_pts[:, 0].min(), mesh_pts[:, 0].max()
    y_min, y_max = mesh_pts[:, 1].min(), mesh_pts[:, 1].max()
    core_pts = np.array(pipeline.wvg_props_input['points'])

    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 400),
                                 np.linspace(y_min, y_max, 400))
    mesh_grid_intensity = griddata((mesh_pts[:, 0], mesh_pts[:, 1]), intensity_mesh,
                                   (grid_x, grid_y), method='cubic', fill_value=0)

    peak_mesh_idx = np.unravel_index(np.argmax(mesh_grid_intensity), mesh_grid_intensity.shape)
    peak_mesh_x = grid_x[0, peak_mesh_idx[1]]
    peak_mesh_y = grid_y[peak_mesh_idx[0], 0]

    dist_to_cores = np.sqrt((core_pts[:, 0] - peak_mesh_x)**2 + (core_pts[:, 1] - peak_mesh_y)**2)
    min_dist = dist_to_cores.min()
    closest_core = np.argmin(dist_to_cores)
    
    print(f"\n[MESH INTERPOLATION CHECK]")
    print(f"  Mesh PSF peak: ({peak_mesh_x:.3f}, {peak_mesh_y:.3f}) μm")
    print(f"  Nearest core (#{closest_core}): distance = {min_dist:.3f} μm")
    
    if min_dist > 2.0:
        print(f"  ⚠️  WARNING: PSF peak is > 2 μm from nearest core")
        print(f"  This suggests a scaling or centering issue in mesh interpolation.")

    # 7. Power interception
    total_power = intensity_mesh.sum()
    core_mask = np.zeros_like(intensity_mesh, dtype=bool)
    for cx, cy in core_pts:
        dist = np.sqrt((mesh_pts[:, 0] - cx)**2 + (mesh_pts[:, 1] - cy)**2)
        core_mask = core_mask | (dist < 2.5)
    power_in_cores = intensity_mesh[core_mask].sum()
    frac_in_cores = power_in_cores / total_power if total_power > 0 else 0
    
    print(f"\n[COUPLING EFFICIENCY CHECK]")
    print(f"  Power in cores (r=2.5 μm): {frac_in_cores:.1%}")
    
    if frac_in_cores < 0.5:
        print(f"  ⚠️  WARNING: < 50% coupling efficiency")
        print(f"  Check beam size, alignment, and numerical aperture matching.")

    # 8. Generate plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: FFT grid with corrected center lines
    ax = axes[0, 0]
    im = ax.imshow(np.log10(intensity_grid + 1e-8), cmap='viridis',
                   extent=[-fft_center_idx, padded_size - fft_center_idx,
                           -fft_center_idx, padded_size - fft_center_idx])
    ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='Center')
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax.plot(offset_px[1], offset_px[0], 'r+', markersize=15, markeredgewidth=2, label='Peak')
    ax.set_title(f"FFT Focal Plane (log)\nOffset: {offset_px}")
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Pixels")
    ax.legend()
    plt.colorbar(im, ax=ax, label="log10 Intensity")

    # Plot 2: Mask
    ax = axes[0, 1]
    im = ax.imshow(mask, cmap='gray', origin='lower')
    ax.axvline(expected_center, color='r', linestyle='--', alpha=0.5)
    ax.axhline(expected_center, color='r', linestyle='--', alpha=0.5)
    ax.plot(mask_center_x, mask_center_y, 'rx', markersize=10, markeredgewidth=2)
    ax.set_title("Pupil Mask\nRed cross = centroid")
    plt.colorbar(im, ax=ax)

    # Plot 3: Cross-section
    ax = axes[0, 2]
    x_axis = np.arange(padded_size) - fft_center_idx
    y_peak = intensity_grid[peak_idx[0], :]
    ax.plot(x_axis, y_peak / y_peak.max(), label='Horizontal')
    x_peak = intensity_grid[:, peak_idx[1]]
    ax.plot(x_axis, x_peak / x_peak.max(), label='Vertical')
    ax.axvline(0, color='r', linestyle='--', alpha=0.3)
    ax.set_title("Normalized PSF Cross-sections")
    ax.set_xlabel("Pixels from center")
    ax.set_ylabel("Normalized intensity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: PSF on mesh
    ax = axes[1, 0]
    im = ax.imshow(mesh_grid_intensity, cmap='inferno',
                   extent=[x_min, x_max, y_min, y_max], origin='lower')
    ax.scatter(core_pts[:, 0], core_pts[:, 1], c='cyan', s=30, 
               edgecolor='k', linewidth=1.5, label='Cores')
    ax.scatter(peak_mesh_x, peak_mesh_y, c='red', s=80, marker='x',
               linewidth=2, label='PSF peak')
    ax.set_title(f"PSF on Mesh\nNearest core: {min_dist:.2f} μm")
    ax.set_xlabel("X (μm)")
    ax.set_ylabel("Y (μm)")
    ax.legend()
    plt.colorbar(im, ax=ax, label="Intensity")

    # Plot 5: Radial profile
    ax = axes[1, 1]
    r = np.sqrt((mesh_pts[:, 0] - peak_mesh_x)**2 + (mesh_pts[:, 1] - peak_mesh_y)**2)
    idx_sort = np.argsort(r)
    r_sorted = r[idx_sort]
    I_sorted = intensity_mesh[idx_sort]
    bins = np.linspace(0, r_sorted.max(), 50)
    r_bin = (bins[1:] + bins[:-1]) / 2
    I_bin = np.zeros_like(r_bin)
    for i in range(len(bins) - 1):
        mask_bin = (r_sorted >= bins[i]) & (r_sorted < bins[i+1])
        if np.any(mask_bin):
            I_bin[i] = I_sorted[mask_bin].mean()
    ax.plot(r_bin, I_bin, 'b-', linewidth=2, label='Measured')
    ax.set_title("Radial PSF Profile")
    ax.set_xlabel("Radius (μm)")
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Pupil phase
    ax = axes[1, 2]
    phase_input = np.angle(Ef_input[0])
    im = ax.imshow(phase_input, cmap='hsv', origin='lower', vmin=-np.pi, vmax=np.pi)
    ax.set_title("Pupil Phase")
    plt.colorbar(im, ax=ax, label="Phase (rad)")

    plt.tight_layout()
    plt.show()

    # Summary report
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    issues = []
    if abs(mask_center_y - expected_center) > 0.1 or abs(mask_center_x - expected_center) > 0.1:
        issues.append("⚠️  Mask is off-center")
    if abs(offset_px[0]) > 0.5 or abs(offset_px[1]) > 0.5:
        issues.append("⚠️  FFT PSF peak is off-center")
    if min_dist > 2.0:
        issues.append("⚠️  Mesh PSF far from cores")
    if frac_in_cores < 0.5:
        issues.append("⚠️  Low coupling efficiency")
    
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
        print("\nRecommendations:")
        print("  1. Check mask centering in IFunc")
        print("  2. Verify pixel_scale_um matches FFT grid spacing")
        print("  3. Check mesh interpolation coordinate mapping")
    else:
        print("✓ All checks passed - PSF appears well-centered")
    print("="*60)

    
def _jax_fft_batch(Ef_input_batch, pad_width):
    """JIT-compiled zero-pad + FFT for a batch of pupil fields (JAX only)."""
    E_padded = jnp.pad(
        Ef_input_batch,
        ((0, 0), (pad_width, pad_width), (pad_width, pad_width)),
        mode='constant',
        constant_values=0,
    )
    return jnp.fft.fftshift(
        jnp.fft.fft2(
            jnp.fft.ifftshift(E_padded, axes=(1, 2)),
            axes=(1, 2),
        ),
        axes=(1, 2),
    )

def batch_collect_subpixel_signals(images: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
    rows = coords[0]  # Shape: (N_SIGNALS, 5, 5)
    cols = coords[1]  # Shape: (N_SIGNALS, 5, 5)
    
    def extract_single_core_patch(img, r_coords, c_coords):
        c_pack = jnp.stack([r_coords, c_coords], axis=0)
        patch = jax.scipy.ndimage.map_coordinates(img, c_pack, order=1, mode='constant', cval=0.0)
        return jnp.mean(patch)

    collect_all_cores_single_frame = jax.vmap(
        lambda img: jax.vmap(lambda r, c: extract_single_core_patch(img, r, c))(rows, cols)
    )
    return collect_all_cores_single_frame(images)


if backend == 'jax':
    _apply_pupil_jax_core = jax.jit(_jax_fft_batch, static_argnames=['pad_width'])
    batch_collect_subpixel_signals = jax.jit(batch_collect_subpixel_signals)

def get_simulation_parameters(nrings=N_RINGS, wavelength_um=DEFAULT_WAVELENGTH_UM):
    """Return a dict of all setup constants and derived parameters."""

    wavelength_nm = wavelength_um * 1000.0
    scaling_factor = wavelength_um / DEFAULT_WAVELENGTH_UM
    # scaling_factor = 1.0

    params = {
        "nrings":        nrings,
        "wl":            wavelength_um,
        "wavelength_nm": wavelength_nm,
        "taper_factor":  12.,
        "rclad":         9.0,
        "rjack":         27,
        "z_ex":          L1 + L2,
        "nclad":         1.444,
        "pad_factor":    4,
        "core_res":      16,
        "clad_res":      60,
        "jack_res":      30,
        "pixel_scale_um": 1,  # Physical scale of each pixel in the padded FFT grid (μm/px)
        "ifunc_file":    '/raid2/gcarla/git/ANDES/andes/PASSATA_scripts/data/ifunc/ANDES_400pix_all_modes.fits',
    }
    params["rcore"]  = 1.8 / params["taper_factor"]
    params["ncore"]  = params["nclad"] + 8.8e-3
    params["njack"]  = params["nclad"] - 5.5e-3
    params["output_positions"] = hex_ring_positions(params["nrings"], params["rclad"] / 2.5)
    params["n_output_positions"] = len(params["output_positions"])
    params["rcores"] = [params["rcore"]] * params["n_output_positions"]
    params["ncores"] = [params["ncore"]] * params["n_output_positions"]
    
    # Focal-plane (post-FFT) pixel scale.
    #
    # The pupil field is sampled on grid_size px and zero-padded to
    # grid_size * pad_factor before fft2, so the diffraction pattern is
    # sampled at exactly `pad_factor` px per (lambda/D).  The Airy first-null
    # radius is therefore 1.22 * pad_factor px -- it does NOT scale with
    # grid_size (the earlier `* grid_size / 2` term shrank pixel_scale_um by
    # ~200x, so the padded grid only spanned +-5 um and ~2/3 of the +-27 um
    # lantern mesh fell outside it -> "mesh points map outside FFT grid").
    #
    # pixel_scale_um is then fixed by requiring the Airy radius to equal
    # psf_fill_factor * rclad in physical units.
    psf_fill_factor = 0.85
    r_airy_px = 1.22 * params["pad_factor"]
    params["pixel_scale_um"] = psf_fill_factor * params["rclad"] / r_airy_px

    return params


# =====================================================================
# LAYER 2: WAVEGUIDE INFRASTRUCTURE
# =====================================================================
def _wavelength_tag(wavelength_nm: float) -> str:
    """Filename tag matching the convention used for the 800 nm cache
    (800.0 nm -> '0800'). Adjust if your cache naming differs, or if your
    native grid isn't on round-nm values -- in that case you likely want
    an explicit {wavelength_nm: tag} mapping instead of this formula."""
    return f"{int(round(wavelength_nm)):04d}"


def build_and_characterize_lantern(p):
    """Set up the PhotonicLantern and return a ChainPropagator."""        
    PL_nrings = PhotonicLantern(
        p["output_positions"], p["rcores"], p["rclad"], p["rjack"],
        p["ncores"], p["nclad"], p["njack"], p["z_ex"],
        p["taper_factor"], p["core_res"], p["clad_res"], p["jack_res"],
    )

    prop1 = Propagator(p["wl"], PL_nrings, p["n_output_positions"]+1)
    prop1.degen_groups  = default_degenerate_groups_front[p["nrings"]]
    prop1.skipped_modes = default_skipped_modes_front[p["nrings"]]

    cache_prefix = "port"
    n_output_positions = str(p["n_output_positions"])
    wavelength_nm = p["wavelength_nm"]
    
    tag = f"{n_output_positions}{cache_prefix}_{_wavelength_tag(wavelength_nm)}" + "_front"

    prop1.characterize(0,L1,save=True,tag=tag)
    #prop1.load(tag)

    prop2 = Propagator(p["wl"], PL_nrings, p["n_output_positions"]+1)    
    prop2.degen_groups  = default_degenerate_groups_back[p["nrings"]]
    prop2.skipped_modes = default_skipped_modes_back[p["nrings"]]

    prop2.load_init_conds(prop1)
    tag = f"{n_output_positions}{cache_prefix}_{_wavelength_tag(wavelength_nm)}" + "_back"
    prop2.characterize(L1,L1+L2,save=True,tag=tag)
    #prop2.load(tag)

    return ChainPropagator([prop1, prop2])


def get_waveguide_properties(prop12, mesh_z=0):
    """Extract waveguide modal properties at a given z position."""
    p_segment  = prop12.get_prop(mesh_z)
    mesh_obj   = p_segment.mesh
    mesh_areas = p_segment.wvg.assign_IOR()
    modes      = p_segment.vs

    if len(modes.shape) == 3:
        z_idx        = np.argmin(np.abs(p_segment.zs - mesh_z))
        active_modes = modes[z_idx]
    else:
        active_modes = modes

    n_mesh_points = mesh_obj.points.shape[0]
    areas         = getattr(p_segment, 'mesh_areas', np.ones(n_mesh_points))

    if active_modes.shape[0] == n_mesh_points:
        active_modes = active_modes.T

    n_modes   = active_modes.shape[0]
    points_2d = np.stack((mesh_obj.points[:, 0], mesh_obj.points[:, 1]), axis=-1)
        
    B     = construct_B(mesh_obj, sparse=True)
    areas = np.array(B.diagonal())
    return {
        'mesh':          mesh_obj,
        'mesh_areas':    areas,
        'points':        points_2d,
        'modes':         active_modes,
        'n_modes':       n_modes,
        'n_mesh_points': n_mesh_points,
    }


# =====================================================================
# LAYER 3: FIELD GENERATION & PROJECTION
# =====================================================================

class ModalProjector:
    """Project spatial field profiles onto the modal basis with normalisation."""

    def __init__(self, wvg_props, xp=np):
        self.xp = xp
        modes   = xp.asarray(wvg_props["modes"])
        areas   = xp.asarray(wvg_props["mesh_areas"])
        self.projection_matrix = modes.conj() * areas

    def project_batch(self, E_batch):
        """
        Project a batch of mesh-sampled electric fields onto the mode basis.

        Returns
        -------
        u0_batch : array, shape (n_fields, n_modes)
            Unit-L2-norm modal coefficients: the overlap integrals
            ``⟨vₖ | E⟩`` divided by their own L2 norm (rows that are exactly
            zero are left as zero).  This is a normalised initial condition,
            ready to hand to ``propagate``.  The absolute amount of pupil
            power that actually coupled into the tracked modes is *not*
            carried here -- it is returned separately as
            ``coupling_efficiency``.
        coupling_efficiency : array, shape (n_fields,)
            L2 norm of each projected vector *before* normalisation.
            ~1.0 means nearly all pupil power coupled into the tracked
            modes; << 1.0 flags misalignment or a pixel-scale problem.
            Retain this if you need output power spectra to be comparable
            across wavelengths -- the normalised ``u0_batch`` alone cannot be.
        """
        u0_batch = E_batch @ self.projection_matrix.T
        coupling_efficiency = self.xp.linalg.norm(u0_batch, axis=1)
        # Avoid division by zero for all-zero inputs (e.g. a masked-out field).
        safe_norms = coupling_efficiency[:, None]
        if self.xp is np:
            safe_norms = np.where(safe_norms == 0, 1.0, safe_norms)
        else:
            safe_norms = self.xp.where(safe_norms == 0, 1.0, safe_norms)
        return u0_batch / safe_norms, coupling_efficiency


class IncidentFieldGenerator:
    """Fixed version with correct centering and axis mapping."""

    def __init__(self, p, ifunc, xp=np, pupil_template=None):
        self.p     = p
        self.ifunc = ifunc
        self.xp    = xp

        if pupil_template is not None:
            # DM mask + influence-function matrix don't depend on
            # propagation wavelength -- reuse a copy built once elsewhere
            # instead of re-reading/re-stacking ifunc.influence_function.
            self.mask_np, self.grid_size, self.num_modes, self.ifunc_matrix = pupil_template
        else:
            self.mask_np   = ifunc.mask_inf_func.get() > 0
            self.grid_size = self.mask_np.shape[0]
            self.num_modes = len(ifunc.influence_function)
            self.ifunc_matrix = xp.stack(
                [f.get() for f in ifunc.influence_function], axis=0)

    def precompute_interpolation_weights(self, mesh_points):
        """
        FIXED VERSION: Corrects FFT center calculation and axis mapping.
        
        Key fixes:
        1. Proper even/odd grid center calculation for FFT convention
        2. Correct axis mapping (mesh Y->Y, mesh X->X)
        3. Validation checks
        """
        padded_size = self.grid_size * self.p["pad_factor"]
        
        # FIX 1: Correct center calculation for FFT convention
        # For even grids: center is at index padded_size // 2
        # For odd grids: center is at index (padded_size - 1) // 2
        if padded_size % 2 == 0:
            center_offset = padded_size / 2.0
        else:
            center_offset = (padded_size - 1) / 2.0
        
        print(f"[Interpolation Setup]")
        print(f"  Padded grid size: {padded_size}")
        print(f"  Grid parity: {'even' if padded_size % 2 == 0 else 'odd'}")
        print(f"  Center offset: {center_offset}")

        # FIX 2: Retrieve and validate pixel scale
        pixel_scale = self.p.get("pixel_scale_um", 1.0)
        print(f"  Pixel scale: {pixel_scale} μm/pixel")
        
        # FIX 3: CORRECT axis mapping
        # mesh_points is shaped (N_points, 2) where:
        #   mesh_points[:, 0] = Y coordinates (vertical)
        #   mesh_points[:, 1] = X coordinates (horizontal)
        # This matches the convention: points = (y, x) in mesh.points
        mesh_y = mesh_points[:, 0]  # Vertical coordinate
        mesh_x = mesh_points[:, 1]  # Horizontal coordinate
        
        # Validate mesh range
        print(f"  Mesh X range: [{mesh_x.min():.3f}, {mesh_x.max():.3f}] μm")
        print(f"  Mesh Y range: [{mesh_y.min():.3f}, {mesh_y.max():.3f}] μm")

        # Convert physical coordinates to continuous pixel indices
        # Grid coordinate = physical_coordinate / pixel_scale + center
        mesh_x_pix = (mesh_x / pixel_scale) + center_offset
        mesh_y_pix = (mesh_y / pixel_scale) + center_offset

        # Check for out-of-bounds points
        n_out_of_bounds = np.sum((mesh_y_pix < 0) | (mesh_y_pix >= padded_size) |
                                  (mesh_x_pix < 0) | (mesh_x_pix >= padded_size))
        if n_out_of_bounds > 0:
            print(f"  ⚠️  WARNING: {n_out_of_bounds} mesh points map outside FFT grid!")
            print(f"     Check pixel_scale_um and mesh extent.")

        # Bilinear interpolation weights
        iy0 = self.xp.floor(mesh_y_pix).astype(self.xp.int32)
        iy1 = iy0 + 1
        ix0 = self.xp.floor(mesh_x_pix).astype(self.xp.int32)
        ix1 = ix0 + 1

        wy1 = mesh_y_pix - iy0
        wy0 = 1.0 - wy1
        wx1 = mesh_x_pix - ix0
        wx0 = 1.0 - wx1

        # Boundary validation
        valid = ((iy0 >= 0) & (iy1 < padded_size) &
                 (ix0 >= 0) & (ix1 < padded_size))

        # Clip indices
        iy0 = self.xp.clip(iy0, 0, padded_size - 1)
        iy1 = self.xp.clip(iy1, 0, padded_size - 1)
        ix0 = self.xp.clip(ix0, 0, padded_size - 1)
        ix1 = self.xp.clip(ix1, 0, padded_size - 1)

        # Store
        self.iy0 = self.xp.asarray(iy0)
        self.iy1 = self.xp.asarray(iy1)
        self.ix0 = self.xp.asarray(ix0)
        self.ix1 = self.xp.asarray(ix1)
        self.w00 = self.xp.asarray((wy0 * wx0 * valid)[:, None])
        self.w01 = self.xp.asarray((wy0 * wx1 * valid)[:, None])
        self.w10 = self.xp.asarray((wy1 * wx0 * valid)[:, None])
        self.w11 = self.xp.asarray((wy1 * wx1 * valid)[:, None])
        
        print(f"  Valid interpolation points: {np.sum(valid)} / {len(valid)}")

    def generate_field_profiles_batch(self, coeff_batch):
        """Generate complex electric field profiles from aberration coefficients."""
        opd_flat_batch   = coeff_batch @ self.ifunc_matrix
        phase_flat_batch = opd_flat_batch * (2 * self.xp.pi / self.p["wavelength_nm"])
        E_flat_batch     = self.xp.exp(1j * phase_flat_batch)

        n_fields      = coeff_batch.shape[0]
        Ef_input_batch = self.xp.zeros(
            (n_fields, self.grid_size, self.grid_size), dtype=self.xp.complex128)
        if self.xp is np:
            Ef_input_batch[:, self.mask_np] = E_flat_batch
        else:
            Ef_input_batch = Ef_input_batch.at[:, self.mask_np].set(E_flat_batch)
        return Ef_input_batch

    def apply_ef_to_lantern_jax(self, Ef_input_batch):
        """Pad and FFT-transform using JAX (runs on the selected device)."""
        pad_width         = (self.grid_size * self.p["pad_factor"] - self.grid_size) // 2
        Ef_input_batch_dev = jax.device_put(Ef_input_batch, _jax_device)
        return _apply_pupil_jax_core(Ef_input_batch_dev, pad_width)

    def apply_ef_to_lantern(self, Ef_input_batch):
        """Pad and FFT-transform using SciPy (CPU, multithreaded)."""
        import scipy.fft as sp_fft
        N           = self.grid_size
        pad_width   = (N * self.p["pad_factor"] - N) // 2
        padded_size = N + 2 * pad_width

        E_padded = np.zeros(
            (Ef_input_batch.shape[0], padded_size, padded_size),
            dtype=np.complex128)
        E_padded[:, pad_width:pad_width+N, pad_width:pad_width+N] = \
            np.asarray(Ef_input_batch)

        return sp_fft.fftshift(
            sp_fft.fft2(
                sp_fft.ifftshift(E_padded, axes=(1, 2)),
                axes=(1, 2),
                workers=-1,
            ),
            axes=(1, 2),
        )

    def resample_to_mesh(self, E_lantern_batch):
        """Bilinear interpolation from the padded FFT grid onto the FE mesh."""
        val00 = E_lantern_batch[:, self.iy0, self.ix0]
        val01 = E_lantern_batch[:, self.iy0, self.ix1]
        val10 = E_lantern_batch[:, self.iy1, self.ix0]
        val11 = E_lantern_batch[:, self.iy1, self.ix1]
        return (val00 * self.w00.T + val01 * self.w01.T +
                val10 * self.w10.T + val11 * self.w11.T)


# =====================================================================
# LAYER 4: BATCH PROPAGATION ENGINE
# =====================================================================

class BatchPropagationPipeline:
    """
    Orchestrates batch propagation of modal fields through a photonic lantern.

    Parameters
    ----------
    prop12 :               ChainPropagator
    p :                    simulation parameter dict
    ifunc :                influence-function object
    field_gen_on_gpu :     bool, optional (default False)
        When True *and* the JAX backend is active, field generation runs on
        the GPU.  When False (default), field generation always uses
        NumPy/SciPy on the CPU, which avoids GPU memory pressure during the
        OPD → pupil → FFT → projection phase.
        Can be overridden per call via the ``use_gpu`` keyword of
        ``generate_batch_modal_coefficients()``.
    field_gen_chunk_size : int or None, optional (default 64)
        Number of fields processed in each chunk of
        ``generate_batch_modal_coefficients()``.  The large intermediate
        arrays (OPD batch, padded FFT grid, mesh-sampled fields) are
        allocated, used, and released chunk by chunk, capping peak memory
        to O(chunk_size) independently of the total batch size and
        independently of which backend is used.  Set to None to disable
        chunking (process the whole batch at once).
        Can be overridden per call via the ``chunk_size`` keyword.
    """

    def __init__(self, prop12, p, ifunc,
                 field_gen_on_gpu=False,
                 field_gen_chunk_size=default_gen_chunk_size,
                 field_gen_pupil_template=None):
        
        self.prop12 = prop12
        self.p      = p

        # The *propagation* xp follows the global backend.
        self.xp = jnp if backend == 'jax' else np

        # Default chunk size for generate_batch_modal_coefficients().
        # None means "no chunking" (process the whole batch at once).
        self.field_gen_chunk_size = field_gen_chunk_size

        # -----------------------------------------------------------------
        # Decide the default array module for field generation.
        # -----------------------------------------------------------------
        self._field_gen_on_gpu_default = (backend == 'jax') and field_gen_on_gpu

        # Always build the numpy generators (cheap, host memory only).
        self._field_gen_pupil_template = field_gen_pupil_template
        self._field_gen_np   = IncidentFieldGenerator(
            p, ifunc, xp=np, pupil_template=field_gen_pupil_template)

        
        self._modal_proj_np  = None   # built after weight precomputation below

        # Build JAX generators only if explicitly requested.
        self._field_gen_jax  = None
        self._modal_proj_jax = None
        self._ifunc_ref      = ifunc  # kept for lazy JAX init

        wvg_props_input = get_waveguide_properties(prop12, mesh_z=0)
        self.wvg_props_input  = wvg_props_input
        self.wvg_props_output = get_waveguide_properties(prop12, mesh_z=p["z_ex"])

        self._field_gen_np.precompute_interpolation_weights(
            wvg_props_input['points'])
        self._modal_proj_np = ModalProjector(wvg_props_input, xp=np)

        if backend == 'jax' and field_gen_on_gpu:
            self._build_jax_field_gen()

        # Delaunay cache for output interpolation.
        self._delaunay_cache = {}
        # not needed:
        # self._delaunay_cache[DEFAULT_GRID_RESOLUTION] = \
        #     self._precompute_delaunay_grid(DEFAULT_GRID_RESOLUTION)

    # ------------------------------------------------------------------
    def _build_jax_field_gen(self):
        """Lazily initialise the JAX field-generation objects."""
        if self._field_gen_jax is not None:
            return
        self._field_gen_jax = IncidentFieldGenerator(
            self.p, self._ifunc_ref, xp=jnp)
        self._field_gen_jax.precompute_interpolation_weights(
            jnp.asarray(self.wvg_props_input['points'],
                        dtype=jnp.float64, device=_jax_device))
        self._modal_proj_jax = ModalProjector(self.wvg_props_input, xp=jnp)

    # ------------------------------------------------------------------
    def enable_gpu_field_gen(self):
        """
        Lazily initialise the JAX field-generation objects after construction.

        Call this if the pipeline was built with ``field_gen_on_gpu=False``
        but you later want to use ``use_gpu=True`` in some calls.
        """
        if backend != 'jax':
            raise RuntimeError(
                "enable_gpu_field_gen() requires the JAX backend.")
        self._build_jax_field_gen()
        print("JAX field-generation objects initialised on device:", _jax_device)

    # ------------------------------------------------------------------
    def _precompute_delaunay_grid(self, grid_resolution):
        """Precompute Delaunay triangulation and barycentric weights."""
        from scipy.spatial import Delaunay

        mesh_pts = self.wvg_props_output['mesh'].points
        x_min, x_max = mesh_pts[:, 0].min(), mesh_pts[:, 0].max()
        y_min, y_max = mesh_pts[:, 1].min(), mesh_pts[:, 1].max()

        plot_x = np.linspace(x_min, x_max, grid_resolution)
        plot_y = np.linspace(y_min, y_max, grid_resolution)
        X_plot, Y_plot = np.meshgrid(plot_x, plot_y)
        grid_pts = np.column_stack((X_plot.ravel(), Y_plot.ravel()))

        tri        = Delaunay(mesh_pts[:, :2])
        v_simplex  = tri.find_simplex(grid_pts)
        valid_grid = v_simplex >= 0

        v0 = np.zeros(grid_pts.shape[0], dtype=np.int32)
        v1 = np.zeros(grid_pts.shape[0], dtype=np.int32)
        v2 = np.zeros(grid_pts.shape[0], dtype=np.int32)
        w0 = np.zeros(grid_pts.shape[0], dtype=np.float64)
        w1 = np.zeros(grid_pts.shape[0], dtype=np.float64)
        w2 = np.zeros(grid_pts.shape[0], dtype=np.float64)

        if np.any(valid_grid):
            valid_simplices = v_simplex[valid_grid]
            v0[valid_grid]  = tri.simplices[valid_simplices, 0]
            v1[valid_grid]  = tri.simplices[valid_simplices, 1]
            v2[valid_grid]  = tri.simplices[valid_simplices, 2]

            transform = tri.transform[valid_simplices]
            r3        = tri.simplices[valid_simplices, 2].astype(np.intp)
            delta     = grid_pts[valid_grid] - tri.points[r3]
            bary      = np.einsum('nij,nj->ni', transform[:, :2], delta)
            w0[valid_grid] = bary[:, 0]
            w1[valid_grid] = bary[:, 1]
            w2[valid_grid] = 1.0 - bary[:, 0] - bary[:, 1]

        return X_plot, Y_plot, v0, v1, v2, w0, w1, w2, valid_grid

    # ------------------------------------------------------------------
    def interpolate_output_to_grid(self, E_output_batch,
                                   grid_resolution=DEFAULT_GRID_RESOLUTION):
        """Interpolate output spatial fields to a regular grid (always CPU)."""
        if grid_resolution not in self._delaunay_cache:
            self._delaunay_cache[grid_resolution] = \
                self._precompute_delaunay_grid(grid_resolution)
        X_plot, Y_plot, v0, v1, v2, w0, w1, w2, valid_grid = \
            self._delaunay_cache[grid_resolution]

        E_np            = np.asarray(E_output_batch)
        intensity_batch = np.abs(E_np) ** 2

        flat_interpolated = (
            intensity_batch[:, v0] * w0 +
            intensity_batch[:, v1] * w1 +
            intensity_batch[:, v2] * w2
        )
        flat_interpolated[:, ~valid_grid] = 0.0

        n_fields    = E_np.shape[0]
        uf_2d_batch = flat_interpolated.reshape(
            n_fields, grid_resolution, grid_resolution)
        return uf_2d_batch, X_plot, Y_plot

    # ------------------------------------------------------------------
    def _process_chunk(self, coeff_chunk, fg, proj, use_gpu):
        """Run the field-generation pipeline on a single coefficient chunk.

        Returns a plain numpy array of shape (chunk_size, n_modes).
        All intermediate arrays are local to this call and are released
        when it returns.
        """
        Ef_input  = fg.generate_field_profiles_batch(coeff_chunk)
        if use_gpu:
            E_lantern = fg.apply_ef_to_lantern_jax(Ef_input)
        else:
            E_lantern = fg.apply_ef_to_lantern(Ef_input)
        Ef_focal_mesh   = fg.resample_to_mesh(E_lantern)
        u0_chunk, eff = proj.project_batch(Ef_focal_mesh)
        return np.asarray(u0_chunk)

    # ------------------------------------------------------------------
    def generate_batch_modal_coefficients(self, aberration_coeff_batch,
                                          use_gpu=None,
                                          chunk_size=default_gen_chunk_size):
        """
        Generate batch modal coefficients from aberration configurations.

        The input batch is split into chunks of *chunk_size* fields.  For
        each chunk the full pipeline (OPD → pupil → FFT → mesh resample →
        modal projection) is executed and its result accumulated in a plain
        numpy array on the host before the next chunk is started.  This
        limits peak memory to O(chunk_size) **regardless of the total batch
        size and regardless of which backend is used**.

        Parameters
        ----------
        aberration_coeff_batch : array-like, shape (n_configs, n_active_modes)
            Each row is one aberration configuration.  Zeros are appended
            automatically if n_active_modes < n_modes_total.
        use_gpu : bool or None, optional
            Override the instance-level ``field_gen_on_gpu`` setting.

            - ``None``  (default) — use the instance default.
            - ``False`` — force CPU/NumPy for this call.
            - ``True``  — force GPU/JAX for this call (requires JAX backend
              and initialised JAX generators).
        chunk_size : int or None, optional
            Number of fields per processing chunk.  Overrides the instance-
            level ``field_gen_chunk_size`` for this call only.

            - ``None``  — use the instance default (``field_gen_chunk_size``).
            - Positive int — use that chunk size.
            - ``0`` or negative — disable chunking (process whole batch).

        Returns
        -------
        u0_batch : np.ndarray, shape (n_configs, n_modes), complex128
            Normalised modal coefficients, always a plain numpy array.
        """
        # ------------------------------------------------------------------
        # Resolve backend choice.
        # ------------------------------------------------------------------
        if use_gpu is None:
            _use_gpu = self._field_gen_on_gpu_default
        else:
            _use_gpu = bool(use_gpu)

        if _use_gpu and backend != 'jax':
            raise RuntimeError(
                "use_gpu=True requires the JAX backend "
                "(set CBEAM_BACKEND=jax before importing cbeam).")
        if _use_gpu and self._field_gen_jax is None:
            raise RuntimeError(
                "JAX field generators have not been initialised.  "
                "Either construct BatchPropagationPipeline with "
                "field_gen_on_gpu=True, or call "
                "pipeline.enable_gpu_field_gen() first.")

        fg   = self._field_gen_jax  if _use_gpu else self._field_gen_np
        proj = self._modal_proj_jax if _use_gpu else self._modal_proj_np

        # ------------------------------------------------------------------
        # Resolve chunk size.
        # ------------------------------------------------------------------
        if chunk_size is None:
            _chunk_size = self.field_gen_chunk_size
        elif chunk_size <= 0:
            _chunk_size = None   # explicit disable
        else:
            _chunk_size = int(chunk_size)

        # ------------------------------------------------------------------
        # Prepare the full coefficient array (host numpy first, then
        # optionally moved to device inside _process_chunk).
        # ------------------------------------------------------------------
        coeff_np = np.asarray(aberration_coeff_batch, dtype=np.float64)
        if coeff_np.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {coeff_np.shape}")

        n_configs, n_active_modes = coeff_np.shape
        n_modes_total = fg.num_modes

        if n_active_modes < n_modes_total:
            padded = np.zeros((n_configs, n_modes_total), dtype=np.float64)
            padded[:, :n_active_modes] = coeff_np
            coeff_np = padded

        # ------------------------------------------------------------------
        # Process in chunks.
        # ------------------------------------------------------------------
        if _chunk_size is None or _chunk_size >= n_configs:
            # Single pass — no chunking overhead.
            coeff_chunk = self._to_device(coeff_np, _use_gpu)
            return self._process_chunk(coeff_chunk, fg, proj, _use_gpu)

        print(f"  Generating modal coefficients for {n_configs} fields "
              f"(chunk_size={_chunk_size}, "
              f"device={'GPU' if _use_gpu else 'CPU'}) ...", flush=True)

        u0_chunks = []
        for start in range(0, n_configs, _chunk_size):
            end         = min(start + _chunk_size, n_configs)
            coeff_chunk = self._to_device(coeff_np[start:end], _use_gpu)
            u0_chunk    = self._process_chunk(coeff_chunk, fg, proj, _use_gpu)
            u0_chunks.append(u0_chunk)
            print(f"    field-gen chunk {start}–{end-1} done.", flush=True)

        print("  Field generation complete.")
        return np.concatenate(u0_chunks, axis=0)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_device(coeff_np, use_gpu):
        """Move a numpy coefficient array to the appropriate device."""
        if use_gpu:
            return jnp.asarray(coeff_np, dtype=jnp.float64, device=_jax_device)
        return coeff_np   # already numpy, no copy needed

    # ------------------------------------------------------------------
    def propagate_batch(self, u0_batch, chunk_size=None):
        """
        Propagate a batch of modal coefficient vectors through the waveguide.

        For the JAX backend the batch is split into chunks of *chunk_size*
        fields.  Each chunk is propagated, its output is immediately moved
        to host memory, and only then is the next chunk started, keeping
        peak GPU memory O(chunk_size) rather than O(n_batch).

        Parameters
        ----------
        u0_batch   : np.ndarray, shape (n_fields, n_modes)
        chunk_size : int or None
            If None a safe default is chosen:
              - JAX backend:   32  (tune upward if GPU has headroom)
              - numpy backend: full batch (no chunking needed)

        Returns
        -------
        uf_batch  : np.ndarray (n_fields, n_modes)
        zs        : array of z values (from the last chunk; all identical)
        us_batch  : np.ndarray (n_fields, n_z, n_modes) or None
                    Full trajectory — only populated for single-chunk runs
                    to avoid excessive memory use.
        """
        n_fields = u0_batch.shape[0]

        if chunk_size is None:
            chunk_size = default_chunk_size if backend == 'jax' else n_fields

        print(f"  Propagating {n_fields} fields "
              f"(chunk_size={chunk_size}) ...", flush=True)

        uf_chunks   = []
        us_chunks   = []
        zs_last     = None
        multi_chunk = (n_fields > chunk_size)

        for start in range(0, n_fields, chunk_size):
            end   = min(start + chunk_size, n_fields)
            chunk = u0_batch[start:end]

            if backend == 'jax':
                chunk = jnp.asarray(chunk, dtype=jnp.complex128,
                                    device=_jax_device)

            zs, us_grid, uf = self.prop12.propagate(chunk)

            uf_chunks.append(np.asarray(uf))
            zs_last = np.asarray(zs)

            if not multi_chunk:
                us_chunks.append(np.asarray(us_grid))

            print(f"    propagation chunk {start}–{end-1} done.", flush=True)

        uf_batch = np.concatenate(uf_chunks, axis=0)

        if multi_chunk:
            us_batch = None
            print("  (us_batch not returned for multi-chunk runs to save memory)")
        else:
            us_grid_full = np.concatenate(us_chunks, axis=0)
            us_batch = (np.transpose(us_grid_full, (1, 0, 2))
                        if us_grid_full.ndim == 3 else us_grid_full)

        print("  Batch propagation complete.")
        return uf_batch, zs_last, us_batch

    # ------------------------------------------------------------------
    def reconstruct_batch_output_fields(self, uf_batch):
        """Reconstruct spatial output fields from modal coefficients (CPU)."""
        modes_out = self.wvg_props_output['modes']
        return np.asarray(uf_batch) @ modes_out


# =====================================================================
# LAYER 5: VISUALISATION & ANALYSIS
# =====================================================================

def visualize_batch_output(uf_2d_batch, X_plot, Y_plot, titles=None, maxv=1,
                           peak_box_size=5, show_arrow=True):
    """
    Visualise a batch of output intensity maps with peak detection and annotation.

    Parameters
    ----------
    uf_2d_batch : ndarray, shape (n_fields, nx, ny)
        Batch of intensity maps on the regular grid.
    X_plot, Y_plot : ndarray, shape (nx, ny) or (ny, nx)
        Meshgrid arrays of spatial coordinates (μm).
    titles : list of str, optional
        Titles for each field.
    maxv : int
        Maximum number of fields to plot.
    peak_box_size : int (odd)
        Size of the square (in pixels) used for averaging around the peak.
    show_arrow : bool
        If True, draw a cyan arrow from the top‑right corner pointing to the peak.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from scipy.ndimage import uniform_filter

    n_fields = min(uf_2d_batch.shape[0], maxv)
    n_cols = min(3, n_fields)
    n_rows = (n_fields + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_fields == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n_fields):
        ax = axes[i]
        data = uf_2d_batch[i]
        
        # Convert complex to intensity
        data_intensity = np.abs(data)
        
        # Log intensity for colormap
        im_log = np.log(data_intensity + 1e-10)
        im = ax.imshow(im_log, cmap='inferno',
                       extent=[X_plot.min(), X_plot.max(),
                               Y_plot.min(), Y_plot.max()],
                       origin='lower')
        ax.set_title(titles[i] if titles else f"Field {i}")
        ax.set_xlabel('x (μm)')
        ax.set_ylabel('y (μm)')
        plt.colorbar(im, ax=ax, label='log(Intensity)')

        # ------- Peak detection as center of box with highest average signal -------
        # Compute average intensity in each box using convolution
        box_averages = uniform_filter(data_intensity, size=peak_box_size, 
                                      mode='constant', cval=0)
        
        # Find the box center with highest average
        peak_idx = np.unravel_index(np.argmax(box_averages), box_averages.shape)
        row, col = peak_idx
        
        # Physical coordinates
        x_peak = X_plot[row, col]
        y_peak = Y_plot[row, col]

        # Get statistics for that box
        half = peak_box_size // 2
        r0 = max(0, row - half)
        r1 = min(data_intensity.shape[0], row + half + 1)
        c0 = max(0, col - half)
        c1 = min(data_intensity.shape[1], col + half + 1)
        peak_region = data_intensity[r0:r1, c0:c1]
        peak_avg = np.mean(peak_region)
        peak_max = np.max(peak_region)

        # ------- Annotations -------
        # Red 'x' marker at the peak
        ax.plot(x_peak, y_peak, 'rx', markersize=12, markeredgewidth=2,
                label='Peak')

        # Text box with peak and average values
        ax.text(x_peak, y_peak,
                f'Peak: {peak_max:.2e}\nAvg({peak_box_size}x{peak_box_size}): {peak_avg:.2e}',
                color='white', fontsize=8, va='bottom', ha='left',
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

        # Optional arrow pointing to the peak
        if show_arrow:
            # Arrow starting near the top‑right corner of the plot
            x_range = X_plot.max() - X_plot.min()
            y_range = Y_plot.max() - Y_plot.min()
            start_x = X_plot.max() - 0.05 * x_range
            start_y = Y_plot.max() - 0.05 * y_range
            arrow = FancyArrowPatch((start_x, start_y), (x_peak, y_peak),
                                    arrowstyle='->', mutation_scale=20,
                                    color='cyan', linewidth=2)
            ax.add_patch(arrow)

        # Legend (shows only 'Peak' entry)
        ax.legend(loc='upper right', fontsize=8)

    # Hide any unused subplots
    for j in range(n_fields, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def display_hex_grid_plots(ideal_centers, standardized_signals):
    """Generates the concurrent side-by-side 3D column and 2D map views."""
    heights = standardized_signals * 1000
    centers = np.array(ideal_centers)
    hex_radius = 0.4
    
    vmin, vmax = heights.min(), heights.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis
    
    min_x, max_x = centers[:, 0].min() - 1, centers[:, 0].max() + 1
    min_y, max_y = centers[:, 1].min() - 1, centers[:, 1].max() + 1
    
    def create_hexagon_vertices(cx, cy, r):
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        return np.column_stack((cx + r * np.cos(angles), cy + r * np.sin(angles)))

    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    
    polys, colors_list, n_sides = [], [], 6
    for i, (x, y) in enumerate(centers):
        height = heights[i]
        color = cmap(norm(height))
        hex_vertices = create_hexagon_vertices(x, y, hex_radius)
        
        base_3d = np.column_stack((hex_vertices, np.zeros(n_sides)))
        top_3d = np.column_stack((hex_vertices, np.full(n_sides, height)))
        vertices_3d = np.vstack((base_3d, top_3d))
        
        polys.extend([vertices_3d[:n_sides], vertices_3d[n_sides:]])
        colors_list.extend([color, color])
        
        for j in range(n_sides):
            side_face = [vertices_3d[j], vertices_3d[(j+1)%n_sides], vertices_3d[(j+1)%n_sides + n_sides], vertices_3d[j + n_sides]]
            polys.append(side_face)
            colors_list.append(color)

    poly_collection = Poly3DCollection(polys, alpha=0.7, edgecolor='black', linewidth=0.5)
    poly_collection.set_facecolor(colors_list)
    ax1.add_collection3d(poly_collection)
    ax1.set_xlim(min_x, max_x); ax1.set_ylim(min_y, max_y); ax1.set_zlim(0, heights.max() * 1.1)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Height')
    ax1.set_title('3D Height Field on Hexagonal Grid')
    
    for i, (x, y) in enumerate(centers):
        height = heights[i]
        hexagon = RegularPolygon((x, y), numVertices=6, radius=hex_radius, orientation=0,
                                 facecolor=cmap(norm(height)), edgecolor='black', linewidth=1.5, alpha=0.8)
        ax2.add_patch(hexagon)
        ax2.text(x, y, f'{height:.1f}', ha='center', va='center', fontsize=8, fontweight='bold')

    ax2.set_xlim(min_x, max_x); ax2.set_ylim(min_y, max_y); ax2.set_aspect('equal')
    ax2.set_xlabel('X', fontsize=12); ax2.set_ylabel('Y', fontsize=12)
    ax2.set_title('Top View: Hexagonal Grid Height Field', fontsize=14)
    ax2.grid(True, alpha=0.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=[ax1, ax2], pad=0.05, shrink=0.7).set_label('Height', fontsize=12)
    plt.show()
    
def visualize_batch_hex_grid_signals(
    pipeline,
    E_output_batch,
    ideal_grid_positions,
    calibrate_subpixel_centers,
    collect_subpixel_signals,
    map_evaluated_to_ideal_geometry,
    display_hex_grid_plots,
    titles=None,
    grid_resolution=DEFAULT_GRID_RESOLUTION,
):
    """Execute centroid sub-pixel tracking and display hex grid configurations."""
    mesh_final            = pipeline.wvg_props_output['mesh']
    waveguide_modes_final = pipeline.wvg_props_output['modes']

    # =====================================================================
    # FIX: Collapse (n_modes, n_mesh_points) into a 1D map (n_mesh_points,)
    # =====================================================================
    total_modes_profile = np.sum(np.abs(waveguide_modes_final) ** 2, axis=0)

    # Pass the 1D combined intensity profile instead of the raw 2D modes matrix
    core_centers, _, _, dx, dy = calibrate_subpixel_centers(
        total_modes_profile, mesh_final)
    # =====================================================================

    ideal_permutation = map_evaluated_to_ideal_geometry(
        core_centers, ideal_grid_positions)

    n_fields = E_output_batch.shape[0]
    print("\n" + "=" * 60)
    print("GEOMETRIC HEX SIGNAL EXTRACTION PROCESSING")
    print("=" * 60)

    uf_2d_batch, X_plot_out, Y_plot_out = pipeline.interpolate_output_to_grid(
        E_output_batch, grid_resolution)

    # Physical -> pixel mapping for the output grid, taken from the grid itself
    # (the core centres live in the same physical frame, over the mesh extent).
    x_min = X_plot_out[0, 0]
    y_min = Y_plot_out[0, 0]
    dx    = X_plot_out[0, 1] - X_plot_out[0, 0]
    dy    = Y_plot_out[1, 0] - Y_plot_out[0, 0]

    for i in range(n_fields):
        field_title = titles[i] if titles else f"Field {i}"
        print(f"Processing core integration tracks for: {field_title}...")

        output_signals = collect_subpixel_signals(
            uf_2d_batch[i], x_min, y_min, dx, dy, core_centers, n=7)

        standardized_signals = np.zeros(len(output_signals))
        standardized_signals[ideal_permutation] = output_signals

        print(f"Displaying Core Matrix for: {field_title}")
        display_hex_grid_plots(ideal_grid_positions, standardized_signals)


def batch_statistics(uf_batch, labels=None):
    """Compute and display statistics of batch propagation outputs."""
    n_fields = uf_batch.shape[0]
    print("\n" + "=" * 60)
    print("BATCH PROPAGATION STATISTICS")
    print("=" * 60)
    for i in range(n_fields):
        label       = labels[i] if labels else f"Field {i}"
        mode_powers = np.abs(uf_batch[i]) ** 2
        total_power = np.sum(mode_powers)
        max_mode    = np.argmax(mode_powers)
        top_3_power = np.sum(np.sort(mode_powers)[-3:])
        print(f"\n{label}:")
        print(f"  Total Power:          {total_power:.4f}")
        print(f"  Max Mode (idx):       {max_mode}")
        print(f"  Max Mode Power:       {mode_powers[max_mode]:.4f}")
        print(f"  Power in Top 3 Modes: {top_3_power:.4f}")


# =====================================================================
# LAYER 6: MAIN EXECUTION PIPELINE
# =====================================================================

def main_batch_propagation():
    """Main execution function for the batch propagation pipeline."""
    print("=" * 60)
    print("PHOTONIC LANTERN BATCH PROPAGATION PIPELINE")
    print("=" * 60)

    p = get_simulation_parameters()
    print(f"\n[1/5] Configuration loaded.")
    print(f"      Wavelength: {p['wl']} μm  |  Lantern length: {p['z_ex']} μm")
    print(f"      Global backend: {backend}")

    print(f"\n[2/5] Loading influence function...")
    import specula
    specula.init(0)
    from specula.data_objects.ifunc import IFunc
    ifunc = IFunc.restore(p["ifunc_file"])
    print(f"      Modes available: {len(ifunc.influence_function)}")

    print(f"\n[3/5] Building waveguide propagator...")
    prop12 = build_and_characterize_lantern(p)
    print(f"      Modes: 20, Segments: 2")

    print(f"\n[4/5] Initialising batch pipeline ...")
    # field_gen_on_gpu=False  → field generation uses NumPy/SciPy (CPU)
    # field_gen_chunk_size=64 → process 64 fields at a time during field gen
    pipeline = BatchPropagationPipeline(
        prop12, p, ifunc,
        field_gen_on_gpu=False,
        field_gen_chunk_size=64,
    )

    mode_indices  = [0, 3, 5]
    amplitudes_nm = [0.0, 100.0, 300.0, 600.0]
    aberration_configs = [
        {'mode_idx': m, 'amplitude_nm': a}
        for m in mode_indices for a in amplitudes_nm
    ]
    print(f"      Batch size: {len(aberration_configs)} fields")
    print(f"      Modes: {mode_indices}, Amplitudes (nm): {amplitudes_nm}")

    print(f"\n[5/5] Generating and propagating batch...")
    n_total      = pipeline._field_gen_np.num_modes
    coeff_matrix = np.zeros((len(aberration_configs), n_total), dtype=np.float64)
    for k, cfg in enumerate(aberration_configs):
        coeff_matrix[k, cfg['mode_idx']] = cfg['amplitude_nm']

    # chunk_size=None → use the instance default (field_gen_chunk_size=64).
    # Pass an explicit int to override, e.g. chunk_size=32 for tighter memory.
    u0_batch = pipeline.generate_batch_modal_coefficients(
        coeff_matrix, use_gpu=False)
    print(f"      Input field batch shape: {u0_batch.shape}")

    # chunk_size=8 is conservative for propagation; increase if GPU allows.
    uf_batch, zs, us_batch = pipeline.propagate_batch(u0_batch, chunk_size=8)
    print(f"      Output modal coefficient batch shape: {uf_batch.shape}")

    E_output_batch = pipeline.reconstruct_batch_output_fields(uf_batch)
    print(f"      Output spatial field batch shape: {E_output_batch.shape}")

    uf_2d_batch, X_plot, Y_plot = pipeline.interpolate_output_to_grid(E_output_batch)
    print(f"      Output grid batch shape: {uf_2d_batch.shape}")

    titles = [f"Mode {c['mode_idx']}, {c['amplitude_nm']:.0f} nm"
              for c in aberration_configs]

    print(f"\n[Results] Visualising batch output...")
    visualize_batch_output(uf_2d_batch, X_plot, Y_plot, titles)
    batch_statistics(uf_batch, titles)

    print("\n" + "=" * 60)
    print("BATCH PROPAGATION COMPLETE")
    print("=" * 60)

    return {
        'u0_batch':       u0_batch,
        'uf_batch':       uf_batch,
        'E_output_batch': E_output_batch,
        'uf_2d_batch':    uf_2d_batch,
        'zs':             zs,
        'us_batch':       us_batch,
        'configs':        aberration_configs,
        'pipeline':       pipeline,
    }


def create_sparse_aberration_configs_mono(n, m, minv, maxv):
    """
    Generate *n* random aberration configurations where each configuration 
    has only one active mode (value) that varies, while others remain zero.

    Args:
        n    (int): Number of configurations.
        m    (int): Number of modes available (indices 0 to m-1).
        minv (float): Minimum amplitude (nm).
        maxv (float): Maximum amplitude (nm).

    Returns:
        np.ndarray of shape (n, m), dtype float64.
    """
    # Initialize all with zeros
    configs = np.zeros((n, m), dtype=np.float64)
    
    # Pick a random mode index for each of the n configurations
    random_mode_indices = np.random.randint(0, m, size=n)
    
    # Generate the random amplitudes for those specific positions
    amplitudes = np.random.uniform(minv, maxv, size=n)
    
    # Assign the values
    configs[np.arange(n), random_mode_indices] = amplitudes
    
    return configs

def create_random_aberration_configs(n, m, minv, maxv):
    """
    Generate *n* random aberration configurations each with *m* active modes.

    Args:
        n    (int):   Number of configurations.
        m    (int):   Number of active modes per config (modes 0 … m-1).
        minv (float): Minimum amplitude (nm).
        maxv (float): Maximum amplitude (nm).

    Returns:
        np.ndarray of shape (n, m), dtype float64.
    """
    return np.random.uniform(minv, maxv, (n, m)).astype(np.float64)

def create_ramp_aberration_configs(modes, n_steps, minv, maxv):
    """
    Generate aberration configurations that ramp each mode individually.

    For every mode index in *modes* a sequence of *n_steps* configurations is
    produced in which that mode's amplitude is varied linearly from *minv* to
    *maxv* while all other modes are kept at zero.  The resulting array has
    ``len(modes) * n_steps`` rows.

    Args:
        modes   (list[int]): mode indices to ramp, e.g. [0, 1, 2].
        n_steps (int):       number of amplitude steps per mode (≥ 2).
        minv    (float):     starting amplitude value (nm).
        maxv    (float):     ending amplitude value (nm).

    Returns:
        coeff_matrix : np.ndarray, shape (len(modes) * n_steps, max(modes)+1),
                       dtype float64.
                       Each row is one configuration; the column index equals
                       the mode index.
        labels       : list[str] of length len(modes) * n_steps, one human-
                       readable label per configuration, e.g.
                       "mode 2 | 150.0 nm".

    Example:
        # Ramp modes 0, 3, 5 over 10 steps from 0 to 500 nm
        coeff, labels = create_ramp_aberration_configs([0, 3, 5], 10, 0, 500)
        u0_batch = pipeline.generate_batch_modal_coefficients(coeff)
    """
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2 to define a ramp.")
    if len(modes) == 0:
        raise ValueError("modes list must not be empty.")

    amplitudes  = np.linspace(minv, maxv, n_steps)   # shape (n_steps,)
    n_cols      = max(modes) + 1
    n_configs   = len(modes) * n_steps

    coeff_matrix = np.zeros((n_configs, n_cols), dtype=np.float64)
    labels       = []

    for row, (mode_idx, amp) in enumerate(
            (m, a) for m in modes for a in amplitudes):
        coeff_matrix[row, mode_idx] = amp
        labels.append(f"mode {mode_idx} | {amp:.1f} nm")

    return coeff_matrix, labels


def calibrate_subpixel_centers(total_modes_profile, mesh_final):
    """Finds core centers to sub-pixel accuracy with a dynamic, self-healing peak finder."""
    plot_x_out = np.linspace(mesh_final.points[:, 0].min(), mesh_final.points[:, 0].max(), 400)
    plot_y_out = np.linspace(mesh_final.points[:, 1].min(), mesh_final.points[:, 1].max(), 400)
    X_plot_out, Y_plot_out = np.meshgrid(plot_x_out, plot_y_out)
    
    dx = plot_x_out[1] - plot_x_out[0]
    dy = plot_y_out[1] - plot_y_out[0]
    
    total_modes_2d = griddata(
        (mesh_final.points[:, 0], mesh_final.points[:, 1]), total_modes_profile, 
        (X_plot_out, Y_plot_out), method='cubic', fill_value=0.0
    )
    
    # --- DYNAMIC SELF-HEALING PEAK FINDER ---
    thresh = 0.1
    peak_rows, peak_cols = [], []
    
    for attempt in range(15):
        is_peak = (total_modes_2d == maximum_filter(total_modes_2d, size=15)) & (total_modes_2d > thresh * np.max(total_modes_2d))
        peak_rows, peak_cols = np.where(is_peak)
        
        if len(peak_rows) == N_SIGNALS:
            print(f"[Peak Success] Target N_SIGNALS cores resolved successfully at threshold {thresh:.3f}.")
            break
        elif len(peak_rows) < N_SIGNALS:
            thresh *= 0.75  # Lower threshold to pick up weaker cores
        else:
            thresh *= 1.25  # Raise threshold to discard noise split-peaks
    else:
        print(f"⚠️ [Warning] Peak finder converged on {len(peak_rows)} peaks instead of {N_SIGNALS}. Using closest configuration.")
    # ----------------------------------------
    
    core_centers = []
    centroid_half_width = 4
    
    for r, c in zip(peak_rows, peak_cols):
        r_min, r_max = max(0, r - centroid_half_width), min(total_modes_2d.shape[0], r + centroid_half_width + 1)
        c_min, c_max = max(0, c - centroid_half_width), min(total_modes_2d.shape[1], c + centroid_half_width + 1)
        
        patch = total_modes_2d[r_min:r_max, c_min:c_max]
        r_indices, c_indices = np.meshgrid(np.arange(r_min, r_max), np.arange(c_min, c_max), indexing='ij')
        
        patch_sum = np.sum(patch)
        if patch_sum > 0:
            sub_pixel_row = np.sum(r_indices * patch) / patch_sum
            sub_pixel_col = np.sum(c_indices * patch) / patch_sum
            cx = plot_x_out[0] + sub_pixel_col * dx
            cy = plot_y_out[0] + sub_pixel_row * dy
        else:
            cx, cy = plot_x_out[c], plot_y_out[r]
            
        core_centers.append((cx, cy))
        
    return sorted(core_centers, key=lambda p: (np.round(p[1], 2), np.round(p[0], 2))), plot_x_out, plot_y_out, dx, dy


def map_evaluated_to_ideal_geometry(core_centers, ideal_centers_list):
    """Calculates optimal rigid alignment (Procrustes SVD) and assigns global indexes."""
    ideal_centers = np.array(ideal_centers_list)
    eval_centers = np.array(core_centers)
    
    ideal_centered = ideal_centers - np.mean(ideal_centers, axis=0)
    eval_centered = eval_centers - np.mean(eval_centers, axis=0)
    
    scale_ideal = np.sqrt(np.mean(np.sum(ideal_centered**2, axis=1)))
    scale_eval = np.sqrt(np.mean(np.sum(eval_centered**2, axis=1)))
    eval_scaled = eval_centered * (scale_ideal / scale_eval)
    
    H = np.dot(eval_scaled.T, ideal_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    eval_aligned = np.dot(eval_scaled, R.T)
    
    diff = eval_aligned[:, np.newaxis, :] - ideal_centered[np.newaxis, :, :]
    cost_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    eval_indices, ideal_permutation = linear_sum_assignment(cost_matrix)
    
    print(f"[Geometric Fit] Estimated physical core pitch: {scale_eval / scale_ideal:.3f} µm")
    print(f"[Geometric Fit] Residual matching RMS error: {np.mean(cost_matrix[eval_indices, ideal_permutation]):.4e}\n")
    
    return ideal_permutation

def collect_subpixel_signals(image_2d, x_min, y_min, dx, dy, centers, n=7):
    """Extracts an n x n subimage centered on fractional continuous coordinates."""
    signals = np.zeros(len(centers))
    half_n = n // 2
    offsets = np.arange(-half_n, half_n + 1)
    
    for idx, (cx, cy) in enumerate(centers):
        f_col = (cx - x_min) / dx
        f_row = (cy - y_min) / dy
        
        sub_cols, sub_rows = np.meshgrid(f_col + offsets, f_row + offsets)
        coords = np.vstack((sub_rows.ravel(), sub_cols.ravel()))
        subimage_flat = map_coordinates(image_2d, coords, order=3, mode='constant', cval=0.0)
        signals[idx] = np.mean(subimage_flat)
        
    return signals

def detect_centers_from_grid(intensity_2d, X_plot, Y_plot, N_SIGNALS=19):
    """Detect centers from a 2D intensity grid."""
    dx = X_plot[0,1] - X_plot[0,0]
    dy = Y_plot[1,0] - Y_plot[0,0]
    x_min, y_min = X_plot[0,0], Y_plot[0,0]
    
    thresh = 0.1
    for attempt in range(15):
        is_peak = (intensity_2d == maximum_filter(intensity_2d, size=15)) & (intensity_2d > thresh * np.max(intensity_2d))
        peak_rows, peak_cols = np.where(is_peak)
        if len(peak_rows) == N_SIGNALS:
            break
        elif len(peak_rows) < N_SIGNALS:
            thresh *= 0.75
        else:
            thresh *= 1.25
    else:
        print(f"Warning: found {len(peak_rows)} peaks, expected {N_SIGNALS}")
    
    centers = []
    half_width = 4
    for r, c in zip(peak_rows, peak_cols):
        r0, r1 = max(0, r-half_width), min(intensity_2d.shape[0], r+half_width+1)
        c0, c1 = max(0, c-half_width), min(intensity_2d.shape[1], c+half_width+1)
        patch = intensity_2d[r0:r1, c0:c1]
        rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing='ij')
        total = np.sum(patch)
        if total > 0:
            sub_r = np.sum(rr * patch) / total
            sub_c = np.sum(cc * patch) / total
            cx = x_min + sub_c * dx
            cy = y_min + sub_r * dy
        else:
            cx = x_min + c * dx
            cy = y_min + r * dy
        centers.append((cx, cy))
    return sorted(centers, key=lambda p: (np.round(p[1],2), np.round(p[0],2)))

if __name__ == "__main__":
    results = main_batch_propagation()

    

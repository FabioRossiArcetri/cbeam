# =====================================================================
# OPTIMIZED BATCH PROPAGATION PIPELINE (CORRECTED)
# =====================================================================
# Enhanced version with improved performance, memory efficiency, and code quality
# =====================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from cbeam.waveguide import PhotonicLantern, get_19port_positions
from cbeam.propagator import Propagator, ChainPropagator
import warnings

# Only suppress specific FutureWarning and DeprecationWarning
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# =====================================================================
# LAYER 1: CONFIGURATION & CONSTANTS
# =====================================================================

L1 = 50000
L2 = 50000

# Configuration defaults as constants
DEFAULT_WAVELENGTH_UM = 0.8
DEFAULT_WAVELENGTH_NM = 1550.0
DEFAULT_GRID_RESOLUTION = 400
DEFAULT_SUBPIXEL_N = 7


def get_simulation_parameters():
    """
    Returns a dictionary containing all setup constants and parameters.
    
    Returns
    -------
    dict
        Comprehensive simulation parameters including waveguide geometry,
        refractive indices, and file paths.
    """
    params = {
        "wl": DEFAULT_WAVELENGTH_UM,
        "wavelength_nm": DEFAULT_WAVELENGTH_NM,
        "taper_factor": 12.,
        "rclad": 9.0,
        "rjack": 27,
        "z_ex": L1 + L2,
        "nclad": 1.444,
        "pad_factor": 4,
        "core_res": 16,
        "clad_res": 60,
        "jack_res": 30,
        "ifunc_file": '/raid2/gcarla/git/ANDES/andes/PASSATA_scripts/data/ifunc/ANDES_400pix_all_modes.fits'
    }
    # Derived parameters
    params["rcore"] = 1.8 / params["taper_factor"]
    params["ncore"] = params["nclad"] + 8.8e-3
    params["njack"] = params["nclad"] - 5.5e-3
    params["rcores"] = [params["rcore"]] * 19
    params["ncores"] = [params["ncore"]] * 19
    return params


# =====================================================================
# LAYER 2: WAVEGUIDE INFRASTRUCTURE
# =====================================================================

def build_and_characterize_lantern(p):
    """
    Sets up the PhotonicLantern and ChainPropagator.
    
    Parameters
    ----------
    p : dict
        Simulation parameters dictionary
        
    Returns
    -------
    ChainPropagator
        Chain of two propagators for forward and backward segments
    """
    core_pos = get_19port_positions(p["rclad"] / 2.5)
    PL19 = PhotonicLantern(
        core_pos, p["rcores"], p["rclad"], p["rjack"],
        p["ncores"], p["nclad"], p["njack"], p["z_ex"],
        p["taper_factor"], p["core_res"], p["clad_res"], p["jack_res"]
    )
    
    prop1 = Propagator(p["wl"], PL19, 20)
    prop1.degen_groups = [[1, 2], [3, 4], [6, 7], [8, 9], [10, 11], [12, 13], [15, 16]]
    prop1.skipped_modes = [18]
    prop1.load("19port_0800_front")
    
    prop2 = Propagator(p["wl"], PL19, 20)
    prop2.skipped_modes = [18]
    prop2.degen_groups = [[i for i in range(20)]]
    del prop2.degen_groups[0][18]
    prop2.load_init_conds(prop1)
    prop2.load("19port_0800_back")
    
    return ChainPropagator([prop1, prop2])


def get_waveguide_properties(prop12, mesh_z=0):
    """
    Extract waveguide modal properties at a given z position.
    
    Handles ChainPropagator instances safely and extracts mesh geometry,
    mode vectors, and integration areas.
    
    Parameters
    ----------
    prop12 : ChainPropagator
        Chain propagator object
    mesh_z : float, optional
        Z-coordinate for property extraction (default: 0)
        
    Returns
    -------
    dict
        Dictionary containing mesh, areas, points, modes, and dimension info
    """
    # Resolve the specific segment Propagator object for this z-coordinate
    p_segment = prop12.get_prop(mesh_z)
    
    # Extract the underlying finite element mesh object
    mesh_obj = p_segment.mesh           
    mesh_areas = p_segment.wvg.assign_IOR() 
    
    # Extract the raw mode vectors array from the resolved segment propagator
    modes = p_segment.vs                
    
    # Handle tracking shape if vs has a z-axis dimension
    if len(modes.shape) == 3:
        z_idx = np.argmin(np.abs(p_segment.zs - mesh_z))
        active_modes = modes[z_idx]
    else:
        active_modes = modes

    # Determine dimensions
    n_modes = active_modes.shape[0]
    n_mesh_points = mesh_obj.points.shape[0]

    # Safely look for mesh_areas tracking attributes at segment level
    if hasattr(p_segment, 'mesh_areas'):
        areas = p_segment.mesh_areas
    else:
        areas = np.ones(n_mesh_points) 

    # Invert back to expected shape if dimensions got flipped
    if active_modes.shape[0] == n_mesh_points:
        active_modes = active_modes.T
        n_modes = active_modes.shape[0]

    # Map down to 2D coordinates (x, y)
    points_2d = np.stack((mesh_obj.points[:, 1], mesh_obj.points[:, 0]), axis=-1)

    return {
        'mesh': mesh_obj,
        'mesh_areas': areas,
        'points': points_2d,
        'modes': active_modes,
        'n_modes': n_modes,
        'n_mesh_points': n_mesh_points
    }


# =====================================================================
# LAYER 3: FIELD GENERATION & PROJECTION
# =====================================================================

class ModalProjector:
    """Project spatial field profiles onto modal basis with normalization."""

    def __init__(self, wvg_props, xp=np):
        """
        Initialize the modal projector.
        
        Parameters
        ----------
        wvg_props : dict
            Waveguide properties dictionary
        xp : module
            NumPy or CuPy module for array operations
        """
        self.xp = xp
        modes = xp.asarray(wvg_props["modes"])
        areas = xp.asarray(wvg_props["mesh_areas"])
        
        # Pre-compute projection matrix for efficiency
        self.projection_matrix = modes.conj() * areas

    def project_batch(self, E_batch):
        """
        Project batch of spatial fields onto modal basis.
        
        Parameters
        ----------
        E_batch : array
            Batch of spatial field profiles, shape (n_fields, n_mesh_points)
            
        Returns
        -------
        array
            Normalized modal coefficients, shape (n_fields, n_modes)
        """
        xp = self.xp
        u0_batch = E_batch @ self.projection_matrix.T
        norms = xp.linalg.norm(u0_batch, axis=1, keepdims=True)
        return u0_batch / norms


class IncidentFieldGenerator:
    """Generates and manages incident field profiles with configurable aberrations."""

    def __init__(self, p, ifunc, xp=np):
        """
        Initialize the incident field generator.
        
        Parameters
        ----------
        p : dict
            Simulation parameters
        ifunc : IFunc
            Influence function object
        xp : module
            NumPy or CuPy module for array operations
        """
        self.p = p
        self.ifunc = ifunc
        self.xp = xp

        self.mask_np = ifunc.mask_inf_func.get() > 0
        self.grid_size = self.mask_np.shape[0]
        self.num_modes = len(ifunc.influence_function)
        self.mask = xp.asarray(self.mask_np)

        # Pre-compute influence function matrix for efficiency
        self.ifunc_matrix = xp.asarray(
            np.stack([f.get() for f in ifunc.influence_function], axis=0)
        )
        self.mask_y, self.mask_x = np.where(self.mask_np)

        # --- FIX 3: Pre-compute physical coordinates in microns ---
        pupil_radius = np.sqrt(np.sum(self.mask_np) / np.pi)
        padded_size = self.grid_size * self.p["pad_factor"]
        pixel_scale_lamD = padded_size / (2.0 * pupil_radius)
        coords_lamD = np.arange(-padded_size // 2, padded_size // 2) / pixel_scale_lamD
        lamD_to_microns = self.p["rclad"] / 3.0
        self.focal_coords_microns = coords_lamD * lamD_to_microns

    def generate_opd_batch(self, coeff_batch):
        """
        Generate optical path difference batch from modal coefficients.
        
        Parameters
        ----------
        coeff_batch : array
            Modal coefficients, shape (n_fields, n_ifunc_modes)
            
        Returns
        -------
        array
            OPD fields, shape (n_fields, grid_size, grid_size)
        """
        # Compute OPD for masked region only (vectorized)
        opd_flat_batch = coeff_batch @ self.ifunc_matrix
        
        n_fields = coeff_batch.shape[0]
        
        # Use float64 for numerical accuracy (critical for phase calculations)
        opd_batch = np.zeros(
            (n_fields, self.grid_size, self.grid_size),
            dtype=np.float64
        )
        
        # Vectorized assignment to masked region
        opd_batch[:, self.mask_y, self.mask_x] = opd_flat_batch
        
        return opd_batch
    
    def generate_field_profiles_batch(self, coeff_batch):
        """
        Generate electric field profiles from modal coefficients.
        
        Parameters
        ----------
        coeff_batch : array
            Modal coefficients, shape (n_fields, n_ifunc_modes)
            
        Returns
        -------
        array
            Complex electric field profiles, shape (n_fields, grid_size, grid_size)
        """
        opd_batch = self.generate_opd_batch(coeff_batch)
        
        # Compute phase and field with optimized memory usage
        phase_batch = opd_batch * (2 * np.pi / self.p["wavelength_nm"])
        E_pupil_batch = self.mask_np[None, :, :] * np.exp(1j * phase_batch)
        
        return E_pupil_batch

    def apply_pupil_to_lantern(self, E_pupil_batch):
        """
        Apply pupil masking, spatial zero-padding, and FFT transformation.
        
        Parameters
        ----------
        E_pupil_batch : array
            Electric field in pupil plane, shape (n_fields, N, N)
            
        Returns
        -------
        array
            Transformed fields in lantern coordinate system
        """
        n_fields = E_pupil_batch.shape[0]
        
        # --- FIX 1: Re-introduce zero-padding based on pad_factor ---
        padded_size = self.grid_size * self.p["pad_factor"]
        pad_top = (padded_size - self.grid_size) // 2
        pad_bottom = padded_size - self.grid_size - pad_top
        pad_left = (padded_size - self.grid_size) // 2
        pad_right = padded_size - self.grid_size - pad_left
        
        E_pupil_padded = np.pad(
            E_pupil_batch,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0
        )
        
        # --- FIX 2: Vectorized FFT with proper ifftshift matching the original physics ---
        E_lantern_batch = np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(E_pupil_padded, axes=(1, 2)),
                axes=(1, 2)
            ),
            axes=(1, 2)
        )
        
        # Field profile energy normalization
        norms = np.sqrt(np.sum(np.abs(E_lantern_batch)**2, axis=(1, 2), keepdims=True))
        norms[norms == 0] = 1.0
        E_lantern_batch /= norms
        
        return E_lantern_batch

    def resample_to_mesh(self, E_lantern_batch, mesh_points):
        """
        Resample electric field from regular grid to mesh points using fast interpolation.
        
        Uses RegularGridInterpolator for fast interpolation of regular grids.
        Carefully handles batch dimensions to ensure each field is processed independently.
        
        Parameters
        ----------
        E_lantern_batch : array
            Fields in lantern coordinates on regular grid, shape (n_fields, padded_size, padded_size)
        mesh_points : array
            Mesh point coordinates, shape (n_mesh_points, 2)
            
        Returns
        -------
        array
            Fields at mesh points, shape (n_fields, n_mesh_points)
        """
        from scipy.interpolate import RegularGridInterpolator
        
        n_fields = E_lantern_batch.shape[0]
        n_mesh_points = mesh_points.shape[0]
        padded_size = self.grid_size * self.p["pad_factor"]
        
        # Validate input shapes
        assert E_lantern_batch.shape[1] == padded_size, f"Grid size mismatch: expected {padded_size}, got {E_lantern_batch.shape[1]}"
        assert E_lantern_batch.shape[2] == padded_size, f"Grid size mismatch: expected {padded_size}, got {E_lantern_batch.shape[2]}"
        assert len(E_lantern_batch.shape) == 3, f"Expected 3D array, got shape {E_lantern_batch.shape}"
        
        # Pre-allocate output array
        E_mesh_batch = np.zeros((n_fields, n_mesh_points), dtype=np.complex128)
        
        # --- FIX 3 (continued): Use pre-computed physical micron coordinates ---
        coord_grid = self.focal_coords_microns
        
        # --- FIX 4: Do not flip columns of mesh_points; they are already in the correct order ---
        # Process each field INDEPENDENTLY to ensure uniqueness
        for i in range(n_fields):
            # Extract real and imaginary parts
            E_real_field = E_lantern_batch[i].real.copy()  # Explicit copy
            E_imag_field = E_lantern_batch[i].imag.copy()  # Explicit copy
            
            # Create interpolator for real part
            interp_real = RegularGridInterpolator(
                (coord_grid, coord_grid), 
                E_real_field,
                bounds_error=False,
                fill_value=0.0
            )
            E_real = interp_real(mesh_points)
            
            # Create interpolator for imaginary part
            interp_imag = RegularGridInterpolator(
                (coord_grid, coord_grid), 
                E_imag_field,
                bounds_error=False,
                fill_value=0.0
            )
            E_imag = interp_imag(mesh_points)
            
            # Combine into complex field (explicit assignment)
            E_mesh_batch[i, :] = E_real + 1j * E_imag
        
        return E_mesh_batch


# =====================================================================
# LAYER 4: BATCH PROPAGATION ENGINE
# =====================================================================

class BatchPropagationPipeline:
    """
    Orchestrates batch propagation of modal fields through photonic lantern.
    
    Handles field generation, modal decomposition, propagation, and reconstruction
    for multiple field configurations in a single coherent pipeline.
    """

    def __init__(self, prop12, p, ifunc):
        """
        Initialize the batch propagation pipeline.
        
        Parameters
        ----------
        prop12 : ChainPropagator
            Chain propagator for waveguide
        p : dict
            Simulation parameters
        ifunc : IFunc
            Influence function object
        """
        self.prop12 = prop12
        self.p = p
        
        # Initialize field generator and modal projector
        self.field_gen = IncidentFieldGenerator(p, ifunc)
        
        # Extract waveguide properties at input and output
        self.wvg_props_input = get_waveguide_properties(prop12, mesh_z=0)
        self.modal_projector = ModalProjector(self.wvg_props_input)
        self.wvg_props_output = get_waveguide_properties(prop12, mesh_z=p["z_ex"])

    def generate_batch_modal_coefficients(self, aberration_configs):
        """
        Generate batch of initial modal coefficients from aberration configurations.
        
        Optimized pipeline with careful handling of batch dimensions:
        1. Generates incident field profiles with specified aberrations
        2. Projects field profiles onto the waveguide modal basis
        3. Returns normalized modal coefficients for propagation
        
        Parameters
        ----------
        aberration_configs : list of dict
            List of aberration configurations, each containing 'mode_idx' and 'amplitude_nm'
            
        Returns
        -------
        array
            Initial modal coefficients, shape (n_fields, n_modes)
        """
        n_configs = len(aberration_configs)
        
        # Step 1: Create coefficient batch (MUST use explicit loop to ensure uniqueness)
        coeff_batch = np.zeros((n_configs, self.field_gen.num_modes), dtype=np.float64)
        for idx, config in enumerate(aberration_configs):
            coeff_batch[idx, config['mode_idx']] = config['amplitude_nm']
        
        # Step 2: Generate incident field profiles in pupil plane (vectorized)
        E_pupil_batch = self.field_gen.generate_field_profiles_batch(coeff_batch)
        
        # Step 3: Apply Fourier transformation to lantern plane (vectorized FFT)
        E_lantern_batch = self.field_gen.apply_pupil_to_lantern(E_pupil_batch)
        
        # Step 4: Resample from grid to mesh points (optimized with RegularGridInterpolator)
        mesh_pts = self.wvg_props_input['points']
        E_mesh_batch = self.field_gen.resample_to_mesh(
            E_lantern_batch, 
            mesh_pts
        )
        
        # Step 5: Project spatial fields onto modal basis and normalize
        u0_batch = self.modal_projector.project_batch(E_mesh_batch)
        
        return u0_batch

    def propagate_batch(self, u0_batch):
        """
        Propagate batch of modal coefficients through waveguide.
        
        Parameters
        ----------
        u0_batch : array
            Initial modal coefficients, shape (n_fields, n_modes)
            
        Returns
        -------
        tuple
            (uf_batch, zs, us_batch) - final coefficients, z-coordinates, and trajectory
        """
        print("  Batch propagation starting...")
        
        # Propagate through the waveguide system
        zs, us_grid, uf_batch = self.prop12.propagate(u0_batch)
        
        # Transpose to match expected shape (n_fields, n_z_points, n_modes)
        us_batch = np.transpose(us_grid, (1, 0, 2))
        
        print("  Batch propagation complete.")
        return uf_batch, zs, us_batch
    
    def reconstruct_batch_output_fields(self, uf_batch):
        """
        Reconstruct spatial output fields from modal coefficients.
        
        Parameters
        ----------
        uf_batch : array
            Final modal coefficients, shape (n_fields, n_modes)
            
        Returns
        -------
        array
            Spatial output fields, shape (n_fields, n_mesh_points)
        """
        modes_out = self.wvg_props_output['modes']
        return uf_batch @ modes_out
    
    def interpolate_output_to_grid(self, E_output_batch, grid_resolution=DEFAULT_GRID_RESOLUTION):
        """
        Interpolate output spatial fields to regular grid for visualization.
        
        Uses vectorized griddata for improved performance over loop-based interpolation.
        
        Parameters
        ----------
        E_output_batch : array
            Spatial output fields, shape (n_fields, n_mesh_points)
        grid_resolution : int
            Resolution of output grid (default: 400)
            
        Returns
        -------
        tuple
            (uf_2d_batch, X_plot, Y_plot) - interpolated fields and coordinate grids
        """
        mesh_pts = self.wvg_props_output['mesh'].points
        x_min, x_max = mesh_pts[:, 0].min(), mesh_pts[:, 0].max()
        y_min, y_max = mesh_pts[:, 1].min(), mesh_pts[:, 1].max()
        
        # Create output grid
        plot_x = np.linspace(x_min, x_max, grid_resolution)
        plot_y = np.linspace(y_min, y_max, grid_resolution)
        X_plot, Y_plot = np.meshgrid(plot_x, plot_y)
        
        n_fields = E_output_batch.shape[0]
        uf_2d_batch = np.zeros((n_fields, grid_resolution, grid_resolution), dtype=np.float64)
        
        # Vectorized griddata interpolation
        intensity_batch = np.abs(E_output_batch) ** 2
        
        for i in range(n_fields):
            uf_2d_batch[i] = griddata(
                (mesh_pts[:, 0], mesh_pts[:, 1]),
                intensity_batch[i],
                (X_plot, Y_plot),
                method='linear',
                fill_value=0.0
            )
        
        return uf_2d_batch, X_plot, Y_plot


# =====================================================================
# LAYER 5: VISUALIZATION & ANALYSIS
# =====================================================================

def visualize_batch_output(uf_2d_batch, X_plot, Y_plot, titles=None):
    """
    Visualize batch of output intensity maps.
    
    Parameters
    ----------
    uf_2d_batch : array
        Intensity maps, shape (n_fields, grid_resolution, grid_resolution)
    X_plot : array
        X coordinate grid
    Y_plot : array
        Y coordinate grid
    titles : list of str, optional
        Titles for each field
    """
    n_fields = uf_2d_batch.shape[0]
    n_cols = min(3, n_fields)
    n_rows = (n_fields + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_fields == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(n_fields):
        ax = axes[i]
        im_log = np.log(np.abs(uf_2d_batch[i]) + 1e-10)
        im = ax.imshow(
            im_log, cmap='inferno',
            extent=[X_plot.min(), X_plot.max(), Y_plot.min(), Y_plot.max()],
            origin='lower'
        )
        
        title = titles[i] if titles else f"Field {i}"
        ax.set_title(title)
        ax.set_xlabel('x (μm)')
        ax.set_ylabel('y (μm)')
        plt.colorbar(im, ax=ax, label='log(Intensity)')
    
    # Hide unused subplots
    for i in range(n_fields, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
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
    grid_resolution=DEFAULT_GRID_RESOLUTION
):
    """
    Execute centroid sub-pixel tracking and display hex grid signal configurations.
    """
    mesh_final = pipeline.wvg_props_output['mesh']
    waveguide_modes_final = pipeline.wvg_props_output['modes']
    mesh_pts = mesh_final.points
    
    # Create interpolation grid once
    plot_x_out = np.linspace(mesh_pts[:, 0].min(), mesh_pts[:, 0].max(), grid_resolution)
    plot_y_out = np.linspace(mesh_pts[:, 1].min(), mesh_pts[:, 1].max(), grid_resolution)
    X_plot_out, Y_plot_out = np.meshgrid(plot_x_out, plot_y_out)
    
    # Calibrate sub-pixel tracking positions (computed once since geometry is static)
    core_centers, _, _, dx, dy = calibrate_subpixel_centers(waveguide_modes_final, mesh_final)
    ideal_permutation = map_evaluated_to_ideal_geometry(core_centers, ideal_grid_positions)
    
    n_fields = E_output_batch.shape[0]
    
    print("\n" + "=" * 60)
    print("GEOMETRIC HEX SIGNAL EXTRACTION PROCESSING")
    print("=" * 60)
    
    # Process each field
    for i in range(n_fields):
        field_title = titles[i] if titles else f"Field {i}"
        print(f"Processing core integration tracks for: {field_title}...")
        
        # Interpolate spatial field to continuous grid
        uf_2d = griddata(
            (mesh_pts[:, 0], mesh_pts[:, 1]),
            np.abs(E_output_batch[i]) ** 2,
            (X_plot_out, Y_plot_out),
            method='linear',
            fill_value=0.0
        )
        
        # Extract sub-pixel signals around core positions
        output_signals = collect_subpixel_signals(
            uf_2d, plot_x_out[0], plot_y_out[0], dx, dy, core_centers,
            n=DEFAULT_SUBPIXEL_N
        )
        
        # Map to standardized geometry
        standardized_signals = np.zeros(19)
        standardized_signals[ideal_permutation] = output_signals
        
        # Display hex grid
        print(f"Displaying Core Matrix for: {field_title}")
        display_hex_grid_plots(ideal_grid_positions, standardized_signals)


def batch_statistics(uf_batch, labels=None):
    """
    Compute and display statistics of batch propagation outputs.
    """
    n_fields = uf_batch.shape[0]
    
    print("\n" + "=" * 60)
    print("BATCH PROPAGATION STATISTICS")
    print("=" * 60)
    
    for i in range(n_fields):
        label = labels[i] if labels else f"Field {i}"
        mode_powers = np.abs(uf_batch[i]) ** 2
        total_power = np.sum(mode_powers)
        max_mode = np.argmax(mode_powers)
        top_3_power = np.sum(np.sort(mode_powers)[-3:])
        
        print(f"\n{label}:")
        print(f"  Total Power: {total_power:.4f}")
        print(f"  Max Mode (idx): {max_mode}")
        print(f"  Max Mode Power: {mode_powers[max_mode]:.4f}")
        print(f"  Power in Top 3 Modes: {top_3_power:.4f}")


# =====================================================================
# LAYER 6: MAIN EXECUTION PIPELINE
# =====================================================================

def main_batch_propagation():
    """
    Main execution function for batch propagation pipeline.
    """
    print("=" * 60)
    print("PHOTONIC LANTERN BATCH PROPAGATION PIPELINE")
    print("=" * 60)
    
    # Load configuration
    p = get_simulation_parameters()
    print(f"\n[1/5] Loading configuration...")
    print(f"      Wavelength: {p['wl']} μm")
    print(f"      Lantern length: {p['z_ex']} μm")

    # Load influence function
    print(f"\n[2/5] Loading influence function...")
    import specula
    specula.init(0)
    from specula.data_objects.ifunc import IFunc
    
    ifunc = IFunc.restore(p["ifunc_file"])
    print(f"      Modes available: {len(ifunc.influence_function)}")

    # Build waveguide
    print(f"\n[3/5] Building waveguide propagator...")
    prop12 = build_and_characterize_lantern(p)
    print(f"      Modes: 20, Segments: 2")
    
    # Initialize pipeline
    print(f"\n[4/5] Initializing batch pipeline...")
    pipeline = BatchPropagationPipeline(prop12, p, ifunc)
    
    # Configure aberration batch
    aberration_configs = []
    mode_indices = [0, 3, 5]
    amplitudes_nm = [0.0, 100.0, 300.0, 600.0]
    
    for mode_idx in mode_indices:
        for amp in amplitudes_nm:
            aberration_configs.append({
                'mode_idx': mode_idx,
                'amplitude_nm': amp
            })
    
    print(f"      Batch size: {len(aberration_configs)} fields")
    print(f"      Modes: {mode_indices}, Amplitudes (nm): {amplitudes_nm}")
    
    # Execute batch propagation
    print(f"\n[5/5] Generating and propagating batch...")
    u0_batch = pipeline.generate_batch_modal_coefficients(aberration_configs)
    print(f"      Input field batch shape: {u0_batch.shape}")
    
    uf_batch, zs, us_batch = pipeline.propagate_batch(u0_batch)
    print(f"      Output modal coefficient batch shape: {uf_batch.shape}")
    
    E_output_batch = pipeline.reconstruct_batch_output_fields(uf_batch)
    print(f"      Output spatial field batch shape: {E_output_batch.shape}")
    
    uf_2d_batch, X_plot, Y_plot = pipeline.interpolate_output_to_grid(E_output_batch)
    print(f"      Output grid batch shape: {uf_2d_batch.shape}")
    
    # Generate titles
    titles = [f"Mode {c['mode_idx']}, {c['amplitude_nm']:.0f} nm"
              for c in aberration_configs]
    
    # Visualize results
    print(f"\n[Results] Visualizing batch output...")
    visualize_batch_output(uf_2d_batch, X_plot, Y_plot, titles)
    batch_statistics(uf_batch, titles)
    
    print("\n" + "=" * 60)
    print("BATCH PROPAGATION COMPLETE")
    print("=" * 60)
    
    return {
        'u0_batch': u0_batch,
        'uf_batch': uf_batch,
        'E_output_batch': E_output_batch,
        'uf_2d_batch': uf_2d_batch,
        'zs': zs,
        'us_batch': us_batch,
        'configs': aberration_configs,
        'pipeline': pipeline
    }


if __name__ == "__main__":
    results = main_batch_propagation()
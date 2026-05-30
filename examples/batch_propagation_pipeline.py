# =====================================================================
# RESTRUCTURED BATCH PROPAGATION PIPELINE
# =====================================================================
# This version enables efficient propagation of a cube of input fields
# generated from an array of modal coefficients in the spectral modal basis.
# =====================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, griddata
from cbeam.waveguide import PhotonicLantern, get_19port_positions
from cbeam.propagator import Propagator, ChainPropagator
import warnings

warnings.filterwarnings('ignore')

# =====================================================================
# LAYER 1: CONFIGURATION & CONSTANTS
# =====================================================================

L1 = 50000
L2 = 50000

def get_simulation_parameters():
    """Returns a dictionary containing all setup constants and parameters."""
    params = {
        "wl": 0.8,
        "wavelength_nm": 1550.0,
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
    """Sets up the PhotonicLantern and ChainPropagator."""
    core_pos = get_19port_positions(p["rclad"] / 2.5)
    PL19 = PhotonicLantern(
        core_pos, p["rcores"], p["rclad"], p["rjack"],
        p["ncores"], p["nclad"], p["njack"], p["z_ex"],
        p["taper_factor"], p["core_res"], p["clad_res"], p["jack_res"]
    )
    
    prop1 = Propagator(p["wl"], PL19, 20)
    prop1.degen_groups = [[1,2], [3,4], [6,7], [8,9], [10,11], [12,13], [15,16]]
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
    Extract waveguide modal properties at a given z position 
    handling ChainPropagator instances safely.
    """
    # 1. Resolve the specific segment Propagator object for this z-coordinate
    p_segment = prop12.get_prop(mesh_z)
    
    # 2. Extract the underlying finite element mesh object
    mesh_obj = p_segment.mesh           
    mesh_areas = p_segment.wvg.assign_IOR() 
    
    # 3. Extract the raw mode vectors array (vs) from the resolved segment propagator
    modes = p_segment.vs                
    
    # Handle tracking shape if vs has a z-axis dimension (e.g., shape: [n_z, n_modes, n_points])
    if len(modes.shape) == 3:
        z_idx = np.argmin(np.abs(p_segment.zs - mesh_z))
        active_modes = modes[z_idx]
    else:
        active_modes = modes

    # Correct dimensions
    n_modes = active_modes.shape[0]          # Row count is number of physical modes (20)
    n_mesh_points = mesh_obj.points.shape[0]   # Total structural coordinate points (12559)

    # Safely look for mesh_areas tracking attributes at segment level
    if hasattr(p_segment, 'mesh_areas'):
        areas = p_segment.mesh_areas
    else:
        areas = np.ones(n_mesh_points) 

    # Invert back to expected shape (n_modes, n_points) if dimensions got flipped
    if active_modes.shape[0] == n_mesh_points:
        active_modes = active_modes.T
        n_modes = active_modes.shape[0]

    # Map down to 2D coordinates (x, y)
    points_2d = np.stack((mesh_obj.points[:, 1], mesh_obj.points[:, 0]), axis=-1)

    return {
        'mesh': mesh_obj,                   # <-- RE-ADDED THIS LINE TO FIX THE KEYERROR
        'mesh_areas': areas,
        'points': points_2d,
        'modes': active_modes,
        'n_modes': n_modes,                 # Cleanly evaluates to 20
        'n_mesh_points': n_mesh_points       # Evaluates to 12559
    }

# =====================================================================
# LAYER 3: FIELD GENERATION & PROJECTION
# =====================================================================

class IncidentFieldGenerator:
    """Generates and manages incident field profiles with configurable aberrations."""
    
    def __init__(self, p, ifunc):
        """Initialize with simulation parameters and influence function."""
        self.p = p
        self.ifunc = ifunc
        self.mask_np = self.ifunc.mask_inf_func.get() > 0
        self.grid_size = self.mask_np.shape[0]
        self.num_modes = len(self.ifunc.influence_function)
        
    def generate_opd_screen(self, mode_coefficients):
        """Generate optical path difference screen from modal coefficients."""
        opd_flat = np.zeros_like(self.ifunc.influence_function[0].get(), dtype=np.float64)
        for idx, coeff in enumerate(mode_coefficients):
            if coeff != 0.0:
                opd_flat += coeff * self.ifunc.influence_function[idx].get()
        
        opd_2d = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        opd_2d[self.mask_np] = opd_flat
        
        return opd_2d
    
    def generate_field_profile(self, mode_coefficients):
        """Generate incident field profile with phase modulation."""
        from scipy.fft import fft2, fftshift, ifftshift
        
        opd_2d = self.generate_opd_screen(mode_coefficients)
        
        # OPD → Phase
        phase_2d_rad = opd_2d * (2.0 * np.pi / self.p["wavelength_nm"])
        
        # Propagate to focal plane
        E_pupil = self.mask_np * np.exp(1j * phase_2d_rad)
        padded_size = self.grid_size * self.p["pad_factor"]
        
        pad_top = (padded_size - self.grid_size) // 2
        pad_bottom = padded_size - self.grid_size - pad_top
        pad_left = (padded_size - self.grid_size) // 2
        pad_right = padded_size - self.grid_size - pad_left
        
        E_pupil_padded = np.pad(E_pupil, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                               mode='constant', constant_values=0)
        
        E_focal_raw = fftshift(fft2(ifftshift(E_pupil_padded)))
        E_focal_raw /= np.sqrt(np.sum(np.abs(E_focal_raw)**2))
        
        return E_focal_raw
    
    def create_interpolator(self, mode_coefficients, mesh_z_properties):
        """Create spatial interpolator for incident field."""
        E_focal = self.generate_field_profile(mode_coefficients)
        
        pupil_radius = np.sqrt(np.sum(self.mask_np) / np.pi)
        pixel_scale_lamD = (self.grid_size * self.p["pad_factor"]) / (2.0 * pupil_radius)
        coords_lamD = np.arange(-self.grid_size * self.p["pad_factor"] // 2, 
                                self.grid_size * self.p["pad_factor"] // 2) / pixel_scale_lamD
        
        lamD_to_microns = self.p["rclad"] / 3.0
        focal_coords_microns = coords_lamD * lamD_to_microns
        
        return RegularGridInterpolator(
            (focal_coords_microns, focal_coords_microns),
            E_focal,
            bounds_error=False,
            fill_value=0.0
        )

def project_field_to_modal_basis(E_spatial, wvg_properties):
    """Project spatial field onto waveguide modal basis."""
    modes = wvg_properties['modes']       # Shape is (n_modes, n_mesh_points)
    mesh_areas = wvg_properties['mesh_areas']
    
    # REMOVED .T: Matrix multiply directly
    u0 = modes.conj() @ (E_spatial * mesh_areas)
    return u0 / np.linalg.norm(u0)


# =====================================================================
# LAYER 4: BATCH FIELD GENERATION & PROPAGATION
# =====================================================================

class BatchPropagationPipeline:
    """Orchestrates batch propagation of multiple incident fields."""
    
    def __init__(self, prop12, p, ifunc):
        self.prop12 = prop12
        self.p = p
        self.field_gen = IncidentFieldGenerator(p, ifunc)
        self.wvg_props_input = get_waveguide_properties(prop12, mesh_z=0)
        self.wvg_props_output = get_waveguide_properties(prop12, mesh_z=prop12.zs[-1])
    
    def generate_batch_modal_coefficients(self, aberration_modes_list):
        """Generate a batch of modal coefficients from list of aberration configurations."""
        n_fields = len(aberration_modes_list)
        
        # FIX THIS LINE: Ensure the second dimension is n_modes (20), NOT mesh points (12559)
        n_modes = self.wvg_props_input['n_modes'] 
        u0_batch = np.zeros((n_fields, n_modes), dtype=np.complex128)
        
        for i, config in enumerate(aberration_modes_list):
            # Create mode coefficient array for this aberration
            mode_coeffs = np.zeros(self.field_gen.num_modes)
            mode_coeffs[config['mode_idx']] = config['amplitude_nm']
            
            # Generate incident field
            interp = self.field_gen.create_interpolator(mode_coeffs, self.wvg_props_input)
            E_input = interp(self.wvg_props_input['points']).astype(np.complex128)
            
            # Project to modal basis (returns shape (20,))
            u0_batch[i] = project_field_to_modal_basis(E_input, self.wvg_props_input)
        
        return u0_batch
    
    def propagate_batch(self, u0_batch):
        """Propagate batch of modal coefficients through the lantern simultaneously."""
        print(f"  Propagating batch of {u0_batch.shape[0]} fields simultaneously...", flush=True)
        
        # Invoke the native batch-ready propagate call
        zs, us_grid, uf_batch = self.prop12.propagate(u0_batch)
        
        # us_grid is shaped (n_z_points, n_fields, n_modes) from the ODE system
        # Transpose it to (n_fields, n_z_points, n_modes) to match expected downstream pipeline behavior
        us_batch = np.transpose(us_grid, (1, 0, 2))
        
        print("  Batch propagation complete.")
        return uf_batch, zs, us_batch
    
    def reconstruct_batch_output_fields(self, uf_batch):
        """Reconstruct spatial output fields from modal coefficients cleanly via matrix mult."""
        modes_out = self.wvg_props_output['modes']  # Shape is (n_modes, n_mesh_points)
        
        return uf_batch @ modes_out
    
    def interpolate_output_to_grid(self, E_output_batch, grid_resolution=400):
        """Interpolate output spatial fields to regular grid for visualization."""
        mesh_pts = self.wvg_props_output['mesh'].points
        x_min, x_max = mesh_pts[:, 0].min(), mesh_pts[:, 0].max()
        y_min, y_max = mesh_pts[:, 1].min(), mesh_pts[:, 1].max()
        
        plot_x = np.linspace(x_min, x_max, grid_resolution)
        plot_y = np.linspace(y_min, y_max, grid_resolution)
        X_plot, Y_plot = np.meshgrid(plot_x, plot_y)
        
        n_fields = E_output_batch.shape[0]
        uf_2d_batch = np.zeros((n_fields, grid_resolution, grid_resolution))
        
        for i in range(n_fields):
            uf_2d_batch[i] = griddata(
                (mesh_pts[:, 0], mesh_pts[:, 1]),
                np.abs(E_output_batch[i])**2,
                (X_plot, Y_plot),
                method='linear',
                fill_value=0.0
            )
        
        return uf_2d_batch, X_plot, Y_plot


# =====================================================================
# LAYER 5: VISUALIZATION & ANALYSIS
# =====================================================================

def visualize_batch_output(uf_2d_batch, X_plot, Y_plot, titles=None):
    """Visualize batch of output intensity maps."""
    n_fields = uf_2d_batch.shape[0]
    n_cols = min(3, n_fields)
    n_rows = (n_fields + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_fields == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(n_fields):
        ax = axes[i]
        im_log = np.log(np.abs(uf_2d_batch[i]) + 1e-10)
        im = ax.imshow(im_log, cmap='inferno', 
                      extent=[X_plot.min(), X_plot.max(), Y_plot.min(), Y_plot.max()],
                      origin='lower')
        
        title = titles[i] if titles else f"Field {i}"
        ax.set_title(title)
        ax.set_xlabel('x (μm)')
        ax.set_ylabel('y (μm)')
        plt.colorbar(im, ax=ax, label='log(Intensity)')
    
    for i in range(n_fields, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def visualize_batch_hex_grid_signals(pipeline, E_output_batch, ideal_grid_positions, calibrate_subpixel_centers, collect_subpixel_signals, map_evaluated_to_ideal_geometry, display_hex_grid_plots, titles=None, grid_resolution=400):
    """
    Executes centroid sub-pixel tracking, extracts core signal integrations, 
    and displays the resulting geometric hex matrix configurations for the entire batch.
    """
    mesh_final = pipeline.wvg_props_output['mesh']
    waveguide_modes_final = pipeline.wvg_props_output['modes']
    mesh_pts = mesh_final.points
    
    # 1. Continuous fine mesh interpolation grids
    plot_x_out = np.linspace(mesh_pts[:, 0].min(), mesh_pts[:, 0].max(), grid_resolution)
    plot_y_out = np.linspace(mesh_pts[:, 1].min(), mesh_pts[:, 1].max(), grid_resolution)
    X_plot_out, Y_plot_out = np.meshgrid(plot_x_out, plot_y_out)
    
    # 2. Calibrate sub-pixel tracking positions (only needed once since geometry is static)
    core_centers, _, _, dx, dy = calibrate_subpixel_centers(waveguide_modes_final, mesh_final)
    ideal_permutation = map_evaluated_to_ideal_geometry(core_centers, ideal_grid_positions)
    
    n_fields = E_output_batch.shape[0]
    
    print("\n" + "="*60)
    print("GEOMETRIC HEX SIGNAL EXTRACTION PROCESSING")
    print("="*60)
    
    for i in range(n_fields):
        field_title = titles[i] if titles else f"Field {i}"
        print(f"Processing core integration tracks for: {field_title}...")
        
        # Linearly interpolate this specific spatial field configuration to continuous 2D space
        uf_2d = griddata(
            (mesh_pts[:, 0], mesh_pts[:, 1]), 
            np.abs(E_output_batch[i])**2, 
            (X_plot_out, Y_plot_out), 
            method='linear', 
            fill_value=0.0
        )
        
        # Execute sub-pixel signal capture around the localized core positions
        output_signals = collect_subpixel_signals(uf_2d, plot_x_out[0], plot_y_out[0], dx, dy, core_centers, n=7)
        
        # Map down via geometric hex permutations
        standardized_signals = np.zeros(19)
        standardized_signals[ideal_permutation] = output_signals
        
        # Display the custom core footprint plots
        print(f"Displaying Core Matrix for: {field_title}")
        # Note: assuming display_hex_grid_plots opens/handles its own plt window
        display_hex_grid_plots(ideal_grid_positions, standardized_signals)
        
def batch_statistics(uf_batch, labels=None):
    """Compute and display statistics of batch outputs."""
    n_fields = uf_batch.shape[0]
    
    print("\n" + "="*60)
    print("BATCH PROPAGATION STATISTICS")
    print("="*60)
    
    for i in range(n_fields):
        label = labels[i] if labels else f"Field {i}"
        mode_powers = np.abs(uf_batch[i])**2
        total_power = np.sum(mode_powers)
        max_mode = np.argmax(mode_powers)
        
        print(f"\n{label}:")
        print(f"  Total Power: {total_power:.4f}")
        print(f"  Max Mode (idx): {max_mode}")
        print(f"  Max Mode Power: {mode_powers[max_mode]:.4f}")
        print(f"  Power in Top 3 Modes: {np.sum(np.sort(mode_powers)[-3:]):.4f}")


# =====================================================================
# LAYER 6: MAIN EXECUTION PIPELINE
# =====================================================================

def main_batch_propagation():
    """Main execution function for batch propagation."""
    print("="*60)
    print("PHOTONIC LANTERN BATCH PROPAGATION PIPELINE")
    print("="*60)
    
    p = get_simulation_parameters()
    print(f"\n[1/5] Loading configuration...")
    print(f"      Wavelength: {p['wl']} μm")
    print(f"      Lantern length: {p['z_ex']} μm")

    print(f"\n[2/5] Loading influence function...")
    import specula
    specula.init(0)
    from specula.data_objects.ifunc import IFunc

    # UPDATE THIS LINE HERE AS WELL:
    ifunc = IFunc.restore(p["ifunc_file"])  
    
    print(f"      Modes available: {len(ifunc.influence_function)}")

    print(f"\n[3/5] Building waveguide propagator...")
    prop12 = build_and_characterize_lantern(p)
    print(f"      Modes: 20, Segments: 2")
    
    print(f"\n[4/5] Initializing batch pipeline...")
    pipeline = BatchPropagationPipeline(prop12, p, ifunc)
    
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
    
    print(f"\n[5/5] Generating and propagating batch...")
    u0_batch = pipeline.generate_batch_modal_coefficients(aberration_configs)
    print(f"      Input field batch shape: {u0_batch.shape}")
    
    uf_batch, zs, us_batch = pipeline.propagate_batch(u0_batch)
    print(f"      Output modal coefficient batch shape: {uf_batch.shape}")
    
    E_output_batch = pipeline.reconstruct_batch_output_fields(uf_batch)
    print(f"      Output spatial field batch shape: {E_output_batch.shape}")
    
    uf_2d_batch, X_plot, Y_plot = pipeline.interpolate_output_to_grid(E_output_batch)
    print(f"      Output grid batch shape: {uf_2d_batch.shape}")
    
    titles = [f"Mode {c['mode_idx']}, {c['amplitude_nm']:.0f} nm" 
              for c in aberration_configs]
    
    print(f"\n[Results] Visualizing batch output...")
    visualize_batch_output(uf_2d_batch, X_plot, Y_plot, titles)
    batch_statistics(uf_batch, titles)
    
    print("\n" + "="*60)
    print("BATCH PROPAGATION COMPLETE")
    print("="*60)
    
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
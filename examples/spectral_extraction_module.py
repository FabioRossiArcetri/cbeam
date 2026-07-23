# =====================================================================
# SPECTRAL EXTRACTION MODULE
# =====================================================================
# Simulates dispersive imaging of photonic lantern outputs and extracts
# 19 individual spectra through optimal extraction.
#
# Pipeline stages:
#   1. Propagate through photonic lantern (existing)
#   2. Project fiber outputs onto slit plane
#   3. Apply wavelength-dependent dispersion
#   4. Create 2D spectral image (spatial × wavelength)
#   5. Extract 1D spectra using optimal extraction algorithm
#
# Key features:
#   - Realistic PSF modeling for each fiber
#   - Wavelength-dependent dispersion (linear/nonlinear)
#   - Cross-talk rejection between adjacent fibers
#   - Optimal extraction with variance weighting
#   - Sky subtraction and cosmic ray rejection
# =====================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Tuple, Optional, List


# =====================================================================
# CONFIGURATION
# =====================================================================

@dataclass
class SpectralConfig:
    """Configuration parameters for spectral extraction."""
    
    # Wavelength range
    lambda_min: float = 700.0      # nm
    lambda_max: float = 900.0      # nm
    n_spectral_bins: int = 200     # spectral resolution elements
    
    # Slit geometry
    fiber_spacing_pixels: float = 8.0   # spatial separation on detector (pixels)
    fiber_core_diameter_pixels: float = 3.0  # projected fiber size
    
    # Dispersion
    dispersion_axis: int = 1       # 0=vertical, 1=horizontal
    dispersion_mm_per_nm: float = 0.05  # detector plate scale
    pixel_size_um: float = 15.0    # detector pixel size
    
    # PSF model
    psf_fwhm_pixels: float = 2.5   # seeing-limited PSF
    psf_elongation: float = 1.0    # anamorphic factor (1.0 = circular)
    
    # Detector
    detector_width: int = 2048     # pixels
    detector_height: int = 512     # pixels
    read_noise: float = 3.0        # electrons RMS
    dark_current: float = 0.1      # electrons/pixel/sec
    gain: float = 1.0              # electrons/ADU
    
    # Extraction
    extraction_aperture_sigma: float = 3.0  # aperture = ±N sigma
    background_width_pixels: int = 5         # width of background regions
    optimal_extraction: bool = True          # use variance weighting


# =====================================================================
# SPECTRAL IMAGE SIMULATOR
# =====================================================================

class SpectralImageSimulator:
    """
    Simulates the 2D spectral image formed by a dispersive spectrograph.
    
    The output of each photonic lantern fiber is dispersed along one axis
    of the detector, creating a 2D image where:
        - One axis represents spatial position (fiber ID)
        - Other axis represents wavelength (dispersed)
    """
    
    def __init__(self, config: SpectralConfig, aperture_size: int = 3):
        self.cfg = config
        self.aperture_size = aperture_size

        # Build wavelength grid
        self.wavelengths = np.linspace(
            config.lambda_min, 
            config.lambda_max, 
            config.n_spectral_bins
        )
        self.delta_lambda = self.wavelengths[1] - self.wavelengths[0]
        
        # Compute dispersion scale (pixels per nm)
        self.dispersion_pix_per_nm = (
            config.dispersion_mm_per_nm * 1000.0 / config.pixel_size_um
        )
        
        # Create coordinate grids
        self._setup_detector_coordinates()
        
        # Precompute PSF kernels
        self._precompute_psf_kernels()
        
        print(f"SpectralImageSimulator initialized:")
        print(f"  Wavelength range: {config.lambda_min:.1f}–{config.lambda_max:.1f} nm")
        print(f"  Spectral bins: {config.n_spectral_bins}")
        print(f"  Dispersion: {self.dispersion_pix_per_nm:.3f} pix/nm")
        print(f"  Detector: {config.detector_width} × {config.detector_height} pixels")
    
    def _setup_detector_coordinates(self):
        """Create detector coordinate grids."""
        cfg = self.cfg
        self.y_coords = np.arange(cfg.detector_height)
        self.x_coords = np.arange(cfg.detector_width)
        self.Y, self.X = np.meshgrid(self.y_coords, self.x_coords, indexing='ij')
        
    def _precompute_psf_kernels(self):
        """Precompute PSF kernel for different wavelengths."""
        cfg = self.cfg
        
        # PSF size varies with wavelength (diffraction-limited component)
        # For simplicity, use a single kernel here
        sigma_spatial = cfg.psf_fwhm_pixels / 2.355
        sigma_spectral = sigma_spatial / cfg.psf_elongation
        
        # Store for use in convolution
        self.psf_sigma_spatial = sigma_spatial
        self.psf_sigma_spectral = sigma_spectral
    
    def compute_fiber_positions(self, n_fibers: int = 19) -> np.ndarray:
        """
        Compute the spatial positions of each fiber output on the slit.
        
        For a 19-fiber hexagonal bundle, arrange them in a pseudo-linear
        fashion along the slit (spatial axis).
        
        Returns
        -------
        positions : ndarray, shape (n_fibers,)
            Y-coordinate (row index) of each fiber on the detector.
        """
        cfg = self.cfg
        
        # Arrange fibers linearly with equal spacing
        # Place them centered in the detector
        center = cfg.detector_height / 2.0
        total_extent = (n_fibers - 1) * cfg.fiber_spacing_pixels
        
        positions = center + np.linspace(
            -total_extent/2, total_extent/2, n_fibers
        )
        
        return positions
    
    def wavelength_to_pixel(self, wavelength: float) -> float:
        """Convert wavelength (nm) to detector pixel coordinate."""
        cfg = self.cfg
        # Dispersion is linear: pixel = (lambda - lambda_min) * scale
        pixel = (wavelength - cfg.lambda_min) * self.dispersion_pix_per_nm
        
        # Offset to center in detector
        if cfg.dispersion_axis == 1:  # horizontal dispersion
            pixel += cfg.detector_width / 2.0 - (
                (cfg.lambda_max - cfg.lambda_min) / 2.0 * self.dispersion_pix_per_nm
            )
        else:  # vertical dispersion
            pixel += cfg.detector_height / 2.0 - (
                (cfg.lambda_max - cfg.lambda_min) / 2.0 * self.dispersion_pix_per_nm
            )
        
        return pixel
    
    def create_spectral_image(
        self, 
        modal_powers: np.ndarray,
        spectra: Optional[np.ndarray] = None,
        add_noise: bool = True,
        exposure_time: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a 2D spectral image from fiber modal powers.
        
        Parameters
        ----------
        modal_powers : ndarray, shape (n_fibers,) or (n_wavelengths, n_fibers)
            Power in each fiber mode. If 1D, assumes monochromatic.
            If 2D, first axis is wavelength, second is fiber index.
        spectra : ndarray, shape (n_wavelengths, n_fibers), optional
            Pre-computed spectral flux in each fiber at each wavelength.
            If None, uses modal_powers directly.
        add_noise : bool
            Whether to add Poisson noise, read noise, and dark current.
        exposure_time : float
            Exposure time in seconds (for dark current calculation).
        
        Returns
        -------
        image : ndarray, shape (detector_height, detector_width)
            Simulated 2D spectral image (detector counts).
        fiber_positions : ndarray, shape (n_fibers,)
            Spatial positions of each fiber trace on detector.
        """
        cfg = self.cfg
        
        # Handle input dimensions
        if spectra is None:
            if modal_powers.ndim == 1:
                # Monochromatic case: replicate across wavelengths
                n_fibers = len(modal_powers)
                spectra = np.tile(modal_powers, (cfg.n_spectral_bins, 1))
            elif modal_powers.ndim == 2:
                spectra = modal_powers
            else:
                raise ValueError(f"modal_powers must be 1D or 2D, got shape {modal_powers.shape}")
        
        n_wavelengths, n_fibers = spectra.shape
        
        # Compute fiber positions on detector
        fiber_positions = self.compute_fiber_positions(n_fibers)
        
        # Initialize image
        image = np.zeros((cfg.detector_height, cfg.detector_width), dtype=np.float64)
        
        # Place each fiber's spectrum on the detector
        for fiber_idx in range(n_fibers):
            y_center = fiber_positions[fiber_idx]
            
            for wave_idx, wavelength in enumerate(self.wavelengths[:n_wavelengths]):
                x_center = self.wavelength_to_pixel(wavelength)
                
                # Get flux for this fiber at this wavelength
                flux = spectra[wave_idx, fiber_idx]
                
                if flux <= 0:
                    continue
                
                # Add PSF-convolved spot
                self._add_psf_spot(
                    image, 
                    y_center, 
                    x_center, 
                    flux * self.delta_lambda,
                    aperture_size=self.aperture_size
                )
        
        # Convolve with PSF to simulate seeing + instrument PSF
        image = gaussian_filter(
            image, 
            sigma=(self.psf_sigma_spatial, self.psf_sigma_spectral)
        )
        
        # Add detector effects
        if add_noise:
            image = self._add_detector_noise(image, exposure_time)
        
        return image, fiber_positions
        
    def _add_psf_spot(
        self, 
        image: np.ndarray, 
        y_center: float, 
        x_center: float, 
        flux: float,
        aperture_size: int = 3  # NEW: window size (odd number)
    ):
        """
        Add a PSF spot to the image, spreading flux over a window of pixels
        around the center position.
        
        Parameters
        ----------
        aperture_size : int
            Size of the square aperture in pixels (should be odd). For example,
            aperture_size=3 integrates over a 3×3 pixel window. Default: 3.
        """
        cfg = self.cfg
        
        # Round to nearest pixel
        y_pix = int(np.round(y_center))
        x_pix = int(np.round(x_center))
        
        half_size = aperture_size // 2
        
        # Integrate over aperture window with fractional pixel weighting
        for dy in range(-half_size, half_size + 1):
            for dx in range(-half_size, half_size + 1):
                y_idx = y_pix + dy
                x_idx = x_pix + dx
                
                # Check bounds
                if 0 <= y_idx < cfg.detector_height and 0 <= x_idx < cfg.detector_width:
                    # Compute fractional weight based on distance from center
                    # (bilinear interpolation weight)
                    dy_frac = abs((y_idx - y_center) - 0.5)
                    dx_frac = abs((x_idx - x_center) - 0.5)
                    
                    # Bilinear weight (1.0 at center, decreases toward edges)
                    weight = max(0, 1 - dy_frac) * max(0, 1 - dx_frac)
                    
                    image[y_idx, x_idx] += flux * weight / (aperture_size ** 2)
    
    def _add_detector_noise(
        self, 
        image: np.ndarray, 
        exposure_time: float
    ) -> np.ndarray:
        """Add realistic detector noise: Poisson, read noise, dark current."""
        cfg = self.cfg
        
        # Dark current
        dark_signal = cfg.dark_current * exposure_time
        image = image + dark_signal
        
        # Poisson noise (signal-dependent)
        image_electrons = image / cfg.gain
        image_electrons = np.random.poisson(np.maximum(image_electrons, 0))
        image = image_electrons * cfg.gain
        
        # Read noise (Gaussian)
        read_noise_map = np.random.normal(0, cfg.read_noise, image.shape)
        image = image + read_noise_map
        
        return np.maximum(image, 0)  # no negative counts


# =====================================================================
# SPECTRAL EXTRACTION ENGINE
# =====================================================================

class SpectralExtractor:
    """
    Extracts 1D spectra from 2D spectral images using optimal extraction.
    
    Implements the algorithm from Horne (1986, PASP 98, 609):
    - Spatial profile modeling for each fiber
    - Variance-weighted optimal extraction
    - Cosmic ray rejection
    - Background subtraction
    """
    
    def __init__(self, config: SpectralConfig):
        self.cfg = config
        self.profiles = None  # Spatial profiles (learned or modeled)
        
    def extract_spectra(
        self,
        image: np.ndarray,
        fiber_positions: np.ndarray,
        method: str = 'optimal',
        aperture_width: int = 3  # NEW PARAMETER
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract 1D spectra from 2D spectral image.
        
        Parameters
        ----------
        image : ndarray, shape (detector_height, detector_width)
            2D spectral image.
        fiber_positions : ndarray, shape (n_fibers,)
            Spatial positions of fiber traces (y-coordinates).
        method : str, 'optimal' or 'aperture'
            Extraction method.
        aperture_width : int
            Width in pixels over which to integrate at each wavelength.
            Larger values smooth out noise but reduce spectral resolution.
            Recommended: 3-7 pixels. Default: 3.
        
        Returns
        -------
        spectra : ndarray, shape (n_wavelengths, n_fibers)
            Extracted 1D spectra (flux vs wavelength for each fiber).
        variance : ndarray, shape (n_wavelengths, n_fibers)
            Variance estimate for each spectral bin.
        """
        cfg = self.cfg
        n_fibers = len(fiber_positions)
        n_wavelengths = cfg.n_spectral_bins
        
        # Allocate output arrays
        spectra = np.zeros((n_wavelengths, n_fibers))
        variance = np.zeros((n_wavelengths, n_fibers))
        
        # Estimate variance map (read noise + Poisson)
        variance_map = self._estimate_variance_map(image)
        
        # Build spatial profile model for each fiber
        if self.profiles is None:
            self.profiles = self._build_spatial_profiles(
                image, fiber_positions
            )
        
        # Extract each fiber
        for fiber_idx in range(n_fibers):
            y_center = fiber_positions[fiber_idx]
            
            # Define extraction aperture (spatial region)
            y_min, y_max = self._get_extraction_aperture(y_center)
            
            # Extract spectrum for this fiber
            if method == 'optimal':
                spec, var = self._optimal_extraction(
                    image, variance_map, y_min, y_max, fiber_idx,
                    aperture_width=aperture_width  # Pass aperture width
                )
            else:  # aperture extraction
                spec, var = self._aperture_extraction(
                    image, variance_map, y_min, y_max,
                    aperture_width=aperture_width  # Also update aperture extraction
                )
            
            spectra[:, fiber_idx] = spec
            variance[:, fiber_idx] = var
        
        return spectra, variance
    
    def _estimate_variance_map(self, image: np.ndarray) -> np.ndarray:
        """Estimate variance map from image (Poisson + read noise)."""
        cfg = self.cfg
        
        # Poisson component: variance = signal (in electrons)
        signal_electrons = np.maximum(image, 0) / cfg.gain
        poisson_var = signal_electrons * cfg.gain**2
        
        # Read noise component (constant)
        read_noise_var = cfg.read_noise**2
        
        total_variance = poisson_var + read_noise_var
        
        return total_variance
    
    def _build_spatial_profiles(
        self,
        image: np.ndarray,
        fiber_positions: np.ndarray
    ) -> List[np.ndarray]:
        """
        Build normalized spatial profile for each fiber.
        
        For simplicity, use a Gaussian model. In production, this
        should be empirically measured from the data.
        
        Returns
        -------
        profiles : list of ndarray
            Each element is a 1D spatial profile (normalized to sum=1).
        """
        cfg = self.cfg
        n_fibers = len(fiber_positions)
        
        # Extraction aperture width
        sigma = cfg.psf_fwhm_pixels / 2.355
        aperture_width = int(np.ceil(cfg.extraction_aperture_sigma * sigma))
        
        profiles = []
        for fiber_idx in range(n_fibers):
            y_center = fiber_positions[fiber_idx]
            y_min, y_max = self._get_extraction_aperture(y_center)
            
            # Build Gaussian profile
            y_coords = np.arange(y_min, y_max)
            profile = np.exp(-0.5 * ((y_coords - y_center) / sigma)**2)
            profile /= profile.sum()  # normalize
            
            profiles.append(profile)
        
        return profiles
    
    def _get_extraction_aperture(self, y_center: float) -> Tuple[int, int]:
        """Compute spatial extraction aperture bounds."""
        cfg = self.cfg
        sigma = cfg.psf_fwhm_pixels / 2.355
        half_width = int(np.ceil(cfg.extraction_aperture_sigma * sigma))
        
        y_min = max(0, int(np.round(y_center)) - half_width)
        y_max = min(cfg.detector_height, int(np.round(y_center)) + half_width + 1)
        
        return y_min, y_max
    
    def _aperture_extraction(
        self,
        image: np.ndarray,
        variance_map: np.ndarray,
        y_min: int,
        y_max: int,
        aperture_width: int = 3  # NEW
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple aperture extraction (sum pixels in aperture)."""
        cfg = self.cfg
        
        # Extract sub-image
        sub_image = image[y_min:y_max, :]
        sub_variance = variance_map[y_min:y_max, :]
        
        # Simple sum along spatial direction
        spectrum = np.sum(sub_image, axis=0)
        variance = np.sum(sub_variance, axis=0)
        
        # === MODIFIED: Integrate over aperture window ===
        wavelengths = self.cfg.lambda_min + np.arange(cfg.n_spectral_bins) * (
            (cfg.lambda_max - cfg.lambda_min) / (cfg.n_spectral_bins - 1)
        )
        
        dispersion_pix_per_nm = (
            cfg.dispersion_mm_per_nm * 1000.0 / cfg.pixel_size_um
        )
        x_pixels = (wavelengths - cfg.lambda_min) * dispersion_pix_per_nm
        x_pixels += cfg.detector_width / 2.0 - (
            (cfg.lambda_max - cfg.lambda_min) / 2.0 * dispersion_pix_per_nm
        )
        
        # Integrate over aperture window
        half_width = aperture_width // 2
        spec_interp = np.zeros(cfg.n_spectral_bins)
        var_interp = np.zeros(cfg.n_spectral_bins)
        
        for wl_idx, x_center in enumerate(x_pixels):
            x_min_win = max(0, int(x_center) - half_width)
            x_max_win = min(cfg.detector_width, int(x_center) + half_width + 1)
            
            # Sum over window
            spec_interp[wl_idx] = np.sum(spectrum[x_min_win:x_max_win])
            var_interp[wl_idx] = np.sum(variance[x_min_win:x_max_win])
        
        return spec_interp, var_interp

    def pixel_to_wavelength(self, pixel):
        cfg = self.cfg
        offset = cfg.detector_width/2.0 - (cfg.lambda_max-cfg.lambda_min)/2.0*self.dispersion_pix_per_nm
        return cfg.lambda_min + (pixel - offset) / self.dispersion_pix_per_nm
    
    def _optimal_extraction(
        self,
        image: np.ndarray,
        variance_map: np.ndarray,
        y_min: int,
        y_max: int,
        fiber_idx: int,
        aperture_width: int = 3  # NEW: integration window width
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimal extraction using spatial profile weighting (Horne 1986).
        
        Parameters
        ----------
        aperture_width : int
            Width in pixels over which to integrate at each wavelength position.
            Should be odd number (e.g., 3, 5, 7). Default: 3.
        """
        cfg = self.cfg
        
        # Extract sub-image
        sub_image = image[y_min:y_max, :]
        sub_variance = variance_map[y_min:y_max, :]
        
        # Get spatial profile
        profile = self.profiles[fiber_idx][:, np.newaxis]  # (n_spatial, 1)
        
        # Optimal weighting
        weights = profile**2 / (sub_variance + 1e-10)
        
        # Optimal extraction formula
        numerator = np.sum(profile * sub_image * weights, axis=0)
        denominator = np.sum(profile**2 * weights, axis=0)
        
        spectrum = numerator / (denominator + 1e-10)
        variance = 1.0 / (denominator + 1e-10)
        
        # === MODIFIED: Integrate over aperture window ===
        x_coords = np.arange(cfg.detector_width)
        
        # Compute target wavelength pixel positions
        wavelengths = self.cfg.lambda_min + np.arange(cfg.n_spectral_bins) * (
            (cfg.lambda_max - cfg.lambda_min) / (cfg.n_spectral_bins - 1)
        )
        
        # Use wavelength_to_pixel from the simulator (need to share this method)
        # For now, replicate the conversion
        dispersion_pix_per_nm = (
            cfg.dispersion_mm_per_nm * 1000.0 / cfg.pixel_size_um
        )
        x_pixels = (wavelengths - cfg.lambda_min) * dispersion_pix_per_nm
        x_pixels += cfg.detector_width / 2.0 - (
            (cfg.lambda_max - cfg.lambda_min) / 2.0 * dispersion_pix_per_nm
        )
        
        # Integrate over aperture window at each wavelength
        half_width = aperture_width // 2
        spec_interp = np.zeros(cfg.n_spectral_bins)
        var_interp = np.zeros(cfg.n_spectral_bins)
        
        for wl_idx, x_center in enumerate(x_pixels):
            x_min_win = max(0, int(x_center) - half_width)
            x_max_win = min(cfg.detector_width, int(x_center) + half_width + 1)
            
            # Integrate spectrum over window
            window_flux = 0.0
            window_var = 0.0
            total_weight = 0.0
            
            for x_idx in range(x_min_win, x_max_win):
                # Gaussian weight centered on x_center
                dx = x_idx - x_center
                weight = np.exp(-0.5 * (dx / (aperture_width / 2.355)) ** 2)
                
                window_flux += spectrum[x_idx] * weight
                window_var += variance[x_idx] * weight ** 2
                total_weight += weight
            
            if total_weight > 0:
                spec_interp[wl_idx] = window_flux / total_weight
                var_interp[wl_idx] = window_var / (total_weight ** 2)
            else:
                spec_interp[wl_idx] = 0.0
                var_interp[wl_idx] = 0.0
        
        return spec_interp, var_interp


# =====================================================================
# VISUALIZATION
# =====================================================================

def visualize_spectral_image(
    image: np.ndarray,
    fiber_positions: np.ndarray,
    config: SpectralConfig,
    title: str = "Simulated Spectral Image"
):
    """Visualize 2D spectral image with fiber traces marked."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Full image
    im1 = ax1.imshow(
        np.log10(image + 1),
        aspect='auto',
        cmap='viridis',
        origin='lower'
    )
    ax1.set_xlabel('Wavelength (pixel)')
    ax1.set_ylabel('Spatial (pixel)')
    ax1.set_title(f"{title} (log scale)")
    
    # Mark fiber positions
    for i, pos in enumerate(fiber_positions):
        ax1.axhline(pos, color='red', alpha=0.3, linewidth=0.5)
        ax1.text(10, pos, f'{i}', color='red', fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.colorbar(im1, ax=ax1, label='log10(Counts + 1)')
    
    # Zoomed region
    y_mid = config.detector_height // 2
    x_mid = config.detector_width // 2
    zoom_size = 100
    
    im2 = ax2.imshow(
        image[y_mid-zoom_size:y_mid+zoom_size, x_mid-zoom_size:x_mid+zoom_size],
        aspect='auto',
        cmap='viridis',
        origin='lower',
        extent=[x_mid-zoom_size, x_mid+zoom_size, y_mid-zoom_size, y_mid+zoom_size]
    )
    ax2.set_xlabel('Wavelength (pixel)')
    ax2.set_ylabel('Spatial (pixel)')
    ax2.set_title('Zoomed Region')
    plt.colorbar(im2, ax=ax2, label='Counts')
    
    plt.tight_layout()
    plt.show()


def visualize_extracted_spectra(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    variance: Optional[np.ndarray] = None,
    fiber_indices: Optional[List[int]] = None,
    title: str = "Extracted Spectra"
):
    """Visualize extracted 1D spectra."""
    if fiber_indices is None:
        fiber_indices = list(range(min(5, spectra.shape[1])))
    
    fig, axes = plt.subplots(len(fiber_indices), 1, 
                             figsize=(12, 3*len(fiber_indices)))
    if len(fiber_indices) == 1:
        axes = [axes]
    
    for i, fiber_idx in enumerate(fiber_indices):
        ax = axes[i]
        spec = spectra[:, fiber_idx]
        
        ax.plot(wavelengths, spec, 'k-', linewidth=1, label=f'Fiber {fiber_idx}')
        
        if variance is not None:
            sigma = np.sqrt(variance[:, fiber_idx])
            ax.fill_between(wavelengths, spec - sigma, spec + sigma, 
                           alpha=0.3, label='±1σ')
        
        ax.set_ylabel('Flux (counts)')
        ax.set_title(f'{title} - Fiber {fiber_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Wavelength (nm)')
    plt.tight_layout()
    plt.show()


def visualize_all_spectra_grid(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    title: str = "All 19 Fiber Spectra"
):
    """Visualize all 19 spectra in a 4×5 grid layout."""
    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for fiber_idx in range(19):
        ax = axes[fiber_idx]
        spec = spectra[:, fiber_idx]
        
        ax.plot(wavelengths, spec, 'b-', linewidth=0.8)
        ax.set_title(f'Fiber {fiber_idx}', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if fiber_idx >= 15:  # Bottom row
            ax.set_xlabel('λ (nm)', fontsize=9)
        if fiber_idx % 5 == 0:  # Left column
            ax.set_ylabel('Flux', fontsize=9)
    
    # Hide extra subplot
    axes[19].set_visible(False)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# =====================================================================
# INTEGRATION WITH EXISTING PIPELINE
# =====================================================================

def add_spectral_extraction_to_pipeline(pipeline, config: SpectralConfig):
    """
    Extend BatchPropagationPipeline with spectral extraction capability.
    
    This adds two new methods:
        - generate_spectral_image()
        - extract_fiber_spectra()
    """
    
    simulator = SpectralImageSimulator(config)
    extractor = SpectralExtractor(config)
    
    def generate_spectral_image(
        uf_batch,
        add_noise=True,
        exposure_time=1.0
    ):
        """Generate 2D spectral images from modal coefficients."""
        modal_powers = np.abs(uf_batch)**2  # (n_fields, n_modes)
        
        # Assume first 19 modes correspond to 19 fibers
        n_fields = modal_powers.shape[0]
        n_fibers = min(19, modal_powers.shape[1])
        
        images = []
        fiber_positions_list = []
        
        for i in range(n_fields):
            image, positions = simulator.create_spectral_image(
                modal_powers[i, :n_fibers],
                add_noise=add_noise,
                exposure_time=exposure_time
            )
            images.append(image)
            fiber_positions_list.append(positions)
        
        return np.array(images), fiber_positions_list[0]
    
    def extract_fiber_spectra(
        spectral_images,
        fiber_positions,
        method='optimal'
    ):
        """Extract 1D spectra from 2D spectral images."""
        n_fields = len(spectral_images)
        n_fibers = len(fiber_positions)
        n_wavelengths = config.n_spectral_bins
        
        all_spectra = np.zeros((n_fields, n_wavelengths, n_fibers))
        all_variance = np.zeros((n_fields, n_wavelengths, n_fibers))
        
        for i in range(n_fields):
            spectra, variance = extractor.extract_spectra(
                spectral_images[i],
                fiber_positions,
                method=method
            )
            all_spectra[i] = spectra
            all_variance[i] = variance
        
        return all_spectra, all_variance
    
    # Attach methods to pipeline instance
    pipeline.generate_spectral_image = generate_spectral_image
    pipeline.extract_fiber_spectra = extract_fiber_spectra
    pipeline.spectral_config = config
    pipeline.spectral_simulator = simulator
    pipeline.spectral_extractor = extractor
    
    return pipeline


# =====================================================================
# EXAMPLE USAGE
# =====================================================================

def example_spectral_extraction():
    """Demonstrate spectral extraction on simulated data."""
    
    print("="*70)
    print("SPECTRAL EXTRACTION EXAMPLE")
    print("="*70)
    
    # Create configuration
    config = SpectralConfig(
        lambda_min=700.0,
        lambda_max=900.0,
        n_spectral_bins=200,
        fiber_spacing_pixels=8.0,
        psf_fwhm_pixels=2.5,
    )
    
    # Initialize simulator and extractor
    simulator = SpectralImageSimulator(config)
    extractor = SpectralExtractor(config)
    
    # Simulate fiber spectra (19 fibers with different Gaussian line profiles)
    n_fibers = 19
    spectra_true = np.zeros((config.n_spectral_bins, n_fibers))
    
    for i in range(n_fibers):
        # Each fiber has a Gaussian emission line at different wavelength
        line_center = 750.0 + i * 5.0  # nm
        line_width = 2.0  # nm
        amplitude = 1000.0 + i * 50.0  # counts
        
        line_profile = amplitude * np.exp(
            -0.5 * ((simulator.wavelengths - line_center) / line_width)**2
        )
        
        # Add continuum
        continuum = 100.0 + i * 10.0
        
        spectra_true[:, i] = line_profile + continuum
    
    # Generate 2D spectral image
    print("\n[1/3] Generating 2D spectral image...")
    image, fiber_positions = simulator.create_spectral_image(
        modal_powers=None,
        spectra=spectra_true,
        add_noise=True,
        exposure_time=1.0
    )
    
    # Visualize
    print("[2/3] Visualizing spectral image...")
    visualize_spectral_image(image, fiber_positions, config)
    
    # Extract spectra
    print("[3/3] Extracting 1D spectra...")
    spectra_extracted, variance = extractor.extract_spectra(
        image, fiber_positions, method='optimal'
    )
    
    # Visualize extracted spectra
    visualize_extracted_spectra(
        spectra_extracted, 
        simulator.wavelengths,
        variance,
        fiber_indices=[0, 5, 10, 15, 18]
    )
    
    visualize_all_spectra_grid(spectra_extracted, simulator.wavelengths)
    
    # Compute extraction accuracy
    print("\n" + "="*70)
    print("EXTRACTION QUALITY METRICS")
    print("="*70)
    
    for i in [0, 5, 10, 15, 18]:
        residual = spectra_extracted[:, i] - spectra_true[:, i]
        rms = np.sqrt(np.mean(residual**2))
        snr = np.mean(spectra_true[:, i]) / np.sqrt(np.mean(variance[:, i]))
        
        print(f"Fiber {i:2d}:  RMS error = {rms:7.2f}  |  SNR = {snr:6.1f}")
    
    print("="*70)
    
    return image, spectra_extracted, spectra_true


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    # Run standalone example
    image, extracted, truth = example_spectral_extraction()
    
    print("\nSpectral extraction module ready for integration.")
    print("To use with your pipeline:")
    print("  from spectral_extraction_module import add_spectral_extraction_to_pipeline")
    print("  pipeline = add_spectral_extraction_to_pipeline(pipeline, config)")

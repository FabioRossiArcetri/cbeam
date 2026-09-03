# Auto-split from the former single-file batch_propagation_pipeline.py.
"""Core-centre detection, sub-pixel signal sampling and geometric alignment.

Pure processing -- this module imports no matplotlib.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import maximum_filter, map_coordinates
from scipy.optimize import linear_sum_assignment

from .constants import backend, jax, jnp, N_SIGNALS, DEFAULT_SUBPIXEL_N


def batch_collect_subpixel_signals(images: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
    rows = coords[0]  # Shape: (N_SIGNALS, DEFAULT_SUBPIXEL_N, DEFAULT_SUBPIXEL_N)
    cols = coords[1]  # Shape: (N_SIGNALS, DEFAULT_SUBPIXEL_N, DEFAULT_SUBPIXEL_N)
    
    def extract_single_core_patch(img, r_coords, c_coords):
        c_pack = jnp.stack([r_coords, c_coords], axis=0)
        patch = jax.scipy.ndimage.map_coordinates(img, c_pack, order=1, mode='constant', cval=0.0)
        return jnp.mean(patch)

    collect_all_cores_single_frame = jax.vmap(
        lambda img: jax.vmap(lambda r, c: extract_single_core_patch(img, r, c))(rows, cols)
    )
    return collect_all_cores_single_frame(images)


if backend == 'jax':
    batch_collect_subpixel_signals = jax.jit(batch_collect_subpixel_signals)


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
    # determinant correction: without it R can come out as a reflection
    # (improper rotation), which mirrors the detected grid before matching.
    d = np.sign(np.linalg.det(np.dot(Vt.T, U.T)))
    R = np.dot(Vt.T, np.dot(np.diag([1.0, d]), U.T))
    eval_aligned = np.dot(eval_scaled, R.T)

    diff = eval_aligned[:, np.newaxis, :] - ideal_centered[np.newaxis, :, :]
    cost_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    eval_indices, ideal_permutation = linear_sum_assignment(cost_matrix)

    print(f"[Geometric Fit] Detected/ideal scale ratio: {scale_eval / scale_ideal:.3f}")
    print(f"[Geometric Fit] Residual matching RMS error: {np.mean(cost_matrix[eval_indices, ideal_permutation]):.4e}\n")
    
    return ideal_permutation


def collect_subpixel_signals(image_2d, x_min, y_min, dx, dy, centers, n=DEFAULT_SUBPIXEL_N):
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


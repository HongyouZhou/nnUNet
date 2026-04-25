
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, label, gaussian_filter
from skimage.segmentation import watershed
from skimage.morphology import ball, binary_dilation, binary_erosion
import argparse
from pathlib import Path



import os
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, label, map_coordinates
from skimage.segmentation import watershed
from skimage.morphology import ball, binary_erosion
import argparse


def normalize_hu(image, min_hu=-1000, max_hu=1000):
    """Normalize HU values to [0, 1] range."""
    image = np.clip(image, min_hu, max_hu)
    return (image - min_hu) / (max_hu - min_hu)


def compute_potential(abbc_prob_map, ct_image, w_dist, w_boundary, w_intensity):
    """
    Compute the potential/elevation map for both Watershed and Flow methods.
    For Watershed: Treating it as 'Elevation' (hydrology).
    For Flow: Treating it as 'Potential Energy' (physics).
    """
    # Probabilities
    prob_boundary = abbc_prob_map[3]
    prob_fg = np.sum(abbc_prob_map[1:4], axis=0)
    foreground_mask = prob_fg > 0.5
    
    # Distance Map (Normalized)
    dist = distance_transform_edt(foreground_mask)
    if dist.max() > 0:
        dist = dist / dist.max()
        
    # Intensity Map (Normalized)
    norm_ct = normalize_hu(ct_image, min_hu=-200, max_hu=1500)
    
    # Potential construction
    # We want: 
    # High Potential (Peaks) at: Cores (Deep inside object)
    # Low Potential (Valleys) at: Background, Boundaries, and High Density Collapsed Zones
    
    # Base: Distance (0 at boundary, 1 at center)
    # Penalties: Boundary prob, Intensity
    potential = dist - (prob_boundary * w_boundary) - (norm_ct * w_intensity)
    
    return potential, foreground_mask


def fragments_from_watershed(potential, foreground_mask, markers):
    """Classic Watershed on inverted potential."""
    # Watershed expects catchment basins (low values). 
    # Our potential has peaks at centers. So we invert it.
    elevation = -potential 
    segmentation = watershed(elevation, markers, mask=foreground_mask)
    return segmentation.astype(np.uint16)


def fragments_from_flow(potential, foreground_mask, core_mask, n_iter=100, step=1.0):
    """
    Gradient Flow Tracking (Cellpose-like).
    Pixels flow towards local potential peaks.
    """
    shape = potential.shape
    ndim = potential.ndim
    
    # 1. Compute Gradients of the Potential field
    # We want to climb UP the potential (towards distance peaks)
    gradients = np.gradient(potential) # list of arrays [dz, dy, dx]
    # Normalize gradients? Not strictly necessary if we rely on distance field slope, 
    # but stabilizing can help. Let's keep it raw first as distance field slope is good behavior.
    
    # Convert vectors to a grid for sampling
    # Gradients are in index space
    grad_fields = np.stack(gradients, axis=0) # (D, H, W, D) or (D, D, H, W) depending on input
    
    # 2. Initialize Particle Positions
    # We only care about foreground pixels
    # Create meshgrid of coordinates
    coords = np.indices(shape).astype(np.float32) # (ndim, H, W, D)
    
    # Mask out background to save computation? 
    # Vectorized update on full volume is usually cleaner in numpy unless extreme sparsity.
    # Let's run on full volume but filter final assignment using mask.
    
    current_coords = coords.copy()
    
    # 3. Iterative Flow Simulation (Euler Integration)
    print(f"  Simulating flow for {n_iter} iterations...")
    for i in range(n_iter):
        # Sample gradients at current coordinates
        # Map coordinates expects (ndim, sample_coords) flattening
        # Using linear interpolation (order=1) for smooth flow
        
        # Optimization: Round to nearest integer for fast lookup (order=0)
        # much faster, slightly more jittery flow, but fine for segmentation
        sampled_grads = []
        for d in range(ndim):
            # We can use map_coordinates to sample the gradient field distored by current flow
            # But the gradient field itself is static in space!
            # We need to know the gradient vector AT the current particle position.
            g = map_coordinates(grad_fields[d], current_coords, order=0, mode='nearest')
            sampled_grads.append(g)
            
        sampled_grads = np.stack(sampled_grads, axis=0)
        
        # Update positions: climb up the gradient
        current_coords += step * sampled_grads
        
        # Clip to boundaries
        for d in range(ndim):
            np.clip(current_coords[d], 0, shape[d]-1, out=current_coords[d])
            
    # 4. Assign Labels based on final position
    # We check which Core (Label 2) the pixel landed in (or nearest to)
    
    # Round final coordinates to integers to lookup the Core Mask
    final_coords_int = np.rint(current_coords).astype(np.int32)
    
    # We need a labeled core map
    core_labels_map, _ = label(core_mask)
    
    # Lookup the label at the landing spot
    # Advanced: if landing spot is not in a core (e.g. local maxima outside core), 
    # we might have over-segmentation. But with our potential, peaks should be cores.
    
    # Use map_coordinates to pull the label from core_labels_map at final_coords
    captured_labels = map_coordinates(core_labels_map, final_coords_int, order=0, mode='nearest')
    
    # Apply foreground mask
    segmentation = captured_labels * foreground_mask
    
    return segmentation.astype(np.uint16)


def process_volume(abbc_prob_map, ct_image, 
                        method='flow',
                        w_dist=1.0, 
                        w_boundary=2.0, 
                        w_intensity=1.0, 
                        min_core_size=10):
    
    # 1. Decode Class Labels & Precursors
    pred_labels = np.argmax(abbc_prob_map, axis=0)
    core_mask = (pred_labels == 2)
    
    # Filter small cores
    if min_core_size > 0:
        core_labels, num_cores = label(core_mask)
        sizes = np.bincount(core_labels.ravel())
        mask_sizes = sizes > min_core_size
        mask_sizes[0] = 0
        core_mask = mask_sizes[core_labels]
        
    print(f"  Found {label(core_mask)[1]} Cores.")
        
    # 2. Compute Potential
    potential, fg_mask = compute_potential(abbc_prob_map, ct_image, w_dist, w_boundary, w_intensity)
    
    # 3. Run Selected Method
    if method == 'watershed':
        print("  Running Watershed...")
        markers = label(core_mask)[0]
        return fragments_from_watershed(potential, fg_mask, markers)
        
    elif method == 'flow':
        print("  Running Gradient Flow...")
        return fragments_from_flow(potential, fg_mask, core_mask, n_iter=60, step=1.0)
        
    else:
        raise ValueError(f"Unknown method: {method}")


def load_nifti(path):
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine, nii.header


def save_nifti(data, affine, header, path):
    nii = nib.Nifti1Image(data, affine, header)
    nib.save(nii, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--abbc", type=str, required=True, help="Path to ABBC probability map (4D Nifti)")
    parser.add_argument("--ct", type=str, required=True, help="Path to original CT image")
    parser.add_argument("--output", type=str, required=True, help="Path to save instance segmentation")
    parser.add_argument("--method", type=str, default='flow', choices=['watershed', 'flow'], help="Segmentation method")
    parser.add_argument("--w_dist", type=float, default=1.0)
    parser.add_argument("--w_boundary", type=float, default=2.0)
    parser.add_argument("--w_intensity", type=float, default=0.5)
    
    args = parser.parse_args()
    
    print(f"Processing {args.ct} with method [{args.method}]...")
    
    # Load Data
    abbc_data, aff, head = load_nifti(args.abbc)
    ct_data, _, _ = load_nifti(args.ct)
    
    # Handle Channel dimension
    if abbc_data.ndim == 4 and abbc_data.shape[-1] == 4:
        # Permute to [C, H, W, D]
        abbc_data = abbc_data.transpose(3, 0, 1, 2)
        
    if abbc_data.shape[0] != 4:
        raise ValueError(f"Expected 4 channels for ABBC (Bg, Shell, Core, Bound), got shape {abbc_data.shape}")

    # Run Post-processing
    instances = process_volume(
        abbc_data, 
        ct_data, 
        method=args.method,
        w_dist=args.w_dist, 
        w_boundary=args.w_boundary, 
        w_intensity=args.w_intensity
    )
    
    # Save
    save_nifti(instances, aff, head, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()


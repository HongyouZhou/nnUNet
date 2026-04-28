#!/usr/bin/env python3
"""
ABBC to Instance Segmentation Converter for Charite Results

This script converts ABBC segmentation results from the Charite fine-tune results
to instance segmentation using the abbc2instance function.

Data structure:
/ssdArray/hongyou/dev/data/charite_results/30_07_charite_fine_tune/images/
├── case1_input.nii.gz      # Original CT image
├── case1_pred.nii.gz       # ABBC segmentation
├── case2_input.nii.gz      # Original CT image
├── case2_pred.nii.gz       # ABBC segmentation
└── ...
"""

import os
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
import time
from typing import Tuple, Optional
import sys
import shutil

# Add the project root to Python path to import abbc2instance
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from tools.PENGWIN.abbc_conversion.abbc2instance import abbc2instance
    from tools.PENGWIN.postprocess.abbc_watershed import process_volume
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please make sure you're running this script from the correct directory")
    sys.exit(1)


def load_nifti(file_path: str) -> Tuple[np.ndarray, dict]:
    """
    Load a NIfTI file and return the array and header information.
    
    Args:
        file_path: Path to the NIfTI file
        
    Returns:
        Tuple of (array, header_info)
    """
    try:
        img = nib.load(file_path)
        return img.get_fdata(), img.header
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def save_nifti(array: np.ndarray, file_path: str, header_info: dict, dtype=np.uint16):
    """
    Save a numpy array as a NIfTI file.
    
    Args:
        array: Array to save
        file_path: Output file path
        header_info: Header information from original image
        dtype: Data type for output
    """
    try:
        # Create new header based on the original
        new_header = header_info.copy()
        new_header.set_data_dtype(dtype)
        
        # Create new image and save
        new_img = nib.Nifti1Image(array.astype(dtype), None, new_header)
        nib.save(new_img, file_path)
        print(f"Saved: {file_path}")
    except Exception as e:
        print(f"Error saving {file_path}: {e}")


def label_to_onehot(label_map: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """
    Convert a label map (H, W, D) to a one-hot probability map (C, H, W, D).
    """
    shape = (num_classes,) + label_map.shape
    onehot = np.zeros(shape, dtype=np.float32)
    for i in range(num_classes):
        onehot[i] = (label_map == i).astype(np.float32)
    return onehot


def convert_abbc_to_instance(abbc_array: np.ndarray, 
                            ct_array: Optional[np.ndarray] = None,
                            core_label: int = 2, 
                            boundary_label: int = 1, 
                            border_label: int = 3,
                            processes: Optional[int] = None,
                            progressbar: bool = True,
                            dtype: np.dtype = np.uint16,
                            method: str = 'original',
                            w_dist: float = 1.0,
                            w_boundary: float = 2.0,
                            w_intensity: float = 1.0) -> Tuple[np.ndarray, int]:
    """
    Convert ABBC segmentation to instance segmentation.
    
    Args:
        abbc_array: Input ABBC segmentation array
        ct_array: Input CT image array (required for watershed/flow methods)
        core_label: Label for core regions (default: 2)
        boundary_label: Label for boundary regions (default: 1)  
        border_label: Label for border regions (default: 3)
        processes: Number of processes to use (default: None for single process)
        progressbar: Whether to show progress bar (default: True)
        dtype: Output data type (default: np.uint16)
        method: Segmentation method ('original', 'watershed', 'flow')
        w_dist: Weight for distance map (for watershed/flow)
        w_boundary: Weight for boundary probability (for watershed/flow)
        w_intensity: Weight for intensity map (for watershed/flow)
        
    Returns:
        Tuple of (instance_segmentation, number_of_instances)
    """
    print(f"Converting ABBC to instance segmentation using method: {method}")
    print(f"Input shape: {abbc_array.shape}")
    print(f"Labels: core={core_label}, boundary={boundary_label}, border={border_label}")
    
    # Check if input contains expected labels
    unique_labels = np.unique(abbc_array)
    print(f"Unique labels in input: {unique_labels}")
    
    if core_label not in unique_labels:
        print(f"Warning: Core label {core_label} not found in input!")
    
    start_time = time.time()
    
    if method == 'original':
        # Convert ABBC to instance using original method
        instances, num_instances = abbc2instance(
            abbc=abbc_array,
            core_label=core_label,
            boundary_label=boundary_label,
            border_label=border_label,
            processes=processes,
            progressbar=progressbar,
            dtype=dtype
        )
    elif method in ['watershed', 'flow']:
        if ct_array is None:
            raise ValueError(f"CT array is required for method {method}")
            
        # Convert label map to one-hot probability map
        # Assuming labels are 0, 1, 2, 3. If custom labels are used, we might need mapping.
        # Here we assume standard mapping: 0=bg, 1=boundary, 2=core, 3=border
        # If user provided custom labels, we need to map them to 0,1,2,3 for process_volume
        
        # Create a standard 0-3 map for process_volume
        standard_map = np.zeros_like(abbc_array, dtype=np.uint8)
        standard_map[abbc_array == boundary_label] = 1
        standard_map[abbc_array == core_label] = 2
        standard_map[abbc_array == border_label] = 3
        
        abbc_prob_map = label_to_onehot(standard_map, num_classes=4)
        
        instances = process_volume(
            abbc_prob_map=abbc_prob_map,
            ct_image=ct_array,
            method=method,
            w_dist=w_dist,
            w_boundary=w_boundary,
            w_intensity=w_intensity
        )
        instances = instances.astype(dtype)
        num_instances = len(np.unique(instances)) - 1 if np.any(instances) else 0
    else:
        raise ValueError(f"Unknown method: {method}")

    end_time = time.time()
    
    print(f"Conversion completed in {end_time - start_time:.2f} seconds")
    print(f"Number of instances found: {num_instances}")
    
    return instances, num_instances


def process_charite_case(input_dir: str, 
                        output_dir: str,
                        case_name: str,
                        core_label: int = 2,
                        boundary_label: int = 1, 
                        border_label: int = 3,
                        processes: Optional[int] = None,
                        progressbar: bool = True,
                        dtype: np.dtype = np.uint16,
                        copy_input: bool = True,
                        method: str = 'original',
                        w_dist: float = 1.0,
                        w_boundary: float = 2.0,
                        w_intensity: float = 1.0) -> bool:
    """
    Process a single Charite case (ABBC prediction + CT input).
    
    Args:
        input_dir: Input directory containing the case files
        output_dir: Output directory for results
        case_name: Name of the case (e.g., "case1")
        core_label: Label for core regions
        boundary_label: Label for boundary regions
        border_label: Label for border regions
        processes: Number of processes to use
        progressbar: Whether to show progress bar
        dtype: Output data type
        copy_input: Whether to copy the input CT image to output
        method: Segmentation method
        w_dist, w_boundary, w_intensity: Weights for watershed/flow
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Processing case: {case_name}")
    print(f"{'='*60}")
    
    # Define file paths
    ct_input_file = os.path.join(input_dir, f"{case_name}_input.nii.gz")
    abbc_pred_file = os.path.join(input_dir, f"{case_name}_pred.nii.gz")
    
    # Check if files exist
    if not os.path.exists(ct_input_file):
        print(f"Error: CT input file not found: {ct_input_file}")
        return False
    
    if not os.path.exists(abbc_pred_file):
        print(f"Error: ABBC prediction file not found: {abbc_pred_file}")
        return False
    
    # Load CT input to check dimensions
    print(f"Loading CT input for dimension check: {ct_input_file}")
    ct_array, ct_header_info = load_nifti(ct_input_file)
    if ct_array is None:
        print(f"Error: Failed to load CT input file: {ct_input_file}")
        return False
    
    print(f"CT dimensions: {ct_array.shape}")

    # Create output directory for this case
    case_output_dir = os.path.join(output_dir, case_name)
    os.makedirs(case_output_dir, exist_ok=True)
    
    # Copy input CT image if requested
    if copy_input:
        ct_output_file = os.path.join(case_output_dir, f"{case_name}_input.nii.gz")
        shutil.copy2(ct_input_file, ct_output_file)
        print(f"Copied CT input: {ct_output_file}")
    
    # Load ABBC segmentation
    print(f"Loading ABBC prediction: {abbc_pred_file}")
    abbc_array, header_info = load_nifti(abbc_pred_file)
    if abbc_array is None:
        return False
    
    # Convert to instance segmentation
    try:
        instances, num_instances = convert_abbc_to_instance(
            abbc_array=abbc_array,
            ct_array=ct_array,
            core_label=core_label,
            boundary_label=boundary_label,
            border_label=border_label,
            processes=processes,
            progressbar=progressbar,
            dtype=dtype,
            method=method,
            w_dist=w_dist,
            w_boundary=w_boundary,
            w_intensity=w_intensity
        )
        
        # Save instance segmentation
        instance_output_file = os.path.join(case_output_dir, f"{case_name}_instance.nii.gz")
        save_nifti(instances, instance_output_file, header_info, dtype)
        
        # Save ABBC prediction for reference
        abbc_output_file = os.path.join(case_output_dir, f"{case_name}_abbc.nii.gz")
        save_nifti(abbc_array, abbc_output_file, header_info, np.int8)
        
        print(f"Successfully processed case {case_name}")
        print(f"Output directory: {case_output_dir}")
        print(f"Files created:")
        print(f"  - {case_name}_input.nii.gz (CT image)")
        print(f"  - {case_name}_abbc.nii.gz (ABBC segmentation)")
        print(f"  - {case_name}_instance.nii.gz (Instance segmentation)")
        print(f"Number of instances: {num_instances}")
        
        return True
        
    except Exception as e:
        print(f"Error converting case {case_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_charite_directory(input_dir: str, 
                            output_dir: str,
                            core_label: int = 2,
                            boundary_label: int = 1,
                            border_label: int = 3,
                            processes: Optional[int] = None,
                            progressbar: bool = True,
                            dtype: np.dtype = np.uint16,
                            copy_input: bool = True,
                            method: str = 'original',
                            w_dist: float = 1.0,
                            w_boundary: float = 2.0,
                            w_intensity: float = 1.0) -> None:
    """
    Process all Charite cases in the input directory.
    
    Args:
        input_dir: Input directory containing Charite case files
        output_dir: Output directory for results
        core_label: Label for core regions
        boundary_label: Label for boundary regions
        border_label: Label for border regions
        processes: Number of processes to use
        progressbar: Whether to show progress bar
        dtype: Output data type
        copy_input: Whether to copy input CT images
        method: Segmentation method
        w_dist, w_boundary, w_intensity: Weights for watershed/flow
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all case names (files ending with _input.nii.gz)
    input_files = list(input_path.glob("*_input.nii.gz"))
    
    if not input_files:
        print(f"No input files found in {input_dir}")
        return
    
    # Extract case names
    case_names = []
    for input_file in input_files:
        case_name = input_file.name.replace('_input.nii.gz', '')
        case_names.append(case_name)
    
    print(f"Found {len(case_names)} cases to process:")
    for case_name in case_names:
        print(f"  - {case_name}")
    
    # Process each case
    successful = 0
    failed = 0
    
    for case_name in case_names:
        if process_charite_case(
            input_dir, output_dir, case_name,
            core_label, boundary_label, border_label,
            processes, progressbar, dtype, copy_input,
            method, w_dist, w_boundary, w_intensity
        ):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Processing Summary:")
    print(f"{'='*60}")
    print(f"Total cases: {len(case_names)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")
    
    if successful > 0:
        print(f"\nResults are saved in:")
        for case_name in case_names[:3]:  # Show first 3 cases
            case_output_dir = os.path.join(output_dir, case_name)
            print(f"  {case_output_dir}/")
        if len(case_names) > 3:
            print(f"  ... and {len(case_names) - 3} more cases")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Charite ABBC segmentation results to instance segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data Structure Expected:
/ssdArray/hongyou/dev/data/charite_results/30_07_charite_fine_tune/images/
├── case1_input.nii.gz      # Original CT image
├── case1_pred.nii.gz       # ABBC segmentation
├── case2_input.nii.gz      # Original CT image
├── case2_pred.nii.gz       # ABBC segmentation
└── ...

Examples:
  # Process all cases in the default Charite directory
  python charite_abbc2instance.py
  
  # Process with custom output directory
  python charite_abbc2instance.py --output_dir /path/to/output
  
  # Use custom labels
  python charite_abbc2instance.py --core_label 5 --boundary_label 3 --border_label 7
  
  # Use multiple processes
  python charite_abbc2instance.py --processes 4
  
  # Use watershed method
  python charite_abbc2instance.py --method watershed
  
  # Use flow method with custom weights
  python charite_abbc2instance.py --method flow --w_dist 1.5 --w_boundary 2.5
        """
    )
    
    # Default paths
    default_input_dir = "/ssdArray/hongyou/dev/data/charite_results/Dataset989_charite/images"
    default_output_dir = "/ssdArray/hongyou/dev/data/charite_results/Dataset989_charite/instances"
    
    # Input/Output options
    parser.add_argument('--input_dir', type=str, default=default_input_dir, 
                       help=f'Input directory containing Charite case files (default: {default_input_dir})')
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                       help=f'Output directory for results (default: {default_output_dir})')
    
    # ABBC label options
    parser.add_argument('--core_label', type=int, default=2, help='Label for core regions (default: 2)')
    parser.add_argument('--boundary_label', type=int, default=1, help='Label for boundary regions (default: 1)')
    parser.add_argument('--border_label', type=int, default=3, help='Label for border regions (default: 3)')
    
    # Processing options
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use (default: single process)')
    parser.add_argument('--no_progressbar', action='store_true', help='Disable progress bar')
    parser.add_argument('--dtype', type=str, default='uint16', choices=['uint8', 'uint16', 'uint32'], help='Output data type (default: uint16)')
    parser.add_argument('--no_copy_input', action='store_true', help='Do not copy input CT images to output')
    
    # Method options
    parser.add_argument('--method', type=str, default='watershed', choices=['original', 'watershed', 'flow'], 
                       help='Segmentation method (default: original)')
    parser.add_argument('--w_dist', type=float, default=1.0, help='Weight for distance map (default: 1.0)')
    parser.add_argument('--w_boundary', type=float, default=2.0, help='Weight for boundary probability (default: 2.0)')
    parser.add_argument('--w_intensity', type=float, default=1.0, help='Weight for intensity map (default: 1.0)')
    
    args = parser.parse_args()
    
    # Convert dtype string to numpy dtype
    dtype_map = {'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32}
    dtype = dtype_map[args.dtype]
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Process all cases
    process_charite_directory(
        args.input_dir, args.output_dir,
        args.core_label, args.boundary_label, args.border_label,
        args.processes, not args.no_progressbar, dtype, not args.no_copy_input,
        args.method, args.w_dist, args.w_boundary, args.w_intensity
    )


if __name__ == "__main__":
    main()

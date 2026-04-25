#!/usr/bin/env python3
"""
This script evaluates seg_inference outputs against ground truth instance segmentation.

seg_inference outputs separate binary masks for each bone structure (seg_femur.nii.gz, seg_fragment_1.nii.gz, etc.)
This script:
1. Loads all binary masks for a case
2. Merges them into a single instance segmentation (each file = one instance)
3. Evaluates against ground truth using the same metrics as evaluate_instance_metrics.py

Data structure:
Ground truth: /home/hongyou/dev/data/10_10/{case_id}/{case_id}_0000_pred.nii.gz.seg.nrrd
seg_inference: /ssdArray/hongyou/dev/data/seg_inference/{case_id}/*.nii.gz
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

# Try to import abbc2instance for GT conversion (optional)
abbc2instance = None
try:
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.datasets.PENGWIN.abbc_conversion.abbc2instance import abbc2instance
except ImportError:
    logging.warning("Could not import abbc2instance. Will use simple connected components for GT conversion if needed.")


def setup_logging(log_file: str | None = None, level=logging.INFO):
    """
    Setup logging to output to both console and file.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter('%(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logging.info(f"Logging to file: {log_file}")
    
    return logger


def load_nifti_instance(file_path: str) -> tuple[np.ndarray | None, dict | None]:
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
        logging.error(f"Error loading {file_path}: {e}")
        return None, None


def load_nrrd_segmentation(file_path: str) -> np.ndarray | None:
    """
    Load a .seg.nrrd file using SimpleITK.
    
    Args:
        file_path: Path to the .seg.nrrd file
        
    Returns:
        Numpy array of the segmentation
    """
    try:
        img_sitk = sitk.ReadImage(str(file_path))
        # SimpleITK reads in (z, y, x) order, transpose to match nibabel (x, y, z)
        img_arr = sitk.GetArrayFromImage(img_sitk).transpose(2, 1, 0)
        return img_arr
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None


def load_seg_inference_masks(case_dir: str) -> tuple[np.ndarray | None, list[str]]:
    """
    Load all seg_inference binary masks for a case and convert to instance segmentation.
    Each seg_*.nii.gz file represents a single instance (no connected component analysis needed).
    
    Args:
        case_dir: Directory containing seg_inference outputs (e.g., .../549/)
        
    Returns:
        Tuple of (instance segmentation array, list of loaded instance files)
    """
    case_path = Path(case_dir)
    
    if not case_path.exists():
        logging.error(f"Case directory does not exist: {case_dir}")
        return None, []
    
    # Find all seg_*.nii.gz files (exclude ct.nii.gz)
    mask_files = sorted([f for f in case_path.glob("seg_*.nii.gz")])
    
    if not mask_files:
        logging.error(f"No segmentation mask files found in {case_dir}")
        return None, []
    
    logging.info(f"Found {len(mask_files)} segmentation mask files")
    
    # Load the first mask to get the shape
    first_mask, _ = load_nifti_instance(str(mask_files[0]))
    if first_mask is None:
        return None, []
    
    # Initialize instance segmentation (all background = 0)
    instance_seg = np.zeros(first_mask.shape, dtype=np.uint16)
    current_instance_id = 1
    loaded_files = []
    
    # Process each segmentation mask file
    for mask_file in mask_files:
        mask, _ = load_nifti_instance(str(mask_file))
        if mask is None:
            logging.warning(f"Failed to load {mask_file}, skipping")
            continue
        
        # Convert to binary
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Check if mask is non-empty
        if binary_mask.sum() > 0:
            # Each file is one instance, assign unique instance ID
            instance_seg[binary_mask > 0] = current_instance_id
            loaded_files.append(mask_file.name)
            current_instance_id += 1
        else:
            logging.warning(f"Empty mask: {mask_file.name}")
    
    total_instances = current_instance_id - 1
    logging.info(f"Processed {len(loaded_files)} segmentation masks")
    logging.info(f"Total instances found: {total_instances}")
    if len(loaded_files) <= 5:
        for file in loaded_files:
            logging.info(f"  - {file}")
    else:
        for file in loaded_files[:3]:
            logging.info(f"  - {file}")
        logging.info(f"  ... and {len(loaded_files) - 3} more")
    
    return instance_seg, loaded_files


def convert_semantic_to_instance(semantic_array: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Convert binary semantic segmentation to instance segmentation using connected components.
    
    Args:
        semantic_array: Binary segmentation array (0 = background, 1 = bone)
        
    Returns:
        Tuple of (instance_array, num_instances)
    """
    logging.info("Converting semantic segmentation to instance segmentation using connected components...")
    start_time = time.time()
    
    # Use scipy's connected components labeling
    # structure defines connectivity (3x3x3 for 26-connectivity)
    structure = ndimage.generate_binary_structure(3, 3)
    labeled_array, num_instances = ndimage.label(semantic_array, structure=structure)
    
    end_time = time.time()
    logging.info(f"Conversion completed in {end_time - start_time:.2f} seconds")
    logging.info(f"Number of instances: {num_instances}")
    
    return labeled_array.astype(np.uint16), num_instances


def convert_abbc_to_instance(semantic_array: np.ndarray,
                              core_label: int = 2,
                              boundary_label: int = 1,
                              border_label: int = 3,
                              processes: int | None = None,
                              progressbar: bool = True) -> tuple[np.ndarray, int]:
    """
    Convert semantic ABBC segmentation to instance segmentation.
    
    Args:
        semantic_array: Semantic segmentation array
        core_label: Label for core regions
        boundary_label: Label for boundary regions
        border_label: Label for border regions
        processes: Number of processes for parallel processing
        progressbar: Whether to show progress bar
        
    Returns:
        Tuple of (instance_array, num_instances)
    """
    logging.info("Converting ABBC segmentation to instance segmentation...")
    start_time = time.time()
    
    if abbc2instance is not None:
        # Use the proper ABBC conversion if available
        instances, num_instances = abbc2instance(
            abbc=semantic_array,
            core_label=core_label,
            boundary_label=boundary_label,
            border_label=border_label,
            processes=processes,
            progressbar=progressbar,
            dtype=np.uint16
        )
    else:
        # Fallback: simple connected components on combined mask
        logging.warning("abbc2instance not available, using simple connected components")
        combined_mask = np.logical_or(
            semantic_array == core_label,
            np.logical_or(semantic_array == boundary_label, semantic_array == border_label)
        ).astype(np.uint8)
        instances, num_instances = convert_semantic_to_instance(combined_mask)
    
    end_time = time.time()
    logging.info(f"Conversion completed in {end_time - start_time:.2f} seconds")
    logging.info(f"Number of instances: {num_instances}")
    
    return instances, num_instances


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute IoU between two binary masks.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        IoU value
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute DICE coefficient between two binary masks.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        DICE value
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    pred_volume = pred_mask.sum()
    gt_volume = gt_mask.sum()
    
    if pred_volume + gt_volume == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2 * intersection / (pred_volume + gt_volume)


def match_instances(pred_array: np.ndarray,
                   gt_array: np.ndarray,
                   iou_threshold: float = 0.1) -> tuple[list[tuple[int, int, float, float]],
                                                         list[int],
                                                         list[int]]:
    """
    Match predicted instances to ground truth instances using greedy IoU-based matching.
    
    Args:
        pred_array: Predicted instance segmentation array
        gt_array: Ground truth instance segmentation array
        iou_threshold: Minimum IoU for a match
        
    Returns:
        Tuple of:
        - matched_pairs: List of (pred_id, gt_id, iou, dice)
        - unmatched_pred: List of unmatched prediction IDs
        - unmatched_gt: List of unmatched GT IDs
    """
    pred_ids = np.unique(pred_array)
    pred_ids = pred_ids[pred_ids > 0]  # Remove background
    
    gt_ids = np.unique(gt_array)
    gt_ids = gt_ids[gt_ids > 0]  # Remove background
    
    logging.info(f"  Found {len(pred_ids)} predicted instances and {len(gt_ids)} GT instances")
    
    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return [], list(pred_ids), list(gt_ids)
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_ids), len(gt_ids)))
    
    for i, pred_id in enumerate(pred_ids):
        pred_mask = (pred_array == pred_id)
        for j, gt_id in enumerate(gt_ids):
            gt_mask = (gt_array == gt_id)
            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)
    
    # Greedy matching: iteratively match the pair with highest IoU
    matched_pairs = []
    matched_pred = set()
    matched_gt = set()
    
    while True:
        # Find maximum IoU in the matrix
        max_iou = np.max(iou_matrix)
        
        if max_iou < iou_threshold:
            break
        
        # Find the indices of maximum IoU
        max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        pred_idx, gt_idx = max_idx
        
        pred_id = pred_ids[pred_idx]
        gt_id = gt_ids[gt_idx]
        
        # Compute DICE for this pair
        pred_mask = (pred_array == pred_id)
        gt_mask = (gt_array == gt_id)
        dice = compute_dice(pred_mask, gt_mask)
        
        matched_pairs.append((int(pred_id), int(gt_id), max_iou, dice))
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        
        # Set this row and column to 0 to avoid rematching
        iou_matrix[pred_idx, :] = 0
        iou_matrix[:, gt_idx] = 0
    
    # Find unmatched instances
    unmatched_pred = [int(pred_ids[i]) for i in range(len(pred_ids)) if i not in matched_pred]
    unmatched_gt = [int(gt_ids[j]) for j in range(len(gt_ids)) if j not in matched_gt]
    
    return matched_pairs, unmatched_pred, unmatched_gt


def evaluate_case(seg_inference_dir: str,
                 gt_file: str,
                 case_id: str,
                 core_label: int = 2,
                 boundary_label: int = 1,
                 border_label: int = 3,
                 iou_threshold: float = 0.1,
                 processes: int | None = None) -> dict | None:
    """
    Evaluate seg_inference output for a single case.
    
    Args:
        seg_inference_dir: Directory containing seg_inference outputs
        gt_file: Path to ground truth file
        case_id: Case identifier
        core_label: Core label for ABBC conversion
        boundary_label: Boundary label for ABBC conversion
        border_label: Border label for ABBC conversion
        iou_threshold: Minimum IoU for matching
        processes: Number of processes for conversion
        
    Returns:
        Dictionary with evaluation results or None if failed
    """
    logging.info(f"\n{'=' * 60}")
    logging.info(f"Evaluating case: {case_id}")
    logging.info(f"{'=' * 60}")
    
    # Load seg_inference masks and convert to instance segmentation
    logging.info(f"Loading seg_inference masks from: {seg_inference_dir}")
    pred_array, loaded_files = load_seg_inference_masks(seg_inference_dir)
    if pred_array is None:
        logging.error("Failed to load seg_inference masks")
        return None
    
    num_pred_instances = len(np.unique(pred_array)) - 1  # Exclude background
    
    # Load ground truth
    logging.info(f"Loading ground truth: {gt_file}")
    if gt_file.endswith('.nrrd') or gt_file.endswith('.seg.nrrd'):
        gt_array = load_nrrd_segmentation(gt_file)
    else:
        gt_array, _ = load_nifti_instance(gt_file)
    
    if gt_array is None:
        logging.error("Failed to load ground truth file")
        return None
    
    logging.info(f"Prediction shape: {pred_array.shape}, GT shape: {gt_array.shape}")
    
    # Check if shapes match
    if pred_array.shape != gt_array.shape:
        logging.warning(f"Warning: Shape mismatch! Prediction: {pred_array.shape}, GT: {gt_array.shape}")
        return None
    
    # Check if GT is semantic and convert if needed
    if is_semantic_segmentation(gt_array):
        logging.info("Ground truth appears to be semantic segmentation (ABBC format)")
        gt_array, num_gt_instances = convert_abbc_to_instance(
            gt_array, core_label, boundary_label, border_label, 
            processes, progressbar=False
        )
    else:
        logging.info("Ground truth appears to be instance segmentation")
        num_gt_instances = len(np.unique(gt_array)) - 1  # Exclude background
    
    # Match instances
    logging.info(f"Matching instances with IoU threshold = {iou_threshold}")
    matched_pairs, unmatched_pred, unmatched_gt = match_instances(
        pred_array, gt_array, iou_threshold
    )
    
    num_matched = len(matched_pairs)
    
    logging.info(f"Matched {num_matched} instance pairs")
    logging.info(f"Unmatched predictions: {len(unmatched_pred)}")
    logging.info(f"Unmatched ground truth: {len(unmatched_gt)}")
    
    # Compute aggregate metrics including unmatched instances as 0
    # For matched pairs: use actual IoU and DICE
    # For unmatched predictions (FP): DICE=0, IoU=0
    # For unmatched GT (FN): DICE=0, IoU=0
    
    ious = [pair[2] for pair in matched_pairs]  # Matched instances
    dices = [pair[3] for pair in matched_pairs]  # Matched instances
    
    # Add zeros for unmatched predictions (false positives)
    ious.extend([0.0] * len(unmatched_pred))
    dices.extend([0.0] * len(unmatched_pred))
    
    # Add zeros for unmatched ground truth (false negatives)
    ious.extend([0.0] * len(unmatched_gt))
    dices.extend([0.0] * len(unmatched_gt))
    
    mean_iou = np.mean(ious) if len(ious) > 0 else 0.0
    mean_dice = np.mean(dices) if len(dices) > 0 else 0.0
    
    # Also track metrics only for matched instances
    matched_mean_iou = np.mean([pair[2] for pair in matched_pairs]) if num_matched > 0 else 0.0
    matched_mean_dice = np.mean([pair[3] for pair in matched_pairs]) if num_matched > 0 else 0.0
    
    # Compute precision, recall, F1
    precision = num_matched / num_pred_instances if num_pred_instances > 0 else 0.0
    recall = num_matched / num_gt_instances if num_gt_instances > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    logging.info(f"Mean IoU (with FP/FN as 0): {mean_iou:.4f}")
    logging.info(f"Mean DICE (with FP/FN as 0): {mean_dice:.4f}")
    logging.info(f"Mean IoU (matched only): {matched_mean_iou:.4f}")
    logging.info(f"Mean DICE (matched only): {matched_mean_dice:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")
    
    return {
        'case_id': case_id,
        'num_gt_instances': num_gt_instances,
        'num_pred_instances': num_pred_instances,
        'num_matched': num_matched,
        'mean_dice': mean_dice,  # Includes FP/FN as 0
        'mean_iou': mean_iou,    # Includes FP/FN as 0
        'matched_mean_dice': matched_mean_dice,  # Only matched instances
        'matched_mean_iou': matched_mean_iou,    # Only matched instances
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'matched_pairs': matched_pairs
    }


def is_semantic_segmentation(array: np.ndarray, max_semantic_label: int = 3) -> bool:
    """
    Check if the segmentation is semantic (ABBC format with labels 0,1,2,3)
    or instance (sequential instance IDs).
    
    Args:
        array: Segmentation array
        max_semantic_label: Maximum label value for semantic segmentation
        
    Returns:
        True if semantic, False if instance
    """
    unique_labels = np.unique(array)
    # If all labels are <= max_semantic_label, likely semantic
    return np.max(unique_labels) <= max_semantic_label


def find_case_files(gt_dir: str, seg_inference_dir: str) -> list[tuple[str, str, str]]:
    """
    Find matching case files in GT and seg_inference directories.
    
    Args:
        gt_dir: Ground truth directory
        seg_inference_dir: seg_inference output directory
        
    Returns:
        List of (case_id, gt_file_path, seg_inference_case_dir) tuples
    """
    gt_path = Path(gt_dir)
    seg_inference_path = Path(seg_inference_dir)
    
    if not gt_path.exists():
        logging.error(f"Error: GT directory does not exist: {gt_dir}")
        return []
    
    if not seg_inference_path.exists():
        logging.error(f"Error: seg_inference directory does not exist: {seg_inference_dir}")
        return []
    
    # Find all case subdirectories in GT
    case_dirs = [d for d in gt_path.iterdir() if d.is_dir()]
    
    matched_files = []
    
    for case_dir in case_dirs:
        case_id = case_dir.name
        
        # Look for GT file (various possible naming patterns)
        gt_candidates = [
            case_dir / f"{case_id}_0000_pred.nii.gz.seg.nrrd",
            case_dir / f"{case_id}_0000_pred.nii.gz.nii.seg.nrrd",
            case_dir / f"{case_id}_0000_pred.nii.seg.nrrd",
        ]
        
        gt_file = None
        for candidate in gt_candidates:
            if candidate.exists():
                gt_file = candidate
                break
        
        if gt_file is None:
            logging.warning(f"Warning: No GT file found for case {case_id}")
            continue
        
        # Look for seg_inference output directory (directly under case_id)
        seg_inference_case_dir = seg_inference_path / case_id
        
        if not seg_inference_case_dir.exists():
            logging.warning(f"Warning: No seg_inference output found for case {case_id}")
            continue
        
        # Check if there are any segmentation mask files
        mask_files = list(seg_inference_case_dir.glob("seg_*.nii.gz"))
        
        if not mask_files:
            logging.warning(f"Warning: No segmentation mask files found in {seg_inference_case_dir}")
            continue
        
        matched_files.append((case_id, str(gt_file), str(seg_inference_case_dir)))
    
    return matched_files


def save_results_to_csv(results: list[dict], output_file: str):
    """
    Save evaluation results to a CSV file, including summary statistics at the end.
    
    Args:
        results: List of result dictionaries
        output_file: Output CSV file path
    """
    if not results:
        logging.warning("No results to save")
        return
    
    fieldnames = ['case_id', 'num_gt_instances', 'num_pred_instances', 'num_matched',
                  'mean_dice', 'mean_iou', 'matched_mean_dice', 'matched_mean_iou', 
                  'precision', 'recall', 'f1']
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write per-case results
        for result in results:
            row = {k: result[k] for k in fieldnames}
            writer.writerow(row)
        
        # Calculate and write summary statistics
        total_cases = len(results)
        total_gt_instances = sum(r['num_gt_instances'] for r in results)
        total_pred_instances = sum(r['num_pred_instances'] for r in results)
        total_matched = sum(r['num_matched'] for r in results)
        
        # Aggregate metrics (weighted by total number of instances per case)
        # mean_dice and mean_iou already include FP/FN as 0
        weighted_dice = []
        weighted_iou = []
        weighted_matched_dice = []
        weighted_matched_iou = []
        
        for result in results:
            # Number of instances considered in this case (GT + Pred - Matched, to avoid double counting)
            num_instances = result['num_gt_instances'] + result['num_pred_instances'] - result['num_matched']
            if num_instances > 0:
                weighted_dice.extend([result['mean_dice']] * num_instances)
                weighted_iou.extend([result['mean_iou']] * num_instances)
            
            # For matched-only metrics, weight by number of matched instances
            if result['num_matched'] > 0:
                weighted_matched_dice.extend([result['matched_mean_dice']] * result['num_matched'])
                weighted_matched_iou.extend([result['matched_mean_iou']] * result['num_matched'])
        
        overall_mean_dice = float(np.mean(weighted_dice)) if weighted_dice else 0.0
        overall_mean_iou = float(np.mean(weighted_iou)) if weighted_iou else 0.0
        overall_matched_mean_dice = float(np.mean(weighted_matched_dice)) if weighted_matched_dice else 0.0
        overall_matched_mean_iou = float(np.mean(weighted_matched_iou)) if weighted_matched_iou else 0.0
        
        overall_precision = total_matched / total_pred_instances if total_pred_instances > 0 else 0.0
        overall_recall = total_matched / total_gt_instances if total_gt_instances > 0 else 0.0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        # Per-case statistics
        case_dices = [r['mean_dice'] for r in results]
        case_ious = [r['mean_iou'] for r in results]
        case_matched_dices = [r['matched_mean_dice'] for r in results if r['num_matched'] > 0]
        case_matched_ious = [r['matched_mean_iou'] for r in results if r['num_matched'] > 0]
        
        per_case_mean_dice = float(np.mean(case_dices)) if case_dices else 0.0
        per_case_std_dice = float(np.std(case_dices)) if case_dices else 0.0
        per_case_mean_iou = float(np.mean(case_ious)) if case_ious else 0.0
        per_case_std_iou = float(np.std(case_ious)) if case_ious else 0.0
        per_case_matched_mean_dice = float(np.mean(case_matched_dices)) if case_matched_dices else 0.0
        per_case_matched_std_dice = float(np.std(case_matched_dices)) if case_matched_dices else 0.0
        per_case_matched_mean_iou = float(np.mean(case_matched_ious)) if case_matched_ious else 0.0
        per_case_matched_std_iou = float(np.std(case_matched_ious)) if case_matched_ious else 0.0
        
        # Write blank row separator
        writer.writerow({})
        
        # Write overall summary (weighted by instances)
        writer.writerow({
            'case_id': 'OVERALL_WEIGHTED',
            'num_gt_instances': total_gt_instances,
            'num_pred_instances': total_pred_instances,
            'num_matched': total_matched,
            'mean_dice': overall_mean_dice,
            'mean_iou': overall_mean_iou,
            'matched_mean_dice': overall_matched_mean_dice,
            'matched_mean_iou': overall_matched_mean_iou,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1
        })
        
        # Write per-case average summary
        writer.writerow({
            'case_id': 'PER_CASE_MEAN',
            'num_gt_instances': total_cases,
            'num_pred_instances': '',
            'num_matched': '',
            'mean_dice': per_case_mean_dice,
            'mean_iou': per_case_mean_iou,
            'matched_mean_dice': per_case_matched_mean_dice,
            'matched_mean_iou': per_case_matched_mean_iou,
            'precision': '',
            'recall': '',
            'f1': ''
        })
        
        # Write per-case std summary
        writer.writerow({
            'case_id': 'PER_CASE_STD',
            'num_gt_instances': '',
            'num_pred_instances': '',
            'num_matched': '',
            'mean_dice': per_case_std_dice,
            'mean_iou': per_case_std_iou,
            'matched_mean_dice': per_case_matched_std_dice,
            'matched_mean_iou': per_case_matched_std_iou,
            'precision': '',
            'recall': '',
            'f1': ''
        })
    
    logging.info(f"\nResults saved to: {output_file}")


def print_summary_statistics(results: list[dict]):
    """
    Print summary statistics across all cases.
    
    Args:
        results: List of result dictionaries
    """
    if not results:
        logging.warning("No results to summarize")
        return
    
    logging.info(f"\n{'=' * 60}")
    logging.info("SUMMARY STATISTICS")
    logging.info(f"{'=' * 60}")
    
    total_cases = len(results)
    total_gt_instances = sum(r['num_gt_instances'] for r in results)
    total_pred_instances = sum(r['num_pred_instances'] for r in results)
    total_matched = sum(r['num_matched'] for r in results)
    
    # Aggregate metrics (weighted by total number of instances)
    weighted_dice = []
    weighted_iou = []
    weighted_matched_dice = []
    weighted_matched_iou = []
    
    for result in results:
        num_instances = result['num_gt_instances'] + result['num_pred_instances'] - result['num_matched']
        if num_instances > 0:
            weighted_dice.extend([result['mean_dice']] * num_instances)
            weighted_iou.extend([result['mean_iou']] * num_instances)
        
        if result['num_matched'] > 0:
            weighted_matched_dice.extend([result['matched_mean_dice']] * result['num_matched'])
            weighted_matched_iou.extend([result['matched_mean_iou']] * result['num_matched'])
    
    overall_mean_dice = np.mean(weighted_dice) if weighted_dice else 0.0
    overall_mean_iou = np.mean(weighted_iou) if weighted_iou else 0.0
    overall_matched_mean_dice = np.mean(weighted_matched_dice) if weighted_matched_dice else 0.0
    overall_matched_mean_iou = np.mean(weighted_matched_iou) if weighted_matched_iou else 0.0
    
    overall_precision = total_matched / total_pred_instances if total_pred_instances > 0 else 0.0
    overall_recall = total_matched / total_gt_instances if total_gt_instances > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    logging.info(f"Total cases evaluated: {total_cases}")
    logging.info(f"Total GT instances: {total_gt_instances}")
    logging.info(f"Total predicted instances: {total_pred_instances}")
    logging.info(f"Total matched instances: {total_matched}")
    logging.info(f"\nOverall Mean DICE (with FP/FN as 0): {overall_mean_dice:.4f}")
    logging.info(f"Overall Mean IoU (with FP/FN as 0): {overall_mean_iou:.4f}")
    logging.info(f"Overall Mean DICE (matched only): {overall_matched_mean_dice:.4f}")
    logging.info(f"Overall Mean IoU (matched only): {overall_matched_mean_iou:.4f}")
    logging.info(f"Overall Precision: {overall_precision:.4f}")
    logging.info(f"Overall Recall: {overall_recall:.4f}")
    logging.info(f"Overall F1-Score: {overall_f1:.4f}")
    
    # Per-case statistics
    case_dices = [r['mean_dice'] for r in results]
    case_ious = [r['mean_iou'] for r in results]
    case_matched_dices = [r['matched_mean_dice'] for r in results if r['num_matched'] > 0]
    case_matched_ious = [r['matched_mean_iou'] for r in results if r['num_matched'] > 0]
    
    if case_dices:
        logging.info(f"\nPer-case Mean DICE (with FP/FN): {np.mean(case_dices):.4f} ± {np.std(case_dices):.4f}")
        logging.info(f"Per-case Mean IoU (with FP/FN): {np.mean(case_ious):.4f} ± {np.std(case_ious):.4f}")
    if case_matched_dices:
        logging.info(f"Per-case Mean DICE (matched only): {np.mean(case_matched_dices):.4f} ± {np.std(case_matched_dices):.4f}")
        logging.info(f"Per-case Mean IoU (matched only): {np.mean(case_matched_ious):.4f} ± {np.std(case_matched_ious):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate seg_inference outputs for PENGWIN bone fragment segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with default settings
  python evaluate_seg_inference.py
  
  # Specify custom directories
  python evaluate_seg_inference.py --gt_dir ~/dev/data/10_10 --seg_inference_dir /path/to/seg_inference
  
  # Use different IoU threshold for matching
  python evaluate_seg_inference.py --iou_threshold 0.2
  
  # Save results to custom CSV file
  python evaluate_seg_inference.py --output_file seg_inference_results.csv
        """
    )
    
    # Directory arguments
    default_gt_dir = os.path.expanduser("~/dev/data/10_10")
    default_seg_inference_dir = "/ssdArray/hongyou/dev/data/seg_inference"
    default_output_file = "seg_inference_evaluation_results.csv"
    
    parser.add_argument('--gt_dir', type=str, default=default_gt_dir,
                       help=f'Ground truth directory (default: {default_gt_dir})')
    parser.add_argument('--seg_inference_dir', type=str, default=default_seg_inference_dir,
                       help=f'seg_inference output directory (default: {default_seg_inference_dir})')
    parser.add_argument('--output_file', type=str, default=default_output_file,
                       help=f'Output CSV file path (default: {default_output_file})')
    
    # Evaluation parameters
    parser.add_argument('--iou_threshold', type=float, default=0.1,
                       help='Minimum IoU threshold for instance matching (default: 0.1)')
    
    # ABBC conversion parameters
    parser.add_argument('--core_label', type=int, default=2,
                       help='Core label for ABBC conversion (default: 2)')
    parser.add_argument('--boundary_label', type=int, default=1,
                       help='Boundary label for ABBC conversion (default: 1)')
    parser.add_argument('--border_label', type=int, default=3,
                       help='Border label for ABBC conversion (default: 3)')
    
    # Processing options
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of processes for parallel ABBC conversion (default: single process)')
    
    # Logging options
    parser.add_argument('--log_file', type=str, default=None,
                       help='Log file path (default: None, only console output)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    
    # Expand paths
    gt_dir = os.path.expanduser(args.gt_dir)
    seg_inference_dir = os.path.expanduser(args.seg_inference_dir)
    
    logging.info("=" * 60)
    logging.info("seg_inference Segmentation Evaluation")
    logging.info("=" * 60)
    logging.info(f"Ground truth directory: {gt_dir}")
    logging.info(f"seg_inference directory: {seg_inference_dir}")
    logging.info(f"IoU threshold: {args.iou_threshold}")
    
    # Find matching case files
    logging.info("\nDiscovering cases...")
    case_files = find_case_files(gt_dir, seg_inference_dir)
    
    if not case_files:
        logging.error("No matching cases found!")
        sys.exit(1)
    
    logging.info(f"Found {len(case_files)} cases to evaluate")
    
    # Evaluate each case
    results = []
    successful = 0
    failed = 0
    
    for case_id, gt_file, seg_inference_case_dir in case_files:
        result = evaluate_case(
            seg_inference_dir=seg_inference_case_dir,
            gt_file=gt_file,
            case_id=case_id,
            core_label=args.core_label,
            boundary_label=args.boundary_label,
            border_label=args.border_label,
            iou_threshold=args.iou_threshold,
            processes=args.processes
        )
        
        if result is not None:
            results.append(result)
            successful += 1
        else:
            failed += 1
    
    logging.info(f"\n{'=' * 60}")
    logging.info(f"Evaluation completed: {successful} successful, {failed} failed")
    logging.info(f"{'=' * 60}")
    
    # Save results to CSV
    if results:
        save_results_to_csv(results, args.output_file)
        print_summary_statistics(results)
    else:
        logging.warning("No successful evaluations to save")


if __name__ == "__main__":
    main()

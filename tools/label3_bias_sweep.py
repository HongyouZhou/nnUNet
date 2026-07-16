import argparse
import csv
import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from torch.backends import cudnn

from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class


DEFAULT_BIASES = (
    -0.40, -0.30, -0.20, -0.10, -0.05,
    0.0,
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70,
)


def parse_biases(value: str) -> list[float]:
    if not value:
        return list(DEFAULT_BIASES)
    return [float(i.strip()) for i in value.split(',') if i.strip()]


def empty_stats() -> dict[str, int]:
    return {
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'TN': 0,
        'n_pred': 0,
        'n_ref': 0,
    }


def compute_label_stats(ref: np.ndarray, pred_mask: np.ndarray, label: int, valid_mask: np.ndarray | None) -> dict:
    ref_mask = ref == label
    if valid_mask is not None:
        ref_mask = ref_mask & valid_mask
        pred_mask = pred_mask & valid_mask

    tp = int(np.count_nonzero(ref_mask & pred_mask))
    fp = int(np.count_nonzero(~ref_mask & pred_mask))
    fn = int(np.count_nonzero(ref_mask & ~pred_mask))
    total = int(np.count_nonzero(valid_mask)) if valid_mask is not None else int(ref.size)
    tn = total - tp - fp - fn

    stats = empty_stats()
    stats.update({'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'n_pred': tp + fp, 'n_ref': tp + fn})
    if tp + fp + fn == 0:
        stats['Dice'] = float('nan')
        stats['IoU'] = float('nan')
    else:
        stats['Dice'] = 2 * tp / (2 * tp + fp + fn)
        stats['IoU'] = tp / (tp + fp + fn)
    return stats


def summarize(metric_per_case: list[dict], labels: tuple[int, ...]) -> dict:
    metric_keys = tuple(metric_per_case[0]['metrics'][str(labels[0])].keys())
    means = {}
    for label in labels:
        label_key = str(label)
        means[label_key] = {}
        for metric_key in metric_keys:
            means[label_key][metric_key] = float(
                np.nanmean([case['metrics'][label_key][metric_key] for case in metric_per_case])
            )

    foreground_mean = {}
    for metric_key in metric_keys:
        foreground_mean[metric_key] = float(np.mean([means[str(label)][metric_key] for label in labels]))

    return {
        'metric_per_case': metric_per_case,
        'mean': means,
        'foreground_mean': foreground_mean,
    }


def source_means(metric_per_case: list[dict], label: int) -> dict:
    label_key = str(label)
    result = {}
    for prefix in ('charite_', 'pengwin_'):
        values = [
            case['metrics'][label_key]['Dice']
            for case in metric_per_case
            if case['case_identifier'].startswith(prefix)
        ]
        result[prefix.rstrip('_')] = {
            'mean': float(np.nanmean(values)),
            'min': float(np.nanmin(values)),
            'max': float(np.nanmax(values)),
            'n': len(values),
        }
    return result


def run(args):
    biases = parse_biases(args.biases)
    if args.device == 'cuda':
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    device = torch.device(args.device)

    trainer = get_trainer_from_args(
        args.dataset,
        args.configuration,
        args.fold,
        args.trainer,
        args.plans,
        continue_training=False,
        device=device,
    )
    trainer.load_checkpoint(args.checkpoint)
    trainer.set_deep_supervision_enabled(False)
    trainer.network.eval()
    if trainer.dataset_class is None:
        trainer.dataset_class = infer_dataset_class(trainer.preprocessed_dataset_folder)

    if torch.cuda.is_available() and device.type == 'cuda':
        cudnn.deterministic = False
        cudnn.benchmark = True

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.manual_initialization(
        trainer.network,
        trainer.plans_manager,
        trainer.configuration_manager,
        None,
        trainer.dataset_json,
        trainer.__class__.__name__,
        trainer.inference_allowed_mirroring_axes,
    )

    _, val_keys = trainer.do_split()
    dataset_val = trainer.dataset_class(
        trainer.preprocessed_dataset_folder,
        val_keys,
        folder_with_segs_from_previous_stage=trainer.folder_with_segs_from_previous_stage,
    )

    labels = tuple(int(i) for i in trainer.label_manager.foreground_labels)
    if args.label not in labels:
        raise RuntimeError(f'label {args.label} is not in foreground labels {labels}')

    bias_cases = {bias: [] for bias in biases}
    gt_folder = Path(trainer.preprocessed_dataset_folder_base) / 'gt_segmentations'
    file_ending = trainer.dataset_json['file_ending']
    image_reader_writer = trainer.plans_manager.image_reader_writer_class()

    for idx, case_identifier in enumerate(dataset_val.identifiers):
        print(f'[{idx + 1:03d}/{len(dataset_val.identifiers):03d}] predicting {case_identifier}', flush=True)
        data, _, seg_prev, properties = dataset_val.load_case(case_identifier)
        data = data[:]
        if trainer.is_cascaded:
            raise RuntimeError('This script does not currently support cascaded validation.')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            data_tensor = torch.from_numpy(data)

        with torch.no_grad():
            logits = predictor.predict_sliding_window_return_logits(data_tensor).cpu()

        _, probabilities = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits,
            trainer.plans_manager,
            trainer.configuration_manager,
            trainer.label_manager,
            properties,
            return_probabilities=True,
        )
        del logits, data_tensor, data

        reference, _ = image_reader_writer.read_seg(str(gt_folder / f'{case_identifier}{file_ending}'))
        reference = reference[0].astype(np.int16, copy=False)
        if probabilities.shape[1:] != reference.shape:
            raise RuntimeError(
                f'Shape mismatch for {case_identifier}: probabilities={probabilities.shape[1:]}, '
                f'reference={reference.shape}'
            )

        valid_mask = None
        if trainer.label_manager.ignore_label is not None:
            valid_mask = reference != trainer.label_manager.ignore_label

        label_probability = probabilities[args.label].copy()
        other_indices = [i for i in range(probabilities.shape[0]) if i != args.label]
        other_max = probabilities[other_indices[0]].copy()
        other_argmax = np.full(other_max.shape, other_indices[0], dtype=np.uint8)
        for other_index in other_indices[1:]:
            other_probability = probabilities[other_index]
            update_mask = other_probability > other_max
            other_max[update_mask] = other_probability[update_mask]
            other_argmax[update_mask] = other_index
        del probabilities

        for bias in biases:
            factor = math.exp(bias)
            pred_label_mask = (label_probability * factor) >= other_max
            case_metrics = {}
            for label in labels:
                if label == args.label:
                    pred_mask = pred_label_mask
                else:
                    pred_mask = (~pred_label_mask) & (other_argmax == label)
                case_metrics[str(label)] = compute_label_stats(reference, pred_mask, label, valid_mask)

            bias_cases[bias].append({
                'case_identifier': case_identifier,
                'reference_file': str(gt_folder / f'{case_identifier}{file_ending}'),
                'prediction_file': f'label{args.label}_bias_{bias:+.3f}/{case_identifier}{file_ending}',
                'metrics': case_metrics,
            })

        del reference, label_probability, other_argmax, other_max
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for bias in biases:
        summary = summarize(bias_cases[bias], labels)
        summary['label3_bias'] = bias
        summary['label3_factor'] = math.exp(bias)
        summary['source_means'] = source_means(summary['metric_per_case'], args.label)
        summary_path = output_dir / f'summary_label{args.label}_bias_{bias:+.3f}.json'
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        summaries.append(summary)

    ranking = sorted(
        summaries,
        key=lambda item: (item['mean'][str(args.label)]['Dice'], item['foreground_mean']['Dice']),
        reverse=True,
    )
    compact = {
        'dataset': args.dataset,
        'configuration': args.configuration,
        'fold': args.fold,
        'trainer': args.trainer,
        'plans': args.plans,
        'checkpoint': args.checkpoint,
        'label': args.label,
        'biases': biases,
        'best_by_label_dice': {
            'bias': ranking[0]['label3_bias'],
            'factor': ranking[0]['label3_factor'],
            'foreground_dice': ranking[0]['foreground_mean']['Dice'],
            'label1_dice': ranking[0]['mean']['1']['Dice'],
            'label2_dice': ranking[0]['mean']['2']['Dice'],
            'label3_dice': ranking[0]['mean'][str(args.label)]['Dice'],
            'label3_iou': ranking[0]['mean'][str(args.label)]['IoU'],
            'source_means': ranking[0]['source_means'],
        },
        'rows': [
            {
                'bias': summary['label3_bias'],
                'factor': summary['label3_factor'],
                'foreground_dice': summary['foreground_mean']['Dice'],
                'foreground_iou': summary['foreground_mean']['IoU'],
                'label1_dice': summary['mean']['1']['Dice'],
                'label2_dice': summary['mean']['2']['Dice'],
                'label3_dice': summary['mean'][str(args.label)]['Dice'],
                'label3_iou': summary['mean'][str(args.label)]['IoU'],
                'charite_label3_mean': summary['source_means']['charite']['mean'],
                'pengwin_label3_mean': summary['source_means']['pengwin']['mean'],
            }
            for summary in summaries
        ],
    }
    (output_dir / 'bias_sweep_summary.json').write_text(json.dumps(compact, indent=2, sort_keys=True))

    with (output_dir / 'bias_sweep_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(compact['rows'][0].keys()))
        writer.writeheader()
        writer.writerows(compact['rows'])

    print('Best by label Dice:')
    print(json.dumps(compact['best_by_label_dice'], indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('configuration')
    parser.add_argument('fold')
    parser.add_argument('-tr', '--trainer', required=True)
    parser.add_argument('-p', '--plans', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--label', type=int, default=3)
    parser.add_argument('--biases', default=','.join(str(i) for i in DEFAULT_BIASES))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

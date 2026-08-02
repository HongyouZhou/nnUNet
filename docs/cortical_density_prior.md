# Cortical density prior experiment

This experiment keeps the Dataset778 separator network at one CT input channel
and three softmax outputs (`background`, `cortical body`, `separator`). Density
is training-only: it changes paired patch sampling and adds a soft auxiliary
separator loss. It never relabels low-HU fractures and does not alter min-cut
costs.

## Frozen training contract

- Fold calibration uses only valid class-1/class-2 voxels from that fold's
  training patients.
- `D = sigmoid((HU-Q50)/max((Q75-Q25)/2, 1 HU))`.
- Normal negatives are the inner fragment-support surface, semantic-valid and
  more than 2 mm from a GT separator.
- Control sampling is 40% separator, 40% normal surface, 20% random.
- Prior sampling is 40% separator, 20% low-density normal surface, 20%
  high-density normal surface, 20% random.
- Prior loss is standard Dice+CE plus `0.5 * L_density`. Positive separator
  weights are `1+2D`; normal-surface negative weights are `1+(1-D)`.

The independent plans name is
`nnUNetResEncUNetMPlansSeparatorDensityPrior`. Its preprocessed target contains
semantic, fragment support, validity, the frozen normal-surface mask, and an
encoded pre-augmentation HU channel. Only normalized CT enters the network.

## HPC workflow

Prepare plans, five training-fold calibrations, the OOF hard audit, and the
independent preprocessed data:

```bash
sbatch slurm/charite_cortical/prepare_density_prior.slurm
```

The job stops unless the patient-macro density ROC-AUC bootstrap 95% lower
bound is above 0.5 and the mean per-patient median density difference is
positive. If it passes, launch the four one-GPU paired pilot tasks:

```bash
sbatch slurm/charite_cortical/train_density_prior_pilot.slurm
```

After OOF inference and the unchanged automatic split pipeline, supply one
metric JSON per trainer to the gate. Each `records` item must contain
`patient_id`, `fold`, `all_child_recovery`, `intact_case`,
`intact_false_split`, and `cortical_union_dice`.

```bash
python -m tools.charite_cortical.density_prior evaluate \
  --control control-pilot-metrics.json \
  --prior prior-pilot-metrics.json \
  --output density-prior-pilot-gate.json
```

Completion requires at least +3 percentage points all-child recovery, at most
5% intact false splits, and no more than 1 percentage point cortical-union Dice
loss. Then submit the remaining paired folds:

```bash
sbatch \
  --export=ALL,CORTICAL_DENSITY_PILOT_GATE=/absolute/path/density-prior-pilot-gate.json \
  slurm/charite_cortical/train_density_prior_remaining.slurm
```

All control/prior inference and splitting must use identical thresholds and
unchanged graph parameters. Local training and validation are intentionally
not part of this workflow.

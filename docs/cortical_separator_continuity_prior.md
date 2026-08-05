# Voxel-space cortical continuity separator pilot

This pilot keeps the ResEnc nnU-Net M network unchanged: one CT input and
three softmax classes (`0 background`, `1 cortical body`, `2 separator`).
Ground-truth cortical instance IDs and HU are training-only target channels;
inference still emits the ordinary separator probability consumed by the
existing min-max cut workflow.

Instance IDs use nearest-neighbour resampling. The HU code is reconstructed
once from the normalized/resampled CT before augmentation and stored with the
target, so intensity augmentation cannot leak into density supervision.

The independent plans name is
`nnUNetResEncUNetMPlansSeparatorContinuityPrior`. Its preprocessing identifier
is separate from all earlier density-prior artifacts. The frozen target order
is:

```text
[semantic, support, validity, normal_surface, hu_code, cortical_instance_id]
```

`normal_surface` is the union of boundaries extracted independently from each
positive cortical instance, restricted to semantic-valid voxels farther than
2 mm from the separator. Extracting each boundary before union preserves the
interface between touching instances.

## Pair and loss contract

All six target channels use one joint nearest-neighbour spatial transform for
rotation and scaling. The trainer validates this at startup; the generic
nnU-Net categorical interpolation remains unchanged for unrelated trainers.
Pairs are rebuilt from the augmented instance channel at loss time. There are
13 undirected local half-edges and three axis-aligned two-voxel edges, giving
16 channels at the frozen 0.5 mm spacing. A pair is usable only when both
endpoints have positive ownership, relation-valid bit 1 is set, neither is the
semantic ignore label, and the complete lifted path is inside the patch.

For a path, `q_cut` is the maximum separator probability on the path. Same-
instance and different-instance BCE terms are averaged separately. Conditional
density uses only pairs whose two endpoints are separator or normal-surface
voxels and uses the mean of the two fold-calibrated endpoint scores—never gap
HU. The three arms are:

```text
nnUNetTrainerCorticalSeparatorMatchedBase
  Dice + CE

nnUNetTrainerCorticalSeparatorContinuity
  Dice + CE + 0.1 continuity

nnUNetTrainerCorticalSeparatorContinuityDensityPrior
  Dice + CE + 0.1 continuity + 0.1 conditional density
```

All arms use 40% separator, 40% normal-surface, and 20% random patch centres,
1000 epochs, and checkpoints every 50 epochs. The two regularizers run only on
the finest output; deep-supervision branches retain Dice+CE.

## HPC workflow

Commit and push local changes first. On HPC, the scripts use a fixed checkout
and `git pull --ff-only`; do not edit the HPC checkout directly.

After preparation, run the fail-closed performance smoke before submitting a
formal pilot:

```bash
sbatch slurm/charite_cortical/prepare_separator_continuity_prior.slurm
bash slurm/charite_cortical/submit_separator_continuity_smoke.sh
```

Preparation requests 256 GB CPU memory for 48 hours, builds the independent
plans and 68-case target, produces five fold calibrations, and writes the raw-
density AUC JSON. The audit is diagnostic-only: `training_allowed=false` is
recorded but never blocks preprocessing or the pilot. The smoke runs two full
fold-0 epochs for each arm in separate output folders: epoch 0 absorbs Torch
compile, and epoch 1 must finish within 120 seconds. It never submits the
formal pilot automatically.
The pilot is a one-GPU
`0-5%4` array with folds 0/1 for matched baseline, continuity, and
continuity+density; `--c` resumes a 48-hour task from its periodic checkpoint.

After unchanged inference, separator threshold, min-max cut parameters, and
downstream evaluation, apply the same gate to each increment:

```bash
python -m tools.charite_cortical.density_prior increment-gate \
  --reference matched-base.json --candidate continuity.json \
  --comparison-name continuity_vs_matched_base \
  --output continuity-gate.json

python -m tools.charite_cortical.density_prior increment-gate \
  --reference continuity.json --candidate continuity-density.json \
  --comparison-name density_vs_continuity \
  --output density-gate.json
```

Each increment needs at least +3 percentage points all-child recovery, no more
than 5% intact false splits, and no more than 1 percentage point loss in
cortical-union Dice before folds 2–4. Submit the remaining arms with:

```bash
CORTICAL_CONTINUITY_GATE=/absolute/path/continuity-gate.json \
CORTICAL_DENSITY_GATE=/absolute/path/density-gate.json \
bash slurm/charite_cortical/submit_separator_continuity_remaining.sh
```

Matched baseline folds 2–4 always run; each regularized arm is submitted only
when its corresponding increment passed. This workflow does not invoke a C+A
predictor, Axial19/Dense39 MWS, or an oracle training gate.

## Complete Slurm DAG

The complete workflow can attach to an already submitted pilot array or submit
one itself, but requires a successful smoke generated from the exact current
Git revision. A dry-run does not submit jobs:

```bash
PROJECT_HOME=/sc-projects/sc-proj-cc09-repair/hongyou \
CORTICAL_REPO_DIR=$PROJECT_HOME/dev/nnUNet-separator-continuity \
bash slurm/charite_cortical/submit_separator_continuity_dag.sh \
  --smoke-run-dir /absolute/path/to/passed-smoke-run \
  --pilot-job 10267993 \
  --dry-run
```

Remove `--dry-run` to create a run contract and attach the controller. The
controller checks all six `checkpoint_final.pth` files after the pilot array.
Missing tasks receive at most three bounded retries with 128 GB host memory.
Training jobs requeue five minutes before the 48-hour limit and resume through
`--c`.

The fail-closed graph is:

```text
HPC tests -> three-arm compile+measured-epoch smoke -> 120-second throughput gate
  -> explicit formal DAG submission
  -> pilot checkpoint guard
  -> OOF inference (3 arms x folds 0/1)
  -> separator minimax watershed and downstream evaluation
  -> continuity and density increment gates
  -> conditional folds 2-4 training
  -> folds 2-4 inference and evaluation
  -> per-arm fivefold merge
  -> final-summary.json
```

The frozen postprocessor uses the predicted cortical union as its mask,
connected low-separator regions (`p_sep < 0.5`) as markers, and `p_sep` as the
minimum-maximum watershed barrier. GT cortical instances are opened only by
the evaluation job. Every job ID is recorded in `<run-dir>/jobs.json`; metrics,
gate decisions, predictions, and final summary remain under the same run
directory.

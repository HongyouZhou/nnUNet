# Cortical geometry: diagnosis and next mechanism test

## Scope and interpretation

The user's intended mechanism is cortical **geometric continuity** as the
strongest evidence of fragment ownership. Two pieces can remain voxel-connected
while a surface step, kink, truncation or malalignment supports different
ownership. Conversely, holes in a damaged cortical shell do not necessarily
imply different fragments.

The original `docs/00.md` explicitly included a partial crack whose bottom is
still connected. The existing separator-continuity pilot instead tested local
0.5–1 mm relation supervision on a three-class network, with a frozen
low-separator-component marker generator. It is an initial ablation, not a
conclusive test of the intended geometric mechanism. Its negative increment
gates terminate those particular loss expansions, not the overall research
direction.

The original output target was per-fragment cortical instances. Extending these
instances into full bone interiors is a later step; lack of that extension alone
does not invalidate the original cortical instance task. The current mechanism
gap is that postprocessing cannot use independent cortical ownership evidence
to reconsider fragments already merged into one marker.

## Frozen-result diagnosis

Run `tools.charite_cortical.diagnose_continuity` with
`slurm/charite_cortical/diagnose_separator_continuity.slurm`. Tasks 0–4 cover all
68 baseline OOF cases; tasks 5–8 cover the same 28 pilot cases for the two loss
variants. All compute and tests run on HPC. Source data, models, predictions,
and the original postprocessing metrics stay frozen.

Each case measures:

- Coverage of each relation-valid cortical GT instance by the predicted union.
  Below 50% coverage, even perfect assignment of covered voxels cannot reach
  IoU 0.5. The reported coverage upper bound excludes all false positives and
  uses GT ownership; it is deliberately optimistic and not a method score.
- Reconstructed markers, including the original 10-voxel filter and fallback.
  A marker is mixed if it contains at least 10% of each of two GT instances.
- Mixed output instances and connected union components, measured with the
  same overlap criterion.
- Annotation pairs that touch in the frozen 26-neighbour convention, including
  pairs sharing a substantial marker. Annotation contact is a screening tool,
  not clinical proof of a surviving anatomical bridge. Inspect the CT before
  interpreting a particular candidate.

Flags can overlap. Neither marker mixing nor coverage loss alone proves the
complete causal explanation of all unrecovered instances. Diagnostic metrics
must reproduce every original per-case metric exactly before a case is accepted.
Arm comparisons use the identical 28-case pilot population; the 68-case baseline
is reported separately.

Representative cases receive a GT-assisted seed counterfactual. Keep both the
predicted cortical mask and the separator field fixed; supply separate seed IDs
using the lowest-separator-probability 10% of covered voxels of each GT instance.
Run the same watershed. Improvement shows that independent ownership markers
can help; it does **not** show that geometric cues can generate those markers
automatically. The CT panels, coordinates and small aligned NIfTI ROIs are saved
for review.

## Following mechanism prototype

First test automatic ownership evidence on the selected connected-fragment
cases, before another training sweep:

1. Derive cortical surface patches and physical-scale features from predicted
   cortex and CT: local tangent consistency, surface fit residual, truncation
   and offset evidence. Validate feature behaviour at normal curved cortex and
   thin cortex as controls. A generic high-curvature threshold is insufficient.
2. Build cortical patch relations with multiple physical scales. Distinguish
   normal surface continuation from fracture evidence. Use separator probability
   as supporting evidence; a single low-probability bridge must not force a
   whole connected region to have one ownership label.
3. Partition the surface graph with multiple ownership candidates per connected
   cortical component. Strong, reliable cortical relations constrain ownership;
   uncertain or disrupted patches remain revisable. Do not equate high HU with
   a rule that cortex must never be cut.
4. Once cortical ownership works, propagate those labels into a complete bone
   mask when full fragment volumes are needed. Interior CT and distance can
   guide the partition, while clear cortical ownership constrains it.

The existing `tools/PENGWIN/postprocess/cortical_anchored_split.py` contains
surface seed and graph-cut infrastructure, but targets four ABBC classes and
has different costs. It cannot be plugged into the three-class pilot by merely
renaming labels. Its solver can be reused after checking seed semantics,
physical spacing, memory and the ability to cut through a residual bridge.

Any prototype developed against these diagnosed OOF cases is exploratory.
Freeze its choices before independent evaluation; do not present tuned-case
improvement as an unbiased OOF result. The next acceptance test must include
both connected fragments that should split and normal continuous cortex that
should remain intact, plus small/thin fragments. Whole-case all-child recovery
alone is too coarse to diagnose this mechanism; report per-pair split success,
child recovery and false splits together.

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

The CT network may still learn implicit geometric information. The absence of
an explicit surface-ownership mechanism is not evidence that it learns no
geometry, and the frozen implementation is not a coding error merely because
it tests a narrower surrogate.

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

## Completed baseline diagnosis (2026-10-05)

All 68 baseline cases reproduced their frozen per-case metrics. 67/68 cases
contain mixed markers, defined by at least 10% overlap with each of two GT
instances. Of 517 children, 427 participate in mixed markers; 401 of those have
at least 50% predicted cortical coverage. 354 adequately covered mixed-marker
children have no predicted instance with individual IoU at least 0.5.

78 children have coverage below 50%, so missing cortex is another genuine
failure mode. Flags overlap. An optimistic GT-ownership assignment that removes
all false positives yields a patient-macro coverage-based recovery upper bound
of 0.8859226, compared with the actual recovery of 0.1783685. This is a diagnosis
of available support, not an achievable or deployable method result.

Three selected GT-assisted marker experiments improved recovery counts from
1/6 to 6/6, 1/4 to 3/4, and 4/19 to 13/19 with the cortical union and separator
field fixed. The last counterfactual also false-split two GT children; independent
markers therefore do not resolve every assignment problem. CT panels and aligned
NIfTI ROIs remain in the experiment artifacts, outside the source repository.

This supports testing independent cortical ownership within a connected region,
rather than treating any low-separator bridge as a compulsory merge. It does not
yet establish that automatic geometry can supply the necessary ownership.

On the same 28 pilot cases (222 children), all three arms have mixed markers
in all 28 cases. The baseline, continuity and continuity+density arms have
180, 189 and 190 children in mixed markers, respectively; 171, 175 and 176 of
those children have at least 50% predicted coverage. Thus the local loss
increments did not remove this marker failure mode.

Two automatic seed controls kept the union and separator field frozen and used
no GT for seed generation: connected predicted cortical-body voxels (argmax
class 1), or low-separator voxels with threshold 0.3. Both retained 26 neighbours
and the 10-voxel seed filter. Recovery stayed at 1/6 and 1/4 in the first two
selected cases and improved only from 4/19 to 5/19 in the third. These selected
case controls do not establish a tuned method's population performance, but
show that those simple marker changes did not reproduce the oracle gains.

## Additional loss-representation audit

The current pair generator includes same-instance pairs whose endpoints are
semantic separator voxels. For instance IDs `[A, A, B, B]`, a perfect semantic
separator field `[0, 1, 1, 0]` gives local maximum-barrier scores `[1, 1, 1]`.
The instance-relation cut targets are `[0, 1, 0]`. The two within-instance edges
therefore oppose the semantic separator target. This is an objective tension
for that arrangement, not evidence of a particular contribution to actual
training failure. Its frequency on augmented training samples has not yet
been measured.

An independently predicted directional edge relation can represent `[0, 1, 0]`
without forcing the separator voxel probability to zero. This is a candidate
representation for the next mechanism test, rather than another increase in
the weight of the same maximum-voxel-probability surrogate.

Neither local smoothness nor high HU should become an unconditional must-link:
a surviving hinge can be locally smooth while the surrounding fracture and
surface context support a global split. Reliable cortical ownership relations
must account for that context. Similarly, a normal anatomical bend must not
become a fracture just because curvature is high.

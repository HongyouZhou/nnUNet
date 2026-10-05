# Automatic cortical ownership mechanism prototype

This development stage follows the completed frozen-output diagnosis. It keeps
the network, native cortical union and separator field fixed. It tests an
independent CT geometry partition inside frozen separator instances. Inference
uses CT, predicted cortex, probabilities and physical spacing only. GT is read
after the prediction is saved, for evaluation and inspection.

## Mechanism

1. Average CT gradient structure tensors at 0.9 and 2.7 mm, after 0.5 mm
   derivative smoothing. The principal direction describes a local cortical
   sheet normal. Its sign is irrelevant; inner and outer shell gradients agree.
2. Form connected patches inside 1.25 mm grid cells, using actual 26-neighbour
   contacts. Separate sheets within one cell remain separate patches. All
   calculations use native physical spacing; sheared grids are rejected.
3. Mark a patch relation as repulsive when reliable fine-scale normals differ
   by at least 20 degrees and exceed the contextual angle by at least 8 degrees.
   Require four connected repulsive edges to reject isolated direction outliers.
   Smooth curvature has similar angular variation at both scales. Low-coherence
   corners cannot independently supply repulsion. These are initial engineering
   choices, not clinically calibrated thresholds.
4. Agglomerate positive patch relations in descending confidence order while
   enforcing all supported repulsive relations globally. A smooth surviving
   hinge or an alternative graph route cannot override a repulsive constraint.
   Smooth cortical edges are strong merge preferences but remain revisable.
   Separator probability contributes at most 15% to merge ordering. HU alone
   never supplies a mandatory merge or a prohibition on cutting.
5. Assign every original cortical voxel a patch ownership. Union coverage and
   union Dice must remain identical; frozen coarse instances cannot merge.

This is a greedy signed-graph approximation, with local geometric constraints
enforced globally. It is not a learned anatomical ownership model. In
particular, the initial direction cue can miss parallel offset/truncated sheets
and a smooth hinge without any observed surrounding direction discontinuity.
The multiscale curvature safeguard cannot certify that a sharp anatomical crease
is a fracture, and noisy or crowded sheets can create incorrect constraints.
The implementation therefore tests a specific part of the original idea, not
the entire intended method.

## Validation and execution

Selected development cases are charite_1434/fold 0, charite_18/fold 1 and
charite_103/fold 1, using matched_base frozen OOF predictions. Their selection
comes from the earlier GT-assisted diagnosis, so results are exploratory.
Parameters above are recorded before this first automatic run. There is no
parameter search against these cases and no claim of independent OOF validation.

HPC tests include graph-level residual-bridge separation, randomized constraint
preservation, sign invariance, connected patch correctness, smooth CT curvature
on isotropic and anisotropic grids, and exact sparse/whole-grid metric equality.
Case reports include frozen vs automatic child recovery and false splits,
dominant-owner separation for annotated touching mixed-marker pairs, and up to
three same-GT 8 mm regions at least 10 mm from other annotated instances.
These regions are annotation-based continuity controls, not clinical proof of
intact cortex. Their sphere and tensor context must lie inside the CT scan, to
exclude acquisition-end truncation. Native three-plane CT figures show a diagnosed contact and a
same-instance control. No GT changes the automatic crop, constraints or labels.

Run `slurm/charite_cortical/geometry_ownership.slurm` after the HPC checks pass;
then run `tools.charite_cortical.run_geometry_ownership --summary` with the same
dataset/run/diagnosis/output arguments. Artifacts are isolated from the previous
DAG. This postprocessing step needs CPUs, not another GPU training allocation.
All A100 variants remain eligible for any later training stage.

Before expansion, inspect both selected contact separation and normal-region
false splits. Geometry that breaks a connection but fragments the same cortex
is not a successful ownership model. A negative result should guide explicit
offset/truncation or learned relation evidence rather than a larger training
sweep with the same separator surrogate.

## First automatic mechanism result (2026-10-05)

All three selected cases completed on HPC. The v1 direction-based hard
constraints failed the ownership mechanism test:

| Case | Frozen recovery | Automatic recovery | Frozen false-split children | Automatic false-split children |
| --- | --- | --- | --- | --- |
| charite_1434 | 1/6 | 1/6 | 0/6 | 4/6 |
| charite_18 | 1/4 | 1/4 | 0/4 | 1/4 |
| charite_103 | 4/19 | 4/19 | 0/19 | 9/19 |

Recovery stayed at 6/29, while false-split children rose from 0/29 to 14/29.
All three selected contact pairs failed the predeclared dominant-ownership
separation criterion. Cortical union Dice stayed exactly unchanged. Hundreds
of extra pieces are produced, rather than coherent per-fragment shells.

The same parameters preserve an ideal planar or smoothly curved synthetic CT
sheet and split a sharp two-plane synthetic sheet. A more complex synthetic
partial hinge produces eight pieces and false-splits both stipulated owners.
The synthetic sharp-sheet ownership is stipulated; it is not clinical evidence
that an intact sharp anatomical ridge should be cut. Graph constraint tests
prove that a cannot-link survives an alternative bridge, but passing those
tests does not validate the geometry that supplies the cannot-link.

Initial source: `9793edfbc0d67cd9098ef2cf8110553a3e0c392d`. Artifacts:
`/sc-projects/sc-proj-cc09-repair/hongyou/dev/data/cortical_geometry_ownership/20261005_9793edf`.
Tests 11080166, cases 11080167_0–2, summary 11080168 and synthetic diagnostic
11080175 all completed with exit 0:0. The initial suite had 152 passing checks.

The follow-up annotation audit saves the prediction-derived patch graph before
reading GT. It evaluates only patches with at least 50% relation-valid owned
voxels and at least 80% ownership purity among those voxels. It reports both
audited and excluded edges, and how many geometric repulsions fall within the
same GT fragment. Its replay must match every v1 ownership voxel and frozen
metric exactly. This audits the existing failure; it does not tune or improve
the partition.

The follow-up completed at source
`5ccd7caaf0d4626ac43d2a991a3422e557e3b187`, with 156 HPC checks passing.
The ownership maps and frozen/automatic metrics replayed voxel-exactly.
Tests 11080369, replay cases 11080370_0–2, and audit 11080371 all completed
with exit 0:0. Final artifacts:
`/sc-projects/sc-proj-cc09-repair/hongyou/dev/data/cortical_geometry_ownership/20261005_5ccd7ca`.

| Case | All repulsive edges | Auditable repulsive edges | Repulsion within the same GT child | Repulsion across different GT children |
| --- | --- | --- | --- | --- |
| charite_1434 | 44,560 | 25,738 | 25,630 | 108 |
| charite_18 | 53,675 | 19,365 | 19,206 | 159 |
| charite_103 | 23,648 | 17,273 | 17,245 | 28 |

99.2–99.8% of the auditable repulsive edges fall inside one annotated child.
Excluded edges have insufficient valid ownership or label purity; this fraction
must not be applied to them or interpreted as an independent population score.
Edges are correlated. Normal-direction features can contain signal, but these
thresholded constraints have insufficient ownership specificity.

The corrected same-instance controls exclude scan boundaries and nearby other
annotated children: 2/3 controls in charite_1434, 0/2 in charite_18, and 2/2 in
charite_103 are false-split. Thus the over-splitting cannot be attributed solely
to control selection at acquisition ends. These remain annotation-based proxy
controls, not clinically verified normal cortex.

## Consequence for the next ownership model

Do not expand v1 to a population experiment or train another separator-loss
increment on the assumption that the mechanism works. Reliable CT gradient
direction is not the same quantity as reliable cortical fragment ownership.
The user's intended strongest evidence is the continuity of a coherent
per-fragment cortical shell, including its thickness, surface trajectory,
truncation and offset, with context around a surviving bridge.

The next candidate needs actual sheet/region hypotheses that avoid assigning
different owners to layers of one cortical thickness. Relations should consider
surface continuation and mismatch across multiple patches, and remain uncertain
when a normal anatomical bend or image artifact explains the observation.
Directly supervised directional/patch ownership relations can express same-GT
connections even on semantic separator voxels without the earlier max-voxel
barrier objective conflict. They must be independently calibrated on training
data, rather than making every thresholded normal change an inviolable cut.

For a learned candidate, each outer fold must exclude its validation subjects
from the entire pipeline, including geometry calibration and any backbone
providing training features. Reusing training features from other OOF backbones
that saw the held-out subjects would not establish that exclusion. Inference
must keep GT absent from crops, candidate counts, sheet models and graph labels.
First retest partial-bridge positives, same-shell curvature/thickness controls,
and small/thin fragments. Full-bone propagation remains later. No learned
ownership training was submitted as part of this v1 mechanism test.

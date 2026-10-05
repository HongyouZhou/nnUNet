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
intact cortex. Native three-plane CT figures show a diagnosed contact and a
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

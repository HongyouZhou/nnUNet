# 基于 cortical 连续性的骨碎片分割实施计划

## 1. 目标与边界

本实施的最终输出是完整骨碎片的 category-agnostic 3D instance map。Expert cortical
不是最终体积标签，而是碎片身份的主要结构证据：

```text
CT
  -> provisional ABBC instance
  -> cortical 连续性预测
  -> unknown-K cortical grouping
  -> 保守的 split/stop
  -> provisional support 内的被动体积传播
  -> 完整 fragment instances
```

首版只修复一个 provisional ABBC instance 内的粘连/欠分割，不跨 provisional instance
做 merge。模型不能使用 fragment GT、真实碎片数、人工 seed 或人工 split 决策。低密度内部
不能决定碎片数，也不能推翻 cortical grouping；它只接受已经冻结的 cortical identity。

## 2. 已冻结的实现选择

| 项目 | 决定 |
|---|---|
| 数据 | `cortical_dataset_68_final_20260728` 的全部 68 例 |
| 评价性质 | 固定五折 OOF development evaluation；没有 locked test，不作 confirmatory claim |
| 五折 | patient/case-level、cohort-stratified、`shuffle=True`、`random_state=20260729` |
| 最终输出 | cortical instances 和完整 fragment instances 均输出 |
| provisional support | 冻结的现有 ABBC 模型和 ABBC-to-instance 转换 |
| v1 修复范围 | 每个 provisional instance 内 split-only |
| 主结构证据 | `C` cortical semantic 和 `A` same-fragment affinity |
| `R` | 从不同 cortical instance 的物理邻近关系派生的 contact target，不称为 expert fracture rim |
| unknown-K solver | deterministic Mutex Watershed |
| split/stop | 保守 abstention；development intact false-split 上限 5% |
| baseline 顺序 | 先实现三分类 cortical separator，再实现 `C+A` |
| 正式初始化 | Charité scratch 为主；mixed Dataset777 checkpoint 仅工程 smoke test |

所有距离、margin、最小支持量和形态学操作均使用毫米或立方毫米，不使用固定 voxel-count
阈值。现有数据包含小 cortical instance 和高度各向异性病例，固定 voxel 阈值会产生系统性
偏差。

## 3. 数据构建

### 3.1 Dataset778 目录

构建版本名为 `Dataset778_ChariteCorticalContinuity`。原始 final68 数据只读，输出采用
manifest 驱动并记录输入 SHA-256：

```text
Dataset778_ChariteCorticalContinuity/
|-- imagesTr/
|   `-- <case>_0000.nii.gz
|-- labelsTr/
|   `-- <case>.nii.gz
|-- corticalInstancesTr/
|   `-- <case>.nii.gz
|-- corticalOverlapTr/
|   `-- <case>.nii.gz
|-- fragmentInstancesTr/
|   `-- <case>.nii.gz
|-- fragmentOverlapTr/
|   `-- <case>.nii.gz
|-- supportTr/
|   `-- <case>.nii.gz
|-- validMasksTr/
|   `-- <case>.nii.gz
|-- rimContactTr/
|   `-- <case>.nii.gz
|-- metadataTr/
|   `-- <case>.json
|-- dataset.json
|-- continuity_manifest.json
`-- splits_final.json
```

`labelsTr` 是三分类 separator baseline：

```text
0 non-cortex
1 cortical body
2 different-fragment cortical contact/separator
3 no-reliable-cortex/ambiguous fragment ignore（由 dataset.json 显式声明，不是 cortex）
```

扫描背景和 support 外区域仍为 0；不能把整幅图填成 ignore=3，因为 nnU-Net fingerprint
会将所有正 label（包括 ignore 值）纳入 foreground intensity 统计。只有未可靠标注的 fragment
support/归属歧义区使用 3。

每例按 canonical fragment name 排序生成稳定、连续的 case-local `int16` ID。cortical
复用对应 fragment 的 ID。跨 NRRD layer 重叠 voxel 的处理规则固定为：

- semantic `C=1`；
- `cortical_instance_id=0`；
- `A`、`R` 和 instance evaluation 均 ignore；
- 不按 layer priority 强制分配所有权。

`validMasksTr` 是版本化 bitmask：bit 0 为 `C` supervision valid，bit 1 为 relation
supervision valid，bit 2 为 `R_contact` supervision valid。loader 必须按位解析，不能把任意
非零值简单转换成一个共同的 boolean valid mask。

`fragmentInstancesTr` 保存无重叠 voxel 的稳定 full-fragment owner ID；
`fragmentOverlapTr` 单独保存多重归属。两者只用于 oracle/评价，不进入 `C+A` 模型输入，
避免 full-volume GT 泄漏进 identity prediction。

10 个没有可靠 cortical 的 fragment 不作为 `C=0` 负样本，其区域在 cortical relation
supervision 中 ignore。CT38 的 synthetic fragment 只参与 cortical identity 诊断，不进入
正常 full-volume 指标。

### 3.2 物理空间与 target

初始工作 spacing 为 `0.5 mm isotropic`。构建阶段必须输出 thin-cortex retention QA；若
重采样后任何 instance 消失则硬失败。粗层病例在低于原生层厚的 through-plane relation 上
不产生监督。

实现中 preprocessor 在 `[C,I,O,U,S]` 后附加三个 case-constant
`native_min_steps_zyx` channel。动态 affinity 会逐 offset 检查其轴向步长，低于原生分辨率的
方向整 channel ignore。为保证这些物理轴在增强后仍有意义，v1 禁止任意旋转增强，保留
mirroring、crop 和 intensity augmentation。

`C` 是任意 cortical mask 的 union。`A` 必须在同步空间增强之后，从变换后的 instance map
动态生成：

- 13 个 half-26-neighbour local offsets；
- 三个正轴上最接近 3 mm 和 6 mm 的 lifted offsets；
- 去重后最多 19 个 affinity channel；
- 两端为同一有效 cortical ID：positive；
- 两端为不同有效 cortical ID：negative；
- 背景、overlap、unknown、crop border：ignore。

若 O2 在 19-offset schema 下 exact-K/identity grouping 低于 95%，先切换预定义的
39-direction schema 重跑 O2；第二次仍未达到 gate 则暂停神经训练，修复 graph coverage。

派生 `R_contact`：

```text
positive: cortex i 到任意不同 cortex j 的距离 <= 2 mm
negative: cortex i 到所有不同 cortex 的距离 >= 4 mm
ignore:   2--4 mm、overlap、no-cortex、不可分辨 through-plane
```

## 4. 分阶段模型

### 4.1 MVP-0：三分类 separator

使用标准 nnU-Net v2 3D full-resolution ResEnc M：

```text
CT -> softmax(non-cortex, cortical-body, separator)
```

后处理使用 `P(C)=P(body)+P(separator)`，在高置信 separator 上断开后做 connected
components/watershed。该模型是必须报告的、无需 directional affinity 的可复现 baseline。

### 4.2 MVP-1：主模型 `C+A`

使用 ResEnc M 和版本化 flat-logit schema，避免修改 nnU-Net 通用 predictor 的单 activation
假设：

```text
flat logits = [C:1, A:19]  # 默认总计 20 channels
```

schema 保存 head slice、activation、voxel/mm offsets、valid policy 和 mirror policy。模型
训练关闭 deep supervision；directional affinity 推理关闭 mirror TTA。fold 和 sliding-window
先平均 logits，再逐 head sigmoid。

损失：

```text
L = masked_dice_bce(C) + balanced_bce(A)
```

affinity positive/negative 分别求均值后各占 0.5，空 valid-edge patch 返回有限的可求导零损失。
训练采样按 case 均衡，再按以下 patch 类型采样：

- 40% cortical contact/hard-negative；
- 30% 先均匀选择 cortical instance，再选择该 instance voxel；
- 20% provisional support 内；
- 10% cropped image 内随机。

正式初始 schedule 为 500 epochs、250 iterations/epoch，五折 scratch。现有 mixed
Dataset777 checkpoint 只允许用于确认权重加载、前向和 artifact 链路，不进入正式比较。

### 4.3 MVP-2：条件加入 `R_contact`

仅当 `C+A` 已通过 oracle/OOF gate 后训练：

```text
flat logits = [C, A, R]
L = 1.0 * L_C + 1.0 * L_A + 0.5 * L_R
```

`R` 使用 masked Dice + focal BCE (`gamma=2`)。只有当它在相同 OOF protocol 下提升
all-child recovery/PQ 且不突破 false-split 安全约束时才保留。

## 5. 推理与后处理

### 5.1 固定输入契约

模型工作网格上的输入：

```text
P: provisional full-volume instances [Z,Y,X]
C: cortical probability [Z,Y,X]
A: same-fragment affinity logits/probabilities [E,Z,Y,X]
O: directional offsets [E,3]
R: optional contact probability [Z,Y,X]
spacing_zyx
```

正式 provisional baseline 固定为：

```text
nnUNetTrainer_L3SamplingCE3_ChariteV3FineTune150
__nnUNetResEncUNetMPlans__3d_fullres
fold_all / checkpoint_final
```

每次 run 必须记录实际 model folder、plans、checkpoint SHA-256、ABBC-to-instance 配置和
代码 revision；不能只记录上述人类可读名称。

### 5.2 Cortical graph

每个 `P == provisional_id` 独立处理：

1. hysteresis support：`C_low=0.30`、`C_high=0.60`，只保留连接到 high anchor 的 low
   support；
2. 按 schema offsets 构建 local/lifted graph；
3. 校准后 `p_same >= 0.70` 为 attractive，`p_same <= 0.30` 为 repulsive，中间丢弃；
4. edge weight 乘以两端 cortex confidence；
5. deterministic Mutex Watershed 一次得到 unknown `K_raw`；
6. 相同权重以 repulsive 优先，再按 canonical endpoints 打破平局。

资源超限、无可靠 cortex、只有不稳定 singleton、传播 invariant 失败时必须返回明确
`abstain_*` reason，并保留原 instance，不能静默换算法。

### 5.3 Split/stop

`K_raw<=1` 直接 no-op。`K_raw>=2` 生成完整 multiway proposal，v1 不递归接受部分 binary
split。固定特征的 L2 logistic scorer 使用：

- normalized signed-graph energy gain；
- attractive-cut/repulsive-uncut violation；
- 每个 cluster 的 high-confidence cortical support；
- cluster support 的最小值/中位数；
- threshold perturbation 下的 partition stability；
- propagation 后最小 child 比例、无 seed component 和 connectivity violations；
- 可选 `R_contact` cut support。

训练标签表示“当前 proposal 是否值得接受”，而不是只表示 GT 是否有多个 fragment。阈值
在 OOF development 上选择，并强制 intact false-split `<=5%`。不存在满足约束的阈值时，
正式配置为 always-abstain/no-go。

### 5.4 完整体积传播

接受的 cortical cluster 是不可合并的固定 seed。在原 provisional support 内运行
spacing-aware uniform-cost multi-source geodesic Voronoi。CT、HU、ABBC core 和低密度
内部异常均不得改变 `K` 或 seed identity。

必须逐 instance 验证：

```text
union(children) == input provisional support
children pairwise disjoint
每个 foreground voxel 恰好归属一次
不新增 foreground
每个 cortical seed 保持自己的 child ID
```

未切 instance 保留原 ID。接受 split 后最大体积 child 保留原 ID，其余 child 按最小
cortical linear index 稳定排序，从全局最大 ID 后依次分配。输出统一为 `uint32`。

## 6. Oracle 与评价

在正式训练前依次运行：

| Oracle | Cortex/grouping | Volume | 目的 |
|---|---|---|---|
| O1 | GT cortical IDs | controlled GT fragment union | 验证纯传播 |
| O2 | GT semantic cortex + GT affinities，unknown K | GT volume | 验证 offsets、图和 solver |
| O3 | GT cortex + predicted A | GT volume | 验证 affinity |
| O4 | predicted C+A | GT volume | 验证 cortical localization |
| O5 | GT cortical IDs | predicted provisional volume | 定位 volume 瓶颈 |
| O6 | predicted C+A | predicted provisional volume | 完整自动链路 |

controlled manifest 包含相邻/表面距离 `<=2 mm` 的 binary merge、3--5 个 fragment 的
multiway merge，以及 intact single-fragment negatives。缺可靠 cortex 的 fragment 只报告
coverage ceiling，不进入 oracle 主分母。

硬 gate：

- O1 binary all-child recovery `>=90%`；
- O1 multiway recovery `>=75%`；
- intact false-split `<=5%`；
- O2 exact-K + identity grouping `>=95%`；
- O1 binary `<70%` 时停止 neural development，先修 propagation；
- O2 未达目标时先扩 offset/修 graph，不用模型掩盖 solver 错误。

evaluator 必须显式声明 prediction/GT kind，不用 label 最大值猜语义。Hungarian IoU matching
报告：

- PQ/SQ/RQ，IoU 0.5 主阈值及 0.7 次阈值；
- all-child unique recovery、FP/FN、count error、matched Dice/IoU；
- `VI_merge`、`VI_split`；
- intact false-split；
- cortical exact-K、ARI/VI、pairwise relation；
- confidence Brier/ECE 和 risk-coverage；
- patient-macro 聚合。

68 例全部进入固定五折，因此所有性能结论都明确标记为 development OOF，不使用
“held-out test”或“confirmatory”表述。

## 7. 实现模块和命令

代码分为三个不互相混淆的层：

```text
tools/charite_cortical/continuity_data/         # final68 -> Dataset778、targets、folds、QA
nnunetv2/training/cortical_continuity/          # schema、affinity、loss、trainer/predictor
tools/charite_cortical/continuity_postprocess/  # MWS、split/stop、propagation、oracle/evaluator
```

统一 CLI 为：

```bash
python -m tools.charite_cortical.continuity_cli --help
```

它提供：

- `build`：final68 -> Dataset778，默认 dry-run；
- `splits`：打印/写出冻结五折；
- `oracle`：从 Dataset778 case 直接运行 O1/O2；
- `gate`：汇总 68 例 O1/O2，冻结 `neural_training_allowed`；
- `refine`：C+A 或 separator 工作网格 NPZ -> full/cortical instances；
- `calibrate`：在 OOF proposal 上拟合 L2 logistic split scorer；
- `evaluate`：显式 instance kind 的 PQ/VI/ARI/pairwise 评价；
- `aggregate`：patient-macro 和 deterministic bootstrap CI；
- `provenance`：对 source/splits/config/plans/checkpoints/ABBC 配置逐项 hash。

所有会写文件的命令要求显式 `--output`，且拒绝覆盖已有结果。`build`、`oracle` 还要求
`--execute`；省略时只检查输入并打印计划。formal `refine` 必须加 `--formal` 和
`--run-metadata`，否则 provenance 契约不成立。未提供校准 scorer 的 C+A refinement 默认
`always-abstain`，不会使用未经 OOF 校准的固定阈值接受 split。

### 7.1 数据、planning 与 preprocessing

只读预检查：

```bash
python -m tools.charite_cortical.continuity_cli build \
  --source /home/hongyou/dev/data/segmentation/derived/cortical_dataset_68_final_20260728
```

正式 HPC 构建和两套 preprocessing（separator 与 C+A）：

```bash
sbatch slurm/charite_cortical/build_plan_preprocess.slurm
```

该 CPU 作业申请 8 CPUs、256 GB RAM 和 48 小时。0.5-mm 全体积重采样会使超长 CT
病例产生数十亿体素，并在并行 worker 中放大峰值内存，因此两套 preprocessor 都在
重采样前执行 supervision ROI crop：训练时取已知 cortical/support 的包围盒并在物理空间
外扩 32 mm，然后把外层 ROI bbox 与 nnU-Net 的 nonzero bbox 组合写回 properties，保证
预测导出仍能恢复到原始全体积坐标。默认每套 preprocessing 使用 1 个 worker；只能在确认
单病例峰值内存后通过 `CORTICAL_SEPARATOR_PREPROCESS_PROCESSES` 和
`CORTICAL_CONTINUITY_PREPROCESS_PROCESSES` 显式增加。

separator 与 C+A 分别写入
`nnUNetResEncUNetMPlansSeparator_3d_fullres` 和
`nnUNetResEncUNetMPlansContinuity_3d_fullres`，不复用标准目录，也不复用此前的部分结果。
训练 ROI 只用于定位和限制计算范围；正式 inference 必须使用冻结 ABBC provisional support
生成同样的 32-mm ROI，fragment identity 的判断证据仍只能来自 cortical prediction。

生成的 plans 名称固定为：

```text
nnUNetResEncUNetMPlansSeparator
nnUNetResEncUNetMPlansContinuity
```

第二个 plans 由以下 CLI 冻结 `axial19` schema；若正式 gate 选择 dense39，则使用独立
`nnUNetResEncUNetMPlansContinuityDense39`，不原位改变已经冻结的 axial19 plans。

```bash
python -m nnunetv2.training.cortical_continuity.configure_plans \
  --plans <nnUNetResEncUNetMPlansContinuity.json> \
  --configuration 3d_fullres \
  --direction-set axial19
```

### 7.2 Oracle gate

单 case 命令直接读取 Dataset778 sidecar、验证 SHA-256，并在内存中完成 XYZ -> ZYX
working-grid 转换：

```bash
python -m tools.charite_cortical.continuity_cli oracle \
  --dataset "$nnUNet_raw/Dataset778_ChariteCorticalContinuity" \
  --case <case_id> \
  --mode O1 \
  --execute \
  --output <case>/o1.json

python -m tools.charite_cortical.continuity_cli oracle \
  --dataset "$nnUNet_raw/Dataset778_ChariteCorticalContinuity" \
  --case <case_id> \
  --mode O2 \
  --direction-set axial19 \
  --execute \
  --output <case>/o2-axial19.json
```

68 例 HPC array：

```bash
sbatch slurm/charite_cortical/run_oracle_axial19_68.slurm
```

汇总 gate：

```bash
python -m tools.charite_cortical.continuity_cli gate \
  --o1 "$CORTICAL_ORACLE_ROOT"/*/o1.json \
  --o2-axial19 "$CORTICAL_ORACLE_ROOT"/*/o2-axial19.json \
  --expected-cases 68 \
  --output <oracle-gate-axial19.json>
```

若 axial19 的 exact-K + identity `<95%`，先运行：

```bash
sbatch slurm/charite_cortical/run_oracle_dense39_68.slurm
```

再用 `--o2-dense39 "$CORTICAL_ORACLE_ROOT"/*/o2-dense39.json` 生成新的 gate。`gate`
要求每例包含完整 controlled-event manifest，缺 case、只跑部分 event、direction-set
不匹配或任一阈值未达标都会写出 `neural_training_allowed=false`。

### 7.3 五折训练

separator baseline 可先独立运行：

```bash
sbatch slurm/charite_cortical/train_separator_5fold.slurm
```

C+A 训练脚本强制要求 `CORTICAL_GATE_FILE`，且只在其中
`neural_training_allowed=true` 时启动：

```bash
mkdir -p logs
sbatch \
  --export=ALL,CORTICAL_GATE_FILE=/absolute/path/frozen-68-case-gate.json \
  slurm/charite_cortical/train_continuity_5fold.slurm
```

这是一个 `0-4` job array，每个 task 使用一张 GPU 训练一个 fold。它沿用原 ABBC
SLURM 的 `repair` 环境、nnU-Net 数据目录和 `--c` 续训方式，同时在真正启动训练前检查
oracle gate、plans、`splits_final.json` 以及 plans 中声明的 preprocessed
`data_identifier`。HPC 上仓库不在 `$PROJECT_HOME/dev/nnUNet` 时，可在提交时一并传入
`CORTICAL_REPO_DIR`；环境名可用 `CORTICAL_ENV_NAME` 覆盖。

若 gate 选择 dense39，先运行：

```bash
sbatch slurm/charite_cortical/prepare_dense39.slurm
```

trainer 名称为 `nnUNetTrainerCorticalContinuity`，由标准 `nnUNetv2_train -tr` class
discovery 找到。正式 schedule 在 trainer 内冻结为 500 epochs、250 iterations/epoch、
initial LR `1e-3`。

### 7.4 OOF 校准、refine 与评价

专用 predictor 只输出 working-grid evidence，不调用 generic semantic exporter；fold 和
sliding-window 先平均 logits，再分别 sigmoid，并强制 mirror TTA 关闭：

```bash
python -m nnunetv2.inference.cortical_continuity_predict \
  --model-folder <nnUNet-results-model-folder> \
  --folds 0 1 2 3 4 \
  --checkpoint checkpoint_final.pth \
  --input <preprocessed-CZYX.npy> \
  --provisional <same-grid-provisional-instances.npy> \
  --output <working-grid-evidence.npz>
```

该命令同时写出 checkpoint/plans/input hash provenance。`--provisional` 省略时只输出 C+A
证据；提供时输出可以直接交给 `refine`。raw-grid NIfTI 映射必须使用 nnU-Net 保存的
preprocessing properties 完成，不能在此处猜测 orientation 或 spacing。

`refine` 的结果是 working grid 上的 ZYX instance 数组。导出 NIfTI 时必须显式提供同一
working grid 的参考图像；如需回到原始 CT geometry，还必须显式允许最近邻重采样：

```bash
python -m tools.charite_cortical.continuity_cli export-nifti \
  --result <refine-result.npz> \
  --working-reference <working-grid-reference.nii.gz> \
  --output-full <full-instances-working.nii.gz> \
  --output-cortical <cortical-instances-working.nii.gz>

python -m tools.charite_cortical.continuity_cli export-nifti \
  --result <refine-result.npz> \
  --working-reference <working-grid-reference.nii.gz> \
  --original-reference <original-ct.nii.gz> \
  --allow-resample-to-original \
  --output-full <full-instances-original.nii.gz> \
  --output-cortical <cortical-instances-original.nii.gz>
```

导出固定使用 `uint32` instance IDs 和 order-0 nearest-neighbor，记录输入、参考图像、输出
SHA-256 与坐标转换诊断；输出或诊断文件已存在时拒绝覆盖。

校准输入 JSON 每条记录包含 `features`、`proposal_correct` 和 `intact_negative`：

```bash
python -m tools.charite_cortical.continuity_cli calibrate \
  --input <oof-proposals.json> \
  --output <split-scorer.json>
```

若样本不足、只有一个类别、intact negatives 不足，或不存在满足 5% risk 约束且能接受
正确 proposal 的 operating point，输出 scorer 类型为 `always_abstain`。

formal refinement 之前生成完整 provenance：

```bash
python -m tools.charite_cortical.continuity_cli provenance \
  --source-manifest <continuity_manifest.json> \
  --splits <splits_final.json> \
  --continuity-config configs/charite_cortical/continuity_v1.json \
  --plans <plans.json> \
  --cortical-checkpoint <checkpoint_final.pth> \
  --provisional-checkpoint <ABBC-checkpoint_final.pth> \
  --abbc-config <abbc-to-instance-config.json> \
  --output <run-metadata.json>

python -m tools.charite_cortical.continuity_cli refine \
  --input <working-grid-evidence.npz> \
  --config configs/charite_cortical/continuity_v1.json \
  --scorer <split-scorer.json> \
  --formal \
  --run-metadata <run-metadata.json> \
  --output <refined-instances.npz>
```

`working-grid-evidence.npz` 必须包含 `provisional_instances`、`cortex_probability`、
`affinity`、`affinity_offsets_zyx`、`spacing_zyx` 和 `affinity_kind`；可选 `rim_probability`。
输出 diagnostics 记录每个 provisional instance 的 accepted/unchanged/abstained reason。

建议执行顺序：

```text
1. validate final68 hashes/census
2. build Dataset778 + fixed fivefold splits
3. run O1 propagation oracle
4. run O2 graph oracle (19 offsets, conditional 39 offsets)
5. train/evaluate MVP-0 separator
6. train fivefold MVP-1 C+A and export OOF logits
7. calibrate graph probabilities and split/stop on OOF only
8. run O3--O6 and compare against frozen ABBC baseline
9. only if justified, train MVP-2 C+A+R
10. freeze deployment configuration
```

## 8. Definition of done

软件实施完成需满足：

- final68 builder 可重复运行且输出 hash 稳定，源文件 hash 不变；
- 68 例各自恰好出现在一个 validation fold；
- overlap/no-cortex/synthetic-fragment policy 有自动测试；
- 19/39 offset schema、增强后 affinity、空 mask loss 有测试；
- MWS 对 node/edge/label permutation 稳定，synthetic `K=1/2/3` 正确；
- propagation 通过 no-gap/no-overlap/no-added-voxel/seed-ownership invariant；
- evaluator 通过 Hungarian greedy counterexample 和 label permutation 测试；
- tiny synthetic end-to-end pipeline 可运行；
- 在可用 nnU-Net GPU 环境完成 trainer forward/backward 和五折训练 smoke test；
- 所有正式 run 保存 source manifest、split、config、plans、checkpoint/code hash 和 diagnostics。

模型训练和科学 gate 是否通过由五折 GPU/OOF 结果决定；软件实现本身不预设 cortical
一定能带来提升。

## 9. 当前代码交付状态

本次交付完成的是面向 HPC 的软件实现，不包含本地训练或性能结论：

- Dataset778 builder、固定五折、overlap/unknown/synthetic policy、manifest/hash/audit
  已实现；
- separator baseline 所需三分类标签与独立 ResEnc M plans 路径已实现；
- separator/C+A 的 32-mm supervision ROI 预处理、全网格导出映射和单 worker 默认值已实现；
- C+A schema、axial19/dense39、动态 affinity、native-resolution masking、loss、
  40/30/20/10 loader、trainer 与专用 predictor 已实现；
- split-only MWS、保守 abstention、传播 invariant、separator 后处理、O1/O2、
  calibration、instance/partition/calibration metrics 和 patient-macro 已实现；
- unified CLI、working/original-grid NIfTI 导出、冻结 config、provenance 和 SLURM arrays
  已实现；
- C+A SLURM 入口被 formal oracle gate 硬阻断，不能绕过 O1/O2 直接启动。
- MVP-2 的 `R_contact` 仍保持条件分支：只有 MVP-1 的正式 OOF gate 通过后才允许实例化和
  训练，不把尚未被数据证明有效的额外 head 混入当前 HPC 主路径。

以下属于 HPC 执行阶段，未在本次代码实现中运行：

- final68 全量 materialization 与 0.5-mm preprocessing；
- 68-case O1、axial19 O2，以及必要时 dense39 O2；
- separator 与 C+A 五折训练、OOF evidence export；
- split scorer OOF calibration、O3--O6 和最终 patient-macro 报告；
- GPU forward/backward、显存/吞吐与大体积 MWS/propagation 性能 profiling。

因此当前状态应表述为“implementation ready for HPC execution”，不能表述为模型已提升、
gate 已通过或存在 held-out/confirmatory 结果。

# SCSR-V4.1：120 Epoch 后机制验证

本轮依据用户提供的验证方案增加独立验证入口 `verify_scsr_v41.py`。默认执行实验 1、2、3、4；实验 5 可选。没有重新训练，没有新增损失或修改 SCSR 结构，原训练代码、单卡/DDP 流程及旧 `diagnose_scsr_v41.py` 均保持原样。

## 推荐命令：一次完成四组核心实验

在服务器 **WHU 目录**内执行。使用物理 GPU 4，程序内部为 `cuda:0`。验证使用单进程 `python`，不使用 `torchrun`。

```bash
CUDA_VISIBLE_DEVICES=4 python verify_scsr_v41.py --config_file configs/SCSR_V41.yml --checkpoint ./logs/SCSR_V41_1000_DDP/transformer_120.pth --output-dir ./logs/SCSR_V41_1000_DDP/mechanisms_fp32 --device cuda:0 DATASETS.ROOT_DIR /project/code/data
```

这是一条完整命令，默认四项全部执行，不需要先后运行四条命令。请将 checkpoint 路径替换为实际训练结果，输出目录必须为空。需同时保留 `transformer_120.pth.json`，脚本使用 sidecar 中的 `epoch=120` 验证轮次，不能只靠文件名判断。

默认从 checkpoint 目录读取 `scsr_v41_init_rank*.json` 和 `scsr_v41_training_diagnostics.jsonl`。如果这些文件存放在另一目录，在 **DATASETS.ROOT_DIR 等配置键之前**添加：

```text
--initial-records-dir 实际初始化JSON目录 --training-diagnostics 实际训练诊断JSONL路径
```

若使用本地已复制的记录，该目录为 `新建文件夹`。只有 JSON/JSONL 而没有原始初始化权重时，不会把重建结果标为精确原始快照。

## 统一协议和状态保护

- WHU-MARS-1000：500 个训练身份和 500 个测试身份；严格要求 Query=6,405、Gallery=93,609。
- 模态顺序固定 RGB/IR/Thermal，即 RGB=1、NIR=2、TIR=3；相机 0–4 为 Ground，5–6 为 Aerial。
- 768 维 pre-BN CLS，随后 L2 归一化；距离使用原 `iter_euclidean_distance_chunks`，即归一化特征的**平方欧氏距离**。身份间隔也使用这个实际距离定义，不另行开平方。
- 排除 Query 同相机的全部 Gallery，完整排序计算 AP/mINP。分块只控制内存，不截断 Gallery 或使用 Top-K 近似 mAP。
- 无 re-ranking/TTA。默认 FP32，关闭诊断进程的 TF32，固定数据顺序、种子和样本。原训练进程/配置的精度逻辑未改。
- 四组核心实验所有参数 `requires_grad=False`，全程 eval/no_grad，不创建优化器或原型银行。参数干预只暂时替换 SCSR 的 7 个张量并在 finally 中恢复；条件干预只在该实例临时替换 forward，异常时同样恢复。没有全局修改 Block.forward。
- 各阶段记录 checkpoint/sidecar SHA256、代码 SHA256、数据顺序哈希、精度、干预规则和种子。完成以及异常退出时核对完整模型 state_dict 哈希；正常完成也检查没有填充参数 `.grad`。

## 实验 1：同权重四组合检索

只加载一次最终权重，提取 ON/OFF 两套 Query 和 Gallery 特征，之后使用：

| 组 | Query | Gallery |
| --- | --- | --- |
| A | OFF | OFF |
| B | ON | ON |
| C | ON | OFF |
| D | OFF | ON |

OFF 仅跳过 SCSR，原 norm1/Attention/残差不变。提取和复用缓存时都验证 RGB/NIR 的 ON/OFF 特征逐元素一致。没有 TIR 的域会复用等价距离计算，但仍输出四组各自的指标。

整体与九模态对：mAP、Rank-1/5/10、mINP。另输出 6 个模态平台域的全部 36 个有向组合，包含 TIR→RGB/NIR/TIR 的全部 G/A 组合，以及 RGB/NIR→TIR。空域或同相机过滤后无有效正样本的域明确标记，不虚构 0 分。

逐 Query 文件保存完整 6,405 行，含 A/B 首名身份/图像、正确性、首个正样本排名、AP、delta_AP、排名改变和四类正确性转移。无有效正样本的 Query 保留原行及原因，不计入有效 Query 分母。

输出：

- `v41_on_off_metrics.csv`：整体和九模态，4×10=40 行。
- `v41_on_off_modality_platform.csv`：36 域四组合，4×36=144 行。
- `v41_on_off_query_comparison.csv`。
- `v41_on_off_manifest.json`。
- `v41_feature_cache.pth`：两套归一化特征及来源身份信息，约 586 MiB，不包含模型和临时排序内存。

缓存复用必须匹配 checkpoint/sidecar、数据顺序、模型/预处理、精度、提取 Batch、PyTorch/backend 及模型源码哈希。只提供路径或 shape 相同不能通过检查。

## 实验 2：初始化与训练后参数干预

优先使用 `--initial-state 原初始化权重路径`，支持纯 SCSR 七张量 state_dict 或完整原始模型中的 `base.scsr_v41.*`。应是**已校准**的初始化，未校准 std=0.02 的权重会被统计核验拒绝。不会将校准倍率重复乘到已提供的快照上。

没有快照时，自动按照记录的**模型 initialization_seed**、完整模型构造顺序、原 RNG 保护逻辑和**完整精度 output_scale_factor**重建。不会直接新建一个随机 SCSR，也不会误用校准 Batch seed（如 42235）。此版本 ImageNet 加载仅复制/插值主干张量，不消耗 RNG、不修改 SCSR，因此重建不需要重新提供 ImageNet 文件。

核验内容：四卡记录与 checkpoint 校准元信息一致；输出层实测 std；如果存在首个零有效更新的诊断记录，比较全部七个参数范数。若不匹配，实验 2 标记 unavailable/skipped，其他实验继续。

**范数/std 相同不证明逐元素相同。** 原训练未保存初始张量哈希，所以当前自动重建始终标为 `reconstructed_initial_control`、`exact_original_initialization=false`。提供快照也记录来源及验证状态；只有存在匹配的原始七张量哈希证据才标记 exact。复用本脚本保存的重建文件不会将其升级为原始快照。

固定约 512 张 TIR 测试图像，G/A 各半，按种子打乱身份后轮流抽取，覆盖不同身份与相机，不依据检索正确性挑选。所有组保持最终主干、BNNeck、classifier 不变：

| 组 | U | 条件网络 |
| --- | --- | --- |
| A | trained | trained |
| B | initial | initial |
| C | initial | trained |
| D | trained | initial |

相对于 A，逐样本记录 V/T/M 的实际元素相对 L2、余弦、RMS、方向分布、饱和比例；记录 norm1、Attention 输入/输出、Block 7 输出和最终 CLS 差异。训练前后参数另记录逐元素差异的汇总、相对变化和余弦，并保存完整 FP64 参数差分。

输出：

- `v41_initial_vs_trained_parameters.csv`。
- `v41_initial_vs_trained_gate.csv`。
- `v41_initial_vs_trained_features.csv`。
- `v41_initial_reconstruction_manifest.json`。
- `v41_initial_control.pth`：附来源标记的初始对照。
- `v41_initial_parameter_deltas.pth`：全部参数逐元素差分及来源标记。

这些结果属于“最终主干固定后的参数干预”，不代表重新训练的初始化模型，也不能替代独立 UAD 基线。

## 实验 3：条件与高频方向干预

使用同一固定 TIR 样本集。先建立原始 V/M 银行，再运行所有干预，避免 donor 配对随 Batch 边界改变。

| 干预名 | 唯一改变的部分 |
| --- | --- |
| `spatial_mean_v` | 对 V 的 8×4 空间求平均后广播，不平均 M 或通道轴 |
| `donor_same_platform_v` | 固定同平台、不同身份 donor 的 V |
| `donor_cross_platform_v` | 固定跨平台、不同身份 donor 的 V，单独标记统计 |
| `zero_ll_condition` | 仅将 W_L(LL) 的16通道置零；IDWT 中仍保留原 LL |
| `disable_lh` / `disable_hl` / `disable_hh` | 仅让指定方向 H'=H；条件与生成的 M 不变 |
| `disable_all` | 三方向都不更新，使用原 Patch 精确回写，避免将 Haar 重建舍入当成机制效果 |

每个干预记录逐样本 gate/方向增量以及 Attention/CLS 差异。高频增量从 FP32 更新后子带减原子带计算，包含实际算术舍入；不是用理想 FP64 M×H 代替。生产 callback 未暴露 H' 的情况下，在同一设备复算原 FP32 算子后采集。

运行时检查：正常诊断算子与原 SCSR 完整前向逐元素一致；所有干预的原 LL/HF 保持一致；有符号 HF 压缩及 RMS 条件不变；子带关闭时生成的 M 不变。反事实均只改变指定条件或实际应用的更新。

输入相关性文件同时比较：不同身份图像的相同空间位置（同平台/跨平台分别记录），以及同图像31个位置相对于固定位置(0,0)的通道增益排序。使用 signed M 降序（等价于1+M降序）、带平均并列秩的 Spearman 和 Top-32 重叠。退化常量门控明确标记，相关系数不填造假的1。

输出：

- `v41_condition_intervention.csv`。
- `v41_frequency_band_intervention.csv`。
- `v41_gate_input_dependence.csv`。
- `v41_condition_manifest.json`。
- `v41_tir_fixed_samples.json`：抽样与固定 donor 配对清单。

本项不重新提取全部 Gallery、不运行每种干预的完整检索。逐位置统计是描述性统计，不能将同图像的多个位置当作独立 Query 计算显著性。

## 实验 4：身份距离与困难近邻

直接复用实验 1 的缓存和距离块，不再次提取图像。整体、九模态和36域中分别分析 A/B：

- 过滤同相机后最近正样本/最近负样本的路径、身份、模态、相机、平台、距离和排名。
- `margin = d_neg - d_pos`，以及 `delta_margin = margin_B - margin_A`。
- Top-10 正确身份数量、首个正样本排名、AP、Rank-1 及正确性转移。
- A 选出的正/负邻居在 A/B 两种特征下的固定对象距离/排名；B 选出的邻居也在两种特征下测量。这样不会把邻居对象更换误读为同一对象的距离变化。

`first_positive_rank_delta` 为 B−A，负数表示正样本排名改善；`delta_margin` 为 B−A，正数表示最近正负间隔增大。

输出：

- `v41_identity_distance_changes.csv`：包含未变化样本和无有效正样本状态的完整分析。
- `v41_hard_negative_transitions.csv`：最近负对象或首名 Gallery 改变的有效 Query 子集，全部对象距离仍在完整文件中。
- `v41_query_transition_summary.csv`：按协议/域/四类正确性转移汇总；缺少正/负样本的间隔不计入该均值。
- `v41_tir_platform_summary.csv`：涉及 TIR 的平台域汇总。
- `v41_representative_cases.json`：每个协议/转移按固定 query_index 取最先3个有效例子，不按增益大小挑案例。

各协议的同一 Query 会有多行，不能把这些重复协议行当成不同 Query 合并统计。正式整体 mAP 取 `protocol=overall`；子域均值不可直接替代它。

## 可选实验 5：最终权重的局部梯度

在同一命令中增加 `--experiments 1,2,3,4,5` 即可；`--proca-gradient` 可另加原 ProCA 单独梯度。

默认固定2个真实训练图像组成的 PKM Batch，P=4/K=4/M=3，即每批48张。这是为了单 GPU FP32 诊断的内存开销，不冒充正式四卡每卡 P=16 的192张 Batch；如显存允许，用 `--gradient-identities 16` 选择正式 P/K 大小，`--gradient-batches` 可选择2–4批。

临时恢复原 `requires_grad` 标志，使用 eval 状态分别计算 CE/WRT（可选 ProCA），通过 `autograd.grad` 获取梯度；不填充参数 `.grad`、不执行优化器更新，不更新 BN/原型。记录梯度 norm/RMS、有限性、CE/WRT同参数余弦，以及单损失下原 max_norm=1 参数裁剪规则的模拟前后值。

**原规则只裁剪参数，不裁剪激活。** Attention 输入和 CLS 梯度的模拟裁剪后值保持原值，并标记 `clipping_applies=false`。单损失模拟裁剪不等于当时联合四损失的实际裁剪。ProCA记录未乘0.01的单损失梯度，并明确给出其训练权重；这只是最终权重附近的局部敏感度。

输出 `v41_final_gradient_probe.csv`。原 checkpoint 没有存原型银行，GPD明确不纳入，绝不随机新建原型冒充原训练 GPD。

## 常用可选参数

| 参数 | 含义 |
| --- | --- |
| `--experiments 1,4` | 仅检索和身份距离；默认已经是1,2,3,4 |
| `--experiments 2,3` | 仅固定样本的内部机制，不重复完整检索 |
| `--feature-cache 已生成的v41_feature_cache.pth` | 在新的空输出目录复用严格匹配的缓存 |
| `--batch-size 16` | 前向图像 Batch；也是缓存来源核验项 |
| `--distance-chunk 16` | CPU距离/排序 Query 分块，始终保留完整Gallery |
| `--tir-samples 512` | 机制样本数，须为偶数，G/A各半 |
| `--workers 4` | 固定图像加载 worker 数 |
| `--amp` | 另做 AMP 对照；必须另存输出目录/对应缓存，不能混用 FP32 缓存 |
| `--loss-scale 65536` | AMP梯度探针的显式诊断缩放，输出梯度除回该比例并保留非有限标志 |

所有脚本选项应放在最后的 `DATASETS.ROOT_DIR /project/code/data` 等 YACS 配置覆盖之前。代码不安装任何新依赖。

## 新增文件及本地核验

本轮新增7个文件，没有修改或删除任何原文件：

| 文件 | 职责 |
| --- | --- |
| `verify_scsr_v41.py` | 统一入口、严格协议/epoch/cache核验、各实验编排和最终状态检查 |
| `diagnostics/v41_ranking.py` | 四组合完整排序、固定邻居与重选邻居、流式CSV和域/转移汇总 |
| `diagnostics/v41_initial.py` | 初始化恢复/重建及证据标记、参数差异 |
| `diagnostics/v41_mechanisms.py` | 仅验证阶段的条件/频率干预、采集、固定样本与排序相关性 |
| `diagnostics/v41_gradients.py` | 最终权重单损失梯度和CE/WRT余弦，无优化器/原型 |
| `tests/test_v41_mechanisms.py` | 干预作用范围、异常恢复、完整AP、对象切换、缓存、初始化和小样本文件生成 |
| `SCSR_V41_MECHANISM_VERIFICATION.md` | 本说明 |

本地 Windows/Python3.9/PyTorch2.8 CPU：10项新验证测试、11项原V4.1回归测试通过；最终采集/激活梯度裁剪语义修正后又复核相关4项，通过。小型真实图像文件测试生成了实验2/3所有必需输出文件；排序测试覆盖四组合、完整AP、同相机排除和最近负对象切换，验证未改变参数/BN。

42个训练相关源码/配置的SHA256与修改前逐一相同；54个Python源文件通过Python3.8 AST解析。初始化重建的输出层 std 为 `0.07494654506444931`，与原记录一致；7个参数范数在 CPU/GPU 浮点容差内一致，仍标记非精确重建对照。

检查日志及源码审计位于工程根 `tmp/v41_mechanisms/`。本地没有第120轮权重、WHU-MARS实际数据或CUDA，未执行全量真实验证、未生成真实检索结果，也没有给出机制有效性的结果结论。GPU/旧PyTorch1.12.1实机验证由服务器上述命令完成。

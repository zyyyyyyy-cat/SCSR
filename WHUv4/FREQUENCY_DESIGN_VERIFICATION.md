# WHU-MARS频率证据验证实现说明

入口为 `verify_frequency_design.py`。它只加载冻结checkpoint并用临时hook采集/干预指定Block的`norm1`输出，不被`train.py`、模型正式前向、损失或优化器引用。运行中不创建Optimizer、不执行Step；结束时用完整`state_dict`哈希及`.grad`检查确认模型没有被修改。

## 模型与选择协议

- `--uad-checkpoint`必须是原UAD第120轮完整state_dict，是实验1、2、3、4、5、6的主要模型。
- `--v41-checkpoint`是可选的V4.1第120轮checkpoint，仅为实验3输出LL条件、V和M统计；必须同时存在其`.pth.json`。
- 参数校准、频率方向比较和层选择全部使用训练身份固定诊断集。默认固定64个有效Query，每个身份最多6个分层Gallery，保证每个Query有不同相机正样本，并加入条件匹配负样本。
- 正式测试确认不会自动选择组。先读训练身份结果，再在新输出目录用显式`--confirm-interventions BAND:+/-:budget`固定选择。
- 一级Haar与干预为FP32。`signed_spatial`保留符号、通道和8×4空间位置；保存到CPU时用FP16降低内存，但距离计算恢复FP32。`channel_rms`、`energy`、`energy_ratio`始终FP32且分开评价。

## 第一阶段：训练身份上的五项核心验证

```bash
CUDA_VISIBLE_DEVICES=4 python verify_frequency_design.py \
  --config_file configs/SCSR_V41.yml \
  --uad-checkpoint ./logs/UAD_1000_DDP/transformer_120.pth \
  --v41-checkpoint ./logs/SCSR_V41_1000_DDP/transformer_120.pth \
  --output-dir ./logs/frequency_design/train_core_fp32 \
  --device cuda:0 \
  --experiments 1,2,3,4,5 \
  DATASETS.ROOT_DIR /project/code/data
```

默认校准目标为0.5%与1.0%的真实Attention输入相对L2扰动。LH/HL/HH的`alpha`上限为20%，LL为10%。每个频率方向带3个固定随机种子的随机频率、空间位置和通道等范数对照。所有CSV都记录实际`r_attn`、Attention输出变化、最终CLS相对L2和余弦。

实验4默认只比较在训练身份上预先规定的`LH+、HL+、HH+`，预算均为0.5%。可以用`--layer-directions`更换，但应在查看正式测试结果前固定。

实验5使用两个固定真实PKM Batch，分别求CE、WRT和原定义ProCA对Block 7频带的梯度与方向导数。checkpoint没有原型银行，因此GPD明确不运行，不使用随机原型替代。

## 第二阶段：固定后的正式测试确认

训练身份结果确定候选后，在新的空目录运行。例如固定`LH增强1%`和`HH抑制0.5%`：

```bash
CUDA_VISIBLE_DEVICES=4 python verify_frequency_design.py \
  --config_file configs/SCSR_V41.yml \
  --uad-checkpoint ./logs/UAD_1000_DDP/transformer_120.pth \
  --output-dir ./logs/frequency_design/formal_confirmation_fp32 \
  --device cuda:0 \
  --experiments 2 \
  --confirm-interventions LH:+:0.01,HH:-:0.005 \
  DATASETS.ROOT_DIR /project/code/data
```

正式确认严格使用6405 Query与93609 Gallery，分块计算完整Gallery排序，无Top-K截断。每个候选分别输出TIR仅在Query、仅在Gallery、两侧同时干预三种状态。候选不会由脚本根据测试集指标自动筛选。

## 辅助实验6

```bash
CUDA_VISIBLE_DEVICES=4 python verify_frequency_design.py \
  --config_file configs/SCSR_V41.yml \
  --uad-checkpoint ./logs/UAD_1000_DDP/transformer_120.pth \
  --output-dir ./logs/frequency_design/control_fp32 \
  --device cuda:0 --experiments 6 \
  DATASETS.ROOT_DIR /project/code/data
```

`frequency_vs_spatial_control.csv`用`row_type`区分总体检索指标、逐Query变化和固定正负样本Margin变化。频率、空间、通道和随机方向对照均以实际Attention输入L2范数匹配，不训练Adapter。

## 输出

- 通用：`diagnostic_manifest.json`、`frequency_fixed_train_split.json`和六个`frequency_experiment_N_manifest.json`。
- 实验1：`frequency_identity_discriminability.csv`、`frequency_distance_distributions.csv`。
- 实验2：`frequency_intervention_calibration.csv`、`frequency_intervention_metrics.csv`、`frequency_intervention_query_changes.csv`、`frequency_intervention_margin.csv`、`frequency_intervention_controls.csv`；正式确认另有三个`frequency_official_confirmation_*.csv`。
- 实验3：`frequency_scene_stability.csv`、`frequency_cross_modality_pairs.csv`、`frequency_camera_platform_statistics.csv`、`v41_condition_identity_scene_statistics.csv`。
- 实验4：`frequency_layer_profile.csv`、`frequency_layer_intervention.csv`、`frequency_layer_identity_sensitivity.csv`。
- 实验5：`frequency_loss_gradient.csv`、`frequency_gradient_alignment.csv`、`frequency_margin_sensitivity.csv`。
- 实验6：`frequency_vs_spatial_control.csv`。

未请求的实验仍生成带`status=not_run`的结果文件和manifest。成功结束时总manifest应为`status=complete`，且`parameters_and_buffers_unchanged`、`parameter_grad_buffers_empty`、`finally_parameters_and_buffers_unchanged`均为`true`。

本地没有WHU-MARS、真实第120轮UAD/V4.1 checkpoint或CUDA，因此未生成真实实验数值。实现已通过Python 3.8语法检查、Haar/干预/配对/相机过滤单元测试及原V4.1诊断回归。

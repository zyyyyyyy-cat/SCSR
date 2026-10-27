# SCSR-V4.1 本地检查记录

日期：2026-10-14。环境：Windows，Python 3.9.13，PyTorch 2.8.0 CPU。没有 WHU-MARS 数据、ImageNet 实际预训练文件或 CUDA/NCCL 环境。以下检查使用随机输入及随机初始化主干，不能替代服务器真实实验。

## 本地结果

最终执行：

```text
python -m unittest discover -s tests -p test_scsr_v41.py -v
Ran 11 tests in 43.799s
OK
Actual Python unittest exit code: 0
```

完整 stdout/stderr：工程根 `tmp/v41_original/final_cpu_tests.txt`。

最终将诊断相对范数的分母对齐为“参考范数 + EPS”后，再复核模态/dtype、残差/ON-OFF、协议/语法和 overflow 日志 4 项，全部通过（3.456s）；记录在 `tmp/v41_original/final_diagnostic_checks.txt`。该对齐只影响诊断读数，没有改变频域调制算子或优化步骤。

| 检查 | 已验证的内容 |
| --- | --- |
| Haar | 显式 2×2 方向结果 `[LL,LH,HL,HH]=[5,-2,-1,0]`；FP64 重建和反向梯度误差满足 1e-12 容差；能量守恒 |
| 原始网络独立比对 | 测试使用修改前原 UAD 的完整 wrapper 和 ViT 源码；相同种子下所有共同参数/BN 缓冲逐元素一致；完整 OFF 检索前向逐元素一致 |
| 单层接入与残差 | 仅 blocks[6] 为新子类；原 Block.forward AST 未改；验证残差使用原 x，norm2/MLP 顺序正确 |
| 模态与 dtype | 非默认模态顺序、混合/交错传感器 ID、CLS 保留、RGB/NIR 原样保留；float32/float16/bfloat16 回写；无 ID/未知 ID 报错 |
| 调制边界 | 全精度重建 Patch 相对变化不超过 0.2（浮点容差）；M 幅度不超过 0.2；LL 保持不变至浮点容差 |
| 优化器 | 40,032 参数，7 个参数张量完整且仅注册一次；新组 WD=0/LR=BASE_LR，原 classifier WD 保持 1e-4 |
| 校准副作用 | 不调用训练 Loader/Sampler 迭代；恢复 Python/NumPy/CPU RNG；单卡 64 位置与四卡 global256 的每卡固定 Batch 选择一致、实际 192 张图 |
| 身份梯度 | eval 状态分别运行真实 CE/WRT 公式；7 个新参数均有有限非零梯度；不更新参数/BN，参数 `.grad` 未被该探针填充 |
| 实际优化 | 三模态、P=2/K=4 合成图像，使用原 CE+WRT+0.01ProCA+GPD、SGD momentum=0.9、全模型梯度裁剪 1.0；两步中 7 个参数张量均发生实际浮点参数变化，原型更新正确执行 |
| Strict checkpoint | 完整保存/严格恢复后输出逐元素一致；V3 版本 sidecar 拒绝加载 |
| 评测与语法 | 小型排序案例排除同相机全部 Gallery；全排序 AP/ON-OFF 一致；48 个 Python 源文件通过 Python 3.8 AST 解析 |
| AMP overflow 日志 | 注入 Inf 梯度，仍能写有效 JSON 并保留非有限标志/真实未更新读数；不会因 JSON 禁止 Infinity 而导致日志报错 |
| 一次性校准比例 | 验证不需调整、上调、下调；零/NaN 或不可同时满足 V/T 范围时拒绝继续调整 |

单独的 `train.py --help`、`test.py --help`、`diagnose_scsr_v41.py --help` 均正常返回。文档的四卡、单卡、200-step 预检和测试命令均已进行 YACS 合并检查，覆盖参数均成对且存在于当前代码。没有真正启动训练或访问服务器。

本地 PyTorch 2.8 对 `torch.cuda.amp`/`torch.cpu.amp` 给出的 FutureWarning 来自旧 API 的弃用提示。本实现有意使用服务器 PyTorch 1.12.1 可用的 CUDA AMP API。Python 3.8 AST 通过和旧 API 检查不等于已在 Python 3.8 / PyTorch 1.12.1 实机验证。

## 必须如实解释的限制

合成输入的两次实际 SGD 相对更新为：

| 参数 | Step 1 | Step 2 |
| --- | ---: | ---: |
| U | 3.59e-12 | 6.61e-12 |
| LL projection | 5.13e-9 | 8.05e-9 |
| 三个 HF projection | 8.74e-10–1.03e-9 | 1.18e-9–1.62e-9 |
| Spatial conv | 2.65e-9 | 4.34e-9 |
| Output conv | 1.41e-7 | 2.43e-7 |

这些数值很小，尤其 U 的更新；“非零”仅说明该数学路径在这次合成测试中连通且发生过浮点更新，**不能据此宣称 V4.1 已获得有效学习，更不能宣称优于 V3/UAD**。本地缺少真实 ImageNet 预训练和 WHU-MARS 特征，无法给出真实训练下的更新幅度。完整训练前应先审查真实 200 次有效更新的梯度、changed_fraction、step_relative、from_init_relative 及 Attention/CLS 变化。

float16/bfloat16 的 CPU 回写检查验证存储 dtype 和舍入后的可读差异，不代表 GPU autocast/GradScaler 已通过。Inf 注入检查只验证监测器的错误记录能力，不代表真实 AMP 跳步已在本机发生。

## V3 证据与源码保留

重新读取 `WHU_v3/logs/SCSR_1000_diagnostics/` 中已有 CSV：Q RMS 均值约 `5.12363e-5`，有效 Patch 相对变化约 `3.33284e-8`，TIR 最终 CLS 相对变化约 `4.78070e-7`；6,405 个 Query 的 Rank-1 Gallery 改变为 0。这些支持指导文件对 V3“实际检索作用极弱”的判断，不能唯一归因于某一个设计因素。

修改前源码归档：`tmp/v41_original/WHU_original_source.zip`；原源码 SHA256 清单：`source_hashes.json`；最终差异：`source_changes.patch`；配置命令和保留文件审计：`final_source_audit.json`。仅 6 个原文件有修改；没有删除原文件。`WHU_v3` 源码哈希无变化。原 UAD 配置、损失文件、数据加载/增强、两种 PKM sampler、scheduler、metrics 和 test.py 均未修改。

## 服务器仍需完成

使用 [实施说明](SCSR_V41_IMPLEMENTATION.md) 中的 200-step 四卡预检命令，保留完整 120 Epoch 调度器和正式训练配置，检查：

1. 实际数据为预期 WHU-MARS-1000，全模态/全平台，不混用 GD/2337；UAD 与 V4.1 使用同一个预训练 SHA256、种子、Batch/LR、Query/Gallery 和有效更新协议。
2. `scsr_v41_init_rank*.json` 的全局 V/T RMS、方向 M、初始 FP32/AMP 注意力/CLS 差异与 effective_output_std 正常；各 Rank 参数由原 DDP 初始化广播保持一致。
3. 没有 NCCL 同步错误/卡死；原 GPD 原型更新正常；真实 AMP skip 数量与有效 step 数可解释。
4. `scsr_v41_training_diagnostics.jsonl` 中每项 CE/WRT/ProCA/GPD/total 有限，7 个参数的真实更新与相对初值变化可解释，门控未严重饱和，Attention 和 CLS 的变化超出数值噪声。Rank 0 日志中的 cuda_memory 提供自进程启动以来（包含校准及探针）的峰值 allocated/reserved 字节数；同时核对各卡实际显存占用。
5. 按相同协议完成独立 UAD 与 V4.1 的 120 Epoch 比较，再运行同 V4.1 checkpoint 的 ON/OFF 检索诊断；ON/OFF 不能代替独立训练基线的性能对比。

本地没有给出真实数据初始化“通过”、GPU 显存保证、AMP 学习有效性或四卡训练“通过”的结论。

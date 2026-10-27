# 冻结120 Epoch检查点：四项SCSR诊断

实现对应用户提供的《SCSR-V3无需重新训练的诊断验证方案》。仅使用现有SCSR权重，不重新训练、不更新参数、BN统计或原型。不修改频率模块公式、身份损失、采样器和原train.py/test.py流程。

## 运行前准备

在服务器WHU工程目录运行。同步新增 `diagnose_scsr.py`、整个 `diagnostics/` 目录，以及修改后的 `model/scsr.py`、`model/backbones/vit_pytorch.py`。无需重装依赖。

必须同时保留以下两份文件，检查点中的classifier也必须完整：

```text
logs/SCSR_1000_DDP/transformer_120.pth
logs/SCSR_1000_DDP/transformer_120.pth.yml
```

新的入口严格加载全部模型参数，包含分类器。不能调用原普通测试的load_param替代，因为该接口跳过classifier，得到的CE诊断会不正确。兼容无前缀和 `module.` 前缀的state_dict，旧PyTorch不支持weights_only时采用已有回退。

数据必须是完整WHU-MARS-1000：500训练身份、500测试身份，RGB/IR/Thermal、Query与Gallery全部平台，不启用GD过滤。平台映射校验零基相机0..6；0..4=Ground，5..6=Aerial。预处理直接复用原验证流程的Resize→ToTensor→Normalize，无训练随机增强。

## 推荐命令

先执行完整四项诊断，默认FP32推理与原test.py一致。下面为单行Bash命令，路径按实际环境替换：

```bash
CUDA_VISIBLE_DEVICES=0 python diagnose_scsr.py --config_file configs/SCSR.yml --checkpoint logs/SCSR_1000_DDP/transformer_120.pth --output-dir logs/SCSR_120_diagnostics --batch-size 32 --distance-chunk 64 --probe-samples-per-modality 32 --gradient-batches 2 --gradient-identities 4 DATASETS.ROOT_DIR /path/to/data
```

如需记录原UAD检查点来源，所有脚本选项必须位于YACS键值参数之前：

```bash
CUDA_VISIBLE_DEVICES=0 python diagnose_scsr.py --config_file configs/SCSR.yml --checkpoint logs/SCSR_1000_DDP/transformer_120.pth --uad-checkpoint logs/UAD_1000_DDP/transformer_120.pth --output-dir logs/SCSR_120_diagnostics DATASETS.ROOT_DIR /path/to/data
```

`--uad-checkpoint`仅用于manifest记录，不会加载它，也不参与ON/OFF对照。不提供时记录null。检查点epoch只能从标准文件名推断；原state_dict不含epoch字段，准确来源仍由使用者确认。

只执行某些项，例如先做P0检索：

```bash
CUDA_VISIBLE_DEVICES=0 python diagnose_scsr.py --config_file configs/SCSR.yml --checkpoint logs/SCSR_1000_DDP/transformer_120.pth --output-dir logs/SCSR_120_on_off --experiments 1 DATASETS.ROOT_DIR /path/to/data
```

训练AMP精度专项复查（保持同一检查点、同一固定样本，另存输出，避免覆盖FP32原测试结果）：

```bash
CUDA_VISIBLE_DEVICES=0 python diagnose_scsr.py --config_file configs/SCSR.yml --checkpoint logs/SCSR_1000_DDP/transformer_120.pth --output-dir logs/SCSR_120_amp_probes --experiments 2,3,4 --amp DATASETS.ROOT_DIR /path/to/data
```

这是数值精度模式选择，不是放大SCSR残差；SCSR内部仍按原实现FP32计算，最后转换回实际Token dtype。`--amp`不能用于CPU。默认不加该参数，避免将原test.py的FP32检索误改成AMP检索。ON/OFF始终采用同一精度。

全部入口采用单进程单GPU，**不用torchrun，不执行DDP诊断**，不会影响已有DDP训练代码。`--device cuda:0`指可见GPU中的第0张。例如CUDA_VISIBLE_DEVICES=3时即物理第3张。运行独立进程，优先空闲卡。

## 六个输出文件

| 文件 | 内容 |
| --- | --- |
| `diagnostic_manifest.json` | 检查点/配置SHA256、环境、精度、数据数量与顺序hash、固定样本及训练批次、分组统计、ON/OFF四类Query计数、模型状态前后hash。 |
| `scsr_on_off_metrics.csv` | ON/OFF各14个协议：整体、9模态对、TIR四个平台组合。每行mAP、Rank-1/5/10、mINP、有效Query数、状态。共28行。 |
| `scsr_on_off_query_comparison.csv` | 所有整体Query（含标记为无效的Query）、路径/身份/相机/模态/平台、ON/OFF首名Gallery路径与身份、首个正样本排名、AP/INP、完整排序hash与排名变化、四类正确性。另附L2身份特征差异。 |
| `scsr_internal_activation.csv` | 固定TIR样本的P6、投影、四子带、Q、门控增量、IDWT、升维、平滑缩放残差、dtype转换和实际回写统计。 |
| `scsr_layerwise_feature_effect.csv` | 各样本ON/OFF在Block6 CLS/Patch、SCSR后CLS/Patch、Block7/9/12 CLS、最终pre-BN CLS及L2 CLS的绝对/相对差异、余弦、变化/相同元素比例与dtype。 |
| `scsr_gradient_probe.csv` | 至少两批真实固定PKM式样本，CE与WRT分别统计各激活和全部SCSR参数的ValueRMS、GradRMS、GradNorm、有限性及梯度是否存在。 |

只选择部分实验时，只生成对应CSV与manifest。输出目录要求全新，避免两次诊断混杂。CSV指标用小数（例如0.11=11%），浮点输出12位科学计数法。空白指标代表无有效Query或没有可用梯度，不能当作0。JSON把非有限数值写成字符串并保留finite标志，不伪造为正常值。

## 实验1：协议与有效Query

ON/OFF采用同一个model，图像只读取/预处理一次，每个Batch在同一Tensor上执行两次前向；runtime开关只跳过L6后的SCSR。模块仍注册，权重不变，不加载UAD作为OFF模型。

所有特征为768D pre-BN，经原L2归一化。距离调用现有 `iter_euclidean_distance_chunks`，CPU FP32计算欧氏距离并用原torch.argsort排序，不截断、不re-ranking、不TTA；每条Query排除该相机的**所有Gallery样本**。AP与mINP计算与原eval_func一致。

ON/OFF有效性仅由相同的身份/相机元数据决定，并断言一致。无效Query仍保留在比较表，以便复查，四类正确性统计只计有效Query。`rank1_gallery_changed`与`rank1_identity_changed`区分“换了同身份图片”和“换了身份”；`ranking_changed`通过完整过滤后Gallery索引hash判断，既不截断也不只看Top-1。

每个分组按同样协议限制Query/Gallery后再计算，含无正样本组的status，不用整体排序简单平均替代组指标。距离分块默认64降低内存；原公式和过滤规则未变，manifest记录分块大小。微小变化附近的距离相等/浮点舍入可能改变排序，因此同时报告特征差异与完整排序变化，不把所有变化都直接视为性能增益。

## 实验2/3：实际张量与固定样本

默认每模态选32条Query，每模态尽量均衡Ground/Aerial，共96张，其中32张TIR。独立seed控制选样，全部路径写入manifest，ON/OFF不重采样。

模型通过默认关闭的callback采集真实前向，不用长期Forward Hooks。每Batch只保留有限的中间张量，统计后释放；不把整套Query/Gallery的中间Token保存到磁盘。

实验2分别记录：

```text
r_generated = ||residual_fp32||F / (||P6||F + eps)
r_cast      = ||residual_cast||F / (||P6||F + eps)
r_effective = ||SCSR实际返回值 - P6||F / (||P6||F + eps)
```

`updated`就是SCSR实际返回并经index_copy传给Block7的Patch，不是诊断重新构造的结果。ChangedRatio、绝对变化均值/最大值/P50/P90/P99测量实际返回数值；为了准确测量差异才将实际值转FP64相减，**不改变模型的加法或运行精度**。

门控记录直接用于ΔH的 `0.5*tanh(Q)`，即数学意义的G−1，不额外构造1+gate再相减造成新的舍入。三个方向分别统计。分组均值写入manifest；实验3区分RGB/NIR/TIR以及各平台，预期Block6和RGB/NIR对照应一致，最终CLS变化需要依据真实数值判断。

## 实验4：固定状态的身份梯度

从训练集选择至少两批不重叠身份，默认每批4身份，每身份每模态K=NUM_INSTANCE=4，共48张图；全部采用验证预处理，禁止随机增强。采样路径与relabel后的分类标签完整写入manifest，模态组顺序与target.repeat完全一致。

模型保持eval，自动求导开启。CE（不做label smoothing）和原WRT分别重新前向，再通过autograd.grad执行反向，保留实际Q、投影、残差等计算图。这不会写入参数.grad，也没有Optimizer实例/Step，不更新BN或原型，不保存任何模型权重。

标准UAD检查点只保存model.state_dict，没有processor局部PrototypeBank。因此明确跳过GPD/联合四项损失检查，不能用新建零原型银行冒充原训练状态。此处梯度是eval-state诊断梯度，不能当作训练模式BN下的实际梯度。

FP32模式loss_scale=1。AMP模式默认诊断loss_scale=4096，反向后所有梯度除以同一scale，降低半精度反向下溢风险；它不修改参数或残差，也不声称恢复了训练时GradScaler状态。可通过 `--gradient-loss-scale` 指定；manifest与CSV记录scale以及非有限梯度。若出现非有限梯度，先检查该scale与dtype，不能直接断言原训练梯度出错。

额外输出每项单独损失的global_grad_norm及“若max_norm=1裁剪时”的比例/GradRMS，仅作计算，不实际裁剪梯度。该比例不是训练四项联合损失的真实裁剪系数，不应用它代替训练日志或决定改学习率。

## 保护训练与检查结果

仅修改两个模型文件，增加无参数、默认关闭的callbacks与默认开启的runtime标志。普通训练/测试不设置callbacks，不做诊断统计；训练模式下禁止旁路SCSR。context在成功或异常退出时恢复标志与callbacks。未改train.py、test.py、processor、loss、solver或datasets实现。

诊断前后对model.state_dict（含BN运行统计）计算SHA256，必须完全一致；不一致则manifest标失败并报错。检查点文件仅读取。科学计数法CSV与manifest在独立目录输出。

本地测试覆盖：原训练回归、零初始化等价、真实回写与Block7输入一致、FP32舍入使有效残差为0的情况、ON/OFF RGB/NIR不变、参数/BN状态不变、异常时开关恢复、CE/WRT梯度、原评测函数数值一致、固定选样和分类器完整恢复。使用合成JPEG还走通六文件生成的完整链路。

本地没有真实WHU-MARS数据或GPU训练权重，因此这些测试不等于已经完成真实四项诊断；最终CSV与120 Epoch权重结果需在服务器运行上述命令获取。

## 本次文件变更

| 文件 | 状态/用途 |
| --- | --- |
| `model/scsr.py` | 修改：默认关闭的真实内部张量callback，返回值算术不变。 |
| `model/backbones/vit_pytorch.py` | 修改：eval-only runtime ON/OFF与逐层callback，正常训练保持开启。 |
| `diagnose_scsr.py` | 新增：独立CLI、严格完整加载、四项调度、manifest及状态不变校验。 |
| `diagnostics/__init__.py` | 新增：诊断包入口。 |
| `diagnostics/core.py` | 新增：可恢复诊断context、精确差异统计、有限capture及梯度探针。 |
| `diagnostics/data.py` | 新增：固定预处理、顺序元数据、代表性Query和对应身份PKM式批次。 |
| `diagnostics/retrieval.py` | 新增：原距离函数、流式ON/OFF完整排序和有效Query/协议统计。 |
| `tests/test_scsr_diagnostics.py` | 新增：6项数值/协议/权重/完整输出测试。 |
| `docs/SCSR_DIAGNOSTICS.md` | 新增：本说明。 |
| `docs/scsr_diagnostic_checks.txt` | 新增：18项回归与诊断测试完整记录。 |
| `docs/SCSR_IMPLEMENTATION.md` | 修改：补充诊断入口与本轮检查结果。 |
| `docs/scsr_changes.patch` | 修改：更新累计源码差异供审阅。 |

未删除文件。train.py、test.py、processor.py、loss、solver、datasets与原metrics.py本轮均未修改。

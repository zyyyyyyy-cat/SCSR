# 实验A：SCSR-V3 完整代码修改指导（正式实施版）

目标： 在现有WHU-MARS/UAD代码基础上，增加一个仅针对TIR的中层频率调制模块，保持原有数据采样、身份损失、原型更新和统一检索方式不变。

本指导以此前确定的SCSR-V3为准，侧重修改位置、数据传递、模块逻辑、接口约定及必须验证的实现细节，不提供具体代码，也不增加新的方法结构。

## 一、正式指导前的两轮复审结果

### 第一轮：检查数学结构和优化机制

核对了完整的计算链：

\\[ P_6\rightarrow W_d\rightarrow\mathrm{DWT} \rightarrow\Phi\rightarrow\Delta H \rightarrow\mathrm{IDWT}\rightarrow W_u \rightarrow P_6'\rightarrow\mathrm{CLS} \\]

结果是：

- 输入输出维度可以严格对应。
- 高频增量能够通过正交Haar逆变换恢复到原16×8空间网格。
- 无偏置逐位置升维投影能够保留增量的零低频性质。
- 有界残差不会阻断梯度，但只能限制中层直接更新幅度，不能保证最终身份空间完全不受干扰。
- 原UAD四项损失能够通过最终CLS反向监督SCSR，无须额外增加频率重建、辅助分类或一致性损失。
- 768→192投影、64维条件瓶颈和高频乘性更新均存在明确的表达能力限制，但没有发现必须修改的数学矛盾。

第一轮结论：保留SCSR-V3，不修改核心公式。

### 第二轮：检查与实际UAD源码的兼容性

重新核对了官方仓库的`make_model.py`、`vit_pytorch.py`、`processor.py`、`make_dataloader.py`及优化器构造文件。

发现三个实施时必须处理的重点：

1. 官方模型的`mode`主要区分训练与测试路径，不能直接作为传感器模态编号使用。
2. 训练数据按RGB、IR、Thermal分别组成Batch，模型再按模态顺序拼接；不能改变拼接后的特征与身份标签顺序，否则会影响CE/WRT以及ProCA/GPD。
3. SCSR必须在优化器创建前完成注册，训练与测试都要走相同的第6层后增强路径；测试时必须使用真实模态信息启用TIR分支。

第二轮结论：整体兼容，重点是正确实现模态路由、ViT中间层插入和原有训练契约。

因此，可以采用下面的正式修改方案。

## 二、参考论文与具体模块来源

首先明确：SCSR不是某一篇论文的直接复现，而是参考几篇论文的局部机制后，针对UAD重新设计的模块。

| 参考论文                | 具体参考位置                                                                 | SCSR采用的思想                                    | 不采用的内容                                  |
| ------------------- | ---------------------------------------------------------------------- | -------------------------------------------- | --------------------------------------- |
| NS-FPN，CVPR 2026    | 第3.1节 Low-frequency Guided Feature Purification（LFP），式（1）—（6）          | DWT分离低频与高频，利用低频上下文指导高频调整，IDWT恢复空间特征          | 固定抑制式门控、高斯噪声过滤、SFS螺旋采样、FPN检测结构          |
| FreqFiT，MICCAI 2025 | 第3节 Frequency Fine-tuning，式（1）—（4）                                     | 在Transformer Blocks之间进行频率处理，以残差方式影响后续Token表示 | FFT复数频率滤波器、完整PEFT冻结策略、原论文的scale/shift结构 |
| PFESA，MICCAI 2025   | 第2.1节Frequency-Domain Feature Decoupling、第2.2节Edge/Structure Attention | 整体结构与局部细节应协同考虑，而不是单独强调高频能量                   | 固定FFT高低频掩码、无参数统计注意力、直接乘性覆盖原特征           |
| SFP，ICCV 2025       | ViT异常Token传播及特征净化部分                                                    | 中间层Patch响应可能影响后续表示，应谨慎处理                     | 异常Token硬筛选、邻域替换和语义分割专用注意力机制             |
| WHU-MARS/UAD        | ProCA、GPD及ViT身份特征学习                                                    | 保留完整基础模型与身份优化目标                              | 不改造原型记忆、不新增检索分支                         |

相关阅读和源码：

- [NS-FPN论文](https://arxiv.org/pdf/2508.06878) · [LFP实现](https://github.com/mengduann/NS-FPN/blob/main/model/NS_FPN.py)
- [FreqFiT论文](https://papers.miccai.org/miccai-2025/paper/3066_paper.pdf) · [频率算子实现](https://github.com/tsly123/FreqFiT_medical/blob/main/models/freqfit.py)
- [PFESA论文](https://papers.miccai.org/miccai-2025/paper/3694_paper.pdf) · [PFESA实现](https://github.com/59-lmq/PFESA/blob/main/models/Attention/pfesa.py)
- [SFP论文](https://openaccess.thecvf.com/content/ICCV2025/html/Jin_Feature_Purification_Matters_Suppressing_Outlier_Propagation_for_Training-Free_Open-Vocabulary_Semantic_ICCV_2025_paper.html) · [SFP实现](https://github.com/Kimsure/SFP/tree/main/SFP_ICCV)
- [WHU-MARS/UAD官方代码](https://github.com/msm8976/WHU-MARS/tree/main/CVPR26_UAD)

特别强调两点：SCSR的低频和高频双向条件门控属于本次重新设计，不是NS-FPN原样复制；它使用Haar-DWT而不是FreqFiT的FFT，因此不能声称直接采用了FreqFiT的滤波器公式。

## 三、代码修改范围：建议只涉及以下位置

以官方UAD目录`CVPR26_UAD/`为参考。

| 修改位置                             | 需要做什么                   | 是否改原有逻辑    |
| -------------------------------- | ----------------------- | ---------- |
| `model/`下新增SCSR模块文件              | 实现完整频域特征调制              | 新增         |
| `model/backbones/vit_pytorch.py` | 注册SCSR，在第6个ViT Block后调用 | 局部修改       |
| `model/make_model.py`            | 建立训练与推理的模态标签传递          | 局部修改       |
| `processor/processor.py`         | 将测试Batch真实模态传给模型        | 仅修改调用接口    |
| `config/defaults.py`             | 增加SCSR配置项及默认值           | 新增配置       |
| `configs/`                       | 为实验A建立独立配置              | 不覆盖原UAD    |
| `loss/`                          | 不修改                     | 保持原样       |
| `datasets/sampler.py`等           | 不修改                     | 保持原样       |
| `solver/`                        | 原则上不修改                  | 使用原优化器与调度器 |
| `utils/metrics.py`               | 不修改                     | 保持统一评测协议   |

建议建立独立实验目录或Git分支，保留原UAD完全不变，便于复现、比较和排查。

其中最重要的是前三处：SCSR模块本体、ViT插入位置、模态路由接口。

## 四、新增SCSR模块：完整实现规范

这一部分可以直接作为开发者的模块需求说明。

### 4.1 模块对外接口

建议命名为`SCSR`，对外只接收当前层的TIR Patch Token：

\\[ P\in\mathbb R^{B_T\times128\times768} \\]

返回：

\\[ P'\in\mathbb R^{B_T\times128\times768} \\]

不要让SCSR自身负责判断样本来自哪个模态，也不要让它读取身份标签、相机编号或原型记忆。

模态筛选由外部Backbone完成，SCSR只负责对传入的TIR特征执行计算。

这样更容易保证职责清楚，同时方便未来对照空间Adapter。

### 4.2 模块内部组件

| 组件    | 固定规格                         | 重要要求                |
| ----- | ---------------------------- | ------------------- |
| 预归一化  | LayerNorm(768)               | 对每个Token的通道归一化      |
| 降维    | Linear 768→192               | 无偏置                 |
| DWT   | Haar，一级二维变换                  | 固定滤波器，不训练小波基        |
| 条件投影  | Conv 1×1，768→64              | 无偏置                 |
| 空间上下文 | Depthwise Conv 3×3，64通道      | padding=1，groups=64 |
| 激活    | GELU                         | 在前两次条件卷积之间及其后使用     |
| 门控输出  | Conv 1×1，64→576              | 带偏置，末层权重和偏置零初始化     |
| 高频更新  | \\(0.5\tanh(Q_j)\odot H_j\\) | 三子带分别更新             |
| IDWT  | 与DWT配套的Haar逆变换               | 低频增量固定为0            |
| 升维    | 逐位置Linear 192→768            | 无偏置，无后置激活           |
| 有界残差  | \\(\eta=0.2\\)               | 每样本独立计算范数           |

在上述偏置设置下，新增可训练参数数目为383,616。这包括LayerNorm、两个通道投影以及条件网络。

### 4.3 模块内部的执行步骤

1. 输入投影

   对TIR Patch做LayerNorm和768→192线性映射，恢复为16×8空间网格，得到192通道中间特征。

   要使用Patch Embedding真实的网格排列顺序，而不是任意reshape。正常的16×8网格按照高度16、宽度8排列。
2. 二维频率分解

   使用正交Haar-DWT得到LL、LH、HL、HH四个子带，形状均为`B_T×192×8×4`。

   保留小波系数的正负号，不对四个子带取绝对值，不做额外的子带归一化。
3. 生成上下文条件

   按固定顺序将LL和三个高频子带在通道维拼接，得到`B_T×768×8×4`。

   通过`1×1卷积 → GELU → 3×3 Depthwise卷积 → GELU → 1×1卷积`，生成576通道门控Logit。

   最后拆分为三组192通道，分别对应LH、HL、HH。
4. 高频双向调制

   将门控Logit通过Tanh限制至(-1,1)，乘以0.5形成实际的高频变化系数。

   对原高频系数逐元素相乘，生成三组高频增量。允许增强与抑制，但不直接生成全新的高频内容。
5. 恢复频域变化

   使用零LL增量和三组高频增量执行IDWT，直接得到192通道、16×8大小的频率变化图。

   注意这里恢复的是变化量，不是一张新的完整特征图。不需要对完整重建结果再次减去原特征。
6. 升维与有界残差

   变化图经过无偏置192→768投影，整理回128个Patch Token。

   按每张图像分别计算原Patch和更新量的Frobenius范数，执行有界残差加法，输出更新后的Patch。

### 4.4 必须特别注意的小波实现细节

NS-FPN公开代码中已经使用`pytorch_wavelets`的`DWTForward`、`DWTInverse`，可以参考其变换实现和高频子带组织方式。

但不要直接把完整`wav_Enhance`类复制到SCSR，因为它包含固定抑制式注意力和额外高斯滤波，而SCSR使用的是重新设计的双向门控。

建议遵守以下约定：

- 正反变换使用同一个小波库、相同Haar小波和边界模式；
- 输入尺寸固定为16×8，一级分解后为8×4，尺寸均为偶数；
- 正反变换完成后直接得到16×8，不进行额外插值；
- 显式核对小波库返回的三个高频方向顺序，避免LH、HL交换；
- DWT、门控频率计算及IDWT可以在FP32中执行，再将最终残差转换回ViT当前Token的dtype。

特别是最后一点，不能因为新增模块使用FP32，就无意中让后续全部ViT Blocks都改变运行精度。

此外，不允许在IDWT后的升维投影后额外添加GELU、LayerNorm或空间门控。否则原来约定的高频增量性质会发生改变。

## 五、在ViT中插入SCSR：这是最重要的结构修改

### 5.1 修改`vit_pytorch.py`

官方主干的主要流程是：

`Patch Embedding → CLS和位置编码 → 所有Transformer Blocks → 最终LayerNorm → CLS输出`

现在需要改为：

`Patch Embedding → CLS和位置编码 → Block 1—6 → SCSR → Block 7—12 → 最终LayerNorm → CLS输出`

对应修改点是`TransReID`的Backbone初始化和`forward_features`。

明确插入在第6个Block执行完成之后。

如果Transformer Blocks采用从0开始的索引，那么插入点位于`blocks[5]`之后、`blocks[6]`之前。

不要误插入第6个Block的Attention与MLP之间，也不要插入到最终`self.norm`之后。

### 5.2 输入输出及CLS处理

第6层输出：

\\[ X_6=[c_6;P_6] \\]

SCSR只处理`P_6`中属于TIR的样本。

完成后：

\\[ X_6'=[c_6;P_6'] \\]

继续执行第7—12层。

需要注意：这里的“CLS保持不变”，仅指SCSR不直接修改第6层CLS，不是整个模型最终CLS不变。

后续Transformer的Self-Attention仍会让CLS与更新后的Patch交互，这是SCSR影响最终身份表征的主要路径。

### 5.3 不要修改原ViT其他结构

保持以下内容不变：

- 原始Patch Embedding；
- CLS Token；
- 位置编码与已有SIE设置；
- 前12个Transformer Blocks内部的Attention和MLP；
- 最终LayerNorm；
- 原BNNeck；
- 原身份分类器。

这样实验A与UAD之间唯一的主要网络差别就是一个L6后的TIR频率调制模块。

### 5.4 SCSR注册与预训练权重加载

SCSR需要作为Backbone内的正式子模块注册，确保其参数出现在模型参数列表、优化器和Checkpoint中。

建议按以下原则处理：

- ImageNet预训练ViT参数仍使用原方式加载；
- SCSR是新增参数，不要求在ImageNet权重中存在；
- 新增模块在模型级通用初始化完成后，执行自己的特定初始化；
- 不要通过后续通用初始化覆盖门控末层的零初始化；
- 创建优化器时，SCSR参数应当已经存在。

不建议先训练完整UAD再追加SCSR微调，然后直接将结果与原120 Epoch UAD比较；本轮应当使用相同的预训练起点和训练预算。

## 六、模态路由：保证训练和推理使用同一条计算路径

### 6.1 修改`model/make_model.py`

当前官方UAD训练接口会收到三组图像，然后将其按模态顺序拼接：

\\[ X=[X\_{\mathrm{RGB}};X\_{\mathrm{NIR}};X\_{\mathrm{TIR}}] \\]

建议保留这个行为，但在模型内部同时建立与之对应的模态标识：

\\[ M=[1,\ldots,1;\ 2,\ldots,2;\ 3,\ldots,3] \\]

这里的1、2、3分别对应官方数据配置中的`RGB`、`IR`、`Thermal`。

具体要求：

训练时：

- 使用现有`imgs`列表的模态顺序建立模态标识。
- 图像拼接与模态标识拼接必须一一对应。
- 模态标识传给Backbone，不参与身份分类或ProCA计算。
- SCSR只对标识为Thermal的样本执行。

推理时：

- 模型接收输入图像及其真实模态编号；
- TIR样本执行SCSR；
- RGB/NIR样本保持原计算路径。

建议在模型接口中明确区分两个概念：

| 参数          | 含义                 |
| ----------- | ------------------ |
| `mode`      | 保留官方原义：训练或评估执行路径   |
| `modal_ids` | RGB/NIR/TIR传感器模态编号 |

不要复用一个变量承担两种语义。

### 6.2 修改`processor.py`中的测试调用

官方验证和测试DataLoader已经返回`modid`，不需要改动数据集格式。

源码中对应结构为：

`img, vid, camid, camidt, modid, imgpath`

目前模型调用没有把`modid`传进Backbone，需要在验证和独立测试两处调用中补上。

我检查了官方`whu_mars.py`，其模态编号按`DATASETS.MODALITIES`排列，从1开始：

| 编号 | 对应模态        |
| -- | ----------- |
| 1  | RGB         |
| 2  | IR/NIR      |
| 3  | Thermal/TIR |

因此，在官方默认配置下可以使用3作为TIR标识。但更稳妥的工程约定是由数据集模态配置建立映射，而不是在SCSR内部写死数字3。

验证与测试的模态标识均须转换为能够与图像Batch逐样本对应的形式。

### 6.3 Batch顺序不能改变

这项要求尤其重要。

官方`train_collate_fn`将三种模态分别堆叠；`make_model.py`再将三组图像拼接；原ProCA则根据固定模态顺序和每身份样本数计算中心。

因此SCSR应该做到：

只更新对应TIR样本在原Batch中的Patch特征，不重新排列Batch，也不重新排列身份标签。

即使为了计算效率暂时提取所有TIR Patch，也必须把结果准确放回原位置。

尽量避免对仍被自动求导使用的中间张量进行不安全的原地覆盖；应采用不会破坏计算图的特征替换或组合方式。

不建议为实现SCSR而修改PKM采样器、训练数据结构或ProCA的中心计算规则。

## 七、损失函数、优化器和训练配置：原则上全部沿用原UAD

### 7.1 不修改`loss/`

实验A的损失仍然是：

\\[ \boxed{ \mathcal L= \mathcal L\_{\mathrm{CE}} + \mathcal L\_{\mathrm{WRT}} + 0.01\mathcal L\_{\mathrm{ProCA}} + \mathcal L\_{\mathrm{GPD}} } \\]

应保持原有监督位置：

| 损失    | 特征来源                | 修改要求 |
| ----- | ------------------- | ---- |
| CE    | BNNeck后的CLS         | 保持原样 |
| WRT   | pre-BN CLS          | 保持原样 |
| ProCA | pre-BN CLS形成的模态身份中心 | 保持原样 |
| GPD   | pre-BN CLS与全局身份原型   | 保持原样 |

这里需要重点检查是否所有损失都拿到了SCSR之后的最终CLS。

不能CE使用增强特征，WRT、ProCA或GPD却意外使用未增强的原CLS。

同样，不添加频率损失、额外TIR分类器、频率稀疏项或一致性损失。

这是为了让实验真正检验频率模块本身，而不是同时改变监督策略。

### 7.2 不修改ProCA/GPD更新规则

官方GPD原型更新采用：

\\[ p\leftarrow \operatorname{norm} (0.8p+0.2p\_{\mathrm{new}}) \\]

即`GPD_MOMENTUM=0.2`表示新中心占20%。

沿用原UAD的原型更新顺序和时机，不要新增第二套原型银行，也不要因为SCSR处理TIR就分别维护TIR专属身份中心。

### 7.3 优化器设置

新增SCSR参数应与共享主干一同参与原SGD优化。

推荐：

| 配置                | 首轮要求             |
| ----------------- | ---------------- |
| Optimizer         | 原UAD SGD         |
| Backbone初始LR      | 0.008            |
| SCSR初始LR          | 0.008            |
| Momentum          | 0.9              |
| Weight Decay      | \\(10^{-4}\\)    |
| Warmup            | 5 Epoch          |
| Epoch             | 120              |
| LR Schedule       | 与对照UAD实际运行配置保持一致 |
| Gradient Clipping | 保留原配置            |
| AMP               | 保留，频域操作内部使用FP32  |

不要为SCSR额外设置5倍或10倍学习率，也不要修改全局损失权重。

如果当前复现工程已经对官方代码做了训练调度调整，以获得对照结果的实际配置为准，不要因为添加SCSR又切回另一套调度。

### 7.4 模块初始化必须明确

| 参数        | 初始化要求     |
| --------- | --------- |
| LayerNorm | 权重1、偏置0   |
| 降维与升维投影   | 小幅非零随机初始化 |
| 条件网络前两层   | 正常非零初始化   |
| 门控最后一层    | 权重0、偏置0   |
| 固定残差系数    | 0.2       |

这样首次前向：

\\[ Q=0\Rightarrow\Delta H=0\Rightarrow P'=P \\]

因此新增模块在初始状态下不改变原ViT的前向特征。

需要注意，初始化时通常只有门控最后一层首先获得有效梯度；其他模块要在门控偏离恒等后逐步开始学习。这是预期行为。

不要为“让所有参数第一步都有梯度”而同时修改门控、投影和残差初始化。

不过，若训练多个更新步后SCSR仍完全没有有效梯度或残差，则属于必须排查的实现或优化问题。

## 八、配置文件：建议建立独立实验配置

不要覆盖官方`UAD.yml`。

建议为实验A建立独立配置，例如`SCSR.yml`，继承现有UAD全部设置，仅参考增加下表参数。

| 新增配置                      | 建议值       |
| ------------------------- | --------- |
| `SCSR.ENABLED`            | `True`    |
| `SCSR.TARGET_MODALITY`    | `Thermal` |
| `SCSR.INSERT_AFTER_BLOCK` | `6`       |
| `SCSR.IN_DIM`             | `768`     |
| `SCSR.SPECTRAL_DIM`       | `192`     |
| `SCSR.CONDITION_DIM`      | `64`      |
| `SCSR.WAVELET`            | `haar`    |
| `SCSR.WAVELET_LEVELS`     | `1`       |
| `SCSR.GATE_AMPLITUDE`     | `0.5`     |
| `SCSR.RESIDUAL_SCALE`     | `0.2`     |
| `SCSR.EPS`                | `1e-6`    |
| `SCSR.LR_MULTIPLIER`      | `1.0`     |

其中`INSERT_AFTER_BLOCK=6`表示经过第6个Block之后插入，并不表示下标为6的Block执行之前插入。

关闭`SCSR.ENABLED`时，应当能完全恢复原UAD前向路径。这对检查实现是否引入无关变化非常重要。

输出目录建议与原UAD分离，Checkpoint同时记录所使用的SCSR配置。评估时必须创建与该Checkpoint匹配的SCSR模型，不能使用原始无SCSR模型结构强行加载。

## 九、DDP和运行效率：沿用原框架，只关注新增路径

你特别提出不需要复杂化DDP，因此这里仅列出SCSR额外引入的注意事项。

### 9.1 DDP

原UAD已有DDP训练流程，不需要为了SCSR重新设计分布式框架。

只需保证：

- SCSR在模型创建时就注册，随后由原DDP统一管理；
- 每张卡训练时都能正确接收模态标识；
- 各卡的TIR样本都执行同一个SCSR模块；
- 新模块参数随原模型同步梯度；
- 不额外为SCSR引入跨卡通信。

正常PKM训练Batch包含三种模态，因此一般不需要额外处理“某张卡永远不使用SCSR参数”的情况。若实际分卡方式不同，再根据真实情况处理未使用参数问题即可。

### 9.2 AMP及精度

建议：

- 主干仍使用原AMP配置；
- SCSR中的DWT、门控计算和IDWT使用FP32；
- 有界残差的范数使用FP32计算；
- 最终更新量转换回原Patch特征dtype，再继续进入共享ViT。

需要确保新增模块没有意外使后续Transformer全部转为FP32，从而改变显存使用和原基线的混合精度执行方式。

### 9.3 运行开销

按照当前配置，SCSR只处理192通道的16×8中层特征，DWT后空间分辨率为8×4。

因此模块参数较少，但仍有小波变换、数据整理及额外卷积开销。

建议正常记录：

- 训练每步耗时；
- 显存峰值；
- TIR推理延迟；
- 新增参数数量。

这些用于说明模块实际代价，不需要额外开发性能分析系统。

## 十、正式训练前的最小验收标准

这部分不属于额外研究实验，而是确保修改没有破坏原UAD训练协议。

| 核对内容         | 必须满足的结果                          |
| ------------ | -------------------------------- |
| SCSR关闭       | 前向结构与原UAD一致                      |
| SCSR开启且初始化恒等 | 同权重、评估模式下最终特征与原UAD数值一致或处于计算精度误差内 |
| Batch维度      | 输入192张时输出仍为192个768D特征            |
| 中层特征         | 129个Token，其中1个CLS、128个Patch      |
| 小波变换         | 16×8 → 8×4 → 16×8                |
| 子带排列         | 三组高频与门控输出一一对应                    |
| 高频残差性质       | 对应Haar单元内的直接残差零低频成立              |
| 梯度           | 门控末层正常启动，后续参数逐步获得梯度              |
| 模态路由         | 只有TIR执行频率调制                      |
| 身份顺序         | 更新后样本顺序与原Batch完全一致               |
| 训练损失         | CE、WRT、ProCA、GPD均正常计算            |
| 推理路径         | TIR Query/Gallery都经过SCSR         |
| 数值稳定         | 无NaN、Inf、错误dtype或维度问题            |

建议先用一个小Batch完成前向与反向，再通过几次正常优化步骤检查梯度、模型保存与重新加载。

这些检查完成后，便可以直接开展完整120 Epoch训练，不需要新增独立的专家训练或诊断实验。

## 十一、正式训练后需要保留的结果

建议完整SCSR首先与本轮同条件UAD比较。

核心结果包括：

| 评价        | 需要输出                            |
| --------- | ------------------------------- |
| 整体AS-ReID | mAP、Rank-1、Rank-5、Rank-10、mINP  |
| 九模态检索     | RGB/NIR/TIR的3×3 Query→Gallery矩阵 |
| TIR同模态    | TIR→TIR Rank-1、mAP、mINP         |
| 平台组合      | G→G、G→A、A→G、A→A                 |
| TIR平台细分   | TIR-G/TIR-A之间四组检索               |
| 训练状态      | 各原损失变化、门控偏离1的幅度、残差比例            |
| 工程成本      | 参数、显存、吞吐量及推理延迟                  |

主实验结束后，如果结果支持继续研究，再开展容量匹配空间Adapter和HF-Only对照。

不需要在当前第一轮就同时运行所有消融。

## 十二、最终交给开发者的实施摘要

### 实验A：SCSR-V3实施任务

在WHU-MARS/UAD的ViT-B/16主干第6个Transformer Block之后，加入一个仅处理TIR Patch Token的SCSR模块。

模块将768维Patch投影到192维，恢复16×8空间网格，使用一级Haar-DWT分解为一个低频与三个高频子带。将完整四子带拼接，通过`768→64→576`的轻量条件网络生成三组高频门控，以`0.5×tanh`对已有高频进行双向调制。

使用零低频增量和三个高频增量执行IDWT，再经过无偏置192→768线性投影，利用相对有界残差更新原Patch，残差系数固定为0.2。原CLS不直接修改，全部Token继续经过ViT第7—12层。

保持原BNNeck、身份分类器、CE/WRT/ProCA/GPD损失、PKM采样、GPD原型更新及768维pre-BN CLS检索方式不变。

训练和推理都需要使用真实模态标签启用TIR分支，保持原Batch和身份顺序，新增参数使用与主干相同的初始学习率，不增加额外损失。

### 最终判断

经过两轮再次审查以及源码对照，SCSR-V3可以作为实验A的正式首轮实现方案，目前不需要继续修改其结构与损失函数。

整个实施中最容易产生实质性错误的地方是：

① L6插入位置与Token空间排列；② 小波子带顺序及逆变换；③ 近恒等初始化的实际梯度路径；④ TIR模态路由；⑤ Batch顺序与UAD损失输入的一致性。

只要严格处理这五点，就能让首轮实验有效地回答核心研究问题：

在保持UAD统一身份优化不变的情况下，针对TIR中层特征增加结构条件频率调制，究竟能否改善TIR身份判别和完整AS-ReID检索？


# UAD + SCSR-V4.1：最终方案审查与完整代码修改指导

实施目标： 在原WHU-MARS论文提出的UAD模型上，加入可学习的TIR中层频率调制机制，尝试增强热红外身份特征的判别能力，同时保留UAD原有的多模态统一身份学习框架。

最终推荐方案： SCSR-V4.1（低秩动态频率调制，接入Block 7 Attention输入）。

本指导按照“方案审查 → 文献依据 → 数学定义 → 源码修改位置 → 初始化与优化 → DDP注意事项 → 实施核验 → 正式训练”的顺序组织。具体实现可以自行编写，但以下接口、张量关系、参数设置及验证条件应保持一致。

## 一、实施前最后一轮方案审查

首先明确：V4.1不是对V3的简单调参，而是针对已经观察到的退化状态重新设计模块的有效作用路径。

此前V3实测：

| 关键数据                        | 结果                        |
| --------------------------- | ------------------------- |
| 高频门控Logit RMS               | \\(5.12\times10^{-5}\\)   |
| 相对Patch更新                   | \\(3.33\times10^{-8}\\)   |
| 最终CLS相对差异                   | 约 \\(4.79\times10^{-7}\\) |
| 同Checkpoint ON/OFF Rank-1变化 | 0/6,405条Query             |
| 训练后投影参数幅度                   | 约为初始化的81%                 |

这些现象证明V3新增分支几乎没有发挥实际检索作用，但不能唯一断言是双投影、零初始化或Weight Decay某一因素独立导致。

### 1.1 V4.1已经针对性修正的问题

| 原V3/V4问题              | V4.1修改                          |
| --------------------- | ------------------------------- |
| 严格零输出门控，前端可能难以启动      | 非零、非对称初始化                       |
| 768→192→768双投影，学习路径较长 | 直接处理768维特征                      |
| 固定通道排序，无法动态改变通道偏好     | Rank-4通道—空间交互                   |
| 原始Patch与新增残差尺度不一致     | 在原Block 7归一化Attention输入中调制      |
| 多级残差缩放                | 仅保留一个频率调制上界 \\(\gamma\\)        |
| 新分支长期受到正则化收缩          | SCSR参数单独不使用Weight Decay         |
| 原CLS难以感知极小Patch更新     | 直接改变当前Block Attention读取的Patch内容 |

### 1.2 最后发现的一个重要剩余风险

将频率调制放到Block 7 `norm1`之后，并不意味着梯度一定比V3更强。

因为原ViT Block使用：

\\[ Y=X+\operatorname{Attention}(\operatorname{LN}\_1(X)) \\]

V4.1实际改变的是Attention分支输入，而不是残差主干 \\(X\\)。

因此，SCSR参数收到的任务梯度仍需要经过Attention反向传播。尤其当CLS很少读取被修改的Patch时，新增分支依然可能获得较弱梯度。

这是本次实施必须重点检查的问题。

我不建议现在为此增加第二条残差支路或新的Attention。那样会重新引入多个融合机制，使实验目的不清晰。

正确做法是保留简洁的V4.1，在真实身份损失下分别检查：

- SCSR各层参数是否有效更新；
- 增强后的TIR Patch是否真正改变Block 7 Attention输出；
- 最终CLS是否获得实际变化。

如果前两项成立、第三项仍长期接近数值噪声，则不能继续单纯调大门控，而应重新考虑融合位置与CLS信息传递路径。

### 1.3 最终审查意见

目前没有发现必须阻止V4.1实施的明确结构错误，可以固定这一候选方案。

但固定的是方法定义及训练协议，不是事先认定它一定能提高Rank-1或mAP。

本次不再增加额外损失、多尺度小波、前景掩码、独立CLS分支或者额外注意力模块。

## 二、模块参考文献与具体来源

需要区分三种关系：

- 直接保留： 原论文的方法、训练协议或网络组件。
- 机制借鉴： 借鉴论文的核心思想，但没有直接复现其完整方法。
- 本方案设计： 为WHU-MARS任务提出的具体组合与实现选择，不能宣称来自某篇参考论文。

| 来源论文                           | 具体位置                                   | 借鉴内容                       | 本实验的实际处理                                   |
| ------------------------------ | -------------------------------------- | -------------------------- | ------------------------------------------ |
| WHU-MARS / UAD，Zhao等，CVPR 2026 | UAD方法部分，ProCA、GPD及AS-ReID实验协议          | 共享ViT、跨模态身份聚合、全局原型判别       | 原样保留UAD训练和最终检索框架                           |
| NS-FPN，Yuan等，CVPR 2026         | §3.1 LFP，公式（1）—（6），Figure 3            | DWT分解、低频引导高频调整、IDWT重建      | 使用一级Haar，并以LL和高频条件生成动态调制；不采用其Gaussian第二阶段  |
| FreqFiT，Ly与Nguyen，MICCAI 2025  | §3，Figure 1、公式（1）—（4）                  | 在ViT Token表示中进行频域调制与特征适配   | 借鉴ViT频域适配思想；本实验用Haar而非FFT，且改为Attention输入融合 |
| PFESA，Li等，MICCAI 2025          | §2.1频率解耦，§2.2 Edge/Structure Attention | 低/高频互补统计与结构、边缘的差异化处理       | 使用有符号低频特征、高频压缩特征及能量统计；不复制其参数免费注意力公式        |
| AdaptFormer，Chen等，NeurIPS 2022 | §3 Approach，Adapter与Transformer集成机制    | 轻量模块与预训练Transformer兼容的设计原则 | 作为融合选择的参考，不直接使用其并行FFN Adapter              |

其中：

NS-FPN确实在§3.1先进行DWT，再根据低频得到空间权重并调制高频，随后还使用门控Gaussian过滤；我们仅借鉴其低频引导和高频处理的基本机制。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arxiv.org

+1



FreqFiT在§3使用FFT、可学习频域滤波器和残差及缩放/平移进行适配；它没有提出我们这里的Rank-4 Haar门控或Block 7 Attention输入调制。

[image](https://www.google.com/s2/favicons?domain=https://papers.miccai.org\&sz=32)

papers.miccai.org

+1



PFESA使用FFT与Gaussian频带分离，并根据高低频统计构建参数免费注意力；它并不是Haar门控网络。

[image](https://www.google.com/s2/favicons?domain=https://papers.miccai.org\&sz=32)

papers.miccai.org

+1



UAD的ProCA与GPD分别负责跨场景类内聚合和类间身份判别，二者仍作为本实验身份监督的基础。

[image](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com\&sz=32)

openaccess.thecvf.com

+1



### 必须明确标注为本方案设计的内容

Rank-4动态通道—空间门控、非对称非零初始化、在UAD第7层归一化Attention输入中进行TIR-only调制，是针对WHU-MARS任务的组合设计。

不能写成“直接采用NS-FPN的门控”“直接复现FreqFiT”或“采用PFESA的小波注意力”。

论文参考链接：

- [WHU-MARS / UAD — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_WHU-MARS_A_Multispectral_Aerial-Ground_Benchmark_Towards_Any-Scenario_Person_Re-Identification_CVPR_2026_paper.html)
- [NS-FPN — §3.1 LFP](https://arxiv.org/pdf/2508.06878)
- [FreqFiT — §3](https://papers.miccai.org/miccai-2025/paper/3066_paper.pdf)
- [PFESA — §2.1—2.2](https://papers.miccai.org/miccai-2025/paper/3694_paper.pdf)
- [AdaptFormer — §3](https://papers.neurips.cc/paper_files/paper/2022/file/69e2f49ab0837b71b0e0cb7c555990f8-Paper-Conference.pdf)

## 三、正式固定的SCSR-V4.1结构

### 3.1 整体设计与插入位置

使用原UAD的ViT-B/16，共12个Transformer Block。

SCSR仅插入一次：

Block 6执行结束 → Block 7 `norm1` → SCSR处理TIR Patch → Block 7原Self-Attention。

注意：这里不是在Block 6的内部，也不是Block 7输出之后。

原Block 7的计算是：

\\[ Z=\operatorname{LN}\_1(X_6) \\]

\\[ Y=X_6+\operatorname{Attn}(Z) \\]

\\[ X_7=Y+\operatorname{MLP}(\operatorname{LN}\_2(Y)) \\]

修改为：

\\[ Z=\operatorname{LN}\_1(X_6) \\]

\\[ Z'=\operatorname{SCSR}\_{V4.1}(Z,\mathbf m) \\]

\\[ \boxed{Y=X_6+\operatorname{Attn}(Z')} \\]

\\[ X_7=Y+\operatorname{MLP}(\operatorname{LN}\_2(Y)) \\]

其中 \\(\mathbf m\\) 表示当前Batch中每张图像的真实模态编号。

SCSR只替换TIR对应的Patch部分，不改变CLS Token，不改变原残差主干 \\(X_6\\)。

其余11个Transformer Block完全保留原实现。

### 3.2 输入与输出规格

原ViT输入尺寸固定：

\\[ 256\times128 \\]

Patch大小及Stride：

\\[ 16\times16 \\]

因此：

\\[ H_p=16,\qquad W_p=8 \\]

共128个Patch，加上CLS共129个Token。

| 对象            | 维度                   |
| ------------- | -------------------- |
| Block 7归一化输入  | \\([B,129,768]\\)    |
| CLS           | \\([B,1,768]\\)      |
| TIR Patch     | \\([B_T,128,768]\\)  |
| TIR Patch二维特征 | \\([B_T,768,16,8]\\) |
| 每个Haar子带      | \\([B_T,768,8,4]\\)  |
| SCSR输出        | 与输入完全相同              |

这里 \\(B_T\\) 是该Batch中真实TIR样本数量。

模态编码继续采用：

- RGB = 1
- IR/NIR = 2
- Thermal/TIR = 3

务必通过真实模态ID判断TIR，不能根据`mode`、平台、相机或者Batch固定位置直接推测模态。

`mode`仍只负责原UAD的训练/评估接口逻辑。

### 3.3 Haar分解及逆变换

对每个通道的2×2局部区域：

\\[ \begin{bmatrix} a&b\\\c&d \end{bmatrix} \\]

固定使用以下正交Haar约定：

\\[ LL=\frac{a+b+c+d}{2} \\]

\\[ LH=\frac{a+b-c-d}{2} \\]

\\[ HL=\frac{a-b+c-d}{2} \\]

\\[ HH=\frac{a-b-c+d}{2} \\]

Haar变换与逆变换必须使用同一符号及顺序约定。

模块不能将`LH`与`HL`交换，也不能使用先拼接子带、再直接reshape为原空间图像的方式代替IDWT。

保留：

\\[ L=LL,\qquad H_1=LH,\ H_2=HL,\ H_3=HH \\]

本实验只允许调整三个高频子带。LL原样参与重建，但其信息可用于预测门控。

### 3.4 低频及高频条件构建

第一部分：有符号低频结构

使用无Bias的1×1卷积：

\\[ W_L:768\rightarrow16 \\]

得到：

\\[ S_L=W_L(LL) \\]

维度：

\\[ [B_T,16,8,4] \\]

保留正负响应，不要先对全部通道做简单有符号平均，否则可能因归一化而抵消有用信息。

第二部分：各方向有符号高频

对每一个高频子带单独使用无Bias的1×1投影：

\\[ W\_{H,j}:768\rightarrow4 \\]

得到三组：

\\[ S\_{H,j}=W\_{H,j}(H_j) \\]

每组维度：

\\[ [B_T,4,8,4] \\]

三个投影不共享参数，避免不同方向被强制使用相同的特征压缩方式。

第三部分：各方向高频能量

对每个高频子带的768个通道计算RMS：

\\[ E_j(h,w)= \sqrt{\frac{1}{768}\sum_c H_j(c,h,w)^2+10^{-6}} \\]

得到：

\\[ E_j\in\mathbb R^{B_T\times1\times8\times4} \\]

这是幅度统计，不是身份置信度，也不能直接据此判断哪些位置是人体。

将全部条件拼接：

\\[ S=[S_L;S\_{H,1};S\_{H,2};S\_{H,3};E_1;E_2;E_3] \\]

最终：

\\[ \boxed{S\in\mathbb R^{B_T\times31\times8\times4}} \\]

### 3.5 动态空间条件生成

使用两层轻量卷积：

\\[ 31 \xrightarrow{\operatorname{Conv}\_{3\times3}} 32 \xrightarrow{\operatorname{GELU}} 32 \xrightarrow{\operatorname{Conv}\_{1\times1}} 12 \\]

具体要求：

| 层       | 输入→输出 | Kernel | Padding | Bias  |
| ------- | ----- | ------ | ------- | ----- |
| 第一层Conv | 31→32 | 3×3    | 1       | False |
| 激活      | 32→32 | GELU   | —       | —     |
| 第二层Conv | 32→12 | 1×1    | 0       | False |

第二层输出：

\\[ V\in\mathbb R^{B_T\times12\times8\times4} \\]

按照：

\\[ 3\times R,\qquad R=4 \\]

重排为：

\\[ \boxed{ V\in\mathbb R^{B_T\times3\times4\times8\times4} } \\]

三个高频方向分别拥有4张输入相关空间条件图。

此处不增加额外的Sigmoid、归一化层或零初始化输出层。

### 3.6 Rank-4动态通道—空间门控

建立可学习参数：

\\[ U\in\mathbb R^{3\times768\times4} \\]

然后按照同一个高频方向进行低秩组合：

\\[ T_j(c,h,w) = \sum\_{r=1}^{4} U\_{j,c,r}V\_{j,r}(h,w) \\]

最终门控：

\\[ \boxed{ M_j(c,h,w)=0.2\tanh(T_j(c,h,w)) } \\]

其中：

\\[ -0.2\<M_j<0.2 \\]

这意味着：

\\[ 0.8<1+M_j<1.2 \\]

这里允许高频增强和抑制，而不是只能进行单向高频过滤。

与上一版静态通道偏置加空间门控的不同之处在于，\\(T_j\\)通过多个动态空间基进行组合。

所以不同通道的门控大小顺序可以随图像和位置变化，不再固定。

但仍应将其准确称为低秩动态通道—空间交互，而不是完全自由的768维逐通道门控。

### 3.7 高频调制与重建

分别计算：

\\[ H_j'=(1+M_j)\odot H_j \\]

然后：

\\[ Z_P'= \operatorname{IDWT}(LL,H_1',H_2',H_3') \\]

恢复维度：

\\[ [B_T,128,768] \\]

SCSR返回的是增强后的完整Patch特征，不是仅返回高频增量。

不再经过：

- 768→192降维；
- 192→768升维；
- 外部0.2残差；
- 基于原始Patch范数的平滑缩放。

这里唯一保留的幅度控制是门控中的：

\\[ \gamma=0.2 \\]

由于正交Haar具有能量守恒性质，在精确计算下：

\\[ \\|Z_P'-Z_P\\|\_F \le0.2\\|H\\|\_F \le0.2\\|Z_P\\|\_F \\]

这保证的是归一化Attention Patch输入的整体频率扰动上界，不是最终Attention输出、CLS变化或身份性能的保证。

### 3.8 参数量

按照上述全部投影和卷积无Bias、无额外仿射层计算：

| 组件                | 参数量    |
| ----------------- | ------ |
| LL投影768→16        | 12,288 |
| 三个高频投影768→4       | 9,216  |
| 3×3空间卷积31→32      | 8,928  |
| 1×1输出卷积32→12      | 384    |
| Rank-4通道基 \\(U\\) | 9,216  |
| 总计                | 40,032 |

因此，新增SCSR模块参数应严格为40,032。

不应因为额外添加Bias、LayerNorm、通道注意力或残差投影，导致参数数量发生未说明的变化。

## 四、固定初始化与优化器设置

这一部分是吸取V3教训后最需要认真落实的工程要求。

### 4.1 不允许再次出现双侧弱初始化

门控为：

\\[ M=0.2\tanh(UV) \\]

对应梯度：

\\[ \frac{\partial M}{\partial U}\propto V,\qquad \frac{\partial M}{\partial V}\propto U \\]

如果 \\(U,V\\) 都极小，就会同时削弱门控输出及两侧梯度。

因此固定如下初始化方案。

| 参数          | 初始化要求                              |
| ----------- | ---------------------------------- |
| LL投影        | 正常非零初始化，建议Kaiming                  |
| 三个高频投影      | 正常非零初始化，建议Kaiming                  |
| 31→32空间卷积   | Kaiming初始化                         |
| 32→12输出卷积   | 权重非零高斯初始化，初始std=0.02               |
| 通道基 \\(U\\) | \\(\mathcal N(0,0.5^2)\\)          |
| 所有卷积Bias    | 不设置                                |
| 门控上界        | \\(\gamma=0.2\\)                   |
| 动态条件初始输出    | 在真实Batch中检查 \\(V\\) 的RMS约0.05—0.10 |

这里的0.5是 \\(U\\) 的标准差，不是门控系数。

在四个低秩分量量级相近、近似独立的情况下，\\(U\\)使用0.5标准差可以使 \\(UV\\) 的典型RMS与 \\(V\\) 处于相近量级。

但这只是初始化尺度推导，不能代替实际检查。

具体操作应当是：在正式创建Optimizer前，用固定的真实UAD训练Batch进行一次无梯度前向，检查 \\(V\\)、\\(UV\\)、门控 \\(M\\) 的分布。

如果初始输出明显偏离目标范围，只允许对最终32→12卷积的初始化尺度作一次调整，重新核验后冻结初始化配置。不能在正式训练过程中持续根据观测结果调整初始化权重。

同时检查：

\\[ \Pr(|UV|>2) \\]

避免门控在训练开始时就大量进入Tanh饱和区。

### 4.2 不使用零初始化融合系数

明确禁止以下做法：

- 最后一层门控权重全部初始化为0；
- \\(U\\)和输出条件层同时初始化为0；
- 在门控之后增加初始为0的可学习残差系数；
- 增加第二个固定0.2残差系数；
- 使用额外的`detach()`截断 \\(U,V\\) 的梯度。

也不要为了恢复严格恒等初始化，将整个模块输出再次乘以一个可学习的零标量。

初始SCSR-ON可以与SCSR-OFF存在适度差异；这不是错误。

### 4.3 优化器参数组

原UAD采用SGD、Momentum=0.9、Weight Decay=1e-4，并使用原学习率调度。

V4.1建议：

| 参数组                | LR        | Weight Decay | Momentum |
| ------------------ | --------- | ------------ | -------- |
| 原ViT               | 原UAD      | 原UAD         | 0.9      |
| 原BNNeck/Classifier | 原UAD      | 原UAD         | 0.9      |
| SCSR-V4.1全部参数      | 与主干基础LR相同 | 0            | 0.9      |

必须注意原UAD优化器会对名称包含`bias`的参数使用额外LR系数。V4.1目前没有Bias，但仍建议对SCSR使用显式参数组，确保之后即使局部实现发生变化，也不会意外采用Bias LR放大。

本轮不要额外设置SCSR两倍或十倍学习率。

也不建议单独给 \\(U\\) 一个更大学习率，而给 \\(V\\) 另一个学习率。那会重新增加优化变量，难以判断V4.1的真实效果。

### 4.4 训练梯度应如何检查？

不能仅检查：

\\[ \nabla\_\theta L\neq0 \\]

还应该记录实际参数更新：

\\[ r\_{\mathrm{update}}= \frac{\\|\theta\_{t+1}-\theta_t\\|\_2} {\\|\theta_t\\|\_2+\epsilon} \\]

并分别记录：

- 通道基 \\(U\\)；
- LL投影；
- 高频投影；
- 第一层空间卷积；
- 第二层输出卷积。

另外需要记录CE、WRT单独梯度，以及完整UAD联合损失对应的梯度或实际优化更新。

只有参数确实发生了持续的、与任务梯度相关的变化，才能说明模块获得了有效优化机会。

但参数更新幅度增大仍然不能证明它学到了正确的身份信息。

## 五、具体源码修改指导

### 5.1 应使用哪份工程？

以已验证的SCSR-V3工程复制出独立的V4.1工程，不直接覆盖V3，也不修改作为对照的原UAD工程。

建议保留：

```
WHU_UAD/          原UAD基线

WHU_SCSR_V3/      原实验A及诊断代码

WHU_SCSR_V41/     本次新实验
```

理由是V3已完成模态路由、DDP训练、统一评估、Checkpoint配置校验、AMP更新统计等基础工作。

这些能力应复用，不必重新开发。

但要注意：保留V3的工程基础，不代表保留V3的网络结构。

V4.1必须完全替换V3频率模块，不能将两个模块串联。

### 5.2 修改文件总览

| 文件                               | 需要修改的内容                   | 重要程度      |
| -------------------------------- | ------------------------- | --------- |
| `model/scsr_v41.py`              | 新建V4.1全部频率调制计算            | 必须        |
| `model/backbones/vit_pytorch.py` | 在第7个Block `norm1`之后接入SCSR | 必须        |
| `model/make_model.py`            | 统一训练/推理模态ID及模块接口          | 必须        |
| `config/defaults.py`             | 增加V4.1配置项                 | 必须        |
| `configs/SCSR_V41.yml`           | 独立实验配置                    | 必须        |
| `solver/make_optimizer.py`       | SCSR独立参数组，WD=0            | 必须        |
| `processor/processor.py`         | 确保训练、验证均传递真实模态ID，并输出诊断    | 视现有V3实现核查 |
| `datasets/make_dataloader.py`    | 核对模态顺序和Batch契约            | 原则上不改     |
| `loss/`及原UAD损失文件                 | 不修改                       | 保持        |
| `utils/metrics.py`               | 不修改检索协议                   | 保持        |
| `train.py`、`test.py`             | 不修改主要运行逻辑                 | 尽量保持      |

以下按文件明确说明。

### 5.3 `model/scsr_v41.py`

新建一个独立模块，负责全部频率计算。

建议接口：

- 输入：经过Block 7 `norm1`的Token，以及当前Batch真实模态ID。
- 输出：形状、Token顺序和dtype与输入一致的Token。
- 只修改TIR的128个Patch。
- CLS与非TIR Token保持不变。

内部处理顺序固定为：

1. 根据真实模态ID选择TIR样本。
2. 分离CLS和128个Patch，恢复16×8空间排列。
3. 在FP32计算路径中执行一级正交Haar。
4. 由LL、有符号高频和高频RMS生成31通道条件。
5. 经过31→32→12条件网络，生成Rank-4动态空间基。
6. 与可学习通道基 \\(U\\) 组合，经过Tanh得到门控。
7. 分别调制LH、HL、HH，LL保持不变。
8. 通过精确IDWT恢复768维增强Patch。
9. 将增强Patch写回原TIR样本位置，保持非TIR和CLS不变。
10. 返回与输入dtype一致的完整Token序列。

实现时尤其注意：

第一，Haar和门控内部使用FP32。 原训练环境为PyTorch 1.12.1，可以在SCSR内部暂时关闭AMP autocast完成计算，之后再转换回原Attention输入dtype。

第二，避免不安全的原地操作。 不建议直接对原输入Token切片执行会影响Autograd版本计数的原地赋值，应采用保持计算图的索引写回方式。

第三，若Batch没有TIR，可以直接返回原输入。 不需要构造虚假的TIR样本，也不需要计算无意义的频率分支。

第四，不要额外创建未使用的前向模块。 只保留实际参与计算的参数。

第五，不能把高频方向当作普通通道随意交换或reshape。

诊断接口可以返回统计值，但正常训练和评估的模型输出契约应与UAD保持一致。

### 5.4 `model/backbones/vit_pytorch.py`

这是整次修改最需要谨慎的文件。

原UAD的Transformer Block核心执行顺序为：

\\[ x=x+\operatorname{dropPath} \left(\operatorname{attn}(\operatorname{norm1}(x))\right) \\]

\\[ x=x+\operatorname{dropPath} \left(\operatorname{mlp}(\operatorname{norm2}(x))\right) \\]

修改时应该只在第7个Block调用Attention之前增加一个可选Token调制入口。

具体要求：

- 仍先使用原Block 7自己的`norm1`；
- 对`norm1`输出执行SCSR；
- 再传入原`attn`；
- Attention结果依旧加回原Block输入主干；
- 原`drop_path`和`norm2`、MLP执行顺序不变。

原始残差主干绝不能被替换为SCSR输出。

即不能错误实现成：

\\[ Y=Z'+\operatorname{Attn}(Z') \\]

正确应当是：

\\[ \boxed{ Y=X_6+\operatorname{Attn}(Z') } \\]

还有几个重要细节：

Block编号： PyTorch中`blocks[6]`是第7个Block，不是第6个。

模块注册： SCSR只注册一次，建议注册在Backbone中。不要在每一个Transformer Block内部各创建一个SCSR，也不要重复注册同一组可学习参数。

禁用开关： 关闭SCSR时应完全跳过调制，恢复原始：

\\[ \operatorname{Attn}(\operatorname{norm1}(X_6)) \\]

预训练加载： 保持原ViT参数名称和形状，不修改Q/K/V投影、Attention头数、MLP或原LayerNorm参数。

最终特征： 仍经过原第12个Block、最终LayerNorm，输出原CLS。

### 5.5 `model/make_model.py`

负责把模态信息正确传入Backbone。

UAD原训练模型接受三模态图像列表，将它们拼接后输入共享ViT。

V4.1必须明确保持这条路径。

训练时：

\\[ [\text{RGB batch};\text{NIR batch};\text{TIR batch}] \\]

对应模态标签：

\\[ [1,\dots,1;\ 2,\dots,2;\ 3,\dots,3] \\]

不能仅依赖拼接后的图像位置在Backbone内部随意推断模态。

建议模型封装层根据实际`cfg.DATASETS.MODALITIES`及传入列表顺序，显式生成与图像一一对应的模态ID。

验证/推理时，直接使用DataLoader提供的`modid`，转换为与图像同设备的整型Tensor，再传入模型。

非常重要：`mode=1/2/3`是原评测流程选择模态数据Loader的接口参数，不能代替真正的`modal_ids`。

即使目前三种模态恰好对应1、2、3，也应在模型接口中区分两者。

这样以后调整评测顺序、混合Query Batch或者进行单模态推理时，才不会发生隐蔽错误。

本文件还需要：

- 增加SCSR启用状态；
- 使Backbone能够接受`modal_ids`；
- 保持训练返回`cls_score, global_feat, feat`；
- 保持推理只返回原规定的身份特征。

### 5.6 `config/defaults.py`和`configs/SCSR_V41.yml`

建议使用独立配置命名空间，例如`SCSR_V41`，避免与V3配置混用。

核心配置固定为：

| 配置项                       | 值                            |
| ------------------------- | ---------------------------- |
| ENABLED                   | True                         |
| TARGET_MODALITY           | Thermal                      |
| INSERT_AT_BLOCK           | 7                            |
| INSERT_POSITION           | norm1_after_before_attention |
| WAVELET                   | haar                         |
| WAVELET_LEVELS            | 1                            |
| EMBED_DIM                 | 768                          |
| GRID_H / GRID_W           | 16 / 8                       |
| LL_CONDITION_DIM          | 16                           |
| HF_SIGNED_DIM             | 4                            |
| CONDITION_HIDDEN_DIM      | 32                           |
| LOW_RANK                  | 4                            |
| GATE_AMPLITUDE            | 0.2                          |
| U_INIT_STD                | 0.5                          |
| CONDITION_OUTPUT_INIT_STD | 0.02                         |
| SCSR_WEIGHT_DECAY         | 0                            |
| SCSR_LR_MULTIPLIER        | 1.0                          |
| EPS                       | 1e-6                         |

其中`INSERT_AT_BLOCK=7`明确表示操作发生于第7个Transformer Block内部。

不要同时使用旧V3的`INSERT_AFTER_BLOCK=6`作为另一个生效控制项，避免同一个位置存在两套不同配置语义。

原UAD的其他超参数、输入分辨率和损失配置继续保持不变。

### 5.7 `solver/make_optimizer.py`

在原参数分组逻辑上增加SCSR独立分组即可。

只需要实现：

- 检测SCSR-V4.1命名空间下的参数；
- 使用基础LR；
- Weight Decay固定为0；
- 保持SGD Momentum=0.9；
- 保持原学习率调度；
- 其余参数沿用原UAD规则。

不要增加第二个Optimizer，也不要单独为SCSR创建另一套Scheduler。

还要检查所有SCSR参数都被Optimizer包含，而且同一参数没有进入多个组。

这比修改原UAD整体优化策略更简单，也便于保持公平对照。

### 5.8 `processor/processor.py`

复用V3已经实现的训练及评测流程，不重写原UAD损失逻辑。

主要核查：

训练阶段

- 三模态图像和身份标签顺序对应；
- 模态ID与实际图像一一对应；
- 分类损失仍使用原BNNeck后分类输出；
- WRT、ProCA、GPD仍使用原规定的pre-BN CLS；
- ProCA身份中心构造与梯度截断方式不变；
- GPD的原型更新、跨进程聚合逻辑不变。

验证和推理阶段

- 必须显式传递真实模态ID；
- SCSR在训练及推理中都使用相同的Block 7接入方式；
- 原768维pre-BN CLS作为检索特征；
- 原L2归一化、距离计算及同相机过滤不变；
- 不引入Re-ranking或TTA。

训练日志增加最小必要诊断：

| 指标                            | 目的                   |
| ----------------------------- | -------------------- |
| \\(V\\)和\\(UV\\)的RMS、极值       | 检查弱启动和Tanh饱和         |
| 门控 \\(\lvert M\rvert\\)均值/RMS | 判断频率调制幅度             |
| 三个高频方向的门控分布                   | 判断是否有方向差异            |
| \\(r\_{\mathrm{attn}}\\)      | 检查实际Attention输入修改    |
| Block 7 Attention输出变化         | 检查调制是否进入注意力计算        |
| 最终CLS ON/OFF差异                | 检查身份表征变化             |
| SCSR参数梯度与实际更新比例               | 检查有效学习               |
| AMP跳过更新次数                     | 判断实际Optimizer Step数量 |

这些信息可在初始化、前100次有效更新、前几个Epoch记录；不需要每个Iteration都保存全部中间Tensor。

### 5.9 原有文件不建议修改的部分

以下内容原则上保持原UAD实现：

| 部分             | 处理 |
| -------------- | -- |
| 数据集划分          | 保持 |
| PKM采样策略        | 保持 |
| WRT损失          | 保持 |
| ProCA          | 保持 |
| GPD            | 保持 |
| 分类器与BNNeck     | 保持 |
| 最终CLS输出        | 保持 |
| CMC/mAP/mINP计算 | 保持 |
| 相机过滤和Gallery定义 | 保持 |

尤其不要为了“让SCSR更有梯度”而偷偷调整GPD权重、ProCA权重、Triplet Margin或训练数据采样规则。

这些都应保留为UAD基线契约。

## 六、DDP与AMP的必要注意事项

不需要重新设计DDP。直接沿用V3经过验证的4卡训练框架。

### 6.1 DDP

原UAD已使用：

- DistributedDataParallel；
- PKM分布式采样；
- 原GPD跨进程原型同步；
- `find_unused_parameters=True`。

V4.1只需要额外确认：

第一，每个Rank的模态ID与本Rank图像顺序一致。 不同Rank不能根据全局Batch下标判断模态。

第二，正常PKM训练时每个Rank均有TIR样本。 若某Rank不存在TIR，分支会被跳过，需要确保原DDP未使用参数处理逻辑兼容该情况。

第三，SCSR内部不要额外添加AllGather、AllReduce或其他进程通信。 它是每张图像独立执行的局部特征操作，不需要跨GPU同步频率统计。

第四，参数初始化在各Rank保持一致。 建议保留原DDP参数广播机制，并检查首次Optimizer Step前SCSR权重一致。

### 6.2 AMP

由于此前V3的FP32诊断发现了极小的特征更新，本次建议：

- 原UAD主干继续使用原AMP；
- SCSR内部Haar、能量统计、门控及IDWT使用FP32；
- 返回Attention之前转换为原`norm1`输出dtype；
- 不改变原GradScaler与梯度裁剪机制。

特别要检查：

转换dtype前后，实际调制是否仍然存在。

记录实际进入Attention的增强Patch与原Patch的差异，而不只看SCSR内部FP32增量。

原UAD的AMP可能跳过Optimizer Step。有效训练更新次数必须以真正完成的Optimizer Step为准。

原GPD原型更新在AMP跳步时是否仍执行，需要核对V3已有实现；本轮不建议只对新方法单独改变原型银行更新规则，否则会影响基线可比性。

## 七、实施完成后必须通过的验证

这是本次最重要的验收部分。

不要重复V3只做数学、形状和梯度非零测试，就直接训练120 Epoch的做法。

需要分为三个层次。

### 7.1 第一层：实现正确性验证

| 测试         | 通过条件                           |
| ---------- | ------------------------------ |
| 参数量        | 精确40,032                       |
| 输入输出       | `[B,129,768]`保持一致              |
| Haar可逆性    | FP32下正逆重建最大误差达到约1e-5或更小的合理数值水平 |
| Haar能量     | 正交变换前后能量在浮点容差内一致               |
| CLS旁路      | SCSR内部不直接修改CLS                 |
| RGB/NIR旁路  | 同权重ON/OFF输出一致                  |
| TIR路由      | 实际只修改TIR样本                     |
| IDWT       | LL不被直接调制，三高频方向对应正确             |
| 原UAD关闭等价性  | 同权重、相同输入，禁用SCSR后的完整网络与原UAD前向一致 |
| 梯度连通       | 所有SCSR参数能够收到预期梯度               |
| Checkpoint | 保存、严格加载后结果一致                   |

其中，关闭等价性是必须通过的检查。

不能因为V4.1修改了Block.forward接口，就意外改变了所有模态的原始Attention计算。

### 7.2 第二层：真实数据初始化验证

使用真正的WHU-MARS训练Batch、真实ViT特征，而不是随机合成Token。

建议检查：

| 测量                       | 初始参考或要求                             |
| ------------------------ | ----------------------------------- |
| \\(V\\) RMS              | 约0.05—0.10                          |
| \\(UV\\) RMS             | 同量级，避免接近零或严重饱和                      |
| \\(\Pr(\|UV\|>2)\\)      | 应很低，初始不应大量饱和                        |
| 门控 \\(M\\)               | 非零、不同样本存在变化                         |
| \\(r\_{\mathrm{attn}}\\) | 初步参考约 \\(10^{-3}\\) 至 \\(10^{-2}\\) |
| 回写dtype后变化               | 仍然可测量                               |
| Block 7 Attention输出      | 对TIR存在非零差异                          |
| 完整CLS                    | 可测量差异，不应只有机器精度附近的扰动                 |
| CE/WRT/ProCA/GPD         | 数值有限，无异常爆炸                          |

上述范围属于初始化数值核查参考，不是需要通过损失函数强制维持的数值目标。

如果达到可测量的Attention输入变化，却几乎不改变Attention输出和CLS，说明问题可能在实际信息读取路径，而不只是初始化尺度。

这时应先检查CLS到Patch的注意力响应以及相应梯度，不能立刻加大 \\(\gamma\\)。

### 7.3 第三层：真实优化过程验证

建议使用正式训练配置进行一次前100—200个有效Optimizer Step的短期核查。

记录每个SCSR参数组的：

\\[ r\_{\mathrm{update}}= \frac{\\|\theta\_{t+1}-\theta_t\\|\_2} {\\|\theta_t\\|\_2+\epsilon} \\]

同时比较：

\\[ \\|U_t-U_0\\|,\quad \\|W\_{L,t}-W\_{L,0}\\|,\quad \\|W\_{H,t}-W\_{H,0}\\| \\]

以及条件卷积的参数变化。

不能只比较参数范数是否增大，因为参数可能改变方向但范数变化很小。

| 检查问题             | 必须得到的证据                 |
| ---------------- | ----------------------- |
| 是否真正更新了全部SCSR参数？ | 实际Optimizer Step前后参数变化  |
| 是否再次出现弱梯度？       | 完整损失下的参数梯度量级            |
| 是否存在更新失衡？        | \\(U\\)、条件网络及高频投影更新比例对比 |
| 是否发生Tanh饱和？      | \\(UV\\)分布及饱和比例         |
| 是否影响最终CLS？       | ON/OFF完整CLS差异           |
| 是否影响原训练稳定性？      | 原四项损失及梯度有限              |
| 是否受AMP影响？        | 实际dtype、跳步次数和有效更新       |

这里建议同时记录同一Checkpoint下开启与关闭V4.1的最终特征差异。

但不要在短期训练中频繁计算全部93,609张Gallery的完整检索，避免诊断成本过高。

### 7.4 明确什么情况下应停止完整训练

只有出现以下实现或优化问题，才建议先停止并修复：

- Haar或模态路由不正确；
- 新增参数没有真正进入Optimizer；
- 大量NaN、Inf或严重Tanh饱和；
- 完整身份损失无法回传至关键条件层；
- 若干有效更新后，SCSR仍几乎完全处于初始状态；
- Attention输入变化持续消失，或者Attention与CLS完全无法获得有效响应；
- 原UAD训练过程发生明显数值异常。

第5—10 Epoch没有mAP提升，不是自动停止训练的理由。

短期验收解决的是“模块有没有正常且有效地参与学习”，而不是判断最终身份检索一定会提升。

## 八、正式完整训练与实验对照

完成上述检查后，按原UAD条件进行完整训练。

| 配置                | 固定要求                         |
| ----------------- | ---------------------------- |
| 数据集               | WHU-MARS-1000                |
| 初始化               | 与原UAD相同ImageNet预训练ViT        |
| 训练方式              | DDP，4 GPU                    |
| 全局采样Batch         | 256                          |
| 每GPU采样位置          | 64                           |
| 每GPU实际三模态图像       | 192                          |
| 基础LR              | 0.032                        |
| 优化器               | SGD，Momentum=0.9             |
| 主干Weight Decay    | 1e-4                         |
| SCSR Weight Decay | 0                            |
| Epoch             | 120                          |
| Warmup            | 5 Epoch                      |
| 随机种子              | 1234                         |
| Loss              | 原CE + WRT + 0.01 ProCA + GPD |
| 检索特征              | 768D pre-BN CLS              |
| Re-ranking        | False                        |
| TTA               | False                        |
| Camera Filter     | 原协议                          |

注意两点。

第一，应该从与UAD一致的ImageNet预训练初始化开始，而不是从已完成120 Epoch的V3权重接着训练。

否则无法与原UAD DDP直接比较。

第二，新增SCSR的参数初始化不能改变原主干和分类器的随机初始化轨迹。

可以保留V3已有的随机数状态保护机制，确保在相同种子下，SCSR-OFF的初始原网络参数与UAD一致。通过逐参数比较确认，而不是仅认为两组使用相同seed就必然等价。

### 最终评测项目

正式训练后至少报告：

| 评测范围                    | 指标                             |
| ----------------------- | ------------------------------ |
| AS-ReID整体               | mAP、Rank-1、Rank-5、Rank-10、mINP |
| 九组模态组合                  | Rank-1、mAP、mINP                |
| TIR→TIR                 | 重点检查身份内部判别                     |
| RGB/NIR↔TIR             | 检查跨模态检索                        |
| Ground↔Aerial           | 检查平台变化影响                       |
| TIR专项Ground/Aerial      | 检查TIR与平台的共同作用                  |
| 同Checkpoint SCSR-ON/OFF | 检查模块直接推理贡献                     |
| 逐Query排名转换              | 分析正确性修复与退化                     |
| SCSR梯度、门控与CLS变化         | 检查机制是否实际得到利用                   |

最终成功标准不是单纯“门控显著非零”，也不是“出现若干Query排名变化”。

真正希望观察的是：TIR身份判别和跨模态/跨平台检索关系获得有意义且较稳定的改善，同时整体AS-ReID性能不因其他方向退化而被抵消。

## 九、交付给代码修改人员的实施边界

为了避免实现过程中重新引入未经审查的变化，建议明确以下范围。

| 允许                       | 不允许                   |
| ------------------------ | --------------------- |
| 新建SCSR-V4.1模块            | 保留V3模块与V4.1串联         |
| 修改第7个Block的Attention输入路径 | 修改全部Transformer Block |
| 新增真实模态ID传递               | 使用评测`mode`冒充模态        |
| 新增SCSR参数组                | 修改原UAD整体Optimizer规则   |
| 增加必要诊断日志                 | 新增身份损失或频率损失           |
| FP32实现Haar和门控            | 改变原UAD主干AMP训练策略       |
| 完整ON/OFF与梯度核验            | 只凭Loss下降宣称模块有效        |
| 保留原UAD、V3独立归档            | 覆盖原实验代码与Checkpoint    |

另外，代码修改完成后应提交一份简短的实现核对说明，明确：

- 实际修改了哪些源文件；
- 哪些模块来自参考论文的机制借鉴；
- 新增网络实际参数量和初始化分布；
- Block 7的具体接入顺序；
- 训练/验证模态ID是否一致；
- 所有测试是否通过；
- 是否实际完成CUDA/AMP/DDP短期验证；
- 是否存在与这份指导不同的实现决策。

不应把CPU单元测试通过写成“已验证真实训练有效”，也不应在没有训练结果前报告性能改善。

## 十、最终实施结论

建议正式固定SCSR-V4.1，按上述规范进行代码修改。

本版本的核心是：

\\[ \boxed{ \begin{gathered} \text{UAD Block 7 LN1输出}\\\ \downarrow\\\ \text{TIR 768维Patch Haar分解}\\\ \downarrow\\\ \text{低频结构 + 有符号高频 + 高频能量条件}\\\ \downarrow\\\ \text{Rank-4动态通道—空间门控}\\\ \downarrow\\\ \text{高频双向调制 + IDWT}\\\ \downarrow\\\ \text{原UAD Self-Attention与CLS聚合} \end{gathered} } \\]

相对于V3，本次不再依靠多个随机通道投影生成微小残差，而是直接在原ViT归一化注意力输入中调整TIR空间频率信息。

同时通过明确的非对称初始化、单一门控幅度、独立优化器参数组，以及真实UAD损失下的有效更新核验，避免再次出现“模块运行正确，但几乎没有学到有效特征调整”的问题。

需要坚持最后一条原则：这版设计提高了模块获得有效学习机会的合理性，但没有提前证明高频调制具有身份判别价值。

因此下一步最合理的工作顺序就是：复制V3工程 → 替换为V4.1 → 实现正确性检查 → 真实Batch初始化与梯度验收 → 短期优化检查 → 完整120 Epoch实验。

在这些步骤之外，暂不添加额外机制，也不继续凭理论猜测反复修改模型。


# 以已经成功复现 `UAD_1000_DDP` 的原始UAD工程为基础，新建SCSR-V4.1实验；从V3中选择性迁移经过验证的工程功能，而不是直接在V3网络结构上继续修改。

这是目前最稳妥、也最有利于后续论文实验归因的方式。

## 一、到底选择原UAD还是V3？

| 方案           | 优点              | 主要风险                      | 建议            |
| ------------ | --------------- | ------------------------- | ------------- |
| 原UAD上直接开发    | 原始网络干净，实验变量明确   | 需要重新接入模态路由、诊断等功能          | 推荐，但应复用V3可靠代码 |
| V3上直接替换模块    | 训练与诊断基础设施完善，改动少 | 可能残留V3旧插入逻辑、旧参数组和残差路径     | 不推荐直接修改       |
| 全新下载官方UAD再修改 | 来源明确            | 可能缺少你本地已验证的DDP、数据集配置和评测扩展 | 不必要           |

推荐结构：

```
UAD_1000_DDP/          # 已成功复现的原UAD，保持不变

SCSR_V3/               # 实验A，代码、结果及诊断全部归档

SCSR_V41/              # 从原UAD复制的新工程
    原UAD完整主干
    + V4.1新频率模块
    + 必要的真实模态ID传递
    + V3中可靠的训练诊断功能
```

注意，我说的“原UAD”是你实际跑出UAD DDP基线的那份工程，不是重新下载一个可能缺少本地改动的官方仓库。

## 二、为什么不建议直接修改V3？

因为V3和V4.1的接入方式存在根本区别。

V3：

\\[ X_6'=X_6+\Delta X\_{\mathrm{SCSR}} \\]

然后进入Block 7。

V4.1：

\\[ Z=\operatorname{LN}\_1(X_6) \\]

\\[ Z'=\operatorname{SCSR}\_{V4.1}(Z) \\]

\\[ Y=X_6+\operatorname{Attention}(Z') \\]

因此，如果直接在V3上叠加V4.1，容易产生：

- V3残差和V4.1门控同时执行；
- V3的768→192→768投影仍然被注册或参与计算；
- V3的残差尺度参数继续生效；
- V3旧插入点与V4.1的新插入点同时存在；
- 旧优化器参数组、配置开关和Checkpoint逻辑干扰V4.1。

这些不是已经证实存在于你当前源码中的Bug，而是从V3迁移到V4.1时必须避免的具体风险。

因此，网络结构应以原UAD为准重新接入，而不是在V3的频率计算路径上做增量叠加。

## 三、前面的修改指导是否完整？

方法设计、数学定义、网络位置、参数数量、初始化、优化器及训练协议已经比较完整，可以作为主要实施依据。

但如果直接交给代码修改人员，我认为还需要补充四项不可省略的工程约束。

### 1. 必须保证仅修改第7个Block

不能通过全局修改Transformer `Block.forward`，使12个Block全部执行SCSR。

应该只在`blocks[6]`内部的`norm1`之后、`attn`之前执行调制。

同时保留原始残差：

\\[ Y=X_6+\operatorname{Attention}(Z') \\]

不要误写成：

\\[ Y=Z'+\operatorname{Attention}(Z') \\]

这属于必须进行单元测试的计算路径。

### 2. V4.1关闭时必须等价于原UAD

这是防止引入隐蔽代码错误的最高优先级检查。

使用相同权重、输入和模态标签，比较：

| 对比对象               | 要求                     |
| ------------------ | ---------------------- |
| 原UAD vs V4.1-OFF   | 完整前向结果一致               |
| RGB的V4.1-ON vs OFF | 完整前向结果一致               |
| NIR的V4.1-ON vs OFF | 完整前向结果一致               |
| TIR的V4.1-ON vs OFF | 存在实际频率更新               |
| V4.1参数量            | 精确40,032               |
| SCSR全部参数           | 均被Optimizer正确包含，不能重复注册 |


### 3. 不能只检查SCSR的梯度非零

这一项此前已经造成过严重教训。

必须在真实UAD身份损失及SGD训练配置下检查：

- \\(U\\)是否发生有效更新；
- 动态条件网络的各层参数是否有效更新；
- 门控是否具有输入相关变化；
- 更新是否真正改变Block 7 Attention输出；
- 最终CLS是否产生实际变化；
- AMP是否造成有效更新丢失。

特别是Rank-4低秩门控，必须同时检查 \\(U\\) 和 \\(V\\) 的实际学习状况，不能只检查门控输出。

### 4. 需要明确Checkpoint初始化契约

正式训练必须满足：

原UAD和V4.1的共同网络参数，从相同ImageNet预训练及相同随机初始化条件开始。

不能从训练完成的V3 Checkpoint加载主干后再训练V4.1。

同时，新建SCSR模块时不能意外改变原分类器等随机初始化结果。可以复用V3已有的随机数状态保护机制。

否则，最终性能差异就可能混入主干初始化差异。

## 四、V3有哪些功能值得迁移？

建议只复用这些已通过此前训练或诊断验证的工程功能：

| V3已有功能                | 处理建议           |
| --------------------- | -------------- |
| 真实模态ID的训练/推理传递        | 迁移             |
| PKM三模态Batch顺序检查       | 迁移或保留原实现       |
| DDP实际Optimizer Step统计 | 迁移             |
| AMP跳步统计               | 迁移             |
| 同Checkpoint模块ON/OFF验证 | 迁移并适配新位置       |
| 逐Query检索变化统计          | 迁移             |
| 九模态及跨平台统一评估           | 迁移或保留本地UAD已有实现 |
| Checkpoint配置一致性检查     | 迁移并更新模块版本      |
| V3 Haar与IDWT单元测试      | 可复用测试逻辑        |
| V3低秩投影、条件网络及残差实现      | 不迁移            |
| V3旧插入点及优化器规则          | 不迁移            |

注意，迁移诊断功能时也要修改采集位置：V4.1需要检查的是Block 7归一化Attention输入和Attention输出，不能继续沿用V3的原始Patch残差统计作为主要诊断指标。

## 六、最终建议

选择“原UAD网络底座 + V4.1全新实现 + V3可靠工具复用”，不要选择“V3原网络上持续叠加修改”。

前面给出的指导在算法设计层面已经足够实施，但上述四项工程约束必须一并写入修改要求。没有实际检查最终`.py`源码之前，也不能保证实现不存在遗漏。

还有一个重要原则：

不要把V3所有历史实现都当成需要继承的资产。 V3最值得保留的是经过验证的实验基础设施和失败诊断经验，而不是已经证明几乎不起作用的频率计算结构。

这样既能够最大程度避免历史代码遗留问题，也有利于保证最终论文实验结论的可信度。
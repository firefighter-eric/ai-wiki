---
type: topic
status: formal
review_scope: evidence_synthesis
reviewed: 2026-10-07
---
# 经典 CNN 架构

## TL;DR（快速导读）

经典 CNN 的演进分别改进深度、连接、多分支与效率。VGG、ResNet、DenseNet 和 MobileNet 改的是不同设计维度。

## 先用一个问题理解

若深层网络难训练，残差连接提供直接路径；若想复用特征，DenseNet 采用拼接；若设备预算低，则看轻量卷积。这样按问题阅读，比把模型排成年表更容易理解结构差别。

## 页面状态

- 状态：正式 topic
- 事实基座：`wiki/summaries/` 优先

## 主题定义

本页讨论的不是“所有卷积网络名字大全”，而是**ImageNet 时代到 ViT 前后过渡期中，经典 CNN backbone 如何围绕深度、连接方式、多分支、效率与现代化设计持续重写自己**。这里的边界需要收紧到 backbone 层，而不是把检测头、分割头、训练技巧或所有视觉任务都混进来。也因此，本页重点放在 `VGG / GoogLeNet / ResNet / DenseNet / ResNeXt / MobileNet / ConvNeXt` 这些真正改写设计坐标系的节点上。

这页值得单独成题，是因为这些模型之间并不是简单的“后者更强于前者”。它们各自改写的维度并不相同：`VGG` 强化深度，`GoogLeNet` 重写模块内部计算分配，`ResNet` 解决深层优化，`DenseNet` 重写跨层信息流，`ResNeXt` 引入 `cardinality`，`MobileNet` 把端侧效率变成第一公民，`ConvNeXt` 则在 `ViT` 时代重新论证纯卷积设计空间。把它们放到同一页，不是为了排排行榜，而是为了看清经典 CNN 主线真正演化的是哪些问题。

## 核心问题

- **深度到底何时带来收益，何时反而变成优化障碍。**
- **跨层连接应该解决什么问题**：是让网络更易优化、让特征更可复用，还是让更深结构真正可训练。
- **多分支与规则结构如何取舍**：`Inception` 式手工多分支和 `ResNet / ResNeXt` 式规则重复模块各自解决什么。
- **卷积 backbone 的竞争目标是否只剩纯精度**，还是必须把延迟、参数量、部署场景一起纳入设计目标。
- **ViT 时代卷积还有没有稳定位置**：是被替代，还是通过 `ConvNeXt` 这类现代化设计被重新解释。

## 主线脉络 / 方法分层

如果按论文年份简单排列，这条主线会变成“模型名流水账”。更有解释力的组织方式，是看经典 CNN 每一步到底在改写哪个核心设计轴。

- **深度可扩张化层**：`VGG` 把视觉 backbone 的主设计变量从杂糅组件调整为“能否用高度规则的小卷积堆叠出更深网络”。它建立了一个清晰但昂贵的基线：深度本身确实重要。
- **计算重分配层**：`GoogLeNet`/`Inception` 说明提升能力不一定只能靠单路径变深。通过多分支多尺度处理与 `1x1` 降维，网络可以在受控预算下同时增加宽度、深度和多尺度表达。这里的关键词不是“更深”，而是“更会分配卷积计算”。
- **优化接口重写层**：`ResNet` 是整条主线最重要的转折点。它把深层网络失败解释为优化问题而非表示力问题，并用 residual learning 改写层间接口。自此之后，深度不再只是理论上的潜力，而成为实践中可稳定利用的资源。
- **信息流强化层**：`DenseNet` 延续“跨层连接能改善深层训练”这一判断，但不给残差相加继续加码，而改为把此前所有层的特征显式串接。它强调的是 feature reuse 与信息流最大化，而不是只求更深。
- **容量维度扩展层**：`ResNeXt` 在残差家族已经成熟后，重新提出 `cardinality` 这一维度。它试图兼顾 `Inception` 的多分支表达力与 `ResNet` 的规则可扩展性，表明卷积网络容量不应只靠 depth / width 调节。
- **效率优先分化层**：`MobileNet` 则明确改变优化目标。它不再把 ImageNet 最高精度视为唯一坐标，而是围绕移动端和嵌入式部署，把 depthwise separable convolution 与全局缩放超参数做成一整套效率优先设计。
- **现代化回写层**：`ConvNeXt` 并没有否定 `ViT`，而是在 `ViT` 已经改写视觉架构讨论框架之后，反过来检验经典卷积到底是输在归纳偏置，还是输在设计老化。结论是：在现代训练和模块选择下，纯卷积 backbone 依然具有强竞争力。

把这些层串起来，可以得到一个比“CNN 被 Transformer 淘汰”更稳的判断：**经典 CNN 的长期演化，不是单向追求更深，而是在持续决定该把复杂度放在深度、模块内部结构、跨层连接、容量维度、部署效率还是现代化设计上。**

### 指标进步需要同时记录训练配方

[VGG](../summaries/Simonyan%2C%20Zisserman%20-%202014%20-%20Very%20Deep%20Convolutional%20Networks%20for%20Large-Scale%20Image%20Recognition.md)固定小卷积核研究深度，[GoogLeNet](../summaries/Szegedy%20et%20al.%20-%202014%20-%20Going%20Deeper%20with%20Convolutions.md)通过多分支和瓶颈组织计算，[ResNeXt](../summaries/Xie%20et%20al.%20-%202016%20-%20Aggregated%20Residual%20Transformations%20for%20Deep%20Neural%20Networks.md)强调分组分支的 cardinality。这些不是在完全相同资源下独立替换一个组件的统一消融；输入尺度、数据增强和训练日程也影响成绩。[ConvNeXt](../summaries/Liu%20et%20al.%20-%202022%20-%20A%20ConvNet%20for%20the%202020s.md)重新检验现代配方下卷积性能，支持不能把老训练配置当成架构能力上限。本文比较设计轴，避免以跨年代最佳数字作因果结论。

## 关键争论与分歧

- **卷积主线的核心突破究竟是深度还是连接**：现有证据更支持 `ResNet` 的接口重写比单纯继续加深更关键。`VGG` 证明深度重要，但 `ResNet` 才让深度真正成为稳定可训练资源。
- **多分支是否优于规则结构**：`GoogLeNet` 证明多分支在受控计算下很强，但其设计更手工化。`ResNeXt` 的意义恰恰在于承认多分支有价值，同时试图把它收敛到更统一的模板里。争论的本质不是谁“更先进”，而是工程可扩展性与模块表达力如何权衡。
- **Dense connectivity 是不是比 residual 更强**：当前证据只支持它代表不同取舍，而不支持对所有场景的绝对优势。`DenseNet` 在参数效率和 feature reuse 上有优势，但系统复杂度和部署友好性并不自动占优。
- **轻量化 CNN 是否只是精度退而求其次**：`MobileNet` 更准确的地位是重新定义目标函数。它把资源预算本身引入 backbone 设计，而不是在高算力精度竞赛中的次优残余。
- **ConvNeXt 是否证明 CNN 重新统治视觉**：不能这样外推。更稳的说法是，`ConvNeXt` 证明许多被归因为 Transformer 的优势，其实部分来自现代化设计空间；但这不等于视觉已重新收敛到纯 CNN。

## 证据基础

- [Simonyan, Zisserman - 2014 - Very Deep Convolutional Networks for Large-Scale Image Recognition](../../wiki/summaries/Simonyan,%20Zisserman%20-%202014%20-%20Very%20Deep%20Convolutional%20Networks%20for%20Large-Scale%20Image%20Recognition.md)
- [Szegedy et al. - 2014 - Going Deeper with Convolutions](../../wiki/summaries/Szegedy%20et%20al.%20-%202014%20-%20Going%20Deeper%20with%20Convolutions.md)
- [He et al. - 2015 - Deep Residual Learning for Image Recognition](../../wiki/summaries/He%20et%20al.%20-%202015%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition.md)
- [Huang et al. - 2016 - Densely Connected Convolutional Networks](../../wiki/summaries/Huang%20et%20al.%20-%202016%20-%20Densely%20Connected%20Convolutional%20Networks.md)
- [Xie et al. - 2016 - Aggregated Residual Transformations for Deep Neural Networks](../../wiki/summaries/Xie%20et%20al.%20-%202016%20-%20Aggregated%20Residual%20Transformations%20for%20Deep%20Neural%20Networks.md)
- [Howard et al. - 2017 - MobileNets Efficient Convolutional Neural Networks for Mobile Vision Applications](../../wiki/summaries/Howard%20et%20al.%20-%202017%20-%20MobileNets%20Efficient%20Convolutional%20Neural%20Networks%20for%20Mobile%20Vision%20Applications.md)
- [Liu et al. - 2022 - A ConvNet for the 2020s](../../wiki/summaries/Liu%20et%20al.%20-%202022%20-%20A%20ConvNet%20for%20the%202020s.md)

## 代表页面

- [VGG](../concepts/VGG.md)
- [GoogLeNet](../concepts/GoogLeNet.md)
- [ResNet](../concepts/ResNet.md)
- [DenseNet](../concepts/DenseNet.md)
- [ResNeXt](../concepts/ResNeXt.md)
- [MobileNet](../concepts/MobileNet.md)
- [ConvNeXt](../concepts/ConvNeXt.md)
- [ViT](../concepts/ViT.md)

## 未解决问题

- 固定数据、日程与算力时，深度、宽度、分支与连接各贡献多少？跨年代配方限制架构归因。
- 分类结构何时能迁移到检测、分割与低数据任务？不能只按 ImageNet 排名选择全部任务。
- 计算量怎样对应实际延迟和量化表现？小卷积、可分离卷积与多分支的效率依后端，FLOPs 不替代测试。

## 关联页面

- [传统 CV](./传统%20CV.md)
- [目标检测](./目标检测.md)
- [ViT](../concepts/ViT.md)
- [VGG](../concepts/VGG.md)
- [GoogLeNet](../concepts/GoogLeNet.md)
- [ResNet](../concepts/ResNet.md)
- [DenseNet](../concepts/DenseNet.md)
- [ResNeXt](../concepts/ResNeXt.md)
- [MobileNet](../concepts/MobileNet.md)
- [ConvNeXt](../concepts/ConvNeXt.md)

## 这里的术语是什么意思

- **backbone**：模型骨干：主要负责提取或变换表示，其他任务模块在它之上工作。

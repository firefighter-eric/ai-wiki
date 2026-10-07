---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Liu et al. - 2022 - A ConvNet for the 2020s

## TL;DR（快速导读）

ConvNeXt 逐步调整 ResNet 的结构与训练方式，研究纯卷积模型在现代视觉任务中仍能达到什么水平。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Liu et al. - 2022 - A ConvNet for the 2020s.pdf
- 原始 HTML：../../raw/html/Liu et al. - 2022 - A ConvNet for the 2020s.html
- 全文文本：../../raw/text/Liu et al. - 2022 - A ConvNet for the 2020s.md
- 作者：Liu et al.
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

论文把卷积网络的模块选择和训练配方向现代设计更新，并用对照分析各项改变。它帮助判断收益来自架构还是训练条件；公平比较需要匹配数据、算力和训练设置。

## 关键事实

- **C1**：ConvNeXt从ResNet出发，按现代训练、macro设计、depthwise、inverted bottleneck和大kernel逐步改造。
- **C2**：stem改为4×4 stride4 patchify，空间/通道混合分离。
- **C3**：T/S/B/L等与Swin相近复杂度比较，ImageNet1K训练300epochs/AdamW。
- **C4**：附录承认一些多模态等任务Transformer可能更灵活。

## 争议与不确定点

- 不同改造顺序和配方会改变消融收益。
- 分类/检测分数并不能覆盖所有多模态应用。

## 关联页面

- 主题：[经典 CNN 架构](../../wiki/topics/经典%20CNN%20架构.md)
- 主题：[传统 CV](../../wiki/topics/传统%20CV.md)
- 概念：[ConvNeXt](../../wiki/concepts/ConvNeXt.md)
- 概念：[ResNet](../../wiki/concepts/ResNet.md)
- 概念：[ViT](../../wiki/concepts/ViT.md)

## 这里的术语是什么意思

- **backbone**：模型骨干：主要负责提取或变换表示，其他任务模块在它之上工作。

## 方法与实验解读

ConvNeXt检验架构差异能否在训练配方相近时解释性能。大kernel扩大局部卷积感受野，depthwise和pointwise把空间与通道变换分开。结果支持现代化CNN仍有竞争力，而不是证明attention没有价值；公平比较必须对齐训练时长、数据和复杂度。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Liu%20et%20al.%20-%202022%20-%20A%20ConvNet%20for%20the%202020s.md#source-section-4 ) | recipe与结构共同对照。 |
| C2 | [原文]( ../../raw/text/Liu%20et%20al.%20-%202022%20-%20A%20ConvNet%20for%20the%202020s.md#source-section-8 ) | 保留纯ConvNet，不使用attention替代卷积。 |
| C3 | [原文]( ../../raw/text/Liu%20et%20al.%20-%202022%20-%20A%20ConvNet%20for%20the%202020s.md#source-section-23 ) | 数据量与训练配方是性能条件。 |
| C4 | [原文]( ../../raw/text/Liu%20et%20al.%20-%202022%20-%20A%20ConvNet%20for%20the%202020s.md#source-section-48 ) | 未证明CNN在所有视觉/跨模态场景占优。 |

## 核证范围

核读modernization路线、patchify/depthwise、训练设置与Limitations。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

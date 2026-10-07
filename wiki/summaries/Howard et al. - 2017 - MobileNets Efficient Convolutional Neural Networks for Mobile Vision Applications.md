---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Howard et al. - 2017 - MobileNets Efficient Convolutional Neural Networks for Mobile Vision Applications

## TL;DR（快速导读）

MobileNet 用深度可分离卷积减少计算，并提供可调节的模型宽度与输入分辨率，适合研究移动端视觉成本。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Howard et al. - 2017 - MobileNets Efficient Convolutional Neural Networks for Mobile Vision Applications.pdf
- 原始 HTML：../../raw/html/Howard et al. - 2017 - MobileNets Efficient Convolutional Neural Networks for Mobile Vision Applications.html
- 全文文本：../../raw/text/Howard et al. - 2017 - MobileNets Efficient Convolutional Neural Networks for Mobile Vision Applications.md
- 作者：Howard et al.
- 年份：2017
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

它将空间卷积与通道组合拆开，再通过缩放参数调整精度、模型大小和延迟。浮点运算量降低只是成本的一部分，实际设备速度还取决于算子实现、内存访问和输入配置。

## 关键事实

- **C1**：depthwise separable convolution 把逐通道空间滤波与1×1 pointwise通道混合分开。
- **C2**：width multiplier 控制通道数，resolution multiplier 控制输入及特征图尺寸。
- **C3**：ImageNet之外，报告检测、细粒度分类、人脸属性和地理定位任务。

## 争议与不确定点

- 模型MAC下降不会自动按同比例降低任意硬件的耗时。
- 降低分辨率会损失小物体等信息，不能只由总参数评估效果。

## 关联页面

- 主题：[经典 CNN 架构](../../wiki/topics/经典%20CNN%20架构.md)
- 主题：[传统 CV](../../wiki/topics/传统%20CV.md)
- 概念：[MobileNet](../../wiki/concepts/MobileNet.md)

## 这里的术语是什么意思

- **backbone**：模型骨干：主要负责提取或变换表示，其他任务模块在它之上工作。

## 方法与实验解读

MobileNet在算子级减少冗余，并用宽度/分辨率把一条网络变成资源可选家族。节省的主要是卷积乘加；真实延迟还受内存访问、算子实现和设备支持影响。因此选型先看目标硬件的实际 latency，再看对应精度，而不只看理论压缩比。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Howard%20et%20al.%20-%202017%20-%20MobileNets%20Efficient%20Convolutional%20Neural%20Networks%20for%20Mobile%20Vision%20Applications.md#source-section-6 ) | depthwise单独不能组合通道信息。 |
| C2 | [原文]( ../../raw/text/Howard%20et%20al.%20-%202017%20-%20MobileNets%20Efficient%20Convolutional%20Neural%20Networks%20for%20Mobile%20Vision%20Applications.md#source-section-5 ) | 两个资源轴影响计算与准确率。 |
| C3 | [原文]( ../../raw/text/Howard%20et%20al.%20-%202017%20-%20MobileNets%20Efficient%20Convolutional%20Neural%20Networks%20for%20Mobile%20Vision%20Applications.md#source-section-2 ) | 主张任务可迁移，不保证无需任务微调。 |

## 核证范围

核读结构、两个multiplier及实验任务范围。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

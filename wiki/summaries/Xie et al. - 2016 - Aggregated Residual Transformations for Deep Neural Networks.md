---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Xie et al. - 2016 - Aggregated Residual Transformations for Deep Neural Networks

## TL;DR（快速导读）

ResNeXt 将统一的多分支变换放进残差块，研究分支数量怎样成为深度和宽度之外的容量选择。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Xie et al. - 2016 - Aggregated Residual Transformations for Deep Neural Networks.pdf
- 原始 HTML：../../raw/html/Xie et al. - 2016 - Aggregated Residual Transformations for Deep Neural Networks.html
- 全文文本：../../raw/text/Xie et al. - 2016 - Aggregated Residual Transformations for Deep Neural Networks.md
- 作者：Xie et al.
- 年份：2016
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

论文用 cardinality 表示并行变换的数量，在规则模块中聚合结果。比较时需控制总计算和参数预算，才能看出收益来自分支组织还是更多资源。

## 关键事实

- **C1**：ResNeXt将同拓扑变换聚合，cardinality为分支数。
- **C2**：固定复杂度比较时调整bottleneck宽度隔离cardinality变量。
- **C3**：ImageNet/COCO实验比较对应ResNet，而非所有结构统一冠军。

## 争议与不确定点

- ILSVRC提交的ensemble和单模型不混。
- classification改善不自动保证所有下游检测/分割提升。

## 关联页面

- 主题：[经典 CNN 架构](../../wiki/topics/经典%20CNN%20架构.md)
- 主题：[传统 CV](../../wiki/topics/传统%20CV.md)
- 概念：[ResNeXt](../../wiki/concepts/ResNeXt.md)
- 概念：[ResNet](../../wiki/concepts/ResNet.md)
- 概念：[GoogLeNet](../../wiki/concepts/GoogLeNet.md)

## 这里的术语是什么意思

- **backbone**：模型骨干：主要负责提取或变换表示，其他任务模块在它之上工作。

## 方法与实验解读

把分支设计统一为重复模块，让内部并行容量成为可调维度。groupconv是实现等价形式之一；设备上分组kernel可能影响效率，需将表示收益与实际运行成本分别验证。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Xie%20et%20al.%20-%202016%20-%20Aggregated%20Residual%20Transformations%20for%20Deep%20Neural%20Networks.md#source-section-8 ) | 与depth/width不同。 |
| C2 | [原文]( ../../raw/text/Xie%20et%20al.%20-%202016%20-%20Aggregated%20Residual%20Transformations%20for%20Deep%20Neural%20Networks.md#source-section-9 ) | 相同FLOPs/参数不保证相同latency。 |
| C3 | [原文]( ../../raw/text/Xie%20et%20al.%20-%202016%20-%20Aggregated%20Residual%20Transformations%20for%20Deep%20Neural%20Networks.md#source-section-12 ) | 数据规模与backbone条件限定。 |

## 核证范围

核读聚合变换、cardinality/宽度控制、实现与ImageNet/COCO对照。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

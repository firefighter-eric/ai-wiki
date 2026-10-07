---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Krishnamoorthi - 2018 - Quantizing deep convolutional networks for efficient inference A whitepaper

## TL;DR（快速导读）

这份量化白皮书比较卷积网络的整数推理方案，包括训练后量化和量化感知训练，重点是精度与部署成本的折中。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

量化把浮点数映射到有限整数范围。资料讨论权重与激活的量化粒度，以及怎样校准或训练以降低误差。结论来自特定网络和任务；部署前需要检查数据分布与目标硬件。

## 具体怎么理解

同一通道的数值范围可能与其他通道不同，分别设定尺度能减少小数值被粗糙压缩的误差。

## 关键事实

- **C1**：weight-only 量化主要减少传输与存储，可仍以浮点执行推理。
- **C2**：QAT 在训练中模拟量化影响，和直接对训练后模型量化不同。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Krishnamoorthi%20-%202018%20-%20Quantizing%20deep%20convolutional%20networks%20for%20efficient%20inference%20A%20whitepaper.pdf)
- 全文文本：[打开全文文本](../../raw/text/Krishnamoorthi%20-%202018%20-%20Quantizing%20deep%20convolutional%20networks%20for%20efficient%20inference%20A%20whitepaper.md)
- 作者：Krishnamoorthi
- 年份：2018
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Krishnamoorthi%20-%202018%20-%20Quantizing%20deep%20convolutional%20networks%20for%20efficient%20inference%20A%20whitepaper.html)

## 争议与不确定点

- 以 CNN 为主的结果不能直接作为 Transformer 量化结论。
- batch normalization、范围和离群值会影响精度。

## 关联页面

- 主题：[传统NLP](../topics/传统%20NLP.md)
- 综合：暂无
- [推理优化：量化、缓存与硬件](../comparisons/%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96%EF%BC%9A%E9%87%8F%E5%8C%96%E3%80%81%E7%BC%93%E5%AD%98%E4%B8%8E%E7%A1%AC%E4%BB%B6.md)：把本篇方法放到相关任务与比较条件中阅读。

## 方法与实验解读

该白皮书把权重、激活、通道粒度与训练策略分开比较。对压缩方案的判断应先问是否只压存储，还是确实执行整数运算，再看精度和目标设备延迟。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Krishnamoorthi%20-%202018%20-%20Quantizing%20deep%20convolutional%20networks%20for%20efficient%20inference%20A%20whitepaper.md#source-section-13 ) | 权重位宽与执行位宽分开 |
| C2 | [原文]( ../../raw/text/Krishnamoorthi%20-%202018%20-%20Quantizing%20deep%20convolutional%20networks%20for%20efficient%20inference%20A%20whitepaper.md#source-section-18 ) | 需校准或训练时处理量化误差 |

## 核证范围

核对 §3.1.1 的 weight-only 与 §3.2 的 QAT，参考训练最佳实践边界。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

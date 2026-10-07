---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Kim et al. - 2021 - I-BERT Integer-only BERT Quantization

## TL;DR（快速导读）

I-BERT 研究只用整数运算运行 BERT，不仅量化权重，也处理非线性运算，以适应高效推理硬件。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

常见量化实现仍会在部分操作中回到浮点数。I-BERT 为 Transformer 的相关运算设计整数方案，目标是降低内存与推理成本。实际加速还取决于硬件和算子支持，位宽降低不能直接换算成相同倍数的端到端速度。

## 具体怎么理解

例如矩阵乘法已是整数，但后续归一化又转成浮点，会增加转换成本；整条路径是否支持整数很重要。

## 关键事实

- **C1**：为 GELU、Softmax 与 LayerNorm 设计整数近似，实现整数推理。
- **C2**：实现中 MatMul 为 INT8、累加为 INT32，非线性算子保留 INT32。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Kim%20et%20al.%20-%202021%20-%20I-BERT%20Integer-only%20BERT%20Quantization.pdf)
- 全文文本：[打开全文文本](../../raw/text/Kim%20et%20al.%20-%202021%20-%20I-BERT%20Integer-only%20BERT%20Quantization.md)
- 作者：Kim et al.
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Kim%20et%20al.%20-%202021%20-%20I-BERT%20Integer-only%20BERT%20Quantization.html)

## 争议与不确定点

- GLUE 上精度保持不证明所有模型和任务都可无损量化。
- 速度收益需要整数算子与目标硬件支持。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无
- [推理优化：量化、缓存与硬件](../comparisons/%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96%EF%BC%9A%E9%87%8F%E5%8C%96%E3%80%81%E7%BC%93%E5%AD%98%E4%B8%8E%E7%A1%AC%E4%BB%B6.md)：把本篇方法放到相关任务与比较条件中阅读。

## 方法与实验解读

I-BERT 补的是整数矩阵乘法之外的非线性计算。部署时不仅看权重位宽，还要确认累加、缩放、归一化与后端算子的实现；否则宣称全整数可能仍藏有浮点开销。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Kim%20et%20al.%20-%202021%20-%20I-BERT%20Integer-only%20BERT%20Quantization.md#source-section-7 ) | 训练和推理数值流程不同 |
| C2 | [原文]( ../../raw/text/Kim%20et%20al.%20-%202021%20-%20I-BERT%20Integer-only%20BERT%20Quantization.md#source-section-24 ) | integer-only 不能简化为所有运算都是 INT8 |

## 核证范围

核对 §3.2 的非线性近似、§4.1 精度比较和附录 C.1 的各算子精度。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

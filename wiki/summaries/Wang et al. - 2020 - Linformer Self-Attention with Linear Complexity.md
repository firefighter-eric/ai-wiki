---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Wang et al. - 2020 - Linformer Self-Attention with Linear Complexity

## TL;DR（快速导读）

Linformer 先在序列维压缩键和值，再计算注意力，以低秩近似降低长序列成本。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

长序列先变成较少的表示再参与计算，省下部分成本，也要检查远距离细节是否损失。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Wang et al. - 2020 - Linformer Self-Attention with Linear Complexity.pdf
- 原始 HTML：../../raw/html/Wang et al. - 2020 - Linformer Self-Attention with Linear Complexity.html
- 全文文本：../../raw/text/Wang et al. - 2020 - Linformer Self-Attention with Linear Complexity.md
- 作者：Wang et al.
- 年份：2020
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

压缩将许多位置汇成更少的表示，减少后续矩阵计算。重要细节是否被保留取决于投影、压缩程度和任务；复杂度优势应与质量及硬件速度一起比较。

## 关键事实

- **C1**：将长度方向K/V从n压到k，计算n×kattention而非n×n。
- **C2**：低秩动机来自RoBERTa数据上的谱分析与近似命题。
- **C3**：可跨heads/layers共享投影，节约参数。

## 争议与不确定点

- 近似保持输出的理论条件不等于任意rank k的全任务保证。
- 预训练/下游/速度实验需要分别比较。

## 关联页面

- 主题：[注意力机制 Attention](../../wiki/topics/注意力机制%20Attention.md)
- 概念：[Transformer](../../wiki/concepts/Transformer.md)

## 方法与实验解读

Linformer修改信息访问，让每个query看压缩后的K/V。它与核特征线性attention、稀疏候选和精确IO优化有不同近似误差。学习投影一般关联长度和mask使用，不能不加修改地作为任意因果LLM训练的无损替换。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Wang%20et%20al.%20-%202020%20-%20Linformer%20Self-Attention%20with%20Linear%20Complexity.md#source-section-10 ) | k固定时相对n线性；k并非零成本。 |
| C2 | [原文]( ../../raw/text/Wang%20et%20al.%20-%202020%20-%20Linformer%20Self-Attention%20with%20Linear%20Complexity.md#source-section-7 ) | 不是证明所有输入attention本身严格低秩。 |
| C3 | [原文]( ../../raw/text/Wang%20et%20al.%20-%202020%20-%20Linformer%20Self-Attention%20with%20Linear%20Complexity.md#source-section-13 ) | 投影与长度配置影响泛化。 |

## 核证范围

核读低秩谱/命题、投影模型、共享与实验设计。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

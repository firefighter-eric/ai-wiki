---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Choromanski et al. - 2021 - Rethinking Attention with Performers

## TL;DR（快速导读）

Performer 用随机特征近似标准注意力，希望降低长序列的计算与存储成本；近似误差是比较时的关键。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

近似目标需要同时看误差、训练和实际硬件效率，不能只凭复杂度表达式判断效果。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Choromanski et al. - 2021 - Rethinking Attention with Performers.pdf
- 原始 HTML：../../raw/html/Choromanski et al. - 2021 - Rethinking Attention with Performers.html
- 全文文本：../../raw/text/Choromanski et al. - 2021 - Rethinking Attention with Performers.md
- 作者：Choromanski et al.
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

FAVOR+ 把 softmax 注意力转写为正随机特征下的计算形式。它研究的是数学近似，因此需同时检查特征数量、近似稳定性、任务质量及实际速度，而不能只凭线性复杂度判断收益。

## 关键事实

- **C1**：Performer 用 FAVOR+ 正值正交随机特征估计 softmax attention kernel。
- **C2**：不依赖稀疏或低秩先验，固定特征数时 attention 可用线性时间/空间实现。
- **C3**：报告讨论无偏或近无偏估计、收敛与降低方差，并通过微调迁移兼容 Transformer。
- **C4**：附录指出固定随机特征数下，query/key 范数增大会损害近似质量，难以表示趋于极尖锐的 hard attention。

## 争议与不确定点

- 复杂度优势必须与随机特征数及近似误差一起报告。
- 理论误差界不能代替语言任务、长依赖任务和数值稳定性测试。

## 关联页面

- 主题：[注意力机制 Attention](../../wiki/topics/注意力机制%20Attention.md)
- 概念：[Transformer](../../wiki/concepts/Transformer.md)

## 方法与实验解读

先把 Q/K 转为可分解的随机特征，计算特征维度上的聚合，再恢复近似 attention 输出，因此无需显式保存全体 token 对。正值特征减少负权和数值不稳，正交化降低方差。该路线保留全局交互的近似形式，与 Longformer 的删边和 FlashAttention 的精确重排不同。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Choromanski%20et%20al.%20-%202021%20-%20Rethinking%20Attention%20with%20Performers.md#source-section-4 ) | 随机特征近似，与 exact attention 分开。 |
| C2 | [原文]( ../../raw/text/Choromanski%20et%20al.%20-%202021%20-%20Rethinking%20Attention%20with%20Performers.md#source-section-2 ) | 特征数、head dimension 和误差要求仍影响实际成本。 |
| C3 | [原文]( ../../raw/text/Choromanski%20et%20al.%20-%202021%20-%20Rethinking%20Attention%20with%20Performers.md#source-section-3 ) | 核估计的理论性质不等于最终归一化输出完全无偏。 |
| C4 | [原文]( ../../raw/text/Choromanski%20et%20al.%20-%202021%20-%20Rethinking%20Attention%20with%20Performers.md#source-section-78 ) | 误差条件来自理论讨论，不保证任意训练轨迹稳定。 |

## 核证范围

核读核心 FAVOR+ 定义、理论主张与 F.6 的范数/特征数边界，未逐行审查全部证明。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

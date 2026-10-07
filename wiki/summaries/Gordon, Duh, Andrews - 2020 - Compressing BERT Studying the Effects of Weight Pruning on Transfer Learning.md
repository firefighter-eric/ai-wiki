---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Gordon, Duh, Andrews - 2020 - Compressing BERT Studying the Effects of Weight Pruning on Transfer Learning

## TL;DR（快速导读）

这篇 BERT 剪枝研究发现，压掉权重对预训练和下游迁移的影响会随剪枝程度变化，不能只用模型压缩率判断效果。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

剪枝通过移除部分权重减少模型规模。论文比较不同剪枝程度对预训练损失与任务迁移的影响，并分析可通过后续训练恢复到什么程度。它提示压缩与知识保留存在条件相关的折中。

## 具体怎么理解

删去一小部分权重可能几乎不影响任务，继续大量删除却可能破坏有用表示；两种情况不能按同一规律外推。

## 关键事实

- **C1**：实验考察 BERT 权重剪枝对下游迁移的影响，区分预训练与下游微调后剪枝。
- **C2**：约 30–40% 权重可在所测设置中去除而保持下游准确率，更激进压缩会影响预训练归纳能力。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Gordon%2C%20Duh%2C%20Andrews%20-%202020%20-%20Compressing%20BERT%20Studying%20the%20Effects%20of%20Weight%20Pruning%20on%20Transfer%20Learning.pdf)
- 全文文本：[打开全文文本](../../raw/text/Gordon%2C%20Duh%2C%20Andrews%20-%202020%20-%20Compressing%20BERT%20Studying%20the%20Effects%20of%20Weight%20Pruning%20on%20Transfer%20Learning.md)
- 作者：Gordon, Duh, Andrews
- 年份：2020
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Gordon%2C%20Duh%2C%20Andrews%20-%202020%20-%20Compressing%20BERT%20Studying%20the%20Effects%20of%20Weight%20Pruning%20on%20Transfer%20Learning.html)

## 争议与不确定点

- 权重置零不自动减少稠密矩阵运行成本。
- 所测 GLUE 任务不足以描述所有迁移能力。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无
- [推理优化：量化、缓存与硬件](../comparisons/%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96%EF%BC%9A%E9%87%8F%E5%8C%96%E3%80%81%E7%BC%93%E5%AD%98%E4%B8%8E%E7%A1%AC%E4%BB%B6.md)：把本篇方法放到相关任务与比较条件中阅读。

## 方法与实验解读

剪枝不能只问目标任务能否拟合，也要问模型是否保留迁移所需的归纳偏置。非结构化稀疏权重减少还需要硬件和算子支持，才能变成真正速度收益。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Gordon%2C%20Duh%2C%20Andrews%20-%202020%20-%20Compressing%20BERT%20Studying%20the%20Effects%20of%20Weight%20Pruning%20on%20Transfer%20Learning.md#source-section-10 ) | 剪枝时机是关键变量 |
| C2 | [原文]( ../../raw/text/Gordon%2C%20Duh%2C%20Andrews%20-%202020%20-%20Compressing%20BERT%20Studying%20the%20Effects%20of%20Weight%20Pruning%20on%20Transfer%20Learning.md#source-section-10 ) | 任务均值与训练设置，不是任意压缩率保证 |

## 核证范围

核对 §3.1 的剪枝矩阵、§3.4 的时机比较与 §7 结论。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Smock, Pesala, Abraham - 2023 - Aligning Benchmark Datasets for Table Structure Recognition

## TL;DR（快速导读）

这篇工作对齐表格基准中的错误与不一致标注，说明评测数据的处理方式也会影响模型比较。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

不同数据集可能对相同结构采用不同标注规则。论文研究清理和对齐这些规则后的训练与评估变化。读者应先确认结构定义一致，再比较模型得分，否则差异可能来自标注而非能力。

## 具体怎么理解

同一个合并表头，一套数据记作一格，另一套拆成多格；未经对齐，正确输出也可能被判错。

## 关键事实

- **C1**：固定 Table Transformer 架构，调整 FinTabNet、PubTables-1M 与 ICDAR 标注，以研究数据一致性。
- **C2**：作者未直接测量自动对齐过程准确率，并承认对齐可能引入新错误。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Smock%2C%20Pesala%2C%20Abraham%20-%202023%20-%20Aligning%20Benchmark%20Datasets%20for%20Table%20Structure%20Recognition.pdf)
- 全文文本：[打开全文文本](../../raw/text/Smock%2C%20Pesala%2C%20Abraham%20-%202023%20-%20Aligning%20Benchmark%20Datasets%20for%20Table%20Structure%20Recognition.md)
- 作者：Smock, Pesala, Abraham
- 年份：2023
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Smock%2C%20Pesala%2C%20Abraham%20-%202023%20-%20Aligning%20Benchmark%20Datasets%20for%20Table%20Structure%20Recognition.html)

## 争议与不确定点

- 过滤样本会改变数据分布。
- 所测数据集的对齐规则不能无条件迁移到全部表格。

## 关联页面

- 主题：[LLM RL](../topics/LLM%20RL.md)
- 综合：暂无
- [Robin Abraham](../authors/Robin%20Abraham.md)：沿作者或机构继续阅读相关来源。
- [Rohith Pesala](../authors/Rohith%20Pesala.md)：沿作者或机构继续阅读相关来源。
- [Brandon Smock](../authors/Brandon%20Smock.md)：沿作者或机构继续阅读相关来源。

## 方法与实验解读

跨数据集表现下降可能来自标签定义不同，而非模型不会读表。论文用固定模型测试标注对齐，说明评测前要先确认合并单元格、表头与内容的定义一致。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Smock%2C%20Pesala%2C%20Abraham%20-%202023%20-%20Aligning%20Benchmark%20Datasets%20for%20Table%20Structure%20Recognition.md#source-section-3 ) | 数据改进与新模型架构改进分开 |
| C2 | [原文]( ../../raw/text/Smock%2C%20Pesala%2C%20Abraham%20-%202023%20-%20Aligning%20Benchmark%20Datasets%20for%20Table%20Structure%20Recognition.md#source-section-12 ) | 质量过滤不能证明标注绝对正确 |

## 核证范围

核对研究设计、§4 的固定模型实验和 §5 的对齐局限。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

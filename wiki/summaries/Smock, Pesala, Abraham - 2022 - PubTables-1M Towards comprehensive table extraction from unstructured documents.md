---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Smock, Pesala, Abraham - 2022 - PubTables-1M Towards comprehensive table extraction from unstructured documents

## TL;DR（快速导读）

PubTables-1M 提供大规模表格抽取标注，重点是完整、清楚的单元格结构，支撑检测和表格恢复研究。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

模型需要一致的真实标注才能学到表格结构。数据集覆盖大量学术文章表格，并整理结构与相关信息。数据规模不代替类型覆盖；迁到票据或复杂报告时仍需测试。

## 具体怎么理解

例如一个表头跨两列，标注要明确跨度；如果只记文字框，就难以学习最终电子表格结构。

## 关键事实

- **C1**：PubTables-1M 区分表格检测、结构识别与功能分析，并为这些任务提供标注。
- **C2**：canonicalization 合并特定条件下的相邻单元格，解决原始结构标注过度分割导致的歧义。
- **C3**：规范化算法不保证零错误，其他数据集可能需要额外假设。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Smock%2C%20Pesala%2C%20Abraham%20-%202022%20-%20PubTables-1M%20Towards%20comprehensive%20table%20extraction%20from%20unstructured%20documents.pdf)
- 全文文本：[打开全文文本](../../raw/text/Smock%2C%20Pesala%2C%20Abraham%20-%202022%20-%20PubTables-1M%20Towards%20comprehensive%20table%20extraction%20from%20unstructured%20documents.md)
- 作者：Smock, Pesala, Abraham
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Smock%2C%20Pesala%2C%20Abraham%20-%202022%20-%20PubTables-1M%20Towards%20comprehensive%20table%20extraction%20from%20unstructured%20documents.html)

## 争议与不确定点

- 学术文档表格的分布与发票、网页或电子表格不同。
- 结构正确仍不保证文字识别和数值关系正确。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无
- [文档与表格的输入输出接口](../comparisons/%E6%96%87%E6%A1%A3%E4%B8%8E%E8%A1%A8%E6%A0%BC%E7%9A%84%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%E6%8E%A5%E5%8F%A3.md)：把本篇方法放到相关任务与比较条件中阅读。
- [Robin Abraham](../authors/Robin%20Abraham.md)：沿作者或机构继续阅读相关来源。
- [Rohith Pesala](../authors/Rohith%20Pesala.md)：沿作者或机构继续阅读相关来源。
- [Brandon Smock](../authors/Brandon%20Smock.md)：沿作者或机构继续阅读相关来源。

## 方法与实验解读

论文重点不只是训练更好的检测器，还在于把表格真值定义得一致。表头与跨格的歧义会让同一视觉结构出现多个标注答案；规范化之后，模型与指标才能比较同一个目标。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Smock%2C%20Pesala%2C%20Abraham%20-%202022%20-%20PubTables-1M%20Towards%20comprehensive%20table%20extraction%20from%20unstructured%20documents.md#source-section-14 ) | 检测到表格不等于恢复单元格内容 |
| C2 | [原文]( ../../raw/text/Smock%2C%20Pesala%2C%20Abraham%20-%202022%20-%20PubTables-1M%20Towards%20comprehensive%20table%20extraction%20from%20unstructured%20documents.md#source-section-10 ) | 基于 PMCOA 标注假设 |
| C3 | [原文]( ../../raw/text/Smock%2C%20Pesala%2C%20Abraham%20-%202022%20-%20PubTables-1M%20Towards%20comprehensive%20table%20extraction%20from%20unstructured%20documents.md#source-section-11 ) | 不能把同一规则无条件应用于全部表格 |

## 核证范围

核对 §3 的规范化和局限、§4 的三类任务以及 §5 的规范化对比。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

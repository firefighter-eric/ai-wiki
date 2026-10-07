---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Smock, Pesala, Abraham - 2022 - GriTS Grid table similarity metric for table structure recognition

## TL;DR（快速导读）

GriTS 直接以表格网格比较预测与真实结构，研究比单纯比较标记字符串更贴近表格形态的指标。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

同一表格可以有不同字符串表示，字符串差异也可能掩盖结构差异。GriTS 从矩阵和共同子结构出发定义相似度。阅读时应分清指标比较的是结构、位置还是内容，并检查近似计算方法。

## 具体怎么理解

一张表把两行错并为一行，文字可能几乎一样，但网格关系明显不同，指标应反映这个错误。

## 关键事实

- **C1**：把表格作为二维网格，允许单元格部分相似，而非只比较序列化 HTML。
- **C2**：Factored 2D-MSS 是启发式近似，并给出真相似度的上下界。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Smock%2C%20Pesala%2C%20Abraham%20-%202022%20-%20GriTS%20Grid%20table%20similarity%20metric%20for%20table%20structure%20recognition.pdf)
- 全文文本：[打开全文文本](../../raw/text/Smock%2C%20Pesala%2C%20Abraham%20-%202022%20-%20GriTS%20Grid%20table%20similarity%20metric%20for%20table%20structure%20recognition.md)
- 作者：Smock, Pesala, Abraham
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Smock%2C%20Pesala%2C%20Abraham%20-%202022%20-%20GriTS%20Grid%20table%20similarity%20metric%20for%20table%20structure%20recognition.html)

## 争议与不确定点

- 实验中的近似误差小不保证所有表格都为零。
- 相似度高仍可能包含关键数值错误，需结合业务内容核查。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无
- [Robin Abraham](../authors/Robin%20Abraham.md)：沿作者或机构继续阅读相关来源。
- [Rohith Pesala](../authors/Rohith%20Pesala.md)：沿作者或机构继续阅读相关来源。
- [Brandon Smock](../authors/Brandon%20Smock.md)：沿作者或机构继续阅读相关来源。

## 方法与实验解读

GriTS 保留行列的二维关系，便于区分哪里出现结构或内容错误。它减少序列化方式对评测的影响，但必须说明使用拓扑、位置还是内容版本；内容分数也会受到 OCR 质量影响。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Smock%2C%20Pesala%2C%20Abraham%20-%202022%20-%20GriTS%20Grid%20table%20similarity%20metric%20for%20table%20structure%20recognition.md#source-section-12 ) | 可分别衡量拓扑、位置与内容 |
| C2 | [原文]( ../../raw/text/Smock%2C%20Pesala%2C%20Abraham%20-%202022%20-%20GriTS%20Grid%20table%20similarity%20metric%20for%20table%20structure%20recognition.md#source-section-14 ) | 不能把近似算法说成精确求解 NP-hard 问题 |

## 核证范围

核对 §2 的属性区分、§3.2–3.4 的相似度与近似算法以及 §4 的误差评测。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

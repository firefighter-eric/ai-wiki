---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Xing et al. - 2023 - LORE Logical Location Regression Network for Table Structure Recognition

## TL;DR（快速导读）

LORE 直接预测单元格的逻辑行列位置，尝试用结构回归恢复表格，减少额外规则和冗长序列解码。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

表格结构识别不仅要检测框，还要知道单元格在逻辑网格里的位置。LORE 围绕逻辑位置组织预测。应检验跨行跨列、无边框和复杂布局，并与预训练版本核对差异。

## 具体怎么理解

“总计”可能跨整行；视觉框较宽并不能单独说明它跨了哪些列，需要逻辑位置表示。

## 关键事实

- **C1**：CNN 提取单元格视觉特征，再用回归头预测空间框与逻辑坐标。
- **C2**：通过级联与单元格内外约束学习结构依赖，区别于只预测邻接或生成序列。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Xing%20et%20al.%20-%202023%20-%20LORE%20Logical%20Location%20Regression%20Network%20for%20Table%20Structure%20Recognition.pdf)
- 全文文本：[打开全文文本](../../raw/text/Xing%20et%20al.%20-%202023%20-%20LORE%20Logical%20Location%20Regression%20Network%20for%20Table%20Structure%20Recognition.md)
- 作者：Xing et al.
- 年份：2023
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Xing%20et%20al.%20-%202023%20-%20LORE%20Logical%20Location%20Regression%20Network%20for%20Table%20Structure%20Recognition.html)

## 争议与不确定点

- 跨格和复杂布局仍需要适配不同标注约定。
- 结构输出不含自动事实核查。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无
- [文档与表格的输入输出接口](../comparisons/%E6%96%87%E6%A1%A3%E4%B8%8E%E8%A1%A8%E6%A0%BC%E7%9A%84%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%E6%8E%A5%E5%8F%A3.md)：把本篇方法放到相关任务与比较条件中阅读。

## 方法与实验解读

LORE 直接学习格子占哪些行列，让表格结构恢复围绕逻辑坐标展开。与后续扩展比较时，要区分算法变化、预训练与数据，不把两版本当成完全相同模型。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Xing%20et%20al.%20-%202023%20-%20LORE%20Logical%20Location%20Regression%20Network%20for%20Table%20Structure%20Recognition.md#source-section-9 ) | 逻辑行列位置不是像素坐标 |
| C2 | [原文]( ../../raw/text/Xing%20et%20al.%20-%202023%20-%20LORE%20Logical%20Location%20Regression%20Network%20for%20Table%20Structure%20Recognition.md#source-section-26 ) | 这是原 LORE，后续 LORE++ 另加入预训练 |

## 核证范围

核对 Methodology、数据范围与结论，和 LORE++ 的新增预训练作版本区分。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

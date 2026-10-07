---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Liao et al. - 2023 - DocTr Document Transformer for Structured Information Extraction in Documents

## TL;DR（快速导读）

DocTr 把文档中的实体表示成锚点词与位置框，再建模实体关系，探索结构化信息抽取的新接口。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

传统标注序列依赖文字顺序，图结构解码又可能复杂。DocTr 借鉴定位式表示，把实体与页面位置相连。任务结果不仅需要识别文字，还要组织字段和关系，适合关注表单、票据等文档抽取。

## 具体怎么理解

在发票里找出金额并将其与对应条目连接，比只读出全部文字多了一层结构判断。

## 关键事实

- **C1**：将实体表示为 anchor word 与框，将实体连接作为独立关联预测。
- **C2**：语言编码器读取 OCR 文字与框，视觉编码器读取图像，解码器联合两者。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Liao%20et%20al.%20-%202023%20-%20DocTr%20Document%20Transformer%20for%20Structured%20Information%20Extraction%20in%20Documents.pdf)
- 全文文本：[打开全文文本](../../raw/text/Liao%20et%20al.%20-%202023%20-%20DocTr%20Document%20Transformer%20for%20Structured%20Information%20Extraction%20in%20Documents.md)
- 作者：Liao et al.
- 年份：2023
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Liao%20et%20al.%20-%202023%20-%20DocTr%20Document%20Transformer%20for%20Structured%20Information%20Extraction%20in%20Documents.html)

## 争议与不确定点

- 收据解析的 schema 与一般文档知识关系不同。
- 检测框正确不能保证实体文字和关系完整。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无
- [文档与表格的输入输出接口](../comparisons/%E6%96%87%E6%A1%A3%E4%B8%8E%E8%A1%A8%E6%A0%BC%E7%9A%84%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%E6%8E%A5%E5%8F%A3.md)：把本篇方法放到相关任务与比较条件中阅读。

## 方法与实验解读

DocTr 把文档结构化抽取写成类似检测的任务，使实体边界与关系连接不完全依赖文字排列。输入 OCR 错误仍会传到下游，评测应分开看实体标注与连接正确性。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Liao%20et%20al.%20-%202023%20-%20DocTr%20Document%20Transformer%20for%20Structured%20Information%20Extraction%20in%20Documents.md#source-section-2 ) | 与 IOB 序列标注的对象定义不同 |
| C2 | [原文]( ../../raw/text/Liao%20et%20al.%20-%202023%20-%20DocTr%20Document%20Transformer%20for%20Structured%20Information%20Extraction%20in%20Documents.md#source-section-10 ) | 仍依赖 OCR 输入，不是 OCR-free 模型 |

## 核证范围

核对问题定义、§3.2 的视觉语言流程及实验的三类任务。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

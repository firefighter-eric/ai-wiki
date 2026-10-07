---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Lysak et al. - 2023 - Optimized Table Tokenization for Table Structure Recognition

## TL;DR（快速导读）

这篇研究优化表格结构的词元表示，关注同一张表怎样编码成更适合模型生成的序列。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

图像到标记序列的表格识别，需要选择 HTML 等结构的编码方式。表示长度和组合方式会影响学习与解码。论文把表示本身作为优化对象，提醒结构识别不仅取决于视觉模型。

## 具体怎么理解

同一个合并单元格可以用较冗长或较紧凑的标记表达；序列长度变化会影响生成难度。

## 关键事实

- **C1**：OTSL 用五种 token 描述二维表格网格，减少结构序列长度。
- **C2**：可以检测非法中间序列，但合法序列不保证结构预测正确。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Lysak%20et%20al.%20-%202023%20-%20Optimized%20Table%20Tokenization%20for%20Table%20Structure%20Recognition.pdf)
- 全文文本：[打开全文文本](../../raw/text/Lysak%20et%20al.%20-%202023%20-%20Optimized%20Table%20Tokenization%20for%20Table%20Structure%20Recognition.md)
- 作者：Lysak et al.
- 年份：2023
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Lysak%20et%20al.%20-%202023%20-%20Optimized%20Table%20Tokenization%20for%20Table%20Structure%20Recognition.html)

## 争议与不确定点

- 收益依赖 TableFormer 等模型及数据集设置。
- 单元格内容准确性仍依赖文字识别。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无
- [文档与表格的输入输出接口](../comparisons/%E6%96%87%E6%A1%A3%E4%B8%8E%E8%A1%A8%E6%A0%BC%E7%9A%84%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%E6%8E%A5%E5%8F%A3.md)：把本篇方法放到相关任务与比较条件中阅读。

## 方法与实验解读

OTSL 改进的是模型要生成什么表示。更短的结构序列降低解码步骤，也便于局部规则检查；错误修复规则只能提高成功机会，不能把所有合法输出都视为真值。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Lysak%20et%20al.%20-%202023%20-%20Optimized%20Table%20Tokenization%20for%20Table%20Structure%20Recognition.md#source-section-8 ) | 结构语言不包含完整事实判断 |
| C2 | [原文]( ../../raw/text/Lysak%20et%20al.%20-%202023%20-%20Optimized%20Table%20Tokenization%20for%20Table%20Structure%20Recognition.md#source-section-10 ) | 语法有效与内容正确分开 |

## 核证范围

核对 §4.1–4.3 的语言和验证规则、§5 的 TableFormer 对照。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

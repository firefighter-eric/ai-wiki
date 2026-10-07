---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Tian et al. - 2024 - SpreadsheetLLM Encoding Spreadsheets for Large Language Models

## TL;DR（快速导读）

SpreadsheetLLM 研究怎样压缩并编码电子表格，保留单元格地址、布局和格式，让 LLM 更有效地处理二维信息。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

电子表格可能很大，直接逐格写进提示既昂贵又容易丢结构。论文设计序列化和编码方法，使模型接收更紧凑的表格表示。压缩时应检查信息是否仍足以支持问答和计算。

## 具体怎么理解

“A3 的值”与“第 3 行总计”需要知道地址和结构；只保留一串数值很难区分。

## 关键事实

- **C1**：SheetCompressor 组合结构锚点、倒排表达和数据格式聚合，以压缩表格输入。
- **C2**：原方案未充分利用背景颜色和边框等线索，作者指出 token 成本限制。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Tian%20et%20al.%20-%202024%20-%20SpreadsheetLLM%20Encoding%20Spreadsheets%20for%20Large%20Language%20Models.pdf)
- 全文文本：[打开全文文本](../../raw/text/Tian%20et%20al.%20-%202024%20-%20SpreadsheetLLM%20Encoding%20Spreadsheets%20for%20Large%20Language%20Models.md)
- 作者：Tian et al.
- 年份：2024
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Tian%20et%20al.%20-%202024%20-%20SpreadsheetLLM%20Encoding%20Spreadsheets%20for%20Large%20Language%20Models.html)

## 争议与不确定点

- 完整压缩与不聚合版本的质量和压缩率有取舍。
- F1 的任务范围需说明，不能当成任意电子表格分析准确率。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无
- [文档与表格的输入输出接口](../comparisons/%E6%96%87%E6%A1%A3%E4%B8%8E%E8%A1%A8%E6%A0%BC%E7%9A%84%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%E6%8E%A5%E5%8F%A3.md)：把本篇方法放到相关任务与比较条件中阅读。

## 方法与实验解读

SpreadsheetLLM 让大表格更容易放进语言模型上下文。它利用重复与布局规律压缩，但聚合可能舍掉具体数值；执行分析时仍需访问原始格值，不能用结构摘要替代全部计算输入。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Tian%20et%20al.%20-%202024%20-%20SpreadsheetLLM%20Encoding%20Spreadsheets%20for%20Large%20Language%20Models.md#source-section-9 ) | 压缩表示与完整原文件不同 |
| C2 | [原文]( ../../raw/text/Tian%20et%20al.%20-%202024%20-%20SpreadsheetLLM%20Encoding%20Spreadsheets%20for%20Large%20Language%20Models.md#source-section-36 ) | data-format-aware 不等于保留全部视觉格式 |

## 核证范围

核对 §3 编码与压缩、§5.2.1 版本差别及 Limitations。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

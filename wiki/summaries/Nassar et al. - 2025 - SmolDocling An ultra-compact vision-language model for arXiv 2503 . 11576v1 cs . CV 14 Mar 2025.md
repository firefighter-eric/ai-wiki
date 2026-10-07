---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Nassar et al. - 2025 - SmolDocling An ultra-compact vision-language model for arXiv 2503 . 11576v1 cs . CV 14 Mar 2025

## TL;DR（快速导读）

SmolDocling 用较小的视觉语言模型读取整页文档，输出包含位置和页面元素的 DocTags，研究紧凑的端到端转换。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

传统文档解析常串联多个专门模型。本文用统一标记描述页面中的文本、结构和位置，探索小模型的整体转换能力。需要检查各类页面元素的识别情况，模型小并不意味着所有复杂文档都能正确解析。

## 具体怎么理解

例如一页同时有标题、正文和表格，输出应表达各部分的角色与位置，而不只是把文字拼起来。

## 关键事实

- **C1**：使用紧凑 SmolVLM 底座面向整页文档转换，输出 DocTags。
- **C2**：DocTags 将正文与结构标签分开，并显式表示页面元素和关系。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Nassar%20et%20al.%20-%202025%20-%20SmolDocling%20An%20ultra-compact%20vision-language%20model%20for%20arXiv%202503%20.%2011576v1%20cs%20.%20CV%2014%20Mar%202025.pdf)
- 全文文本：[打开全文文本](../../raw/text/Nassar%20et%20al.%20-%202025%20-%20SmolDocling%20An%20ultra-compact%20vision-language%20model%20for%20arXiv%202503%20.%2011576v1%20cs%20.%20CV%2014%20Mar%202025.md)
- 作者：Nassar et al.
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Nassar%20et%20al.%20-%202025%20-%20SmolDocling%20An%20ultra-compact%20vision-language%20model%20for%20arXiv%202503%20.%2011576v1%20cs%20.%20CV%2014%20Mar%202025.html)

## 争议与不确定点

- 紧凑模型的基准胜出不代表所有领域胜过大模型。
- 生成遗漏与重复应单独监测。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无

## 方法与实验解读

SmolDocling 通过专用格式与训练数据降低文档转换成本。对人阅读时应把 DocTags 转成可读正文，同时保留布局或结构信息；输出合法仍需检查文字、公式和代码缩进。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Nassar%20et%20al.%20-%202025%20-%20SmolDocling%20An%20ultra-compact%20vision-language%20model%20for%20arXiv%202503%20.%2011576v1%20cs%20.%20CV%2014%20Mar%202025.md#source-section-8 ) | 模型小不代表每类文档都完整准确 |
| C2 | [原文]( ../../raw/text/Nassar%20et%20al.%20-%202025%20-%20SmolDocling%20An%20ultra-compact%20vision-language%20model%20for%20arXiv%202503%20.%2011576v1%20cs%20.%20CV%2014%20Mar%202025.md#source-section-9 ) | 需解析后转换到用户阅读格式 |

## 核证范围

核对 §3.1–3.2 的底座与 DocTags、任务数据的代码缩进要求及结果范围。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

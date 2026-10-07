---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Pradhan, Moschitti, Uryupina - 2012 - CoNLL-2012 Shared Task Modeling Multilingual Unrestricted Coreference in OntoNotes

## TL;DR（快速导读）

CoNLL-2012 在 OntoNotes 上评估英文、中文和阿拉伯文共指消解，为跨语言指代研究提供统一任务与数据。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

任务要求将文档中指向同一对象的表达归为一组。数据覆盖较广的指代现象，便于比较系统。阅读时应关注标注范围、语言差异和评测口径，不把数据集成绩直接当成所有文本的实际表现。

## 具体怎么理解

“李明”“他”“这名学生”可能指同一人，系统需要在全文中把这些提及连起来。

## 关键事实

- **C1**：共享任务用 OntoNotes v5.0 评测英语、中文与阿拉伯语共指。
- **C2**：使用 MUC、B-CUBED、CEAF 等不同共指指标，计分对象各有侧重。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Pradhan%2C%20Moschitti%2C%20Uryupina%20-%202012%20-%20CoNLL-2012%20Shared%20Task%20Modeling%20Multilingual%20Unrestricted%20Coreference%20in%20OntoNotes.pdf)
- 全文文本：[打开全文文本](../../raw/text/Pradhan%2C%20Moschitti%2C%20Uryupina%20-%202012%20-%20CoNLL-2012%20Shared%20Task%20Modeling%20Multilingual%20Unrestricted%20Coreference%20in%20OntoNotes.md)
- 作者：Pradhan, Moschitti, Uryupina
- 年份：2012
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 争议与不确定点

- OntoNotes 的标注范围不是现实全部共指现象。
- 语言和文体分布限制外推。

## 关联页面

- 主题：[传统NLP](../topics/传统%20NLP.md)
- 综合：暂无

## 方法与实验解读

共指基准不仅检查两段文字是否同义，还检查哪些提及属于同一实体。gold mentions 与自动检测 mentions 的设定也会改变任务难度；比较后续模型必须使用同一边界与计分协议。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/pdf/Pradhan%2C%20Moschitti%2C%20Uryupina%20-%202012%20-%20CoNLL-2012%20Shared%20Task%20Modeling%20Multilingual%20Unrestricted%20Coreference%20in%20OntoNotes.pdf#page=1 ) | 每种语言的标注和表现需分别解释 |
| C2 | [原文]( ../../raw/pdf/Pradhan%2C%20Moschitti%2C%20Uryupina%20-%202012%20-%20CoNLL-2012%20Shared%20Task%20Modeling%20Multilingual%20Unrestricted%20Coreference%20in%20OntoNotes.pdf#page=19 ) | 关系链接、提及与实体层指标不同 |

## 核证范围

核对 PDF 第 1 页语言与版本、第 19 页评价体系及补充 gold mention 条件。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Wang et al. - 2022 - DAMO-NLP at SemEval-2022 Task 11 A Knowledge-based System for Multilingual Named Entity Recognition

## TL;DR（快速导读）

DAMO-NLP 为短文本实体识别补充 Wikipedia 知识上下文，帮助分辨缺少上下文的多语言实体。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

短句中的名字可能对应多个对象，单靠句子很难判断。方法检索相关知识并提供给实体识别模型。需要检查检索结果是否对应正确实体，否则补充知识也可能带来新的错误。

## 具体怎么理解

“苹果发布了更新”与“苹果价格上涨”里的同一名字可能指不同对象，相关背景能帮助判断。

## 关键事实

- **C1**：利用多语种 Wikipedia 知识库检索上下文，帮助低上下文复杂实体识别。
- **C2**：将检索文本与输入连接后送入 XLM-R，按长度切块；迭代检索也会使用预测实体。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Wang%20et%20al.%20-%202022%20-%20DAMO-NLP%20at%20SemEval-2022%20Task%2011%20A%20Knowledge-based%20System%20for%20Multilingual%20Named%20Entity%20Recognition.pdf)
- 全文文本：[打开全文文本](../../raw/text/Wang%20et%20al.%20-%202022%20-%20DAMO-NLP%20at%20SemEval-2022%20Task%2011%20A%20Knowledge-based%20System%20for%20Multilingual%20Named%20Entity%20Recognition.md)
- 作者：Wang et al.
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Wang%20et%20al.%20-%202022%20-%20DAMO-NLP%20at%20SemEval-2022%20Task%2011%20A%20Knowledge-based%20System%20for%20Multilingual%20Named%20Entity%20Recognition.html)

## 争议与不确定点

- 知识库语言覆盖与实体新鲜度影响效果。
- 共享任务冠军成绩绑定当时数据与规则。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无

## 方法与实验解读

该系统用外部知识补充短句中缺少的实体语境。检索与识别互相帮助，也可能互相放大错误，因此应检查目标实体是否有正确文档支持，而不仅看名字看起来合理。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Wang%20et%20al.%20-%202022%20-%20DAMO-NLP%20at%20SemEval-2022%20Task%2011%20A%20Knowledge-based%20System%20for%20Multilingual%20Named%20Entity%20Recognition.md#source-section-6 ) | 任务是 NER，不是直接生成问答 |
| C2 | [原文]( ../../raw/text/Wang%20et%20al.%20-%202022%20-%20DAMO-NLP%20at%20SemEval-2022%20Task%2011%20A%20Knowledge-based%20System%20for%20Multilingual%20Named%20Entity%20Recognition.md#source-section-11 ) | 错误实体预测可能影响后续检索 |

## 核证范围

核对 §3.1–3.2 的知识检索与 NER、MultiCoNER 任务范围。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

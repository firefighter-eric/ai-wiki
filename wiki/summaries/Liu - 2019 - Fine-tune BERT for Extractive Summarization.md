---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Liu - 2019 - Fine-tune BERT for Extractive Summarization

## TL;DR（快速导读）

BERTSUM 用 BERT 编码文档并选择重要句子生成抽取式摘要，输出主要来自原文，而不是自由改写。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

抽取式摘要要判断哪些句子值得保留。论文改造 BERT 的文档表示，用于句子选择任务。它能够保留原文措辞，但仍可能遗漏上下文、重复内容或打断逻辑，因此选择准确和最终可读性都要检查。

## 具体怎么理解

例如长报道有二十段，模型选出几句核心信息拼成摘要；它不会自动把这些句子改写成新的解释。

## 关键事实

- **C1**：在每句前插入 CLS、句后插入 SEP，并交替使用 segment embedding 获得句级表示。
- **C2**：任务为抽取式摘要：对原句判断是否入选，再用句间模型建模关系。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Liu%20-%202019%20-%20Fine-tune%20BERT%20for%20Extractive%20Summarization.pdf)
- 全文文本：[打开全文文本](../../raw/text/Liu%20-%202019%20-%20Fine-tune%20BERT%20for%20Extractive%20Summarization.md)
- 作者：Liu
- 年份：2019
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Liu%20-%202019%20-%20Fine-tune%20BERT%20for%20Extractive%20Summarization.html)

## 争议与不确定点

- ROUGE 衡量与参考摘要的重叠，不直接保证事实完整或阅读流畅。
- CNN/DailyMail 与 NYT 的新闻结构不能代表论文摘要或知识库综述。

## 关联页面

- 主题：[传统NLP](../topics/传统%20NLP.md)
- 综合：暂无

## 方法与实验解读

BertSum 将 BERT 的编码能力用于选择新闻中的关键句。每句都有表示，句间层再考虑文档结构。它适合保留原句的信息抽取，但复制原句并不自动解决冗余、逻辑衔接或事实背景缺失。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Liu%20-%202019%20-%20Fine-tune%20BERT%20for%20Extractive%20Summarization.md#source-section-6 ) | 区别于仅用文档开头一个 CLS |
| C2 | [原文]( ../../raw/text/Liu%20-%202019%20-%20Fine-tune%20BERT%20for%20Extractive%20Summarization.md#source-section-4 ) | 不能描述成自动改写原句的生成模型 |

## 核证范围

核对 §2 的句表示与抽取任务、§3.2 的数据和 §4 的结果说明。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Xue et al. - 2021 - mT5 A Massively Multilingual Pre-trained Text-to-Text Transformer

## TL;DR（快速导读）

mT5 把 T5 的文本到文本接口扩展到多语言预训练，让同一框架处理不同语言的理解和生成任务。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

模型在多语言语料上训练，并研究迁移任务与生成中的语言控制。语言覆盖广不表示各语言表现相同，低资源语言与意外切换语言的情况值得单独观察。

## 具体怎么理解

要求用中文回答时，内容正确但突然转成其他语言也会影响可用性；多语言评价不能只看整体平均分。

## 关键事实

- **C1**：mT5 将 T5 的 text-to-text 方案扩展到 101 种语言的 mC4 预训练。
- **C2**：生成式抽取问答可能产生不属于上下文的输出，零样本还可能出现意外翻译。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Xue%20et%20al.%20-%202021%20-%20mT5%20A%20Massively%20Multilingual%20Pre-trained%20Text-to-Text%20Transformer.pdf)
- 全文文本：[打开全文文本](../../raw/text/Xue%20et%20al.%20-%202021%20-%20mT5%20A%20Massively%20Multilingual%20Pre-trained%20Text-to-Text%20Transformer.md)
- 作者：Xue et al.
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Xue%20et%20al.%20-%202021%20-%20mT5%20A%20Massively%20Multilingual%20Pre-trained%20Text-to-Text%20Transformer.html)

## 争议与不确定点

- 多语言生成的非法输出会影响抽取式指标。
- 数据语言覆盖与目标域分布不同，不能把模型规模收益视为所有语言一致改善。

## 关联页面

- 主题：[传统NLP](../topics/传统%20NLP.md)
- 综合：暂无

## 方法与实验解读

mT5 提供多语言统一生成接口，但它与直接选取上下文跨度的编码器模型不同：答案可能被翻译、改写或生成到上下文外。评估时必须区分英语零样本、翻译训练与目标语言标注训练。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Xue%20et%20al.%20-%202021%20-%20mT5%20A%20Massively%20Multilingual%20Pre-trained%20Text-to-Text%20Transformer.md#source-section-2 ) | 预训练覆盖数不等于每种语言达到同样能力 |
| C2 | [原文]( ../../raw/text/Xue%20et%20al.%20-%202021%20-%20mT5%20A%20Massively%20Multilingual%20Pre-trained%20Text-to-Text%20Transformer.md#source-section-13 ) | 生成合法性不是编码器式候选跨度的硬约束 |

## 核证范围

核对预训练说明、§4 的多种评测协议与 §5 的非法输出和意外翻译问题。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

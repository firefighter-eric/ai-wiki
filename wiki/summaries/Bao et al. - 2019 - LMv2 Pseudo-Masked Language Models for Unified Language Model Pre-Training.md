---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# UniLMv2：Pseudo-Masked Language Models（2020）

## TL;DR（快速导读）

PMLM 用普通遮挡和伪遮挡结合的预训练方式，让同一语言模型同时学习理解上下文和逐步生成被遮住的内容。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

理解任务需要利用上下文，生成任务还需要处理输出之间的先后依赖。论文在共享模型中安排两类遮挡信号，分别学习可见上下文与被遮片段之间的关系。阅读重点是注意力可见范围和训练目标怎样配合。

## 具体怎么理解

例如补全一个连续短语时，既要看句子前后文，也要处理短语内部词语之间的依赖。

## 关键事实

- **C1**：PMLM 联合 autoencoding 与部分自回归目标，用 pseudo mask 学习遮挡片段内部关系。
- **C2**：通过位置与注意力设计复用上下文编码，实验比较理解和生成下游任务。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Bao%20et%20al.%20-%202019%20-%20LMv2%20Pseudo-Masked%20Language%20Models%20for%20Unified%20Language%20Model%20Pre-Training.pdf)
- 全文文本：[打开全文文本](../../raw/text/Bao%20et%20al.%20-%202019%20-%20LMv2%20Pseudo-Masked%20Language%20Models%20for%20Unified%20Language%20Model%20Pre-Training.md)
- 作者：Hangbo Bao、Li Dong、Furu Wei 等
- 年份：2020
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Bao%20et%20al.%20-%202019%20-%20LMv2%20Pseudo-Masked%20Language%20Models%20for%20Unified%20Language%20Model%20Pre-Training.html)
- 归档说明：保留历史文件名以维持来源对应和链接；标题、作者与年份以上述核对信息为准。

## 争议与不确定点

- 与 BERT-base 等比较绑定训练配方与预算。
- 统一目标不代表无需下游适配。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无

## 方法与实验解读

UniLMv2 希望一个预训练模型兼顾双向理解和条件生成。它不只独立猜被遮挡词，还学习遮挡词之间的依赖；注意力 mask 的设计决定哪些位置能看到哪些内容，避免目标泄漏。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Bao%20et%20al.%20-%202019%20-%20LMv2%20Pseudo-Masked%20Language%20Models%20for%20Unified%20Language%20Model%20Pre-Training.md#source-section-2 ) | 统一理解与生成训练目标，并非一般单向 LM |
| C2 | [原文]( ../../raw/text/Bao%20et%20al.%20-%202019%20-%20LMv2%20Pseudo-Masked%20Language%20Models%20for%20Unified%20Language%20Model%20Pre-Training.md#source-section-16 ) | 计算复用与能力收益分别理解 |

## 核证范围

核对摘要、§3.1 的预训练目标和 §4 的下游实验范围。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

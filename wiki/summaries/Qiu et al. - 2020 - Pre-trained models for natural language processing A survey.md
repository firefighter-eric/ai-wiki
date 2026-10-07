---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Qiu et al. - 2020 - Pre-trained models for natural language processing A survey

## TL;DR（快速导读）

这篇 NLP 预训练综述整理表示方式、训练目标和适配方法，帮助理解不同语言模型为何适合不同任务。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

预训练先从大量文本学习，再迁移到下游任务。综述比较多种路线和应用，并讨论挑战。读者应关注编码与生成接口、监督信号和适配成本，具体效果回到各论文核对。

## 具体怎么理解

做文本分类和做摘要生成都能利用预训练，但任务输出不同，后续适配方式也会不同。

## 关键事实

- **C1**：综述区分静态词表示与上下文化预训练编码，静态向量无法随语境变化处理多义词。
- **C2**：讨论从特征使用到预训练模型适配的变化，早期上下文编码器也可作为冻结特征。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Qiu%20et%20al.%20-%202020%20-%20Pre-trained%20models%20for%20natural%20language%20processing%20A%20survey.pdf)
- 全文文本：[打开全文文本](../../raw/text/Qiu%20et%20al.%20-%202020%20-%20Pre-trained%20models%20for%20natural%20language%20processing%20A%20survey.md)
- 作者：Qiu et al.
- 年份：2020
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Qiu%20et%20al.%20-%202020%20-%20Pre-trained%20models%20for%20natural%20language%20processing%20A%20survey.html)

## 争议与不确定点

- 2020 年综述不覆盖后来完整的指令与偏好训练路线。
- 模型范式介绍不是统一可比的性能实验。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无

## 方法与实验解读

这篇综述提供 NLP 预训练发展的背景。阅读时应先识别表示是否依赖上下文，再识别下游是否更新底座；不要把所有预训练模型统称为现在的聊天 LLM。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Qiu%20et%20al.%20-%202020%20-%20Pre-trained%20models%20for%20natural%20language%20processing%20A%20survey.md#source-section-7 ) | 模型表示层面的限制 |
| C2 | [原文]( ../../raw/text/Qiu%20et%20al.%20-%202020%20-%20Pre-trained%20models%20for%20natural%20language%20processing%20A%20survey.md#source-section-19 ) | 冻结特征、微调和生成式适配应区分 |

## 核证范围

核对词向量局限、第二代上下文编码器与模型分析的范围。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

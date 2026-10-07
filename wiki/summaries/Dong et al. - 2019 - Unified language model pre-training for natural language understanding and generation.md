---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Dong et al. - 2019 - Unified language model pre-training for natural language understanding and generation

## TL;DR（快速导读）

UniLM 用不同注意力遮挡方式，让共享 Transformer 学习单向、双向和序列到序列任务，连接语言理解与生成。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

同一网络可以通过限制词语能看到哪些位置，形成不同的学习任务。UniLM 将这些任务放入预训练，使模型随后可用于问答、摘要等应用。关键是可见范围与目标的安排，而不只是把多个任务名称放在一起。

## 具体怎么理解

做理解时可以读前后文；做生成时，输出位置只能利用允许的历史信息，不能偷看尚未生成的答案。

## 关键事实

- **C1**：共享 Transformer 参数，用不同 self-attention mask 实现双向、单向与 seq2seq 语言建模。
- **C2**：预训练混合多个 LM 目标，下游同时评测理解与生成。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Dong%20et%20al.%20-%202019%20-%20Unified%20language%20model%20pre-training%20for%20natural%20language%20understanding%20and%20generation.pdf)
- 全文文本：[打开全文文本](../../raw/text/Dong%20et%20al.%20-%202019%20-%20Unified%20language%20model%20pre-training%20for%20natural%20language%20understanding%20and%20generation.md)
- 作者：Dong et al.
- 年份：2019
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Dong%20et%20al.%20-%202019%20-%20Unified%20language%20model%20pre-training%20for%20natural%20language%20understanding%20and%20generation.html)

## 争议与不确定点

- 微调任务效果不能直接当成通用零样本能力。
- ROUGE 等指标不保证生成内容事实正确。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无

## 方法与实验解读

UniLM 通过控制 token 之间的可见关系，将理解与生成放入同一网络。看生成结果时要区分条件输入与目标输出的注意力权限；共享权重并不意味着推理时所有 token 都双向可见。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Dong%20et%20al.%20-%202019%20-%20Unified%20language%20model%20pre-training%20for%20natural%20language%20understanding%20and%20generation.md#source-section-6 ) | 可见信息由 mask 决定，不是三个完全独立模型 |
| C2 | [原文]( ../../raw/text/Dong%20et%20al.%20-%202019%20-%20Unified%20language%20model%20pre-training%20for%20natural%20language%20understanding%20and%20generation.md#source-section-12 ) | 目标采样比例与下游微调都是配方的一部分 |

## 核证范围

核对 §2.1–2.4 的输入、骨干与训练目标及 §3 的下游范围。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

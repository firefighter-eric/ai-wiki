---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Xiong et al. - 2021 - Nyströmformer A Nystrom-Based Algorithm for Approximating Self-Attention

## TL;DR（快速导读）

Nyströmformer 用少量代表位置重建注意力的近似关系，减少长序列的计算。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

少数代表位置参与构造近似关系；它们是否覆盖重要信息，会影响最终结果。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Xiong et al. - 2021 - Nyströmformer A Nystrom-Based Algorithm for Approximating Self-Attention.pdf
- 原始 HTML：../../raw/html/Xiong et al. - 2021 - Nyströmformer A Nystrom-Based Algorithm for Approximating Self-Attention.html
- 全文文本：../../raw/text/Xiong et al. - 2021 - Nyströmformer A Nystrom-Based Algorithm for Approximating Self-Attention.md
- 作者：Xiong et al.
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

方法利用代表性行列形成近似，仍试图保留全局信息交互。代表位置的选择、近似误差和任务效果是核心条件；理论复杂度降低不保证所有实现都更快。

## 关键事实

- **C1**：Nyströmformer用Q/Klandmarks近似softmaxattention，不访问完整QKᵀ。
- **C2**：以segmentmeans取landmarks，近似伪逆，并加入V的depthwiseconvskip。
- **C3**：评测含预训练、GLUE与LRA长序列任务。

## 争议与不确定点

- m固定的复杂度优势不保证小batch/短序列更快。
- encoder任务结果不证明任意causaldecode适用。

## 关联页面

- 主题：[注意力机制 Attention](../../wiki/topics/注意力机制%20Attention.md)
- 概念：[Transformer](../../wiki/concepts/Transformer.md)

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。

## 方法与实验解读

用少量地标重建全局关系，保留全局交互意图但改变计算结果。与Linformer的学习投影相比，landmarks的取法、伪逆稳定与分段方式决定近似质量；与FlashAttention的精确重排不能混为一类。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Xiong%20et%20al.%20-%202021%20-%20Nystr%C3%B6mformer%20A%20Nystrom-Based%20Algorithm%20for%20Approximating%20Self-Attention.md#source-section-8 ) | landmark数m远小于n时才相对n线性。 |
| C2 | [原文]( ../../raw/text/Xiong%20et%20al.%20-%202021%20-%20Nystr%C3%B6mformer%20A%20Nystrom-Based%20Algorithm%20for%20Approximating%20Self-Attention.md#source-section-13 ) | 误差与稳定性依这些实现。 |
| C3 | [原文]( ../../raw/text/Xiong%20et%20al.%20-%202021%20-%20Nystr%C3%B6mformer%20A%20Nystrom-Based%20Algorithm%20for%20Approximating%20Self-Attention.md#source-section-17 ) | 非现代LLM所有部署情形。 |

## 核证范围

核读landmarks/近似、伪逆/skip与GLUE/LRA设定。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

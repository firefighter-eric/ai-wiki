---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Beltagy, Peters, Cohan - 2020 - Longformer The Long-Document Transformer

## TL;DR（快速导读）

Longformer 让大多数位置只看附近内容，少数任务关键位置看全篇，以较少连接处理长文。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

普通正文主要交流附近信息，问题或关键标记可访问全篇；这种连接选择与精确算子优化不同。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Beltagy, Peters, Cohan - 2020 - Longformer The Long-Document Transformer.pdf
- 原始 HTML：../../raw/html/Beltagy, Peters, Cohan - 2020 - Longformer The Long-Document Transformer.html
- 全文文本：../../raw/text/Beltagy, Peters, Cohan - 2020 - Longformer The Long-Document Transformer.md
- 作者：Beltagy, Peters, Cohan
- 年份：2020
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

这种局部窗口加全局位置的结构主动减少注意力连接，适合研究长文档中的信息传播。它改变了连接方式；哪些位置需要全局访问、远处证据能否有效到达，取决于任务设计与训练。

## 关键事实

- **C1**：Longformer 用局部窗口与少量全局 token 稀疏化注意力。
- **C2**：局部窗口每 token 的成本为 O(nw)，堆叠层扩大感受野。
- **C3**：分类可令 CLS 为全局 token，问答可令问题 token 为全局 token；全局交互采用单独的 Q/K/V 投影。
- **C4**：预训练版从 RoBERTa 继续 MLM，使用 4096 长度和长短文混合语料，再做下游微调。
- **C5**：报告引入 LED 扩展长文生成，并在 arXiv 摘要任务验证。

## 争议与不确定点

- 固定窗口的线性成本不表示所有长文推理问题都已解决。
- 不同 kernel 对速度与 dilation 支持不同；本页不将理论复杂度当硬件吞吐。

## 关联页面

- 主题：[注意力机制 Attention](../../wiki/topics/注意力机制%20Attention.md)
- 概念：[Transformer](../../wiki/concepts/Transformer.md)

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **encoder**：编码器：把输入转成模型内部表示。
- **decoder**：解码器：根据已有表示产生文字、图像或其他输出。

## 方法与实验解读

局部窗口建立邻域表示，全局 token 给任务关键位置直接访问全篇的通道。计算复杂度、CUDA 内核速度和下游准确率分别回答能否存下、能否快速计算与信息是否充分交互。作者的 RoBERTa 继续预训练也说明，已有稠密模型改成稀疏结构后，需要让模型适应新的信息路径。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Beltagy%2C%20Peters%2C%20Cohan%20-%202020%20-%20Longformer%20The%20Long-Document%20Transformer.md#source-section-11 ) | 线性复杂度需要窗口与全局 token 数不随序列长度同比增加。 |
| C2 | [原文]( ../../raw/text/Beltagy%2C%20Peters%2C%20Cohan%20-%202020%20-%20Longformer%20The%20Long-Document%20Transformer.md#source-section-9 ) | 固定 w 时随 n 线性；长距离交互仍受模式设计影响。 |
| C3 | [原文]( ../../raw/text/Beltagy%2C%20Peters%2C%20Cohan%20-%202020%20-%20Longformer%20The%20Long-Document%20Transformer.md#source-section-12 ) | 任务配置是模型的一部分，不能只替换掩码就认为等价。 |
| C4 | [原文]( ../../raw/text/Beltagy%2C%20Peters%2C%20Cohan%20-%202020%20-%20Longformer%20The%20Long-Document%20Transformer.md#source-section-24 ) | 与字符自回归实验路线分开。 |
| C5 | [原文]( ../../raw/text/Beltagy%2C%20Peters%2C%20Cohan%20-%202020%20-%20Longformer%20The%20Long-Document%20Transformer.md#source-section-2 ) | 当前保存 HTML 对 LED 的细节较少，本页不推断 encoder/decoder 的全部实现。 |

## 核证范围

核读局部/全局注意力、投影与实现、继续 MLM 配方，并限定 LED 结论到当前来源明确覆盖的范围。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

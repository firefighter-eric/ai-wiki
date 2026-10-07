---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
source_id: arxiv:2109.10282
---
# Li et al. - 2021 - TrOCR Transformer-based Optical Character Recognition with Pre-trained Models

## TL;DR（快速导读）

TrOCR 将预训练图像 Transformer 与文本 Transformer 组合，直接把文字图像解码成文本，研究端到端文字识别。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

传统识别系统常分为视觉编码、字符生成和语言后处理。TrOCR 用预训练的编码器与解码器完成图像到文本转换。这里重点是识别文本区域；整页版面、阅读顺序和表格结构仍需另外讨论。

## 具体怎么理解

输入一张裁剪的手写单词图片，输出对应文字；找到页面上所有文本块与排序并不自动包含在这一步中。

## 关键事实

- **C1**：TrOCR 将裁剪后的文本图像分成 patch，以视觉 Transformer 编码，再由带交叉注意力的文本 decoder 自回归生成 wordpiece。
- **C2**：编码器以 DeiT/BEiT 初始化，解码器以 RoBERTa/MiniLM 初始化，再通过合成文本行和真实数据训练。
- **C3**：SROIE 采用词级 P/R/F1，IAM 采用字符错误率；场景文本结果需区分仅合成微调和加入基准训练数据的设置。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Li%20et%20al.%20-%202021%20-%20TrOCR%20Transformer-based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.pdf)
- 全文文本：[打开全文文本](../../raw/text/Li%20et%20al.%20-%202021%20-%20TrOCR%20Transformer-based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.md)
- 作者：Li et al.
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Li%20et%20al.%20-%202021%20-%20TrOCR%20Transformer-based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.html)

## 争议与不确定点

- 场景文本中符号是否计入答案的标注不一致，会导致部分数据集表现变差。
- 识别准确不等于文档问答或表格结构正确。
- 同 arXiv ID 的另一归档不能作为独立重复验证。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无

## 方法与实验解读

先把一行字裁出来，编码器理解字形，解码器逐步生成文字。这样可以利用现成视觉和语言预训练权重，并让解码器承担语言建模；如果输入是整页发票，仍要另外解决字在哪、先读哪行以及字段关系。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Li%20et%20al.%20-%202021%20-%20TrOCR%20Transformer-based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.md#source-section-6 ) | 核心任务是文本识别，不自动完成整页检测、阅读顺序与字段理解。 |
| C2 | [原文]( ../../raw/text/Li%20et%20al.%20-%202021%20-%20TrOCR%20Transformer-based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.md#source-section-17 ) | 端到端识别仍依赖训练数据与图像裁剪；合成语料与伪标签有噪声。 |
| C3 | [原文]( ../../raw/text/Li%20et%20al.%20-%202021%20-%20TrOCR%20Transformer-based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.md#source-section-25 ) | 不同数据集、输出标点约定与训练数据会影响排名，不能混合指标。 |

## 核证范围

核对 Encoder/Decoder、预训练数据、SROIE 与 IAM、Scene Text 数据与符号失败情况。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

## 来源归档关系

本页作为该论文的主阅读入口；另一归档是 [Li et al. - 2023 - TrOCR Transformer-Based Optical Character Recognition with Pre-trained Models](Li%20et%20al.%20-%202023%20-%20TrOCR%20Transformer-Based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.md)。按 arxiv:2109.10282 合并计数；不同保存版本可用于核对修订，不能当作独立实验或独立来源复现。

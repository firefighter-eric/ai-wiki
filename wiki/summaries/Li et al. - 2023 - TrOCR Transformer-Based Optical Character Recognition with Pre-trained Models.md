---
type: summary
status: refined
canonical_summary: wiki/summaries/Li et al. - 2021 - TrOCR Transformer-based Optical Character Recognition with Pre-trained Models.md
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
source_id: arxiv:2109.10282
---
# Li et al. - 2023 - TrOCR Transformer-Based Optical Character Recognition with Pre-trained Models

## TL;DR（快速导读）

这份 TrOCR 来源讨论利用预训练编码器和解码器识别文字，与库中另一份 TrOCR 来源可能是同一工作的不同版本。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

TrOCR 从文字图像生成字符序列，尝试统一视觉理解和文本生成。比较库中两份来源时，应核对 arXiv ID、版本和出版信息，再判断内容差异；不能把重复归档当成两份独立支持证据。

## 具体怎么理解

例如同一篇论文的预印本和正式出版版都被保存，研究结论的独立证据数量仍然是一项工作。

## 关键事实

- **C1**：TrOCR 将裁剪后的文本图像分成 patch，以视觉 Transformer 编码，再由带交叉注意力的文本 decoder 自回归生成 wordpiece。
- **C2**：编码器以 DeiT/BEiT 初始化，解码器以 RoBERTa/MiniLM 初始化，再通过合成文本行和真实数据训练。
- **C3**：SROIE 采用词级 P/R/F1，IAM 采用字符错误率；场景文本结果需区分仅合成微调和加入基准训练数据的设置。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Li%20et%20al.%20-%202023%20-%20TrOCR%20Transformer-Based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.pdf)
- 全文文本：[打开全文文本](../../raw/text/Li%20et%20al.%20-%202023%20-%20TrOCR%20Transformer-Based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.md)
- 作者：Li et al.
- 年份：2023
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Li%20et%20al.%20-%202023%20-%20TrOCR%20Transformer-Based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.html)
- 核对说明：与 2021 TrOCR 条目指向同一 arXiv 研究，保留历史路径并按一次独立证据计。

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
| C1 | [原文]( ../../raw/text/Li%20et%20al.%20-%202023%20-%20TrOCR%20Transformer-Based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.md#source-section-6 ) | 核心任务是文本识别，不自动完成整页检测、阅读顺序与字段理解。 |
| C2 | [原文]( ../../raw/text/Li%20et%20al.%20-%202023%20-%20TrOCR%20Transformer-Based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.md#source-section-17 ) | 端到端识别仍依赖训练数据与图像裁剪；合成语料与伪标签有噪声。 |
| C3 | [原文]( ../../raw/text/Li%20et%20al.%20-%202023%20-%20TrOCR%20Transformer-Based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.md#source-section-25 ) | 不同数据集、输出标点约定与训练数据会影响排名，不能混合指标。 |

## 核证范围

核对 Encoder/Decoder、预训练数据、SROIE 与 IAM、Scene Text 数据与符号失败情况。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

## 来源归档关系

同一论文的另一归档；主题综合优先引用 [Li et al. - 2021 - TrOCR Transformer-based Optical Character Recognition with Pre-trained Models](Li%20et%20al.%20-%202021%20-%20TrOCR%20Transformer-based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.md)。按 arxiv:2109.10282 合并计数；不同保存版本可用于核对修订，不能当作独立实验或独立来源复现。

---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Wei, Sun, Li - 2026 - DeepSeek-OCR 2 Visual Causal Flow

## TL;DR（快速导读）

DeepSeek-OCR 2 研究怎样按语义组织视觉词元的顺序，再交给语言模型读取复杂页面。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

输入一整页扫描文档，先以视觉表示压缩，再输出文本；图表、顺序和公式应逐项核对。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Wei, Sun, Li - 2026 - DeepSeek-OCR 2 Visual Causal Flow.pdf
- 全文文本：../../raw/text/Wei, Sun, Li - 2026 - DeepSeek-OCR 2 Visual Causal Flow.md
- 作者：Wei, Sun, Li
- 年份：2026
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

方法通过更新视觉编码器和因果流查询调整视觉信息顺序，减少机械逐行扫描与实际阅读关系的差异。它延续视觉文本压缩路线，但顺序正确与内容识别正确仍是不同验收目标。

## 关键事实

- **C1**：DeepEncoderV2将双向视觉tokens与因果query流结合，只把query输出送decoder。
- **C2**：global256加k个local144query，k0–6，总256–1120。
- **C3**：OmniDocBench1.5报告91.09，对初代增3.73点，readingorderED0.085降0.057。

## 争议与不确定点

- 类似数据来源不意味着完全相同训练样本/算力，因果归因仍需对应消融。
- 与通用VLM比较要对齐visualbudget、prompt和输出格式。

## 关联页面

- 概念：[DeepSeek-OCR](../../wiki/concepts/DeepSeek-OCR.md)
- 家族前序：[Wei, Sun, Li - 2025 - DeepSeek-OCR Contexts Optical Compression](./Wei,%20Sun,%20Li%20-%202025%20-%20DeepSeek-OCR%20Contexts%20Optical%20Compression.md)
- 概念：[DeepSeek](../../wiki/concepts/DeepSeek.md)
- 主题：[OCR](../../wiki/topics/OCR.md)
- [DeepSeek](../authors/DeepSeek.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **OCR**：文字识别：从图像读取文字；整页任务还需处理布局和阅读顺序。

## 方法与实验解读

初代侧重token压缩，二代在相近预算下重新组织视觉信息，使阅读顺序成为结构性目标。因果query能利用前序聚合结果，但仍需检查数值与表结构，而不是看到更高overall就推断文档完整准确。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Wei%2C%20Sun%2C%20Li%20-%202026%20-%20DeepSeek-OCR%202%20Visual%20Causal%20Flow.md#source-section-12 ) | 语义重排通过learnedqueries，不是显式排序每个像素。 |
| C2 | [原文]( ../../raw/text/Wei%2C%20Sun%2C%20Li%20-%202026%20-%20DeepSeek-OCR%202%20Visual%20Causal%20Flow.md#source-section-13 ) | multi-crop具体预算。 |
| C3 | [原文]( ../../raw/text/Wei%2C%20Sun%2C%20Li%20-%202026%20-%20DeepSeek-OCR%202%20Visual%20Causal%20Flow.md#source-section-23 ) | 点数不是相对改善3.73%；固定benchmark版本。 |

## 核证范围

核读dualstream/attentionmask、query预算、training与主结果/顺序指标。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

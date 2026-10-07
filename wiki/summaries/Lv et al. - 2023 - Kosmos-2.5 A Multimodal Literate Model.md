---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Lv et al. - 2023 - Kosmos-2.5 A Multimodal Literate Model

## TL;DR（快速导读）

Kosmos-2.5 读取文字密集图像，既能输出带位置的文本块，也能输出保留结构与样式的 Markdown。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

整页文档需要同时保留内容和组织。模型学习空间相关转写与结构化文本输出，使读者能分别检查位置、文字和格式。良好转写仍不代表已经完成文档问答或事实推理。

## 具体怎么理解

例如同一页可输出“这段文字位于哪里”，也可输出标题、列表和正文结构；两种表示各有用途。

## 关键事实

- **C1**：采用视觉编码器、Resampler 与语言解码器，输出带位置框的文字行或 Markdown；不能把整个视觉系统描述成只有语言解码器。
- **C2**：文字位置通过离散坐标 token 表示，另一任务直接生成 Markdown。
- **C3**：本文 NED / NTED 定义为 1 减归一化编辑距离，因此越高越好；OCR 另用词级 precision、recall、F1。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Lv%20et%20al.%20-%202023%20-%20Kosmos-2.5%20A%20Multimodal%20Literate%20Model.pdf)
- 全文文本：[打开全文文本](../../raw/text/Lv%20et%20al.%20-%202023%20-%20Kosmos-2.5%20A%20Multimodal%20Literate%20Model.md)
- 作者：Lv et al.
- 年份：2023
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Lv%20et%20al.%20-%202023%20-%20Kosmos-2.5%20A%20Multimodal%20Literate%20Model.html)

## 争议与不确定点

- 与 Nougat 的差异还包含训练数据覆盖，不能归因于架构一项因素。
- 生成格式正确不保证所有文字、公式和布局均准确；本页未把演示图当作全面可靠性证据。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无

## 方法与实验解读

模型把文字密集图像转成两种可读表达：需要位置时输出文字和框，需要文档结构时输出 Markdown。视觉编码器负责读取图像，Resampler 压缩视觉表示，语言解码器负责序列生成。评估时须把文字识别与结构保持分开。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Lv%20et%20al.%20-%202023%20-%20Kosmos-2.5%20A%20Multimodal%20Literate%20Model.md#source-section-5 ) | 视觉组件与生成头分开理解 |
| C2 | [原文]( ../../raw/text/Lv%20et%20al.%20-%202023%20-%20Kosmos-2.5%20A%20Multimodal%20Literate%20Model.md#source-section-6 ) | 页面文字定位与结构生成是不同输出格式 |
| C3 | [原文]( ../../raw/text/Lv%20et%20al.%20-%202023%20-%20Kosmos-2.5%20A%20Multimodal%20Literate%20Model.md#source-section-20 ) | 该定义不能直接套用通常越低越好的编辑距离指标 |

## 核证范围

核对 §2.1–2.2、§3.1 的指标定义、§3.3–3.4 的结果与讨论。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

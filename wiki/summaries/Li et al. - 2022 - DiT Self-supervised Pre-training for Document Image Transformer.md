---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Li et al. - 2022 - DiT Self-supervised Pre-training for Document Image Transformer

## TL;DR（快速导读）

文档 DiT 在大量未标注文档图像上自监督预训练，为版面分析等文档视觉任务提供表示底座。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

自然照片与文档页面的结构不同。本文把图像 Transformer 预训练用于文档图像，研究迁移到文档 AI 任务的效果。这里的 DiT 指 Document Image Transformer，需与图像生成领域的 Diffusion Transformer 区分。

## 具体怎么理解

例如发票、论文和表单的视觉布局有各自规律，专门学习文档图像可能帮助后续识别这些结构。

## 关键事实

- **C1**：这里 DiT 指 Document Image Transformer，采用 ViT 编码文档图像 patch。
- **C2**：用 Masked Image Modeling 预测视觉 token，不要求人工页面标签预训练。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Li%20et%20al.%20-%202022%20-%20DiT%20Self-supervised%20Pre-training%20for%20Document%20Image%20Transformer.pdf)
- 全文文本：[打开全文文本](../../raw/text/Li%20et%20al.%20-%202022%20-%20DiT%20Self-supervised%20Pre-training%20for%20Document%20Image%20Transformer.md)
- 作者：Li et al.
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Li%20et%20al.%20-%202022%20-%20DiT%20Self-supervised%20Pre-training%20for%20Document%20Image%20Transformer.html)

## 争议与不确定点

- 无标签预训练不等于下游全程无监督。
- 训练文档与扫描、语言或格式变化仍会影响迁移。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无
- [文档与表格的输入输出接口](../comparisons/%E6%96%87%E6%A1%A3%E4%B8%8E%E8%A1%A8%E6%A0%BC%E7%9A%84%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%E6%8E%A5%E5%8F%A3.md)：把本篇方法放到相关任务与比较条件中阅读。

## 方法与实验解读

DiT 在大量文档图像上学习视觉表示，再用于分类、布局和检测。它不是直接生成 Markdown 的 OCR 系统；下游文字检测与字符识别也要分别理解。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Li%20et%20al.%20-%202022%20-%20DiT%20Self-supervised%20Pre-training%20for%20Document%20Image%20Transformer.md#source-section-6 ) | 与 Diffusion Transformer 的同名缩写区别 |
| C2 | [原文]( ../../raw/text/Li%20et%20al.%20-%202022%20-%20DiT%20Self-supervised%20Pre-training%20for%20Document%20Image%20Transformer.md#source-section-7 ) | 下游任务仍需对应训练和头部 |

## 核证范围

核对 §3.1–3.2 的架构与 MIM、预训练设置和下游检测范围。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

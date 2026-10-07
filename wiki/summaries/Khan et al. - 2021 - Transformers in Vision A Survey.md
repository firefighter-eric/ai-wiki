---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Khan et al. - 2021 - Transformers in Vision A Survey

## TL;DR（快速导读）

这篇综述整理 Transformer 在视觉任务中的用法，帮助比较全局关系建模、图像表示和计算成本。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

视觉 Transformer 需要把图像转换成适合序列处理的表示，并与分类、检测等任务结合。综述讨论方法结构、优点与挑战。读者应按任务和输入表示归类，避免把所有名称带 Transformer 的模型当成同一种方案。

## 具体怎么理解

图像分类关注整张图的类别，检测还要定位多个对象；二者虽然都能用 Transformer，输出接口却不同。

## 关键事实

- **C1**：综述整理 Transformer 在视觉多任务中的应用，而不是提出统一胜出模型。
- **C2**：将计算成本与数据需求列为主要挑战。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Khan%20et%20al.%20-%202021%20-%20Transformers%20in%20Vision%20A%20Survey.pdf)
- 全文文本：[打开全文文本](../../raw/text/Khan%20et%20al.%20-%202021%20-%20Transformers%20in%20Vision%20A%20Survey.md)
- 作者：Khan et al.
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Khan%20et%20al.%20-%202021%20-%20Transformers%20in%20Vision%20A%20Survey.html)

## 争议与不确定点

- 2021 年后的方法不由本文覆盖。
- 分类结论不能直接迁移到视频、检测或三维任务。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无

## 方法与实验解读

这篇综述帮助辨认全局注意力、局部结构和混合网络在不同视觉任务中的取舍。其关于数据和归纳偏置的讨论应结合具体模型：Transformer 仍有位置、mask 与结构设计，不能理解为完全没有先验。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Khan%20et%20al.%20-%202021%20-%20Transformers%20in%20Vision%20A%20Survey.md#source-section-47 ) | 历史研究地图，不是当前排行榜 |
| C2 | [原文]( ../../raw/text/Khan%20et%20al.%20-%202021%20-%20Transformers%20in%20Vision%20A%20Survey.md#source-section-40 ) | 随架构、训练配方和任务而变化 |

## 核证范围

核对结论的综述范围、§4.1 计算成本及 §4.2 数据需求。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

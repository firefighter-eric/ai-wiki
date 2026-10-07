---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Xiao et al. - 2023 - Florence-2 Advancing a Unified Representation for a Variety of Vision Tasks

## TL;DR（快速导读）

Florence-2 用文字提示指定视觉任务，再生成对应描述或位置等输出，把多种视觉能力放进统一接口。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

视觉任务的语义层次不同，既有整图描述也有区域定位。Florence-2 用提示驱动的生成式表示组织这些任务。需要分别检验输出结构、定位与内容质量，不能用一种任务表现概括全部能力。

## 具体怎么理解

输入同一张图，要求“描述场景”得到一句话，要求“找出汽车”则需要相应位置输出。

## 关键事实

- **C1**：以图像与任务提示为输入，通过 seq2seq 生成任务输出，共用一套权重和架构。
- **C2**：评测分别报告 zero-shot 与任务微调，零样本 caption 分数是特定数据与提示下的结果。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Xiao%20et%20al.%20-%202023%20-%20Florence-2%20Advancing%20a%20Unified%20Representation%20for%20a%20Variety%20of%20Vision%20Tasks.pdf)
- 全文文本：[打开全文文本](../../raw/text/Xiao%20et%20al.%20-%202023%20-%20Florence-2%20Advancing%20a%20Unified%20Representation%20for%20a%20Variety%20of%20Vision%20Tasks.md)
- 作者：Xiao et al.
- 年份：2023
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Xiao%20et%20al.%20-%202023%20-%20Florence-2%20Advancing%20a%20Unified%20Representation%20for%20a%20Variety%20of%20Vision%20Tasks.html)

## 争议与不确定点

- 与超大模型的比较含数据、任务和训练差异，并非纯参数规模控制实验。
- 坐标和结构输出仍需正确解析与验证。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无
- [Microsoft Research](../authors/Microsoft%20Research.md)：沿作者或机构继续阅读相关来源。

## 方法与实验解读

Florence-2 把图像级描述、区域定位与更细粒度视觉任务纳入生成接口。统一接口便于复用，但质量来自视觉编码、任务表示和多粒度训练数据的组合，不能只用参数小来解释能力。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Xiao%20et%20al.%20-%202023%20-%20Florence-2%20Advancing%20a%20Unified%20Representation%20for%20a%20Variety%20of%20Vision%20Tasks.md#source-section-6 ) | 统一的是任务建模，输出仍需按任务解析 |
| C2 | [原文]( ../../raw/text/Xiao%20et%20al.%20-%202023%20-%20Florence-2%20Advancing%20a%20Unified%20Representation%20for%20a%20Variety%20of%20Vision%20Tasks.md#source-section-29 ) | 不能把微调或预训练数据覆盖混称未见任务泛化 |

## 核证范围

核对 §3 模型、§6.1 设置、§6.2 零样本评测及结论的数据范围。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

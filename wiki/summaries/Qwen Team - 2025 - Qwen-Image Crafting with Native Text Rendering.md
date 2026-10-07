---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Qwen Team - 2025 - Qwen-Image Crafting with Native Text Rendering

## TL;DR（快速导读）

Qwen-Image 以文字渲染和图像编辑为重点，适合研究生成图中的中英文文字怎样更可控。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

生成一张带中文招牌的图片，需要既检查场景与构图，也逐字检查招牌文字。

## 来源信息

- 类型：官方博客 / 技术发布
- 来源链接：https://qwenlm.github.io/blog/qwen-image/
- 全文文本：../../raw/text/Qwen Team - 2025 - Qwen-Image Crafting with Native Text Rendering.md
- 作者：Qwen Team
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

官方介绍图像基础模型及复杂文字、双语内容与精确编辑能力。实际使用时要分别检查文字准确性、布局和未编辑区域是否稳定；图像审美分数不能替代这些具体要求。

## 关键事实

- **C1**：Qwen-Image是20B MMDiT image foundation model，发布重点为文字渲染和编辑。
- **C2**：公开报告覆盖GenEval/DPG/OneIG、GEdit/ImgEdit/GSO和LongText/ChineseWord/TextCraft。
- **C3**：demo包含多行中文/英文、文字编辑、增删对象与姿态变化。

## 争议与不确定点

- 图表数字未在正文完整文本化，本文不补猜分数。
- 段落和小字可能出现遗漏或变形，生产文档需逐字审核。

## 关联页面

- 概念：[Qwen-Image](../../wiki/concepts/Qwen-Image.md)
- 主题：[扩散模型与文生图](../../wiki/topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
- 主题：[Qwen 系列](../../wiki/topics/Qwen%20系列.md)

## 方法与实验解读

文字任务同时要求内容拼写、排版、风格和图像语境，单纯画面好看不足以证明字符准确。编辑还要看未编辑区域是否保持。官方跨基准领先声明应绑定发布版本，不能由展示结果推断现在仍排名第一。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Qwen%20Team%20-%202025%20-%20Qwen-Image%20Crafting%20with%20Native%20Text%20Rendering.md#source-section-0 ) | 官方介绍，不补推训练细节。 |
| C2 | [原文]( ../../raw/text/Qwen%20Team%20-%202025%20-%20Qwen-Image%20Crafting%20with%20Native%20Text%20Rendering.md#source-section-1 ) | 生成、编辑、文字任务不同。 |
| C3 | [原文]( ../../raw/text/Qwen%20Team%20-%202025%20-%20Qwen-Image%20Crafting%20with%20Native%20Text%20Rendering.md#source-section-2 ) | 精挑例子不等于未筛选成功率。 |

## 核证范围

核读原始介绍、Performance与示例任务，限定官方发布声明。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

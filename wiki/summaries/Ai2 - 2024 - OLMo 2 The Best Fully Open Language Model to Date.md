---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Ai2 - 2024 - OLMo 2 The Best Fully Open Language Model to Date

## TL;DR（快速导读）

OLMo 2 的价值在于把权重、训练数据、代码与评测一起公开，并展示稳定训练和后期数据课程如何改善 7B/13B 模型。性能结论应按发布时的英语基准理解。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

可下载权重与可检查训练数据、代码和过程，是不同层次的开放；复现仍需匹配资源和设置。

## 来源信息

- 类型：官方博客 / 模型发布说明
- 原始文件：../../raw/html/Ai2 - 2024 - OLMo 2 The Best Fully Open Language Model to Date.html
- 全文文本：../../raw/text/Ai2 - 2024 - OLMo 2 The Best Fully Open Language Model to Date.md
- 作者：Ai2
- 年份：2024
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

这份来源主要帮助理解可检查的模型训练：读者可以沿数据、训练配方和中间产物研究模型如何形成。比较开放程度时，需逐项检查这些材料，而不能只看权重是否可以下载。

## 关键事实

- **C1**：发布版包括 7B 与 13B，训练量最高约 5T tokens。
- **C2**：博客将 fully open 定义为权重、训练数据、代码和评测完整开放。
- **C3**：首阶段使用 OLMo-Mix-1124，后期转向高质量 Dolmino-Mix，并衰减学习率。
- **C4**：7B 后期训练三个不同数据顺序的 50B-token 分支，再做 model souping；13B 使用另一组分支规模。
- **C5**：英语基准区分开发集和开发期间未看的任务；最优 fully-open 是作者在该比较集合中的判断。

## 争议与不确定点

- 官方发布材料，性能表是作者报告，未在本库独立复现。
- to-date 和 best 等表述具有发布日期与候选集合限制，不作为当前排行榜结论。

## 关联页面

- 概念：[OLMo 2](../../wiki/concepts/OLMo%202.md)
- 概念：[BLOOM](../../wiki/concepts/BLOOM.md)
- 主题：[LLM 预训练](../../wiki/topics/LLM%20预训练.md)
- 比较：[开放模型家族与中国重要家族对照](../../wiki/comparisons/开放模型家族与中国重要家族对照.md)

## 方法与实验解读

模型改进同时涉及 RMSNorm/QK-Norm 等稳定性措施、后期数据课程、学习率退火和模型合并。OLMES 将开发过程中的反馈与最后的未见任务分开，有助于判断调参收益是否迁移；但博客中的帕累托位置仍依赖所选模型、训练计算估计和英语基准。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Ai2%20-%202024%20-%20OLMo%202%20The%20Best%20Fully%20Open%20Language%20Model%20to%20Date.md#source-section-2 ) | 2024 年发布博客口径。 |
| C2 | [原文]( ../../raw/text/Ai2%20-%202024%20-%20OLMo%202%20The%20Best%20Fully%20Open%20Language%20Model%20to%20Date.md#source-section-2 ) | 开放程度与推理能力是两个维度。 |
| C3 | [原文]( ../../raw/text/Ai2%20-%202024%20-%20OLMo%202%20The%20Best%20Fully%20Open%20Language%20Model%20to%20Date.md#source-section-3 ) | 两个阶段的数据比例和训练时长不能混为一个总配方。 |
| C4 | [原文]( ../../raw/text/Ai2%20-%202024%20-%20OLMo%202%20The%20Best%20Fully%20Open%20Language%20Model%20to%20Date.md#source-section-3 ) | 这是该版本配方，不是所有 OLMo 模型的固定配置。 |
| C5 | [原文]( ../../raw/text/Ai2%20-%202024%20-%20OLMo%202%20The%20Best%20Fully%20Open%20Language%20Model%20to%20Date.md#source-section-2 ) | 未见开发指标不保证其他模型未使用该任务，也不消除数据污染可能。 |

## 核证范围

核读 Announcing OLMo 2、模型开放分类、OLMES 比较与 Pretraining 两阶段配方。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

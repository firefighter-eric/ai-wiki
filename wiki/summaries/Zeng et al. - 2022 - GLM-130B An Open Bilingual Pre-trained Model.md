---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Zeng et al. - 2022 - GLM-130B An Open Bilingual Pre-trained Model

## TL;DR（快速导读）

GLM-130B 将 GLM 的填空式训练路线扩展到大型中英双语模型，是追踪后续家族的基础资料。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

GLM 预训练论文解释方法，ChatGLM 或后续报告描述具体模型；两类材料的结论不能直接互换。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Zeng et al. - 2022 - GLM-130B An Open Bilingual Pre-trained Model.pdf
- 全文文本：../../raw/text/Zeng et al. - 2022 - GLM-130B An Open Bilingual Pre-trained Model.md
- 作者：Zeng et al.
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

报告介绍双语预训练与开放模型。阅读时应区分基础训练目标、语言覆盖、能力评测和后来的聊天版本；家族关系不能替代每代模型具体训练与发布信息。

## 关键事实

- **C1**：GLM130B基于autoregressiveblankinfilling与双向上下文结构，面向中英。
- **C2**：95%tokens自监督，另混入小比例multitask指令学习。
- **C3**：采用平台感知3Dparallel、稳定化DeepNorm等技术。

## 争议与不确定点

- 不同pretraining目标与fewshot格式影响跨模型公平比较。
- 量化可用性不保证每个设备和任务无精度损失。

## 关联页面

- 概念：[GLM](../../wiki/concepts/GLM.md)
- 主题：[LLM 预训练](../../wiki/topics/LLM%20预训练.md)
- 比较：[开放模型家族与中国重要家族对照](../../wiki/comparisons/开放模型家族与中国重要家族对照.md)

## 方法与实验解读

被mask的跨度按序生成，而已知上下文可双向读取。模型通过训练稳定性与量化部署使超大规模更可用；“开放双语”不能仅用参数量解释，要回到数据、mask、instruction配方和语言评测。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Zeng%20et%20al.%20-%202022%20-%20GLM-130B%20An%20Open%20Bilingual%20Pre-trained%20Model.md#source-section-5 ) | 不等同纯GPTcausal目标。 |
| C2 | [原文]( ../../raw/text/Zeng%20et%20al.%20-%202022%20-%20GLM-130B%20An%20Open%20Bilingual%20Pre-trained%20Model.md#source-section-6 ) | 训练目标贡献需要拆开。 |
| C3 | [原文]( ../../raw/text/Zeng%20et%20al.%20-%202022%20-%20GLM-130B%20An%20Open%20Bilingual%20Pre-trained%20Model.md#source-section-7 ) | 硬件/配方与架构共同决定效果。 |

## 核证范围

核读GLM目标、双语预训练/多任务、稳定化/并行与评测设计。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

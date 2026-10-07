---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Qwen Team - 2024 - Hello Qwen2

## TL;DR（快速导读）

Qwen2 发布页介绍语言覆盖、长上下文及代码数学能力的更新，适合辨认家族代际变化。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

密集模型与专家混合型号的总参数、每次计算和部署方式不同，不能只按名称中的数字比较。

## 来源信息

- 类型：官方博客 / 技术发布
- 来源链接：https://qwenlm.github.io/blog/qwen2/
- 全文文本：../../raw/text/Qwen Team - 2024 - Hello Qwen2.md
- 作者：Qwen Team
- 年份：2024
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

它将预训练和后训练改进放在模型家族中说明。核对能力时，应选择具体尺寸、基础或聊天版本，并查看对应任务与上下文设置；官方总体介绍不能替代每个成员的验证。

## 关键事实

- **C1**：Qwen2涵盖0.5B/1.5B/7B/57B-A14B/72B的base与instruct，并增加27种语言。
- **C2**：GQA扩展到各尺寸；instruct7B/72B报告128K支持，小模型32K、57B-A14B64K。
- **C3**：所有instruct先在32K训练，再用YARN/DualChunkAttention外推。
- **C4**：72B采用Qianwen License，其余此批采用Apache2.0。

## 争议与不确定点

- 博客性能是官方报告，跨模型prompt和评测条件需核对。
- 一百万token文档处理agent方案不等于所有模型原生窗口一百万。

## 关联页面

- 概念：[Qwen](../../wiki/concepts/Qwen.md)
- 概念：[Qwen2](../../wiki/concepts/Qwen2.md)
- 主题：[Qwen 系列](../../wiki/topics/Qwen%20系列.md)

## 这里的术语是什么意思

- **reward model**：奖励模型：根据训练信号给回答或行为打分，分数是目标的近似。
- **SFT**：监督微调：用输入与参考输出继续训练已有模型。
- **DPO**：直接偏好优化：用成对偏好数据直接调整模型概率，简化部分奖励训练流程。
- **post-training**：后训练：在预训练底座上继续调整指令遵循、偏好或其他行为。

## 方法与实验解读

本代将多语言、GQA和长文支持做成系列配方。把尺寸、激活量、KV结构和长度一并比较，才能解释部署成本；语言覆盖数量也不能代替分语言质量。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Qwen%20Team%20-%202024%20-%20Hello%20Qwen2.md#source-section-1 ) | dense与MoE不能只按总参数排成本。 |
| C2 | [原文]( ../../raw/text/Qwen%20Team%20-%202024%20-%20Hello%20Qwen2.md#source-section-2 ) | 训练长度与外推评测分开。 |
| C3 | [原文]( ../../raw/text/Qwen%20Team%20-%202024%20-%20Hello%20Qwen2.md#source-section-6 ) | needle-in-haystack不覆盖全部长文推理。 |
| C4 | [原文]( ../../raw/text/Qwen%20Team%20-%202024%20-%20Hello%20Qwen2.md#source-section-9 ) | 发布时型号许可，不能全家族一概而论。 |

## 核证范围

核读介绍、型号表、GQA、长文与许可。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

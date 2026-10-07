---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models

## TL;DR（快速导读）

DeepSeekMath 同时研究数学语料和强化学习，其中 GRPO 用同题多份回答的相对分数估计更新参照。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models.pdf
- 全文文本：../../raw/text/Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models.md
- 作者：Zhihong Shao et al.
- 年份：2024
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

论文把数学继续训练、指令训练和强化学习接起来。GRPO 不再单独训练价值模型，但仍依赖组内采样与奖励；数学数据质量和训练方法的作用应通过对应对照分别理解。

## 关键事实

- **C1**：GRPO用同问题多输出的相对reward估计advantage，省去同规模critic。
- **C2**：本报告RL从Instruct7B出发，约144K问题、每题64outputs、长度1024、KL0.04。
- **C3**：报告GSM8K88.2/MATH51.7的CoT成绩，in-domain与out-of-domain分开。
- **C4**：几何/定理证明与few-shot仍较弱，数学数据偏差可能参与解释。

## 争议与不确定点

- GRPO的名称在后续实现有多种改动，需固定loss/length normalization/reward版本。
- 可验证奖励也有覆盖盲点和格式偏差，得分高不代表推理过程真实可靠。

## 关联页面

- 主题：[LLM RL](../../wiki/topics/LLM%20RL.md)
- 概念：[GRPO](../../wiki/concepts/GRPO.md)
- 概念：[DeepSeek-R1](../../wiki/concepts/DeepSeek-R1.md)
- [DeepSeek](../authors/DeepSeek.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **critic**：价值模型：估计状态或行为的预期回报，为策略更新提供参照。
- **DPO**：直接偏好优化：用成对偏好数据直接调整模型概率，简化部分奖励训练流程。
- **GRPO**：组相对策略优化：利用同一问题多份回答的相对奖励进行更新。

## 方法与实验解读

组内中心化把同一道题的相对好坏变成更新信号，降低value训练成本；奖励无差异时却缺少区分信号。此报告同时改变数学数据、SFT和RL，GRPO贡献需看相应消融，不能把全部最终能力归因单一优化器。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Shao%20et%20al.%20-%202024%20-%20DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models.md#source-section-21 ) | 仍有policy/reference/reward或verifier，不能说无需其他模型。 |
| C2 | [原文]( ../../raw/text/Shao%20et%20al.%20-%202024%20-%20DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models.md#source-section-25 ) | 2024算法实验配置，不继承到R1所有配置。 |
| C3 | [原文]( ../../raw/text/Shao%20et%20al.%20-%202024%20-%20DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models.md#source-section-25 ) | 单样本与投票/工具分数不可混。 |
| C4 | [原文]( ../../raw/text/Shao%20et%20al.%20-%202024%20-%20DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models.md#source-section-40 ) | 作者明确局限。 |

## 核证范围

核读PPO到GRPO公式、outcome/process形式、训练配置/评测与限制。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

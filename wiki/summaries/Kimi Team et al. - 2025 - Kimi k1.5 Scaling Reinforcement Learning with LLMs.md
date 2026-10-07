---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Kimi Team et al. - 2025 - Kimi k1.5 Scaling Reinforcement Learning with LLMs

## TL;DR（快速导读）

Kimi k1.5 把强化学习、长上下文与多模态推理一起研究，是理解 Kimi 推理训练路线的资料。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

某代模型只提供接口，不代表后续所有型号也如此；先确认具体版本再讨论能力与使用方式。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Kimi Team et al. - 2025 - Kimi k1.5 Scaling Reinforcement Learning with LLMs.pdf
- 全文文本：../../raw/text/Kimi Team et al. - 2025 - Kimi k1.5 Scaling Reinforcement Learning with LLMs.md
- 作者：Kimi Team et al.
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

这份来源主要讨论怎样训练更复杂的推理行为，本身不是开放权重发布。它的公开范围只对应这个节点，不能把当时的开放性判断外推到后来整个家族；能力、训练机制与权重可得性需分开检查。

## 关键事实

- **C1**：k1.5训练阶段包含pretrain、vanilla SFT、long-CoT SFT与RL。
- **C2**：RL prompts覆盖可评估的STEM/竞赛/推理及图文任务。
- **C3**：作者把RL context扩到128K，并用partial rollouts复用轨迹提高效率。
- **C4**：报告提出long-to-short能力迁移，强调上下文与policy optimization联合影响。

## 争议与不确定点

- 可自动验证的问题与开放知识工作有不同奖励条件。
- 模型整体基准提升不能证明所有增益都由RL或context长度单独贡献。

## 关联页面

- 概念：[Kimi](../../wiki/concepts/Kimi.md)
- 概念：[Kimi K3](../../wiki/concepts/Kimi%20K3.md)
- 主题：[LLM 预训练](../../wiki/topics/LLM%20预训练.md)
- 主题：[LLM RL](../../wiki/topics/LLM%20RL.md)
- 比较：[开放模型家族与中国重要家族对照](../../wiki/comparisons/开放模型家族与中国重要家族对照.md)
- [Moonshot AI](../authors/Moonshot%20AI.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。
- **agentic**：代理执行：模型使用工具并根据结果继续行动，可靠性要看完整流程。

## 方法与实验解读

模型通过可检查奖励探索解题轨迹，较长rollout给多步推理空间；partial rollout减少重采样成本。长思考可以提高一些任务结果，却同时增加延迟和训练资源。后续开放K2/K3具有独立来源，不能把k1.5时期的API发布形态写成Kimi家族永久属性。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Kimi%20Team%20et%20al.%20-%202025%20-%20Kimi%20k1.5%20Scaling%20Reinforcement%20Learning%20with%20LLMs.md#source-section-4 ) | 本文重点是RL，不把全部能力归因单个优化步骤。 |
| C2 | [原文]( ../../raw/text/Kimi%20Team%20et%20al.%20-%202025%20-%20Kimi%20k1.5%20Scaling%20Reinforcement%20Learning%20with%20LLMs.md#source-section-5 ) | reward可验证性限制任务分布。 |
| C3 | [原文]( ../../raw/text/Kimi%20Team%20et%20al.%20-%202025%20-%20Kimi%20k1.5%20Scaling%20Reinforcement%20Learning%20with%20LLMs.md#source-section-3 ) | 128K训练预算与真实任务所需长度不同。 |
| C4 | [原文]( ../../raw/text/Kimi%20Team%20et%20al.%20-%202025%20-%20Kimi%20k1.5%20Scaling%20Reinforcement%20Learning%20with%20LLMs.md#source-section-2 ) | 具体收益需按推理长度和测试任务比较。 |

## 核证范围

核读Approach、RL prompt筛选、长上下文/partial-rollout与总结。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

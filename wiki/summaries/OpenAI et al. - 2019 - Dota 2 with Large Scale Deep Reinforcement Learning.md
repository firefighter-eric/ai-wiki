---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# OpenAI et al. - 2019 - Dota 2 with Large Scale Deep Reinforcement Learning

## TL;DR（快速导读）

OpenAI Five 把已有强化学习方法扩大到复杂的 Dota 2 环境，研究长时间决策、部分可见信息与大规模训练。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

Dota 2 要求在不完整信息下连续行动，单步正确并不足以赢得整局。报告描述大规模经验采样和策略训练，说明系统规模在复杂任务中的作用。游戏结果依赖规则与训练设置，不能直接推成通用现实代理能力。

## 具体怎么理解

一次行动可能到几分钟后才体现收益；模型要在短期动作和长期结果之间建立学习信号。

## 关键事实

- **C1**：OpenAI Five 使用 PPO 优化游戏策略，训练扩展包含大批量与长时间自博弈。
- **C2**：训练期用固定参考 agent 与 TrueSkill 跟踪能力，最终人类比赛是另一种验证。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/OpenAI%20et%20al.%20-%202019%20-%20Dota%202%20with%20Large%20Scale%20Deep%20Reinforcement%20Learning.pdf)
- 全文文本：[打开全文文本](../../raw/text/OpenAI%20et%20al.%20-%202019%20-%20Dota%202%20with%20Large%20Scale%20Deep%20Reinforcement%20Learning.md)
- 作者：OpenAI et al.
- 年份：2019
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/OpenAI%20et%20al.%20-%202019%20-%20Dota%202%20with%20Large%20Scale%20Deep%20Reinforcement%20Learning.html)

## 争议与不确定点

- 比赛规则与英雄、动作等限制需要随结果说明。
- 单场胜利不排除被新策略利用，论文另讨论开放挑战中的表现。

## 关联页面

- 主题：[LLM RL](../topics/LLM%20RL.md)
- 综合：暂无
- [OpenAI](../authors/OpenAI.md)：沿作者或机构继续阅读相关来源。

## 方法与实验解读

Dota 2 研究展示了强化学习在长程、部分可观测多人游戏中的工程规模。它优化的是特定游戏接口与奖励，不能因超越职业选手就推断系统具有通用世界理解或能解决任意开放任务。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/OpenAI%20et%20al.%20-%202019%20-%20Dota%202%20with%20Large%20Scale%20Deep%20Reinforcement%20Learning.md#source-section-7 ) | 环境、动作和观测接口均有特定设计 |
| C2 | [原文]( ../../raw/text/OpenAI%20et%20al.%20-%202019%20-%20Dota%202%20with%20Large%20Scale%20Deep%20Reinforcement%20Learning.md#source-section-10 ) | 代理评分与实际对局结果分开 |

## 核证范围

核对 §3.1–3.2 的策略接口与 PPO、§4.1 的评测逻辑；不外推为通用智能证明。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

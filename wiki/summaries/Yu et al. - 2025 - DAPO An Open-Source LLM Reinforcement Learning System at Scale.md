---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale

## TL;DR（快速导读）

DAPO 将长推理强化学习当作系统问题，分别处理探索收缩、采样、损失权重和过长回答。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

同一批题目若全答对或全答错，组内奖励可能缺少区分；有效采样与算法目标需要配合。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale.pdf
- 全文文本：../../raw/text/Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale.md
- 作者：Qiying Yu et al.
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

论文在 GRPO 基础上调整剪切、动态采样、词元级损失与长度相关奖励。它讨论的是把训练稳定跑起来的组合方案；各部分收益与奖励质量、生成长度和训练预算相关，应查看消融和复现条件。

## 关键事实

- **C1**：DAPO在GRPO框架上采用cliphigher、dynamicsampling、tokenloss与overlongrewardshaping。
- **C2**：过滤组内reward无差异prompt保持有效梯度，但需要额外rollouts。
- **C3**：tokenlevel归约改变长/短response权重，长度截断奖励避免把超长等同错误。
- **C4**：Qwen2.5-32B数学设定AIME24报告50%，不等同通用assistant性能。

## 争议与不确定点

- 数学可验证奖励结果不自动迁移客服/创作偏好。
- 作者公开系统不意味着硬件/数据不同也能原样复现分数。

## 关联页面

- 主题：[LLM RL](../../wiki/topics/LLM%20RL.md)
- 概念：[DAPO](../../wiki/concepts/DAPO.md)
- 概念：[GRPO](../../wiki/concepts/GRPO.md)

## 这里的术语是什么意思

- **GRPO**：组相对策略优化：利用同一问题多份回答的相对奖励进行更新。

## 方法与实验解读

训练崩塌可能来自探索不足、无信息样本、长度权重和奖励噪声，DAPO分别处理这些问题。完整RL成本包括rollout和额外采样，不能只数optimizerupdates。向别的任务迁移需要新的reward、长度与任务成功测试。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Yu%20et%20al.%20-%202025%20-%20DAPO%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.md#source-section-9 ) | 系统性组合，不是仅换一个clip值。 |
| C2 | [原文]( ../../raw/text/Yu%20et%20al.%20-%202025%20-%20DAPO%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.md#source-section-11 ) | sampleefficiency与总生成成本分开。 |
| C3 | [原文]( ../../raw/text/Yu%20et%20al.%20-%202025%20-%20DAPO%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.md#source-section-12 ) | loss与reward定义直接影响行为。 |
| C4 | [原文]( ../../raw/text/Yu%20et%20al.%20-%202025%20-%20DAPO%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.md#source-section-17 ) | 历史训练步数对照，不是总FLOPs减半。 |

## 核证范围

核读四项改进、verl/AdamW/rollout设置、AIME与消融。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

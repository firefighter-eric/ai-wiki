---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Wu et al. - 2025 - On the Generalization of SFT A Reinforcement Learning Perspective with Reward Rectification

## TL;DR（快速导读）

DFT 从强化学习视角分析监督微调的泛化，并根据词元概率调整训练目标，探索对标准 SFT 的简洁改进。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

作者分析标准监督微调梯度隐含的奖励结构，提出动态重标定以稳定更新，并在不同模型和任务上评估。原文的理论假设和具体实现应一起阅读，不能只凭代码改动小就认定收益普遍。

## 具体怎么理解

同样是学习参考答案，不同词元的当前概率可能不同；调整它们对目标的贡献，会改变模型更新方向。

## 关键事实

- **C1**：DFT 动态按目标 token 概率缩放微调目标，改变标准 SFT 的梯度权重。
- **C2**：实验集中在数学数据，其他任务、更大模型和多模态尚未系统验证。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Wu%20et%20al.%20-%202025%20-%20On%20the%20Generalization%20of%20SFT%20A%20Reinforcement%20Learning%20Perspective%20with%20Reward%20Rectification.pdf)
- 全文文本：[打开全文文本](../../raw/text/Wu%20et%20al.%20-%202025%20-%20On%20the%20Generalization%20of%20SFT%20A%20Reinforcement%20Learning%20Perspective%20with%20Reward%20Rectification.md)
- 作者：Wu et al.
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Wu%20et%20al.%20-%202025%20-%20On%20the%20Generalization%20of%20SFT%20A%20Reinforcement%20Learning%20Perspective%20with%20Reward%20Rectification.html)

## 争议与不确定点

- 正文的部分增益倍数与简单算术不完全一致，本页保留可验证的目标定义，不沿用这些倍数。
- 局限段的规模表述与主结果出现 8B 模型不完全一致，本页不使用其严格参数上限作为事实。

## 关联页面

- 主题：[LLM RL](../topics/LLM%20RL.md)
- 综合：暂无

## 方法与实验解读

DFT 用强化学习视角解释监督微调的权重问题，再给出更轻的目标改动。理论联系应按论文假设理解；数学任务增益值得复测，但不足以证明所有任务中 DFT 都优于 SFT。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Wu%20et%20al.%20-%202025%20-%20On%20the%20Generalization%20of%20SFT%20A%20Reinforcement%20Learning%20Perspective%20with%20Reward%20Rectification.md#source-section-2 ) | 属于训练目标修改，不是必须在线 rollout 的 RL |
| C2 | [原文]( ../../raw/text/Wu%20et%20al.%20-%202025%20-%20On%20the%20Generalization%20of%20SFT%20A%20Reinforcement%20Learning%20Perspective%20with%20Reward%20Rectification.md#source-section-30 ) | 结果不能写成普遍替代 SFT |

## 核证范围

核对摘要目标、Dataset and Models、主结果与 Limitations；明确原文规模与倍率表述的内部不一致。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

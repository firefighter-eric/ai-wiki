---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model

## TL;DR（快速导读）

ORPO 把示范学习和偏好学习放进同一训练目标，并尝试省去单独的参考模型。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

同一问题的优选回答既作为学习目标，也与较差回答形成偏好信号；两部分如何平衡会影响行为。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model.pdf
- 全文文本：../../raw/text/Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model.md
- 作者：Jiwoo Hong et al.
- 年份：2024
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

训练既鼓励模型学习较好的回答，也用优势比偏好项区分好坏回答。它减少了部分流程和资源要求，但仍依赖成对偏好数据；单阶段设计是否更合适，要看数据、稳定性和任务效果。

## 关键事实

- **C1**：ORPO 将 chosen 的NLL与 chosen/rejected log-odds-ratio惩罚合并在单阶段。
- **C2**：不需要 reference model，也不需要先独立做SFT再偏好对齐。
- **C3**：作者在125M–7B及HH-RLHF/UltraFeedback上验证，报告AlpacaEval等结果。

## 争议与不确定点

- 报告将超过7B、更多算法和更多领域比较列为后续工作。
- 排行榜结果含自动评审器，不能直接等同全面人类偏好。

## 关联页面

- 主题：[LLM RL](../../wiki/topics/LLM%20RL.md)
- 概念：[ORPO](../../wiki/concepts/ORPO.md)
- 概念：[DPO](../../wiki/concepts/DPO.md)

## 这里的术语是什么意思

- **SFT**：监督微调：用输入与参考输出继续训练已有模型。
- **RLHF**：人类反馈强化学习：把人类偏好转成奖励，并用它调整模型行为。
- **DPO**：直接偏好优化：用成对偏好数据直接调整模型概率，简化部分奖励训练流程。

## 方法与实验解读

ORPO 用概率的 odds 来比较回答，而不是相对参考模型的 log-ratio。chosen 的监督项维持任务学习，rejected 惩罚提供方向性偏好，两个目标一起训练。reference-free 减少模型副本的需求，但并不取消数据质量、长度归一化和偏好分布问题。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Hong%20et%20al.%20-%202024%20-%20ORPO%20Monolithic%20Preference%20Optimization%20without%20Reference%20Model.md#source-section-3 ) | 仍使用成对偏好数据。 |
| C2 | [原文]( ../../raw/text/Hong%20et%20al.%20-%202024%20-%20ORPO%20Monolithic%20Preference%20Optimization%20without%20Reference%20Model.md#source-section-3 ) | 训练目标内部仍保留监督学习项。 |
| C3 | [原文]( ../../raw/text/Hong%20et%20al.%20-%202024%20-%20ORPO%20Monolithic%20Preference%20Optimization%20without%20Reference%20Model.md#source-section-2 ) | 资源节省与质量结论限所测范围。 |

## 核证范围

核读联合目标、数据集、评测与Limitations。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

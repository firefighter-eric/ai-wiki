---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# DeepSeek-R1：奖励驱动推理与多阶段训练（2025）

## TL;DR（快速导读）

DeepSeek-R1 区分纯强化学习探索的 R1-Zero 与加入冷启动、多阶段训练的 R1，并将推理能力蒸馏到较小模型。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

R1-Zero 用强化学习激励推理行为，但作者也报告可读性与语言混杂问题。R1 加入冷启动数据和多阶段流程，尝试兼顾推理与可用性。比较时要分清模型版本、训练阶段和测试设置；旧归档名中的 2024 年存在误标。

## 具体怎么理解

会解题与能清楚、稳定地表达解题过程是不同目标；训练流程可能需要同时处理这两项。

## 关键事实

- **C1**：R1-Zero 在 DeepSeek-V3-Base 上直接进行 GRPO 强化学习，不先进行冷启动 SFT；GRPO 通过同一道题的一组输出奖励估计相对优势，省去单独的 critic。
- **C2**：R1 使用少量长 CoT 冷启动、推理 RL、拒绝采样与 SFT、全场景 RL 的多阶段流程；语言一致性奖励改善可读性，但作者观察到轻微性能代价。
- **C3**：报告默认以温度 0.6、top-p 0.95、最长 32768 tokens 采样多次，并用正确样本比例估计 pass@1；这与单次贪心解码或多数投票指标不同。
- **C4**：作者报告 R1 在函数调用、多轮交互、复杂角色扮演和 JSON 输出上仍弱于 DeepSeek-V3，few-shot 提示在其评测中会降低性能。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.pdf)
- 全文文本：[打开全文文本](../../raw/text/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md)
- 作者：DeepSeek-AI
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.html)
- 归档说明：文件名保留以维持已有链接；本页标题按原文识别内容整理，旧文件名不作为作者或年份依据。
- 归档说明：保留历史文件名以维持来源对应和链接；标题、作者与年份以上述核对信息为准。

## 争议与不确定点

- 长推理消耗更多生成预算；不同温度、采样次数和提示会改变比较结果。
- 模型仍有语言混杂和通用能力短板；软件工程 RL 因评估耗时没有同等充分覆盖。
- 所谓 aha moment 是训练行为观察，不能单凭一个轨迹证明模型具有人类式理解。

## 关联页面

- 主题：[LLM预训练](../topics/LLM%20预训练.md)
- 综合：暂无
- [DeepSeek](../authors/DeepSeek.md)：沿作者或机构继续阅读相关来源。

## 方法与实验解读

可以把训练流程理解为先让模型尝试可验证的解题，再把可读、正确的推理轨迹筛出来用于监督训练，最后兼顾一般使用中的帮助性与安全性。规则奖励适合答案可检查的数学与代码任务；它并不自动覆盖开放写作和真实软件工程。蒸馏实验主要证明教师生成的数据能改善小模型，不能证明学生独立用同等 RL 预算也会达到相同效果。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md#source-section-12 ) | 无冷启动 SFT 不等于没有预训练或没有训练任务；奖励仍决定优化方向。 |
| C2 | [原文]( ../../raw/text/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md#source-section-22 ) | 必须区分 R1 与 R1-Zero，不能把纯 RL 的结论套到整个 R1 流程。 |
| C3 | [原文]( ../../raw/text/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md#source-section-32 ) | 采样次数通常为 4–64，随测试集变化；比较需控制推理预算。 |
| C4 | [原文]( ../../raw/text/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md#source-section-40 ) | 这是该报告版本的边界，不能把推理基准提升概括为全面能力提升。 |

## 核证范围

核对 §2.2 GRPO 与规则奖励、§2.3 多阶段流程、§3 评测设置与表格解释、§5 局限。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Yang et al. - 2026 - Learning beyond Teacher Generalized On-Policy Distillation with Reward Extrapolation

## TL;DR（快速导读）

G-OPD 让学生在自己的回答上学习教师分布，并调整奖励与约束的相对权重，研究更一般的在线蒸馏。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 原始 PDF：[raw/pdf/Yang et al. - 2026 - Learning beyond Teacher Generalized On-Policy Distillation with Reward Extrapolation.pdf](../../raw/pdf/Yang%20et%20al.%20-%202026%20-%20Learning%20beyond%20Teacher%20Generalized%20On-Policy%20Distillation%20with%20Reward%20Extrapolation.pdf)
- 原始 HTML：[raw/html/Yang et al. - 2026 - Learning beyond Teacher Generalized On-Policy Distillation with Reward Extrapolation.html](../../raw/html/Yang%20et%20al.%20-%202026%20-%20Learning%20beyond%20Teacher%20Generalized%20On-Policy%20Distillation%20with%20Reward%20Extrapolation.html)
- 全文文本：[raw/text/Yang et al. - 2026 - Learning beyond Teacher Generalized On-Policy Distillation with Reward Extrapolation.md](../../raw/text/Yang%20et%20al.%20-%202026%20-%20Learning%20beyond%20Teacher%20Generalized%20On-Policy%20Distillation%20with%20Reward%20Extrapolation.md)
- 来源 URL：https://arxiv.org/abs/2602.12125
- 作者：Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, Yankai Lin
- 机构：Renmin University of China; Tencent

## 摘要

论文将在线策略蒸馏解释为带约束的强化学习，并用奖励缩放及不同参考模型扩展目标。它主要在数学和代码任务中实验；能否超越教师取决于设置，不能把局部结果解释成通用能力保证。

### 方法与背景细节

这篇论文把 `OPD`（On-Policy Distillation）放进 LLM 后训练与 reasoning RL 的交界处讨论：学生模型先从自己的当前策略采样轨迹，再在这些 student-generated trajectories 上对齐 teacher 的 logit 分布。作者认为，这种做法同时保留了 on-policy 训练与 token 级密集监督，因此不同于用 teacher-generated trajectories 做 `SFT` 的 off-policy distillation，也不同于只依赖最终正确性或 outcome reward 的常规 RL。

论文的核心贡献不是首次提出 OPD，而是把 OPD 重新解释为一种 **dense KL-constrained RL** 的特殊情形：OPD 中的 teacher / reference log-probability ratio 可以被理解成 token 级隐式 reward，且 reward 项与 KL 正则项在标准 OPD 中固定为同等权重。在此基础上，作者提出 `G-OPD`，通过引入可调 reward scaling factor 与更灵活的 reference model，把标准 OPD 扩展为更一般的后训练目标。

实验主要覆盖数学推理与代码生成。论文提出的 `ExOPD` 使用大于 1 的 reward scaling factor 做 reward extrapolation；在同尺寸多 teacher 合并和 strong-to-weak distillation 场景中，作者报告其相较标准 OPD 有更高表现，并且在某些设置下能让统一 student 超过多个 domain teacher。但论文也承认，reward correction 需要访问 teacher 的 pre-RL base model，会增加计算成本；过强 extrapolation 也可能带来隐式 reward hacking、响应长度膨胀和训练不稳定。

## 关键事实

- **C1**：OPD在student生成轨迹上优化student到teacher的reverseKL。
- **C2**：G-OPD将implicitreward与KL的相对权重用lambda解耦，大于1为extrapolation。
- **C3**：strong-to-weak的rewardcorrection可需teacher的preRLbase作reference。
- **C4**：同尺寸multi-teacher任务有超过各teacher的结果；更大规模/跨家族仍待验证。

## 争议与不确定点

- 需要teacher概率/参考模型的设定与只能调用文本API不同。
- rewardscaling稳定区间和跨家族泛化尚无通用保证。

## 关联页面

- 主题：[LLM RL](../topics/LLM%20RL.md)
- 概念：[OPD](../concepts/OPD.md)
- 概念：[GRPO](../concepts/GRPO.md)
- 概念：[DAPO](../concepts/DAPO.md)
- 概念：[DPO](../concepts/DPO.md)

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **distillation**：蒸馏：利用教师模型提供的答案或分布训练学生模型。
- **SFT**：监督微调：用输入与参考输出继续训练已有模型。

## 方法与实验解读

密集logprob差提供每token反馈，reference决定隐式奖励的含义。超过teacher的结果来自组合/外推及特定任务评估，不违反蒸馏信息约束，也不证明teacher输出都正确。实践要按评测预算和响应长度对齐再比。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Yang%20et%20al.%20-%202026%20-%20Learning%20beyond%20Teacher%20Generalized%20On-Policy%20Distillation%20with%20Reward%20Extrapolation.md#source-section-7 ) | on-policy与teacher离线轨迹不同。 |
| C2 | [原文]( ../../raw/text/Yang%20et%20al.%20-%202026%20-%20Learning%20beyond%20Teacher%20Generalized%20On-Policy%20Distillation%20with%20Reward%20Extrapolation.md#source-section-8 ) | reward放大也可能放大teacher错误。 |
| C3 | [原文]( ../../raw/text/Yang%20et%20al.%20-%202026%20-%20Learning%20beyond%20Teacher%20Generalized%20On-Policy%20Distillation%20with%20Reward%20Extrapolation.md#source-section-9 ) | 额外访问/算力条件。 |
| C4 | [原文]( ../../raw/text/Yang%20et%20al.%20-%202026%20-%20Learning%20beyond%20Teacher%20Generalized%20On-Policy%20Distillation%20with%20Reward%20Extrapolation.md#source-section-19 ) | 不应标题推出所有student必胜teacher。 |

## 核证范围

核读OPD形式、G-OPD/ExOPD、rewardcorrection、实验设置与讨论。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

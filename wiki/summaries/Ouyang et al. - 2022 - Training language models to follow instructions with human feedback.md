---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Ouyang et al. - 2022 - Training language models to follow instructions with human feedback

## TL;DR（快速导读）

InstructGPT 先学习示范，再学习人类偏好并做强化学习，研究让语言模型更符合用户意图。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Ouyang et al. - 2022 - Training language models to follow instructions with human feedback.pdf
- 原始 HTML：../../raw/html/Ouyang et al. - 2022 - Training language models to follow instructions with human feedback.html
- 全文文本：../../raw/text/Ouyang et al. - 2022 - Training language models to follow instructions with human feedback.md
- 作者：Ouyang et al.
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

基础模型的语言能力与指令遵循不是同一件事。论文把示范数据、回答排序、奖励模型与策略更新连接起来；人类偏好只覆盖已收集的判断条件，仍要检查事实可靠性与分布变化。

## 关键事实

- **C1**：InstructGPT先人工示范SFT，再比较数据训练RM，最后用PPO优化策略。
- **C2**：训练prompt混合真实API与标注员写作数据。
- **C3**：1.3B InstructGPT可在人工偏好上优于175B GPT3。
- **C4**：truthfulness与toxicity改善具有任务/提示条件，bias未显著改善，仍会编造事实。

## 争议与不确定点

- 标签偏好、API样本及筛选标注员都有代表性限制。
- RM作为代理可被策略利用，不能把RLHF说成已经解决幻觉或偏见。

## 关联页面

- 概念：[InstructGPT](../../wiki/concepts/InstructGPT.md)
- 概念：[RLHF](../../wiki/concepts/RLHF.md)
- 主题：[AI 智能问答与智能客服](../../wiki/topics/AI%20%E6%99%BA%E8%83%BD%E9%97%AE%E7%AD%94%E4%B8%8E%E6%99%BA%E8%83%BD%E5%AE%A2%E6%9C%8D.md)
- 主题：[指令对齐与 post-training](../../wiki/topics/指令对齐与%20post-training.md)
- [OpenAI](../authors/OpenAI.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **benchmark**：评测基准：特定数据、任务与规则的组合，分数只在这些条件下成立。
- **reward model**：奖励模型：根据训练信号给回答或行为打分，分数是目标的近似。
- **SFT**：监督微调：用输入与参考输出继续训练已有模型。
- **RLHF**：人类反馈强化学习：把人类偏好转成奖励，并用它调整模型行为。

## 方法与实验解读

SFT学期望回复，RM拟合比较偏好，PPO用代理奖励塑形。人工偏好优势揭示参数量与用户意图遵循是两个维度；混入预训练梯度可以缓解alignment tax，但奖励高不保证事实真。对知识问答，需要另有检索证据和引用核验。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md#source-section-18 ) | 三步各有数据与误差来源。 |
| C2 | [原文]( ../../raw/text/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md#source-section-22 ) | 作者采样的API分布，不代表全体用户。 |
| C3 | [原文]( ../../raw/text/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md#source-section-4 ) | 人评交互质量，不是所有知识/推理能力。 |
| C4 | [原文]( ../../raw/text/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md#source-section-40 ) | truthfulness见§4.2前节，错误见§4.3。 |

## 核证范围

核读三步方法、数据分布、模型/评测、API人工偏好、truthfulness/toxicity与局限。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

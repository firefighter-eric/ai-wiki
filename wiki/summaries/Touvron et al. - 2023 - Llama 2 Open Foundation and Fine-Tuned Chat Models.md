---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Touvron et al. - 2023 - Llama 2 Open Foundation and Fine-Tuned Chat Models

## TL;DR（快速导读）

Llama 2 同时提供基础模型与聊天模型，后者通过专门后训练改善指令遵循和对话行为。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

基础模型主要延续文本，对话模型按交互格式响应；比较时需要确认使用的检查点。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Touvron et al. - 2023 - Llama 2 Open Foundation and Fine-Tuned Chat Models.pdf
- 全文文本：../../raw/text/Touvron et al. - 2023 - Llama 2 Open Foundation and Fine-Tuned Chat Models.md
- 作者：Touvron et al.
- 年份：2023
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

报告将基础预训练与聊天对齐分开介绍，涉及示范和人类反馈等训练。选择与比较时应说明使用哪个版本；基础能力、对话风格和安全行为并不是同一个评测目标。

## 关键事实

- **C1**：公开7B/13B/70B的base/chat，2Ttokens预训练。
- **C2**：相对初代增加上下文和GQA，具体attention配置按型号。
- **C3**：Chat通过SFT、偏好RM与RLHF，含rejection sampling/PPO。

## 争议与不确定点

- 开放获取受专门许可，不等同Apache2.0。
- 安全评测不充分覆盖对抗/多轮分布，报告自己说明边界。

## 关联页面

- 概念：[Llama 家族](../../wiki/concepts/Llama%20家族.md)
- 概念：[LLaMA（初代）](../../wiki/concepts/LLaMA%20初代.md)
- 概念：[Llama 2](../../wiki/concepts/Llama%202.md)
- 概念：[Code Llama](../../wiki/concepts/Code%20Llama.md)
- 主题：[LLM预训练](../../wiki/topics/LLM%20预训练.md)
- [Hugo Touvron](../authors/Hugo%20Touvron.md)：沿作者或机构继续阅读相关来源。
- [Meta AI](../authors/Meta%20AI.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **benchmark**：评测基准：特定数据、任务与规则的组合，分数只在这些条件下成立。
- **SFT**：监督微调：用输入与参考输出继续训练已有模型。
- **RLHF**：人类反馈强化学习：把人类偏好转成奖励，并用它调整模型行为。

## 方法与实验解读

基座改善语言分布，后训练改善交互与偏好；helpfulness和safety分别建模避免把两个目标混成一句“更好”。人评近似某闭源模型是特定提示集合的判断，不能替代事实、工具或安全极端输入测试。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Touvron%20et%20al.%20-%202023%20-%20Llama%202%20Open%20Foundation%20and%20Fine-Tuned%20Chat%20Models.md#source-section-5 ) | base与chat评测分离。 |
| C2 | [原文]( ../../raw/text/Touvron%20et%20al.%20-%202023%20-%20Llama%202%20Open%20Foundation%20and%20Fine-Tuned%20Chat%20Models.md#source-section-6 ) | 不能说全型号都GQA。 |
| C3 | [原文]( ../../raw/text/Touvron%20et%20al.%20-%202023%20-%20Llama%202%20Open%20Foundation%20and%20Fine-Tuned%20Chat%20Models.md#source-section-18 ) | 多阶段后训练，PPO见§3.2.3。 |

## 核证范围

核读预训练/型号、SFT、RM、拒绝采样/PPO与人评和安全局限。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

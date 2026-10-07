---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Inan et al. - 2023 - Llama Guard LLM-based Input-Output Safeguard for Human-AI Conversations

## TL;DR（快速导读）

Llama Guard 用独立模型检查用户输入和模型回答的风险，适合放在对话系统的审核环节。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

在生成回答前后检查内容类别，可帮助决定是否继续处理；高分类分数并不保证完整系统不存在风险。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Inan et al. - 2023 - Llama Guard LLM-based Input-Output Safeguard for Human-AI Conversations.pdf
- 原始 HTML：../../raw/html/Inan et al. - 2023 - Llama Guard LLM-based Input-Output Safeguard for Human-AI Conversations.html
- 全文文本：../../raw/text/Inan et al. - 2023 - Llama Guard LLM-based Input-Output Safeguard for Human-AI Conversations.md
- 作者：Inan et al.
- 年份：2023
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

论文把提示与回答分类设为专门任务，并使用可调整的风险类别。它负责判定与护栏，不负责生成客服答案；类别覆盖、误报和漏报都需要按实际业务样本检查。

## 关键事实

- **C1**：LlamaGuard是基于Llama2-7B的输入/输出安全分类器。
- **C2**：输入包含taxonomy/guidelines，允许zero/few-shot改换分类口径。
- **C3**：评测区分on-policy和不同taxonomy的off-policy条件。
- **C4**：OpenAI Moderation/ToxicChat结果来自作者评测。

## 争议与不确定点

- 政策口径变化会改变标签分布，必须检验映射与阈值。
- 安全benchmark不能证明未知对抗请求的全面稳健性。

## 关联页面

- 概念：[Llama Guard](../../wiki/concepts/Llama%20Guard.md)
- 主题：[AI 智能问答与智能客服](../../wiki/topics/AI%20%E6%99%BA%E8%83%BD%E9%97%AE%E7%AD%94%E4%B8%8E%E6%99%BA%E8%83%BD%E5%AE%A2%E6%9C%8D.md)
- 主题：[LLM RL](../../wiki/topics/LLM%20RL.md)

## 这里的术语是什么意思

- **benchmark**：评测基准：特定数据、任务与规则的组合，分数只在这些条件下成立。

## 方法与实验解读

门控模型把用户输入风险与生成输出风险分成两个检查点，并以分类规则约束输出。实际客服需要明确哪些风险分类适用、误拒答和漏报如何计量，再决定前置/后置使用。它不能单独解决知识来源、权限控制或工具执行错误。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Inan%20et%20al.%20-%202023%20-%20Llama%20Guard%20LLM-based%20Input-Output%20Safeguard%20for%20Human-AI%20Conversations.md#source-section-10 ) | 判定模型，不是一般聊天能力增强。 |
| C2 | [原文]( ../../raw/text/Inan%20et%20al.%20-%202023%20-%20Llama%20Guard%20LLM-based%20Input-Output%20Safeguard%20for%20Human-AI%20Conversations.md#source-section-3 ) | 可配置不保证任意政策都准确遵循。 |
| C3 | [原文]( ../../raw/text/Inan%20et%20al.%20-%202023%20-%20Llama%20Guard%20LLM-based%20Input-Output%20Safeguard%20for%20Human-AI%20Conversations.md#source-section-12 ) | 类别映射影响比较的公平性。 |
| C4 | [原文]( ../../raw/text/Inan%20et%20al.%20-%202023%20-%20Llama%20Guard%20LLM-based%20Input-Output%20Safeguard%20for%20Human-AI%20Conversations.md#source-section-18 ) | 基准覆盖有限，不推导零漏报。 |

## 核证范围

核读风险分类、训练、on/off-policy和整体评测。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

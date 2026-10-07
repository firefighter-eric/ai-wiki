---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Qwen Team - 2025 - Qwen3 Think Deeper Act Faster

## TL;DR（快速导读）

Qwen3 的混合思考模式让同一模型在深入推理与快速回答之间切换，思考预算成为使用条件之一。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

简单问答未必需要长推理，复杂题目可能需要更多预算；模式切换应结合延迟和正确率评估。

## 来源信息

- 类型：官方博客 / 技术发布
- 来源链接：https://qwenlm.github.io/blog/qwen3/
- 全文文本：../../raw/text/Qwen Team - 2025 - Qwen3 Think Deeper Act Faster.md
- 作者：Qwen Team
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

发布页同时介绍专家混合规模和代理能力。比较时应注明是否启用思考、允许多少生成预算及使用什么工具；更长思考并不在每个问题上都值得付出额外延迟。

## 关键事实

- **C1**：开放两个MoE235B-A22B/30B-A3B与六个dense尺寸，提供thinking/nonthinking模式。
- **C2**：数据约36Ttokens、119种语言方言，含PDF抽取改进与合成数学/代码数据。
- **C3**：post-training四步为长CoT冷启动、reasoningRL、模式融合与generalRL。
- **C4**：本文未披露预训练优化器名称，应标未披露。

## 争议与不确定点

- 比较大/小模型需同时记录推理长度，否则成本和效果都不公平。
- 优化器以及部分数据配方没有在该博客披露，仍保留未知。

## 关联页面

- 概念：[Qwen3](../../wiki/concepts/Qwen3.md)
- 概念：[Qwen3.5](../../wiki/concepts/Qwen3.5.md)
- 主题：[Qwen 系列](../../wiki/topics/Qwen%20系列.md)
- 对比：[Muon 与 AdamW](../../wiki/comparisons/Muon%20与%20AdamW.md)
- [Qwen Team - Alibaba](../authors/Qwen%20Team%20-%20Alibaba.md)：沿作者或机构继续阅读相关来源。
- [Qwen Team](../authors/Qwen%20Team.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。
- **CoT**：思维链：模型写出的中间推理文本，不能自动视为忠实的内部思考。
- **agentic**：代理执行：模型使用工具并根据结果继续行动，可靠性要看完整流程。
- **post-training**：后训练：在预训练底座上继续调整指令遵循、偏好或其他行为。

## 方法与实验解读

模式融合将推理预算变成可调使用条件，评测需要带thinking设置与输出budget。数据提取与合成改善同时发生，不能把跨代收益单独归因MoE或RL。MCP/agent支持是接口和训练能力线索，端到端工具正确率还需要实际任务证据。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Qwen%20Team%20-%202025%20-%20Qwen3%20Think%20Deeper%20Act%20Faster.md#source-section-1 ) | 参数总量与激活量分别记。 |
| C2 | [原文]( ../../raw/text/Qwen%20Team%20-%202025%20-%20Qwen3%20Think%20Deeper%20Act%20Faster.md#source-section-3 ) | token规模与质量不是同一指标。 |
| C3 | [原文]( ../../raw/text/Qwen%20Team%20-%202025%20-%20Qwen3%20Think%20Deeper%20Act%20Faster.md#source-section-4 ) | generalRL含20余任务，不能写成只数学可验证奖励。 |
| C4 | [原文]( ../../raw/text/Qwen%20Team%20-%202025%20-%20Qwen3%20Think%20Deeper%20Act%20Faster.md#source-section-3 ) | absence只限保存来源，不猜AdamW/Muon。 |

## 核证范围

核读型号表、模式、预训练、后训练与agent用法；全文检索未披露项。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

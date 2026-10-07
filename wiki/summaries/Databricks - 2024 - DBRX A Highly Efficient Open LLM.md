---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# DBRX：Databricks 官方模型发布说明

## TL;DR（快速导读）

DBRX 用更细的专家划分控制每个 token 的计算：总参数 132B、激活约 36B，每次选择 16 个专家中的 4 个。正确来源是 Databricks 官方发布说明，旧附件已确认误配。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

总参数量决定要保存多少权重，激活参数量影响每次计算；二者都不足以单独判断服务速度。

## 来源信息

- 类型：论文 / 技术报告
- 历史误配附件（不作为证据）：../../raw/pdf/Databricks - 2024 - DBRX A Highly Efficient Open LLM.pdf
- 全文文本：../../raw/text/Databricks - 2024 - DBRX A Highly Efficient Open LLM.md
- 作者：Databricks
- 年份：2024
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开有效正文快照](../../raw/html/verified/DBRX-official-release-2026-10-07.html)
- 核对说明：正确来源为 https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm；旧数学论文附件原样封存，排除出本页证据。

## 摘要

这份发布资料介绍 Databricks 的开放语言模型及其稀疏计算路线。比较部署成本时，应分别看总权重存储、每次激活的参数、专家通信和任务质量，而不能把总参数量直接当成每次推理成本。

## 关键事实

- **C1**：DBRX 是 next-token decoder MoE，总参数 132B、激活 36B，16 个专家中选择 4 个。
- **C2**：预训练使用 12T 文本与代码 tokens，上下文最大 32K，并使用数据课程。
- **C3**：结构包含 RoPE、GLU 与 GQA，使用 tiktoken tokenizer。
- **C4**：长文评测中 DBRX 与 Mixtral Instruct 整体接近，GPT-4 Turbo 通常更好；更长输入时仍有明显下降。
- **C5**：发行方报告 MoE 推理与训练效率收益，且明确收益来自结构、数据、优化和分词等共同变化。

## 争议与不确定点

- 论文名和 arXiv 2403.08275 曾误配；旧 PDF/HTML 是 fractional KdV 数值方法论文，不再作为 DBRX 证据。
- 厂商的效率与质量比较依赖所列设备、模型和基准，未在本库独立复跑。

## 关联页面

- 概念：[DBRX](../../wiki/concepts/DBRX.md)
- 概念：[Mixtral](../../wiki/concepts/Mixtral.md)
- 概念：[MoE](../../wiki/concepts/MoE.md)
- 主题：[LLM 预训练](../../wiki/topics/LLM%20预训练.md)

## 这里的术语是什么意思

- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。

## 方法与实验解读

细粒度 MoE 增加可选专家组合，使每 token 只激活部分计算，但完整模型的存储和跨设备路由仍有成本。评测应同时记录总参数、激活参数、吞吐和任务质量。开放权重的发布形式也应与开放训练数据区分；本来源不证明其具备 OLMo 式完整可复现材料。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Databricks%20-%202024%20-%20DBRX%20A%20Highly%20Efficient%20Open%20LLM.md#source-section-2 ) | 2024 年官方发布版。 |
| C2 | [原文]( ../../raw/text/Databricks%20-%202024%20-%20DBRX%20A%20Highly%20Efficient%20Open%20LLM.md#source-section-2 ) | 窗口上限不是长文任务准确率。 |
| C3 | [原文]( ../../raw/text/Databricks%20-%202024%20-%20DBRX%20A%20Highly%20Efficient%20Open%20LLM.md#source-section-2 ) | 本页依据官方技术介绍，不是原误配的 arXiv 附件。 |
| C4 | [原文]( ../../raw/text/Databricks%20-%202024%20-%20DBRX%20A%20Highly%20Efficient%20Open%20LLM.md#source-section-5 ) | KV-Pairs/HotpotQAXL、当时 API 版本与长度条件。 |
| C5 | [原文]( ../../raw/text/Databricks%20-%202024%20-%20DBRX%20A%20Highly%20Efficient%20Open%20LLM.md#source-section-6 ) | 不能把全部效率改善归因于专家数。 |

## 核证范围

核读正确官方发布文的 Architecture、Training Data、Long Context 与 Efficiency，并检查原附件标题确认误配。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

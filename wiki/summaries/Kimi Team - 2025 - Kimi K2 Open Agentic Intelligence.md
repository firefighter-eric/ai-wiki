---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Kimi Team - 2025 - Kimi K2: Open Agentic Intelligence

## TL;DR（快速导读）

Kimi K2 报告使用 MuonClip 限制过大的注意力分数，研究把矩阵优化器扩展到大型专家模型时的稳定性。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

某代模型只提供接口，不代表后续所有型号也如此；先确认具体版本再讨论能力与使用方式。

## 来源信息

- 类型：技术报告 / arXiv 论文
- arXiv：https://arxiv.org/abs/2507.20534
- 原始 PDF：../../raw/pdf/Kimi Team - 2025 - Kimi K2 Open Agentic Intelligence.pdf
- 发布页快照：../../raw/html/Kimi Team - 2025 - Kimi K2 Open Agentic Intelligence.html
- 全文文本：../../raw/text/Kimi Team - 2025 - Kimi K2 Open Agentic Intelligence.md
- 作者：Kimi Team
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

方法在参数更新后，根据批次中各注意力头的最大分数缩放查询和键的投影权重，缓解分数爆炸。它在专家混合模型中使用，阅读时需区分优化器更新、稳定性约束与模型容量各自的影响。

## 关键事实

- **C1**：K2为1T总参数/32B激活MoE，预训练15.5T tokens。
- **C2**：QK-Clip在optimizer更新后，依据forward中per-head最大logit缩放Q/K权重。
- **C3**：超阈值时γ=min(1,τ/Smax)，Q与K各乘sqrt(γ)；MLA只缩放head-specific部分。
- **C4**：主训练用τ=100的MuonClip，作者报告无loss spikes。
- **C5**：算法描述二维矩阵Muon路径；本报告未逐项确认所有AdamW fallback groups。
- **C6**：后训练包含大规模agent合成数据与真实/模拟环境联合RL。

## 争议与不确定点

- 无spike是作者记录的一次训练结果，仍需复现实测。
- fallback optimizer、完整数据和部署细节未逐项公开；不确定保留。

## 关联页面

- 概念：[Muon](../concepts/Muon.md)
- 概念：[Kimi](../concepts/Kimi.md)
- 概念：[Kimi K3](../concepts/Kimi%20K3.md)
- 对比：[Muon 与 AdamW](../comparisons/Muon%20与%20AdamW.md)

## 这里的术语是什么意思

- **SFT**：监督微调：用输入与参考输出继续训练已有模型。
- **embedding**：向量表示：把文字、图片等编码成一组数，用于模型计算或相似度比较。
- **weight decay**：权重衰减：训练中使权重逐步缩小的机制，需看它如何与梯度更新结合。
- **agentic**：代理执行：模型使用工具并根据结果继续行动，可靠性要看完整流程。

## 方法与实验解读

MuonClip用矩阵优化提高token效率，同时以QK约束避免attention极端logits。稳定性措施属于参数更新后的控制，不是标准gradient clipping。数据增强、SFT和环境RL也影响最终能力；工具定义不清时过长生成与截断说明harness与模型共同决定成功率。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Kimi%20Team%20-%202025%20-%20Kimi%20K2%20Open%20Agentic%20Intelligence.md#source-section-2 ) | 总参数、激活参数与训练量不同分母。 |
| C2 | [原文]( ../../raw/text/Kimi%20Team%20-%202025%20-%20Kimi%20K2%20Open%20Agentic%20Intelligence.md#source-section-7 ) | 不改当前step的forward/backward。 |
| C3 | [原文]( ../../raw/text/Kimi%20Team%20-%202025%20-%20Kimi%20K2%20Open%20Agentic%20Intelligence.md#source-section-7 ) | 按head约束，不能粗暴缩放共享latent通道。 |
| C4 | [原文]( ../../raw/text/Kimi%20Team%20-%202025%20-%20Kimi%20K2%20Open%20Agentic%20Intelligence.md#source-section-8 ) | 特定训练轨迹，不证明任意配置稳定。 |
| C5 | [原文]( ../../raw/text/Kimi%20Team%20-%202025%20-%20Kimi%20K2%20Open%20Agentic%20Intelligence.md#source-section-8 ) | 不把Moonlight配置未经确认套入K2。 |
| C6 | [原文]( ../../raw/text/Kimi%20Team%20-%202025%20-%20Kimi%20K2%20Open%20Agentic%20Intelligence.md#source-section-2 ) | agent表现不能单独归因Muon。 |

## 核证范围

核读模型范围、§2QK-Clip/MuonClip、agent后训练与局限。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

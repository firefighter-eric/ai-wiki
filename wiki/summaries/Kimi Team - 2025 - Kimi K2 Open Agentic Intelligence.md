---
type: summary
status: refined
---
# Kimi Team - 2025 - Kimi K2: Open Agentic Intelligence

## 来源信息

- 类型：技术报告 / arXiv 论文
- arXiv：https://arxiv.org/abs/2507.20534
- 原始 PDF：../../raw/pdf/Kimi Team - 2025 - Kimi K2 Open Agentic Intelligence.pdf
- 发布页快照：../../raw/html/Kimi Team - 2025 - Kimi K2 Open Agentic Intelligence.html
- 全文文本：../../raw/text/Kimi Team - 2025 - Kimi K2 Open Agentic Intelligence.md
- 作者：Kimi Team
- 年份：2025
- 状态：已精读 MuonClip、训练 recipe 与模型规模部分

## 摘要

`Kimi K2` 是总参数约 1T、每 token 激活 32B 的 MoE。它把 Moonlight 版 Muon 的 weight decay 与 consistent update RMS 扩展成 `MuonClip`：在每次参数更新后，根据实际 batch 中每个 attention head 的最大 logit，按 head 缩放 Q/K projection weights，从而缓解 Muon 扩展时出现的 attention-logit explosion。

## 关键事实

- QK-Clip 读取 forward 已计算的 per-head 最大 attention logit `S_max^h`；当它超过阈值 `τ` 时，令 `γ_h=min(1, τ/S_max^h)`，再把 head-specific Q/K 权重各乘 `sqrt(γ_h)`。
- 该操作发生在 optimizer update 后，不改变当前 step 的 forward/backward；它约束下一步产生的 QK 点积。
- K2 使用 `τ=100`、weight decay `0.1`、MuonClip 和 WSD learning-rate schedule，预训练 15.5T tokens；作者报告全程无 loss spike。
- 论文把 QK-Clip 做成 per-head，是因为实验中只有少数 heads 出现极端 logits；对 MLA 只缩放未共享的 head-specific components。
- K2 的 Algorithm 1 只定义对二维 weight matrices 的 Muon update，并通过引用 Moonlight 继承 consistent update RMS recipe；Moonlight 明确把 RMSNorm、LM head 与 embedding 交给 AdamW。但 K2 报告自身没有逐项重申这组 AdamW parameter groups。
- K2 的 SFT / RL 阶段也披露使用 Muon，但论文没有给出足以把最终 agentic 能力单独归因给优化器的消融。

## 争议与不确定点

- “零 loss spike”是单次超大训练 run 的作者报告，不代表所有 MuonClip 配置天然稳定。
- QK-Clip 修复的是该模型/attention 设计下观察到的 logit instability，不应视作每个 Muon 模型都必须使用的组成部分；DeepSeek-V4 通过 Q/KV normalization 选择不使用它。
- 因此可以高置信判断 K2 不是“所有参数都由 Muon 更新”，但把 `embedding / LM head / RMSNorm -> AdamW` 写成 K2 独立披露会过度陈述；更准确的标记是“沿用 Moonlight 混合 recipe 的强推断，K2 未公布精确 parameter-group 配置或 AdamW betas/epsilon”。
- K2 的能力同时来自模型规模、MoE/MLA 架构、数据和 post-training，不能由 MuonClip 一项解释。

## 关联页面

- 概念：[Muon](../concepts/Muon.md)
- 概念：[Kimi](../concepts/Kimi.md)
- 概念：[Kimi K3](../concepts/Kimi%20K3.md)
- 对比：[Muon 与 AdamW](../comparisons/Muon%20与%20AdamW.md)

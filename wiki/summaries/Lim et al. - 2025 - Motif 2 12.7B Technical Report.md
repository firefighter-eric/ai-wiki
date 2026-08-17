---
type: summary
status: refined
---
# Lim et al. - 2025 - Motif-2-12.7B Technical Report

## 来源信息

- 类型：技术报告 / arXiv 论文
- arXiv：https://arxiv.org/abs/2511.07464
- 原始 PDF：../../raw/pdf/Lim et al. - 2025 - Motif 2 12.7B Technical Report.pdf
- 发布页快照：../../raw/html/Lim et al. - 2025 - Motif 2 12.7B Technical Report.html
- 全文文本：../../raw/text/Lim et al. - 2025 - Motif 2 12.7B Technical Report.md
- 作者：Motif Technologies / Lim 等
- 年份：2025
- 状态：已精读 MuonClip 与 Parallel Muon 部分

## 摘要

`Motif-2-12.7B` 在 5.5T-token 预训练中使用 MuonClip，并实现 `Parallel Muon` 处理 Newton–Schulz 的分布式瓶颈。它不让所有 ranks all-gather 后重复正交化全部矩阵，而用 all-to-all 把完整矩阵分配给不同 ranks 并行计算，再 scatter 回原有 shards。

## 关键事实

- 预训练有效 batch 从 16M tokens 扩到 80M tokens，优化器为 MuonClip；模型、数据调度与 Grouped Differential Attention 同时变化。
- Parallel Muon 的流程是 `all-to-all gather -> 每个 rank 处理分配到的完整矩阵 -> Newton–Schulz -> all-to-all scatter`。
- 实现按 Newton–Schulz FLOPs 对矩阵排序并 round-robin 分配，缓解 ranks 间负载不平衡；还用 chunked pipeline 重叠 gather、计算与 scatter。
- 论文给出的 8×H200 单节点吞吐对比显示 Parallel Muon 明显快于其 Distributed Muon baseline；这是特定模型、分片和硬件下的 optimizer-kernel 测量，不是端到端模型训练提速倍数。
- 该来源证明 Muon 的工程扩散不只发生在 Moonshot，也推动了 all-to-all、混合并行和专用 kernel 的新实现。

## 争议与不确定点

- 模型质量变化不能单独归因给 MuonClip 或 Parallel Muon，因为架构、数据、batch、精度和训练系统均有变化。
- 单节点 benchmark 不能直接外推到更大集群；all-to-all 的收益高度依赖网络拓扑、矩阵形状、chunk size 与负载平衡。
- 报告由模型开发者发布，性能数字仍需独立复现。

## 关联页面

- 概念：[Muon](../concepts/Muon.md)
- 对比：[Muon 与 AdamW](../comparisons/Muon%20与%20AdamW.md)
- 主题：[LLM 预训练](../topics/LLM%20预训练.md)

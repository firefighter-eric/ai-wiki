# Liu et al. - 2025 - Muon is Scalable for LLM Training

## 来源信息

- 类型：技术报告 / arXiv 论文
- arXiv：https://arxiv.org/abs/2502.16982
- 原始 PDF：../../raw/pdf/Liu et al. - 2025 - Muon is Scalable for LLM Training.pdf
- 发布页快照：../../raw/html/Liu et al. - 2025 - Muon is Scalable for LLM Training.html
- 全文文本：../../raw/text/Liu et al. - 2025 - Muon is Scalable for LLM Training.md
- 作者：Kimi Team / Jingyuan Liu 等
- 年份：2025
- 状态：已精读 Muon scale-up、scaling law、SFT 与分布式实现部分

## 摘要

该报告把原始 Muon 扩展到大规模 LLM 预训练，识别出两个关键条件：对 Muon 参数加入 weight decay，以及根据矩阵形状控制每个参数的 update RMS。基于这一 recipe，团队以 Muon 训练 3B activated / 16B total 的 MoE `Moonlight`，总计 5.7T tokens，并给出与调优 AdamW baseline 的 scaling-law 对比。

## 关键事实

- 不做 scale control 时，semi-orthogonal update 的自然 RMS 随矩阵形状变化，导致不同形状参数拥有不一致的有效更新尺度。
- 作者将 Muon update RMS 统一重标定，并用 weight decay 控制训练后期的权重增长；这两点使 AdamW 的部分超参数更容易复用。
- 报告明确说明实际训练采用 Muon/AdamW 混合分组：matrix-based hidden parameters 使用 Muon，`RMSNorm`、`LM head` 与 embedding parameters 由 AdamW 处理。因此 Moonlight 所称“使用 Muon”并不表示全参数都走 Newton–Schulz。
- 作者拟合的 compute-optimal scaling law 显示，Muon 达到 AdamW 可比 loss 约需 `52%` 的训练 FLOPs，即论文所称约 `2×` compute efficiency。
- 这一 `52%` 是特定模型族、数据、超参数搜索和 loss 拟合下的作者实验结论，不是“单步快 2 倍”，也不是普适定律。
- Distributed Muon 在 ZeRO-1 风格分片后先更新本地 momentum，再收集完整矩阵做 Newton–Schulz，只保留本 rank 对应 update shard。
- 论文的配置中，Muon 参数只保存一个 momentum buffer，作者称其额外 optimizer-state memory 为分布式 AdamW 的一半；通信工作量则略高于 AdamW。
- SFT 消融显示：Muon 预训练且 Muon 微调的 Moonlight 较强，但把 AdamW 预训练 checkpoint 改用 Muon 微调没有显示稳定优势；公开 Qwen2.5-7B 上 Muon-SFT 与 Adam-SFT 大致相当。

## 争议与不确定点

- scaling-law 与 Moonlight 结果来自提出者团队，仍需要不同架构、数据与独立团队复现。
- 更高 token / compute efficiency 不等于更低每步 wall-clock；分布式矩阵收集、Newton–Schulz 与硬件利用率会影响端到端成本。
- 论文没有证明 Muon 是所有微调、持续训练或 RL 场景的 drop-in replacement，且明确暴露 pretrain–finetune optimizer mismatch。

## 关联页面

- 概念：[Muon](../concepts/Muon.md)
- 对比：[Muon 与 AdamW](../comparisons/Muon%20与%20AdamW.md)
- 概念：[Kimi](../concepts/Kimi.md)
- 主题：[LLM 预训练](../topics/LLM%20预训练.md)

---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Liu et al. - 2025 - Muon is Scalable for LLM Training

## TL;DR（快速导读）

Moonlight 报告研究把 Muon 用于大规模语言模型训练，强调权重衰减和随矩阵形状调整更新尺度。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

一张权重矩阵包含多个方向；Muon 处理整体矩阵，而 AdamW 主要按元素自适应调整。效率需按完整训练比较。

## 来源信息

- 类型：技术报告 / arXiv 论文
- arXiv：https://arxiv.org/abs/2502.16982
- 原始 PDF：../../raw/pdf/Liu et al. - 2025 - Muon is Scalable for LLM Training.pdf
- 发布页快照：../../raw/html/Liu et al. - 2025 - Muon is Scalable for LLM Training.html
- 全文文本：../../raw/text/Liu et al. - 2025 - Muon is Scalable for LLM Training.md
- 作者：Kimi Team / Jingyuan Liu 等
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

作者将这些条件组成训练配方，并与调优的 AdamW 对照。它说明优化器效果依赖具体实现和缩放规则；比较时应核对同样的模型、数据、训练预算及实际耗时。

## 关键事实

- **C1**：不做 scale control 时，semi-orthogonal update 的自然 RMS 随矩阵形状变化，导致不同形状参数拥有不一致的有效更新尺度。
- **C2**：作者将 Muon update RMS 统一重标定，并用 weight decay 控制训练后期的权重增长；这两点使 AdamW 的部分超参数更容易复用。
- **C3**：报告明确说明实际训练采用 Muon/AdamW 混合分组：matrix-based hidden parameters 使用 Muon，`RMSNorm`、`LM head` 与 embedding parameters 由 AdamW 处理。因此 Moonlight 所称“使用 Muon”并不表示全参数都走 Newton–Schulz。
- **C4**：作者拟合的 compute-optimal scaling law 显示，Muon 达到 AdamW 可比 loss 约需 `52%` 的训练 FLOPs，即论文所称约 `2×` compute efficiency。
- **C5**：这一 `52%` 是特定模型族、数据、超参数搜索和 loss 拟合下的作者实验结论，不是“单步快 2 倍”，也不是普适定律。
- **C6**：Distributed Muon 在 ZeRO-1 风格分片后先更新本地 momentum，再收集完整矩阵做 Newton–Schulz，只保留本 rank 对应 update shard。
- **C7**：论文的配置中，Muon 参数只保存一个 momentum buffer，作者称其额外 optimizer-state memory 为分布式 AdamW 的一半；通信工作量则略高于 AdamW。
- **C8**：SFT 消融显示：Muon 预训练且 Muon 微调的 Moonlight 较强，但把 AdamW 预训练 checkpoint 改用 Muon 微调没有显示稳定优势；公开 Qwen2.5-7B 上 Muon-SFT 与 Adam-SFT 大致相当。

## 争议与不确定点

- 52%训练FLOPs仅适用于作者模型族与拟合，不是普适规律。
- 从AdamW底座换Muon做SFT并未呈现稳定优势，训练来源是重要条件。

## 关联页面

- 概念：[Muon](../concepts/Muon.md)
- 对比：[Muon 与 AdamW](../comparisons/Muon%20与%20AdamW.md)
- 概念：[Kimi](../concepts/Kimi.md)
- 主题：[LLM 预训练](../topics/LLM%20预训练.md)

## 这里的术语是什么意思

- **SFT**：监督微调：用输入与参考输出继续训练已有模型。
- **embedding**：向量表示：把文字、图片等编码成一组数，用于模型计算或相似度比较。
- **FLOPs**：浮点运算量：描述计算数量，不能直接等同于实际耗时。
- **momentum**：动量：用历史梯度的累积信息平滑和组织参数更新。
- **weight decay**：权重衰减：训练中使权重逐步缩小的机制，需看它如何与梯度更新结合。
- **Newton–Schulz**：Newton–Schulz 迭代：用重复矩阵运算近似目标矩阵变换，迭代次数影响成本与近似。

## 方法与实验解读

Muon可扩展需要把更新RMS、衰减、parameter groups与分布式矩阵处理一起设计。Moonlight的实验通过loss scaling比较整体计算效率，另外用状态内存和通信评估工程成本。其混合AdamW配置是已披露事实，不能用来填补K3等别的报告没有说明的参数分组。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Liu%20et%20al.%20-%202025%20-%20Muon%20is%20Scalable%20for%20LLM%20Training.md#source-section-11 ) | 矩阵形状影响semi-orthogonal update RMS。 |
| C2 | [原文]( ../../raw/text/Liu%20et%20al.%20-%202025%20-%20Muon%20is%20Scalable%20for%20LLM%20Training.md#source-section-13 ) | scale matching与weight decay共同作用。 |
| C3 | [原文]( ../../raw/text/Liu%20et%20al.%20-%202025%20-%20Muon%20is%20Scalable%20for%20LLM%20Training.md#source-section-13 ) | embedding/head物理上可为矩阵，但本配方明确排除出Muon。 |
| C4 | [原文]( ../../raw/text/Liu%20et%20al.%20-%202025%20-%20Muon%20is%20Scalable%20for%20LLM%20Training.md#source-section-3 ) | 作者compute-optimal loss拟合，不是单步吞吐。 |
| C5 | [原文]( ../../raw/text/Liu%20et%20al.%20-%202025%20-%20Muon%20is%20Scalable%20for%20LLM%20Training.md#source-section-3 ) | 模型族、数据和超参搜索条件限定。 |
| C6 | [原文]( ../../raw/text/Liu%20et%20al.%20-%202025%20-%20Muon%20is%20Scalable%20for%20LLM%20Training.md#source-section-17 ) | 完整矩阵NS后只留本rank更新分片。 |
| C7 | [原文]( ../../raw/text/Liu%20et%20al.%20-%202025%20-%20Muon%20is%20Scalable%20for%20LLM%20Training.md#source-section-18 ) | 仅optimizer-state，未计临时空间和权重/梯度。 |
| C8 | [原文]( ../../raw/text/Liu%20et%20al.%20-%202025%20-%20Muon%20is%20Scalable%20for%20LLM%20Training.md#source-section-31 ) | SFT初始化来源与optimizer交互，不能泛化。 |

## 核证范围

保留原有8项优化器分析，核读更新尺度、衰减、parameter groups、distributed方案、scaling和SFT对照。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

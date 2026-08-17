---
type: summary
status: refined
---
# Keller Jordan - 2024 - Muon: An Optimizer for Hidden Layers in Neural Networks

## 来源信息

- 类型：作者技术说明 / 官方实现入口
- 来源：https://kellerjordan.github.io/posts/muon/
- 原始页面：../../raw/html/Keller Jordan - 2024 - Muon An Optimizer for Hidden Layers in Neural Networks.html
- 全文文本：../../raw/text/Keller Jordan - 2024 - Muon An Optimizer for Hidden Layers in Neural Networks.md
- 作者：Keller Jordan 及 Muon contributors
- 年份：2024；页面后续持续修订
- 状态：已精读定义、Newton–Schulz、适用范围与运行成本部分

## 摘要

`Muon` 全称 `MomentUm Orthogonalized by Newton–Schulz`。它先生成 SGD momentum / Nesterov momentum 更新，再把每个二维隐藏层参数的更新矩阵送入若干轮 Newton–Schulz 多项式迭代，近似取其 polar factor `UV^T`。这一步把更新矩阵的非零奇异值推向相近尺度，目标是避免少数强奇异方向长期支配学习。

## 关键事实

- 若动量矩阵的 SVD 为 `UΣV^T`，Muon 的正交化目标近似为 `UV^T`；被正交化的是 update / momentum，不是模型权重本身。
- 实现先用 Frobenius norm 归一化矩阵，再以 `a=3.4445, b=-4.7750, c=2.0315` 做通常 5 轮 Newton–Schulz 多项式迭代。
- Newton–Schulz 主要由矩阵乘法组成，可在 BF16 中运行；作者选择它而不是直接 SVD，是因为后者过慢。
- Muon 主要用于隐藏层二维矩阵。标量、向量、embedding 与最终 classifier / prediction head 默认仍交给 AdamW。
- Transformer 中把 Q、K、V 分开处理优于把 fused QKV 当成一张矩阵；Nesterov momentum 是公开实现默认值。
- 核心持久状态与 SGD momentum 相同，通常每个 Muon 参数只需一个 momentum buffer；但正交化仍需要临时工作空间。

## 争议与不确定点

- “稀有方向被放大因而更好学习”是作者基于高条件数更新的经验解释，不是完整因果证明。
- 原始结果首先来自 NanoGPT 竞赛和小到中等模型；大规模有效性需要后续 Moonlight、Kimi 与 DeepSeek 等来源支撑。
- Muon 不是 Hessian-based second-order optimizer；把矩阵更新做正交化也不等于计算完整协方差或 Kronecker preconditioner。

## 关联页面

- 概念：[Muon](../concepts/Muon.md)
- 对比：[Muon 与 AdamW](../comparisons/Muon%20与%20AdamW.md)
- 主题：[LLM 预训练](../topics/LLM%20预训练.md)

---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Keller Jordan - 2024 - Muon: An Optimizer for Hidden Layers in Neural Networks

## TL;DR（快速导读）

Muon 对矩阵参数的动量更新做近似正交化，尝试让不同更新方向获得更均衡的尺度。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

一张权重矩阵包含多个方向；Muon 处理整体矩阵，而 AdamW 主要按元素自适应调整。效率需按完整训练比较。

## 来源信息

- 类型：作者技术说明 / 官方实现入口
- 来源：https://kellerjordan.github.io/posts/muon/
- 原始页面：../../raw/html/Keller Jordan - 2024 - Muon An Optimizer for Hidden Layers in Neural Networks.html
- 全文文本：../../raw/text/Keller Jordan - 2024 - Muon An Optimizer for Hidden Layers in Neural Networks.md
- 作者：Keller Jordan 及 Muon contributors
- 年份：2024；页面后续持续修订
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

它先产生动量，再用 Newton–Schulz 迭代近似矩阵变换，使非零奇异方向的更新尺度更接近。它利用二维参数的结构；迭代开销、矩阵形状及其他参数使用什么优化器，都属于比较条件。

## 关键事实

- **C1**：Muon正交化的是梯度/动量更新；若矩阵分解为UΣVᵀ，其目标方向为UVᵀ。
- **C2**：示例Newton–Schulz先按norm归一化，BF16执行5轮，系数3.4445/-4.7750/2.0315。
- **C3**：隐藏二维矩阵用Muon；标量、向量及输入/输出层建议用AdamW等标准优化器。
- **C4**：与Orthogonal-SGDM区别包括先动量后正交化和用NS代替SVD。

## 争议与不确定点

- 速度记录依赖调参、硬件与任务，作者也强调充分调优基线的重要性。
- embedding/head的处理因模型而异，不能用此博客填补其他报告未披露的配置。

## 关联页面

- 概念：[Muon](../concepts/Muon.md)
- 对比：[Muon 与 AdamW](../comparisons/Muon%20与%20AdamW.md)
- 主题：[LLM 预训练](../topics/LLM%20预训练.md)

## 这里的术语是什么意思

- **embedding**：向量表示：把文字、图片等编码成一组数，用于模型计算或相似度比较。
- **momentum**：动量：用历史梯度的累积信息平滑和组织参数更新。
- **Newton–Schulz**：Newton–Schulz 迭代：用重复矩阵运算近似目标矩阵变换，迭代次数影响成本与近似。

## 方法与实验解读

Muon在矩阵整体上调整更新方向，与Adam逐元素缩放不同。有限轮NS用矩阵乘法近似目标，降低SVD开销但不保证每个奇异值精确变为1。通常动量只需一个持久buffer，正交化仍需临时张量；本页的状态内存说明来自算法实现分析。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Keller%20Jordan%20-%202024%20-%20Muon%20An%20Optimizer%20for%20Hidden%20Layers%20in%20Neural%20Networks.md#source-section-11 ) | 不是把模型权重改成正交矩阵。 |
| C2 | [原文]( ../../raw/text/Keller%20Jordan%20-%202024%20-%20Muon%20An%20Optimizer%20for%20Hidden%20Layers%20in%20Neural%20Networks.md#source-section-2 ) | 示例实现，不是所有后续Muon变体的固定值。 |
| C3 | [原文]( ../../raw/text/Keller%20Jordan%20-%202024%20-%20Muon%20An%20Optimizer%20for%20Hidden%20Layers%20in%20Neural%20Networks.md#source-section-2 ) | 后续模型是否沿用这些parameter groups需分别核对。 |
| C4 | [原文]( ../../raw/text/Keller%20Jordan%20-%202024%20-%20Muon%20An%20Optimizer%20for%20Hidden%20Layers%20in%20Neural%20Networks.md#source-section-12 ) | 效率与训练收益分别评估。 |

## 核证范围

核读Definition代码、parameter groups、与Shampoo/Orthogonal-SGDM关系及证据讨论。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

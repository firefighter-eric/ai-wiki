---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Kingma and Ba - 2015 - Adam: A Method for Stochastic Optimization

## TL;DR（快速导读）

Adam 记录梯度的平均趋势与平方大小，为每个参数调整更新尺度；它是一阶优化器。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

持续变化幅度较大的参数与较小的参数可获得不同尺度的更新；它并不是直接计算 Hessian 的二阶法。

## 来源信息

- 类型：ICLR 2015 论文
- arXiv：https://arxiv.org/abs/1412.6980
- 原始 PDF：../../raw/pdf/Kingma and Ba - 2015 - Adam A Method for Stochastic Optimization.pdf
- 发布页快照：../../raw/html/Kingma and Ba - 2015 - Adam A Method for Stochastic Optimization.html
- 全文文本：../../raw/text/Kingma and Ba - 2015 - Adam A Method for Stochastic Optimization.md
- 作者：Diederik P. Kingma、Jimmy Ba
- 年份：2015
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

一阶矩积累更新方向，二阶矩估计梯度幅度，再经过偏差校正进行逐元素归一化。这里的二阶矩不是 Hessian：Adam 不直接计算损失曲率，也不按完整矩阵结构进行更新。

## 关键事实

- **C1**：Adam保存梯度一阶矩与平方梯度二阶矩，以指数衰减更新。
- **C2**：bias correction除以1-β^t，校正零初始化的早期偏差。
- **C3**：默认β1=0.9、β2=0.999、ε=1e-8，方向为m_hat/(sqrt(v_hat)+ε)。
- **C4**：两个moment张量构成主要持久optimizer状态。

## 争议与不确定点

- 论文中的理论条件不等于任意非凸训练全局收敛。
- Adam与decoupled weight-decay的AdamW不是相同算法，本页不混写。

## 关联页面

- 概念：[Muon](../concepts/Muon.md)
- 对比：[Muon 与 AdamW](../comparisons/Muon%20与%20AdamW.md)
- 主题：[LLM 预训练](../topics/LLM%20预训练.md)

## 方法与实验解读

Adam在每个参数元素上按历史梯度尺度调节步长，使用一阶梯度而非二阶求导。理论部分在online convex条件下分析，神经网络实验属于经验结果。比较Muon时要区分逐元素与矩阵方向处理，并说明内存统计是否包含权重和梯度。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/pdf/Kingma%20and%20Ba%20-%202015%20-%20Adam%20A%20Method%20for%20Stochastic%20Optimization.pdf#page=2 ) | 二阶矩不是Hessian。 |
| C2 | [原文]( ../../raw/pdf/Kingma%20and%20Ba%20-%202015%20-%20Adam%20A%20Method%20for%20Stochastic%20Optimization.pdf#page=2 ) | 初始化与衰减参数条件。 |
| C3 | [原文]( ../../raw/pdf/Kingma%20and%20Ba%20-%202015%20-%20Adam%20A%20Method%20for%20Stochastic%20Optimization.pdf#page=2 ) | 原论文的推荐默认值，不是所有问题最优。 |
| C4 | [原文]( ../../raw/pdf/Kingma%20and%20Ba%20-%202015%20-%20Adam%20A%20Method%20for%20Stochastic%20Optimization.pdf#page=2 ) | 未计主权重、梯度和混合精度master copy。 |

## 核证范围

核读第1–4页算法、bias correction与理论条件。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

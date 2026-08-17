---
type: summary
status: refined
---
# Kingma and Ba - 2015 - Adam: A Method for Stochastic Optimization

## 来源信息

- 类型：ICLR 2015 论文
- arXiv：https://arxiv.org/abs/1412.6980
- 原始 PDF：../../raw/pdf/Kingma and Ba - 2015 - Adam A Method for Stochastic Optimization.pdf
- 发布页快照：../../raw/html/Kingma and Ba - 2015 - Adam A Method for Stochastic Optimization.html
- 全文文本：../../raw/text/Kingma and Ba - 2015 - Adam A Method for Stochastic Optimization.md
- 作者：Diederik P. Kingma、Jimmy Ba
- 年份：2015
- 状态：已精读优化器定义与算法部分；全文由 PDF 重建

## 摘要

`Adam` 是逐元素自适应一阶优化器：它同时维护梯度的一阶矩估计与未中心化二阶矩估计，经过 bias correction 后，用二阶矩的平方根对一阶矩逐元素归一化。它结合了 momentum 与按坐标自适应学习率，但不显式利用二维权重的矩阵结构，也不计算 Hessian。

## 关键事实

- 一阶矩 `m_t` 追踪梯度均值，二阶矩 `v_t` 追踪梯度平方；默认参数为 `β1=0.9`、`β2=0.999`、`ε=10^-8`。
- 更新核心是 `m_hat / (sqrt(v_hat) + ε)`，因此不同参数元素根据自己的历史梯度尺度获得不同有效步长。
- bias correction 用于修正一、二阶矩从零初始化带来的早期偏差。
- Adam 是 first-order 方法；使用梯度的二阶矩不等于使用目标函数的二阶导数。
- 对一个参数通常需要保存一阶矩和二阶矩两个持久状态张量，这也是其优化器状态内存的重要来源。

## 争议与不确定点

- 原始 Adam 论文不包含后来提出的 decoupled weight decay；现代 LLM 论文所写的 `AdamW` 不能直接与原始 Adam 完全等同。
- Adam 的通用性和工程成熟度很高，但逐元素归一化不会直接校正一个权重矩阵不同奇异方向之间的尺度差异。

## 关联页面

- 概念：[Muon](../concepts/Muon.md)
- 对比：[Muon 与 AdamW](../comparisons/Muon%20与%20AdamW.md)
- 主题：[LLM 预训练](../topics/LLM%20预训练.md)

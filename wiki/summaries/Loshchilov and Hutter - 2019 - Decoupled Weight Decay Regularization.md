---
type: summary
status: refined
---
# Loshchilov and Hutter - 2019 - Decoupled Weight Decay Regularization

## 来源信息

- 类型：ICLR 2019 论文
- arXiv：https://arxiv.org/abs/1711.05101
- 原始 PDF：../../raw/pdf/Loshchilov and Hutter - 2019 - Decoupled Weight Decay Regularization.pdf
- 发布页快照：../../raw/html/Loshchilov and Hutter - 2019 - Decoupled Weight Decay Regularization.html
- 全文文本：../../raw/text/Loshchilov and Hutter - 2019 - Decoupled Weight Decay Regularization.md
- 作者：Ilya Loshchilov、Frank Hutter
- 年份：2019
- 状态：已精读核心方法与结论

## 摘要

该论文指出，`L2 regularization` 与 weight decay 只在标准 SGD 的特定缩放下等价；对 Adam 这类自适应优化器，两者并不等价。`AdamW` 将参数衰减从 loss gradient 的自适应更新中解耦，使 weight decay 不再被逐元素二阶矩缩放。

## 关键事实

- 把 `λW` 加入梯度再交给 Adam，会让正则项受到自适应分母影响；这不是经典 weight decay 的同一操作。
- AdamW 在参数更新之外直接施加衰减，可概括为 `W <- (1-ηλ)W - η AdamUpdate`。
- 解耦后，学习率与 weight-decay coefficient 的调参关系更清晰。
- 现代大模型语境中的“Adam”常实际指 AdamW，但论文或实现仍应按其明确披露区分。

## 争议与不确定点

- 论文的主要实验对象不是现代超大规模 Transformer；它支撑的是 weight decay 的算法定义，不直接证明某个 LLM 训练 recipe 的最优性。
- 不同框架对 `weight_decay` 的默认参数范围与排除参数组可能不同，不能只看优化器名称判断实际更新。

## 关联页面

- 概念：[Muon](../concepts/Muon.md)
- 对比：[Muon 与 AdamW](../comparisons/Muon%20与%20AdamW.md)
- 主题：[LLM 预训练](../topics/LLM%20预训练.md)

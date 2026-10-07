---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# He, Girshick, Dollar - 2019 - Rethinking imageNet pre-training

## TL;DR（快速导读）

这篇检测研究说明，在合适数据和更长训练下，从随机初始化训练也能得到有竞争力的结果，重新审视 ImageNet 预训练的必要性。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

预训练常用于加快收敛和改善下游任务。作者在检测与分割实验中比较预训练和随机初始化，并增加训练迭代让后者充分收敛。结果约束于模型、数据和预算，不能写成预训练在所有场景都没有价值。

## 具体怎么理解

若两个模型只训练同样短的时间，预训练模型可能占优；延长随机初始化模型的训练后，比较结论可能变化。

## 关键事实

- **C1**：在 COCO 上使用适当归一化和更长训练，随机初始化的 Mask R-CNN 可达到预训练模型相近效果。
- **C2**：ImageNet 预训练主要加快早期收敛，小数据区域仍可能有益。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/He%2C%20Girshick%2C%20Dollar%20-%202019%20-%20Rethinking%20imageNet%20pre-training.pdf)
- 全文文本：[打开全文文本](../../raw/text/He%2C%20Girshick%2C%20Dollar%20-%202019%20-%20Rethinking%20imageNet%20pre-training.md)
- 作者：He, Girshick, Dollar
- 年份：2019
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/He%2C%20Girshick%2C%20Dollar%20-%202019%20-%20Rethinking%20imageNet%20pre-training.html)

## 争议与不确定点

- 额外训练时间本身就是成本，最终 AP 相近不等于计算成本相同。
- 非常少数据、不同任务和不同骨干应分别验证。

## 关联页面

- 主题：[目标检测](../../wiki/topics/目标检测.md)
- 主题：[传统 CV](../../wiki/topics/传统%20CV.md)
- 概念：[Faster R-CNN](../../wiki/concepts/Faster%20R-CNN.md)

## 方法与实验解读

论文通过控制实验区分预训练带来的初始化优势与最终性能上限。比较从头训练时，必须给足收敛时间并处理归一化；否则所谓预训练优势可能只是训练配方不适合随机初始化。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/He%2C%20Girshick%2C%20Dollar%20-%202019%20-%20Rethinking%20imageNet%20pre-training.md#source-section-7 ) | 不是保持相同训练步数即可达成 |
| C2 | [原文]( ../../raw/text/He%2C%20Girshick%2C%20Dollar%20-%202019%20-%20Rethinking%20imageNet%20pre-training.md#source-section-28 ) | 论文的检测与分割设置，不能外推到全部视觉任务 |

## 核证范围

核对 §3 的必要修改、§5.1 的 COCO 协议与 §6 讨论。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

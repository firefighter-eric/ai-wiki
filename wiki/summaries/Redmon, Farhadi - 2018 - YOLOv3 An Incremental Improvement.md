---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Redmon, Farhadi - 2018 - YOLOv3 An Incremental Improvement

## TL;DR（快速导读）

YOLOv3 结合更强特征提取、多尺度预测与目标性判断，研究稳定的实时检测。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Redmon, Farhadi - 2018 - YOLOv3 An Incremental Improvement.pdf
- 原始 HTML：../../raw/html/Redmon, Farhadi - 2018 - YOLOv3 An Incremental Improvement.html
- 全文文本：../../raw/text/Redmon, Farhadi - 2018 - YOLOv3 An Incremental Improvement.md
- 作者：Redmon, Farhadi
- 年份：2018
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

不同尺度上的预测帮助处理尺寸变化，类别判断与框中是否有目标的判断分开。它是多个设计的组合；比较时需看输入分辨率、后处理和相同任务下的精度与速度。

## 关键事实

- **C1**：YOLOv3使用三尺度预测和Darknet53特征提取。
- **C2**：类别采用独立logistic而非互斥softmax，支持多标签。
- **C3**：报告AP50很强但更严格IoU下的表现相对较弱。

## 争议与不确定点

- 速度/精度数字限定TitanX与输入大小。
- 后续YOLO不同作者的架构不能自动继承本代所有假设。

## 关联页面

- 主题：[目标检测](../../wiki/topics/目标检测.md)
- 概念：[YOLO](../../wiki/concepts/YOLO.md)
- [Ali Farhadi](../authors/Ali%20Farhadi.md)：沿作者或机构继续阅读相关来源。
- [Joseph Redmon](../authors/Joseph%20Redmon.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **backbone**：模型骨干：主要负责提取或变换表示，其他任务模块在它之上工作。

## 方法与实验解读

多尺度让较细特征参与小对象检测，residual backbone提高表示能力；类别独立概率改变输出语义。称为incremental improvement意味着路线与配方继续优化，不能把后来的neck/head术语当本文已经定义的唯一标准。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Redmon%2C%20Farhadi%20-%202018%20-%20YOLOv3%20An%20Incremental%20Improvement.md#source-section-7 ) | backbone细节见§2.4。 |
| C2 | [原文]( ../../raw/text/Redmon%2C%20Farhadi%20-%202018%20-%20YOLOv3%20An%20Incremental%20Improvement.md#source-section-6 ) | 类别定义需匹配数据。 |
| C3 | [原文]( ../../raw/text/Redmon%2C%20Farhadi%20-%202018%20-%20YOLOv3%20An%20Incremental%20Improvement.md#source-section-10 ) | AP50与COCO AP不能互换。 |

## 核证范围

核读框预测、多标签、三尺度、Darknet53与结果讨论。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

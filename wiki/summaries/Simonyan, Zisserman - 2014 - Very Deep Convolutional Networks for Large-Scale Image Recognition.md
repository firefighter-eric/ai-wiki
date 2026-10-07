---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Simonyan, Zisserman - 2014 - Very Deep Convolutional Networks for Large-Scale Image Recognition

## TL;DR（快速导读）

VGG 重复堆叠小卷积，系统研究更深网络如何改善图像识别，并提供规则的特征提取结构。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Simonyan, Zisserman - 2014 - Very Deep Convolutional Networks for Large-Scale Image Recognition.pdf
- 原始 HTML：../../raw/html/Simonyan, Zisserman - 2014 - Very Deep Convolutional Networks for Large-Scale Image Recognition.html
- 全文文本：../../raw/text/Simonyan, Zisserman - 2014 - Very Deep Convolutional Networks for Large-Scale Image Recognition.md
- 作者：Simonyan, Zisserman
- 年份：2014
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

小卷积逐层扩大感受范围，结构便于扩展和迁移。论文关注深度与识别效果，但深度也增加计算与存储；它应与后来的残差和轻量设计在各自训练条件下比较。

## 关键事实

- **C1**：VGG以重复3×3卷积构成规则深网，A到E为11至19带权重层。
- **C2**：堆叠小kernel扩大感受野并增加非线性，相同通道下参数少于单大kernel。
- **C3**：ILSVRC2014定位冠军报告top5testerror25.3%。

## 争议与不确定点

- 分类、定位、多尺度/ensemble的数字须按设置分开。
- 更深网络也带来优化和显存问题，不能无条件说深度越大越好。

## 关联页面

- 主题：[经典 CNN 架构](../../wiki/topics/经典%20CNN%20架构.md)
- 主题：[传统 CV](../../wiki/topics/传统%20CV.md)
- 概念：[VGG](../../wiki/concepts/VGG.md)

## 这里的术语是什么意思

- **backbone**：模型骨干：主要负责提取或变换表示，其他任务模块在它之上工作。

## 方法与实验解读

规则化结构让深度成为较清晰的实验变量。有效感受野变大不代表每个位置实际贡献相同，参数量也不代表运行成本；全连接层和高分辨率激活仍很重。它是比较深度、训练与迁移的历史基线。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Simonyan%2C%20Zisserman%20-%202014%20-%20Very%20Deep%20Convolutional%20Networks%20for%20Large-Scale%20Image%20Recognition.md#source-section-6 ) | 统计包括FC，VGG16/19不是16/19个卷积层。 |
| C2 | [原文]( ../../raw/text/Simonyan%2C%20Zisserman%20-%202014%20-%20Very%20Deep%20Convolutional%20Networks%20for%20Large-Scale%20Image%20Recognition.md#source-section-7 ) | 比较忽略边界效应和不同训练因素。 |
| C3 | [原文]( ../../raw/text/Simonyan%2C%20Zisserman%20-%202014%20-%20Very%20Deep%20Convolutional%20Networks%20for%20Large-Scale%20Image%20Recognition.md#source-section-30 ) | 定位与分类任务区分。 |

## 核证范围

核读结构配置、感受野分析、分类/定位与迁移。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

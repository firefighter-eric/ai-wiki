---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Huang et al. - 2016 - Densely Connected Convolutional Networks

## TL;DR（快速导读）

DenseNet 将前面各层的特征直接拼接给后面的层，研究怎样复用特征并改善信息传播。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Huang et al. - 2016 - Densely Connected Convolutional Networks.pdf
- 原始 HTML：../../raw/html/Huang et al. - 2016 - Densely Connected Convolutional Networks.html
- 全文文本：../../raw/text/Huang et al. - 2016 - Densely Connected Convolutional Networks.md
- 作者：Huang et al.
- 年份：2016
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

这种连接与残差相加不同：后续层接收已有特征集合，再学习新的特征。它可减少重复学习，但特征存储和拼接也有成本；参数效率和实际内存需求需要分别比较。

## 关键事实

- **C1**：DenseNet让每层接收同尺寸block内全部先前特征的concatenation。
- **C2**：与ResNet相加不同，串接保留各层特征通道供后续复用。
- **C3**：报告在CIFAR/SVHN/ImageNet验证参数效率；表格区分数据增强和dropout。

## 争议与不确定点

- 强串接可能增加激活和数据搬运，部署内存需实测。
- 论文的参数优势依赖所选深度、growth rate、压缩与数据设置。

## 关联页面

- 主题：[经典 CNN 架构](../../wiki/topics/经典%20CNN%20架构.md)
- 主题：[传统 CV](../../wiki/topics/传统%20CV.md)
- 概念：[DenseNet](../../wiki/concepts/DenseNet.md)
- 概念：[ResNet](../../wiki/concepts/ResNet.md)

## 方法与实验解读

Dense connectivity把特征保留与新增特征分开：增长率控制每层加入的通道数，后层复用早期表示。瓶颈/压缩设计控制串接带来的通道膨胀。比较ResNet时应同时看参数、激活、计算和任务精度，不能把参数效率泛化为所有资源维度占优。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Huang%20et%20al.%20-%202016%20-%20Densely%20Connected%20Convolutional%20Networks.md#source-section-7 ) | 跨block用transition改变空间尺寸。 |
| C2 | [原文]( ../../raw/text/Huang%20et%20al.%20-%202016%20-%20Densely%20Connected%20Convolutional%20Networks.md#source-section-26 ) | 是结构性差异，不是简单多一条shortcut。 |
| C3 | [原文]( ../../raw/text/Huang%20et%20al.%20-%202016%20-%20Densely%20Connected%20Convolutional%20Networks.md#source-section-14 ) | 较少参数不等于峰值激活或实际运行内存更低。 |

## 核证范围

核读dense定义、ResNet差异、实验表与discussion。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

---
type: concept
---
# GoogLeNet

## TL;DR（快速导读）

GoogLeNet 的 Inception 模块用多种尺度的并行分支处理图像，再合并信息，在预算内扩展表示能力。

## 简介

GoogLeNet 的 Inception 模块用多种尺度的并行分支处理图像，再合并信息，在预算内扩展表示能力。

## 具体怎么理解

同一局部区域可以通过不同感受范围的分支观察，再组合成下一层特征。

## 关键属性

- 类型：视觉 backbone / 经典 CNN
- 代表来源：[Szegedy et al. - 2014 - Going Deeper with Convolutions](../../wiki/summaries/Szegedy%20et%20al.%20-%202014%20-%20Going%20Deeper%20with%20Convolutions.md)
- 当前角色：承接 `Inception` 模块化多尺度设计主线

## 相关主张

- `GoogLeNet` 通过 `Inception module` 在受控复杂度下结合多尺度分支。
- 它把经典 CNN 的主要创新方向之一从单一路径堆叠扩展为多分支结构工程。
- 在当前知识库里，它也是 `ResNeXt` 想要继承但简化的多分支思想来源之一。

## 来源支持

- [Szegedy et al. - 2014 - Going Deeper with Convolutions](../../wiki/summaries/Szegedy%20et%20al.%20-%202014%20-%20Going%20Deeper%20with%20Convolutions.md)

## 关联页面

- [经典 CNN 架构](../topics/经典%20CNN%20架构.md)
- [传统 CV](../topics/传统%20CV.md)
- [ResNeXt](./ResNeXt.md)

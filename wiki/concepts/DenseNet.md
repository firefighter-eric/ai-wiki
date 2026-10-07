---
type: concept
---
# DenseNet

## TL;DR（快速导读）

DenseNet 把前面层的特征直接传给后面层，促进特征复用；它使用拼接，区别于 ResNet 的残差相加。

## 简介

DenseNet 把前面层的特征直接传给后面层，促进特征复用；它使用拼接，区别于 ResNet 的残差相加。

## 具体怎么理解

后层既能使用新计算的特征，也能直接读早期边缘等特征；需要同时考虑连接带来的内存成本。

## 关键属性

- 类型：视觉 backbone / 经典 CNN
- 代表来源：[Huang et al. - 2016 - Densely Connected Convolutional Networks](../../wiki/summaries/Huang%20et%20al.%20-%202016%20-%20Densely%20Connected%20Convolutional%20Networks.md)
- 当前角色：承接强跨层连接与 feature reuse 路线

## 相关主张

- `DenseNet` 通过跨层特征串接强化信息流与特征复用。
- 它与 `ResNet` 的关键差别是连接方式从残差相加改为显式拼接。
- 在当前知识库里，`DenseNet` 代表经典 CNN 中“更强连接结构换取参数效率”的重要分支。

## 来源支持

- [Huang et al. - 2016 - Densely Connected Convolutional Networks](../../wiki/summaries/Huang%20et%20al.%20-%202016%20-%20Densely%20Connected%20Convolutional%20Networks.md)

## 关联页面

- [经典 CNN 架构](../topics/经典%20CNN%20架构.md)
- [传统 CV](../topics/传统%20CV.md)
- [ResNet](./ResNet.md)

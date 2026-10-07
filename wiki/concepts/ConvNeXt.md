---
type: concept
---
# ConvNeXt

## TL;DR（快速导读）

ConvNeXt 重新设计纯卷积骨干的结构和训练细节，研究在视觉 Transformer 时代卷积模型能做到什么。

## 简介

ConvNeXt 重新设计纯卷积骨干的结构和训练细节，研究在视觉 Transformer 时代卷积模型能做到什么。

## 具体怎么理解

比较卷积与 Transformer 时，需要把训练配方和预算放在同一条件下，不能把所有收益都归给注意力。

## 关键属性

- 类型：视觉 backbone / 现代 CNN
- 代表来源：[Liu et al. - 2022 - A ConvNet for the 2020s](../../wiki/summaries/Liu%20et%20al.%20-%202022%20-%20A%20ConvNet%20for%20the%202020s.md)
- 当前角色：连接 `ResNet` 主线与 `ViT` 时代对卷积归纳偏置的重新评估

## 相关主张

- `ConvNeXt` 通过系统现代化 `ResNet` 证明纯卷积 backbone 仍可与层级 Transformer 竞争。
- 它的重要性在于说明卷积的弱势不完全来自归纳偏置本身，也来自旧式设计和训练配方。
- 在当前知识库里，`ConvNeXt` 是经典 CNN topic 与 `ViT` topic 之间的重要桥梁概念。

## 来源支持

- [Liu et al. - 2022 - A ConvNet for the 2020s](../../wiki/summaries/Liu%20et%20al.%20-%202022%20-%20A%20ConvNet%20for%20the%202020s.md)
- [He et al. - 2015 - Deep Residual Learning for Image Recognition](../../wiki/summaries/He%20et%20al.%20-%202015%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition.md)

## 关联页面

- [经典 CNN 架构](../topics/经典%20CNN%20架构.md)
- [传统 CV](../topics/传统%20CV.md)
- [ResNet](./ResNet.md)
- [ViT](./ViT.md)

## 这里的术语是什么意思

- **backbone**：模型骨干：主要负责提取或变换表示，其他任务模块在它之上工作。

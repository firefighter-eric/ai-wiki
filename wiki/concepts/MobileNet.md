---
type: concept
---
# MobileNet

## TL;DR（快速导读）

MobileNet 通过深度可分离卷积及规模调节面向移动端效率，关注准确率、延迟与模型大小的折中。

## 简介

MobileNet 通过深度可分离卷积及规模调节面向移动端效率，关注准确率、延迟与模型大小的折中。

## 具体怎么理解

先分别在每个通道处理空间信息，再混合通道；目标设备的实际算子效率决定最终收益。

## 关键属性

- 类型：视觉 backbone / 轻量 CNN
- 代表来源：[Howard et al. - 2017 - MobileNets Efficient Convolutional Neural Networks for Mobile Vision Applications](../../wiki/summaries/Howard%20et%20al.%20-%202017%20-%20MobileNets%20Efficient%20Convolutional%20Neural%20Networks%20for%20Mobile%20Vision%20Applications.md)
- 当前角色：承接移动端高效视觉模型路线

## 相关主张

- `MobileNet` 通过 depthwise separable convolution 显著降低计算量。
- 它把 width multiplier 与 resolution multiplier 做成全局调节手柄，强调工程可调性。
- 在当前知识库里，`MobileNet` 代表经典 CNN 中效率优先而非纯精度优先的独立分支。

## 来源支持

- [Howard et al. - 2017 - MobileNets Efficient Convolutional Neural Networks for Mobile Vision Applications](../../wiki/summaries/Howard%20et%20al.%20-%202017%20-%20MobileNets%20Efficient%20Convolutional%20Neural%20Networks%20for%20Mobile%20Vision%20Applications.md)

## 关联页面

- [经典 CNN 架构](../topics/经典%20CNN%20架构.md)
- [传统 CV](../topics/传统%20CV.md)
- [ConvNeXt](./ConvNeXt.md)

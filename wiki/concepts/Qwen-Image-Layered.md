---
type: concept
---
# Qwen-Image-Layered

## TL;DR（快速导读）

Qwen-Image-Layered 将图像分成可独立处理的 RGBA 图层，关注编辑时保留内容与合成关系。

## 简介

Qwen-Image-Layered 将图像分成可独立处理的 RGBA 图层，关注编辑时保留内容与合成关系。

## 具体怎么理解

修改背景时，应能保留角色层；移动角色后，边缘透明度与原来被遮挡区域的完整性都影响合成。

## 关键属性

- 类型：图层分解与可编辑图像生成模型
- 代表来源：
  - [Yin et al. - 2025 - Qwen-Image-Layered Towards Inherent Editability via Layer Decomposition](../../wiki/summaries/Yin%20et%20al.%20-%202025%20-%20Qwen-Image-Layered%20Towards%20Inherent%20Editability%20via%20Layer%20Decomposition.md)
- 当前角色：Qwen 图像生成支线中的 layered decomposition 节点

## 相关主张

- 它的重点不是继续提升单张 RGB 出图质量，而是把图像编辑所需的图层结构直接建模出来。
- 它说明 Qwen 图像生成路线已经从“文本渲染与编辑能力”进一步扩展到“表示级可编辑性”。
- 在方法上，`Qwen-Image-Layered` 与 `AlphaVAE` 共享一个关键判断：`RGBA` 不应只是附属通道，而应作为独立潜表示对象来训练。

报告区分三训练阶段和四任务，使用 Adam。RGBA 分解不是唯一原设计的恢复；评测允许层对齐与合并，实际编辑应检查重组保真、alpha 边界和层粒度。依据：[论文摘要](../summaries/Yin%20et%20al.%20-%202025%20-%20Qwen-Image-Layered%20Towards%20Inherent%20Editability%20via%20Layer%20Decomposition.md)。

## 来源支持

- [Yin et al. - 2025 - Qwen-Image-Layered Towards Inherent Editability via Layer Decomposition](../../wiki/summaries/Yin%20et%20al.%20-%202025%20-%20Qwen-Image-Layered%20Towards%20Inherent%20Editability%20via%20Layer%20Decomposition.md)

## 关联页面

- [Qwen-Image](./Qwen-Image.md)
- [AlphaVAE](./AlphaVAE.md)
- [RGBA 图层图像](./RGBA%20%E5%9B%BE%E5%B1%82%E5%9B%BE%E5%83%8F.md)
- [扩散模型与文生图](../topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
- [Qwen 系列](../topics/Qwen%20%E7%B3%BB%E5%88%97.md)

## 这里的术语是什么意思

- **RGBA**：颜色加透明度的四通道表示，适合透明图像和图层合成。

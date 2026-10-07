---
type: concept
---
# Qwen-Image

## TL;DR（快速导读）

Qwen-Image 是 Qwen 的图像生成与编辑支线，关注视觉内容与文字渲染，需与图像理解模型区分。

## 简介

Qwen-Image 是 Qwen 的图像生成与编辑支线，关注视觉内容与文字渲染，需与图像理解模型区分。

## 具体怎么理解

生成一张带中文招牌的图片，需要既检查场景与构图，也逐字检查招牌文字。

## 关键属性

- 类型：图像生成基础模型
- 代表来源：
  - [Qwen Team - 2025 - Qwen-Image Crafting with Native Text Rendering](../../wiki/summaries/Qwen%20Team%20-%202025%20-%20Qwen-Image%20Crafting%20with%20Native%20Text%20Rendering.md)
- 当前角色：Qwen 家族的 image generation 分支起点

## 相关主张

- Qwen-Image 的差异化重点不是抽象地“图像质量更高”，而是把复杂文本渲染与一致性编辑做成核心能力。
- 其 `20B MMDiT` 定位表明 Qwen 家族已经不只是在理解型多模态上扩展，也开始进入原生图像生成基础模型赛道。
- 从知识组织角度看，Qwen-Image 既属于扩散/文生图研究线，也应被放回 `Qwen 系列` 中理解家族分叉。
- 随着 `Qwen-Image-Layered` 出现，Qwen 图像生成分支已经不只关注 RGB 出图与编辑，也开始向 `RGBA` 图层表示和可编辑工作流延伸。

## 来源支持

- [Qwen Team - 2025 - Qwen-Image Crafting with Native Text Rendering](../../wiki/summaries/Qwen%20Team%20-%202025%20-%20Qwen-Image%20Crafting%20with%20Native%20Text%20Rendering.md)

## 关联页面

- [Qwen](./Qwen.md)
- [Qwen-Image-Layered](./Qwen-Image-Layered.md)
- [RGBA 图层图像](./RGBA%20%E5%9B%BE%E5%B1%82%E5%9B%BE%E5%83%8F.md)
- [Qwen2.5-Omni](./Qwen2.5-Omni.md)
- [Qwen3.5-Omni](./Qwen3.5-Omni.md)
- [Stable Diffusion](./Stable%20Diffusion.md)
- [扩散模型与文生图](../topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
- [Qwen 系列](../topics/Qwen%20系列.md)

## 这里的术语是什么意思

- **RGBA**：颜色加透明度的四通道表示，适合透明图像和图层合成。

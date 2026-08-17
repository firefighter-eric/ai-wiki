---
type: concept
---
# Vision Banana

## 简介

`Vision Banana` 是 Google DeepMind 在论文 `Image Generators are Generalist Vision Learners` 中提出的通用视觉模型。它的关键意义不在于新增一个分割或深度估计专门架构，而在于把强图像生成模型 `Nano Banana Pro` 当作视觉基础模型底座，并通过轻量 instruction tuning 把分割、深度、表面法线等感知任务统一为 `RGB` 图像生成任务。

## 关键属性

- 类型：生成式视觉基础模型 / 通用视觉理解模型
- 底座：`Nano Banana Pro`
- 代表来源：
  - [Gabeur et al. - 2026 - Image Generators are Generalist Vision Learners](../../wiki/summaries/Gabeur%20et%20al.%20-%202026%20-%20Image%20Generators%20are%20Generalist%20Vision%20Learners.md)
- 当前角色：连接图像生成预训练、传统 CV 感知任务和统一 `RGB` 输出接口的关键节点

## 相关主张

- `Vision Banana` 把“图像生成模型是否隐含视觉理解能力”从直觉论断推进到 benchmark 证据：轻量 instruction tuning 后，它在多个分割、metric depth 和 surface normal 任务上达到或接近 SOTA。
- 它的统一接口不是文本，而是可解码 `RGB` 图像：模型生成 mask、depth colormap 或 normal map，再解析回标准视觉任务输出。
- 与 `CLIP / ViT / SigLIP` 这类表征学习路线相比，`Vision Banana` 更强调生成式预训练本身可成为视觉理解底座。
- 与 `Stable Diffusion / FLUX.2 / Qwen-Image` 这类图像生成主线相比，它把生成模型的能力边界从“产出可控图像”推向“用生成接口承接感知任务”。
- 这一路线仍受闭源模型、数据透明度和推理成本限制；当前更适合作为“生成模型正在进入 CV 基础模型层”的强信号，而不是完全替代专门视觉模型的最终证据。

## 来源支持

- [Gabeur et al. - 2026 - Image Generators are Generalist Vision Learners](../../wiki/summaries/Gabeur%20et%20al.%20-%202026%20-%20Image%20Generators%20are%20Generalist%20Vision%20Learners.md)

## 关联页面

- [扩散模型与文生图](../topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
- [传统 CV](../topics/传统%20CV.md)
- [CLIP](./CLIP.md)
- [ViT](./ViT.md)
- [FLUX.2](./FLUX.2.md)
- [Stable Diffusion](./Stable%20Diffusion.md)
- [Qwen-Image](./Qwen-Image.md)

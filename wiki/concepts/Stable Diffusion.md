---
type: concept
---
# Stable Diffusion

## TL;DR（快速导读）

Stable Diffusion 在压缩后的潜空间进行图像扩散，将生成模型推向可本地运行和适配的开放生态。

## 简介

Stable Diffusion 在压缩后的潜空间进行图像扩散，将生成模型推向可本地运行和适配的开放生态。

## 具体怎么理解

模型先在较小的表示中生成，再解码成图片；图像质量、显存与速度仍随版本和配置变化。

## 关键属性

- 类型：开放文生图模型家族
- 代表来源：
  - [Rombach et al. - 2022 - High-Resolution Image Synthesis with Latent Diffusion Models](../../wiki/summaries/Rombach%20et%20al.%20-%202022%20-%20High-Resolution%20Image%20Synthesis%20with%20Latent%20Diffusion%20Models.md)
  - [Stability AI - 2022 - Stable Diffusion Launch Announcement](../../wiki/summaries/Stability%20AI%20-%202022%20-%20Stable%20Diffusion%20Launch%20Announcement.md)
- 当前角色：开放扩散模型主线的奠基家族

## 相关主张

- Stable Diffusion 的真正突破不只是“能文生图”，而是把扩散生成的成本压低到消费级 GPU 可运行的范围。
- 它把 latent diffusion 从研究论文扩展为开放权重、开放代码、社区微调与插件生态的公共基础设施。
- 后续大量图像生成路线，无论是强调开放、强调控制、还是强调文本渲染，基本都要回应 Stable Diffusion 所建立的基线。

## 来源支持

- [Rombach et al. - 2022 - High-Resolution Image Synthesis with Latent Diffusion Models](../../wiki/summaries/Rombach%20et%20al.%20-%202022%20-%20High-Resolution%20Image%20Synthesis%20with%20Latent%20Diffusion%20Models.md)
- [Stability AI - 2022 - Stable Diffusion Launch Announcement](../../wiki/summaries/Stability%20AI%20-%202022%20-%20Stable%20Diffusion%20Launch%20Announcement.md)

## 关联页面

- [FLUX.2](./FLUX.2.md)
- [Qwen-Image](./Qwen-Image.md)
- [Qwen](./Qwen.md)
- [扩散模型与文生图](../topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)

## 这里的术语是什么意思

- **latent**：潜表示：原始数据经过模型编码后的内部表示，通常更紧凑。

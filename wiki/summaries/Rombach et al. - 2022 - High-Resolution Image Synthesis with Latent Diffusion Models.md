---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Rombach et al. - 2022 - High-Resolution Image Synthesis with Latent Diffusion Models

## TL;DR（快速导读）

潜空间扩散先把图像压缩成表示，再在较小表示中去噪，降低图像生成的计算成本。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / CVPR 2022
- 来源链接：https://arxiv.org/abs/2112.10752
- 全文文本：../../raw/text/Rombach et al. - 2022 - High-Resolution Image Synthesis with Latent Diffusion Models.md
- 作者：Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Bjorn Ommer
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

自编码器负责图像与潜表示之间的转换，扩散模型在潜空间学习生成。成本降低依赖压缩质量，也可能丢失细节；阅读时应同时检查重建能力、条件控制和最终生成效果。

## 关键事实

- **C1**：LDM先训练感知压缩autoencoder，再在低维latent上训练扩散模型。
- **C2**：重建使用perceptual与patch-adversarial目标，研究多个下采样因子。
- **C3**：cross-attention可加入文本等条件，支持不同条件生成任务。
- **C4**：作者承认采样仍慢于GAN，像素精度任务受重建瓶颈影响。

## 争议与不确定点

- 原论文训练集/实验不能代表所有后来StableDiffusion版本。
- FID等分布指标不检验单图事实或文本拼写。

## 关联页面

- 概念：[Stable Diffusion](../../wiki/concepts/Stable%20Diffusion.md)
- 主题：[扩散模型与文生图](../../wiki/topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
- [Stability AI](../authors/Stability%20AI.md)：沿作者或机构继续阅读相关来源。

## 方法与实验解读

降低空间维度减少去噪成本，却先丢掉了一部分像素信息。条件attention让生成接收文本/布局等控制；应用成败由压缩、生成、条件理解和采样一起决定。不能把美观重建等同数值准确，文档小字和精密细节尤其需核对。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Rombach%20et%20al.%20-%202022%20-%20High-Resolution%20Image%20Synthesis%20with%20Latent%20Diffusion%20Models.md#source-section-7 ) | 压缩和生成阶段分离。 |
| C2 | [原文]( ../../raw/text/Rombach%20et%20al.%20-%202022%20-%20High-Resolution%20Image%20Synthesis%20with%20Latent%20Diffusion%20Models.md#source-section-8 ) | latent不是无损编码。 |
| C3 | [原文]( ../../raw/text/Rombach%20et%20al.%20-%202022%20-%20High-Resolution%20Image%20Synthesis%20with%20Latent%20Diffusion%20Models.md#source-section-10 ) | 条件接口不保证prompt完全遵循。 |
| C4 | [原文]( ../../raw/text/Rombach%20et%20al.%20-%202022%20-%20High-Resolution%20Image%20Synthesis%20with%20Latent%20Diffusion%20Models.md#source-section-20 ) | 效率与精度折中。 |

## 核证范围

核读两阶段方法、autoencoder、条件机制、压缩对照与明确局限。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

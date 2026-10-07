---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Stability AI - 2022 - Stable Diffusion Launch Announcement

## TL;DR（快速导读）

Stable Diffusion 发布公告记录潜空间扩散模型进入公开权重与代码阶段，适合了解早期使用入口。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

模型先在较小的表示中生成，再解码成图片；图像质量、显存与速度仍随版本和配置变化。

## 来源信息

- 类型：官方公告 / 产品发布
- 来源链接：https://stability.ai/news-updates/stable-diffusion-announcement
- 全文文本：../../raw/text/Stability AI - 2022 - Stable Diffusion Launch Announcement.md
- 作者：Stability AI
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

公告介绍文生图、消费级 GPU 使用与社区发布。它说明当时的公开范围与产品定位；图像生成机制应看潜空间扩散论文，许可、具体权重和设备要求应按对应版本核对。

## 关键事实

- **C1**：2022年公告描述先向研究者放行StableDiffusion，公开发布仍在筹备。
- **C2**：路线承接CompVis/Runway的CVPR22 LatentDiffusion，并由LAION/HuggingFace等协作。
- **C3**：宣布公开代码/模型卡、由HuggingFace承载可获准的权重，强调消费级GPU运行。

## 争议与不确定点

- 硬件与生成速度宣传需要精度、尺寸和采样步骤条件。
- 开放获取和协作名单不构成全部使用场景的权利结论。

## 关联页面

- 概念：[Stable Diffusion](../../wiki/concepts/Stable%20Diffusion.md)
- 主题：[扩散模型与文生图](../../wiki/topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
- [Stability AI](../authors/Stability%20AI.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **latent**：潜表示：原始数据经过模型编码后的内部表示，通常更紧凑。

## 方法与实验解读

研究方法将去噪搬到latent，发布协作再把它变成可下载的模型和开发生态。本文用于开放生成路线的历史节点，具体结构回到LDM论文，当前软件/授权回到具体模型卡。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Stability%20AI%20-%202022%20-%20Stable%20Diffusion%20Launch%20Announcement.md#source-section-1 ) | 区别首阶段与后来publicrelease。 |
| C2 | [原文]( ../../raw/text/Stability%20AI%20-%202022%20-%20Stable%20Diffusion%20Launch%20Announcement.md#source-section-1 ) | 公告署名与组织贡献，不自动分配版权归属。 |
| C3 | [原文]( ../../raw/text/Stability%20AI%20-%202022%20-%20Stable%20Diffusion%20Launch%20Announcement.md#source-section-1 ) | 该版本发布声明，不是所有后来模型规格。 |

## 核证范围

核读完整发布公告，修正将首阶段公告写成全面公开发布的混淆。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

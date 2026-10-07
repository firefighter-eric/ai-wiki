---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Luo et al. - 2024 - IntrinsicDiffusion Joint Intrinsic Layers from Latent Diffusion Models

## TL;DR（快速导读）

IntrinsicDiffusion 将图像分解为材质颜色、照明和几何等内在因素，研究可控的物理属性编辑。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / Adobe Research
- 来源链接：https://research.adobe.com/publication/intrinsicdiffusion-joint-intrinsic-layers-from-latent-diffusion-models/
- 原始文件：../../raw/html/Luo et al. - 2024 - IntrinsicDiffusion Joint Intrinsic Layers from Latent Diffusion Models.html
- 全文文本：../../raw/text/Luo et al. - 2024 - IntrinsicDiffusion Joint Intrinsic Layers from Latent Diffusion Models.md
- 作者：Jundan Luo, Duygu Ceylan, Jae Shin Yoon, Nanxuan Zhao, Julien Philip, Anna Frühstück, Wenbin Li, Christian Richardt, Tuanfeng Y. Wang
- 年份：2024
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

它利用图像扩散模型学到的先验，联合预测反照率、光照和表面几何。这与前景背景透明图层不同：前者解释图像如何形成，后者主要组织可移动的对象和图层。

## 关键事实

- **C1**：官方论文简介讨论albedo、illumination与surface geometry的intrinsic decomposition。
- **C2**：方法在预训练生成模型上加入conditioning，联合预测多种intrinsic modality。
- **C3**：作者称可混合仅标注部分模态的数据，并展示relighting/retexturing用途。

## 争议与不确定点

- 本库此来源只保存官方publication简介，尚无可用论文全文；核证范围限简介明确主张。
- 与对象/可编辑设计图层属于不同任务，不能按layer一词合并。

## 关联页面

- 主题：[图像分层 layered](../../wiki/topics/%E5%9B%BE%E5%83%8F%E5%88%86%E5%B1%82%20layered.md)
- 主题：[传统 CV](../../wiki/topics/%E4%BC%A0%E7%BB%9F%20CV.md)
- 主题：[扩散模型与文生图](../../wiki/topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)

## 方法与实验解读

当前来源是Adobe的论文简介，能够确认问题定义、联合条件预测和编辑应用。单张图像的光照/材质/几何分解本身存在歧义，生成先验给出合理解并不表示唯一物理解。本页不把简介里的SOTA宣传升级成已经核读完整评测的结论。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Luo%20et%20al.%20-%202024%20-%20IntrinsicDiffusion%20Joint%20Intrinsic%20Layers%20from%20Latent%20Diffusion%20Models.md#source-section-5 ) | 内在属性层，不是对象分层或PSD图层恢复。 |
| C2 | [原文]( ../../raw/text/Luo%20et%20al.%20-%202024%20-%20IntrinsicDiffusion%20Joint%20Intrinsic%20Layers%20from%20Latent%20Diffusion%20Models.md#source-section-5 ) | 简介未披露具体网络和完整实验。 |
| C3 | [原文]( ../../raw/text/Luo%20et%20al.%20-%202024%20-%20IntrinsicDiffusion%20Joint%20Intrinsic%20Layers%20from%20Latent%20Diffusion%20Models.md#source-section-5 ) | 联合训练能力声明，不补猜损失或评测分数。 |

## 核证范围

核读官方publication页全文与作者/会议信息；未声称读过外链ACM正文。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

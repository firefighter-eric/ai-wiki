---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Yin et al. - 2025 - Qwen-Image-Layered Towards Inherent Editability via Layer Decomposition

## TL;DR（快速导读）

Qwen-Image-Layered 把一张图片拆成多个语义图层，让移动、改色等操作尽量只影响目标层。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

修改背景时，应能保留角色层；移动角色后，边缘透明度与原来被遮挡区域的完整性都影响合成。

## 来源信息

- 类型：论文 / arXiv
- 来源链接：https://arxiv.org/abs/2512.15603
- 原始文件：../../raw/pdf/Yin et al. - 2025 - Qwen-Image-Layered Towards Inherent Editability via Layer Decomposition.pdf
- 全文文本：../../raw/text/Yin et al. - 2025 - Qwen-Image-Layered Towards Inherent Editability via Layer Decomposition.md
- 作者：Shengming Yin, Zekai Zhang, Zecheng Tang, Kaiyuan Gao, Xiao Xu, Kun Yan, Jiahao Li, Yilei Chen, Yuxiang Chen, Heung-Yeung Shum, Lionel M. Ni, Jingren Zhou, Junyang Lin, Chenfei Wu
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

扩散分解器输出带透明度的图层，使编辑可以围绕独立对象进行。真正可编辑还要求层关系、边缘和重组结果稳定；层数增加本身不证明未编辑区域已经得到保护。

## 关键事实

- **C1**：从单RGB分解成可变数量语义RGBA图层，重组采用按序alpha-blending。
- **C2**：RGBA-VAE共用RGB/RGBAlatent，VLD-MMDiT用Layer3DRoPE组织层内/层间关系。
- **C3**：三训练阶段由textRGB/RGBA，再multilayer，再image-to-multilayer。
- **C4**：数据取自真实PSD的过滤/合并/标注；Crello用order-awareDTW对齐并允许层合并。
- **C5**：报告训练使用Adam与lr1e-5，不替换成惯常AdamW。

## 争议与不确定点

- 透明层与可编辑图层并不保证已恢复原PSD的文字/矢量/效果参数。
- 小型Crello域与复杂真实PSD有分布差异，整体泛化仍受数据约束。

## 关联页面

- 概念：[Qwen-Image-Layered](../../wiki/concepts/Qwen-Image-Layered.md)
- 概念：[RGBA 图层图像](../../wiki/concepts/RGBA%20%E5%9B%BE%E5%B1%82%E5%9B%BE%E5%83%8F.md)
- 概念：[Qwen-Image](../../wiki/concepts/Qwen-Image.md)
- 主题：[扩散模型与文生图](../../wiki/topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
- 主题：[Qwen 系列](../../wiki/topics/Qwen%20%E7%B3%BB%E5%88%97.md)
- [Jingren Zhou](../authors/Jingren%20Zhou.md)：沿作者或机构继续阅读相关来源。
- [Junyang Lin](../authors/Junyang%20Lin.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **RGBA**：颜色加透明度的四通道表示，适合透明图像和图层合成。

## 方法与实验解读

多层输出把编辑操作放在表示层，使增删、移动和替换更直接。可变层数不保证天然最佳粒度，评价允许合并层正说明groundtruth不是唯一正确结构；还要检查重组保真、alpha边界、跨层遮挡与实际编辑可用性。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Yin%20et%20al.%20-%202025%20-%20Qwen-Image-Layered%20Towards%20Inherent%20Editability%20via%20Layer%20Decomposition.md#source-section-8 ) | 设计层分解多解，不是唯一物理分解。 |
| C2 | [原文]( ../../raw/text/Yin%20et%20al.%20-%202025%20-%20Qwen-Image-Layered%20Towards%20Inherent%20Editability%20via%20Layer%20Decomposition.md#source-section-10 ) | 表征与关系建模不同职责。 |
| C3 | [原文]( ../../raw/text/Yin%20et%20al.%20-%202025%20-%20Qwen-Image-Layered%20Towards%20Inherent%20Editability%20via%20Layer%20Decomposition.md#source-section-11 ) | 四任务、三阶段，修正任务链与阶段数混淆。 |
| C4 | [原文]( ../../raw/text/Yin%20et%20al.%20-%202025%20-%20Qwen-Image-Layered%20Towards%20Inherent%20Editability%20via%20Layer%20Decomposition.md#source-section-13 ) | Crello评价协议见§4.3.1，承认分解歧义。 |
| C5 | [原文]( ../../raw/text/Yin%20et%20al.%20-%202025%20-%20Qwen-Image-Layered%20Towards%20Inherent%20Editability%20via%20Layer%20Decomposition.md#source-section-14 ) | 明确optimizer披露。 |

## 核证范围

核读RGBA-VAE、VLD/Layer3D、三阶段、PSD、DTW评测与训练配置。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

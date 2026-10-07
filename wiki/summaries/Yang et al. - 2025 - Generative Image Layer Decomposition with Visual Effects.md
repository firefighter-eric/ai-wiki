---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Yang et al. - 2025 - Generative Image Layer Decomposition with Visual Effects

## TL;DR（快速导读）

LayerDecomp 将图像拆为干净背景和带透明效果的前景，帮助移动物体时保留阴影与反射。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / arXiv / Adobe Research
- 来源链接：https://arxiv.org/abs/2411.17864
- 原始文件：../../raw/pdf/Yang et al. - 2025 - Generative Image Layer Decomposition with Visual Effects.pdf
- 全文文本：../../raw/text/Yang et al. - 2025 - Generative Image Layer Decomposition with Visual Effects.md
- 作者：Jinrui Yang, Qing Liu, Yijun Li, Soo Ye Kim, Daniil Pakhomov, Mengwei Ren, Jianming Zhang, Zhe Lin, Cihang Xie, Yuyin Zhou
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

只抠出物体轮廓往往不能自然重组图像，论文因此把视觉效果纳入前景层。分解后要检查背景补全、透明边缘和效果归属，才能判断移动或缩放后的合成是否可信。

## 关键事实

- **C1**：LayerDecomp从composite与objectmask输出cleanbackground及带视觉效果的RGBAforeground。
- **C2**：consistencyloss在pixelspace将前景与背景重新alpha-blend回输入。
- **C3**：训练结合模拟triplets与真实camera-capturedpairs。
- **C4**：作者承认现有数据偏常见shadow/reflection，smoke/mist仍待扩展。

## 争议与不确定点

- 遮挡背景由生成补全，不是原始隐藏像素的可验证恢复。
- 多层递归/扩展与一次可变层生成需要分开。

## 关联页面

- 概念：[RGBA 图层图像](../../wiki/concepts/RGBA%20%E5%9B%BE%E5%B1%82%E5%9B%BE%E5%83%8F.md)
- 概念：[Qwen-Image-Layered](../../wiki/concepts/Qwen-Image-Layered.md)
- 主题：[扩散模型与文生图](../../wiki/topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)
- 主题：[图像分层 layered](../../wiki/topics/%E5%9B%BE%E5%83%8F%E5%88%86%E5%B1%82%20layered.md)

## 这里的术语是什么意思

- **RGBA**：颜色加透明度的四通道表示，适合透明图像和图层合成。

## 方法与实验解读

将投影阴影/反射放进可编辑前景，移动或移除对象时能更自然地处理随对象变化的效果。重组约束补标注不足，但存在多解；用户研究衡量视觉自然度，不能证明准确恢复了拍摄时物理分量。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Yang%20et%20al.%20-%202025%20-%20Generative%20Image%20Layer%20Decomposition%20with%20Visual%20Effects.md#source-section-8 ) | 默认两层目标，不是通用PSD恢复。 |
| C2 | [原文]( ../../raw/text/Yang%20et%20al.%20-%202025%20-%20Generative%20Image%20Layer%20Decomposition%20with%20Visual%20Effects.md#source-section-9 ) | 重组一致不保证分解唯一正确。 |
| C3 | [原文]( ../../raw/text/Yang%20et%20al.%20-%202025%20-%20Generative%20Image%20Layer%20Decomposition%20with%20Visual%20Effects.md#source-section-10 ) | 数据来源不等同全部自然视觉效果覆盖。 |
| C4 | [原文]( ../../raw/text/Yang%20et%20al.%20-%202025%20-%20Generative%20Image%20Layer%20Decomposition%20with%20Visual%20Effects.md#source-section-16 ) | 具体局限。 |

## 核证范围

核读DiT/RGB-RGBA、重组loss、模拟/真实数据、消融与局限。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

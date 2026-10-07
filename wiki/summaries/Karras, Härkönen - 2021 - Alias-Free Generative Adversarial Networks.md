---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Karras, Härkönen - 2021 - Alias-Free Generative Adversarial Networks

## TL;DR（快速导读）

StyleGAN3 的无混叠设计针对生成视频中纹理像粘在像素坐标上的问题，让细节随对象运动更一致。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

论文将问题追溯到生成网络中的信号处理与混叠，重新考虑卷积生成过程的采样和变换。关注点是变换一致性，不只是静态图片是否清晰；评测也需观察移动与旋转中的细节表现。

## 具体怎么理解

当人物转头时，皮肤纹理应跟着脸移动；如果细节像固定在屏幕上，就会出现不自然的“粘纹理”现象。

## 关键事实

- **C1**：把特征图解释为连续信号并抑制 aliasing，以减少生成细节黏在绝对像素位置的现象。
- **C2**：只修改生成器，作者指出判别器仍可能引入位置偏好。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Karras%2C%20H%C3%A4rk%C3%B6nen%20-%202021%20-%20Alias-Free%20Generative%20Adversarial%20Networks.pdf)
- 全文文本：[打开全文文本](../../raw/text/Karras%2C%20H%C3%A4rk%C3%B6nen%20-%202021%20-%20Alias-Free%20Generative%20Adversarial%20Networks.md)
- 作者：Karras, Härkönen
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Karras%2C%20H%C3%A4rk%C3%B6nen%20-%202021%20-%20Alias-Free%20Generative%20Adversarial%20Networks.html)

## 争议与不确定点

- 不同配置针对平移或旋转，结果不应混用。
- 仍存在牙齿等细节运动错误。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无
- [神经渲染](../concepts/%E7%A5%9E%E7%BB%8F%E6%B8%B2%E6%9F%93.md)：把本篇方法放到相关任务与比较条件中阅读。

## 方法与实验解读

StyleGAN3 关注画面移动时纹理应随物体一起移动。连续信号与抗混叠设计改善这一点，但头部转动、遮挡与三维结构还涉及更复杂的一致性问题，不能只凭等变性指标判断。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Karras%2C%20H%C3%A4rk%C3%B6nen%20-%202021%20-%20Alias-Free%20Generative%20Adversarial%20Networks.md#source-section-4 ) | 目标是变换等变性，不是任意视角三维一致性 |
| C2 | [原文]( ../../raw/text/Karras%2C%20H%C3%A4rk%C3%B6nen%20-%202021%20-%20Alias-Free%20Generative%20Adversarial%20Networks.md#source-section-22 ) | 不能宣称整个 GAN 已完全等变 |

## 核证范围

核对 §2 信号解释、§3.1 输入修改与 §5 的生成器和判别器边界。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

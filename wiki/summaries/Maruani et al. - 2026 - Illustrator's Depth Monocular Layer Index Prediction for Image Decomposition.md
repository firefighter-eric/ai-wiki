---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Maruani et al. - 2026 - Illustrator's Depth Monocular Layer Index Prediction for Image Decomposition

## TL;DR（快速导读）

Illustrator's Depth 为像素预测有序图层编号，帮助把插画拆成可以重新排列和编辑的层。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / arXiv / Adobe Research
- 来源链接：https://arxiv.org/abs/2511.17454
- 原始文件：../../raw/pdf/Maruani et al. - 2026 - Illustrator's Depth Monocular Layer Index Prediction for Image Decomposition.pdf
- 全文文本：../../raw/text/Maruani et al. - 2026 - Illustrator's Depth Monocular Layer Index Prediction for Image Decomposition.md
- 作者：Nissim Maruani, Peiying Zhang, Siddhartha Chaudhuri, Matthew Fisher, Nanxuan Zhao, Vladimir G. Kim, Pierre Alliez, Mathieu Desbrun, Wang Yifan
- 年份：2026
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

这里的深度表示编辑中的前后层关系，不是相机到物体的几何距离。论文关注插画、海报、阴影和轮廓线等内容，阅读时应检查层序一致性与分解后能否支持目标编辑。

## 关键事实

- **C1**：Illustrator’s Depth预测每像素的绘画层序，含义是艺术构图中的前后关系，不是物理距离。
- **C2**：用有合理层序的MMSVG-Illustration，合并连续同色层、剔除同色非连续重叠歧义并光栅化层索引。
- **C3**：网络从Depth Pro初始化，直接学习层索引与相对顺序。
- **C4**：预测可用于矢量化、text-to-vector、relief与基于深度的编辑。
- **C5**：作者明确指出白背景单对象训练造成忽略背景的失败。

## 争议与不确定点

- 设计层次有主观性；数据整理固定了部分歧义，并没有消除任务多解性。
- 纹理破损可被识别成前景，复杂背景和顺序错误见失败章节。

## 关联页面

- 主题：[图像分层 layered](../../wiki/topics/%E5%9B%BE%E5%83%8F%E5%88%86%E5%B1%82%20layered.md)
- 主题：[传统 CV](../../wiki/topics/%E4%BC%A0%E7%BB%9F%20CV.md)
- 主题：[Slide 理解与生成](../../wiki/topics/Slide%20%E7%90%86%E8%A7%A3%E4%B8%8E%E7%94%9F%E6%88%90.md)

## 这里的术语是什么意思

- **RGBA**：颜色加透明度的四通道表示，适合透明图像和图层合成。

## 方法与实验解读

从SVG的绘制次序获得可监督的层序，再迁移深度网络的边界/遮挡先验。预测层图为后续矢量化和编辑提供排序线索，与amodal segmentation恢复被遮挡形状、intrinsic decomposition分材质光照是不同目标。使用时先看背景是否丢失，再看层间顺序，不宜只用视觉效果判断。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Maruani%20et%20al.%20-%202026%20-%20Illustrator%27s%20Depth%20Monocular%20Layer%20Index%20Prediction%20for%20Image%20Decomposition.md#source-section-11 ) | 相邻物理深度可能属于不同设计层。 |
| C2 | [原文]( ../../raw/text/Maruani%20et%20al.%20-%202026%20-%20Illustrator%27s%20Depth%20Monocular%20Layer%20Index%20Prediction%20for%20Image%20Decomposition.md#source-section-14 ) | §3.2数据规则，不能等同现成PSD分层。 |
| C3 | [原文]( ../../raw/text/Maruani%20et%20al.%20-%202026%20-%20Illustrator%27s%20Depth%20Monocular%20Layer%20Index%20Prediction%20for%20Image%20Decomposition.md#source-section-17 ) | 几何先验与任务监督共同作用。 |
| C4 | [原文]( ../../raw/text/Maruani%20et%20al.%20-%202026%20-%20Illustrator%27s%20Depth%20Monocular%20Layer%20Index%20Prediction%20for%20Image%20Decomposition.md#source-section-21 ) | 用途分别见§4.2–4.4，不表示输出完整可编辑工程文件。 |
| C5 | [原文]( ../../raw/text/Maruani%20et%20al.%20-%202026%20-%20Illustrator%27s%20Depth%20Monocular%20Layer%20Index%20Prediction%20for%20Image%20Decomposition.md#source-section-56 ) | 域偏移是具体限制。 |

## 核证范围

核读任务定义、SVG数据、初始化/loss、应用与失败案例。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Bochkovskiy, Wang, Liao - 2020 - YOLOv4 Optimal Speed and Accuracy of Object Detection

## TL;DR（快速导读）

YOLOv4 通过组合网络结构、数据增强和训练技巧，提高单阶段目标检测的实用性。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Bochkovskiy, Wang, Liao - 2020 - YOLOv4 Optimal Speed and Accuracy of Object Detection.pdf
- 原始 HTML：../../raw/html/Bochkovskiy, Wang, Liao - 2020 - YOLOv4 Optimal Speed and Accuracy of Object Detection.html
- 全文文本：../../raw/text/Bochkovskiy, Wang, Liao - 2020 - YOLOv4 Optimal Speed and Accuracy of Object Detection.md
- 作者：Bochkovskiy, Wang, Liao
- 年份：2020
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

论文整合 CSP、Mosaic、CIoU 与 PAN 等组件：有的改变特征流，有的改善训练，有的帮助定位。阅读重点是哪些改动增加推理成本、哪些主要发生在训练阶段，以及组合收益如何通过消融验证。

## 关键事实

- **C1**：YOLOv4 采用 CSPDarknet53、SPP/PAN 与 YOLOv3 检测头组合。
- **C2**：区分推理不增成本的 Bag of Freebies 与少量增成本的 Bag of Specials。
- **C3**：Mosaic 将四张训练图合并；SAT 在两个阶段先扰动图像、再训练模型。
- **C4**：作者在 ImageNet 分类和 COCO 检测上考察组合，并强调单 GPU 的可训练性。

## 争议与不确定点

- 最快/最准确属于发布时比较，依赖分辨率、硬件和实现。
- 组合收益不应全部归功于某一个技巧；本页未复现实验。

## 关联页面

- 主题：[目标检测](../../wiki/topics/目标检测.md)
- 概念：[YOLO](../../wiki/concepts/YOLO.md)

## 这里的术语是什么意思

- **backbone**：模型骨干：主要负责提取或变换表示，其他任务模块在它之上工作。

## 方法与实验解读

YOLOv4 的贡献是把候选结构、损失、增强与正则化组合成可复用的实时检测配方。它仍是 one-stage anchor-based 检测，变化主要在特征提取和训练信号。分类中更好的 backbone 未必在检测中更好，因此需要任务匹配的消融和实际吞吐，不能只按 BFLOPs 做选择。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Bochkovskiy%2C%20Wang%2C%20Liao%20-%202020%20-%20YOLOv4%20Optimal%20Speed%20and%20Accuracy%20of%20Object%20Detection.md#source-section-12 ) | 该版本的 backbone/neck/head 配方。 |
| C2 | [原文]( ../../raw/text/Bochkovskiy%2C%20Wang%2C%20Liao%20-%202020%20-%20YOLOv4%20Optimal%20Speed%20and%20Accuracy%20of%20Object%20Detection.md#source-section-6 ) | 分类是训练/推理成本分类，不是效果排名。 |
| C3 | [原文]( ../../raw/text/Bochkovskiy%2C%20Wang%2C%20Liao%20-%202020%20-%20YOLOv4%20Optimal%20Speed%20and%20Accuracy%20of%20Object%20Detection.md#source-section-11 ) | 数据增强与结构升级分开归因。 |
| C4 | [原文]( ../../raw/text/Bochkovskiy%2C%20Wang%2C%20Liao%20-%202020%20-%20YOLOv4%20Optimal%20Speed%20and%20Accuracy%20of%20Object%20Detection.md#source-section-13 ) | 单 GPU 可训练不等于任意 GPU 均达到论文 FPS。 |

## 核证范围

核读方法配置、BoF/BoS、Mosaic/SAT、实验设计与结论。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

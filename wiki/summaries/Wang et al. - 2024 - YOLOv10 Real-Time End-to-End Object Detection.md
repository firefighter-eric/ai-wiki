---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Wang et al. - 2024 - YOLOv10 Real-Time End-to-End Object Detection

## TL;DR（快速导读）

YOLOv10 用两种样本分配协同训练，推理采用一对一预测，研究省去 NMS 后处理的目标检测。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Wang et al. - 2024 - YOLOv10 Real-Time End-to-End Object Detection.pdf
- 原始 HTML：../../raw/html/Wang et al. - 2024 - YOLOv10 Real-Time End-to-End Object Detection.html
- 全文文本：../../raw/text/Wang et al. - 2024 - YOLOv10 Real-Time End-to-End Object Detection.md
- 作者：Wang et al.
- 年份：2024
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

一对多分配提供训练监督，一对一分支承担推理输出，两者需要一致地学习。省去去重步骤改变了整体流程，但最终延迟仍应包括网络与输出处理，并按相同输入和硬件测试。

## 关键事实

- **C1**：consistentdualassignments训练时保留one-to-many监督，推理用one-to-one以去NMS。
- **C2**：轻分类head、空间/通道降采样与rank-guidedblock等协同设计效率/精度。
- **C3**：作者承认one-to-one与原one-to-manybranch仍有性能差距，未研究大数据预训练。

## 争议与不确定点

- 端到端latency与只网络前向FPS不能混比。
- 拥挤小物体、域转移与训练数据范围限制仍需要具体测试。

## 关联页面

- 主题：[目标检测](../../wiki/topics/目标检测.md)
- 概念：[YOLO](../../wiki/concepts/YOLO.md)
- 概念：[DETR](../../wiki/concepts/DETR.md)

## 这里的术语是什么意思

- **NMS**：非极大值抑制：检测后处理中的去重步骤，保留较合适的候选框。

## 方法与实验解读

训练分支用丰富正样本提高优化，推理分支用唯一匹配减少重复框。NMS-free路线可降低后处理成本，但性能必须以完整推理链和实际硬件测；与DETR的唯一匹配相通，模型结构和训练成本仍不同。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Wang%20et%20al.%20-%202024%20-%20YOLOv10%20Real-Time%20End-to-End%20Object%20Detection.md#source-section-6 ) | 去重复通过训练匹配而非简单删除NMS。 |
| C2 | [原文]( ../../raw/text/Wang%20et%20al.%20-%202024%20-%20YOLOv10%20Real-Time%20End-to-End%20Object%20Detection.md#source-section-7 ) | 结构多因素。 |
| C3 | [原文]( ../../raw/text/Wang%20et%20al.%20-%202024%20-%20YOLOv10%20Real-Time%20End-to-End%20Object%20Detection.md#source-section-21 ) | 明确limit而非全面胜出。 |

## 核证范围

核读dualassignments、效率架构、COCO设置与明确局限。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

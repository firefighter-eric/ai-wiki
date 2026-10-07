---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Ultralytics - 2026 - Ultralytics YOLO Docs Home

## TL;DR（快速导读）

这份 Ultralytics 首页快照用于辨认当时的 YOLO 产品版本与官方入口，不能代替原论文的机制说明。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

交通画面需要逐帧找出车和人，速度与漏检都重要；网络延迟还要与完整后处理成本一起计算。

## 来源信息

- 类型：官方文档 / 产品总览
- 原始 HTML：../../raw/html/Ultralytics - 2026 - Ultralytics YOLO Docs Home.html
- 全文文本：../../raw/text/Ultralytics - 2026 - Ultralytics YOLO Docs Home.md
- 作者 / 机构：Ultralytics
- 年份：2026
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

页面记录归档时的版本推荐和文档导航。可据此进入具体模型的训练、推理与任务文档，再回到论文核对结构；页面中的“最新”和推荐会变化，应始终连同快照日期阅读。

## 关键事实

- **C1**：保存文档首页把YOLO26列为最新，并称NMS-free/edge优化。
- **C2**：首页建议生产使用YOLO26/YOLO11，并覆盖detect/segment/classify/pose/OBB/track。
- **C3**：文档列AGPL3.0与Enterprise许可两个入口。

## 争议与不确定点

- 官方推荐和latest均需固定保存日期。
- 页面宣传的edge效率没有在此给出统一测量协议。

## 关联页面

- 主题：[目标检测](../../wiki/topics/目标检测.md)
- 概念：[YOLO](../../wiki/concepts/YOLO.md)

## 这里的术语是什么意思

- **NMS**：非极大值抑制：检测后处理中的去重步骤，保留较合适的候选框。

## 方法与实验解读

首页是产品版本与任务入口，不提供YOLO历史各代统一架构或训练证据。核对NMS-free须进入具体型号资料，早期论文则回到原始summary；同一YOLO名字还包含不同团队的实现。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Ultralytics%20-%202026%20-%20Ultralytics%20YOLO%20Docs%20Home.md#source-section-1 ) | 当前页快照声明，后续latest会漂移。 |
| C2 | [原文]( ../../raw/text/Ultralytics%20-%202026%20-%20Ultralytics%20YOLO%20Docs%20Home.md#source-section-2 ) | 库能力与单模型任务分别看。 |
| C3 | [原文]( ../../raw/text/Ultralytics%20-%202026%20-%20Ultralytics%20YOLO%20Docs%20Home.md#source-section-4 ) | 只是许可导航，不为具体业务作法律归类。 |

## 核证范围

核读首页、任务导航、历史与许可段，限定快照用途。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

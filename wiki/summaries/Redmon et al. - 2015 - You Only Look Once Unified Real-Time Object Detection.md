---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Redmon et al. - 2015 - You Only Look Once Unified Real-Time Object Detection

## TL;DR（快速导读）

YOLOv1 一次前向计算就预测目标框和类别，把检测组织成单阶段回归任务。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Redmon et al. - 2015 - You Only Look Once Unified Real-Time Object Detection.pdf
- 原始 HTML：../../raw/html/Redmon et al. - 2015 - You Only Look Once Unified Real-Time Object Detection.html
- 全文文本：../../raw/text/Redmon et al. - 2015 - You Only Look Once Unified Real-Time Object Detection.md
- 作者：Redmon et al.
- 年份：2015
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

模型在全图上直接产生预测，以减少候选区域流水线的复杂性。阅读时可看速度与定位、召回之间的取舍，尤其关注不同大小和密集目标；实时速度需要结合硬件与输入尺寸判断。

## 关键事实

- **C1**：YOLOv1把整图到框/类别作为单网络回归，网格负责中心落入其中的对象。
- **C2**：推理仍进行NMS，不是只一次网络前向就取消全部后处理。
- **C3**：论文错误分析显示背景误检较少，但定位错误更多。
- **C4**：小目标密集群与精确定位受网格和空间限制。

## 争议与不确定点

- 历史VOC mAP、COCO AP与AP50不可直接拼榜。
- 实时性依赖硬件、输入尺寸和是否计后处理。

## 关联页面

- 主题：[目标检测](../../wiki/topics/目标检测.md)
- 主题：[传统 CV](../../wiki/topics/传统%20CV.md)
- 概念：[YOLO](../../wiki/concepts/YOLO.md)

## 方法与实验解读

整图统一学习能利用上下文，省去独立proposal生成的流程；固定网格却容易让同格对象争用有限预测槽。检测比较同时看背景误检、漏检和定位精度，不能只看FPS。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Redmon%20et%20al.%20-%202015%20-%20You%20Only%20Look%20Once%20Unified%20Real-Time%20Object%20Detection.md#source-section-4 ) | 每格有限boxes和类别约束。 |
| C2 | [原文]( ../../raw/text/Redmon%20et%20al.%20-%202015%20-%20You%20Only%20Look%20Once%20Unified%20Real-Time%20Object%20Detection.md#source-section-7 ) | one-stage与end-to-end无后处理不是同义词。 |
| C3 | [原文]( ../../raw/text/Redmon%20et%20al.%20-%202015%20-%20You%20Only%20Look%20Once%20Unified%20Real-Time%20Object%20Detection.md#source-section-12 ) | VOC2007对照，非所有领域规律。 |
| C4 | [原文]( ../../raw/text/Redmon%20et%20al.%20-%202015%20-%20You%20Only%20Look%20Once%20Unified%20Real-Time%20Object%20Detection.md#source-section-8 ) | 作者承认的失败模式。 |

## 核证范围

核读grid建模、训练/推理、错误分析和局限。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

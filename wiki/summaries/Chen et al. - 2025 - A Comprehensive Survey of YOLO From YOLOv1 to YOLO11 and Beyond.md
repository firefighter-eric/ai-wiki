---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Chen et al. - 2025 - A Comprehensive Survey of YOLO From YOLOv1 to YOLO11 and Beyond

## TL;DR（快速导读）

这篇 YOLO 综述按特征提取、特征融合、预测与训练等环节解释版本差异，适合建立家族阅读地图。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

交通画面需要逐帧找出车和人，速度与漏检都重要；网络延迟还要与完整后处理成本一起计算。

## 来源信息

- 类型：综述 / 调研
- 原始文件：../../raw/pdf/Chen et al. - 2025 - A Comprehensive Survey of YOLO From YOLOv1 to YOLO11 and Beyond.pdf
- 原始 HTML：../../raw/html/Chen et al. - 2025 - A Comprehensive Survey of YOLO From YOLOv1 to YOLO11 and Beyond.html
- 全文文本：../../raw/text/Chen et al. - 2025 - A Comprehensive Survey of YOLO From YOLOv1 to YOLO11 and Beyond.md
- 作者：Chen et al.
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

综述覆盖从 YOLOv1 到 YOLO11 等工作，帮助比较各版本到底改了哪一步。使用它时应按问题追踪结构、损失和样本分配的变化，再回到原论文核对；版本号本身不能说明速度或精度更好。

## 关键事实

- **C1**：综述按 YOLO 版本回顾实时检测路线，范围延伸至 YOLO11 及后续讨论。
- **C2**：YOLOv1 章节解释网格预测框与类别的统一接口。
- **C3**：YOLO11 的速度和精度数字来自文中引用的模型/应用材料。
- **C4**：结论把家族演化组织成兼顾准确率、实时性与部署效率的问题。

## 争议与不确定点

- 名称相近的 YOLO 分支不一定有同一作者或代码谱系。
- 综述中的性能汇总不是统一环境复跑；部分版本细节来自官方文档而非独立论文。

## 关联页面

- 主题：[目标检测](../../wiki/topics/目标检测.md)
- 概念：[YOLO](../../wiki/concepts/YOLO.md)

## 这里的术语是什么意思

- **backbone**：模型骨干：主要负责提取或变换表示，其他任务模块在它之上工作。

## 方法与实验解读

这篇综述提供历史地图，适合发现需要补读的版本与变化点。本库据此沿特征网络、融合、预测头、分配/损失和训练配方比较家族；这些比较轴是知识组织方法，不能冒称作者的唯一正式分类。版本数字跨越不同设备和指标时，应拆回原文实验条件。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Chen%20et%20al.%20-%202025%20-%20A%20Comprehensive%20Survey%20of%20YOLO%20From%20YOLOv1%20to%20YOLO11%20and%20Beyond.md#source-section-2 ) | 综述的截止时间，不是当前全部版本目录。 |
| C2 | [原文]( ../../raw/text/Chen%20et%20al.%20-%202025%20-%20A%20Comprehensive%20Survey%20of%20YOLO%20From%20YOLOv1%20to%20YOLO11%20and%20Beyond.md#source-section-11 ) | 网格责任分配属于初代结构，不能概括所有后续版本。 |
| C3 | [原文]( ../../raw/text/Chen%20et%20al.%20-%202025%20-%20A%20Comprehensive%20Survey%20of%20YOLO%20From%20YOLOv1%20to%20YOLO11%20and%20Beyond.md#source-section-64 ) | COCO mAP 与农业任务 mAP@50 不可直接横比。 |
| C4 | [原文]( ../../raw/text/Chen%20et%20al.%20-%202025%20-%20A%20Comprehensive%20Survey%20of%20YOLO%20From%20YOLOv1%20to%20YOLO11%20and%20Beyond.md#source-section-72 ) | 综述综合判断，具体版本事实优先回溯该版原始报告。 |

## 核证范围

核读摘要、初代接口、后期比较段落与总括结论，未把全部引用表逐项独立复现。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

# Wang et al. - 2024 - YOLOv10 Real-Time End-to-End Object Detection

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Wang et al. - 2024 - YOLOv10 Real-Time End-to-End Object Detection.pdf
- 原始 HTML：../../raw/html/Wang et al. - 2024 - YOLOv10 Real-Time End-to-End Object Detection.html
- 全文文本：../../raw/text/Wang et al. - 2024 - YOLOv10 Real-Time End-to-End Object Detection.md
- 作者：Wang et al.
- 年份：2024
- 状态：已基于 arXiv HTML 整理

## 自动抽取摘要或人工摘要

`YOLOv10` 的关键价值是把传统 YOLO 系列推进到 `NMS-free`、更接近真正 end-to-end 的阶段。它通过 consistent dual assignments 同时保留一对多监督的训练优势与一对一推理接口，说明 YOLO 路线正在吸收 `DETR` 式 end-to-end 思想，而不是一直停留在经典 NMS 依赖形态。

## 关键事实

- 论文明确把传统 YOLO 的后处理瓶颈定位为 `NMS` 依赖。
- `YOLOv10` 通过 dual assignments 在训练时同时使用一对多与一对一分配，在推理时使用一对一 head，实现 `NMS-free` 部署。
- 它还系统优化了头部、下采样、block design 与部分 self-attention，以推进速度/精度边界。
- 从系列脉络看，`YOLOv10` 说明 YOLO 与 `DETR` 并非完全对立，而是开始在 end-to-end 训练接口上接近。
- 在 `目标检测` topic 中，它是连接 classic YOLO 与新一代 end-to-end real-time detector 的关键节点。

## 争议与不确定点

- 虽然论文使用 “end-to-end” 表述，但它与 `DETR` 的 set prediction 路径并不完全相同，不能简单视为同一范式。
- 当前 summary 仅提炼其方法位置，尚未继续评估其在不同部署环境中的泛化稳定性。

## 关联页面

- 主题：[目标检测](../../wiki/topics/目标检测.md)
- 概念：[YOLO](../../wiki/concepts/YOLO.md)
- 概念：[DETR](../../wiki/concepts/DETR.md)

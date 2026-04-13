# YOLO

## 简介

`YOLO` 是目标检测中最重要的 one-stage 模型家族之一。在当前知识库中，它不仅表示 `YOLOv1` 这一开端，也表示从 `YOLOv2/9000`、`YOLOv3`、`YOLOv4` 到 `YOLOv10/11`，再到截至 `2026-04-13` 官方文档中的 `YOLO26` 这一整条实时检测系列。

## 关键属性

- 类型：目标检测模型家族
- 代表来源：
  - [Redmon et al. - 2015 - You Only Look Once Unified Real-Time Object Detection](../../raw/summary/Redmon%20et%20al.%20-%202015%20-%20You%20Only%20Look%20Once%20Unified%20Real-Time%20Object%20Detection.md)
  - [Redmon, Farhadi - 2016 - YOLO9000 Better Faster Stronger](../../raw/summary/Redmon,%20Farhadi%20-%202016%20-%20YOLO9000%20Better%20Faster%20Stronger.md)
  - [Redmon, Farhadi - 2018 - YOLOv3 An Incremental Improvement](../../raw/summary/Redmon,%20Farhadi%20-%202018%20-%20YOLOv3%20An%20Incremental%20Improvement.md)
  - [Bochkovskiy, Wang, Liao - 2020 - YOLOv4 Optimal Speed and Accuracy of Object Detection](../../raw/summary/Bochkovskiy,%20Wang,%20Liao%20-%202020%20-%20YOLOv4%20Optimal%20Speed%20and%20Accuracy%20of%20Object%20Detection.md)
  - [Wang et al. - 2024 - YOLOv10 Real-Time End-to-End Object Detection](../../raw/summary/Wang%20et%20al.%20-%202024%20-%20YOLOv10%20Real-Time%20End-to-End%20Object%20Detection.md)
  - [Chen et al. - 2025 - A Comprehensive Survey of YOLO From YOLOv1 to YOLO11 and Beyond](../../raw/summary/Chen%20et%20al.%20-%202025%20-%20A%20Comprehensive%20Survey%20of%20YOLO%20From%20YOLOv1%20to%20YOLO11%20and%20Beyond.md)
  - [Ultralytics - 2026 - Ultralytics YOLO Docs Home](../../raw/summary/Ultralytics%20-%202026%20-%20Ultralytics%20YOLO%20Docs%20Home.md)
- 当前角色：`目标检测` topic 中 one-stage 主线的总入口

## 相关主张

- `YOLOv1` 把检测改写成单阶段回归问题，奠定 one-stage 实时检测主线。
- `YOLOv2 / YOLO9000`、`YOLOv3` 把该主线推进为更成熟的 anchor-based、多尺度、可扩展家族。
- `YOLOv4` 代表 YOLO 进入 recipe / engineering optimization 阶段，强调单 GPU 可训练、可部署的高效实践。
- `YOLOv10` 说明新一代 YOLO 已开始吸收 `NMS-free` 与一对一推理接口思想，向更强 end-to-end 部署形式靠近。
- 根据 `2025` 综述，YOLO 系列可以按 backbone、neck、head、loss/assignment、training strategy 五条改进轴理解，而不应只按版本号罗列。
- 根据官方文档，截至 `2026-04-13`，Ultralytics 将 `YOLO26` 作为最新版本，并同时推荐 `YOLO26` 与 `YOLO11` 用于稳定生产负载。

## 来源支持

- [Redmon et al. - 2015 - You Only Look Once Unified Real-Time Object Detection](../../raw/summary/Redmon%20et%20al.%20-%202015%20-%20You%20Only%20Look%20Once%20Unified%20Real-Time%20Object%20Detection.md)
- [Redmon, Farhadi - 2016 - YOLO9000 Better Faster Stronger](../../raw/summary/Redmon,%20Farhadi%20-%202016%20-%20YOLO9000%20Better%20Faster%20Stronger.md)
- [Redmon, Farhadi - 2018 - YOLOv3 An Incremental Improvement](../../raw/summary/Redmon,%20Farhadi%20-%202018%20-%20YOLOv3%20An%20Incremental%20Improvement.md)
- [Bochkovskiy, Wang, Liao - 2020 - YOLOv4 Optimal Speed and Accuracy of Object Detection](../../raw/summary/Bochkovskiy,%20Wang,%20Liao%20-%202020%20-%20YOLOv4%20Optimal%20Speed%20and%20Accuracy%20of%20Object%20Detection.md)
- [Wang et al. - 2024 - YOLOv10 Real-Time End-to-End Object Detection](../../raw/summary/Wang%20et%20al.%20-%202024%20-%20YOLOv10%20Real-Time%20End-to-End%20Object%20Detection.md)
- [Chen et al. - 2025 - A Comprehensive Survey of YOLO From YOLOv1 to YOLO11 and Beyond](../../raw/summary/Chen%20et%20al.%20-%202025%20-%20A%20Comprehensive%20Survey%20of%20YOLO%20From%20YOLOv1%20to%20YOLO11%20and%20Beyond.md)
- [Ultralytics - 2026 - Ultralytics YOLO Docs Home](../../raw/summary/Ultralytics%20-%202026%20-%20Ultralytics%20YOLO%20Docs%20Home.md)
- [目标检测](../topics/目标检测.md)

## 关联页面

- [Faster R-CNN](./Faster%20R-CNN.md)
- [DETR](./DETR.md)
- [目标检测](../topics/目标检测.md)
- [传统 CV](../topics/传统%20CV.md)

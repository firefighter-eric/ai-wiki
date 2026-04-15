# Bochkovskiy, Wang, Liao - 2020 - YOLOv4 Optimal Speed and Accuracy of Object Detection

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Bochkovskiy, Wang, Liao - 2020 - YOLOv4 Optimal Speed and Accuracy of Object Detection.pdf
- 原始 HTML：../../raw/html/Bochkovskiy, Wang, Liao - 2020 - YOLOv4 Optimal Speed and Accuracy of Object Detection.html
- 全文文本：../../raw/text/Bochkovskiy, Wang, Liao - 2020 - YOLOv4 Optimal Speed and Accuracy of Object Detection.md
- 作者：Bochkovskiy, Wang, Liao
- 年份：2020
- 状态：已基于 arXiv HTML 整理

## 自动抽取摘要或人工摘要

`YOLOv4` 的意义在于把 YOLO 系列推进为“高效工程优化集合”的代表路线。它并不主要依赖单一新原理，而是系统整合 `CSP`、`Mosaic`、`CIoU loss`、`PAN` 等 bag-of-freebies / bag-of-specials，使 one-stage 检测在单 GPU 可训练、可部署语境中达到更高实用性。

## 关键事实

- 论文将检测器拆成 backbone / neck / head，并系统讨论可带来收益的训练与结构技巧。
- `YOLOv4` 强调单 GPU 即可训练并达到高实时性能，体现了很强的工程可用性导向。
- 其路线核心是“组合优化”，而不是重新定义检测任务接口。
- 从知识库视角看，`YOLOv4` 代表 YOLO 进入大规模工程 recipe 竞争阶段。
- 它也帮助确立 YOLO 在实时检测中的主流地位，而不再只是单一论文模型。

## 争议与不确定点

- 由于其贡献高度组合化，哪些提升来自结构设计、哪些来自训练 recipe，需要更细化的比较页才能稳定表达。
- 当前 summary 关注其系列定位，尚未细拆所有 bag-of-freebies 的边际贡献。

## 关联页面

- 主题：[目标检测](../../wiki/topics/目标检测.md)
- 概念：[YOLO](../../wiki/concepts/YOLO.md)

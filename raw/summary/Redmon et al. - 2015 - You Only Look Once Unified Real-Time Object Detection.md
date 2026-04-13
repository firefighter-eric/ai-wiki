# Redmon et al. - 2015 - You Only Look Once Unified Real-Time Object Detection

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Redmon et al. - 2015 - You Only Look Once Unified Real-Time Object Detection.pdf
- 原始 HTML：../../raw/html/Redmon et al. - 2015 - You Only Look Once Unified Real-Time Object Detection.html
- 全文文本：../../raw/text/Redmon et al. - 2015 - You Only Look Once Unified Real-Time Object Detection.md
- 作者：Redmon et al.
- 年份：2015
- 状态：已基于 arXiv HTML 整理

## 自动抽取摘要或人工摘要

`YOLOv1` 的关键突破是把目标检测从“proposal + 分类 + 后处理”的复杂流水线重写成单阶段回归问题：模型在一次前向传播中直接预测边界框与类别概率。它强调全图上下文建模与实时推理速度，标志着 `one-stage detection` 作为独立主线的正式成形。

## 关键事实

- 论文将目标检测明确表述为从整图到边界框与类别概率的单一回归问题。
- `YOLOv1` 以单网络直接输出检测结果，弱化了 `R-CNN` 系列依赖的 proposal pipeline。
- 文中强调其优势是高吞吐实时检测与更少的背景误检，而不是绝对最强的定位精度。
- 论文也明确承认其局限：小目标、密集目标与精确定位能力不足。
- 从知识库主线看，`YOLOv1` 不是 proposal-based 检测的微调版本，而是 one-stage 检测范式的出发点。

## 争议与不确定点

- `YOLOv1` 的“end-to-end”主要指单阶段统一预测，并不等于今天 `NMS-free` 的 end-to-end 定义。
- 其速度优势建立在 2015 年检测基线与硬件语境下，不能直接拿来和 2024-2026 的实时 detector 横向等价比较。
- 当前 summary 聚焦范式意义，尚未继续细拆损失函数与误差分析细节。

## 关联页面

- 主题：[目标检测](../../wiki/topics/目标检测.md)
- 主题：[传统 CV](../../wiki/topics/传统%20CV.md)
- 概念：[YOLO](../../wiki/concepts/YOLO.md)

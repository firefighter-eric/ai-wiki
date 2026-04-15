# DETR

## 简介

`DETR` 是把目标检测重写为 `set prediction` 的代表模型。在当前知识库中，它对应“从 anchor / NMS 依赖的经典检测流程转向 end-to-end 检测”的关键概念。

## 关键属性

- 类型：目标检测模型
- 代表来源：[Carion et al. - 2020 - End-to-End Object Detection with Transformers](../../wiki/summaries/Carion%20et%20al.%20-%202020%20-%20End-to-End%20Object%20Detection%20with%20Transformers.md)
- 当前角色：end-to-end 检测主线的起点页

## 相关主张

- `DETR` 通过 bipartite matching 与 object queries，把检测从 proposal/NMS 组合流程改写为一对一集合预测。
- 它的核心价值不只是“用了 Transformer”，而是改变了检测任务的监督形式与推理接口。
- 在当前知识库里，`RT-DETR` 和 `Co-DETR` 等来源都可被看作对 `DETR` 训练效率、实时性和监督稀疏问题的后续修正。

## 来源支持

- [Carion et al. - 2020 - End-to-End Object Detection with Transformers](../../wiki/summaries/Carion%20et%20al.%20-%202020%20-%20End-to-End%20Object%20Detection%20with%20Transformers.md)
- [Zhao et al. - 2023 - DETRs Beat YOLOs on Real-time Object Detection](../../wiki/summaries/Zhao%20et%20al.%20-%202023%20-%20DETRs%20Beat%20YOLOs%20on%20Real-time%20Object%20Detection.md)
- [Zong, Song, Liu - 2024 - DETRs with Collaborative Hybrid Assignments Training](../../wiki/summaries/Zong,%20Song,%20Liu%20-%202024%20-%20DETRs%20with%20Collaborative%20Hybrid%20Assignments%20Training.md)

## 关联页面

- [Faster R-CNN](./Faster%20R-CNN.md)
- [目标检测](../topics/目标检测.md)
- [传统 CV](../topics/传统%20CV.md)

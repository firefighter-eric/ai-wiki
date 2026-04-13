# Faster R-CNN

## 简介

`Faster R-CNN` 是经典两阶段目标检测范式的代表模型。在当前知识库中，它对应“proposal-based detection 被深度网络统一化”的关键节点。

## 关键属性

- 类型：目标检测模型
- 代表来源：[Ren et al. - 2015 - Faster R-CNN Towards Real-Time Object Detection with Region Proposal Networks](../../raw/summary/Ren%20et%20al.%20-%202015%20-%20Faster%20R-CNN%20Towards%20Real-Time%20Object%20Detection%20with%20Region%20Proposal%20Networks.md)
- 当前角色：承接 `R-CNN / Fast R-CNN` 后的统一两阶段检测主线

## 相关主张

- `Faster R-CNN` 把外部 proposal 模块替换为可学习的 `RPN`，使候选区域生成与检测网络共享特征。
- 它确立了 proposal-based、anchor-based、两阶段检测在工程上的标准范式。
- 在当前知识库里，它既是 `目标检测` topic 的历史基线，也是 `DETR` 等 end-to-end 检测方法的重要对照对象。

## 来源支持

- [Ren et al. - 2015 - Faster R-CNN Towards Real-Time Object Detection with Region Proposal Networks](../../raw/summary/Ren%20et%20al.%20-%202015%20-%20Faster%20R-CNN%20Towards%20Real-Time%20Object%20Detection%20with%20Region%20Proposal%20Networks.md)
- [Carion et al. - 2020 - End-to-End Object Detection with Transformers](../../raw/summary/Carion%20et%20al.%20-%202020%20-%20End-to-End%20Object%20Detection%20with%20Transformers.md)
- [目标检测](../topics/目标检测.md)

## 关联页面

- [DETR](./DETR.md)
- [目标检测](../topics/目标检测.md)
- [传统 CV](../topics/传统%20CV.md)

---
type: concept
---
# Tip-Adapter

## TL;DR（快速导读）

Tip-Adapter 用少量任务示例的特征与标签缓存辅助 CLIP 分类，减少重新训练的成本；还需区分可训练变体。

## 简介

Tip-Adapter 用少量任务示例的特征与标签缓存辅助 CLIP 分类，减少重新训练的成本；还需区分可训练变体。

## 具体怎么理解

新图片既与类别描述比较，也与缓存中的示例比较；缓存样本能否代表实际分布很重要。

## 关键属性

- 类型：轻量适配方法
- 代表来源：[Zhang et al. - 2022 - Tip-Adapter Training-Free Adaption of CLIP for Few-Shot Classification](../../wiki/summaries/Zhang%20et%20al.%20-%202022%20-%20Tip-Adapter%20Training-Free%20Adaption%20of%20CLIP%20for%20Few-Shot%20Classification.md)
- 当前角色：CLIP 生态中的下游适配节点

## 相关主张

- Tip-Adapter 说明强视觉基础模型可通过极轻量机制快速迁移到 few-shot 任务。
- 在当前知识库里，它是“基础模型 + 轻量适配”组合的视觉范例。

## 来源支持

- [Zhang et al. - 2022 - Tip-Adapter Training-Free Adaption of CLIP for Few-Shot Classification](../../wiki/summaries/Zhang%20et%20al.%20-%202022%20-%20Tip-Adapter%20Training-Free%20Adaption%20of%20CLIP%20for%20Few-Shot%20Classification.md)

## 关联页面

- [CLIP](./CLIP.md)
- [Prompt Tuning](./Prompt Tuning.md)
- [传统 CV](../topics/传统%20CV.md)

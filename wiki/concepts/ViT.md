---
type: concept
---
# ViT

## TL;DR（快速导读）

ViT 把图片切成图块序列交给 Transformer，在大规模训练下用于视觉任务，改变了图像的建模接口。

## 简介

ViT 把图片切成图块序列交给 Transformer，在大规模训练下用于视觉任务，改变了图像的建模接口。

## 具体怎么理解

图块像词语一样形成输入序列，但图像的位置与任务性质不同，不能直接照搬语言模型评测。

## 关键属性

- 类型：视觉基础模型
- 代表来源：[Dosovitskiy et al. - 2020 - An Image is Worth 16x16 Words Transformers for Image Recognition at Scale](../../wiki/summaries/Dosovitskiy%20et%20al.%20-%202020%20-%20An%20Image%20is%20Worth%2016x16%20Words%20Transformers%20for%20Image%20Recognition%20at%20Scale.md)
- 当前角色：连接 Transformer 与视觉规模化路线

## 相关主张

- ViT 把图像 patch 序列化，建立视觉任务上的通用 Transformer 路线。
- 在当前知识库里，它也是 CLIP、Scaling Vision Transformers 等后续方向的前置概念。

## 来源支持

- [Dosovitskiy et al. - 2020 - An Image is Worth 16x16 Words Transformers for Image Recognition at Scale](../../wiki/summaries/Dosovitskiy%20et%20al.%20-%202020%20-%20An%20Image%20is%20Worth%2016x16%20Words%20Transformers%20for%20Image%20Recognition%20at%20Scale.md)

## 关联页面

- [Transformer](./Transformer.md)
- [CLIP](./CLIP.md)
- [传统 CV](../topics/传统%20CV.md)

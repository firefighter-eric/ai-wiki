---
type: concept
---
# CLIP

## TL;DR（快速导读）

CLIP 把图片和文字映射到可比较的表示空间，用自然语言描述进行图像分类或图文检索。

## 简介

CLIP 把图片和文字映射到可比较的表示空间，用自然语言描述进行图像分类或图文检索。

## 具体怎么理解

把一张图与“猫在沙发上”“狗在草地上”两段文字比较，选择更匹配的描述；提示措辞也会影响结果。

## 关键属性

- 类型：视觉语言预训练模型
- 代表来源：[Radford et al. - 2021 - Learning Transferable Visual Models From Natural Language Supervision](../../wiki/summaries/Radford%20et%20al.%20-%202021%20-%20Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision.md)
- 当前角色：连接视觉编码器、图文对齐与下游适配方法

## 相关主张

- CLIP 把自然语言配对监督变成开放词汇视觉能力的来源。
- 在当前知识库里，它也是 Tip-Adapter 等轻量适配方法的基础前提。

## 来源支持

- [Radford et al. - 2021 - Learning Transferable Visual Models From Natural Language Supervision](../../wiki/summaries/Radford%20et%20al.%20-%202021%20-%20Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision.md)

## 关联页面

- [ViT](./ViT.md)
- [Tip-Adapter](./Tip-Adapter.md)
- [Kosmos-2](./Kosmos-2.md)
- [传统 CV](../topics/传统%20CV.md)
- [Fang et al. - 2021 - Injecting Semantic Concepts into End-to-End Image Captioning](../summaries/Fang%20et%20al.%20-%202021%20-%20Injecting%20Semantic%20Concepts%20into%20End-to-End%20Image%20Captioning.md)：ViTCAP：从视觉表示与语义概念生成 caption，区别于检索。
- [Mokady, Hertz, Bermano - 2021 - ClipCap CLIP Prefix for Image Captioning](../summaries/Mokady%2C%20Hertz%2C%20Bermano%20-%202021%20-%20ClipCap%20CLIP%20Prefix%20for%20Image%20Captioning.md)：ClipCap：把 CLIP 表示作为语言生成前缀，不能把 caption 当作事实识别保证。


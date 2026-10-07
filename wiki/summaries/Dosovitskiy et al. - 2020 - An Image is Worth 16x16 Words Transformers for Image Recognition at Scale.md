---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Dosovitskiy et al. - 2020 - An Image is Worth 16x16 Words Transformers for Image Recognition at Scale

## TL;DR（快速导读）

ViT 把图像切成小块，将图块当成序列输入 Transformer，说明图像分类也可以通过序列建模来完成。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

模型先把图块映射成向量，再用 Transformer 在图块之间交换信息。论文研究这一结构在大规模预训练下的图像分类效果。数据规模和训练方式是结论的重要条件，不能仅凭结构名称断言其总优于卷积网络。

## 具体怎么理解

例如把照片分成网格，每个格子形成一个输入元素；模型再结合不同格子的内容判断整张图。

## 关键事实

- **C1**：ViT 将图像分成固定 patch，线性投影并加位置嵌入，以 Transformer encoder 和分类 token 做图像分类。
- **C2**：在较小预训练数据上大 ViT 弱于有卷积归纳偏置的 ResNet，较大数据规模时优势出现。
- **C3**：控制规模研究以 JFT-300M 预训练，比较多种 ResNet、ViT 和混合模型的迁移效果与预训练成本。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Dosovitskiy%20et%20al.%20-%202020%20-%20An%20Image%20is%20Worth%2016x16%20Words%20Transformers%20for%20Image%20Recognition%20at%20Scale.pdf)
- 全文文本：[打开全文文本](../../raw/text/Dosovitskiy%20et%20al.%20-%202020%20-%20An%20Image%20is%20Worth%2016x16%20Words%20Transformers%20for%20Image%20Recognition%20at%20Scale.md)
- 作者：Dosovitskiy et al.
- 年份：2020
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Dosovitskiy%20et%20al.%20-%202020%20-%20An%20Image%20is%20Worth%2016x16%20Words%20Transformers%20for%20Image%20Recognition%20at%20Scale.html)

## 争议与不确定点

- JFT 等大数据预训练条件与普通小数据项目差距很大。
- 分类结果不直接证明检测、分割或文档阅读能力。
- 位置插值与更高分辨率微调改变输入序列长度，成本随之变化。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无

## 方法与实验解读

把图像当作一串小块可以复用序列模型，但模型需从训练数据学会图像结构。卷积预先带来局部和平移方面的偏置，ViT 用更大数据换取更灵活的表示，两条路线的优劣取决于资源和任务。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Dosovitskiy%20et%20al.%20-%202020%20-%20An%20Image%20is%20Worth%2016x16%20Words%20Transformers%20for%20Image%20Recognition%20at%20Scale.md#source-section-6 ) | patch 大小是模型超参数；标题 16×16 不代表所有实验都只使用该 patch。 |
| C2 | [原文]( ../../raw/text/Dosovitskiy%20et%20al.%20-%202020%20-%20An%20Image%20is%20Worth%2016x16%20Words%20Transformers%20for%20Image%20Recognition%20at%20Scale.md#source-section-13 ) | 纯 Transformer 的成功依赖数据规模与训练设置，不能推出小数据总优于 CNN。 |
| C3 | [原文]( ../../raw/text/Dosovitskiy%20et%20al.%20-%202020%20-%20An%20Image%20is%20Worth%2016x16%20Words%20Transformers%20for%20Image%20Recognition%20at%20Scale.md#source-section-14 ) | 预训练成本比较要同时控制数据、epoch 与模型配置；不是部署速度比较。 |

## 核证范围

核对 §3.1 patch 与 encoder、§4.1 训练设置、§4.3 数据需求、§4.4 控制规模比较。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

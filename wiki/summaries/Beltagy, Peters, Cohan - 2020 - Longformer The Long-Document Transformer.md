# Beltagy, Peters, Cohan - 2020 - Longformer The Long-Document Transformer

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Beltagy, Peters, Cohan - 2020 - Longformer The Long-Document Transformer.pdf
- 原始 HTML：../../raw/html/Beltagy, Peters, Cohan - 2020 - Longformer The Long-Document Transformer.html
- 全文文本：../../raw/text/Beltagy, Peters, Cohan - 2020 - Longformer The Long-Document Transformer.md
- 作者：Beltagy, Peters, Cohan
- 年份：2020
- 状态：已基于 arXiv HTML 整理

## 自动抽取摘要或人工摘要

`Longformer` 代表长文档建模里最实用的一类 attention：`local window + task-motivated global attention`。它不试图近似 full attention 的所有连接，而是承认大多数 token 主要依赖局部邻域，仅为少数任务关键 token 开放全局访问，因此可以把复杂度降为线性并保持对文档任务友好。

## 关键事实

- `Longformer` 使用固定窗口的局部 attention 处理大多数 token，并为少量关键 token 分配全局 attention。
- 该设计直接面向长文档分类、问答、摘要等任务，而不是只服务于自回归语言建模。
- 论文强调它可以作为 pretrained Transformer 的 drop-in replacement，说明其工程目标是“兼容现有预训练范式”。
- `Longformer` 还扩展出 `LED`，把这一路线推进到 encoder-decoder 的长文档生成任务。
- 在 attention 主线中，`Longformer` 代表“局部窗口 + 全局 token”的长文档 attention 路线。

## 争议与不确定点

- 该路线依赖对“哪些 token 应该拥有全局视野”的任务设计，因此通用性部分来自经验选择，而非完全数据驱动。
- 线性复杂度不等于任意任务都优于 full attention；它本质上对连接图做了结构性偏置。

## 关联页面

- 主题：[注意力机制 Attention](../../wiki/topics/注意力机制%20Attention.md)
- 概念：[Transformer](../../wiki/concepts/Transformer.md)

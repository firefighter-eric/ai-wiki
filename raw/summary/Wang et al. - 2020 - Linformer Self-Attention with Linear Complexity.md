# Wang et al. - 2020 - Linformer Self-Attention with Linear Complexity

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Wang et al. - 2020 - Linformer Self-Attention with Linear Complexity.pdf
- 原始 HTML：../../raw/html/Wang et al. - 2020 - Linformer Self-Attention with Linear Complexity.html
- 全文文本：../../raw/text/Wang et al. - 2020 - Linformer Self-Attention with Linear Complexity.md
- 作者：Wang et al.
- 年份：2020
- 状态：已基于 arXiv HTML 整理

## 自动抽取摘要或人工摘要

`Linformer` 代表 efficient attention 中最典型的低秩投影路线。它的核心判断是：标准 self-attention 形成的上下文映射矩阵在实践中可被低秩近似，因此可以先把序列维上的 `K/V` 投影到更低维空间，再计算 attention，把复杂度从二次压到线性。

## 关键事实

- 论文明确将标准 self-attention 的瓶颈定位为序列长度上的二次时间与空间复杂度。
- `Linformer` 的主要方法是对 keys / values 做线性投影，以低秩近似注意力矩阵，而不是改变 softmax 形式本身。
- 作者主张 self-attention matrix 具有低秩结构，这一经验性和理论性观察构成了其方法前提。
- 该路线的优点是结构简单、易嵌入原有 Transformer；代价是需要接受低秩假设不总是精确成立。
- 在 attention 主线中，`Linformer` 适合作为“低秩近似 attention”的代表节点。

## 争议与不确定点

- 低秩假设在不同任务、层数和序列长度下成立程度并不一致，因此它更像一类近似假设，而不是对标准 attention 的普适等价重写。
- 线性复杂度在理论上成立，但实际吞吐仍取决于投影维度、kernel 实现和序列长度。

## 关联页面

- 主题：[注意力机制 Attention](../../wiki/topics/注意力机制%20Attention.md)
- 概念：[Transformer](../../wiki/concepts/Transformer.md)

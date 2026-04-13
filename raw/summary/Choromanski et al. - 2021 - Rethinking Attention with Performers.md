# Choromanski et al. - 2021 - Rethinking Attention with Performers

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Choromanski et al. - 2021 - Rethinking Attention with Performers.pdf
- 原始 HTML：../../raw/html/Choromanski et al. - 2021 - Rethinking Attention with Performers.html
- 全文文本：../../raw/text/Choromanski et al. - 2021 - Rethinking Attention with Performers.md
- 作者：Choromanski et al.
- 年份：2021
- 状态：已基于 arXiv HTML 整理

## 自动抽取摘要或人工摘要

`Performer` 是线性 attention 路线中最有代表性的“核函数近似 softmax”方案之一。它用 `FAVOR+` 把 softmax attention 重写为正随机特征下的线性形式，目标是在不预设稀疏结构或低秩结构的前提下，以线性时间和空间近似标准 attention。

## 关键事实

- 论文将其方法定位为对标准 softmax full attention 的近似，而不是重新定义一套完全不同的注意力语义。
- `FAVOR+` 的核心是用正交随机特征近似 softmax kernel，从而避免显式构造 `n x n` attention matrix。
- 与 `Linformer` 的低秩假设、`Longformer` 的结构稀疏不同，`Performer` 属于核技巧驱动的线性化路线。
- 论文强调其理论性质，如近似精度、低方差与可兼容标准 Transformer 的微调迁移。
- 在当前 topic 中，`Performer` 代表“kernelized / random feature linear attention”分支。

## 争议与不确定点

- 尽管复杂度线性，但近似质量依赖随机特征数与具体实现，工程上并非“免费替代”标准 attention。
- 它近似的是 softmax attention，而不是在任何场景下都能复现 full attention 的全部行为。

## 关联页面

- 主题：[注意力机制 Attention](../../wiki/topics/注意力机制%20Attention.md)
- 概念：[Transformer](../../wiki/concepts/Transformer.md)

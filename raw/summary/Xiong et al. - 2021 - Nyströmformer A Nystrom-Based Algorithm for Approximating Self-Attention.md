# Xiong et al. - 2021 - Nyströmformer A Nystrom-Based Algorithm for Approximating Self-Attention

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Xiong et al. - 2021 - Nyströmformer A Nystrom-Based Algorithm for Approximating Self-Attention.pdf
- 原始 HTML：../../raw/html/Xiong et al. - 2021 - Nyströmformer A Nystrom-Based Algorithm for Approximating Self-Attention.html
- 全文文本：../../raw/text/Xiong et al. - 2021 - Nyströmformer A Nystrom-Based Algorithm for Approximating Self-Attention.md
- 作者：Xiong et al.
- 年份：2021
- 状态：已基于 arXiv HTML 整理

## 自动抽取摘要或人工摘要

`Nyströmformer` 代表 efficient attention 中基于 landmark / Nyström 近似的路线。它把 softmax attention matrix 看成可由少量代表性列与行重建的对象，通过 landmark points 逼近完整 attention，从而在保持 global interaction 语义的同时把复杂度降到线性级别。

## 关键事实

- 论文的核心思想是把 Nyström method 引入 self-attention，对 softmax matrix 做低成本重建。
- 它与 `Linformer` 同属“近似 full attention”路线，但近似方式不是固定投影，而是依赖 landmark 表示。
- `Nyströmformer` 明确以长序列任务为目标，并在 `LRA` 上与其他 efficient attention 方法比较。
- 该方法保留“全局 token 两两潜在交互”的建模意图，只是通过近似避免显式构造全矩阵。
- 在 attention 主线中，它适合归为“landmark / Nyström 近似 attention”分支。

## 争议与不确定点

- 近似质量依赖 landmark 数量与构造方式，实际效果受任务分布影响较大。
- 它和 `Linformer`、`Performer` 一样，提供的是不同近似偏置，而不是普适最优替代。

## 关联页面

- 主题：[注意力机制 Attention](../../wiki/topics/注意力机制%20Attention.md)
- 概念：[Transformer](../../wiki/concepts/Transformer.md)

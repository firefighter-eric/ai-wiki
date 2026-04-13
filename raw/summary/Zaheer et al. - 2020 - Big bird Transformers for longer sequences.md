# Zaheer et al. - 2020 - Big bird Transformers for longer sequences

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Zaheer et al. - 2020 - Big bird Transformers for longer sequences.pdf
- 原始 HTML：../../raw/html/Zaheer et al. - 2020 - Big bird Transformers for longer sequences.html
- 全文文本：../../raw/text/Zaheer et al. - 2020 - Big bird Transformers for longer sequences.md
- 作者：Zaheer et al.
- 年份：2020
- 状态：已基于 arXiv HTML 整理

## 自动抽取摘要或人工摘要

`BigBird` 的核心贡献是把长序列 sparse attention 从“经验上能跑得更长”推进到“仍保留 Transformer 关键表达性质”的理论与实践组合。它使用 `global + local + random` 的混合稀疏连接模式，把复杂度压到近线性，同时试图保留全连接 attention 的信息流与表达能力。

## 关键事实

- `BigBird` 的注意力模式由三部分组成：少量全局 token、局部窗口 token、随机连接 token。
- 该设计直接针对标准全连接 self-attention 的 `O(n^2)` 代价，目标是在长文档、问答、摘要等任务中支持更长上下文。
- 论文强调其理论性质：在加入少量全局 token 后，稀疏 attention 仍可保持 universal approximation 与 Turing completeness 这类关键表达结论。
- 与只做局部窗口的路线相比，`BigBird` 更强调“长程信息桥接”与理论可解释性，而不是仅靠卷积式邻域传播。
- 在当前 topic 语境中，`BigBird` 代表的是“混合稀疏 attention”路线，而不是所有 sparse attention 的唯一实现。

## 争议与不确定点

- `BigBird` 的理论结论依赖特定稀疏模式与全局 token 假设，不能简单外推到任意稀疏掩码。
- 尽管复杂度是线性的，但实际工程收益仍受 kernel 实现、硬件和任务长度分布影响。
- 当前 summary 把重点放在稀疏模式与主线地位，未展开全部实验与证明细节。

## 关联页面

- 主题：[注意力机制 Attention](../../wiki/topics/注意力机制%20Attention.md)
- 概念：[Transformer](../../wiki/concepts/Transformer.md)

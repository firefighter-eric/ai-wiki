# Kitaev, Kaiser, Levskaya - 2020 - Reformer The Efficient Transformer

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Kitaev, Kaiser, Levskaya - 2020 - Reformer The Efficient Transformer.pdf
- 原始 HTML：../../raw/html/Kitaev, Kaiser, Levskaya - 2020 - Reformer The Efficient Transformer.html
- 全文文本：../../raw/text/Kitaev, Kaiser, Levskaya - 2020 - Reformer The Efficient Transformer.md
- 作者：Kitaev, Kaiser, Levskaya
- 年份：2020
- 状态：已基于 arXiv HTML 整理

## 自动抽取摘要或人工摘要

`Reformer` 代表的是基于 `LSH attention` 的近邻检索式稀疏路线。它不再让每个 token 与全部 token 做精确配对，而是先用局部敏感哈希把相似 token 分桶，再在桶内做 attention，从而把复杂度从 `O(n^2)` 下降到约 `O(n log n)`；同时它还结合 reversible layers 解决训练显存问题。

## 关键事实

- `Reformer` 的 attention 近似核心是 `LSH attention`，即通过哈希分桶来减少需要比较的 token 对。
- 论文同时引入 reversible residual layers 与分块前馈计算，说明它是“attention 近似 + 训练内存优化”的组合方案，而不只是掩码设计。
- 与局部窗口 attention 不同，`Reformer` 试图通过内容相似性而不是固定相对位置来组织稀疏连接。
- 其目标场景是超长序列，其中全连接 attention 的计算与显存成本最难承受。
- 在当前 topic 中，`Reformer` 代表“基于哈希 / 检索的稀疏 attention”分支。

## 争议与不确定点

- 由于哈希分桶、排序与多轮哈希带来额外常数项，理论复杂度优势并不总能直接转换成短序列上的实际优势。
- 这种方法对近似误差与实现细节较敏感，因此后续影响更多体现在“提供一种方向”，而不是成为统一标准实现。

## 关联页面

- 主题：[注意力机制 Attention](../../wiki/topics/注意力机制%20Attention.md)
- 概念：[Transformer](../../wiki/concepts/Transformer.md)

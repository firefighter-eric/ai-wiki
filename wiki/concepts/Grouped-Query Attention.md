# Grouped-Query Attention

## 简介

`Grouped-Query Attention (GQA)` 是介于标准 `Multi-Head Attention (MHA)` 与 `Multi-Query Attention (MQA)` 之间的折中结构。它让一组 query heads 共享一组 `K/V`，以较小质量损失换取更低的推理 `KV cache` 成本。

## 关键属性

- 类型：attention 结构变体 / 推理优化
- 代表来源：[Ainslie et al. - 2023 - GQA Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](../../wiki/summaries/Ainslie%20et%20al.%20-%202023%20-%20GQA%20Training%20Generalized%20Multi-Query%20Transformer%20Models%20from%20Multi-Head%20Checkpoints.md)
- 相关前身：[Shazeer - 2019 - Fast Transformer Decoding One Write-Head is All You Need](../../wiki/summaries/Shazeer%20-%202019%20-%20Fast%20Transformer%20Decoding%20One%20Write-Head%20is%20All%20You%20Need.md)

## 相关主张

- `GQA` 的主要价值是把 `MHA` 与 `MQA` 之间的离散选择改写为可调节的结构谱系。
- 它是现代 LLM 解码加速里非常常见的 attention 设计，因为它通常比 `MQA` 更稳、又比 `MHA` 更省缓存。

## 来源支持

- [Shazeer - 2019 - Fast Transformer Decoding One Write-Head is All You Need](../../wiki/summaries/Shazeer%20-%202019%20-%20Fast%20Transformer%20Decoding%20One%20Write-Head%20is%20All%20You%20Need.md)
- [Ainslie et al. - 2023 - GQA Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](../../wiki/summaries/Ainslie%20et%20al.%20-%202023%20-%20GQA%20Training%20Generalized%20Multi-Query%20Transformer%20Models%20from%20Multi-Head%20Checkpoints.md)

## 关联页面

- [Transformer](./Transformer.md)
- [注意力机制 Attention](../topics/注意力机制%20Attention.md)

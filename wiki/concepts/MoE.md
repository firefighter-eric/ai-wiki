---
type: concept
---
# MoE

## 简介

MoE 是 Mixture-of-Experts 的缩写。在当前知识库中，它表示“只激活部分专家参数以提升容量与效率”的稀疏模型架构路线。

## 关键属性

- 类型：模型架构 / 稀疏激活方法
- 代表来源：
  - [Fedus, Zoph, Shazeer - 2022 - Switch Transformers Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](../../wiki/summaries/Fedus,%20Zoph,%20Shazeer%20-%202022%20-%20Switch%20Transformers%20Scaling%20to%20Trillion%20Parameter%20Models%20with%20Simple%20and%20Efficient%20Sparsity.md)
  - [Unknown - 2024 - DeepSeek-V3 Technical Report](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)
  - [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](../../wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)
  - [Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence](../../wiki/summaries/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)
  - [Google DeepMind - 2026 - Gemma 4 Model Card](../summaries/Google%20DeepMind%20-%202026%20-%20Gemma%204%20Model%20Card.md)
  - [Google DeepMind - 2026 - DiffusionGemma 26B A4B IT Model Card](../summaries/Google%20DeepMind%20-%202026%20-%20DiffusionGemma%2026B%20A4B%20IT%20Model%20Card.md)
  - [Zuo et al. - 2022 - MoEBERT from BERT to Mixture-of-Experts via Importance-Guided Adaptation](../../wiki/summaries/Zuo%20et%20al.%20-%202022%20-%20MoEBERT%20from%20BERT%20to%20Mixture-of-Experts%20via%20Importance-Guided%20Adaptation.md)
- 当前角色：连接大规模稀疏训练与开源模型效率工程的结构概念

## 相关主张

- `Fedus et al. 2022` 强调 MoE 通过稀疏激活在近似固定计算成本下扩展总参数规模。
- 在当前知识库里，MoE 不只属于早期 Switch Transformer 路线，也已通过 `DeepSeek-V3 / DeepSeek-V4` 进入当代开放大模型主线。
- `DeepSeek-V4` 进一步说明 MoE 的竞争点已经从“总参数更大”延伸到“激活参数、长上下文 attention 成本与 KV cache 成本如何共同优化”。
- `Kimi K3` 把这一竞争推进到约 `2.78T / 104.2B activated` 的原生多模态模型，并用 `Stable LatentMoE` 在 896 个 routed experts 中激活 16 个；其重点不是专家数本身，而是 latent routed path、activation stabilization、Quantile Balancing 与 MoonEP 的联合可执行性。
- K3 还说明 MoE load balancing 至少分两层：`Quantile Balancing` 改善 expert-level dispatch 分布，`MoonEP` 则在给定路由结果上保证 EP rank-level 完全平衡；二者不能合并成同一个 router 算法。
- `Gemma 4 26B A4B` 与 `DiffusionGemma` 说明 MoE 也正在成为开放多模态和非自回归文本生成实验的效率底座。
- `Zuo et al. 2022` 说明 MoE 也可作为从致密模型迁移到专家结构的一种改造思路，而不只服务于超大预训练模型。

## 来源支持

- [Fedus, Zoph, Shazeer - 2022 - Switch Transformers Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](../../wiki/summaries/Fedus,%20Zoph,%20Shazeer%20-%202022%20-%20Switch%20Transformers%20Scaling%20to%20Trillion%20Parameter%20Models%20with%20Simple%20and%20Efficient%20Sparsity.md)
- [Unknown - 2024 - DeepSeek-V3 Technical Report](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)
- [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](../../wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)
- [Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence](../../wiki/summaries/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)
- [Google DeepMind - 2026 - Gemma 4 Model Card](../summaries/Google%20DeepMind%20-%202026%20-%20Gemma%204%20Model%20Card.md)
- [Google DeepMind - 2026 - DiffusionGemma 26B A4B IT Model Card](../summaries/Google%20DeepMind%20-%202026%20-%20DiffusionGemma%2026B%20A4B%20IT%20Model%20Card.md)
- [Zuo et al. - 2022 - MoEBERT from BERT to Mixture-of-Experts via Importance-Guided Adaptation](../../wiki/summaries/Zuo%20et%20al.%20-%202022%20-%20MoEBERT%20from%20BERT%20to%20Mixture-of-Experts%20via%20Importance-Guided%20Adaptation.md)

## 关联页面

- [DeepSeek](./DeepSeek.md)
- [DeepSeek-V3](./DeepSeek-V3.md)
- [DeepSeek-V4](./DeepSeek-V4.md)
- [Kimi K3](./Kimi%20K3.md)
- [Stable LatentMoE](./Stable%20LatentMoE.md)
- [Quantile Balancing](./Quantile%20Balancing.md)
- [MoonEP](./MoonEP.md)
- [Gemma 4](./Gemma%204.md)
- [DiffusionGemma](./DiffusionGemma.md)
- [LLM 预训练](../topics/LLM%20预训练.md)
- [传统 NLP](../topics/传统%20NLP.md)

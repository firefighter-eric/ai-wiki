---
type: comparison
---
# Muon 与 AdamW

## 比较范围

本页比较现代 LLM 训练中的 `Muon family` 与 `AdamW`。重点不是把它们写成互斥阵营，而是回答三个问题：更新几何有什么不同；Muon's reported efficiency 来自哪里；前沿模型在什么参数和训练阶段实际采用了什么。

## 核心结论

`AdamW` 是逐元素自适应优化器，通用、成熟、易于 ZeRO/FSDP 分片；`Muon` 是二维矩阵专用优化器，用 momentum + Newton–Schulz 近似 `UV^T`，尝试平衡矩阵更新的奇异方向。Muon 在若干预训练报告中显示更强 loss/token 或 loss/FLOP efficiency，但需要矩阵重组、更新尺度校准与额外稳定性机制。

现实中的最常见 Muon 配置是混合式：大型 hidden matrices 用 Muon，embedding、normalization、bias 和 prediction head 用 AdamW。

## 算法与工程对照

| 维度 | AdamW | Muon |
| --- | --- | --- |
| 更新粒度 | 参数元素 | 完整二维矩阵 |
| 主要状态 | first moment + second moment | momentum；另有 NS 临时 buffers |
| 预条件/归一化 | `m / (sqrt(v)+ε)`，逐元素 | 将 momentum matrix 的奇异值推向相近尺度 |
| 是否 Hessian 二阶法 | 否 | 否 |
| weight decay | 与 loss-gradient update 解耦 | 可扩展版本同样使用 decoupled WD |
| 参数范围 | 矩阵、向量、标量均可 | 主要是二维 hidden matrices |
| 每步运算 | element-wise 为主 | 多轮 BF16 GEMMs |
| 分布式实现 | 分片后可局部更新 | NS 需要 logically complete matrix |
| 典型优势 | 稳健、生态成熟、适合各种训练阶段 | 预训练 sample/compute efficiency、较少持久状态、matrix-aware geometry |
| 典型风险 | 两个 state tensors 占内存；不直接利用矩阵结构 | 通信/同步、scale tuning、attention logits、pretrain-finetune mismatch |

## `52% FLOPs` 应如何理解

Moonlight 的 scaling-law 结论是：在作者定义的 compute-optimal 训练设定下，Muon 达到 AdamW 可比 validation loss 约需要 `52%` 的 training FLOPs。因此它近似表示 `1 / 0.52 ≈ 1.92×` 的达到目标 loss 的计算效率。

它不表示：

- 模型参数扩大到 2 倍；
- 单步 wall-clock 加速 2 倍；
- loss 数值降低到 AdamW 的 1/2；
- 任意模型和数据都能复现同一倍率。

严谨说法应是“Moonlight 作者在其 scaling-law 设置下报告约 2× compute efficiency”，并同时说明 Muon 单步还有 Newton–Schulz 与通信开销。

## 模型采用证据

| 证据类别 | 模型 | 公开披露 |
| --- | --- | --- |
| Muon family | Moonlight | Scalable Muon：WD + update RMS + Distributed Muon |
| Muon family | Kimi K2 | MuonClip；继承 Moonlight recipe，但未独立公布 AdamW fallback 清单 |
| Muon family | Kimi K3 | matrix parameters 用 Muon，Q/K/V 用 Per-Head Muon；其余 parameter groups 未披露 |
| Muon family | DeepSeek-V4 | 多数参数用 Hybrid Muon，特殊参数保留 AdamW |
| Muon family | Motif-2-12.7B | MuonClip + all-to-all Parallel Muon |
| AdamW | Llama 3 405B | 明确披露 AdamW |
| AdamW | DeepSeek-V3 | AdamW `.9/.95`，WD `.1` |
| AdamW | OLMo 2 | AdamW `ε=1e-8`，embedding 不做 WD |
| 未披露 | Qwen3 | 公开报告/发布材料没有给出 optimizer 名称 |
| 未披露 | Gemma 3 | 只披露 ZeRO-3 optimizer-state sharding，未披露算法 |

这张表不能当作排行榜：各模型的架构、数据、token budget、训练精度与系统栈不同，只有同模型、同数据、调优 baseline 的受控实验才能直接比较 optimizer quality。

K2/K3 也不能与 DeepSeek-V4 使用同一“已确认混合分组”标签：Moonlight 明确把 RMSNorm、LM head、embedding 留给 AdamW；K2 只通过上游 recipe 间接支持这一判断；K3 仅明确 matrix parameters 用 Muon。RMSNorm 等 1-D 参数确定不走 Muon，但 K3 没有命名 fallback；embedding/head 是 2-D，是否排除也未直接披露。

## 什么时候更适合选哪一个

### 优先 AdamW

- 微调现有 AdamW-pretrained checkpoint，尤其预算有限且没有 optimizer 消融时。
- 参数形态复杂，包含大量向量、标量、embedding 或小矩阵。
- 训练栈依赖成熟的 ZeRO/FSDP、checkpoint、offload 与 optimizer fusion。
- 首要目标是低工程风险和可复现性，而不是探索预训练 loss/FLOP frontier。

### 可以评估 Muon

- 从头预训练 Transformer，主要计算来自大二维矩阵。
- token / training FLOPs 很昂贵，值得用小规模 scaling runs 验证 loss/token 与 loss/FLOP。
- 能实现 Q/K/V 独立参数组、shape-aware RMS scaling、AdamW fallback 与分布式完整矩阵处理。
- 有能力监控 max attention logits、update RMS、weight RMS、loss/grad spikes 与端到端 tokens/sec。

## 证据边界

- Muon 的大规模正面证据目前主要来自 Moonshot/Kimi、DeepSeek 与 Motif 的团队报告，独立、跨架构复现仍有限。
- Moonlight 的 SFT 消融提示 optimizer mismatch：预训练优化器和微调优化器不同，Muon 的优势可能明显减弱。
- K2 的 QK-Clip 和 DeepSeek-V4 的 Q/KV normalization 表明，Muon 的稳定性必须与 attention architecture 一起设计。
- “一个 momentum buffer”是 Muon 参数组的状态优势，不等于完整训练进程显存正好减半。

## 来源支持

- [Adam](../summaries/Kingma%20and%20Ba%20-%202015%20-%20Adam%20A%20Method%20for%20Stochastic%20Optimization.md)
- [AdamW](../summaries/Loshchilov%20and%20Hutter%20-%202019%20-%20Decoupled%20Weight%20Decay%20Regularization.md)
- [原始 Muon](../summaries/Keller%20Jordan%20-%202024%20-%20Muon%20An%20Optimizer%20for%20Hidden%20Layers%20in%20Neural%20Networks.md)
- [Muon is Scalable / Moonlight](../summaries/Liu%20et%20al.%20-%202025%20-%20Muon%20is%20Scalable%20for%20LLM%20Training.md)
- [Kimi K2](../summaries/Kimi%20Team%20-%202025%20-%20Kimi%20K2%20Open%20Agentic%20Intelligence.md)
- [Kimi K3](../summaries/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)
- [DeepSeek-V4](../summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)
- [Motif-2-12.7B](../summaries/Lim%20et%20al.%20-%202025%20-%20Motif%202%2012.7B%20Technical%20Report.md)
- [Llama 3](../summaries/Team,%20Meta%20-%202024%20-%20The%20Llama%203%20Herd%20of%20Models.md)
- [DeepSeek-V3](../summaries/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)
- [OLMo 2](../summaries/Team%20OLMo%20-%202025%20-%202%20OLMo%202%20Furious.md)
- [Qwen3](../summaries/Qwen%20Team%20-%202025%20-%20Qwen3%20Think%20Deeper%20Act%20Faster.md)
- [Gemma 3](../summaries/Team,%20Deepmind%20-%202025%20-%20Gemma%203%20Technical%20Report.md)

## 关联页面

- [Muon](../concepts/Muon.md)
- [Kimi K3](../concepts/Kimi%20K3.md)
- [DeepSeek-V4](../concepts/DeepSeek-V4.md)
- [LLM 预训练](../topics/LLM%20预训练.md)

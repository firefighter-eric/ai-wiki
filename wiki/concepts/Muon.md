# Muon

## 简介

`Muon`（`MomentUm Orthogonalized by Newton–Schulz`）是面向神经网络二维隐藏层权重的矩阵级优化器。它不是给 Adam 再换一个二阶矩，也不是 Hessian 型二阶优化：Muon 先形成 SGD momentum / Nesterov momentum 更新，再对整个更新矩阵做近似 polar orthogonalization，最后按目标 update RMS 重标定并施加 weight decay。

一句话概括：`AdamW` 逐元素调步长；`Muon` 调整整张矩阵更新的奇异方向。

## 关键属性

- 类型：一阶、矩阵感知、基于 momentum 与 Newton–Schulz 的优化器
- 主要适用对象：attention、FFN、MoE experts 等二维 hidden weight matrices
- 通常不使用 Muon 的参数：embedding、prediction head、normalization scale、bias、部分 router / gating parameters
- 核心持久状态：每个 Muon 参数通常一个 momentum buffer；AdamW 通常有一阶矩与二阶矩两个 state tensors
- 主要额外计算：若干轮 BF16 matrix multiplications 形式的 Newton–Schulz iterations
- 当前成熟度：已进入多个大型预训练项目，但仍是与 AdamW 混合使用的快速演进路线，不是全行业默认

## Muon 一步是怎么做的

以下写法刻意分开“算法不变量”和不同代码库的系数约定。是否把 `(1-μ)` 吸收到 gradient 或 learning rate 中，会让公式外观不同，但不改变核心结构。

### 1. 选出独立二维矩阵

将 Transformer 的 Q、K、V、O 与 FFN / expert projections 作为 logically independent matrices。原始实现建议 Q、K、V 分开处理；向量、标量、embedding 和输出 head 交给 AdamW。

### 2. 累积 momentum，并可用 Nesterov look-ahead

一种常见写法是：

$$
M_t = \mu M_{t-1} + G_t,
$$

$$
B_t = \mu M_t + G_t.
$$

其中 `B_t` 是送去正交化的 Nesterov-style update。原始公开实现通常使用 `momentum=0.95` 且默认开启 Nesterov。

### 3. 归一化后做 Newton–Schulz

先令：

$$
X_0 = \frac{B_t}{\lVert B_t\rVert_F+\epsilon}.
$$

为减少运算，若矩阵是 tall matrix，实现通常先转置，使迭代在较小维度形成 Gram matrix。随后重复约 5 次：

$$
X_{k+1}=aX_k+b(X_kX_k^\top)X_k+c(X_kX_k^\top)^2X_k,
$$

原始 tuned coefficients 为：

$$
(a,b,c)=(3.4445,-4.7750,2.0315).
$$

若 `B_t=U\Sigma V^\top`，理想目标是：

$$
\operatorname{Ortho}(B_t)=UV^\top.
$$

也就是把非零奇异值推到接近 1，而保留左右奇异向量。五轮 tuned iteration 是快速近似，不等于每次都精确计算 SVD。

### 4. 重标定 update RMS

`UV^T` 的自然 RMS 会随矩阵宽高比变化。可扩展版本因此乘以 shape-aware scale，使不同矩阵获得一致目标 RMS：

$$
O_t=s(W)\,X_K,
$$

其中 `s(W)` 由矩阵形状与目标 update RMS 决定。Moonlight 报告选用约 `0.2` 的目标 RMS；DeepSeek-V4 使用 `0.18`，说明这不是不可变常数。

### 5. 参数更新与 decoupled weight decay

最后可概括为：

$$
W_t=(1-\eta\lambda)W_{t-1}-\eta O_t.
$$

Muon 的现代大模型版本通常保留 decoupled weight decay。这里被近似正交化的是 momentum / update matrix，模型权重本身不会被强制变成正交矩阵。

## 为什么可能有效

### 更新方向更均衡

Transformer 矩阵的 SGD-momentum / Adam updates 常呈高条件数、近似低秩：少量奇异方向尺度很大，其他方向很弱。Muon 近似把不同非零奇异值拉到相近尺度，因此不会让最大方向单独支配整张矩阵更新。

这可理解为一种 matrix-aware preconditioning-like effect，但 Muon 不保存 Hessian、完整 covariance 或 Kronecker factors，不能简单称为传统二阶优化器。

### 预训练样本与计算效率

Moonlight 的 compute-optimal scaling-law 实验中，作者报告 Muon 达到 AdamW 可比 loss 约只需 `52%` training FLOPs，约等于其所称 `2×` compute efficiency。这个数字表示“达到同等 loss 所需总训练计算更少”，不是模型变大 2 倍、loss 降成 1/2.5，也不是单个 optimizer step 快 2 倍。

### 较少持久 optimizer state

对由 Muon 接管的矩阵，核心持久状态通常只有一个 momentum buffer；AdamW 要保存 first moment 与 second moment。Moonlight 的分布式实现据此报告 optimizer-state 额外内存约为 AdamW 的一半。但全模型仍有 AdamW 参数组，Newton–Schulz 还有临时 buffers，端到端显存节省要以具体实现为准。

### 适合 accelerator matmul

直接 SVD 过慢，而 Newton–Schulz 可写成少量 BF16 GEMMs。大 token batch 下，optimizer FLOPs 相对 forward/backward 可能较小；在分布式训练中，完整矩阵收集与网络拓扑往往比纯 FLOPs 更关键。

## 优势不能脱离代价理解

- **不是通用全参数替换**：标量、向量、embedding、head 和 normalization 通常仍用 AdamW。
- **完整矩阵依赖**：Newton–Schulz 需要 logically complete matrix，与 ZeRO/FSDP 的 element-wise sharding 天然冲突。
- **每步可能更慢**：额外 GEMMs、gather/scatter 和同步会增加 step time；应比较达到目标 loss 的端到端成本。
- **更新尺度需要校准**：不做 shape-aware RMS scaling 或 weight decay，Muon 扩展到大型模型时可能出现权重增长或不一致有效学习率。
- **attention stability 不是自动解决**：K2 观察到 Muon 更易出现 attention-logit explosion，因此加入 QK-Clip；DeepSeek-V4 则依靠 query/KV normalization，不使用 QK-Clip。
- **pretrain 与 finetune 可能不匹配**：Moonlight 的消融中，AdamW-pretrained checkpoint 改用 Muon-SFT 没有稳定优势；公开 Qwen2.5-7B 上 Muon-SFT 仅与 Adam-SFT 大致相当。
- **主要证据仍来自开发者报告**：不同模型同时改变架构、数据、batch 和训练系统，最终能力不能单独归因给优化器。

## Muon 家族如何演进

| 版本 | 核心增加项 | 解决的问题 |
| --- | --- | --- |
| 原始 Muon | momentum + 约 5 轮 Newton–Schulz | 快速近似 `UV^T`，平衡矩阵更新奇异方向 |
| Scalable Muon / Moonlight | weight decay + consistent update RMS + Distributed Muon | scale transfer、权重增长、ZeRO-1 分布式训练 |
| MuonClip / Kimi K2 | per-head QK weight clipping | Muon 扩展时的 attention-logit explosion |
| Per-Head Muon / Kimi K3 | Q/K/V momentum 按 head 分块正交化 | 避免不同 attention heads 在一张矩阵中互相支配 |
| Hybrid Muon / DeepSeek-V4 | 8 轮 tuned NS + 2 轮收敛型 NS | 先快速拉近奇异值，再更精确稳定到 1 |
| Parallel Muon / Motif-2 | all-to-all 分配完整矩阵、并行 NS、流水 gather/scatter | 避免所有 ranks 重复计算全部 Newton–Schulz |

## 先进模型实际用了什么

| 模型 / 报告 | 预训练优化器披露 | 重要细节 |
| --- | --- | --- |
| Moonlight（2025） | Muon | 3B activated / 16B total MoE，5.7T tokens；加入 WD 与 update RMS scaling |
| Kimi K2（2025） | MuonClip | Muon + per-head QK-Clip；继承 Moonlight recipe，但未独立列出 AdamW parameter groups |
| Motif-2-12.7B（2025） | MuonClip | 另实现 Parallel Muon 解决分布式 NS 重复计算 |
| Kimi K3（2026） | Muon + Per-Head Muon | 明确 matrix parameters 用 Muon；非矩阵 fallback 与 embedding/head 分组未披露 |
| DeepSeek-V4（2026） | Muon / AdamW 混合 | 多数模块 Muon；embedding、head、RMSNorm、部分 mHC 参数 AdamW |
| Llama 3 405B（2024） | AdamW | 明确披露 AdamW 与 FSDP-sharded optimizer states |
| DeepSeek-V3（2024） | AdamW | `β1=.9`、`β2=.95`、WD `.1`；一、二阶矩以 BF16 保存 |
| OLMo 2（2025） | AdamW | `ε` 调到 `1e-8`，embedding 排除 weight decay |
| Qwen3（2025） | 未披露 | 当前公开报告/发布材料未给出 optimizer 名称，不能推测 |
| Gemma 3（2025） | 未披露 | 只披露 optimizer state 用 ZeRO-3 分片，未给出算法名称 |

因此更准确的行业判断是：`AdamW` 仍是通用默认；Muon 已从实验性优化器进入少数前沿预训练栈，并主要沿 Moonshot 系列扩散，也被 DeepSeek-V4 与 Motif-2 采用。它目前更像“二维矩阵的专用优化路径 + AdamW fallback”，而不是彻底替代 AdamW。

### K2/K3 是否也把 embedding、head、RMSNorm 留给 AdamW

- **Moonlight：明确是。** 原论文写明 RMSNorm、LM head 与 embedding parameters 由 AdamW 处理，隐藏层矩阵由 Muon 处理。
- **K2：强推断是，但没有独立清单。** K2 使用 Moonlight 的 weight decay 与 consistent update RMS recipe，算法只描述二维 weights 的 Muon step；报告没有再次列出 fallback groups 或 AdamW 超参数。
- **K3：更应保留“不完全披露”。** 报告明确 Muon 用于 matrix parameters，因此 RMSNorm 等 1-D 参数不走 Muon；但没有说 fallback 就是 AdamW。Embedding/head 是二维参数，仅凭这一句话无法判断它们是否被排除。结合 Moonlight/K2 谱系，继续使用 AdamW 是最合理推断，却不是 K3 报告的直接事实。
- **DeepSeek-V4：明确是。** 它逐项写出 embedding、prediction head、RMSNorm 与部分 mHC 参数保留 AdamW，因此证据等级高于 K2/K3。

## 相关主张

- Muon 的关键不是“所有方向都同样重要”，而是避免 update spectrum 被少数最大奇异值完全支配；这种归一化仍可能放大噪声方向，因此收益依赖 momentum、矩阵粒度与训练阶段。
- `52% FLOPs` 是 Moonlight scaling law 的作者报告，需要保留模型族、数据和超参数搜索边界。
- K2 的 QK-Clip、K3 的 Per-Head Muon、DeepSeek-V4 的 Hybrid Newton–Schulz 分别处理 stability、head coupling 与正交化精度，不是同一个改动的不同名字。
- 分布式 Muon 的主要系统问题不是矩阵乘法本身，而是怎样在参数已分片时恢复 logically complete matrices，并避免通信、冗余计算和 rank imbalance。

## 来源支持

- [Keller Jordan - 2024 - Muon: An Optimizer for Hidden Layers in Neural Networks](../summaries/Keller%20Jordan%20-%202024%20-%20Muon%20An%20Optimizer%20for%20Hidden%20Layers%20in%20Neural%20Networks.md)
- [Liu et al. - 2025 - Muon is Scalable for LLM Training](../summaries/Liu%20et%20al.%20-%202025%20-%20Muon%20is%20Scalable%20for%20LLM%20Training.md)
- [Kimi Team - 2025 - Kimi K2: Open Agentic Intelligence](../summaries/Kimi%20Team%20-%202025%20-%20Kimi%20K2%20Open%20Agentic%20Intelligence.md)
- [Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence](../summaries/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)
- [DeepSeek AI - 2026 - DeepSeek-V4](../summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)
- [Lim et al. - 2025 - Motif-2-12.7B Technical Report](../summaries/Lim%20et%20al.%20-%202025%20-%20Motif%202%2012.7B%20Technical%20Report.md)
- [Team, Meta - 2024 - The Llama 3 Herd of Models](../summaries/Team,%20Meta%20-%202024%20-%20The%20Llama%203%20Herd%20of%20Models.md)
- [Unknown - 2024 - DeepSeek-V3 Technical Report](../summaries/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)
- [OLMo Team - 2025 - 2 OLMo 2 Furious](../summaries/Team%20OLMo%20-%202025%20-%202%20OLMo%202%20Furious.md)
- [Qwen Team - 2025 - Qwen3 Think Deeper Act Faster](../summaries/Qwen%20Team%20-%202025%20-%20Qwen3%20Think%20Deeper%20Act%20Faster.md)
- [Team, DeepMind - 2025 - Gemma 3 Technical Report](../summaries/Team,%20Deepmind%20-%202025%20-%20Gemma%203%20Technical%20Report.md)

## 关联页面

- [Muon 与 AdamW](../comparisons/Muon%20与%20AdamW.md)
- [Kimi K3](./Kimi%20K3.md)
- [DeepSeek-V4](./DeepSeek-V4.md)
- [Kimi](./Kimi.md)
- [OLMo 2](./OLMo%202.md)
- [LLM 预训练](../topics/LLM%20预训练.md)

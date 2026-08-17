---
type: concept
---
# Kimi K3

## 简介

`Kimi K3` 是 Moonshot AI 在 2026 年发布的开放权重、原生多模态、agent-oriented `MoE` 模型。它以约 `2.78T` 总参数、`104.2B` 激活参数和 `1M` token context 把 Kimi 家族从此前的高影响 API 模型推进到集群级 open-weight frontier model。其技术特征不是单个新算子，而是 `KDA + Gated MLA`、`AttnRes`、`Stable LatentMoE`、多档 reasoning-effort RL 与 KDA-aware serving 的联合设计。

## 关键属性

- 类型：开放权重原生多模态 MoE / 长上下文 agent 模型
- 规模：约 `2.78T` 总参数、`104.2B` 激活参数、93 层
- 上下文：`1,048,576` token
- 核心结构：`69 KDA + 24 Gated MLA`、`Block AttnRes`、`Stable LatentMoE`
- MoE：896 routed experts、top-16、2 shared experts
- 视觉：从头联合训练的 `MoonViT-V2`
- Post-training：SFT、三领域 × 三 reasoning efforts 的 RL、MOPD、MXFP4/MXFP8 QAT
- 开放性：完整权重可下载，但适用自定义 `Kimi K3 License`，包含 MaaS 与超大商业产品条件
- 部署：官方列出 vLLM、SGLang、TokenSpeed；推荐 64+ accelerators 的 supernode 形态

## 技术结构

| 信息流维度 | K3 机制 | 解决的主要问题 | 新代价或约束 |
|---|---|---|---|
| 序列 | `KDA` 与周期性 `Gated MLA` | 让大部分层用 fixed-size recurrent state 处理长序列，同时保留周期性全局交互 | KDA 状态递推、context parallelism 与 prefix cache 需专用 kernel/manager |
| 深度 | `Block AttnRes` | 让当前层选择性读取 embedding 与先前 block 表示，避免所有历史只被均匀压进单条 residual stream | 需要维护 block states 和专用 prefill/decode kernel |
| 宽度 | `Stable LatentMoE` | 用低维 routed path 扩大 expert pool 与 top-k，同时控制通信和激活不稳定 | 极端 sparsity 依赖 QB、静态 EP 和专门 MoE kernels |
| 模态 | `MoonViT-V2` | 从训练开始统一 text/image/video 表示，而非后接视觉 adapter | 视觉样本形状与计算量使 PP/CP 负载更不规则 |
| 行为 | 多领域、多 effort RL + MOPD | 把 reasoning、coding、agentic 与视觉执行能力压回统一模型 | preserved thinking history 成为实际调用协议约束 |

## Muon 在 K3 中的用法

Kimi K3 延续 Kimi K2，把 `Muon` 用于 **matrix parameters**。这一定义很重要：技术报告并没有说所有标量、向量和非矩阵参数都由 Muon 更新；其明确讨论的核心对象是 Transformer 中的二维权重矩阵。

### AdamW fallback 的证据边界

K3 报告没有像 DeepSeek-V4 那样列出 `embedding / head / RMSNorm -> AdamW` 的完整参数清单，全文也没有披露 AdamW 的 betas 或 epsilon。当前能分三层判断：

- **明确事实**：matrix parameters 使用 Muon；Q/K/V 进一步使用 Per-Head Muon。
- **结构上确定**：RMSNorm scale 等 1-D 参数不在 Muon matrix path 中，必然由另一条更新路径处理，但 K3 没有命名该 optimizer。
- **基于谱系的强推断**：Moonlight 明确以 AdamW 处理 RMSNorm、LM head 与 embedding，K3 又写明“Following Kimi K2”；所以沿用 AdamW fallback 很合理。不过 embedding/head 本身是 2-D，不能仅凭“matrix parameters”措辞证明它们在 K3 中是否被排除。

因此知识库把 K3 标成“Muon 为矩阵主路径，AdamW fallback 很可能存在但精确 parameter groups 未披露”，而不采用 DeepSeek-V4 那种已确认的逐模块表述。

### 从 full-matrix 到 Per-Head Muon

普通 Muon 会先对一个完整权重矩阵的 momentum 做 Newton–Schulz orthogonalization。对 attention 的 Q/K/V projections，如果直接处理拼接后的完整 projection matrix，相当于把所有 attention heads 看成一个耦合块：

1. 先累计完整 Q、K 或 V projection 的 momentum matrix。
2. 对完整矩阵做近似正交化。
3. 所有 heads 共同决定最终 update geometry。

K3 将这一路径改成 `Per-Head Muon`：

1. 沿 attention-head dimension 切分 Q/K/V momentum matrices。
2. 对每个 head 对应的 tall matrix block 分别执行 Newton–Schulz iterations。
3. 把各 head blocks 的结果重新组合成 projection update。

这样做是因为 full-matrix orthogonalization 可能让 gradient 或 momentum scale 较大的 heads 主导共同更新方向，使较弱 heads 得不到充分归一化。Per-Head Muon 把正交化边界收缩到单个 head，目标是让不同 heads 获得更平衡的更新尺度和学习动态。官方报告还指出，小于完整 projection matrix 的 tall per-head blocks 能略微降低 Newton–Schulz 的计算开销。

### 在预训练 recipe 中的位置

Per-Head Muon 不是孤立使用的。K3 预训练把它与以下机制共同组合：

- 继承自 Kimi K2 的 weight clipping。
- 用于 MoE load balancing 的 `Quantile Balancing (QB)`。
- cosine learning-rate schedule 与 `1%` linear warmup。
- 全程 `0.1` weight decay。

因此 Per-Head Muon 主要负责 attention projection updates 的 head-wise geometry 与大尺度训练稳定性；QB 负责 expert dispatch balance，weight clipping 和有界激活负责控制数值异常。它们解决的是不同层次的问题。

### 分布式实现

K3 的 distributed optimizer 会把参数分散到 data-parallel ranks，但 Newton–Schulz orthogonalization 需要完整的 logically independent matrix。朴素实现是在所有 ranks 上 all-gather 整个 parameter buffer，这会同时增加通信量和峰值显存。

K3 改为 `P2P-based Muon orthogonalization`：每个 rank 只从相应 owner ranks 取回自己负责参数所缺少的 shards，不为所有参数建立完整全局 buffer；通信和正交化再以 model-chunk buffer 为粒度流水化，从而隐藏一部分通信开销。这说明 Muon 在 K3 中不仅是优化公式，也需要与 parameter sharding 和 pipeline execution 协同设计。

### 能解释什么，不能解释什么

- 可以支持的判断：Per-Head Muon 是 K3 用来平衡 attention heads 更新尺度、改善大规模训练稳定性的组成部分。
- 不能直接推出：K3 的 `2.5× scaling efficiency` 全部来自 Muon。该数字是 architecture、data、training recipe、model shape 和超参数重新搜索的联合 scaling-law 结果。
- 不能直接推出：采用 Muon 会让推理更快。Muon 作用于训练优化；推理效率主要由 KDA/MLA、MoE、量化、cache 与 serving system 决定。

## 相关主张

- K3 的 `2.5× scaling efficiency` 是相对 K2 的官方 scaling-law 结果，指在作者的验证损失拟合与超参数搜索下更有效地使用计算，不等于下游 benchmark 全面提升 2.5 倍。
- KDA 并不是简单删除 full attention：K3 每三个 KDA 层保留一个 Gated MLA 层，并让最后一层始终执行全局 attention。这是一种 recurrent/linear mixing 与 global latent attention 的混合架构。
- Stable LatentMoE 的贡献不只是把 expert 数量增到 896，而是把低维 routed path、RMSNorm、SiTU-GLU 与 Quantile Balancing 组合起来，使 top-16 routing 在 3T 级训练中可控。
- K3 的 native multimodality 指视觉与语言从预训练开始共享 next-token objective；公开模型仓库明确列出 text/image 输入，而技术报告说明训练语料还覆盖 video。
- agent 能力来自 post-training 与环境系统的共同作用：partial rollout、budgeted effort、MOPD、不同 harness 配置、可验证环境和持久化 microVM 共同支持长程轨迹。
- 1M context 的工程核心不是只把最大长度设大，而是训练 curriculum、KDA Context Parallelism、外部 cache pool、KDA-aware prefix caching 与 fleet scheduling 的共同实现。
- K3 是 open-weight，但不是 fully open research release：训练数据与完整训练代码没有达到 OLMo 式全面公开，且自定义许可证对特定商业规模设置条件。
- 官方评测覆盖广，但不同模型与任务使用不同 harness、effort、tools 和数据来源；适合观察能力覆盖面，不适合脱离配置给出永久总排名。
- 产品上 K3 永远 thinking，且依赖 preserved thinking history。部署方如果只回传可见 answer、遗漏 `reasoning_content`，会触发官方明确警告的质量不稳定。
- 官方还警告 K3 可能过度主动；在有权限、资金、外部写入或不可逆操作的 agent 场景中，system prompt、approval gate 和 verifier 不能省略。

## 使用判断

- 适合：超长代码库、复杂 knowledge work、视觉参与的多步任务、可以保留完整会话状态且有集群级部署条件的 agent 系统。
- 需要谨慎：低延迟短对话、必须关闭 thinking、会在同一 session 中频繁切换模型、不能保留 reasoning history、缺少权限边界或 verifier 的自动化流程。
- Self-hosting 评估不能只看权重下载成功，还要核对 MXFP4 kernel、KDA/AttnRes runtime、EP 拓扑、cache manager 和具体框架版本。

## 来源支持

- [Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence](../../wiki/summaries/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)
- [Kimi - 2026 - Kimi K3 Open Frontier Intelligence Release](../../wiki/summaries/Kimi%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence%20Release.md)
- [Moonshot AI - 2026 - Kimi K3 Model Repository](../../wiki/summaries/Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20Model%20Repository.md)
- [Moonshot AI - 2026 - Kimi K3 License](../../wiki/summaries/Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20License.md)
- [Kimi - 2026 - Kimi API Model Selection](../../wiki/summaries/Kimi%20-%202026%20-%20Kimi%20API%20Model%20Selection.md)

## 关联页面

- [Kimi](./Kimi.md)
- [Kimi Delta Attention](./Kimi%20Delta%20Attention.md)
- [Attention Residuals](./Attention%20Residuals.md)
- [Stable LatentMoE](./Stable%20LatentMoE.md)
- [Quantile Balancing](./Quantile%20Balancing.md)
- [MoonViT-V2](./MoonViT-V2.md)
- [MoonEP](./MoonEP.md)
- [MoE](./MoE.md)
- [Muon](./Muon.md)
- [SGLang](./SGLang.md)
- [vLLM](./vLLM.md)
- [LLM 预训练](../topics/LLM%20预训练.md)
- [LLM RL](../topics/LLM%20RL.md)
- [注意力机制 Attention](../topics/注意力机制%20Attention.md)
- [开放模型家族与中国重要家族对照](../comparisons/开放模型家族与中国重要家族对照.md)
- [Moonshot AI](../authors/Moonshot%20AI.md)

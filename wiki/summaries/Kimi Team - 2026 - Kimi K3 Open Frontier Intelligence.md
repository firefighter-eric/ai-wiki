# Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence

## 来源信息

- 类型：技术报告 / arXiv 论文
- arXiv：https://arxiv.org/abs/2607.24653
- 原始 PDF：../../raw/pdf/Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence.pdf
- 发布页快照：../../raw/html/Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence.html
- 全文文本：../../raw/text/Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence.md
- 作者：Kimi Team
- 年份：2026
- 状态：已精读；PDF 共 47 页，并对架构、系统与评测页面做过渲染核验

## 摘要

`Kimi K3` 是一个原生多模态、面向长程 agent 工作负载的 `2.78T` 参数 `MoE` 模型，每个 token 激活约 `104.2B` 参数，最大上下文为 `1,048,576` token。报告的核心不是单一规模纪录，而是把信息流沿三个轴共同扩展：序列轴由 `Kimi Delta Attention (KDA)` 与周期性 `Gated MLA` 组成混合注意力；深度轴由 `Attention Residuals (AttnRes)` 选择性访问先前层表示；宽度轴由 `Stable LatentMoE` 在 896 个 routed experts 中激活 16 个。作者报告这些改动与数据、训练 recipe 共同带来相对 `Kimi K2` 约 `2.5×` 的 scaling efficiency 提升。

报告同时把预训练、post-training 与系统实现写成一个端到端设计。预训练从一开始联合优化文本和视觉 token，并用 `MoonViT-V2` 作为从头训练的视觉编码器；post-training 依次经过 `SFT -> 多领域、多 reasoning effort 的 RL -> Multi-Teacher On-Policy Distillation (MOPD)`，还从 SFT 起引入 `MXFP4` 权重、`MXFP8` 激活的 quantization-aware training。长程 agent RL 通过 partial rollout、持久化 sandbox、外部 cache pool 和可恢复 microVM 环境维持跨迭代轨迹。

系统层与架构层高度耦合。KDA 用固定大小 recurrent state 替代随序列增长的完整 KV state，但带来串行递推、context parallelism 和 prefix-cache 语义的新问题；报告为训练、prefill、decode 分别设计 kernel，并让 KDA state 与 MLA paged KV cache 在统一缓存布局中共同分配、淘汰和恢复。对 896-expert MoE，`MoonEP` 通过动态冗余专家、在线规划和静态形状实现每个 expert-parallel rank 的严格负载平衡。

在评测上，K3 覆盖 reasoning、coding、agentic、vision 与内部长程执行任务。报告明确承认总体表现仍落后于最强 proprietary baselines `Claude Fable 5` 与 `GPT-5.6 Sol`。因此本报告更适合支撑“架构与训练系统如何构成开放前沿模型”的判断，不应被压缩成无条件 benchmark 冠军叙事。

## 关键事实

### 模型与架构

- 总参数约 `2.78T`，激活参数约 `104.2B`；93 层，其中 1 个 dense layer；hidden size `7,168`，96 个 attention heads，词表 `160K`。
- attention 由 `69 KDA + 24 Gated MLA` 构成。每个主 block 使用 `3 KDA : 1 Gated MLA`，backbone 末尾再放一个 Gated MLA，以保证最终层执行全局 attention。
- KDA 是带 channel-wise forget gate 的 delta-rule recurrence。K3 把 log-decay 改为下界为 `-5` 的 scaled sigmoid，使 16-token tile 的倒数缩放保持在 BF16 动态范围内，从而让 causal diagonal 与 off-diagonal tiles 都能走 dense Tensor Core matmul；输出门也改为 input-dependent full-rank gate。
- Gated MLA 保留低维 KV latent 的全局交互，但在 K3 中不使用显式位置编码；KDA 提供 recency 与 position-sensitive mixing，周期性 MLA 提供 unrestricted global content interaction。
- `Block AttnRes` 将层分成 8 个 12-layer blocks，并把 embedding 计入来源后形成 9 个 block-level states；它把 full AttnRes 的存储与跨 stage 通信从 `O(Ld)` 降到 `O(Nd)`。
- `Stable LatentMoE` 使用 `3,584` 维 latent routed path、896 个 routed experts、每 token top-16、2 个 full-width shared experts，每个 expert hidden size 为 `3,072`。
- Stable LatentMoE 通过 routed aggregate 后的 `RMSNorm`、有界的 `SiTU-GLU` 和 `Quantile Balancing (QB)` 共同处理极端 sparsity 下的 activation explosion 与 load imbalance。
- `QB` 从 router-score quantile 直接推导下一步 expert bias；bias 只参与 top-k dispatch，不进入 mixture weights，因此其目标是调节负载而不直接改写 router gradient。
- K3 延续 K2，对 matrix parameters 使用 `Muon`；其中 Q/K/V attention projections 采用 `Per-Head Muon`，不对拼接后的完整 momentum matrix 一次正交化，而是沿 attention-head dimension 分块执行 Newton–Schulz orthogonalization，以减少大尺度下不同 head 更新幅度互相支配的问题。报告称这种做法使 head 间学习动态更平衡，并因 tall per-head blocks 较小而略降 optimizer overhead。
- 报告只明确划定“matrix parameters 使用 Muon”，全文没有出现 AdamW parameter-group 配置。RMSNorm scale 等 1-D 参数显然不属于该 Muon matrix path；但报告没有命名其 fallback optimizer。Embedding 与 output head 本身是 2-D，是否像 Moonlight 一样从 Muon 中排除也没有被 K3 独立确认。
- `MoonViT-V2` 是约 `401M` 参数、27 层、patch size 14、12 heads 的视觉编码器；报告称它从随机初始化开始与 LLM 联合训练，而不是先做 SigLIP 式对比预训练再接入。

### 预训练与长上下文

- 数据覆盖 Web Text、Code、Mathematics、Knowledge 与大规模视觉语料；视觉数据包含 caption、图文交错文档、OCR、perception、video 与 visual coding。
- K3 从训练开始就把视觉与文本 token 置于统一 next-token prediction 目标下联合优化，而不是在语言模型完成后再做视觉 adapter 对齐。
- training context 先从 `8K` 扩展到 `64K`，cooldown 阶段再从 `256K` 扩展到 `1M`；报告称由于 KDA 隐式提供位置信息，扩窗不需要重新缩放或插值 RoPE。
- 长上下文数据经过去重、质量过滤与结构验证，并额外合成只有跨越完整 1M context 才能解决的多模态子任务，避免模型只依赖局部模式。
- `2.5×` scaling efficiency 是作者在独立 scaling-law 搜索和 held-out OOD validation loss 上相对 Kimi K2 的综合结果，不是“相同参数下所有下游任务均提升 2.5 倍”。

### Post-training 与 agent 环境

- post-training 由 SFT、RL、MOPD 三阶段组成。RL 分 general、general agents、coding agents 三个领域，并为 low、high、max 三档 reasoning effort 训练 9 个 expert policies。
- partial rollout 在一部分轨迹完成后启动优化，将未完成轨迹暂停并跨 iteration 恢复；per-token regularization 用于容忍由此产生的 stale/off-policy data。
- reasoning-effort RL 对每题设置 token budget，并对超过预算的轨迹覆盖负奖励；agentic task 的预算同时计入 reasoning trace 与 tool-call arguments。
- `MOPD` 用九个领域/effort teacher 的 token-level dense reward 将专门能力合并到单一 student；作者称更细粒度 top-k distillation 在该设定下未显示清晰优势。
- 统一 white-box RL 环境把 tools、system prompts、context management、skills、memories 与 subagents 模块化，动态模拟 Kimi Code、Claude Code、Codex、OpenClaw、Hermes 等不同 harness，减少对单一 agent protocol 的过拟合。
- 可验证任务覆盖搜索、专业知识工作、视觉推理、GPU kernel、网页开发、个人助理和 Autonomous Execution Tasks；最终 reward 尽量落到可检查的环境状态，而不是模型自报完成。
- 从 SFT 开始，routed experts 以 `MXFP4` 权重和 `MXFP8` 激活做 QAT；非 expert attention、latent projection、shared experts 与 router 保持更高精度。
- 预训练的 MTP layer 被进一步训练为 EAGLE-3 风格 draft model，并直接优化 speculative decoding 的 acceptance-rate surrogate。

### 训练与推理基础设施

- KDA 使用固定大小 recurrent state，缓解长序列 KV 增长，但其状态递推不天然适合 GPU 并行；报告为 training/prefill 设计 `FlashKDA`，为跨设备长序列设计 KDA Context Parallelism，并为 decode 设计可在 speculative rejection 后重建 state 的 replay kernel。
- 3T 级训练组合 Pipeline Parallelism、virtual stages、Expert Parallelism、ZeRO-1、Pipeline ZeRO-2 和 Context Parallelism，并用统一 activation manager 组合 recomputation、quantization、local/remote offload。
- K3 的 distributed optimizer 按 DP ranks 切分参数，但 Muon 正交化需要完整 matrix；其实现不在每个 rank 上 all-gather 整个 parameter buffer，而是让各 rank 通过 P2P 只取回自己负责参数的缺失 shards，并按 model-chunk buffers 流水化通信与正交化计算。
- `MoonEP` 用动态冗余专家、GPU online planner、zero-copy communication 与 static shapes 保证每个 EP rank 接收完全相同的 token 负载，避免逐层 host-device shape synchronization。
- 1M agentic RL 将可复用 prefix 状态写回 CPU DRAM 外部 cache pool，并在训练/rollout 阶段间复用显存和主存；scheduler 根据 active/queued requests 与 cache utilization 自动节流。
- `AgentENV` 基于 Firecracker microVM，支持 pause/resume、fork、snapshot 与增量 checkpoint；报告给出的 133ms checkpoint、49ms resume 和 6.5× memory overcommit 都是作者系统中的测量值。
- KDA-aware prefix cache 将 fixed-size KDA state 与 sequence-growing MLA KV pages 放在统一 paged layout 下，但只有持久化了命中边界的 KDA checkpoint 时，MLA prefix hit 才能被完整复用。
- fleet-level serving 用 cache-affinity 将 session 路由到持有其 prefix cache 的集群，并以双集群 consistent hashing 控制故障影响；budget-based admission control 隔离短请求和 1M-token 请求的资源预算。

### 评测

- 官方主表统一把 K3 设为 `reasoning effort=max`、`temperature=1.0`；single-step tasks 多用 `top-p=0.95`，agentic tasks 用 `top-p=1.0`。
- coding 与 agentic 结果混用 Kimi Code、Claude Code、Codex 等 harness；部分成绩来自官方 leaderboard、Artificial Analysis、Vals AI，另一些来自 Moonshot 自测，因此不能把全部单元格视为同一评测环境下的严格 controlled comparison。
- 报告给出 K3 在第三方榜单上的当时排名，但明确说明 Elo 会随投票漂移；此类数字只应当作 `2026-07-23` 左右的快照。
- 官方结论是 K3 接近但总体仍落后于 Claude Fable 5 与 GPT-5.6 Sol，同时强于报告中测试的其他模型；知识库不把这一自评推广为跨平台、跨版本的永久排名。

## 争议与不确定点

- 技术报告由模型开发者发布，架构描述是一手证据，但 benchmark、成本和案例研究仍带有供应方自评性质；第三方结果虽被引用，汇总与叙事仍由 Moonshot 完成。
- `2.5×` scaling efficiency 依赖作者选择的模型 family、超参数搜索和 OOD loss 拟合方法，目前缺少独立复现。
- K3 沿 K2/Moonlight 路线，因而“非矩阵参数以及 embedding/head 很可能继续使用 AdamW”是合理推断；但证据等级低于 DeepSeek-V4 的明确清单，不能写成 K3 报告已经确认的事实。
- KDA、AttnRes、Stable LatentMoE、MoonEP 与 KDA-aware cache 同时变化，报告没有提供足以把最终能力精确归因到每个组件的完整外部消融。
- 1M context 表示训练与接口上限，不保证任意任务在 1M token 上都能保持等价检索、推理和生成质量；真实成本也高度依赖 cache hit、请求形态与集群拓扑。
- “open”在此首先表示权重可获得；实际使用受自定义 Kimi K3 License 约束，不能自动等同于 OSI 意义的软件开源或 fully open research release。

## 关联页面

- 概念：[Kimi](../../wiki/concepts/Kimi.md)
- 概念：[Kimi K3](../../wiki/concepts/Kimi%20K3.md)
- 概念：[Kimi Delta Attention](../../wiki/concepts/Kimi%20Delta%20Attention.md)
- 概念：[Attention Residuals](../../wiki/concepts/Attention%20Residuals.md)
- 概念：[Stable LatentMoE](../../wiki/concepts/Stable%20LatentMoE.md)
- 概念：[Quantile Balancing](../../wiki/concepts/Quantile%20Balancing.md)
- 概念：[MoonViT-V2](../../wiki/concepts/MoonViT-V2.md)
- 概念：[MoonEP](../../wiki/concepts/MoonEP.md)
- 概念：[MoE](../../wiki/concepts/MoE.md)
- 概念：[Muon](../../wiki/concepts/Muon.md)
- 主题：[LLM 预训练](../../wiki/topics/LLM%20预训练.md)
- 主题：[LLM RL](../../wiki/topics/LLM%20RL.md)
- 主题：[注意力机制 Attention](../../wiki/topics/注意力机制%20Attention.md)
- 比较：[开放模型家族与中国重要家族对照](../../wiki/comparisons/开放模型家族与中国重要家族对照.md)

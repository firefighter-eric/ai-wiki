---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence

## TL;DR（快速导读）

Kimi K3 报告同时调整序列、层间和专家信息流，并结合多模态训练与后训练处理长程任务。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

完成代码任务可能要读取仓库、运行工具和根据结果继续行动；模型分数与整条执行系统的可靠性要分开测。

## 来源信息

- 类型：技术报告 / arXiv 论文
- arXiv：https://arxiv.org/abs/2607.24653
- 原始 PDF：../../raw/pdf/Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence.pdf
- 发布页快照：../../raw/html/Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence.html
- 全文文本：../../raw/text/Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence.md
- 作者：Kimi Team
- 年份：2026
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 核对说明：原HTML为arXiv摘要页2607.24653，全文层已改为完整47页PDF提取，证据定位使用PDF页码。

## 摘要

序列方向使用 KDA 与门控 MLA，层间使用注意力残差，专家方向使用 Stable LatentMoE。报告还涉及视觉编码、强化学习、蒸馏与低精度训练；各改动的贡献需看对应消融，整体收益不能直接归给某一个模块。

### 方法与背景细节

`Kimi K3` 是一个原生多模态、面向长程 agent 工作负载的 `2.78T` 参数 `MoE` 模型，每个 token 激活约 `104.2B` 参数，最大上下文为 `1,048,576` token。报告的核心不是单一规模纪录，而是把信息流沿三个轴共同扩展：序列轴由 `Kimi Delta Attention (KDA)` 与周期性 `Gated MLA` 组成混合注意力；深度轴由 `Attention Residuals (AttnRes)` 选择性访问先前层表示；宽度轴由 `Stable LatentMoE` 在 896 个 routed experts 中激活 16 个。作者报告这些改动与数据、训练 recipe 共同带来相对 `Kimi K2` 约 `2.5×` 的 scaling efficiency 提升。

报告同时把预训练、post-training 与系统实现写成一个端到端设计。预训练从一开始联合优化文本和视觉 token，并用 `MoonViT-V2` 作为从头训练的视觉编码器；post-training 依次经过 `SFT -> 多领域、多 reasoning effort 的 RL -> Multi-Teacher On-Policy Distillation (MOPD)`，还从 SFT 起引入 `MXFP4` 权重、`MXFP8` 激活的 quantization-aware training。长程 agent RL 通过 partial rollout、持久化 sandbox、外部 cache pool 和可恢复 microVM 环境维持跨迭代轨迹。

系统层与架构层高度耦合。KDA 用固定大小 recurrent state 替代随序列增长的完整 KV state，但带来串行递推、context parallelism 和 prefix-cache 语义的新问题；报告为训练、prefill、decode 分别设计 kernel，并让 KDA state 与 MLA paged KV cache 在统一缓存布局中共同分配、淘汰和恢复。对 896-expert MoE，`MoonEP` 通过动态冗余专家、在线规划和静态形状实现每个 expert-parallel rank 的严格负载平衡。

在评测上，K3 覆盖 reasoning、coding、agentic、vision 与内部长程执行任务。报告明确承认总体表现仍落后于最强 proprietary baselines `Claude Fable 5` 与 `GPT-5.6 Sol`。因此本报告更适合支撑“架构与训练系统如何构成开放前沿模型”的判断，不应被压缩成无条件 benchmark 冠军叙事。

## 关键事实

- **C1**：总参数约 `2.78T`，激活参数约 `104.2B`；93 层，其中 1 个 dense layer；hidden size `7,168`，96 个 attention heads，词表 `160K`。
- **C2**：attention 由 `69 KDA + 24 Gated MLA` 构成。每个主 block 使用 `3 KDA : 1 Gated MLA`，backbone 末尾再放一个 Gated MLA，以保证最终层执行全局 attention。
- **C3**：KDA 是带 channel-wise forget gate 的 delta-rule recurrence。K3 把 log-decay 改为下界为 `-5` 的 scaled sigmoid，使 16-token tile 的倒数缩放保持在 BF16 动态范围内，从而让 causal diagonal 与 off-diagonal tiles 都能走 dense Tensor Core matmul；输出门也改为 input-dependent full-rank gate。
- **C4**：Gated MLA 保留低维 KV latent 的全局交互，但在 K3 中不使用显式位置编码；KDA 提供 recency 与 position-sensitive mixing，周期性 MLA 提供 unrestricted global content interaction。
- **C5**：`Block AttnRes` 将层分成 8 个 12-layer blocks，并把 embedding 计入来源后形成 9 个 block-level states；它把 full AttnRes 的存储与跨 stage 通信从 `O(Ld)` 降到 `O(Nd)`。
- **C6**：`Stable LatentMoE` 使用 `3,584` 维 latent routed path、896 个 routed experts、每 token top-16、2 个 full-width shared experts，每个 expert hidden size 为 `3,072`。
- **C7**：Stable LatentMoE 通过 routed aggregate 后的 `RMSNorm`、有界的 `SiTU-GLU` 和 `Quantile Balancing (QB)` 共同处理极端 sparsity 下的 activation explosion 与 load imbalance。
- **C8**：`QB` 从 router-score quantile 直接推导下一步 expert bias；bias 只参与 top-k dispatch，不进入 mixture weights，因此其目标是调节负载而不直接改写 router gradient。
- **C9**：K3 延续 K2，对 matrix parameters 使用 `Muon`；其中 Q/K/V attention projections 采用 `Per-Head Muon`，不对拼接后的完整 momentum matrix 一次正交化，而是沿 attention-head dimension 分块执行 Newton–Schulz orthogonalization，以减少大尺度下不同 head 更新幅度互相支配的问题。报告称这种做法使 head 间学习动态更平衡，并因 tall per-head blocks 较小而略降 optimizer overhead。
- **C10**：报告只明确划定“matrix parameters 使用 Muon”，全文没有出现 AdamW parameter-group 配置。RMSNorm scale 等 1-D 参数显然不属于该 Muon matrix path；但报告没有命名其 fallback optimizer。Embedding 与 output head 本身是 2-D，是否像 Moonlight 一样从 Muon 中排除也没有被 K3 独立确认。
- **C11**：`MoonViT-V2` 是约 `401M` 参数、27 层、patch size 14、12 heads 的视觉编码器；报告称它从随机初始化开始与 LLM 联合训练，而不是先做 SigLIP 式对比预训练再接入。
- **C12**：数据覆盖 Web Text、Code、Mathematics、Knowledge 与大规模视觉语料；视觉数据包含 caption、图文交错文档、OCR、perception、video 与 visual coding。
- **C13**：K3 从训练开始就把视觉与文本 token 置于统一 next-token prediction 目标下联合优化，而不是在语言模型完成后再做视觉 adapter 对齐。
- **C14**：training context 先从 `8K` 扩展到 `64K`，cooldown 阶段再从 `256K` 扩展到 `1M`；报告称由于 KDA 隐式提供位置信息，扩窗不需要重新缩放或插值 RoPE。
- **C15**：长上下文数据经过去重、质量过滤与结构验证，并额外合成只有跨越完整 1M context 才能解决的多模态子任务，避免模型只依赖局部模式。
- **C16**：`2.5×` scaling efficiency 是作者在独立 scaling-law 搜索和 held-out OOD validation loss 上相对 Kimi K2 的综合结果，不是“相同参数下所有下游任务均提升 2.5 倍”。
- **C17**：post-training 由 SFT、RL、MOPD 三阶段组成。RL 分 general、general agents、coding agents 三个领域，并为 low、high、max 三档 reasoning effort 训练 9 个 expert policies。
- **C18**：partial rollout 在一部分轨迹完成后启动优化，将未完成轨迹暂停并跨 iteration 恢复；per-token regularization 用于容忍由此产生的 stale/off-policy data。
- **C19**：reasoning-effort RL 对每题设置 token budget，并对超过预算的轨迹覆盖负奖励；agentic task 的预算同时计入 reasoning trace 与 tool-call arguments。
- **C20**：`MOPD` 用九个领域/effort teacher 的 token-level dense reward 将专门能力合并到单一 student；作者称更细粒度 top-k distillation 在该设定下未显示清晰优势。
- **C21**：统一 white-box RL 环境把 tools、system prompts、context management、skills、memories 与 subagents 模块化，动态模拟 Kimi Code、Claude Code、Codex、OpenClaw、Hermes 等不同 harness，减少对单一 agent protocol 的过拟合。
- **C22**：可验证任务覆盖搜索、专业知识工作、视觉推理、GPU kernel、网页开发、个人助理和 Autonomous Execution Tasks；最终 reward 尽量落到可检查的环境状态，而不是模型自报完成。
- **C23**：从 SFT 开始，routed experts 以 `MXFP4` 权重和 `MXFP8` 激活做 QAT；非 expert attention、latent projection、shared experts 与 router 保持更高精度。
- **C24**：预训练的 MTP layer 被进一步训练为 EAGLE-3 风格 draft model，并直接优化 speculative decoding 的 acceptance-rate surrogate。
- **C25**：KDA 使用固定大小 recurrent state，缓解长序列 KV 增长，但其状态递推不天然适合 GPU 并行；报告为 training/prefill 设计 `FlashKDA`，为跨设备长序列设计 KDA Context Parallelism，并为 decode 设计可在 speculative rejection 后重建 state 的 replay kernel。
- **C26**：3T 级训练组合 Pipeline Parallelism、virtual stages、Expert Parallelism、ZeRO-1、Pipeline ZeRO-2 和 Context Parallelism，并用统一 activation manager 组合 recomputation、quantization、local/remote offload。
- **C27**：K3 的 distributed optimizer 按 DP ranks 切分参数，但 Muon 正交化需要完整 matrix；其实现不在每个 rank 上 all-gather 整个 parameter buffer，而是让各 rank 通过 P2P 只取回自己负责参数的缺失 shards，并按 model-chunk buffers 流水化通信与正交化计算。
- **C28**：`MoonEP` 用动态冗余专家、GPU online planner、zero-copy communication 与 static shapes 保证每个 EP rank 接收完全相同的 token 负载，避免逐层 host-device shape synchronization。
- **C29**：1M agentic RL 将可复用 prefix 状态写回 CPU DRAM 外部 cache pool，并在训练/rollout 阶段间复用显存和主存；scheduler 根据 active/queued requests 与 cache utilization 自动节流。
- **C30**：`AgentENV` 基于 Firecracker microVM，支持 pause/resume、fork、snapshot 与增量 checkpoint；报告给出的 133ms checkpoint、49ms resume 和 6.5× memory overcommit 都是作者系统中的测量值。
- **C31**：KDA-aware prefix cache 将 fixed-size KDA state 与 sequence-growing MLA KV pages 放在统一 paged layout 下，但只有持久化了命中边界的 KDA checkpoint 时，MLA prefix hit 才能被完整复用。
- **C32**：fleet-level serving 用 cache-affinity 将 session 路由到持有其 prefix cache 的集群，并以双集群 consistent hashing 控制故障影响；budget-based admission control 隔离短请求和 1M-token 请求的资源预算。
- **C33**：官方主表统一把 K3 设为 `reasoning effort=max`、`temperature=1.0`；single-step tasks 多用 `top-p=0.95`，agentic tasks 用 `top-p=1.0`。
- **C34**：coding 与 agentic 结果混用 Kimi Code、Claude Code、Codex 等 harness；部分成绩来自官方 leaderboard、Artificial Analysis、Vals AI，另一些来自 Moonshot 自测，因此不能把全部单元格视为同一评测环境下的严格 controlled comparison。
- **C35**：报告给出 K3 在第三方榜单上的当时排名，但明确说明 Elo 会随投票漂移；此类数字只应当作 `2026-07-23` 左右的快照。
- **C36**：官方结论是 K3 接近但总体仍落后于 Claude Fable 5 与 GPT-5.6 Sol，同时强于报告中测试的其他模型；知识库不把这一自评推广为跨平台、跨版本的永久排名。

## 争议与不确定点

- 报告没有确认全部非矩阵或embedding/head的fallback optimizer，明确保留不确定。
- 评测使用多种harness、max effort与来源，排行榜只代表2026-07附近快照。
- 1M窗口、fixed recurrent state与压缩不会保证任意长距离细节都准确保留；需目标任务测试。

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
- [Moonshot AI](../authors/Moonshot%20AI.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **KV cache**：键值缓存：保存已经处理过的位置表示，生成新内容时可复用，避免全部重算。
- **prefill**：提示计算阶段：先处理输入提示，再开始逐步生成输出。
- **decode**：解码阶段：利用已有输入与生成历史，产生后续输出。
- **scheduler**：调度器：决定请求何时进入计算、每次处理多少，以及如何共享资源。
- **backbone**：模型骨干：主要负责提取或变换表示，其他任务模块在它之上工作。

## 方法与实验解读

K3 把长序列信息流、跨层信息流、专家容量和执行环境一起设计。KDA 控制状态增长，周期性 MLA 提供全局交互，AttnRes 调节跨层聚合；LatentMoE、矩阵优化、低精度与通信设计共同决定可训练规模。长程 agent 的能力还依赖保留 thinking/history、可恢复 sandbox 和多样 harness，因此不能把完整工作流成功率归因于其中一项结构。阅读可先看 C1–C16 的基础模型，再看 C17–C32 的训练/部署，最后用 C33–C36 约束性能比较。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=11 ) | 架构规格以本报告表1为准；总参数与激活参数分开。 |
| C2 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=4 ) | 组成见表1；模式见第4页。 |
| C3 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=5 ) | 16-token tile与BF16范围的结构/数值配套设计。 |
| C4 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=5 ) | 混合attention的作者解释；不使用显式PE不等于没有位置敏感性。 |
| C5 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=6 ) | 最后一个block不满12层，不能把8×12机械当实际层数。 |
| C6 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=11 ) | routed/shared通路分开，参数细目见表1。 |
| C7 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=7 ) | 多项措施共同控制数值与负载，未隔离全部贡献。 |
| C8 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=9 ) | 训练更新在下一step生效，推理时bias固定。 |
| C9 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=10 ) | 矩阵按head分块；作者效率/稳定性解释。 |
| C10 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=10 ) | 全文搜索AdamW为0；fallback未披露，保留不确定。 |
| C11 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=11 ) | 精确参数见表1；随机初始化训练见第9页。 |
| C12 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=10 ) | 数据组成，不保证各子域覆盖相同。 |
| C13 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=9 ) | 视觉encoder从头训练不等于所有数据或目标都公开。 |
| C14 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=12 ) | 训练长度课程与实际可用长距离信息不同。 |
| C15 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=12 ) | 合成全局依赖任务用于补足长文数据，不证明无失真。 |
| C16 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=10 ) | 相同validation loss下的整体计算效率，不是下游分数倍数。 |
| C17 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=12 ) | 三个domain×三个effort形成九个teacher。 |
| C18 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=13 ) | staleness与policy regularization同时存在。 |
| C19 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=13 ) | general计thinking，agent计thinking与工具参数，预算分母不同。 |
| C20 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=14 ) | 九teacher按domain/effort选择；top-k无益属于该设定。 |
| C21 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=14 ) | 白盒harness多样化不证明所有真实工具协议兼容。 |
| C22 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=16 ) | 可检验环境状态是奖励设计目标，仍需防reward hacking。 |
| C23 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=14 ) | 量化对象仅routed专家；shared等保持高精度。 |
| C24 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=14 ) | draft与目标模型分开，目标冻结。 |
| C25 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=17 ) | FlashKDA/CP见第17–18页；decode replay见第24页。 |
| C26 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=19 ) | 内存/并行组合细节延续第20页。 |
| C27 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=20 ) | 分片状态与逻辑矩阵更新配合，P2P只取必要shards。 |
| C28 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=19 ) | 完美rank负载是系统实现条件，不等于逐专家样本数都相同。 |
| C29 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=21 ) | 训练/rollout交换资源，依赖scheduler与cache命中。 |
| C30 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=22 ) | 133/49ms和6.5×均是作者系统测量，不是任意microVM指标。 |
| C31 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=23 ) | 命中边界同时满足MLA与KDA状态条件。 |
| C32 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=25 ) | 在线流量和cache分布影响收益。 |
| C33 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=26 ) | 所列采样用于作者评测，工具/无工具分开。 |
| C34 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=26 ) | 跨harness与外部来源，不作严格同环境比较。 |
| C35 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=31 ) | July23榜单快照会随投票漂移。 |
| C36 | [原文]( ../../raw/pdf/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.pdf#page=1 ) | 总体表述与个别任务胜负不同，不生成永久排名。 |

## 核证范围

保留原有36项技术事实，逐项核对PDF表1、§2–§6相关正文与原文关键句；对AdamW缺失作全文字符串检查。数学证明附录与训练数据全量未独立审查，实验未复现。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

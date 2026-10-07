---
type: topic
status: formal
review_scope: evidence_synthesis
reviewed: 2026-10-07
---
# LLM RL

## TL;DR（快速导读）

先看训练信号来自哪里：人类偏好、可验证结果，还是 teacher 概率。再看是否需要当前策略采样、reference、critic 和环境。DPO、GRPO、DAPO 与 OPD 不能只按“是否叫 RL”来比较。

阅读重点：先按问题选择路线，再核对比较条件与证据边界。

## 先用一个问题理解

“回答更合用户偏好”“数学题答对”和“在代码仓库里完成修改”可能使用不同奖励。阅读方法时先问奖励怎样得到、回答由谁生成、是否在线采样，再看优化算法；这样更容易分清 DPO、GRPO 和代理训练。

## 页面状态

正式 topic；2026-10-07 复核核心来源并补充方法比较。正文区分论文实验、作者报告和本文综合判断；开放问题表示研究证据的边界。

## 主题定义

本页讨论 **LLM 后训练中以奖励、偏好与策略优化为核心接口** 的方法族。它覆盖 `RLHF`、`DPO`、`ORPO`、`KTO`、`GRPO`、`DAPO`、`OPD`，以及 `DeepSeek-R1 / DeepSeek-V3.2 / Kimi K3` 一类 reasoning-oriented、thinking tool-use 与 long-horizon agentic RL 路线，但 **不覆盖一般性的 instruction tuning 细节**，也不把所有 post-training 方法都笼统写成 RL。本页的边界是：只有当方法明确围绕 **奖励信号、偏好信号、策略更新、在线 rollout、行为激励、密集 token 级监督或其等价重写** 展开时，才进入 `LLM RL`。

因此，本页处理的不是“模型如何学会遵循指令”这一宽泛问题，而是更窄也更关键的问题：**当预训练能力已经存在后，是否需要通过奖励驱动的后训练机制来改变模型行为，乃至直接激励推理能力。** 这也是本页与 [指令对齐与 post-training](./指令对齐与%20post-training.md) 的区别。后者讨论行为塑形这一总框架；本页则讨论其中最具争议、最容易分化、也最接近策略优化语言的一支。

当前知识库中的稳定证据支持一个较强判断：**LLM RL 已经不是单一技术名词，而是从经典 RLHF 管线分化出的一个方法族。** 其中有些路线试图显式学习奖励并在线优化策略，有些路线把 RLHF 折叠为更直接的偏好目标，有些路线则把 RL 从“让模型更符合人类偏好”推进到“直接塑造推理行为与求解策略”。这些路线共享的是奖励驱动语法，而不是统一的训练配方。

## 核心问题

- **RL 在 LLM 后训练中到底解决什么问题**：是行为校正、偏好拟合、在线探索，还是直接提升推理能力。
- `RLHF`、`DPO`、`ORPO`、`KTO` 与 reasoning RL 的关系究竟是阶段演化、方法分叉，还是不同约束下的并行接口。
- 哪些场景可以把复杂的 `reward model + online RL` 简化为离线偏好优化，哪些场景又必须保留在线 rollout 与策略更新。
- `GRPO`、`DAPO` 这类 reasoning RL 方法的增益主要来自 **优化目标**、**奖励设计**，还是 **大规模训练工程**。
- 当 RL 的目标从“更 helpful / truthful / harmless”转向“更会推理 / 更会解题”时，评估、风险与可监控性会如何变化。

## 主线脉络 / 方法分层

本页不采用“`RLHF -> DPO -> R1`”的线性叙述，而按 **训练接口与目标函数的变化** 来分层。这样做的原因是：很多后续方法并不是简单替代前一代，而是在改变数据接口、参考模型依赖、在线性要求与奖励对象。

- **经典 RLHF 管线**：`Ouyang et al. 2022` 给出的不是一个局部 trick，而是一个完整范式：先用 demonstrations 做监督微调，再用 preference rankings 训练 reward model，最后用 RL 优化策略。其成立前提是，**人类偏好可以被近似建模，并作为比 next-token likelihood 更贴近产品目标的训练信号。** 这一层的关键贡献不在于 PPO 本身，而在于把“帮助性、真实性、无害性”转写成一个可迭代优化的后训练流程。
- **reference-based 偏好优化层**：`Rafailov et al. 2023` 的 `DPO` 之所以重要，不只是因为它“更简单”，而是因为它指出在一定假设下，RLHF 的最优策略可以被改写成更直接的 preference objective。这里的方法分层依据是：**奖励模型与在线 RL 是否必须显式存在。** `DPO` 保留了“偏好决定策略”的核心思想，但把优化形式从显式 RL 管线折叠为闭式损失。
- **reference-free 或弱化监督接口的偏好优化层**：`ORPO` 与 `KTO` 的价值，不是单纯再造一个对齐 loss，而是继续削弱 RLHF 管线中对外部部件与标注形式的依赖。`ORPO` 把 `SFT + preference alignment` 合并为单阶段目标；`KTO` 则把二元成对偏好改写为 unary desirable / undesirable 信号。这一层反映出 post-training 的一个稳定趋势：**方法正在从“完整 RL 管线”向“更轻量、数据接口更便宜的偏好优化族”扩散。**
- **reasoning-oriented 在线 RL 层**：`Shao et al. 2024` 的 `DeepSeekMath` 以及 `DeepSeek-R1` 表明，RL 在 LLM 中的角色已经发生变化。这里不再只是通过奖励让回答“更像人偏好的答案”，而是让模型在数学、代码或长链推理任务中 **形成更有效的中间行为模式**。`GRPO` 的意义在于，它把 critic-free、group-relative advantage 的在线优化方案带入 reasoning 训练，使“推理行为激励”成为一个可规模化讨论的对象。
- **大规模 reasoning RL 工程化层**：`DAPO` 说明 reasoning RL 的瓶颈并不止于“有没有一个好优化器”。当训练目标转向长链推理，长度偏置、reward noise、entropy collapse、sample efficiency、token-level loss、rollout 管理等问题会快速上升为一等公民。也就是说，**算法层与系统层在 reasoning RL 中已经高度耦合**，单独讨论某个 loss 往往不足以解释最终效果。
- **long-horizon agentic RL 层**：`Kimi K3` 把 reasoning RL 的系统问题推进到跨 iteration、百万 token 和持久环境。partial rollout 允许未完成轨迹暂停并恢复；per-problem token budget 训练 low/high/max effort；general、general-agent、coding 三域产生九个 teacher policies，再通过 `MOPD` 的 token-level dense reward 合并回单一模型。更关键的是，tools、system prompts、context management、skills、memories 与 subagents 被做成可组合 white-box harness，奖励尽量落到 verifier 检查的最终环境状态。这一层说明，当 rollout 跨越数百乃至上千次工具调用时，**sandbox lifecycle、cache persistence、任务合成与 verifier 隔离已经成为 RL 方法的一部分**。
- **on-policy distillation 层**：`OPD` 把 teacher-student distillation 拉回到 student 自己的 rollout 分布上：student 先生成轨迹，再在这些轨迹上接受 teacher 的 token 级分布监督。它与 `GRPO / RLVR` 共享 on-policy 语法，但用 dense distillation signal 缓解 outcome reward 稀疏的问题；与 `SFT` 式 off-policy distillation 相比，它又更强调训练分布与推理分布的一致性。`G-OPD / ExOPD` 进一步把 OPD 解释为 dense KL-constrained RL 的特殊情形，说明蒸馏与 RL 在 LLM 后训练中并不是完全分离的两条线。
- **thinking tool-use 层**：`DeepSeek-R1-0528` 与 `DeepSeek-V3.2` 说明 reasoning model 正在从“会推理”走向“能用工具持续执行”。`R1-0528` 增加 JSON output 与 function calling，更多是可用性接口；`V3.2` 则把 thinking 直接集成进 tool-use，并引入覆盖 `1,800+` environments 与 `85k+` complex instructions 的 agent 训练数据合成。这个层次不是一般预训练能力，也不是纯 API 功能，而是 reasoning 后训练与 agent 系统接口开始合流的证据。

如果从知识组织角度再压缩一次，可以把本页方法族粗分为七类：**偏好建模型 RLHF**、**离线化偏好优化**、**online reasoning RL**、**reasoning RL 工程系统**、**long-horizon agentic RL**、**on-policy dense distillation**、**thinking tool-use**。这样切分比按论文时间顺序更稳定，因为它对应的是不同的训练接口与目标边界。

### 用数据接口和成本拆开方法族

| 路线 | 更新时的核心数据 | 关键依赖 | 主要失效风险 |
| --- | --- | --- | --- |
| PPO 型 RLHF | 当前策略输出与学得的偏好奖励 | reward、reference，常含 value/critic | 奖励偏差被策略放大 |
| DPO | 已有 chosen/rejected 对 | reference 与偏好数据覆盖 | 训练对分布与新策略偏离 |
| ORPO / KTO | 偏好对或好/坏样本 | 各自损失定义；KTO 仍有参考基线 | 不能把标注接口变化当信息免费 |
| GRPO / DAPO | 同题多条 rollout 与奖励 | 采样、reference、验证器 | 无差异组、长度偏置、奖励漏洞 |
| OPD / G-OPD | student 自己生成轨迹上的 teacher 分布 | teacher logprob，部分设定另需 reference | teacher 错误、概率访问与生成成本 |

[DPO](../summaries/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)的等价推导依赖 KL 约束和偏好模型假设，实际离线训练不获得无限在线探索。[KTO](../summaries/Ethayarajh%20et%20al.%20-%202024%20-%20KTO%20Model%20Alignment%20as%20Prospect%20Theoretic%20Optimization.md)改变成对标注要求，不意味着所有参考模型依赖消失。[GRPO 来源](../summaries/Shao%20et%20al.%20-%202024%20-%20DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models.md)省去 critic，但模型、奖励和采样仍存在。[DAPO](../summaries/Yu%20et%20al.%20-%202025%20-%20DAPO%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.md)的动态采样过滤无信息组会多生成轨迹，因此训练步数减少不能直接写成总算力减半。

[G-OPD / ExOPD](../summaries/Yang%20et%20al.%20-%202026%20-%20Learning%20beyond%20Teacher%20Generalized%20On-Policy%20Distillation%20with%20Reward%20Extrapolation.md)在 student 分布上给密集反馈，将奖励强度与 KL 权重分开。大于一的外推可能强化有效信号，也可能强化 teacher 错误；“超过 teacher”是给定模型、任务与组合方式的结果，不是蒸馏普遍规律。[K3](../summaries/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)把长期环境、预算与可恢复轨迹纳入训练；本文据此把总成本分为训练更新、轨迹生成、teacher 评分、验证和环境维持。这个成本分解是可复用的分析框架，而非报告已提供的统一 FLOPs 排行。

训练目标改善不自动说明过程真实可信。[推理监控研究](../summaries/Baker%20et%20al.%20-%20Unknown%20-%20Monitoring%20Reasoning%20Models%20for%20Misbehavior%20and%20the%20Risks%20of%20Promoting%20Obfuscation.md)在特定代码 agent 环境中观察奖励作弊与监控；[DFT](../summaries/Wu%20et%20al.%20-%202025%20-%20On%20the%20Generalization%20of%20SFT%20A%20Reinforcement%20Learning%20Perspective%20with%20Reward%20Rectification.md)则调整 SFT 梯度权重。这两个邻接来源让“奖励优化”等价于“全部后训练”的说法更不成立：过程是否可监控、微调是否泛化，仍需要各自的评价协议。

## 关键争论与分歧

- **`RLHF` 是否只是过渡技术**：`DPO / ORPO / KTO` 的出现说明，传统 `reward model + PPO` 并非唯一实现路径；但这并不足以推出“RLHF 已经过时”。这一争论只有在区分“概念范式”和“具体工程配方”后才成立。更稳妥的结论是：**完整 RLHF 与简化偏好目标并存，现有方法论文不足以统计其实际使用比例。**
- **偏好优化是否等于“不要 RL”**：从 `DPO` 的推导，到 `DeepSeekMath` 对不同优化形式的统一理解，现有证据更支持把很多“非 RL”方法理解为 **对 RLHF 的离线化、闭式化或重参数化**，而不是与 RL 完全断裂。只有在把“是否显式在线 rollout”误写成“是否仍属于奖励驱动策略优化”时，这个争论才会被过度简化。
- **对齐 RL 与推理 RL 是否应放在同一主题**：本页把两者放在一起，不是因为目标相同，而是因为它们共享奖励与策略优化语法。争论真正成立的前提是：必须承认 **“更符合用户偏好”** 与 **“更会求解复杂问题”** 不是同一个目标函数。也因此，`DeepSeek-R1` 不应被直接当作 `InstructGPT` 的自然后续，而应视为 RL 在 LLM 中功能重心的一次转移。
- **`SFT` 是否仍然必要**：`DeepSeek-R1-Zero` 在特定推理训练中探索了不以 SFT 为前置的路线；ORPO 合并监督与偏好，KTO 改变标签形式，二者不能直接证明 SFT 不必要；但当前可追溯证据同样显示，完全绕开 SFT 往往会带来可读性、语言稳定性与训练可控性问题。因此更稳妥的判断不是“有无 SFT 的二选一”，而是：**SFT 是否必要依训练目标、起点与任务；现有材料没有建立所有后训练的统一必要性定理。**
- **reasoning RL 的主要瓶颈是算法还是系统**：`GRPO` 给出了 reasoning RL 的代表性算法接口，但 `DAPO` 更强调系统工程细节的决定性作用。只要当前证据仍主要来自技术报告而不是统一对照实验，就不能草率地把收益归因给单一算法创新。
- **长程 agent RL 的“算法”边界在哪里**：`Kimi K3` 把 partial rollout、stale-data regularization、effort budget、MOPD、外部 cache pool、resumable microVM 与 verifiable environments 放入同一管线。其优势是端到端可执行，代价是难以用单一消融判断收益来自 policy objective、task distribution 还是 infrastructure。当前更稳妥的结论是：长程 agent RL 的优化对象已经从“单次回答”扩展到“持续变化的环境状态和计算预算”。
- **OPD 是蒸馏还是 RL**：`OPD` 表面上是 teacher-student distillation，但 `Yang et al. 2026` 把它解释为 dense KL-constrained RL 的特殊情形。这个争论的关键不是命名，而是训练信号来源：如果只看优化形式，它更像带 teacher implicit reward 的密集 RL；如果看监督接口，它仍依赖 teacher logits。当前更稳妥的写法是把它放在 `LLM RL` 的相邻层，而不是把它硬塞进 `DPO / ORPO / KTO` 偏好优化分支。
- **tool-use 是否仍属于 RL / post-training**：`DeepSeek-V3.2` 使边界变得更复杂。工具调用格式本身不是 RL，但如果模型通过大规模环境、复杂指令和 thinking-in-tool-use 训练获得持续执行能力，就不能只把它视为产品 API。当前更稳妥的组织方式是把它放在 `LLM RL` 与后续 agent topic 的交叉位置，而不是写入 `LLM 预训练` 的基础规律。
- **RL 收益应如何评估**：现有 summary 多集中于 benchmark、偏好胜率、数学与代码成绩；但对 reward hacking、过程可监控性、链式思维可读性与长期行为稳定性的证据仍不足。因此当前 topic 可以较稳地讨论“性能收益”，却还不能对“安全收益”或“长期可控性收益”下过强结论。

### 不能由方法名字推出的结论

ORPO 合并监督和偏好目标，不能作为 SFT 不必要的证据；KTO 使用单样本好坏标签，也没有普遍否定 SFT。现有报告证明了一些训练路线可行，没有证明经典 RLHF 的使用比例已经下降，本文不作市场趋势判断。R1-Zero 的结果也不足以将所有任务的 SFT 定位为可选项。对齐偏好、数学正确率、工具最终状态和安全监控是不同目标，必须分别报告。

## 证据基础

- [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](../../wiki/summaries/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)
- [Rafailov et al. - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model](../../wiki/summaries/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)
- [Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model](../../wiki/summaries/Hong%20et%20al.%20-%202024%20-%20ORPO%20Monolithic%20Preference%20Optimization%20without%20Reference%20Model.md)
- [Ethayarajh et al. - 2024 - KTO Model Alignment as Prospect Theoretic Optimization](../../wiki/summaries/Ethayarajh%20et%20al.%20-%202024%20-%20KTO%20Model%20Alignment%20as%20Prospect%20Theoretic%20Optimization.md)
- [Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models](../../wiki/summaries/Shao%20et%20al.%20-%202024%20-%20DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models.md)
- [DeepSeek-R1：奖励驱动推理与多阶段训练（2025）](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md)
- [DeepSeek AI - 2025 - DeepSeek-R1-0528 Release](../../wiki/summaries/DeepSeek%20AI%20-%202025%20-%20DeepSeek-R1-0528%20Release.md)
- [DeepSeek AI - 2025 - DeepSeek-V3.2 Release](../../wiki/summaries/DeepSeek%20AI%20-%202025%20-%20DeepSeek-V3.2%20Release.md)
- [Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale](../../wiki/summaries/Yu%20et%20al.%20-%202025%20-%20DAPO%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.md)
- [Yang et al. - 2026 - Learning beyond Teacher Generalized On-Policy Distillation with Reward Extrapolation](../../wiki/summaries/Yang%20et%20al.%20-%202026%20-%20Learning%20beyond%20Teacher%20Generalized%20On-Policy%20Distillation%20with%20Reward%20Extrapolation.md)
- [Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence](../../wiki/summaries/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)
- [Baker et al. - Unknown - Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation](../summaries/Baker%20et%20al.%20-%20Unknown%20-%20Monitoring%20Reasoning%20Models%20for%20Misbehavior%20and%20the%20Risks%20of%20Promoting%20Obfuscation.md)：补充代码 agent 中奖励作弊与推理监控的受限实验。
- [Wu et al. - 2025 - On the Generalization of SFT A Reinforcement Learning Perspective with Reward Rectification](../summaries/Wu%20et%20al.%20-%202025%20-%20On%20the%20Generalization%20of%20SFT%20A%20Reinforcement%20Learning%20Perspective%20with%20Reward%20Rectification.md)：补充 SFT 梯度修正的数学任务结果，避免把后训练等同于 RL。

## 代表页面

- [RLHF](../concepts/RLHF.md)
- [DPO](../concepts/DPO.md)
- [ORPO](../concepts/ORPO.md)
- [KTO](../concepts/KTO.md)
- [GRPO](../concepts/GRPO.md)
- [DAPO](../concepts/DAPO.md)
- [OPD](../concepts/OPD.md)
- [Instruction Tuning](../concepts/Instruction%20Tuning.md)
- [InstructGPT](../concepts/InstructGPT.md)
- [DeepSeek-R1](../concepts/DeepSeek-R1.md)
- [DeepSeek 系列](./DeepSeek%20系列.md)
- [Kimi K3](../concepts/Kimi%20K3.md)
- [RLHF vs DPO vs ORPO vs KTO](../comparisons/RLHF%20vs%20DPO%20vs%20ORPO%20vs%20KTO.md)

## 未解决问题

- 偏好奖励、结果验证与 teacher 概率各会带来什么行为偏差？高分可能来自奖励漏洞，监控实验不证明全部推理链忠实。
- 如何在采样、更新、teacher 评分、验证和长期环境间分配总预算？DAPO 步数和 K3 系统报告没有统一成本最优。
- 长轨迹、延迟验证和分布漂移如何影响稳定性？partial rollout 和概率外推的跨模型、跨任务有效区间仍需对照。
- 何种任务需要 SFT 起点？R1-Zero 的受限结果不足以为全部任务取消示范训练。

## 关联页面

- [指令对齐与 post-training](./指令对齐与%20post-training.md)
- [LLM 预训练](../topics/LLM%20预训练.md)
- [DeepSeek](../concepts/DeepSeek.md)
- [FLAN](../concepts/FLAN.md)
- [LoRA](../concepts/LoRA.md)
- [OPT-IML](../concepts/OPT-IML.md)
- [Prompt Tuning](../concepts/Prompt%20Tuning.md)
- [RLHF](../concepts/RLHF.md)
- [DPO](../concepts/DPO.md)
- [ORPO](../concepts/ORPO.md)
- [KTO](../concepts/KTO.md)
- [GRPO](../concepts/GRPO.md)
- [DAPO](../concepts/DAPO.md)
- [OPD](../concepts/OPD.md)
- [DeepSeek 系列](./DeepSeek%20系列.md)
- [Kimi](../concepts/Kimi.md)
- [Kimi K3](../concepts/Kimi%20K3.md)
- [Toolformer](../concepts/Toolformer.md)
- [Llama Guard](../concepts/Llama%20Guard.md)
- [RLHF vs DPO vs ORPO vs KTO](../comparisons/RLHF%20vs%20DPO%20vs%20ORPO%20vs%20KTO.md)
- [AI 能力评测：任务、过程与预测](../comparisons/AI%20%E8%83%BD%E5%8A%9B%E8%AF%84%E6%B5%8B%EF%BC%9A%E4%BB%BB%E5%8A%A1%E3%80%81%E8%BF%87%E7%A8%8B%E4%B8%8E%E9%A2%84%E6%B5%8B.md)：基准成绩、探索案例、过程监控、社会趋势与未来预测是不同证据。先判断材料属于哪一种，再看数据、评价和外推条件；多篇材料不能自动拼成 AGI 已实现的证明。

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **rollout**：采样执行：用当前策略生成回答或连续行动，形成后续训练与评价的材料。
- **critic**：价值模型：估计状态或行为的预期回报，为策略更新提供参照。
- **reward model**：奖励模型：根据训练信号给回答或行为打分，分数是目标的近似。
- **distillation**：蒸馏：利用教师模型提供的答案或分布训练学生模型。
- **SFT**：监督微调：用输入与参考输出继续训练已有模型。

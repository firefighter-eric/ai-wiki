---
type: topic
status: formal
review_scope: evidence_synthesis
reviewed: 2026-10-07
---
# DeepSeek 系列

## TL;DR（快速导读）

DeepSeek 的效率架构、推理后训练和文档视觉压缩是不同技术线。V3/V4 改计算与缓存，Math/R1 改奖励驱动行为，OCR 研究页面转写与压缩；它们的成绩需要分别看实验条件。

阅读重点：先按问题选择路线，再核对比较条件与证据边界。

## 先用一个问题理解

研究训练成本先看 V3，研究奖励怎样形成推理行为看 R1，研究长文推理成本看缓存和压缩。把这些结果合成家族判断前，应确认每份报告的版本与实验条件。

## 页面状态

正式 topic；2026-10-07 复核核心来源并补充方法比较。正文区分论文实验、作者报告和本文综合判断；开放问题表示研究证据的边界。

## 主题定义

本页讨论 DeepSeek 作为一个模型技术系列的主线：它如何从高效 MoE 预训练骨干，扩展到 reasoning-oriented RL，再进一步进入 thinking tool-use、百万 token 长上下文 agent 工作流，以及把 OCR 作为 long-context compression 实验场的文档模型分支。这里的重点不是给 DeepSeek 做总排名，也不是讨论商业价格、芯片生态或政策争议，而是解释这些节点之间的技术关系。

DeepSeek 系列最容易被误读为几个彼此独立的爆点：`DeepSeek-V3` 是高效开放 MoE，`DeepSeek-R1` 是推理模型，`DeepSeek-V4` 是百万上下文，`DeepSeek-OCR` 是文档识别。但从当前知识库的 evidence base 看，更稳妥的理解是：这些节点共同围绕一个问题展开，即 **如何在可控推理成本下获得更大的有效容量、更长的可用上下文和更强的长程任务执行能力**。

因此，本页把 DeepSeek 组织成一条技术主线，而不是简单模型目录：`V3` 提供 MoE 与高效训练骨干，`DeepSeekMath / GRPO / R1` 把后训练目标推进到推理行为塑形，`R1-0528` 与 `V3.2` 把 reasoning 接入更稳定的 JSON / function calling / tool-use 接口，`V4` 把长上下文成本和 agent workflow 作为架构目标，`DeepSeek-OCR` 则从文档视觉压缩侧面回应同一个长上下文效率问题。

## 核心问题

- **DeepSeek 的核心创新到底在模型规模、训练算法，还是系统效率**：不同节点给出的答案不同，必须分层讨论。
- **MoE 与 reasoning RL 如何连接**：`V3` 的高效基座并不自动解释 `R1`，但它提供了后续 reasoning model 的能力底座。
- **长上下文是否只是窗口变大**：`V4` 与 `DeepSeek-OCR` 都说明问题重点在 token / KV / vision token 的成本结构，而不是只把 context length 写大。
- **tool-use 与 agent 能力应归入预训练、后训练还是系统层**：`V3.2` 与 `V4` 都把这些能力放到发布主张里，但 topic 组织上需要把 base model 架构、reasoning mode 和工具接口拆开。
- **DeepSeek-OCR 是否只是 OCR 分支**：它当然属于 OCR，但在 DeepSeek 系列里还承担 long-context compression 的实验意义。

## 主线脉络 / 方法分层

- **高效 MoE 基座层**：`DeepSeek-V3` 的核心位置在于把 DeepSeek 家族的能力底座绑定到 `MoE`、`MLA` 与多 token 预测等效率设计上。它不是单纯“更大参数”的节点，而是把总参数容量、激活参数成本和开放模型能力放在同一套工程叙事里。后续 `R1` 与 `V4` 都借用了这个底座叙事：前者在其上做 reasoning 行为塑形，后者则继续把稀疏效率推到百万上下文。
- **reasoning RL 起点层**：`DeepSeekMath` 的重要性不只是数学能力提升，而是提出并使用 `GRPO`，把 critic-free、group-relative advantage 的策略优化方式引入 reasoning 训练。`DeepSeek-R1` 进一步把这种路线放大到通用推理模型：`R1-Zero` 显示大规模 RL 可以直接激发推理行为，但也暴露可读性和语言混杂问题；`R1` 则通过冷启动数据和多阶段训练把能力与可用性重新平衡。
- **推理模型可用性层**：`DeepSeek-R1-0528` 说明 `R1` 主线没有停在研究报告里，而是继续补 API 与交互接口，包括 JSON output、function calling、前端能力和幻觉降低。这里的重点不是重新定义 RL 算法，而是说明 reasoning model 正在向工具可用、格式可控的产品接口移动。
- **thinking tool-use 层**：`DeepSeek-V3.2` 是 `R1` 到 `V4` 之间最关键的桥。它把 reasoning-first model 明确写成 agent 目标，官方发布页强调 `1,800+` environments、`85k+` complex instructions 的 agent 训练数据合成，并把 thinking 直接集成到 tool-use。这个节点说明 DeepSeek 的推理路线开始从“会解复杂题”转向“能在工具环境中持续执行任务”。
- **百万上下文效率层**：`DeepSeek-V4` 把 DeepSeek 的效率问题推到更长上下文场景。它继续使用 MoE，但关键不只是总参数，而是 `CSA / HCA` 混合 attention、`mHC` 和 `Muon` 等设计如何压低超长上下文中的 attention FLOPs、`KV cache` 成本和训练不稳定性。`V4-Pro / Flash` 的分层也说明 DeepSeek 在同一代模型里显式区分能力上限与推理成本。
- **V4 架构机制层**：`CSA` 与 `HCA` 应被拆开理解。`CSA` 先压缩 `KV cache`，再通过 sparse selection 从 compressed KV entries 中选取 top-k 参与 attention，保留压缩后的相关性选择；`HCA` 则用更大的压缩率做重度 KV 合并，更偏向把百万上下文的缓存成本压到足够低。`mHC` 不处理 attention，而是约束 residual mapping 来增强深层 Transformer block 间的信号传播稳定性；`Muon` 也不是模型结构，而是用于多数模块的优化器，服务于收敛速度与训练稳定性。这样看，`V4` 的创新不是一个单点 attention trick，而是 **长上下文 attention 压缩 + residual 稳定连接 + optimizer 工程** 的组合。
- **视觉压缩与文档分支层**：`DeepSeek-OCR` 与 `DeepSeek-OCR 2` 看似是 OCR 模型，实际和 DeepSeek 主线共享一个问题意识：长上下文太贵时，能否通过视觉表示压缩文本信息。初代 `DeepSeek-OCR` 把 OCR 写成 `vision-text compression` 实验；`DeepSeek-OCR 2` 再用 visual causal flow 尝试让视觉 token 按语义顺序重排。它们在 OCR topic 中是专门文档 VLM，在 DeepSeek topic 中则是长上下文 token economy 的旁支证据。

如果压缩成一句话，DeepSeek 系列的技术主线是：**先用稀疏 MoE 降低能力扩张的单位成本，再用 RL 塑造推理行为，随后把推理行为接入工具和长上下文工作流，最后从文本与视觉两侧同时压缩上下文成本。**

### 将架构收益、后训练收益和系统收益分开

[V3 报告](../summaries/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)的 MLA 压缩 KV 表示，MoE 控制每 token 激活容量；训练配置使用 AdamW，不能因为后来的 V4 使用 Muon 而回写 V3。[V4 报告](../summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)进一步引入 CSA/HCA、mHC 和优化器配方。缓存变小主要影响长序列资源，残差映射主要影响信号传播，优化器改变训练更新：三类指标分别是推理内存/算量、训练稳定性和收敛，不能把它们合成一个无条件加速数字。

[DeepSeekMath](../summaries/Shao%20et%20al.%20-%202024%20-%20DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models.md)中的 GRPO 从同一题的一组输出估计相对优势，移除独立 critic；它仍依赖奖励、reference 和 rollout，不是“不需要别的模型”。[R1](../summaries/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md)中的 Zero 与正式 R1 也必须区分：前者验证无 SFT 冷启动的 RL 路线，后者通过冷启动和多阶段训练改善可读性。学生蒸馏成绩则是第三种设定，不能当作同一个大模型的训练消融。

| 分支 | 主要改变 | 支持的判断 | 尚不能支持的判断 |
| --- | --- | --- | --- |
| V3 / V4 | 激活容量、注意力与缓存、训练稳定性 | 组合设计降低特定成本 | 任一组件单独解释总成绩 |
| Math / R1 | 奖励与策略更新、训练阶段 | 可验证任务上的行为改善 | 无监督获得任意知识或可靠长程代理 |
| OCR / OCR 2 | 页面视觉编码与 token 分配 | 文档转写和压缩设定下的结果 | 无损保存所有长期记忆 |

[DeepSeek-OCR](../summaries/Wei%2C%20Sun%2C%20Li%20-%202025%20-%20DeepSeek-OCR%20Contexts%20Optical%20Compression.md)的压缩率与恢复成绩来自特定页面、视觉 token 数及输出协议；语言模型补全可能得到可读文本，也可能掩盖字符错误。[OCR 2](../summaries/Wei%2C%20Sun%2C%20Li%20-%202026%20-%20DeepSeek-OCR%202%20Visual%20Causal%20Flow.md)重排视觉信息流，不能由此推导可以对任意历史对话无限压缩。本文把两者放入 token 成本主线是研究动机的联系，不表示它们已验证 V4 的文本缓存算法。

## 关键争论与分歧

- **DeepSeek 的主线是否应写成“开源模型追赶闭源模型”**：这只覆盖一部分事实。`V3 / V4` 的 open-weight 与成本效率很重要，但 `R1` 的意义在 reasoning RL，`V3.2` 的意义在 thinking tool-use，`DeepSeek-OCR` 的意义在 vision-text compression。把它们都写成“开源追赶”会损失技术结构。
- **reasoning 能力主要来自基座还是 RL**：当前证据支持分层写法。`V3` 提供强基座，但 `DeepSeekMath / R1` 的核心主张是后训练能直接激励推理行为；同时 `R1-Zero` 的可读性问题又说明单靠 RL 并不足以给出稳定交互模型。
- **tool-use 是否只是产品功能**：`V3.2` 的发布页把 tool-use 与大规模 agent 数据合成直接绑定，因此它不只是 UI 功能。但它也不应被写成预训练规律；更合适的位置是 reasoning / agent 后训练与系统接口之间。
- **1M context 是否等于真正解决长上下文**：`V4` 与 `DeepSeek-OCR` 都提示，长上下文的关键是成本与可用性。百万 token 是能力上限和系统目标，实际效果仍受推理框架、KV 存储、任务形态和第三方复现影响。
- **V4 的四个机制能否外推成通用范式**：`CSA / HCA / mHC / Muon` 目前更稳妥地说是 `DeepSeek-V4` 的组合式工程答案。`CSA / HCA` 解决长上下文 attention 与 KV 成本，`mHC` 解决深层信号传播，`Muon` 解决大规模训练优化；它们可能互相配合，而不是任一组件单独解释全部收益。当前不应把它们写成已被社区验证的通用最优解。
- **DeepSeek-OCR 是否应纳入 DeepSeek 系列主线**：应纳入，但要限定其角色。它不是通用 LLM 主干，而是 DeepSeek 对 long-context compression 的视觉侧实验，事实来源仍应主要放在 OCR 与文档理解语境中使用。

### 官方发布与技术因果的证据等级

R1-0528 与 V3.2 发布页能证明团队声明了哪些接口与训练范围，不能替代独立的能力复现。V4 技术报告有更具体的架构和实验条件，但跨硬件、跨任务的通用优势仍需对照。页面里的“百万上下文”应解释为给定模型和运行设定的支持规模，读取过一百万 token 不等于能可靠回答其中每个跨文档关系。

## 证据基础

- [Unknown - 2024 - DeepSeek-V3 Technical Report](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)
- [Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models](../../wiki/summaries/Shao%20et%20al.%20-%202024%20-%20DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models.md)
- [DeepSeek-R1：奖励驱动推理与多阶段训练（2025）](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md)
- [DeepSeek AI - 2025 - DeepSeek-R1-0528 Release](../../wiki/summaries/DeepSeek%20AI%20-%202025%20-%20DeepSeek-R1-0528%20Release.md)
- [DeepSeek AI - 2025 - DeepSeek-V3.2 Release](../../wiki/summaries/DeepSeek%20AI%20-%202025%20-%20DeepSeek-V3.2%20Release.md)
- [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](../../wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)
- [Wei, Sun, Li - 2025 - DeepSeek-OCR Contexts Optical Compression](../../wiki/summaries/Wei,%20Sun,%20Li%20-%202025%20-%20DeepSeek-OCR%20Contexts%20Optical%20Compression.md)
- [Wei, Sun, Li - 2026 - DeepSeek-OCR 2 Visual Causal Flow](../../wiki/summaries/Wei,%20Sun,%20Li%20-%202026%20-%20DeepSeek-OCR%202%20Visual%20Causal%20Flow.md)

## 代表页面

- [DeepSeek](../concepts/DeepSeek.md)
- [DeepSeek-V3](../concepts/DeepSeek-V3.md)
- [DeepSeek-R1](../concepts/DeepSeek-R1.md)
- [DeepSeek-V4](../concepts/DeepSeek-V4.md)
- [DeepSeek-OCR](../concepts/DeepSeek-OCR.md)
- [GRPO](../concepts/GRPO.md)
- [MoE](../concepts/MoE.md)
- [Compressed Sparse Attention](../concepts/Compressed%20Sparse%20Attention.md)
- [Heavily Compressed Attention](../concepts/Heavily%20Compressed%20Attention.md)
- [Manifold-Constrained Hyper-Connections](../concepts/Manifold-Constrained%20Hyper-Connections.md)
- [Muon](../concepts/Muon.md)
- [LLM 预训练](./LLM%20预训练.md)
- [LLM RL](./LLM%20RL.md)
- [OCR](./OCR.md)

## 未解决问题

- V4 的注意力压缩、残差设计和优化器在不同长度与硬件上各贡献多少？组合报告不足以给组件的普遍因果结论。
- Math/R1 的可验证奖励怎样迁移到开放任务？推理分数、可读性与工具最终状态仍需分别评价。
- 视觉压缩何时丢掉决定答案的字符和结构？OCR 恢复成绩还不能证明长期记忆可靠性。

## 关联页面

- [LLM 预训练](./LLM%20预训练.md)
- [LLM RL](./LLM%20RL.md)
- [OCR](./OCR.md)
- [开放模型家族与中国重要家族对照](../comparisons/开放模型家族与中国重要家族对照.md)
- [DeepSeek](../concepts/DeepSeek.md)
- [DeepSeek-V3](../concepts/DeepSeek-V3.md)
- [DeepSeek-R1](../concepts/DeepSeek-R1.md)
- [DeepSeek-V4](../concepts/DeepSeek-V4.md)
- [DeepSeek-OCR](../concepts/DeepSeek-OCR.md)
- [GRPO](../concepts/GRPO.md)
- [MoE](../concepts/MoE.md)
- [Compressed Sparse Attention](../concepts/Compressed%20Sparse%20Attention.md)
- [Heavily Compressed Attention](../concepts/Heavily%20Compressed%20Attention.md)
- [Manifold-Constrained Hyper-Connections](../concepts/Manifold-Constrained%20Hyper-Connections.md)
- [Muon](../concepts/Muon.md)

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **KV cache**：键值缓存：保存已经处理过的位置表示，生成新内容时可复用，避免全部重算。
- **critic**：价值模型：估计状态或行为的预期回报，为策略更新提供参照。
- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。
- **sparse**：稀疏计算或连接：只使用选中的部分，具体省略什么取决于方法。
- **OCR**：文字识别：从图像读取文字；整页任务还需处理布局和阅读顺序。

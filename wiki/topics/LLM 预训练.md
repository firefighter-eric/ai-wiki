---
type: topic
status: formal
review_scope: evidence_synthesis
reviewed: 2026-10-07
---
# LLM 预训练

## TL;DR（快速导读）

预训练比较要同时看目标、数据、参数激活、计算预算和训练公开程度。模型最终的聊天、推理或 agent 成绩还受后训练与运行系统影响，不能全部归因于预训练。

阅读重点：先按问题选择路线，再核对比较条件与证据边界。

## 先用一个问题理解

固定预算训练语言模型时，要同时决定模型多大、读哪些数据和训练多久。较大模型看得不够，与较小模型训练充分，可能产生不同结果；先区分这类底座问题，再看指令或推理适配。

## 页面状态

正式 topic；2026-10-07 复核核心来源并补充方法比较。正文区分论文实验、作者报告和本文综合判断；开放问题表示研究证据的边界。

## 主题定义

本页讨论 **大语言模型在 post-training 之前，如何通过大规模自监督训练获得通用能力底座**。这里的重点是预训练目标、规模化规律、数据与计算预算配置、开放模型家族的训练取向，以及 dense 与 sparse 路线的结构性分化。它 **不讨论** RLHF、DPO、GRPO 这类后训练行为塑形方法，也不把工具调用、多模态或 agent 行为当作预训练页的直接中心，除非它们能被明确回收到“能力底座如何形成”这一问题。

本页的边界必须严格，因为当前很多模型报告会把预训练、SFT、RL、部署工程与产品能力写在同一份技术文档里。知识组织上，如果不把 **“能力底座”** 与 **“行为改写”** 分开，`LLM 预训练` 就会退化成一个总目录页。当前更稳妥的理解是：**预训练决定模型大致会不会、能不能、会到什么程度；post-training 决定这些能力怎样被组织成可交互、可约束、可产品化的行为接口。**

从现有 summary 来看，本页的主线不是“某一家模型赢了什么 benchmark”，而是三层连续变化：第一，`GPT-3 / PaLM` 所代表的 **dense scaling** 如何证明大规模自回归预训练能产生通用 few-shot 能力；第二，`Chinchilla` 如何把讨论从“继续变大”修正为“在固定算力下合理配置参数量与 token 数”；第三，开放模型家族如何在这个框架下分化出多语言、代码、MoE、本地部署与 fully open 等不同竞争方向。

## 核心问题

- **通用语言能力主要如何从预训练中形成**，以及这种能力与后训练行为改善应如何切分。
- dense scaling、compute-optimal 修正与 sparse/MoE 路线之间的关系是什么，哪些是补充，哪些是路线分化。
- 开放模型家族之间真正可比的维度是什么：参数规模、数据规模、训练效率、语言覆盖、透明度，还是部署友好性。
- “更大的模型”与“更合理的数据和计算配置”之间，哪个更应被视为预训练阶段的核心驱动力。
- 当前知识库对 LLM 主线的叙述，应该以闭源标杆为骨架，还是以开放家族竞争格局为骨架。

## 主线脉络 / 方法分层

本页按 **能力形成逻辑** 分层，而不是按模型发布时间罗列。因为对预训练的理解，关键不在于记住家族名单，而在于把“为何能力出现”“如何更有效训练”“为何开放家族分叉”放到同一结构里。

- **dense scaling 证明期**：`Brown et al. 2020` 与 `Chowdhery et al. 2022` 共同支撑了预训练时代的第一个核心判断：**在自回归语言建模框架下，随着参数、数据与训练系统规模扩大，模型会出现更强的 few-shot 与跨任务泛化能力。** `GPT-3` 的意义在于让 prompt 成为任务接口；`PaLM` 的意义在于说明这条路线在更大训练系统、更多语言与代码场景下仍然成立。
- **compute-optimal 修正期**：`Hoffmann et al. 2022` 并没有推翻 dense scaling，而是修正其粗糙版本。它指出许多早期大模型不是“参数不够大”，而是 **在既定计算预算下 token 训练不足**。因此，本页理解 `Chinchilla` 的正确方式，不是“从大模型转向小模型”，而是从“只扩参数”转向 **参数量与数据量的联合最优配置**。这是预训练叙事里最重要的纠偏节点。
- **能力底座与行为塑形分层期**：`Ouyang et al. 2022` 之所以应在本页中被提及，不是因为它属于预训练，而是因为它为“预训练页的边界”提供了反证。即使 base model 已很强，它仍不会自动变成 helpful、truthful、harmless 的交互系统。这一事实支持一个重要结构判断：**预训练主要建立底座，后训练主要适配行为和任务；这是一种分析分工，并非知识贡献的绝对划线。**
- **开放模型家族并行竞争期**：`LLaMA / Llama 2 / Llama 3 / Mistral / Mixtral / Gemma / Gemma 4 / OLMo 2 / DBRX / OpenELM / Falcon 3 / BLOOM / StarCoder 2 / GLM-130B / Qwen / DeepSeek-V3 / DeepSeek-V4 / Kimi K3` 等来源共同表明，预训练主线已从闭源演示阶段进入多家族并行推进阶段。这里真正的分化并不只是“是否开源”，而是 **多语言覆盖、代码能力、上下文长度、训练效率、MoE 采用、研究透明度与部署形态** 的组合差异。
- **sparse scaling 与效率导向期**：`Mixtral`、`DBRX`、`DeepSeek-V3 / DeepSeek-V4` 与 `Kimi K3` 等节点说明，预训练不再只沿 dense Transformer 一条线扩张。MoE 的引入使“总参数规模”与“单 token 激活成本”发生脱钩，预训练讨论因此从“模型有多大”转向“**每单位计算预算能激活多强的有效容量**”。`DeepSeek-V4` 把问题推进到百万 token 下的 attention FLOPs、KV compression 与 residual stability；`Kimi K3` 则用 896-expert `Stable LatentMoE`、KDA/MLA hybrid attention、AttnRes、Quantile Balancing 与 Per-Head Muon，把 sparse scaling 进一步绑定到 recurrent state、跨深度信息流和 expert-parallel balance。因此 sparse scaling 已经不只是训练容量问题，也变成了长上下文推理工程、优化稳定性与 distributed execution 的共同问题。
- **中国重要家族与全球开放主线交叉期**：`GLM-130B`、`Qwen`、`DeepSeek-V3 / V4` 与 `Kimi` 相关来源说明，中国模型竞争不应被简化为 “Qwen 对其他一切”。Kimi 过去只能作为高影响非 open-weight 对照，但 `Kimi K3` 已发布完整权重，使该家族也进入 open-weight frontier 主线；与此同时，自定义许可证、训练透明度与集群级部署门槛仍要求把 `open-weight`、`fully open` 和 API 可用性分开。

如果进一步压缩，本页方法分层可概括为：**dense 能力形成**、**compute-optimal 预算修正**、**开放家族分叉**、**sparse 效率扩张**。这四层共同构成当前 LLM 预训练叙事的稳定骨架。

### 从模型名单转为训练条件比较

| 比较轴 | 需要一起记录的条件 | 会造成的误判 |
| --- | --- | --- |
| 模型规模 | 总参数、每 token 激活参数、层数与宽度 | MoE 总参数直接等同 dense 计算 |
| 数据规模 | token 数、重复、语言/代码比例和来源 | 更多 token 必然更高质量 |
| 训练目标 | causal、blank infilling、混合任务目标 | 同参数模型都在优化同一任务 |
| 训练资源 | 总 FLOPs、精度、通信和硬件利用率 | 实际训练时间只由参数决定 |
| 评价阶段 | base / instruction / reasoning，提示与采样 | post-training 成绩反写为基座收益 |

[LLaMA](../summaries/Touvron%20et%20al.%20-%202023%20-%20LLaMA%20Open%20and%20Efficient%20Foundation%20Language%20Models.md)提供了小于某些早期模型、但训练 token 更充分的实践；[Chinchilla](../summaries/Hoffmann%20et%20al.%20-%202022%20-%20Training%20Compute-Optimal%20Large%20Language%20Models.md)研究固定训练计算预算，两者都说明参数不是唯一投入，但不能因此建立永久的 token/参数比例。[GLM-130B](../summaries/Zeng%20et%20al.%20-%202022%20-%20GLM-130B%20An%20Open%20Bilingual%20Pre-trained%20Model.md)采用 blank infilling 与少量多任务目标，不能纳入纯 causal 目标的无差别横比。[Mixtral](../summaries/Jiang%20et%20al.%20-%202024%20-%20Mixtral%20of%20Experts.md)与[DBRX](../summaries/Databricks%20-%202024%20-%20DBRX%20A%20Highly%20Efficient%20Open%20LLM.md)改变专家选择；DBRX 本页依据官方发布快照，历史误配数学 PDF 不再提供证据。

“开放”也应分为权重、训练数据、代码、检查点和许可条件。[OLMo 2](../summaries/Ai2%20-%202024%20-%20OLMo%202%20The%20Best%20Fully%20Open%20Language%20Model%20to%20Date.md)强调训练链公开，[K3 许可证](../summaries/Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20License.md)则有明确商业条件；二者不能通过同一个 open-weight 标签消除差异。公开权重降低访问门槛，不说明其训练实验可完整重建，也不说明一台消费级设备能运行全部型号。

对家族技术效果作因果判断，需要控制其他变量。V4/K3 报告将架构、优化器、数据和后训练共同改动；即使最终基准进步，也不能将全部收益分给单个注意力设计。本文较稳定的综合结论是：预训练决定可供后训练使用的表示与生成底座，最终系统表现还由数据接口、推理预算和工具环境共同决定。

## 关键争论与分歧

- **更大是否仍是最主要驱动力**：现有证据支持“规模仍然关键”，但不再支持“只扩参数即可”的朴素版本。这个争论真正成立的前提是：区分 **规模本身有效** 与 **规模配置是否合理**。`Chinchilla` 修正的是后者，而不是前者。
- **dense 与 sparse 哪条更代表未来主线**：当前 summary 仍以 dense scaling 为能力讨论的共同语言，但 `Mixtral`、`DBRX`、`DeepSeek-V3` 说明 sparse/MoE 已经成为现实工程路线。现阶段更稳妥的结论不是“dense 被 sparse 替代”，而是：**dense 仍提供主干理论语言，sparse 则在工程竞争中不断扩大实际权重。**
- **预训练与后训练应如何分界**：许多技术报告会把预训练、SFT、RL 一并叙述，尤其是 `DeepSeek-V3 / DeepSeek-V4 / Kimi K3` 一类综合性报告更容易模糊边界。但只要当前知识库仍把“能力底座”与“行为塑形”视为两阶段结构，就不应把强 post-training 或 agent benchmark 效果反写成预训练规律本身。
- **开放模型是否主要只是分发策略差异**：当前证据不支持这种过窄理解。`BLOOM`、`OLMo 2` 强调研究透明度；`Gemma` 到 `Gemma 4` 强调 practical size、开放多模态、`MoE` 与本地/工作站部署；`OpenELM` 与 `Phi-3` 强调端侧与效率；`Qwen`、`Llama`、`DeepSeek` 强调家族化延展。也就是说，开放模型之间存在真实技术分化，而不仅是 license 分化。
- **家族级开放标签是否稳定**：`Kimi` 说明答案是否定的。`k1.5` 时代的来源只能支撑高影响 API/报告节点，而 `Kimi K3` 已是完整权重发布；因此开放性必须绑定具体代际和许可证。反过来，K3 的自定义商业条件与未完全公开的训练链路也说明 `open-weight` 仍不能自动升级为 `fully open research release`。
- **预训练是否已经足以解释当前模型差异**：随着 agent、多模态与 tool use 路线扩张，单靠预训练已难解释全部产品能力差异。当前证据仍支持本页把预训练当作能力骨干，但也支持一个限制性判断：**预训练已不再独自解释最终系统表现。**

### 本库可以比较路线，不能给统一冠军

GPT-3、PaLM、LLaMA 和近期 MoE 报告使用不同基准、数据时间与提示。历史基准还有污染检查和训练数据公开程度的差异。没有共同协议时，按技术条件比较比汇总最高分更可靠；若要形成性能排名，应另做同任务、同推理预算的评测。文中的“能力底座”与“行为塑形”是分析分工，实际后训练也会增加知识、工具模式和新任务经验，不能解释成知识只在预训练产生。

## 证据基础

- [Brown et al. - 2020 - Language models are few-shot learners](../../wiki/summaries/Brown%20et%20al.%20-%202020%20-%20Language%20models%20are%20few-shot%20learners.md)
- [Chowdhery et al. - 2022 - PaLM Scaling Language Modeling with Pathways](../../wiki/summaries/Chowdhery%20et%20al.%20-%202022%20-%20PaLM%20Scaling%20Language%20Modeling%20with%20Pathways.md)
- [Hoffmann et al. - 2022 - Training Compute-Optimal Large Language Models](../../wiki/summaries/Hoffmann%20et%20al.%20-%202022%20-%20Training%20Compute-Optimal%20Large%20Language%20Models.md)
- [Touvron et al. - 2023 - LLaMA Open and Efficient Foundation Language Models](../../wiki/summaries/Touvron%20et%20al.%20-%202023%20-%20LLaMA%20Open%20and%20Efficient%20Foundation%20Language%20Models.md)
- [Touvron et al. - 2023 - Llama 2 Open Foundation and Fine-Tuned Chat Models](../../wiki/summaries/Touvron%20et%20al.%20-%202023%20-%20Llama%202%20Open%20Foundation%20and%20Fine-Tuned%20Chat%20Models.md)
- [Roziere et al. - 2023 - Code Llama Open Foundation Models for Code](../../wiki/summaries/Roziere%20et%20al.%20-%202023%20-%20Code%20Llama%20Open%20Foundation%20Models%20for%20Code.md)
- [Scao et al. - 2022 - BLOOM A 176B-Parameter Open-Access Multilingual Language Model](../../wiki/summaries/Scao%20et%20al.%20-%202022%20-%20BLOOM%20A%20176B-Parameter%20Open-Access%20Multilingual%20Language%20Model.md)
- [MosaicML - 2023 - MPT-7B](../../wiki/summaries/MosaicML%20-%202023%20-%20MPT-7B.md)
- [Jiang et al. - 2023 - Mistral 7B](../../wiki/summaries/Jiang%20et%20al.%20-%202023%20-%20Mistral%207B.md)
- [Jiang et al. - 2024 - Mixtral of Experts](../../wiki/summaries/Jiang%20et%20al.%20-%202024%20-%20Mixtral%20of%20Experts.md)
- [Team, Google - 2024 - Gemma Open Models Based on Gemini Research and Technology](../../wiki/summaries/Team,%20Google%20-%202024%20-%20Gemma%20Open%20Models%20Based%20on%20Gemini%20Research%20and%20Technology.md)
- [Team, Google DeepMind - 2024 - Gemma 2 Improving Open Language Models at a Practical Size](../../wiki/summaries/Team,%20Google%20DeepMind%20-%202024%20-%20Gemma%202%20Improving%20Open%20Language%20Models%20at%20a%20Practical%20Size.md)
- [Google DeepMind - 2026 - Gemma 4 Model Card](../../wiki/summaries/Google%20DeepMind%20-%202026%20-%20Gemma%204%20Model%20Card.md)
- [Lozhkov et al. - 2024 - StarCoder 2 and The Stack v2 The Next Generation](../../wiki/summaries/Lozhkov%20et%20al.%20-%202024%20-%20StarCoder%202%20and%20The%20Stack%20v2%20The%20Next%20Generation.md)
- [DBRX：Databricks 官方模型发布说明](../../wiki/summaries/Databricks%20-%202024%20-%20DBRX%20A%20Highly%20Efficient%20Open%20LLM.md)
- [Mehta et al. - 2024 - OpenELM An Efficient Language Model Family with Open Training and Inference Framework](../../wiki/summaries/Mehta%20et%20al.%20-%202024%20-%20OpenELM%20An%20Efficient%20Language%20Model%20Family%20with%20Open%20Training%20and%20Inference%20Framework.md)
- [Abdin et al. - 2024 - Phi-3 Technical Report A Highly Capable Language Model Locally on Your Phone](../../wiki/summaries/Abdin%20et%20al.%20-%202024%20-%20Phi-3%20Technical%20Report%20A%20Highly%20Capable%20Language%20Model%20Locally%20on%20Your%20Phone.md)
- [Ai2 - 2024 - OLMo 2 The Best Fully Open Language Model to Date](../../wiki/summaries/Ai2%20-%202024%20-%20OLMo%202%20The%20Best%20Fully%20Open%20Language%20Model%20to%20Date.md)
- [TII - 2024 - Falcon 3](../../wiki/summaries/TII%20-%202024%20-%20Falcon%203.md)
- [Zeng et al. - 2022 - GLM-130B An Open Bilingual Pre-trained Model](../../wiki/summaries/Zeng%20et%20al.%20-%202022%20-%20GLM-130B%20An%20Open%20Bilingual%20Pre-trained%20Model.md)
- [Kimi Team et al. - 2025 - Kimi k1.5 Scaling Reinforcement Learning with LLMs](../../wiki/summaries/Kimi%20Team%20et%20al.%20-%202025%20-%20Kimi%20k1.5%20Scaling%20Reinforcement%20Learning%20with%20LLMs.md)
- [Bai et al. - 2023 - Qwen Technical Report](../../wiki/summaries/Bai%20et%20al.%20-%202023%20-%20Qwen%20Technical%20Report.md)
- [Dubey et al. - 2024 - The Llama 3 Herd of Models](../../wiki/summaries/Dubey%20et%20al.%20-%202024%20-%20The%20Llama%203%20Herd%20of%20Models.md)
- [Unknown - 2024 - DeepSeek-V3 Technical Report](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)
- [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](../../wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)
- [Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence](../../wiki/summaries/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)
- [Moonshot AI - 2026 - Kimi K3 License](../summaries/Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20License.md)：补充具体代际许可边界，不作为训练能力证据。

## 代表页面

- [GPT-3](../concepts/GPT-3.md)
- [PaLM](../concepts/PaLM.md)
- [Chinchilla](../concepts/Chinchilla.md)
- [指令对齐与 post-training](./指令对齐与%20post-training.md)
- [LLM RL](./LLM%20RL.md)
- [T5](../concepts/T5.md)
- [Switch Transformer](../concepts/Switch%20Transformer.md)
- [OPT](../concepts/OPT.md)
- [mT5](../concepts/mT5.md)
- [Qwen](../concepts/Qwen.md)
- [Llama 家族](../concepts/Llama%20家族.md)
- [LLaMA（初代）](../concepts/LLaMA%20初代.md)
- [Llama 2](../concepts/Llama%202.md)
- [Code Llama](../concepts/Code%20Llama.md)
- [Llama 3](../concepts/Llama%203.md)
- [BLOOM](../concepts/BLOOM.md)
- [MPT](../concepts/MPT.md)
- [Mistral 7B](../concepts/Mistral%207B.md)
- [Mixtral](../concepts/Mixtral.md)
- [Gemma](../concepts/Gemma.md)
- [Gemma 2](../concepts/Gemma%202.md)
- [Gemma 3](../concepts/Gemma%203.md)
- [Gemma 4](../concepts/Gemma%204.md)
- [DiffusionGemma](../concepts/DiffusionGemma.md)
- [StarCoder2](../concepts/StarCoder2.md)
- [DBRX](../concepts/DBRX.md)
- [OpenELM](../concepts/OpenELM.md)
- [Phi-3](../concepts/Phi-3.md)
- [OLMo 2](../concepts/OLMo%202.md)
- [Falcon 3](../concepts/Falcon%203.md)
- [MiniCPM](../concepts/MiniCPM.md)
- [GLM](../concepts/GLM.md)
- [Kimi](../concepts/Kimi.md)
- [Kimi K3](../concepts/Kimi%20K3.md)
- [DeepSeek 系列](./DeepSeek%20系列.md)
- [DeepSeek-V3](../concepts/DeepSeek-V3.md)
- [DeepSeek-V4](../concepts/DeepSeek-V4.md)
- [Compressed Sparse Attention](../concepts/Compressed%20Sparse%20Attention.md)
- [Heavily Compressed Attention](../concepts/Heavily%20Compressed%20Attention.md)
- [Manifold-Constrained Hyper-Connections](../concepts/Manifold-Constrained%20Hyper-Connections.md)
- [Muon](../concepts/Muon.md)
- [Kimi Delta Attention](../concepts/Kimi%20Delta%20Attention.md)
- [Attention Residuals](../concepts/Attention%20Residuals.md)
- [Stable LatentMoE](../concepts/Stable%20LatentMoE.md)
- [Quantile Balancing](../concepts/Quantile%20Balancing.md)
- [MoonViT-V2](../concepts/MoonViT-V2.md)
- [MoonEP](../concepts/MoonEP.md)
- [MoE](../concepts/MoE.md)
- [Scaling 与 compute-optimal training](./Scaling%20与%20compute-optimal%20training.md)
- [开放模型家族与中国重要家族对照](../comparisons/开放模型家族与中国重要家族对照.md)
- [Qwen 系列演进](../timelines/Qwen%20系列演进.md)

## 未解决问题

- 数据质量、重复率与语言/代码混配怎样改变预算最优？家族数据不完全公开，参数和 token 数都不是全部原因。
- MoE 容量、通信与路由负载怎样共同决定训练和服务成本？总参数或激活参数不能单独预测硬件代价。
- 怎样分离基座、后训练、推理预算与工具环境的贡献？组合成绩需要共同协议和控制实验才能归因。

## 关联页面

- [Scaling 与 compute-optimal training](./Scaling%20与%20compute-optimal%20training.md)
- [指令对齐与 post-training](./指令对齐与%20post-training.md)
- [LLM RL](./LLM%20RL.md)
- [文本扩散语言模型](./%E6%96%87%E6%9C%AC%E6%89%A9%E6%95%A3%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)
- [DeepSeek 系列](./DeepSeek%20系列.md)
- [DeepSeek](../concepts/DeepSeek.md)
- [开放模型家族与中国重要家族对照](../comparisons/开放模型家族与中国重要家族对照.md)
- [Qwen 系列演进](../timelines/Qwen%20系列演进.md)
- [推理优化：量化、缓存与硬件](../comparisons/%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96%EF%BC%9A%E9%87%8F%E5%8C%96%E3%80%81%E7%BC%93%E5%AD%98%E4%B8%8E%E7%A1%AC%E4%BB%B6.md)：先定位瓶颈再选优化：量化减表示成本，剪枝改有效权重，缓存复用已有计算，调度提高资源利用率。它们可以配合，但速度收益不能简单相乘。

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。
- **sparse**：稀疏计算或连接：只使用选中的部分，具体省略什么取决于方法。
- **FLOPs**：浮点运算量：描述计算数量，不能直接等同于实际耗时。

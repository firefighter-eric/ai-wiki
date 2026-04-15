# AI 智能问答与智能客服

## 页面状态

- 状态：正式 topic
- 事实基座：`wiki/summaries/` 优先

## 主题定义

本页讨论与 AI 直接相关的智能问答、智能客服与 support assistant 问题：用户围绕产品、服务、订单、规则、故障、售后与常见问题发起咨询，系统需要在多轮对话中理解意图、检索知识、生成回复，并在必要时调用工具或触发人工转接。这里的重点不是通用闲聊，也不是呼叫中心排班、纯语音识别或情感计算，而是“问答系统、客服机器人与 LLM assistant 如何成为可控、可追溯、可落地的企业服务系统”。

与 [指令对齐与 post-training](./指令对齐与%20post-training.md) 相比，本页更强调客服场景中的系统分层与应用约束；与 [传统 NLP](./传统%20NLP.md) 相比，本页不只关心检索或分类本身，而是关心这些能力如何拼成客服链路；与 [LLM RL](./LLM%20RL.md) 相比，本页把偏好优化与安全看作客服系统中的行为控制层，而不是独立算法主题。

## 核心问题

- 高频标准问答、长尾知识问答与事务型客服请求，分别适合 FAQ 检索、RAG 生成，还是工具调用。
- 多轮智能客服对话中的历史轮次如何建模，哪些上下文应被保留，哪些应被忽略。
- 通用 LLM 如何通过 instruction tuning、偏好学习与安全护栏转成企业可用的智能问答 / 智能客服助手。
- 智能客服回复的正确性、可执行性、礼貌性、合规性与拒答边界应如何同时优化。
- 智能问答与智能客服系统应如何评测，才能避免只看离线 NLP 指标却忽视真实业务风险。

## 主线脉络 / 方法分层

- FAQ 检索起点：`Sakata et al. 2019` 把客服问题首先建模为 FAQ pair 检索，而不是自由生成。它同时考虑 `query-question similarity` 与 `query-answer relevance`，说明客服场景不能只比“用户问得像不像历史问题”，还要比“候选答案是否真能解决当前诉求”。这一路线对高频、标准化、答案边界清晰的问题最有效，优点是稳定、可控、易审计，缺点是对组合式、跨文档、需要状态更新的长尾问题覆盖有限。
- 多轮对话建模层：`Vlasov et al. 2019` 的 Dialogue Transformers 把对话状态表示建立在 turn-level self-attention 上，核心含义不是“把更强模型塞进客服”，而是指出客服对话历史并非都同样重要。真实客服里常见主题跳转、补充信息、纠错与插话，Transformer 对不同 turn 的选择性关注比简单 RNN 汇总更贴近客服对话结构。但该路线主要解决“如何编码对话状态”，并不自动解决事实依据与企业知识接入。
- 语义检索取代词匹配：`Karpukhin et al. 2020` 用 DPR 证明 dense retrieval 可以显著超过 BM25 式稀疏检索。对客服而言，这意味着知识库召回不应只依赖关键词重叠，因为用户往往用口语、错误术语或模糊表述提问。DPR 的贡献在于把“问法变化”从规则问题转成表征问题，为后续 RAG 型客服提供了知识入口层。
- 领域匹配检索强化：`Oğuz et al. 2021` 进一步说明，检索能力在客服里不是拿通用 embedding 就够了，领域匹配的预训练任务会明显影响效果。论文使用合成问题与对话式 post-comment 数据提升 IR 与 dialogue retrieval，背后的客服含义很直接：企业 FAQ、工单、聊天记录和知识文章本身就是关键训练资源，客服检索的瓶颈往往不只是模型结构，而是有没有把“本域问法”灌进去。
- 从检索到知识增强客服：`Wang - PIKE-RAG` 把工业场景下的 RAG 问题进一步拆成 specialized knowledge extraction、task decomposition 与 rationale construction，说明复杂客服已不只是“召回一段文档再复述”。当问题涉及规则组合、异常情况解释或多条件判断时，客服系统需要把知识拆解、重组并逐步形成可解释回答。这一方向更接近复杂企业 support，而不只是通用开放问答。
- 通用指令能力层：`Wei et al. 2021` 的 FLAN 说明 instruction tuning 能显著提升未见任务上的 zero-shot 泛化。对客服来说，这一层解决的是“模型能否理解工单改写、道歉、解释、澄清、总结、转接建议等多种指令样式”，从而摆脱完全基于固定 intent schema 的旧式客服机器人。
- 指令数据设计层：`Iyer et al. 2022` 的 OPT-IML 不只是重复 FLAN 结论，而是把 instruction tuning 的决策变量拆开分析，包括任务多样性、采样策略、是否加入 demonstrations、是否加入 specialized dialogue / reasoning 数据。它对客服的启发是：企业要做客服模型，不只是“多喂点指令数据”，还要关心客服专属数据是否被当作独立分布建模，否则模型容易在泛化上看似很强，在真实客服对话里却不稳。
- 对齐与服务风格层：`Ouyang et al. 2022` 的 InstructGPT 明确指出，大模型更大并不自动更 helpful、truthful、harmless。客服场景尤其依赖这一点，因为“回答得像人”不等于“回答得可交付”。该论文把 demonstrations、preference ranking 与 RLHF 连成完整管线，提供了把通用模型塑造成服务型交互代理的工程框架。
- 偏好优化简化层：`Rafailov et al. 2023` 的 DPO 说明，很多客服风格偏好其实可以通过偏好对直接优化，而不必总是走 reward model + PPO 的重管线。论文中它在 summarization 和 single-turn dialogue 上表现良好，对客服团队的现实意义是：如果企业已有“更好的客服回复 vs 更差的客服回复”这类排序数据，就能以较低工程复杂度做行为塑形。
- 工具使用层：`Schick et al. 2023` 的 Toolformer 表明模型可以学会何时调用外部 API。客服问题里，很多真正高价值场景不是回答知识，而是查订单、查物流、算赔付、查日程、创建工单。它提醒我们，生成式客服若没有工具层，常常只能停留在“解释型客服”；而要进入“处理型客服”，必须把工具调用纳入主线。
- 安全与护栏层：`Inan et al. 2023` 的 Llama Guard 将 Human-AI conversation 中的输入与输出安全审查建模成专门 safeguard 任务。对客服系统来说，这意味着安全不应完全寄托在主模型“自觉守规矩”上，而应有独立的门控层处理越权请求、敏感内容、违规建议与危险回复。
- 评测与治理层：`Liang et al. 2022` 的 HELM 说明语言模型评测不能只看单一准确率，而应把鲁棒性、公平性、毒性、校准与效率等维度同时纳入。客服系统尤其需要这种 holistic evaluation，因为真实业务里的失败往往不是“答错一道题”，而是“答得很像对、但在合规或行动层面出错”。

## 关键争论与分歧

- 客服应以检索为主还是生成为主：`Sakata 2019` 更接近“先找标准答案”，`Karpukhin 2020` 与 `PIKE-RAG` 则把客服推进到“检索后生成”。当前证据更支持把两者视为分层关系，而不是二选一替代。
- 多轮对话能力是否足以代表客服能力：`Vlasov 2019` 说明上下文编码很关键，但客服失败常常并不发生在上下文建模，而是发生在知识缺失、政策执行错误或工具无法调用。
- 指令微调是否足够：`Wei 2021` 与 `Iyer 2022` 说明 instruction tuning 能显著提升客服泛化，但 `Ouyang 2022` 与 `Rafailov 2023` 表明，若要稳定地符合企业偏好与服务风格，偏好优化仍然常常必要。
- 安全应由主模型内化还是外部护栏承担：`Ouyang 2022` 倾向于通过对齐改善主模型行为，`Inan 2023` 则说明独立 safeguard 模型在系统层仍有不可替代性。
- 评测应看离线基准还是业务指标：`Liang 2022` 支持多维评测，但客服落地还需要把首次解决率、升级率、误拒率、错误执行率与人工接管成本纳入评估；这部分在当前知识库里仍缺少更直接的 summary 支撑。

## 证据基础

- [Sakata et al. - 2019 - FAQ retrieval using query-question similarity and BERT-based query-answer relevance](../../wiki/summaries/Sakata%20et%20al.%20-%202019%20-%20FAQ%20retrieval%20using%20query-question%20similarity%20and%20BERT-based%20query-answer%20relevance.md)：支撑“高频标准客服问题首先可被建模为 FAQ 检索”的起点。
- [Vlasov, Mosig, Nichol - 2019 - Dialogue Transformers](../../wiki/summaries/Vlasov,%20Mosig,%20Nichol%20-%202019%20-%20Dialogue%20Transformers.md)：支撑“客服需要多轮上下文选择性编码，而不是简单串接历史”。
- [Karpukhin et al. - 2020 - Dense passage retrieval for open-domain question answering](../../wiki/summaries/Karpukhin%20et%20al.%20-%202020%20-%20Dense%20passage%20retrieval%20for%20open-domain%20question%20answering.md)：支撑“客服知识召回从词匹配转向语义检索”的关键方法节点。
- [Oğuz et al. - 2021 - Domain-matched Pre-training Tasks for Dense Retrieval](../../wiki/summaries/O%C4%9Fuz%20et%20al.%20-%202021%20-%20Domain-matched%20Pre-training%20Tasks%20for%20Dense%20Retrieval.md)：支撑“客服检索效果高度依赖领域匹配数据与预训练任务”。
- [Wei et al. - 2021 - Finetuned Language Models Are Zero-Shot Learners](../../wiki/summaries/Wei%20et%20al.%20-%202021%20-%20Finetuned%20Language%20Models%20Are%20Zero-Shot%20Learners.md)：支撑“指令微调可把通用模型拉向客服式任务接口”。
- [Iyer et al. - 2022 - OPT-IML Scaling Language Model Instruction Meta Learning through the Lens of Generalization](../../wiki/summaries/Iyer%20et%20al.%20-%202022%20-%20OPT-IML%20Scaling%20Language%20Model%20Instruction%20Meta%20Learning%20through%20the%20Lens%20of%20Generalization.md)：支撑“客服型 instruction tuning 的关键在任务规模、分布与 specialized dialogue 数据设计”。
- [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](../../wiki/summaries/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)：支撑“客服助手需要 helpful / truthful / harmless 风格对齐”的核心框架。
- [Liang et al. - 2022 - Holistic Evaluation of Language Models](../../wiki/summaries/Liang%20et%20al.%20-%202022%20-%20Holistic%20Evaluation%20of%20Language%20Models.md)：支撑“客服评测不能只看单一任务准确率”。
- [Rafailov, Mitchell, Jul - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model](../../wiki/summaries/Rafailov,%20Mitchell,%20Jul%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)：支撑“客服风格偏好可通过更轻量的偏好优化实现”。
- [Schick et al. - 2023 - Toolformer Language Models Can Teach Themselves to Use Tools](../../wiki/summaries/Schick%20et%20al.%20-%202023%20-%20Toolformer%20Language%20Models%20Can%20Teach%20Themselves%20to%20Use%20Tools.md)：支撑“客服从回答型代理走向处理型代理需要工具调用层”。
- [Inan et al. - 2023 - Llama Guard LLM-based Input-Output Safeguard for Human-AI Conversations](../../wiki/summaries/Inan%20et%20al.%20-%202023%20-%20Llama%20Guard%20LLM-based%20Input-Output%20Safeguard%20for%20Human-AI%20Conversations.md)：支撑“客服系统需要独立的输入输出安全门控”。
- [Wang - Unknown - PIKE-RAG sPecIalized KnowledgE and Rationale Augmented Generation](../../wiki/summaries/Wang%20-%20Unknown%20-%20PIKE-RAG%20sPecIalized%20KnowledgE%20and%20Rationale%20Augmented%20Generation.md)：支撑“工业客服中的复杂知识应用与分解式 RAG”。

## 代表页面

- [Dense Retrieval](../concepts/Dense%20Retrieval.md)
- [DPR](../concepts/DPR.md)
- [Instruction Tuning](../concepts/Instruction%20Tuning.md)
- [InstructGPT](../concepts/InstructGPT.md)
- [RLHF](../concepts/RLHF.md)
- [DPO](../concepts/DPO.md)
- [Llama Guard](../concepts/Llama%20Guard.md)
- [指令对齐与 post-training](./指令对齐与%20post-training.md)
- [LLM RL](./LLM%20RL.md)

## 未解决问题

- 当前知识库还缺少直接围绕工单分流、人工转接决策、客服行动执行与 CRM 集成的 summary，因此“客服代理如何可靠闭环”仍未形成稳定结论。
- 多轮记忆与账户级个性化如何与知识检索、安全审查共同工作，当前证据仍偏方法碎片，缺少一条完整系统线。
- 客服中的拒答、澄清、追问与升级到人工之间应如何做最优策略切换，现有页面能说明必要性，但不足以支撑成熟方法学。
- 业务评测仍是明显空白：首次解决率、重复来访率、误操作率、投诉风险与人工节省之间的关系，当前尚无足够 `wiki/summaries/` 可作为结论基座。

## 关联页面

- [传统 NLP](./传统%20NLP.md)
- [LLM RL](./LLM%20RL.md)
- [指令对齐与 post-training](./指令对齐与%20post-training.md)
- [Dense Retrieval](../concepts/Dense%20Retrieval.md)
- [Instruction Tuning](../concepts/Instruction%20Tuning.md)
- [RLHF](../concepts/RLHF.md)
- [Llama Guard](../concepts/Llama%20Guard.md)

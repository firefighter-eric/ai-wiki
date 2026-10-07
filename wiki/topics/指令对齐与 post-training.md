---
type: topic
status: formal
review_scope: evidence_synthesis
reviewed: 2026-10-07
---
# 指令对齐与 post-training

## TL;DR（快速导读）

后训练分别塑造任务接口、输出分布和偏好行为。示范、偏好对、可验证奖励与 teacher 分布提供不同信息；改善人工偏好不表示事实、安全和执行能力都同时得到保证。

阅读重点：先按问题选择路线，再核对比较条件与证据边界。

## 先用一个问题理解

对“请用三句话解释”的回答，可分别检查内容正确、句数遵守和表达清楚。训练示范、成对偏好与在线奖励会强调不同方面；阅读时先确认目标，再比较数据和算法。

## 页面状态

正式 topic；2026-10-07 复核核心来源并补充方法比较。正文区分论文实验、作者报告和本文综合判断；开放问题表示研究证据的边界。

## 主题定义

本页讨论 **预训练之后，模型如何被重写为更可用的交互系统**。它的中心不是“模型还会不会继续学知识”，而是“已有能力如何被组织成更能遵循指令、响应偏好、保持安全边界并适应产品接口的行为形态”。这里的 `post-training` 主要覆盖 instruction tuning、监督示范、偏好数据、偏好优化与 `RLHF` 总框架，但 **不把 reasoning-oriented RL 的细节作为本页主角**；那部分应主要回收到 [LLM RL](./LLM%20RL.md)。

本页与 `LLM 预训练` 的边界在于：预训练解释 **能力底座的形成**，而 post-training 解释 **行为接口的塑形**。与 `LLM RL` 的边界在于：本页讨论从 instruction following 到 preference alignment 的总流程与方法结构；`LLM RL` 则专门处理奖励驱动、策略优化与 reasoning RL 的细部问题。

当前证据支持一个已经相当稳定的判断：**post-training 不是预训练的附属补丁，而是把 base model 转换为可交互系统的独立阶段。** 这也是为什么 `InstructGPT` 在知识史上的地位，不只是又一个微调技巧，而是明确建立了“预训练能力”与“面向用户的行为质量”之间的阶段性分工。

## 核心问题

- 为什么强预训练模型仍可能 **不 helpful、不 truthful、不 harmless**，以及这些缺陷为何不能仅靠扩大预训练规模自动消失。
- instruction tuning、监督示范、偏好学习与 `RLHF` 在后训练管线中分别解决什么问题。
- 行为改善为何能够在 **参数规模不占优** 的情况下显著提升产品体验，这种收益与基础能力提升应如何区分。
- `DPO` 一类方法到底是在替代 `RLHF`，还是在改写其实现方式。
- 当 post-training 从“让模型更会回答”延伸到“让模型更会推理、更会使用工具”时，主题边界应如何收束。

## 主线脉络 / 方法分层

本页按 **后训练解决的问题类型** 分层，而不是按“哪篇论文先发表”来写。因为 instruction tuning、偏好对齐与 RLHF 真正的差异，在于它们改变的是不同层级的行为接口。

- **指令接口建立层**：`Wei et al. 2021` 的 `Finetuned Language Models Are Zero-Shot Learners` 之所以关键，不只是因为它提升了 zero-shot 表现，而是因为它说明：**把任务表达统一成自然语言指令，本身就是一种可泛化的接口设计。** 在这一层，模型开始从“会续写文本”过渡到“能把指令当作任务约束来执行”。
- **监督示范塑形层**：在后训练管线中，监督示范的作用并不是提供全部知识，而是把模型拉入更接近用户预期的输出分布。它通常解决的是语气、结构、任务完成格式与初始行为稳定性问题。即便许多来源没有单独把这一层展开为独立论文，它仍然是理解 `InstructGPT` 管线不可省略的中间层。
- **偏好建模与 `RLHF` 层**：`Ouyang et al. 2022` 给出的关键不是“又做了一次微调”，而是证明了 demonstrations、preference rankings、reward model 与 RL 可以被组织成一个统一的对齐框架。在这里，后训练的目标从“预测下一个 token”转向 **优化更接近用户价值判断的行为分布**。也正是在这一层，post-training 被明确写成一个独立于预训练的产品化阶段。
- **偏好优化简化层**：`Rafailov et al. 2023` 的 `DPO` 指出，偏好对齐不一定必须经过显式 reward model + PPO 这一完整管线。它把 post-training 方法族进一步分化为“完整 RLHF 管线”与“更直接的 preference optimization”。因此，本页更适合把 `DPO` 理解为 **post-training 的方法内部分化**，而不是将其简单记作“RLHF 的替代者”。
- **reasoning-oriented 后训练外溢层**：`DeepSeek-R1` 说明后训练目标已经开始从经典 alignment 外溢到推理行为塑形。但在本页中，这一层只作为边界说明出现：它表明 post-training 不再只关心“更听话”，也开始关心“更会做题、更会长链推理”；其更细的优化与奖励问题仍应下沉到 `LLM RL`。

从知识组织上看，本页最稳定的分层是：**instruction interface 建立**、**监督行为塑形**、**偏好建模与 RLHF**、**偏好优化分化**。这样写能保持 post-training 的主题边界，而不会让页面被 reasoning RL 全面接管。

### 一条后训练管线如何被拆成可判断的阶段

[FLAN](../summaries/Wei%20et%20al.%20-%202021%20-%20Finetuned%20Language%20Models%20Are%20Zero-Shot%20Learners.md)以自然语言指令连接多任务，[OPT-IML](../summaries/Iyer%20et%20al.%20-%202022%20-%20OPT-IML%20Scaling%20Language%20Model%20Instruction%20Meta%20Learning%20through%20the%20Lens%20of%20Generalization.md)进一步强调训练/测试任务分布与采样。它们说明任务表述可迁移，但没有证明任何新指令都被正确理解。[InstructGPT](../summaries/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)把 demonstration、偏好排序、reward model 和 PPO 组合起来；1.3B 模型被评审偏好于更大的原始 GPT-3，是该提示分布和评审规则下的行为结果，不是全知识与推理能力反超。

| 阶段 | 信号提供了什么 | 仍需另行验证什么 |
| --- | --- | --- |
| 监督示范 | 在给定输入上如何回答 | 未见任务、错误示范与分布外泛化 |
| 偏好排序 | 多个回答中评审更喜欢哪个 | 偏好是否与事实、安全一致 |
| 在线奖励优化 | 当前策略输出获得何种分数 | 奖励漏洞、长度偏置、探索成本 |
| 蒸馏 | teacher 在特定轨迹上如何分配概率 | teacher 错误、可用概率接口与迁移 |
| 运行时护栏 | 输入输出是否触发策略类别 | 账户授权、工具状态与漏拦 |

[DPO](../summaries/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)让离线偏好对直接更新策略，省略显式 reward model/PPO 管线，但保留 reference 和数据假设。[ORPO](../summaries/Hong%20et%20al.%20-%202024%20-%20ORPO%20Monolithic%20Preference%20Optimization%20without%20Reference%20Model.md)将监督与偏好合并，[KTO](../summaries/Ethayarajh%20et%20al.%20-%202024%20-%20KTO%20Model%20Alignment%20as%20Prospect%20Theoretic%20Optimization.md)调整标注接口。简化训练系统不意味着降低全部数据成本，也不保证在新的策略分布上更稳健。

[DFT](../summaries/Wu%20et%20al.%20-%202025%20-%20On%20the%20Generalization%20of%20SFT%20A%20Reinforcement%20Learning%20Perspective%20with%20Reward%20Rectification.md)提醒：只在 SFT 内改变梯度权重，也可能改变数学泛化，后训练不能全部叫 RL。[Hermes 3](../summaries/Teknium%2C%20Quesnelle%2C%20Guang%20-%20Unknown%20-%20arXiv%202408%20.%2011857v1%20cs%20.%20CL%2015%20Aug%202024.md)是一条实际模型训练与行为目标的来源；它的报告能说明开放权重如何适配，不能与不同评审体系直接拼出最优配方。[Llama Guard](../summaries/Inan%20et%20al.%20-%202023%20-%20Llama%20Guard%20LLM-based%20Input-Output%20Safeguard%20for%20Human-AI%20Conversations.md)属于运行时内容判别，与训练内化互补，但分类器的存在本身不证明系统已安全。

本文据此将后训练结果分别解释为任务遵循、偏好一致、验证正确和过程可监控。对齐可能改变拒答、风格和回答长度，这些变化会影响自动 judge 与人工偏好；横比时应检查相同 prompt、评审群体和长度/采样设置。

## 关键争论与分歧

- **instruction tuning 与 `RLHF` 的关系是什么**：当前证据更支持把 instruction tuning 视为后训练的前置层或相邻层，而不是 `RLHF` 的完整替代。只有在区分“让模型理解指令接口”与“让模型按人类偏好优化行为”之后，这一争论才有意义。
- **对齐收益来自哪一层**：`Ouyang 2022` 的经典结果常被概括为“小模型经过后训练可优于更大但未对齐的模型”。这一结论成立，但其适用边界是 **行为质量与交互可用性**，而不是“基础知识和推理能力已完全可被后训练替代”。因此不能把该结论过度外推为“预训练规模不再重要”。
- **`DPO` 是否会取代 `RLHF`**：现有证据更支持“`DPO` 是重要分化方向”而不是“全面替代”。这一争论只有在区分 **工程复杂度** 与 **概念目标** 后才站得住。`DPO` 改变了实现路径，但并没有让“偏好决定后训练目标”这个中心前提消失。
- **reasoning RL 是否仍属于 alignment**：`DeepSeek-R1` 使这一边界开始模糊。当前更稳妥的做法不是强行划一，而是承认：后训练正在从“helpfulness / harmlessness”扩展到“问题求解行为塑形”，但这并不意味着传统 alignment 议题已经失效。
- **post-training 是否只是产品层技巧**：当前证据并不支持这种降格理解。无论是 `InstructGPT` 的阶段性影响，还是后续偏好优化方法族的扩展，都说明 post-training 已经是现代 LLM 系统设计中的核心组成部分，而非上线前的小修补。

### “更受偏好”不能改写成“已保证正确”

人类偏好可以包含语气、完整度和价值选择，也可能奖励自信但不准确的回答。真实性、无害性与帮助性之间仍有冲突；某一维的改进不能当作另一维的实验证据。本文支持把后训练设为独立分析阶段，但不将预训练与后训练的知识贡献强行隔离，也不声称一种偏好损失已经覆盖所有服务目标。

## 证据基础

- [Wei et al. - 2021 - Finetuned Language Models Are Zero-Shot Learners](../../wiki/summaries/Wei%20et%20al.%20-%202021%20-%20Finetuned%20Language%20Models%20Are%20Zero-Shot%20Learners.md)
- [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](../../wiki/summaries/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)
- [Rafailov et al. - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model](../../wiki/summaries/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)
- [DeepSeek-R1：奖励驱动推理与多阶段训练（2025）](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md)
- [Iyer et al. - 2022 - OPT-IML Scaling Language Model Instruction Meta Learning through the Lens of Generalization](../summaries/Iyer%20et%20al.%20-%202022%20-%20OPT-IML%20Scaling%20Language%20Model%20Instruction%20Meta%20Learning%20through%20the%20Lens%20of%20Generalization.md)：指令泛化受任务分布和采样影响。
- [Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model](../summaries/Hong%20et%20al.%20-%202024%20-%20ORPO%20Monolithic%20Preference%20Optimization%20without%20Reference%20Model.md)：监督与偏好合并的方案。
- [Ethayarajh et al. - 2024 - KTO Model Alignment as Prospect Theoretic Optimization](../summaries/Ethayarajh%20et%20al.%20-%202024%20-%20KTO%20Model%20Alignment%20as%20Prospect%20Theoretic%20Optimization.md)：单样本偏好标签及参考依赖。
- [Wu et al. - 2025 - On the Generalization of SFT A Reinforcement Learning Perspective with Reward Rectification](../summaries/Wu%20et%20al.%20-%202025%20-%20On%20the%20Generalization%20of%20SFT%20A%20Reinforcement%20Learning%20Perspective%20with%20Reward%20Rectification.md)：数学任务中的 SFT 梯度修正。
- [Hermes 3：开放模型的指令与行为训练（2024）](../summaries/Teknium%2C%20Quesnelle%2C%20Guang%20-%20Unknown%20-%20arXiv%202408%20.%2011857v1%20cs%20.%20CL%2015%20Aug%202024.md)：Hermes 3 的行为训练报告。
- [Inan et al. - 2023 - Llama Guard LLM-based Input-Output Safeguard for Human-AI Conversations](../summaries/Inan%20et%20al.%20-%202023%20-%20Llama%20Guard%20LLM-based%20Input-Output%20Safeguard%20for%20Human-AI%20Conversations.md)：运行时内容风险分类的边界。

## 代表页面

- [FLAN](../concepts/FLAN.md)
- [Instruction Tuning](../concepts/Instruction%20Tuning.md)
- [InstructGPT](../concepts/InstructGPT.md)
- [OPT-IML](../concepts/OPT-IML.md)
- [LoRA](../concepts/LoRA.md)
- [RLHF](../concepts/RLHF.md)
- [DPO](../concepts/DPO.md)
- [DeepSeek-R1](../concepts/DeepSeek-R1.md)

## 未解决问题

- 评审偏好何时与事实、安全及任务完成冲突？风格、长度和自信会影响 judge，真实性仍需独立评价。
- 离线偏好优化怎样覆盖变化后的策略分布？接口简化没有自动解决探索与奖励偏差。
- 示范、奖励和蒸馏各提供哪些迁移信息？DFT、R1 与 OPD 目标不同，不能用一个算法统一全部后训练。

## 关联页面

- [LLM 预训练](./LLM%20预训练.md)
- [LLM RL](./LLM%20RL.md)
- [GPT-3](../concepts/GPT-3.md)
- [RLHF](../concepts/RLHF.md)
- [DPO](../concepts/DPO.md)
- [Prompt Tuning](../concepts/Prompt%20Tuning.md)
- [AI 能力评测：任务、过程与预测](../comparisons/AI%20%E8%83%BD%E5%8A%9B%E8%AF%84%E6%B5%8B%EF%BC%9A%E4%BB%BB%E5%8A%A1%E3%80%81%E8%BF%87%E7%A8%8B%E4%B8%8E%E9%A2%84%E6%B5%8B.md)：基准成绩、探索案例、过程监控、社会趋势与未来预测是不同证据。先判断材料属于哪一种，再看数据、评价和外推条件；多篇材料不能自动拼成 AGI 已实现的证明。

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **reward model**：奖励模型：根据训练信号给回答或行为打分，分数是目标的近似。
- **RLHF**：人类反馈强化学习：把人类偏好转成奖励，并用它调整模型行为。
- **DPO**：直接偏好优化：用成对偏好数据直接调整模型概率，简化部分奖励训练流程。
- **post-training**：后训练：在预训练底座上继续调整指令遵循、偏好或其他行为。

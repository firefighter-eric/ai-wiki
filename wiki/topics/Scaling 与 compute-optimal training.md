---
type: topic
status: formal
review_scope: evidence_synthesis
reviewed: 2026-10-07
---
# Scaling 与 compute-optimal training

## TL;DR（快速导读）

Chinchilla 研究固定训练 FLOPs 下怎样分配参数和 token；部署还要计算长期推理成本。幂律是实验区间内的拟合，不能当作所有模型、数据和任务通用的预算公式。

阅读重点：先按问题选择路线，再核对比较条件与证据边界。

## 先用一个问题理解

同一笔预算可训练大模型较少步，或小模型更多步。Chinchilla 主要研究训练配置；上线时若有大量请求，还要计入持续推理成本。比较结论时应先确认优化的是哪一笔成本。

## 页面状态

正式 topic；2026-10-07 复核核心来源并补充方法比较。正文区分论文实验、作者报告和本文综合判断；开放问题表示研究证据的边界。

## 主题定义

本页聚焦 **LLM 训练中的规模化规律**，尤其是两个紧密相连但不能混写的问题：第一，为什么能力会随着模型、数据与训练系统规模扩大而提升；第二，在 **固定计算预算** 下，参数量与 token 数应如何配置才更接近最优。也就是说，本页讨论的是 **训练规律**，而不是家族盘点，也不是完整的预训练总论。

与 [LLM 预训练](./LLM%20预训练.md) 相比，本页更窄，专门处理“规模为何有效”和“预算如何最优”这两个理论与工程中间层问题。与 `MoE` 或某些具体模型页相比，本页也更抽象；只有当具体模型能为规模化规律提供稳定证据时，才进入讨论。

当前知识库中，这一 topic 的最稳固骨架是：`GPT-3` 证明大规模 dense 自回归预训练会带来明显 few-shot 与跨任务能力；`PaLM` 说明这种收益在更大训练系统中仍持续存在；`Chinchilla` 则把“scaling”从粗糙的“继续变大”修正为 **在既定 FLOPs 下平衡参数量与训练 token 数**。因此，本页真正要解释的不是单篇论文，而是 **dense scaling 与 compute-optimal 修正之间的连续关系**。

## 核心问题

- 为什么扩大规模会提升 few-shot、跨任务泛化与通用语言能力。
- 在固定训练计算预算下，**参数量** 与 **训练 token 数** 的平衡应如何理解。
- dense scaling 的成功与 `Chinchilla` 的 under-trained 修正之间究竟是冲突关系，还是连续纠偏关系。
- 训练阶段的 compute-optimal 结论，是否能够直接外推到部署阶段的成本最优与产品最优。
- sparse/MoE 路线是否会改写当前基于 dense 模型建立的 compute-optimal 讨论。

## 主线脉络 / 方法分层

本页按 **问题演进逻辑** 分层。也就是说，不是“哪篇论文更有名”，而是“它回答了规模化问题中的哪一层”。

- **规模化有效性层**：`Brown et al. 2020` 给出的核心贡献，是让“随着模型规模扩大，few-shot 能力显著增强”成为一个可被广泛接受的事实陈述。这里最重要的不是某个单项 benchmark，而是 prompt 被证明能成为任务接口，从而让大模型具备跨任务迁移的统一表达形式。
- **系统级规模扩展层**：`PaLM` 进一步把这种规模化收益推向更大训练系统，并把多语言、代码与推理能力纳入同一 scaling 语境。它支撑的不是一个全新理论，而是一个关键经验判断：**dense Transformer 的收益并未在 GPT-3 后立即触顶。**
- **compute-optimal 修正层**：`Hoffmann et al. 2022` 是这一 topic 的真正分水岭。它表明许多模型在固定 FLOPs 下并不是“太小”，而是“训练不够久、token 不够多”。因此 compute-optimal training 的核心不是否定大模型，而是把规模化问题重写为 **预算分配问题**：在总算力既定时，应如何在参数量与训练数据上取得更优平衡。
- **路线解释层**：从现有 summary 出发，更准确的结论不是“`GPT-3` 被 `Chinchilla` 推翻”，而是：**早期 dense scaling 证明了规模有效，`Chinchilla` 修正了如何更有效地使用规模。** 这一区分很重要，因为它决定了本页应把 `Chinchilla` 写成“纠偏”，而不是“反例”。
- **外推边界层**：compute-optimal 是训练阶段命题，但产品系统关心的是推理延迟、显存占用、服务成本与吞吐。当前证据提醒我们，**训练最优并不天然等于部署最优**。这也是为什么本页必须单独保留“外推边界”这一层，而不把训练规律直接写成系统结论。
- **sparse/MoE 潜在改写层**：现有 topic 还缺少直接以 MoE 重写 compute-optimal 的强证据，但 `Mixtral`、`DBRX`、`DeepSeek-V3` 所代表的 sparse 路线已足以提出一个结构性问题：当总参数与单 token 激活参数脱钩后，原本建立在 dense 假设上的最优配置结论，在多大程度上仍然成立。当前还不能下定论，但这个问题已经构成该 topic 的自然延伸。

### 固定预算到底固定了什么

[Chinchilla](../summaries/Hoffmann%20et%20al.%20-%202022%20-%20Training%20Compute-Optimal%20Large%20Language%20Models.md)使用三种估计：固定模型扫描训练量、比较 IsoFLOP 曲线、拟合参数化损失。论文在约 $C=6ND$ 的 dense Transformer 训练约束下，拟合得到参数 $N$ 和训练 token $D$ 的增长指数约 0.46 与 0.54。读作“二者近似共同增长”比读作永远固定比例更准确；公式的常数、数据分布和拟合区间都参与结果。

70B/1.4T 的 Chinchilla 与 Gopher 在相近训练 FLOPs 下比较，支持原训练配置可以重新分配预算。它并没有证明所有任务都最优，也没有把产品生命周期的推理计算算进去。若上线请求很多，小模型多训练一些可能省后续服务成本；这是由成本项推导出的设计假设，本文来源没有测量特定业务的盈亏点。

| 问题 | 最优化对象 | 证据能外推到哪里 |
| --- | --- | --- |
| Chinchilla 配置 | 固定训练 FLOPs 的语言建模损失 | 相近 dense 目标和数据条件 |
| 数学推理 scaling | 预训练损失、SFT/增强数据与数学成绩 | 对应任务与解码协议 |
| ViT 规模与形状 | 参数、宽深比、数据和视觉任务 | 视觉分类/迁移，非语言 token 公式 |
| 产品预算 | 训练、内存、服务延迟与请求量 | 需另建部署成本模型 |

[数学推理 scaling](../summaries/Yuan%20et%20al.%20-%202023%20-%20Scaling%20Relationship%20on%20Learning%20Mathematical%20Reasoning%20with%20Large%20Language%20Models.md)发现预训练损失比仅用参数规模更能解释特定数学成绩；这不表示损失已经充分解释所有推理能力。[ViT scaling](../summaries/Zhai%20et%20al.%20-%202022%20-%20Scaling%20Vision%20Transformers.md)和[SoViT](../summaries/Alabdulmohsin%20et%20al.%20-%202023%20-%20Getting%20ViT%20in%20Shape%20Scaling%20Laws%20for%20Compute-Optimal%20Model%20Design.md)进一步展示“扩大规模”还包含模型形状选择。SoViT 在稠密分割存在局限，说明一个分类上较优的形状不能无条件迁移为全部视觉任务最优。

这些来源共同支持的抽象是：先定义预算、目标和评价分布，再拟合投入与结果的关系。本文据此拒绝把参数规模、训练 token、解码采样数量与 agent 工具预算合并成一条无需测量的通用曲线。

## 关键争论与分歧

- **是否还应使用一般性的 scaling law 叙述**：当前知识库更适合聚焦 `compute-optimal training`，而不是泛泛而谈“模型越大越强”。这一争论真正成立的前提是：必须承认 scaling 既是经验规律，也是预算配置问题，而非单一口号。
- **模型更大还是数据更多更关键**：`Hoffmann 2022` 的结论经常被误读为“数据比参数更重要”。更准确的说法是：**在固定计算预算下，只增参数而不相应增加训练 token 会导致 under-trained。** 因此这不是“参数 vs 数据”的简单二选一，而是联合配置问题。
- **训练最优是否等于产品最优**：当前 summary 并未提供足够证据把 FLOPs 最优直接转写为服务成本最优。只要推理阶段仍受显存、延迟、吞吐与硬件友好性约束，训练结论就不能无条件外推到部署层。
- **dense scaling 是否已被 sparse 路线改写**：目前还不能这么写。dense scaling 仍是理解能力增长与 compute-optimal 讨论的基础语言；MoE 更像是在工程实现上引入新的效率维度。只有在有更多直接对照 summary 后，才能更强地讨论“dense law 是否需要重写”。
- **`Chinchilla` 是否否定了早期大模型叙事**：现有证据不支持这种断裂式写法。更稳妥的说法是，`Chinchilla` 让 scaling 从“继续做大”变成“更精确地配置预算做大”，它修正的是策略，不是抹去 dense scaling 的事实基础。

### 共识与开放问题的边界

在原研究区间，参数和数据共同扩展、预算分配影响损失，是较稳定的经验结论；新的数据质量、MoE、不同上下文和推理计算如何改变系数，仍需拟合与控制实验。GPT-3 的 few-shot 趋势是多个任务上的观察，不能解释成每个任务随参数严格单调提升，更不能当成已找到能力增长的全部因果机制。

## 证据基础

- [Brown et al. - 2020 - Language models are few-shot learners](../../wiki/summaries/Brown%20et%20al.%20-%202020%20-%20Language%20models%20are%20few-shot%20learners.md)
- [Chowdhery et al. - 2022 - PaLM Scaling Language Modeling with Pathways](../../wiki/summaries/Chowdhery%20et%20al.%20-%202022%20-%20PaLM%20Scaling%20Language%20Modeling%20with%20Pathways.md)
- [Hoffmann et al. - 2022 - Training Compute-Optimal Large Language Models](../../wiki/summaries/Hoffmann%20et%20al.%20-%202022%20-%20Training%20Compute-Optimal%20Large%20Language%20Models.md)
- [Yuan et al. - 2023 - Scaling Relationship on Learning Mathematical Reasoning with Large Language Models](../summaries/Yuan%20et%20al.%20-%202023%20-%20Scaling%20Relationship%20on%20Learning%20Mathematical%20Reasoning%20with%20Large%20Language%20Models.md)：区分数学任务、预训练损失和解码口径。
- [Zhai et al. - 2022 - Scaling Vision Transformers](../summaries/Zhai%20et%20al.%20-%202022%20-%20Scaling%20Vision%20Transformers.md)：补充视觉模型规模与数据条件。
- [Alabdulmohsin et al. - 2023 - Getting ViT in Shape Scaling Laws for Compute-Optimal Model Design](../summaries/Alabdulmohsin%20et%20al.%20-%202023%20-%20Getting%20ViT%20in%20Shape%20Scaling%20Laws%20for%20Compute-Optimal%20Model%20Design.md)：补充固定预算下的深宽形状与迁移局限。

## 代表页面

- [GPT-3](../concepts/GPT-3.md)
- [PaLM](../concepts/PaLM.md)
- [Chinchilla](../concepts/Chinchilla.md)
- [LLM 预训练](../topics/LLM%20预训练.md)

## 未解决问题

- 质量、去重和重复训练如何改变幂律系数？Chinchilla 拟合依研究分布，不能脱离数据照搬常数。
- MoE、模型形状与长上下文怎样改变 dense 成本近似？视觉形状和数学 scaling 显示任务也会改变最优配置。
- 训练最优与生命周期服务最优何时分离？请求量、硬件和质量阈值不同，训练 FLOPs 不能给统一产品规模。

## 关联页面

- [LLM 预训练](../topics/LLM%20预训练.md)
- [Chinchilla](../concepts/Chinchilla.md)
- [MoE](../concepts/MoE.md)
- [AI 能力评测：任务、过程与预测](../comparisons/AI%20%E8%83%BD%E5%8A%9B%E8%AF%84%E6%B5%8B%EF%BC%9A%E4%BB%BB%E5%8A%A1%E3%80%81%E8%BF%87%E7%A8%8B%E4%B8%8E%E9%A2%84%E6%B5%8B.md)：基准成绩、探索案例、过程监控、社会趋势与未来预测是不同证据。先判断材料属于哪一种，再看数据、评价和外推条件；多篇材料不能自动拼成 AGI 已实现的证明。

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **benchmark**：评测基准：特定数据、任务与规则的组合，分数只在这些条件下成立。
- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。
- **sparse**：稀疏计算或连接：只使用选中的部分，具体省略什么取决于方法。
- **FLOPs**：浮点运算量：描述计算数量，不能直接等同于实际耗时。

# RLHF vs DPO vs ORPO vs KTO

## 比较目标

本页比较四条常见的 LLM 偏好对齐路线：经典 `RLHF` 管线，以及三条简化 / 改写后的偏好优化方法 `DPO`、`ORPO`、`KTO`。重点不是列公式，而是澄清它们在数据接口、优化对象、工程复杂度与适用边界上的真实差异。

## 核心结论

- `RLHF` 是完整管线：示范数据、偏好数据、reward model 与在线 RL 组成闭环，灵活但工程最重。
- `DPO` 用闭式目标折叠了 reward model + PPO 的一部分角色，更适合已有成对偏好数据、希望降低训练复杂度的场景。
- `ORPO` 把监督微调与偏好对齐更紧地合并，进一步压缩流程，但对数据分布和训练配方更敏感。
- `KTO` 改写了偏好接口，允许只提供 unary desirable / undesirable 信号，在弱标注或便宜反馈场景下更实用。

## 对照维度

### 1. 数据接口

- `RLHF`：通常需要 demonstrations、pairwise preference 与额外 rollout 数据。
- `DPO`：依赖 pairwise preference。
- `ORPO`：仍建立在偏好样本之上，但训练时更偏向单阶段整合。
- `KTO`：只要求 unary desirable / undesirable 标签，不强制成对比较。

### 2. 优化方式

- `RLHF`：显式 reward model，再用在线 RL 更新策略。
- `DPO`：直接优化 reference-based preference objective，不显式单独训练 reward model。
- `ORPO`：把 SFT 与偏好优化并入单阶段目标，弱化 reference model 依赖。
- `KTO`：从前景理论视角定义损失，对“好 / 坏”反馈做非对称建模。

### 3. 工程复杂度

- `RLHF` 最高：训练链路长，调参点多，rollout 与奖励建模成本大。
- `DPO` 中等：显著轻于 `RLHF`，但仍需要高质量成对偏好数据。
- `ORPO` 中低：流程更短，但 recipe 稳定性要求更高。
- `KTO` 中低：标注接口更轻，但其效果高度依赖 unary 反馈质量。

### 4. 适用边界

- `RLHF`：适合需要在线探索、显式奖励塑形、复杂行为约束的场景。
- `DPO`：适合偏好数据较成熟、希望替代重型 RL pipeline 的场景。
- `ORPO`：适合想把 instruction tuning 与 preference alignment 更紧密合并的场景。
- `KTO`：适合拿不到稳定 pairwise preference、但能拿到粗粒度好坏反馈的场景。

## 关键分歧

- `DPO / ORPO / KTO` 不是“完全不要 RL”，更准确地说，它们是在不同前提下把 RLHF 的一部分结构离线化、折叠化或弱化数据接口。
- 偏好优化是否足以取代经典 `RLHF`，取决于是否需要在线探索、显式奖励建模与复杂 rollout。
- 当目标从“更符合偏好”转向“更强 reasoning 行为”时，很多系统仍会回到更强的在线 RL 设定。

## 证据基础

- [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](../summaries/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)
- [Rafailov et al. - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model](../summaries/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)
- [Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model](../summaries/Hong%20et%20al.%20-%202024%20-%20ORPO%20Monolithic%20Preference%20Optimization%20without%20Reference%20Model.md)
- [Ethayarajh et al. - 2024 - KTO Model Alignment as Prospect Theoretic Optimization](../summaries/Ethayarajh%20et%20al.%20-%202024%20-%20KTO%20Model%20Alignment%20as%20Prospect%20Theoretic%20Optimization.md)

## 关联页面

- [LLM RL](../topics/LLM%20RL.md)
- [指令对齐与 post-training](../topics/指令对齐与%20post-training.md)
- [RLHF](../concepts/RLHF.md)
- [DPO](../concepts/DPO.md)
- [ORPO](../concepts/ORPO.md)
- [KTO](../concepts/KTO.md)

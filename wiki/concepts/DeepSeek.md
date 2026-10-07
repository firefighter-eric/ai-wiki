---
type: concept
---
# DeepSeek

## TL;DR（快速导读）

DeepSeek 家族可沿三条线阅读：通用模型的高效训练、R1 的推理后训练，以及长上下文和 OCR 的信息压缩。

## 简介

DeepSeek 家族可沿三条线阅读：通用模型的高效训练、R1 的推理后训练，以及长上下文和 OCR 的信息压缩。

## 具体怎么理解

想理解推理先看 R1；想理解训练成本先看 V3；想处理长文再看缓存与 OCR，三者不是一个问题。

## 关键属性

- 类型：模型家族 / 研究系列 / 机构技术路线
- 代表来源：
  - [Unknown - 2024 - DeepSeek-V3 Technical Report](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)
  - [Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models](../../wiki/summaries/Shao%20et%20al.%20-%202024%20-%20DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models.md)
  - [DeepSeek-R1：奖励驱动推理与多阶段训练（2025）](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md)
  - [DeepSeek AI - 2025 - DeepSeek-R1-0528 Release](../../wiki/summaries/DeepSeek%20AI%20-%202025%20-%20DeepSeek-R1-0528%20Release.md)
  - [DeepSeek AI - 2025 - DeepSeek-V3.2 Release](../../wiki/summaries/DeepSeek%20AI%20-%202025%20-%20DeepSeek-V3.2%20Release.md)
  - [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](../../wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)
  - [Wei, Sun, Li - 2025 - DeepSeek-OCR Contexts Optical Compression](../../wiki/summaries/Wei,%20Sun,%20Li%20-%202025%20-%20DeepSeek-OCR%20Contexts%20Optical%20Compression.md)
  - [Wei, Sun, Li - 2026 - DeepSeek-OCR 2 Visual Causal Flow](../../wiki/summaries/Wei,%20Sun,%20Li%20-%202026%20-%20DeepSeek-OCR%202%20Visual%20Causal%20Flow.md)
- 当前角色：连接 `LLM 预训练`、`LLM RL`、长上下文效率、agent tool-use 与 OCR / document parsing 的家族导航页

## 相关主张

- `DeepSeek-V3` 代表 DeepSeek 家族的高效 MoE 基座：其重点是稀疏激活、训练效率和开放模型能力，而不是单纯参数规模。
- `DeepSeekMath` 与 `DeepSeek-R1` 把 DeepSeek 主线推进到 reasoning-oriented RL：`GRPO` 与大规模 RL 直接服务于数学、代码和长链推理行为塑形。
- `DeepSeek-R1-0528` 表明 R1 主线继续向更稳定的交互接口发展，包括 JSON output、function calling 与幻觉降低等可用性更新。
- `DeepSeek-V3.2` 是 R1 与 V4 之间的 agent 桥接节点：它把 thinking 集成到 tool-use，并用大规模 agent 环境与复杂指令合成支撑工具使用训练。
- `DeepSeek-V4` 把 DeepSeek 的效率叙事推进到 `1M` 上下文、`CSA / HCA` 混合 attention、`mHC` residual 稳定连接、`Muon` optimizer 与长程 agent 工作流。
- `DeepSeek-OCR` 与 `DeepSeek-OCR 2` 是 OCR 分支，但它们对 DeepSeek 主线的意义在于把文档视觉表示作为 long-context compression 的实验场。

## 来源支持

- [Unknown - 2024 - DeepSeek-V3 Technical Report](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)
- [Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models](../../wiki/summaries/Shao%20et%20al.%20-%202024%20-%20DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models.md)
- [DeepSeek-R1：奖励驱动推理与多阶段训练（2025）](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md)
- [DeepSeek AI - 2025 - DeepSeek-R1-0528 Release](../../wiki/summaries/DeepSeek%20AI%20-%202025%20-%20DeepSeek-R1-0528%20Release.md)
- [DeepSeek AI - 2025 - DeepSeek-V3.2 Release](../../wiki/summaries/DeepSeek%20AI%20-%202025%20-%20DeepSeek-V3.2%20Release.md)
- [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](../../wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)
- [Wei, Sun, Li - 2025 - DeepSeek-OCR Contexts Optical Compression](../../wiki/summaries/Wei,%20Sun,%20Li%20-%202025%20-%20DeepSeek-OCR%20Contexts%20Optical%20Compression.md)
- [Wei, Sun, Li - 2026 - DeepSeek-OCR 2 Visual Causal Flow](../../wiki/summaries/Wei,%20Sun,%20Li%20-%202026%20-%20DeepSeek-OCR%202%20Visual%20Causal%20Flow.md)

## 关联页面

- [DeepSeek 系列](../topics/DeepSeek%20系列.md)
- [DeepSeek-V3](./DeepSeek-V3.md)
- [DeepSeek-R1](./DeepSeek-R1.md)
- [DeepSeek-V4](./DeepSeek-V4.md)
- [DeepSeek-OCR](./DeepSeek-OCR.md)
- [GRPO](./GRPO.md)
- [MoE](./MoE.md)
- [Compressed Sparse Attention](./Compressed%20Sparse%20Attention.md)
- [Heavily Compressed Attention](./Heavily%20Compressed%20Attention.md)
- [Manifold-Constrained Hyper-Connections](./Manifold-Constrained%20Hyper-Connections.md)
- [Muon](./Muon.md)
- [LLM 预训练](../topics/LLM%20预训练.md)
- [LLM RL](../topics/LLM%20RL.md)
- [OCR](../topics/OCR.md)

## 这里的术语是什么意思

- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。
- **OCR**：文字识别：从图像读取文字；整页任务还需处理布局和阅读顺序。
- **GRPO**：组相对策略优化：利用同一问题多份回答的相对奖励进行更新。
- **agent**：代理：围绕任务读材料、调用工具和连续执行的系统；名称本身不保证自主性或质量。

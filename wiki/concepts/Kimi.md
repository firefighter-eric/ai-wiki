---
type: concept
---
# Kimi

## TL;DR（快速导读）

Kimi 家族入口连接长上下文、推理训练与后续开放模型，开放性和部署条件要逐代核对。

## 简介

Kimi 家族入口连接长上下文、推理训练与后续开放模型，开放性和部署条件要逐代核对。

## 具体怎么理解

某代模型只提供接口，不代表后续所有型号也如此；先确认具体版本再讨论能力与使用方式。

## 关键属性

- 类型：大模型家族 / reasoning、长上下文与 agent 路线
- 机构：Moonshot AI / Kimi Team
- 当前直接覆盖：`Kimi k1.5`、`Kimi K3`
- 开放性：按具体代际区分；k1.5 来源主要是技术报告，K3 发布完整权重并采用自定义许可证
- 当前角色：中国重要模型家族，同时通过 K3 进入全球 open-weight frontier model 主线

## 相关主张

- Kimi 家族的连续主轴是 long context、reasoning 与 agent execution，但不同代际的模型结构、模态与开放策略不能合并成一个静态标签。
- `Kimi k1.5` 的主要知识价值是 long-context RL 与 reasoning scaling；它不能为 K3 的 open-weight、MoE、KDA 或 native multimodality 提供直接证据。
- `Kimi K3` 是当前家族的结构性转折：2.78T/104.2B MoE、KDA/Gated MLA、AttnRes、Stable LatentMoE、MoonViT-V2 与 1M agentic RL 共同构成新基座。
- K3 的权重开放使 Kimi 不再只是 closed/API 对照节点，但自定义许可证、训练透明度和 64+ accelerator 推荐部署形态意味着它也不等同于 fully open research release 或低门槛本地模型。
- 家族级 benchmark 结论必须绑定具体 model、reasoning effort、harness、tools、context management 与评测日期；不能从 K3 的单次官方主表反推整个 Kimi 家族的永久排名。

## 来源支持

- [Kimi Team et al. - 2025 - Kimi k1.5 Scaling Reinforcement Learning with LLMs](../../wiki/summaries/Kimi%20Team%20et%20al.%20-%202025%20-%20Kimi%20k1.5%20Scaling%20Reinforcement%20Learning%20with%20LLMs.md)
- [Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence](../../wiki/summaries/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)
- [Kimi - 2026 - Kimi K3 Open Frontier Intelligence Release](../../wiki/summaries/Kimi%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence%20Release.md)
- [Moonshot AI - 2026 - Kimi K3 Model Repository](../../wiki/summaries/Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20Model%20Repository.md)
- [Moonshot AI - 2026 - Kimi K3 License](../../wiki/summaries/Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20License.md)

## 关联页面

- [Kimi K3](./Kimi%20K3.md)
- [Kimi Delta Attention](./Kimi%20Delta%20Attention.md)
- [Attention Residuals](./Attention%20Residuals.md)
- [Stable LatentMoE](./Stable%20LatentMoE.md)
- [MoonViT-V2](./MoonViT-V2.md)
- [MoonEP](./MoonEP.md)
- [MoE](./MoE.md)
- [Muon](./Muon.md)
- [LLM 预训练](../topics/LLM%20预训练.md)
- [LLM RL](../topics/LLM%20RL.md)
- [开放模型家族与中国重要家族对照](../comparisons/开放模型家族与中国重要家族对照.md)
- [Moonshot AI](../authors/Moonshot%20AI.md)

## 这里的术语是什么意思

- **benchmark**：评测基准：特定数据、任务与规则的组合，分数只在这些条件下成立。
- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。
- **agentic**：代理执行：模型使用工具并根据结果继续行动，可靠性要看完整流程。
- **agent**：代理：围绕任务读材料、调用工具和连续执行的系统；名称本身不保证自主性或质量。

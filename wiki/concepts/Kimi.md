---
type: concept
---
# Kimi

## 简介

`Kimi` 是 Moonshot AI 的模型与 agent 产品家族。当前知识库中的两条直接证据分别代表家族的不同阶段：`Kimi k1.5` 说明其在 long-context reasoning RL 上的早期路线；`Kimi K3` 则把家族推进到开放权重、原生多模态、3T 级 MoE、1M context 与长程 agent systems co-design。因而旧的“整个 Kimi 家族均为 API/闭源”判断已经失效。

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

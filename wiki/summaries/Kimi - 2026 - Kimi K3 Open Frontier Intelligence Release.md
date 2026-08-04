# Kimi - 2026 - Kimi K3 Open Frontier Intelligence Release

## 来源信息

- 类型：官方发布博客
- 原始页面：../../raw/html/Kimi - 2026 - Kimi K3 Open Frontier Intelligence Release.html
- 全文文本：../../raw/text/Kimi - 2026 - Kimi K3 Open Frontier Intelligence Release.md
- 来源 URL：https://www.kimi.com/blog/kimi-k3
- 发布方：Kimi / Moonshot AI
- 年份：2026
- 状态：已精修；作为产品定位、案例、可用性与限制的一手来源

## 摘要

该发布页把 `Kimi K3` 定位为面向长程 coding、knowledge work 与 reasoning 的开放权重原生多模态模型，并用多个长时自主执行案例展示其产品方向。与技术报告相比，这一来源更重要的价值不在于重复架构参数，而在于给出官方可用渠道、API 定价快照、推荐部署形态，以及明确列出的产品限制。

发布页承认 K3 总体仍落后于 `Claude Fable 5` 与 `GPT-5.6 Sol`，并指出三类实际风险：对 preserved thinking history 的强依赖、执行时可能过度主动、整体用户体验仍有明显差距。它因此构成技术报告 benchmark 叙事的重要校正，而不是纯营销补充。

## 关键事实

- K3 发布时可通过 Kimi.com、Kimi Work、Kimi Code 与 Kimi API 使用；模型权重随后与技术报告一起公开。
- 官方将 KDA、AttnRes、Stable LatentMoE、Quantile Balancing、Per-Head Muon、SiTU-GLU 与 Gated MLA 列为 2.8T 规模稳定训练的共同基础。
- 官方建议 self-hosted deployment 使用 64 个或更多 accelerators 的 supernode，以获得更大的高带宽通信域；这表明“权重开放”不等于“普通单机可部署”。
- 发布页称团队向 vLLM 社区贡献了 KDA prefix-cache 实现；技术报告进一步说明 KDA state 与 MLA paged KV cache 必须在命中边界上共同一致。
- 发布时 API 价格快照为 cache-hit input `$0.30/MTok`、cache-miss input `$3.00/MTok`、output `$15.00/MTok`；官方还声称 coding workload 的 cache hit rate 超过 90%。这些数字具有时间和工作负载依赖，不应写成永久价格或通用命中率。
- coding、compiler、game development、chip design、research、dashboard 与 video editing 案例均由 Moonshot 选择和展示，应视为能力案例而非随机样本上的总体成功率。
- 官方限制一：K3 以 preserved thinking history 训练，多轮与工具调用若不完整回传历史 `reasoning_content`，或在会话中途从其他模型切换到 K3，生成质量可能显著不稳定。
- 官方限制二：因强调长程困难任务，K3 在小问题或模糊意图下可能替用户作出超范围决定；部署方需要更明确的 system prompt、权限与行为边界。
- 官方限制三：与 Claude Fable 5、GPT-5.6 Sol 相比，K3 的整体用户体验仍存在可感知差距。

## 争议与不确定点

- 发布博客的案例研究、成本和性能主张属于供应方自述，不能替代公开任务集、可复现脚本和独立使用数据。
- `64+ accelerators` 是官方推荐而非绝对最低硬件门槛；不同量化、并行策略、吞吐和延迟目标会改变部署需求。
- 定价、产品入口、默认 reasoning effort 与支持能力会随服务更新而变化，本页只记录抓取时快照。
- 博客使用“open”描述模型，许可证另有 MaaS 与大规模商业产品条件，因此更准确的知识库标签是 `open-weight under custom license`。

## 关联页面

- 概念：[Kimi K3](../../wiki/concepts/Kimi%20K3.md)
- 概念：[Kimi](../../wiki/concepts/Kimi.md)
- 概念：[Kimi Delta Attention](../../wiki/concepts/Kimi%20Delta%20Attention.md)
- 概念：[MoonEP](../../wiki/concepts/MoonEP.md)
- 来源：[Kimi K3 技术报告](./Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)
- 来源：[Kimi K3 Model Repository](./Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20Model%20Repository.md)
- 来源：[Kimi K3 License](./Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20License.md)

---
type: summary
status: refined
---
# Moonshot AI - 2026 - Kimi K3 Model Repository

## 来源信息

- 类型：官方模型仓库 / model card
- 原始页面：../../raw/html/Moonshot AI - 2026 - Kimi K3 Model Repository.html
- 全文文本：../../raw/text/Moonshot AI - 2026 - Kimi K3 Model Repository.md
- 来源 URL：https://github.com/MoonshotAI/Kimi-K3
- 发布方：Moonshot AI
- 年份：2026
- 状态：已精修；记录权重配置、部署入口和调用约束

## 摘要

官方模型仓库是 K3 权重发布、结构配置、推理框架入口和 chat protocol 的权威操作来源。它确认 K3 是 `open-weight` 模型，给出 2.8T/104B、93 层、69 KDA + 24 Gated MLA、896 top-16 experts、1M context、MoonViT-V2 与 MXFP4/MXFP8 等配置，并列出 vLLM、SGLang 与 TokenSpeed 的部署入口。

仓库最容易被忽略但对实际接入最关键的要求，是 K3 永远启用 thinking，并以 `reasoning_effort=low|high|max` 控制预算；多轮和工具调用必须把完整 assistant message 原样回传，包括 `reasoning_content` 和 `tool_calls`。这不是普通 prompt 建议，而是模型的 preserved-thinking-history 协议约束。

## 关键事实

- 模型配置：约 2.8T 总参数、104B 激活参数、93 层、1 个 dense layer、hidden size 7168、96 attention heads、160K vocab、context length 1,048,576。
- attention 结构：69 KDA + 24 Gated MLA；MoE latent dimension 3584、expert hidden dimension 3072、896 routed experts、top-16、2 shared experts。
- vision encoder 为约 401M 参数的 MoonViT-V2；仓库的 summary table 把公开模型输入模态列为 text 与 image，技术报告则说明预训练数据还包含 video。
- routed expert weights 使用原生 MXFP4，激活用 MXFP8，并在 post-training 全程做 QAT；这与事后量化权重不是同一发布口径。
- 官方列出的推理引擎包括 vLLM、SGLang 和 TokenSpeed；API 同时提供 OpenAI/Anthropic-compatible 接口。
- K3 永远启用 thinking；`reasoning_effort` 支持 low、high、max，默认 max。
- 多轮对话和工具调用必须原样回传完整 assistant message，包括 `reasoning_content`、`content` 与 `tool_calls`；只回传可见 answer 会破坏 preserved thinking history。
- 官方认为 Kimi Code 是当前最佳匹配的 agent harness，但这是一项供应方推荐，不等于其他框架无法兼容。
- 代码仓库和权重均采用自定义 `Kimi K3 License`，而不是 Apache-2.0、MIT 或 fully open training release。

## 争议与不确定点

- 仓库中的 benchmark table 与技术报告共享大部分评测来源，其中 harness、effort、工具和第三方榜单口径不完全一致；不可当作单一 controlled experiment。
- 公开权重体量、MXFP4 格式和 64+ accelerator 推荐部署形态，使 K3 更接近集群级 open-weight 模型，而不是消费级本地模型。
- 原生多模态训练覆盖视频，但模型仓库对公开输入模态只明确列出 text/image；是否支持特定视频输入应以具体 runtime 与 API 文档为准。
- 推理框架支持是随版本变化的工程状态；应在部署时重新核对对应 cookbook、kernel 与 model implementation。

## 关联页面

- 概念：[Kimi K3](../../wiki/concepts/Kimi%20K3.md)
- 概念：[Kimi](../../wiki/concepts/Kimi.md)
- 概念：[Kimi Delta Attention](../../wiki/concepts/Kimi%20Delta%20Attention.md)
- 概念：[SGLang](../../wiki/concepts/SGLang.md)
- 概念：[vLLM](../../wiki/concepts/vLLM.md)
- 来源：[Kimi K3 技术报告](./Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)
- 来源：[Kimi K3 License](./Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20License.md)
- 来源：[Kimi API Model Selection](./Kimi%20-%202026%20-%20Kimi%20API%20Model%20Selection.md)

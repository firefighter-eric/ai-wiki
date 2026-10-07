---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Moonshot AI - 2026 - Kimi K3 Model Repository

## TL;DR（快速导读）

Kimi K3 模型仓库提供结构、部署入口和消息协议，其中多轮工具调用需要保留完整助手历史。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

完成代码任务可能要读取仓库、运行工具和根据结果继续行动；模型分数与整条执行系统的可靠性要分开测。

## 来源信息

- 类型：官方模型仓库 / model card
- 原始页面：../../raw/html/Moonshot AI - 2026 - Kimi K3 Model Repository.html
- 全文文本：../../raw/text/Moonshot AI - 2026 - Kimi K3 Model Repository.md
- 来源 URL：https://github.com/MoonshotAI/Kimi-K3
- 发布方：Moonshot AI
- 年份：2026
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

仓库说明思考预算、模型配置与推理框架接入。尤其需要按协议回传 reasoning_content 和 tool_calls 等助手内容；遗漏历史可能影响行为，这与单纯改变提示措辞不同，且配置只对应归档版本。

## 关键事实

- **C1**：**C1**：模型配置：约 2.8T 总参数、104B 激活参数、93 层、1 个 dense layer、hidden size 7168、96 attention heads、160K vocab、context length 1,048,576。
- **C2**：**C2**：attention 结构：69 KDA + 24 Gated MLA；MoE latent dimension 3584、expert hidden dimension 3072、896 routed experts、top-16、2 shared experts。
- **C3**：**C3**：vision encoder 为约 401M 参数的 MoonViT-V2；仓库的 summary table 把公开模型输入模态列为 text 与 image，技术报告则说明预训练数据还包含 video。
- **C4**：**C4**：routed expert weights 使用原生 MXFP4，激活用 MXFP8，并在 post-training 全程做 QAT；这与事后量化权重不是同一发布口径。
- **C5**：**C5**：官方列出的推理引擎包括 vLLM、SGLang 和 TokenSpeed；API 同时提供 OpenAI/Anthropic-compatible 接口。
- **C6**：**C6**：K3 永远启用 thinking；`reasoning_effort` 支持 low、high、max，默认 max。
- **C7**：**C7**：多轮对话和工具调用必须原样回传完整 assistant message，包括 `reasoning_content`、`content` 与 `tool_calls`；只回传可见 answer 会破坏 preserved thinking history。
- **C8**：**C8**：官方认为 Kimi Code 是当前最佳匹配的 agent harness，但这是一项供应方推荐，不等于其他框架无法兼容。
- **C9**：**C9**：代码仓库和权重均采用自定义 `Kimi K3 License`，而不是 Apache-2.0、MIT 或 fully open training release。

## 争议与不确定点

- 配置与推理框架会随仓库版本变化，本页限定保存快照。
- 官网推荐某coding harness是厂商建议，基准结果也依赖具体harness与context管理。

## 关联页面

- 概念：[Kimi K3](../../wiki/concepts/Kimi%20K3.md)
- 概念：[Kimi](../../wiki/concepts/Kimi.md)
- 概念：[Kimi Delta Attention](../../wiki/concepts/Kimi%20Delta%20Attention.md)
- 概念：[SGLang](../../wiki/concepts/SGLang.md)
- 概念：[vLLM](../../wiki/concepts/vLLM.md)
- 来源：[Kimi K3 技术报告](./Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)
- 来源：[Kimi K3 License](./Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20License.md)
- 来源：[Kimi API Model Selection](./Kimi%20-%202026%20-%20Kimi%20API%20Model%20Selection.md)
- [Moonshot AI](../authors/Moonshot%20AI.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。
- **latent**：潜表示：原始数据经过模型编码后的内部表示，通常更紧凑。
- **encoder**：编码器：把输入转成模型内部表示。
- **agent**：代理：围绕任务读材料、调用工具和连续执行的系统；名称本身不保证自主性或质量。
- **post-training**：后训练：在预训练底座上继续调整指令遵循、偏好或其他行为。

## 方法与实验解读

仓库是部署契约，技术报告是训练与实验说明，两者版本和模态表述需分别记录。模型卡写Text/Image，报告包含视频评测；不能由后者推定任意服务端都接受视频。长对话工具调用需要回传完整assistant状态，preserved thinking是模型用法要求。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20Model%20Repository.md#source-section-3 ) | 保存版本明确表述；配置、解释和独立实验结果分别理解。 |
| C2 | [原文]( ../../raw/text/Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20Model%20Repository.md#source-section-3 ) | 保存版本明确表述；配置、解释和独立实验结果分别理解。 |
| C3 | [原文]( ../../raw/text/Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20Model%20Repository.md#source-section-3 ) | 保存版本明确表述；配置、解释和独立实验结果分别理解。 |
| C4 | [原文]( ../../raw/text/Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20Model%20Repository.md#source-section-5 ) | 保存版本明确表述；配置、解释和独立实验结果分别理解。 |
| C5 | [原文]( ../../raw/text/Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20Model%20Repository.md#source-section-6 ) | 保存版本明确表述；配置、解释和独立实验结果分别理解。 |
| C6 | [原文]( ../../raw/text/Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20Model%20Repository.md#source-section-7 ) | 保存版本明确表述；配置、解释和独立实验结果分别理解。 |
| C7 | [原文]( ../../raw/text/Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20Model%20Repository.md#source-section-7 ) | 保存版本明确表述；配置、解释和独立实验结果分别理解。 |
| C8 | [原文]( ../../raw/text/Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20Model%20Repository.md#source-section-8 ) | 保存版本明确表述；配置、解释和独立实验结果分别理解。 |
| C9 | [原文]( ../../raw/text/Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20Model%20Repository.md#source-section-9 ) | 保存版本明确表述；配置、解释和独立实验结果分别理解。 |

## 核证范围

保留9项详细配置，核读模型表、QAT、部署、thinking历史、harness与许可说明。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

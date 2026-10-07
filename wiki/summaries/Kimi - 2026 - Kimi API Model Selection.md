---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Kimi - 2026 - Kimi API Model Selection

## TL;DR（快速导读）

这份 Kimi API 快照帮助辨认 K3 与 K2.6 的模式、上下文和思考预算，适合核对调用接口。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

某代模型只提供接口，不代表后续所有型号也如此；先确认具体版本再讨论能力与使用方式。

## 来源信息

- 类型：官方 API 帮助文档
- 原始页面：../../raw/html/Kimi - 2026 - Kimi API Model Selection.html
- 全文文本：../../raw/text/Kimi - 2026 - Kimi API Model Selection.md
- 来源 URL：https://www.kimi.com/help/kimi-api/api-model-selection
- 发布方：Kimi / Moonshot AI
- 抓取日期：2026-08-04
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

页面说明不同模型的思考模式和 reasoning_effort 参数。接口上限、预算选项是归档时的服务约定，可能随版本变化；它不能替代权重部署配置，也不能省略多轮工具调用的消息协议要求。

## 关键事实

- **C1**：快照中kimi-k3始终thinking，reasoning_effort支持low/high/max、默认max，窗口最高1M。
- **C2**：kimi-k2.6支持thinking开关与256K。
- **C3**：图像支持URL/Base64，快照按每图1024 tokens计费。
- **C4**：快照明确PPT generation和Deep Research未开放通用API。

## 争议与不确定点

- 页面不提供全任务性能比较，选择维度不是排行榜。
- 计费和API支持随版本变化，本页只陈述存档规则。

## 关联页面

- 概念：[Kimi K3](../../wiki/concepts/Kimi%20K3.md)
- 概念：[Kimi](../../wiki/concepts/Kimi.md)
- 来源：[Kimi K3 Model Repository](./Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20Model%20Repository.md)
- 来源：[Kimi K3 官方发布](./Kimi%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence%20Release.md)

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。

## 方法与实验解读

选模型先确定上下文、thinking开关、输入模态与调用成本，再测试目标任务。K3的长程定位通常带来更高推理预算，低延迟应用应记录实际输出长度和effort。文档里的尚未支持是快照状态，后续接入应复核当前API。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Kimi%20-%202026%20-%20Kimi%20API%20Model%20Selection.md#source-section-3 ) | 版本化API文档；发布首日仅max与后续三档不要混写。 |
| C2 | [原文]( ../../raw/text/Kimi%20-%202026%20-%20Kimi%20API%20Model%20Selection.md#source-section-4 ) | 需要non-thinking时两型号非等价替换。 |
| C3 | [原文]( ../../raw/text/Kimi%20-%202026%20-%20Kimi%20API%20Model%20Selection.md#source-section-6 ) | 计费规则时间依赖，不保证后来仍相同。 |
| C4 | [原文]( ../../raw/text/Kimi%20-%202026%20-%20Kimi%20API%20Model%20Selection.md#source-section-7 ) | 产品UI功能不能自动推导API能力。 |

## 核证范围

核读完整model selection及图像/未支持能力段落。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

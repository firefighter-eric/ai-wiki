---
type: summary
status: refined
---
# Kimi - 2026 - Kimi API Model Selection

## 来源信息

- 类型：官方 API 帮助文档
- 原始页面：../../raw/html/Kimi - 2026 - Kimi API Model Selection.html
- 全文文本：../../raw/text/Kimi - 2026 - Kimi API Model Selection.md
- 来源 URL：https://www.kimi.com/help/kimi-api/api-model-selection
- 发布方：Kimi / Moonshot AI
- 抓取日期：2026-08-04
- 状态：当前服务快照；接口能力可能变化

## 摘要

该页面说明 Kimi 官方 API 中 `kimi-k3` 与 `kimi-k2.6` 的定位差异。`kimi-k3` 是面向长程 coding、端到端 knowledge work 和视觉理解的旗舰模型，始终运行在 thinking mode，并通过顶层 `reasoning_effort` 参数控制 low/high/max 三档预算；`kimi-k2.6` 则保留 thinking / non-thinking 切换，最大上下文较短。

本来源的价值是把技术报告中的多档 reasoning effort 落到实际 API contract，并确认 K3 的官方服务上限为 1M context。它不提供权重部署细节，也不应替代模型仓库的 preserved-thinking-history 要求。

## 关键事实

- API model id 为 `kimi-k3`；官方定位是 long-horizon coding、end-to-end knowledge work 与 native visual understanding。
- K3 始终启用 thinking；顶层 `reasoning_effort` 支持 low、high、max，默认 max。
- 官方最大上下文窗口为 1M token。
- 与之对照，`kimi-k2.6` 支持 thinking/non-thinking 切换并提供 256K context，因此需要关闭 thinking 的场景不能直接把 K3 当作等价替代。
- API 的 image input 可通过 URL 或 Base64 传入，常见 JPEG/PNG/WebP 等格式受支持；页面快照称每张图固定按 1024 tokens 计费。
- 该页面快照中，PPT generation API 与 Deep Research API 尚未开放；产品 UI 能力不应自动推断为通用 API 能力。

## 争议与不确定点

- 这是 living documentation，model id、默认 effort、图像计费和 unsupported capability 均可能在后续更新。
- 1M 是协议上限，不代表每个请求都应使用完整上下文，也不保证在所有任务上保持等价质量或成本。
- 页面没有完整列出 preserved thinking history 的回传要求，实际接入还必须同时阅读模型仓库和 API schema。

## 关联页面

- 概念：[Kimi K3](../../wiki/concepts/Kimi%20K3.md)
- 概念：[Kimi](../../wiki/concepts/Kimi.md)
- 来源：[Kimi K3 Model Repository](./Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20Model%20Repository.md)
- 来源：[Kimi K3 官方发布](./Kimi%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence%20Release.md)

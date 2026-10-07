---
type: summary
status: refined
evidence_schema: 1
reviewed: 2026-10-07
---
# Anthropic - 2025 - Effective Context Engineering for AI Agents

## TL;DR（快速导读）

长任务要按问题读取材料，并把已确认的结论与待办写成笔记，让有限上下文保存真正需要的状态。

## 先看一个例子

整理长论文时，先读目录定位方法与实验，再按需要打开原文；已核证结论写回 wiki，避免每次都重读全篇。

## 来源信息

- 类型：官方工程文章；发布于 2025-09-29，核验快照为 2026-10-07。
- 官方来源：[文章](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- 原始 HTML：[快照](../../raw/html/Anthropic%20-%202025%20-%20Effective%20Context%20Engineering%20for%20AI%20Agents.html)
- 全文文本：[正文](../../raw/text/Anthropic%20-%202025%20-%20Effective%20Context%20Engineering%20for%20AI%20Agents.md)
- 状态：精修 summary，已核对相关章节。

## 摘要

文章把上下文当作有限资源，讨论按需加载材料、清晰的工具接口、压缩和结构化笔记。用于本库时，索引负责定位，原文负责核对，wiki 保存已经整理的知识；笔记应保留证据和未解决问题。

## 关键事实

- C1：文件路径、查询与链接可以作为轻量标识，让 agent 按需加载内容，逐步发现上下文。
- C2：工具和系统指令应清晰，返回与任务有关的信息；上下文体积大不自动意味着使用效果好。
- C3：长任务可采用上下文压缩、结构化笔记等机制；文章也讨论多 agent，但其适用性依赖任务。

## 证据定位

| 主张 | 原文章节与定位 | 适用条件 |
|---|---|---|
| C1 | [上下文检索](../../raw/text/Anthropic%20-%202025%20-%20Effective%20Context%20Engineering%20for%20AI%20Agents.md#source-section-4) | 工具能可靠地读取相应文件或数据 |
| C2 | [上下文组织](../../raw/text/Anthropic%20-%202025%20-%20Effective%20Context%20Engineering%20for%20AI%20Agents.md#source-section-3) | 工程建议，需按任务验证 |
| C3 | [长任务](../../raw/text/Anthropic%20-%202025%20-%20Effective%20Context%20Engineering%20for%20AI%20Agents.md#source-section-5) | 笔记与压缩须保留必要信息 |

## 争议与不确定点

- 这是工程经验文章，不能证明某种架构对所有知识库更优。
- 按需探索会增加运行时读取；本库采用该思路的工具改动仍需回归评测。
- 多 agent 讨论不构成本次会话的分工授权，也不是实施本流程的必要条件。

## 关联页面

- [LLM Wiki 文档处理流程](../concepts/LLM%20Wiki%20文档处理流程.md)
- [LLM Wiki 与检索和文档解析方法](../comparisons/LLM%20Wiki%20与检索和文档解析方法.md)

## 这里的术语是什么意思

- **agent**：代理：围绕任务读材料、调用工具和连续执行的系统；名称本身不保证自主性或质量。

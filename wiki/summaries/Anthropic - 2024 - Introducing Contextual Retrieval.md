---
type: summary
status: refined
evidence_schema: 1
reviewed: 2026-10-07
---
# Anthropic - 2024 - Introducing Contextual Retrieval

## TL;DR（快速导读）

Contextual Retrieval 在片段进入检索索引前补上整篇文档的背景，减少断章取义的匹配。

## 先看一个例子

只有“增长了 20%”的片段无法判断对象和时期；把它所属文档与章节带上，才便于检索和核对。

## 来源信息

- 类型：官方工程文章；发布于 2024-09-19，核验快照为 2026-10-07。
- 官方来源：[文章](https://www.anthropic.com/engineering/contextual-retrieval)
- 原始 HTML：[快照](../../raw/html/Anthropic%20-%202024%20-%20Introducing%20Contextual%20Retrieval.html)
- 全文文本：[正文](../../raw/text/Anthropic%20-%202024%20-%20Introducing%20Contextual%20Retrieval.md)
- 状态：精修 summary，已核对方法与评测范围。

## 摘要

孤立片段可能丢失人物、时期或适用条件。文章介绍为片段添加特定上下文，再结合语义检索、词项检索与重排序；对本库，保留标题、章节和来源状态是可以先做的一步，收益仍需检索评测。

## 关键事实

- C1：片段特定的上下文同时用于语义与 BM25 索引；这不同于给全部片段追加同一份通用摘要。
- C2：重排序是召回后的筛选步骤，涉及质量、开销与延迟的权衡。
- C3：原文评测基于特定数据、模型与 top-20 口径，并要求按用例评估片段边界等配置。

## 证据定位

| 主张 | 原文章节与定位 | 适用条件 |
|---|---|---|
| C1 | [方法](../../raw/text/Anthropic%20-%202024%20-%20Introducing%20Contextual%20Retrieval.md#source-section-4) | 补充文本需准确反映全文上下文 |
| C2 | [重排序](../../raw/text/Anthropic%20-%202024%20-%20Introducing%20Contextual%20Retrieval.md#source-section-10) | 需衡量额外推理成本 |
| C3 | [评测方法](../../raw/text/Anthropic%20-%202024%20-%20Introducing%20Contextual%20Retrieval.md#source-section-7)；[配置](../../raw/text/Anthropic%20-%202024%20-%20Introducing%20Contextual%20Retrieval.md#source-section-9) | 作者测试范围，不能直接外推本库 |

## 争议与不确定点

- 本轮实现来源上下文与定位、词面召回回归，没有实施该文完整的生成式片段上下文、embedding 与 reranker 管线。
- 来源摘要和知识综述具有长期阅读价值，不能从本篇检索实验推导“wiki summary 没有用”。
- 作者自报的检索改进不能作为本库已经获得同等效果的证据。

## 关联页面

- [LLM Wiki 文档处理流程](../concepts/LLM%20Wiki%20文档处理流程.md)
- [LLM Wiki 与检索和文档解析方法](../comparisons/LLM%20Wiki%20与检索和文档解析方法.md)

---
type: summary
status: refined
evidence_schema: 1
reviewed: 2026-10-07
---
# Microsoft - 2026 - GraphRAG Query Engine

## TL;DR（快速导读）

GraphRAG 按问题范围选择查询路径：具体实体问题与全库综合问题需要不同的材料组织方式。

## 先看一个例子

问一个模型的缓存机制，可进入局部证据；问整个领域的路线变化，则需要组织多份来源并比较。

## 来源信息

- 类型：官方软件文档；2026-10-07 快照，年份表示核验时间。
- 官方来源：[文档](https://microsoft.github.io/graphrag/query/overview/)
- 原始 HTML：[快照](../../raw/html/Microsoft%20-%202026%20-%20GraphRAG%20Query%20Engine.html)
- 全文文本：[正文](../../raw/text/Microsoft%20-%202026%20-%20GraphRAG%20Query%20Engine.md)
- 状态：精修 summary，已核对查询模式。

## 摘要

官方文档区分局部实体、全局综合及由社区信息扩展的查询。用于本库时，可以先让 concept、topic、comparison 和 summary 承担这些不同入口，是否增加额外图索引再由评测决定。

## 关键事实

- C1：Local search 联合知识图谱与原文片段，面向具体实体问题。
- C2：Global search 对社区报告进行 map-reduce，面向全局问题，文档明确其资源开销较高。
- C3：DRIFT 使用社区信息拓展局部检索，并细化后续问题。

## 证据定位

| 主张 | 原文章节与定位 | 适用条件 |
|---|---|---|
| C1 | [Local](../../raw/text/Microsoft%20-%202026%20-%20GraphRAG%20Query%20Engine.md#source-section-2) | 已有完成的 GraphRAG 索引 |
| C2 | [Global](../../raw/text/Microsoft%20-%202026%20-%20GraphRAG%20Query%20Engine.md#source-section-3) | 已生成社区报告 |
| C3 | [DRIFT](../../raw/text/Microsoft%20-%202026%20-%20GraphRAG%20Query%20Engine.md#source-section-4) | 依赖图和社区数据 |

## 争议与不确定点

- 本库的 Markdown 链接图与任务路由不是完整 GraphRAG 实现。
- 本页没有比较中文论文综述场景的成本、引用准确性或查全率。
- 文档持续变化；引入完整图索引前需要用本库问题评测其额外收益。

## 关联页面

- [GraphRAG 索引方法](./Microsoft%20-%202026%20-%20GraphRAG%20Indexing%20Methods.md)
- [LLM Wiki 文档处理流程](../concepts/LLM%20Wiki%20文档处理流程.md)
- [LLM Wiki 与检索和文档解析方法](../comparisons/LLM%20Wiki%20与检索和文档解析方法.md)

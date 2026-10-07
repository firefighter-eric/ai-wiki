---
type: summary
status: refined
evidence_schema: 1
reviewed: 2026-10-07
---
# Microsoft - 2026 - GraphRAG Indexing Methods

## TL;DR（快速导读）

GraphRAG 的两种索引路线在关系描述、噪声和建库成本之间取舍，生成更多连接并不等于知识更准确。

## 先看一个例子

两个人名同现于一篇文档，只能说明共现；若要断言合作或支持关系，还需要明确内容与来源。

## 来源信息

- 类型：官方软件文档；2026-10-07 快照，年份表示核验时间。
- 官方来源：[文档](https://microsoft.github.io/graphrag/index/methods/)
- 原始 HTML：[快照](../../raw/html/Microsoft%20-%202026%20-%20GraphRAG%20Indexing%20Methods.html)
- 全文文本：[正文](../../raw/text/Microsoft%20-%202026%20-%20GraphRAG%20Indexing%20Methods.md)
- 状态：精修 summary，已核对两种索引路线与限制。

## 摘要

Standard GraphRAG 用语言模型描述实体、关系和社区；FastGraphRAG 在部分环节改用词组与共现图，同时保留社区报告。共现只说明一起出现，若要当作实际关系，仍需原文支持。

## 关键事实

- C1：Standard 对实体、关系及其汇总使用 LLM，并支持可选 claim extraction。
- C2：Fast 使用 NLP 词组与同片段共现，省去实体、关系描述，但仍生成社区报告。
- C3：官方说明 Fast 默认 NLP 主要适合英文，图更嘈杂；高保真图探索和全局摘要存在不同取舍。

## 证据定位

| 主张 | 原文章节与定位 | 适用条件 |
|---|---|---|
| C1 | [Standard](../../raw/text/Microsoft%20-%202026%20-%20GraphRAG%20Indexing%20Methods.md#source-section-2) | 需要模型与索引配置 |
| C2 | [Fast](../../raw/text/Microsoft%20-%202026%20-%20GraphRAG%20Indexing%20Methods.md#source-section-3) | 共现关系不等于语义或因果关系 |
| C3 | [取舍](../../raw/text/Microsoft%20-%202026%20-%20GraphRAG%20Indexing%20Methods.md#source-section-4)；[语言条件](../../raw/text/Microsoft%20-%202026%20-%20GraphRAG%20Indexing%20Methods.md#source-section-3) | 需针对中文 / 英文混合材料验证 |

## 争议与不确定点

- FastGraphRAG 与 LazyGraphRAG 不应只因名称相似就视为同一实现。
- 官方成本估计依赖配置，不是本库实测；本轮未安装 GraphRAG 或 NLP 模型。
- 共现图不能替代 summary 的证据关系或原文核证。

## 关联页面

- [GraphRAG 查询模式](./Microsoft%20-%202026%20-%20GraphRAG%20Query%20Engine.md)
- [LLM Wiki 与检索和文档解析方法](../comparisons/LLM%20Wiki%20与检索和文档解析方法.md)

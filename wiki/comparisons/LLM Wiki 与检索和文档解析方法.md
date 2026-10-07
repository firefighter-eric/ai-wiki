---
type: comparison
---
# LLM Wiki 与检索和文档解析方法

## TL;DR（快速导读）

本库继续把可读、可修订的 wiki 作为知识产物，再用按需读取、证据定位、查询分层与评测改善执行。额外工具要用真实问题验证收益。

## 用一个例子看差别

整理一篇新论文时，解析器负责提供结构，检索负责找到已有相关页，agent 负责核证与修改判断。某个工具召回了片段，不等于已经消化论文；换解析后端也应先用困难页面比较。

## 比较目标

在大量导入论文前，判断哪些文档处理与 agent 方法能改进本库。资料核验日期为 2026-10-07；参考包含 2024–2025 年工程文章和当前官方软件文档，不将历史方法包装成刚发布的新算法。

## 核心判断

本库适合继续以 persistent wiki 为知识产物，在上下文读取、来源定位、查询路由与评测上补工具。该选择是基于本库规模、现有 Markdown 页面和以下来源的工程判断；目前没有本库同任务、同成本的实验能证明它胜过所有替代架构。

| 方法 | 主要解决的问题 | 与 LLM Wiki 的关系 | 本轮采用 |
|---|---|---|---|
| 按需上下文与结构化笔记 | 有限上下文和长任务状态 | 帮助 agent 阅读并持续回写 wiki | 目录、范围明确的阅读包、追加日志 |
| Contextual Retrieval | 孤立片段丢失主体与条件 | 可作为 wiki 候选召回的增强层 | 保留来源 / 章节 / 成熟度；建立候选召回评测 |
| GraphRAG 查询分层 | 实体局部与全局综合问题范围不同 | 参考其路由思路组织 wiki 导航 | 比较、演进、实体、宽主题分别进入对应页面 |
| Standard / FastGraphRAG 索引 | 图信息丰富度与成本、噪声的取舍 | 未来可增加辅助索引，仍需 summary 证据层 | 先用已有证据链接生成队列，评测后再决定额外图索引 |
| Docling 结构化解析 | 版面、表格、公式、OCR | 在来源到全文的环节提供可核对结构 | HTML 结构保留、PDF 页码；困难样本再评估专用后端 |

功能与边界分别由下方五篇 summary 支持；本轮采用项是本库的实现选择。

## 关键取舍

**知识编写与候选检索承担不同职责。** summary 与综述用于维护可读、可修订的知识；检索返回相关候选。Contextual Retrieval 对检索片段的设计不能直接否定长期维护 summary 的价值；GraphRAG 的社区报告也说明预先汇总有独立用途。参见 [检索上下文](../summaries/Anthropic%20-%202024%20-%20Introducing%20Contextual%20Retrieval.md) 与 [GraphRAG 查询](../summaries/Microsoft%20-%202026%20-%20GraphRAG%20Query%20Engine.md)。

**关系图需要区分连接与证据。** 文件链接适合导航，topic 的证据链接适合识别精读依赖；共现关系不能据此升级为因果或支持关系。FastGraphRAG 的描述与语言限制见 [索引方法 summary](../summaries/Microsoft%20-%202026%20-%20GraphRAG%20Indexing%20Methods.md)。本轮精读优先级明确标为启发式。

**解析成功与理解正确需要分开验证。** HTML 章节和 PDF 页码可以定位来源，复杂表格和图形仍需原文检查。专用解析器也需用困难样本验证，功能列表无法保证全库正确。见 [Docling summary](../summaries/Docling%20Project%20-%202026%20-%20Document%20Processing%20Overview.md)。

**agent 职责分阶段即可实施。** 本库把计划、阅读、核证、整合变成明确职责，并按需读取上下文；是否使用更多 agent 由实际任务与会话授权决定。长任务笔记的依据见 [上下文工程 summary](../summaries/Anthropic%20-%202025%20-%20Effective%20Context%20Engineering%20for%20AI%20Agents.md)。

## 本轮交付与后续门槛

- 已落地：精读队列与读取包、章节 / 页码 / SHA256、精修证据定位协议、qmd 独立缓存及备用检索、成熟度返回、已知目标页面的回归评测。
- 后续可评估：困难 PDF 样本上的 Docling / OCR、语义召回与重排序、全局图索引。需要代表问题或文档与实测收益，再决定启用。
- 仍需实际阅读解决：旧自动摘要的精修、topic 主线成熟、导航缺口与相互矛盾的研究判断。新流程提供队列和门禁，不自动制造这些内容。

## 证据基础

- [Anthropic - 2025 - Effective Context Engineering for AI Agents](../summaries/Anthropic%20-%202025%20-%20Effective%20Context%20Engineering%20for%20AI%20Agents.md)
- [Anthropic - 2024 - Introducing Contextual Retrieval](../summaries/Anthropic%20-%202024%20-%20Introducing%20Contextual%20Retrieval.md)
- [Microsoft - 2026 - GraphRAG Query Engine](../summaries/Microsoft%20-%202026%20-%20GraphRAG%20Query%20Engine.md)
- [Microsoft - 2026 - GraphRAG Indexing Methods](../summaries/Microsoft%20-%202026%20-%20GraphRAG%20Indexing%20Methods.md)
- [Docling Project - 2026 - Document Processing Overview](../summaries/Docling%20Project%20-%202026%20-%20Document%20Processing%20Overview.md)

## 关联页面

- [LLM Wiki 文档处理流程](../concepts/LLM%20Wiki%20文档处理流程.md)
- [arXiv 与 Hugging Face 论文发现入口](./arXiv%20与%20Hugging%20Face%20论文发现入口.md)
- [LLM Wiki 方法](../../LLM%20Wiki.md)

---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Wu et al. - 2020 - CorefQA Coreference Resolution as Query-based Span Prediction

## TL;DR（快速导读）

CorefQA 把共指消解改写成基于提及生成问题、在文档里找答案跨度的任务，利用问答式接口寻找指代对象。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

每个候选提及连同周围上下文形成查询，模型预测与它共指的文本跨度。这种方式让指代决策能结合具体提及进行。需要检查候选覆盖、跨度预测与聚类结果，问答接口本身不会消除歧义。

## 具体怎么理解

看到“她”时，可用周围句子作为线索，询问文档里哪个名字与“她”对应。

## 关键事实

- **C1**：将共指检索写成针对候选 mention 的 query-based span prediction。
- **C2**：QA 形式可找回提议阶段遗漏的 mention，但仍保留候选提议模型。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Wu%20et%20al.%20-%202020%20-%20CorefQA%20Coreference%20Resolution%20as%20Query-based%20Span%20Prediction.pdf)
- 全文文本：[打开全文文本](../../raw/text/Wu%20et%20al.%20-%202020%20-%20CorefQA%20Coreference%20Resolution%20as%20Query-based%20Span%20Prediction.md)
- 作者：Wu et al.
- 年份：2020
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Wu%20et%20al.%20-%202020%20-%20CorefQA%20Coreference%20Resolution%20as%20Query-based%20Span%20Prediction.html)

## 争议与不确定点

- 候选、窗口和说话人信息会影响结果。
- 共指指标提高不等于所有实体合并都可信。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无

## 方法与实验解读

CorefQA 用问答式跨度定位寻找相关提及，补充传统候选成对打分。下游仍需要聚成实体簇，检查一致性；底座 SpanBERT 与 QA 数据增强也应纳入性能归因。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Wu%20et%20al.%20-%202020%20-%20CorefQA%20Coreference%20Resolution%20as%20Query-based%20Span%20Prediction.md#source-section-33 ) | 预测同一实体的提及，不是一般开放问答 |
| C2 | [原文]( ../../raw/text/Wu%20et%20al.%20-%202020%20-%20CorefQA%20Coreference%20Resolution%20as%20Query-based%20Span%20Prediction.md#source-section-18 ) | 召回改进不表示取消候选与计算限制 |

## 核证范围

核对查询化任务、§3.3 的候选提议、§3.8–3.9 的 QA 增强和设计边界。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

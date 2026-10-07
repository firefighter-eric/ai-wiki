---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Wu et al. - 2021 - Recursively Summarizing Books with Human Feedback

## TL;DR（快速导读）

这篇整本书摘要工作把长任务递归拆成小摘要，再用人类反馈改善各层结果，研究超长材料怎样逐步压缩。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

人类难以一次检查整本书的摘要。方法先处理较小部分，再将结果组合成更高层摘要，并在过程中收集反馈。分层减少评价负担，也可能逐层传递遗漏和误解，需要保留回看原文的能力。

## 具体怎么理解

先总结每节，再合成章节，最后总结全书；某节遗漏关键事件，后面的层级可能再也看不到它。

## 关键事实

- **C1**：把难以整体监督的长文任务拆成较小部分，再递归组合摘要。
- **C2**：标注者观察到模型可能遗漏故事核心，如人物背景或关键世界设定。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Wu%20et%20al.%20-%202021%20-%20Recursively%20Summarizing%20Books%20with%20Human%20Feedback.pdf)
- 全文文本：[打开全文文本](../../raw/text/Wu%20et%20al.%20-%202021%20-%20Recursively%20Summarizing%20Books%20with%20Human%20Feedback.md)
- 作者：Wu et al.
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Wu%20et%20al.%20-%202021%20-%20Recursively%20Summarizing%20Books%20with%20Human%20Feedback.html)

## 争议与不确定点

- 拆解提升可监督性，但未证明整体事实正确。
- 自动重叠指标与读者对整书理解的评判不同。

## 关联页面

- 主题：[LLM RL](../topics/LLM%20RL.md)
- 综合：暂无
- [OpenAI](../authors/OpenAI.md)：沿作者或机构继续阅读相关来源。

## 方法与实验解读

递归摘要让人类可以审核小片段，再逐层形成整书摘要。其风险是下层遗漏会被上层放大：合并时看不到原文，就难以补回核心信息。本库因此保留来源定位，综合页的关键判断仍回到单篇摘要与原文核对。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Wu%20et%20al.%20-%202021%20-%20Recursively%20Summarizing%20Books%20with%20Human%20Feedback.md#source-section-5 ) | 局部摘要与整体连贯性是两层问题 |
| C2 | [原文]( ../../raw/text/Wu%20et%20al.%20-%202021%20-%20Recursively%20Summarizing%20Books%20with%20Human%20Feedback.md#source-section-71 ) | ROUGE / BERTScore 不能直接替代完整情节核查 |

## 核证范围

核对任务拆解、整书评估设计、BookSum 指标与附录 J.1 的遗漏案例。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

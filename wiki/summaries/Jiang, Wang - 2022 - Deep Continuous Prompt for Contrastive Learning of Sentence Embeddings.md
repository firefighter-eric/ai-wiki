---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Jiang, Wang - 2022 - Deep Continuous Prompt for Contrastive Learning of Sentence Embeddings

## TL;DR（快速导读）

这篇句向量方法冻结语言模型，只训练少量深层连续提示，再用对比学习适配句子表示，降低全参数微调成本。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

句向量对比训练通常需要更新整个模型。本文把可训练部分集中到前缀式连续提示，保持底座参数固定。应关注提示放在哪些层、数据构造和效果成本比较；参数少也不等于任务一定不损失质量。

## 具体怎么理解

可以把连续提示理解为可学习的输入控制量；它不是人在提示框里手写的一句话。

## 关键事实

- **C1**：PromCSE 在 PLM 各层加入可学习 soft prompt，再以对比目标优化句向量。
- **C2**：STS 改善并未转化为所有有监督迁移任务改善，作者在局限中明确说明。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Jiang%2C%20Wang%20-%202022%20-%20Deep%20Continuous%20Prompt%20for%20Contrastive%20Learning%20of%20Sentence%20Embeddings.pdf)
- 全文文本：[打开全文文本](../../raw/text/Jiang%2C%20Wang%20-%202022%20-%20Deep%20Continuous%20Prompt%20for%20Contrastive%20Learning%20of%20Sentence%20Embeddings.md)
- 作者：Jiang, Wang
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Jiang%2C%20Wang%20-%202022%20-%20Deep%20Continuous%20Prompt%20for%20Contrastive%20Learning%20of%20Sentence%20Embeddings.html)

## 争议与不确定点

- 监督版的 hard negative 信息与自监督版不同。
- 领域偏移实验范围有限，不证明任意新领域鲁棒。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无
- [句向量、稠密召回与交互排序](../comparisons/%E5%8F%A5%E5%90%91%E9%87%8F%E3%80%81%E7%A8%A0%E5%AF%86%E5%8F%AC%E5%9B%9E%E4%B8%8E%E4%BA%A4%E4%BA%92%E6%8E%92%E5%BA%8F.md)：把本篇方法放到相关任务与比较条件中阅读。

## 方法与实验解读

PromCSE 用少量可学习前缀改变句表示，同时研究能量视角下的对比目标。对知识库检索而言，STS 和跨域相似度提供线索，但最终仍需查询召回、长文与领域评测。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Jiang%2C%20Wang%20-%202022%20-%20Deep%20Continuous%20Prompt%20for%20Contrastive%20Learning%20of%20Sentence%20Embeddings.md#source-section-9 ) | 连续提示训练与自然语言提示不同 |
| C2 | [原文]( ../../raw/text/Jiang%2C%20Wang%20-%202022%20-%20Deep%20Continuous%20Prompt%20for%20Contrastive%20Learning%20of%20Sentence%20Embeddings.md#source-section-33 ) | 句子相似性收益不等于所有下游收益 |

## 核证范围

核对 §3.1 方法、CxC-STS 结果和 Limitations 的迁移边界。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

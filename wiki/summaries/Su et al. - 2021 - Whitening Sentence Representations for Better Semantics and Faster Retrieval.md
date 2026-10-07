---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Su et al. - 2021 - Whitening Sentence Representations for Better Semantics and Faster Retrieval

## TL;DR（快速导读）

句向量白化通过后处理改善表示空间分布，探索不用复杂新模型就改善语义比较与检索的方式。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

BERT 句向量可能集中在狭窄方向，影响相似度利用。论文用白化变换处理表示，并研究维度与检索效果。后处理参数依赖统计样本，不能保证对所有语料和任务同样有效。

## 具体怎么理解

若大多数句子向量挤在同一方向，余弦相似度难分辨语义；重整空间分布是这类方法的出发点。

## 关键事实

- **C1**：对句向量做 whitening，以改善各向异性，并可截取部分维度压缩表示。
- **C2**：实验分别报告有无 NLI 监督的 STS 结果，低维效果依赖模型与维度选择。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Su%20et%20al.%20-%202021%20-%20Whitening%20Sentence%20Representations%20for%20Better%20Semantics%20and%20Faster%20Retrieval.pdf)
- 全文文本：[打开全文文本](../../raw/text/Su%20et%20al.%20-%202021%20-%20Whitening%20Sentence%20Representations%20for%20Better%20Semantics%20and%20Faster%20Retrieval.md)
- 作者：Su et al.
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Su%20et%20al.%20-%202021%20-%20Whitening%20Sentence%20Representations%20for%20Better%20Semantics%20and%20Faster%20Retrieval.html)

## 争议与不确定点

- 用于估计均值协方差的语料会影响域迁移。
- 维度过低可能丢掉区分关键实体的信号。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无
- [句向量、稠密召回与交互排序](../comparisons/%E5%8F%A5%E5%90%91%E9%87%8F%E3%80%81%E7%A8%A0%E5%AF%86%E5%8F%AC%E5%9B%9E%E4%B8%8E%E4%BA%A4%E4%BA%92%E6%8E%92%E5%BA%8F.md)：把本篇方法放到相关任务与比较条件中阅读。

## 方法与实验解读

whitening 让不同方向的方差更均衡，减少句向量集中在少数方向的现象。降维能减少存储与距离计算，但召回是否保持要用真实查询测试，不能只凭 STS 相关系数判断。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Su%20et%20al.%20-%202021%20-%20Whitening%20Sentence%20Representations%20for%20Better%20Semantics%20and%20Faster%20Retrieval.md#source-section-7 ) | 线性分布校准，不是重新训练语言模型 |
| C2 | [原文]( ../../raw/text/Su%20et%20al.%20-%202021%20-%20Whitening%20Sentence%20Representations%20for%20Better%20Semantics%20and%20Faster%20Retrieval.md#source-section-16 ) | 表示压缩不保证所有任务无损 |

## 核证范围

核对 §3.2–3.3 的变换与降维、无 NLI 结果及 STS 指标定义。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

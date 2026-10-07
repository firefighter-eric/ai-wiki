---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Xu, Choi - 2020 - Revealing the Myth of Higher-Order Inference in Coreference Resolution

## TL;DR（快速导读）

这篇共指研究重新检验高阶推理的收益，比较多种方法，提醒复杂推理模块是否有效需要受控实验支持。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

一些系统会根据已预测的指代关系继续更新表示。论文分析这些步骤相对表示学习的实际贡献。阅读重点是不同模块与训练设置的对照，不能因为方法看起来更复杂就推定更好。

## 具体怎么理解

若加一轮关系传播提高了得分，还需排除网络容量、训练时长或其他配置同时改变的影响。

## 关键事实

- **C1**：在 SpanBERT 底座下，多种高阶推断没有清晰优势，部分方法有负影响。
- **C2**：cluster merging 结合先行词和簇信息，区别于只精炼 span 表示。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Xu%2C%20Choi%20-%202020%20-%20Revealing%20the%20Myth%20of%20Higher-Order%20Inference%20in%20Coreference%20Resolution.pdf)
- 全文文本：[打开全文文本](../../raw/text/Xu%2C%20Choi%20-%202020%20-%20Revealing%20the%20Myth%20of%20Higher-Order%20Inference%20in%20Coreference%20Resolution.md)
- 作者：Xu, Choi
- 年份：2020
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Xu%2C%20Choi%20-%202020%20-%20Revealing%20the%20Myth%20of%20Higher-Order%20Inference%20in%20Coreference%20Resolution.html)

## 争议与不确定点

- CoNLL-2012 的发现不证明所有数据上 HOI 无用。
- 最优方法与平均趋势要分别报告。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无

## 方法与实验解读

这篇研究要求把编码器升级与推断机制增益分开。一个模块在旧底座上有效，换成更强表示后可能冗余；因此是否保留高阶机制要看同底座消融，而不是沿用领域惯例。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Xu%2C%20Choi%20-%202020%20-%20Revealing%20the%20Myth%20of%20Higher-Order%20Inference%20in%20Coreference%20Resolution.md#source-section-14 ) | 与较弱编码器上的旧结果不矛盾，条件已改变 |
| C2 | [原文]( ../../raw/text/Xu%2C%20Choi%20-%202020%20-%20Revealing%20the%20Myth%20of%20Higher-Order%20Inference%20in%20Coreference%20Resolution.md#source-section-12 ) | 不同 HOI 方法不是同一个操作 |

## 核证范围

核对 cluster merging 定义、§4.1 同底座比较及结论。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Scao et al. - 2022 - BLOOM A 176B-Parameter Open-Access Multilingual Language Model

## TL;DR（快速导读）

BLOOM 由国际协作建设多语言大模型，阅读重点包括语言覆盖、数据、公开材料与发布条件。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

若要研究多语言表现，先确认目标语言的数据与评测，再比较对应任务；不能仅凭多语言标签推断所有语言同样强。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Scao et al. - 2022 - BLOOM A 176B-Parameter Open-Access Multilingual Language Model.pdf
- 全文文本：../../raw/text/Scao et al. - 2022 - BLOOM A 176B-Parameter Open-Access Multilingual Language Model.md
- 作者：Scao et al.
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

这份报告不仅介绍模型规模，也讨论协作训练和治理方式。开放访问、多语言覆盖和各语言实际能力需要分别核对；广泛参与不能自动替代数据、评测与许可检查。

## 关键事实

- **C1**：BLOOM为176B自回归多语言模型，由BigScience协作开发。
- **C2**：采用causal decoder-only与ALiBi，架构选择依赖零样本任务设计。
- **C3**：多语言数据选择由语言社区参与，包含治理、过滤、去重和隐私处理。
- **C4**：模型RAIL含行为限制，代码Apache2.0，公开获取不等于二者授权完全相同。

## 争议与不确定点

- zero-shot与multitask-finetuned结果不能混合为基座分数。
- 数据与评测偏差仍存在，模型开放不意味着自动适合高风险用途。

## 关联页面

- 概念：[BLOOM](../../wiki/concepts/BLOOM.md)
- 主题：[LLM 预训练](../../wiki/topics/LLM%20预训练.md)
- 比较：[开放模型家族与中国重要家族对照](../../wiki/comparisons/开放模型家族与中国重要家族对照.md)

## 方法与实验解读

报告把模型架构、ROOTS数据、tokenizer、训练工程与治理放在一条可检查链里。开放权重扩大研究入口，公开数据处理和限制说明使读者能解释风险；多语言覆盖还需逐语言质量，不能以176B规模替代。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Scao%20et%20al.%20-%202022%20-%20BLOOM%20A%20176B-Parameter%20Open-Access%20Multilingual%20Language%20Model.md#source-section-87 ) | model与协作治理均为本文主题。 |
| C2 | [原文]( ../../raw/text/Scao%20et%20al.%20-%202022%20-%20BLOOM%20A%20176B-Parameter%20Open-Access%20Multilingual%20Language%20Model.md#source-section-33 ) | 不证明decoder在所有transfer任务最好。 |
| C3 | [原文]( ../../raw/text/Scao%20et%20al.%20-%202022%20-%20BLOOM%20A%20176B-Parameter%20Open-Access%20Multilingual%20Language%20Model.md#source-section-20 ) | 流程不表示语料全部无问题。 |
| C4 | [原文]( ../../raw/text/Scao%20et%20al.%20-%202022%20-%20BLOOM%20A%20176B-Parameter%20Open-Access%20Multilingual%20Language%20Model.md#source-section-57 ) | 源码和权重许可分别看。 |

## 核证范围

核读架构选择/ALiBi、数据语言治理、许可、评测设计与总结。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Bommasani et al. - 2021 - On the Opportunities and Risks of Foundation Models

## TL;DR（快速导读）

这份报告把基础模型看作可适配多种任务的共同底座，讨论其技术机会与社会风险；它适合建立问题地图。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

基础模型先在广泛数据上训练，再被用于不同应用。报告从训练、适配和应用展开讨论，也关注同一种底层模型被大量系统复用后，偏差、失效和治理问题怎样扩散。它提供跨领域分析框架，具体应用是否有效仍需领域证据。

## 具体怎么理解

如果很多产品共享一个底层模型，底座的一项偏差可能同时影响多个产品；应用层的好表现也不能说明底座所有能力都可靠。

## 关键事实

- **C1**：foundation model 指在广泛数据上大规模训练、可适配多类下游任务的模型。
- **C2**：报告用 emergence 与 homogenization 描述范式变化，并指出共享模型可能形成共同失败点。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Bommasani%20et%20al.%20-%202021%20-%20On%20the%20Opportunities%20and%20Risks%20of%20Foundation%20Models.pdf)
- 全文文本：[打开全文文本](../../raw/text/Bommasani%20et%20al.%20-%202021%20-%20On%20the%20Opportunities%20and%20Risks%20of%20Foundation%20Models.md)
- 作者：Bommasani et al.
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Bommasani%20et%20al.%20-%202021%20-%20On%20the%20Opportunities%20and%20Risks%20of%20Foundation%20Models.html)

## 争议与不确定点

- 2021 年的论述和例子需要按历史背景理解。
- 报告汇总的法律、医疗等应用可能性不能当成可直接部署的专业结论。

## 关联页面

- 主题：[传统NLP](../topics/传统%20NLP.md)
- 综合：暂无
- [AI 能力评测：任务、过程与预测](../comparisons/AI%20%E8%83%BD%E5%8A%9B%E8%AF%84%E6%B5%8B%EF%BC%9A%E4%BB%BB%E5%8A%A1%E3%80%81%E8%BF%87%E7%A8%8B%E4%B8%8E%E9%A2%84%E6%B5%8B.md)：把本篇方法放到相关任务与比较条件中阅读。

## 方法与实验解读

报告把技术、应用与社会影响放入同一框架。广泛复用一个底座降低适配门槛，也让数据缺陷和偏差影响多个应用；评价具体系统时必须继续检查底座、适配方式和部署场景。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/pdf/Bommasani%20et%20al.%20-%202021%20-%20On%20the%20Opportunities%20and%20Risks%20of%20Foundation%20Models.pdf#page=3 ) | 概念定义，不等于已具备所有任务能力 |
| C2 | [原文]( ../../raw/pdf/Bommasani%20et%20al.%20-%202021%20-%20On%20the%20Opportunities%20and%20Risks%20of%20Foundation%20Models.pdf#page=3 ) | 机会与风险分析，不是单项实验因果结论 |

## 核证范围

核对 PDF 第 3–6 页的定义、emergence / homogenization 与范式边界，限定概念框架摘要。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

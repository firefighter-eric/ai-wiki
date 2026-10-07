---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Jiang et al. - 2023 - Mistral 7B

## TL;DR（快速导读）

Mistral 7B 将查询分组共享与滑动窗口注意力结合，用较紧凑的模型研究语言能力和推理效率。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

比较它与更大模型时，应确认任务、上下文长度和推理配置；参数少不自动代表任何设备上都更快。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Jiang et al. - 2023 - Mistral 7B.pdf
- 全文文本：../../raw/text/Jiang et al. - 2023 - Mistral 7B.md
- 作者：Jiang et al.
- 年份：2023
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

GQA 减少键值缓存，滑动窗口限制部分位置连接，两者解决的成本不同。报告中的模型比较应结合数据、任务和部署条件阅读，不能仅按参数规模判断替代关系。

## 关键事实

- **C1**：Mistral7B使用GQA降低KV成本，并使用SWA限制单层可访问位置。
- **C2**：rolling cache与prefill chunking控制长输入的缓存和计算。
- **C3**：作者报告在所测基准上超过Llama2-13B，并另发Instruct版。
- **C4**：模型以Apache2.0发布。

## 争议与不确定点

- 窗口外信息需经过层传播，极长距离检索仍要任务评测。
- 报告不公开全部数据细节，知识压缩解释是作者观点而非完整归因实验。

## 关联页面

- 概念：[Mistral 7B](../../wiki/concepts/Mistral%207B.md)
- 概念：[Mixtral](../../wiki/concepts/Mixtral.md)
- 主题：[LLM 预训练](../../wiki/topics/LLM%20预训练.md)
- 比较：[开放模型家族与中国重要家族对照](../../wiki/comparisons/开放模型家族与中国重要家族对照.md)

## 方法与实验解读

模型将缓存结构和局部信息路径一起优化。GQA改变K/V共享，SWA改变token间交互范围，二者可叠加但不是同一种节省。较小模型在报告基准有竞争力说明训练配方的重要性，而不是证明任意任务都能压缩到同样规模。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Jiang%20et%20al.%20-%202023%20-%20Mistral%207B.md#source-section-4 ) | 长距离影响通过多层传播，不是每层完整全局注意力。 |
| C2 | [原文]( ../../raw/text/Jiang%20et%20al.%20-%202023%20-%20Mistral%207B.md#source-section-4 ) | cache策略不等于无限可靠的长文记忆。 |
| C3 | [原文]( ../../raw/text/Jiang%20et%20al.%20-%202023%20-%20Mistral%207B.md#source-section-5 ) | base与chat及所测任务分开。 |
| C4 | [原文]( ../../raw/text/Jiang%20et%20al.%20-%202023%20-%20Mistral%207B.md#source-section-2 ) | 许可证事实与性能主张分开。 |

## 核证范围

核读architectural details、cache/prefill和base/instruct结果。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Unknown - 2024 - DeepSeek-V3 Technical Report

## TL;DR（快速导读）

DeepSeek-V3 的 MLA 将键和值压缩到较小的联合表示，降低生成时缓存的内存与读写压力。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

先训练通用语言底座，再做推理或指令适配，是不同阶段；模型名称不能代替训练阶段的说明。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Unknown - 2024 - DeepSeek-V3 Technical Report.pdf
- 全文文本：../../raw/text/Unknown - 2024 - DeepSeek-V3 Technical Report.md
- 作者：Unknown
- 年份：2024
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

该摘要主要沿注意力问题阅读技术报告，解释低秩缓存压缩。它与减少注意力连接的路线不同；完整模型还包含其他设计，不能把整体评测提升全部归于 MLA。

## 关键事实

- **C1**：V3为671B/37B MoE，预训练14.8Ttokens；AdamW beta0.9/0.95/decay0.1。
- **C2**：优化器moments以BF16、masterweights与累积gradientsFP32。
- **C3**：MLA低秩联合压缩KV，decode缓存latent与位置相关部分。
- **C4**：FP8、DualPipe、路由负载和MTP一起组成系统配方。

## 争议与不确定点

- 2.788MH800hours不等于完整研发成本。
- 服务部署单元大、小batch效率与多节点成本是明确局限。

## 关联页面

- 主题：[注意力机制 Attention](../../wiki/topics/注意力机制%20Attention.md)
- 主题：[LLM 预训练](../../wiki/topics/LLM%20预训练.md)
- 概念：[DeepSeek](../../wiki/concepts/DeepSeek.md)
- 对比：[Muon 与 AdamW](../../wiki/comparisons/Muon%20与%20AdamW.md)
- [DeepSeek](../authors/DeepSeek.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **KV cache**：键值缓存：保存已经处理过的位置表示，生成新内容时可复用，避免全部重算。
- **latent**：潜表示：原始数据经过模型编码后的内部表示，通常更紧凑。

## 方法与实验解读

MLA减KV状态，MoE改激活计算，FP8减GEMM与存储成本，DualPipe改通信/计算重叠。它们的成本口径不同；与Muon报告比较先看明确披露的optimizer及parameter groups，不能由相近架构推测未披露配置。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md#source-section-51 ) | 4K预训练长度，longcontext扩展另阶段。 |
| C2 | [原文]( ../../raw/text/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md#source-section-34 ) | 低精度方案不能写所有参数都是FP8。 |
| C3 | [原文]( ../../raw/text/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md#source-section-6 ) | 仍执行attention，不将序列复杂度变成线性。 |
| C4 | [原文]( ../../raw/text/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md#source-section-85 ) | 成本为披露训练核算，不含全部研究/数据成本。 |

## 核证范围

核读MLA、低精度state、训练超参、基础设施与deployment限制。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

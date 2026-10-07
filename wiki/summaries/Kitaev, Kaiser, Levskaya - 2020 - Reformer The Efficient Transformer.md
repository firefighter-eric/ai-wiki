---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Kitaev, Kaiser, Levskaya - 2020 - Reformer The Efficient Transformer

## TL;DR（快速导读）

Reformer 用哈希分桶筛选相似位置，并用可逆层节省训练存储，研究长序列的低成本处理。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

先按表示相近程度组织候选，再计算部分连接；这种筛选会有自己的假设与近似条件。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Kitaev, Kaiser, Levskaya - 2020 - Reformer The Efficient Transformer.pdf
- 原始 HTML：../../raw/html/Kitaev, Kaiser, Levskaya - 2020 - Reformer The Efficient Transformer.html
- 全文文本：../../raw/text/Kitaev, Kaiser, Levskaya - 2020 - Reformer The Efficient Transformer.md
- 作者：Kitaev, Kaiser, Levskaya
- 年份：2020
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

局部敏感哈希使注意力主要在相关桶内计算，可逆层减少保存中间激活的需求。这两项分别影响连接选择与存储；分桶误差、任务质量和实际速度仍需按具体配置核对。

## 关键事实

- **C1**：LSH attention以内容哈希组织候选，把序列attention成本目标从O(L²)降到O(LlogL)。
- **C2**：Reformer还用reversible residual减少跨层保存激活，并分块执行FFN。
- **C3**：LSH配置共享Q/K表示，哈希次数影响检索覆盖和质量。
- **C4**：合成任务显示增加评测hash数可改善准确率；稠密模型直接换LSH有精度损失。

## 争议与不确定点

- bucket碰撞与漏检可能影响长依赖。
- 理论复杂度不保证哈希、排序与gather在任意设备上快。

## 关联页面

- 主题：[注意力机制 Attention](../../wiki/topics/注意力机制%20Attention.md)
- 概念：[Transformer](../../wiki/concepts/Transformer.md)

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。

## 方法与实验解读

内容相近的token倾向落在同bucket，以候选稀疏化代替全对全比较。可逆层在反向重建输入，另减少激活存储。三种策略的作用要拆开：哈希影响信息访问，重建/分块主要影响运行资源；改变一个不等于复用全部收益。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Kitaev%2C%20Kaiser%2C%20Levskaya%20-%202020%20-%20Reformer%20The%20Efficient%20Transformer.md#source-section-2 ) | 哈希数、bucket大小和实现参与常数成本。 |
| C2 | [原文]( ../../raw/text/Kitaev%2C%20Kaiser%2C%20Levskaya%20-%202020%20-%20Reformer%20The%20Efficient%20Transformer.md#source-section-3 ) | attention近似与训练内存策略是不同部分。 |
| C3 | [原文]( ../../raw/text/Kitaev%2C%20Kaiser%2C%20Levskaya%20-%202020%20-%20Reformer%20The%20Efficient%20Transformer.md#source-section-8 ) | 不能当无误差替换。 |
| C4 | [原文]( ../../raw/text/Kitaev%2C%20Kaiser%2C%20Levskaya%20-%202020%20-%20Reformer%20The%20Efficient%20Transformer.md#source-section-14 ) | 受控任务证据，不是任意LLM免训迁移。 |

## 核证范围

核读LSH、sharedQK、可逆/分块与合成任务hash消融。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

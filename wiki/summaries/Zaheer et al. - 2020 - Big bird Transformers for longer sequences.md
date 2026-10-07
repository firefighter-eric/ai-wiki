---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Zaheer et al. - 2020 - Big bird Transformers for longer sequences

## TL;DR（快速导读）

BigBird 混合局部、全局和随机连接，用较少注意力关系处理长序列，并分析表达能力。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

减少每个位置的连接能省计算，但哪些远处信息仍可传播，需要由连接设计与任务评估。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Zaheer et al. - 2020 - Big bird Transformers for longer sequences.pdf
- 原始 HTML：../../raw/html/Zaheer et al. - 2020 - Big bird Transformers for longer sequences.html
- 全文文本：../../raw/text/Zaheer et al. - 2020 - Big bird Transformers for longer sequences.md
- 作者：Zaheer et al.
- 年份：2020
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

不同连接承担附近交流、关键位置聚合和远距离传播等作用。它主动改变连接图；理论性质有前提，实际文档任务仍要核对远处信息是否能有效传递和使用。

## 关键事实

- **C1**：BigBird结合局部窗口、随机连接和少量全局token；itc将已有token设为全局，etc额外添加全局token。
- **C2**：通用逼近证明要求注意力图包含星形结构；Turing completeness证明使用任意精度等理想化条件。
- **C3**：论文给出稀疏注意力需更多层才能解决的最远向量任务，否定稀疏与全注意力在所有任务上等价。
- **C4**：NLP实验使用4096长度；QA按数据集开发集选配置，摘要仅encoder改为稀疏、decoder仍全注意力。

## 争议与不确定点

- 通用逼近和Turing completeness不能证明固定层数、有限精度与优化过程等价。
- 摘要结果来自稀疏encoder与全attention decoder，不是完全稀疏的自回归LLM。
- 复杂度下降需要固定稀疏规模和有效硬件实现，短序列不保证更快。

## 关联页面

- 主题：[注意力机制 Attention](../../wiki/topics/注意力机制%20Attention.md)
- 概念：[Transformer](../../wiki/concepts/Transformer.md)

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **sparse**：稀疏计算或连接：只使用选中的部分，具体省略什么取决于方法。

## 方法与实验解读

这条路线以可控的信息传播图替代全连接计算。局部边保留邻域关系，随机边缩短传播距离，全局token收集跨文档信息；它改变了单层能看到的交互。长文QA和摘要的收益要同时看输入长度、预训练初始化与任务调参，不能用理论表达能力直接推导实际生成质量。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Zaheer%20et%20al.%20-%202020%20-%20Big%20bird%20Transformers%20for%20longer%20sequences.md#source-section-5 ) | 窗口、随机与全局数量不随n线性增长时，边数与n成线性关系。 |
| C2 | [原文]( ../../raw/text/Zaheer%20et%20al.%20-%202020%20-%20Big%20bird%20Transformers%20for%20longer%20sequences.md#source-section-10 )、[原文]( ../../raw/text/Zaheer%20et%20al.%20-%202020%20-%20Big%20bird%20Transformers%20for%20longer%20sequences.md#source-section-11 ) | 表达能力存在性不等于有限深度或实际浮点模型保证。 |
| C3 | [原文]( ../../raw/text/Zaheer%20et%20al.%20-%202020%20-%20Big%20bird%20Transformers%20for%20longer%20sequences.md#source-section-12 ) | 结论依赖Orthogonal Vector Conjecture等复杂性假设。 |
| C4 | [原文]( ../../raw/text/Zaheer%20et%20al.%20-%202020%20-%20Big%20bird%20Transformers%20for%20longer%20sequences.md#source-section-14 )、[原文]( ../../raw/text/Zaheer%20et%20al.%20-%202020%20-%20Big%20bird%20Transformers%20for%20longer%20sequences.md#source-section-16 )、[原文]( ../../raw/text/Zaheer%20et%20al.%20-%202020%20-%20Big%20bird%20Transformers%20for%20longer%20sequences.md#source-section-18 ) | 任务、上下文和初始化不同，不能把全部收益归因于稀疏模式。 |

## 核证范围

核读架构、星形通用逼近、Turing精度条件、稀疏下界及NLP/摘要实验设置。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Jiang et al. - 2024 - Mixtral of Experts

## TL;DR（快速导读）

Mixtral 为每个输入选择部分专家，扩大总容量，同时控制每次实际参与计算的规模。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

保存全部专家仍需要内存，多卡部署还可能产生通信；只看激活参数量会漏掉这些成本。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Jiang et al. - 2024 - Mixtral of Experts.pdf
- 全文文本：../../raw/text/Jiang et al. - 2024 - Mixtral of Experts.md
- 作者：Jiang et al.
- 年份：2024
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

专家混合把总权重数量与激活参数数量分开。实际效率还取决于路由、负载和跨设备通信；阅读报告时，需同时检查任务质量、权重内存和单次推理成本。

## 关键事实

- **C1**：Mixtral8x7B每层有8个FFN专家，每token选2个并加权合并。
- **C2**：总参数约47B、每token激活约13B，上下文训练32K。
- **C3**：基准比较由作者用同一评测pipeline重跑。
- **C4**：作者明确内存随总参数而非激活参数计，硬件利用率另影响速度。

## 争议与不确定点

- 不同专家的负载和跨设备通信可能降低稀疏收益。
- passkey合成检索不覆盖真实长文推理的全部条件。

## 关联页面

- 概念：[Mixtral](../../wiki/concepts/Mixtral.md)
- 概念：[Mistral 7B](../../wiki/concepts/Mistral%207B.md)
- 概念：[MoE](../../wiki/concepts/MoE.md)
- 主题：[LLM 预训练](../../wiki/topics/LLM%20预训练.md)

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。
- **sparse**：稀疏计算或连接：只使用选中的部分，具体省略什么取决于方法。

## 方法与实验解读

路由让不同token选择不同容量，在控制计算的同时增加总知识存储。部署必须保存全部专家，或设计offload与并行；因此激活参数适合比较算术，却不能单独估计显存和吞吐。报告只支持8x7B这一版，后续更大Mixtral版本需各自来源。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Jiang%20et%20al.%20-%202024%20-%20Mixtral%20of%20Experts.md#source-section-5 ) | 稀疏性位于FFN，attention并非因此稀疏。 |
| C2 | [原文]( ../../raw/text/Jiang%20et%20al.%20-%202024%20-%20Mixtral%20of%20Experts.md#source-section-2 ) | 8x7B名称不能简单当56B独立完整模型。 |
| C3 | [原文]( ../../raw/text/Jiang%20et%20al.%20-%202024%20-%20Mixtral%20of%20Experts.md#source-section-6 ) | 任务版本和prompt条件仍需保存。 |
| C4 | [原文]( ../../raw/text/Jiang%20et%20al.%20-%202024%20-%20Mixtral%20of%20Experts.md#source-section-6 ) | 少算不等于只需装入13B权重。 |

## 核证范围

核读专家层、参数分母、统一评测、长文与成本说明。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

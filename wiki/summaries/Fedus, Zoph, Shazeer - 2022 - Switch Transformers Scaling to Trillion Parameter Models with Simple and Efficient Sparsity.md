---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Fedus, Zoph, Shazeer - 2022 - Switch Transformers Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

## TL;DR（快速导读）

Switch Transformer 让每个输入只路由到少数专家，扩大模型总容量，同时控制每个词元实际参与的计算。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

普通网络对所有输入使用同一组参数；专家混合模型按输入选择不同子网络。Switch 简化路由，并研究通信和训练稳定性。总参数很大不表示每个输入都计算全部参数，也不表示系统通信与显存成本随之消失。

## 具体怎么理解

可以把专家理解成不同的处理分支：路由器选一条来处理当前输入，但所有专家的权重仍需由系统保存和组织。

## 关键事实

- **C1**：Switch 层为每个 token 路由至单个专家，仍通过门控概率加权输出。
- **C2**：专家容量固定，负载不均会使部分 token 在该专家层不被处理；增大容量也增加计算和通信。
- **C3**：64 专家 Switch-Base 在 32 TPUv3、相同每样本 FLOPs 下，以约七分之一时间达到 T5-Base 相近困惑度。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Fedus%2C%20Zoph%2C%20Shazeer%20-%202022%20-%20Switch%20Transformers%20Scaling%20to%20Trillion%20Parameter%20Models%20with%20Simple%20and%20Efficient%20Sparsity.pdf)
- 全文文本：[打开全文文本](../../raw/text/Fedus%2C%20Zoph%2C%20Shazeer%20-%202022%20-%20Switch%20Transformers%20Scaling%20to%20Trillion%20Parameter%20Models%20with%20Simple%20and%20Efficient%20Sparsity.md)
- 作者：Fedus, Zoph, Shazeer
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Fedus%2C%20Zoph%2C%20Shazeer%20-%202022%20-%20Switch%20Transformers%20Scaling%20to%20Trillion%20Parameter%20Models%20with%20Simple%20and%20Efficient%20Sparsity.html)

## 争议与不确定点

- 训练步数效率不直接等于墙钟效率，论文专门分别评测两者。
- 更多总参数需要存储与通信支持；微调稳定性和专家负载仍是工程条件。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无

## 方法与实验解读

Switch 把 MoE 路由简化为 top-1，允许总参数量增加而每个 token 的激活计算保持较少。节省计算后，瓶颈可能转向路由、跨设备通信和容量填充，因此模型大小、激活计算和墙钟时间必须分别记录。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Fedus%2C%20Zoph%2C%20Shazeer%20-%202022%20-%20Switch%20Transformers%20Scaling%20to%20Trillion%20Parameter%20Models%20with%20Simple%20and%20Efficient%20Sparsity.md#source-section-5 ) | 稀疏的是专家前馈计算，不是全部参数或注意力 |
| C2 | [原文]( ../../raw/text/Fedus%2C%20Zoph%2C%20Shazeer%20-%202022%20-%20Switch%20Transformers%20Scaling%20to%20Trillion%20Parameter%20Models%20with%20Simple%20and%20Efficient%20Sparsity.md#source-section-6 ) | 容量、负载均衡与设备划分共同影响效率 |
| C3 | [原文]( ../../raw/text/Fedus%2C%20Zoph%2C%20Shazeer%20-%202022%20-%20Switch%20Transformers%20Scaling%20to%20Trillion%20Parameter%20Models%20with%20Simple%20and%20Efficient%20Sparsity.md#source-section-11 ) | 论文的预训练困惑度与硬件设置，非任意下游任务的七倍加速 |

## 核证范围

核对 §2.1–2.4、§3.1–3.2 与 §8，支持 top-1 路由、容量限制和特定预训练加速结论。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

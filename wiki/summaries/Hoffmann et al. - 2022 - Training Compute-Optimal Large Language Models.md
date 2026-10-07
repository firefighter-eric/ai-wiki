---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Hoffmann et al. - 2022 - Training Compute-Optimal Large Language Models

## TL;DR（快速导读）

Chinchilla 研究固定训练预算下参数量和数据量的搭配，发现一味增大模型而不给足训练数据会浪费计算。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

作者用一系列不同规模和训练量的实验拟合计算最优配置，再训练 Chinchilla 检验推断。核心问题是模型应有多大、应看多少数据。结论有实验范围与成本目标，不能把某个比例当成所有训练和部署场景的固定公式。

## 具体怎么理解

同一笔训练预算，可以训练更大的模型较少步，也可以训练较小模型更多步；论文研究这两个选择如何平衡。

## 关键事实

- **C1**：研究在固定训练计算量下分配模型参数与训练 token，通过固定模型扫描、IsoFLOP 曲线和参数化损失拟合三种方法估计最优分配。
- **C2**：在约 FLOPs=6ND 的约束下，参数拟合给出 N 与 D 随计算量增长的指数约 0.46 与 0.54，支持两者近似同步扩展。
- **C3**：Chinchilla 用 70B 参数和 1.4T tokens 训练，在与 Gopher 相同训练 FLOPs 下优于它的多数测量任务。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Hoffmann%20et%20al.%20-%202022%20-%20Training%20Compute-Optimal%20Large%20Language%20Models.pdf)
- 全文文本：[打开全文文本](../../raw/text/Hoffmann%20et%20al.%20-%202022%20-%20Training%20Compute-Optimal%20Large%20Language%20Models.md)
- 作者：Hoffmann et al.
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Hoffmann%20et%20al.%20-%202022%20-%20Training%20Compute-Optimal%20Large%20Language%20Models.html)

## 争议与不确定点

- 原扫描主要少于一遍数据；重复多轮训练、数据质量变化和其他模态需要重新验证。
- 高预算处出现曲率，大规模只有两次可比训练，幂律外推有误差。
- 计算最优与用户长期推理成本最优是不同目标。

## 关联页面

- 主题：[LLM预训练](../topics/LLM%20预训练.md)
- 综合：暂无
- [DeepMind](../authors/DeepMind.md)：沿作者或机构继续阅读相关来源。

## 方法与实验解读

固定预算好比同一笔训练费用：可以造更大的模型，也可以让较小模型多读高质量数据。论文发现当时不少大模型读得不够，调整两项分配比单纯增大参数更有效。部署时若要降低大量未来推理成本，最优选择还需要把推理费用加入目标，不能机械照抄训练预算最优点。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Hoffmann%20et%20al.%20-%202022%20-%20Training%20Compute-Optimal%20Large%20Language%20Models.md#source-section-10 ) | 超过 400 次训练的经验拟合；目标是训练计算最优。 |
| C2 | [原文]( ../../raw/text/Hoffmann%20et%20al.%20-%202022%20-%20Training%20Compute-Optimal%20Large%20Language%20Models.md#source-section-14 ) | 不是在所有数据、架构和预算上都精确等于 0.5，也不是固定 token/参数比的自然定律。 |
| C3 | [原文]( ../../raw/text/Hoffmann%20et%20al.%20-%202022%20-%20Training%20Compute-Optimal%20Large%20Language%20Models.md#source-section-16 ) | 验证模型处在预测 40–70B 区间的较大端；大规模直接对照主要只有 Chinchilla 与 Gopher。 |

## 核证范围

核对 §3 三种估计方法与有效前沿、§4 Chinchilla 实际预算、§5 局限和单遍数据条件。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

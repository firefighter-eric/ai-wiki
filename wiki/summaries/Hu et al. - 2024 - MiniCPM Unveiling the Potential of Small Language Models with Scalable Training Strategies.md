---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Hu et al. - 2024 - MiniCPM Unveiling the Potential of Small Language Models with Scalable Training Strategies

## TL;DR（快速导读）

MiniCPM 研究如何把小语言模型训练得更充分，用规模实验和学习率安排提高有限参数预算下的能力。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

小模型部署成本较低，但训练配方仍影响其上限。报告通过模型缩放实验选择训练配置，并引入预热、稳定、衰减三个阶段的学习率计划，支持持续训练与数据扩展。作者的能力比较依赖具体任务和模型版本。

## 具体怎么理解

例如先稳定训练，再在收尾阶段降低学习率；继续加入数据时，训练计划如何衔接会影响结果。

## 关键事实

- **C1**：以小模型进行 Model Wind Tunnel Experiments，分析超参数、batch 与学习率。
- **C2**：WSD 将训练拆成 warmup、stable、decay 三段，方便从稳定阶段继续扩展训练。
- **C3**：作者承认未实际训练更大 LLM 验证该扩展规律。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Hu%20et%20al.%20-%202024%20-%20MiniCPM%20Unveiling%20the%20Potential%20of%20Small%20Language%20Models%20with%20Scalable%20Training%20Strategies.pdf)
- 全文文本：[打开全文文本](../../raw/text/Hu%20et%20al.%20-%202024%20-%20MiniCPM%20Unveiling%20the%20Potential%20of%20Small%20Language%20Models%20with%20Scalable%20Training%20Strategies.md)
- 作者：Hu et al.
- 年份：2024
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Hu%20et%20al.%20-%202024%20-%20MiniCPM%20Unveiling%20the%20Potential%20of%20Small%20Language%20Models%20with%20Scalable%20Training%20Strategies.html)

## 争议与不确定点

- 部分任务小模型仍落后，平均分接近不等于所有能力相同。
- 端侧部署还受内存、量化和算子支持约束。

## 关联页面

- 主题：[LLM RL](../topics/LLM%20RL.md)
- 综合：暂无
- [MiniCPM - ModelBest](../authors/MiniCPM%20-%20ModelBest.md)：沿作者或机构继续阅读相关来源。

## 方法与实验解读

MiniCPM 用便宜的小模型实验探索训练策略，再训练 1.2B 和 2.4B 非嵌入参数模型。WSD 的价值在于让继续训练和衰减阶段更可控；规模外推仍需新的实验证据。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Hu%20et%20al.%20-%202024%20-%20MiniCPM%20Unveiling%20the%20Potential%20of%20Small%20Language%20Models%20with%20Scalable%20Training%20Strategies.md#source-section-5 ) | 小尺度实验为配方选择提供线索 |
| C2 | [原文]( ../../raw/text/Hu%20et%20al.%20-%202024%20-%20MiniCPM%20Unveiling%20the%20Potential%20of%20Small%20Language%20Models%20with%20Scalable%20Training%20Strategies.md#source-section-11 ) | 阶段长度和学习率仍是具体训练设置 |
| C3 | [原文]( ../../raw/text/Hu%20et%20al.%20-%202024%20-%20MiniCPM%20Unveiling%20the%20Potential%20of%20Small%20Language%20Models%20with%20Scalable%20Training%20Strategies.md#source-section-28 ) | 不能把小模型规律视为已验证的大模型规律 |

## 核证范围

核对 §3、§4.2、§6.5 与 Limitations，限定训练策略及小尺度证据。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Alabdulmohsin et al. - 2023 - Getting ViT in Shape Scaling Laws for Compute-Optimal Model Design

## TL;DR（快速导读）

SoViT 研究同样的训练预算该怎样分配给 ViT 的宽度和深度；模型的形状也影响效率，参数总量不能解释一切。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

已有规模化研究常先确定模型应有多少参数。本文进一步估计宽度、深度等结构配置，在视觉 Transformer 上寻找更适合给定计算预算的形状。作者报告形状优化后的模型能与更大模型竞争；需要同时比较数据、训练计算和评测任务。

## 具体怎么理解

例如两个模型参数量接近，一个更宽、一个更深，它们的训练速度和最终表现仍可能不同。

## 关键事实

- **C1**：研究计算预算下的宽度与深度形状，而不只优化参数总数。
- **C2**：优化后的 SoViT 在部分任务接近更大模型，但稠密分割显示该形状存在局限。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Alabdulmohsin%20et%20al.%20-%202023%20-%20Getting%20ViT%20in%20Shape%20Scaling%20Laws%20for%20Compute-Optimal%20Model%20Design.pdf)
- 全文文本：[打开全文文本](../../raw/text/Alabdulmohsin%20et%20al.%20-%202023%20-%20Getting%20ViT%20in%20Shape%20Scaling%20Laws%20for%20Compute-Optimal%20Model%20Design.md)
- 作者：Alabdulmohsin et al.
- 年份：2023
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Alabdulmohsin%20et%20al.%20-%202023%20-%20Getting%20ViT%20in%20Shape%20Scaling%20Laws%20for%20Compute-Optimal%20Model%20Design.html)

## 争议与不确定点

- 形状规律依赖模型家族和训练范围。
- 微调、线性探测与零样本迁移不是同一效果测量。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无

## 方法与实验解读

这篇工作将 scaling 从模型多大推进到模型长什么样。宽、深及训练数据共同影响资源分配；所谓 compute-optimal 必须连同优化的任务、预算和损失定义理解。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Alabdulmohsin%20et%20al.%20-%202023%20-%20Getting%20ViT%20in%20Shape%20Scaling%20Laws%20for%20Compute-Optimal%20Model%20Design.md#source-section-2 ) | 同等参数量可对应不同计算和能力 |
| C2 | [原文]( ../../raw/text/Alabdulmohsin%20et%20al.%20-%202023%20-%20Getting%20ViT%20in%20Shape%20Scaling%20Laws%20for%20Compute-Optimal%20Model%20Design.md#source-section-13 ) | 分类目标下的最优形状不必然适合分割 |

## 核证范围

核对优化目标、§5 的多任务验证和 §5.4 的分割局限。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

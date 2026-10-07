---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Li, Fan, Ai - Unknown - Scaling Language-Image Pre-training via Masking

## TL;DR（快速导读）

FLIP 在训练图文模型时移除大量图块，把节省的计算用于更多样本，研究精度与训练成本的折中。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

完整处理所有图块会消耗视觉计算。FLIP 通过随机遮挡减少每张图参与训练的部分，使相同时间或显存下能处理更多图文对。要同时看遮挡比例、训练样本量和最终评测，不能只比较单次计算量。

## 具体怎么理解

如果一张图片只处理部分图块，同样预算可用于更多图片；代价是每次观察的信息更少。

## 关键事实

- **C1**：FLIP 在训练时删除大量图像 patch，仅编码可见部分，以扩大 batch 或增加同预算下样本数。
- **C2**：不同预训练数据会带来明显鲁棒性差异，论文专门提醒 WIT 与 LAION 的比较条件。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Li%2C%20Fan%2C%20Ai%20-%20Unknown%20-%20Scaling%20Language-Image%20Pre-training%20via%20Masking.pdf)
- 全文文本：[打开全文文本](../../raw/text/Li%2C%20Fan%2C%20Ai%20-%20Unknown%20-%20Scaling%20Language-Image%20Pre-training%20via%20Masking.md)
- 作者：Li, Fan, Ai
- 年份：Unknown
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Li%2C%20Fan%2C%20Ai%20-%20Unknown%20-%20Scaling%20Language-Image%20Pre-training%20via%20Masking.html)

## 争议与不确定点

- 文本编码、通信和训练阶段的其他成本仍存在。
- 图像遮挡比率与数据分布影响细粒度能力。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无

## 方法与实验解读

FLIP 将图文对比训练的预算重新分配：每张图少算一些，换取更多图文对。遮挡带来信息损失，收益来自批量、样本和计算之间的取舍，不能将图像编码减少比例当成整个系统同比加速。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Li%2C%20Fan%2C%20Ai%20-%20Unknown%20-%20Scaling%20Language-Image%20Pre-training%20via%20Masking.md#source-section-8 ) | 不是对完整图像做同等计算后再遮挡 |
| C2 | [原文]( ../../raw/text/Li%2C%20Fan%2C%20Ai%20-%20Unknown%20-%20Scaling%20Language-Image%20Pre-training%20via%20Masking.md#source-section-28 ) | 架构收益与数据收益需要区分 |

## 核证范围

核对 Image masking、§3 方法和零样本鲁棒性比较条件。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Kim et al. - Unknown - Full Stack Optimization of Transformer Inference a Survey

## TL;DR（快速导读）

这篇综述从模型、软件到硬件整理 Transformer 推理优化，说明速度问题需要沿整条执行链定位。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

推理成本可能来自模型规模、数据搬运、算子执行和系统组织。全栈优化关注这些环节如何相互配合。阅读时先区分单个算子、一次生成和完整服务的指标，避免把局部改进当成整机加速。

## 具体怎么理解

一个算子快了一倍，若它原来只占总耗时的小部分，用户看到的响应时间可能只改善一点。

## 关键事实

- **C1**：从模型算法、计算图、调度和硬件等层面讨论 Transformer 推理优化。
- **C2**：Transformer 的矩阵形状、内存层级与非线性开销不同于 CNN，优化配置需重新评估。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Kim%20et%20al.%20-%20Unknown%20-%20Full%20Stack%20Optimization%20of%20Transformer%20Inference%20a%20Survey.pdf)
- 全文文本：[打开全文文本](../../raw/text/Kim%20et%20al.%20-%20Unknown%20-%20Full%20Stack%20Optimization%20of%20Transformer%20Inference%20a%20Survey.md)
- 作者：Kim et al.
- 年份：Unknown
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Kim%20et%20al.%20-%20Unknown%20-%20Full%20Stack%20Optimization%20of%20Transformer%20Inference%20a%20Survey.html)

## 争议与不确定点

- 综述中的速度比例绑定具体测量或模拟设置。
- 编码器推理分析不能直接覆盖所有自回归 decode 瓶颈。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无
- [推理优化：量化、缓存与硬件](../comparisons/%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96%EF%BC%9A%E9%87%8F%E5%8C%96%E3%80%81%E7%BC%93%E5%AD%98%E4%B8%8E%E7%A1%AC%E4%BB%B6.md)：把本篇方法放到相关任务与比较条件中阅读。

## 方法与实验解读

推理瓶颈可能在计算、内存搬运或调度。该综述说明为什么参数变少、算子融合或照搬 CNN 加速器不一定更快：优化要测整个执行路径，并对齐 batch、序列长度、精度和设备。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/pdf/Kim%20et%20al.%20-%20Unknown%20-%20Full%20Stack%20Optimization%20of%20Transformer%20Inference%20a%20Survey.pdf#page=1 ) | 多层共同约束，不是单个压缩技巧 |
| C2 | [原文]( ../../raw/pdf/Kim%20et%20al.%20-%20Unknown%20-%20Full%20Stack%20Optimization%20of%20Transformer%20Inference%20a%20Survey.pdf#page=2 ) | 具体收益受序列长度与硬件资源影响 |

## 核证范围

核对 PDF 第 1–2 页全栈分类与主要硬件观察；不把各实验峰值拼成总体加速承诺。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

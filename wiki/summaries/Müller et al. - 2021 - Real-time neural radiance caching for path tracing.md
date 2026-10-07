---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Instant Neural Graphics Primitives：多分辨率哈希编码（2022）

## TL;DR（快速导读）

现有原文实际是 Instant-NGP：用多分辨率哈希特征和小网络加速神经图形表示。旧文件名误写成了神经辐射缓存论文。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

PDF 与 HTML 都对应《Instant Neural Graphics Primitives with a Multiresolution Hash Encoding》。方法把空间坐标映射到可训练的多尺度哈希特征，再交给较小网络，并配合高效 GPU 实现。它覆盖多种图形任务；本页不把旧归档名当成另一篇论文的证据。

## 具体怎么理解

可以把坐标看作地图上的位置，多层不同分辨率的特征表提供局部信息，小网络再组合它们预测颜色或其他场属性。

## 关键事实

- **C1**：实际原文是 Instant Neural Graphics Primitives，核心为多分辨率 hash encoding。
- **C2**：速度来自 hash 表示、小型 MLP 与 CUDA / fully-fused 实现的组合。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/M%C3%BCller%20et%20al.%20-%202021%20-%20Real-time%20neural%20radiance%20caching%20for%20path%20tracing.pdf)
- 全文文本：[打开全文文本](../../raw/text/M%C3%BCller%20et%20al.%20-%202021%20-%20Real-time%20neural%20radiance%20caching%20for%20path%20tracing.md)
- 作者：Thomas Müller 等
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/M%C3%BCller%20et%20al.%20-%202021%20-%20Real-time%20neural%20radiance%20caching%20for%20path%20tracing.html)
- 归档说明：文件名保留以维持已有链接；本页标题按原文识别内容整理，旧文件名不作为作者或年份依据。
- 归档说明：保留历史文件名以维持来源对应和链接；标题、作者与年份以上述核对信息为准。

## 争议与不确定点

- 低维空间坐标是编码的主要适用范围。
- 实现优化与 GPU 条件限制速度外推。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无
- [神经渲染](../concepts/%E7%A5%9E%E7%BB%8F%E6%B8%B2%E6%9F%93.md)：把本篇方法放到相关任务与比较条件中阅读。

## 方法与实验解读

Instant-NGP 将不同分辨率的可学习网格特征送入小网络，让空间细节更多由编码承担。它可用于 NeRF 等任务，但这里的训练与渲染加速不等于任意三维生成问题都解决。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/M%C3%BCller%20et%20al.%20-%202021%20-%20Real-time%20neural%20radiance%20caching%20for%20path%20tracing.md#source-section-7 ) | 修正历史文件名对应关系，保留稳定路径 |
| C2 | [原文]( ../../raw/text/M%C3%BCller%20et%20al.%20-%202021%20-%20Real-time%20neural%20radiance%20caching%20for%20path%20tracing.md#source-section-12 ) | 算法与实现共同贡献，不能只归因于网络结构 |

## 核证范围

核对正文题名、§3 hash encoding、§4 实现与非空间维度讨论。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Jiang, Lu - Unknown - InfiniteYou Flexible Photo Recrafting While Preserving Your Identity

## TL;DR（快速导读）

InfiniteYou 研究保持人物身份的图像再创作，让同一个人能出现在不同场景或风格中，同时关注文本匹配与画面质量。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

身份保持生成需要在改变姿态、场景和外观表达时仍保留人物特征。本文基于扩散 Transformer 研究这一任务，针对身份相似度、文本对齐和生成质量的取舍提出框架。身份相似和整张图相似不是同一个评价目标。

## 具体怎么理解

例如把同一个人物从室内头像换到户外场景，应保留身份特征，也应遵循新的环境描述。

## 关键事实

- **C1**：面向 DiT 的身份保持生成，使用 InfuseNet 与多阶段训练；不是照搬 U-Net 的 IP-Adapter。
- **C2**：身份相似度、CLIPScore 与图像质量分别评测。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Jiang%2C%20Lu%20-%20Unknown%20-%20InfiniteYou%20Flexible%20Photo%20Recrafting%20While%20Preserving%20Your%20Identity.pdf)
- 全文文本：[打开全文文本](../../raw/text/Jiang%2C%20Lu%20-%20Unknown%20-%20InfiniteYou%20Flexible%20Photo%20Recrafting%20While%20Preserving%20Your%20Identity.md)
- 作者：Jiang, Lu
- 年份：Unknown
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Jiang%2C%20Lu%20-%20Unknown%20-%20InfiniteYou%20Flexible%20Photo%20Recrafting%20While%20Preserving%20Your%20Identity.html)

## 争议与不确定点

- 比较含不同模型与基线适配程度，不能单独归因于一个模块。
- 身份一致不意味着每次生成都无失真。

## 关联页面

- 主题：[LLM RL](../topics/LLM%20RL.md)
- 综合：暂无
- [扩散模型与文生图](../topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)：回到相邻方法，核对任务边界。

## 方法与实验解读

InfiniteYou 试图在保留身份的同时保留 DiT 的生成能力。使用时应分别检查脸部相似、提示一致和画面瑕疵；一个人脸编码相似度不能代替对整张图的审核。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Jiang%2C%20Lu%20-%20Unknown%20-%20InfiniteYou%20Flexible%20Photo%20Recrafting%20While%20Preserving%20Your%20Identity.md#source-section-7 ) | 网络设计与训练策略共同影响结果 |
| C2 | [原文]( ../../raw/text/Jiang%2C%20Lu%20-%20Unknown%20-%20InfiniteYou%20Flexible%20Photo%20Recrafting%20While%20Preserving%20Your%20Identity.md#source-section-11 ) | 身份指标好不保证文字、画质与所有细节都好 |

## 核证范围

核对 §3.2 的架构选择与 §4.2 的多维评测说明。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

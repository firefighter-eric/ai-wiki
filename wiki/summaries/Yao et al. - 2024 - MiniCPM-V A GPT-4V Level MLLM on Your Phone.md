---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Yao et al. - 2024 - MiniCPM-V A GPT-4V Level MLLM on Your Phone

## TL;DR（快速导读）

MiniCPM-V 面向更轻量的视觉语言部署，研究怎样在有限模型规模下提供图像理解能力，成本和质量都需按设备测试。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

大规模多模态模型部署昂贵。报告从模型与训练设计探索较小模型的能力，面向实际使用约束。标题中的能力类比是作者主张，必须结合具体基准、版本和硬件理解。

## 具体怎么理解

在手机上读图片，除了回答是否正确，还要看加载内存、处理时间与持续运行成本。

## 关键事实

- **C1**：由视觉编码器、Perceiver 风格压缩层与 LLM 组成，视觉 token 压缩后进入语言模型。
- **C2**：端侧部署是独立工程问题，论文分别讨论部署挑战、实现和设备结果。
- **C3**：作者承认多模态能力深度与宽度仍有限，图像结果不能替代音视频能力验证。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Yao%20et%20al.%20-%202024%20-%20MiniCPM-V%20A%20GPT-4V%20Level%20MLLM%20on%20Your%20Phone.pdf)
- 全文文本：[打开全文文本](../../raw/text/Yao%20et%20al.%20-%202024%20-%20MiniCPM-V%20A%20GPT-4V%20Level%20MLLM%20on%20Your%20Phone.md)
- 作者：Yao et al.
- 年份：2024
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Yao%20et%20al.%20-%202024%20-%20MiniCPM-V%20A%20GPT-4V%20Level%20MLLM%20on%20Your%20Phone.html)

## 争议与不确定点

- 不同 MiniCPM-V 版本与底座不同，不能合并成一个统一分数。
- 部分基准胜出不等于对闭源模型所有场景全面等效。

## 关联页面

- 主题：[LLM RL](../topics/LLM%20RL.md)
- 综合：暂无
- [MiniCPM - ModelBest](../authors/MiniCPM%20-%20ModelBest.md)：沿作者或机构继续阅读相关来源。

## 方法与实验解读

MiniCPM-V 把高分辨率图像编码和 token 压缩结合，以控制语言模型的视觉输入成本。标题中的 GPT-4V level 指部分基准比较，端侧体验还取决于量化、设备、图片大小和执行后端。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Yao%20et%20al.%20-%202024%20-%20MiniCPM-V%20A%20GPT-4V%20Level%20MLLM%20on%20Your%20Phone.md#source-section-9 ) | 压缩减少 token，不等于原图信息完整保留 |
| C2 | [原文]( ../../raw/text/Yao%20et%20al.%20-%202024%20-%20MiniCPM-V%20A%20GPT-4V%20Level%20MLLM%20on%20Your%20Phone.md#source-section-30 ) | 模型基准与手机延迟需分开判断 |
| C3 | [原文]( ../../raw/text/Yao%20et%20al.%20-%202024%20-%20MiniCPM-V%20A%20GPT-4V%20Level%20MLLM%20on%20Your%20Phone.md#source-section-65 ) | 具体 MiniCPM-V 版本，不指整个后续家族 |

## 核证范围

核对 §3.1、§5 的端侧部署、§6 主结果和 Limitations；不外推为任意手机可用性。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

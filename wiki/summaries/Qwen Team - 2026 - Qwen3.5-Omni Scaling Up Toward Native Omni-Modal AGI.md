---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Qwen Team - 2026 - Qwen3.5-Omni Scaling Up Toward Native Omni-Modal AGI

## TL;DR（快速导读）

这份 Qwen3.5-Omni 索引快照介绍多模态家族的扩展方向，关注长上下文和音视频理解。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

输入视频并要求语音回应时，需要确认模型版本、是否支持流式以及声音输出的限制。

## 来源信息

- 类型：官方研究索引条目 / 发布摘要
- 来源链接：https://qwen.ai/research
- 全文文本：../../raw/text/Qwen Team - 2026 - Qwen3.5-Omni Scaling Up Toward Native Omni-Modal AGI.md
- 作者：Qwen Team
- 年份：2026
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开有效正文快照](../../raw/html/verified/qwen3.5-omni-official-blog-2026-10-07.html)
- 核对说明：使用官方公开article接口返回的content补足旧SPA外壳；保存正文版本而非登录状态。

## 摘要

它帮助定位家族研究资料，但并不等于完整技术报告。尺寸、输入输出、开放范围和实际能力仍需逐个成员核对；归档日期也限定了这里对版本关系的说明。

## 关键事实

- **C1**：Plus/Flash/Light支持全模态输入，声明256K、超过10小时音频或400秒720P/1FPS视频。
- **C2**：Thinker/Talker均采用HybridAttentionMoE，通过TMRoPE与交织输入同步音视频。
- **C3**：ARIA动态对齐text/speech units以改善漏读、误读和数字发音。
- **C4**：官方215项SOTA包含benchmark与面向语言的子任务，不是215个独立数据集。

## 争议与不确定点

- 博客36种语音生成含语言/方言口径，不能与技术报告另一设定的10种直接合并。
- 长输入和厂商SOTA声明不意味着每种语言、每种组合均有同等保证。

## 关联页面

- 概念：[Qwen3.5-Omni](../../wiki/concepts/Qwen3.5-Omni.md)
- 概念：[Qwen2.5-Omni](../../wiki/concepts/Qwen2.5-Omni.md)
- 主题：[Qwen 系列](../../wiki/topics/Qwen%20系列.md)

## 这里的术语是什么意思

- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。

## 方法与实验解读

Thinker理解、Talker发声，流式chunk与ARIA协调文本和语音的不同token速率。语义打断、搜索工具和音色控制属于交互功能；准确率、时序稳定性与端到端响应要分别评测。原生多模态仍需要注明是否包含音轨和视频采样率。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Qwen%20Team%20-%202026%20-%20Qwen3.5-Omni%20Scaling%20Up%20Toward%20Native%20Omni-Modal%20AGI.md#source-section-0 ) | 长视频规格依赖采样率，不是全帧原始连续流。 |
| C2 | [原文]( ../../raw/text/Qwen%20Team%20-%202026%20-%20Qwen3.5-Omni%20Scaling%20Up%20Toward%20Native%20Omni-Modal%20AGI.md#source-section-7 ) | 输入路径、输出路径分别看。 |
| C3 | [原文]( ../../raw/text/Qwen%20Team%20-%202026%20-%20Qwen3.5-Omni%20Scaling%20Up%20Toward%20Native%20Omni-Modal%20AGI.md#source-section-7 ) | 目标与声明，不保证错误清零。 |
| C4 | [原文]( ../../raw/text/Qwen%20Team%20-%202026%20-%20Qwen3.5-Omni%20Scaling%20Up%20Toward%20Native%20Omni-Modal%20AGI.md#source-section-0 ) | 不同指标不可累加成统一优势。 |

## 核证范围

核读官方article正文、完整架构、代际对照、音视频脚注与offline/realtime说明。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

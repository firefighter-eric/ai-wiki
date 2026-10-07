---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Qwen Team - 2025 - Qwen2.5-Omni See Hear Talk Write Do It All

## TL;DR（快速导读）

Qwen2.5-Omni 接收文字、图片、音频和视频，并生成文字或语音，研究流式多模态交互。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

边听边看视频再口头回应，需要对齐画面、声音和输出时序；不是单纯把录音转成文字。

## 来源信息

- 类型：官方博客 / 技术发布
- 来源链接：https://qwenlm.github.io/blog/qwen2.5-omni/
- 全文文本：../../raw/text/Qwen Team - 2025 - Qwen2.5-Omni See Hear Talk Write Do It All.md
- 作者：Qwen Team
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

来源强调在同一模型中处理多种模态和实时输入输出。理解输入内容、及时回应和生成自然声音是不同能力，需要分别核对；端到端也不意味着每个模态都没有瓶颈。

## 关键事实

- **C1**：模型处理文本、图像、音频、视频，流式输出文本与语音。
- **C2**：Thinker负责多模态理解/文本，Talker接收高维表示与文本，流式生成speech tokens。
- **C3**：TMRoPE用于同步视频时间戳与音频时间轴。
- **C4**：报告在OmniBench与多种单模态基准比较。

## 争议与不确定点

- 发布页未给可复现的完整吞吐/延迟配方。
- 厂商所述同尺寸比较不表示替代所有单模态专家。

## 关联页面

- 概念：[Qwen2.5-Omni](../../wiki/concepts/Qwen2.5-Omni.md)
- 概念：[Qwen3.5-Omni](../../wiki/concepts/Qwen3.5-Omni.md)
- 主题：[Qwen 系列](../../wiki/topics/Qwen%20系列.md)

## 方法与实验解读

理解与说话有不同输出时钟：Talker在接收Thinker表示时可开始输出，减少整段生成后的等待。流式接口、延迟和回答正确性仍是不同评测项；多模态合并也会引入跨模态冲突和时序错误。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Qwen%20Team%20-%202025%20-%20Qwen2.5-Omni%20See%20Hear%20Talk%20Write%20Do%20It%20All.md#source-section-0 ) | 输入输出模态必须分开。 |
| C2 | [原文]( ../../raw/text/Qwen%20Team%20-%202025%20-%20Qwen2.5-Omni%20See%20Hear%20Talk%20Write%20Do%20It%20All.md#source-section-1 ) | 单一协同结构，而非独立模型简单接API。 |
| C3 | [原文]( ../../raw/text/Qwen%20Team%20-%202025%20-%20Qwen2.5-Omni%20See%20Hear%20Talk%20Write%20Do%20It%20All.md#source-section-0 ) | 时间位置编码并非任意视频都音画无误。 |
| C4 | [原文]( ../../raw/text/Qwen%20Team%20-%202025%20-%20Qwen2.5-Omni%20See%20Hear%20Talk%20Write%20Do%20It%20All.md#source-section-2 ) | 不同任务有不同指标，不能混作统一SOTA。 |

## 核证范围

核读intro/TMRoPE、Thinker-Talker与具体benchmark维度。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

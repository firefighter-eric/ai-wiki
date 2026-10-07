---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Wang et al. - 2017 - Tacotron Towards end-To-end speech synthesis

## TL;DR（快速导读）

Tacotron 从字符直接预测语音声学表示，减少传统文字转语音系统中大量手工模块，是端到端 TTS 的代表路线。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

传统系统通常拆成文本分析、声学预测和声音合成。Tacotron 用文字与音频配对数据学习前面的映射，再产生可用于波形合成的声学输出。内容、对齐和最终音质仍需要分别检查。

## 具体怎么理解

“今天下雨了”不仅要读出正确字符，也要学到字与声学时间段的对应及自然停顿。

## 关键事实

- **C1**：Tacotron 从字符经 attention seq2seq 生成谱图，再转为波形。
- **C2**：原版 post-net 预测线性频谱，使用 Griffin-Lim；与 Tacotron 2 的 WaveNet 不同。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Wang%20et%20al.%20-%202017%20-%20Tacotron%20Towards%20end-To-end%20speech%20synthesis.pdf)
- 全文文本：[打开全文文本](../../raw/text/Wang%20et%20al.%20-%202017%20-%20Tacotron%20Towards%20end-To-end%20speech%20synthesis.md)
- 作者：Wang et al.
- 年份：2017
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Wang%20et%20al.%20-%202017%20-%20Tacotron%20Towards%20end-To-end%20speech%20synthesis.html)

## 争议与不确定点

- 长句可能出现对齐与停止错误。
- 单说话人英语 MOS 不能代表所有语言和韵律控制。

## 关联页面

- 主题：[传统NLP](../topics/传统%20NLP.md)
- 综合：暂无
- [语音表示、识别与合成](../concepts/%E8%AF%AD%E9%9F%B3%E8%A1%A8%E7%A4%BA%E3%80%81%E8%AF%86%E5%88%AB%E4%B8%8E%E5%90%88%E6%88%90.md)：把本篇方法放到相关任务与比较条件中阅读。

## 方法与实验解读

Tacotron 学习文字与声学帧的对齐，减少手工语言学前端。它在帧级生成，速度与音质都受到对齐、post-net 和波形恢复影响，不能把 end-to-end 理解为取消所有处理阶段。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Wang%20et%20al.%20-%202017%20-%20Tacotron%20Towards%20end-To-end%20speech%20synthesis.md#source-section-5 ) | 主要学习声学表示，仍有波形合成步骤 |
| C2 | [原文]( ../../raw/text/Wang%20et%20al.%20-%202017%20-%20Tacotron%20Towards%20end-To-end%20speech%20synthesis.md#source-section-9 ) | 两代系统的声码器不能混用 |

## 核证范围

核对 §3 架构、§3.4 波形合成和模型细节的停止问题。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

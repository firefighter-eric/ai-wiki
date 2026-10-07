---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# AV-HuBERT：遮挡多模态聚类的音视频语音表示（2022）

## TL;DR（快速导读）

AV-HuBERT 同时看嘴唇运动和听声音，用遮挡后的预测任务学习语音表示，为音视频语音识别提供预训练基础。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

说话时，口型与声音包含相关信息。论文把音频和视频的部分内容遮住，让模型预测由聚类得到的隐藏单元；这些单元还能迭代更新。它学习的是可迁移的表示，之后仍需接入语音识别等具体任务。

## 具体怎么理解

比如一段视频里的声音被噪声盖住，嘴唇的变化仍能提供线索；预训练希望模型学会利用两种输入之间的对应关系。

## 关键事实

- **C1**：AV-HuBERT 交替进行音视频特征聚类与遮挡预测，利用语音和嘴部运动关联。
- **C2**：实验结合 LRS3 与 VoxCeleb2 的英语部分，并区分无标签预训练及有标签微调。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Ai%20-%202022%20-%20Learning%20Audio-Visual%20Speech%20Representation%20by%20Masked%20Multimodal%20Cluster%20Prediction.pdf)
- 全文文本：[打开全文文本](../../raw/text/Ai%20-%202022%20-%20Learning%20Audio-Visual%20Speech%20Representation%20by%20Masked%20Multimodal%20Cluster%20Prediction.md)
- 作者：Bowen Shi、Wei-Ning Hsu、Kushal Lakhotia、Abdelrahman Mohamed
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Ai%20-%202022%20-%20Learning%20Audio-Visual%20Speech%20Representation%20by%20Masked%20Multimodal%20Cluster%20Prediction.html)
- 归档说明：保留历史文件名以维持来源对应和链接；标题、作者与年份以上述核对信息为准。

## 争议与不确定点

- 英语视频和嘴部可见度限制外推。
- 唇读与完整音视频 ASR 的 WER 不可混为同一成绩。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无
- [语音表示、识别与合成](../concepts/%E8%AF%AD%E9%9F%B3%E8%A1%A8%E7%A4%BA%E3%80%81%E8%AF%86%E5%88%AB%E4%B8%8E%E5%90%88%E6%88%90.md)：把本篇方法放到相关任务与比较条件中阅读。

## 方法与实验解读

AV-HuBERT 用音频帮助学习视觉语音表示，再在相应识别任务上微调。比较低标注收益时，无标注训练量和推理使用模态都必须对齐，否则容易把数据优势误认为纯架构优势。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Ai%20-%202022%20-%20Learning%20Audio-Visual%20Speech%20Representation%20by%20Masked%20Multimodal%20Cluster%20Prediction.md#source-section-8 ) | 多模态预训练与只用视觉唇读的推理设置分开 |
| C2 | [原文]( ../../raw/text/Ai%20-%202022%20-%20Learning%20Audio-Visual%20Speech%20Representation%20by%20Masked%20Multimodal%20Cluster%20Prediction.md#source-section-11 ) | 使用额外无标签数据不等于没有额外训练数据 |

## 核证范围

核对 §3.3 的迭代学习、§4.1 的数据及 §4.2 的标注条件。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

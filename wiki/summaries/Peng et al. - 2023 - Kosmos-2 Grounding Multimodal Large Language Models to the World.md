---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Peng et al. - 2023 - Kosmos-2 Grounding Multimodal Large Language Models to the World

## TL;DR（快速导读）

Kosmos-2 把语言中的对象描述与图像中的位置绑定起来，使模型生成文字时也能指出“说的是哪里”。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

普通图像描述只输出文字，难以定位指代对象。论文用位置词元和图文定位数据训练模型，把文字片段与边界框相连。定位是否准确与描述是否正确需要分别评估。

## 具体怎么理解

模型说“左边的狗”时，还应给出图像中对应区域，读者才知道它指的是哪一个对象。

## 关键事实

- **C1**：GrIT 将 caption 中名词短语与图像区域关联，为 grounding 提供训练对。
- **C2**：将连续框坐标离散成位置 token，与文字放入统一序列。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Peng%20et%20al.%20-%202023%20-%20Kosmos-2%20Grounding%20Multimodal%20Large%20Language%20Models%20to%20the%20World.pdf)
- 全文文本：[打开全文文本](../../raw/text/Peng%20et%20al.%20-%202023%20-%20Kosmos-2%20Grounding%20Multimodal%20Large%20Language%20Models%20to%20the%20World.md)
- 作者：Peng et al.
- 年份：2023
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Peng%20et%20al.%20-%202023%20-%20Kosmos-2%20Grounding%20Multimodal%20Large%20Language%20Models%20to%20the%20World.html)

## 争议与不确定点

- 自动生成图文关联的质量会影响训练。
- Flickr30k 指代表达成绩与通用图像问答成绩不同。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无
- [Microsoft Research](../authors/Microsoft%20Research.md)：沿作者或机构继续阅读相关来源。

## 方法与实验解读

Kosmos-2 让文字中的对象指代能落到图像框上。它建立语言与区域的联系，但 grounding 命中不能自动证明复杂关系、计数或常识判断正确；这些能力要分别评估。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Peng%20et%20al.%20-%202023%20-%20Kosmos-2%20Grounding%20Multimodal%20Large%20Language%20Models%20to%20the%20World.md#source-section-4 ) | 自动构造的区域关联有误差可能 |
| C2 | [原文]( ../../raw/text/Peng%20et%20al.%20-%202023%20-%20Kosmos-2%20Grounding%20Multimodal%20Large%20Language%20Models%20to%20the%20World.md#source-section-8 ) | 离散位置精度受网格设置约束 |

## 核证范围

核对 §2 GrIT 构造、§3.1 坐标表示和 §4 的任务范围。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

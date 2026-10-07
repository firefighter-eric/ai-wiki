---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Wang et al. - 2024 - CDM A Reliable Metric for Fair and Accurate Formula Recognition Evaluation

## TL;DR（快速导读）

CDM 研究公式识别的评价方式，避免仅用字符串差异惩罚形式不同但内容相近的公式表达。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

同一个公式可以有多种标记写法，传统文本指标未必公平。论文针对公式表示与视觉内容提出更合适的度量。评估时应区分书写形式、结构与数学意义，不能凭单一分数判断所有错误。

## 具体怎么理解

两段 LaTeX 可能渲染成一样的公式；反过来，只差一个上标的字符串也可能改变数学含义。

## 关键事实

- **C1**：CDM 先渲染 LaTeX，再进行字符空间匹配，以减轻等价表示在字符串指标中的惩罚。
- **C2**：元素匹配采用二分图匹配，不直接对全部像素逐点比较。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Wang%20et%20al.%20-%202024%20-%20CDM%20A%20Reliable%20Metric%20for%20Fair%20and%20Accurate%20Formula%20Recognition%20Evaluation.pdf)
- 全文文本：[打开全文文本](../../raw/text/Wang%20et%20al.%20-%202024%20-%20CDM%20A%20Reliable%20Metric%20for%20Fair%20and%20Accurate%20Formula%20Recognition%20Evaluation.md)
- 作者：Wang et al.
- 年份：2024
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Wang%20et%20al.%20-%202024%20-%20CDM%20A%20Reliable%20Metric%20for%20Fair%20and%20Accurate%20Formula%20Recognition%20Evaluation.html)

## 争议与不确定点

- 需要可渲染输出；渲染失败应与内容错误分别报告。
- 视觉匹配不证明公式表达的科学含义正确。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无

## 方法与实验解读

相同公式可有不同 LaTeX 写法，BLEU 或编辑距离会把这些写法差异算成错误。CDM 把比较移到渲染后的字符和位置，适合核查公式是否被正确识别，但不会自动判断代数恒等或推导正确。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Wang%20et%20al.%20-%202024%20-%20CDM%20A%20Reliable%20Metric%20for%20Fair%20and%20Accurate%20Formula%20Recognition%20Evaluation.md#source-section-8 ) | 视觉表达相同与数学语义等价并不是同一个概念 |
| C2 | [原文]( ../../raw/text/Wang%20et%20al.%20-%202024%20-%20CDM%20A%20Reliable%20Metric%20for%20Fair%20and%20Accurate%20Formula%20Recognition%20Evaluation.md#source-section-10 ) | 仍依赖渲染、字符定位和匹配质量 |

## 核证范围

核对 §3 的旧指标局限、§4 的渲染与匹配设计和文档级评测说明。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

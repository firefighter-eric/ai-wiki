---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Li et al. - 2025 - dots.ocr Multilingual Document Layout Parsing in a Single Vision-Language Model

## TL;DR（快速导读）

dots.ocr 用同一视觉语言模型理解文档版面、识别内容和组织阅读关系，研究减少多阶段误差累积。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

一页双栏资料要同时读对文字、区分区域和排好顺序；这三项中任何一项出错都会影响最终文本。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Li et al. - 2025 - dots.ocr Multilingual Document Layout Parsing in a Single Vision-Language Model.pdf
- 全文文本：../../raw/text/Li et al. - 2025 - dots.ocr Multilingual Document Layout Parsing in a Single Vision-Language Model.md
- 作者：Li et al.
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

方法试图把区域、文字和阅读顺序放进一次生成中联合学习。单模型并不意味着各项都可靠，仍需分别核对多语言文字、表格、公式和顺序，尤其关注复杂页面上的遗漏与结构错误。

## 关键事实

- **C1**：dots.ocr把每个block输出为bbox、类别、内容组成的有序序列，共同学习布局、识别与关系。
- **C2**：架构为从头训练的1.2B视觉encoder加约1.7B语言decoder。
- **C3**：作者引入126语言XDocParse并报告OmniDocBench EN87.5/CH84.0。
- **C4**：XDocParse OverallEdit约0.177，数值越低越好。

## 争议与不确定点

- 126语言存在不保证每种语言相同质量；应看分语言结果。
- 作者统一化主张和基准成绩不等于任意复杂PDF都可无后处理。

## 关联页面

- 概念：[dots.ocr](../../wiki/concepts/dots.ocr.md)
- 概念：[PaddleOCR](../../wiki/concepts/PaddleOCR.md)
- 概念：[GLM-OCR](../../wiki/concepts/GLM-OCR.md)
- 概念：[DeepSeek-OCR](../../wiki/concepts/DeepSeek-OCR.md)
- 主题：[OCR](../../wiki/topics/OCR.md)
- 主题：[传统 CV](../../wiki/topics/传统%20CV.md)

## 这里的术语是什么意思

- **encoder**：编码器：把输入转成模型内部表示。
- **decoder**：解码器：根据已有表示产生文字、图像或其他输出。
- **OCR**：文字识别：从图像读取文字；整页任务还需处理布局和阅读顺序。

## 方法与实验解读

统一序列让布局、文本和结构关系互相约束，合成数据引擎补多语言与版式覆盖。作者指出pipeline误差传播，但端到端路线也面临长输出、坐标和顺序错误，不能把pipeline一概当过时。选择应按可定位错误、语言覆盖与全页面准确率比较。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Li%20et%20al.%20-%202025%20-%20dots.ocr%20Multilingual%20Document%20Layout%20Parsing%20in%20a%20Single%20Vision-Language%20Model.md#source-section-9 ) | sequence顺序承担reading-order语义。 |
| C2 | [原文]( ../../raw/text/Li%20et%20al.%20-%202025%20-%20dots.ocr%20Multilingual%20Document%20Layout%20Parsing%20in%20a%20Single%20Vision-Language%20Model.md#source-section-10 ) | 约2.9B核心模型，不能仅按decoder大小比较。 |
| C3 | [原文]( ../../raw/text/Li%20et%20al.%20-%202025%20-%20dots.ocr%20Multilingual%20Document%20Layout%20Parsing%20in%20a%20Single%20Vision-Language%20Model.md#source-section-3 ) | 原报告benchmark版本；不与后来v1.5新分数直接拼榜。 |
| C4 | [原文]( ../../raw/text/Li%20et%20al.%20-%202025%20-%20dots.ocr%20Multilingual%20Document%20Layout%20Parsing%20in%20a%20Single%20Vision-Language%20Model.md#source-section-18 ) | edit distance与quality score方向相反。 |

## 核证范围

核读§3统一格式/架构、OmniDocBench与XDocParse评测。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

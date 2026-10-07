---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Zheng et al. - 2025 - PPTAgent Generating and Evaluating Presentations Beyond Text-to-Slides

## TL;DR（快速导读）

PPTAgent 把演示文稿生成组织为分阶段编辑流程，同时评估内容、视觉效果和跨页结构，超出只把文字放进幻灯片。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

生成 PPT 需要决定信息顺序、页面功能和布局，再执行编辑。论文以两阶段方案处理这些问题，并将多方面质量纳入评价。生成成功还应通过实际页面阅读与跨页逻辑检查。

## 具体怎么理解

一份五页汇报可以每页文字都正确，却没有开场、论据和结论的顺序；结构连贯性需要单独判断。

## 关键事实

- **C1**：先分析参考演示文稿并提取内容 schema，再生成大纲并通过可执行代码编辑参考页。
- **C2**：编辑阶段迭代修改所选参考页，而非只把段落转换成幻灯片。
- **C3**：复杂嵌套 group shape 的解析仍是瓶颈，成功率不是百分之百。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Zheng%20et%20al.%20-%202025%20-%20PPTAgent%20Generating%20and%20Evaluating%20Presentations%20Beyond%20Text-to-Slides.pdf)
- 全文文本：[打开全文文本](../../raw/text/Zheng%20et%20al.%20-%202025%20-%20PPTAgent%20Generating%20and%20Evaluating%20Presentations%20Beyond%20Text-to-Slides.md)
- 作者：Zheng et al.
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Zheng%20et%20al.%20-%202025%20-%20PPTAgent%20Generating%20and%20Evaluating%20Presentations%20Beyond%20Text-to-Slides.html)

## 争议与不确定点

- 超过 95% 的成功率对应特定任务和模型组合，不等于设计合格率。
- 复杂对象和模板可能需要专门处理。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无

## 方法与实验解读

PPTAgent 利用现有参考页承接布局，并把新内容填入结构。这个流程把规划、页面编辑和执行检查连接起来；选错参考页、解析失败或文字溢出仍会使最终演示文稿不合格。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202025%20-%20PPTAgent%20Generating%20and%20Evaluating%20Presentations%20Beyond%20Text-to-Slides.md#source-section-6 ) | 参考材料与页面结构是生成前提 |
| C2 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202025%20-%20PPTAgent%20Generating%20and%20Evaluating%20Presentations%20Beyond%20Text-to-Slides.md#source-section-9 ) | 执行成功与设计、内容质量分别评价 |
| C3 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202025%20-%20PPTAgent%20Generating%20and%20Evaluating%20Presentations%20Beyond%20Text-to-Slides.md#source-section-39 ) | 论文样本的执行成功率不能代表任意 PPT |

## 核证范围

核对 §2.2–2.3 的两阶段流程、实验成功率说明及 §7 局限。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

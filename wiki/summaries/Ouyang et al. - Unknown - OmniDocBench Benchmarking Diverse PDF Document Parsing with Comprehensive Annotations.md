---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Ouyang et al. - Unknown - OmniDocBench Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations

## TL;DR（快速导读）

OmniDocBench 用多来源 PDF 和细致标注评估文档解析，帮助发现正文、公式、表格等不同内容的识别短板。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

只用少数干净论文测试，会掩盖解析器在真实文档上的问题。基准扩展文档类型和内容标注，支持更全面的比较。应分别看元素识别、阅读顺序和结构保真，不把一个总分当成每类页面都可靠。

## 具体怎么理解

一个工具可能正文很好、公式很差；另一工具可能表格准确，却把双栏顺序读乱，评测应显示这些差异。

## 关键事实

- **C1**：原报告 OmniDocBench 包含 981 个 PDF 页面、九类页面，并标注语言、布局、模糊、水印与背景等属性。
- **C2**：评价流水线包括内容提取、Adjacency Search Match 对齐和指标计算；合并与拆分段落减少不同分段方式对评分的干扰。
- **C3**：文本用归一化编辑距离，表格用 TEDS 等，公式用 CDM/编辑距离/BLEU；阅读顺序只评价参加计算的文字组件。
- **C4**：原实验专用解析工具整体较强，但模糊、水印和复杂背景子集中部分 VLM 更稳健。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Ouyang%20et%20al.%20-%20Unknown%20-%20OmniDocBench%20Benchmarking%20Diverse%20PDF%20Document%20Parsing%20with%20Comprehensive%20Annotations.pdf)
- 全文文本：[打开全文文本](../../raw/text/Ouyang%20et%20al.%20-%20Unknown%20-%20OmniDocBench%20Benchmarking%20Diverse%20PDF%20Document%20Parsing%20with%20Comprehensive%20Annotations.md)
- 作者：Ouyang et al.
- 年份：Unknown
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Ouyang%20et%20al.%20-%20Unknown%20-%20OmniDocBench%20Benchmarking%20Diverse%20PDF%20Document%20Parsing%20with%20Comprehensive%20Annotations.html)

## 争议与不确定点

- 忽略规则使某些真实需求没有进入总分，caption 与页脚仍需单独抽查。
- 总榜单是归档版本与评测配置的结果，不代表当前软件版本。
- 页级数据不能直接证明跨页表格、跨页推理与完整论文解析效果。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无

## 方法与实验解读

论文解析有多个不同问题：识字、恢复表格、表达公式、决定阅读顺序。OmniDocBench 把这些分开测，并尝试对齐不同系统的分段。用于本库验收时，应同时抽查文字、公式和表格，不能只凭正文字符数判断解析成功。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Ouyang%20et%20al.%20-%20Unknown%20-%20OmniDocBench%20Benchmarking%20Diverse%20PDF%20Document%20Parsing%20with%20Comprehensive%20Annotations.md#source-section-13 ) | 这是本地归档版本的数据规模，不能与后续扩展版本的规模或榜单混用。 |
| C2 | [原文]( ../../raw/text/Ouyang%20et%20al.%20-%20Unknown%20-%20OmniDocBench%20Benchmarking%20Diverse%20PDF%20Document%20Parsing%20with%20Comprehensive%20Annotations.md#source-section-16 ) | 对齐算法自身选择会影响分数，仍需检查真实遗漏与错配。 |
| C3 | [原文]( ../../raw/text/Ouyang%20et%20al.%20-%20Unknown%20-%20OmniDocBench%20Benchmarking%20Diverse%20PDF%20Document%20Parsing%20with%20Comprehensive%20Annotations.md#source-section-17 ) | 页眉页脚、页码、部分脚注和 caption 被忽略；高阅读顺序分不代表整页所有元素都完整。 |
| C4 | [原文]( ../../raw/text/Ouyang%20et%20al.%20-%20Unknown%20-%20OmniDocBench%20Benchmarking%20Diverse%20PDF%20Document%20Parsing%20with%20Comprehensive%20Annotations.md#source-section-20 ) | 按页面属性与组件看误差，不能把某工具的整体榜首当作每种页面都最佳。 |

## 核证范围

核对 §3 数据取得和统计、§4 对齐与指标忽略规则、§5.2 整体及干扰页面评价。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

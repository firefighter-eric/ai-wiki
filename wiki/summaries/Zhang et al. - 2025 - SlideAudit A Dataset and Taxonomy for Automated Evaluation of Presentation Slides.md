---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Zhang et al. - 2025 - SlideAudit A Dataset and Taxonomy for Automated Evaluation of Presentation Slides

## TL;DR（快速导读）

SlideAudit 用专家整理的设计缺陷分类和标注幻灯片，研究如何自动发现演示页面中的具体问题。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

幻灯片评价涉及排版、内容与视觉组织，单纯测文字质量不够。数据集将缺陷类型明确化，便于模型识别和评测。需要检查标注一致性和缺陷覆盖，检测到某些问题不等于完成整份汇报的沟通评价。

## 具体怎么理解

例如字号过小、元素拥挤与对齐错误应有不同标签，这样结果才能告诉作者具体该改哪里。

## 关键事实

- **C1**：通过专家评估与反思迭代建立幻灯片缺陷 taxonomy，用于定位问题及生成改进建议。
- **C2**：设计评价依赖受众、语境和演示目标，作者承认其主观性。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Zhang%20et%20al.%20-%202025%20-%20SlideAudit%20A%20Dataset%20and%20Taxonomy%20for%20Automated%20Evaluation%20of%20Presentation%20Slides.pdf)
- 全文文本：[打开全文文本](../../raw/text/Zhang%20et%20al.%20-%202025%20-%20SlideAudit%20A%20Dataset%20and%20Taxonomy%20for%20Automated%20Evaluation%20of%20Presentation%20Slides.md)
- 作者：Zhang et al.
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Zhang%20et%20al.%20-%202025%20-%20SlideAudit%20A%20Dataset%20and%20Taxonomy%20for%20Automated%20Evaluation%20of%20Presentation%20Slides.html)

## 争议与不确定点

- 缺陷定位 IoU、建议有用性和最终作品质量是不同指标。
- 样本和专家偏好限制分类体系的普适性。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无

## 方法与实验解读

SlideAudit 给设计反馈提供可讨论的词汇和标注对象。反馈应说明哪里有问题、为什么影响表达、怎样修改；模型提出计划后，仍须查看实际改版，而非把用户觉得建议有用当成设计已经改善。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Zhang%20et%20al.%20-%202025%20-%20SlideAudit%20A%20Dataset%20and%20Taxonomy%20for%20Automated%20Evaluation%20of%20Presentation%20Slides.md#source-section-9 ) | 缺陷分类与整页审美偏好分开 |
| C2 | [原文]( ../../raw/text/Zhang%20et%20al.%20-%202025%20-%20SlideAudit%20A%20Dataset%20and%20Taxonomy%20for%20Automated%20Evaluation%20of%20Presentation%20Slides.md#source-section-39 ) | 统一 taxonomy 不能替代用户偏好 |

## 核证范围

核对 §3.2 的研究流程、§5 的定位与建议评测以及 §6.2 局限。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

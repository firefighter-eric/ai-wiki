---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Schick et al. - 2023 - Toolformer Language Models Can Teach Themselves to Use Tools

## TL;DR（快速导读）

Toolformer 用少量示范和自监督筛选，让语言模型学习何时调用工具、传什么参数以及如何使用结果。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

遇到算术题时，可先调用计算器，再将返回数值放回文本；选对工具与正确解释结果同样重要。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Schick et al. - 2023 - Toolformer Language Models Can Teach Themselves to Use Tools.pdf
- 原始 HTML：../../raw/html/Schick et al. - 2023 - Toolformer Language Models Can Teach Themselves to Use Tools.html
- 全文文本：../../raw/text/Schick et al. - 2023 - Toolformer Language Models Can Teach Themselves to Use Tools.md
- 作者：Schick et al.
- 年份：2023
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

方法将 API 调用嵌入生成过程，并根据调用对后续预测是否有帮助筛选训练材料。它研究的是工具使用学习；工具返回错误、参数不适合或任务需要多步执行时，仍需额外检查可靠性。

## 关键事实

- **C1**：Toolformer用少量API示范引导生成调用、执行工具，再按是否减少后续语言建模loss筛选。
- **C2**：工具包括calculator/QA/search/translation/calendar。
- **C3**：当前方法不能串联工具或交互式迭代浏览，调用对提示措辞敏感。

## 争议与不确定点

- 独立生成调用的数据不覆盖有状态多步链。
- 工具输出错误和训练/运行分布偏移仍影响结果。

## 关联页面

- 概念：[Toolformer](../../wiki/concepts/Toolformer.md)
- 主题：[AI 智能问答与智能客服](../../wiki/topics/AI%20%E6%99%BA%E8%83%BD%E9%97%AE%E7%AD%94%E4%B8%8E%E6%99%BA%E8%83%BD%E5%AE%A2%E6%9C%8D.md)

## 这里的术语是什么意思

- **agent**：代理：围绕任务读材料、调用工具和连续执行的系统；名称本身不保证自主性或质量。

## 方法与实验解读

自动插入的调用只有在结果帮助预测后文时留下，降低了人工标注依赖；然而loss改善不保证调用必要、参数合法或业务执行成功。客服应用可以借此设计调用学习，但需要另行证明权限、状态与任务完成。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Schick%20et%20al.%20-%202023%20-%20Toolformer%20Language%20Models%20Can%20Teach%20Themselves%20to%20Use%20Tools.md#source-section-7 ) | 筛选代理是tokenloss，不是完整用户任务效用。 |
| C2 | [原文]( ../../raw/text/Schick%20et%20al.%20-%202023%20-%20Toolformer%20Language%20Models%20Can%20Teach%20Themselves%20to%20Use%20Tools.md#source-section-10 ) | 静态API任务，未验证创建订单等真实业务。 |
| C3 | [原文]( ../../raw/text/Schick%20et%20al.%20-%202023%20-%20Toolformer%20Language%20Models%20Can%20Teach%20Themselves%20to%20Use%20Tools.md#source-section-36 ) | 明确局限，不能称已完整解决agent规划。 |

## 核证范围

核读采样/执行/loss过滤、API集合、下游与明确限制。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

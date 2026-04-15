# Schick et al. - 2023 - Toolformer Language Models Can Teach Themselves to Use Tools

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Schick et al. - 2023 - Toolformer Language Models Can Teach Themselves to Use Tools.pdf
- 原始 HTML：../../raw/html/Schick et al. - 2023 - Toolformer Language Models Can Teach Themselves to Use Tools.html
- 全文文本：../../raw/text/Schick et al. - 2023 - Toolformer Language Models Can Teach Themselves to Use Tools.md
- 作者：Schick et al.
- 年份：2023
- 状态：已整理

## 摘要

Toolformer 讨论的不是“模型会不会工具调用”这种泛泛问题，而是语言模型能否在少量示范下自监督学习何时调用外部 API、传什么参数、怎样把返回结果并入后续生成。它的重要意义在于把外部工具从“手工编排插件”推进到“模型内化的行动接口”。对智能客服来说，这意味着 AI 不再只能解释知识，还可以执行查询、检索和事务动作。

## 关键事实

- Toolformer 让模型学习四件事：是否调用工具、调用哪个工具、传什么参数、如何使用工具返回结果。
- 论文使用少量 demonstrations 为每种 API 引导自监督数据构造，而不是要求大规模人工工具调用标注。
- 所接入工具包含计算器、问答系统、搜索、翻译和日历等，说明工具调用被视为通用能力而非单一插件特化。
- 对智能客服 / support assistant 而言，这一方向支撑“处理型客服”能力，例如查订单、查状态、算费用、创建工单或查询知识系统。
- 它也为后续 agent 型客服提供方法论前身：问题不只是生成回复，而是决定何时借助外部系统。

## 争议与不确定点

- Toolformer 展示了工具调用学习的可行性，但企业客服中的权限控制、失败回退、审计与幂等问题超出论文范围。
- 论文中的工具集合偏通用，不等于真实客服中的 CRM、订单、支付、工单系统接入难度。

## 关联页面

- 概念：[Toolformer](../../wiki/concepts/Toolformer.md)
- 主题：[AI 智能问答与智能客服](../../wiki/topics/AI%20%E6%99%BA%E8%83%BD%E9%97%AE%E7%AD%94%E4%B8%8E%E6%99%BA%E8%83%BD%E5%AE%A2%E6%9C%8D.md)

---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Roziere et al. - 2023 - Code Llama Open Foundation Models for Code

## TL;DR（快速导读）

Code Llama 在 Llama 2 基础上继续学习代码，并加入长上下文和补全中间代码的训练目标。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

补全一个函数与修改整个代码库的任务不同；模型能写出代码，还需要测试才能判断它是否正确。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Roziere et al. - 2023 - Code Llama Open Foundation Models for Code.pdf
- 全文文本：../../raw/text/Roziere et al. - 2023 - Code Llama Open Foundation Models for Code.md
- 作者：Roziere et al.
- 年份：2023
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

它是面向编程任务的专门分支，补全既可从前文继续，也可利用前后文填入缺失部分。任务能力需按具体成员和输入方式检查；能生成代码不等于代码已经运行正确。

## 关键事实

- **C1**：CodeLlama以Llama2初始化，原尺寸7B/13B/34B，保存更新版本还包含70B。
- **C2**：7B/13B/70B在此版本训练infilling，34B没有该目标。
- **C3**：FIM重排prefix/middle/suffix以自回归训练代码补全。
- **C4**：使用16K长文微调并测试至100K，代码任务以HumanEval/MBPP等分别评估。

## 争议与不确定点

- HumanEval pass率不能替代真实代码库测试通过与维护性。
- 长文perplexity/检索与复杂跨文件推理有不同边界。

## 关联页面

- 概念：[Llama 家族](../../wiki/concepts/Llama%20家族.md)
- 概念：[Llama 2](../../wiki/concepts/Llama%202.md)
- 概念：[Code Llama](../../wiki/concepts/Code%20Llama.md)
- 主题：[LLM预训练](../../wiki/topics/LLM%20预训练.md)
- [Hugo Touvron](../authors/Hugo%20Touvron.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **benchmark**：评测基准：特定数据、任务与规则的组合，分数只在这些条件下成立。

## 方法与实验解读

领域继续预训练提供代码知识，FIM改变缺口补全形式，instruction数据改变交互习惯。小型Python专用在部分代码题胜过通用大模型，只说明任务专门化有效；不能推定仓库修改、依赖解析和软件工程任务全部相同。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Roziere%20et%20al.%20-%202023%20-%20Code%20Llama%20Open%20Foundation%20Models%20for%20Code.md#source-section-6 ) | 70B后续版本，不混作2023首发。 |
| C2 | [原文]( ../../raw/text/Roziere%20et%20al.%20-%202023%20-%20Code%20Llama%20Open%20Foundation%20Models%20for%20Code.md#source-section-6 ) | 按型号能力区分。 |
| C3 | [原文]( ../../raw/text/Roziere%20et%20al.%20-%202023%20-%20Code%20Llama%20Open%20Foundation%20Models%20for%20Code.md#source-section-11 ) | infilling与纯续写不等价。 |
| C4 | [原文]( ../../raw/text/Roziere%20et%20al.%20-%202023%20-%20Code%20Llama%20Open%20Foundation%20Models%20for%20Code.md#source-section-12 ) | 训练长度与外推效果分开。 |

## 核证范围

核读型号/版本、data、infilling、长文微调与评测/消融。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

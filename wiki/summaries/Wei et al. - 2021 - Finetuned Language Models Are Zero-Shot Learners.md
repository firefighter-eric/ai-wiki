---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Wei et al. - 2021 - Finetuned Language Models Are Zero-Shot Learners

## TL;DR（快速导读）

FLAN 通过多任务自然语言指令微调，改善模型在未见任务上的零样本表现，展示了指令数据的迁移价值。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

模型在预训练后不一定能理解各种任务要求。论文把已有任务写成指令形式，用多任务训练强化这种接口。需要区分是否见过测试任务，以及指令模板和任务覆盖对结果的影响。

## 具体怎么理解

训练时做过翻译与问答，测试时给出新的任务说明；如果不再提供示例仍能完成，才是在观察零样本迁移。

## 关键事实

- **C1**：FLAN 把 62 个公开文本数据集改写为自然语言指令，按 12 类任务簇组织，每个数据集编写十种模板并加入部分反向任务。
- **C2**：零样本评估按整类任务簇留出，而不仅是留出一个数据集；测试任务所属簇不能出现在指令微调中。
- **C3**：主要模型为 137B 的 LaMDA-PT；指令微调提高部分留出任务表现，但在更小的 8B 及以下模型设置中会降低留出任务表现。
- **C4**：commonsense 与 coreference 的句子补全任务只有七项中的三项受益，说明原始 LM 目标已接近任务时，指令微调未必增加优势。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Wei%20et%20al.%20-%202021%20-%20Finetuned%20Language%20Models%20Are%20Zero-Shot%20Learners.pdf)
- 全文文本：[打开全文文本](../../raw/text/Wei%20et%20al.%20-%202021%20-%20Finetuned%20Language%20Models%20Are%20Zero-Shot%20Learners.md)
- 作者：Wei et al.
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Wei%20et%20al.%20-%202021%20-%20Finetuned%20Language%20Models%20Are%20Zero-Shot%20Learners.html)

## 争议与不确定点

- 任务簇分类含主观选择；不同任务可能共享能力，留出任务簇仍不等于完全独立。
- 论文只覆盖较短的单句指令，不能直接证明复杂多步 agent 工作流能力。
- 预训练污染分析未发现明显优势来源，但无法把没有检测到污染当作绝对无污染。

## 关联页面

- 主题：[LLM预训练](../topics/LLM%20预训练.md)
- 综合：暂无

## 方法与实验解读

训练时用‘判断这两句话是否相互矛盾’等指令把原数据变成问答式任务，测试时换成训练没有覆盖的任务簇。真正要观察的是模型能否迁移‘照指令做事’的方式，而不是是否记住一张训练表。任务模板、留出方法与模型规模一起决定这项实验的含义。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Wei%20et%20al.%20-%202021%20-%20Finetuned%20Language%20Models%20Are%20Zero-Shot%20Learners.md#source-section-5 ) | 这是原始 FLAN 设置，不能与后续 FLAN 系列的更大任务合集混用。 |
| C2 | [原文]( ../../raw/text/Wei%20et%20al.%20-%202021%20-%20Finetuned%20Language%20Models%20Are%20Zero-Shot%20Learners.md#source-section-6 ) | 零样本指未在指令微调阶段见该任务簇，不保证预训练从未接触类似信息。 |
| C3 | [原文]( ../../raw/text/Wei%20et%20al.%20-%202021%20-%20Finetuned%20Language%20Models%20Are%20Zero-Shot%20Learners.md#source-section-12 ) | 该规模消融受训练配方与任务分割约束，不能推成小模型普遍不适合指令微调。 |
| C4 | [原文]( ../../raw/text/Wei%20et%20al.%20-%202021%20-%20Finetuned%20Language%20Models%20Are%20Zero-Shot%20Learners.md#source-section-9 ) | 按具体任务看收益，不能只看总体平均。 |

## 核证范围

核对 §2 数据模板与留出、§2.4 模型设置、§3 各任务结果、§4.2 规模消融、§6 讨论及污染检查说明。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

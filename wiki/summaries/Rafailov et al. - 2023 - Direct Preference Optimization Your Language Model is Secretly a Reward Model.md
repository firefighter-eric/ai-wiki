---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
source_id: arxiv:2305.18290
---
# Rafailov et al. - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model

## TL;DR（快速导读）

DPO 直接利用“偏好回答与不偏好回答”的成对数据训练语言模型，简化显式奖励模型和在线强化学习的部分流程。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

传统偏好对齐常先拟合奖励，再进行策略优化。DPO 通过目标变换，把偏好信息直接作用于模型概率。它仍需要合适的偏好数据和参考条件；离线训练的简化不代表自动获得所有在线探索能力。

## 具体怎么理解

同一问题有两个回答，标注者选出更好的一个；训练推动模型更倾向这个回答，并约束相对变化。

## 关键事实

- **C1**：DPO 从带 KL 约束的奖励最大化目标出发，将奖励写成策略与参考策略的对数概率比；在 Bradley–Terry 成对偏好模型下，配分函数在奖励差中抵消，得到直接训练策略的分类损失。
- **C2**：训练使用同一提示的偏好答案与非偏好答案，优化两者相对于参考模型的对数概率比差，不需要单独拟合显式奖励模型或运行 PPO 训练循环。
- **C3**：实验涵盖情感控制、TL;DR 摘要与单轮对话；摘要和对话主要以 GPT-4 对参考答案的胜率评价，并对摘要做了人类判断核对。
- **C4**：论文讨论指出分布外泛化、奖励过度优化和更大规模模型仍需进一步研究，原实验评估模型规模最高约 6B。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.pdf)
- 全文文本：[打开全文文本](../../raw/text/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)
- 作者：Rafailov et al.
- 年份：2023
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.html)

## 争议与不确定点

- 偏好标签可能含噪声、覆盖不足或长度偏好；训练目标不会自行消除这些偏差。
- 报告使用 GPT-4 胜率的部分实验需连同评判提示与人类验证一起读。
- 同 arXiv ID 的另一归档是同一研究，不应重复计作独立证据。

## 关联页面

- 主题：[LLM RL](../topics/LLM%20RL.md)
- 综合：暂无

## 这里的术语是什么意思

- **DPO**：直接偏好优化：用成对偏好数据直接调整模型概率，简化部分奖励训练流程。

## 方法与实验解读

对一个问题准备较好与较差两个回答，DPO 不直接要求把较好回答概率推到最大，而是增加两者相对于参考模型的概率比差。参考模型和 β 约束改变幅度，因此它与只对好答案做 SFT 有不同目标。论文的简化发生在偏好优化训练环节，不代表数据收集、评价与上线选择也被消除。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md#source-section-6 ) | 推导依赖给定偏好模型与参考策略；不是任意奖励、任意标注下的无条件等价。 |
| C2 | [原文]( ../../raw/text/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md#source-section-6 ) | 仍需要偏好数据、参考模型与 β 等训练选择。 |
| C3 | [原文]( ../../raw/text/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md#source-section-19 ) | GPT-4 提示会影响胜率，尤其长度偏好；自动评判不是普适质量真值。 |
| C4 | [原文]( ../../raw/text/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md#source-section-20 ) | 原报告优于或接近 PPO 的结果受任务、规模与超参数设置约束。 |

## 核证范围

核对 §4 目标与公式 4–7、§6 任务与评判流程、§6.4 人类核对、§7 局限。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

## 来源归档关系

本页作为该论文的主阅读入口；另一归档是 [Rafailov, Mitchell, Jul - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model](Rafailov%2C%20Mitchell%2C%20Jul%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)。按 arxiv:2305.18290 合并计数；不同保存版本可用于核对修订，不能当作独立实验或独立来源复现。

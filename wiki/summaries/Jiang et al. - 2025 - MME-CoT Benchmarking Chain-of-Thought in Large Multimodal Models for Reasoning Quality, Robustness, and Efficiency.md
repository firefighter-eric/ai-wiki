---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Jiang et al. - 2025 - MME-CoT Benchmarking Chain-of-Thought in Large Multimodal Models for Reasoning Quality, Robustness, and Efficiency

## TL;DR（快速导读）

MME-CoT 分别评估多模态模型推理过程的质量、稳健性和效率，研究“让模型多想几步”是否总有帮助。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

语言模型中的思维链收益不能直接套到看图模型。基准覆盖数学、科学、OCR、逻辑和场景等问题，从多个角度观察推理表现。需要同时检查答案、过程和额外计算，而不是用更长输出代替更好推理。

## 具体怎么理解

模型先写出一大段解释再答错，并不算推理更强；同一道图像题还应比较直接回答与分步回答。

## 关键事实

- **C1**：评测多模态 CoT 时同时考察推理质量、鲁棒性与效率，不只看最终答案。
- **C2**：基准包含推理与感知问题，用于检查 CoT 是否在需要和不需要推理的场景都合适。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Jiang%20et%20al.%20-%202025%20-%20MME-CoT%20Benchmarking%20Chain-of-Thought%20in%20Large%20Multimodal%20Models%20for%20Reasoning%20Quality%2C%20Robustness%2C%20and%20Efficiency.pdf)
- 全文文本：[打开全文文本](../../raw/text/Jiang%20et%20al.%20-%202025%20-%20MME-CoT%20Benchmarking%20Chain-of-Thought%20in%20Large%20Multimodal%20Models%20for%20Reasoning%20Quality%2C%20Robustness%2C%20and%20Efficiency.md)
- 作者：Jiang et al.
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Jiang%20et%20al.%20-%202025%20-%20MME-CoT%20Benchmarking%20Chain-of-Thought%20in%20Large%20Multimodal%20Models%20for%20Reasoning%20Quality%2C%20Robustness%2C%20and%20Efficiency.html)

## 争议与不确定点

- 过程评分依赖标注与评判规则，不能自动证明真实内部推理。
- 模型版本和生成设置影响质量、鲁棒性与成本。

## 关联页面

- 主题：[LLM RL](../topics/LLM%20RL.md)
- 综合：暂无
- [AI 能力评测：任务、过程与预测](../comparisons/AI%20%E8%83%BD%E5%8A%9B%E8%AF%84%E6%B5%8B%EF%BC%9A%E4%BB%BB%E5%8A%A1%E3%80%81%E8%BF%87%E7%A8%8B%E4%B8%8E%E9%A2%84%E6%B5%8B.md)：把本篇方法放到相关任务与比较条件中阅读。

## 这里的术语是什么意思

- **OCR**：文字识别：从图像读取文字；整页任务还需处理布局和阅读顺序。

## 方法与实验解读

MME-CoT 提醒读者，增加推理文字可能改善答案，也可能增加无关步骤和成本。评估应同时看感知信息是否正确、推理是否支持结论，以及额外 token 是否带来可测收益。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/pdf/Jiang%20et%20al.%20-%202025%20-%20MME-CoT%20Benchmarking%20Chain-of-Thought%20in%20Large%20Multimodal%20Models%20for%20Reasoning%20Quality%2C%20Robustness%2C%20and%20Efficiency.pdf#page=1 ) | 回答正确与推理过程可信、成本合理需分开 |
| C2 | [原文]( ../../raw/pdf/Jiang%20et%20al.%20-%202025%20-%20MME-CoT%20Benchmarking%20Chain-of-Thought%20in%20Large%20Multimodal%20Models%20for%20Reasoning%20Quality%2C%20Robustness%2C%20and%20Efficiency.pdf#page=4 ) | 基准类别不等于完整覆盖现实多模态任务 |

## 核证范围

核对 PDF 第 1 页评测目标、第 3–4 页任务组成；本页不引用抽取顺序混乱的图表数字作为模型排名。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

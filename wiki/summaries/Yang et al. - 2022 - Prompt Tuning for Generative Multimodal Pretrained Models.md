---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Yang et al. - 2022 - Prompt Tuning for Generative Multimodal Pretrained Models

## TL;DR（快速导读）

这篇工作将提示微调用于生成式多模态预训练模型，研究只训练少量提示参数能否适配理解和生成任务。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

提示微调在语言模型上较常见，本文探索其在统一多模态序列到序列模型中的迁移。关键是提示如何进入模型、任务数据和可训练参数范围。低参数成本与最终任务效果需要一起评估。

## 具体怎么理解

同一个多模态底座接不同可学习提示，以适应图像描述或其他任务；底座能力仍限制可达到的效果。

## 关键事实

- **C1**：在 OFA 类生成式多模态模型上训练可学习 prefix，冻结主体模型。
- **C2**：作者明确指出 prompt tuning 还不能替代完整微调。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Yang%20et%20al.%20-%202022%20-%20Prompt%20Tuning%20for%20Generative%20Multimodal%20Pretrained%20Models.pdf)
- 全文文本：[打开全文文本](../../raw/text/Yang%20et%20al.%20-%202022%20-%20Prompt%20Tuning%20for%20Generative%20Multimodal%20Pretrained%20Models.md)
- 作者：Yang et al.
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Yang%20et%20al.%20-%202022%20-%20Prompt%20Tuning%20for%20Generative%20Multimodal%20Pretrained%20Models.html)

## 争议与不确定点

- 更快处理样本不一定带来相同总收敛成本。
- 结论依赖基础模型规模、任务与前缀设置。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无

## 方法与实验解读

这篇工作将多层可学习前缀用于多模态任务适配。节省的是更新参数，但仍需执行主体模型前向和反向相关计算；训练速度、显存和任务效果要各自测量。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Yang%20et%20al.%20-%202022%20-%20Prompt%20Tuning%20for%20Generative%20Multimodal%20Pretrained%20Models.md#source-section-10 ) | 软前缀训练与人工写提示词不同 |
| C2 | [原文]( ../../raw/text/Yang%20et%20al.%20-%202022%20-%20Prompt%20Tuning%20for%20Generative%20Multimodal%20Pretrained%20Models.md#source-section-28 ) | 参数少不保证所有任务都达到同样质量 |

## 核证范围

核对 Basic Implementation、§5 讨论及训练效率补充说明。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

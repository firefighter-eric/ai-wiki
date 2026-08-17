---
type: summary
status: refined
---
# OLMo Team - 2025 - 2 OLMo 2 Furious

## 来源信息

- 类型：技术报告 / arXiv 论文
- arXiv：https://arxiv.org/abs/2501.00656
- 原始 PDF：../../raw/pdf/Team OLMo - 2025 - 2 OLMo 2 Furious.pdf
- 发布页快照：../../raw/html/Team OLMo - 2025 - 2 OLMo 2 Furious.html
- 全文文本：../../raw/text/Team OLMo - 2025 - 2 OLMo 2 Furious.md
- 作者：OLMo Team
- 年份：2025
- 状态：已精读与优化器、训练稳定性相关部分

## 摘要

`OLMo 2` 是 fully open 训练路线的代表之一。其报告明确使用 AdamW，并把优化器细节本身当作训练稳定性消融对象：把 `ε` 从 `10^-5` 降到 `10^-8`，并从 weight decay 中排除 embedding，而不是切换到 Muon。

## 关键事实

- OLMo 2 明确使用 AdamW；`ε=10^-8` 让训练早期更新更大，作者观察到梯度范数更快进入稳定区间。
- weight decay 系数为 `0.1`，但 embedding 被排除；作者认为对 embedding 衰减过强会造成 embedding norm 过小，并放大早期层梯度。
- 该例说明“使用 AdamW”仍包含参数组、`ε`、学习率计划和稳定化措施等关键 recipe 差异，不能仅凭优化器名称比较模型。

## 争议与不确定点

- OLMo 2 与 Muon 系列并非在同一数据、架构和预算下的直接 optimizer ablation，因此它只能证明先进开放模型仍在使用 AdamW，不能用于判定二者谁更优。
- 报告中的稳定性结论依赖 OLMo 架构和训练栈；`ε=10^-8` 不是每个模型的通用最优值。

## 关联页面

- 概念：[OLMo 2](../concepts/OLMo%202.md)
- 概念：[Muon](../concepts/Muon.md)
- 对比：[Muon 与 AdamW](../comparisons/Muon%20与%20AdamW.md)
- 主题：[LLM 预训练](../topics/LLM%20预训练.md)

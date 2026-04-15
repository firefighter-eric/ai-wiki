# Ouyang et al. - 2022 - Training language models to follow instructions with human feedback

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Ouyang et al. - 2022 - Training language models to follow instructions with human feedback.pdf
- 原始 HTML：../../raw/html/Ouyang et al. - 2022 - Training language models to follow instructions with human feedback.html
- 全文文本：../../raw/text/Ouyang et al. - 2022 - Training language models to follow instructions with human feedback.md
- 作者：Ouyang et al.
- 年份：2022
- 状态：已整理

## 摘要

InstructGPT 论文是“模型更大”与“模型更符合用户意图”之间分离的标志性来源。作者明确指出，语言模型即使能力很强，也可能不 helpful、不 truthful、不 harmless；因此需要用 demonstrations、偏好排序和 RLHF 把模型行为塑形成更符合用户意图的交互代理。对智能问答与智能客服来说，这篇论文提供的是服务型 AI 的核心行为框架。

## 关键事实

- 论文把后训练流程拆成三层：基于人工 demonstrations 的 SFT、基于偏好排序的 reward model，以及基于该 reward 的 RLHF。
- 训练数据不仅包含标注员撰写的 prompts，也包含 API 用户提交的 prompts，这使它更接近真实交互分布，而不是纯 benchmark 任务。
- 文中一个重要结果是，1.3B 的 InstructGPT 在人工偏好评估中可以优于 175B GPT-3，说明行为对齐与参数规模是不同维度。
- 论文同时报告 truthfulness 改善与 toxic output 降低，表明对齐不仅是“更礼貌”，也是“更接近可部署交互系统”。
- 在智能客服语境中，这篇论文支撑的不是知识检索层，而是“回复风格、用户意图遵循、拒答边界与服务型行为”的塑形层。

## 争议与不确定点

- InstructGPT 解决的是 general instruction-following 与 preference alignment，不直接处理企业知识库接入和工单闭环。
- helpful / truthful / harmless 是通用对齐目标；企业客服通常还需要额外的品牌语气、政策一致性和业务规则约束。

## 关联页面

- 概念：[InstructGPT](../../wiki/concepts/InstructGPT.md)
- 概念：[RLHF](../../wiki/concepts/RLHF.md)
- 主题：[AI 智能问答与智能客服](../../wiki/topics/AI%20%E6%99%BA%E8%83%BD%E9%97%AE%E7%AD%94%E4%B8%8E%E6%99%BA%E8%83%BD%E5%AE%A2%E6%9C%8D.md)
- 主题：[指令对齐与 post-training](../../wiki/topics/指令对齐与%20post-training.md)

# LLM 预训练

## 页面状态

- 状态：正式 topic
- 事实基座：`wiki/summaries/` 优先

## 主题定义

本页讨论语言模型在 post-training 之前如何通过大规模预训练获得通用能力。这里的重点是预训练目标、规模化规律、模型家族与训练工程，而不是 RLHF、DPO 等行为对齐方法。当前知识库中，LLM 的骨干叙事也统一收敛到本页：大规模预训练如何带来通用能力、compute-optimal 视角如何修正“只扩参数”的早期叙述，以及为什么后训练应被视为独立于能力底座的下一阶段。

## 核心问题

- 通用语言能力主要如何从大规模预训练中获得。
- dense scaling、compute-optimal 修正与 sparse scaling 之间的关系是什么。
- 开放模型家族的技术差异主要体现在哪些训练与架构选择上。
- 预训练阶段的能力边界与后训练阶段的行为改写边界应如何切分。

## 主线脉络 / 方法分层

- dense scaling 主线：`Brown et al. 2020` 与 `Chowdhery et al. 2022` 支撑了“继续扩大模型与训练系统能持续提升 few-shot 能力”这一方向，GPT-3 与 PaLM 是其中最典型的 dense scaling 代表。
- compute-optimal 修正：`Hoffmann et al. 2022` 把讨论从“能否继续变大”转向“在固定 FLOPs 下如何平衡参数量与 token 数”，Chinchilla 因此成为 dense scaling 的关键纠偏点。
- 能力形成与行为塑形的阶段切分：当前证据更支持把 LLM 主线拆成“预训练获得通用能力”与“post-training 改写交互行为”两个阶段。`Ouyang et al. 2022` 之所以重要，不是因为它改变了预训练规律，而是因为它说明 strong base model 仍不足以自动变成更 helpful、truthful、harmless 的交互系统。
- 开放模型家族阶段：`Touvron et al. 2023`、`Bai et al. 2023`、`Dubey et al. 2024` 和 `Unknown 2024 DeepSeek-V3` 说明预训练主线已从闭源标杆转向多个开放家族并行推进，关注点同时包括多语言、代码、长上下文与工程可复现性。LLaMA/Llama 2/Llama 3 构成了其中最连续的一条开放家族线。
- sparse scaling 与高效训练：`DeepSeek-V3` 与 `MoE` 路线表明预训练不再只沿着 dense Transformer 扩张，而是转向“总参数更大、单 token 激活更少”的效率设计。

## 关键争论与分歧

- 更大是否仍是最主要驱动力：当前证据表明规模仍重要，但 `Hoffmann 2022` 已经否定了“只扩参数即可”的简单叙述。
- dense 与 sparse 哪条更代表未来主线：从现有 summary 看，dense scaling 仍是能力讨论的基础语言，但 sparse/MoE 已经成为工程与开源竞争中的现实路线。
- 预训练与后训练如何分界：`DeepSeek-V3` 这类报告同时覆盖预训练、SFT 与 RL，说明现实系统往往跨阶段联合报告，但知识组织上仍需把“能力底座”与“行为对齐”分开。
- LLM 主线是否应只写预训练：当前页承担的是“能力骨干”的总入口，因此需要保留从 GPT-3 到 Chinchilla、再到“为何 post-training 成为独立阶段”的连续叙事；但具体对齐与 RL 方法仍应下沉到 `指令对齐与 post-training` 与 `LLM RL`。
- 开放模型是否主要是发布策略差异：当前证据不足以把开放模型仅视为分发差异，它们在多语言、工具支持、上下文长度与训练效率上也形成了真实技术分化。

## 证据基础

- [Brown et al. - 2020 - Language models are few-shot learners](../../wiki/summaries/Brown%20et%20al.%20-%202020%20-%20Language%20models%20are%20few-shot%20learners.md)
- [Chowdhery et al. - 2022 - PaLM Scaling Language Modeling with Pathways](../../wiki/summaries/Chowdhery%20et%20al.%20-%202022%20-%20PaLM%20Scaling%20Language%20Modeling%20with%20Pathways.md)
- [Hoffmann et al. - 2022 - Training Compute-Optimal Large Language Models](../../wiki/summaries/Hoffmann%20et%20al.%20-%202022%20-%20Training%20Compute-Optimal%20Large%20Language%20Models.md)
- [Touvron et al. - 2023 - LLaMA Open and Efficient Foundation Language Models](../../wiki/summaries/Touvron%20et%20al.%20-%202023%20-%20LLaMA%20Open%20and%20Efficient%20Foundation%20Language%20Models.md)
- [Touvron et al. - 2023 - Llama 2 Open Foundation and Fine-Tuned Chat Models](../../wiki/summaries/Touvron%20et%20al.%20-%202023%20-%20Llama%202%20Open%20Foundation%20and%20Fine-Tuned%20Chat%20Models.md)
- [Roziere et al. - 2023 - Code Llama Open Foundation Models for Code](../../wiki/summaries/Roziere%20et%20al.%20-%202023%20-%20Code%20Llama%20Open%20Foundation%20Models%20for%20Code.md)
- [Bai et al. - 2023 - Qwen Technical Report](../../wiki/summaries/Bai%20et%20al.%20-%202023%20-%20Qwen%20Technical%20Report.md)
- [Dubey et al. - 2024 - The Llama 3 Herd of Models](../../wiki/summaries/Dubey%20et%20al.%20-%202024%20-%20The%20Llama%203%20Herd%20of%20Models.md)
- [Unknown - 2024 - DeepSeek-V3 Technical Report](../../wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)

## 代表页面

- [GPT-3](../concepts/GPT-3.md)
- [PaLM](../concepts/PaLM.md)
- [Chinchilla](../concepts/Chinchilla.md)
- [指令对齐与 post-training](./指令对齐与%20post-training.md)
- [LLM RL](./LLM%20RL.md)
- [T5](../concepts/T5.md)
- [Switch Transformer](../concepts/Switch%20Transformer.md)
- [OPT](../concepts/OPT.md)
- [mT5](../concepts/mT5.md)
- [Qwen](../concepts/Qwen.md)
- [Llama](../concepts/Llama.md)
- [LLaMA](../concepts/LLaMA.md)
- [Llama 2](../concepts/Llama%202.md)
- [Code Llama](../concepts/Code%20Llama.md)
- [Llama 3](../concepts/Llama%203.md)
- [Gemma 3](../concepts/Gemma%203.md)
- [MiniCPM](../concepts/MiniCPM.md)
- [DeepSeek-V3](../concepts/DeepSeek-V3.md)
- [MoE](../concepts/MoE.md)
- [Scaling 与 compute-optimal training](./Scaling%20与%20compute-optimal%20training.md)
- [Qwen 系列演进](../timelines/Qwen%20系列演进.md)

## 未解决问题

- 当前知识库尚未建立 `dense vs MoE` 的独立比较页，因此 sparse scaling 的优劣仍只停留在主题级概述。
- 预训练数据质量、token 去重、长上下文训练成本等问题，在现有 summary 中仍未形成细粒度证据链。
- Qwen、Llama 3、DeepSeek-V3 之间更系统的横向比较仍有待 comparison 层承接。
- 预训练骨干与 post-training 分界目前已有稳定主线，但“tool use、多模态与 agent 化”是否应被视为同一主干的自然延展，仍需要更多 summary 支撑。

## 关联页面

- [Scaling 与 compute-optimal training](./Scaling%20与%20compute-optimal%20training.md)
- [指令对齐与 post-training](./指令对齐与%20post-training.md)
- [LLM RL](./LLM%20RL.md)
- [DeepSeek](../concepts/DeepSeek.md)
- [Qwen 系列演进](../timelines/Qwen%20系列演进.md)

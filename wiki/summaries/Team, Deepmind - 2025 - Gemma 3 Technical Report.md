---
type: summary
status: auto
---
# Team, Deepmind - 2025 - Gemma 3 Technical Report

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Team, Deepmind - 2025 - Gemma 3 Technical Report.pdf
- 全文文本：../../raw/text/Team, Deepmind - 2025 - Gemma 3 Technical Report.md
- 作者：Team, Deepmind
- 年份：2025
- 状态：已核对优化器披露范围；其余部分仍待精读

## 摘要

2025-03-12 Gemma 3 Technical Report Gemma Team, Google DeepMind1 We introduce Gemma 3, a multimodal addition to the Gemma family of lightweight open models, ranging in scale from 1 to 27 billion parameters. This version introduces vision understanding abilities, a wider coverage of languages and longer context – at least 128K tokens. We also change the architecture of the model to reduce the KV-cache memory that tends to explode with long context. This is achieved by increasing the ratio of local to global attention layers, and keeping the span on local attention short. The Gemma 3 models are trained with distillation and achieve superior performance to Gemma 2 for both pre-trained and instruction finetuned versions. In particular, our novel post-training recipe significantly improves the math, chat, instruction-following and multilingual abilities, making Gemma3- 4B-IT competitive with 

## 当前 ingest 判断

- 当前页面为批量重建后的统一来源页，目标是先把全部 PDF 纳入知识库可引用范围。
- 摘要内容来自 `raw/text/` 自动抽取结果，后续需要人工或 LLM 精修。
- 候选主题暂按文件名与摘要关键词自动归类，允许后续调整。

## 关键事实

- 报告只明确说明 optimizer state 通过 ZeRO-3 分片，没有披露优化器具体是 AdamW、Muon 或其他方法；横向研究中应标为“未披露”。
- 已存在可读全文文本，可直接从 `raw/text/Team, Deepmind - 2025 - Gemma 3 Technical Report.md` 继续做深入整理。
- 当前尚未对方法细节、实验设置和局限做系统提炼。
- 若该来源对主题主干重要，下一步应提升为精修版来源页。

## 争议与不确定点

- 自动抽取摘要可能存在 PDF 文本切分误差。
- 主题归类是启发式结果，不等于最终主题归属。
- 当前页面不应被视为最终综述，只应作为后续精修入口。
- “optimizer state 使用 ZeRO-3”描述的是分片方式，不能据此反推优化器算法。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 对比：[Muon 与 AdamW](../comparisons/Muon%20与%20AdamW.md)

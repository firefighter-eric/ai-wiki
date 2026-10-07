---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# vLLM Project - 2026 - vLLM V1 Guide

## TL;DR（快速导读）

vLLM V1 指南说明按统一词元预算调度请求，并记录重构后的功能与兼容边界。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

多个请求长度不同，系统要决定哪些先处理、保存多少缓存及如何分配 GPU；模型权重相同也会因服务系统产生不同表现。

## 来源信息

- 类型：官方文档 / V1 migration 与 feature support guide
- 发布者：vLLM Project
- 原始 HTML：[vLLM V1 Guide](../../raw/html/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.html)
- 全文文本：[vLLM V1 Guide](../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md)
- 官方页面：[vLLM V1](https://docs.vllm.ai/en/stable/usage/v1_guide/)
- 快照日期：2026-08-04
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

每轮用请求与待处理词元数组织计算，让分块提示处理、前缀缓存和推测生成共享调度表示。指南也列出支持状态；它是持续更新的文档，部署时应核对实际版本，不能照搬快照中的所有结论。

### 方法与背景细节

这份 guide 描述 vLLM 从 V0 到 V1 的核心重构和 2026-08-04 时点的功能边界。V1 保留既有 models、GPU kernels 与 utilities，但重写 scheduler、`KV cache manager`、worker、sampler 和 API server，目标是形成更简单、模块化、低 CPU overhead 且默认开启关键优化的统一架构。该快照已宣布 V0 fully deprecated；不过页面也明确自称 living guide，支持状态仍会随 PR / RFC 持续变化。

V1 scheduler 的关键抽象是统一 token budget：它不再先把工作严格分成 prefill 与 decode 两类，而是在每轮预算内用 `{request_id: num_tokens}` 表示各请求本轮要处理的 token 数。这样 chunked prefill、prefix caching 与 speculative decoding 可以共享同一调度表示。调度策略既支持 `FCFS`，也支持 priority-based scheduling；后者以请求 priority 排序，同 priority 时仍以 FCFS 打破平局。

这份文档同时是一份 compatibility checklist：chunked prefill 默认尽可能开启，CUDA graph capture 比 V0 占更多内存，默认 logprobs 语义变为 logits post-processing 之前的 raw output；部分功能处于 functional 或 in progress，另有 `best_of`、per-request logits processors、GPU↔CPU `KV cache` swapping 与 request-level structured-output backend 被明确移除。页面宣称 V1 尤其在 long-context 场景有显著性能改进，但对应 performance benchmark 仍标为 “To be added”，因此不能把该说法当作可复核的性能证据。

## 关键事实

- **C1**：**重构范围**：V1 复用成熟的 model implementations、GPU kernels 与 utilities，同时重构 scheduler、`KV cache manager`、worker、sampler 和 API server。
- **C2**：**设计目标**：官方列出的目标包括易修改的 modular codebase、near-zero CPU overhead、把关键优化合进统一架构，以及尽量 zero-config 地默认启用优化；这些是项目目标，不等于该页面已逐项 benchmark 验证。
- **C3**：**V0 状态**：在这份 2026-08-04 stable 快照中，V0 已被标为 fully deprecated；V0 可用而 V1 不可用的 use case 被引导到 GitHub 或 vLLM Slack 反馈。
- **C4**：**unified scheduler**：scheduler 以 `{request_id: num_tokens}` 的简单字典，在固定 token budget 下动态决定每条请求本轮处理多少 token，不要求 prefill 与 output/decode tokens 进入两套严格分离的调度路径。
- **C5**：**调度能力的组合**：同一 token-budget 表示被用于组合 chunked prefills、prefix caching 与 speculative decoding，而不是分别维护互不兼容的 feature-specific scheduler。
- **C6**：**调度策略**：`--scheduling-policy` 可选择 `FCFS` 或 priority-based scheduling；priority 相同时使用 FCFS 作为 tie-breaker。
- **C7**：**chunked prefill**：V1 在条件允许时默认启用；V0 则会依据模型特性有条件开启。这是默认行为变化，部署迁移时不能假设两代配置语义一致。
- **C8**：**CUDA graphs**：文档明确 V1 的 CUDA graph capture 比 V0 占用更多 memory，但没有在本页给出统一增量数字。
- **C9**：**默认 logprobs 语义**：V1 默认在 temperature、penalties、bad-words processor、`top_k / top_p` 等 logits post-processing 之前返回模型 raw output 对应的 logprobs，因此不一定等于最终 sampling distribution。
- **C10**：**logprobs modes**：`--logprobs-mode` 支持 `raw_logprobs`（默认）、`processed_logprobs`、`raw_logits`、`processed_logits`；raw / processed 的分界是是否经过全部 logits processors。
- **C11**：**prompt logprobs + prefix cache**：接口组合被标为 functional，但当请求需要 prompt logprobs 时，engine 会忽略 prefix cache 并重新 prefill 完整 prompt，因为 V1 不缓存 logprobs。
- **C12**：**硬件支持快照**：NVIDIA、AMD、Intel GPU、TPU 与 CPU 在页面中均标为 functional；Ascend、Spyre、Gaudi、OpenVINO 等更多平台通过各自 plugins 扩展，需查对应 repository。
- **C13**：**模型支持快照**：decoder-only、pooling、Mamba、multimodal 被标为 functional；Whisper 获得 native encoder-decoder support，其他 encoder-decoder models 不在 core support matrix 内，可通过 plugin pattern 扩展。
- **C14**：**pooling 边界**：last-pooling models 新支持 prefix caching 与 chunked prefill；文档仍在为更多 pooling categories 扩展这两项能力。
- **C15**：**Mamba 边界**：Mamba-1、Mamba-2、attention-Mamba hybrid 与文档列举的其他 hybrid mechanisms 可运行，但该快照明确这些模型均尚不支持 prefix caching。
- **C16**：**functional features**：Prefix Caching、Chunked Prefill、LoRA、Logprobs Calculation、FP8 KV Cache、Spec Decode、Prompt Logprobs with Prefix Caching、Structured Output Alternative Backends 均为绿色 functional。
- **C17**：**in-progress feature**：Concurrent Partial Prefills 在该快照中仍标为 in progress。
- **C18**：**移除 `best_of`**：官方理由是使用有限；该 sampling feature 不再属于 V1。
- **C19**：**移除 per-request logits processors**：V1 改为支持服务启动时配置的 global logits processors，不再允许每个请求传入自定义 processing function。
- **C20**：**移除 GPU↔CPU KV swapping**：V1 的 simplified core architecture 不再依靠该机制处理 request preemption，这与 2023 vLLM 论文把 swapping 列为恢复路径的设计不同。
- **C21**：**structured output 变化**：request-level backend 选择被移除；`outlines`、`guidance` 等 alternative backends 及 fallback 仍被支持。

## 争议与不确定点

- “nearzeroCPUoverhead”是项目目标，页面benchmark待补处不能自填数字。
- Mamba/hybrid的prefixcache限制依快照，后续新版本可修正。

## 关联页面

- 概念：[vLLM](../concepts/vLLM.md)
- 概念：[PagedAttention](../concepts/PagedAttention.md)
- 比较：[SGLang 与 vLLM 架构对比](../comparisons/SGLang%20与%20vLLM%20架构对比.md)
- 官方架构：[Architecture Overview](./vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md)
- 官方设计：[Automatic Prefix Caching](./vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md)
- 原始论文：[Kwon et al. - 2023 - Efficient Memory Management for Large Language Model Serving with PagedAttention](./Kwon%20et%20al.%20-%202023%20-%20Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention.md)

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **KV cache**：键值缓存：保存已经处理过的位置表示，生成新内容时可复用，避免全部重算。
- **prefill**：提示计算阶段：先处理输入提示，再开始逐步生成输出。
- **decode**：解码阶段：利用已有输入与生成历史，产生后续输出。
- **scheduler**：调度器：决定请求何时进入计算、每次处理多少，以及如何共享资源。
- **benchmark**：评测基准：特定数据、任务与规则的组合，分数只在这些条件下成立。

## 方法与实验解读

统一tokenbudget组织prefill/decode/features，降低多套调度路径的维护成本。版本迁移还改变logprobs、feature支持与preemption方式，不能把2023论文直接当当前运行说明。列为functional的组合也有重算例外，尤其promptlogprobs会忽略prefixcache。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-1 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C2 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-1 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C3 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-1 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C4 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-8 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C5 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-8 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C6 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-8 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C7 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-3 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C8 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-4 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C9 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-6 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C10 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-6 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C11 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-7 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C12 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-9 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C13 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-10 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C14 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-11 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C15 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-12 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C16 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-14 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C17 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-14 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C18 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-16 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C19 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-16 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C20 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-17 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C21 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md#source-section-18 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |

## 核证范围

保留21项迁移细节，核读defaults、tokenbudget、logprobs、models/features与removed列表。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

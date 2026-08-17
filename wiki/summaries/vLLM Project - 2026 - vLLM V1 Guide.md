---
type: summary
status: refined
---
# vLLM Project - 2026 - vLLM V1 Guide

## 来源信息

- 类型：官方文档 / V1 migration 与 feature support guide
- 发布者：vLLM Project
- 原始 HTML：[vLLM V1 Guide](../../raw/html/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.html)
- 全文文本：[vLLM V1 Guide](../../raw/text/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md)
- 官方页面：[vLLM V1](https://docs.vllm.ai/en/stable/usage/v1_guide/)
- 快照日期：2026-08-04
- 状态：精修 summary；基于官方 living guide 快照，支持矩阵可变且本页不是 benchmark

## 摘要

这份 guide 描述 vLLM 从 V0 到 V1 的核心重构和 2026-08-04 时点的功能边界。V1 保留既有 models、GPU kernels 与 utilities，但重写 scheduler、`KV cache manager`、worker、sampler 和 API server，目标是形成更简单、模块化、低 CPU overhead 且默认开启关键优化的统一架构。该快照已宣布 V0 fully deprecated；不过页面也明确自称 living guide，支持状态仍会随 PR / RFC 持续变化。

V1 scheduler 的关键抽象是统一 token budget：它不再先把工作严格分成 prefill 与 decode 两类，而是在每轮预算内用 `{request_id: num_tokens}` 表示各请求本轮要处理的 token 数。这样 chunked prefill、prefix caching 与 speculative decoding 可以共享同一调度表示。调度策略既支持 `FCFS`，也支持 priority-based scheduling；后者以请求 priority 排序，同 priority 时仍以 FCFS 打破平局。

这份文档同时是一份 compatibility checklist：chunked prefill 默认尽可能开启，CUDA graph capture 比 V0 占更多内存，默认 logprobs 语义变为 logits post-processing 之前的 raw output；部分功能处于 functional 或 in progress，另有 `best_of`、per-request logits processors、GPU↔CPU `KV cache` swapping 与 request-level structured-output backend 被明确移除。页面宣称 V1 尤其在 long-context 场景有显著性能改进，但对应 performance benchmark 仍标为 “To be added”，因此不能把该说法当作可复核的性能证据。

## 关键事实

- **重构范围**：V1 复用成熟的 model implementations、GPU kernels 与 utilities，同时重构 scheduler、`KV cache manager`、worker、sampler 和 API server。
- **设计目标**：官方列出的目标包括易修改的 modular codebase、near-zero CPU overhead、把关键优化合进统一架构，以及尽量 zero-config 地默认启用优化；这些是项目目标，不等于该页面已逐项 benchmark 验证。
- **V0 状态**：在这份 2026-08-04 stable 快照中，V0 已被标为 fully deprecated；V0 可用而 V1 不可用的 use case 被引导到 GitHub 或 vLLM Slack 反馈。
- **unified scheduler**：scheduler 以 `{request_id: num_tokens}` 的简单字典，在固定 token budget 下动态决定每条请求本轮处理多少 token，不要求 prefill 与 output/decode tokens 进入两套严格分离的调度路径。
- **调度能力的组合**：同一 token-budget 表示被用于组合 chunked prefills、prefix caching 与 speculative decoding，而不是分别维护互不兼容的 feature-specific scheduler。
- **调度策略**：`--scheduling-policy` 可选择 `FCFS` 或 priority-based scheduling；priority 相同时使用 FCFS 作为 tie-breaker。
- **chunked prefill**：V1 在条件允许时默认启用；V0 则会依据模型特性有条件开启。这是默认行为变化，部署迁移时不能假设两代配置语义一致。
- **CUDA graphs**：文档明确 V1 的 CUDA graph capture 比 V0 占用更多 memory，但没有在本页给出统一增量数字。
- **默认 logprobs 语义**：V1 默认在 temperature、penalties、bad-words processor、`top_k / top_p` 等 logits post-processing 之前返回模型 raw output 对应的 logprobs，因此不一定等于最终 sampling distribution。
- **logprobs modes**：`--logprobs-mode` 支持 `raw_logprobs`（默认）、`processed_logprobs`、`raw_logits`、`processed_logits`；raw / processed 的分界是是否经过全部 logits processors。
- **prompt logprobs + prefix cache**：接口组合被标为 functional，但当请求需要 prompt logprobs 时，engine 会忽略 prefix cache 并重新 prefill 完整 prompt，因为 V1 不缓存 logprobs。
- **硬件支持快照**：NVIDIA、AMD、Intel GPU、TPU 与 CPU 在页面中均标为 functional；Ascend、Spyre、Gaudi、OpenVINO 等更多平台通过各自 plugins 扩展，需查对应 repository。
- **模型支持快照**：decoder-only、pooling、Mamba、multimodal 被标为 functional；Whisper 获得 native encoder-decoder support，其他 encoder-decoder models 不在 core support matrix 内，可通过 plugin pattern 扩展。
- **pooling 边界**：last-pooling models 新支持 prefix caching 与 chunked prefill；文档仍在为更多 pooling categories 扩展这两项能力。
- **Mamba 边界**：Mamba-1、Mamba-2、attention-Mamba hybrid 与文档列举的其他 hybrid mechanisms 可运行，但该快照明确这些模型均尚不支持 prefix caching。
- **functional features**：Prefix Caching、Chunked Prefill、LoRA、Logprobs Calculation、FP8 KV Cache、Spec Decode、Prompt Logprobs with Prefix Caching、Structured Output Alternative Backends 均为绿色 functional。
- **in-progress feature**：Concurrent Partial Prefills 在该快照中仍标为 in progress。
- **移除 `best_of`**：官方理由是使用有限；该 sampling feature 不再属于 V1。
- **移除 per-request logits processors**：V1 改为支持服务启动时配置的 global logits processors，不再允许每个请求传入自定义 processing function。
- **移除 GPU↔CPU KV swapping**：V1 的 simplified core architecture 不再依靠该机制处理 request preemption，这与 2023 vLLM 论文把 swapping 列为恢复路径的设计不同。
- **structured output 变化**：request-level backend 选择被移除；`outlines`、`guidance` 等 alternative backends 及 fallback 仍被支持。

## 争议与不确定点

- 页面明确是会持续更新的 living guide；绿色、黄色、红色状态只代表 2026-08-04 stable 文档快照，后续版本可能新增、移除或重新定义功能。
- 页面声称升级 V1 core engine 带来显著性能改善，尤其是 long context，但链接位置仍写着 performance benchmark “To be added”。因此本来源不能支撑具体吞吐、latency、CPU overhead 或 V1/V0 speedup 数字。
- 文档对 “Functional” 的定义是功能可运行且优化达到或超过 V0，但这不是所有 model × hardware × quantization × feature 组合均已验证的保证；plugin 平台尤其需要独立核对其仓库。
- “zero configs” 与 “near-zero CPU overhead” 是设计目标而非绝对性质。实际 deployment 仍会受 parallelism、memory、scheduler policy、multimodal preprocessing 与 workload 影响。
- 统一 token-budget scheduler 简化了 feature composition，但本页没有给出请求优先级、公平性、chunk 大小、preemption 代价或 mixed-prefill/decode latency trade-off 的实测分析。
- Prompt Logprobs with Prefix Caching 虽标为 functional，却会为相关请求 bypass prefix cache 并 full-prefill；只看支持矩阵可能高估这一组合的缓存收益。
- V1 默认 raw logprobs 是对外可观察的语义变化。依赖“返回概率等于最终采样概率”的客户端必须显式选择 processed mode，不能仅做无配置版本升级。
- `GPU <> CPU KV Cache Swapping` 的移除说明 V1 改变了 preemption recovery 机制，不代表 CPU offload / connector 生态中的所有数据迁移概念都被一并否定。

## 关联页面

- 概念：[vLLM](../concepts/vLLM.md)
- 概念：[PagedAttention](../concepts/PagedAttention.md)
- 比较：[SGLang 与 vLLM 架构对比](../comparisons/SGLang%20与%20vLLM%20架构对比.md)
- 官方架构：[Architecture Overview](./vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md)
- 官方设计：[Automatic Prefix Caching](./vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md)
- 原始论文：[Kwon et al. - 2023 - Efficient Memory Management for Large Language Model Serving with PagedAttention](./Kwon%20et%20al.%20-%202023%20-%20Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention.md)

---
type: comparison
---
# SGLang 与 vLLM 架构对比

## 比较目标

本页比较 SGLang 与 vLLM 在系统边界、KV cache、scheduler、structured generation 和分布式执行上的设计重心。比较分成两个时间层：2023–2024 年论文用于解释各自的原始技术主张；2026 年官方文档快照用于校正已经发生的能力收敛。它不是脱离 model、hardware、version 和 workload 的性能排行榜。

## 核心结论

- **vLLM 的原始切入点是通用高吞吐 serving engine**：PagedAttention 把动态 KV cache 变成 logical-to-physical block mapping，并与 centralized scheduling、preemption 和 distributed workers 协同。
- **SGLang 的原始切入点是 language/runtime co-design**：前端表达多次 generation、prompt state、分支与约束；runtime 用 RadixAttention、cache-aware scheduling 和 constrained decoding 利用这些程序结构。
- **PagedAttention 与 RadixAttention 不在完全相同的抽象层。** PagedAttention 首先解决物理显存分配和 block-based execution；RadixAttention首先解决已计算 KV 的 prefix indexing、保留、复用与 locality-aware scheduling。SGLang 的 KV tensors 本身也使用 paged layout。
- **“SGLang 有 prefix cache、vLLM 没有”已经过时。** 当前 vLLM V1 有 hash-based Automatic Prefix Caching；当前 SGLang 继续采用 radix-tree 路线，并通过 HiCache 扩展到 GPU / host / distributed storage。
- 当前两者都覆盖 continuous batching、prefix caching、structured outputs、并行和 OpenAI-compatible serving。真正的选择应落在 workload locality、模型与硬件支持、SLO、运维接口以及实测结果上；speculative execution / decoding 的具体路径与兼容组合则应按目标版本另行核对。

## 架构总览

| 维度 | vLLM | SGLang | 实际含义 |
| --- | --- | --- | --- |
| 原始优化边界 | 独立 requests 的通用 inference / serving engine | 多次调用与控制流组成的 structured LM programs | vLLM 从 engine 抽象出发；SGLang 从 application structure 与 runtime 协同出发 |
| 论文核心 | PagedAttention、block-level KV manager、scheduler / preemption | frontend language + runtime、RadixAttention、compressed FSM | 一个先解决动态显存，一个先利用跨调用结构 |
| 当前 KV 物理层 | block pool、block table、paged KV cache | paged KV pool，可配置 page granularity | 两者都不是“每个 request 一块最大连续显存” |
| 当前 prefix index | chained block hash + cache table + LRU free queue | radix tree / HiRadixTree + LRU / storage-tier metadata | 都能复用前缀，数据结构和 locality policy 不同 |
| 当前 scheduler 重心 | V1 unified token-budget scheduler；FCFS / priority | cache-aware / longest-prefix policies；overlap CPU scheduling 与 GPU execution | vLLM 强调统一和简化；SGLang 更显式把 cache locality 与 host overhead 纳入调度 |
| 进程边界 | API server → Engine Core → per-GPU workers，DP 时增加 coordinator | 论文层是 frontend interpreter/compiler → SGLang Runtime；runtime 可独立作为 server 使用 | 当前源码拓扑会随版本变化，不应只凭 2024 论文推断 |
| structured generation | 当前 V1 提供 structured-output backends | 论文以 compressed FSM 为核心优化，当前也接入 grammar backend | 现在是共同能力；差异更多在程序接口和执行整合方式 |

## 1. 系统边界：engine-first 与 program/runtime co-design

vLLM 论文把问题定义为：在 online serving 中，请求长度和生成长度不可预测，KV cache 又占用大量显存；若按最大长度连续预留，fragmentation 与 duplication 会限制 batch size。因此 vLLM 的核心抽象是一个 centralized scheduler、KV cache manager 和 distributed GPU workers 组成的 serving engine。当前 V1 又把职责拆成 API Server、Engine Core、per-GPU Worker，以及可选 DP Coordinator；Engine Core 负责 scheduler、KV cache 和 worker coordination。

SGLang 论文则从另一端出发：agent、few-shot、multi-turn、tree-of-thought 和 structured outputs 不是单个无状态 request，而是多个相互依赖的 model calls。它用嵌入 Python 的 frontend language 表达 `gen / select / fork / join` 等 primitive，再由 runtime 利用跨调用共享前缀、并行和约束。前端与 runtime 可以协同，也可以独立使用；所以把 SGLang 只理解成“另一个 OpenAI API server”会漏掉它最初的系统主张。

这一区别是设计重心，不是永久功能边界。今天 vLLM 也服务 agent / structured-output workload，SGLang Runtime 也能作为通用 OpenAI-compatible server。

## 2. KV cache：物理分页与语义前缀索引

### vLLM / PagedAttention

PagedAttention 把 sequence 的 KV cache 切成 fixed-size logical blocks，再按需映射到不连续 physical blocks。scheduler 在每一步为需要增长的 request 分配 blocks，attention kernel 通过 block table 读取历史 KV。这样无需按最大输出长度预留连续空间，request 完成后 blocks 可立即回收；parallel sampling 和 beam search 还能让多个 sequence 共享 physical blocks，并在写入时 copy-on-write。

它最初解决的是 **capacity 与 fragmentation**：如何把更多 active sequences 放进 GPU，并让动态长度与 preemption 可管理。

### SGLang / RadixAttention

RadixAttention 把 token sequences 组织为 radix tree，tree nodes 指向 paged KV tensors。completed request 的 KV 不立即丢弃，而是作为可淘汰 cache 保留；新请求先做最长前缀匹配，再只 prefill 未命中的 suffix。node split、reference count、leaf-first LRU 与 cache-aware scheduling 共同维护复用生命周期。

它最初解决的是 **reuse 与 locality**：哪些历史 KV 与当前 prompt 语义上共享 prefix，怎样保留、匹配和安排请求顺序以提高命中率。

### 当前已经收敛，但实现仍不同

vLLM V1 的 Automatic Prefix Caching 会把 parent block hash、当前 block token IDs 与 LoRA / multimodal / cache-salt 等 extra hashes 组成 chained hash key；KV manager 用 hash table 找到已计算 full blocks，并以 reference count、block pool、free queue 与 LRU 管理生命周期。它只缓存完整 blocks，所以未对齐 block boundary 的公共尾段仍需重算。SGLang 仍以 radix tree 表达 prefix relationship；2024 论文实现是一 token 一 page，而当前 HiCache 在 `page_size > 1` 时也按 page granularity 匹配，并进一步记录一段 KV 位于 GPU、host memory、distributed storage 中的哪一层。因此也不能把当前 RadixAttention 简化成“永远逐 token 命中”。

因此今天最准确的表述不是“一个做 memory、一个做 cache”，而是：**二者都做 paged memory 和 prefix cache，但 vLLM 的当前 prefix index 以 chained block hash 为中心，SGLang 以 radix tree 及其 cache-aware policy 为中心。**

## 3. Scheduler：统一 token budget 与 cache locality / overlap

vLLM V1 用 unified scheduler 把 prompt tokens 与 output tokens 统一放入每轮 token budget，不再用严格分离的 prefill / decode scheduler path。这个抽象让 chunked prefill、prefix caching 与 speculative decoding 共享较简单的 scheduling interface；官方文档列出的主要 policy 是 FCFS 与 priority。

SGLang 从原论文起就让 matched prefix length 参与 request ordering，以避免 cache thrashing；v0.4 又把 scheduler 做成 one-batch-ahead pipeline，让 CPU 在 GPU 执行 batch N 时准备 batch N+1，隐藏 scheduling、memory allocation 与 radix operations 的 host overhead。其 cache-aware load balancer还在 workers 之外维护近似 radix tree，把新请求路由到预期 prefix hit 更高的 worker。

这里同样不能推导“vLLM 不做 overlap”或“SGLang 不支持 priority”。两边持续吸收类似优化；上面的区别只描述各自官方架构材料中更核心的组织方式。生产选择必须看目标版本的实际 flags、backend 和 benchmark。

## 4. Structured generation：从差异点变成共同能力

SGLang 论文把 structured LM program 当作第一等对象。compressed FSM 会压缩连续的单一合法 transition，使固定 JSON / regex 片段有机会在一次 forward 中推进多个 tokens；API speculative execution 则面向只能调用远程 API 的 multi-call program。这里的独特性是 language primitives、constraint 与 runtime optimization 被放在同一个设计里。

当前 vLLM V1 也提供 structured-output backends，SGLang 当前版本同样使用 XGrammar 等 grammar backend。因而“是否能输出合法 JSON”已经不再是稳定分界线。更值得比较的是 grammar compilation latency、batch interaction、speculative decoding compatibility、reasoning / tool parser，以及同一 schema workload 下的 TTFT、ITL 与吞吐。

## 5. 哪些常被误当成核心区别

- `FlashAttention / FlashInfer / Triton kernels`：这是 execution backend 与 kernel 层，两边都可能使用，不等同于 PagedAttention 或 RadixAttention。
- `continuous batching`：两边都有；它是现代 serving engine 的共同基础。
- `CUDA Graph`、quantization、LoRA、speculative decoding：高度版本相关，不能用一张静态 feature checklist 判定长期胜负。
- `OpenAI-compatible API`：两边都支持；它只说明客户端接口相近，不说明内部 scheduler 与 cache 相同。
- 某篇论文中的 `x×`：vLLM 论文与 SGLang 论文使用了不同年代、baseline、hardware、model 和 workload，数字不能互相拼成当前排行榜。

## 6. 选择建议

- **长 system prompt、multi-turn、many-query-over-document、parallel sampling、agent 分支较多**：SGLang 的 radix-tree cache、locality-aware scheduling 与 hierarchical cache 与这类 workload 结构更直接对齐，值得优先 benchmark。
- **希望把一个通用 engine 嵌入多种 serving / offline inference 场景，重视 V1 的清晰 process separation、广泛模型/硬件插件与标准 request abstraction**：vLLM 是自然候选，但仍需逐项核对目标模型和 backend。
- **简单单轮、prefix reuse 低、decode 很长**：两者在“独特缓存结构”上的差异可能被 decode kernel、quantization、batch size 和 hardware backend 淹没；应以实测为准。
- **大规模集群或 MoE**：不要仅凭框架名称选择。固定相同 model weights、precision、parallel topology、attention / MoE backend、PD 配置与 SLO，再比较 TTFT、TPOT / ITL、request throughput、token throughput、P95/P99、显存占用和故障恢复。

## 证据边界

- vLLM 原始论文对应 2023 年系统；SGLang 最终论文的 vLLM baseline 是 v0.2.5。两篇论文只用于解释设计起点，不用于证明当前性能高低。
- 2026 官方文档是本知识库在 `2026-08-04` 保存的可变快照。feature status、process topology、CLI 参数与 backend 支持可能继续变化。
- SGLang v0.4 博客的 throughput / cache-hit 数字是项目方自报；没有与本页其余来源构成同版本、同硬件、同 workload 的独立横评。
- 本页没有足够证据给出“哪个框架普遍更快”的结论；稳定结论只到架构重心、数据结构和适用 workload。

## 证据基础

- [Kwon et al. - 2023 - Efficient Memory Management for Large Language Model Serving with PagedAttention](../summaries/Kwon%20et%20al.%20-%202023%20-%20Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention.md)
- [Zheng et al. - 2024 - SGLang Efficient Execution of Structured Language Model Programs](../summaries/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md)
- [vLLM Project - 2026 - Architecture Overview](../summaries/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md)
- [vLLM Project - 2026 - vLLM V1 Guide](../summaries/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md)
- [vLLM Project - 2026 - Automatic Prefix Caching](../summaries/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md)
- [SGLang Team - 2024 - SGLang v0.4 Zero-Overhead Batch Scheduler Cache-Aware Load Balancer Faster Structured Outputs](../summaries/SGLang%20Team%20-%202024%20-%20SGLang%20v0.4%20Zero-Overhead%20Batch%20Scheduler%20Cache-Aware%20Load%20Balancer%20Faster%20Structured%20Outputs.md)
- [SGLang Project - 2026 - HiCache System Design and Optimization](../summaries/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md)

## 关联页面

- [vLLM](../concepts/vLLM.md)
- [SGLang](../concepts/SGLang.md)
- [PagedAttention](../concepts/PagedAttention.md)
- [RadixAttention](../concepts/RadixAttention.md)
- [FlashAttention](../concepts/FlashAttention.md)
- [注意力机制 Attention](../topics/注意力机制%20Attention.md)

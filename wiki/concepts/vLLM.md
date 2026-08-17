---
type: concept
---
# vLLM

## 简介

`vLLM` 是面向自回归大语言模型的高吞吐 inference / serving engine。它不是一种新的语言模型架构，也不靠改变模型权重或近似 attention 提升性能。理解 vLLM 必须区分技术层与时间层：`PagedAttention` 是从非连续物理 `KV block` 读取 key/value 的 attention kernel 与寻址抽象；vLLM 则是使用该抽象并持续演进的完整 serving system。

在 **2023 年论文版本**中，系统以 `PagedAttention` 打破“每条序列的逻辑连续 KV cache 必须占据连续物理 GPU 内存”的约束，并围绕它共同设计 centralized scheduling、block-level memory management、preemption recovery、decoding-state sharing 与 tensor-parallel execution。在 **2026-08-04 V1 官方文档快照**中，核心系统已经重构为 API Server、Engine Core、per-GPU Worker 与可选 DP Coordinator 的多进程拓扑；scheduler 改用统一 token budget，prefix reuse 也发展为 hash-based Automatic Prefix Caching（APC）。因此不能把首篇论文架构、当前 V1 实现与 `PagedAttention` 这个单一 primitive 当成同义词。

## 关键属性

- 类型：LLM inference / serving system
- 首篇系统论文：SOSP 2023
- 当前架构时间点：2026-08-04 官方 `stable` / living documentation snapshot
- 核心 primitive：`PagedAttention`
- 优化目标：在相近 latency 下提高 serving throughput
- 主要瓶颈假设：自回归解码受 GPU memory capacity / bandwidth 与动态 `KV cache` 容量约束
- 内存单位：固定 token 数量的 logical / physical `KV block`
- 核心元数据：每条序列的 `block table`、每个物理块的引用计数、已填 slot 数量
- 论文版调度：iteration-level batching、`FCFS`、preemption、sequence-group gang scheduling
- 论文版恢复：GPU-to-CPU swapping 或 `KV cache` recomputation
- 论文版共享：parallel sampling、beam search、预定义 shared prefix，以及 block-level `copy-on-write`
- V1 进程拓扑：API Server ↔ Engine Core → per-GPU Workers；`DP > 1` 时增加 DP Coordinator
- V1 scheduler：统一 prompt / output tokens 的 per-iteration token budget，支持 `FCFS` 与 priority policy
- V1 prefix cache：parent hash + full-block tokens + extra hashes 的 chained block hash，配合 block pool、free queue、reference count 与 LRU
- 分布式路径：论文支持 Megatron-LM 风格 tensor parallelism + NCCL `all-reduce`；V1 文档进一步明确 `DP × PP × TP` 的 worker/process 组织
- 代表来源：[Kwon et al. - 2023 - Efficient Memory Management for Large Language Model Serving with PagedAttention](../summaries/Kwon%20et%20al.%20-%202023%20-%20Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention.md)

## 相关主张

### 2023 论文：从 OS paging 到 KV cache paging

论文用操作系统虚拟内存解释 vLLM 的核心映射，但这是针对 LLM 语义改造后的类比，而不是直接复用 GPU 的通用 virtual memory：

| OS 抽象 | vLLM 中的对应物 | 作用 |
| --- | --- | --- |
| process 的逻辑地址空间 | request / sequence 的逻辑 token 序列 | 上层保持连续上下文视图 |
| fixed-size page | fixed-size logical `KV block` | 把动态增长状态切成可管理单位 |
| physical page frame | GPU DRAM 中的 physical `KV block` | 允许逻辑相邻块物理不相邻 |
| page table | `block table` | 记录 logical-to-physical mapping 与已填位置 |
| shared page + reference count | 共享 physical `KV block` | 复用相同 prompt / beam prefix 的状态 |
| `copy-on-write` | 改写共享末块时复制一个 block | 在维持隔离的同时延迟复制 |
| swap / page recovery | CPU RAM swapping 或 recomputation | 抢占后恢复整条序列的 `KV cache` |

这一抽象消除了变长请求对连续 chunk 的依赖。每条序列只在已有 blocks 填满后申请新物理块，因而 external fragmentation 被 fixed-size blocks 消除，单序列 internal fragmentation 被限制在最后一个 block。它同时把共享从“复制整条连续 tensor”改为“让多个 logical blocks 指向同一 physical block”。

### 2023 论文：控制面与执行面

论文版 vLLM 的控制面集中在 scheduler：

1. scheduler 按本轮状态选择可运行的 sequence groups，并由 `KV cache manager` 为即将新增的 token 分配 blocks；
2. scheduler 为 batch 准备 input token IDs 与每条序列的 block table，并广播给 GPU workers；
3. worker 的 cache engine 按物理 block ID 读写本 worker 所持 attention heads 的 `KV cache`，model shard 完成对应计算；
4. tensor-parallel workers 通过 `all-reduce` 同步模型中间结果，采样 token 再返回 scheduler，进入下一轮调度。

这种分工让所有 workers 共享同一套逻辑到物理映射，却不需要在每次内存分配时彼此协商。scheduler 负责全局一致性，worker 只消费本轮控制消息并执行本地 shard。

### 内存效率如何转化为吞吐

vLLM 的性能链条是：减少预留和碎片 → 同一 GPU 容纳更多活动序列的 `KV cache` → iteration-level batch 可以更大 → memory-bound decode 更充分利用 GPU → 单位时间完成更多请求。由此可见，PagedAttention 的价值主要发生在系统容量层；论文的 kernel microbenchmark 反而测到相对 FasterTransformer `20%–26%` 的 attention latency 开销。

因此，vLLM 的收益依赖 workload：长序列、大模型、输出长度不确定、复杂 decoding 或共享前缀较多时更有利；当序列短、`KV cache` 容量宽裕、系统已经 compute-bound 时，分页管理能释放的额外 batching 空间有限。

### Decoding state sharing

- parallel sampling 中，多条输出序列可直接共享 prompt 对应的完整 blocks；
- beam search 中，候选序列随搜索过程动态 fork、共享和释放前缀 blocks；
- 预定义 shared prefix 可像 shared library 一样常驻物理 blocks，新请求只计算各自 task input；
- `copy-on-write` 只在某条序列需要写入仍被多方引用的末块时触发，因此复制范围最多是一个 block，而非整段历史 `KV cache`；
- 实现以 `fork`、`append`、`free` 三个操作统一上述不同访问模式，使底层 kernel 无需理解 beam 或 sampling 的业务语义。

### 2023 论文：Scheduling、preemption 与恢复

论文版本采用 `FCFS` 保证早到请求优先，并在内存不足时优先抢占最新请求。由于一个 sequence group 内可能存在共享 blocks，vLLM 对其执行 gang scheduling；eviction 也是 all-or-nothing，即一条序列的 blocks 要么全部在 GPU，要么全部被驱逐。

恢复有两条路径。Swapping 把被驱逐 blocks 复制到由 CPU block allocator 管理的 RAM；recomputation 则把已生成 tokens 与原 prompt 合并，在一次 prompt phase 中重建 `KV cache`。前者受 CPU-GPU 传输粒度和 PCIe bandwidth 影响，后者受 prompt computation 成本影响，论文没有宣称其中一条在所有 block size 和硬件上恒优。

### 2026 V1：多进程 serving topology

V1 把在线服务拆成边界更明确的进程：

1. **API Server** 处理 HTTP / OpenAI-compatible API、tokenization、多模态加载与 response streaming，通过 ZMQ 连接所有 Engine Cores；多个 API 与多个 cores 形成 many-to-many topology。
2. **Engine Core** 每个 data-parallel rank 一个，运行 scheduler busy loop、维护 `KV cache`，并协调属于该 rank 的 GPU workers。
3. **GPU Worker** 每张 GPU 一个，加载该 rank 的模型权重、执行 forward pass 并管理 GPU memory；每个 Engine Core 下有 `TP × PP` 个 workers。
4. **DP Coordinator** 仅在 `DP > 1` 时出现，负责 DP ranks 间 load balancing，并为 MoE 模型协调 synchronized forward passes。

若 API Server 数为 `A`、GPU 数为 `N = DP × PP × TP`，文档列出的主进程总数为 `A + DP + N + (DP > 1 时的 1 个 coordinator)`。这套拓扑是 2026-08-04 官方文档快照，而不是 2023 论文图 4 的简单改名；它也不等于操作系统中全部辅助 threads、通信进程或 plugin sidecars。

### 2026 V1：Unified token-budget scheduler

V1 scheduler 不再把 prompt prefill 与 output decode 视为两条严格分离的调度路径。它在每轮固定 token budget 内，用 `{request_id: num_tokens}` 表示各请求本轮要计算的 token 数，使 chunked prefill、prefix caching 与 speculative decoding 能复用同一 scheduling abstraction。官方 guide 列出 `FCFS` 与 priority-based 两种 policy；priority 相同时仍以 FCFS 作为 tie-breaker。

这一统一性也伴随行为变化：chunked prefill 在条件允许时默认开启；默认 logprobs 是 logits post-processing 之前的 raw values；需要 prompt logprobs 的请求会 bypass prefix cache 并 full-prefill；V1 移除了 2023 论文版用于 preemption recovery 的 GPU↔CPU `KV cache` swapping。因而“论文中支持某机制”不能直接推出 V1 仍以相同方式实现它。

### 2026 V1：Hash-based Automatic Prefix Caching

V1 的 APC 对每个 **完整 block** 构造 chained hash：key 包含 parent block hash、当前 block 的精确 token tuple，以及 LoRA ID、多模态 input hash、`cache_salt` 等 extra hashes。parent hash 将此前 prefix 递归纳入身份；只有执行上下文一致、且落在完整 block 边界内的 prefix 才能命中。

`KV cache manager` 在初始化时预分配全部 `KVCacheBlock` 形成 block pool，并维护 hash-to-block-IDs cache mapping、request-to-block-IDs mapping 与 intrusive doubly linked free queue。Cache hit 会 “touch” block、增加 `ref_cnt` 并把无人使用的命中块移出 free queue；新分配从队首取 block，若它仍被缓存则先执行 LRU eviction。`ref_cnt = 0` 的 cached block 可以继续留在 free queue 等待复用，也可以在显存需要时被淘汰。可选 `cache_salt` 注入首块并沿 hash chain 传播，把 prefix reuse 限制在共享同一 salt 的 trust group，以降低跨租户 timing inference 风险。

这与论文中的预定义 shared prefix / block-level COW 是时间上不同的能力层。原始机制说明物理 blocks 可以共享；V1 APC 进一步提供跨请求发现、索引、保留和淘汰任意完整前缀的缓存生命周期。

### 与相邻优化的边界

- 与 `FlashAttention` 相比，PagedAttention 主要解决在线 serving 中跨请求、跨时间动态增长的 `KV cache` 放置与共享；FlashAttention 主要减少一次 attention 计算在 GPU 内存层级间的 IO。论文在 prefill 中仍可使用 conventional self-attention implementation，例如 FlashAttention，在 decode 中再按 block table 使用 PagedAttention。
- 与 Orca 相比，Orca 的核心贡献是 iteration-level scheduling / request interleaving，vLLM 论文的核心新增面是让更多 working sets 真正装入显存的分页式 `KV cache` 管理。论文明确把两者视为互补，而不是互斥架构。
- 与模型级 `MQA / GQA` 等技术相比，vLLM 不改变每个 token 需要生成哪些 K/V 表示；模型结构决定单位 token 的 cache 体积，vLLM 决定这些状态怎样被放置、复用和调度。
- 与 `RadixAttention` 相比，`PagedAttention` 首先回答 KV tensors 的物理放置、增长和 block-based execution；`RadixAttention` 首先回答已计算 token prefixes 怎样由 radix tree 索引、保留、匹配并参与 cache-aware scheduling。二者不在完全相同的抽象层，SGLang 的 RadixAttention 也建立在 paged KV layout 上。
- 与当前 `SGLang` 相比，V1 vLLM 同样拥有 prefix cache，准确差异不是“有或没有缓存”，而是 vLLM APC 以 chained block hash + cache table + LRU free queue 为中心，SGLang 以 radix tree / HiRadixTree 与 locality-aware policy 为中心。完整比较见 [SGLang 与 vLLM 架构对比](../comparisons/SGLang%20与%20vLLM%20架构对比.md)。
- `Kimi K3` 是模型架构迫使 serving cache abstraction 扩展的实例：K3 的 MLA cache 随 token 增长，而 KDA state 固定大小且只能在已持久化 checkpoint 的 boundary 恢复。Moonshot 报告称两者被放进统一 paged layout 并共同完成 allocation、hit validation、eviction 与 transfer；这属于 K3-specific runtime path，不能倒推为通用 vLLM APC 的既有语义。

### 证据的时间与适用范围

论文在 A100 上评估 OPT-13B/66B/175B 与 LLaMA-13B，并用 ShareGPT / Alpaca 的输入输出长度合成 Poisson arrival traces。其总体结论是相对当时的 FasterTransformer 与作者重实现 Orca baseline 获得约 `2–4×` throughput；这证明了 2023 论文设计在相应 memory-bound 设置中的有效性，但不能直接外推为任意后续 vLLM 版本、模型、硬件或生产 workload 的固定倍数。

三份 2026 官方文档则是 2026-08-04 保存的可变快照：Architecture Overview 描述 process topology，V1 Guide 描述 scheduler 语义与当时 feature matrix，Automatic Prefix Caching 描述缓存数据结构和操作。它们都不是 benchmark；V1 Guide 的 performance benchmark 位置仍标为 “To be added”，APC 页面也没有提供 hit rate、TTFT 或 throughput 数据。因此这些来源可以校正“当前架构如何组织”，不能证明 vLLM 在任意 workload 中比其他 engine 更快。

## 来源支持

- [Kwon et al. - 2023 - Efficient Memory Management for Large Language Model Serving with PagedAttention](../summaries/Kwon%20et%20al.%20-%202023%20-%20Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention.md)
- [vLLM Project - 2026 - Architecture Overview](../summaries/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md)
- [vLLM Project - 2026 - vLLM V1 Guide](../summaries/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md)
- [vLLM Project - 2026 - Automatic Prefix Caching](../summaries/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md)
- [Zheng et al. - 2024 - SGLang Efficient Execution of Structured Language Model Programs](../summaries/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md)
- [SGLang Team - 2024 - SGLang v0.4 Zero-Overhead Batch Scheduler Cache-Aware Load Balancer Faster Structured Outputs](../summaries/SGLang%20Team%20-%202024%20-%20SGLang%20v0.4%20Zero-Overhead%20Batch%20Scheduler%20Cache-Aware%20Load%20Balancer%20Faster%20Structured%20Outputs.md)
- [SGLang Project - 2026 - HiCache System Design and Optimization](../summaries/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md)
- [Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence](../summaries/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)
- [Moonshot AI - 2026 - Kimi K3 Model Repository](../summaries/Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20Model%20Repository.md)

## 关联页面

- [PagedAttention](./PagedAttention.md)
- [RadixAttention](./RadixAttention.md)
- [SGLang](./SGLang.md)
- [SGLang 与 vLLM 架构对比](../comparisons/SGLang%20与%20vLLM%20架构对比.md)
- [Transformer](./Transformer.md)
- [FlashAttention](./FlashAttention.md)
- [注意力机制 Attention](../topics/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%20Attention.md)
- [Kimi K3](./Kimi%20K3.md)
- [Kimi Delta Attention](./Kimi%20Delta%20Attention.md)

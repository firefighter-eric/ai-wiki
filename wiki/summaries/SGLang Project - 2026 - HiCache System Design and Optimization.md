---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# SGLang Project - 2026 - HiCache System Design and Optimization

## TL;DR（快速导读）

HiCache 用 GPU、主机内存和远端存储三级缓存保存重复前缀，在容量、读写速度与命中率之间取舍。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

先提取信息，再并行判断，最后合成答案，是一个多调用流程；共享前缀与调度会影响整体成本。

## 来源信息

- 类型：项目技术文档 / 系统设计文档
- 来源标题：HiCache System Design and Optimization
- 来源 URL：https://docs.sglang.io/docs/advanced_features/hicache_design
- 原始 HTML：[SGLang Project - 2026 - HiCache System Design and Optimization.html](../../raw/html/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.html)
- 全文文本：[SGLang Project - 2026 - HiCache System Design and Optimization.md](../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md)
- 作者 / 维护者：SGLang Project
- 年份：2026（按当前知识库接入快照标记）
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

请求先找本地内容，再按需预取远端数据，计算后按策略写回。它减少的是重复提示计算，并不改变注意力公式；远端延迟、主机带宽和缓存策略可能决定额外容量是否带来实际收益。

### 方法与背景细节

HiCache 把 SGLang 的 RadixAttention 从“使用 GPU 闲置空间保存 prefix KV cache”扩展为三级 hierarchical KV cache：GPU memory 为 L1、host memory 为 L2、distributed storage 为 L3。L1/L2 属于单个 inference instance 的本地缓存，L3 则可由集群内实例共享。其目标不是改变 attention 数学，而是在 multi-QA、long-context 等重复 prefix 较多的 workload 中，用更大容量的缓存层级减少重复 prefill，并在 GPU 容量、host bandwidth、远端存储 latency 与 cache hit rate 之间做可配置折中。

系统的关键不是简单把 KV tensors offload 到 CPU 或磁盘，而是让 metadata、data movement 和 eviction/write policy 共同支持三级层次。HiRadixTree 负责表达连续 token span 及其在本地 L1/L2 的精确位置；L3 metadata 不持续同步进本地树，而是在需要时查询 backend。一次请求依次经历 local match、L3 prefetch、GPU computation 和 write-back。page size 与 memory layout 决定匹配粒度和 I/O batching，prefetch policy 决定愿意等待远端命中的时长，write policy 则决定何时把新生成或热点 KV 数据向更慢层级传播。

## 关键事实

- **C1**：HiCache 将 GPU memory、host memory、distributed storage 分别定义为 L1、L2、L3。类比 CPU cache hierarchy，L1/L2 对每个 inference instance 私有，L3 在 cluster 内共享；该类比描述的是容量、速度与共享范围，并不意味着三层具有硬件 CPU cache 的一致性协议。
- **C2**：HiRadixTree 建立在 RadixAttention 的 radix tree 上。每个 node 对应一段连续 token 的 KV cache，root-to-leaf path 表达请求 prefix；共享 prefix 的请求复用同一组 nodes。
- **C3**：扩展后的 node 会记录对应 KV cache 存在哪些层级。对本地 GPU/CPU 数据，HiRadixTree 保存精确 storage address；为降低 metadata overhead，它不保存或持续同步 L3 的详细位置，而是在访问时向 L3 backend 实时查询数据是否存在及其 server/location。
- **C4**：local match 从 root 沿匹配 token prefix 遍历 HiRadixTree，返回一段连续命中，其中前段可位于 L1、后段位于 L2。若命中终止于 node 内部，tree 会 split node 形成精确 boundary；该阶段只操作 metadata，不复制 tensor data。
- **C5**：local match 后，系统对 L1/L2 未命中的后续连续 prefix 查询 L3。若 L3 hit length 超过阈值便触发 L3→L2 prefetch；文档给出的默认阈值是 256 tokens，可配置。
- **C6**：`best_effort` 在 GPU 已可开始 prefill 时立即停止等待，偏向低 latency；`wait_complete` 等待全部 prefetch 完成，偏向高 hit rate；`timeout` 在完成或超时两者先到时停止，用于折中 SLO 与缓存收益。
- **C7**：`timeout` 的默认预算由固定 2 秒、每 1024 tokens 增加 0.1 秒、最高 30 秒组成。prefetch 停止后，系统把已经取回的数据与本地命中一起用于 prefill，而不是要求远端请求必须全量完成。
- **C8**：write-back 负责把 L1 中的 KV cache 传播到 L2/L3，以获得更大容量、更长保留时间和跨实例共享。`write_through` 每次访问立即写向下一层；`write_through_selective` 仅在访问频率超过阈值后备份热点；`write_back` 只在上层 eviction 时下写，以较低 I/O 压力换取较弱的提前缓存。
- **C9**：L2→L3 write-back 只传输 L3 尚不存在的数据。存入 L3 的 KV cache 能否被全部 SGLang instances 共享，仍取决于具体 storage backend 的实现与部署范围。
- **C10**：HiCache 的 L3 以 page 为存取和传输粒度，`--page-size` 指定每页 token 数。较大 page 能减少 metadata overhead、扩大 I/O batch 并提升 storage backend 效率，但部分 page 匹配时会损失 cache hit；长公共前缀倾向较大 page，多样化 prefix 可能更适合小 page。
- **C11**：当 `page_size > 1` 时，HiRadixTree 也按 page granularity 匹配。page size 因而同时影响 metadata boundary、可复用前缀精度与实际 I/O unit，并不只是一个底层存储参数。
- **C12**：`layer_first` 与 GPU 按层计算 KV 的访问方式一致；`page_first` 把同一 page 的数据放在 contiguous memory，便于作为单个对象 zero-copy 传给 L3，却可能导致 L2→GPU 时按“每层每 token”做细碎传输；`page_first_direct` 把 page 内同一 layer 的 tokens 聚合，以 page-layer granularity 缓和这一冲突。
- **C13**：L2→L3 路径可直接传递 memory address 与 size，减少中间 copies。CPU→GPU 路径在 prefill 中让 layer N+1 的 KV transfer 与 layer N computation 重叠，并提供基于 `cudaMemcpyAsync` 之上的 GPU-assisted I/O kernels。
- **C14**：项目文档自报 GPU-assisted I/O kernels 相对其 baseline transfer path 最高可达 3× transfer speed。该数字只描述 CPU↔GPU KV transfer micro-path，不等同于端到端 request throughput 或 latency 提升。
- **C15**：tensor parallelism 等 multi-rank 执行中，各 ranks 必须对 cache hit 与成功 prefetch 长度形成一致判断。文档使用 `all_reduce(op=min)` 同步 L3 hit 数和最终成功获取的 prefix length，避免不同 ranks 对 threshold 或可用 KV 长度产生分歧。
- **C16**：MHA 的 tensor-parallel ranks 各持有一个 token 的部分 KV 数据；MLA 场景下各 ranks 持有完整且相同的数据。HiCache 对 MLA 只允许一个 rank 发起 write-back，避免重复写入相同 KV cache。
- **C17**：L3 通过 `HiCacheStorage(ABC)` 统一 read、write、query interfaces。文档列出的 built-in backends 包括 Mooncake、DeepSeek 3FS（HF3FS）、NIXL、AIBrix KVCache 与示例性的 HiCacheFile，也支持 dynamic backend；LMCache 被列为另一套 hierarchical cache 方案。
- **C18**：在 prefill-decode disaggregation 中，HiCache 可同时部署在 prefill nodes 与 decode nodes；若 decode nodes 启用，decode outputs 也会 write back 到 L3。
- **C19**：当前文档要求 host KV pool 大于 device KV pool，可用 ratio 或每 rank 的 GB 数配置。容量增大通常提高 hit rate，但文档明确指出关系不是线性的：热点数据已覆盖后，继续扩容的边际收益会下降。

## 争议与不确定点

- 默认阈值和容量参数限定保存文档版本。
- 缓存共享依赖模型/权重、token、状态及后端部署一致，不能只相同文本就安全复用。

## 关联页面

- 概念：[SGLang](../concepts/SGLang.md)
- 概念：[RadixAttention](../concepts/RadixAttention.md)
- 比较：[SGLang 与 vLLM 架构对比](../comparisons/SGLang%20与%20vLLM%20架构对比.md)
- 来源：[Zheng et al. - 2024 - SGLang Efficient Execution of Structured Language Model Programs](./Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md)
- 来源：[SGLang Team - 2024 - SGLang v0.4 Zero-Overhead Batch Scheduler Cache-Aware Load Balancer Faster Structured Outputs](./SGLang%20Team%20-%202024%20-%20SGLang%20v0.4%20Zero-Overhead%20Batch%20Scheduler%20Cache-Aware%20Load%20Balancer%20Faster%20Structured%20Outputs.md)
- 概念：[Transformer](../concepts/Transformer.md)
- 主题：[注意力机制 Attention](../topics/注意力机制%20Attention.md)

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **KV cache**：键值缓存：保存已经处理过的位置表示，生成新内容时可复用，避免全部重算。
- **prefill**：提示计算阶段：先处理输入提示，再开始逐步生成输出。
- **decode**：解码阶段：利用已有输入与生成历史，产生后续输出。
- **baseline**：对照方案：用于判断改动有没有带来收益，条件是否公平尤其重要。

## 方法与实验解读

HiCache把前缀KV复用扩展到GPU、host和远端存储。先metadata匹配，再按阈值prefetch、按策略writeback；命中提升须抵消查询、传输和同步代价。页粒度同时改变复用精度和I/O效率，因此不能按缓存容量或micro-path速度直接预测端到端吞吐。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md#source-section-4 ) | 保存实现/文档版本；收益限定相应数据流与硬件配置。 |
| C2 | [原文]( ../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md#source-section-5 ) | 保存实现/文档版本；收益限定相应数据流与硬件配置。 |
| C3 | [原文]( ../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md#source-section-5 ) | 保存实现/文档版本；收益限定相应数据流与硬件配置。 |
| C4 | [原文]( ../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md#source-section-7 ) | 保存实现/文档版本；收益限定相应数据流与硬件配置。 |
| C5 | [原文]( ../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md#source-section-8 ) | 保存实现/文档版本；收益限定相应数据流与硬件配置。 |
| C6 | [原文]( ../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md#source-section-8 ) | 保存实现/文档版本；收益限定相应数据流与硬件配置。 |
| C7 | [原文]( ../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md#source-section-8 ) | 保存实现/文档版本；收益限定相应数据流与硬件配置。 |
| C8 | [原文]( ../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md#source-section-9 ) | 保存实现/文档版本；收益限定相应数据流与硬件配置。 |
| C9 | [原文]( ../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md#source-section-9 ) | 保存实现/文档版本；收益限定相应数据流与硬件配置。 |
| C10 | [原文]( ../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md#source-section-11 ) | 保存实现/文档版本；收益限定相应数据流与硬件配置。 |
| C11 | [原文]( ../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md#source-section-7 ) | 保存实现/文档版本；收益限定相应数据流与硬件配置。 |
| C12 | [原文]( ../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md#source-section-11 ) | 保存实现/文档版本；收益限定相应数据流与硬件配置。 |
| C13 | [原文]( ../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md#source-section-11 ) | 保存实现/文档版本；收益限定相应数据流与硬件配置。 |
| C14 | [原文]( ../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md#source-section-11 ) | 保存实现/文档版本；收益限定相应数据流与硬件配置。 |
| C15 | [原文]( ../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md#source-section-10 ) | 保存实现/文档版本；收益限定相应数据流与硬件配置。 |
| C16 | [原文]( ../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md#source-section-5 ) | 保存实现/文档版本；收益限定相应数据流与硬件配置。 |
| C17 | [原文]( ../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md#source-section-13 ) | 保存实现/文档版本；收益限定相应数据流与硬件配置。 |
| C18 | [原文]( ../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md#source-section-12 ) | 保存实现/文档版本；收益限定相应数据流与硬件配置。 |
| C19 | [原文]( ../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md#source-section-14 ) | 保存实现/文档版本；收益限定相应数据流与硬件配置。 |

## 核证范围

保留19项详细系统分析，核读三层/metadata、匹配、prefetch/writeback、同步、layout与后端配置。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

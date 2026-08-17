---
type: summary
status: refined
---
# SGLang Project - 2026 - HiCache System Design and Optimization

## 来源信息

- 类型：项目技术文档 / 系统设计文档
- 来源标题：HiCache System Design and Optimization
- 来源 URL：https://docs.sglang.io/docs/advanced_features/hicache_design
- 原始 HTML：[SGLang Project - 2026 - HiCache System Design and Optimization.html](../../raw/html/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.html)
- 全文文本：[SGLang Project - 2026 - HiCache System Design and Optimization.md](../../raw/text/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md)
- 作者 / 维护者：SGLang Project
- 年份：2026（按当前知识库接入快照标记）
- 状态：精修 summary，已基于项目文档 HTML 与全文文本核对

## 摘要

HiCache 把 SGLang 的 RadixAttention 从“使用 GPU 闲置空间保存 prefix KV cache”扩展为三级 hierarchical KV cache：GPU memory 为 L1、host memory 为 L2、distributed storage 为 L3。L1/L2 属于单个 inference instance 的本地缓存，L3 则可由集群内实例共享。其目标不是改变 attention 数学，而是在 multi-QA、long-context 等重复 prefix 较多的 workload 中，用更大容量的缓存层级减少重复 prefill，并在 GPU 容量、host bandwidth、远端存储 latency 与 cache hit rate 之间做可配置折中。

系统的关键不是简单把 KV tensors offload 到 CPU 或磁盘，而是让 metadata、data movement 和 eviction/write policy 共同支持三级层次。HiRadixTree 负责表达连续 token span 及其在本地 L1/L2 的精确位置；L3 metadata 不持续同步进本地树，而是在需要时查询 backend。一次请求依次经历 local match、L3 prefetch、GPU computation 和 write-back。page size 与 memory layout 决定匹配粒度和 I/O batching，prefetch policy 决定愿意等待远端命中的时长，write policy 则决定何时把新生成或热点 KV 数据向更慢层级传播。

## 关键事实

### 三层缓存与 HiRadixTree

- HiCache 将 GPU memory、host memory、distributed storage 分别定义为 L1、L2、L3。类比 CPU cache hierarchy，L1/L2 对每个 inference instance 私有，L3 在 cluster 内共享；该类比描述的是容量、速度与共享范围，并不意味着三层具有硬件 CPU cache 的一致性协议。
- HiRadixTree 建立在 RadixAttention 的 radix tree 上。每个 node 对应一段连续 token 的 KV cache，root-to-leaf path 表达请求 prefix；共享 prefix 的请求复用同一组 nodes。
- 扩展后的 node 会记录对应 KV cache 存在哪些层级。对本地 GPU/CPU 数据，HiRadixTree 保存精确 storage address；为降低 metadata overhead，它不保存或持续同步 L3 的详细位置，而是在访问时向 L3 backend 实时查询数据是否存在及其 server/location。
- local match 从 root 沿匹配 token prefix 遍历 HiRadixTree，返回一段连续命中，其中前段可位于 L1、后段位于 L2。若命中终止于 node 内部，tree 会 split node 形成精确 boundary；该阶段只操作 metadata，不复制 tensor data。

### Prefetch 与 Write-back

- local match 后，系统对 L1/L2 未命中的后续连续 prefix 查询 L3。若 L3 hit length 超过阈值便触发 L3→L2 prefetch；文档给出的默认阈值是 256 tokens，可配置。
- `best_effort` 在 GPU 已可开始 prefill 时立即停止等待，偏向低 latency；`wait_complete` 等待全部 prefetch 完成，偏向高 hit rate；`timeout` 在完成或超时两者先到时停止，用于折中 SLO 与缓存收益。
- `timeout` 的默认预算由固定 2 秒、每 1024 tokens 增加 0.1 秒、最高 30 秒组成。prefetch 停止后，系统把已经取回的数据与本地命中一起用于 prefill，而不是要求远端请求必须全量完成。
- write-back 负责把 L1 中的 KV cache 传播到 L2/L3，以获得更大容量、更长保留时间和跨实例共享。`write_through` 每次访问立即写向下一层；`write_through_selective` 仅在访问频率超过阈值后备份热点；`write_back` 只在上层 eviction 时下写，以较低 I/O 压力换取较弱的提前缓存。
- L2→L3 write-back 只传输 L3 尚不存在的数据。存入 L3 的 KV cache 能否被全部 SGLang instances 共享，仍取决于具体 storage backend 的实现与部署范围。

### Page Granularity 与 Data Transfer

- HiCache 的 L3 以 page 为存取和传输粒度，`--page-size` 指定每页 token 数。较大 page 能减少 metadata overhead、扩大 I/O batch 并提升 storage backend 效率，但部分 page 匹配时会损失 cache hit；长公共前缀倾向较大 page，多样化 prefix 可能更适合小 page。
- 当 `page_size > 1` 时，HiRadixTree 也按 page granularity 匹配。page size 因而同时影响 metadata boundary、可复用前缀精度与实际 I/O unit，并不只是一个底层存储参数。
- `layer_first` 与 GPU 按层计算 KV 的访问方式一致；`page_first` 把同一 page 的数据放在 contiguous memory，便于作为单个对象 zero-copy 传给 L3，却可能导致 L2→GPU 时按“每层每 token”做细碎传输；`page_first_direct` 把 page 内同一 layer 的 tokens 聚合，以 page-layer granularity 缓和这一冲突。
- L2→L3 路径可直接传递 memory address 与 size，减少中间 copies。CPU→GPU 路径在 prefill 中让 layer N+1 的 KV transfer 与 layer N computation 重叠，并提供基于 `cudaMemcpyAsync` 之上的 GPU-assisted I/O kernels。
- 项目文档自报 GPU-assisted I/O kernels 相对其 baseline transfer path 最高可达 3× transfer speed。该数字只描述 CPU↔GPU KV transfer micro-path，不等同于端到端 request throughput 或 latency 提升。

### 分布式一致性与 Storage Backends

- tensor parallelism 等 multi-rank 执行中，各 ranks 必须对 cache hit 与成功 prefetch 长度形成一致判断。文档使用 `all_reduce(op=min)` 同步 L3 hit 数和最终成功获取的 prefix length，避免不同 ranks 对 threshold 或可用 KV 长度产生分歧。
- MHA 的 tensor-parallel ranks 各持有一个 token 的部分 KV 数据；MLA 场景下各 ranks 持有完整且相同的数据。HiCache 对 MLA 只允许一个 rank 发起 write-back，避免重复写入相同 KV cache。
- L3 通过 `HiCacheStorage(ABC)` 统一 read、write、query interfaces。文档列出的 built-in backends 包括 Mooncake、DeepSeek 3FS（HF3FS）、NIXL、AIBrix KVCache 与示例性的 HiCacheFile，也支持 dynamic backend；LMCache 被列为另一套 hierarchical cache 方案。
- 在 prefill-decode disaggregation 中，HiCache 可同时部署在 prefill nodes 与 decode nodes；若 decode nodes 启用，decode outputs 也会 write back 到 L3。
- 当前文档要求 host KV pool 大于 device KV pool，可用 ratio 或每 rank 的 GB 数配置。容量增大通常提高 hit rate，但文档明确指出关系不是线性的：热点数据已覆盖后，继续扩容的边际收益会下降。

## 争议与不确定点

- 本来源是会持续更新的 SGLang 工程文档，不是固定版本论文。这里记录的是知识库标记为 2026 的页面快照；参数默认值、支持的 backends、memory layout 与实现细节都可能随之后版本改变。
- 文档的核心内容是 architecture 与 tuning reference，不包含一套完整端到端对照实验。它链接了单独的 benchmark blog，但不能用该链接替代本页自身的证据；本 summary 因而不主张 HiCache 对整体推理有固定倍数提升。
- 唯一明确写在本页的“最高 3×”是项目方对 GPU-assisted I/O kernel transfer speed 的自报结果。文档未在同一段给出硬件、KV dtype、page size、message size 分布、重复次数或端到端贡献，不能外推为所有 workload 的加速保证。
- L3 并非无代价的“大缓存”：backend lookup、network/storage latency、bandwidth contention 与 multi-rank synchronization 都可能抵消 cache hit 的计算节省。三种 prefetch policies 正是对这一不确定性的工程折中。
- page size 不存在普适最优值。扩大 page 可提高顺序 I/O 效率，却会降低不完整 page 的复用精度；具体结果依赖 prefix distribution、模型 KV footprint、backend transaction cost 与 host/GPU bandwidth。
- `write_through`、`write_through_selective`、`write_back` 分别在命中机会、I/O 压力和存储容量之间取舍。文档说明了语义，但没有为不同 workload 给出经独立验证的统一选择规则。
- L3 cross-instance sharing 取决于 backend 的一致性、可见范围和部署方式；“共享 L3”不能自动等同于所有实例都能低延迟、无重复地复用同一 cache。

## 关联页面

- 概念：[SGLang](../concepts/SGLang.md)
- 概念：[RadixAttention](../concepts/RadixAttention.md)
- 比较：[SGLang 与 vLLM 架构对比](../comparisons/SGLang%20与%20vLLM%20架构对比.md)
- 来源：[Zheng et al. - 2024 - SGLang Efficient Execution of Structured Language Model Programs](./Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md)
- 来源：[SGLang Team - 2024 - SGLang v0.4 Zero-Overhead Batch Scheduler Cache-Aware Load Balancer Faster Structured Outputs](./SGLang%20Team%20-%202024%20-%20SGLang%20v0.4%20Zero-Overhead%20Batch%20Scheduler%20Cache-Aware%20Load%20Balancer%20Faster%20Structured%20Outputs.md)
- 概念：[Transformer](../concepts/Transformer.md)
- 主题：[注意力机制 Attention](../topics/注意力机制%20Attention.md)

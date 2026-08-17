---
type: concept
---
# RadixAttention

## 简介

`RadixAttention` 是 SGLang 提出的跨请求 KV cache 复用机制。它用 radix tree 把 token prefix 映射到相应 KV tensors，在请求结束后仍把可复用的 prompt 与 generation states 作为 cache 保留；新请求到来时执行最长前缀匹配，只计算尚未缓存的 suffix。

这个名称容易让人误解为一种新的模型级 attention 公式。更准确地说，它是 serving runtime 中的 **prefix index + cache lifecycle + cache-aware scheduling** 设计，底层 KV tensors 仍可存放在 paged memory layout 中。

## 关键属性

- 类型：prefix-aware KV cache / serving runtime optimization
- 索引结构：CPU 侧 radix tree；edge 表示一段 token sequence，root-to-node path 表示可复用前缀
- 缓存内容：prompt 和 generation 对应的 KV tensors，而不只是在当前 request 生命周期内有效的 decode state
- 复用过程：prefix matching、node split、insertion、reference tracking 与 eviction 统一在同一棵树上
- 淘汰：论文采用 leaf-first LRU；running requests 引用的节点通过 reference counter 锁定，不可被淘汰
- 调度：longest-prefix-match / cache-aware policy 优先处理命中更长前缀的 request，以减少 cache thrashing
- 存储布局：论文使用 non-contiguous paged KV layout，因此 RadixAttention 可与 PagedAttention、continuous batching 和 tensor parallelism组合
- 当前扩展：HiCache 用 `HiRadixTree` 把同一 prefix metadata 延伸到 GPU、host memory 与 distributed storage 三层
- 代表来源：[Zheng et al. - 2024 - SGLang Efficient Execution of Structured Language Model Programs](../summaries/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md)

## 相关主张

- Radix tree 的价值不只是查找重复 system prompt。它能自然表达多轮对话的增长路径、few-shot examples 的多级共享、parallel sampling 和 tree-shaped agent / reasoning branches。
- SGLang 把 cache locality 提升为调度信号：在高并发等待队列中，请求顺序会影响下一批的 cache hit rate，因此 scheduler 与 cache index 需要协同。
- `RadixAttention` 的收益依赖 workload。共享前缀长、复用次数高、prefill 占比大时收益更明显；短 prompt、低复用或长 decode 主导的流量不会自动获得同样优势。
- 当代 vLLM 也有 Automatic Prefix Caching，采用 chained block hashes 而不是 radix tree。因此今天的准确差异是 **prefix cache 的索引和调度设计不同**，而不是“SGLang 能复用前缀、vLLM 不能”。
- 当前 SGLang 的 page size、attention backend、hierarchical storage 与 routing 已超出 2024 论文；任何性能判断都应固定 framework version、model、hardware 和 prompt-prefix distribution。

## 来源支持

- [Zheng et al. - 2024 - SGLang Efficient Execution of Structured Language Model Programs](../summaries/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md)
- [SGLang Team - 2024 - SGLang v0.4 Zero-Overhead Batch Scheduler Cache-Aware Load Balancer Faster Structured Outputs](../summaries/SGLang%20Team%20-%202024%20-%20SGLang%20v0.4%20Zero-Overhead%20Batch%20Scheduler%20Cache-Aware%20Load%20Balancer%20Faster%20Structured%20Outputs.md)
- [SGLang Project - 2026 - HiCache System Design and Optimization](../summaries/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md)

## 关联页面

- [SGLang](./SGLang.md)
- [PagedAttention](./PagedAttention.md)
- [vLLM](./vLLM.md)
- [SGLang 与 vLLM 架构对比](../comparisons/SGLang%20与%20vLLM%20架构对比.md)
- [注意力机制 Attention](../topics/注意力机制%20Attention.md)

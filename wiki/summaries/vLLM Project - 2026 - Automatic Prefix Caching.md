---
type: summary
status: refined
---
# vLLM Project - 2026 - Automatic Prefix Caching

## 来源信息

- 类型：官方文档 / `KV cache` 与 Automatic Prefix Caching 设计
- 发布者：vLLM Project
- 原始 HTML：[Automatic Prefix Caching](../../raw/html/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.html)
- 全文文本：[Automatic Prefix Caching](../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md)
- 官方页面：[vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- 快照日期：2026-08-04
- 状态：精修 summary；基于 vLLM V1 官方设计文档快照，不是缓存性能 benchmark

## 摘要

Automatic Prefix Caching（APC）复用先前请求已经计算过的完整 `KV-cache blocks`，从而跳过相同 prompt prefix 的重复 prefill。vLLM V1 采用 hash-chain identification：一个 block 的 hash 不只包含本 block 的精确 token tuple，还包含 parent block hash，以及 LoRA ID、多模态输入 hash、`cache_salt` 等 extra hashes。parent hash 把前缀历史递归编码进当前 key，因此只有内容和相关执行上下文一致的完整 blocks 才能被命中。

实现上，`KV cache manager` 启动时预分配全部 `KVCacheBlock` 组成 block pool。每个 block 持有不可变 `block_id`、可重置 `block_hash`、当前 `ref_cnt`，以及嵌入对象自身的双向 free-queue pointers。系统同时维护 hash key 到 block IDs 的 cache mapping、request ID 到 allocated block IDs 的 request mapping，以及只有 head / tail 外部指针的 intrusive free queue。处于 cache mapping 的 block 在 `ref_cnt = 0` 时仍可位于 free queue：它可以先被后续 prefix hit “touch” 并重新占用，也可以在内存需要时按 LRU 从队首被驱逐和复用。

这一机制的重点是缓存状态机，而不是一种通用速度保证。文档没有提供 cache hit rate、TTFT、throughput、hash overhead 或不同 block size 的 benchmark；收益取决于请求之间是否存在 block-aligned 公共前缀、缓存容量及淘汰压力。文档称 prefix caching 不改变模型输出，但 non-cryptographic hash、multi-tenant cache sharing 与 timing side channel 仍需要显式安全设计，其中 `cache_salt` 用于把不同 trust groups 的 hash chain 隔离。

## 关键事实

- **优化对象**：APC 避免相同 prefix 的重复 prompt prefill，复用的是已经算好的 `KV cache`；它不减少不同后续 output tokens 各自需要的 decode computation。
- **hash chain**：每个 full block 的 key 由 `hash(parent_hash, block_tokens, extra_hashes)` 构成。parent hash 使后续 block 的身份依赖此前所有 blocks，而不必在每个 key 中重复保存完整 prefix token list。
- **精确 block tokens**：hash components 中保留当前 block 的完整 token tuple，用于降低不同内容落到同一 key 的 collision 风险。
- **extra hashes**：LoRA IDs、多模态 input hashes 与 cache salts 等会进入 key，避免 token placeholders 相同但 adapter、图像或隔离域不同的请求错误共享状态。
- **只缓存 full blocks**：部分填充 block 不进入 APC。若 block size 为 `4`，两请求只有前 `10` tokens 相同，则最多命中前 `8` 个完整 block-aligned tokens。
- **默认 hash 算法**：文档称从 `v0.11` 起默认使用 `sha256`；它降低旧 hash key 的 collision 风险，但默认以 Python pickle serialization，hash 未必能跨 Python / vLLM version 复现。
- **可复现 hash**：`sha256_cbor` 使用 CBOR serialization，适合需要 cross-language / cross-environment deterministic key 的场景。
- **xxHash 选项**：`xxhash` 使用 Pickle + 128-bit xxHash，`xxhash_cbor` 使用 canonical CBOR + xxHash；两者需要可选 `xxhash` package，速度更高但不是 cryptographically secure。
- **collision 安全边界**：官方警告 non-cryptographic hash 理论上会增加 collision 风险，可能导致 undefined behavior，甚至在 multi-tenant 环境泄露 private information；选择算法需要在性能与安全容忍度之间权衡。
- **多模态 key**：图像 placeholder tokens 本身不足以识别实际视觉输入，因此 frontend image processor 生成的 image hash 会作为 extra hash 注入覆盖相关 placeholders 的 blocks。
- **`cache_salt` 隔离**：可选 per-request salt 被注入第一个 block 的 hash，并经 parent hash 传播到后续 chain；只有使用相同 salt 的请求才能互相 reuse cached blocks。
- **隔离目标**：`cache_salt` 用于降低攻击者通过 cache-hit latency 差异推测他人 prefix 是否已缓存的 timing attack；相同 salt 等价于显式加入同一 cache-sharing trust group。
- **block pool**：所有 `KVCacheBlock` 在 manager 初始化时一次性创建，避免运行时 Python object creation，并让 manager 始终能追踪全部 blocks。
- **block 元数据**：`block_id` 不变；`block_hash` 在 block 填满时赋值、eviction 时清除；`ref_cnt` 表示当前使用该 block 的请求数；`prev_free_block / next_free_block` 构成 intrusive doubly linked list。
- **free queue 设计**：manager 只保存 head / tail，链表指针直接位于 block 对象中，因此可以 `O(1)` 把中间元素移到队尾，也避免再用一个 Python `deque` wrapper 持有同一批对象。
- **三张核心索引**：Block Pool 保存所有 block objects；Cache Blocks 将 hash key 映射到一个或多个 block IDs；Request Blocks 将 request ID 映射到其 allocated block IDs；Free Block Queue 管理当前可重用 blocks。
- **新请求命中**：scheduler 先调用 `get_computed_blocks()`，对 prompt tokens 构造 hash chain 并查 cache mapping，得到已经计算的连续 prefix blocks。
- **Touch 操作**：`allocate_slots()` 对命中 blocks 增加 `ref_cnt`；若 block 此前无人使用而位于 free queue，则将其从队列移除，防止同一轮 allocation 把它 evict / reuse。
- **新 block 分配**：manager 从 free queue head 取 block；若队首仍是 cached block，这次分配同时执行 eviction，使旧 hash mapping 不再可命中，然后把物理 block 交给新请求。
- **运行中 append**：running request 把 token IDs 追加到已有或新 blocks 的 slots；一个 block 一旦填满，就立即加入 cache mapping，因此同 batch 中其他请求也可能复用。
- **V1 duplicate blocks**：V1 block table 是 append-only；若新生成的 full block 与既有 cached block 得到相同 hash，系统不会把已追加的物理 block ID 改写为旧 ID，所以同一 hash 可暂时对应 duplicate blocks，直到相关 request 被 free 后消除。
- **Free 顺序**：请求结束时先释放引用；`ref_cnt` 降到 `0` 的 blocks 以反向顺序加入 free queue tail，使包含更长 prefix、通常更难复用的后部 blocks 更早靠近队首并被淘汰。
- **LRU eviction**：free queue head 是 least-recently-used candidate。若它仍在 cache mapping，eviction 会弹出队首、从该 hash 对应 block IDs 中移除其 ID，并清空 block hash，随后才能复用该物理 block。

## 争议与不确定点

- 这是 2026-08-04 的官方 `stable` 设计快照。默认 hash algorithm、serialization、block metadata、append-only policy 与 eviction implementation 都可能随 vLLM 版本改变。
- “almost a free lunch” 是文档的定性表述，本页没有任何 benchmark。hash computation、Python metadata、cache lookup、touch、eviction 与安全 hash 都有成本，净收益取决于 hit rate、prefix 长度、block size 和 workload locality。
- APC 只缓存 full blocks，因此公共前缀若没有落在完整 block 边界上，尾部相同 tokens 仍需重算；“相同 prefix”不能简单等同于逐 token 的任意长度复用。
- `sha256` 大幅降低 collision 风险，但 hash identity 仍依赖 serialization 和所有必要 extra components 是否被正确纳入；文档没有把它描述成形式化 collision-free proof。
- xxHash 变体的 non-cryptographic 性质在 multi-tenant serving 中可能造成比单租户更高的安全后果。不能只因其更快就忽略文档对 undefined behavior / information leakage 的警告。
- `cache_salt` 隔离的是 prefix-cache reuse 与相关 timing signal，不是 encryption、authentication 或完整 tenant isolation。把哪些请求放入同一 salt trust group 仍是服务方的 policy responsibility。
- V1 append-only block table 允许相同 hash 暂时对应多个物理 blocks。这简化运行中映射不变性，但可能短时增加重复占用；文档没有量化 duplicate frequency 或 memory overhead。
- LRU 是对 free queue 中可回收 blocks 的局部策略，实际 cache effectiveness 仍受 active-reference blocks、请求完成顺序与 prefix distribution 影响；本页不支持“LRU 对所有 serving trace 最优”的结论。

## 关联页面

- 概念：[vLLM](../concepts/vLLM.md)
- 概念：[PagedAttention](../concepts/PagedAttention.md)
- 概念：[RadixAttention](../concepts/RadixAttention.md)
- 比较：[SGLang 与 vLLM 架构对比](../comparisons/SGLang%20与%20vLLM%20架构对比.md)
- 官方架构：[Architecture Overview](./vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md)
- 官方指南：[vLLM V1 Guide](./vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md)
- 原始论文：[Kwon et al. - 2023 - Efficient Memory Management for Large Language Model Serving with PagedAttention](./Kwon%20et%20al.%20-%202023%20-%20Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention.md)

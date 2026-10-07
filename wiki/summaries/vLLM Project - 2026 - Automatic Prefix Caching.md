---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# vLLM Project - 2026 - Automatic Prefix Caching

## TL;DR（快速导读）

vLLM 前缀缓存复用内容与执行条件一致的完整缓存块，跳过重复提示计算；它不直接加速后续逐词生成。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

多个请求长度不同，系统要决定哪些先处理、保存多少缓存及如何分配 GPU；模型权重相同也会因服务系统产生不同表现。

## 来源信息

- 类型：官方文档 / `KV cache` 与 Automatic Prefix Caching 设计
- 发布者：vLLM Project
- 原始 HTML：[Automatic Prefix Caching](../../raw/html/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.html)
- 全文文本：[Automatic Prefix Caching](../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md)
- 官方页面：[vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- 快照日期：2026-08-04
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

块标识包含当前位置内容、前面的历史和相关执行条件，避免把相似但不同的请求误当成可共享。命中率、块完整性、隔离配置和缓存回收共同影响收益，不能只看是否打开了开关。

### 方法与背景细节

Automatic Prefix Caching（APC）复用先前请求已经计算过的完整 `KV-cache blocks`，从而跳过相同 prompt prefix 的重复 prefill。vLLM V1 采用 hash-chain identification：一个 block 的 hash 不只包含本 block 的精确 token tuple，还包含 parent block hash，以及 LoRA ID、多模态输入 hash、`cache_salt` 等 extra hashes。parent hash 把前缀历史递归编码进当前 key，因此只有内容和相关执行上下文一致的完整 blocks 才能被命中。

实现上，`KV cache manager` 启动时预分配全部 `KVCacheBlock` 组成 block pool。每个 block 持有不可变 `block_id`、可重置 `block_hash`、当前 `ref_cnt`，以及嵌入对象自身的双向 free-queue pointers。系统同时维护 hash key 到 block IDs 的 cache mapping、request ID 到 allocated block IDs 的 request mapping，以及只有 head / tail 外部指针的 intrusive free queue。处于 cache mapping 的 block 在 `ref_cnt = 0` 时仍可位于 free queue：它可以先被后续 prefix hit “touch” 并重新占用，也可以在内存需要时按 LRU 从队首被驱逐和复用。

这一机制的重点是缓存状态机，而不是一种通用速度保证。文档没有提供 cache hit rate、TTFT、throughput、hash overhead 或不同 block size 的 benchmark；收益取决于请求之间是否存在 block-aligned 公共前缀、缓存容量及淘汰压力。文档称 prefix caching 不改变模型输出，但 non-cryptographic hash、multi-tenant cache sharing 与 timing side channel 仍需要显式安全设计，其中 `cache_salt` 用于把不同 trust groups 的 hash chain 隔离。

## 关键事实

- **C1**：**优化对象**：APC 避免相同 prefix 的重复 prompt prefill，复用的是已经算好的 `KV cache`；它不减少不同后续 output tokens 各自需要的 decode computation。
- **C2**：**hash chain**：每个 full block 的 key 由 `hash(parent_hash, block_tokens, extra_hashes)` 构成。parent hash 使后续 block 的身份依赖此前所有 blocks，而不必在每个 key 中重复保存完整 prefix token list。
- **C3**：**精确 block tokens**：hash components 中保留当前 block 的完整 token tuple，用于降低不同内容落到同一 key 的 collision 风险。
- **C4**：**extra hashes**：LoRA IDs、多模态 input hashes 与 cache salts 等会进入 key，避免 token placeholders 相同但 adapter、图像或隔离域不同的请求错误共享状态。
- **C5**：**只缓存 full blocks**：部分填充 block 不进入 APC。若 block size 为 `4`，两请求只有前 `10` tokens 相同，则最多命中前 `8` 个完整 block-aligned tokens。
- **C6**：**默认 hash 算法**：文档称从 `v0.11` 起默认使用 `sha256`；它降低旧 hash key 的 collision 风险，但默认以 Python pickle serialization，hash 未必能跨 Python / vLLM version 复现。
- **C7**：**可复现 hash**：`sha256_cbor` 使用 CBOR serialization，适合需要 cross-language / cross-environment deterministic key 的场景。
- **C8**：**xxHash 选项**：`xxhash` 使用 Pickle + 128-bit xxHash，`xxhash_cbor` 使用 canonical CBOR + xxHash；两者需要可选 `xxhash` package，速度更高但不是 cryptographically secure。
- **C9**：**collision 安全边界**：官方警告 non-cryptographic hash 理论上会增加 collision 风险，可能导致 undefined behavior，甚至在 multi-tenant 环境泄露 private information；选择算法需要在性能与安全容忍度之间权衡。
- **C10**：**多模态 key**：图像 placeholder tokens 本身不足以识别实际视觉输入，因此 frontend image processor 生成的 image hash 会作为 extra hash 注入覆盖相关 placeholders 的 blocks。
- **C11**：**`cache_salt` 隔离**：可选 per-request salt 被注入第一个 block 的 hash，并经 parent hash 传播到后续 chain；只有使用相同 salt 的请求才能互相 reuse cached blocks。
- **C12**：**隔离目标**：`cache_salt` 用于降低攻击者通过 cache-hit latency 差异推测他人 prefix 是否已缓存的 timing attack；相同 salt 等价于显式加入同一 cache-sharing trust group。
- **C13**：**block pool**：所有 `KVCacheBlock` 在 manager 初始化时一次性创建，避免运行时 Python object creation，并让 manager 始终能追踪全部 blocks。
- **C14**：**block 元数据**：`block_id` 不变；`block_hash` 在 block 填满时赋值、eviction 时清除；`ref_cnt` 表示当前使用该 block 的请求数；`prev_free_block / next_free_block` 构成 intrusive doubly linked list。
- **C15**：**free queue 设计**：manager 只保存 head / tail，链表指针直接位于 block 对象中，因此可以 `O(1)` 把中间元素移到队尾，也避免再用一个 Python `deque` wrapper 持有同一批对象。
- **C16**：**三张核心索引**：Block Pool 保存所有 block objects；Cache Blocks 将 hash key 映射到一个或多个 block IDs；Request Blocks 将 request ID 映射到其 allocated block IDs；Free Block Queue 管理当前可重用 blocks。
- **C17**：**新请求命中**：scheduler 先调用 `get_computed_blocks()`，对 prompt tokens 构造 hash chain 并查 cache mapping，得到已经计算的连续 prefix blocks。
- **C18**：**Touch 操作**：`allocate_slots()` 对命中 blocks 增加 `ref_cnt`；若 block 此前无人使用而位于 free queue，则将其从队列移除，防止同一轮 allocation 把它 evict / reuse。
- **C19**：**新 block 分配**：manager 从 free queue head 取 block；若队首仍是 cached block，这次分配同时执行 eviction，使旧 hash mapping 不再可命中，然后把物理 block 交给新请求。
- **C20**：**运行中 append**：running request 把 token IDs 追加到已有或新 blocks 的 slots；一个 block 一旦填满，就立即加入 cache mapping，因此同 batch 中其他请求也可能复用。
- **C21**：**V1 duplicate blocks**：V1 block table 是 append-only；若新生成的 full block 与既有 cached block 得到相同 hash，系统不会把已追加的物理 block ID 改写为旧 ID，所以同一 hash 可暂时对应 duplicate blocks，直到相关 request 被 free 后消除。
- **C22**：**Free 顺序**：请求结束时先释放引用；`ref_cnt` 降到 `0` 的 blocks 以反向顺序加入 free queue tail，使包含更长 prefix、通常更难复用的后部 blocks 更早靠近队首并被淘汰。
- **C23**：**LRU eviction**：free queue head 是 least-recently-used candidate。若它仍在 cache mapping，eviction 会弹出队首、从该 hash 对应 block IDs 中移除其 ID，并清空 block hash，随后才能复用该物理 block。

## 争议与不确定点

- 缓存命中取决于相同token/权重/adapter/多模态身份与隔离域。
- 版本默认hash与序列化行为会变化，保存文档不作永久默认承诺。

## 关联页面

- 概念：[vLLM](../concepts/vLLM.md)
- 概念：[PagedAttention](../concepts/PagedAttention.md)
- 概念：[RadixAttention](../concepts/RadixAttention.md)
- 比较：[SGLang 与 vLLM 架构对比](../comparisons/SGLang%20与%20vLLM%20架构对比.md)
- 官方架构：[Architecture Overview](./vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md)
- 官方指南：[vLLM V1 Guide](./vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md)
- 原始论文：[Kwon et al. - 2023 - Efficient Memory Management for Large Language Model Serving with PagedAttention](./Kwon%20et%20al.%20-%202023%20-%20Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention.md)

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **KV cache**：键值缓存：保存已经处理过的位置表示，生成新内容时可复用，避免全部重算。
- **prefill**：提示计算阶段：先处理输入提示，再开始逐步生成输出。
- **decode**：解码阶段：利用已有输入与生成历史，产生后续输出。
- **scheduler**：调度器：决定请求何时进入计算、每次处理多少，以及如何共享资源。
- **benchmark**：评测基准：特定数据、任务与规则的组合，分数只在这些条件下成立。

## 方法与实验解读

APC复用相同状态的完整KV块，hashchain连接前缀、token和额外输入身份；block引用与LRU控制回收。缓存可降低重复prefill，但不同输出仍各自decode。正文示例使10个共享tokens在blocksize4时只命中8个，是边界对齐而非无缘由丢失。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-1 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C2 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-1 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C3 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-1 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C4 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-1 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C5 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-7 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C6 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-1 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C7 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-1 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C8 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-1 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C9 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-1 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C10 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-1 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C11 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-1 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C12 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-1 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C13 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-2 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C14 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-2 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C15 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-2 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C16 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-2 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C17 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-4 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C18 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-4 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C19 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-4 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C20 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-4 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C21 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-4 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C22 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-5 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C23 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md#source-section-6 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |

## 核证范围

保留23项分析，核读hash/extra/隔离、pool、分配/重复/释放/eviction及完整例子。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

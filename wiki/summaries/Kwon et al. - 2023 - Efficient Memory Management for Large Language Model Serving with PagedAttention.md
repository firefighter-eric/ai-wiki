---
type: summary
status: refined
---
# Kwon et al. - 2023 - Efficient Memory Management for Large Language Model Serving with PagedAttention

## 来源信息

- 类型：论文 / LLM serving system
- 原始文件：[PDF](../../raw/pdf/Kwon%20et%20al.%20-%202023%20-%20Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention.pdf)
- 原始 HTML：[HTML](../../raw/html/Kwon%20et%20al.%20-%202023%20-%20Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention.html)
- 全文文本：[Markdown](../../raw/text/Kwon%20et%20al.%20-%202023%20-%20Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention.md)
- 作者：Woosuk Kwon、Zhuohan Li、Siyuan Zhuang、Ying Sheng、Lianmin Zheng、Cody Hao Yu、Joseph E. Gonzalez、Hao Zhang、Ion Stoica
- 会议：ACM SIGOPS 29th Symposium on Operating Systems Principles（SOSP '23）
- 年份：2023
- DOI：`10.1145/3600006.3613165`
- arXiv：`2309.06180`
- 状态：精修 summary；已交叉核对原始 HTML、PDF 与全文文本

## 摘要

这篇论文把高吞吐 LLM serving 的关键瓶颈定位到动态 `KV cache` 的内存管理。自回归生成中，每个请求的 `KV cache` 会随 token 逐步增长，最终长度又无法预知；而当时的 serving system 通常按最大序列长度预留连续内存，从而同时产生未来 token 的预留空间、内部碎片与外部碎片。论文测得，相关既有方案中实际用于保存 token state 的 `KV cache` 容量仅占 `20.4%–38.2%`，这直接压缩了可并发 batch 的规模。

论文提出 `PagedAttention`：将一条序列逻辑连续的 `KV cache` 切成固定 token 数量的 `KV block`，通过 `block table` 把逻辑块映射到可以物理不连续的 GPU 内存块。基于这一 primitive，作者构建 `vLLM`，把集中式调度器、`KV cache manager`、CPU/GPU block allocator、抢占恢复、block-level sharing 与分布式 GPU worker 组织成端到端 serving engine。它不仅按需增长 `KV cache`，还通过引用计数和 `copy-on-write` 在 parallel sampling、beam search 及预定义 shared prefix 中复用物理块。

论文的核心结论不是“PagedAttention kernel 比连续内存 attention 更快”，而是：在 `KV cache` 主导容量、系统吞吐受内存约束的场景中，更高的内存利用率允许更多请求同时进入 batch，从而抵消间接寻址的 kernel 开销并提高端到端吞吐。作者在 2023 年的 OPT / LLaMA、A100 与合成 serving trace 设置下，报告 vLLM 相比 FasterTransformer 和其重实现的 Orca 基线总体可获得约 `2–4×` 吞吐提升；收益在长序列、大模型和可共享 `KV cache` 的复杂 decoding 中更明显。

## 关键事实

- **问题规模**：以论文使用的 FP16 OPT-13B 为例，一个 token 的 `KV cache` 约为 `800 KB`，`2048` token 的单请求最多约需 `1.6 GB`。该数字由模型层数、hidden size、attention 结构与精度共同决定，不是所有模型的固定常数。
- **既有内存浪费**：连续 chunk 预分配包含三类占用——尚未生成 token 的 reserved slots、按最大长度过度配置造成的 internal fragmentation、以及不同 chunk 尺寸造成的 external fragmentation。论文 profiling 中只有 `20.4%–38.2%` 的 `KV cache` 空间保存了真实 token state。
- **PagedAttention 语义**：逻辑 `KV block` 按固定数量的 token positions 切分 key/value；论文实现为不同层和 attention heads 分别维护 blocks / block tables。kernel 根据映射分块读取非连续物理块，但执行的仍是原有 attention 计算，并未用近似结果换取内存效率。
- **按需分配**：逻辑块从左到右填充，只有末块可能留有未填 slot；新物理块仅在前一块填满后分配。因此单序列由碎片造成的浪费被限制在一个 block 内，所有请求结束后其物理块可立即回收。
- **控制面架构**：`vLLM` 使用 centralized scheduler 协调 distributed GPU workers；scheduler 内的 `KV cache manager` 维护每条序列的逻辑块到物理块映射，并通过 CPU/GPU block allocator 管理可用块。
- **每轮执行**：scheduler 先选择本轮候选序列并分配新块，再把 token IDs 与各请求的 block table 广播给 GPU workers；worker 按映射读取旧 `KV cache`、写入新状态、执行模型并把采样 token 返回 scheduler。
- **共享与 `copy-on-write`**：parallel sampling 可共享 prompt blocks；beam search 可共享仍相同的候选前缀；预定义 shared prefix 也可映射到预留物理块。共享块由引用计数管理，只有某条序列需要改写仍被多方引用的最后一块时才复制该 block。
- **统一 decoding 抽象**：实现用 `fork`、`append`、`free` 三个基本操作表示 parallel sampling、beam search 和 prefix sharing，使模型 kernel 只看到物理 block ID，而不直接处理上层共享模式。
- **调度与抢占**：论文版本采用 `FCFS`；到达最早的请求优先，最新请求优先被抢占。同一请求中的多条序列作为 sequence group 进行 gang scheduling，并采用 all-or-nothing eviction，避免拆散存在共享关系的 blocks。
- **恢复机制**：被抢占序列可以把全部 blocks swap 到 CPU RAM，或在恢复时把已生成 token 与原 prompt 拼成一次 prompt phase 来 recompute `KV cache`。论文发现小 block 更利于 recomputation，大 block 更利于 swapping，`16–64` 的中等 block size 下两者端到端表现接近。
- **分布式执行**：论文支持 Megatron-LM 风格 tensor parallelism。各 worker 接收相同的物理 block ID 映射，但只保存其 attention heads 对应的 `KV cache` 分片；模型 shard 之间用 NCCL `all-reduce` 同步中间结果，内存管理不要求 workers 彼此同步。
- **实现构成**：论文版本以 FastAPI 提供扩展 OpenAI API 的 frontend；控制面 scheduler / block manager 主要用 Python，实现关键 `PagedAttention`、block reshape / write 和 block copy 的 fused CUDA kernels，并以 PyTorch / Transformers 实现 GPT、OPT、LLaMA executor。
- **基础 sampling 结果**：在 ShareGPT trace 上，vLLM 在相近 normalized latency 下可承受比 `Orca (Oracle)` 高 `1.7–2.7×`、比 `Orca (Max)` 高 `2.7–8×` 的请求率；相对 FasterTransformer 的最高请求率差距可达 `22×`，但这同时包含后者缺乏细粒度调度的影响，不能归因于 PagedAttention 单一因素。
- **共享收益**：Alpaca 实验中，parallel sampling 节省 `6.1%–9.8%` blocks，beam search 节省 `37.6%–55.2%`；ShareGPT 中相应范围为 `16.2%–30.5%` 与 `44.3%–66.3%`。预定义 one-shot / few-shot prefix sharing 相比 `Orca (Oracle)` 分别报告 `1.67×` / `3.58×` 吞吐。
- **block size 取舍**：block 越小，碎片更少、共享概率更高，但 GPU 并行读取效率可能下降；block 越大则相反。论文实验选择 `16` tokens 作为默认值，而不是宣称存在适合所有 workload 的固定最优值。

## 争议与不确定点

- `Orca` 当时没有公开实现，论文中的三种 Orca baseline 均由作者重实现；其中 `Orca (Oracle)` 预先知道真实输出长度，只是不可实现的性能上界。因此跨系统倍数需要结合这一 baseline 构造理解。
- `PagedAttention` 的动态映射会增加 block table 访问、分支和变长序列处理；论文 microbenchmark 中其 attention kernel latency 比高度优化的 FasterTransformer 高 `20%–26%`。端到端收益来自更大的有效 batch，而不是 kernel 层无条件加速。
- “near-zero waste” 不等于绝对零浪费：每条活动序列的最后一个 block 仍可能未填满，block size 也会在碎片、共享概率和 GPU 利用率之间产生权衡。
- 当可用 `KV cache` 空间本来充裕、序列很短或 workload 已转为 compute-bound 时，vLLM 相对优势会缩小；论文也指出，在不具备动态分配且 memory-bound 特征的 GPU workload 中，分页间接寻址甚至可能降低性能。
- 论文实验基于 2023 年的软件版本、A100、OPT-13B/66B/175B、LLaMA-13B，以及由 ShareGPT / Alpaca 长度分布合成的 Poisson arrival trace。其结果不能直接代表后续 vLLM 版本、其他 attention 结构、更新硬件或真实生产流量。
- 论文没有把 iteration-level scheduling 归为 vLLM 独创；它明确认为 Orca 的细粒度调度与 PagedAttention 的内存管理是互补技术。系统优势来自两者与 block-level scheduling / sharing 的组合。

## 关联页面

- 概念：[vLLM](../concepts/vLLM.md)
- 概念：[PagedAttention](../concepts/PagedAttention.md)
- 概念：[RadixAttention](../concepts/RadixAttention.md)
- 比较：[SGLang 与 vLLM 架构对比](../comparisons/SGLang%20与%20vLLM%20架构对比.md)
- 当前架构：[vLLM Project - 2026 - Architecture Overview](./vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md)
- 概念：[Transformer](../concepts/Transformer.md)
- 概念：[FlashAttention](../concepts/FlashAttention.md)
- 主题：[注意力机制 Attention](../topics/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%20Attention.md)

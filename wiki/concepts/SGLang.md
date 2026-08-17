---
type: concept
---
# SGLang

## 简介

`SGLang` 是面向 structured Language Model Programs 的 frontend/runtime 系统。它优化的对象不只是一次 LLM generation call，而是由多次调用、prompt state、structured inputs/outputs、分支、同步、Python control flow 与多模态输入构成的完整程序。arXiv v2 / NeurIPS 2024 论文把架构概括为两部分：嵌入 Python 的 Structured Generation Language，以及为 open-weight models 提供的 SGLang Runtime（SRT）和面向 API-only models 的 endpoint backends。

论文中的三项核心 runtime contribution 是：用 RadixAttention 自动复用 prefix KV cache，用 Compressed Finite State Machine 加速 regex-constrained structured decoding，以及用 API speculative execution 合并 black-box endpoint 上可能重复的多次调用。interpreter 是正文默认执行模式；compiler 是附录中的受限、探索性路径。本页严格描述该 v2 版本，不把之后项目版本的能力倒推回论文，也不代表当前 SGLang 与当前 vLLM 的实时比较。

## 关键属性

- 类型：LLM programming language / compiler / inference runtime
- 优化粒度：跨 primitive、跨 generation call、跨 program 的整体执行，而不只是一条独立 request
- 前端语言：Python-embedded DSL；primitives 包括 `extend` / `+=`、`gen`、`select`、变量读取、`fork` / `join`、`image` / `video`，其中 `gen(regex=...)` 可声明输出约束
- Interpreter：把 prompt state 建模为 asynchronous stream，通过后台 stream executor、non-blocking submission 与 event synchronization 自动执行 intra-program parallelism
- Compiler：通过 tracing 生成带 stream 内和 stream 间依赖的 computational graph，再由 graph executor 执行；论文正文默认使用 interpreter，compiler 不支持 data-dependent control flow
- Open-weight runtime：SGLang Runtime（SRT），实现 RadixAttention 与 Compressed FSM，并与 continuous batching、paged attention、tensor parallelism 等推理优化组合
- API runtime path：调用 OpenAI、Anthropic 等 black-box endpoints，并可使用 API speculative execution 尝试复用一次调用越过 stop condition 后的额外输出
- KV cache 组织：以 token sequence 为 key、KV tensors 为 value 的 radix tree；树在 CPU 维护，cache tensors 位于 paged memory pool
- 淘汰与并发安全：LRU leaf eviction 加 reference counter，正在被 running requests 引用的节点不可淘汰
- 前后端协同：runtime 可自动匹配完整 prompt 的最长缓存前缀；interpreter 还能把 `fork` 暴露出的共享结构作为 scheduling hint
- Structured decoding：把 regex 转为 character/string FSM，压缩连续 singular transitions，并通过 jump-forward 在一个 model forward 中处理多个确定 tokens
- 版本边界：本文结论对应 arXiv 2312.07104v2、2024 年实验栈与 SRT；后续 v0.4、HiCache 等设计属于项目演进，不是 v2 的既有架构
- 代表来源：[Zheng et al. - 2024 - SGLang Efficient Execution of Structured Language Model Programs](../summaries/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md)

## 相关主张

- SGLang 的核心判断是：复杂 LLM 应用中的性能机会存在于 **program structure** 中。前端表达 prompt state、约束、依赖与分支后，runtime 才能系统利用跨调用并行、KV cache sharing、structured-output jump-forward 或 endpoint-call reuse。
- RadixAttention 与一般的 per-request KV cache allocation 不同：完成请求留下的 prompt 和 generation KV cache 仍可成为后续请求的 prefix cache；radix tree 使规则前缀、分叉树和多轮增长 prompt 等复用模式共享同一套匹配与淘汰机制。
- `fork` / `join` 不只是语法便利。它们显式暴露 prompt state 的复制、分支与汇合，使 interpreter 能自动并行独立 generation calls，并帮助 runtime 提前建立共享前缀。
- Compressed FSM 的关键不是只把非法 token probability 设为零，而是识别约束中未来字符串已唯一确定的区段，在一次 forward pass 中推进多个 tokens。retokenization 处理 token boundary，却不能消除不同 tokenizations 造成的 probability distortion。
- API speculative execution 与常见 draft-model speculative decoding 不同：它让一次 black-box call 多生成一段，再由 interpreter 与后续 primitives 做模板匹配。只有匹配准确时，才能减少 endpoint calls、latency 与重复 input-token cost。
- SGLang 的性能不是由 RadixAttention 单独产生。v2 的结果来自 prefix reuse、interpreter parallelism、cache-aware scheduling、Compressed FSM 和 workload structure；open-weight 与 API-only paths 使用的优化机制也不同。
- SGLang 与传统单次 Completion-style serving API 的首要差异是优化边界：后者把 requests 当作彼此独立，论文中的 SGLang 保留跨调用的程序结构。这不意味着所有 workload 都会更快；论文在 long-output multi-turn chat 上就观察到 decode 主导时几乎没有 speedup。
- compiler 路径提供 graph rewriting、serialization 和较低解释开销，但论文版本只能覆盖可 tracing、无 data-dependent control flow 的子集；完整 Python flexibility 主要由 interpreter 路径保留。
- 论文中的 GPT-4-assisted code movement 是探索性机制：它可以增加可共享前缀，却可能改变自然语言 prompt 的语义，因此不能与严格 semantics-preserving compiler optimization 等同。
- `Kimi K3` 官方模型仓库把 SGLang 列为推荐 inference engine，但 K3 的 hybrid KDA–MLA state 与普通 full-attention KV cache 不同；部署时必须核对对应 K3 cookbook、kernel 和 preserved-thinking-history 支持，不能从“支持 SGLang”反推任意历史版本都可直接运行。

## 来源支持

- [Zheng et al. - 2024 - SGLang Efficient Execution of Structured Language Model Programs](../summaries/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md)
- [Moonshot AI - 2026 - Kimi K3 Model Repository](../summaries/Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20Model%20Repository.md)

## 关联页面

- [RadixAttention](./RadixAttention.md)
- [PagedAttention](./PagedAttention.md)
- [vLLM](./vLLM.md)
- [SGLang 与 vLLM 架构对比](../comparisons/SGLang%20与%20vLLM%20架构对比.md)
- [SGLang v0.4 版本说明](../summaries/SGLang%20Team%20-%202024%20-%20SGLang%20v0.4%20Zero-Overhead%20Batch%20Scheduler%20Cache-Aware%20Load%20Balancer%20Faster%20Structured%20Outputs.md)
- [HiCache System Design and Optimization](../summaries/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md)
- [Transformer](./Transformer.md)
- [注意力机制 Attention](../topics/注意力机制%20Attention.md)
- [Kimi K3](./Kimi%20K3.md)
- [Kimi Delta Attention](./Kimi%20Delta%20Attention.md)

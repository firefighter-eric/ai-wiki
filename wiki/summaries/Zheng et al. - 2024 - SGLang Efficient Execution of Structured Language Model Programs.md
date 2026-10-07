---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Zheng et al. - 2024 - SGLang Efficient Execution of Structured Language Model Programs

## TL;DR（快速导读）

SGLang 将多次模型调用组织成程序，并用运行时复用共同前缀、加速约束输出与执行。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

先提取信息，再并行判断，最后合成答案，是一个多调用流程；共享前缀与调度会影响整体成本。

## 来源信息

- 类型：论文 / 系统论文
- 论文正文标题：SGLang: Efficient Execution of Structured Language Model Programs
- arXiv：2312.07104v2
- 版本：arXiv v2 / NeurIPS 2024 最终版本
- 原始 PDF：[Zheng et al. - 2024 - SGLang Efficient Execution of Structured Language Model Programs.pdf](../../raw/pdf/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.pdf)
- 原始 HTML：[Zheng et al. - 2024 - SGLang Efficient Execution of Structured Language Model Programs.html](../../raw/html/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.html)
- 全文文本：[Zheng et al. - 2024 - SGLang Efficient Execution of Structured Language Model Programs.md](../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md)
- 作者：Lianmin Zheng、Liangsheng Yin、Zhiqiang Xie、Chuyue Sun、Jeff Huang、Cody Hao Yu、Shiyi Cao、Christos Kozyrakis、Ion Stoica、Joseph E. Gonzalez、Clark Barrett、Ying Sheng
- 年份：2024
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

前端管理生成状态、控制流和并行，后端利用程序结构安排计算。论文重点包括前缀缓存、压缩状态机与 API 推测执行；它们解决不同瓶颈，不能把效果概括为所有模型请求都同样加速。

### 方法与背景细节

这篇论文把包含多次 LLM calls、control flow、structured inputs/outputs 与多模态数据的应用抽象为 **Language Model Programs（LM Programs）**。SGLang 由可独立使用、又可协同优化的两部分组成：前端是嵌入 Python 的 Structured Generation Language，用 primitives 管理 generation、prompt state、约束和并行；后端是 SGLang Runtime（SRT），从程序结构中提取传统单请求 serving API 看不到的复用与执行机会。interpreter 是论文正文默认执行方式，compiler mode 则作为附录中的受限路径讨论。

arXiv v2 / NeurIPS 版本把 runtime contribution 明确收敛为三条主线。第一，RadixAttention 用 radix-tree LRU cache 自动复用跨 calls、跨 program instances 的 prefix KV cache；第二，Compressed Finite State Machine（Compressed FSM）把约束中连续的唯一转移压缩，使 structured decoding 可以一次前向跳过多个确定 tokens；第三，API speculative execution 面向 GPT-4 等 black-box endpoints，让前一次调用越过 stop condition 多生成一段，并由 interpreter 尝试把多余输出复用于后续 primitives。三者分别优化 open-weight model 的 prefix computation、constrained decoding，以及 API-only model 的调用 latency 与 input-token cost。

## 关键事实

- **C1**：SGLang 是 Python-embedded DSL。`gen` 触发 generation 并可通过 `regex` 参数约束输出，`select` 从候选中选择最高概率项，`extend` / `+=` 追加 prompt，`[variable_name]` 读取生成变量，`fork` / `join` 表达 prompt state 的分叉与汇合，`image` / `video` 接收多模态输入。
- **C2**：interpreter 把 prompt state 视为 asynchronous stream。`extend`、`gen`、`select` 等操作以 non-blocking 方式提交到后台 stream executor；只有读取尚未完成的结果时才阻塞，从而在保持 Python control flow 的同时自动暴露 intra-program parallelism。
- **C3**：open-weight models 通过 SGLang Runtime（SRT）执行；OpenAI、Anthropic 等 API models 走 endpoint backend。论文强调 frontend 与 runtime 可协同，也可分别独立使用。
- **C4**：compiler mode 通过 tracing 把可支持的程序转成 computational graph，IR 表达 primitive nodes、stream 内顺序依赖和 stream 间同步依赖。它是附录中的探索性路径，不是论文主实验的默认模式，也不支持 data-dependent control flow。
- **C5**：compiler case study 用 GPT-4 重排自然语言 graph nodes 以增加 constant prefix：15 个测试 templates 中有 12 个经人工检查被认为未改变语义，平均增加 60 个 shareable prefix tokens；失败来自对 prompt 语义的误判，因此该变换不严格 preserve semantics。
- **C6**：RadixAttention 把 token sequence 到 KV cache tensors 的映射保存在 radix tree 中，对 prompt 与 generation results 统一执行 prefix search、reuse、insertion 与 eviction。KV tensors 采用 non-contiguous paged layout，论文实现中一页对应一个 token；tree metadata 位于 CPU。
- **C7**：cache 与 running requests 共享同一 memory pool，而不是预留固定 cache partition。等待请求需要更大 batch 时，系统可以逐出全部可淘汰 cache tokens，把空间让给运行中请求。
- **C8**：eviction 从 least-recently-used leaves 开始，以尽量保留仍能被多个分支复用的 common ancestors。每个 node 有 reference counter，当前 batch 正在使用的 node 不可淘汰。
- **C9**：frontend 发送完整 prompt，runtime 自动 prefix-match。执行 `fork` 时，interpreter 先发送公共 prefix 作为 frontend hint，确保共享节点进入 tree，再提交剩余 branches；ablation 表明取消 hint 或 frontend parallelism 都会降低性能。
- **C10**：cache-aware scheduler 优先 matched prefix 较长的 requests，近似以 DFS 顺序访问 request radix tree。论文证明在 offline batch、cache size 不小于最大 request length 时，DFS / longest-shared-prefix-first 可达到最优 hit rate；但也明确指出 greedy policy 可能造成 starvation，公平调度仍是 future work。
- **C11**：RadixAttention 与 continuous batching、paged attention、tensor parallelism 兼容。tensor parallelism 下各 GPU 保存 sharded KV cache；data-parallel appendix 则让 workers 各维护 sub-tree、router 维护 weakly consistent meta-tree，在 locality 与并行负载之间做 dispatch。
- **C12**：structured output 约束先由 regex 转为基于 character/string 的 FSM。若一个 state 只有一个 successor，且 edge 只有唯一允许 string/character，系统把连续 singular-transition edges 合并成 compressed edge；运行时的 jump-forward 可在一个 forward pass 中处理多个确定 tokens，而不是逐 token mask-and-decode。
- **C13**：compressed edge 的字符串仍需按原 tokenizer 重新 tokenization。论文通过 retokenization 对齐模型的真实 token boundaries，并报告只有少量额外开销；但也承认字符串约束与 token probability 之间可能出现 distorted probability，特别是不同 choice 对应不等长 tokenizations 时，尚无根本解决方案。
- **C14**：API speculative execution 用于不能修改 model runner 的 black-box API。第一次 call 可以忽略 stop condition 多生成少量 tokens，interpreter 保存这些额外输出，并与后续 primitives 的预期模板匹配；若匹配成功，就能少发一次 endpoint call，节省 latency 与重复 context 的 input-token fee。
- **C15**：API speculative execution 的正确复用依赖 prompt engineering 和模型是否按模板继续生成。它不是 open-weight speculative decoding，也不通过 draft model 验证 token；其投机对象是后续 LM-program calls。
- **C16**：baselines 为 Guidance 0.1.8（llama.cpp）、vLLM 0.2.5（default API server）与 LMQL 0.7.3（Hugging Face Transformers）。论文脚注明确说更新版 vLLM 已部分集成 RadixAttention 作为 optional experimental feature，因此选择更早版本比较。
- **C17**：open-weight models 覆盖 Llama-2 7B/70B、Mixtral-8x7B、LLaVA image/video models，API test 使用 GPT-3.5；主要硬件是 A10G 24GB，另有 A100 80GB。workloads 包括 MMLU、HellaSwag、ReAct / generative agents、Tree-of-thought、Skeleton-of-thought、JSON decoding、multi-turn chat 与 DSPy RAG。
- **C18**：作者报告相对所选 systems 的最大 throughput improvement 为 6.4×、最大 latency reduction 为 3.7×。各 benchmark 的收益来源不同：prefix reuse、单程序并行或 Compressed FSM，不能把这些最大值理解为每个 workload 的统一提升。
- **C19**：benchmark cache hit rate 为 50%–99%；cache-aware scheduling 平均达到各 workload 理论 optimal hit rate 的 96%。multi-turn chat 中短输出收益更明显，而长输出因 decode time 主导且 session 间共享少，论文观察到几乎没有 speedup。
- **C20**：multi-modal benchmark 相对 model authors 的原始 Hugging Face implementations 自报最高约 6× throughput。Chatbot Arena 一个月的 production observation 中，LLaVA-NeXT-34B 与 Vicuna-33B 的 RadixAttention hit rate 分别为 52.4% 和 74.1%，Vicuna-33B first-token latency 平均降低 1.7×。
- **C21**：GPT-3.5 三字段抽取实验中，few-shot prompting 下 API speculative execution 被描述为具有较高匹配准确率，并把 input-token cost 降至约三分之一；论文未在该段给出精确 accuracy 数值。
- **C22**：ablation 中，Compressed FSM 使 JSON decoding throughput 提升 1.6×；若不跨 batch 复用预处理后的 FSM、而为每个 request 重做 preprocessing，throughput 会低 2.4×。无 cache-reuse 的 ShareGPT 测试总计 74.3 秒，其中 RadixAttention data-structure management 为 0.2 秒，即低于 0.3%。

## 争议与不确定点

- 历史baseline版本较旧，最高倍数不能当当前框架比较。
- 约束保证可解析结构，不保证生成事实或业务执行正确。

## 关联页面

- 概念：[SGLang](../concepts/SGLang.md)
- 概念：[RadixAttention](../concepts/RadixAttention.md)
- 概念：[PagedAttention](../concepts/PagedAttention.md)
- 比较：[SGLang 与 vLLM 架构对比](../comparisons/SGLang%20与%20vLLM%20架构对比.md)
- 后续版本：[SGLang Team - 2024 - SGLang v0.4 Zero-Overhead Batch Scheduler Cache-Aware Load Balancer Faster Structured Outputs](./SGLang%20Team%20-%202024%20-%20SGLang%20v0.4%20Zero-Overhead%20Batch%20Scheduler%20Cache-Aware%20Load%20Balancer%20Faster%20Structured%20Outputs.md)
- 后续设计：[SGLang Project - 2026 - HiCache System Design and Optimization](./SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md)
- 概念：[Transformer](../concepts/Transformer.md)
- 主题：[注意力机制 Attention](../topics/注意力机制%20Attention.md)

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **KV cache**：键值缓存：保存已经处理过的位置表示，生成新内容时可复用，避免全部重算。
- **decode**：解码阶段：利用已有输入与生成历史，产生后续输出。
- **scheduler**：调度器：决定请求何时进入计算、每次处理多少，以及如何共享资源。
- **runtime**：运行时：负责实际执行程序、管理状态和安排计算的系统。
- **benchmark**：评测基准：特定数据、任务与规则的组合，分数只在这些条件下成立。

## 方法与实验解读

SGLang把多调用程序、共享prefix和格式约束交给runtime协同优化。RadixAttention减少重复prefill，FSM跨唯一路径减少调用，API投机复用后续模板；前端并行、缓存和decode收益不能相乘。编译重排是探索性附录且不严格保持prompt语义，应与主实验分开。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-4 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C2 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-4 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C3 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-4 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C4 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-29 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C5 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-30 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C6 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-5 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C7 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-5 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C8 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-5 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C9 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-5 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C10 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-5 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C11 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-5 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C12 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-7 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C13 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-7 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C14 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-8 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C15 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-8 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C16 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-10 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C17 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-10 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C18 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-11 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C19 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-11 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C20 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-11 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C21 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-11 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C22 | [原文]( ../../raw/text/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md#source-section-12 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |

## 核证范围

保留22项细节，核读DSL、异步、Radix、FSM/retokenization、API投机、setup/结果/消融和compiler附录。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

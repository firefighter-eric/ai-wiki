---
type: summary
status: refined
---
# SGLang Team - 2024 - SGLang v0.4 Zero-Overhead Batch Scheduler Cache-Aware Load Balancer Faster Structured Outputs

## 来源信息

- 类型：项目发布博客 / 版本说明
- 来源标题：SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs
- 来源 URL：https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/
- 原始 HTML：[SGLang Team - 2024 - SGLang v0.4 Zero-Overhead Batch Scheduler Cache-Aware Load Balancer Faster Structured Outputs.html](../../raw/html/SGLang%20Team%20-%202024%20-%20SGLang%20v0.4%20Zero-Overhead%20Batch%20Scheduler%20Cache-Aware%20Load%20Balancer%20Faster%20Structured%20Outputs.html)
- 全文文本：[SGLang Team - 2024 - SGLang v0.4 Zero-Overhead Batch Scheduler Cache-Aware Load Balancer Faster Structured Outputs.md](../../raw/text/SGLang%20Team%20-%202024%20-%20SGLang%20v0.4%20Zero-Overhead%20Batch%20Scheduler%20Cache-Aware%20Load%20Balancer%20Faster%20Structured%20Outputs.md)
- 作者：SGLang Team
- 日期：2024-12-04
- 对应版本：SGLang v0.4
- 状态：精修 summary，已基于发布博客全文核对

## 摘要

SGLang v0.4 的更新重点不是重新定义 SGLang 的编程语言，而是继续把 RadixAttention 所暴露的 prefix reuse 扩展到更完整的 serving path。该版本同时处理四类瓶颈：用 overlap scheduler 隐藏 CPU batch scheduling 开销；用 cache-aware load balancer 在 data-parallel workers 之间保留 prefix locality；针对 DeepSeek MLA 设计 Data Parallelism Attention（DPA）以减少 tensor parallelism 下的 KV cache 重复；并以 XGrammar 替换更慢的 grammar backend，加速 JSON 等 structured outputs。

这四项改进位于不同层级。overlap scheduler 优化单个 worker 内 CPU/GPU pipeline；cache-aware router 决定请求应落到哪个 worker；DPA 改变 DeepSeek 模型的多 GPU attention/MoE 数据流；XGrammar 则优化 constrained decoding。发布博客将它们并列为 v0.4 性能升级，但各自的数字来自不同硬件、模型与 workload，不能相加，也不能视为所有请求共享的统一加速比例。

## 关键事实

### Zero-Overhead Batch Scheduler

- LLM serving 除 GPU computation 外，还需要 CPU 完成 batch scheduling、memory allocation 与 prefix matching。v0.4 让 scheduler 提前一个 batch 准备下一批所需 metadata，使 CPU scheduling 与当前 batch 的 GPU computation 重叠，从而隐藏 radix cache operations 等 CPU 开销。
- 实现需要用 future tokens 处理前后 batch 的依赖，并精细安排 CUDA events 与 synchronization；因此“zero-overhead”指 CPU 开销被 GPU 计算覆盖，而不是调度工作本身消失。
- 项目方用 Nsight profile 展示连续五个 decoding batches 间 GPU 没有 idle gap。该观察对应 Triton attention backend；博客同时承认 FlashInfer backend 当时仍有 minor gap。
- overlap scheduler 在 v0.4 默认开启，可用 `--disable-overlap` 回退到旧路径。项目方自报相对 v0.3 为 1.1× speedup、相对其他未逐一说明的 state-of-the-art baselines 为 1.3×，且小模型和较大 tensor-parallel size 下收益更明显。

### Cache-Aware Load Balancer

- 普通 round-robin 会把共享 prefix 的请求分散到不同 workers，降低每个 worker 的 KV cache hit rate。v0.4 router 预测各 worker 的 prefix match，优先选择命中率较高者，同时仍做负载均衡以避免 worker imbalance。
- router 在自身维护 workers 实际 radix trees 的 approximate radix tree，并进行 lazy update；它利用请求传递的信息近似 cache state，不要求 workers 之间同步 cache metadata，因此被描述为 communication-free design。
- router 支持多机 workers，由独立 Rust package `sglang-router` 提供 CLI 与 Python bindings。博客还自报 Rust 实现相对 Python alternatives 有约 2× 性能，但未在正文提供这一数字的完整 benchmark protocol。
- 项目方在 8 张 A100 80GB、多个长共享 prefix groups 且各组 perfectly balanced 的合成 workload 上报告：throughput 从 82,665 token/s 增至 158,596 token/s，cache hit rate 从 20% 增至 75%，分别约为 1.9× throughput 和 3.8× hit rate。博客明确提示结果会随 workload characteristics 变化，并称 worker 数增加时收益扩大。

### Data Parallelism Attention（DPA）

- DeepSeek 模型的 MLA 只有一个 KV head；若直接使用 8-way tensor parallelism，KV cache 会跨 GPU 重复。v0.4 对 attention 部分采用 data parallelism，使各 DP worker 独立处理不同的 prefill、decode 或 idle batches，以减少 KV cache 占用并扩大可用 batch size。
- attention 处理后的数据会在进入 Mixture-of-Experts（MoE）层前在 workers 间执行 all-gather，经过 MoE 后再分发回各 worker。因此 DPA 不是整个模型完全无通信的数据并行，而是 attention 与 MoE 之间有显式 collective/data redistribution 的混合执行方式。
- 项目方在 8 张 H100 80GB、DeepSeek-Coder-V2-Instruct-FP8、10,000 个 random requests 的设置下，自报相对 v0.3 有 1.9× decoding throughput。复现命令使用 1-token input、512-token output，并在两个对照配置中关闭 radix cache；该结果针对 decode-heavy benchmark。
- 发布时 DPA 仅支持 DeepSeek models，需显式启用 `--enable-dp-attention`。博客把 expert parallelism for MoE 列为后续继续优化的方向。

### XGrammar 与 Structured Outputs

- SGLang 原有 structured decoding 使用 Compressed Finite State Machine；v0.4 新增 XGrammar backend，以更快的 grammar execution 支持 JSON 等受约束输出。
- 用户可通过 `--grammar-backend xgrammar` 启用，并继续使用 OpenAI-compatible API 请求 structured output。
- 项目方声称 SGLang + XGrammar 在 JSON decoding tasks 上相对其他 open-source solutions 最高可达 10×，但本篇博客没有给出对手版本、硬件、schema 复杂度和完整测量表，而是把详细论证链接到 XGrammar 的另一篇博客。

## 争议与不确定点

- 本来源是项目方发布博客，不是独立评测或 peer-reviewed paper。所有 1.1×、1.3×、1.9×、2×、3.8×、10× 数字均为 SGLang 团队在特定版本与设置下的自报结果；正文未统一提供误差范围、重复次数或完整 baseline controls。
- 四项优化的 benchmark 不共享同一模型、硬件、流量与指标，不能把倍数相乘或汇总成“SGLang v0.4 总体快多少”。
- cache-aware router 的主结果来自长共享前缀、分组完全平衡的合成 workload，天然有利于 cache locality；prefix 较短、重复率低、到达分布不均或负载压力主导时，实际收益可能明显不同。
- “zero-overhead”是 pipeline overlap 的效果描述，并非 CPU scheduling、allocation 或 radix lookup 没有成本；当 GPU computation 不足以覆盖 CPU 工作，或使用博客所述仍有 gap 的 backend 时，残余 overhead 仍可能暴露。
- DPA 的 1.9× 数字针对 DeepSeek-Coder-V2、8×H100 和极短输入/长输出 benchmark，且关闭 radix cache；不能直接外推到 prefill-heavy、非 MLA、较少 GPU 或不同 MoE communication topology。
- XGrammar 的最高 10× 缺少本来源内可独立复核的完整实验细节，应视为发布时的方向性性能声明，而不是跨 schema、模型与 backend 的稳定保证。
- 页面记录的是 2024-12-04 的 v0.4 版本快照。SGLang、FlashInfer、XGrammar、DeepSeek support 与 router 此后均可能演进，不能据此直接判断当前版本能力。

## 关联页面

- 概念：[SGLang](../concepts/SGLang.md)
- 概念：[RadixAttention](../concepts/RadixAttention.md)
- 比较：[SGLang 与 vLLM 架构对比](../comparisons/SGLang%20与%20vLLM%20架构对比.md)
- 来源：[Zheng et al. - 2024 - SGLang Efficient Execution of Structured Language Model Programs](./Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md)
- 概念：[Transformer](../concepts/Transformer.md)
- 主题：[注意力机制 Attention](../topics/注意力机制%20Attention.md)

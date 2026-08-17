---
type: summary
status: refined
---
# vLLM Project - 2026 - Architecture Overview

## 来源信息

- 类型：官方文档 / 系统架构说明
- 发布者：vLLM Project
- 原始 HTML：[Architecture Overview](../../raw/html/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.html)
- 全文文本：[Architecture Overview](../../raw/text/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md)
- 官方页面：[vLLM Architecture Overview](https://docs.vllm.ai/en/stable/design/arch_overview/)
- 快照日期：2026-08-04
- 状态：精修 summary；基于官方 stable 文档快照，不是性能 benchmark

## 摘要

这份文档从 entrypoint、操作系统进程和模型对象三个层次解释 vLLM。对使用者，主要入口是离线推理的 Python `LLM` class 与在线服务的 `vllm serve`；对部署者，V1 把 HTTP/input processing、调度与 `KV cache` 管理、GPU model execution 拆进不同进程；对扩展开发者，每个 worker 内部再由 model runner 持有实际的 `torch.nn.Module`，并用统一的 `VllmConfig` 与模型构造接口连接各层。

V1 在线服务的核心拓扑是 `API Server ↔ Engine Core → GPU Workers`。API Server 负责请求接入、tokenization、多模态媒体加载和流式返回，通过 ZMQ 与所有 Engine Cores 建立 many-to-many 连接；每个 data-parallel rank 有一个 Engine Core，运行持续调度的 busy loop，维护 `KV cache` 并派发模型执行；每张 GPU 由一个独立 worker process 管理。启用 data parallelism 时，还会额外出现一个 DP Coordinator，负责 DP ranks 间负载均衡，并为 MoE 模型协调同步 forward pass。

该页面的价值是给出 CPU/process sizing 与职责边界，而不是证明某种拓扑具有多少性能优势。它说明 V1 如何通过多进程隔离关注点，但没有提供 latency、throughput、CPU utilization 或扩展效率数据；页面中的数量公式应视为 2026-08-04 stable 文档所描述的默认部署模型，而不是所有后端、插件和未来版本不变的 ABI。

## 关键事实

- **离线入口**：`vllm.LLM` 是不启动独立 inference server 的主要 Python interface；文档示例通过 `LLM.generate()` 对一组 prompts 执行生成。
- **在线入口**：推荐使用 `vllm serve <model>`。直接运行 `python -m vllm.entrypoints.openai.api_server` 已被文档标为 deprecated，未来可能停止支持。
- **API Server 职责**：处理 HTTP / OpenAI-compatible API、input processing、tokenization、多模态数据加载与 response streaming；它不承担 GPU forward pass。
- **API Server 数量**：无 data parallelism 时默认 `1` 个；启用 DP 后默认自动扩展到 `DP size`，也可用 `--api-server-count` 手工设置为 `A`。
- **API 到 core 的拓扑**：每个 API Server 都通过 ZMQ 连接全部 Engine Cores，形成 many-to-many 路由，因此任一 API Server 可以把请求送往任一 Engine Core。
- **CPU thread 提示**：每个 API Server 会为 media loading 使用多个 CPU threads，数量由 `VLLM_MEDIA_LOADING_THREAD_COUNT` 控制，文档快照中的默认值为 `8`。
- **Engine Core 职责**：运行 scheduler、管理 `KV cache`、协调其所属 GPU workers，并在 busy loop 中持续选择请求和下发工作。
- **Engine Core 数量**：每个 data-parallel rank 一个，即数量为 `DP`；例如 `--data-parallel-size 4` 对应四个 Engine Cores。
- **GPU Worker 职责**：一张 GPU 对应一个 worker process；worker 加载本 rank 的模型权重、执行 forward pass、管理 GPU memory，并只与拥有它的 Engine Core 通信。
- **并行维度与 worker 数量**：每个 Engine Core 下的 worker 数量为 `TP × PP`；全局 GPU worker 数量 `N = DP × PP × TP`。
- **DP Coordinator**：仅当 `DP > 1` 时额外创建一个 coordinator process，用于 DP ranks 间 load balancing，并协调 MoE 模型需要的 synchronized forward passes。
- **进程总数公式**：若 API Server 数量为 `A`、GPU 数量为 `N`，文档给出的 vLLM 进程数为 `A + DP + N + (DP > 1 时的 1 个 coordinator)`。
- **拓扑示例**：单机 `-tp=4` 的四 GPU 服务为 `1 API + 1 Engine Core + 4 workers = 6` 个进程；`-tp=2 -dp=4` 的八 GPU 服务默认是 `4 API + 4 Engine Cores + 8 workers + 1 coordinator = 17` 个进程。
- **worker 内部对象**：每个 worker 有一个 model runner，负责模型加载与运行、输入 tensor 准备和 CUDA graph capture；model runner 再持有一个实际的 `torch.nn.Module` model object。
- **rank 语义**：worker 的 `rank` 用于全局编排，`local_rank` 主要用于 accelerator assignment 与访问本地文件系统、shared memory 等资源。
- **统一配置**：文档把 `VllmConfig` 视为 engine-level global state，各层接收完整配置对象；新增只影响 model runner 的功能时，无需逐层改变 engine / worker / model constructor 参数。
- **统一模型接口**：vLLM 内置 model 使用 keyword-only `__init__(*, vllm_config: VllmConfig, prefix: str = "")`，以统一不同模型与视觉/语言子模型的创建方式；out-of-tree registered model 需要适配这一签名。
- **初始化时 sharding / quantization**：tensor-parallel sharding 与 quantization 在各 layer 初始化时完成，使每个 worker 只创建所需权重 shard，避免先在每张 GPU 完整加载超大模型再变换的峰值内存。

## 争议与不确定点

- 这是 `stable` 官方文档在 2026-08-04 的 living snapshot。进程默认值、CLI、class path、支持的 executor/backend 以及 DP 协调方式都可能在后续版本变化，不能只凭本页推断当前安装版本。
- 总进程数公式描述的是文档列出的 vLLM 主进程，不应机械当作整台机器的完整 process / thread 数；媒体加载 threads、通信库、外部 launcher、sidecar 与插件可能带来额外资源占用。
- API Server 与 Engine Core 的 many-to-many 连接说明“可以路由”，但本页没有量化负载均衡质量、ZMQ overhead、head-of-line blocking 或多 API scaling efficiency。
- `LLMEngine / AsyncLLMEngine` 的 class-level overview 与 V1 process topology 是不同抽象层，不能把每个 class 和每个 OS process 一一对应。在线路径中的具体对象生命周期仍需以对应版本源码为准。
- `VllmConfig` 降低跨层传参改动，但文档也承认完整配置对象会增加组件 unit test 难度；default config helper 是测试便利方案，不代表全局状态没有耦合成本。
- 文档用 405B / 16×H100 举例解释 initialization-time sharding 的内存动机，这是一项结构说明，不是实际加载成功、速度或吞吐 benchmark。

## 关联页面

- 概念：[vLLM](../concepts/vLLM.md)
- 概念：[PagedAttention](../concepts/PagedAttention.md)
- 比较：[SGLang 与 vLLM 架构对比](../comparisons/SGLang%20与%20vLLM%20架构对比.md)
- 论文：[Kwon et al. - 2023 - Efficient Memory Management for Large Language Model Serving with PagedAttention](./Kwon%20et%20al.%20-%202023%20-%20Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention.md)
- 官方文档：[vLLM V1 Guide](./vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md)
- 官方文档：[Automatic Prefix Caching](./vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md)

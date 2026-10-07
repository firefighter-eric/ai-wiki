---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# vLLM Project - 2026 - Architecture Overview

## TL;DR（快速导读）

vLLM 架构文档解释请求怎样经过接口服务、引擎调度和 GPU 执行，适合定位不同层的瓶颈。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

多个请求长度不同，系统要决定哪些先处理、保存多少缓存及如何分配 GPU；模型权重相同也会因服务系统产生不同表现。

## 来源信息

- 类型：官方文档 / 系统架构说明
- 发布者：vLLM Project
- 原始 HTML：[Architecture Overview](../../raw/html/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.html)
- 全文文本：[Architecture Overview](../../raw/text/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md)
- 官方页面：[vLLM Architecture Overview](https://docs.vllm.ai/en/stable/design/arch_overview/)
- 快照日期：2026-08-04
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

接口层处理输入输出，引擎管理调度和缓存，工作进程执行模型。离线入口、在线入口和多设备协调各有职责；具体进程与配置随版本变化，应连同归档快照阅读。

### 方法与背景细节

这份文档从 entrypoint、操作系统进程和模型对象三个层次解释 vLLM。对使用者，主要入口是离线推理的 Python `LLM` class 与在线服务的 `vllm serve`；对部署者，V1 把 HTTP/input processing、调度与 `KV cache` 管理、GPU model execution 拆进不同进程；对扩展开发者，每个 worker 内部再由 model runner 持有实际的 `torch.nn.Module`，并用统一的 `VllmConfig` 与模型构造接口连接各层。

V1 在线服务的核心拓扑是 `API Server ↔ Engine Core → GPU Workers`。API Server 负责请求接入、tokenization、多模态媒体加载和流式返回，通过 ZMQ 与所有 Engine Cores 建立 many-to-many 连接；每个 data-parallel rank 有一个 Engine Core，运行持续调度的 busy loop，维护 `KV cache` 并派发模型执行；每张 GPU 由一个独立 worker process 管理。启用 data parallelism 时，还会额外出现一个 DP Coordinator，负责 DP ranks 间负载均衡，并为 MoE 模型协调同步 forward pass。

该页面的价值是给出 CPU/process sizing 与职责边界，而不是证明某种拓扑具有多少性能优势。它说明 V1 如何通过多进程隔离关注点，但没有提供 latency、throughput、CPU utilization 或扩展效率数据；页面中的数量公式应视为 2026-08-04 stable 文档所描述的默认部署模型，而不是所有后端、插件和未来版本不变的 ABI。

## 关键事实

- **C1**：**离线入口**：`vllm.LLM` 是不启动独立 inference server 的主要 Python interface；文档示例通过 `LLM.generate()` 对一组 prompts 执行生成。
- **C2**：**在线入口**：推荐使用 `vllm serve <model>`。直接运行 `python -m vllm.entrypoints.openai.api_server` 已被文档标为 deprecated，未来可能停止支持。
- **C3**：**API Server 职责**：处理 HTTP / OpenAI-compatible API、input processing、tokenization、多模态数据加载与 response streaming；它不承担 GPU forward pass。
- **C4**：**API Server 数量**：无 data parallelism 时默认 `1` 个；启用 DP 后默认自动扩展到 `DP size`，也可用 `--api-server-count` 手工设置为 `A`。
- **C5**：**API 到 core 的拓扑**：每个 API Server 都通过 ZMQ 连接全部 Engine Cores，形成 many-to-many 路由，因此任一 API Server 可以把请求送往任一 Engine Core。
- **C6**：**CPU thread 提示**：每个 API Server 会为 media loading 使用多个 CPU threads，数量由 `VLLM_MEDIA_LOADING_THREAD_COUNT` 控制，文档快照中的默认值为 `8`。
- **C7**：**Engine Core 职责**：运行 scheduler、管理 `KV cache`、协调其所属 GPU workers，并在 busy loop 中持续选择请求和下发工作。
- **C8**：**Engine Core 数量**：每个 data-parallel rank 一个，即数量为 `DP`；例如 `--data-parallel-size 4` 对应四个 Engine Cores。
- **C9**：**GPU Worker 职责**：一张 GPU 对应一个 worker process；worker 加载本 rank 的模型权重、执行 forward pass、管理 GPU memory，并只与拥有它的 Engine Core 通信。
- **C10**：**并行维度与 worker 数量**：每个 Engine Core 下的 worker 数量为 `TP × PP`；全局 GPU worker 数量 `N = DP × PP × TP`。
- **C11**：**DP Coordinator**：仅当 `DP > 1` 时额外创建一个 coordinator process，用于 DP ranks 间 load balancing，并协调 MoE 模型需要的 synchronized forward passes。
- **C12**：**进程总数公式**：若 API Server 数量为 `A`、GPU 数量为 `N`，文档给出的 vLLM 进程数为 `A + DP + N + (DP > 1 时的 1 个 coordinator)`。
- **C13**：**拓扑示例**：单机 `-tp=4` 的四 GPU 服务为 `1 API + 1 Engine Core + 4 workers = 6` 个进程；`-tp=2 -dp=4` 的八 GPU 服务默认是 `4 API + 4 Engine Cores + 8 workers + 1 coordinator = 17` 个进程。
- **C14**：**worker 内部对象**：每个 worker 有一个 model runner，负责模型加载与运行、输入 tensor 准备和 CUDA graph capture；model runner 再持有一个实际的 `torch.nn.Module` model object。
- **C15**：**rank 语义**：worker 的 `rank` 用于全局编排，`local_rank` 主要用于 accelerator assignment 与访问本地文件系统、shared memory 等资源。
- **C16**：**统一配置**：文档把 `VllmConfig` 视为 engine-level global state，各层接收完整配置对象；新增只影响 model runner 的功能时，无需逐层改变 engine / worker / model constructor 参数。
- **C17**：**统一模型接口**：vLLM 内置 model 使用 keyword-only `__init__(*, vllm_config: VllmConfig, prefix: str = "")`，以统一不同模型与视觉/语言子模型的创建方式；out-of-tree registered model 需要适配这一签名。
- **C18**：**初始化时 sharding / quantization**：tensor-parallel sharding 与 quantization 在各 layer 初始化时完成，使每个 worker 只创建所需权重 shard，避免先在每张 GPU 完整加载超大模型再变换的峰值内存。

## 争议与不确定点

- processcount是本快照拓扑，plugins或外部launcher可增加进程。
- 稳定URL内容会更新，部署时应锁定软件版本并实测资源。

## 关联页面

- 概念：[vLLM](../concepts/vLLM.md)
- 概念：[PagedAttention](../concepts/PagedAttention.md)
- 比较：[SGLang 与 vLLM 架构对比](../comparisons/SGLang%20与%20vLLM%20架构对比.md)
- 论文：[Kwon et al. - 2023 - Efficient Memory Management for Large Language Model Serving with PagedAttention](./Kwon%20et%20al.%20-%202023%20-%20Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention.md)
- 官方文档：[vLLM V1 Guide](./vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md)
- 官方文档：[Automatic Prefix Caching](./vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md)

## 这里的术语是什么意思

- **KV cache**：键值缓存：保存已经处理过的位置表示，生成新内容时可复用，避免全部重算。
- **scheduler**：调度器：决定请求何时进入计算、每次处理多少，以及如何共享资源。
- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。

## 方法与实验解读

API负责输入/输出，enginecore调度和KV，GPUworker执行模型。DP/TP/PP影响进程数、权重分片和通信，CPU媒体线程也参与服务容量。隔离职责的价值在可定位瓶颈；OpenAI-compatible只是API形式，不表示内部用相同推理架构。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md#source-section-3 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C2 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md#source-section-4 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C3 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md#source-section-6 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C4 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md#source-section-6 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C5 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md#source-section-6 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C6 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md#source-section-6 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C7 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md#source-section-7 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C8 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md#source-section-7 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C9 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md#source-section-8 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C10 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md#source-section-8 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C11 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md#source-section-9 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C12 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md#source-section-10 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C13 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md#source-section-10 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C14 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md#source-section-15 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C15 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md#source-section-14 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C16 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md#source-section-17 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C17 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md#source-section-17 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |
| C18 | [原文]( ../../raw/text/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md#source-section-17 ) | 保存论文/文档版本；涉及不同模型、阶段和后端时按原文条件。 |

## 核证范围

保留18项架构细节，核读入口、V1进程拓扑/数量、worker/runner/model与配置/分片。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

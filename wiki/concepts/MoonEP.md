---
type: concept
---
# MoonEP

## 简介

`MoonEP` 是 Kimi K3 3T 级 MoE 预训练中的 expert-parallel execution scheme。它通过动态冗余 experts、在线 placement planning、zero-copy dispatch/combine 与 static computation shapes，使每个 EP rank 接收完全相同数量的 token，而不是只追求平均均衡。

## 关键属性

- 类型：MoE expert parallelism / distributed training system
- 目标：消除 EP ranks 之间的 token-load imbalance 与动态 shape synchronization
- 关键机制：每 rank 有界冗余 experts、GPU online planner、expert migration、zero-copy communication、workload-aware GEMM scheduling
- 与 QB 的关系：QB 改善 expert-level routing 分布；MoonEP 在给定路由结果上保证 rank-level execution balance

## 相关主张

- 传统 EP 中，不同 ranks 收到的 token 数变化会造成 compute imbalance、activation fragmentation 与逐层 host-device synchronization。
- MoonEP 允许复制少量热门 experts，并证明每个 rank 预留不超过 `E/R` 个冗余 expert slots 就总能找到完全平衡方案；online GPU planner 近似离线 ILP 解。
- 完全平衡使每个 rank 的 token volume 和 tensor shapes 静态可知，从而消除每层读取动态 expert counts 的 host synchronization。
- Planner 预先决定每个 token 的目的地址，dispatch 可以直接写入远端 expert-grouped positions，减少中间 permute/copy buffer。
- Rank-level 完全平衡不自动保证 rank 内每个 expert GEMM 同时结束；K3 仍需要 workload-aware schedule 和 shared-expert overlap。
- MoonEP 是训练吞吐与稳定性的系统技术，不是模型 inference API 本身；其收益数字与可移植性目前主要来自 Moonshot 的 K3 系统。

## 来源支持

- [Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence](../../wiki/summaries/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)

## 关联页面

- [Kimi K3](./Kimi%20K3.md)
- [Stable LatentMoE](./Stable%20LatentMoE.md)
- [Quantile Balancing](./Quantile%20Balancing.md)
- [MoE](./MoE.md)
- [LLM 预训练](../topics/LLM%20预训练.md)

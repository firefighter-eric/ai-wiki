---
type: concept
---
# Quantile Balancing

## 简介

`Quantile Balancing (QB)` 是 Kimi K3 为近千 routed experts 设计的 auxiliary-loss-free load-balancing 方法。它不通过额外训练 loss 强迫 router 均衡，而是根据 router-score margins 的分位数，直接估计使每个 expert 接近目标 token load 的 dispatch bias。

## 关键属性

- 类型：MoE router load balancing / auxiliary-loss-free routing
- 调节对象：只加入 top-k selection score 的 expert-specific bias
- 不调节对象：最终 mixture weights 不包含该 bias
- K3 使用场景：896 routed experts、每 token top-16

## 相关主张

- 固定步长 bias update 需要在响应速度和负载震荡之间调参；专家数扩展到近千时，这个敏感超参数会成为稳定性与吞吐问题。
- QB 先做 Top-(k+1) 以取得每个 token 的进入 top-k cutoff，再通过 expert column 的 margin quantile 找到达到目标 load 的下一步 bias。
- 因为 bias 不进入 mixture weight，QB 的直接作用是 dispatch allocation，而不是改变被选 experts 的组合系数或 router gradient。
- 单个 expert 的 quantile target 能改善平均负载，但实际集群吞吐还依赖 expert placement、EP rank balance、冗余 experts 和通信调度；K3 因而另外使用 MoonEP 保证 rank-level 完全平衡。
- QB 在 K3 报告中有推导和内部效果证据，但目前缺少跨模型家族的独立复现，不能直接当作所有 MoE routing 的通用最优方案。

## 来源支持

- [Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence](../../wiki/summaries/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)
- [Kimi - 2026 - Kimi K3 Open Frontier Intelligence Release](../../wiki/summaries/Kimi%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence%20Release.md)

## 关联页面

- [Kimi K3](./Kimi%20K3.md)
- [Stable LatentMoE](./Stable%20LatentMoE.md)
- [MoonEP](./MoonEP.md)
- [MoE](./MoE.md)

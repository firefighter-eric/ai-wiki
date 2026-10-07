---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence

## TL;DR（快速导读）

DeepSeek-V4 报告把长上下文、代理任务和专家模型效率一起设计，重点检查注意力与缓存怎样承担更长输入。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

能接收整份长报告，不代表每个位置的信息都同样容易利用；需要测试实际问题和长文证据定位。

## 来源信息

- 类型：技术报告 / 模型卡发布资料
- 原始文件：../../raw/pdf/DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence.pdf
- 发布页快照：../../raw/html/DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence.html
- 全文文本：../../raw/text/DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence.md
- 作者：DeepSeek AI
- 年份：2026
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

报告介绍 Pro 与 Flash 两条规模路线，并讨论 CSA、HCA 等混合注意力降低长上下文计算和缓存负担。架构、后训练与代理评测需分开阅读；能容纳更长输入，不等于所有远距离信息都能可靠利用。

## 关键事实

- **C1**：预览系列 Pro 为 1.6T/49B 激活，Flash 为 284B/13B 激活，均支持 1M 上下文。
- **C2**：1M 场景下 Pro 的估计单 token FLOPs 为 V3.2 的 27%、KV cache 为 10%。
- **C3**：CSA 压缩 KV 后用 DSA 选 top-k；HCA 更强压缩并保持压缩条目上的 dense attention。
- **C4**：mHC 将残差映射约束到双随机矩阵，并约束输入输出映射以改善深层信号稳定性。
- **C5**：mHC 的实现使用融合、重算和流水线重叠控制额外开销。
- **C6**：Muon 累积 momentum，进行混合 Newton–Schulz 正交化近似，再重标定并执行衰减和更新。
- **C7**：Flash/Pro 分别预训练 32T/33T tokens，后训练包含领域专家 SFT/GRPO 与统一 on-policy distillation。
- **C8**：报告将低精度、定制 kernel 与混合 ZeRO 等列为训练/推理效率的配套条件。

## 争议与不确定点

- 1M 窗口并不保证任何位置的细粒度信息都无损保留；压缩路线需额外测长距离细节。
- 作者基准与内部评测尚未在本库复现，不能把 preview 的作者比较写成永久排名。

## 关联页面

- 主题：[LLM 预训练](../../wiki/topics/LLM%20预训练.md)
- 主题：[LLM RL](../../wiki/topics/LLM%20RL.md)
- 概念：[DeepSeek](../../wiki/concepts/DeepSeek.md)
- 概念：[DeepSeek-V4](../../wiki/concepts/DeepSeek-V4.md)
- 概念：[DeepSeek-V3](../../wiki/concepts/DeepSeek-V3.md)
- 概念：[MoE](../../wiki/concepts/MoE.md)
- 概念：[Compressed Sparse Attention](../../wiki/concepts/Compressed%20Sparse%20Attention.md)
- 概念：[Heavily Compressed Attention](../../wiki/concepts/Heavily%20Compressed%20Attention.md)
- 概念：[Manifold-Constrained Hyper-Connections](../../wiki/concepts/Manifold-Constrained%20Hyper-Connections.md)
- 概念：[Muon](../../wiki/concepts/Muon.md)
- 比较：[开放模型家族与中国重要家族对照](../../wiki/comparisons/开放模型家族与中国重要家族对照.md)
- [DeepSeek](../authors/DeepSeek.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **KV cache**：键值缓存：保存已经处理过的位置表示，生成新内容时可复用，避免全部重算。
- **SFT**：监督微调：用输入与参考输出继续训练已有模型。
- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。
- **sparse**：稀疏计算或连接：只使用选中的部分，具体省略什么取决于方法。
- **embedding**：向量表示：把文字、图片等编码成一组数，用于模型计算或相似度比较。

## 方法与实验解读

V4 将序列压缩、稀疏选择、残差信号约束和矩阵级优化共同用于超长上下文。CSA 重点保留相关片段，HCA 进一步压缩远处信息，局部窗口保留细节；这是一种质量与缓存成本的分配。报告的高推理预算 Max 结果应与普通模式分开，也要把 FLOPs/KV 估算和实际任务延迟、检索准确率分开。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/pdf/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.pdf#page=1 ) | 保存报告的 preview 版本。 |
| C2 | [原文]( ../../raw/pdf/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.pdf#page=5 ) | 等效 FP8 FLOPs 的估算，不是端到端延迟实测。 |
| C3 | [原文]( ../../raw/pdf/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.pdf#page=9 ) | 两者均与局部窗口分支结合，压缩率和候选预算影响信息保留。 |
| C4 | [原文]( ../../raw/pdf/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.pdf#page=8 ) | 数学约束与训练系统开销要分开评估。 |
| C5 | [原文]( ../../raw/pdf/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.pdf#page=21 ) | 报告开销属于其训练实现，不是任意硬件保证。 |
| C6 | [原文]( ../../raw/pdf/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.pdf#page=14 ) | 矩阵级更新依赖完整逻辑权重，不能等同逐元素 AdamW。 |
| C7 | [原文]( ../../raw/pdf/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.pdf#page=5 ) | 预训练、领域训练和模型合并是不同阶段。 |
| C8 | [原文]( ../../raw/pdf/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.pdf#page=4 ) | 结构的理论节省并非脱离系统即可实现。 |

## 核证范围

从完整 PDF 核读第 1、4–14、21 页的架构、效率口径、mHC、Muon 和训练路线；旧 HTML 不含正文，全文已改为 PDF 提取。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

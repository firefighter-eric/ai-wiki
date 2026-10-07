---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Kimi - 2026 - Kimi K3 Open Frontier Intelligence Release

## TL;DR（快速导读）

Kimi K3 发布页提供可用入口与长任务案例，也记录思考历史、主动执行和用户体验方面的限制。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

完成代码任务可能要读取仓库、运行工具和根据结果继续行动；模型分数与整条执行系统的可靠性要分开测。

## 来源信息

- 类型：官方发布博客
- 原始页面：../../raw/html/Kimi - 2026 - Kimi K3 Open Frontier Intelligence Release.html
- 全文文本：../../raw/text/Kimi - 2026 - Kimi K3 Open Frontier Intelligence Release.md
- 来源 URL：https://www.kimi.com/blog/kimi-k3
- 发布方：Kimi / Moonshot AI
- 年份：2026
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

它将模型定位为处理代码、知识工作与推理的多模态系统。发布案例可帮助理解目标任务，限制说明则帮助判断接入风险；架构与训练细节应回到技术报告，价格和渠道只代表归档快照。

## 关键事实

- **C1**：发布时提供Kimi.com/Work/Code/API，并计划后续公开权重和报告。
- **C2**：自部署推荐64或更多accelerators的supernode。
- **C3**：从SFT起使用MXFP4权重/MXFP8激活QAT，配合负载均衡与KDA prefix-cache。
- **C4**：官方指出必须保持完整thinking历史，否则多轮/工具或中途换模型可能不稳定。
- **C5**：限制还包括对小问题/模糊意图可能过度执行，以及整体体验仍落后于所列强闭源模型。

## 争议与不确定点

- 演示不等于平均可靠性，kernel等比较还包含不同来源的评测与fallback条件。
- 公告价格和高缓存命中率不自动适用于新工作负载。

## 关联页面

- 概念：[Kimi K3](../../wiki/concepts/Kimi%20K3.md)
- 概念：[Kimi](../../wiki/concepts/Kimi.md)
- 概念：[Kimi Delta Attention](../../wiki/concepts/Kimi%20Delta%20Attention.md)
- 概念：[MoonEP](../../wiki/concepts/MoonEP.md)
- 来源：[Kimi K3 技术报告](./Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)
- 来源：[Kimi K3 Model Repository](./Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20Model%20Repository.md)
- 来源：[Kimi K3 License](./Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20License.md)
- [Moonshot AI](../authors/Moonshot%20AI.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **KV cache**：键值缓存：保存已经处理过的位置表示，生成新内容时可复用，避免全部重算。

## 方法与实验解读

发布页用长程案例展示能力，并披露运行规模与harness条件。案例由团队选择，可用于理解输出形式，不能当随机任务的成功率。对于知识工作，明确工具边界和回传历史比单看基准更影响实际行为；价格、入口和缓存命中率则应按发布快照理解。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Kimi%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence%20Release.md#source-section-1 ) | 公告中的将来计划与后来已落库报告分开。 |
| C2 | [原文]( ../../raw/text/Kimi%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence%20Release.md#source-section-16 ) | 发行方建议，权重开放不表示单机低成本。 |
| C3 | [原文]( ../../raw/text/Kimi%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence%20Release.md#source-section-16 ) | 算法与系统共同组成部署条件。 |
| C4 | [原文]( ../../raw/text/Kimi%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence%20Release.md#source-section-23 ) | harness兼容性是模型质量的成立条件。 |
| C5 | [原文]( ../../raw/text/Kimi%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence%20Release.md#source-section-23 ) | 作者自评，不能当跨平台永久排名。 |

## 核证范围

核读发布入口、架构/基础设施、案例评测条件与完整Limitations。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

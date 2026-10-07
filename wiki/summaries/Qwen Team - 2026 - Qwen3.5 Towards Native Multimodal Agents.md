---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Qwen Team - 2026 - Qwen3.5 Towards Native Multimodal Agents

## TL;DR（快速导读）

这份 Qwen3.5 官方索引快照将视觉语言与多模态代理作为家族方向，主要提供研究入口。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

能分析图片与能连续调用工具完成任务是两类能力；应分别查看原文与应用测试。

## 来源信息

- 类型：官方研究索引条目 / 发布摘要
- 来源链接：https://qwen.ai/blog?id=qwen3.5
- 全文文本：../../raw/text/Qwen Team - 2026 - Qwen3.5 Towards Native Multimodal Agents.md
- 作者：Qwen Team
- 年份：2026
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开有效正文快照](../../raw/html/verified/qwen3.5-official-blog-2026-10-07.html)
- 核对说明：旧快照是SPA外壳；有效正文来自官方公开article接口的content，原文件保留。

## 摘要

页面反映归档时的官方定位，可据此寻找技术报告和具体模型。索引说明的能力方向比完整训练和评测证据更简略，不能仅凭标题推断所有成员已经具备相同的视觉执行能力。

## 关键事实

- **C1**：首个开放权重型号397B-A17B，混合GatedDeltaNet与稀疏MoE，定位原生视觉语言模型。
- **C2**：语言/方言从119增加到201，文本与视觉在早期融合训练。
- **C3**：Qwen3.5-Plus的API提供1M上下文与官方工具；不能给全部权重型号继承该窗口。
- **C4**：报告异构多模态并行、FP8敏感层BF16与异步RL训推分离。

## 争议与不确定点

- 厂商吞吐倍数限定上下文/模型/软件配置，不能按激活参数直接推算成本。
- 尚未披露完整训练与每项改动独立消融，本页不补猜优化器。

## 关联页面

- 概念：[Qwen3.5](../../wiki/concepts/Qwen3.5.md)
- 概念：[Qwen3.5-Omni](../../wiki/concepts/Qwen3.5-Omni.md)
- 主题：[Qwen 系列](../../wiki/topics/Qwen%20系列.md)

## 这里的术语是什么意思

- **agent**：代理：围绕任务读材料、调用工具和连续执行的系统；名称本身不保证自主性或质量。

## 方法与实验解读

线性与全局注意力混合影响长文运行成本，MoE决定每token激活的容量，多模态融合决定视觉信息何时参与学习。这三种设计不是同一个因果变量。博客评测脚注还说明工具、harness与协议修正，因此结果需带具体设定再比较。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Qwen%20Team%20-%202026%20-%20Qwen3.5%20Towards%20Native%20Multimodal%20Agents.md#source-section-0 ) | 总参数397B/激活17B；该博客不是每个后续型号配置。 |
| C2 | [原文]( ../../raw/text/Qwen%20Team%20-%202026%20-%20Qwen3.5%20Towards%20Native%20Multimodal%20Agents.md#source-section-4 ) | 覆盖数量不代表每语言同等质量。 |
| C3 | [原文]( ../../raw/text/Qwen%20Team%20-%202026%20-%20Qwen3.5%20Towards%20Native%20Multimodal%20Agents.md#source-section-0 ) | 产品服务与权重规格分别看。 |
| C4 | [原文]( ../../raw/text/Qwen%20Team%20-%202026%20-%20Qwen3.5%20Towards%20Native%20Multimodal%20Agents.md#source-section-5 ) | 多个系统改动共同作用。 |

## 核证范围

核读恢复的官方正文介绍、预训练、基础设施、API范围及评测脚注。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

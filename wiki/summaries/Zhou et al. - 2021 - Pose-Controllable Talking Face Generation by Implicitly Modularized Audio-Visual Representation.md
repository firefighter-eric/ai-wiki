---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Zhou et al. - 2021 - Pose-Controllable Talking Face Generation by Implicitly Modularized Audio-Visual Representation

## TL;DR（快速导读）

这篇说话人生成方法把音频驱动的口型与头部姿态控制分开考虑，研究保持同步时怎样控制动作。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

嘴部同步不能决定全部头部运动。论文通过模块化的音视频表示处理姿态控制，并关注极端条件下的问题。需要分别观察身份、口型、姿态与画面稳定性。

## 具体怎么理解

同一段语音可以配上不同朝向的头部动作；控制姿态不应让嘴部节奏随之错位。

## 关键事实

- **C1**：PC-AVS 将身份、音频和姿态表示分开，使用其他视频提供姿态控制。
- **C2**：同步指标与画面质量分别衡量，作者也提醒高于真值的 SyncNet 分数不代表视觉全面更好。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Zhou%20et%20al.%20-%202021%20-%20Pose-Controllable%20Talking%20Face%20Generation%20by%20Implicitly%20Modularized%20Audio-Visual%20Representation.pdf)
- 全文文本：[打开全文文本](../../raw/text/Zhou%20et%20al.%20-%202021%20-%20Pose-Controllable%20Talking%20Face%20Generation%20by%20Implicitly%20Modularized%20Audio-Visual%20Representation.md)
- 作者：Zhou et al.
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Zhou%20et%20al.%20-%202021%20-%20Pose-Controllable%20Talking%20Face%20Generation%20by%20Implicitly%20Modularized%20Audio-Visual%20Representation.html)

## 争议与不确定点

- Celeb 视频和正脸基准不能覆盖全部身份、姿态与遮挡。
- 姿态可控不证明所有动作自然。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无

## 方法与实验解读

PC-AVS 在让静态照片说话的同时提供姿态驱动。模块化控制便于组合，但各表示未必完全解耦；实际输出仍需检查身份保持、嘴型、头动与背景稳定。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Zhou%20et%20al.%20-%202021%20-%20Pose-Controllable%20Talking%20Face%20Generation%20by%20Implicitly%20Modularized%20Audio-Visual%20Representation.md#source-section-7 ) | 控制输入存在，不是只从音频恢复唯一真实姿态 |
| C2 | [原文]( ../../raw/text/Zhou%20et%20al.%20-%202021%20-%20Pose-Controllable%20Talking%20Face%20Generation%20by%20Implicitly%20Modularized%20Audio-Visual%20Representation.md#source-section-11 ) | 自动同步指标存在解释边界 |

## 核证范围

核对 §3.2 的表示模块、§4.2 的指标讨论和结论的驱动方式。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

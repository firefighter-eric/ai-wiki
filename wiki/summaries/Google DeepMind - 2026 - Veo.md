---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Google DeepMind - 2026 - Veo

## TL;DR（快速导读）

Veo 官方页面介绍带音频的视频生成及参考控制；它展示产品能力，具体质量仍要通过对应任务核对。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：官方模型页 / 产品说明
- 原始文件：../../raw/html/Google DeepMind - 2026 - Veo.html
- 全文文本：../../raw/text/Google DeepMind - 2026 - Veo.md
- 来源链接：https://deepmind.google/models/veo/
- 作者：Google DeepMind
- 年份：2026
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

归档页面重点讨论画面、指令遵循、物理表现、声音与产品入口。页面中的版本和比较是快照信息，宣传样例也不是完整评测；实际创作应分别检查镜头、人物、声音和持续状态。

## 关键事实

- **C1**：保存网页将 Veo3.1 列为最新版本，强调音视频生成及创作控制。
- **C2**：网页提供 Gemini、Flow 和开发接口三类入口。
- **C3**：音视频整体偏好测试使用527个 MovieGenBench prompts。
- **C4**：参考图/扩展/插入等内部比较含364例、720p；Veo8秒、对照10秒，部分指标关闭音频。

## 争议与不确定点

- 内部人评依赖 prompts、对照版本、时长与音频开关。
- state-of-the-art 属于该快照与比较范围，本页不生成当前竞品排名。

## 关联页面

- 概念：[Veo 3.1](../../wiki/concepts/Veo%203.1.md)
- 主题：[视频生成](../../wiki/topics/视频生成.md)
- 作者：[DeepMind](../../wiki/authors/DeepMind.md)

## 方法与实验解读

Veo 网页同时介绍创作接口和人评。参考图、首尾帧、扩展与对象插入分别控制生成的不同阶段，不能只凭整体偏好判定每种控制都可靠。音画同步、可视物理真实感和任务遵循需要各自指标；演示视频是示例，不代表错误率。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Google%20DeepMind%20-%202026%20-%20Veo.md#source-section-2 ) | 该快照信息，不当作当前永恒版本。 |
| C2 | [原文]( ../../raw/text/Google%20DeepMind%20-%202026%20-%20Veo.md#source-section-1 ) | 入口与具体能力/额度需要分别确认。 |
| C3 | [原文]( ../../raw/text/Google%20DeepMind%20-%202026%20-%20Veo.md#source-section-45 ) | 人类偏好指标，不是物理规律正确率。 |
| C4 | [原文]( ../../raw/text/Google%20DeepMind%20-%202026%20-%20Veo.md#source-section-48 ) | 时长和音频条件不相同，解释时保留这些差异。 |

## 核证范围

核读版本入口、控制能力、偏好指标与评测脚注。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

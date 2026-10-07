---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Alibaba Cloud - 2025 - Alibaba Unveils Wan2.6 Series Enabling Everyone to Star in Videos

## TL;DR（快速导读）

Wan2.6 的发布稿介绍参考视频、文字和图片驱动的视频生成；重点是把同一角色带入新的场景与镜头。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

两个镜头讲同一段情节时，人物状态与空间关系应衔接；增加镜头数量也增加一致性检查。

## 来源信息

- 类型：官方新闻稿 / 产品发布
- 原始文件：../../raw/html/Alibaba Cloud - 2025 - Alibaba Unveils Wan2.6 Series Enabling Everyone to Star in Videos.html
- 全文文本：../../raw/text/Alibaba Cloud - 2025 - Alibaba Unveils Wan2.6 Series Enabling Everyone to Star in Videos.md
- 来源链接：https://www.alibabacloud.com/press-room/alibaba-unveils-wan2-6-series-enabling-everyone
- 作者：Alibaba Cloud / Alibaba Group
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

其中参考视频路线允许用角色外观和声音作为输入，再以文字描述新内容。阅读这份产品资料时，可关注角色一致性、多镜头控制及素材要求；发布稿展示的功能与完整方法、独立效果验证应分开看。

## 关键事实

- **C1**：2025-12-16 发布 Wan2.6 系列，新增 R2V，并更新 T2V、I2V、image 与 T2I。
- **C2**：R2V 接收含外观与声音的角色参考视频，再按文本提示生成新场景。
- **C3**：公告介绍多镜头叙事、多主体对话与音画同步，并称视频输出最长 15 秒。
- **C4**：图片路线支持图文交错输出和图像编辑，公告称支持长中英文提示。
- **C5**：发布时入口是 Model Studio 与 wan.video；Qwen App 集成是公告中的后续计划。

## 争议与不确定点

- 中国首个是发行方表述，缺少独立竞品和发布日期审计。
- 参考人物外观/声音一致、电影级质量等应通过具体工作流测试确认，公告没有给出误差分布。

## 关联页面

- 概念：[Wan2.6](../../wiki/concepts/Wan2.6.md)
- 主题：[视频生成](../../wiki/topics/视频生成.md)
- 作者：[Alibaba Group](../../wiki/authors/Alibaba%20Group.md)

## 方法与实验解读

这是一份产品公告，适合核对版本、接口种类和创作工作流。它没有披露训练数据、网络结构、消融或统一评测，因而可支撑产品功能导航，不能单独支撑关于某种算法优越性的研究判断。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Alibaba%20Cloud%20-%202025%20-%20Alibaba%20Unveils%20Wan2.6%20Series%20Enabling%20Everyone%20to%20Star%20in%20Videos.md#source-section-1 ) | 官方发布公告。 |
| C2 | [原文]( ../../raw/text/Alibaba%20Cloud%20-%202025%20-%20Alibaba%20Unveils%20Wan2.6%20Series%20Enabling%20Everyone%20to%20Star%20in%20Videos.md#source-section-1 ) | 公告描述的产品输入输出，未提供独立身份或声音一致性评测。 |
| C3 | [原文]( ../../raw/text/Alibaba%20Cloud%20-%202025%20-%20Alibaba%20Unveils%20Wan2.6%20Series%20Enabling%20Everyone%20to%20Star%20in%20Videos.md#source-section-1 ) | 系列能力概述；不同端点的实际限制须查各自 API 文档。 |
| C4 | [原文]( ../../raw/text/Alibaba%20Cloud%20-%202025%20-%20Alibaba%20Unveils%20Wan2.6%20Series%20Enabling%20Everyone%20to%20Star%20in%20Videos.md#source-section-1 ) | 不将宣传性质量描述换算为实验分数。 |
| C5 | [原文]( ../../raw/text/Alibaba%20Cloud%20-%202025%20-%20Alibaba%20Unveils%20Wan2.6%20Series%20Enabling%20Everyone%20to%20Star%20in%20Videos.md#source-section-1 ) | 历史发布信息，不保证当前产品入口。 |

## 核证范围

核读公告正文、发布日期、产品系列组成与发布入口，未将网页页脚作为证据。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

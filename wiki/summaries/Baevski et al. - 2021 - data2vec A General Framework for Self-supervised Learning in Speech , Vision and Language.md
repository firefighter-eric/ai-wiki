---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# data2vec：跨语音、视觉与语言的自监督框架（2022）

## TL;DR（快速导读）

data2vec 用同一种自监督思路处理语音、图像和文本：遮住输入的一部分，预测教师模型看到完整输入时产生的表示。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

不同模态通常使用各自的预训练目标。data2vec 尝试统一这个接口，让学生从局部可见内容预测教师给出的上下文表示。这里的统一是学习方式的统一，不表示三种输入必然由一个未经适配的模型同时处理。

## 具体怎么理解

例如学生只看到一张图的一部分，却要预测教师看完整图后形成的表示；语音和文本采用相应的遮挡任务。

## 关键事实

- **C1**：student 从被遮挡输入预测 teacher 对完整输入产生的表示；teacher 使用模型权重的指数滑动平均。
- **C2**：学习范式跨语音、视觉和语言复用，但输入编码器和遮挡策略仍按模态设计。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Baevski%20et%20al.%20-%202021%20-%20data2vec%20A%20General%20Framework%20for%20Self-supervised%20Learning%20in%20Speech%20%2C%20Vision%20and%20Language.pdf)
- 全文文本：[打开全文文本](../../raw/text/Baevski%20et%20al.%20-%202021%20-%20data2vec%20A%20General%20Framework%20for%20Self-supervised%20Learning%20in%20Speech%20%2C%20Vision%20and%20Language.md)
- 作者：Alexei Baevski 等
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Baevski%20et%20al.%20-%202021%20-%20data2vec%20A%20General%20Framework%20for%20Self-supervised%20Learning%20in%20Speech%20%2C%20Vision%20and%20Language.html)
- 归档说明：保留历史文件名以维持来源对应和链接；标题、作者与年份以上述核对信息为准。

## 争议与不确定点

- 不同模态的结果来自各自训练与任务设置。
- 跨模态检索或联合音视频学习属于后续方向，未由此处结果直接证明。

## 关联页面

- 主题：[Slide  理解与生成](../topics/Slide%20理解与生成.md)
- 综合：暂无

## 方法与实验解读

data2vec 将预测目标从原始数据或离散标签改为完整输入的模型表示。语音、图像和文本可使用同一种 student–teacher 思路，却仍有各自编码与评测，不能把这篇工作读成已经完成跨模态对齐。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Baevski%20et%20al.%20-%202021%20-%20data2vec%20A%20General%20Framework%20for%20Self-supervised%20Learning%20in%20Speech%20%2C%20Vision%20and%20Language.md#source-section-9 ) | 目标是上下文化表示，不是重建原始像素或 token 本身 |
| C2 | [原文]( ../../raw/text/Baevski%20et%20al.%20-%202021%20-%20data2vec%20A%20General%20Framework%20for%20Self-supervised%20Learning%20in%20Speech%20%2C%20Vision%20and%20Language.md#source-section-32 ) | 统一算法不等于一个模型联合掌握所有模态 |

## 核证范围

核对 §3 方法、§4 模型设置和 §7 的模态边界。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Wang et al. - 2022 - OFA Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework

## TL;DR（快速导读）

OFA 把多种视觉与语言任务写成统一序列到序列接口，通过指令与输出序列表达不同任务。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

图像描述、视觉定位和语言任务通常各用一套接口。OFA 用共享框架处理多模态输入和生成输出。统一形式有助于复用，但任务本身的监督、指标与错误仍有区别。

## 具体怎么理解

同一框架可以按指令生成图像描述，也可以输出某个对象的位置；读者应先确认要求的输出是什么。

## 关键事实

- **C1**：用 encoder–decoder 和统一序列目标组织任务；输入图像仍使用视觉特征提取，目标图像通过离散编码进入词表。
- **C2**：图文、纯文本和纯图像任务一起预训练，但具体任务的收益并不一致。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Wang%20et%20al.%20-%202022%20-%20OFA%20Unifying%20Architectures%2C%20Tasks%2C%20and%20Modalities%20Through%20a%20Simple%20Sequence-to-Sequence%20Learning%20Framework.pdf)
- 全文文本：[打开全文文本](../../raw/text/Wang%20et%20al.%20-%202022%20-%20OFA%20Unifying%20Architectures%2C%20Tasks%2C%20and%20Modalities%20Through%20a%20Simple%20Sequence-to-Sequence%20Learning%20Framework.md)
- 作者：Wang et al.
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Wang%20et%20al.%20-%202022%20-%20OFA%20Unifying%20Architectures%2C%20Tasks%2C%20and%20Modalities%20Through%20a%20Simple%20Sequence-to-Sequence%20Learning%20Framework.html)

## 争议与不确定点

- 微调结果与零样本迁移需要分别引用，不能用微调排行榜证明零样本能力。
- 不同预训练任务可能产生负迁移；统一架构并不保证所有任务同时受益。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无
- [Jingren Zhou](../authors/Jingren%20Zhou.md)：沿作者或机构继续阅读相关来源。
- [Shuai Bai](../authors/Shuai%20Bai.md)：沿作者或机构继续阅读相关来源。
- [Junyang Lin](../authors/Junyang%20Lin.md)：沿作者或机构继续阅读相关来源。

## 方法与实验解读

OFA 将问答、定位、分类和生成写成序列任务，模型共享架构与输出词表。它统一的是建模接口：文本分词、图像特征与离散图像码仍各有转换步骤。多任务预训练是否有益，需要看目标任务的消融而不是只看整体名次。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Wang%20et%20al.%20-%202022%20-%20OFA%20Unifying%20Architectures%2C%20Tasks%2C%20and%20Modalities%20Through%20a%20Simple%20Sequence-to-Sequence%20Learning%20Framework.md#source-section-10 ) | 统一任务接口不等于取消模态预处理 |
| C2 | [原文]( ../../raw/text/Wang%20et%20al.%20-%202022%20-%20OFA%20Unifying%20Architectures%2C%20Tasks%2C%20and%20Modalities%20Through%20a%20Simple%20Sequence-to-Sequence%20Learning%20Framework.md#source-section-20 ) | 消融中部分任务收益伴随其他任务退步 |

## 核证范围

核对 §3.1–3.4 的表示与任务设计、§4.3 的迁移设定及 §4.4 的消融。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Liang et al. - 2021 - R-Drop Regularized Dropout for Neural Networks arXiv 2106 . 14448v2 cs . LG 29 Oct 2021

## TL;DR（快速导读）

R-Drop 让同一输入经过两次不同 dropout 后，输出分布仍保持一致，用额外一致性约束改善训练。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

Dropout 会随机改变参与计算的子网络。论文通过约束同一输入的两次预测差异，减少这种随机性带来的不一致。需要评估额外训练计算和任务收益，不能把它理解成取消 dropout。

## 具体怎么理解

同一句话做两次前向计算，随机遮掉的神经元不同；训练希望它们仍对类别或下一个词给出相近预测。

## 关键事实

- **C1**：同一输入产生两个 dropout 视图，并最小化两者输出分布的双向 KL。
- **C2**：实现可在 batch 内重复输入来获得两次随机视图，仍增加训练计算。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Liang%20et%20al.%20-%202021%20-%20R-Drop%20Regularized%20Dropout%20for%20Neural%20Networks%20arXiv%202106%20.%2014448v2%20cs%20.%20LG%2029%20Oct%202021.pdf)
- 全文文本：[打开全文文本](../../raw/text/Liang%20et%20al.%20-%202021%20-%20R-Drop%20Regularized%20Dropout%20for%20Neural%20Networks%20arXiv%202106%20.%2014448v2%20cs%20.%20LG%2029%20Oct%202021.md)
- 作者：Liang et al.
- 年份：2021
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Liang%20et%20al.%20-%202021%20-%20R-Drop%20Regularized%20Dropout%20for%20Neural%20Networks%20arXiv%202106%20.%2014448v2%20cs%20.%20LG%2029%20Oct%202021.html)

## 争议与不确定点

- KL 权重与任务会影响正则效果。
- 损失曲线按步数更好可能仍对应更多计算。

## 关联页面

- 主题：[传统CV](../topics/传统%20CV.md)
- 综合：暂无

## 方法与实验解读

R-Drop 要求模型在 dropout 随机性下给出一致预测，降低训练与推理的不一致。它不改变任务目标本身，也不保证预测正确，因此应同时比较验证成绩和墙钟成本。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Liang%20et%20al.%20-%202021%20-%20R-Drop%20Regularized%20Dropout%20for%20Neural%20Networks%20arXiv%202106%20.%2014448v2%20cs%20.%20LG%2029%20Oct%202021.md#source-section-23 ) | 正则化作用在预测分布而非要求隐藏向量相同 |
| C2 | [原文]( ../../raw/text/Liang%20et%20al.%20-%202021%20-%20R-Drop%20Regularized%20Dropout%20for%20Neural%20Networks%20arXiv%202106%20.%2014448v2%20cs%20.%20LG%2029%20Oct%202021.md#source-section-6 ) | 单次批处理不等于没有额外成本 |

## 核证范围

核对 §2.1–2.2 的目标与批实现、§4.1 的成本分析及结论。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

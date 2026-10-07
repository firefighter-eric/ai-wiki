---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# He et al. - 2015 - Deep Residual Learning for Image Recognition

## TL;DR（快速导读）

ResNet 让网络层学习对输入的增量修正，用跳跃连接改善很深网络的训练。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/He et al. - 2015 - Deep Residual Learning for Image Recognition.pdf
- 原始 HTML：../../raw/html/He et al. - 2015 - Deep Residual Learning for Image Recognition.html
- 全文文本：../../raw/text/He et al. - 2015 - Deep Residual Learning for Image Recognition.md
- 作者：He et al.
- 年份：2015
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

残差块输出可以理解为原输入加上学到的变化，提供更直接的信息与梯度路径。论文围绕深度增加带来的优化问题验证这一设计；比较网络时仍需控制训练配方、结构和资源。

## 关键事实

- **C1**：degradation 指更深 plain 网络训练误差变差，不能仅归因过拟合。
- **C2**：残差块学习F(x)=H(x)-x，以F(x)+x输出；维度变化时可用投影shortcut。
- **C3**：报告ImageNet最高152层，并以ensemble取得3.57% top-5 test error。
- **C4**：论文还把残差特征用于检测等视觉任务。

## 争议与不确定点

- 尺寸变化、归一化和训练配方参与效果，shortcut不是单独保证收敛的定理。
- 竞赛ensemble结果不能当作每个ResNet部署实例的准确率。

## 关联页面

- 主题：[经典 CNN 架构](../../wiki/topics/经典%20CNN%20架构.md)
- 主题：[传统 CV](../../wiki/topics/传统%20CV.md)
- 概念：[ResNet](../../wiki/concepts/ResNet.md)
- 概念：[ResNeXt](../../wiki/concepts/ResNeXt.md)
- 概念：[ConvNeXt](../../wiki/concepts/ConvNeXt.md)

## 这里的术语是什么意思

- **backbone**：模型骨干：主要负责提取或变换表示，其他任务模块在它之上工作。

## 方法与实验解读

如果理想映射接近恒等，残差参数化让网络只需学偏离部分，而不必重新建立整个传递。作者通过 plain/residual 对照显示优化差异，而不是主张深度永远越大越好。后续架构继承残差接口是知识库的谱系组织判断，不能反过来作为原论文事实。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/He%20et%20al.%20-%202015%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition.md#source-section-3 ) | 训练与测试误差分别看。 |
| C2 | [原文]( ../../raw/text/He%20et%20al.%20-%202015%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition.md#source-section-6 ) | 同维度 identity shortcut 与跨维投影有不同成本。 |
| C3 | [原文]( ../../raw/text/He%20et%20al.%20-%202015%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition.md#source-section-2 ) | ensemble、single model和不同crop设置不能混合。 |
| C4 | [原文]( ../../raw/text/He%20et%20al.%20-%202015%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition.md#source-section-2 ) | 分类训练优化证据与下游任务迁移分别验证。 |

## 核证范围

核读 degradation、§3残差定义/结构与ImageNet和下游结果口径。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

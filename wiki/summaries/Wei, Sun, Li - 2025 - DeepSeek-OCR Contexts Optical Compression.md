---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Wei, Sun, Li - 2025 - DeepSeek-OCR Contexts Optical Compression

## TL;DR（快速导读）

DeepSeek-OCR 将文档图像编码为较少视觉词元，研究视觉压缩能否节省长文本上下文。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

输入一整页扫描文档，先以视觉表示压缩，再输出文本；图表、顺序和公式应逐项核对。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Wei, Sun, Li - 2025 - DeepSeek-OCR Contexts Optical Compression.pdf
- 全文文本：../../raw/text/Wei, Sun, Li - 2025 - DeepSeek-OCR Contexts Optical Compression.md
- 作者：Wei, Sun, Li
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

方法用视觉编码器和语言解码器恢复文档内容，OCR 是检验压缩保真度的任务。节省词元并不自动等于读取可靠；文字遗漏、公式、布局和压缩条件都需要分别核对。

## 关键事实

- **C1**：DeepEncoder低视觉token配MoE decoder，问题是光学文本压缩。
- **C2**：压缩测试为Fox英文100页、600–1300texttokens，Tiny64/Small100visiontokens。
- **C3**：约20×压缩仍只保留约60%准确率，是有损边界。
- **C4**：没有SFTchat阶段，一些视觉能力用completionprompt激活。

## 争议与不确定点

- 论文长期记忆/无限context只是未来方向。
- OCR accuracy、顺序、表格结构和下游问答要分别评测。

## 关联页面

- 概念：[DeepSeek-OCR](../../wiki/concepts/DeepSeek-OCR.md)
- 来源后续：[Wei, Sun, Li - 2026 - DeepSeek-OCR 2 Visual Causal Flow](./Wei,%20Sun,%20Li%20-%202026%20-%20DeepSeek-OCR%202%20Visual%20Causal%20Flow.md)
- 概念：[DeepSeek](../../wiki/concepts/DeepSeek.md)
- 主题：[OCR](../../wiki/topics/OCR.md)
- [DeepSeek](../authors/DeepSeek.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **benchmark**：评测基准：特定数据、任务与规则的组合，分数只在这些条件下成立。
- **MoE**：专家混合：路由器为不同输入选择部分子网络，总容量与每次实际计算不同。
- **OCR**：文字识别：从图像读取文字；整页任务还需处理布局和阅读顺序。

## 方法与实验解读

把文本画成二维图像以减少LLM输入token，但可恢复字数不等于回答保真；小字/稀有字符/多栏和历史细节还会丢失。生产页/天包含并行与数据生成配置，不能由它推出单请求延迟或知识库零误差。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Wei%2C%20Sun%2C%20Li%20-%202025%20-%20DeepSeek-OCR%20Contexts%20Optical%20Compression.md#source-section-8 ) | 不是已验证无限记忆方案。 |
| C2 | [原文]( ../../raw/text/Wei%2C%20Sun%2C%20Li%20-%202025%20-%20DeepSeek-OCR%20Contexts%20Optical%20Compression.md#source-section-22 ) | 10×约97%声明限定此测试。 |
| C3 | [原文]( ../../raw/text/Wei%2C%20Sun%2C%20Li%20-%202025%20-%20DeepSeek-OCR%20Contexts%20Optical%20Compression.md#source-section-28 ) | 不能宣称任意文本近无损。 |
| C4 | [原文]( ../../raw/text/Wei%2C%20Sun%2C%20Li%20-%202025%20-%20DeepSeek-OCR%20Contexts%20Optical%20Compression.md#source-section-27 ) | OCR输出不等同通用聊天。 |

## 核证范围

核读架构/数据、Fox压缩设定、OCR解析与discussion/非chat限制。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

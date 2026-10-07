---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Duan et al. - 2026 - GLM-OCR Technical Report

## TL;DR（快速导读）

GLM-OCR 先分析页面区域，再识别文字、公式和表格，研究紧凑模型与完整文档处理流程的配合。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

识别票据正文、恢复表格和读取公式需要不同检查项；平均识别分数可能掩盖某类错误。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Duan et al. - 2026 - GLM-OCR Technical Report.pdf
- 全文文本：../../raw/text/Duan et al. - 2026 - GLM-OCR Technical Report.md
- 作者：Duan et al.
- 年份：2026
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

报告描述较小规模模型、多词元预测及版面检测加区域识别的两阶段设计。阅读时应区分文本识别、结构恢复、关键信息抽取和部署速度；一个识别分数不能代表整页文档理解的全部质量。

## 关键事实

- **C1**：GLM-OCR 总参数约 0.9B，由 0.4B CogViT、连接器和 0.5B GLM decoder 构成。
- **C2**：文档解析先用 PP-DocLayout-V3 分区，再并行识别各区域并恢复阅读顺序。
- **C3**：MTP 使用共享参数预测多个未来 token；报告平均每步约 5.2 tokens，吞吐提升约 50%。
- **C4**：训练经历视觉训练、VLM 预训练、带 MTP 的 SFT 和 RL。
- **C5**：OmniDocBench v1.5 overall 约 94.6；闭源参考模型在表中不参与 best-score 排名。

## 争议与不确定点

- 分区错误会向后传递，复杂版式和长结构输出需要分别评测。
- 部署支持是接口可用性信息，实际速度须在目标文档和硬件上测试。

## 关联页面

- 概念：[GLM-OCR](../../wiki/concepts/GLM-OCR.md)
- 概念：[GLM](../../wiki/concepts/GLM.md)
- 概念：[PaddleOCR](../../wiki/concepts/PaddleOCR.md)
- 主题：[OCR](../../wiki/topics/OCR.md)

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **benchmark**：评测基准：特定数据、任务与规则的组合，分数只在这些条件下成立。
- **encoder**：编码器：把输入转成模型内部表示。
- **decoder**：解码器：根据已有表示产生文字、图像或其他输出。
- **OCR**：文字识别：从图像读取文字；整页任务还需处理布局和阅读顺序。

## 方法与实验解读

核心 VLM 处理区域的文字和结构，layout 层决定分区与顺序，MTP 改善长结构输出的解码效率。三者需要分别检查：识别准确不代表页面分区正确，吞吐改善也不能掩盖合并顺序或表格结构错误。紧凑模型与 PaddleOCR 组件可以互补，选型应计算整个 pipeline 的延迟和显存。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Duan%20et%20al.%20-%202026%20-%20GLM-OCR%20Technical%20Report.md#source-section-6 ) | 核心模型规模，不含外部 layout 模型的全部系统成本。 |
| C2 | [原文]( ../../raw/text/Duan%20et%20al.%20-%202026%20-%20GLM-OCR%20Technical%20Report.md#source-section-6 ) | 系统为分阶段 pipeline，不是单网络直接解决所有任务。 |
| C3 | [原文]( ../../raw/text/Duan%20et%20al.%20-%202026%20-%20GLM-OCR%20Technical%20Report.md#source-section-2 ) | 作者测试条件，不能视为所有文档和部署端点的常数。 |
| C4 | [原文]( ../../raw/text/Duan%20et%20al.%20-%202026%20-%20GLM-OCR%20Technical%20Report.md#source-section-7 ) | 训练阶段承担不同职责。 |
| C5 | [原文]( ../../raw/text/Duan%20et%20al.%20-%202026%20-%20GLM-OCR%20Technical%20Report.md#source-section-14 ) | 指标和候选集合固定；不扩展为所有 OCR 任务第一。 |

## 核证范围

核读架构、训练配方、MTP 声明、公共评测及部署/局限章节。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

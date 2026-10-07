---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# PaddlePaddle Team et al. - 2025 - PaddleOCR 3.0 Technical Report

## TL;DR（快速导读）

PaddleOCR 3.0 将文字识别、文档结构恢复和信息抽取组织成工具链，适合按处理环节选择组件。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

处理一张票据可能先检测文字，再识别、排序和抽取字段；应分别定位是哪一步出错。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/PaddlePaddle Team et al. - 2025 - PaddleOCR 3.0 Technical Report.pdf
- 全文文本：../../raw/text/PaddlePaddle Team et al. - 2025 - PaddleOCR 3.0 Technical Report.md
- 作者：PaddlePaddle Team et al.
- 年份：2025
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

PP-OCRv5 负责识别，PP-StructureV3 处理文档结构，PP-ChatOCRv4 将结果用于信息抽取等任务。它不是一个统一分数能概括的模型；应分别验收每个环节与端到端结果，关注误差如何传递。

## 关键事实

- **C1**：PaddleOCR3.0是工具包，由PP-OCRv5、PP-StructureV3和PP-ChatOCRv4承担不同职责。
- **C2**：PP-OCRv5包含预处理、检测、方向分类与识别，支持简繁中文/拼音/英/日；报告17类场景评测。
- **C3**：PP-StructureV3将多个模型组成pipeline，输出JSON/Markdown并处理阅读顺序。
- **C4**：PP-ChatOCRv4结合解析、向量检索、LLM与VLM进行信息抽取和问答。
- **C5**：系统提供训练/推理工具、服务/端侧部署和MCP server。

## 争议与不确定点

- 模型0.07B比较限定OCR文本子任务，不能据此断言胜过VLM所有文档能力。
- 表格合并单元、复杂公式和排序仍可能出错，输出需按任务核验。

## 关联页面

- 概念：[PaddleOCR](../../wiki/concepts/PaddleOCR.md)
- 概念：[TrOCR](../../wiki/concepts/TrOCR.md)
- 概念：[LayoutLMv3](../../wiki/concepts/LayoutLMv3.md)
- 概念：[DocLLM](../../wiki/concepts/DocLLM.md)
- 主题：[OCR](../../wiki/topics/OCR.md)
- 主题：[传统 CV](../../wiki/topics/传统%20CV.md)

## 这里的术语是什么意思

- **RAG**：检索增强生成：先找外部材料，再利用这些材料生成回答。
- **OCR**：文字识别：从图像读取文字；整页任务还需处理布局和阅读顺序。
- **agent**：代理：围绕任务读材料、调用工具和连续执行的系统；名称本身不保证自主性或质量。

## 方法与实验解读

专用OCR可以用较小模型解决文本识别；全页面解析还需要布局、顺序和结构协调。报告将pipeline工具、专家VLM和通用VLM放在同一基准，但每个子任务仍要分开看，尤其1-edit和edit方向相反。部署时保留每个模块的错误定位，而不是只保留一份成功Markdown。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/PaddlePaddle%20Team%20et%20al.%20-%202025%20-%20PaddleOCR%203.0%20Technical%20Report.md#source-section-2 ) | OCR/解析/问答不可用单一分数替代。 |
| C2 | [原文]( ../../raw/text/PaddlePaddle%20Team%20et%20al.%20-%202025%20-%20PaddleOCR%203.0%20Technical%20Report.md#source-section-5 ) | 默认server版本，与mobile分别看。 |
| C3 | [原文]( ../../raw/text/PaddlePaddle%20Team%20et%20al.%20-%202025%20-%20PaddleOCR%203.0%20Technical%20Report.md#source-section-6 ) | 版面、表格与公式是额外结构任务。 |
| C4 | [原文]( ../../raw/text/PaddlePaddle%20Team%20et%20al.%20-%202025%20-%20PaddleOCR%203.0%20Technical%20Report.md#source-section-7 ) | 回答可靠性还受检索与生成影响。 |
| C5 | [原文]( ../../raw/text/PaddlePaddle%20Team%20et%20al.%20-%202025%20-%20PaddleOCR%203.0%20Technical%20Report.md#source-section-11 ) | 工程接口不等同所有硬件都相同性能。 |

## 核证范围

核读组件设计、17场景评测、解析指标、ChatOCR与部署；保留工具包和单模型区别。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

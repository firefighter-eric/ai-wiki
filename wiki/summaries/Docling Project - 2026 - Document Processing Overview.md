---
type: summary
status: refined
evidence_schema: 1
reviewed: 2026-10-07
---
# Docling Project - 2026 - Document Processing Overview

## TL;DR（快速导读）

Docling 把文档解析为结构化表示；检查论文时，要分别验收文字、阅读顺序、表格与公式。

## 先看一个例子

选一页双栏加公式表格的文档，分别检查文字、顺序与结构，再决定是否替换现有抽取。

## 来源信息

- 类型：官方软件文档；2026-10-07 快照，年份表示核验时间。
- 官方来源：[文档](https://docling-project.github.io/docling/)
- 原始 HTML：[快照](../../raw/html/Docling%20Project%20-%202026%20-%20Document%20Processing%20Overview.html)
- 全文文本：[正文](../../raw/text/Docling%20Project%20-%202026%20-%20Document%20Processing%20Overview.md)
- 状态：精修 summary，已核对功能声明及用途。

## 摘要

官方概览列出 PDF 版面分析、OCR、表格与公式处理，以及 Markdown、JSON 输出。功能存在不能保证每份文件都正确解析；可用双栏、扫描页和公式表格样本检查，再决定是否加入本库流程。

## 关键事实

- C1：官方功能列表包括多格式读取及 PDF 阅读顺序、表格、公式等结构处理。
- C2：统一表示可导出为 Markdown、HTML 和无损 JSON 等格式。
- C3：提供 OCR、本地执行和 CLI / MCP 等接入方式。

## 证据定位

| 主张 | 原文章节与定位 | 适用条件 |
|---|---|---|
| C1、C2、C3 | [功能列表](../../raw/text/Docling%20Project%20-%202026%20-%20Document%20Processing%20Overview.md#source-section-3) | 项目方功能声明，具体能力依赖版本、后端和模型 |

## 争议与不确定点

- 功能支持不保证所有跨栏、合并单元格、公式和扫描件都正确。
- 本轮核验官方文档，未安装或运行 Docling；未把功能声明写成本库解析质量已经提升的实测证据。
- 新后端应先用有页码的代表样本评测，再决定部署方式和模型资源。

## 关联页面

- [LLM Wiki 文档处理流程](../concepts/LLM%20Wiki%20文档处理流程.md)
- [LLM Wiki 与检索和文档解析方法](../comparisons/LLM%20Wiki%20与检索和文档解析方法.md)

## 这里的术语是什么意思

- **OCR**：文字识别：从图像读取文字；整页任务还需处理布局和阅读顺序。

---
type: concept
---
# GLM-OCR

## TL;DR（快速导读）

GLM-OCR 是面向文字和文档理解的专门模型，适用性要按页面类型、输出结构与部署成本评估。

## 简介

GLM-OCR 是面向文字和文档理解的专门模型，适用性要按页面类型、输出结构与部署成本评估。

## 具体怎么理解

识别票据正文、恢复表格和读取公式需要不同检查项；平均识别分数可能掩盖某类错误。

## 关键属性

- 类型：OCR / 文档理解模型
- 代表来源：[Duan et al. - 2026 - GLM-OCR Technical Report](../../wiki/summaries/Duan%20et%20al.%20-%202026%20-%20GLM-OCR%20Technical%20Report.md)
- 当前角色：compact production-minded OCR 路线代表页

## 相关主张

- GLM-OCR 通过 `0.9B` 模型规模、`MTP` 与两阶段 layout + region recognition 设计，在性能、吞吐和部署性之间做折中。
- 在当前知识库里，它是 specialized OCR model 向生产系统收敛的重要节点。

## 来源支持

- [Duan et al. - 2026 - GLM-OCR Technical Report](../../wiki/summaries/Duan%20et%20al.%20-%202026%20-%20GLM-OCR%20Technical%20Report.md)

## 关联页面

- [GLM](./GLM.md)
- [OCR](../topics/OCR.md)
- [PaddleOCR](./PaddleOCR.md)
- [DeepSeek-OCR](./DeepSeek-OCR.md)
- [dots.ocr](./dots.ocr.md)
- [传统 CV](../topics/传统%20CV.md)

## 这里的术语是什么意思

- **OCR**：文字识别：从图像读取文字；整页任务还需处理布局和阅读顺序。

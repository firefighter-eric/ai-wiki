---
type: concept
---
# PaddleOCR

## TL;DR（快速导读）

PaddleOCR 是 OCR 与文档处理工具链，既包含文字识别，也涉及版面、结构和相关应用模块。

## 简介

PaddleOCR 是 OCR 与文档处理工具链，既包含文字识别，也涉及版面、结构和相关应用模块。

## 具体怎么理解

处理一张票据可能先检测文字，再识别、排序和抽取字段；应分别定位是哪一步出错。

## 关键属性

- 类型：OCR / 文档解析工具链
- 代表来源：[PaddlePaddle Team et al. - 2025 - PaddleOCR 3.0 Technical Report](../../wiki/summaries/PaddlePaddle%20Team%20et%20al.%20-%202025%20-%20PaddleOCR%203.0%20Technical%20Report.md)
- 当前角色：OCR 工程系统与开源生态的重要节点

## 相关主张

- PaddleOCR 说明 OCR 主线不只是在比单模型识别精度，也在比谁能提供更完整的识别、解析、抽取与部署链路。
- 在当前知识库里，它把 `PP-OCRv5`、`PP-StructureV3` 与 `PP-ChatOCRv4` 收敛为一个生产级 toolkit 入口。

## 来源支持

- [PaddlePaddle Team et al. - 2025 - PaddleOCR 3.0 Technical Report](../../wiki/summaries/PaddlePaddle%20Team%20et%20al.%20-%202025%20-%20PaddleOCR%203.0%20Technical%20Report.md)

## 关联页面

- [OCR](../topics/OCR.md)
- [TrOCR](./TrOCR.md)
- [LayoutLMv3](./LayoutLMv3.md)
- [DocLLM](./DocLLM.md)
- [Qwen2.5-VL](./Qwen2.5-VL.md)
- [传统 CV](../topics/传统%20CV.md)

## 这里的术语是什么意思

- **OCR**：文字识别：从图像读取文字；整页任务还需处理布局和阅读顺序。

---
type: concept
---
# DocLayNet

## TL;DR（快速导读）

DocLayNet 是文档版面标注数据集，提供不同页面类型的区域标签，用于训练和评估版面分析。

## 简介

DocLayNet 是文档版面标注数据集，提供不同页面类型的区域标签，用于训练和评估版面分析。

## 具体怎么理解

模型需要区分标题、正文、表格等区域；检测到这些区域后，还要做文字识别和阅读顺序恢复。

## 关键属性

- 类型：数据集
- 代表来源：[Pfitzmann et al. - 2022 - DocLayNet A Large Human-Annotated Dataset for Document-Layout Segmentation](../../wiki/summaries/Pfitzmann%20et%20al.%20-%202022%20-%20DocLayNet%20A%20Large%20Human-Annotated%20Dataset%20for%20Document-Layout%20Segmentation.md)
- 当前角色：版面理解方向的基础页

## 相关主张

- DocLayNet 为文档版面检测和分割提供标准化标注基准。
- 在当前知识库里，它补足了文档理解不仅有 OCR，也有版面结构层。

## 来源支持

- [Pfitzmann et al. - 2022 - DocLayNet A Large Human-Annotated Dataset for Document-Layout Segmentation](../../wiki/summaries/Pfitzmann%20et%20al.%20-%202022%20-%20DocLayNet%20A%20Large%20Human-Annotated%20Dataset%20for%20Document-Layout%20Segmentation.md)

## 关联页面

- [LayoutLMv3](./LayoutLMv3.md)
- [DocLLM](./DocLLM.md)
- [PubTables-1M](./PubTables-1M.md)
- [传统 CV](../topics/传统%20CV.md)

## 这里的术语是什么意思

- **OCR**：文字识别：从图像读取文字；整页任务还需处理布局和阅读顺序。

---
type: concept
---
# TrOCR

## TL;DR（快速导读）

TrOCR 用预训练图像编码器和文本解码器，将文字图像直接生成字符序列，重点是识别环节。

## 简介

TrOCR 用预训练图像编码器和文本解码器，将文字图像直接生成字符序列，重点是识别环节。

## 具体怎么理解

裁剪一行手写文字后输出文本；整页检测、表格和阅读顺序不自动由这一步解决。

## 关键属性

- 类型：OCR 模型
- 代表来源：[Li et al. - 2021 - TrOCR Transformer-based Optical Character Recognition with Pre-trained Models](../../wiki/summaries/Li%20et%20al.%20-%202021%20-%20TrOCR%20Transformer-based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.md)
- 当前角色：文档文字识别主线的重要模型页

## 相关主张

- TrOCR 用预训练 Transformer 重写 OCR 的编码解码流程。
- 在当前知识库里，它连接文档理解、OCR 与生成式视觉文本建模。

## 来源支持

- [Li et al. - 2021 - TrOCR Transformer-based Optical Character Recognition with Pre-trained Models](../../wiki/summaries/Li%20et%20al.%20-%202021%20-%20TrOCR%20Transformer-based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.md)

## 关联页面

- [OCR](../topics/OCR.md)
- [LayoutLMv3](./LayoutLMv3.md)
- [DocLLM](./DocLLM.md)
- [PubTables-1M](./PubTables-1M.md)
- [传统 CV](../topics/传统%20CV.md)

## 这里的术语是什么意思

- **OCR**：文字识别：从图像读取文字；整页任务还需处理布局和阅读顺序。

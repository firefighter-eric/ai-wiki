---
type: concept
---
# DocLLM

## TL;DR（快速导读）

DocLLM 将文字与页面位置关系一起纳入语言模型，帮助理解字段、表单和票据，而不只读取文字内容。

## 简介

DocLLM 将文字与页面位置关系一起纳入语言模型，帮助理解字段、表单和票据，而不只读取文字内容。

## 具体怎么理解

“金额”和右侧数字属于同一字段，布局提供了这层联系；丢掉位置后，纯文本可能很难判断对应关系。

## 关键属性

- 类型：文档语言模型
- 代表来源：[DocLLM：布局感知文档语言模型（2024）](../../wiki/summaries/Wang%20et%20al.%20-%202023%20-%20DocLLM%20A%20layout-aware%20generative%20language%20model%20for%20multimodal%20document%20understanding.md)
- 当前角色：连接文档基础模型与生成式 LLM

## 相关主张

- DocLLM 强调布局感知信息对生成式文档理解的重要性。
- 在当前知识库里，它把 LayoutLM 类编码器路线推进到更 generative 的模型接口。

## 来源支持

- [DocLLM：布局感知文档语言模型（2024）](../../wiki/summaries/Wang%20et%20al.%20-%202023%20-%20DocLLM%20A%20layout-aware%20generative%20language%20model%20for%20multimodal%20document%20understanding.md)

## 关联页面

- [OCR](../topics/OCR.md)
- [LayoutLMv3](./LayoutLMv3.md)
- [Kosmos-2.5](./Kosmos-2.5.md)
- [TrOCR](./TrOCR.md)
- [传统 CV](../topics/传统%20CV.md)

---
type: concept
---
# PubTables-1M

## TL;DR（快速导读）

PubTables-1M 提供大规模表格抽取与结构标注，帮助模型学习单元格、行列和合并关系。

## 简介

PubTables-1M 提供大规模表格抽取与结构标注，帮助模型学习单元格、行列和合并关系。

## 具体怎么理解

“总计”跨两列时，标注必须表达跨度；仅检测到文字框还无法恢复正确网格。

## 关键属性

- 类型：数据集
- 代表来源：[Smock, Pesala, Abraham - 2022 - PubTables-1M Towards comprehensive table extraction from unstructured documents](../../wiki/summaries/Smock,%20Pesala,%20Abraham%20-%202022%20-%20PubTables-1M%20Towards%20comprehensive%20table%20extraction%20from%20unstructured%20documents.md)
- 当前角色：表格理解主线的基础支撑页

## 相关主张

- PubTables-1M 为表格检测、结构识别与抽取提供了大规模评测与训练资源。
- 在当前知识库里，它也是 GriTS、表格模型与 benchmark 对齐讨论的关键依托。

## 来源支持

- [Smock, Pesala, Abraham - 2022 - PubTables-1M Towards comprehensive table extraction from unstructured documents](../../wiki/summaries/Smock,%20Pesala,%20Abraham%20-%202022%20-%20PubTables-1M%20Towards%20comprehensive%20table%20extraction%20from%20unstructured%20documents.md)

## 关联页面

- [DocLayNet](./DocLayNet.md)
- [TrOCR](./TrOCR.md)
- [传统 CV](../topics/传统%20CV.md)

## 这里的术语是什么意思

- **benchmark**：评测基准：特定数据、任务与规则的组合，分数只在这些条件下成立。

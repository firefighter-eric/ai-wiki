---
type: concept
---
# DeepSeek-OCR

## TL;DR（快速导读）

DeepSeek-OCR 研究将文档视觉信息压缩成较少词元再转写，关注识别质量与长文处理成本的关系。

## 简介

DeepSeek-OCR 研究将文档视觉信息压缩成较少词元再转写，关注识别质量与长文处理成本的关系。

## 具体怎么理解

输入一整页扫描文档，先以视觉表示压缩，再输出文本；图表、顺序和公式应逐项核对。

## 关键属性

- 类型：OCR / 文档解析模型家族
- 代表来源：[Wei, Sun, Li - 2026 - DeepSeek-OCR 2 Visual Causal Flow](../../wiki/summaries/Wei,%20Sun,%20Li%20-%202026%20-%20DeepSeek-OCR%202%20Visual%20Causal%20Flow.md)
- 当前角色：vision-text compression 与 visual causal flow 路线代表页

## 相关主张

- DeepSeek-OCR 把 OCR 当作验证视觉压缩与高效长上下文建模的试验场，而不只把它视为传统识别任务。
- 在当前知识库里，它代表 OCR 与 long-context efficiency 交叉的一条独特路线，并从视觉侧补充 [DeepSeek 系列](../topics/DeepSeek%20系列.md) 的上下文压缩主线。

视觉压缩实验针对特定页面与视觉 token 配置；恢复率并不证明无损长期记忆。OCR 2 的视觉信息流重排也不能保证任意文本历史完全恢复。依据：[初代来源摘要](../summaries/Wei%2C%20Sun%2C%20Li%20-%202025%20-%20DeepSeek-OCR%20Contexts%20Optical%20Compression.md)与[OCR 2 摘要](../summaries/Wei%2C%20Sun%2C%20Li%20-%202026%20-%20DeepSeek-OCR%202%20Visual%20Causal%20Flow.md)。

## 来源支持

- [Wei, Sun, Li - 2025 - DeepSeek-OCR Contexts Optical Compression](../../wiki/summaries/Wei,%20Sun,%20Li%20-%202025%20-%20DeepSeek-OCR%20Contexts%20Optical%20Compression.md)
- [Wei, Sun, Li - 2026 - DeepSeek-OCR 2 Visual Causal Flow](../../wiki/summaries/Wei,%20Sun,%20Li%20-%202026%20-%20DeepSeek-OCR%202%20Visual%20Causal%20Flow.md)

## 关联页面

- [DeepSeek](./DeepSeek.md)
- [DeepSeek 系列](../topics/DeepSeek%20系列.md)
- [OCR](../topics/OCR.md)
- [GLM-OCR](./GLM-OCR.md)
- [dots.ocr](./dots.ocr.md)
- [传统 CV](../topics/传统%20CV.md)

## 这里的术语是什么意思

- **OCR**：文字识别：从图像读取文字；整页任务还需处理布局和阅读顺序。

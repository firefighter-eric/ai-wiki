# Vidu - 2026 - Pricing

## 来源信息

- 类型：官方文档 / 定价与能力矩阵
- 原始文件：../../raw/html/Vidu - 2026 - Pricing.html
- 全文文本：../../raw/text/Vidu - 2026 - Pricing.md
- 来源链接：https://platform.vidu.com/docs/pricing
- 作者：Vidu
- 年份：2026
- 状态：已整理

## 摘要

虽然这页名义上是 pricing 文档，但它同时暴露了 `Vidu Q2` 与 `Q2-Pro` 的真实能力边界，因此对知识库来说它不是单纯价格表，而是产品能力矩阵。页面表明 Vidu 路线并不只做 `text-to-video`，还把 `reference-to-video`、`video extension`、`motion sync`、`text/audio` 相关能力组织成 API 能力面。其中 `Q2-Pro` 是 reference-to-video 与更高分辨率工作流的关键节点。

## 关键事实

- `Vidu Q2` 支持 `text-to-video`，覆盖 `540P / 720P / 1080P`。
- `Vidu Q2-Pro` 支持 `reference-to-video`，覆盖 `540P / 720P / 1080P`。
- `Vidu Q2-Pro` 也支持 `image-to-video` 与 `start-end to video` 的更高规格版本。
- 文档明确列出 `video extension`，当前支持在原视频基础上增加 `1-7s`。
- 文档列出 `Motion Sync` 作为独立能力，说明 Vidu 把动作同步作为显式接口能力暴露。
- 文档还出现 `text2audio`、`timing2audio` 与 lip-sync / text-to-audio 等条目，说明其视频产品线已与音频接口联动。
- 对 `img2video / reference2video` 启用音频会额外计费，意味着音频是其正式能力面的一部分，而非旁路工具。

## 争议与不确定点

- 这页是定价与接口文档，不提供系统性的模型介绍或 benchmark，因此更适合用来确认能力边界，而不是确认综合质量。
- `Vidu Q2`、`Q2-Pro` 与 `Q3` 系列在同一页面共存，说明产品线迭代很快；其中哪些节点应视为代表模型，后续仍需更多官方技术说明补充。
- 当前页面可确认能力支持，但难以独立判断这些能力的实际效果上限。

## 关联页面

- 概念：[Vidu Q2-Pro](../../wiki/concepts/Vidu%20Q2-Pro.md)
- 主题：[视频生成](../../wiki/topics/视频生成.md)
- 作者：[ShengShu Technology](../../wiki/authors/ShengShu%20Technology.md)

# Google DeepMind - 2026 - Gemini Diffusion

## 来源信息

- 类型：Google DeepMind 模型介绍页
- 原始 HTML：[raw/html/Google DeepMind - 2026 - Gemini Diffusion.html](../../raw/html/Google%20DeepMind%20-%202026%20-%20Gemini%20Diffusion.html)
- 全文文本：[raw/text/Google DeepMind - 2026 - Gemini Diffusion.md](../../raw/text/Google%20DeepMind%20-%202026%20-%20Gemini%20Diffusion.md)
- 来源 URL：https://deepmind.google/models/gemini-diffusion/
- 作者：Google DeepMind
- 年份：2026
- 状态：已整理

## 摘要

`Gemini Diffusion` 是 Google DeepMind 用于展示 text diffusion 方向的实验模型页面。它没有像 DiffusionGemma 那样提供开放权重，但给出了 Google 对文本扩散路线的核心论点：不同于自回归模型逐 token 生成，diffusion language model 通过从噪声逐步 refinement 来生成文本，因此更适合快速响应、迭代修正和编辑类任务。

在当前知识库中，`Gemini Diffusion` 更适合作为 DiffusionGemma 的上游研究背景，而不是一个可本地部署的开放模型节点。它解释了为什么 Google 会把 text diffusion 作为一个值得从 Gemini 研究转写到 Gemma 开放家族中的方向。

## 关键事实

- 页面标题将 Gemini Diffusion 定义为 experimental text diffusion model。
- Google DeepMind 称该方向用于探索更高 control、creativity 和 speed 的文本生成。
- 页面把 text diffusion 与传统自回归语言模型对比：后者逐 token 生成，前者通过逐步 refinement 从噪声生成文本。
- 官方列出的能力包括 rapid response、more coherent text、iterative refinement。
- 页面给出 benchmark 与速度展示，包含 `1479 tokens/sec` 的 sampling speed 口径。
- 官方称 Gemini Diffusion 当前作为实验 demo 存在，用于开发和改进未来模型。

## 争议与不确定点

- 该页面不是开放模型卡，缺少完整架构、训练和部署细节。
- benchmark 与速度数据来自官方展示，不应直接外推到 DiffusionGemma 或其他开放部署场景。
- 与 DiffusionGemma 的关系应表述为研究路线背景，而不是同一模型的不同版本。

## 关联页面

- 概念：[DiffusionGemma](../concepts/DiffusionGemma.md)
- 主题：[文本扩散语言模型](../topics/%E6%96%87%E6%9C%AC%E6%89%A9%E6%95%A3%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)


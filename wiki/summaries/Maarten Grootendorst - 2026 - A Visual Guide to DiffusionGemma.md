# Maarten Grootendorst - 2026 - A Visual Guide to DiffusionGemma

## 来源信息

- 类型：第三方解释性博客 / visual guide
- 原始 HTML：[raw/html/Maarten Grootendorst - 2026 - A Visual Guide to DiffusionGemma.html](../../raw/html/Maarten%20Grootendorst%20-%202026%20-%20A%20Visual%20Guide%20to%20DiffusionGemma.html)
- 全文文本：[raw/text/Maarten Grootendorst - 2026 - A Visual Guide to DiffusionGemma.md](../../raw/text/Maarten%20Grootendorst%20-%202026%20-%20A%20Visual%20Guide%20to%20DiffusionGemma.md)
- 来源 URL：https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-diffusiongemma
- 作者：Maarten Grootendorst
- 年份：2026
- 状态：已整理；解释性来源，事实主张优先回到官方模型卡和文档核对

## 摘要

这篇 visual guide 用较直观的方式解释 DiffusionGemma 为什么不同于自回归 LLM。它把核心差异概括为：自回归模型在单用户场景中逐 token 生成，常受 memory bandwidth 限制；DiffusionGemma 则把一个 `256-token` canvas 作为并行工作单元，通过多步 denoising 逐步修正，从而把更多计算集中到单个用户请求上。

该文对概念理解很有帮助，尤其是 masked diffusion、uniform state diffusion、self-conditioning、multi-canvas sampling、scheduler 与 entropy-bounded sampler 的解释。但它不是官方模型卡，因此在知识库中应作为“解释层”使用，而不是优先事实源。

## 关键事实

- 文章把 DiffusionGemma 的核心思想解释为：把单用户场景中的空闲 compute 用来并行预测一个 `256-token` canvas。
- 文中区分 autoregressive LLM 的 memory-bound 单用户解码和 diffusion LLM 的 compute-bound block generation。
- `Uniform State Diffusion` 被解释为用随机 token 替代原 token，而不是只用 `[MASK]` token，从而允许后续步骤反复修正先前 token。
- 文中解释了为什么低置信 token 需要 re-noise：保持与训练时随机噪声分布接近，并避免模型围绕错误 token 继续规划。
- 架构解释中，作者把 DiffusionGemma 描述为在同一个 `Gemma 4 26B A4B` 模型上切换 encoder mode 与 denoiser mode。
- 该文详细解释了 self-conditioning：将上一步 softmax 概率与 embedding matrix 相乘，形成每个位置的概率分布表示，再传入下一步。
- multi-canvas sampling 被解释为 diffusion block 与 autoregressive stitching 的结合：每个 canvas 内部并行 denoise，canvas 之间按顺序追加。
- scheduler 由最大步数、logits temperature schedule 和 adaptive stopping 组成。
- entropy-bounded sampler 负责 canvas initialization、token acceptance 和 token re-noising。

## 争议与不确定点

- 该文是第三方解释性材料，部分机制细节需要与官方代码或技术报告互证。
- 文章的教学类比有助于理解，但不应替代模型卡中的正式参数、benchmark 或限制说明。
- 当前仓库尚缺完整论文或代码级 summary 来验证所有内部实现细节。

## 关联页面

- 概念：[DiffusionGemma](../concepts/DiffusionGemma.md)
- 主题：[文本扩散语言模型](../topics/%E6%96%87%E6%9C%AC%E6%89%A9%E6%95%A3%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)


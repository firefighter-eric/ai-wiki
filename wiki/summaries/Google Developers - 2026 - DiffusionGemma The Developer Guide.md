---
type: summary
status: refined
---
# Google Developers - 2026 - DiffusionGemma The Developer Guide

## 来源信息

- 类型：Google Developers 官方开发者指南
- 原始 HTML：[raw/html/Google Developers - 2026 - DiffusionGemma The Developer Guide.html](../../raw/html/Google%20Developers%20-%202026%20-%20DiffusionGemma%20The%20Developer%20Guide.html)
- 全文文本：[raw/text/Google Developers - 2026 - DiffusionGemma The Developer Guide.md](../../raw/text/Google%20Developers%20-%202026%20-%20DiffusionGemma%20The%20Developer%20Guide.md)
- 来源 URL：https://developers.googleblog.com/en/diffusiongemma-the-developer-guide/
- 作者：Omar Sanseviero、Ian Ballantyne
- 年份：2026
- 状态：已整理

## 摘要

这篇开发者指南是理解 DiffusionGemma 工程机制的核心来源。它把模型解释为在 `Gemma 4` backbone 上加入 text diffusion 生成路径：模型用 causal prefill 读取 prompt 并写入 `KV cache`，再用 bidirectional denoising 在 `256-token canvas` 上并行修正 token；当一个 canvas 完成后，再把它追加进上下文并生成下一块。

该指南还给出可运行的 vLLM serving 配置，并展示 Sudoku fine-tuning 示例。Sudoku 示例的价值不在于证明模型通用能力更强，而是说明并行双向 denoising 适合有全局约束、需要反复修正的结构化问题。

## 关键事实

- DiffusionGemma 通过 compute-bound parallel generation 绕过传统自回归模型本地推理中的 memory bandwidth 瓶颈。
- `Uniform State Diffusion`：从随机 placeholder token canvas 开始，多次并行 denoise。
- `Block Autoregressive Diffusion`：单块 canvas 为 `256` tokens，长输出通过“完成一块、提交到 `KV cache`、再生成下一块”的方式扩展。
- prefill / incremental prefill 使用 causal attention，denoising 阶段使用 bidirectional attention。
- 双向 denoising 允许一个 canvas 内任意 token 彼此可见，因此比自回归生成更适合 Sudoku 一类强全局约束问题。
- 开发者指南提供 vLLM 命令，包含 `--max-model-len 262144`、`--diffusion-config '{"canvas_length": 256}'`、`diffusion_sampler: entropy_bound`、`diffusion_entropy_bound: 0.1` 等关键参数。
- Sudoku 示例中，base DiffusionGemma 对 Sudoku 几乎不能解出，但简单 JAX SFT 后 correctness 提高到 `80%`，并能在更少 denoising steps 下提前停止。
- 指南列出可用框架：vLLM、Hugging Face Transformers、SGLang、MLX；微调可参考 Hackable Diffusion、Unsloth、NVIDIA NeMo。

## 争议与不确定点

- Sudoku 是一个很适合展示双向约束传播的任务，不能直接代表通用问答、长文本写作或复杂 agent 任务的全面提升。
- 文中 vLLM 配置是官方推荐起点，不等于所有硬件和 workload 的最优配置。
- 对 block diffusion 的训练细节、损失函数、数据构成仍需更完整技术报告或代码补充。

## 关联页面

- 概念：[DiffusionGemma](../concepts/DiffusionGemma.md)
- 主题：[文本扩散语言模型](../topics/%E6%96%87%E6%9C%AC%E6%89%A9%E6%95%A3%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)


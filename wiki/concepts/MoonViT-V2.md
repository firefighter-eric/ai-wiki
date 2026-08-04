# MoonViT-V2

## 简介

`MoonViT-V2` 是 Kimi K3 的视觉编码器。与先对视觉塔做对比学习、再接入语言模型的常见路径不同，K3 报告称 MoonViT-V2 从随机初始化开始，与语言 backbone 在统一 next-token prediction 目标下联合训练。

## 关键属性

- 类型：原生多模态视觉编码器 / Vision Transformer
- 规模：约 401M 参数、27 层、patch size 14、12 attention heads
- 输入：图像与视频在训练中共享参数处理
- 接口：视觉特征通过轻量 MLP projector 映射到 LLM embedding space
- 训练：与文本 token 从预训练起联合优化

## 相关主张

- K3 将 native multimodality 定义为“从预训练开始联合优化”，而不是语言基座完成后再接一个冻结或预训练视觉塔。
- 报告比较了从头训练的 MoonViT-V2 与 SigLIP-initialized MoonViT-3D，称前者在联合训练中 gradient norm 更低、spikes 更少，同时在视觉评测上达到相当水平。
- 视觉数据不只包含 caption，还包含 OCR、图文交错、perception、video，以及 SVG、3D、Web、Game、CAD 等 code-rendered visuals；因此视觉路线与 K3 的 coding/agent 训练直接交叉。
- 长图像与视频造成的计算不均衡由 dynamic context parallelism 和把 ViT computation 填入 pipeline bubbles 的系统优化处理。
- 当前主要证据来自 K3 官方报告，尚不足以断言“从头联合训练”在所有模型规模和数据预算下都优于 contrastive initialization。

## 来源支持

- [Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence](../../wiki/summaries/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)
- [Moonshot AI - 2026 - Kimi K3 Model Repository](../../wiki/summaries/Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20Model%20Repository.md)

## 关联页面

- [Kimi K3](./Kimi%20K3.md)
- [ViT](./ViT.md)
- [CLIP](./CLIP.md)
- [LLM 预训练](../topics/LLM%20预训练.md)

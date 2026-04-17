# Team Seedance et al. - 2026 - Seedance 2.0 Advancing Video Generation for World Complexity

## 来源信息

- 类型：论文 / model card
- 原始文件：../../raw/pdf/Team Seedance et al. - 2026 - Seedance 2.0 Advancing Video Generation for World Complexity.pdf
- 全文文本：../../raw/text/Team Seedance et al. - 2026 - Seedance 2.0 Advancing Video Generation for World Complexity.md
- 来源链接：https://arxiv.org/abs/2604.14148
- 作者：Team Seedance et al.
- 年份：2026
- 状态：已整理

## 摘要

Seedance 2.0 是 ByteDance Seed 在 2026 年发布的原生多模态音视频生成模型。该模型把 `text / image / audio / video` 四类输入统一进同一音视频联合生成框架，不只覆盖 `T2V / I2V`，也强调多模态参考、视频编辑、续写与 extension 等更接近真实创作工作流的能力。与许多只给出单轮生成 demo 的视频模型相比，这篇来源更突出其“可控生产”定位：复杂动作稳定性、专业镜头语言、多镜头叙事与音画同步被当作核心竞争维度。

## 关键事实

- Seedance 2.0 支持直接生成 `4-15` 秒音视频内容，原生输出分辨率为 `480p` 与 `720p`。
- 当前开放平台支持最多 `3` 段视频、`9` 张图片和 `3` 段音频作为参考输入，并支持文本、图像、音频、视频的组合条件控制。
- 论文把其能力范围组织为 `T2V / I2V / R2V`、视频编辑、continuation 与 extension，而不是只讨论单一文生视频任务。
- 模型显式强调复杂动作、多人交互、镜头调度、叙事节奏、风格参考、特效参考和角色一致性等接近 production workflow 的维度。
- 音频侧主打双声道 / 双耳沉浸式生成、环境音与配乐分层，以及与视觉动作的严格时间同步。
- 按文中自建 `SeedVideoBench 2.0` 结果，Seedance 2.0 在 `T2V / I2V / R2V` 的已评估维度上均排第一；对比对象包括 `Kling 3.0`、`Veo 3.1`、`Sora 2 Pro`、`Wan 2.6`、`Vidu Q2 Pro` 等。
- 论文同时声称，在 `Arena.AI` 视频榜单中，`Dreamina Seedance 2.0 720p` 于 `2026-04-08`（文中注明 Eastern Time）在 `Text-to-Video` 与 `Image-to-Video` 两个榜单都排第一。
- 这篇来源也明确暴露了短板：在 `R2V` 的 extension 任务上，Seedance 2.0 虽然支持更广输入，但质量仍落后于 `Veo 3.1`。

## 争议与不确定点

- 这篇来源更接近带 benchmark 的官方 model card，而不是完整公开方法论文；架构、训练数据与训练细节披露明显少于评测部分。
- 绝大多数领先结论来自作者自建的 `SeedVideoBench 2.0` 与作者选择的商用模型对比，属于强信息量但非独立第三方评测。
- `Arena.AI` 排名结论带有明确时间点 `2026-04-08`，后续可能变化，不应把它视为长期稳定事实。
- 论文显示 Seedance 2.0 在视频 extension 上仍弱于 `Veo 3.1`，说明“全任务领先”不能机械外推到所有子任务。

## 关联页面

- 概念：[Seedance 2.0](../../wiki/concepts/Seedance%202.0.md)
- 主题：[视频生成](../../wiki/topics/视频生成.md)
- 作者：[ByteDance Seed](../../wiki/authors/ByteDance%20Seed.md)

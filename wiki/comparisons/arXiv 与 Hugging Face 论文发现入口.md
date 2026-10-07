---
type: comparison
---
# arXiv 与 Hugging Face 论文发现入口

## TL;DR（快速导读）

方向还不明确时，可用 Hugging Face 发现社区关注的候选；有具体问题后，用 arXiv 分类、关键词和引用继续扩展。热度只是筛选线索。

## 用一个例子看差别

例如先从周榜发现缓存优化论文，再到 arXiv 搜索相关关键词，顺着原论文引用补齐早期方法。进入精读前，仍要看问题是否相关、原文质量及有无独立验证。

## 比较目标

回答“想找很多 AI paper 阅读，应该从哪里开始”。比较的是发现与筛选论文的方式，而不是给两个平台的论文质量打分。页面功能核验日期为 2026-10-07；使用建议是基于三篇入口 summary 的综合判断。

## 核心判断

**方向尚未确定、希望先广泛浏览时，优先从 Hugging Face Papers 开始；确定研究问题后，再用 arXiv 的分类目录和关键词检索扩展。** Hugging Face 帮助发现社区正在关注的候选与相关资源，arXiv 帮助沿具体学科和研究问题持续搜索。这一建议由 [Trending Papers](../summaries/Hugging%20Face%20-%202026%20-%20Trending%20Papers.md)、[Paper Pages](../summaries/Hugging%20Face%20-%202026%20-%20Paper%20Pages.md) 与 [arXiv recent 目录](../summaries/arXiv%20-%202026%20-%20Artificial%20Intelligence%20Recent%20Submissions.md) 的不同组织方式支持。

Hugging Face 论文页可以通过 arXiv ID 建立，所以两边常常是同一篇论文的不同入口。初筛放在 Hugging Face、精读回到原文是自然的组合，并不需要二选一。这个关系由 [Paper Pages 文档 summary](../summaries/Hugging%20Face%20-%202026%20-%20Paper%20Pages.md) 支持。

## 发现方式与适用场景

| 比较维度 | Hugging Face Papers | arXiv 官方目录与检索 |
|---|---|---|
| 主要入口 | Daily / Weekly / Monthly 与 Trending | 学科 recent / new、搜索与 Advanced search |
| 可见筛选信号 | 社区提交、Upvote、摘要、部分代码入口 | 日期、标题、作者、学科标签与原文入口 |
| 适合的起点 | 尚未确定方向，想快速形成阅读候选集 | 已有关键词、作者或研究问题，想扩大检索 |
| 深入阅读的帮助 | 关联模型、数据集、Spaces 与社区讨论 | 回到摘要页、PDF 和可用 HTML 原文 |
| 需要补足的判断 | 热度是否与自己的问题相关、资源是否可用 | 哪些候选值得精读、实验是否支持结论 |

表中的功能分别回溯上方三篇 summary；“适合的起点”是使用建议。目前没有同一任务下的查全率、质量或效率测量，可以证明一方普遍优于另一方。

## “AI 趋势榜”的解释边界

arXiv 官方 recent 页面按学科与日期组织；目录顺序不能当作热度排行。若“arXiv AI 趋势榜”指第三方产品，还需要核对具体网址、数据来源、更新窗口与排序规则，不能仅因名称包含 arXiv 就把它当作官方推荐。该边界见 [arXiv recent 目录 summary](../summaries/arXiv%20-%202026%20-%20Artificial%20Intelligence%20Recent%20Submissions.md)。

Hugging Face 展示 Upvote，但可见页面不足以还原 Trending 的完整公式。应把社区热度用于召回候选，随后单独检查研究问题、方法、baseline、实验设置、消融与局限。此处的依据见 [Trending Papers summary](../summaries/Hugging%20Face%20-%202026%20-%20Trending%20Papers.md)。

## 建议的阅读流程

以下数量是可调整的阅读节奏示例，不是平台统计或质量保证：

1. 在 [Hugging Face Papers](https://huggingface.co/papers) 先看 Weekly，再用 Daily 跟进；若要补一段时间的热点，可看 Monthly。
2. 从标题与摘要挑出约 20–30 篇候选，围绕自己的问题留下 3–5 篇优先精读。初筛重点看解决什么问题、新方法改变了什么、实验是否能检验这个改变。
3. 用选中论文的关键词、作者、引用文献继续在 arXiv 搜索。AI 相关论文分散在多个类别，可从 [cs.LG](https://arxiv.org/list/cs.LG/recent)、[cs.CL](https://arxiv.org/list/cs.CL/recent)、[cs.CV](https://arxiv.org/list/cs.CV/recent) 和 [cs.AI](https://arxiv.org/list/cs.AI/recent) 等入口扩展；这些类别页在本次查询中已在线核验。
4. 合并候选时按 arXiv ID 去重，并记录年份 / 版本，避免把跨类别和跨平台重复展示当成新增来源。
5. 对真正要纳入本库的论文，执行 `原始 HTML / PDF → raw/text → summary`，然后把新证据写回相关 topic / concept / comparison。入口页中的摘要不能代替独立论文的精修 summary。

对于当前知识库，选题可以从已有 [LLM 预训练](../topics/LLM%20预训练.md)、[LLM RL](../topics/LLM%20RL.md)、[视频生成](../topics/视频生成.md) 与 [注意力机制 Attention](../topics/注意力机制%20Attention.md) 页面进入，沿未解决问题找来源。前三个 topic 目前为待建设状态，应把它们用于组织选题，并继续回溯精修 summary 判断具体结论。

## 证据基础

- [arXiv - 2026 - Artificial Intelligence Recent Submissions](../summaries/arXiv%20-%202026%20-%20Artificial%20Intelligence%20Recent%20Submissions.md)：支持官方目录的日期、分类、分页、搜索与原文导航结构。
- [Hugging Face - 2026 - Trending Papers](../summaries/Hugging%20Face%20-%202026%20-%20Trending%20Papers.md)：支持社区提交、Upvote、时间窗口与部分代码入口，并限定热度的解释范围。
- [Hugging Face - 2026 - Paper Pages](../summaries/Hugging%20Face%20-%202026%20-%20Paper%20Pages.md)：支持 arXiv ID 关联、模型 / 数据集 / Spaces 与讨论功能，以及覆盖和验证的边界。

## 关联页面

- [LLM 预训练](../topics/LLM%20预训练.md)：待建设 topic，提供基础模型研究选题入口。
- [LLM RL](../topics/LLM%20RL.md)：待建设 topic，提供强化学习与后训练选题入口。
- [视频生成](../topics/视频生成.md)：待建设 topic，提供生成媒体选题入口。
- [注意力机制 Attention](../topics/注意力机制%20Attention.md)：正式 topic，可沿其分层与开放问题继续找论文。

---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Bai et al. - 2023 - Qwen Technical Report

## TL;DR（快速导读）

初代 Qwen 报告同时介绍基础、聊天及专门模型，并说明它们如何训练成能遵循指令和使用工具的系统。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 先看一个例子

看图问答选视觉理解资料，输出语音看 Omni，制作图片看 Image；这些能力不是同一种任务。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdf/Bai et al. - 2023 - Qwen Technical Report.pdf
- 全文文本：../../raw/text/Bai et al. - 2023 - Qwen Technical Report.md
- 作者：Bai et al.
- 年份：2023
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。

## 摘要

报告把中英为主的多语言编码、预训练、示范微调和人类偏好训练放在同一套家族中，还涉及代码、数学与工具使用。阅读时先区分基础模型和聊天模型，再核对某项能力属于哪种模型与训练阶段。

## 关键事实

- **C1**：报告覆盖 Qwen 1.8B、7B、14B base/chat，并讨论代码、数学、工具和多模态扩展。
- **C2**：训练数据包含网页、书籍、百科和代码，进行精确/模糊去重和质量筛选，总语料规模最高约 3T tokens。
- **C3**：BPE 词表从 tiktoken cl100k_base 扩充到约 152K，加入多语词项并按单数字切分。
- **C4**：decoder 使用 untied embedding、RoPE、QKV bias、RMSNorm 与 SwiGLU。
- **C5**：长上下文外推结合 dynamic NTK、LogN scaling 与分层窗口注意力，报告用长文本 perplexity 验证。
- **C6**：chat 后训练包括 SFT、偏好/奖励模型与 PPO，PPO 使用 policy/value/reference/reward 四模型。
- **C7**：工具、代码解释器与多模态代理能力有单独任务评测。

## 争议与不确定点

- 人类聊天比较使用作者构建的中文指令集，不能直接代表任意语言和生产任务。
- 部分基线分数来自官方结果或 OpenCompass，比较并非所有模型同环境重跑。

## 关联页面

- 概念：[Qwen](../../wiki/concepts/Qwen.md)
- 概念：[Qwen1.5](../../wiki/concepts/Qwen1.5.md)
- 主题：[LLM 预训练](../topics/LLM%20预训练.md)
- 主题：[Qwen 系列](../../wiki/topics/Qwen%20系列.md)
- [Jingren Zhou](../authors/Jingren%20Zhou.md)：沿作者或机构继续阅读相关来源。
- [Jinze Bai](../authors/Jinze%20Bai.md)：沿作者或机构继续阅读相关来源。
- [Qwen Team - Alibaba](../authors/Qwen%20Team%20-%20Alibaba.md)：沿作者或机构继续阅读相关来源。
- [Shuai Bai](../authors/Shuai%20Bai.md)：沿作者或机构继续阅读相关来源。
- [Junyang Lin](../authors/Junyang%20Lin.md)：沿作者或机构继续阅读相关来源。

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **reward model**：奖励模型：根据训练信号给回答或行为打分，分数是目标的近似。
- **SFT**：监督微调：用输入与参考输出继续训练已有模型。
- **RLHF**：人类反馈强化学习：把人类偏好转成奖励，并用它调整模型行为。
- **embedding**：向量表示：把文字、图片等编码成一组数，用于模型计算或相似度比较。
- **decoder**：解码器：根据已有表示产生文字、图像或其他输出。

## 方法与实验解读

报告把数据清洗、词表、decoder 配置、上下文外推和聊天后训练连成完整模型路线。其 base 的 few-shot 学术分数、chat 的人类排序和工具成功率测量不同能力。阅读时先追踪数据与结构，再核对表格中的模型版本和 shot 设置；若要在部署中复用，另行验证长上下文中的检索与工具失败恢复。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/pdf/Bai%20et%20al.%20-%202023%20-%20Qwen%20Technical%20Report.pdf#page=4 ) | 2023 年报告版本，与后来 Qwen2/3 分开。 |
| C2 | [原文]( ../../raw/pdf/Bai%20et%20al.%20-%202023%20-%20Qwen%20Technical%20Report.pdf#page=5 ) | 不同模型的实际训练 token 数见第 7 页，不一律等于 3T。 |
| C3 | [原文]( ../../raw/pdf/Bai%20et%20al.%20-%202023%20-%20Qwen%20Technical%20Report.pdf#page=6 ) | tokenizer 设计，不是多语言性能保证。 |
| C4 | [原文]( ../../raw/pdf/Bai%20et%20al.%20-%202023%20-%20Qwen%20Technical%20Report.pdf#page=7 ) | 结构选择需要与报告版本一致。 |
| C5 | [原文]( ../../raw/pdf/Bai%20et%20al.%20-%202023%20-%20Qwen%20Technical%20Report.pdf#page=9 ) | 低 PPL 不是检索、问答或位置鲁棒性的完整评测。 |
| C6 | [原文]( ../../raw/pdf/Bai%20et%20al.%20-%202023%20-%20Qwen%20Technical%20Report.pdf#page=11 ) | 学术 base 评测与对齐后聊天评测分别解释。 |
| C7 | [原文]( ../../raw/pdf/Bai%20et%20al.%20-%202023%20-%20Qwen%20Technical%20Report.pdf#page=13 ) | 依赖外部工具环境；不等于模型内部完成全部功能。 |

## 核证范围

核读报告第 4–13 页的模型家族、预训练、词表、结构、上下文、后训练与工具评测。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

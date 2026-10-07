---
type: summary
status: refined
evidence_schema: 1
review_scope: core_claims
reviewed: 2026-10-07
---
# Zhang et al. - 2022 - OPT Open Pre-trained Transformer Language Models

## TL;DR（快速导读）

OPT 开放不同规模的解码器语言模型与研究材料，为研究大规模语言模型提供更可访问的训练和评测入口。

阅读重点：先看方法如何解决问题，再看实验条件与适用边界。

## 摘要

大型模型训练昂贵，API 访问又难以支持全部研究问题。OPT 以模型家族形式提供研究接口。需要区分开放权重、训练材料和许可条件，具体适用范围应核对对应版本。

## 具体怎么理解

研究者可检查和微调本地权重，而不只是向远程接口发请求；但开放权重仍不自动表示训练全过程完全可复现。

## 关键事实

- **C1**：OPT 是 125M 至 175B 的自回归 Transformer 家族，训练配方主要参考 GPT-3。
- **C2**：论文指出 OPT-175B 对直接指令和问句可能续写对话场景而非执行要求。

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：[打开原始文件](../../raw/pdf/Zhang%20et%20al.%20-%202022%20-%20OPT%20Open%20Pre-trained%20Transformer%20Language%20Models.pdf)
- 全文文本：[打开全文文本](../../raw/text/Zhang%20et%20al.%20-%202022%20-%20OPT%20Open%20Pre-trained%20Transformer%20Language%20Models.md)
- 作者：Zhang et al.
- 年份：2022
- 状态：精修摘要；已核证本页核心方法、实验条件与局限。
- 原始 HTML：[打开快照](../../raw/html/Zhang%20et%20al.%20-%202022%20-%20OPT%20Open%20Pre-trained%20Transformer%20Language%20Models.html)

## 争议与不确定点

- 公开研究访问与具体权重许可证需逐项核对，不能笼统视为无条件商用。
- 训练过程透明不消除偏见、毒性和事实错误。

## 关联页面

- 主题：[LLM预训练](../topics/LLM%20预训练.md)
- 综合：暂无

## 方法与实验解读

OPT 的贡献包括公开模型和训练过程，让研究者能检查大模型训练的现实问题。基础模型预测续写，用户要求的动作未必就是最高概率续文；这也是后续指令训练要解决的差距。

## 证据定位

本页主张按下表回到原文；数字与比较只适用于对应论文版本和评测条件。

| 主张 | 原文定位 | 成立条件与解读范围 |
| --- | --- | --- |
| C1 | [原文]( ../../raw/text/Zhang%20et%20al.%20-%202022%20-%20OPT%20Open%20Pre-trained%20Transformer%20Language%20Models.md#source-section-5 ) | 基础模型家族，尚不是指令对齐模型 |
| C2 | [原文]( ../../raw/text/Zhang%20et%20al.%20-%202022%20-%20OPT%20Open%20Pre-trained%20Transformer%20Language%20Models.md#source-section-27 ) | 语言续写能力与助手指令遵循能力不同 |

## 核证范围

核对 §2.1–2.2 的家族与训练、§3.1 的提示评测和 §5 的指令局限。

核证日期：2026-10-07。本文是可复用的来源摘要；核证范围限定于本页列出的主张，不表示独立复现实验或审阅了每个附录细节。

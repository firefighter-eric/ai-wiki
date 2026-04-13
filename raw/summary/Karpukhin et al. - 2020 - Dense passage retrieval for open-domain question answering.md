# Karpukhin et al. - 2020 - Dense passage retrieval for open-domain question answering

## 来源信息

- 类型：论文 / 技术报告
- 原始文件：../../raw/pdfs/Karpukhin et al. - 2020 - Dense passage retrieval for open-domain question answering.pdf
- 原始 HTML：../../raw/html/Karpukhin et al. - 2020 - Dense passage retrieval for open-domain question answering.html
- 全文文本：../../raw/text/Karpukhin et al. - 2020 - Dense passage retrieval for open-domain question answering.md
- 作者：Karpukhin et al.
- 年份：2020
- 状态：已整理

## 摘要

这篇论文是 Dense Retrieval 路线的关键节点。它用简单的双编码器结构证明，开放域问答中的知识召回并不一定要依赖 BM25 或 TF-IDF 这类词项匹配方法；只要训练设置得当，稠密向量检索本身就能成为主力召回层。对智能问答与智能客服而言，它的重要性在于把“用户问法的多样性”从规则匹配问题转成语义表征问题。

## 关键事实

- DPR 使用两个独立编码器分别编码问题与 passage，并以点积作为相似度，用 FAISS 做高效近邻检索。
- 论文主张只用已有 question-passage 对进行 fine-tuning，就能显著超过 BM25，而不必依赖更复杂的额外预训练任务。
- 文中报告 dense retriever 在 top-20 passage retrieval accuracy 上相对强 BM25 基线有明显提升，并把更高召回转化为端到端 QA 性能提升。
- 该文不仅提出方法，也明确了一个系统分层：检索器先缩小候选上下文，再由 reader 或后续生成模块完成答案抽取或组织。
- 在客服 / 企业问答语境中，这一路线解释了为什么知识库问答需要 dense retrieval 或 embedding retrieval：用户常用口语、错别字、近义表达或业务别称提问，稀疏匹配往往不稳。

## 争议与不确定点

- 该文是开放域 QA 检索论文，不直接覆盖多轮对话状态、企业政策执行和客服安全边界，因此在智能客服 topic 中主要支撑“知识召回层”而非完整系统层。
- DPR 证明 dense retrieval 可行，但并不意味着所有客服系统都应抛弃 BM25；很多工业系统仍采用 hybrid retrieval。

## 关联页面

- 概念：[Dense Retrieval](../../wiki/concepts/Dense%20Retrieval.md)
- 概念：[DPR](../../wiki/concepts/DPR.md)
- 主题：[AI 智能问答与智能客服](../../wiki/topics/AI%20智能问答与智能客服.md)

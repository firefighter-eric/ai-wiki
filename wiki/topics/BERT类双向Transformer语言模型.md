---
type: topic
status: formal
review_scope: evidence_synthesis
reviewed: 2026-10-07
---
# BERT类双向Transformer语言模型

## TL;DR（快速导读）

BERT 类编码器提供上下文表示，后续工作分别改训练配方、跨度目标、句向量空间和多语言迁移。先确认要的是 token 标签、句子相似度还是 query–document 匹配，再选模型。

阅读重点：先按问题选择路线，再核对比较条件与证据边界。

## 先用一个问题理解

若要给评论分类，可以从编码器微调读起；若要检索相似问题，应看 Sentence-BERT、SimCSE 等表示适配；若要抽答案跨度，还要检查跨度建模。三种任务共享底座，却需要不同训练和输出。

## 页面状态

正式 topic；2026-10-07 复核核心来源并补充方法比较。正文区分论文实验、作者报告和本文综合判断；开放问题表示研究证据的边界。

## 主题定义

本页讨论以 `BERT` 为起点的一组 **双向 Transformer 编码器语言模型**，包括 `RoBERTa`、`SpanBERT`、`Sentence-BERT`、`SimCSE`、`XLM-R` 等代表节点。它们共享的核心特征不是“名字里带不带 BERT”，而是 **以 encoder-only 或双塔 encoder 为主，目标优先落在理解、匹配、抽取、检索与多语言表征，而不是开放式自回归生成**。

这个 topic 的重点不是罗列“BERT 之后出现了哪些变体”，而是说明为什么 BERT 家族会沿几条相对稳定的子线分化：一条围绕 **预训练范式和训练配方**，一条围绕 **任务结构感知的编码目标**，一条围绕 **句向量和稠密检索**，一条围绕 **多语言统一编码**。这几条线解决的并不是同一个问题，因此不能把所有 BERT 变体粗暴理解为“更强的 BERT”。

它与 [LLM 预训练](../topics/LLM%20预训练.md) 的边界在于：后者讨论 decoder-only foundation model 如何通过规模化预训练形成生成能力，而本页关注 **双向编码器为什么长期构成 NLP 理解任务的底座**。它与 [传统 NLP](./传统%20NLP.md) 的边界在于：传统 NLP 是更宽的历史与方法谱系，本页则专门聚焦到 BERT 类双向编码器这一成熟家族。

## 核心问题

- **双向编码器范式为什么成立**：为什么 masked language modeling 能把深层上下文表征变成统一可迁移底座。
- **BERT 家族的主要改进轴到底是什么**：提升来自训练更充分、目标更贴近任务结构，还是把模型变成更适合句向量与检索的接口。
- **为什么句向量和稠密检索会从 BERT 主线中分化出来**：这是否意味着原始 BERT 的表征几何并不天然适合相似度空间。
- **多语言编码器为什么是一个独立子线**：多语言扩展解决的是共享参数与语言不平衡问题，而不是英文 BERT 的简单放大版。
- **在 decoder-only LLM 兴起后，BERT 类模型还剩下什么不可替代性**：需要区分“被抢走了哪些任务”与“仍然保有结构优势的任务”。

## 主线脉络 / 方法分层

- **范式奠基层**：`Devlin et al. 2019` 的 `BERT` 建立了双向 Transformer 编码器的主范式，即通过 masked language modeling 学习上下文化 token 表征，再以微调方式迁移到分类、抽取、问答等下游任务。它真正奠定的不是某个具体网络细节，而是 **“统一预训练编码器 + 任务头”** 这一工作模式。BERT 的成功意味着 NLP 不再需要为每个任务分别设计完全不同的特征工程或模型骨架。
- **训练配方重估层**：`Liu et al. 2019` 的 `RoBERTa` 表明，早期对 BERT 的很多判断其实混杂了训练不充分因素。更大的数据、更长的训练、更合理的 batch 与 masking 策略，可以在不根本改变架构的前提下显著抬高性能。这条线的重要含义是：**BERT 家族内部的很多“结构改进”评价，必须先扣除训练配方差异**。如果不先承认这一点，就容易把训练资源收益误判成架构创新。
- **任务结构感知层**：`Joshi et al. 2020` 的 `SpanBERT` 代表另一种不同于 RoBERTa 的改进逻辑。它不是单纯把 BERT 训得更久，而是认为某些核心任务，例如抽取、问答、共指消解，本质上依赖 span 级语义单元，因此预训练目标也应围绕 span 而不是独立 token 设计。这条线说明，**BERT 家族的演进并不只有“规模化”一条路，任务结构本身也会反过来塑造预训练目标**。
- **句向量化层**：`Sentence-BERT`、`SimCSE`、`DeCLUTR`、`ConSERT` 等 summary 共同指向一个稳定判断：原始 BERT 很强，但 **并不天然提供良好的句向量空间**。这是因为 token-level contextual encoding 的目标，并不等于句级距离结构已经被整理好。于是 BERT 主线中分化出一条专门研究 pooling、双塔结构、对比学习和表征几何的路线，其目标不是“让模型更懂语言”，而是 **让句子空间更适合检索、聚类、匹配和排序**。
- **检索化与双塔接口层**：`Karpukhin et al. 2020` 的 DPR 把句向量化进一步推进到开放域问答与大规模召回场景。这里 BERT 类模型的角色发生了变化：它不再只是下游任务的编码器，而变成 **高维语义索引的表征函数**。这一步很关键，因为它说明 BERT 家族并不只服务“理解任务”，还深度进入了 retrieval stack，成为后续 RAG 之前史的重要一环。
- **多语言统一编码层**：`Conneau et al. 2020` 的 `XLM-R` 与 `Conneau 2021` 的多语言 MLM 工作说明，双向编码器可以扩展为跨语言统一表征底座。这条线真正要解决的问题不是把 BERT 翻译成多语版，而是 **在共享参数下平衡高资源语言、低资源语言与跨语言迁移**。多语言编码器之所以构成独立子线，是因为它面对的瓶颈已经从单语言建模转向语言分布不平衡与迁移效率。
- **向任务系统外延层**：`Liu, Lapata 2020` 的预训练编码器摘要工作提醒，BERT 家族虽以理解为主，但其表征可以作为摘要等生成/半生成任务的强编码底座。这并不意味着 BERT 进入了 decoder-only 赛道，而是说明 **编码器底座可以外接更复杂的任务结构**。因此 BERT 家族的价值不应只按“它能不能直接生成文本”来评价。

### 共享骨架怎样变成三种不同接口

[BERT](../summaries/Devlin%20et%20al.%20-%202019%20-%20BERT%20Pre-training%20of%20deep%20bidirectional%20transformers%20for%20language%20understanding.md)在微调时允许输入 token 相互读取；MLM 则通过遮住一部分输入制造预测任务。它既不是对每个可见 token 都做完整语言建模，也不提供自回归聊天接口。[RoBERTa](../summaries/Liu%20et%20al.%20-%202019%20-%20RoBERTa%20A%20Robustly%20Optimized%20BERT%20Pretraining%20Approach.md)改变数据、训练规模、动态 masking 与 NSP 配方，[SpanBERT](../summaries/Joshi%20et%20al.%20-%202020%20-%20Spanbert%20Improving%20pre-training%20by%20representing%20and%20predicting%20spans.md)改变遮蔽单元并从跨度边界预测内部 token。前者说明基线资源不足会误导架构比较，后者说明任务结构仍能影响目标设计，两条证据不能彼此抵消。

把一句话编码成向量是另一个目标。[Sentence-BERT（Reimers 与 Gurevych，2019）](../summaries/Devlin%2C%20Liu%20-%202014%20-%20Sentence-BERT%20Sentence%20Embeddings%20using%20Siamese%20BERT-Networks.md)用孪生编码器训练可复用的句表示；历史文件名有误，以摘要中的核对信息为准。[SimCSE](../summaries/Gao%2C%20Yao%2C%20Chen%20-%202021%20-%20SimCSE%20Simple%20Contrastive%20Learning%20of%20Sentence%20Embeddings.md)无监督版本用独立 dropout 生成视图，监督版本额外用 NLI 标签。无监督不表示不需要文本，监督结果也不能归为“只换 pooling 的收益”。[BERT-flow](../summaries/Li%20et%20al.%20-%202019%20-%20On%20the%20Sentence%20Embeddings%20from%20Pre-trained%20Language%20Models.md)和[Whitening](../summaries/Su%20et%20al.%20-%202021%20-%20Whitening%20Sentence%20Representations%20for%20Better%20Semantics%20and%20Faster%20Retrieval.md)侧重几何变换，与重训编码器的对比学习不同。

| 接口 | 是否可提前计算文档 | 训练信号在约束什么 | 常见误用 |
| --- | --- | --- | --- |
| token/span 预测 | 可复用隐藏表示，但还需任务头 | 标签或跨度结构 | 把 token 精度当句子检索质量 |
| 双塔句/段落向量 | 可以，支持离线建索引 | 正负例距离与相似度 | STS 高分直接等于检索召回高 |
| cross-encoder 配对评分 | 通常随 query 重新计算 | query 与候选共同交互 | 把较贵重排接口当大库首轮召回 |

表中区别由模型接口推导；实际延迟取决于尺寸、长度、批量和硬件。[DPR](../summaries/Karpukhin%20et%20al.%20-%202020%20-%20Dense%20passage%20retrieval%20for%20open-domain%20question%20answering.md)训练的是问题到段落的检索，不是对称句子相似度。[Trans-Encoder](../summaries/Liu%20et%20al.%20-%202021%20-%20Trans-Encoder%20Unsupervised%20sentence-pair%20modelling%20through%20self-%20and%20mutual-distillations.md)把双塔与交互评分器通过蒸馏连接，正说明二者有不同训练与服务角色。

## 关键争论与分歧

- **BERT 之后的提升主要来自架构改动，还是训练更充分**：`RoBERTa` 强烈支持后者至少长期被低估，但 `SpanBERT` 又清楚表明，若下游任务的核心单元是 span，则目标函数设计确实会带来独立收益。更稳妥的判断不是偏向某一边，而是：**训练配方决定基线能到哪里，目标设计决定能力是否对准特定任务结构**。
- **句向量是否能被视为 BERT 预训练的自动副产物**：现有证据更支持否定答案。`Sentence-BERT` 和 `SimCSE` 的对照实验说明，token-level MLM 目标与句级相似度目标不同，额外训练可在所测任务上改善句表示。因此当任务从分类转向检索或匹配时，模型接口已经发生了本质变化，而不是简单换一个 pooling。
- **双向编码器是否已被 decoder-only LLM 淘汰**：如果问题是开放式生成或聊天，主导权确实已经转向 decoder-only；但若问题是 reranking、dense retrieval、分类、抽取、低延迟多语言理解，则 encoder 的双塔、任务头等接口仍有结构上的适配价值；具体效率需同预算测试。这个争论能成立的前提是 **先区分任务接口**；如果不区分接口，结论只会滑向空泛的“LLM 更强”。
- **BERT 家族应按模型名字组织，还是按功能分化组织**：当前知识库更适合后者。因为许多关键节点并未根本改动 backbone，而是改变训练目标、池化方式、对比目标或部署接口。按名字罗列很容易丢失“这些工作究竟在解决哪一层问题”的主线。

### 编码器的定位要靠同任务比较

本库证据足以支持双塔可离线编码、交互评分需重新计算，以及 MLM 不直接优化句间距离；不足以证明编码器在所有分类或检索任务上都比生成模型更快、更准。关于它在 LLM 时代的优势，应把低延迟看作由接口带来的候选优势，并在相同任务、质量阈值和服务预算下验证。多语言迁移也必须按语言数据量和评测集比较，不能从英文 STS 平均成绩推出低资源语言效果。

## 证据基础

- [Devlin et al. - 2019 - BERT Pre-training of deep bidirectional transformers for language understanding](../../wiki/summaries/Devlin%20et%20al.%20-%202019%20-%20BERT%20Pre-training%20of%20deep%20bidirectional%20transformers%20for%20language%20understanding.md)：支撑双向预训练编码器范式的建立。
- [Liu et al. - 2019 - RoBERTa A Robustly Optimized BERT Pretraining Approach](../../wiki/summaries/Liu%20et%20al.%20-%202019%20-%20RoBERTa%20A%20Robustly%20Optimized%20BERT%20Pretraining%20Approach.md)：支撑训练配方对 BERT 家族上限的决定性影响。
- [Joshi et al. - 2020 - Spanbert Improving pre-training by representing and predicting spans](../../wiki/summaries/Joshi%20et%20al.%20-%202020%20-%20Spanbert%20Improving%20pre-training%20by%20representing%20and%20predicting%20spans.md)：支撑围绕 span 级任务结构重写预训练目标的路线。
- [Liu, Lapata - 2020 - Text summarization with pretrained encoders](../../wiki/summaries/Liu,%20Lapata%20-%202020%20-%20Text%20summarization%20with%20pretrained%20encoders.md)：支撑双向编码器向摘要等复杂任务系统外延的能力。
- [Sentence-BERT：孪生编码器句向量（Reimers 与 Gurevych，2019）](../../wiki/summaries/Devlin,%20Liu%20-%202014%20-%20Sentence-BERT%20Sentence%20Embeddings%20using%20Siamese%20BERT-Networks.md)：支撑原始 BERT 不天然等于可直接使用的句向量空间，以及双塔句嵌入路线的必要性。
- [Gao, Yao, Chen - 2021 - SimCSE Simple Contrastive Learning of Sentence Embeddings](../../wiki/summaries/Gao,%20Yao,%20Chen%20-%202021%20-%20SimCSE%20Simple%20Contrastive%20Learning%20of%20Sentence%20Embeddings.md)：支撑以对比学习整理句向量几何的代表路径。
- [Giorgi et al. - 2021 - DeCLUTR Deep contrastive learning for unsupervised textual representations](../../wiki/summaries/Giorgi%20et%20al.%20-%202021%20-%20DeCLUTR%20Deep%20contrastive%20learning%20for%20unsupervised%20textual%20representations.md)：支撑无监督句表示学习也是 BERT 家族中的独立方向。
- [Yan et al. - 2021 - ConSERT A contrastive framework for self-supervised sentence representation transfer](../../wiki/summaries/Yan%20et%20al.%20-%202021%20-%20ConSERT%20A%20contrastive%20framework%20for%20self-supervised%20sentence%20representation%20transfer.md)：支撑句向量支线中自监督对比方法的代表证据。
- [Karpukhin et al. - 2020 - Dense passage retrieval for open-domain question answering](../../wiki/summaries/Karpukhin%20et%20al.%20-%202020%20-%20Dense%20passage%20retrieval%20for%20open-domain%20question%20answering.md)：支撑 BERT 类编码器进一步进入大规模稠密检索接口。
- [Conneau et al. - 2020 - Unsupervised cross-lingual representation learning at scale](../../wiki/summaries/Conneau%20et%20al.%20-%202020%20-%20Unsupervised%20cross-lingual%20representation%20learning%20at%20scale.md)：支撑多语言双向编码器的统一表示路线。
- [Conneau - 2021 - Larger-Scale Transformers for Multilingual Masked Language Modeling](../../wiki/summaries/Conneau%20-%202021%20-%20Larger-Scale%20Transformers%20for%20Multilingual%20Masked%20Language%20Modeling.md)：支撑多语言 MLM 的规模化与平衡问题。
- [Li et al. - 2019 - On the Sentence Embeddings from Pre-trained Language Models](../summaries/Li%20et%20al.%20-%202019%20-%20On%20the%20Sentence%20Embeddings%20from%20Pre-trained%20Language%20Models.md)：补充 BERT-flow 的几何校正与对应任务条件。
- [Su et al. - 2021 - Whitening Sentence Representations for Better Semantics and Faster Retrieval](../summaries/Su%20et%20al.%20-%202021%20-%20Whitening%20Sentence%20Representations%20for%20Better%20Semantics%20and%20Faster%20Retrieval.md)：补充线性白化与降维的区别。
- [Liu et al. - 2021 - Trans-Encoder Unsupervised sentence-pair modelling through self- and mutual-distillations](../summaries/Liu%20et%20al.%20-%202021%20-%20Trans-Encoder%20Unsupervised%20sentence-pair%20modelling%20through%20self-%20and%20mutual-distillations.md)：支撑交互评分与可索引向量接口之间的蒸馏联系。

## 代表页面

- [BERT](../concepts/BERT.md)
- [RoBERTa](../concepts/RoBERTa.md)
- [SpanBERT](../concepts/SpanBERT.md)
- [Sentence-BERT](../concepts/Sentence-BERT.md)
- [SimCSE](../concepts/SimCSE.md)
- [XLM-R](../concepts/XLM-R.md)
- [Dense Retrieval](../concepts/Dense%20Retrieval.md)
- [DPR](../concepts/DPR.md)

## 未解决问题

- 短句 STS 的几何改善何时能迁移到长文、实体和领域外检索？比较页明确了接口，但实验没有建立统一迁移关系。
- 如何在相同预算下分离训练配方与目标结构的贡献？RoBERTa 和 SpanBERT 指向不同改进轴。
- 低资源语言与噪声输入下，双塔、交互模型和小型生成器怎样分工？实际效率与质量还需同任务测试。

## 关联页面

- [传统 NLP](../topics/传统%20NLP.md)
- [LLM 预训练](../topics/LLM%20预训练.md)
- [搜索排序](./搜索排序.md)
- [BERT](../concepts/BERT.md)
- [RoBERTa](../concepts/RoBERTa.md)
- [SpanBERT](../concepts/SpanBERT.md)
- [Sentence-BERT](../concepts/Sentence-BERT.md)
- [SimCSE](../concepts/SimCSE.md)
- [XLM-R](../concepts/XLM-R.md)
- [句向量、稠密召回与交互排序](../comparisons/%E5%8F%A5%E5%90%91%E9%87%8F%E3%80%81%E7%A8%A0%E5%AF%86%E5%8F%AC%E5%9B%9E%E4%B8%8E%E4%BA%A4%E4%BA%92%E6%8E%92%E5%BA%8F.md)：句子相似度、找到相关段落和给候选排序是三个目标。双塔便于离线索引，交互模型能细看词间关系；训练与几何修正则决定向量是否适合对应任务。

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **RAG**：检索增强生成：先找外部材料，再利用这些材料生成回答。
- **decoder**：解码器：根据已有表示产生文字、图像或其他输出。

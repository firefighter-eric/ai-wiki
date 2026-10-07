---
type: topic
status: formal
review_scope: evidence_synthesis
reviewed: 2026-10-07
---
# 传统 NLP

## TL;DR（快速导读）

传统 NLP 的价值在任务结构：词、句子、跨度、依存树和检索候选需要不同预测接口。共享预训练表示并没有消除这些差别，理解它们有助于判断何时用生成模型、编码器或专门推断。

阅读重点：先按问题选择路线，再核对比较条件与证据边界。

## 先用一个问题理解

一个问答系统可先用句向量召回，再用匹配模型排序，最后生成回答。改进其中一步不保证全链正确；阅读本页时可把表示、检索和生成放回各自的位置。

## 页面状态

正式 topic；2026-10-07 复核核心来源并补充方法比较。正文区分论文实验、作者报告和本文综合判断；开放问题表示研究证据的边界。

## 主题定义

本页讨论 **LLM 时代之前形成、且在 LLM 时代仍持续影响方法结构的 NLP 主线**。这里的“传统”不是指纯手工规则或统计时代的全部历史，而是指 **以神经机器翻译、双向编码器、句向量、稠密检索、抽取式/编码器式摘要等为核心的现代 NLP 中间层**。它们通常不以开放式通用生成能力为目标，而以翻译、理解、匹配、召回、排序和结构化预测为主。

因此，本页并不试图覆盖从 `n-gram` 到 CRF 的完整史前谱系，也不把 GPT、PaLM、Llama 一类 decoder-only foundation model 纳入主干。更合适的理解是：这里整理的是 **通向 LLM 时代之前后的“表示学习与任务接口 NLP”主线**，也就是后来很多 RAG、检索增强问答、encoder-decoder 生成系统和 text-to-text 预训练的技术前史。

它与 [BERT类双向Transformer语言模型](./BERT%E7%B1%BB%E5%8F%8C%E5%90%91Transformer%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md) 的区别在于：后者聚焦 BERT 家族本身；本页则把 BERT 视为传统 NLP 后期的中心节点之一，并同时纳入句向量、稠密检索与编码器式摘要等相邻路线。它与 [LLM 预训练](../topics/LLM%20预训练.md) 的边界在于：本页关心 **任务接口与表征结构如何演进**，而不是通用生成模型如何通过规模化训练获得能力。

## 核心问题

- **传统 NLP 是如何从任务专用模型转向统一表征底座的**：这决定了后续为何能出现“一个编码器服务多任务”的工作模式。
- **神经机器翻译如何把序列任务改写成统一条件生成接口**：这决定了后续为什么能把翻译、摘要、问答等任务收敛到 encoder-decoder 或 text-to-text 框架中。
- **哪些问题促使传统 NLP 从 token 表征走向句向量和检索向量**：这背后是任务接口变化，而不是简单的模型更换。
- **为什么检索、排序、摘要、问答虽相邻却不能混成一个主题**：它们共享编码器底座，但解决的结构问题并不相同。
- **传统 NLP 与 LLM 时代的分界应划在哪里**：若边界划得过宽，会把一切都写成“LLM 前史”；划得过窄，又会丢失现代检索和 RAG 的技术根系。

## 主线脉络 / 方法分层

- **神经机器翻译与 seq2seq 接口层**：`Sutskever, Vinyals, Le 2014` 将机器翻译从短语式 SMT pipeline 推向端到端条件生成：一个 `LSTM` 编码源序列，另一个 `LSTM` 按自回归方式生成目标序列。这条线的意义不是 RNN 本身，而是把 variable-length input 到 variable-length output 的任务抽象成 `Seq2Seq` 接口。后来的 attention、Transformer 与 T5 都在不同程度上继承这个接口，同时修正固定向量瓶颈、训练并行性和任务统一范围。
- **统一预训练编码器层**：`Devlin et al. 2019` 的 `BERT` 与 `Liu et al. 2019` 的 `RoBERTa` 共同标志传统 NLP 后期最重要的转折，即从大量任务专用建模转向 **统一预训练编码器底座**。这里的关键不是“Transformer 更强”这么简单，而是大量分类、抽取、问答、摘要任务开始共享同一套表示学习基础设施。这使传统 NLP 从“每个任务一套特征工程”过渡到“一个底座适配多任务”。
- **任务结构感知层**：`Joshi et al. 2020` 的 `SpanBERT` 表明，即使进入统一编码器时代，任务结构仍不会消失。抽取、问答、共指等任务高度依赖 span 单元，因此预训练目标也会围绕 span 重写。这个层次说明，**统一底座并没有消解任务差异，而是把任务差异转移到预训练目标与任务接口上**。
- **句向量与语义匹配层**：`SimCSE` 及相关句表示工作说明，分类型编码器并不天然等于高质量相似度空间。传统 NLP 在这一阶段分化出一条独立的句向量路线，其核心不是让模型“更懂句子”，而是让语义空间 **足够适合检索、聚类、匹配和迁移**。这条线后来直接影响 dense retrieval、reranking 与 RAG 中的表征接口。
- **稠密检索层**：`Karpukhin et al. 2020` 的 DPR 把问答系统的前端从稀疏召回推进到双塔语义检索。这条线在知识史上的价值非常高，因为它说明传统 NLP 后期已经开始把表征学习直接用于 **可索引的大规模召回问题**，为后来的检索增强生成提供了清晰前史。DPR 的出现也意味着“理解”不再只是做分类和抽取，而是成为信息访问系统的一部分。
- **编码器式摘要与任务外延层**：`Liu, Lapata 2020` 以及 `Liu 2019` 的抽取式摘要工作共同说明，传统 NLP 后期并不只停留在理解类任务。编码器底座已经被用来支撑摘要这类更复杂的任务结构，但它采取的仍是 **编码器主导的生成或半生成接口**，而不是 today LLM 式的通用开放生成。它们证明了传统 NLP 在生成任务上的延展能力，同时也暴露出其与 decoder-only 路线的边界。
- **不依赖大规模预训练的反思层**：`Yao et al. 2021` 的 “NLP From Scratch Without Large-Scale Pretraining” 代表一个重要提醒，即传统 NLP 的后期并非所有问题都自动收敛到“大规模预训练越大越好”。这条线提示我们，**传统 NLP 并不是单向走向更大模型**，而是始终存在对效率、任务结构和训练成本的反思。

### 从表示学习进入结构化预测

[word2vec](../summaries/Mikolov%20et%20al.%20-%202013%20-%20Efficient%20estimation%20of%20word%20representations%20in%20vector%20space.md)将词放到可计算相似性的空间，但没有上下文 token 的动态表示；[BERT](../summaries/Devlin%20et%20al.%20-%202019%20-%20BERT%20Pre-training%20of%20deep%20bidirectional%20transformers%20for%20language%20understanding.md)改变表示，任务输出仍需设计。[UniLM](../summaries/Dong%20et%20al.%20-%202019%20-%20Unified%20language%20model%20pre-training%20for%20natural%20language%20understanding%20and%20generation.md)用注意力 mask 统一部分理解与生成目标，[UniLMv2（2020）](../summaries/Bao%20et%20al.%20-%202019%20-%20LMv2%20Pseudo-Masked%20Language%20Models%20for%20Unified%20Language%20Model%20Pre-Training.md)通过 pseudo mask 建模遮蔽片段内部关系。统一指的是训练目标与骨架复用，不表示生成、分类和检索拥有同一评测。

结构化任务对一致性有额外要求。[biaffine parsing](../summaries/Dozat%2C%20Manning%20-%202017%20-%20Deep%20biaffine%20attention%20for%20neural%20dependency%20parsing.md)预测依存头与标签，UAS/LAS 对应不同错误；[二阶 TreeCRF](../summaries/Zhang%2C%20Li%2C%20Zhang%20-%202020%20-%20Efficient%20Second-Order%20TreeCRF%20for%20Neural%20Dependency%20Parsing.md)加入 sibling 因子及树上的全局归一化。句法依存是语言结构，不应直接作为事实知识图谱。[实体与关系联合抽取](../summaries/Bekoulis%20et%20al.%20-%202018%20-%20Joint%20entity%20recognition%20and%20relation%20extraction%20as%20a%20multi-head%20selection%20problem.md)与[Matching the Blanks](../summaries/Soares%20et%20al.%20-%202020%20-%20Matching%20the%20blanks%20Distributional%20similarity%20for%20relation%20learning.md)处理实体及关系表示，仍依赖标注定义、负例和语料条件；[知识图谱综述](../summaries/Xu%20et%20al.%20-%202016%20-%20Review%20on%20knowledge%20graph%20techniques.md)提供上层组织语境，但不是自动抽取正确性的证明。

共指给出了有用的反例对照。[高阶共指（Lee 等，2018）](../summaries/Allen%2C%20Science%20-%202018%20-%20Higher-order%20Coreference%20Resolution%20with%20Coarse-to-fine%20Inference.md)用先行词分布迭代修正跨度表示，[高阶推断复核](../summaries/Xu%2C%20Choi%20-%202020%20-%20Revealing%20the%20Myth%20of%20Higher-Order%20Inference%20in%20Coreference%20Resolution.md)在 SpanBERT 设定下未发现多种高阶机制稳定优于基线。因此结论应是“收益依底座与推断方式”，不是高阶推断永远必要或完全无用。[CoNLL-2012](../summaries/Pradhan%2C%20Moschitti%2C%20Uryupina%20-%202012%20-%20CoNLL-2012%20Shared%20Task%20Modeling%20Multilingual%20Unrestricted%20Coreference%20in%20OntoNotes.md)规定共享任务与语料，[CorefQA](../summaries/Wu%20et%20al.%20-%202020%20-%20CorefQA%20Coreference%20Resolution%20as%20Query-based%20Span%20Prediction.md)再把共指组织成 query-based span prediction。接口改变后仍须回到同一簇与先行词评价。

| 输出单元 | 例子 | 需要守住的约束 |
| --- | --- | --- |
| 标签集合 | NER、层次分类 | 标签定义、多标签与父子结构 |
| 跨度与关系 | 共指、关系抽取 | 候选覆盖、跨度边界、关系一致性 |
| 依存树 | parsing | 树结构、标签与语言协议 |
| 文档摘要 | 抽取/生成 | 事实忠实、覆盖与长程组织 |

[BERT 分类微调](../summaries/Sun%20et%20al.%20-%202019%20-%20How%20to%20Fine-Tune%20BERT%20for%20Text%20Classification.md)、[EDA](../summaries/Wei%2C%20Zou%20-%202019%20-%20EDA%20Easy%20data%20augmentation%20techniques%20for%20boosting%20performance%20on%20text%20classification%20tasks.md)和[R-Drop](../summaries/Liang%20et%20al.%20-%202021%20-%20R-Drop%20Regularized%20Dropout%20for%20Neural%20Networks%20arXiv%202106%20.%2014448v2%20cs%20.%20LG%2029%20Oct%202021.md)分别改变适配、数据增强和训练一致性；这些收益不应当作新语言表示的单一突破。[HPT](../summaries/Wang%20et%20al.%20-%202022%20-%20HPT%20Hierarchy-aware%20Prompt%20Tuning%20for%20Hierarchical%20Text%20Classification.md)依赖 MLM 与标签层级，[多语言 NER 系统](../summaries/Wang%20et%20al.%20-%202022%20-%20DAMO-NLP%20at%20SemEval-2022%20Task%2011%20A%20Knowledge-based%20System%20for%20Multilingual%20Named%20Entity%20Recognition.md)依赖外部知识与任务配置，迁移时要把这些前提带上。

## 关键争论与分歧

- **传统 NLP 的边界到底应划在哪里**：若把它等同于“LLM 之前的一切 NLP”，页面会失去结构；若只把它理解为 CRF、HMM 和词袋时代，它又无法解释为何 BERT、DPR、句向量仍应被视为传统 NLP 的延展。当前更合理的边界是：**把非开放式 foundation model、但已形成统一表示学习底座的一整段现代 NLP 主线纳入本页**。
- **Seq2Seq 应算传统 NLP 还是 LLM 预训练前史**：从今天看，T5、mT5、OFA 等工作都继承了 seq2seq 接口；但从方法史看，Sutskever et al. 2014 首先是在神经机器翻译中证明端到端条件生成可行。因此本页把 RNN seq2seq 作为传统 NLP 的生成式前史，同时在概念页中连接到 LLM 预训练与多模态统一接口。
- **统一编码器是否已经消解任务差异**：现有证据支持否定判断。`SpanBERT`、DPR、摘要模型都表明，统一底座只解决“共享表示”的问题，不解决“任务结构相同”的问题。只要任务的最优单元、评价方式和执行接口不同，特化设计就仍然成立。
- **稠密检索应算传统 NLP，还是应直接纳入 LLM 主题**：从今天的应用语境看，它常被写进 RAG 叙事；但从方法史看，它首先是编码器表征学习与信息检索结合的结果。因此在当前知识库中，把它保留为传统 NLP 的后期分支更能保留历史连续性。
- **生成式任务是否已经把传统 NLP 推向终点**：`Liu, Lapata 2020` 已经表明，传统编码器路线可以外延到摘要等任务；但它的生成能力仍然 strongly 受限于任务结构与模型接口。因此更稳妥的判断是：**传统 NLP 并未被直接终结，而是在开放生成问题上逐步把主导权让给了 decoder-only 路线**。

### 共享模型后，任务证据仍要分别保存

结构化任务比开放续写多了输出约束，但不能由此推导小模型天然更准确。评价还受语料版本、标注体系、候选剪枝和训练预算影响。摘要研究中的 ROUGE 反映文本重合，不能单独证明事实正确；依存解析高分不能替代关系抽取或知识库正确率。本文按任务接口组织，是为了使这种边界在选方案时可见。

## 证据基础

- [Sutskever, Vinyals, Le - 2014 - Sequence to Sequence Learning with Neural Networks](../../wiki/summaries/Sutskever,%20Vinyals,%20Le%20-%202014%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.md)：支撑神经机器翻译与 `Seq2Seq` 条件生成接口的早期成型。
- [Devlin et al. - 2019 - BERT Pre-training of deep bidirectional transformers for language understanding](../../wiki/summaries/Devlin%20et%20al.%20-%202019%20-%20BERT%20Pre-training%20of%20deep%20bidirectional%20transformers%20for%20language%20understanding.md)：支撑统一预训练编码器底座的建立。
- [Liu et al. - 2019 - RoBERTa A Robustly Optimized BERT Pretraining Approach](../../wiki/summaries/Liu%20et%20al.%20-%202019%20-%20RoBERTa%20A%20Robustly%20Optimized%20BERT%20Pretraining%20Approach.md)：支撑训练配方重估在传统 NLP 后期的重要性。
- [Liu - 2019 - Fine-tune BERT for Extractive Summarization](../../wiki/summaries/Liu%20-%202019%20-%20Fine-tune%20BERT%20for%20Extractive%20Summarization.md)：支撑编码器底座向抽取式摘要外延的代表节点。
- [Joshi et al. - 2020 - Spanbert Improving pre-training by representing and predicting spans](../../wiki/summaries/Joshi%20et%20al.%20-%202020%20-%20Spanbert%20Improving%20pre-training%20by%20representing%20and%20predicting%20spans.md)：支撑任务结构感知预训练目标仍然必要。
- [Karpukhin et al. - 2020 - Dense passage retrieval for open-domain question answering](../../wiki/summaries/Karpukhin%20et%20al.%20-%202020%20-%20Dense%20passage%20retrieval%20for%20open-domain%20question%20answering.md)：支撑稠密检索作为传统 NLP 后期的重要分支。
- [Liu, Lapata - 2020 - Text summarization with pretrained encoders](../../wiki/summaries/Liu,%20Lapata%20-%202020%20-%20Text%20summarization%20with%20pretrained%20encoders.md)：支撑预训练编码器向更复杂摘要系统的迁移。
- [Gao, Yao, Chen - 2021 - SimCSE Simple Contrastive Learning of Sentence Embeddings](../../wiki/summaries/Gao,%20Yao,%20Chen%20-%202021%20-%20SimCSE%20Simple%20Contrastive%20Learning%20of%20Sentence%20Embeddings.md)：支撑句向量支线从通用编码器中独立出来。
- [Yao et al. - 2021 - NLP From Scratch Without Large-Scale Pretraining A Simple and Efficient Framework](../../wiki/summaries/Yao%20et%20al.%20-%202021%20-%20NLP%20From%20Scratch%20Without%20Large-Scale%20Pretraining%20A%20Simple%20and%20Efficient%20Framework.md)：支撑对“大规模预训练必然主导一切”这一叙事的反思。
- [Mikolov et al. - 2013 - Efficient estimation of word representations in vector space](../summaries/Mikolov%20et%20al.%20-%202013%20-%20Efficient%20estimation%20of%20word%20representations%20in%20vector%20space.md)：静态词表示与上下文表示的起点。
- [Dong et al. - 2019 - Unified language model pre-training for natural language understanding and generation](../summaries/Dong%20et%20al.%20-%202019%20-%20Unified%20language%20model%20pre-training%20for%20natural%20language%20understanding%20and%20generation.md)：多种注意力 mask 的统一预训练。
- [UniLMv2：Pseudo-Masked Language Models（2020）](../summaries/Bao%20et%20al.%20-%202019%20-%20LMv2%20Pseudo-Masked%20Language%20Models%20for%20Unified%20Language%20Model%20Pre-Training.md)：pseudo mask 与片段内部关系。
- [Dozat, Manning - 2017 - Deep biaffine attention for neural dependency parsing](../summaries/Dozat%2C%20Manning%20-%202017%20-%20Deep%20biaffine%20attention%20for%20neural%20dependency%20parsing.md)：依存头与标签评分。
- [Zhang, Li, Zhang - 2020 - Efficient Second-Order TreeCRF for Neural Dependency Parsing](../summaries/Zhang%2C%20Li%2C%20Zhang%20-%202020%20-%20Efficient%20Second-Order%20TreeCRF%20for%20Neural%20Dependency%20Parsing.md)：二阶树结构推断。
- [Bekoulis et al. - 2018 - Joint entity recognition and relation extraction as a multi-head selection problem](../summaries/Bekoulis%20et%20al.%20-%202018%20-%20Joint%20entity%20recognition%20and%20relation%20extraction%20as%20a%20multi-head%20selection%20problem.md)：实体与关系联合输出。
- [Soares et al. - 2020 - Matching the blanks Distributional similarity for relation learning](../summaries/Soares%20et%20al.%20-%202020%20-%20Matching%20the%20blanks%20Distributional%20similarity%20for%20relation%20learning.md)：关系表示训练。
- [Xu et al. - 2016 - Review on knowledge graph techniques](../summaries/Xu%20et%20al.%20-%202016%20-%20Review%20on%20knowledge%20graph%20techniques.md)：知识图谱邻接语境。
- [Allen, Science - 2018 - Higher-order Coreference Resolution with Coarse-to-fine Inference](../summaries/Allen%2C%20Science%20-%202018%20-%20Higher-order%20Coreference%20Resolution%20with%20Coarse-to-fine%20Inference.md)：高阶共指和候选剪枝。
- [Xu, Choi - 2020 - Revealing the Myth of Higher-Order Inference in Coreference Resolution](../summaries/Xu%2C%20Choi%20-%202020%20-%20Revealing%20the%20Myth%20of%20Higher-Order%20Inference%20in%20Coreference%20Resolution.md)：更强底座下的高阶推断复核。
- [Pradhan, Moschitti, Uryupina - 2012 - CoNLL-2012 Shared Task Modeling Multilingual Unrestricted Coreference in OntoNotes](../summaries/Pradhan%2C%20Moschitti%2C%20Uryupina%20-%202012%20-%20CoNLL-2012%20Shared%20Task%20Modeling%20Multilingual%20Unrestricted%20Coreference%20in%20OntoNotes.md)：CoNLL-2012 共享任务定义。
- [Wu et al. - 2020 - CorefQA Coreference Resolution as Query-based Span Prediction](../summaries/Wu%20et%20al.%20-%202020%20-%20CorefQA%20Coreference%20Resolution%20as%20Query-based%20Span%20Prediction.md)：共指问答接口。
- [Sun et al. - 2019 - How to Fine-Tune BERT for Text Classification](../summaries/Sun%20et%20al.%20-%202019%20-%20How%20to%20Fine-Tune%20BERT%20for%20Text%20Classification.md)：文本分类适配条件。
- [Wei, Zou - 2019 - EDA Easy data augmentation techniques for boosting performance on text classification tasks](../summaries/Wei%2C%20Zou%20-%202019%20-%20EDA%20Easy%20data%20augmentation%20techniques%20for%20boosting%20performance%20on%20text%20classification%20tasks.md)：数据增强在低数据条件下的作用。
- [Liang et al. - 2021 - R-Drop Regularized Dropout for Neural Networks arXiv 2106 . 14448v2 cs . LG 29 Oct 2021](../summaries/Liang%20et%20al.%20-%202021%20-%20R-Drop%20Regularized%20Dropout%20for%20Neural%20Networks%20arXiv%202106%20.%2014448v2%20cs%20.%20LG%2029%20Oct%202021.md)：dropout 一致性正则。
- [Wang et al. - 2022 - HPT Hierarchy-aware Prompt Tuning for Hierarchical Text Classification](../summaries/Wang%20et%20al.%20-%202022%20-%20HPT%20Hierarchy-aware%20Prompt%20Tuning%20for%20Hierarchical%20Text%20Classification.md)：MLM 与层次标签约束。
- [Wang et al. - 2022 - DAMO-NLP at SemEval-2022 Task 11 A Knowledge-based System for Multilingual Named Entity Recognition](../summaries/Wang%20et%20al.%20-%202022%20-%20DAMO-NLP%20at%20SemEval-2022%20Task%2011%20A%20Knowledge-based%20System%20for%20Multilingual%20Named%20Entity%20Recognition.md)：多语言 NER 的知识依赖。
- [Rothe, Narayan, Severyn - 2020 - Leveraging pre-trained checkpoints for sequence generation tasks](../summaries/Rothe%2C%20Narayan%2C%20Severyn%20-%202020%20-%20Leveraging%20pre-trained%20checkpoints%20for%20sequence%20generation%20tasks.md)：编码器/解码器检查点迁移。
- [Pang et al. - 2022 - Long Document Summarization with Top-down and Bottom-up Inference](../summaries/Pang%20et%20al.%20-%202022%20-%20Long%20Document%20Summarization%20with%20Top-down%20and%20Bottom-up%20Inference.md)：长摘要的跨层文档组织。
- [Qiu et al. - 2020 - Pre-trained models for natural language processing A survey](../summaries/Qiu%20et%20al.%20-%202020%20-%20Pre-trained%20models%20for%20natural%20language%20processing%20A%20survey.md)：预训练语言模型方法组织。

## 代表页面

- [BERT](../concepts/BERT.md)
- [Seq2Seq](../concepts/Seq2Seq.md)
- [RoBERTa](../concepts/RoBERTa.md)
- [SpanBERT](../concepts/SpanBERT.md)
- [SimCSE](../concepts/SimCSE.md)
- [Dense Retrieval](../concepts/Dense%20Retrieval.md)
- [DPR](../concepts/DPR.md)

## 未解决问题

- 更强表示是否降低高阶推断价值？共指来源在不同底座上得出不同结果，收益依候选和推断结构。
- 输出约束与开放生成怎样组合才能保持标签、树和事实一致？格式合法仍不等于关系正确。
- 相同质量阈值和服务预算下，编码器、任务头与小型生成器怎样分工？方法接口没有给全部任务统一成本排名。

## 关联页面

- [LLM 预训练](../topics/LLM%20预训练.md)
- [BERT类双向Transformer语言模型](./BERT%E7%B1%BB%E5%8F%8C%E5%90%91Transformer%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)
- [搜索排序](./搜索排序.md)
- [BERT](../concepts/BERT.md)
- [Transformer](../concepts/Transformer.md)
- [Seq2Seq](../concepts/Seq2Seq.md)
- [Sentence-BERT](../concepts/Sentence-BERT.md)
- [Dense Retrieval](../concepts/Dense%20Retrieval.md)
- [句向量、稠密召回与交互排序](../comparisons/%E5%8F%A5%E5%90%91%E9%87%8F%E3%80%81%E7%A8%A0%E5%AF%86%E5%8F%AC%E5%9B%9E%E4%B8%8E%E4%BA%A4%E4%BA%92%E6%8E%92%E5%BA%8F.md)：句子相似度、找到相关段落和给候选排序是三个目标。双塔便于离线索引，交互模型能细看词间关系；训练与几何修正则决定向量是否适合对应任务。
- [语音表示、识别与合成](../concepts/%E8%AF%AD%E9%9F%B3%E8%A1%A8%E7%A4%BA%E3%80%81%E8%AF%86%E5%88%AB%E4%B8%8E%E5%90%88%E6%88%90.md)：语音识别把音频变文字，TTS 把文字变音频，声码器把声学表示变波形。自监督表示和音视频输入可以帮助其中部分阶段；它们的质量与训练条件不能互相替代。

## 这里的术语是什么意思

- **RAG**：检索增强生成：先找外部材料，再利用这些材料生成回答。
- **decoder**：解码器：根据已有表示产生文字、图像或其他输出。

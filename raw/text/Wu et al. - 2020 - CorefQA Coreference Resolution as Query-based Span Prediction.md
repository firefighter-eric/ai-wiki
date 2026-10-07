# Wu et al. - 2020 - CorefQA Coreference Resolution as Query-based Span Prediction

- Source HTML: `raw/html/Wu et al. - 2020 - CorefQA Coreference Resolution as Query-based Span Prediction.html`
- Source SHA256: `8a6aa32ee926c512ea2a7abc06a73d3cdf7098d3f1548b71ff8cabf2f6ca2a51`
- Source URL: https://ar5iv.labs.arxiv.org/html/1911.01746
- Generated from: `scripts/fetch_web_text.py`
- Extraction: `structured-html-v2` (headings, links, MathML/TeX and tables; figures require visual review)

## Extracted Text

<a id="source-section-0"></a>

<a id="source-section-1"></a>

# CorefQA: Coreference Resolution as Query-based Span Prediction


Wei Wu♣, Fei Wang♣,
Arianna Yuan◆♣,
Fei Wu♠ and Jiwei Li♠♣

♠ Department of Computer Science and Technology, Zhejiang University

◆ Computer Science Department, Stanford University

♣ ShannonAI

xfyuan@stanford.edu, wufei@zju.edu.cn

{wei_wu, fei_wang,jiwei_li}@shannonai.com


<a id="source-section-2"></a>

###### Abstract


In this paper, we present CorefQA, an accurate and extensible approach for the coreference resolution task.
We formulate the problem as a span prediction task, like in question answering:
A query is generated for each candidate mention using its surrounding context, and a span prediction module is employed to extract the text spans of the coreferences within the document using the generated query.
This formulation comes with the following key advantages:
(1) The span prediction strategy provides the flexibility of retrieving mentions left out at the mention proposal stage;
(2) In the question answering framework, encoding the mention and its context explicitly in a query makes it possible to have a deep and thorough examination of cues embedded in the context of coreferent mentions; and
(3) A plethora of existing question answering datasets can be used for data augmentation to improve the model’s generalization capability.
Experiments demonstrate significant performance boost over previous models, with 83.1 (+3.5) F1 score on the CoNLL-2012 benchmark and 87.5 (+2.5) F1 score on the GAP benchmark.
111[https://github.com/ShannonAI/CorefQA](https://github.com/ShannonAI/CorefQA)


<a id="source-section-3"></a>

## 1 Introduction


Recent coreference resolution systems (Lee et al., [2017](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib20), [2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib21); Zhang et al., [2018a](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib43); Kantor and Globerson, [2019](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib17)) consider all text spans in a document as potential mentions and learn to find an antecedent for each possible mention. There are two key issues with this paradigm, in terms of task formalization and the algorithm.


At the task formalization level, mentions left out at the mention proposal stage can never be recovered since the downstream module only operates on the proposed mentions. Existing models often suffer from mention proposal (Zhang et al., [2018a](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib43)). The coreference datasets can only provide a weak signal for spans that correspond to entity mentions because singleton mentions are not explicitly labeled.
Due to the inferiority of the mention proposal model, it would be favorable if a coreference framework had a mechanism to retrieve left-out mentions.


Original Passage
In addition , many people were poisoned when toxic gas was released. They were poisoned and did not know how to protect themselves against the poison.
Converted Questions
Q1: Who were poisoned when toxic gas was released?
A1: [They, themselves]
Q2: What was released when many people were poisoned?
A2: [the poison]
Q3: Who were poisoned and did not know how to protect themselves against the poison?
A3: [many people, themselves]
Q4: Whom did they not know how to protect against the poison?
A4: [many people, They]
Q5: They were poisoned and did not know how to protect themselves against what?
A5: [toxic gas]


Figure 1: An illustration of the paradigm shift from coreference resolution to query-based span prediction. Spans with the same color represent coreferent mentions. Note that we use a more direct strategy to generate the questions based on the mentions.


At the algorithm level, existing end-to-end methods (Lee et al., [2017](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib20), [2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib21); Zhang et al., [2018a](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib43)) score each pair of mentions only based on mention representations from the output layer of a contextualization model. This means that the model lacks the connection between mentions and their contexts. Semantic matching operations between two mentions (and their contexts) are performed only at the output layer and are relatively superficial. Therefore it is hard for their models to capture all the lexical, semantic and syntactic cues in the context.


To alleviate these issues, we propose CorefQA, a new approach that formulates the coreference resolution problem as a span prediction task, akin to the question answering setting.
A query is generated for each candidate mention using its surrounding context, and a span prediction module is further employed to extract the text spans of the coreferences within the document using the generated query. Some concrete examples are shown in Figure [1](https://ar5iv.labs.arxiv.org/html/1911.01746#S1.F1). 222This is an illustration of the question formulation. The actual operation is described in Section [3.4](https://ar5iv.labs.arxiv.org/html/1911.01746#S3.SS4).


This formulation provides benefits at both the task formulation level and the algorithm level.
At the task formulation level, since left-out mentions can still be retrieved at the span prediction stage, the negative effect of undetected mentions is significantly alleviated. At the algorithm level, by generating a query for each candidate mention using its surrounding context, the CorefQA model explicitly considers the surrounding context of the target mentions, the influence of which will later be propagated to each input word using the self-attention mechanism.
Additionally, unlike existing end-to-end methods (Lee et al., [2017](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib20), [2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib21); Zhang et al., [2018a](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib43)), where the interactions between two mentions are only superficially modeled at the output layer of contextualization, span prediction requires a more thorough and deeper examination of the lexical, semantic and syntactic cues within the context, which will potentially lead to better performance.


Moreover, the proposed question answering formulation allows us to take advantage of existing question answering datasets. Coreference annotation is expensive, cumbersome and often requires linguistic expertise from annotators. Under the proposed formulation, the coreference resolution has the same format as the existing question answering datasets (Rajpurkar et al., [2016a](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib35), [2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib34); Dasigi et al., [2019a](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib7)).
Those datasets can thus readily be used for data augmentation. We show that pre-training on existing question answering datasets improves the model’s generalization and transferability, leading to additional performance boost.


Experiments show that the proposed framework significantly outperforms previous models on two widely-used datasets. Specifically, we achieve new state-of-the-art scores of 83.1 (+3.5) on the CoNLL-2012 benchmark and 87.5 (+2.5) on the GAP benchmark.


<a id="source-section-4"></a>

## 2 Related Work


<a id="source-section-5"></a>

### 2.1 Coreference Resolution


Coreference resolution is a fundamental problem in natural language processing and is considered as a good test of machine intelligence (Morgenstern et al., [2016](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib30)). Neural network models have shown promising results over the years. Earlier neural-based models (Wiseman et al., [2016](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib41); Clark and Manning, [2015](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib5), [2016](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib6)) rely on parsers and hand-engineered mention proposal algorithms.
Recent work (Lee et al., [2017](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib20), [2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib21); Kantor and Globerson, [2019](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib17)) tackled the problem in an end-to-end fashion by jointly detecting mentions and predicting coreferences. Based on how entity-level information is incorporated, they can be further categorized as (1) entity-level models Björkelund and Kuhn ([2014](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib3)); Clark and Manning ([2015](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib5), [2016](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib6)); Wiseman et al. ([2016](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib41)) that directly model the representation of real-world entities and (2) mention-ranking models (Durrett and Klein, [2013](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib10); Wiseman et al., [2015](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib42); Lee et al., [2017](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib20)) that learn to select the antecedent of each anaphoric mention.
Our CorefQA model is essentially a mention-ranking model, but we identify coreference using question answering.


[图片：Refer to caption]


Figure 2: The overall architecture of our CorefQA model. The input passage is first fed into the Mention Proposal Module [3.3](https://ar5iv.labs.arxiv.org/html/1911.01746#S3.SS3) to obtain candidate mentions. Then the Mention Linking Module [3.4](https://ar5iv.labs.arxiv.org/html/1911.01746#S3.SS4) is used to extract coreferent mentions from the passage for each proposed mention. The coreference clusters are obtained using the scores produced in the above two stages.


<a id="source-section-6"></a>

### 2.2 Formalizing NLP Tasks as question answering


Machine reading comprehension is a general and extensible task form. Many tasks in natural language processing can be framed as reading comprehension while abstracting away the task-specific modeling constraints.


McCann et al. ([2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib29)) introduced the decaNLP challenge, which converts a set of 10 core tasks in NLP to reading comprehension. He et al. ([2015](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib13)) showed that semantic role labeling annotations could be solicited by using question-answer pairs to represent the predicate-argument structure. Levy et al. ([2017](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib22)) reduced relation extraction to answering simple reading comprehension questions, yielding models that generalize better in the zero-shot setting. Li et al. ([2019a](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib25), [b](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib26)) cast the tasks of named entity extraction and relation extraction as a reading comprehension problem. In parallel to our work, Aralikatte et al. ([2019](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib1)) converted coreference and ellipsis resolution in a question answering format, and showed the benefits of training joint models for these tasks. Their models are built under the assumption that gold mentions are provided at inference time, whereas our model does not need that assumption – it jointly trains the mention proposal model and the coreference resolution model in an end-to-end manner.
Chada ([2019](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib4))
proposed an extractive QA model for the resolution of ambiguous pronouns,
and showed better results on the GAP dataset by only fine-tuning the pre-trained BERT model.


<a id="source-section-7"></a>

### 2.3 Data Augmentation


Data augmentation is a strategy that enables practitioners to significantly increase the diversity of data available for training models. Data augmentation techniques have been explored in various fields such as question answering (Talmor and Berant, [2019](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib37)), text classification (Kobayashi, [2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib19)) and dialogue language understanding (Hou et al., [2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib14)). In coreference resolution, Zhao et al. ([2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib46)); Emami et al. ([2019](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib11)); Zhao et al. ([2019](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib45)) focused on debiasing the gender bias problem; Aralikatte et al. ([2019](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib1)) explored the effectiveness of joint modeling of ellipsis and coreference resolution. To the best of our knowledge, we are the first to use existing question answering datasets as data augmentation for coreference resolution.


<a id="source-section-8"></a>

## 3 Model


In this section, we describe our CorefQA model in detail. The overall architecture is illustrated in Figure [2](https://ar5iv.labs.arxiv.org/html/1911.01746#S2.F2).


<a id="source-section-9"></a>

### 3.1 Notations


Given a sequence of input tokens $X=\{x_{1},x_{2},...,x_{n}\}$ in a document, where $n$ denotes the length of the document.
$N=n*(n+1)/2$ denotes the number of all possible text spans in $X$. Let $e_{i}$ denotes the $i$-th span representation $1\leq i\leq N$, with the start index first(i) and the end index last(i).
$e_{i}=\{x_{\textsc{first}(i)},x_{\textsc{first}(i)+1},...,x_{\textsc{last}(i)-1},x_{\textsc{last}(i)}\}$.
The task of coreference resolution is to determine the antecedents for all possible spans.
If a candidate span $e_{i}$ does not represent an entity mention or is not coreferent with any other mentions, a dummy token $\epsilon$ is assigned as its antecedent. The linking between all possible spans $e$ defines the final clustering.


<a id="source-section-10"></a>

### 3.2 Input Representations


We use the SpanBERT model 333[https://github.com/facebookresearch/SpanBERT](https://github.com/facebookresearch/SpanBERT) to obtain input representations following Joshi et al. ([2019a](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib15)). Each token $x_{i}$ is associated with a SpanBERT representation $\bm{x}_{i}$.
Since the speaker information is indispensable for coreference resolution, previous methods Wiseman et al. ([2016](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib41)); Lee et al. ([2017](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib20)); Joshi et al. ([2019a](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib15)) usually convert the speaker information into binary features indicating whether two mentions are from the same speaker. However, we use a straightforward strategy that directly concatenates the speaker’s name with the corresponding utterance.
This strategy is inspired by recent research in personalized dialogue modeling that use persona information to represent speakers (Li et al., [2016](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib24); Zhang et al., [2018b](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib44); Mazaré et al., [2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib28)).
In subsection [5.2](https://ar5iv.labs.arxiv.org/html/1911.01746#S5.SS2), we will empirically demonstrate its superiority over the feature-based method in Lee et al. ([2017](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib20)).


To fit long documents into SpanBERT, we use a sliding-window approach that creates a $T$-sized segment after every $T$/2 tokens. Segments are then passed to the SpanBERT encoder independently. The final token representations are derived by taking the token representations with maximum context.


<a id="source-section-11"></a>

### 3.3 Mention Proposal


Similar to Lee et al. ([2017](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib20)), our model considers all spans up to a maximum length $L$ as potential mentions. To improve computational efficiency, we further prune the candidate spans greedily during both training and evaluation. To do so, the mention score of each candidate span consists of three parts:
(1) $\bm{x}_{\textsc{first}(i)}$ is the start of a span;
(2) $\bm{x}_{\textsc{last}(i)}$ is the end of a span; and
(3) $\bm{x}_{\textsc{first}(i)}$ and $\bm{x}_{\textsc{last}(i)}$ form a valid span.
The third part (i.e., (3)) is necessary because each sentence can contain multiple spans.
The first part is computed by feeding $\bm{x}_{\textsc{first}(i)}$ into a feed-forward layer:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>s_{\text{m}}(\bm{x}_{\textsc{first}(i)})=\textsc{ffnn}_{\text{}}([\bm{x}_{\textsc{first}(i)}])<br>$$ | | (1) |


Similarly,
the first part is computed by feeding $\bm{x}_{\textsc{last}(i)}$ into a feed-forward layer:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>s_{\text{m}}(\bm{x}_{\textsc{last}(i)})=\textsc{ffnn}_{\text{}}([\bm{x}_{\textsc{last}(i)}])<br>$$ | | (2) |


The third part
computed by feeding the
concatenation of $\bm{x}_{\textsc{first}(i)}$ and $\bm{x}_{\textsc{last}(i)}$ into
into a feed-forward layer:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>s_{\text{m}}(\bm{x}_{\textsc{first}(i)},\bm{x}_{\textsc{last}(i)})=\textsc{ffnn}_{\text{m}}([\bm{x}_{\textsc{first}(i)},\bm{x}_{\textsc{last}(i)}])<br>$$ | | (3) |


$\textsc{ffnn}_{\text{}}($) denotes the feed-forward neural network that computes a nonlinear mapping from the input vector to the mention score.
The three involved $\textsc{ffnn}_{\text{}}($) use separate sets of parameters.
The overall score for span $i$ being a mention is the average of the three parts:


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | $\displaystyle s_{m}(i)=$ | $\displaystyle[s_{\text{m}}(\bm{x}_{\textsc{first}(i)})+s_{\text{m}}(\bm{x}_{\textsc{end}(i)})$ | | (4) [rowspan=2] |
| | $\displaystyle+$ | $\displaystyle s_{\text{m}}(\bm{x}_{\textsc{first}(i)},\bm{x}_{\textsc{last}(i)})]/3$ | | |


We only keep up to $\lambda n$ (where $n$ is the document length) spans with the highest mention scores.


<a id="source-section-12"></a>

#### Mention Proposal Pretraining


It is crucial that the mention proposal model is pretrained. Otherwise, most of the proposed mentions that are fed to the linking stage are invalid mentions.
The mention proposal model is pretrained by jointly training three binary classification models: (1) whether
$\bm{x}_{\textsc{first}(i)}$ is the start of a span;
(2) whether $\bm{x}_{\textsc{last}(i)}$ is the end of a span; and
(3) whether $\bm{x}_{\textsc{first}(i)}$ and $\bm{x}_{\textsc{last}(i)}$ should be combined.
This leads to the objective of the mention proposal model as follows:


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | $\displaystyle\text{Loss(m)}=$ | $\displaystyle\text{sigmoid}(s_{\text{m}}(\bm{x}_{\textsc{first}(i)}))$ | | (5) [rowspan=3] |
| | $\displaystyle+$ | $\displaystyle\text{sigmoid}(s_{\text{m}}(\bm{x}_{\textsc{end}(i)}))$ | | |
| | $\displaystyle+$ | $\displaystyle\text{sigmoid}(s_{\text{m}}(\bm{x}_{\textsc{first}(i)},\bm{x}_{\textsc{last}(i)}))$ | | |


<a id="source-section-13"></a>

### 3.4 Mention Linking as Span Prediction


Given a mention $e_{i}$ proposed by the mention proposal network, the role of the mention linking network is to give a score $s_{a}(i,j)$ for any text span $e_{j}$, indicating whether $e_{i}$ and $e_{j}$ are coreferent.
We propose to use the question answering framework as the backbone to compute $s_{a}(i,j)$. It operates on the triplet {context (X), query (q), answers (a)}.
The context $X$ is the input document.
The query $q(e_{i})$ is constructed as follows: given $e_{i}$, we use the sentence that $e_{i}$ resides in as the query, with the minor modification that we encapsulates $e_{i}$ with special tokens $<mention></mention>$ .
The answers $a$ are the coreferent mentions of $e_{i}$.
A query $i$ is considered unanswerable in the following scenarios: (1) the candidate span $e_{i}$ does not represent an entity mention or (2) the candidate span $e_{i}$ represents an entity mention but is not coreferent with any other mentions in $X$.


Following Devlin et al. ([2019](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib9)), we represent the input query and the context as a single packed sequence.
The for any
span $j=[{\textsc{first}(j)},...,{\textsc{last}(j)}]$,
we first compute the score of $i$ being the answer for query $q(e_{i})$, denoted by $s_{a}(j|i)$.
Let $\bm{x}_{\textsc{first}(j)}|i$ and $\bm{x}_{\textsc{last}(j)}|i$ respectively denote the representations
for first(j) and last(j) from BERT, where $q(e_{i})$ is used as query concatenated to the context.
$s_{a}(j|i)$
is computed by feeding the first and the last of its constituent token representations (i.e., $\bm{x}_{\textsc{first}(j)}|i$ and $\bm{x}_{\textsc{last}(j)}|i$ ) into a feed-forward layer:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>s_{a}(j\|i)=\textsc{ffnn}_{j\|i}{[\bm{x}_{\textsc{first}(j)\|i},\bm{x}_{\textsc{last}(j)\|i}]}<br>$$ | | (6) |


$\textsc{ffnn}_{j|i}$ denotes the feed-forward neural network that computes a nonlinear mapping from the input vector to the mention score.
Comparing Eq.[8](https://ar5iv.labs.arxiv.org/html/1911.01746#S3.E8) with Eq.[3](https://ar5iv.labs.arxiv.org/html/1911.01746#S3.E3), we can observe their relatedness and difference:
both of the equations compute scores for a span. But for Eq.[8](https://ar5iv.labs.arxiv.org/html/1911.01746#S3.E8), the query $q(e_{i})$ is additionally used to check whether span $j$ is the answer for $q(e_{i})$.


A closer look at Eq.[8](https://ar5iv.labs.arxiv.org/html/1911.01746#S3.E8) reveals that it only models the uni-directional coreference relation from $e_{i}$ to $e_{j}$, i.e., $e_{j}$ is the answer for query $q(e_{i})$.
This is suboptimal since if $e_{i}$ is a coreference mention of $e_{j}$, then $e_{j}$ should also be the coreference mention $e_{i}$.
We thus need to optimize the bi-directional relation between $e_{i}$ and $e_{j}$.444This bidirectional relationship is actually referred to as mutual dependency and has shown to benefit a wide range of NLP tasks such as machine translation Hassan et al. ([2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib12)) or dialogue generation Li et al. ([2015](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib23)).
The final score $s_{a}(i,j)$ is thus given as follows:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>s_{a}(i,j)=\frac{1}{2}(s_{a}(j\|i)+s_{a}(i\|j))<br>$$ | | (7) |


$s_{a}(i|j)$ can be computed in the same way as $s_{a}(j|i)$, in which $q(e_{i})$ is used as the query:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>s_{a}(i\|j)=\textsc{ffnn}_{i\|j}{[\bm{x}_{\textsc{first}(i)\|j},\bm{x}_{\textsc{last}(i)\|j}]}<br>$$ | | (8) |


where $\bm{x}_{\textsc{first}(i)}|j$ and $\bm{x}_{\textsc{last}(i)}|j$ respectively denote the representations
for first(i) and last(i) from BERT, where $q(e_{j})$ is used as query concatenated to the context.


For a pair of text span $e_{i}$ and $e_{j}$, the premises for them being coreferent mentions are (1) they are mentions and (2) they are coreferent.
This makes the overall score $s(i,j)$ for $e_{i}$ and $e_{j}$ the combination of Eq.[3](https://ar5iv.labs.arxiv.org/html/1911.01746#S3.E3) and Eq.[7](https://ar5iv.labs.arxiv.org/html/1911.01746#S3.E7):


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>s(i,j)=\lambda[s_{\text{m}}(i)+s_{\text{m}}(j)]+(1-\lambda)s_{a}(i,j)<br>$$ | | (9) |


$\lambda$ is the hyperparameter to control
the tradeoff between mention proposal and
mention linking.


<a id="source-section-14"></a>

### 3.5 Antecedent Pruning


Given a document $X$ with length $n$ and the number of spans $O(n^{2})$, the computation of Eq.[9](https://ar5iv.labs.arxiv.org/html/1911.01746#S3.E9) for all mention pairs is intractable with the complexity of $O(n^{4})$.
Given an extracted mention $e_{i}$, the computation of Eq.[9](https://ar5iv.labs.arxiv.org/html/1911.01746#S3.E9) for $(e_{i},e_{j})$ regarding all $e_{j}$ is still extremely intensive since the computation of the backward span prediction score $s_{a}(i|j)$ requires running question answering models on all query $q(e_{j})$.
A further pruning procedure is thus needed: For each query $q(e_{i})$, we collect $C$ span candidates only based on the $s_{a}(j|i)$ scores, and then use Eq. [9](https://ar5iv.labs.arxiv.org/html/1911.01746#S3.E9)
to compute the overall scores.


<a id="source-section-15"></a>

### 3.6 Training


For each mention $e_{i}$ proposed by the mention proposal network, it is associated with $C$ potential spans proposed by the mention linking network based on $s(j|i)$,
we aim to optimize the marginal log-likelihood of all correct antecedents implied by the gold clustering.
Following Lee et al. ([2017](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib20)), we append a dummy token $\epsilon$ to the $C$ candidates. The model will output it if none of the $C$ span candidates is coreferent with $e_{i}$.
For each mention $e_{i}$, the model learns a distribution $P(\cdot)$ over all possible antecedent spans $e_{j}$ based on the global score $s(i,j)$ from Eq. [9](https://ar5iv.labs.arxiv.org/html/1911.01746#S3.E9):


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>P(e_{j})=\frac{e^{s(i,j)}}{\sum_{j^{\prime}\in C}e^{s(i,j^{\prime})}}<br>$$ | | (10) |


The mention proposal module and the mention linking module are jointly trained in an end-to-end fashion using training signals from Eq.[10](https://ar5iv.labs.arxiv.org/html/1911.01746#S3.E10), with the
SpanBERT parameters shared.


<a id="source-section-16"></a>

### 3.7 Inference


Given an input document, we can obtain an undirected graph using the overall score, each node of which represents a candidate mention from either the mention proposal module or the mention linking module.
We prune the graph by keeping the edge whose weight is the largest for each node based on Eq.[10](https://ar5iv.labs.arxiv.org/html/1911.01746#S3.E10). Nodes whose closest neighbor is the dummy token $\epsilon$ are abandoned. Therefore, the mention clusters can be decoded from the graph.


<a id="source-section-17"></a>

### 3.8 Data Augmentation using Question Answering Datasets


We hypothesize that the reasoning (such as synonymy, world knowledge, syntactic variation, and multiple sentence reasoning) required to answer the questions are also indispensable for coreference resolution.
Annotated question answering datasets are usually significantly larger than the coreference datasets due to the high linguistic expertise required for the latter.
Under the proposed QA formulation, coreference resolution has the same format as the
existing question answering datasets (Rajpurkar et al., [2016a](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib35), [2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib34); Dasigi et al., [2019a](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib7)). In this way, they can readily be used for data augmentation. We thus propose to pre-train the mention linking network on the Quoref dataset Dasigi et al. ([2019b](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib8)),
and the SQuAD dataset Rajpurkar et al. ([2016b](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib36)).


<a id="source-section-18"></a>

### 3.9 Summary and Discussion


Comparing with existing models Lee et al. ([2017](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib20), [2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib21)); Joshi et al. ([2019b](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib16)), the proposed question answering formalization has the flexibility of retrieving mentions left out at the mention proposal stage. However, since we still have the mention proposal model, we need to know in which situation missed mentions could be retrieved and in which situation they cannot.
We use the example in Figure [1](https://ar5iv.labs.arxiv.org/html/1911.01746#S1.F1) as an illustration, in which {many people, They, themselves} are coreferent mentions: If partial mentions are missed by the mention proposal model, e.g., many people and They, they can still be retrieved in the mention linking stage when the not-missed mention (i.e., themselves) is used as query. But, if all the mentions within the cluster are missed, none of them can be used for query construction, which means they all will be irreversibly left out.
Given the fact that the proposal mention network proposes a significant number of mentions, the chance that mentions within a mention cluster are all missed is relatively low (which exponentially decreases as the number of entities increases).
This explains the superiority (though far from perfect) of the proposed model. However, how to completely remove the mention proposal network remains a problem in the field of coreference resolution.


<a id="source-section-19"></a>

## 4 Experiments


<a id="source-section-20"></a>

### 4.1 Implementation Details


The special tokens used to denote the speaker’s name ($<speaker></speaker>$) and the special tokens used to denote the queried mentions ($<mention></mention>$) are initialized by randomly taking the unused tokens from the SpanBERT vocabulary. The sliding window size $T$ = 512, and the mention keep ratio $\lambda$ = 0.2. The maximum length $L$ for mention proposal = 10 and the maximum number of antecedents kept for each mention $C$ = 50. The SpanBERT parameters are updated by the Adam optimizer (Kingma and Ba, [2015](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib18)) with initial learning rate $1\times 10^{-5}$ and the task parameters are updated by the Range optimizer 555[https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer) with initial learning rate $2\times 10^{-4}$.


<a id="source-section-21"></a>

### 4.2 Baselines


We compare the CorefQA model with previous neural models that are trained end-to-end:


- •


e2e-coref (Lee et al., [2017](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib20)) is the first end-to-end coreference system that learns which spans are entity mentions and how to best cluster them jointly. Their token representations are built upon the GLoVe (Pennington et al., [2014](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib31)) and Turian (Turian et al., [2010](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib38)) embeddings.


- •


c2f-coref + ELMo (Lee et al., [2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib21)) extends Lee et al. ([2017](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib20)) by combining a coarse-to-fine pruning with a higher-order inference mechanism. Their representations are built upon ELMo embeddings (Peters et al., [2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib32)).


- •


c2f-coref + BERT-large(Joshi et al., [2019b](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib16)) builds the c2f-coref system on top of BERT (Devlin et al., [2019](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib9)) token representations.


- •


EE + BERT-large (Kantor and Globerson, [2019](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib17)) represents each mention in a cluster via an approximation of the sum of all mentions in the cluster.


- •


c2f-coref + SpanBERT-large (Joshi et al., [2019a](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib15)) focuses on pre-training span representations to better represent and predict spans of text.


| | MUC [colspan=3] | $\text{B}^{3}$ [colspan=3] | $\text{CEAF}_{\phi_{4}}$ [colspan=3] | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | P | R | F1 | P | R | F1 | P | R | F1 | Avg. F1 |
| e2e-coref(Lee et al., [2017](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib20)) | 78.4 | 73.4 | 75.8 | 68.6 | 61.8 | 65.0 | 62.7 | 59.0 | 60.8 | 67.2 |
| c2f-coref + ELMo (Lee et al., [2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib21)) | 81.4 | 79.5 | 80.4 | 72.2 | 69.5 | 70.8 | 68.2 | 67.1 | 67.6 | 73.0 |
| EE + BERT-large (Kantor and Globerson, [2019](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib17)) | 82.6 | 84.1 | 83.4 | 73.3 | 76.2 | 74.7 | 72.4 | 71.1 | 71.8 | 76.6 |
| c2f-coref + BERT-large (Joshi et al., [2019b](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib16)) | 84.7 | 82.4 | 83.5 | 76.5 | 74.0 | 75.3 | 74.1 | 69.8 | 71.9 | 76.9 |
| c2f-coref + SpanBERT-large (Joshi et al., [2019a](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib15)) | 85.8 | 84.8 | 85.3 | 78.3 | 77.9 | 78.1 | 76.4 | 74.2 | 75.3 | 79.6 |
| CorefQA + SpanBERT-base | 85.2 | 87.4 | 86.3 | 78.7 | 76.5 | 77.6 | 76.0 | 75.6 | 75.8 | 79.9 (+0.3) |
| CorefQA + SpanBERT-large | 88.6 | 87.4 | 88.0 | 82.4 | 82.0 | 82.2 | 79.9 | 78.3 | 79.1 | 83.1 (+3.5) |


Table 1: Evaluation results on the English CoNLL-2012 shared task. The average F1 of MUC, $\text{B}^{3}$, and $\text{CEAF}_{\phi_{4}}$ is the main evaluation metric. Ensemble models are not included in the table for a fair comparison.


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| Model | M | F | B | O |
| e2e-coref | 67.2 | 62.2 | 0.92 | 64.7 |
| c2f-coref + ELMo | 75.8 | 71.1 | 0.94 | 73.5 |
| c2f-coref + BERT-large | 86.9 | 83.0 | 0.95 | 85.0 |
| c2f-coref + SpanBERT-large | 88.8 | 84.9 | 0.96 | 86.8 |
| CorefQA + SpanBERT-large | 88.9 | 86.1 | 0.97 | 87.5 |


Table 2: CorefQA achieves the state-of-the-art performance on all metrics including F1 scores on Masculine and Feminine examples, a Bias factor (F / M) and the Overall F1 score.


<a id="source-section-22"></a>

### 4.3 Results on CoNLL-2012 Shared Task


The English data of CoNLL-2012 shared task (Pradhan et al., [2012](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib33)) contains 2,802/343/348 train/development/test documents in 7 different genres. The main evaluation is the average of three metrics – MUC (Vilain et al., [1995](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib39)), $\text{B}^{3}$ (Bagga and Baldwin, [1998](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib2)), and $\text{CEAF}_{\phi_{4}}$ (Luo, [2005](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib27)) on the test set according to the official CoNLL-2012 evaluation scripts 666[http://conll.cemantix.org/2012/software.html](http://conll.cemantix.org/2012/software.html).


We compare the CorefQA model with several baseline models in Table [1](https://ar5iv.labs.arxiv.org/html/1911.01746#S4.T1). Our CorefQA system achieves a huge performance boost over existing systems: With SpanBERT-base, it achieves an F1 score of 79.9, which already outperforms the previous SOTA model using SpanBERT-large by 0.3. With SpanBERT-large, it achieves an F1 score of 83.1, with a 3.5 performance boost over the previous SOTA system.


<a id="source-section-23"></a>

### 4.4 Results on GAP


The GAP dataset (Webster et al., [2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib40)) is a gender-balanced dataset that targets the challenges of resolving naturally occurring ambiguous pronouns. It comprises 8,908 coreference-labeled pairs of (ambiguous pronoun, antecedent name) sampled from Wikipedia.


We follow the protocols in Webster et al. ([2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib40)); Joshi et al. ([2019b](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib16)) and use the off-the-shelf resolver trained on the CoNLL-2012 dataset to get the performance of the GAP dataset. Table [2](https://ar5iv.labs.arxiv.org/html/1911.01746#S4.T2) presents the results. We can see that the proposed CorefQA model achieves state-of-the-art performance on all metrics on the GAP dataset.


<a id="source-section-24"></a>

## 5 Ablation Study and Analysis


| | Avg. F1 | $\Delta$ |
| --- | --- | --- |
| CorefQA | 83.4 | |
| $-\text{--}$ SpanBERT | 79.6 | -3.8 |
| $-\text{--}$ Mention Proposal Pre-training | 75.9 | -7.5 |
| $-\text{--}$ Question Answering | 75.0 | -8.4 |
| $-\text{--}$ Quoref Pre-training | 82.7 | -0.7 |
| $-\text{--}$ Squad Pre-training | 83.1 | -0.3 |


Table 3: Ablation studies on the CoNLL-2012 development set. SpanBERT token representations, the mention-proposal pre-training, and the question answering pre-training all contribute significantly to the good performance of the full model.


We perform comprehensive ablation studies and analyses on the CoNLL-2012 development dataset. Results are shown in Table [3](https://ar5iv.labs.arxiv.org/html/1911.01746#S5.T3).


<a id="source-section-25"></a>

### 5.1 Effects of Different Modules in the Proposed Framework


<a id="source-section-26"></a>

#### Effect of SpanBERT


Replacing SpanBERT with vanilla BERT leads to a 3.5 F1 degradation. This verifies the importance of span-level pre-training for coreference resolution and is consistent with previous findings (Joshi et al., [2019a](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib15)).


<a id="source-section-27"></a>

#### Effect of Pre-training Mention Proposal Network


Skipping the pre-training of the mention proposal network using golden mentions results in a 7.2 F1 degradation, which is in line with our expectation.
A randomly initialized mention proposal model implies that mentions are randomly selected. Randomly selected mentions will mostly be transformed to unanswerable queries. This makes it hard for the question answering model to learn at the initial training stage, leading to inferior performance.


<a id="source-section-28"></a>

#### Effect of QA pre-training on the augmented datasets


One of the most valuable strengths of converting anaphora resolution to question answering is that existing QA datasets can be readily used for data augmentation purposes. We see a contribution of 0.7 F1 from pre-training on the Quoref dataset (Dasigi et al., [2019a](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib7)) and a contribution of 0.3 F1 from pre-training on the SQuAD dataset (Rajpurkar et al., [2016a](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib35)).


<a id="source-section-29"></a>

#### Effect of Question Answering


We aim to study the pure performance gain of the paradigm shift from mention-pair scoring to query-based span prediction. For this purpose, we replace the mention linking module with the mention-pair scoring module described in Lee et al. ([2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib21)), while others remain unchanged. We observe an 8.1 F1 degradation in performance, demonstrating the significant superiority of the proposed question answering framework over the mention-pair scoring framework.


<a id="source-section-30"></a>

### 5.2 Analyses on speaker modeling strategies


Figure 3: Performance on the development set of the CoNLL-2012 dataset with various number of speakers. F1(Speaker as feature): F1 score for the strategy that treats speaker information as a mention-pair feature. F1(Speaker as input): F1 score for our strategy that treats speaker names as token input. Frequency: percentage of documents with specific number of speakers.


We compare our speaker modeling strategy (denoted by Speaker as input), which directly concatenates the speaker’s name with the corresponding utterance, with the strategy in
Wiseman et al. ([2016](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib41)); Lee et al. ([2017](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib20)); Joshi et al. ([2019a](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib15)) (denoted by Speaker as feature), which converts speaker information into binary features indicating whether two mentions are from the same speaker.
We show the average F1 scores breakdown by documents according to the number of their constituent speakers in Figure [3](https://ar5iv.labs.arxiv.org/html/1911.01746#S5.F3).


Results show that the proposed strategy performs significantly better on documents with a larger number of speakers. Compared with the coarse modeling of whether two utterances are from the same speaker, a speaker’s name can be thought of as speaker ID in persona dialogue learning Li et al. ([2016](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib24)); Zhang et al. ([2018b](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib44)); Mazaré et al. ([2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib28)). Representations learned for names have the potential to better generalize the global information of the speakers in the multi-party dialogue situation, leading to better context modeling and thus better results.


<a id="source-section-31"></a>

### 5.3 Analysis on the Overall Mention Recall


Figure 4: Change of mention recalls as we increase the number of spans $\lambda$ kept per word.


Since the proposed framework has the potential to retrieve
mentions missed at the mention proposal stage, we expect it to have higher overall mention recall rate than previous models
(Lee et al., [2017](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib20), [2018](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib21); Zhang et al., [2018a](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib43); Kantor and Globerson, [2019](https://ar5iv.labs.arxiv.org/html/1911.01746#bib.bib17)).


We examine the proportion of gold mentions covered in the development set as we increase the hyperparameter $\lambda$ (the number of spans kept per word) in Figure [4](https://ar5iv.labs.arxiv.org/html/1911.01746#S5.F4).
Our model consistently outperforms the baseline model with various values of $\lambda$. Notably, our model is less sensitive to smaller values of $\lambda$. This is because missed mentions can still be retrieved at the mention linking stage.


<a id="source-section-32"></a>

### 5.4 Qualitative Analysis


| 1 | [Freddie Mac] is giving golden parachutes to two of its ousted executives. …Yesterday Federal Prosecutions announced a criminal probe into [the company]. |
| --- | --- |
| 2 | [A traveling reporter] now on leave and joins us to tell [her] story. Thank [you] for coming in to share this with us. |
| 3 | Paula Zahn: [Thelma Gutierrez] went inside the forensic laboratory where scientists are trying to solve this mystery.<br>Thelma Gutierrez: In this laboratory alone [I] ’m surrounded by the remains of at least twenty different service members who are in the process of being identified so that they too can go home. |


Table 4: Example mention clusters that were correctly predicted by our model, but wrongly predicted by c2f-coref + SpanBERT-large. Bold spans in brackets represent coreferent mentions. Italic spans represent the speaker’s name of the utterance.


We provide qualitative analyses to highlight the strengths of our model in Table [4](https://ar5iv.labs.arxiv.org/html/1911.01746#S5.T4).


Shown in Example 1, by explicitly formulating the anaphora identification of the company as a query, our model uses more information from a local context, and successfully identifies Freddie Mac as the answer from a longer distance.


The model can also efficiently harness the speaker information in a conversational setting. In Example 3, it would be difficult to identify that [Thelma Gutierrez] is the correct antecedent of mention [I] without knowing that Thelma Gutierrez is the speaker of the second utterance. However, our model successfully identifies it by directly feeding the speaker’s name at the input level.


<a id="source-section-33"></a>

## 6 Conclusion


In this paper, we present CorefQA, a coreference resolution model that casts anaphora identification as the task of query-based span prediction in question answering. We showed that the proposed formalization can successfully retrieve mentions left out at the mention proposal stage. It also makes data augmentation using a plethora of existing question answering datasets possible. Furthermore, a new speaker modeling strategy can also boost the performance in dialogue settings. Empirical results on two widely-used coreference datasets demonstrate the effectiveness of our model. In future work, we will explore novel approaches to generate the questions based on each mention, and evaluate the influence of different question generation methods on the coreference resolution task.


<a id="source-section-34"></a>

## Acknowledgement


We thank all anonymous reviewers for their comments and suggestions.
The work is supported by the National Natural Science Foundation of China (NSFC No. 61625107 and 61751209).


<a id="source-section-35"></a>

## References


- Aralikatte et al. (2019)

Rahul Aralikatte, Matthew Lamm, Daniel Hardt, and Anders Søgaard. 2019.


Ellipsis and coreference resolution as question answering.


CoRR, abs/1908.11141.


- Bagga and Baldwin (1998)

Amit Bagga and Breck Baldwin. 1998.


Algorithms for scoring coreference chains.


In In The First International Conference on Language Resources
and Evaluation Workshop on Linguistics Coreference, pages 563–566.


- Björkelund and Kuhn (2014)

Anders Björkelund and Jonas Kuhn. 2014.


Learning structured perceptrons for coreference resolution with
latent antecedents and non-local features.


In Proceedings of the 52nd Annual Meeting of the Association
for Computational Linguistics, ACL 2014, June 22-27, 2014, Baltimore, MD,
USA, Volume 1: Long Papers, pages 47–57.


- Chada (2019)

Rakesh Chada. 2019.


Gendered pronoun resolution using bert and an extractive question
answering formulation.


arXiv preprint arXiv:1906.03695.


- Clark and Manning (2015)

Kevin Clark and Christopher D. Manning. 2015.


Entity-centric coreference resolution with model stacking.


In Proceedings of the 53rd Annual Meeting of the Association
for Computational Linguistics and the 7th International Joint Conference on
Natural Language Processing of the Asian Federation of Natural Language
Processing, ACL 2015, July 26-31, 2015, Beijing, China, Volume 1: Long
Papers, pages 1405–1415.


- Clark and Manning (2016)

Kevin Clark and Christopher D. Manning. 2016.


Improving coreference resolution by learning entity-level distributed
representations.


In Proceedings of the 54th Annual Meeting of the Association
for Computational Linguistics, ACL 2016, August 7-12, 2016, Berlin,
Germany, Volume 1: Long Papers.


- Dasigi et al. (2019a)

Pradeep Dasigi, Nelson F. Liu, Ana Marasovic, Noah A. Smith, and Matt Gardner.
2019a.


Quoref: A reading comprehension dataset with questions requiring
coreferential reasoning.


CoRR, abs/1908.05803.


- Dasigi et al. (2019b)

Pradeep Dasigi, Nelson F Liu, Ana Marasovic, Noah A Smith, and Matt Gardner.
2019b.


Quoref: A reading comprehension dataset with questions requiring
coreferential reasoning.


arXiv preprint arXiv:1908.05803.


- Devlin et al. (2019)

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.


BERT: pre-training of deep bidirectional transformers for language
understanding.


In Proceedings of the 2019 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume
1 (Long and Short Papers), pages 4171–4186.


- Durrett and Klein (2013)

Greg Durrett and Dan Klein. 2013.


Easy victories and uphill battles in coreference resolution.


In Proceedings of the 2013 Conference on Empirical Methods in
Natural Language Processing, EMNLP 2013, 18-21 October 2013, Grand Hyatt
Seattle, Seattle, Washington, USA, A meeting of SIGDAT, a Special Interest
Group of the ACL, pages 1971–1982.


- Emami et al. (2019)

Ali Emami, Paul Trichelair, Adam Trischler, Kaheer Suleman, Hannes Schulz, and
Jackie Chi Kit Cheung. 2019.


The knowref coreference corpus: Removing gender and number cues for
difficult pronominal anaphora resolution.


In Proceedings of the 57th Conference of the Association for
Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2,
2019, Volume 1: Long Papers, pages 3952–3961.


- Hassan et al. (2018)

Hany Hassan, Anthony Aue, Chang Chen, Vishal Chowdhary, Jonathan Clark,
Christian Federmann, Xuedong Huang, Marcin Junczys-Dowmunt, William Lewis,
Mu Li, et al. 2018.


Achieving human parity on automatic chinese to english news
translation.


arXiv preprint arXiv:1803.05567.


- He et al. (2015)

Luheng He, Mike Lewis, and Luke Zettlemoyer. 2015.


Question-answer driven semantic role labeling: Using natural language
to annotate natural language.


In Proceedings of the 2015 Conference on Empirical Methods in
Natural Language Processing, EMNLP 2015, Lisbon, Portugal, September 17-21,
2015, pages 643–653.


- Hou et al. (2018)

Yutai Hou, Yijia Liu, Wanxiang Che, and Ting Liu. 2018.


Sequence-to-sequence data augmentation for dialogue language
understanding.


In Proceedings of the 27th International Conference on
Computational Linguistics, COLING 2018, Santa Fe, New Mexico, USA, August
20-26, 2018, pages 1234–1245.


- Joshi et al. (2019a)

Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer, and
Omer Levy. 2019a.


Spanbert: Improving pre-training by representing and predicting
spans.


CoRR, abs/1907.10529.


- Joshi et al. (2019b)

Mandar Joshi, Omer Levy, Daniel S. Weld, and Luke Zettlemoyer.
2019b.


BERT for coreference resolution: Baselines and analysis.


CoRR, abs/1908.09091.


- Kantor and Globerson (2019)

Ben Kantor and Amir Globerson. 2019.


Coreference resolution with entity equalization.


In Proceedings of the 57th Conference of the Association for
Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2,
2019, Volume 1: Long Papers, pages 673–677.


- Kingma and Ba (2015)

Diederik P. Kingma and Jimmy Ba. 2015.


Adam: A method for stochastic optimization.


In 3rd International Conference on Learning Representations,
ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track
Proceedings.


- Kobayashi (2018)

Sosuke Kobayashi. 2018.


Contextual augmentation: Data augmentation by words with paradigmatic
relations.


In Proceedings of the 2018 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, NAACL-HLT, New Orleans, Louisiana, USA, June 1-6, 2018, Volume
2 (Short Papers), pages 452–457.


- Lee et al. (2017)

Kenton Lee, Luheng He, Mike Lewis, and Luke Zettlemoyer. 2017.


End-to-end neural coreference resolution.


In Proceedings of the 2017 Conference on Empirical Methods in
Natural Language Processing, EMNLP 2017, Copenhagen, Denmark, September
9-11, 2017, pages 188–197.


- Lee et al. (2018)

Kenton Lee, Luheng He, and Luke Zettlemoyer. 2018.


Higher-order coreference resolution with coarse-to-fine inference.


In Proceedings of the 2018 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, NAACL-HLT, New Orleans, Louisiana, USA, June 1-6, 2018, Volume
2 (Short Papers), pages 687–692.


- Levy et al. (2017)

Omer Levy, Minjoon Seo, Eunsol Choi, and Luke Zettlemoyer. 2017.


Zero-shot relation extraction via reading comprehension.


In Proceedings of the 21st Conference on Computational Natural
Language Learning (CoNLL 2017), Vancouver, Canada, August 3-4, 2017, pages
333–342.


- Li et al. (2015)

Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and Bill Dolan. 2015.


A diversity-promoting objective function for neural conversation
models.


arXiv preprint arXiv:1510.03055.


- Li et al. (2016)

Jiwei Li, Michel Galley, Chris Brockett, Georgios P Spithourakis, Jianfeng Gao,
and Bill Dolan. 2016.


A persona-based neural conversation model.


arXiv preprint arXiv:1603.06155.


- Li et al. (2019a)

Xiaoya Li, Jingrong Feng, Yuxian Meng, Qinghong Han, Fei Wu, and Jiwei Li.
2019a.


A unified mrc framework for named entity recognition.


arXiv preprint arXiv:1910.11476.


- Li et al. (2019b)

Xiaoya Li, Fan Yin, Zijun Sun, Xiayu Li, Arianna Yuan, Duo Chai, Mingxin Zhou,
and Jiwei Li. 2019b.


Entity-relation extraction as multi-turn question answering.


In Proceedings of the 57th Conference of the Association for
Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2,
2019, Volume 1: Long Papers, pages 1340–1350.


- Luo (2005)

Xiaoqiang Luo. 2005.


On coreference resolution performance metrics.


In HLT/EMNLP 2005, Human Language Technology Conference and
Conference on Empirical Methods in Natural Language Processing, Proceedings
of the Conference, 6-8 October 2005, Vancouver, British Columbia, Canada,
pages 25–32.


- Mazaré et al. (2018)

Pierre-Emmanuel Mazaré, Samuel Humeau, Martin Raison, and Antoine Bordes.
2018.


Training millions of personalized dialogue agents.


arXiv preprint arXiv:1809.01984.


- McCann et al. (2018)

Bryan McCann, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. 2018.


The natural language decathlon: Multitask learning as question
answering.


CoRR, abs/1806.08730.


- Morgenstern et al. (2016)

Leora Morgenstern, Ernest Davis, and Charles L. Ortiz Jr. 2016.


Planning, executing, and evaluating the winograd schema challenge.


AI Magazine, 37(1):50–54.


- Pennington et al. (2014)

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.


Glove: Global vectors for word representation.


In Proceedings of the 2014 Conference on Empirical Methods in
Natural Language Processing, EMNLP 2014, October 25-29, 2014, Doha, Qatar,
A meeting of SIGDAT, a Special Interest Group of the ACL, pages
1532–1543.


- Peters et al. (2018)

Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark,
Kenton Lee, and Luke Zettlemoyer. 2018.


Deep contextualized word representations.


In Proceedings of the 2018 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, NAACL-HLT 2018, New Orleans, Louisiana, USA, June 1-6, 2018,
Volume 1 (Long Papers), pages 2227–2237.


- Pradhan et al. (2012)

Sameer Pradhan, Alessandro Moschitti, Nianwen Xue, Olga Uryupina, and Yuchen
Zhang. 2012.


Conll-2012 shared task: Modeling multilingual unrestricted
coreference in ontonotes.


In Joint Conference on Empirical Methods in Natural Language
Processing and Computational Natural Language Learning - Proceedings of the
Shared Task: Modeling Multilingual Unrestricted Coreference in OntoNotes,
EMNLP-CoNLL 2012, July 13, 2012, Jeju Island, Korea, pages 1–40.


- Rajpurkar et al. (2018)

Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018.


Know what you don’t know: Unanswerable questions for squad.


In Proceedings of the 56th Annual Meeting of the Association
for Computational Linguistics, ACL 2018, Melbourne, Australia, July 15-20,
2018, Volume 2: Short Papers, pages 784–789.


- Rajpurkar et al. (2016a)

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang.
2016a.


Squad: 100, 000+ questions for machine comprehension of text.


In Proceedings of the 2016 Conference on Empirical Methods in
Natural Language Processing, EMNLP 2016, Austin, Texas, USA, November 1-4,
2016, pages 2383–2392.


- Rajpurkar et al. (2016b)

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang.
2016b.


Squad: 100,000+ questions for machine comprehension of text.


arXiv preprint arXiv:1606.05250.


- Talmor and Berant (2019)

Alon Talmor and Jonathan Berant. 2019.


Multiqa: An empirical investigation of generalization and transfer in
reading comprehension.


In Proceedings of the 57th Conference of the Association for
Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2,
2019, Volume 1: Long Papers, pages 4911–4921.


- Turian et al. (2010)

Joseph P. Turian, Lev-Arie Ratinov, and Yoshua Bengio. 2010.


Word representations: A simple and general method for
semi-supervised learning.


In ACL 2010, Proceedings of the 48th Annual Meeting of the
Association for Computational Linguistics, July 11-16, 2010, Uppsala,
Sweden, pages 384–394.


- Vilain et al. (1995)

Marc B. Vilain, John D. Burger, John S. Aberdeen, Dennis Connolly, and Lynette
Hirschman. 1995.


A model-theoretic coreference scoring scheme.


In Proceedings of the 6th Conference on Message Understanding,
MUC 1995, Columbia, Maryland, USA, November 6-8, 1995, pages 45–52.


- Webster et al. (2018)

Kellie Webster, Marta Recasens, Vera Axelrod, and Jason Baldridge. 2018.


Mind the GAP: A balanced corpus of gendered ambiguous pronouns.


TACL, 6:605–617.


- Wiseman et al. (2016)

Sam Wiseman, Alexander M. Rush, and Stuart M. Shieber. 2016.


Learning global features for coreference resolution.


In NAACL HLT 2016, The 2016 Conference of the North
American Chapter of the Association for Computational Linguistics: Human
Language Technologies, San Diego California, USA, June 12-17, 2016, pages
994–1004.


- Wiseman et al. (2015)

Sam Wiseman, Alexander M. Rush, Stuart M. Shieber, and Jason Weston. 2015.


Learning anaphoricity and antecedent ranking features for coreference
resolution.


In Proceedings of the 53rd Annual Meeting of the Association
for Computational Linguistics and the 7th International Joint Conference on
Natural Language Processing of the Asian Federation of Natural Language
Processing, ACL 2015, July 26-31, 2015, Beijing, China, Volume 1: Long
Papers, pages 1416–1426.


- Zhang et al. (2018a)

Rui Zhang, Cícero Nogueira dos Santos, Michihiro Yasunaga, Bing Xiang,
and Dragomir R. Radev. 2018a.


Neural coreference resolution with deep biaffine attention by joint
mention detection and mention clustering.


In Proceedings of the 56th Annual Meeting of the Association
for Computational Linguistics, ACL 2018, Melbourne, Australia, July 15-20,
2018, Volume 2: Short Papers, pages 102–107.


- Zhang et al. (2018b)

Saizheng Zhang, Emily Dinan, Jack Urbanek, Arthur Szlam, Douwe Kiela, and Jason
Weston. 2018b.


Personalizing dialogue agents: I have a dog, do you have pets too?


arXiv preprint arXiv:1801.07243.


- Zhao et al. (2019)

Jieyu Zhao, Tianlu Wang, Mark Yatskar, Ryan Cotterell, Vicente Ordonez, and
Kai-Wei Chang. 2019.


Gender bias in contextualized word embeddings.


In Proceedings of the 2019 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume
1 (Long and Short Papers), pages 629–634.


- Zhao et al. (2018)

Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang.
2018.


Gender bias in coreference resolution: Evaluation and debiasing
methods.


In Proceedings of the 2018 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, NAACL-HLT, New Orleans, Louisiana, USA, June 1-6, 2018, Volume
2 (Short Papers), pages 15–20.

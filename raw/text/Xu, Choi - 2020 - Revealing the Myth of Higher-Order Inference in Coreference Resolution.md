# Xu, Choi - 2020 - Revealing the Myth of Higher-Order Inference in Coreference Resolution

- Source HTML: `raw/html/Xu, Choi - 2020 - Revealing the Myth of Higher-Order Inference in Coreference Resolution.html`
- Source SHA256: `a3ea7f597f3c1fd6f56a8871304aaec79ff76f9f2a7bf7bb7f45ea914bf43601`
- Source URL: https://ar5iv.labs.arxiv.org/html/2009.12013
- Generated from: `scripts/fetch_web_text.py`
- Extraction: `structured-html-v2` (headings, links, MathML/TeX and tables; figures require visual review)

## Extracted Text

<a id="source-section-0"></a>

<a id="source-section-1"></a>

# Revealing the Myth of Higher-Order Inference in Coreference Resolution


Liyan Xu

Computer Science

Emory University, Atlanta, GA

liyan.xu@emory.edu

&Jinho D. Choi

Computer Science

Emory University, Atlanta, GA

jinho.choi@emory.edu


<a id="source-section-2"></a>

###### Abstract


This paper analyzes the impact of higher-order inference (HOI) on the task of coreference resolution.
HOI has been adapted by almost all recent coreference resolution models without taking much investigation on its true effectiveness over representation learning.
To make acomprehensive analysis, we implement an end-to-end coreference system as well as four HOI approaches, attended antecedent, entity equalization, span clustering, and cluster merging, where the latter two are our original methods.
We find that given a high-performing encoder such as SpanBERT, the impact of HOI is negative to marginal, providing a new perspective of HOI to this task.
Our best model using cluster merging shows the Avg-F1 of 80.2 on the CoNLL 2012 shared task dataset in English.


<a id="source-section-3"></a>

## 1 Introduction


Coreference resolution has always been considered one of the unsolved NLP tasks due to its challenging aspect of document-level understanding Wiseman et al. ([2015](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib13), [2016](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib14)); Clark and Manning ([2015](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib1), [2016](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib2)); Lee et al. ([2017](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib8)).
Nonetheless, it has made a tremendous progress in recent years by adapting contextualized embedding encoders such as ELMo Lee et al. ([2018](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib9)); Fei et al. ([2019](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib4)) and BERT Kantor and Globerson ([2019](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib7)); Joshi et al. ([2019](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib6), [2020](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib5)).
The latest state-of-the-art model shows the improvement of 12.4% over the model introduced 2.5 years ago, where the major portion of the improvement is derived by representation learning (Figure [1](https://ar5iv.labs.arxiv.org/html/2009.12013#S1.F1)).


Most of these previous models have also adapted higher-order inference (HOI) for the global optimization of coreference links, although HOI clearly has not been the focus of those works, for the fact that gains from HOI have been reported marginal.
This has inspired us to analyze the impact of HOI on modern coreference resolution models in order to envision the future direction of this research.


To make thorough ablation studies among different approaches, we implement an end-to-end coreference system in PyTorch (Sec [3.1](https://ar5iv.labs.arxiv.org/html/2009.12013#S3.SS1)),
and two HOI approaches proposed by previous work, attended antecedent and entity equalization (Sec [3.2](https://ar5iv.labs.arxiv.org/html/2009.12013#S3.SS2)), along with two of our original approaches, span clustering and cluster merging (Sec [3.3](https://ar5iv.labs.arxiv.org/html/2009.12013#S3.SS3)).
These approaches are experimented with two Transformer encoders, BERT and SpanBERT, to assess how effective HOI is even when coupled with those high-performing encoders (Sec [4](https://ar5iv.labs.arxiv.org/html/2009.12013#S4)).
To the best of our knowledge,this is the first work to make a comprehensive analysis on multiple HOI approaches side-by-side for the task of coreference resolution.111Source codes and models are available at [https://github.com/lxucs/coref-hoi](https://github.com/lxucs/coref-hoi).


[图片：Refer to caption]


Figure 1: Performance breakdown of the recent state-of-the-art models on the CoNLL 2012 shared task. W-16: Wiseman et al. ([2016](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib14)), C-16: Clark and Manning ([2016](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib2)), L-17: Lee et al. ([2017](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib8)), L-18: Lee et al. ([2018](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib9)), F-19: Fei et al. ([2019](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib4)), K-19: Kantor and Globerson ([2019](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib7)), J-19: Joshi et al. ([2019](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib6)), J-20: Joshi et al. ([2020](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib5)).


<a id="source-section-4"></a>

## 2 Related Work


Most neural network-based coreference resolution models have adapted antecedent-ranking Wiseman et al. ([2015](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib13)); Clark and Manning ([2015](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib1)); Lee et al. ([2017](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib8), [2018](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib9)); Joshi et al. ([2019](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib6), [2020](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib5)), which relies on the local decisions between each mention and its antecedents.
To achieve deeper global optimization,Wiseman et al. ([2016](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib14)); Clark and Manning ([2016](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib2)); Yu et al. ([2020](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib16)) built entity representations in the ranking process, whereas Lee et al. ([2018](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib9)); Kantor and Globerson ([2019](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib7)) refined the mention representation by aggregating its antecedents’ information.


It is no secret that the integration of contextualized embeddings has played the most critical role in this task. While the following are based on the same end-to-end coreference model Lee et al. ([2017](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib8)),
Lee et al. ([2018](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib9)); Fei et al. ([2019](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib4)) reported 3.3% improvement by adapting ELMo in the encoders Peters et al. ([2018](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib11)).
Kantor and Globerson ([2019](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib7)); Joshi et al. ([2019](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib6)) gained additional 3.3% by adapting BERT Devlin et al. ([2019](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib3)).
Joshi et al. ([2020](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib5)) introduced SpanBERT that gave another 2.7% improvement over Joshi et al. ([2019](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib6)).


Most recently, Wu et al. ([2020](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib15)) proposes a new model that adapts question-answering framework on coreference resolution, and achieves state-of-the-art result of 83.1 on the CoNLL’12 shared task.


<a id="source-section-5"></a>

## 3 Approach


<a id="source-section-6"></a>

### 3.1 End-to-End Coreference System


We reimplement the end-to-end c2f-coref model introduced by Lee et al. ([2018](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib9)) that has been adapted by every coreference resolution model since then. It detects mention candidates through span enumeration and aggressive pruning.
For each candidate span $x$, the model learns the distribution over its antecedents $y\in\mathcal{Y}(x)$:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>P(y)=\frac{e^{s(x,y)}}{\sum_{y^{\prime}\in\mathcal{Y}(x)}e^{s(x,y^{\prime})}}<br>$$ | | (1) |


where $s(x,y)$ is the local score involving two parts: how likely the spans $x$ and $y$ are valid mentions, and how likely they refer to the same entity:


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | $\displaystyle s(x,y)$ | $\displaystyle=s_{m}(x)+s_{m}(y)+s_{c}(x,y)$ | | (2) |
| | $\displaystyle s_{m}(x)$ | $\displaystyle=w_{m}\texttt{FFNN}_{m}(g_{x})$ | | |
| | $\displaystyle s_{c}(x,y)$ | $\displaystyle=w_{c}\texttt{FFNN}_{c}(g_{x},g_{y},\phi(x,y))$ | | |


$g_{x},g_{y}$ are the span embeddings of $x$ and $y$, $\phi(x,y)$ is the meta-information (e.g., speakers, distance), and $w_{m},w_{c}$ are the mention and coreference scores, respectively (FFNN: feedforward neural network).


We use different Transformers-based encoders, and follow the “independent” setup for long documents as suggested by Joshi et al. ([2019](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib6)).


<a id="source-section-7"></a>

### 3.2 Span Refinement


Two HOI methods presented by recent coreference work are based on span refinement that aggregates non-local features to enrich the span representation with more “global” information.
The updated span representation $g_{x}^{\prime}$ can be derived as in Eq. [3](https://ar5iv.labs.arxiv.org/html/2009.12013#S3.E3), where $g_{x}^{\prime}$ is the interpolation between the current and refined representation $g_{x}$ and $a_{x}$, and $W_{f}$ is the gate parameter.
$g_{x}^{\prime}$ is used to perform another round of antecedent-ranking in replacement of $g_{x}$.


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | $\displaystyle g_{x}^{\prime}$ | $\displaystyle=f_{x}\circ g_{x}+(1-f_{x})\circ a_{x}$ | | (3) |
| | $\displaystyle f_{x}$ | $\displaystyle=\sigma(W_{f}[g_{x},a_{x}])$ | | |


The following two methods share the same updating process for $g_{x}^{\prime}$, but with different ways to obtain the refined span representation $a_{x}$.


<a id="source-section-8"></a>

#### Attended Antecedent (AA)


takes the antecedent information to enrich $g_{x}^{\prime}$ (Lee et al., [2018](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib9); Fei et al., [2019](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib4); Joshi et al., [2019](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib6), [2020](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib5)).
The refined span $a_{x}$ is the attended antecedent representation over the current antecedent distribution $P(y)$, where $g_{y\in\mathcal{Y}(x)}$ is the antecedent representation:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>a_{x}=\sum_{y\in\mathcal{Y}(x)}P(y)\cdot g_{y}<br>$$ | | (4) |


<a id="source-section-9"></a>

#### Entity Equalization (EE)


takes the clustering relaxation as in Eq. [3.2](https://ar5iv.labs.arxiv.org/html/2009.12013#S3.Ex4) to model the entity distribution (Kantor and Globerson, [2019](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib7)), where $Q(x\in E_{y^{\prime}})$ is the probability of the span $x$ referring to an entity $E_{y^{\prime}}$ in which the span $y^{\prime}$ is the first mention. $P(y)$ is the current antecedent distribution.


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | | $\displaystyle Q(x\in E_{y^{\prime}})=$ | | |
| | | $\displaystyle\begin{cases}\sum_{k=y^{\prime}}^{x-1}P(y=k)\cdot Q(k\in E_{y^{\prime}})&y^{\prime}<x\\<br>P(y=\epsilon)&y^{\prime}=x\\<br>0&y^{\prime}>x\end{cases}$ | | (5) |


The refined span $a_{x}$ is the attended entity representation, where $e_{y}^{(x)}$ is the entity representation to which the span $y$ belongs till the span $x$:


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | $\displaystyle e_{x}^{(t)}$ | $\displaystyle=\sum_{y=1}^{t}Q(y\in E_{x})\cdot g_{y}$ | | (6) |
| | $\displaystyle a_{x}$ | $\displaystyle=\sum_{y=1}^{x}Q(x\in E_{y})\cdot e_{y}^{(x)}$ | | (7) |


<a id="source-section-10"></a>

### 3.3 HOI with Clustering


This section introduces two new HOI methods for a more extensive study in HOI.


<a id="source-section-11"></a>

#### Span Clustering (SC)


is also based on span refinement, and it
constructs the actual clusters and obtains the “true” predicted entities using $P(y)$ instead of modeling the “soft” entity clusters through the relaxation as in EE (Section [3.2](https://ar5iv.labs.arxiv.org/html/2009.12013#S3.SS2)).
This way, although we lose the differentiable property, the obtaining of true entities with the same empirical inference time as EE has made SC desirable.


The entity representation $e_{i}$ for an entity cluster $C_{i}$ is given by the attended spans in this cluster:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $\displaystyle\alpha_{t}$ | $\displaystyle=w_{\alpha}\text{FFNN}_{\alpha}(g_{t})$ | |
| | $\displaystyle\alpha_{i,t}$ | $\displaystyle=\frac{\exp(\alpha_{t})}{\sum_{k\in C_{i}}\exp(\alpha_{k})}$ | |
| | $\displaystyle e_{i}$ | $\displaystyle=\sum_{t\in C_{i}}\alpha_{i,t}\cdot g_{t}$ | |


The entity clusters $C_{i}$ are constructed in the same way as in the final cluster prediction.
The refined span $a_{x}$ is then equal to the representation of entity $e_{i}$ to which it belongs ($g_{x}\in C_{i}$).


<a id="source-section-12"></a>

#### Cluster Merging (CM)


performs sequential antecedent ranking combining both antecedent and entity information to gradually build up the entity clusters, which is distinguished from span refinement methods that simply re-rank antecedents.
Algorithm [1](https://ar5iv.labs.arxiv.org/html/2009.12013#alg1) describes the ranking process for CM.
$g_{i}$ is the $i$’th span, $\mathcal{Y}(i)$ is the indices of $g_{i}$’s antecedents, and $C_{i}$ is the cluster that $g_{i}$ belongs to.
The ranking score $s_{x}(y)$ consists of both antecedent score $f_{a}$ (see Eq. [2](https://ar5iv.labs.arxiv.org/html/2009.12013#S3.E2)) and cluster score $f_{c}$.
To avoid overlapping between $f_{a}$ and $f_{c}$, we set $f_{c}$ as $0$ if the cluster is the initial cluster (L6).
Thus, $f_{c}$ becomes the consultation such that when $f_{c}>0$, the span $g_{x}$ is likely to match the cluster $C_{y}$, and vice versa.
$f_{c}$ is computed by FFNN similar to $f_{a}$, and $\phi(C_{y})$ is the meta-feature such as the cluster size.


Algorithm 1 Antecedent Ranking for CM


1:procedure ranking($g_{1},\cdots,g_{N}$)


2:     $C_{i=1,\cdots,N}\leftarrow g_{i}$


3:     $R\leftarrow$ ranking_order($g_{1},\cdots,g_{N}$)


4:     for $x=R_{1}\cdots R_{N}$ do


5:         for $y\in\mathcal{Y}(x)$ do $\triangleright$ Parallelized


6:              $f_{c}(g_{x},C_{y})\leftarrow 0$ if $C_{y}=g_{y}$


7:              $s_{x}(y)\leftarrow f_{a}(g_{x},g_{y})+f_{c}(g_{x},C_{y},\phi(C_{y}))$


8:         $y^{\prime}\leftarrow\text{argmax}_{y\in\mathcal{Y}(x)}s_{x}(y)$


9:         if $y^{\prime}\neq\epsilon$ then


10:              merge $C_{x}$ and $C_{y^{\prime}}$


11:     return $s_{1},\cdots,s_{N}$


Two simple configurations can be tuned for CM. We can have the sequential left-to-right ranking order or the easy-first order (L3) whose sequence is ordered by each span’s max antecedent score, building the most confident clusters first (Ng and Cardie, [2002](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib10); Clark and Manning, [2016](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib2)). There can be element-wise mean or max-reduction among the spans in the two merging clusters (L10).


Distinguished from Wiseman et al. ([2016](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib14)), clusters in CM are searched and merged in training without the use of oracle clusters, closing the gap between training and test time.


| | MUC [colspan=3] | | B3 [colspan=3] | | CEAF$\phi_{4}$ [colspan=3] | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | P | R | F1 | | P | R | F1 | | P | R | F1 | Avg. F1 | Avg-M |
| L-17 | 78.4 | 73.4 | 75.8 | | 68.6 | 61.8 | 65.0 | | 62.7 | 59.0 | 60.8 | 67.2 | - |
| L-18 | 81.4 | 79.5 | 80.4 | | 72.2 | 69.5 | 70.8 | | 68.2 | 67.1 | 67.6 | 73.0 | - |
| F-19 | 85.4 | 77.9 | 81.4 | | 77.9 | 66.4 | 71.7 | | 70.6 | 66.3 | 68.4 | 73.8 | - |
| K-19 | 82.6 | 84.1 | 83.4 | | 73.3 | 76.2 | 74.7 | | 72.4 | 71.1 | 71.8 | 76.6 | - |
| J-19 | 84.7 | 82.4 | 83.5 | | 76.5 | 74.0 | 75.3 | | 74.1 | 69.8 | 71.9 | 76.9 | - |
| J-20 | 85.8 | 84.8 | 85.3 | | 78.3 | 77.9 | 78.1 | | 76.4 | 74.2 | 75.3 | 79.6 | - |
| BERT | 85.0 | 82.5 | 83.8 | | 77.3 | 74.0 | 75.6 | | 74.9 | 70.7 | 72.8 | 77.4 | 77.3 ($\pm$0.1) |
| SpanBERT | 85.7 | 85.3 | 85.5 | | 78.6 | 78.6 | 78.6 | | 76.8 | 74.8 | 75.8 | 79.9 | 79.7 ($\pm$0.1) |
| + AA | 86.1 | 84.8 | 85.4 | | 79.3 | 77.3 | 78.3 | | 76.0 | 74.7 | 75.4 | 79.7 | 79.4 ($\pm$0.2) |
| + EE | 85.7 | 84.5 | 85.1 | | 78.5 | 77.4 | 77.9 | | 76.7 | 73.4 | 75.0 | 79.4 | 78.9 ($\pm$0.4) |
| + SC | 85.5 | 85.2 | 85.4 | | 78.4 | 78.5 | 78.4 | | 76.5 | 74.1 | 75.2 | 79.7 | 79.2 ($\pm$0.3) |
| + CM | 85.9 | 85.5 | 85.7 | | 79.0 | 78.9 | 79.0 | | 76.7 | 75.2 | 75.9 | 80.2 | 79.9 ($\pm$0.2) |


Table 1: Best results on the test set of the CoNLL’12 English shared task. The averaged F1 of MUC, B3, CEAF$\phi_{4}$ is the main evaluation metric. Avg-M: the mean Avg-F1 and its standard deviation from five developments. The mean and stdev of other metrics are provided in Appendix [A.2](https://ar5iv.labs.arxiv.org/html/2009.12013#A1.SS2). See Figure [1](https://ar5iv.labs.arxiv.org/html/2009.12013#S1.F1) for acronyms of the previous works.


<a id="source-section-13"></a>

## 4 Experiments


For our experiments, the CoNLL 2012 English shared task dataset is used (Pradhan et al., [2012](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib12)).
Given the end-to-end coreference system in Section [3.1](https://ar5iv.labs.arxiv.org/html/2009.12013#S3.SS1), six models are developed as follows:222Appdendix [A.1](https://ar5iv.labs.arxiv.org/html/2009.12013#A1.SS1) provides details of our experimental settings.


- •


BERT: BERT Devlin et al. ([2019](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib3)) as the encoder


- •


SpanBERT: SpanBERT Joshi et al. ([2020](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib5)) as the encoder


- •


+AA: SpanBERT with attended antecedent (§[3.2](https://ar5iv.labs.arxiv.org/html/2009.12013#S3.SS2))


- •


+EE: SpanBERT with entity equalization (§[3.2](https://ar5iv.labs.arxiv.org/html/2009.12013#S3.SS2))


- •


+SC: SpanBERT with span clustering (§[3.3](https://ar5iv.labs.arxiv.org/html/2009.12013#S3.SS3))


- •


+CM: SpanBERT with cluster merging (§[3.3](https://ar5iv.labs.arxiv.org/html/2009.12013#S3.SS3))


Note that BERT and SpanBERT completely rely on only local decisions without any HOI.
Particularly, +AA is equivalent to Joshi et al. ([2020](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib5)).


<a id="source-section-14"></a>

### 4.1 Results


Table [1](https://ar5iv.labs.arxiv.org/html/2009.12013#S3.T1) shows the best results in comparison to previous state-of-the-art systems. We also report the mean scores and standard deviations from 5 repeated developments, which we could not find from the previous works.


The impact of SpanBERT over BERT is clear, showing 2.4% improvement on average.
However, none of the HOI models shows a clear advantage over SpanBERT which adapts no HOI.
In fact, all HOI models except for CM show negative impact.
The best result is achieved by CM with the Avg-F1 of 80.2, surpassing the previous best result of 79.6 based on c2f-coref reported by Joshi et al. ([2020](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib5)).


<a id="source-section-15"></a>

### 4.2 Impact Analysis of HOI


Three HOI methods based on span refinement, AA, EE, and SC, show negative impact upon local decisions.
We suspect that error propagation from antecedent-ranking may downgrade the quality of refinement.
On the other hand, CM shows marginal improvement, suggesting that maintaining entity clusters can be superior to span refinement, at the cost of more inference time from the sequential ranking process.
To analyze the direct impact of HOI, we take the trained models of each HOI method and evaluate them on the test set while turning off HOI, making it compatible to SpanBERT.


The averaged performance drop w.r.t Avg-F1 after turning off HOI is less than 0.2 for all methods(Appendix [A.3](https://ar5iv.labs.arxiv.org/html/2009.12013#A1.SS3)), implying that none of the HOI method has a significantly direct impact to the final performance of the model using SpanBERT.


| | W2C | C2W | C2C | W2W |
| --- | --- | --- | --- | --- |
| + AA | 240.8 (1.3) | 241.2 (1.3) | 16262.2 | 2168.4 |
| + EE | 244.1 (1.3) | 245.3 (1.3) | 16183.3 | 2136.3 |
| + SC | 248.2 (1.3) | 262.0 (1.4) | 16184.4 | 2146.0 |
| + CM | 226.4 (1.2) | 235.0 (1.2) | 16446.0 | 2180.0 |


Table 2: Averaged statistics on the test set prediction of four HOI approaches. W2C represents the number of mentions that are linked to a Wrong antecedent before HOI and are linked to a Correct antecedent after HOI; vice versa for C2W. C2C/W2W is the number of mentions that are both linked to Correct/Wrong antecedents before and after HOI. Parentheses indicate the percentage of corresponding numbers per row.


In further investigation, we examine the change of coreferent links w.r.t their correctness. Specifically, Table [2](https://ar5iv.labs.arxiv.org/html/2009.12013#S4.T2) shows the four types of link changes before and after HOI. It demonstrates that the benefits from HOI is diminished because the effects are two-sided: there are roughly same amounts of links (about 1%) becoming correct or wrong after HOI, therefore neither HOI method leads to much improvement overall.


It is worth mentioning that the impact of HOI is not limited to only global decisions.
HOI implicitlyserves as a way of regularization that impacts local decisions as well, since HOI and local ranking are mutually dependent during training.
Such indirect influence of HOI makes it difficult to assess its true impact, which we will explore more in the future.


<a id="source-section-16"></a>

### 4.3 Analysis of Pronoun Resolution


| | SP | PS | FL | WL | BC |
| --- | --- | --- | --- | --- | --- |
| BERT | 2.3 | 6.5 | 213.8 | 186.3 | 48.8 (3.5) |
| SpanBERT | 2.8 | 6.6 | 218.3 | 168.0 | 43.8 (2.7) |
| + AA | 1.8 | 8.8 | 214.2 | 159.4 | 44.8 (2.4) |
| + EE | 1.8 | 5.5 | 210.0 | 165.3 | 44.0 (2.5) |
| + SC | 3.8 | 7.2 | 223.6 | 170.0 | 45.4 (3.0) |
| + CM | 3.0 | 6.6 | 208.0 | 162.2 | 43.8 (2.6) |


Table 3: Averaged statistics on the test set prediction of different approaches. SP is the number of coreferent links from Singular to Plural personal pronouns; vice versa for PS. FL (False Link) and WL (Wrong Link) is the number of conreferent link errors that involve two personal pronouns. BC is the number of clusters that contain both singular and plural pronouns, and the parentheses indicate the numbers of BC that contain ambiguous pronouns such as “you”.


<a id="source-section-17"></a>

#### Direct Inference


For the error analysis, we examine the direct inference between two personal pronouns.333Ambiguous pronouns such as “you” are excluded in direct inference analysis, and included in indirect inference analysis.
SP/PS in Table [3](https://ar5iv.labs.arxiv.org/html/2009.12013#S4.T3) shows the numbers of links that one pronoun incorrectly selects another pronoun with different plurality as its antecedent.
We find that adapting HOI shows slightly higher impact than switching to a more advanced encoder.
AA can reinforce the pronoun representation to bias towards singularity and lead to lower SP error and higher PS error, while the difference between BERT and SpanBERT is trivial on SP/PS.


We also look at the general types of coreferent errors involving two pronouns.
False Link (FL) falsely links a non-anaphoric pronoun to another pronoun as antecedent; Wrong Link (WL) links an anaphoric pronoun to another wrong pronoun as antecedent. Table [3](https://ar5iv.labs.arxiv.org/html/2009.12013#S4.T3) shows that EE and CM reduce FL errors by 4+%, suggesting that the aggregation of non-local features indeed leads to more conservative linking decisions.
However, adapting an advanced encoder shows higher impact on WL errors, as SpanBERT reduces almost 10% compared to BERT, implying that representation learning is still more important for semantic matching.


<a id="source-section-18"></a>

#### Indirect Inference


The plurality of ambiguous pronouns such as you depends on the context. Two indirect links of (he, you) and (you, they) can be common to induce incorrect clusters that contain both singular and plural pronouns (Wiseman et al., [2016](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib14); Lee et al., [2018](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib9)). Table [3](https://ar5iv.labs.arxiv.org/html/2009.12013#S4.T3) shows the numbers of these erroneous clusters in prediction. Surprisingly, very few of these clusters contain ambiguous pronouns in either approach. This observation moderates the long-standing movitation of HOI.


Additionally, the change of representation from BERT to SpanBERT has far more impact that reduces 10% of these erroneous clusters, while the four HOI methods fail to show significant difference compared to SpanBERT.


<a id="source-section-19"></a>

## 5 Conclusion


We implement the end-to-end coreference resolution model and investigate four higher-order inference methods, including two of our own methods.
Our best model shows the new result of 80.2 on the CoNLL 2012 dataset.
We thoroughly analyze the empirical effectiveness of HOI and demonstrate why it fails to boost performance on the CoNLL 2012 dataset compared to the improvement from encoders. We show that current HOI does not meet up with the original motivation, suggesting that a new perspective of HOI is needed for this task in the era of deep learning-based NLP.


<a id="source-section-20"></a>

## Acknowledgments


We gratefully acknowledge the support of the AWS Machine Learning Research Awards (MLRA).
Any contents in this material are those of the authors and do not necessarily reflect the views of AWS.


<a id="source-section-21"></a>

## References


- Clark and Manning (2015)

Kevin Clark and Christopher D. Manning. 2015.


[Entity-centric
coreference resolution with model stacking](https://doi.org/10.3115/v1/P15-1136).


In Proceedings of the 53rd Annual Meeting of the Association
for Computational Linguistics and the 7th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers), pages 1405–1415,
Beijing, China. Association for Computational Linguistics.


- Clark and Manning (2016)

Kevin Clark and Christopher D. Manning. 2016.


[Improving coreference
resolution by learning entity-level distributed representations](https://doi.org/10.18653/v1/P16-1061).


In Proceedings of the 54th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), pages 643–653,
Berlin, Germany. Association for Computational Linguistics.


- Devlin et al. (2019)

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.


[BERT: Pre-training of
deep bidirectional transformers for language understanding](https://doi.org/10.18653/v1/N19-1423).


In Proceedings of the 2019 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long and Short Papers), pages 4171–4186,
Minneapolis, Minnesota. Association for Computational Linguistics.


- Fei et al. (2019)

Hongliang Fei, Xu Li, Dingcheng Li, and Ping Li. 2019.


[End-to-end deep
reinforcement learning based coreference resolution](https://doi.org/10.18653/v1/P19-1064).


In Proceedings of the 57th Annual Meeting of the Association
for Computational Linguistics, pages 660–665, Florence, Italy. Association
for Computational Linguistics.


- Joshi et al. (2020)

Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer, and
Omer Levy. 2020.


[Spanbert: Improving
pre-training by representing and predicting spans](https://doi.org/10.1162/tacl_a_00300).


Transactions of the Association for Computational Linguistics,
8:64–77.


- Joshi et al. (2019)

Mandar Joshi, Omer Levy, Luke Zettlemoyer, and Daniel Weld. 2019.


[BERT for coreference
resolution: Baselines and analysis](https://doi.org/10.18653/v1/D19-1588).


In Proceedings of the 2019 Conference on Empirical Methods in
Natural Language Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP), pages 5803–5808, Hong Kong,
China. Association for Computational Linguistics.


- Kantor and Globerson (2019)

Ben Kantor and Amir Globerson. 2019.


[Coreference resolution
with entity equalization](https://doi.org/10.18653/v1/P19-1066).


In Proceedings of the 57th Annual Meeting of the Association
for Computational Linguistics, pages 673–677, Florence, Italy. Association
for Computational Linguistics.


- Lee et al. (2017)

Kenton Lee, Luheng He, Mike Lewis, and Luke Zettlemoyer. 2017.


[End-to-end neural
coreference resolution](https://doi.org/10.18653/v1/D17-1018).


In Proceedings of the 2017 Conference on Empirical Methods in
Natural Language Processing, pages 188–197, Copenhagen, Denmark.
Association for Computational Linguistics.


- Lee et al. (2018)

Kenton Lee, Luheng He, and Luke Zettlemoyer. 2018.


[Higher-order
coreference resolution with coarse-to-fine inference](https://doi.org/10.18653/v1/N18-2108).


In Proceedings of the 2018 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 2 (Short Papers), pages 687–692, New Orleans,
Louisiana. Association for Computational Linguistics.


- Ng and Cardie (2002)

Vincent Ng and Claire Cardie. 2002.


[Improving machine
learning approaches to coreference resolution](https://doi.org/10.3115/1073083.1073102).


In Proceedings of the 40th Annual Meeting of the Association
for Computational Linguistics, pages 104–111, Philadelphia, Pennsylvania,
USA. Association for Computational Linguistics.


- Peters et al. (2018)

Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark,
Kenton Lee, and Luke Zettlemoyer. 2018.


[Deep contextualized
word representations](https://doi.org/10.18653/v1/N18-1202).


In Proceedings of the 2018 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long Papers), pages 2227–2237, New Orleans,
Louisiana. Association for Computational Linguistics.


- Pradhan et al. (2012)

Sameer Pradhan, Alessandro Moschitti, Nianwen Xue, Olga Uryupina, and Yuchen
Zhang. 2012.


[CoNLL-2012
shared task: Modeling multilingual unrestricted coreference in
OntoNotes](https://www.aclweb.org/anthology/W12-4501).


In Joint Conference on EMNLP and CoNLL - Shared Task,
pages 1–40, Jeju Island, Korea. Association for Computational Linguistics.


- Wiseman et al. (2015)

Sam Wiseman, Alexander M. Rush, Stuart Shieber, and Jason Weston. 2015.


[Learning anaphoricity
and antecedent ranking features for coreference resolution](https://doi.org/10.3115/v1/P15-1137).


In Proceedings of the 53rd Annual Meeting of the Association
for Computational Linguistics and the 7th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers), pages 1416–1426,
Beijing, China. Association for Computational Linguistics.


- Wiseman et al. (2016)

Sam Wiseman, Alexander M. Rush, and Stuart M. Shieber. 2016.


[Learning global
features for coreference resolution](https://doi.org/10.18653/v1/N16-1114).


In Proceedings of the 2016 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, pages 994–1004, San Diego, California. Association for
Computational Linguistics.


- Wu et al. (2020)

Wei Wu, Fei Wang, Arianna Yuan, Fei Wu, and Jiwei Li. 2020.


[CorefQA:
Coreference resolution as query-based span prediction](https://doi.org/10.18653/v1/2020.acl-main.622).


In Proceedings of the 58th Annual Meeting of the Association
for Computational Linguistics, pages 6953–6963, Online. Association for
Computational Linguistics.


- Yu et al. (2020)

Juntao Yu, Alexandra Uma, and Massimo Poesio. 2020.


[A cluster
ranking model for full anaphora resolution](https://www.aclweb.org/anthology/2020.lrec-1.2).


In Proceedings of The 12th Language Resources and Evaluation
Conference, pages 11–20, Marseille, France. European Language Resources
Association.


| | MUC | | B3 | | CEAF$\phi_{4}$ | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | F1 | | F1 | | F1 | | Avg. F1 |
| BERT | 83.7 ($\pm$ 0.1) | | 75.5 ($\pm$ 0.1) | | 72.6 ($\pm$ 0.1) | | 77.3 ($\pm$ 0.1) |
| SpanBERT | 85.3 ($\pm$ 0.1) | | 78.4 ($\pm$ 0.1) | | 75.5 ($\pm$ 0.3) | | 79.7 ($\pm$ 0.1) |
| + AA | 85.2 ($\pm$ 0.2) | | 78.1 ($\pm$ 0.2) | | 75.0 ($\pm$ 0.2) | | 79.4 ($\pm$ 0.2) |
| + EE | 85.0 ($\pm$ 0.1) | | 77.7 ($\pm$ 0.2) | | 74.7 ($\pm$ 0.2) | | 78.9 ($\pm$ 0.4) |
| + SC | 85.1 ($\pm$ 0.2) | | 77.9 ($\pm$ 0.3) | | 74.7 ($\pm$ 0.3) | | 79.2 ($\pm$ 0.3) |
| + CM | 85.5 ($\pm$ 0.2) | | 78.5 ($\pm$ 0.3) | | 75.6 ($\pm$ 0.2) | | 79.9 ($\pm$ 0.2) |


Table 4: Results on the test set of the CoNLL’12 English shared task data. Macro-average is reported for each F1 score from 5 repeated developments of each approach. See Section [4](https://ar5iv.labs.arxiv.org/html/2009.12013#S4) for the approaches.


<a id="source-section-22"></a>

## Appendix A Appendices


<a id="source-section-23"></a>

### A.1 Experimental Settings


We implement the experimented models using PyTorch. BERTLarge and SpanBERTLarge are used as encoders. For each experiment, the best performed model on the development set is selected and evaluated on the test set.


<a id="source-section-24"></a>

#### Hyperparameters and Implementation


Similar to Joshi et al. ([2019](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib6), [2020](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib5)), documents are split into independent segments with maximum 384 word pieces for BERTLarge and 512 for SpanBERTLarge. In our final setting, BERT-parameters and task-parameters have separate learning rates ($1\times 10^{-5}$ and $3\times 10^{-4}$ respectively), separate linear decay schedule, and separate weight decay rates ($10^{-2}$ and $0$ respectively). Models are trained 24 epochs with dropout rate 0.3.


The implementation of EE is based on the Tensorflow implementation from Kantor and Globerson ([2019](https://ar5iv.labs.arxiv.org/html/2009.12013#bib.bib7)) which requires $\mathcal{O}(k^{2})$ memory with $k$ being the number of extracted spans, while other HOI approaches only requires $\mathcal{O}(k)$ memory 444The maximum number of antecedents for all models is set to 50 which is constant.. To keep the GPU memory usage within 32GB, we limit the maximum number of span candidates for EE to be 300, which may have a negative impact on the performance.


Experiments are conducted on Nvidia Tesla V100 GPUs with 32GB memory. The average training time is around 7 hours for BERT and SpanBERT without HOI, and ranges from 9 - 15 hours with HOI methods.


<a id="source-section-25"></a>

### A.2 Results


Table [4](https://ar5iv.labs.arxiv.org/html/2009.12013#A0.T4) reports the macro-average F1 scores out of 5 repeated developments of each approach. CM still has the best performance with 79.9 averaged F1 score. Span refinement-based HOI approaches, AA, EE, and SC, still have lower F1 scores than the local-only SpanBERT.


We do not find different configurations for CM make any huge impact to the performance. The final configuration for CM is sequential order and max reduction (Algorithm [1](https://ar5iv.labs.arxiv.org/html/2009.12013#alg1)).


<a id="source-section-26"></a>

### A.3 Analysis


| AA | -0.02 ($\pm$ 0.06) |
| --- | --- |
| EE | 0.03 ($\pm$ 0.07) |
| SC | 0.11 ($\pm$ 0.10) |
| CM | 0.04 ($\pm$ 0.04) |


Table 5: Performance drop on CoNLL’12 English test set after turning off the corresponding HOI in trained models.


Table [5](https://ar5iv.labs.arxiv.org/html/2009.12013#A1.T5) shows the averaged performance drop and its standard deviations w.r.t Avg-F1 after turning off the corresponding HOI in trained models, to see the direct performance impact of HOI over local decisions.


<a id="source-section-27"></a>

#### Pronoun Resolution


In our analysis, the following personal pronouns are regarded as ambiguous pronouns: “you”, “your”, “yours”.

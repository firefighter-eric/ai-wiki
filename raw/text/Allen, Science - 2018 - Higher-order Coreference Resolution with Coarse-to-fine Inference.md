# Allen, Science - 2018 - Higher-order Coreference Resolution with Coarse-to-fine Inference

- Source HTML: `raw/html/Allen, Science - 2018 - Higher-order Coreference Resolution with Coarse-to-fine Inference.html`
- Source SHA256: `6633ce0bc3f94f751d4349d926a65de5dc619abb53380d8386725a75a92b8885`
- Source URL: https://ar5iv.labs.arxiv.org/html/1804.05392v1
- Generated from: `scripts/fetch_web_text.py`
- Extraction: `structured-html-v2` (headings, links, MathML/TeX and tables; figures require visual review)

## Extracted Text

<a id="source-section-0"></a>

<a id="source-section-1"></a>

# Higher-order Coreference Resolution with Coarse-to-fine Inference


Kenton Lee   Luheng He   Luke Zettlemoyer

Paul G. Allen School of Computer Science & Engineering

University of Washington, Seattle WA

{kentonl, luheng, lsz}@cs.washington.edu


<a id="source-section-2"></a>

###### Abstract


We introduce a fully differentiable approximation to higher-order inference for coreference resolution. Our approach uses the antecedent distribution from a span-ranking architecture as an attention mechanism to iteratively refine span representations. This enables the model to softly consider multiple hops in the predicted clusters. To alleviate the computational cost of this iterative process, we introduce a coarse-to-fine approach that incorporates a less accurate but more efficient bilinear factor, enabling more aggressive pruning without hurting accuracy. Compared to the existing state-of-the-art span-ranking approach, our model significantly improves accuracy on the English OntoNotes benchmark, while being far more computationally efficient.


<a id="source-section-3"></a>

## 1 Introduction


Recent coreference resolution systems have heavily relied on first order models Clark and Manning ([2016a](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib4)); Lee et al. ([2017](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib10)), where only pairs of entity mentions are scored by the model. These models are computationally efficient and scalable to long documents. However, because they make independent decisions about coreference links, they are susceptible to predicting clusters that are locally consistent but globally inconsistent. Figure [1](https://ar5iv.labs.arxiv.org/html/1804.05392v1#S1.F1) shows an example from Wiseman et al. ([2016](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib16)) that illustrates this failure case. The plurality of [you] is underspecified, making it locally compatible with both [I] and [all of you], while the full cluster would have mixed plurality, resulting in global inconsistency.


We introduce an approximation of higher-order inference that uses the span-ranking architecture from Lee et al. ([2017](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib10)) in an iterative manner. At each iteration, the antecedent distribution is used as an attention mechanism to optionally update existing span representations, enabling later coreference decisions to softly condition on earlier coreference decisions. For the example in Figure [1](https://ar5iv.labs.arxiv.org/html/1804.05392v1#S1.F1), this enables the linking of [you] and [all of you] to depend on the linking of [I] and [you].


To alleviate computational challenges from this higher-order inference, we also propose a coarse-to-fine approach that is learned with a single end-to-end objective. We introduce a less accurate but more efficient coarse factor in the pairwise scoring function. This additional factor enables an extra pruning step during inference that reduces the number of antecedents considered by the more accurate but inefficient fine factor. Intuitively, the model cheaply computes a rough sketch of likely antecedents before applying a more expensive scoring function.


Our experiments show that both of the above contributions improve the performance of coreference resolution on the English OntoNotes benchmark. We observe a significant increase in average F1 with a second-order model, but returns quickly diminish with a third-order model. Additionally, our analysis shows that the coarse-to-fine approach makes the model performance relatively insensitive to more aggressive antecedent pruning, compared to the distance-based heuristic pruning from previous work.


Speaker 1: Um and [I] think that is what’s - Go ahead Linda.
Speaker 2: Well and uh thanks goes to [you] and to the media to help us… So our hat is off to [all of you] as well.


Figure 1: Example of consistency errors to which first-order span-ranking models are susceptible. Span pairs (I, you) and (you, all of you) are locally consistent, but the span triplet (I, you, all of you) is globally inconsistent. Avoiding this error requires modeling higher-order structures.


<a id="source-section-4"></a>

## 2 Background


<a id="source-section-5"></a>

#### Task definition


We formulate the coreference resolution task as a set of antecedent assignments $y_{i}$ for each of span $i$ in the given document, following Lee et al. ([2017](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib10)). The set of possible assignments for each $y_{i}$ is $\mathcal{Y}(i)=\{\epsilon,1,\ldots,i-1\}$, a dummy antecedent $\epsilon$ and all preceding spans. Non-dummy antecedents represent coreference links between $i$ and $y_{i}$. The dummy antecedent $\epsilon$ represents two possible scenarios: (1) the span is not an entity mention or (2) the span is an entity mention but it is not coreferent with any previous span. These decisions implicitly define a final clustering, which can be recovered by grouping together all spans that are connected by the set of antecedent predictions.


<a id="source-section-6"></a>

#### Baseline


We describe the baseline model Lee et al. ([2017](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib10)), which we will improve to address the modeling and computational limitations discussed previously. The goal is to learn a distribution $P(y_{i})$ over antecedents for each span $i$ :


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | $\displaystyle P(y_{i})$ | $\displaystyle=\frac{e^{s(i,y_{i})}}{\sum_{y^{\prime}\in\mathcal{Y}(i)}e^{s(i,y^{\prime})}}$ | | (1) |


where $s(i,j)$ is a pairwise score for a coreference link between span $i$ and span $j$. The baseline model includes three factors for this pairwise coreference score: (1) $s_{\text{m}}(i)$, whether span $i$ is a mention, (2) $s_{\text{m}}(j)$, whether span $j$ is a mention, and (3) $s_{\text{a}}(i,j)$ whether $j$ is an antecedent of $i$:


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | $\displaystyle s(i,j)$ | $\displaystyle=s_{\text{m}}(i)+s_{\text{m}}(j)+s_{\text{a}}(i,j)$ | | (2) |


In the special case of the dummy antecedent, the score $s(i,\epsilon)$ is instead fixed to 0. A common component used throughout the model is the vector representations $\bm{g}_{i}$ for each possible span $i$. These are computed via bidirectional LSTMs Hochreiter and Schmidhuber ([1997](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib9)) that learn context-dependent boundary and head representations. The scoring functions $s_{\text{m}}$ and $s_{\text{a}}$ take these span representations as input:


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | $\displaystyle s_{\text{m}}(i)$ | $\displaystyle=\bm{w}_{\text{m}}^{\top}\textsc{ffnn}_{\text{m}}(\bm{g}_{i})$ | | (3) |
| | $\displaystyle s_{\text{a}}(i,j)$ | $\displaystyle=\bm{w}_{\text{a}}^{\top}\textsc{ffnn}_{\text{a}}([\bm{g}_{i},\bm{g}_{j},\bm{g}_{i}\circ\bm{g}_{j},\phi(i,j)])$ | | (4) |


where $\circ$ denotes element-wise multiplication, ffnn denotes a feed-forward neural network, and the antecedent scoring function $s_{\text{a}}(i,j)$ includes explicit element-wise similarity of each span $\bm{g}_{i}\circ\bm{g}_{j}$ and a feature vector $\phi(i,j)$ encoding speaker and genre information from the metadata and the distance between the two spans.


The model above is factored to enable a two-stage beam search. A beam of up to $M$ potential mentions is computed (where $M$ is proportional to the document length) based on the spans with the highest mention scores $s_{\text{m}}(i)$. Pairwise coreference scores are only computed between surviving mentions during both training and inference.


Given supervision of gold coreference clusters, the model is learned by optimizing the marginal log-likelihood of the possibly correct antecedents. This marginalization is required since the best antecedent for each span is a latent variable.


<a id="source-section-7"></a>

## 3 Higher-order Coreference Resolution


The baseline above is a first-order model, since it only considers pairs of spans. First-order models are susceptible to consistency errors as demonstrated in Figure [1](https://ar5iv.labs.arxiv.org/html/1804.05392v1#S1.F1). Unlike in sentence-level semantics, where higher-order decisions can be implicitly modeled by the LSTMs, modeling these decisions at the document-level requires explicit inference due to the potentially very large surface distance between mentions.


We propose an inference procedure that allows the model to condition on higher-order structures, while being fully differentiable. This inference involves $N$ iterations of refining span representations, denoted as $\bm{g}_{i}^{n}$ for the representation of span $i$ at iteration $n$. At iteration $n$, $\bm{g}_{i}^{n}$ is computed with an attention mechanism that averages over previous representations $\bm{g}_{j}^{n-1}$ weighted according to how likely each mention $j$ is to be an antecedent for $i$, as defined below.


The baseline model is used to initialize the span representation at $\bm{g}_{i}^{1}$. The refined span representations allow the model to also iteratively refine the antecedent distributions $P_{n}(y_{i})$:


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | $\displaystyle P_{n}(y_{i})$ | $\displaystyle=\frac{e^{s(\bm{g}_{i}^{n},\bm{g}_{y_{i}}^{n})}}{\sum_{y\in\mathcal{Y}(i)}e^{s(\bm{g}_{i}^{n},\bm{g}_{y}^{n}))}}$ | | (5) |


where $s$ is the coreference scoring function of the baseline architecture. The scoring function uses the same parameters at every iteration, but it is given different span representations.


At each iteration, we first compute the expected antecedent representation $\bm{a}_{i}^{n}$ of each span $i$ by using the current antecedent distribution $P_{n}(y_{i})$ as an attention mechanism:


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | $\displaystyle\bm{a}_{i}^{n}$ | $\displaystyle=\sum_{y_{i}\in\mathcal{Y}(i)}P_{n}(y_{i})\cdot\bm{g}_{y_{i}}^{n}$ | | (6) |


The current span representation $\bm{g}_{i}^{n}$ is then updated via interpolation with its expected antecedent representation $\bm{a}_{i}^{n}$:


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | $\displaystyle\bm{f}_{i}^{n}$ | $\displaystyle=\sigma(\mathbf{W}_{\text{f}}[\bm{g}_{i}^{n},\bm{a}_{i}^{n}])$ | | (7) |
| | $\displaystyle\bm{g}_{i}^{n+1}$ | $\displaystyle=\bm{f}_{i}^{n}\circ\bm{g}_{i}^{n}+(\bm{1}-\bm{f}_{i}^{n})\circ\bm{a}_{i}^{n}$ | | (8) |


The learned gate vector $\bm{f}_{i}^{n}$ determines for each dimension whether to keep the current span information or to integrate new information from its expected antecedent.
At iteration $n$, $\bm{g}_{i}^{n}$ is an element-wise weighted average of approximately $n$ span representations (assuming $P_{n}(y_{i})$ is peaked), allowing $P_{n}(y_{i})$ to softly condition on up to $n$ other spans in the predicted cluster.


Span-ranking can be viewed as predicting latent antecedent trees Fernandes et al. ([2012](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib8)); Martschat and Strube ([2015](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib11)), where the predicted antecedent is the parent of a span and each tree is a predicted cluster. By iteratively refining the span representations and antecedent distributions, another way to interpret this model is that the joint distribution $\prod_{i}P_{N}(y_{i})$ implicitly models every directed path of up to length $N+1$ in the latent antecedent tree.


<a id="source-section-8"></a>

## 4 Coarse-to-fine Inference


The model described above scales poorly to long documents. Despite heavy pruning of potential mentions, the space of possible antecedents for every surviving span is still too large to fully consider. The bottleneck is in the antecedent score $s_{\text{a}}(i,j)$, which requires computing a tensor of size $M\times M\times(3|\bm{g}|+|\phi|)$.


This computational challenge is even more problematic with the iterative inference from Section [3](https://ar5iv.labs.arxiv.org/html/1804.05392v1#S3), which requires recomputing this tensor at every iteration.


<a id="source-section-9"></a>

### 4.1 Heuristic antecedent pruning


To reduce computation, Lee et al. ([2017](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib10)) heuristically consider only the nearest $K$ antecedents of each span, resulting in a smaller input of size $M\times K\times(3|\bm{g}|+|\phi|)$.


The main drawback to this solution is that it imposes an a priori limit on the maximum distance of a coreference link. The previous work only considers up to $K=250$ nearest mentions, whereas coreference links can reach much further in natural language discourse.


Figure 2: Comparison of accuracy on the development set for the two antecedent pruning strategies with various beams sizes $K$. The distance-based heuristic pruning performance drops by almost 5 F1 when reducing $K$ from 250 to 50, while the coarse-to-fine pruning results in an insignificant drop of less than 0.2 F1.


<a id="source-section-10"></a>

### 4.2 Coarse-to-fine antecedent pruning


We instead propose a coarse-to-fine approach that can be learned end-to-end and does not establish an a priori maximum coreference distance. The key component of this coarse-to-fine approach is an alternate bilinear scoring function:


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | $\displaystyle s_{\text{c}}(i,j)$ | $\displaystyle=\bm{g}_{i}^{\top}\mathbf{W}_{\text{c}}\;\bm{g}_{j}$ | | (9) |


where $\mathbf{W}_{\text{c}}$ is a learned weight matrix. In contrast to the concatenation-based $s_{\text{a}}(i,j)$, the bilinear $s_{\text{c}}(i,j)$ is far less accurate. A direct replacement of $s_{\text{a}}(i,j)$ with $s_{\text{c}}(i,j)$ results in a performance loss of over 3 F1 in our experiments. However, $s_{\text{c}}(i,j)$ is much more efficient to compute. Computing $s_{\text{c}}(i,j)$ only requires manipulating matrices of size $M\times|\bm{g}|$ and $M\times M$.


| | MUC [colspan=4] | $\text{B}^{3}$ [colspan=4] | $\text{CEAF}_{\phi_{4}}$ [colspan=4] | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | Prec. | Rec. | F1 | | Prec. | Rec. | F1 | | Prec. | Rec. | F1 | | Avg. F1 |
| Martschat and Strube ([2015](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib11)) | 76.7 | 68.1 | 72.2 | | 66.1 | 54.2 | 59.6 | | 59.5 | 52.3 | 55.7 | | 62.5 |
| Clark and Manning ([2015](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib3)) | 76.1 | 69.4 | 72.6 | | 65.6 | 56.0 | 60.4 | | 59.4 | 53.0 | 56.0 | | 63.0 |
| Wiseman et al. ([2015](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib17)) | 76.2 | 69.3 | 72.6 | | 66.2 | 55.8 | 60.5 | | 59.4 | 54.9 | 57.1 | | 63.4 |
| Wiseman et al. ([2016](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib16)) | 77.5 | 69.8 | 73.4 | | 66.8 | 57.0 | 61.5 | | 62.1 | 53.9 | 57.7 | | 64.2 |
| Clark and Manning ([2016b](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib5)) | 79.9 | 69.3 | 74.2 | | 71.0 | 56.5 | 63.0 | | 63.8 | 54.3 | 58.7 | | 65.3 |
| Clark and Manning ([2016a](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib4)) | 79.2 | 70.4 | 74.6 | | 69.9 | 58.0 | 63.4 | | 63.5 | 55.5 | 59.2 | | 65.7 |
| Lee et al. ([2017](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib10)) | 78.4 | 73.4 | 75.8 | | 68.6 | 61.8 | 65.0 | | 62.7 | 59.0 | 60.8 | | 67.2 |
| + ELMo Peters et al. ([2018](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib14)) | 80.1 | 77.2 | 78.6 | | 69.8 | 66.5 | 68.1 | | 66.4 | 62.9 | 64.6 | | 70.4 |
| + hyperparameter tuning | 80.7 | 78.8 | 79.8 | | 71.7 | 68.7 | 70.2 | | 67.2 | 66.8 | 67.0 | | 72.3 |
| + coarse-to-fine inference | 80.4 | 79.9 | 80.1 | | 71.0 | 70.0 | 70.5 | | 67.5 | 67.2 | 67.3 | | 72.6 |
| + second-order inference | 81.4 | 79.5 | 80.4 | | 72.2 | 69.5 | 70.8 | | 68.2 | 67.1 | 67.6 | | 73.0 |


Table 1: Results on the test set on the English CoNLL-2012 shared task. The average F1 of MUC, $\text{B}^{3}$, and $\text{CEAF}_{\phi_{4}}$is the main evaluation metric. We show only non-ensembled models for fair comparison.


Therefore, we instead propose to use $s_{\text{c}}(i,j)$ to compute a rough sketch of likely antecedents. This is accomplished by including it as an additional factor in the model:


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | $\displaystyle s(i,j)$ | $\displaystyle=s_{\text{m}}(i)+s_{\text{m}}(j)+s_{\text{c}}(i,j)+s_{\text{a}}(i,j)$ | | (10) |


Similar to the baseline model, we leverage this additional factor to perform an additional beam pruning step. The final inference procedure involves a three-stage beam search:


<a id="source-section-11"></a>

#### First stage


Keep the top $M$ spans based on the mention score $s_{\text{m}}(i)$ of each span.


<a id="source-section-12"></a>

#### Second stage


Keep the top $K$ antecedents of each remaining span $i$ based on the first three factors, $s_{\text{m}}(i)+s_{\text{m}}(j)+s_{\text{c}}(i,j)$.


<a id="source-section-13"></a>

#### Third stage


The overall coreference $s(i,j)$ is computed based on the remaining span pairs. The soft higher-order inference from Section [3](https://ar5iv.labs.arxiv.org/html/1804.05392v1#S3) is computed in this final stage.


While the maximum-likelihood objective is computed over only the span pairs from this final stage, this coarse-to-fine approach expands the set of coreference links that the model is capable of learning. It achieves better performance while using a much smaller $K$ (see Figure [2](https://ar5iv.labs.arxiv.org/html/1804.05392v1#S4.F2)).


<a id="source-section-14"></a>

## 5 Experimental Setup


We use the English coreference resolution data from the CoNLL-2012 shared task Pradhan et al. ([2012](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib15)) in our experiments. The code for replicating these results is publicly available.111[https://github.com/kentonl/e2e-coref](https://github.com/kentonl/e2e-coref)


Our models reuse the hyperparameters from  Lee et al. ([2017](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib10)), with a few exceptions mentioned below. In our results, we report two improvements that are orthogonal to our contributions.


- •


We used embedding representations from a language model Peters et al. ([2018](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib14)) at the input to the LSTMs (ELMo in the results).


- •


We changed several hyperparameters:


- 1.


increasing the maximum span width from 10 to 30 words.


- 2.


using 3 highway LSTMs instead of 1.


- 3.


using GloVe word embeddings  Pennington et al. ([2014](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib13)) with a window size of 2 for the head word embeddings and a window size of 10 for the LSTM inputs.


The baseline model considers up to 250 antecedents per span. As shown in Figure [2](https://ar5iv.labs.arxiv.org/html/1804.05392v1#S4.F2), the coarse-to-fine model is quite insensitive to more aggressive pruning. Therefore, our final model considers only 50 antecedents per span.


On the development set, the second-order model ($N=2$) outperforms the first-order model by 0.8 F1, but the third order model only provides an additional 0.1 F1 improvement. Therefore, we only compute test results for the second-order model.


<a id="source-section-15"></a>

## 6 Results


We report the precision, recall, and F1 of the the MUC, $\text{B}^{3}$, and $\text{CEAF}_{\phi_{4}}$metrics using the official CoNLL-2012 evaluation scripts. The main evaluation is the average F1 of the three metrics.


Results on the test set are shown in Table [1](https://ar5iv.labs.arxiv.org/html/1804.05392v1#S4.T1). We include performance of systems proposed in the past 3 years for reference. The baseline relative to our contributions is the span-ranking model from  Lee et al. ([2017](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib10)) augmented with both ELMo and hyperparameter tuning, which achieves 72.3 F1. Our full approach achieves 73.0 F1, setting a new state of the art for coreference resolution.


Compared to the heuristic pruning with up to 250 antecedents, our coarse-to-fine model only computes the expensive scores $s_{\text{a}}(i,j)$ for 50 antecedents. Despite using far less computation, it outperforms the baseline because the coarse scores $s_{\text{c}}(i,j)$ can be computed for all antecedents, enabling the model to potentially predict a coreference link between any two spans in the document. As a result, we observe a much higher recall when adopting the coarse-to-fine approach.


We also observe further improvement by including the second-order inference (Section [3](https://ar5iv.labs.arxiv.org/html/1804.05392v1#S3)). The improvement is largely driven by the overall increase in precision, which is expected since the higher-order inference mainly serves to rule out inconsistent clusters. It is also consistent with findings from Martschat and Strube ([2015](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib11)) who report mainly improvements in precision when modeling latent trees to achieve a similar goal.


<a id="source-section-16"></a>

## 7 Related Work


In addition to the end-to-end span-ranking model Lee et al. ([2017](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib10)) that our proposed model builds upon, there is a large body of literature on coreference resolvers that fundamentally rely on scoring span pairs Ng and Cardie ([2002](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib12)); Bengtson and Roth ([2008](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib1)); Denis and Baldridge ([2008](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib6)); Fernandes et al. ([2012](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib8)); Durrett and Klein ([2013](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib7)); Wiseman et al. ([2015](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib17)); Clark and Manning ([2016a](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib4)).


Motivated by structural consistency issues discussed above, significant effort has also been devoted towards cluster-level modeling. Since global features are notoriously difficult to define Wiseman et al. ([2016](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib16)), they often depend heavily on existing pairwise features or architectures Björkelund and Kuhn ([2014](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib2)); Clark and Manning ([2015](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib3), [2016b](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib5)). We similarly use an existing pairwise span-ranking architecture as a building block for modeling more complex structures. In contrast to  Wiseman et al. ([2016](https://ar5iv.labs.arxiv.org/html/1804.05392v1#bib.bib16)) who use highly expressive recurrent neural networks to model clusters, we show that the addition of a relatively lightweight gating mechanism is sufficient to effectively model higher-order structures.


<a id="source-section-17"></a>

## 8 Conclusion


We presented a state-of-the-art coreference resolution system that models higher order interactions between spans in predicted clusters. Additionally, our proposed coarse-to-fine approach alleviates the additional computational cost of higher-order inference, while maintaining the end-to-end learnability of the entire model.


<a id="source-section-18"></a>

### Acknowledgements


The research was supported in part by DARPA under the DEFT program (FA8750-13-2-0019), the ARO (W911NF-16-1-0121), the NSF (IIS-1252835, IIS-1562364), gifts from Google and Tencent, and an Allen Distinguished Investigator Award. We also thank the UW NLP group for helpful conversations and comments on the work.


<a id="source-section-19"></a>

## References


- Bengtson and Roth (2008)

Eric Bengtson and Dan Roth. 2008.


Understanding the value of features for coreference resolution.


In EMNLP.


- Björkelund and Kuhn (2014)

Anders Björkelund and Jonas Kuhn. 2014.


Learning structured perceptrons for coreference resolution with
latent antecedents and non-local features.


In ACL.


- Clark and Manning (2015)

Kevin Clark and Christopher D. Manning. 2015.


Entity-centric coreference resolution with model stacking.


In ACL.


- Clark and Manning (2016a)

Kevin Clark and Christopher D. Manning. 2016a.


Deep reinforcement learning for mention-ranking coreference models.


In EMNLP.


- Clark and Manning (2016b)

Kevin Clark and Christopher D. Manning. 2016b.


Improving coreference resolution by learning entity-level distributed
representations.


In ACL.


- Denis and Baldridge (2008)

Pascal Denis and Jason Baldridge. 2008.


Specialized models and ranking for coreference resolution.


In EMNLP.


- Durrett and Klein (2013)

Greg Durrett and Dan Klein. 2013.


Easy victories and uphill battles in coreference resolution.


In EMNLP.


- Fernandes et al. (2012)

Eraldo Rezende Fernandes, Cícero Nogueira Dos Santos, and Ruy Luiz
Milidiú. 2012.


Latent structure perceptron with feature induction for unrestricted
coreference resolution.


In CoNLL.


- Hochreiter and Schmidhuber (1997)

Sepp Hochreiter and Jürgen Schmidhuber. 1997.


Long Short-term Memory.


Neural computation .


- Lee et al. (2017)

Kenton Lee, Luheng He, Mike Lewis, and Luke S. Zettlemoyer. 2017.


End-to-end neural coreference resolution.


In EMNLP.


- Martschat and Strube (2015)

Sebastian Martschat and Michael Strube. 2015.


Latent structures for coreference resolution.


TACL .


- Ng and Cardie (2002)

Vincent Ng and Claire Cardie. 2002.


Identifying anaphoric and non-anaphoric noun phrases to improve
coreference resolution.


Computational linguistics .


- Pennington et al. (2014)

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.


Glove: Global vectors for word representation.


In EMNLP.


- Peters et al. (2018)

Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark,
Kenton Lee, and Luke Zettlemoyer. 2018.


Deep contextualized word representations.


In HLT-NAACL.


- Pradhan et al. (2012)

Sameer Pradhan, Alessandro Moschitti, Nianwen Xue, Olga Uryupina, and Yuchen
Zhang. 2012.


Conll-2012 shared task: Modeling multilingual unrestricted
coreference in ontonotes.


In CoNLL.


- Wiseman et al. (2016)

Sam Wiseman, Alexander M Rush, and Stuart M Shieber. 2016.


Learning global features for coreference resolution.


In NAACL-HLT.


- Wiseman et al. (2015)

Sam Wiseman, Alexander M. Rush, Stuart M. Shieber, and Jason Weston. 2015.


Learning anaphoricity and antecedent ranking features for coreference
resolution.


In ACL.

# Liu - 2019 - Fine-tune BERT for Extractive Summarization

- Source HTML: `raw/html/Liu - 2019 - Fine-tune BERT for Extractive Summarization.html`
- Source SHA256: `7bd7dc8a2821512b85226cbd1168fe4274ff8a97871ce775b5587f75df87ee42`
- Source URL: https://ar5iv.labs.arxiv.org/html/1903.10318
- Generated from: `scripts/fetch_web_text.py`
- Extraction: `structured-html-v2` (headings, links, MathML/TeX and tables; figures require visual review)

## Extracted Text

<a id="source-section-0"></a>

<a id="source-section-1"></a>

# Fine-tune BERT for Extractive Summarization


Yang Liu

Institute for Language, Cognition and Computation


School of Informatics, University of Edinburgh


10 Crichton Street, Edinburgh EH8 9AB

yang.liu2@ed.ac.uk


<a id="source-section-2"></a>

###### Abstract


BERT (Devlin et al., [2018](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib6)), a pre-trained Transformer (Vaswani et al., [2017](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib16)) model, has achieved ground-breaking performance on multiple NLP tasks.
In
this paper, we describe Bertsum, a simple variant of BERT, for extractive summarization. Our system is the state of the art on the CNN/Dailymail dataset, outperforming the previous best-performed system by 1.65 on ROUGE-L. The codes
to reproduce our results are available at [https://github.com/nlpyang/BertSum](https://github.com/nlpyang/BertSum)


22footnotetext: Please see [https://arxiv.org/abs/1908.08345](https://arxiv.org/abs/1908.08345) for the full and most current version of this
paper


<a id="source-section-3"></a>

## 1 Introduction


Single-document summarization is the task of automatically generating
a shorter version of a document while retaining its most important
information. The task has received much attention in the natural
language processing community due to its potential for various
information access applications. Examples include tools which digest
textual content (e.g., news, social media, reviews), answer
questions, or provide recommendations.


The task is often divided into two paradigms, abstractive
summarization and extractive summarization.
In abstractive
summarization, target summaries contains words or phrases that were not in the original text and usually require various text rewriting operations to generate, while extractive approaches
form summaries by
copying and concatenating the most important spans (usually
sentences) in a document. In this paper, we focus on extractive summarization.


Although many neural models have been proposed for extractive summarization recently (Cheng and Lapata, [2016](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib5); Nallapati et al., [2017](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib11); Narayan et al., [2018](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib12); Dong et al., [2018](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib7); Zhang et al., [2018](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib17); Zhou et al., [2018](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib18)), the improvement on automatic metrics like ROUGE has reached a bottleneck due to the complexity of the task.
In this paper, we argue that, BERT (Devlin et al., [2018](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib6)), with its pre-training on a huge dataset and the powerful architecture for learning complex features, can further boost the performance of extractive summarization .


In this paper, we focus on designing different variants of using BERT on the extractive summarization task and showing their results on CNN/Dailymail and NYT datasets.
We found that a flat architecture with inter-sentence Transformer layers performs the best, achieving the state-of-the-art results on this task.


[图片：Refer to caption]


Figure 1: The overview architecture of the Bertsum model.


<a id="source-section-4"></a>

## 2 Methodology


Let $d$ denote a document containing several sentences
$[sent_{1},sent_{2},\cdots,sent_{m}]$, where $sent_{i}$ is the $i$-th
sentence in the document. Extractive summarization can be defined as
the task of assigning a label $y_{i}\in\{0,1\}$ to each $sent_{i}$,
indicating whether the sentence should be included in the summary. It
is assumed that summary sentences represent the most important content of the document.


<a id="source-section-5"></a>

### 2.1 Extractive Summarization with BERT


To use BERT for extractive summarization, we require it to output the representation for each sentence. However, since BERT is trained as a masked-language model, the output vectors are grounded to tokens instead of sentences. Meanwhile, although BERT has segmentation embeddings for indicating different sentences, it only has two labels (sentence A or sentence B), instead of multiple sentences as in extractive summarization.
Therefore, we modify the input sequence and embeddings of BERT to make it possible for extracting summaries.


<a id="source-section-6"></a>

#### Encoding Multiple Sentences


As illustrated in Figure 1, we insert a [CLS] token before each sentence and a [SEP] token after each sentence.
In vanilla BERT, The [CLS] is used as a symbol to aggregate features from one sentence or a pair of sentences. We modify the model by using multiple [CLS] symbols to get features for sentences ascending the symbol.


<a id="source-section-7"></a>

#### Interval Segment Embeddings


We use interval segment embeddings to distinguish multiple sentences within a document. For $sent_{i}$ we will assign a segment embedding $E_{A}$ or $E_{B}$ conditioned on $i$ is odd or even. For example, for $[sent_{1},sent_{2},sent_{3},sent_{4},sent_{5}]$ we will assign $[E_{A},E_{B},E_{A},E_{B},E_{A}]$.


The vector $T_{i}$ which is the vector of the $i$-th [CLS] symbol from the top BERT layer will be used as the representation for $sent_{i}$.


<a id="source-section-8"></a>

### 2.2 Fine-tuning with Summarization Layers


After obtaining the sentence vectors from BERT, we build several summarization-specific layers stacked on top of the BERT outputs, to capture document-level features for extracting summaries.
For each sentence $sent_{i}$, we will calculate the final predicted score $\hat{Y}_{i}$.
The loss of the whole model is the Binary Classification Entropy of $\hat{Y}_{i}$ against gold label $Y_{i}$.
These summarization layers are jointly fine-tuned with BERT.


<a id="source-section-9"></a>

#### Simple Classifier


Like in the original BERT paper, the Simple Classifier only adds a linear layer on the BERT outputs and use a sigmoid function to get the predicted score:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>\hat{Y}_{i}=\sigma(W_{o}T_{i}+b_{o})<br>$$ | | (1) |


where $\sigma$ is the Sigmoid function.


<a id="source-section-10"></a>

#### Inter-sentence Transformer


Instead of a simple sigmoid classifier, Inter-sentence Transformer applies more Transformer layers only on sentence representations, extracting document-level features focusing on summarization tasks from the BERT outputs:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>\displaystyle\tilde{h}^{l}=\mathrm{LN}(h^{l-1}+\mathrm{MHAtt}(h^{l-1}))<br>$$ | | (2) |
| | $$<br>\displaystyle h^{l}=\mathrm{LN}(\tilde{h}^{l}+\mathrm{FFN}(\tilde{h}^{l}))<br>$$ | | (3) |


where $h^{0}=\mathrm{PosEmb}(T)$ and $T$ are the sentence vectors output by BERT, $\mathrm{PosEmb}$ is the function of adding positional embeddings (indicating the position of each sentence) to $T$;
$\mathrm{LN}$ is the layer normalization operation Ba et al. ([2016](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib1)); $\mathrm{MHAtt}$ is the multi-head attention operation Vaswani et al. ([2017](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib16));
the superscript $l$ indicates the depth of the stacked layer.


The final output layer is still a sigmoid classifier:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>\hat{Y}_{i}=\sigma(W_{o}h_{i}^{L}+b_{o})<br>$$ | | (4) |


where $h^{L}$ is the vector for $sent_{i}$ from the top layer (the $L$-th layer ) of the Transformer. In experiments, we implemented Transformers with $L=1,2,3$ and found Transformer with $2$ layers performs the best.


<a id="source-section-11"></a>

#### Recurrent Neural Network


Although the Transformer model achieved great results on several tasks, there are evidence that Recurrent Neural Networks still have their advantages, especially when combining with techniques in Transformer Chen et al. ([2018](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib4)). Therefore, we apply an LSTM layer over the BERT outputs to learn summarization-specific features.


To stabilize the training, pergate layer normalization Ba et al. ([2016](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib1)) is applied within each LSTM cell. At time step $i$, the input to the LSTM layer is the BERT output $T_{i}$, and the output is calculated as:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>\displaystyle\left(\begin{tabular}[]{c}$F_{i}$\\<br>$I_{i}$\\<br>$O_{i}$\\<br>$G_{i}$\end{tabular}\right)=\mathrm{LN}_{h}(W_{h}h_{i-1})+\mathrm{LN}_{x}(W_{x}T_{i})<br>$$ | | (9) |
| | $$<br>\displaystyle\begin{aligned} C_{i}=&~{}\sigma(F_{i})\odot C_{i-1}\\<br>&+\sigma(I_{i})\odot\mathrm{tanh}(G_{i-1})\\<br>h_{i}=&\sigma(O_{t})\odot\mathrm{tanh}(\mathrm{LN}_{c}(C_{t}))\end{aligned}<br>$$ | | |


where $F_{i},I_{i},O_{i}$ are forget gates, input gates, output gates; $G_{i}$ is the hidden vector and $C_{i}$ is the memory vector; $h_{i}$ is the output vector; $\mathrm{LN}_{h},\mathrm{LN}_{x},\mathrm{LN}_{c}$ are there difference layer normalization operations; Bias terms are not shown.


The final output layer is also a sigmoid classifier:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>\hat{Y}_{i}=\sigma(W_{o}h_{i}+b_{o})<br>$$ | | (10) |


| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | --- | --- | --- |
| Pgn∗ | 39.53 | 17.28 | 37.98 |
| Dca∗ | 41.69 | 19.47 | 37.92 |
| Lead | 40.42 | 17.62 | 36.67 |
| Oracle | 52.59 | 31.24 | 48.87 |
| Refresh∗ | 41.0 | 18.8 | 37.7 |
| Neusum∗ | 41.59 | 19.01 | 37.98 |
| Transformer | 40.90 | 18.02 | 37.17 |
| Bertsum+Classifier | 43.23 | 20.22 | 39.60 |
| Bertsum+Transformer | 43.25 | 20.24 | 39.63 |
| Bertsum+LSTM | 43.22 | 20.17 | 39.59 |


Table 1: Test set results on the CNN/DailyMail dataset using ROUGE $F_{1}$. Results with $*$ mark
are taken from the corresponding papers.


<a id="source-section-12"></a>

## 3 Experiments


In this section we present our implementation, describe the
summarization datasets and our evaluation protocol, and analyze our results.


<a id="source-section-13"></a>

### 3.1 Implementation Details


We use PyTorch, OpenNMT Klein et al. ([2017](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib10)) and the ‘bert-base-uncased’***https://github.com/huggingface/pytorch-pretrained-BERT version of BERT to implement the model.
BERT and summarization layers are jointly fine-tuned.
Adam with $\beta_{1}=0.9$, $\beta_{2}=0.999$ is used for fine-tuning. Learning rate schedule is following Vaswani et al. ([2017](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib16)) with warming-up on first 10,000 steps:


| 列 1 | 列 2 | 列 3 |
| --- | --- | --- |
| | $$<br>lr=2e^{-3}\cdot min(step^{-0.5},step\cdot warmup^{-1.5})<br>$$ | |


All models are trained for 50,000 steps on 3 GPUs (GTX 1080 Ti) with gradient accumulation per two steps, which makes the batch size approximately equal to $36$.
Model checkpoints are saved and evaluated on the validation set every 1,000 steps. We select the top-3 checkpoints based on their evaluation losses on the validations set, and report the averaged results on the test set.


When predicting summaries for a new document, we first use the models to obtain the score for each sentence.
We then rank these sentences by the scores from higher to lower, and select the top-3 sentences as the summary.


<a id="source-section-14"></a>

#### Trigram Blocking


During the predicting process, Trigram Blocking is used to reduce redundancy.
Given selected summary $S$ and a candidate sentence $c$, we will skip $c$ is there exists a trigram overlapping between $c$ and $S$. This is similar to the Maximal Marginal Relevance (MMR) Carbonell and Goldstein ([1998](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib2)) but much simpler.


<a id="source-section-15"></a>

### 3.2 Summarization Datasets


We evaluated on two benchmark datasets, namely the
CNN/DailyMail news highlights dataset Hermann et al. ([2015](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib9)) and
the New York Times Annotated Corpus (NYT; Sandhaus [2008](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib14)).
The CNN/DailyMail dataset contains news articles and associated
highlights, i.e., a few bullet points giving a brief overview of the
article. We used the standard splits of Hermann et al. ([2015](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib9))
for training, validation, and testing (90,266/1,220/1,093 CNN
documents and 196,961/12,148/10,397 DailyMail documents). We did not
anonymize entities.
We first split sentences by CoreNLP and pre-process the dataset following methods in See et al. ([2017](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib15)).


The NYT dataset contains 110,540 articles with abstractive
summaries. Following Durrett et al. ([2016](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib8)), we split these into
100,834 training and 9,706 test examples, based on date of publication
(test is all articles published on January 1, 2007 or later).
We took 4,000 examples from the training set as the validation set.
We also
followed their filtering procedure, documents with summaries that
are shorter than 50 words were removed from the raw dataset. The
filtered test set (NYT50) includes 3,452 test examples.
We first split sentences by CoreNLP and pre-process the dataset following methods in Durrett et al. ([2016](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib8)).


Both datasets contain abstractive gold summaries, which are not
readily suited to training extractive summarization models. A greedy
algorithm was used to
generate an oracle summary for each document. The algorithm
greedily select sentences which can maximize the ROUGE scores as the oracle sentences.
We assigned label 1 to sentences selected in the oracle
summary and 0 otherwise.


<a id="source-section-16"></a>

## 4 Experimental Results


The experimental results on CNN/Dailymail datasets are shown in Table 1.
For comparison, we implement a non-pretrained Transformer baseline which uses the same architecture as BERT, but with smaller parameters. It is randomly initialized and only trained on the summarization task. The Transformer baseline has 6 layers, the hidden size is $512$ and the feed-forward filter size is $2048$. The model is trained with same settings following Vaswani et al. ([2017](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib16)).
We also compare our model with several previously proposed systems.


- •


Lead is an extractive baseline which uses the first-3 sentences of the document as a summary.


- •


Refresh (Narayan et al., [2018](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib12)) is an extractive
summarization system trained by globally optimizing the ROUGE
metric with reinforcement learning.


- •


Neusum (Zhou et al., [2018](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib18)) is the state-of-the-art extractive system that jontly score and select sentences.


- •


Pgn (See et al., [2017](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib15)), is the Pointer Generator Network, an abstractive summarization system based
on an encoder-decoder architecture.


- •


Dca (Celikyilmaz et al., [2018](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib3)) is the Deep Communicating Agents, a
state-of-the-art abstractive summarization system with
multiple agents to represent the document as well as
hierarchical attention mechanism over the agents for decoding.


As illustrated in the table, all BERT-based models outperformed previous state-of-the-art models by a large margin. Bertsum with Transformer achieved the best performance on all three metrics. The Bertsum with LSTM model does not have an obvious influence on the summarization performance compared to the Classifier model.


Ablation studies are conducted to show the contribution of different components of Bertsum. The results are shown in in Table 2. Interval segments increase the performance of base model. Trigram blocking is able to greatly improve the summarization results. This is consistent to previous conclusions that a sequential extractive decoder is helpful to generate more informative summaries. However, here we use the trigram blocking as a simple but robust alternative.


| Model | R-1 | R-2 | R-L |
| --- | --- | --- | --- |
| Bertsum+Classifier | 43.23 | 20.22 | 39.60 |
| -interval segments | 43.21 | 20.17 | 39.57 |
| -trigram blocking | 42.57 | 19.96 | 39.04 |


Table 2: Results of ablation studies of Bertsum on CNN/Dailymail test set using ROUGE $F_{1}$ (R-1 and R-2 are
shorthands for unigram and bigram overlap, R-L is the
longest common subsequence).


The experimental results on NYT datasets are shown in Table 3. Different from CNN/Dailymail, we use the limited-length recall evaluation, following Durrett et al. ([2016](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib8)). We truncate the predicted summaries to the lengths of the gold summaries and evaluate summarization quality with ROUGE Recall.
Compared baselines are (1) First-$k$ words, which is a simple baseline by extracting first $k$ words of the input article; (2) Full is the best-performed extractive model in Durrett et al. ([2016](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib8)); (3) Deep Reinforced Paulus et al. ([2018](https://ar5iv.labs.arxiv.org/html/1903.10318#bib.bib13)) is an abstractive model, using reinforce learning and encoder-decoder structure. The Bertsum+Classifier can achieve the state-of-the-art results on this dataset.


| Model | R-1 | R-2 | R-L |
| --- | --- | --- | --- |
| First-$k$ words | 39.58 | 20.11 | 35.78 |
| Full∗ | 42.2 | 24.9 | - |
| Deep Reinforced∗ | 42.94 | 26.02 | - |
| Bertsum+Classifier | 46.66 | 26.35 | 42.62 |


Table 3: Test set results on the NYT50 dataset using ROUGE Recall. The predicted summary are truncated to the length of the gold-standard summary. Results with $*$ mark
are taken from the corresponding papers.


<a id="source-section-17"></a>

## 5 Conclusion


In this paper, we explored how to use BERT for extractive summarization.
We proposed the Bertsum model and tried several summarization layers can be applied with BERT. We did experiments on two large-scale datasets and found the Bertsum with inter-sentence Transformer layers can achieve the best performance.


<a id="source-section-18"></a>

## References


- Ba et al. (2016)

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. 2016.


Layer normalization.


arXiv preprint arXiv:1607.06450.


- Carbonell and Goldstein (1998)

Jaime G Carbonell and Jade Goldstein. 1998.


The use of mmr and diversity-based reranking for reodering documents
and producing summaries.


- Celikyilmaz et al. (2018)

Asli Celikyilmaz, Antoine Bosselut, Xiaodong He, and Yejin Choi. 2018.


Deep communicating agents for abstractive summarization.


In Proceedings of the NAACL Conference.


- Chen et al. (2018)

Mia Xu Chen, Orhan Firat, Ankur Bapna, Melvin Johnson, Wolfgang Macherey,
George Foster, Llion Jones, Niki Parmar, Mike Schuster, Zhifeng Chen, et al.
2018.


The best of both worlds: Combining recent advances in neural machine
translation.


In Proceedings of the ACL Conference.


- Cheng and Lapata (2016)

Jianpeng Cheng and Mirella Lapata. 2016.


Neural summarization by extracting sentences and words.


In Proceedings of the ACL Conference.


- Devlin et al. (2018)

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018.


Bert: Pre-training of deep bidirectional transformers for language
understanding.


- Dong et al. (2018)

Yue Dong, Yikang Shen, Eric Crawford, Herke van Hoof, and Jackie Chi Kit
Cheung. 2018.


Banditsum: Extractive summarization as a contextual bandit.


In Proceedings of the EMNLP Conference.


- Durrett et al. (2016)

Greg Durrett, Taylor Berg-Kirkpatrick, and Dan Klein. 2016.


Learning-based single-document summarization with compression and
anaphoricity constraints.


In Proceedings of the ACL Conference.


- Hermann et al. (2015)

Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will
Kay, Mustafa Suleyman, and Phil Blunsom. 2015.


Teaching machines to read and comprehend.


In Advances in Neural Information Processing Systems, pages
1693–1701.


- Klein et al. (2017)

Guillaume Klein, Yoon Kim, Yuntian Deng, Jean Senellart, and Alexander M Rush.
2017.


Opennmt: Open-source toolkit for neural machine translation.


In arXiv preprint arXiv:1701.02810.


- Nallapati et al. (2017)

Ramesh Nallapati, Feifei Zhai, and Bowen Zhou. 2017.


Summarunner: A recurrent neural network based sequence model for
extractive summarization of documents.


In Proceedings of the AAAI Conference.


- Narayan et al. (2018)

Shashi Narayan, Shay B Cohen, and Mirella Lapata. 2018.


Ranking sentences for extractive summarization with reinforcement
learning.


In Proceedings of the NAACL Conference.


- Paulus et al. (2018)

Romain Paulus, Caiming Xiong, and Richard Socher. 2018.


A deep reinforced model for abstractive summarization.


In Proceedings of the ICLR Conference.


- Sandhaus (2008)

Evan Sandhaus. 2008.


The New York Times Annotated Corpus.


Linguistic Data Consortium, Philadelphia, 6(12).


- See et al. (2017)

Abigail See, Peter J. Liu, and Christopher D. Manning. 2017.


Get to the point: Summarization with pointer-generator networks.


In Proceedings of the ACL Conference.


- Vaswani et al. (2017)

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017.


Attention is all you need.


In Advances in Neural Information Processing Systems, pages
5998–6008.


- Zhang et al. (2018)

Xingxing Zhang, Mirella Lapata, Furu Wei, and Ming Zhou. 2018.


Neural latent extractive document summarization.


In Proceedings of the EMNLP Conference.


- Zhou et al. (2018)

Qingyu Zhou, Nan Yang, Furu Wei, Shaohan Huang, Ming Zhou, and Tiejun Zhao.
2018.


Neural document summarization by jointly learning to score and select
sentences.


In Proceedings of the ACL Conference.

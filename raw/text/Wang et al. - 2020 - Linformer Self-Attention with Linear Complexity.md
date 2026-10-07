# Wang et al. - 2020 - Linformer Self-Attention with Linear Complexity

- Source HTML: `raw/html/Wang et al. - 2020 - Linformer Self-Attention with Linear Complexity.html`
- Source SHA256: `b6a5432e8a236d86f62cb713076928fc264f3022d39ec4f32ad395e66544b34c`
- Source URL: https://ar5iv.labs.arxiv.org/html/2006.04768
- Generated from: `scripts/fetch_web_text.py`
- Extraction: `structured-html-v2` (headings, links, MathML/TeX and tables; figures require visual review)

## Extracted Text

<a id="source-section-0"></a>

<a id="source-section-1"></a>

# Linformer: Self-Attention with Linear Complexity


Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma

Facebook AI, Seattle, WA

{sinongwang, belindali, hanfang, mkhabsa, haom}@fb.com


<a id="source-section-2"></a>

###### Abstract


Large transformer models have shown extraordinary success in achieving state-of-the-art results in many natural language processing applications. However, training and deploying these models can be prohibitively costly for long sequences, as the standard self-attention mechanism of the Transformer uses $O(n^{2})$ time and space with respect to sequence length. In this paper, we demonstrate that the self-attention mechanism can be approximated by a low-rank matrix. We further exploit this finding to propose a new self-attention mechanism, which reduces the overall self-attention complexity from $O(n^{2})$ to $O(n)$ in both time and space. The resulting linear transformer, the Linformer, performs on par with standard Transformer models, while being much more memory- and time-efficient.


<a id="source-section-3"></a>

## 1 Introduction


Transformer models (Vaswani et al., [2017](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib29)) have become ubiquitous for wide variety of problems in natural language processing (NLP), including translation (Ott et al., [2018](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib21)), text classification, question answering, among others (Raffel et al., [2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib25); Mohamed et al., [2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib20)).
Over the last couple of years, the number of parameters in state-of-the-art NLP transformers has grown drastically, from the original 340 million introduced in BERT-Large to 175 billion in GPT-3 (Brown et al., [2020](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib3)). Although these large-scale models yield impressive results on wide variety of tasks,
training and deploying such model are slow in practice. For example, the original BERT-Large model (Devlin et al., [2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib7)) takes four days to train on 16 Cloud TPUs, and the recent GPT-3 (Brown et al., [2020](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib3))
consumed orders of magnitude more petaflops / day to train compared to its predecessor, GPT-2 (Radford et al., [2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib24)).
Beyond training, deploying Transformer models to real world applications is also expensive, usually requiring extensive distillation (Hinton et al., [2015](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib10)) or compression.


The main efficiency bottleneck in Transformer models is its self-attention mechanism.
Here, each token’s representation is updated by attending to all other tokens in the previous layer.
This operation is key for retaining long-term information, giving Transformers the edge over recurrent models on long sequences.
However, attending to all tokens at each layer incurs a complexity of $O(n^{2})$ with respect to sequence length.
Thus, in this paper, we seek to answer the question:
can Transformer models be optimized to avoid this quadratic operation, or is this operation required to maintain strong performance?


Prior work has proposed several techniques for improving the efficiency of self-attention.
One popular technique is introducing sparsity into attention layers (Child et al., [2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib6); Qiu et al., [2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib23); Beltagy et al., [2020](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib2)) by having each token attend to only a subset of tokens in the whole sequence. This reduces the overall complexity of the attention mechanism to $O(n\sqrt{n})$ (Child et al., [2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib6)). However, as shown in Qiu et al. ([2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib23)), this approach suffers from a large performance drop with limited efficiency gains, i.e., a 2% drop with only 20% speed up.
More recently, the Reformer (Kitaev et al., [2020](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib13)) used
locally-sensitive hashing (LSH) to reduce the self-attention complexity to $O(n\log(n))$.
However, in practice, the Reformer’s efficiency gains only appear
on sequences with length $>2048$ (Figure 5 in Kitaev et al. ([2020](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib13))). Furthermore, the Reformer’s multi-round hashing approach actually increases the number of sequential operations, which further undermines their final efficiency gains.


In this work, we introduce a novel approach for tackling the self-attention bottleneck in Transformers. Our approach is inspired by the key observation that self-attention is low rank. More precisely, we show both theoretically and empirically that the stochastic matrix formed by self-attention can be approximated by a low-rank matrix. Empowered by this observation, we introduce a novel mechanism that reduces self-attention to an $O(n)$ operation in both space- and time-complexity:
we decompose
the original scaled dot-product attention into multiple smaller attentions through linear projections, such that the combination of these operations forms a low-rank factorization of the original attention.
A summary of runtimes for various Transformer architectures, including ours, can be found in Table [1](https://ar5iv.labs.arxiv.org/html/2006.04768#S1.T1).


One predominant application of Transformers, that has seen the most gains, is using them as pretrained language models, whereby models are first pretrained with a language modeling objective on a large corpus, then finetuned on target tasks using supervised data (Devlin et al., [2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib7); Liu et al., [2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib16); Lewis et al., [2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib14)).
Following Devlin et al. ([2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib7)), we pretrain our model on BookCorpus (Zhu et al., [2015](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib31)) plus English Wikipedia using masked-language-modeling objective. We observe similar pretraining performance to the standard Transformer model. We then finetune our pretrained models on three tasks from GLUE (Wang et al., [2018](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib30)) and one sentiment analysis task, IMDB reviews (Maas et al., [2011](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib17)). On these tasks, we find that our model performs comparably, or even slightly better, than the standard pretrained Transformer, while observing significant training and inference speedups.


Table 1: Per-layer time complexity and minimum number of sequential operations as a function of sequence length ($n$) for various architectures.


| Model Architecture | Complexity per Layer | Sequential Operation |
| --- | --- | --- |
| Recurrent | $O(n)$ | $O(n)$ |
| Transformer, (Vaswani et al., [2017](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib29)) | $O(n^{2})$ | $O(1)$ |
| Sparse Tansformer, (Child et al., [2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib6)) | $O(n\sqrt{n})$ | $O(1)$ |
| Reformer, (Kitaev et al., [2020](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib13)) | $O(n\log(n))$ | $O(\log(n))$ |
| Linformer | $O(n)$ | $O(1)$ |


<a id="source-section-4"></a>

## 2 Backgrounds and Related works


<a id="source-section-5"></a>

### 2.1 Transformer and Self-Attention


The Transformer is built upon the idea of Multi-Head Self-Attention (MHA), which allows the model to jointly attend to information at different positions from different representation subspaces. MHA is defined as


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>\mbox{MultiHead}(Q,K,V)=\mbox{Concat}(\mbox{head}_{1},\mbox{head}_{2},\ldots,\mbox{head}_{h})W^{O},<br>$$ | | (1) |


where $Q,K,V\in\mathbb{R}^{n\times d_{m}}$ are input embedding matrices, $n$ is sequence length, $d_{m}$ is the embedding dimension, and $h$ is the number of heads. Each head is defined as:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>\mbox{head}_{i}=\mbox{Attention}(QW_{i}^{Q},KW_{i}^{K},VW_{i}^{V})=\underbrace{\mbox{softmax}\left[\frac{QW_{i}^{Q}(KW_{i}^{K})^{T}}{\sqrt{d_{k}}}\right]}_{P}VW_{i}^{V},<br>$$ | | (2) |


where $W_{i}^{Q},W_{i}^{K}\in\mathbb{R}^{d_{m}\times d_{k}},W_{i}^{V}\in\mathbb{R}^{d_{m}\times d_{v}},W^{O}\in\mathbb{R}^{hd_{v}\times d_{m}}$ are learned matrices and $d_{k},d_{v}$ are the hidden dimensions of the projection subspaces. For the rest of this paper, we will not differentiate between $d_{k}$ and $d_{v}$ and just use $d$.


The self-attention defined in ([2](https://ar5iv.labs.arxiv.org/html/2006.04768#S2.E2)) refers to a context mapping matrix $P\in\mathbb{R}^{n\times n}$. The Transformer uses $P$ to capture the input context for a given token, based on a combination of all tokens in the sequence.
However, computing
$P$ is expensive. It requires multiplying two $n\times d$ matrices, which is $O(n^{2})$ in time and space complexity.
This quadratic dependency on the sequence length has become a bottleneck for Transformers.


<a id="source-section-6"></a>

### 2.2 Related works


There has been much prior literature on improving the efficiency of Transformers, especially the self-attention bottleneck. The most common techniques for model efficiency that can be applied to Transformers (some specific to Transformers, others more general-purpose) include:


Mixed Precision (Micikevicius et al., [2017](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib19)):
Using half-precision or mixed-precision representations of floating points is popular in deep learning, and is also widely used in training Transformers (Ott et al., [2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib22)). This technique can be further improved through Quantization Aware Training (Jacob et al., [2018](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib12); Fan et al., [2020](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib9)), where the weights are quantized during training and the gradients are approximated with the Straight-Through Estimator. This line of work is orthogonal to our approach, and we use mixed-precision training by default.


Knowledge Distillation (Hinton et al., [2015](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib10)): Knowledge distillation aims to transfer the “knowledge" from a large teacher model to a lightweight student model. The student model is then used during inference. However this approach has drawbacks: It does not address speeding up the teacher model during training, and moreover, student models usually suffer performance degradation compared to the teacher model. For example, when distilling a 12-layer BERT to a 6-layer BERT, the student model experiences an average 2.5% performance drop on several benchmark tasks (Sanh et al., [2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib27)).


Sparse Attention (Child et al., [2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib6)): This technique improves the efficiency of self-attention by
adding sparsity in the context mapping matrix $P$. For example, the Sparse Transformer (Child et al., [2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib6)) only computes $P_{ij}$ around the diagonal of matrix $P$ (instead of the all $P_{ij}$). Meanwhile, blockwise self-attention (Qiu et al., [2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib23)) divides $P$ into multiple blocks and only computes $P_{ij}$ within the selected blocks. However, these techniques also suffer a large performance degradation, while having only limited additional speed-up, i.e., 2% drop with 20% speed up.


LSH Attention (Kitaev et al., [2020](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib13)): Locally-sensitive hashing (LSH) attention utilizes a multi-round hashing scheme when computing dot-product attention, which in theory reduces the self-attention complexity to $O(n\log(n))$. However, in practice, their complexity term has a large constant $128^{2}$
and it is only more efficient than the vanilla transformer when sequence length is extremely long.


Improving Optimizer Efficiency:
Microbatching (Huang et al., [2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib11)) splits a batch into small microbatches (which can be fit into memory), and then separately runs forward and backward passes on them with gradient accumulation. Gradient checkpointing (Chen et al., [2016](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib4)) saves memory by only caching activations of a subset of layers. The uncached activations are recomputed during backpropagation from the latest checkpoint. Both techniques trade off time for memory, and do not speed up inference.


As we’ve noted, most common techniques have limitations in reducing both the training and inference time/memory consumption, we investigate how to optimize the self-attention layers and introduce our approach next.


<a id="source-section-7"></a>

## 3 Self-Attention is Low Rank


In this section, we demonstrate that the self-attention mechanism, i.e., the context mapping matrix $P$, is low-rank.


[图片：Refer to caption]


Figure 1: Left two figures are spectrum analysis of the self-attention matrix in pretrained transformer model (Liu et al., [2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib16)) with $n=512$. The Y-axis is the normalized cumulative singular value of context mapping matrix $P$, and the X-axis the index of largest eigenvalue. The results are based on both RoBERTa-base and large model in two public datasets: Wiki103 and IMDB. The right figure plots the heatmap of normalized cumulative eigenvalue at the 128-th largest eigenvalue across different layers and heads in Wiki103 data.


We first provide a spectrum analysis of the context mapping matrix $P$. We use two pretrained transformer models, RoBERTa-base (12-layer stacked transformer) and RoBERTa-large (24-layer stacked transformer) (Liu et al., [2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib16)) on two tasks: masked-language-modeling task on Wiki103 (Merity et al., [2016](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib18)) and classification task on IMDB (Maas et al., [2011](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib17)). In Figure [1](https://ar5iv.labs.arxiv.org/html/2006.04768#S3.F1) (left), we apply singular value decomposition into $P$ across different layers and different heads of the model, and plot the normalized cumulative singular value averaged over 10k sentences.
The results exhibit a clear long-tail spectrum distribution across each layer, head and task.
This implies that most of the information of matrix $P$ can be recovered from the first few largest singular values.
In Figure [1](https://ar5iv.labs.arxiv.org/html/2006.04768#S3.F1) (right), we plot a heatmap of the normalized cumulative singular value at the 128-th largest singular value (out of 512). We observe that the spectrum distribution in higher layers is more skewed than in lower layers, meaning that, in higher layers, more information is concentrated in the largest singular values and the rank of $P$ is lower.


Below, we provide a theoretical analysis of the above spectrum results.


<a id="source-section-8"></a>

###### Theorem 1.


(self-attention is low rank)

For any $Q,K,V\in\mathbb{R}^{n\times d}$ and $W^{Q}_{i},W^{K}_{i},W^{V}_{i}\in\mathbb{R}^{d\times d}$, for any column vector $w\in\mathbb{R}^{n}$ of matrix $VW^{V}_{i}$, there exists a low-rank matrix $\tilde{P}\in\mathbb{R}^{n\times n}$ such that


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>\Pr(\\|\tilde{P}w^{T}-Pw^{T}\\|<\epsilon\\|Pw^{T}\\|)>1-o(1)\mbox{ and }\text{rank}(\tilde{P})=\Theta(\log(n)),<br>$$ | | (3) |


where the context mapping matrix $P$ is defined in ([2](https://ar5iv.labs.arxiv.org/html/2006.04768#S2.E2)).


<a id="source-section-9"></a>

###### Proof.


Based on the definition of the context mapping matrix $P$, we can write


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>P=\mbox{ softmax}\underbrace{\left[\frac{QW_{i}^{Q}(KW_{i}^{K})^{T}}{\sqrt{d}}\right]}_{A}=\exp{(A)}\cdot D_{A}^{-1},<br>$$ | | (4) |


where $D_{A}$ is an $n\times n$ diagonal matrix. The main idea of this proof is based on the distributional Johnson–Lindenstrauss lemma (Lindenstrauss, [1984](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib15)) (JL for short). We construct the approximate low rank matrix as $\tilde{P}=\exp{(A)}\cdot D_{A}^{-1}R^{T}R$, where $R\in\mathbb{R}^{k\times n}$ with i.i.d. entries from $N(0,1/k)$. We can then use the JL lemma to show that, for any column vector $w\in\mathbb{R}^{n}$ of matrix $VW_{i}^{V}$, when $k=5\log(n)/(\epsilon^{2}-\epsilon^{3})$, we have


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>\Pr\left(\\|PR^{T}Rw^{T}-Pw^{T}\\|\leq\epsilon\\|Pw^{T}\\|\right)>1-o(1).<br>$$ | | (5) |


For more details, refer to the supplementary materials.
∎


Given the low-rank property of the context mapping matrix $P$, one straightforward idea is to use singular value decomposition (SVD) to approximate $P$ with a low-rank matrix $P_{\text{low}}$, as follows


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>P\approx P_{\mbox{low}}=\sum\limits_{i=1}^{k}\sigma_{i}u_{i}v_{i}^{T}=\underbrace{\begin{bmatrix}\\<br>u_{1},\cdots,u_{k}\\<br>\\<br>\end{bmatrix}}_{k}\mbox{diag}\{\sigma_{1},\cdots,\sigma_{k}\}\left.\begin{aligned} \begin{bmatrix}&v_{1}&\\<br>&\vdots&\\<br>&v_{k}&\\<br>\end{bmatrix}\end{aligned}\right\}k<br>$$ | | (6) |


where $\sigma_{i}$, $u_{i}$ and $v_{i}$ are the $i$ largest singular values and their corresponding singular vectors. Based on the results in Theorem [1](https://ar5iv.labs.arxiv.org/html/2006.04768#Thmtheorem1) and the Eckart–Young–Mirsky Theorem (Eckart & Young, [1936](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib8)), one can use
$P_{\text{low}}$
to approximate self-attention ([2](https://ar5iv.labs.arxiv.org/html/2006.04768#S2.E2)) with $\epsilon$ error and $O(nk)$ time and space complexity.
However, this approach requires performing an SVD decomposition in each self-attention matrix, which adds additional complexity. Therefore, we propose another approach for low-rank approximation that avoids this added complexity.


<a id="source-section-10"></a>

## 4 Model


In this section, we propose a new self-attention mechanism which allows us to compute the contextual mapping $P\cdot VW_{i}^{V}$ in linear time and memory complexity with respect to sequence length.


The main idea of our proposed linear self-attention (Figure [2](https://ar5iv.labs.arxiv.org/html/2006.04768#S4.F2)) is to add two linear projection matrices
$E_{i},F_{i}\in\mathbb{R}^{n\times k}$ when computing key and value. We first project the original $(n\times d)$-dimensional key and value layers $KW_{i}^{K}$ and $VW_{i}^{V}$ into $(k\times d)$-dimensional projected key and value layers. We then compute an $(n\times k)$-dimensional context mapping matrix $\bar{P}$ using scaled dot-product attention.


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | $\displaystyle\overline{\mbox{head}_{i}}$ | $\displaystyle=\mbox{Attention}(QW_{i}^{Q},E_{i}KW_{i}^{K},F_{i}VW_{i}^{V})$ | | |
| | | $\displaystyle=\underbrace{\mbox{softmax}\left(\frac{QW_{i}^{Q}(E_{i}KW_{i}^{K})^{T}}{\sqrt{d_{k}}}\right)}_{\bar{P}:n\times k}\cdot\underbrace{F_{i}VW_{i}^{V}}_{k\times d},$ | | (7) |


Finally, we compute context embeddings for each headi using $\bar{P}\cdot(F_{i}VW_{i}^{V})$.
Note the above operations only require $O(nk)$ time and space complexity.
Thus, if we can choose a very small projected dimension $k$, such that $k\ll n$, then we can significantly reduce the memory and space consumption. The following theorem states that, when $k=O(d/\epsilon^{2})$ (independent of $n$), one can approximate $P\cdot VW_{i}^{V}$ using linear self-attention ([7](https://ar5iv.labs.arxiv.org/html/2006.04768#S4.E7)) with $\epsilon$ error.


[图片：Refer to caption]


Figure 2: Left and bottom-right show architecture and example of our proposed multihead linear self-attention. Top right shows inference time vs. sequence length for various Linformer models.


<a id="source-section-11"></a>

###### Theorem 2.


(Linear self-attention)
For any $Q_{i},K_{i},V_{i}\in\mathbb{R}^{n\times d}$ and $W_{i}^{Q},W_{i}^{K},W_{i}^{V}\in\mathbb{R}^{d\times d}$, if $k=\min\{\Theta(9d\log(d)/\epsilon^{2}),5\Theta(\log(n)/\epsilon^{2})\}$, then there exists matrices $E_{i},F_{i}\in\mathbb{R}^{n\times k}$ such that, for any row vector $w$ of matrix $QW_{i}^{Q}(KW_{i}^{K})^{T}/\sqrt{d}$, we have


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>\Pr\left(\\|\mbox{\emph{softmax}}(wE_{i}^{T})F_{i}VW_{i}^{V}-\mbox{\emph{softmax}}(w)VW_{i}^{V}\\|\leq\epsilon\\|\mbox{\emph{softmax}}(w)\\|\\|VW_{i}^{V}\\|\right)>1-o(1)<br>$$ | | (8) |


<a id="source-section-12"></a>

###### Proof.


The main idea of proof is based on the distributional Johnson–Lindenstrauss lemma (Lindenstrauss, [1984](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib15)). We first prove that for any row vector $x\in\mathbb{R}^{n}$ of matrix $QW_{i}^{Q}(KW_{i}^{K})^{T}/\sqrt{d_{k}}$ and column vector $y\in\mathbb{R}^{n}$ of matrix $VW_{i}^{V}$,


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $\displaystyle\Pr\left(\\|\exp(xE_{i}^{T})F_{i}y^{T}-\exp(x)y^{T}\\|\leq\epsilon\\|\exp(x)y^{T}\\|\right)>1-2e^{-(\epsilon^{2}-\epsilon^{3})k/4},$ | | (9) |


where $E_{i}=\delta R$ and $F_{i}=e^{-\delta}R$, where $R\in\mathbb{R}^{k\times n}$ with i.i.d. entries from $N(0,1/k)$ and $\delta$ is a small constant. Applying the result in ([9](https://ar5iv.labs.arxiv.org/html/2006.04768#S4.E9)) to every row vector of matrix $A$ and every column vector of matrix $V$, one can directly prove that, for any row vector $A_{i}$ of matrix $A$,


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $\displaystyle\Pr\left(\\|\exp(A_{i}E_{i}^{T})F_{i}V-\exp(A_{i})V\\|\leq\epsilon\\|\exp(A_{i})V\\|\right)>1-o(1),$ | | (10) |


by setting $k=5\log(nd)/(\epsilon^{2}-\epsilon^{3})$. This result does not utilize the low rank property of matrix $A$ (rank($A$)=$d$) and the resultant $k$ has a dependency on sequence length $n$. We will further utlize the fact that rank($A$)=$d$ to prove the choice of $k$ can be constant and independent of sequence length $n$. For more details, refer to the supplementary materials.
∎


In Figure [2](https://ar5iv.labs.arxiv.org/html/2006.04768#S4.F2) (top right), we plot the inference speed of Linformer and standard Transformer versus sequence length, while holding the total number of tokens fixed. We see that while standard Transformer becomes slower at longer sequence lengths, the Linformer speed remains relatively flat and is significantly faster at long sequences.


<a id="source-section-13"></a>

#### Additional Efficiency Techniques


Several additional techniques can be introduced on top of Linformer to further optimize for both performance and efficiency:


Parameter sharing between projections: One can share parameters for the
linear projection matrices $E_{i},F_{i}$ across layers and heads. In particular, we experimented with 3 levels of sharing:


- •


Headwise sharing: for each layer, we share two projection matrices $E$ and $F$ such that $E_{i}=E$ and $F_{i}=F$ across all heads $i$.


- •


Key-value sharing: we do headwise sharing, with the additional constraint of sharing the key and value projections. For each layer, we create a single projection matrix $E$ such that $E_{i}=F_{i}=E$ for each key-value projection matrix across all head $i$.


- •


Layerwise sharing: we use a single projection matrix $E$ across all layers, for all heads, and for both key and value.


For example, in a 12-layer, 12-head stacked Transformer model, headwise sharing, key-value sharing and layerwise sharing will introduce 24, 12, and 1 distinct linear projection matrices, respectively.


Nonuniform projected dimension: One can choose a different projected dimension $k$ for different heads and layers. As shown in Figure [1](https://ar5iv.labs.arxiv.org/html/2006.04768#S3.F1) (right), the contextual mapping matrices in different heads and layers have distinct spectrum distributions, and heads in higher layer tend towards a more skewed distributed spectrum (lower rank). This implies one can choose a smaller projected dimension $k$ for higher layers.


General projections: One can also choose different kinds of low-dimensional projection methods instead of a simple linear projection. For example, one can choose mean/max pooling, or convolution where the kernel and stride is set to $n/k$. The convolutional functions contain parameters that require training.


<a id="source-section-14"></a>

## 5 Experiments


In this section, we present experimental results for the the techniques described above. We analyze the techniques one-by-one and explore how they impact performance.


<a id="source-section-15"></a>

### 5.1 Pretraining Perplexities


We first compare the pretraining performance of our proposed architecture against RoBERTa (Liu et al., [2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib16)), which is based on the Transformer. Following Devlin et al. ([2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib7)), we use BookCorpus (Zhu et al., [2015](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib31)) plus English Wikipedia as our pretraining set (3300M words).
All models are pretrained with the masked-language-modeling (MLM) objective, and the training for all experiments are parallelized across 64 Tesla V100 GPUs with 250k updates.


Effect of projected dimension: We experiment with various values for the projected dimension $k$. (We use the same $k$ across all layers and heads of Linformer.)
In the Figure [3](https://ar5iv.labs.arxiv.org/html/2006.04768#S5.F3)(a) and (b), we plot the validation perplexity curves for both the standard Transformer and the Linformer across different $k$, for maximum sequence lengths $n=512$ and $n=1024$.
As expected, the Linformer performs better as projected dimension $k$ increases.
However, even at $k=128$ for $n=512$ and $k=256$ for $n=1024$, Linformer’s performance is already nearly on par with the original Transformer.


[图片：Refer to caption]


Figure 3: Pretraining validation perplexity versus number of updates.


Effect of sharing projections: In Figure [3](https://ar5iv.labs.arxiv.org/html/2006.04768#S5.F3)(c), we plot the validation perplexity curves for the three parameter sharing strategies (headwise, key-value, and layerwise) with $n=512$. Note that when we use just a single projection matrix (i.e. for layerwise sharing), the resulting Linformer model’s validation perplexity almost matches that of the the non-shared model.
This suggests that we can decrease the number of additional parameters in our model, and consequently, it’s memory consumption, without much detriment to performance.


Effect of longer sequences: We evaluate the effect of sequence length during Linformer pretraining. In the Figure [3](https://ar5iv.labs.arxiv.org/html/2006.04768#S5.F3)(d), we plot the validation perplexity for Linformer with $n\in\{512,1024,2048,4096\}$, holding projected dimension $k$ fixed at $256$.
Note that as sequence length increases, even though our projected dimension is fixed, the final perplexities after convergence remain about the same. This further empirically supports our assertion that the Linformer is linear-time.


Table 2: Dev set results on benchmark natural language understanding tasks. The RoBERTa-base model here is pretrained with same corpus as BERT.


| $n$ | Model | SST-2 | IMDB | QNLI | QQP | Average |
| --- | --- | --- | --- | --- | --- | --- |
| 512 [rowspan=7] | Liu et al. ([2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib16)), RoBERTa-base | 93.1 | 94.1 | 90.9 | 90.9 | 92.25 |
| Linformer, 128 | 92.4 | 94.0 | 90.4 | 90.2 | 91.75 | |
| Linformer, 128, shared kv | 93.4 | 93.4 | 90.3 | 90.3 | 91.85 | |
| Linformer, 128, shared kv, layer | 93.2 | 93.8 | 90.1 | 90.2 | 91.83 | |
| Linformer, 256 | 93.2 | 94.0 | 90.6 | 90.5 | 92.08 | |
| Linformer, 256, shared kv | 93.3 | 93.6 | 90.6 | 90.6 | 92.03 | |
| Linformer, 256, shared kv, layer | 93.1 | 94.1 | 91.2 | 90.8 | 92.30 | |
| 512 [rowspan=2] | Devlin et al. ([2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib7)), BERT-base | 92.7 | 93.5 | 91.8 | 89.6 | 91.90 |
| Sanh et al. ([2019](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib27)), Distilled BERT | 91.3 | 92.8 | 89.2 | 88.5 | 90.45 | |
| 1024 [rowspan=3] | Linformer, 256 | 93.0 | 93.8 | 90.4 | 90.4 | 91.90 |
| Linformer, 256, shared kv | 93.0 | 93.6 | 90.3 | 90.4 | 91.83 | |
| Linformer, 256, shared kv, layer | 93.2 | 94.2 | 90.8 | 90.5 | 92.18 | |


<a id="source-section-16"></a>

### 5.2 Downstream Results


Thus far, we have only examined the pretraining perplexities of our model.
However, we wish to show that our conclusions hold after finetuning on downstream tasks.
We finetune our Linformer on IMDB (Maas et al., [2011](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib17)) and SST-2 (Socher et al., [2013](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib28)) (sentiment classification), as well as QNLI (natural language inference) (Rajpurkar et al., [2016](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib26)), and QQP (textual similarity) (Chen et al., [2018](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib5))
We do the same with RoBERTa, 12-layer BERT-base and 6-layer distilled BERT. All of our models, including the Transformer baselines, were pretrained with the same objective, pretraining corpus, and up to 250k updates (although our Linformer takes much less wall-clock time to get to 250k updates, and was consequently trained for less time). Results are listed in Table [2](https://ar5iv.labs.arxiv.org/html/2006.04768#S5.T2).


We observe that the Linformer model ($n=512,k=128$) has comparable downstream performance to the RoBERTa model, and in fact even slightly outperforms it at $k=256$. Moreover, we note that although the Linformer’s layerwise sharing strategy shares a single projection matrix across the entire model, it actually exhibits the best accuracy result of all three parameter sharing strategies.
Furthermore, the Linformer pretrained with longer sequence length $(n=1024,k=256)$ has similar results to the one pretrained with shorter length $(n=512,k=256)$,
this empirically supports the notion that the performance of Linformer model is mainly determined by the projected dimension $k$ instead of the ratio $n/k$.


<a id="source-section-17"></a>

### 5.3 Inference-time Efficiency Results


In Table [3](https://ar5iv.labs.arxiv.org/html/2006.04768#S5.T3),
we report the inference efficiencies of Linformer (with layerwise sharing) against a standard Transformer. We benchmark both models’ inference speed and memory on a 16GB Tesla V100 GPU card.
We randomly generate data up to some sequence length $n$ and perform a full forward pass on a multiple batches. We also choose batch size based on the maximum batch size that can fit in memory, and our memory savings are computed based on this number.


| length $n$ [rowspan=2] | projected dimensions $k$ [colspan=5] | | | | |
| --- | --- | --- | --- | --- | --- |
| 128 | 256 | 512 | 1024 | 2048 | |
| 512 | 1.5x | 1.3x | - | - | - |
| 1024 | 1.7x | 1.6x | 1.3x | - | - |
| 2048 | 2.6x | 2.4x | 2.1x | 1.3x | - |
| 4096 | 3.4x | 3.2x | 2.8x | 2.2x | 1.3x |
| 8192 | 5.5x | 5.0x | 4.4x | 3.5x | 2.1x |
| 16384 | 8.6x | 7.8x | 7.0x | 5.6x | 3.3x |
| 32768 | 13x | 12x | 11x | 8.8x | 5.0x |
| 65536 | 20x | 18x | 16x | 14x | 7.9x |


| length $n$ [rowspan=2] | projected dimensions $k$ [colspan=5] | | | | |
| --- | --- | --- | --- | --- | --- |
| 128 | 256 | 512 | 1024 | 2048 | |
| 512 | 1.7x | 1.5x | - | - | - |
| 1024 | 3.0x | 2.9x | 1.8x | - | - |
| 2048 | 6.1x | 5.6x | 3.6x | 2.0x | - |
| 4096 | 14x | 13x | 8.3x | 4.3x | 2.3x |
| 8192 | 28x | 26x | 17x | 8.5x | 4.5x |
| 16384 | 56x | 48x | 32x | 16x | 8x |
| 32768 | 56x | 48x | 36x | 18x | 16x |
| 65536 | 60x | 52x | 40x | 20x | 18x |


Table 3: Inference-time efficiency improvements of the Linformer over the Transformer, across various projected dimensions $k$ and sequence lengths $n$.
Left table shows time saved. Right table shows memory saved.


From Table [3](https://ar5iv.labs.arxiv.org/html/2006.04768#S5.T3), we see that even with $n=512$ and $k=128$, Linformer has $1.5\times$ faster inference time and allows for
a $1.7\times$ larger maximum batch size than the Transformer.
As sequence length increases, the inference-time speed-up and memory savings are even more dramatic.
We also plot inference times of both Linformer and Transformer on the 100 data samples in the top right of Figure [2](https://ar5iv.labs.arxiv.org/html/2006.04768#S4.F2).


<a id="source-section-18"></a>

## 6 Conclusion


Transformer models are notoriously slow to train and deploy
in practice since their self-attention operations have $O(n^{2})$ time and space complexity with respect to sequence length $n$. In this paper, we demonstrate, both theoretically and empirically, that the stochastic matrix formed by self-attention mechanism is low-rank. We further leverage this observation to propose a new, highly efficient self-attention mechanism. Through a combination of theoretical and empirical analysis, we demonstrate that our proposed approach is $O(n)$ with respect to sequence length.


<a id="source-section-19"></a>

## Broader Impact


Our work focuses on making Transformers more efficient by introducing a mechanism that reduces self-attention to linear-time complexity. Potential positive impacts of efficient transformers include increasing the accessibility of our models, both for deployment on devices, as well as during training for research purposes. It also has potential impact on training transformer on images since we can support very long sequences.
Furthermore, there are positive environmental benefits associated with decreasing the power consumption of models.
As such, we see no immediate negative ethical or societal impacts of our work
beyond what applies to other core building blocks of deep learning.


<a id="source-section-20"></a>

## References


- Arriaga & Vempala (2006)

Rosa I Arriaga and Santosh Vempala.


An algorithmic theory of learning: Robust concepts and random
projection.


Machine Learning, 63(2):161–182, 2006.


- Beltagy et al. (2020)

Iz Beltagy, Matthew E Peters, and Arman Cohan.


Longformer: The long-document transformer.


arXiv preprint arXiv:2004.05150, 2020.


- Brown et al. (2020)

Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla
Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell,
et al.


Language models are few-shot learners.


arXiv preprint arXiv:2005.14165, 2020.


- Chen et al. (2016)

Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin.


Training deep nets with sublinear memory cost.


arXiv preprint arXiv:1604.06174, 2016.


- Chen et al. (2018)

Zihan Chen, Hongbo Zhang, Xiaoji Zhang, and Leqi Zhao.


Quora question pairs, 2018.


- Child et al. (2019)

Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever.


Generating long sequences with sparse transformers.


arXiv preprint arXiv:1904.10509, 2019.


- Devlin et al. (2019)

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.


Bert: Pre-training of deep bidirectional transformers for language
understanding.


In Proceedings of the 2019 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long and Short Papers), pp.  4171–4186, 2019.


- Eckart & Young (1936)

Carl Eckart and Gale Young.


The approximation of one matrix by another of lower rank.


Psychometrika, 1(3):211–218, 1936.


- Fan et al. (2020)

Angela Fan, Pierre Stock, Benjamin Graham, Edouard Grave, Remi Gribonval, Herve
Jegou, and Armand Joulin.


Training with quantization noise for extreme fixed-point compression.


arXiv preprint arXiv:2004.07320, 2020.


- Hinton et al. (2015)

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.


Distilling the knowledge in a neural network.


arXiv preprint arXiv:1503.02531, 2015.


- Huang et al. (2019)

Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Chen,
HyoukJoong Lee, Jiquan Ngiam, Quoc V Le, Yonghui Wu, et al.


Gpipe: Efficient training of giant neural networks using pipeline
parallelism.


In Advances in Neural Information Processing Systems, pp. 103–112, 2019.


- Jacob et al. (2018)

Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew
Howard, Hartwig Adam, and Dmitry Kalenichenko.


Quantization and training of neural networks for efficient
integer-arithmetic-only inference.


In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition, pp.  2704–2713, 2018.


- Kitaev et al. (2020)

Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya.


Reformer: The efficient transformer.


In International Conference on Learning Representations, 2020.


- Lewis et al. (2019)

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed,
Omer Levy, Ves Stoyanov, and Luke Zettlemoyer.


Bart: Denoising sequence-to-sequence pre-training for natural
language generation, translation, and comprehension.


ACL, 2019.


- Lindenstrauss (1984)

W Johnson J Lindenstrauss.


Extensions of lipschitz maps into a hilbert space.


Contemp. Math, 26:189–206, 1984.


- Liu et al. (2019)

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer
Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov.


Roberta: A robustly optimized bert pretraining approach.


arXiv preprint arXiv:1907.11692, 2019.


- Maas et al. (2011)

Andrew L Maas, Raymond E Daly, Peter T Pham, Dan Huang, Andrew Y Ng, and
Christopher Potts.


Learning word vectors for sentiment analysis.


In Proceedings of the 49th annual meeting of the association
for computational linguistics: Human language technologies-volume 1, pp. 142–150. Association for Computational Linguistics, 2011.


- Merity et al. (2016)

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher.


Pointer sentinel mixture models.


arXiv preprint arXiv:1609.07843, 2016.


- Micikevicius et al. (2017)

Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen,
David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh
Venkatesh, et al.


Mixed precision training.


arXiv preprint arXiv:1710.03740, 2017.


- Mohamed et al. (2019)

Abdelrahman Mohamed, Dmytro Okhonko, and Luke Zettlemoyer.


Transformers with convolutional context for asr.


arXiv preprint arXiv:1904.11660, 2019.


- Ott et al. (2018)

Myle Ott, Sergey Edunov, David Grangier, and Michael Auli.


Scaling neural machine translation.


In Proceedings of the Third Conference on Machine Translation:
Research Papers, pp.  1–9, 2018.


- Ott et al. (2019)

Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng,
David Grangier, and Michael Auli.


fairseq: A fast, extensible toolkit for sequence modeling.


In Proceedings of the 2019 Conference of the North American
Chapter of the Association for Computational Linguistics (Demonstrations),
pp.  48–53, 2019.


- Qiu et al. (2019)

Jiezhong Qiu, Hao Ma, Omer Levy, Scott Wen-tau Yih, Sinong Wang, and Jie Tang.


Blockwise self-attention for long document understanding.


arXiv preprint arXiv:1911.02972, 2019.


- Radford et al. (2019)

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya
Sutskever.


Language models are unsupervised multitask learners.


OpenAI Blog, 1(8):9, 2019.


- Raffel et al. (2019)

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael
Matena, Yanqi Zhou, Wei Li, and Peter J Liu.


Exploring the limits of transfer learning with a unified text-to-text
transformer.


arXiv preprint arXiv:1910.10683, 2019.


- Rajpurkar et al. (2016)

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang.


Squad: 100,000+ questions for machine comprehension of text.


In Proceedings of the 2016 Conference on Empirical Methods in
Natural Language Processing, pp.  2383–2392, 2016.


- Sanh et al. (2019)

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf.


Distilbert, a distilled version of bert: smaller, faster, cheaper and
lighter.


arXiv preprint arXiv:1910.01108, 2019.


- Socher et al. (2013)

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning,
Andrew Y Ng, and Christopher Potts.


Recursive deep models for semantic compositionality over a sentiment
treebank.


In Proceedings of the 2013 conference on empirical methods in
natural language processing, pp.  1631–1642, 2013.


- Vaswani et al. (2017)

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.


Attention is all you need.


In Advances in neural information processing systems, pp. 5998–6008, 2017.


- Wang et al. (2018)

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and
Samuel R. Bowman.


GLUE: A multi-task benchmark and analysis platform for natural
language understanding.


CoRR, abs/1804.07461, 2018.


URL [http://arxiv.org/abs/1804.07461](http://arxiv.org/abs/1804.07461).


- Zhu et al. (2015)

Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun,
Antonio Torralba, and Sanja Fidler.


Aligning books and movies: Towards story-like visual explanations by
watching movies and reading books.


In Proceedings of the IEEE international conference on computer
vision, pp.  19–27, 2015.


<a id="source-section-21"></a>

## Appendix A Proof of Theorem 1


<a id="source-section-22"></a>

###### Proof.


The main proof idea is based on the distributional Johnson–Lindenstrauss lemma (Lindenstrauss, [1984](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib15)) (JL, for short), the following version is from (Arriaga & Vempala, [2006](https://ar5iv.labs.arxiv.org/html/2006.04768#bib.bib1)).


<a id="source-section-23"></a>

###### Lemma 1.


Let $R$ be an $k\times n$ matrix, $1\leq k\leq n$, with i.i.d. entries from $N(0,1/k)$. For any $x,y\in\mathbb{R}^{n}$, we have


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $\displaystyle\Pr\left(\\|Rx\\|\leq(1+\epsilon)\\|x\\|\right)>1-e^{-(\epsilon^{2}-\epsilon^{3})k/4},$ | | (11) |
| | $\displaystyle\Pr\left(\\|xR^{T}Ry^{T}-xy^{T}\\|\leq\epsilon\\|xy\\|\right)>1-2e^{-(\epsilon^{2}-\epsilon^{3})k/4}.$ | | (12) |


For simplicity, we will omit the subscript $i$ for matrix $W_{i}^{K}$, $W_{i}^{Q}$, $W_{i}^{V}$, $E_{i}$ and $F_{i}$. We will regard $Q$ as $QW^{Q}$, $K$ as $KW^{K}$ and $V$ as $VW^{V}$. Define


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>A=\frac{QW_{i}^{Q}(KW_{i}^{K})^{T}}{\sqrt{d}}<br>$$ | | (13) |


Based on the definition of contextual mapping matrix $P$, we have


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | $\displaystyle P=$ | $\displaystyle\mbox{ softmax}\left[\frac{QW_{i}^{Q}(KW_{i}^{K})^{T}}{\sqrt{d}}\right]$ | | |
| | $\displaystyle=$ | $\displaystyle\exp{(A)}\cdot D_{A}^{-1},$ | | (14) |


where $D_{A}$ is an $n\times n$ diagonal matrix such that


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>(D_{A})_{ii}=\sum\limits_{j=1}^{n}\exp{\left(A_{ji}\right)}<br>$$ | | (15) |


Here we provide a constructive proof. Given any approximation error $\epsilon>0$, define the following matrix.


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>\tilde{P}=\exp{(A)}\cdot D_{A}^{-1}R^{T}R,<br>$$ | | (16) |


where $R$ be an $k\times n$ matrix, $1\leq k\leq n$, with i.i.d. entries from $N(0,1/k)$. Clearly the rank of matrix $\tilde{P}$ satisifies


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>\mbox{rank}(\tilde{P})\leq\mbox{rank}(R)=k.<br>$$ | | (17) |


We further show that, when $k=\log(n)$, we have that, for any column vector $w\in\mathbb{R}^{n}$,


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>\Pr\left(\\|\tilde{P}h-Ph\\|\leq\epsilon\\|Ph\\|\right)>1-o(1).<br>$$ | | (18) |


This concludes the theorem. For any row vector $u\in\mathbb{R}^{n}$ of matrix $P$ and any column vector $w\in\mathbb{R}^{n}$ of matrix $VW^{V}$, applying the JL Lemma, we can obtain


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>\Pr\left(\\|uR^{t}Rw^{T}-uw^{T}\\|\leq\epsilon\\|uw^{T}\\|\right)>1-2e^{-(\epsilon^{2}-\epsilon^{3})k/4}.<br>$$ | | (19) |


Therefore, we have


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | $\displaystyle\Pr\left(\\|\tilde{P}w^{T}-Pw^{T}\\|\leq\epsilon\\|Pw^{T}\\|\right)=$ | $\displaystyle\Pr\left(\\|PR^{T}Rw^{T}-Pw^{T}\\|\leq\epsilon\\|Pw^{T}\\|\right)$ | | |
| | $\displaystyle\overset{(a)}{\geq}$ | $\displaystyle 1-\sum\limits_{x\in P}\Pr\left(\\|xR^{T}Rw^{T}-xw^{T}\\|>\epsilon\\|xw^{T}\\|\right)$ | | |
| | $\displaystyle\overset{(b)}{>}$ | $\displaystyle 1-2ne^{-(\epsilon^{2}-\epsilon^{3})k/4}.$ | | (20) |


The above, step (a) is based on the union bound. The step (b) is utilizing the result of JL Lemma. Let $k=5\log(n)/(\epsilon^{2}-\epsilon^{3})$, then theorem follows.
∎


<a id="source-section-24"></a>

## Appendix B Proof of Theorem 2


<a id="source-section-25"></a>

###### Proof.


Define $E=\delta R$ and $F=e^{-\delta}R$, where $R\in\mathbb{R}^{n\times k}$ with i.i.d. entries from $N(0,1/k)$, $\delta$ is a constant with $\delta=1/2^{n}$. We will first prove that for any row vector $x\in\mathbb{R}^{n}$ of matrix $QK^{T}$ and column vector $y\in\mathbb{R}^{n}$ of matrix $V$,


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $\displaystyle\Pr\left(\\|\exp(xE^{T})Fy^{T}-\exp(x)y^{T}\\|\leq\epsilon\\|\exp(x)y^{T}\\|\right)>1-2e^{-(\epsilon^{2}-\epsilon^{3})k/4}.$ | | (21) |


Based on the triangle inequality, we have


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | $\displaystyle\\|\exp(xE^{T})Fy\exp(x)y^{T}\\|$ | $\displaystyle\leq\\|\exp(xE^{T})Fy-\exp(x)R^{T}Ry\\|+\\|\exp(x)R^{T}Ry-\exp(x)y^{T}\\|$ | | |
| | | $\displaystyle\overset{(a)}{\leq}(1+\epsilon)\\|y\\|\\|\exp(xE^{T})-\exp(x)R^{T}\\|+\\|\exp(x)R^{T}Ry-\exp(x)y^{T}\\|$ | | |
| | | $\displaystyle\overset{(b)}{\leq}\\|\exp(x)R^{T}Ry-\exp(x)y^{T}\\|+o(\\|\exp(x)\\|\\|y\\|)$ | | |
| | | $\displaystyle\overset{(c)}{\leq}\epsilon\\|\exp(x)\\|\\|y\\|+o(\\|\exp(x)\\|\\|y\\|)$ | | (22) |


The above, step (a) is based on the Cauchy inequality and JL Lemma in ([11](https://ar5iv.labs.arxiv.org/html/2006.04768#A1.E11)). The step (b) utilizes the fact that exponential function is Lipchitz continuous in a compact region. Then we can choose a small enough $\delta$, i.e., $\delta=\theta(1/n)$ such that


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>\\|\exp(\delta xR)-\exp(\delta x)R\\|=o(\\|\exp(x)\\|)<br>$$ | | (23) |


The step (c) is based on the JL Lemma defined in ([12](https://ar5iv.labs.arxiv.org/html/2006.04768#A1.E12)).


Applying the result in ([21](https://ar5iv.labs.arxiv.org/html/2006.04768#A2.E21)) to every row vector of matrix $A$ and every column vector of matrix $V$, one can directly prove that, for any row vector $A_{i}$ of matrix $A$,


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $\displaystyle\Pr\left(\\|\exp(A_{i}E^{T})FV-\exp(A_{i})V\\|\leq\epsilon\\|\exp(A_{i})\\|\\|V\\|\right)>1-o(1),$ | | (24) |


by setting $k=5\log(nd)/(\epsilon^{2}-\epsilon^{3})$. This result does not utilize the low rank property of matrix $A$ (rank($A$)=$d$) and the resultant $k$ has a dependency on sequence length $n$. We will further prove the choice of $k$ can be constant and independent of sequence length $n$.


Based on the fact that rank($A$)=$d$, we can find a row submatrix $A_{s}\in\mathbb{R}^{2d\times d}$ of matrix $\exp(AE^{T})FH$ such that rank($A_{s}$)=$d$. Applying the result in ([21](https://ar5iv.labs.arxiv.org/html/2006.04768#A2.E21)) to every row vector of matrix $A_{s}$ and every column vector of matrix $V$, and $k=9\log(d)/(\epsilon^{2}-\epsilon^{3})$, we can obtain that, for any row vector $A_{i}^{s}$ of matrix $A^{s}$,


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $\displaystyle\Pr\left(\\|\exp(A_{i}^{s}E^{T})FV-\exp(A_{i}^{s})V\\|\leq\epsilon\\|\exp(A_{i}^{s})\\|\\|V\\|\right)>1-o(1),$ | | (25) |


Furthermore, define the matrix $\Gamma\in\mathbb{R}^{n\times 2d}$ as


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>\Gamma=\begin{bmatrix}\exp(AE^{T})FV\\<br>\exp(A)V\end{bmatrix}\cdot\begin{bmatrix}\exp(A_{s}E^{T})FV\\<br>\exp(A_{s})V\end{bmatrix}^{-1}<br>$$ | | (26) |


We have that, for any row vector $A_{i}$ of matrix $A$, $1\leq i\leq n$.


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $\displaystyle\\|\exp(A_{i}E^{T})FV-\exp(A_{i})V\\|=$ | $\displaystyle\\|\Gamma_{i}\exp(A^{s}E^{T})FV-\Gamma_{i}\exp(A^{s})V\\|$ | |
| | $\displaystyle\overset{(a)}{\leq}$ | $\displaystyle\left\\|[\exp(A^{s}E^{T})FV-\exp(A^{s})V]^{T}\right\\|_{2}\\|\Gamma_{i}\\|$ | |
| | $\displaystyle\overset{(b)}{\leq}$ | $\displaystyle\Theta(d)\\|\exp(A^{s}E^{T})FV-\exp(A^{s})V\\|_{F}$ | |
| | $\displaystyle=$ | $\displaystyle\Theta(d)\sum\limits_{i=1}^{2d}\\|\exp(A_{i}^{s}E^{T})FV-\exp(A_{i}^{s})V\\|$ | |
| | $\displaystyle\overset{(c)}{\leq}$ | $\displaystyle\epsilon\Theta(d)\sum\limits_{i=1}^{2d}\\|\exp(A_{i}^{s})\\|\\|V\\|$ | |
| | $\displaystyle\leq$ | $\displaystyle\epsilon\Theta(d)\\|\exp(A^{s})\\|\\|V\\|$ | |


The above, step (a) utilizes the inequality $\|Ax\|\leq\|A\|_{2}\cdot\|x\|$, where $\|A\|_{2}=\sqrt{\lambda_{\max}(A^{T}A})$ ($\lambda_{\max}(\cdot)$ is the largest eigenvalue) is the spectrum norm of a matrix $A$. The step (b) is based on matrix norm inequality $\|A\|_{2}\leq\|A\|_{F}$, where $\|A\|_{F}=(\sum_{1\leq i,j\leq n}A_{ij}^{2})^{1/2}$ is the Frobenius norm of matrix $A$. The step (c) is based on the results of ([24](https://ar5iv.labs.arxiv.org/html/2006.04768#A2.E24)).
∎

# Zaheer et al. - 2020 - Big bird Transformers for longer sequences

- Source HTML: `raw/html/Zaheer et al. - 2020 - Big bird Transformers for longer sequences.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2007.14062
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Big Bird: Transformers for Longer Sequences

Manzil Zaheer, Guru Guruganesh, Avinava Dubey, 
Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, 
Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed 
Google Research
{manzilz, gurug, avinavadubey}@google.com

###### Abstract

Transformers-based models, such as BERT, have been one of the most successful deep learning models for NLP. Unfortunately, one of their core limitations is the quadratic dependency (mainly in terms of memory) on the sequence length due to their full attention mechanism. To remedy this, we propose, BigBird, a sparse attention mechanism that reduces this quadratic dependency to linear. We show that BigBird is a universal approximator of sequence functions and is Turing complete, thereby preserving these properties of the quadratic, full attention model. Along the way, our theoretical analysis reveals some of the benefits of having O​(1)𝑂1O(1) global tokens (such as CLS), that attend to the entire sequence as part of the sparse attention mechanism. The proposed sparse attention can handle sequences of length up to 8x of what was previously possible using similar hardware. As a consequence of the capability to handle longer context, BigBird drastically improves performance on various NLP tasks such as question answering and summarization. We also propose novel applications to genomics data.

## 1 Introduction

Models based on Transformers [91], such as BERT [22, 63],
are wildly successful for a wide variety of Natural Language Processing (NLP) tasks and consequently are mainstay of modern NLP research.
Their versatility and robustness are the primary drivers behind the wide-scale adoption of Transformers.
The model is easily adapted for a diverse range of sequence based tasks – as a seq2seq model for translation [91], summarization [66], generation [15], etc. or as a standalone encoders for sentiment analysis [83], POS tagging [65], machine reading comprehension [93], etc. – and it is known to vastly outperform previous
sequence models like LSTM [37].
The key innovation in Transformers is the introduction of a self-attention mechanism, which can be evaluated in parallel for each token of the input sequence, eliminating the sequential dependency in recurrent neural networks, like LSTM.
This parallelism enables Transformers to leverage the full power of modern SIMD hardware accelerators like GPUs/TPUs, thereby facilitating training of NLP models on datasets of unprecedented size.
This ability to train on large scale data has led to surfacing of models like BERT [22] and T5 [75], which pretrain transformers on large general purpose corpora and transfer the knowledge to down-stream task. The pretraining has led to significant improvement in low data regime downstream tasks [51] as well as tasks with sufficient data [101] and thus have been a major force behind the ubiquity of transformers in contemporary NLP.

The self-attention mechanism overcomes constraints of RNNs (namely the sequential nature of RNN)
by allowing each token in the input sequence to attend independently to every other token
in the sequence. This design choice has several interesting repercussions.
In particular, the full self-attention have computational and memory requirement that
is quadratic in the sequence length. We note that while the corpus can be large, the sequence length, which provides the context in many applications is very limited.
Using commonly available current hardware and model sizes, this
requirement translates to roughly being able to handle input sequences of length 512 tokens.
This reduces its direct applicability to tasks that require larger context,
like QA [60], document classification, etc.

However, while we know that self-attention and Transformers are useful, our theoretical
understanding is rudimentary. What aspects of the self-attention model are necessary for its performance?
What can we say about the expressivity of Transformers and similar models? Apriori, it was not
even clear from the design if the proposed self-attention mechanism was as effective as RNNs.
For example, the self-attention does not even obey sequence order as it is permutation equivariant.
This concern has been partially resolved, as Yun et al. [104] showed that transformers are expressive
enough to capture all continuous sequence to sequence functions with a compact domain.
Meanwhile, Pérez et al. [72] showed that the full transformer is Turing Complete
(i.e. can simulate a full Turing machine). Two natural questions arise: Can we achieve the empirical
benefits of a fully quadratic self-attention scheme using fewer inner-products? Do these
sparse attention mechanisms preserve the expressivity and flexibility of the original network?

In this paper, we address both the above questions and produce a sparse attention mechanism that
improves performance on a multitude of tasks that require long contexts.
We systematically develop BigBird, an attention mechanism whose complexity
is linear in the number of tokens (Sec. 2). We take inspiration from graph
sparsification methods and understand where the proof for expressiveness of Transformers
breaks down when full-attention is relaxed to form the proposed attention pattern. This understanding
helped us develop BigBird, which is theoretically as expressive and also empirically useful.
In particular, our BigBird consists of three main part:

- •

A set of g𝑔g global tokens attending on all parts of the sequence.

- •

All tokens attending to a set of w𝑤w local neighboring tokens.

- •

All tokens attending to a set of r𝑟r random tokens.

This leads to a high performing attention mechanism scaling to much longer sequence lengths (8x).

To summarize, our main contributions are:

- 1.

BigBird satisfies all the known theoretical properties of full transformer (Sec. 3). In particular, we show that adding extra tokens allows one to express all continuous sequence to sequence functions with only O​(n)𝑂𝑛O(n)-inner products. Furthermore, we show that under standard assumptions regarding precision, BigBird is Turing complete.

- 2.

Empirically, we show that the extended context modelled by BigBird benefits variety of NLP tasks.
We achieve state of the art results for question answering and document summarization on a number of different datasets.
Summary of these results are presented in Sec. 4.

- 3.

Lastly, we introduce a novel application of attention based models where long contexts
are beneficial: extracting contextual representations of genomics sequences like DNA.
With longer masked LM pretraining, BigBird improves performance on downstream tasks such as promoter-region and chromatin profile prediction (Sec. 5).

### 1.1 Related Work

There have been a number of interesting attempts, that were aimed at alleviating the quadratic dependency of Transformers, which can broadly categorized into two directions.
First line of work embraces the length limitation and develops method around it.
Simplest methods in this category just employ sliding window [93], but in general most work fits in the following general paradigm:
using some other mechanism select a smaller subset of relevant contexts to feed in the transformer and optionally iterate, i.e. call transformer block multiple time with different contexts each time.
Most prominently, SpanBERT [42], ORQA [54], REALM [34], RAG [57] have achieved strong performance for different tasks. However, it is worth noting that these methods often require significant engineering efforts (like back prop through large scale nearest neighbor search) and are hard to train.

Second line of work questions if full attention is essential and have tried to come up with approaches that do not require full attention, thereby reducing the memory and computation requirements.
Prominently, Dai et al. [21], Sukhbaatar et al. [82], Rae et al. [74] have proposed auto-regresive models that work well for left-to-right language modeling but suffer in tasks which require bidirectional context.
Child et al. [16] proposed a sparse model that reduces the complexity to O​(n​n)𝑂𝑛𝑛O(n\sqrt{n}), Kitaev et al. [49] further reduced the complexity to O​(n​log⁡(n))𝑂𝑛𝑛O(n\log(n)) by using LSH to compute nearest neighbors.
Ye et al. [103] proposed binary partitions of the data where as Qiu et al. [73] reduced complexity by using block sparsity.
Recently, Longformer [beltagy2020longformer] introduced a localized sliding
window based mask with few global mask to reduce computation and extended BERT to longer sequence based tasks.
Finally, our work is closely related to and built on the work of Extended Transformers Construction [ainslie2020etc].
This work was designed to encode structure in text for transformers. The idea of global tokens was used extensively by them to achieve their goals. Our theoretical work can be seen as providing a justification for the success of these models as well.
It is important to note that most of the aforementioned methods are heuristic based and empirically are not as versatile and robust as the original transformer, i.e. the same architecture do not attain SoTA on multiple standard benchmarks. (There is one exception of Longformer which we include in all our comparisons, see Sec. E.3 for a more detailed comparison).
Moreover, these approximations do not come with theoretical guarantees.

(a) Random attention

(b) Window attention

(c) Global Attention

(d) BigBird

## 2 BigBird Architecture

In this section, we describe the BigBird model using the generalised attention mechanism that is used in each layer of transformer operating on an input sequence 𝑿=(𝒙1,…,𝒙n)∈ℝn×d𝑿subscript𝒙1…subscript𝒙𝑛superscriptℝ𝑛𝑑{\bm{X}}=({\bm{x}}_{1},...,{\bm{x}}_{n})\in\mathbb{R}^{n\times d}.
The generalized attention mechanism is described by a directed graph D𝐷D whose vertex set is [n]={1,…,n}delimited-[]𝑛1…𝑛[n]=\{1,\dots,n\}.
The set of arcs (directed edges) represent the set of inner products that the attention mechanism will consider.
Let N​(i)𝑁𝑖N(i) denote the out-neighbors set of node i𝑖i in D𝐷D, then the ithsuperscript𝑖thi^{\text{th}} output vector of the generalized attention mechanism is defined as \useshortskip

AttnD​(𝑿)i=𝒙i+∑h=1Hσ​(Qh​(𝒙i)​Kh​(𝑿N​(i))T)⋅Vh​(𝑿N​(i))subscriptAttn𝐷subscript𝑿𝑖subscript𝒙𝑖superscriptsubscriptℎ1𝐻⋅𝜎subscript𝑄ℎsubscript𝒙𝑖subscript𝐾ℎsuperscriptsubscript𝑿𝑁𝑖𝑇subscript𝑉ℎsubscript𝑿𝑁𝑖\vspace{-2mm}\small\textsc{Attn}_{D}({\bm{X}})_{i}={\bm{x}}_{i}+\sum_{h=1}^{H}\sigma\left(Q_{h}({\bm{x}}_{i})K_{h}({\bm{X}}_{N(i)})^{T}\right)\cdot V_{h}({\bm{X}}_{N(i)})

(AT)

where Qh,Kh:ℝd→ℝm:subscript𝑄ℎsubscript𝐾ℎ→superscriptℝ𝑑superscriptℝ𝑚Q_{h},K_{h}:\mathbb{R}^{d}\to\mathbb{R}^{m} are query and key functions respectively, Vh:ℝd→ℝd:subscript𝑉ℎ→superscriptℝ𝑑superscriptℝ𝑑V_{h}:\mathbb{R}^{d}\to\mathbb{R}^{d} is a value function, σ𝜎\sigma is a scoring function (e.g. softmax or hardmax) and H𝐻H denotes the number of heads.
Also note XN​(i)subscript𝑋𝑁𝑖X_{N(i)} corresponds to the matrix formed by only stacking {𝒙j:j∈N​(i)}conditional-setsubscript𝒙𝑗𝑗𝑁𝑖\{{\bm{x}}_{j}:j\in N(i)\} and not all the inputs.
If D𝐷D is the complete digraph, we recover the full quadratic attention mechanism of vaswani2017attention.
To simplify our exposition, we will operate on the adjacency matrix A𝐴A of the graph D𝐷D even though the underlying graph maybe sparse.
To elaborate, A∈[0,1]n×n𝐴superscript01𝑛𝑛A\in[0,1]^{n\times n} with A​(i,j)=1𝐴𝑖𝑗1A(i,j)=1 if query i𝑖i attends to key j𝑗j and is zero otherwise.
For example, when A𝐴A is the ones matrix (as in BERT), it leads to quadratic complexity, since all tokens attend on every other token.
This view of self-attention as a fully connected graph allows us to exploit existing graph theory to help reduce its complexity.
The problem of reducing the quadratic complexity of self-attention can now be seen as a graph sparsification problem.
It is well-known that random graphs are expanders and can approximate complete graphs in a number of different contexts including in their spectral properties [spielman2011spectral, hoory2006expander].
We believe sparse random graph for attention mechanism should have two desiderata: small average path length between nodes and a notion of locality, each of which we discuss below.

Let us consider the simplest random graph construction, known as Erdős-Rényi model, where each edge is independently chosen with a fixed probability.
In such a random graph with just Θ~​(n)~Θ𝑛\tilde{\Theta}(n) edges, the shortest path between any two nodes is logarithmic in the number of nodes [chung2002average, katzav2018distribution].
As a consequence, such a random graph approximates the complete graph spectrally and its second eigenvalue (of the adjacency matrix) is quite far from the first eigenvalue [benaych2019largest, benaych2020spectral, alt2019extremal].
This property leads to a rapid mixing time for random walks in the grpah, which informally suggests that information can flow fast between any pair of nodes.
Thus, we propose a sparse attention where each query attends over r𝑟r random number of keys i.e. A​(i,⋅)=1𝐴𝑖⋅1A(i,\cdot)=1 for r𝑟r randomly chosen keys (see Fig. 1(a)).

The second viewpoint which inspired the creation of BigBird is that most contexts within NLP and computational biology have data which displays a great deal of locality of reference.
In this phenomenon, a great deal of information about a token can be derived from its neighboring tokens.
Most pertinently, clark2019does investigated self-attention models in NLP tasks and concluded that that neighboring inner-products are extremely important.
The concept of locality, proximity of tokens in linguistic structure, also forms the basis of various linguistic theories such as transformational-generative grammar.
In the terminology of graph theory, clustering coefficient is a measure of locality of connectivity, and is high when the graph contains many cliques or near-cliques (subgraphs that are almost fully interconnected).
Simple Erdős-Rényi random graphs do not have a high clustering coefficient [sussman2017clusteringcoeff], but
a class of random graphs, known as small world graphs, exhibit high clustering coefficient [watts1998collective].
A particular model introduced by watts1998collective is of high relevance to us as it achieves a good balance between average shortest path and the notion of locality.
The generative process of their model is as follows:
Construct a regular ring lattice, a graph with n𝑛n nodes each connected to w𝑤w neighbors, w𝑤w/2 on each side.

Model
MLM
SQuAD
MNLI

BERT-base
64.2
88.5
83.4

Random (R)
60.1
83.0
80.2

Window (W)
58.3
76.4
73.1

R + W
62.7
85.1
80.5

In other words we begin with a sliding window on the nodes.
Then a random subset (k𝑘k%) of all connections is replaced with a random connection.
The other (100 - k𝑘k)% local connections are retained.
However, deleting such random edges might be inefficient on modern hardware, so we retain it, which will not affect its properties.
In summary, to capture these local structures in the context, in BigBird, we define a sliding window attention, so that during self attention of width w𝑤w, query at location i𝑖i attends from i−w/2𝑖𝑤2i-w/2 to i+w/2𝑖𝑤2i+w/2 keys.
In our notation, A(i,i−w/2:i+w/2)=1A(i,i-w/2:i+w/2)=1 (see Fig. 1(b)).
As an initial sanity check, we performed basic experiments to test whether these intuitions are sufficient in getting performance close to BERT like models, while keeping attention linear in the number of tokens.
We found that random blocks and local window were insufficient in capturing all the context necessary to
compete with the performance of BERT.

The final piece of BigBird is inspired from our theoretical analysis (Sec. 3), which is critical for empirical performance.
More specifically, our theory utilizes the importance of “global tokens” (tokens that attend to all tokens in the sequence and to whom all tokens attend to (see Fig. 1(c)).
These global tokens can be defined in two ways:

- •

BigBird-itc: In internal transformer construction (itc), we make some existing tokens “global”, which attend over the entire sequence. Concretely, we choose a subset G𝐺G of indices
(with g:=|G|assign𝑔𝐺g:=|G|), such that A​(i,:)=1𝐴𝑖:1A(i,:)=1 and A​(:,i)=1𝐴:𝑖1A(:,i)=1 for all i∈G𝑖𝐺i\in G.

- •

BigBird-etc: In extended transformer construction (etc), we include additional “global” tokens such as CLS. Concretely, we add g𝑔g global tokens that attend to all existing tokens. In our notation, this corresponds to creating a new matrix B∈[0,1](N+g)×(N+g)𝐵superscript01𝑁𝑔𝑁𝑔B\in[0,1]^{(N+g)\times(N+g)} by adding g𝑔g rows to matrix A𝐴A, such that B​(i,:)=1𝐵𝑖:1B(i,:)=1, and B​(:,i)=1𝐵:𝑖1B(:,i)=1 for all i∈{1,2,…​g}𝑖12…𝑔i\in\{1,2,\ldots g\}, and B​(g+i,g+j)=A​(i,j)​∀i,j∈{1,…,N}formulae-sequence𝐵𝑔𝑖𝑔𝑗𝐴𝑖𝑗for-all𝑖𝑗1…𝑁B(g+i,g+j)=A(i,j)\forall\ i,j\in\{1,\ldots,N\}. This adds extra location to store context and as we will see in the experiments improves performance.

The final attention mechanism for BigBird (Fig. 1(d)) has all three of these properties: queries attend to r𝑟r random keys, each query attends to w/2𝑤2w/2 tokens to the left of its location and w/2𝑤2w/2 to the right of its location and they contain g𝑔g global tokens (The global tokens can be from existing tokens or extra added tokens).
We provide implementation details in App. D.

## 3 Theoretical Results about Sparse Attention Mechanism

In this section, we will show that that sparse attention mechanisms are as
powerful and expressive as full-attention mechanisms in two respects.
First, we show that when sparse attention mechanisms are used in a standalone encoder
(such as BERT), they are Universal Approximators of sequence
to sequence functions in the style of Yun19.
We note that this property was also explored theoretically in contemporary work yun2020on.
Second, unlike [yun2020on], we further show that sparse encoder-decoder transformers are Turing Complete
(assuming the same conditions defined in [Perez19]).
Complementing the above positive results, we also show that moving to a sparse-attention mechanism
incurs a cost, i.e. there is no free lunch. In Sec. 3.4, we show lower bounds
by exhibiting a natural task where any sufficiently sparse mechanism will
require polynomially more layers.

### 3.1 Notation

The complete Transformer encoder stack is nothing but the repeated application of a single-layer encoder (with independent parameters).
We denote class of such Transformer encoders stack, defined using generalized encoder (Sec. 2), by 𝒯DH,m,qsuperscriptsubscript𝒯𝐷𝐻𝑚𝑞\mathcal{T}_{D}^{H,m,q} which consists of H𝐻H-heads with head size m𝑚m and q𝑞q is the hidden layer size of the output network, and the attention layer is defined by the directed graph D𝐷D.

The key difference between our proposed attention mechanism to that of vaswani2017attention, Yun19 is that we add a special token at the beginning of each sequence and assign it a special vector.
We will refer to this as 𝒙0subscript𝒙0{\bm{x}}_{0}.
Therefore our graph D𝐷D will have vertex set {0}∪[n]={0,1,2,…,n}0delimited-[]𝑛012…𝑛\{0\}\cup[n]=\{0,1,2,\dots,n\}.
We will assume that this extra node and its respective vector will be dropped at the final output layer of transformer.
To avoid cumbersome notation, we will still treat transformer as mapping sequences
𝑿∈ℝn×d𝑿superscriptℝ𝑛𝑑{\bm{X}}\in\mathbb{R}^{n\times d} to ℝn×dsuperscriptℝ𝑛𝑑\mathbb{R}^{n\times d}. We will also allow the
transformer to append position embeddings E∈ℝd×n𝐸superscriptℝ𝑑𝑛E\in\mathbb{R}^{d\times n} to
matrix X𝑋X in the input layer.

Finally, we need to define the function class and distance measure for proving universal approximation property.
Let ℱC​Dsubscriptℱ𝐶𝐷\mathcal{F}_{CD} denote the set of continuous functions f:[0,1]n×d→ℝn×d:𝑓→superscript01𝑛𝑑superscriptℝ𝑛𝑑f:[0,1]^{n\times d}\to\mathbb{R}^{n\times d}
which are continuous with respect to the topology defined by ℓpsubscriptℓ𝑝\ell_{p} norm.
Recall for any p≥1𝑝1p\geq 1, the ℓpsubscriptℓ𝑝\ell_{p} distance is dp​(f1,f2)=(∫‖f1​(X)−f2​(X)‖pp​𝑑X)1/psubscript𝑑𝑝subscript𝑓1subscript𝑓2superscriptsuperscriptsubscriptnormsubscript𝑓1𝑋subscript𝑓2𝑋𝑝𝑝differential-d𝑋1𝑝d_{p}(f_{1},f_{2})=\left(\int\|f_{1}(X)-f_{2}(X)\|_{p}^{p}dX\right)^{1/p}.

### 3.2 Universal Approximators

###### Definition 1.

The star-graph S𝑆S centered at 00 is the graph defined on {0,…,n}0…𝑛\{0,\dots,n\}.
The neighborhood of all vertices i𝑖i is N​(i)={0,i}𝑁𝑖0𝑖N(i)=\{0,i\} for i∈{1​…​n}𝑖1…𝑛i\in\{1\dots n\} and
N​(0)={1,…​n}𝑁01…𝑛N(0)=\{1,\dots n\}.

Our main theorem is that the sparse attention mechanism defined by any graph containing S𝑆S
is a universal approximator:

###### Theorem 1.

Given 1<p<∞1𝑝1<p<\infty and ϵ>0italic-ϵ0\epsilon>0, for any f∈ℱC​D𝑓subscriptℱ𝐶𝐷f\in\mathcal{F}_{CD}, there exists a
transformer with sparse-attention, g∈𝒯DH,m,q𝑔superscriptsubscript𝒯𝐷𝐻𝑚𝑞g\in\mathcal{T}_{D}^{H,m,q} such that
dp​(f,g)≤ϵsubscript𝑑𝑝𝑓𝑔italic-ϵd_{p}(f,g)\leq\epsilon where D𝐷D is any graph containing star graph S𝑆S.

To prove the theorem, we will follow the standard proof structure outlined in [Yun19].

Step 1: Approximate ℱC​Dsubscriptℱ𝐶𝐷\mathcal{F}_{CD} by piece-wise constant functions.
Since f𝑓f is a continuous function with bounded domain [0,1)n×dsuperscript01𝑛𝑑[0,1)^{n\times d}, we will
approximate it with a suitable piece-wise constant function. This is accomplished by a suitable partition of the region [0,1)01[0,1) into a grid of granularity δ𝛿\delta to get a discrete set 𝔾δsubscript𝔾𝛿\mathbb{G}_{\delta}. Therefore, we can assume that we are dealing with a function
f¯:𝔾δ→ℝn×d:¯𝑓→subscript𝔾𝛿superscriptℝ𝑛𝑑\bar{f}:\mathbb{G}_{\delta}\to\mathbb{R}^{n\times d}, where dp​(f,f¯)≤ϵ3subscript𝑑𝑝𝑓¯𝑓italic-ϵ3d_{p}(f,\bar{f})\leq\frac{\epsilon}{3}.

Step 2: Approximate piece-wise constant functions by modified transformers.
This is the key step of the proof where the self-attention mechanism is used to generate a contextual-mapping of the input. Informally, a contextual mapping is a unique code
for the pair consisting of a matrix (𝑿,𝒙i)𝑿subscript𝒙𝑖({\bm{X}},{\bm{x}}_{i}) and a column. Its uniqueness allows
the Feed forward layers to use each code to map it to a unique output column.

The main technical challenge is computing the contextual mapping using only sparse attention mechanism.
This was done in [Yun19] using a “selective” shift operator which shift up entries that are in a
specific interval. Key to their proof was the fact that the shift, was exactly the range of the largest
entry to the smallest entry.

Creating a contextual mapping with a sparse attention mechanism is quite a challenge.
In particular, because each query only attends to a few keys, it is not at all clear
that sufficient information can be corralled to make a contextual embedding of the
entire matrix. To get around this, we develop a sparse shift operator which shifts the entries of the matrices if they lie in a certain range. The exact amount of the shift is controlled by the directed sparse attention graphg D𝐷D. The second key ingredient is the use of additional global token. By carefully applying the operator
to a set of chosen ranges, we will show that each column will contain a unique mapping of the
full mapping. Therefore, we can
augment the loss of inner-products in the self attention mechanism by using multiple
layers and an auxiliary global token.

Step 3: Approximate modified transformers by original Transformers: The final
step is to approximate the modified transformers by the original transformer which uses ReLU and
softmax.

We provide the full details in App. A.

### 3.3 Turing Completeness

Transformers are a very general class. In the original paper of vaswani2017attention, they were used
in both an encoder and a decoder. While the previous section outlined how powerful just the encoders were, another natural question is to ask what the
additional power of both a decoder along with an encoder is? Perez19 showed that the
full transformer based on a quadratic attention mechanism is Turing Complete. This result
makes one unrealistic assumption, which is that the model works on arbitrary precision
model. Of course, this is necessary as otherwise, Transformers are bounded finite
state machines and cannot be Turing Complete.

It is natural to ask if the full attention mechanism is necessary. Or can a
sparse attention mechanism also be used to simulate any Turing Machine?
We show that this is indeed the case: we can use a sparse encoder and sparse decoder
to simulate any Turing Machine.

To use the sparse attention mechanism in the transformer architecture, we need to
define a suitable modification where each token only reacts to previous tokens.
Unlike the case for BERT, where the entire attention mechanism is applied once, in full
transformers, the sparse attention mechanism at decoder side is used token by token.
Secondly the work of Perez19, uses each token as a representation of the tape
history and uses the full attention to move and retrieve the correct tape symbol.
Most of the construction of Perez19 goes through for sparse attentions, except for their
addressing scheme to point back in history (Lemma B.4 in [Perez19]).
We show how to simulate this using a sparse attention mechanism and defer the
details to App. B.

### 3.4 Limitations

We demonstrate a natural task which can be solved by the full attention mechanism in O​(1)𝑂1O(1)-layers.
However, under standard complexity theoretic assumptions, this problem requires
Ω~​(n)~Ω𝑛\tilde{\Omega}(n)-layers for any sparse attention layers with O~​(n)~𝑂𝑛\tilde{O}(n) edges (not just BigBird). (Here O~~𝑂\tilde{O} hides poly-logarthmic factors).
Consider the simple problem of finding the corresponding furthest vector
for each vector in the given sequence of length n𝑛n. Formally,

Task 1. Given n𝑛n unit vectors {u1,…,un}subscript𝑢1…subscript𝑢𝑛\{u_{1},\dots,u_{n}\}, find f​(u1,…,un)→(u1∗,…,un∗)→𝑓subscript𝑢1…subscript𝑢𝑛subscript𝑢superscript1…subscript𝑢superscript𝑛f(u_{1},\dots,u_{n})\to(u_{1^{*}},\dots,u_{n^{*}}) where for a fixed j∈[n]𝑗delimited-[]𝑛j\in[n], we define j∗=arg​maxk⁡‖uk−uj‖22superscript𝑗subscriptargmax𝑘superscriptsubscriptnormsubscript𝑢𝑘subscript𝑢𝑗22j^{*}=\operatorname*{arg\,max}_{k}\|u_{k}-u_{j}\|_{2}^{2}.

Finding vectors that are furthest apart boils down to minimize
inner product search in case of unit vectors. For a full-attention mechanism
with appropriate query
and keys, this task is very easy as we can evaluate all pair-wise inner products.

The impossibility for sparse-attention follows from hardness results
stemming from Orthogonal Vector Conjecture(OVC) [abboud2014consequences, abboud2015tight, backurs2015edit, williams2005new]. The OVC is a widely used assumption
in fine-grained complexity. Informally, it states that one cannot determine if the minimum inner product among
n𝑛n boolean vectors is 00 in subquadratic time. In App. C, we show a
reduction using OVC to show that if a transformer g∈𝒯DH=1,m=2​d,q=0𝑔superscriptsubscript𝒯𝐷formulae-sequence𝐻1formulae-sequence𝑚2𝑑𝑞0g\in\mathcal{T}_{D}^{H=1,m=2d,q=0} for
any sparse directed graph D𝐷D can evaluate the Task 111, it can solve the orthogonal vector problem.

###### Proposition 1.

There exists a single layer full self-attention g∈𝒯H=1,m=2​d,q=0𝑔superscript𝒯formulae-sequence𝐻1formulae-sequence𝑚2𝑑𝑞0g\in\mathcal{T}^{H=1,m=2d,q=0} that can
evaluate Task 1, i.e. g​(u1,…,un)=[u1∗,…,un∗]𝑔subscript𝑢1…subscript𝑢𝑛subscript𝑢superscript1…subscript𝑢superscript𝑛g(u_{1},...,u_{n})=[u_{1^{*}},\dots,u_{n^{*}}], but for any sparse-attention
graph D𝐷D with O~​(n)~𝑂𝑛\tilde{O}(n) edges (i.e. inner product evaluations), would require Ω~​(n1−o​(1))~Ωsuperscript𝑛1𝑜1\tilde{\Omega}(n^{1-o(1)}) layers.

We give a formal proof of this fact in App. C.

## 4 Experiments: Natural Language Processing

In this section our goal is to showcase benefits of modeling longer input sequence for NLP tasks, for which we select three representative tasks.
We begin with basic masked language modeling (MLM; devlin2018bert) to check if better contextual representations can be learnt by utilizing longer contiguous sequences.
Next, we consider QA with supporting evidence, for which capability to handle longer sequence would allow us to retrieve more evidence using crude systems like TF-IDF/BM25.
Finally, we tackle long document classification where discriminating information may not be located in first 512 tokens.
Below we summarize the results for BigBird using sequence length 4096111code available at http://goo.gle/bigbird-transformer, while we defer all other setup details including computational resources, batch size, step size, to App. E.

##### Pretraining and MLM

We follow [devlin2018bert, liu2019roberta] to create base and large versions of BigBird and pretrain it using MLM objective.
This task involves predicting a random subset of tokens which have been masked out.
We use four standard data-sets for pretraining (listed in Sec. E.1, Tab. 10), warm-starting from the public RoBERTa checkpoint222https://github.com/pytorch/fairseq/tree/master/examples/roberta.
We compare performance in predicting the masked out tokens in terms of bits per character, following [beltagy2020longformer].
As seen in Sec. E.1, Tab. 10, both BigBird and Longformer perform better than limited length RoBERTa, with BigBird-etc performing the best.
We note that we trained our models on a
reasonable 16​G​B16𝐺𝐵16GB memory/chip with batch size of 32-64.
Our memory efficiency is due to efficient blocking and sparsity structure of
the sparse attention mechanism described in Sec. 2.

Model

HotpotQA

NaturalQ

TriviaQA

WikiHop

Ans
Sup
Joint

LA
SA

Full

MCQ

RoBERTa

73.5
83.4
63.5

-
-

74.3

72.4

Longformer

74.3
84.4
64.4

-
-

75.2

75.0

BigBird-itc

75.7
86.8
67.7

70.8
53.3

79.5

75.9

BigBird-etc

75.5
87.1
67.8

73.9
54.9

78.7

75.9

Model
HotpotQA

NaturalQ

TriviaQA

WikiHop

Ans
Sup
Joint

LA
SA

Full
Verified

MCQ

HGN [fang2019hierarchical]

82.2
88.5
74.2

-
-

-
-

-

GSAN
81.6
88.7
73.9

-
-

-
-

-

ReflectionNet [gong2020reflection]

-
-
-

77.1
64.1

-
-

-

RikiNet-v2 [liu2020rikinet]

-
-
-

76.1
61.3

-
-

-

Fusion-in-Decoder [izacard2020fid]

-
-
-

-
-

84.4
90.3

-

SpanBERT [joshi2020spanbert]

-
-
-

-
-

79.1
86.6

-

MRC-GCN [tang2020multi]

-
-
-

-
-

-
-

78.3

MultiHop [chen2019multi]

-
-
-

-
-

-
-

76.5

Longformer [beltagy2020longformer]

81.2
88.3
73.2

-
-

77.3
85.3

81.9

BigBird-etc

81.2
89.1
73.6

77.8
57.9

84.5
92.4

82.3

##### Question Answering (QA)

We considered following four challenging datasets:

- 1.

Natural Questions [kwiatkowski2019natural]:
For the given question, find a short span of answer (SA) from the given evidences as well highlight the paragraph from the given evidences
containing information about the correct answer (LA).

- 2.

HotpotQA-distractor [yang2018hotpotqa]: Similar to natural questions, it requires finding the answer (Ans) as well as the supporting facts (Sup) over different documents needed for multi-hop reasoning from the given evidences.

- 3.

TriviaQA-wiki [JoshiTriviaQA2017]: We need to provide an answer for the given question using provided Wikipedia evidence, however, the answer might not be present in the given evidence. On a smaller verified subset of question, the given evidence is guaranteed to contain the answer. Nevertheless, we model the answer as span selection problem in this case as well.

- 4.

WikiHop [welbl2018constructing]: Chose correct option from multiple-choice questions (MCQ), by aggregating information spread across multiple documents given in the evidences.

As these tasks are very competitive, multiple highly engineered systems have been designed specific each dataset confirming to respective output formats.
For a fair comparison, we had to use some additional regularization for training BigBird, details of which are provided in Sec. E.2 along with exact architecture description.
We experiment using the base sized model and select the best configuration on the development set for each dataset (as reported in Tab. 2).
We can see that BigBird-etc, with expanded global tokens consistently outperforms all other models.
Thus, we chose this configuration to train a large sized model to be used for evaluation on the hidden test set.

In Tab. 3, we compare BigBird-etc model to top-3 entries from the leaderboard excluding BigBird.
One can clearly see the importance of using longer context as both Longformer and BigBird outperform models with smaller contexts.
Also, it is worth noting that BigBird submission is a single model, whereas the other top-3 entries for Natural Questions are ensembles, which might explain the slightly lower accuracy in exact answer phrase selection.

##### Classification

We experiment on datasets of different lengths and contents, specifically various document classification and GLUE tasks.
Following BERT, we used one layer with cross entropy loss on top of the first [CLS] token.
We see that gains of using BigBird are more significant when we have longer documents and fewer training examples.
For instance, using base sized model,
BigBird improves state-of-the-art for Arxiv dataset by about 𝟓%percent5\bm{5\%} points.
On Patents dataset, there is improvement over using simple BERT/RoBERTa, but given the large size of training data the improvement over SoTA (which is not BERT based) is not significant.
Note that this performance gain is not seen for much
smaller IMDb dataset.
Along with experimental setup detail, we present detailed results in Sec. E.4 which show competitive performance.

Model

Arxiv

PubMed

BigPatent

R-1
R-2
R-L

R-1
R-2
R-L

R-1
R-2
R-L

Prior Art

SumBasic [nenkova2005impact]

29.47
6.95
26.30

37.15
11.36
33.43

27.44
7.08
23.66

LexRank [erkan2004lexrank]

33.85
10.73
28.99

39.19
13.89
34.59

35.57
10.47
29.03

LSA [wiseman2017challenges]

29.91
7.42
25.67

33.89
9.93
29.70

-
-
-

Attn-Seq2Seq [sutskever2014sequence]

29.30
6.00
25.56

31.55
8.52
27.38

28.74
7.87
24.66

Pntr-Gen-Seq2Seq [see2017get]

32.06
9.04
25.16

35.86
10.22
29.69

33.14
11.63
28.55

Long-Doc-Seq2Seq [cohan2018discourse]

35.80
11.05
31.80

38.93
15.37
35.21

-
-
-

Sent-CLF [subramanian2019extractive]

34.01
8.71
30.41

45.01
19.91
41.16

36.20
10.99
31.83

Sent-PTR [subramanian2019extractive]

42.32
15.63
38.06

43.30
17.92
39.47

34.21
10.78
30.07

Extr-Abst-TLM [subramanian2019extractive]

41.62
14.69
38.03

42.13
16.27
39.21

38.65
12.31
34.09

Dancer [gidiotis2020divide]

42.70
16.54
38.44

44.09
17.69
40.27

-
-
-

Base

Transformer

28.52
6.70
25.58

31.71
8.32
29.42

39.66
20.94
31.20

+ RoBERTa [rothe2019leveraging]

31.98
8.13
29.53

35.77
13.85
33.32

41.11
22.10
32.58

+ Pegasus [zhang2019pegasus]

34.81
10.16
30.14

39.98
15.15
35.89

43.55
20.43
31.80

BigBird-RoBERTa

41.22
16.43
36.96

43.70
19.32
39.99

55.69
37.27
45.56

Large

Pegasus (Reported) [zhang2019pegasus]

44.21
16.95
38.83

45.97
20.15
41.34

52.29
33.08
41.75

Pegasus (Re-eval)

43.85
16.83
39.17

44.53
19.30
40.70

52.25
33.04
41.80

BigBird-Pegasus

46.63
19.02
41.77

46.32
20.65
42.33

60.64
42.46
50.01

### 4.1 Encoder-Decoder Tasks

For an encoder-decoder setup, one can easily see that both suffer from quadratic complexity due to the full self attention.
We focus on introducing the sparse attention mechanism of BigBird only at the encoder side.
This is because, in practical generative applications, the length of output sequence is typically small as compared to the input.
For example for text summarization, we see in realistic scenarios (c.f. Sec. E.5 Tab. 18) that the median output sequence length is ∼200similar-toabsent200\sim 200 where as the input sequence’s median length is >3000absent3000>3000.
For such applications, it is more efficient to use sparse attention mechanism for the encoder and full self-attention for the decoder.

##### Summarization

Document summarization is a task of creating a short and accurate summary of a text document.
We used three long document datasets for testing our model details of which are mention in Tab. 18.
In this paper we focus on abstractive summarization of long documents where using a longer contextual encoder should improve performance.
The reasons are two fold:
First, the salient content can be evenly distributed in the long document, not just in first 512 tokens, and this is by design in the BigPatents dataset [sharma2019bigpatent].
Second, longer documents exhibit a richer discourse structure and summaries are considerably more abstractive, thereby observing more context helps.
As has been pointed out recently [rothe2019leveraging, zhang2019pegasus], pretraining helps in generative tasks, we warm start from our general purpose MLM pretraining on base-sized models as well as utilizing state-of-the-art summarization specific pretraining from Pegasus [zhang2019pegasus] on large-sized models.
The results of training BigBird sparse encoder along with full decoder on these long document datasets are presented in Tab. 4.
We can clearly see modeling longer context brings significant improvement.
Along with hyperparameters, we also present results on shorter but more widespread datasets in Sec. E.5, which show that using sparse attention does not hamper performance either.

## 5 Experiments: Genomics

There has been a recent upsurge in using deep learning for genomics data [tampuu2019viraminer, zhang2019ncnet, busia2019deep], which has resulted in improved performance on several biologically-significant tasks such as
promoter site prediction [oubounyt2019deepromoter], methylation analysis [levy2020methylnet],
predicting functional effects of non-coding variant [zhou2015predicting], etc.
These approaches consume DNA sequence fragments as inputs, and therefore we believe longer input sequence handling capability of BigBird would be beneficial as many functional effects in DNA are highly non-local [buldyrev1995long].
Furthermore, taking inspiration from NLP, we learn powerful contextual representations for DNA fragments utilizing abundant unlabeled data (e.g. human reference genome, Saccharomyces Genome Database) via MLM pretraining.
Next, we showcase that our long input BigBird along with the proposed pretraining significantly improves performances in two downstream tasks.
Detailed experimental setup for the two tasks are provided in App. F.

Model
BPC

SRILM [liang2012segmenting]

1.57

BERT (sqln. 512)
1.23

BigBird (sqln. 4096)

1.12

##### Pre-training and MLM

As explored in liang2012segmenting, instead of operating on base pairs, we propose to first segment DNA into tokens so as to further increase the context length (App. F, Fig. 7).
In particular, we build a byte-pair encoding [kudo2018sentencepiece] table for the DNA sequence of size 32K, with each token representing 8.78 base pairs on average.
We learn contextual representation of these token on the human reference genome (GRCh37)333https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.13/ using MLM objective.
We then report the bits per character (BPC)
on a held-out set in Tab. 5.
We find that attention based contextual representation of DNA does improve BPC, which is further improved by using longer context.

Model
F1

CNNProm [umarov2017recognition]

69.7

DeePromoter [oubounyt2019deepromoter]

95.6

BigBird
99.9

##### Promoter Region Prediction

Promoter is a DNA region typically located upstream of the gene, which is the site of transcription initiation.
Multiple methods have been proposed to identify the promoter regions in a given DNA sequence [yang2017exploiting, lin2017identifying, bharanikumar2018promoterpredict, xiao2019ipsw, oubounyt2019deepromoter], as it is an important first step in understanding gene regulation.
The corresponding machine learning task is to classify a given DNA fragment as promoter or non-promoter sequence. We use the dataset compiled by oubounyt2019deepromoter which was built from Eukaryotic Promoter Database (EPDnew) [dreos2013epd]
444 https://epd.epfl.ch/human/human_database.php?db=human.
We finetuned the pretrained BigBird model from above, using the training data and report F1 on test dataset.
We compare our results to the previously reported best method in Tab. 6.
We see that
BigBird achieve nearly perfect accuracy with a 5%percent55\% jump from the previous best reported accuracy.

Model
TF
HM
DHS

gkm-SVM [ghandi2014enhanced]

89.6
-
-

DeepSea [zhou2015predicting]

95.8
85.6
92.3

BigBird
96.1
88.7
92.1

##### Chromatin-Profile Prediction

Non-coding regions of DNA do not code for proteins.
Majority of diseases and other trait associated single-nucleotide polymorphism are correlated to non-coding genomic variations [zhou2015predicting, khurana2016role].
Thus, understanding the functional effects of non-coding regions of DNA is a very important task. An important step in this process, as defined by zhou2015predicting, is to predict large-scale chromatin-profiling from non-coding genomic sequence.
To this effect, DeepSea [zhou2015predicting], compiled 919 chromatin-profile of 2.4M non-coding variants from Encyclopedia of DNA Elements (ENCODE)555https://www.encodeproject.org/ and Roadmap Epigenomics projects666http://www.roadmapepigenomics.org/.
The corresponding ML task is to predict, for a given non-coding region of DNA, these 919 chromatin-profile including 690690690 transcription factors (TF) binding profiles for 160160160 different TFs, 125125125 DNase I sensitivity (DHS) profiles and 104104104 histone-mark (HM) profiles.
We jointly learn 919 binary classifiers to predict these functional effects from sequence of DNA fragments.
On held-out chromosomes, we compare AUC with the baselines in Tab. 7 and see that we significantly improve on performance on the harder task HM, which is known to have longer-range correlations [gates2017histone] than others.

## 6 Conclusion

We propose BigBird: a sparse attention mechanism that is linear in the
number of tokens. BigBird satisfies a number of theoretical results:
it is a universal approximator of sequence to sequence functions and is also
Turing complete. Theoretically, we use the power of extra global tokens
preserve the expressive powers of the model.
We complement these results by showing that moving to sparse attention mechanism
do incur a cost.
Empirically, BigBird gives state-of-the-art performance on a number of NLP tasks such as
question answering and long document classification. We further introduce attention based
contextual language model for DNA and fine-tune it for down
stream tasks such as promoter region prediction and predicting effects of non-coding variants.

## References

- Abboud et al. [2014]

A. Abboud, V. V. Williams, and O. Weimann.

Consequences of faster alignment of sequences.

In International Colloquium on Automata, Languages, and
Programming, pages 39–51. Springer, 2014.

- Abboud et al. [2015]

A. Abboud, A. Backurs, and V. V. Williams.

Tight hardness results for lcs and other sequence similarity
measures.

In 2015 IEEE 56th Annual Symposium on Foundations of Computer
Science, pages 59–78. IEEE, 2015.

- Abreu et al. [2019]

J. Abreu, L. Fred, D. Macêdo, and C. Zanchettin.

Hierarchical attentional hybrid neural networks for document
classification.

In International Conference on Artificial Neural Networks,
pages 396–402. Springer, 2019.

- Ainslie et al. [2020]

J. Ainslie, S. Ontanon, C. Alberti, P. Pham, A. Ravula, and S. Sanghai.

Etc: Encoding long and structured data in transformers.

arXiv preprint arXiv:2004.08483, 2020.

- Alberti et al. [2019]

C. Alberti, K. Lee, and M. Collins.

A bert baseline for the natural questions.

arXiv preprint arXiv:1901.08634, 2019.

- Alt et al. [2019]

J. Alt, R. Ducatez, and A. Knowles.

Extremal eigenvalues of critical erd\\\backslashh {{\{o}}\}
sr\\\backslash’enyi graphs.

arXiv preprint arXiv:1905.03243, 2019.

- Backurs and Indyk [2015]

A. Backurs and P. Indyk.

Edit distance cannot be computed in strongly subquadratic time
(unless seth is false).

In Proceedings of the forty-seventh annual ACM symposium on
Theory of computing, pages 51–58, 2015.

- Beltagy et al. [2020]

I. Beltagy, M. E. Peters, and A. Cohan.

Longformer: The long-document transformer.

arXiv preprint arXiv:2004.05150, 2020.

- Benaych-Georges et al. [2019]

F. Benaych-Georges, C. Bordenave, A. Knowles, et al.

Largest eigenvalues of sparse inhomogeneous erdős–rényi
graphs.

Annals of Probability, 47(3):1653–1676,
2019.

- Benaych-Georges et al. [2020]

F. Benaych-Georges, C. Bordenave, A. Knowles, et al.

Spectral radii of sparse random matrices.

In Annales de l’Institut Henri Poincaré, Probabilités
et Statistiques, volume 56, pages 2141–2161. Institut Henri Poincaré,
2020.

- Bharanikumar et al. [2018]

R. Bharanikumar, K. A. R. Premkumar, and A. Palaniappan.

Promoterpredict: sequence-based modelling of escherichia coli
σ𝜎\sigma70 promoter strength yields logarithmic dependence between promoter
strength and sequence.

PeerJ, 6:e5862, 2018.

- Buldyrev et al. [1995]

S. Buldyrev, A. Goldberger, S. Havlin, R. Mantegna, M. Matsa, C.-K. Peng,
M. Simons, and H. Stanley.

Long-range correlation properties of coding and noncoding dna
sequences: Genbank analysis.

Physical Review E, 51(5):5084, 1995.

- Busia et al. [2019]

A. Busia, G. E. Dahl, C. Fannjiang, D. H. Alexander, E. Dorfman, R. Poplin,
C. Y. McLean, P.-C. Chang, and M. DePristo.

A deep learning approach to pattern recognition for short dna
sequences.

BioRxiv, page 353474, 2019.

- Chen et al. [2019a]

J. Chen, S.-t. Lin, and G. Durrett.

Multi-hop question answering via reasoning chains.

arXiv preprint arXiv:1910.02610, 2019a.

- Chen et al. [2019b]

Y.-C. Chen, Z. Gan, Y. Cheng, J. Liu, and J. Liu.

Distilling the knowledge of bert for text generation.

arXiv preprint arXiv:1911.03829, 2019b.

- Child et al. [2019]

R. Child, S. Gray, A. Radford, and I. Sutskever.

Generating long sequences with sparse transformers.

arXiv preprint arXiv:1904.10509, 2019.

- Chung and Lu [2002]

F. Chung and L. Lu.

The average distances in random graphs with given expected degrees.

Proceedings of the National Academy of Sciences, 99(25):15879–15882, 2002.

- Clark and Gardner [2017]

C. Clark and M. Gardner.

Simple and effective multi-paragraph reading comprehension.

arXiv preprint arXiv:1710.10723, 2017.

- Clark et al. [2019]

K. Clark, U. Khandelwal, O. Levy, and C. D. Manning.

What does bert look at? an analysis of bert’s attention.

arXiv preprint arXiv:1906.04341, 2019.

- Cohan et al. [2018]

A. Cohan, F. Dernoncourt, D. S. Kim, T. Bui, S. Kim, W. Chang, and N. Goharian.

A discourse-aware attention model for abstractive summarization of
long documents.

arXiv preprint arXiv:1804.05685, 2018.

- Dai et al. [2019]

Z. Dai, Z. Yang, Y. Yang, J. Carbonell, Q. V. Le, and R. Salakhutdinov.

Transformer-xl: Attentive language models beyond a fixed-length
context.

arXiv:1901.02860, 2019.

- Devlin et al. [2018]

J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova.

Bert: Pre-training of deep bidirectional transformers for language
understanding.

arXiv preprint arXiv:1810.04805, 2018.

- Dong et al. [2019]

L. Dong, N. Yang, W. Wang, F. Wei, X. Liu, Y. Wang, J. Gao, M. Zhou, and H.-W.
Hon.

Unified language model pre-training for natural language
understanding and generation.

In Advances in Neural Information Processing Systems, pages
13042–13054, 2019.

- Dreos et al. [2013]

R. Dreos, G. Ambrosini, R. Cavin Périer, and P. Bucher.

Epd and epdnew, high-quality promoter resources in the
next-generation sequencing era.

Nucleic acids research, 41(D1):D157–D164,
2013.

- Erkan and Radev [2004]

G. Erkan and D. R. Radev.

Lexrank: Graph-based lexical centrality as salience in text
summarization.

Journal of artificial intelligence research, 22:457–479, 2004.

- Fang et al. [2019]

Y. Fang, S. Sun, Z. Gan, R. Pillai, S. Wang, and J. Liu.

Hierarchical graph network for multi-hop question answering.

arXiv preprint arXiv:1911.03631, 2019.

- Gates et al. [2017]

L. A. Gates, C. E. Foulds, and B. W. O’Malley.

Histone marks in the ‘driver’s seat’: functional roles in
steering the transcription cycle.

Trends in biochemical sciences, 42(12):977–989, 2017.

- Gehring et al. [2017]

J. Gehring, M. Auli, D. Grangier, D. Yarats, and Y. N. Dauphin.

Convolutional sequence to sequence learning.

In Proceedings of the 34th International Conference on Machine
Learning-Volume 70, pages 1243–1252. JMLR. org, 2017.

- Gehrmann et al. [2018]

S. Gehrmann, Y. Deng, and A. M. Rush.

Bottom-up abstractive summarization.

arXiv preprint arXiv:1808.10792, 2018.

- Ghandi et al. [2014]

M. Ghandi, D. Lee, M. Mohammad-Noori, and M. A. Beer.

Enhanced regulatory sequence prediction using gapped k-mer features.

PLoS computational biology, 10(7), 2014.

- Gidiotis and Tsoumakas [2020]

A. Gidiotis and G. Tsoumakas.

A divide-and-conquer approach to the summarization of academic
articles.

arXiv preprint arXiv:2004.06190, 2020.

- Gong [2020 (accessed June 3, 2020]

M. Gong.

ReflectionNet, 2020 (accessed June 3, 2020).

URL https://www.microsoft.com/en-us/research/people/migon/.

- Gray et al. [2017]

S. Gray, A. Radford, and D. P. Kingma.

Gpu kernels for block-sparse weights.

arXiv preprint arXiv:1711.09224, 3, 2017.

- Guu et al. [2020]

K. Guu, K. Lee, Z. Tung, P. Pasupat, and M.-W. Chang.

Realm: Retrieval-augmented language model pre-training.

arXiv preprint arXiv:2002.08909, 2020.

- He et al. [2019]

J. He, L. Wang, L. Liu, J. Feng, and H. Wu.

Long document classification from local word glimpses via recurrent
attention learning.

IEEE Access, 7:40707–40718, 2019.

- Hermann et al. [2015]

K. M. Hermann, T. Kocisky, E. Grefenstette, L. Espeholt, W. Kay, M. Suleyman,
and P. Blunsom.

Teaching machines to read and comprehend.

In Advances in neural information processing systems, pages
1693–1701, 2015.

- Hochreiter and Schmidhuber [1997]

S. Hochreiter and J. Schmidhuber.

Long short-term memory.

Neural computation, 9(8):1735–1780, 1997.

- Hoory et al. [2006]

S. Hoory, N. Linial, and A. Wigderson.

Expander graphs and their applications.

Bulletin of the American Mathematical Society, 43(4):439–561, 2006.

- Izacard and Grave [2020]

G. Izacard and E. Grave.

Leveraging passage retrieval with generative models for open domain
question answering.

arXiv preprint arXiv:2007.01282, 2020.

- Jiang et al. [2019]

Y. Jiang, J. Petrak, X. Song, K. Bontcheva, and D. Maynard.

Team bertha von suttner at semeval-2019 task 4: Hyperpartisan news
detection using elmo sentence representation convolutional network.

In Proceedings of the 13th International Workshop on Semantic
Evaluation, pages 840–844, 2019.

- Joshi et al. [2017]

M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer.

Triviaqa: A large scale distantly supervised challenge dataset for
reading comprehension.

In Proceedings of the 55th Annual Meeting of the Association
for Computational Linguistics, Vancouver, Canada, July 2017. Association for
Computational Linguistics.

- Joshi et al. [2020]

M. Joshi, D. Chen, Y. Liu, D. S. Weld, L. Zettlemoyer, and O. Levy.

Spanbert: Improving pre-training by representing and predicting
spans.

Transactions of the Association for Computational Linguistics,
8:64–77, 2020.

- Katzav et al. [2018]

E. Katzav, O. Biham, and A. K. Hartmann.

Distribution of shortest path lengths in subcritical
erdős-rényi networks.

Physical Review E, 98(1):012301, 2018.

- Kent et al. [2002]

W. J. Kent, C. W. Sugnet, T. S. Furey, K. M. Roskin, T. H. Pringle, A. M.
Zahler, and D. Haussler.

The human genome browser at ucsc.

Genome research, 12(6):996–1006, 2002.

- Khandelwal et al. [2019]

U. Khandelwal, K. Clark, D. Jurafsky, and L. Kaiser.

Sample efficient text summarization using a single pre-trained
transformer.

arXiv preprint arXiv:1905.08836, 2019.

- Khurana et al. [2016]

E. Khurana, Y. Fu, D. Chakravarty, F. Demichelis, M. A. Rubin, and M. Gerstein.

Role of non-coding sequence variants in cancer.

Nature Reviews Genetics, 17(2):93, 2016.

- Kiesel et al. [2019]

J. Kiesel, M. Mestre, R. Shukla, E. Vincent, P. Adineh, D. Corney, B. Stein,
and M. Potthast.

Semeval-2019 task 4: Hyperpartisan news detection.

In Proceedings of the 13th International Workshop on Semantic
Evaluation, pages 829–839, 2019.

- Kim et al. [2018]

B. Kim, H. Kim, and G. Kim.

Abstractive summarization of reddit posts with multi-level memory
networks.

arXiv preprint arXiv:1811.00783, 2018.

- Kitaev et al. [2019]

N. Kitaev, L. Kaiser, and A. Levskaya.

Reformer: The efficient transformer.

In International Conference on Learning Representations, 2019.

- Kudo and Richardson [2018]

T. Kudo and J. Richardson.

Sentencepiece: A simple and language independent subword tokenizer
and detokenizer for neural text processing.

arXiv preprint arXiv:1808.06226, 2018.

- Kumar et al. [2020]

V. Kumar, A. Choudhary, and E. Cho.

Data augmentation using pre-trained transformer models.

arXiv preprint arXiv:2003.02245, 2020.

- Kwiatkowski et al. [2019]

T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh, C. Alberti,
D. Epstein, I. Polosukhin, J. Devlin, K. Lee, et al.

Natural questions: a benchmark for question answering research.

Transactions of the Association for Computational Linguistics,
7:453–466, 2019.

- Lee and Hsiang [2020]

J.-S. Lee and J. Hsiang.

Patent classification by fine-tuning bert language model.

World Patent Information, 61:101965, 2020.

- Lee et al. [2019]

K. Lee, M.-W. Chang, and K. Toutanova.

Latent retrieval for weakly supervised open domain question
answering.

arXiv preprint arXiv:1906.00300, 2019.

- Levy et al. [2020]

J. J. Levy, A. J. Titus, C. L. Petersen, Y. Chen, L. A. Salas, and B. C.
Christensen.

Methylnet: an automated and modular deep learning approach for dna
methylation analysis.

BMC bioinformatics, 21(1):1–15, 2020.

- Lewis et al. [2019]

M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy, V. Stoyanov,
and L. Zettlemoyer.

Bart: Denoising sequence-to-sequence pre-training for natural
language generation, translation, and comprehension.

arXiv preprint arXiv:1910.13461, 2019.

- Lewis et al. [2020]

P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal,
H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, et al.

Retrieval-augmented generation for knowledge-intensive nlp tasks.

arXiv preprint arXiv:2005.11401, 2020.

- Liang [2012]

W. Liang.

Segmenting dna sequence into words based on statistical language
model.

Nature Precedings, pages 1–1, 2012.

- Lin et al. [2017]

H. Lin, Z.-Y. Liang, H. Tang, and W. Chen.

Identifying sigma70 promoters with novel pseudo nucleotide
composition.

IEEE/ACM transactions on computational biology and
bioinformatics, 2017.

- Lin et al. [2003]

J. Lin, D. Quan, V. Sinha, K. Bakshi, D. Huynh, B. Katz, and D. R. Karger.

What makes a good answer? the role of context in question answering.

In Proceedings of the Ninth IFIP TC13 International Conference
on Human-Computer Interaction (INTERACT 2003), pages 25–32, 2003.

- Liu et al. [2020]

D. Liu, Y. Gong, J. Fu, Y. Yan, J. Chen, D. Jiang, J. Lv, and N. Duan.

Rikinet: Reading wikipedia pages for natural question answering.

arXiv preprint arXiv:2004.14560, 2020.

- Liu and Lapata [2019]

Y. Liu and M. Lapata.

Text summarization with pretrained encoders.

arXiv preprint arXiv:1908.08345, 2019.

- Liu et al. [2019]

Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis,
L. Zettlemoyer, and V. Stoyanov.

Roberta: A robustly optimized bert pretraining approach.

arXiv preprint arXiv:1907.11692, 2019.

- Maas et al. [2011]

A. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng, and C. Potts.

Learning word vectors for sentiment analysis.

In Proceedings of the 49th annual meeting of the association
for computational linguistics: Human language technologies, pages 142–150,
2011.

- Martin et al. [2019]

L. Martin, B. Muller, P. J. O. Suárez, Y. Dupont, L. Romary, É. V.
de la Clergerie, D. Seddah, and B. Sagot.

Camembert: a tasty french language model.

arXiv preprint arXiv:1911.03894, 2019.

- Miller [2019]

D. Miller.

Leveraging bert for extractive text summarization on lectures.

arXiv preprint arXiv:1906.04165, 2019.

- Narayan et al. [2018]

S. Narayan, S. B. Cohen, and M. Lapata.

Don’t give me the details, just the summary! topic-aware
convolutional neural networks for extreme summarization.

arXiv preprint arXiv:1808.08745, 2018.

- Nenkova and Vanderwende [2005]

A. Nenkova and L. Vanderwende.

The impact of frequency on summarization.

Microsoft Research, Redmond, Washington, Tech. Rep.
MSR-TR-2005, 101, 2005.

- Olson et al. [2019]

M. L. Olson, L. Zhang, and C.-N. Yu.

Adapting pretrained language models for long document classification.

OpenReview, 2019.

- Oord et al. [2018]

A. v. d. Oord, Y. Li, and O. Vinyals.

Representation learning with contrastive predictive coding.

arXiv preprint arXiv:1807.03748, 2018.

- Oubounyt et al. [2019]

M. Oubounyt, Z. Louadi, H. Tayara, and K. T. Chong.

Deepromoter: Robust promoter predictor using deep learning.

Frontiers in genetics, 10, 2019.

- Pérez et al. [2019]

J. Pérez, J. Marinković, and P. Barceló.

On the turing completeness of modern neural network architectures.

arXiv preprint arXiv:1901.03429, 2019.

- Qiu et al. [2019]

J. Qiu, H. Ma, O. Levy, S. W.-t. Yih, S. Wang, and J. Tang.

Blockwise self-attention for long document understanding.

arXiv preprint arXiv:1911.02972, 2019.

- Rae et al. [2019]

J. W. Rae, A. Potapenko, S. M. Jayakumar, and T. P. Lillicrap.

Compressive transformers for long-range sequence modelling.

arXiv preprint arXiv:1911.05507, 2019.

- Raffel et al. [2019]

C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou,
W. Li, and P. J. Liu.

Exploring the limits of transfer learning with a unified text-to-text
transformer.

arXiv preprint arXiv:1910.10683, 2019.

- Rothe et al. [2019]

S. Rothe, S. Narayan, and A. Severyn.

Leveraging pre-trained checkpoints for sequence generation tasks.

arXiv preprint arXiv:1907.12461, 2019.

- See et al. [2017]

A. See, P. J. Liu, and C. D. Manning.

Get to the point: Summarization with pointer-generator networks.

arXiv preprint arXiv:1704.04368, 2017.

- Sharma et al. [2019]

E. Sharma, C. Li, and L. Wang.

Bigpatent: A large-scale dataset for abstractive and coherent
summarization.

arXiv preprint arXiv:1906.03741, 2019.

- Shaw et al. [2018]

P. Shaw, J. Uszkoreit, and A. Vaswani.

Self-attention with relative position representations.

arXiv preprint arXiv:1803.02155, 2018.

- Spielman and Teng [2011]

D. A. Spielman and S.-H. Teng.

Spectral sparsification of graphs.

SIAM Journal on Computing, 40(4):981–1025, 2011.

- Subramanian et al. [2019]

S. Subramanian, R. Li, J. Pilault, and C. Pal.

On extractive and abstractive neural document summarization with
transformer language models.

arXiv preprint arXiv:1909.03186, 2019.

- Sukhbaatar et al. [2019]

S. Sukhbaatar, E. Grave, P. Bojanowski, and A. Joulin.

Adaptive attention span in transformers.

arXiv preprint arXiv:1905.07799, 2019.

- Sun et al. [2019]

C. Sun, L. Huang, and X. Qiu.

Utilizing bert for aspect-based sentiment analysis via constructing
auxiliary sentence.

arXiv preprint arXiv:1903.09588, 2019.

- Sussman [2017 (accessed June 3, 2020]

D. Sussman.

Lecture Notes for Boston University MA 882 Spring 2017, 2017
(accessed June 3, 2020).

URL
http://math.bu.edu/people/sussman/MA882_2017/2017-01-26-Lecture-2.html.

- Sutskever et al. [2014]

I. Sutskever, O. Vinyals, and Q. V. Le.

Sequence to sequence learning with neural networks.

In Advances in neural information processing systems, pages
3104–3112, 2014.

- Tampuu et al. [2019]

A. Tampuu, Z. Bzhalava, J. Dillner, and R. Vicente.

Viraminer: Deep learning on raw dna sequences for identifying viral
genomes in human samples.

PloS one, 14(9), 2019.

- Tang et al. [2020]

Z. Tang, Y. Shen, X. Ma, W. Xu, J. Yu, and W. Lu.

Multi-hop reading comprehension across documents with path-based
graph convolutional network.

arXiv:2006.06478, 2020.

- Thongtan and Phienthrakul [2019]

T. Thongtan and T. Phienthrakul.

Sentiment classification using document embeddings trained with
cosine similarity.

In Proceedings of the 57th Annual Meeting of the Association
for Computational Linguistics: Student Research Workshop, pages 407–414,
2019.

- Trinh and Le [2018]

T. H. Trinh and Q. V. Le.

A simple method for commonsense reasoning.

arXiv preprint arXiv:1806.02847, 2018.

- Umarov and Solovyev [2017]

R. K. Umarov and V. V. Solovyev.

Recognition of prokaryotic and eukaryotic promoters using
convolutional deep learning neural networks.

PloS one, 12(2), 2017.

- Vaswani et al. [2017]

A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
Ł. Kaiser, and I. Polosukhin.

Attention is all you need.

In Advances in neural information processing systems, pages
5998–6008, 2017.

- Wang et al. [2018]

A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman.

Glue: A multi-task benchmark and analysis platform for natural
language understanding.

arXiv preprint arXiv:1804.07461, 2018.

- Wang et al. [2019]

Z. Wang, P. Ng, X. Ma, R. Nallapati, and B. Xiang.

Multi-passage bert: A globally normalized bert model for open-domain
question answering.

arXiv preprint arXiv:1908.08167, 2019.

- Watts and Strogatz [1998]

D. J. Watts and S. H. Strogatz.

Collective dynamics of ‘small-world’networks.

nature, 393(6684):440–442, 1998.

- Welbl et al. [2018]

J. Welbl, P. Stenetorp, and S. Riedel.

Constructing datasets for multi-hop reading comprehension across
documents.

Transactions of the Association for Computational Linguistics,
6:287–302, 2018.

- Williams [2005]

R. Williams.

A new algorithm for optimal 2-constraint satisfaction and its
implications.

Theoretical Computer Science, 348(2-3):357–365, 2005.

- Wiseman et al. [2017]

S. Wiseman, S. M. Shieber, and A. M. Rush.

Challenges in data-to-document generation.

arXiv preprint arXiv:1707.08052, 2017.

- Xiao et al. [2019]

X. Xiao, Z.-C. Xu, W.-R. Qiu, P. Wang, H.-T. Ge, and K.-C. Chou.

ipsw (2l)-pseknc: A two-layer predictor for identifying promoters and
their strength by hybrid features via pseudo k-tuple nucleotide composition.

Genomics, 111(6):1785–1793, 2019.

- Yang et al. [2017]

Y. Yang, R. Zhang, S. Singh, and J. Ma.

Exploiting sequence-based features for predicting enhancer–promoter
interactions.

Bioinformatics, 33(14):i252–i260, 2017.

- Yang et al. [2018]

Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. W. Cohen, R. Salakhutdinov, and C. D.
Manning.

Hotpotqa: A dataset for diverse, explainable multi-hop question
answering.

arXiv preprint arXiv:1809.09600, 2018.

- Yang et al. [2019]

Z. Yang, Z. Dai, Y. Yang, J. Carbonell, R. R. Salakhutdinov, and Q. V. Le.

Xlnet: Generalized autoregressive pretraining for language
understanding.

In Advances in neural information processing systems, pages
5754–5764, 2019.

- Yao et al. [2019]

Z. Yao, S. Cao, W. Xiao, C. Zhang, and L. Nie.

Balanced sparsity for efficient dnn inference on gpu.

In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 33, pages 5676–5683, 2019.

- Ye et al. [2019]

Z. Ye, Q. Guo, Q. Gan, X. Qiu, and Z. Zhang.

Bp-transformer: Modelling long-range context via binary partitioning.

arXiv preprint arXiv:1911.04070, 2019.

- Yun et al. [2019]

C. Yun, S. Bhojanapalli, A. S. Rawat, S. J. Reddi, and S. Kumar.

Are transformers universal approximators of sequence-to-sequence
functions?

arXiv preprint arXiv:1912.10077, 2019.

- Yun et al. [2020]

C. Yun, Y.-W. Chang, S. Bhojanapalli, A. S. Rawat, S. J. Reddi, and S. Kumar.

o​(n)𝑜𝑛o(n) connections are expressive enough: Universal approximability
of sparse transformers.

In Advances in Neural Information Processing Systems, 2020.

- Zhang et al. [2019a]

H. Zhang, C.-L. Hung, M. Liu, X. Hu, and Y.-Y. Lin.

Ncnet: Deep learning network models for predicting function of
non-coding dna.

Frontiers in genetics, 10, 2019a.

- Zhang et al. [2019b]

J. Zhang, Y. Zhao, M. Saleh, and P. J. Liu.

Pegasus: Pre-training with extracted gap-sentences for abstractive
summarization.

arXiv preprint arXiv:1912.08777, 2019b.

- Zhang et al. [2015]

X. Zhang, J. Zhao, and Y. LeCun.

Character-level convolutional networks for text classification.

In Advances in neural information processing systems, pages
649–657, 2015.

- Zhou and Troyanskaya [2015]

J. Zhou and O. G. Troyanskaya.

Predicting effects of noncoding variants with deep learning–based
sequence model.

Nature methods, 12(10):931–934, 2015.

- Zhu et al. [2015]

Y. Zhu, R. Kiros, R. Zemel, R. Salakhutdinov, R. Urtasun, A. Torralba, and
S. Fidler.

Aligning books and movies: Towards story-like visual explanations by
watching movies and reading books.

In IEEE international conference on computer vision, pages
19–27, 2015.

Big Bird: Transformers for Longer Sequences – Appendix

## Appendix A Universal Approximators

### A.1 Notation

We begin by setting up some notations following Perez19 to formally describe the complete architecture of Transformers.
A single layer of Transformer encoder is a parametric function EncEnc\operatorname{Enc} receiving a sequence 𝑿=(𝒙1,…,𝒙n)𝑿subscript𝒙1…subscript𝒙𝑛{\bm{X}}=({\bm{x}}_{1},...,{\bm{x}}_{n}) of vectors in ℝdsuperscriptℝ𝑑\mathbb{R}^{d} and returning a sequence 𝒁=(𝒛1,…,𝒛n)𝒁subscript𝒛1…subscript𝒛𝑛{\bm{Z}}=({\bm{z}}_{1},...,{\bm{z}}_{n}) of the same length.
Each 𝒛isubscript𝒛𝑖{\bm{z}}_{i} is a d𝑑d dimensional vector as well.
We interchangeably treat the sequence 𝑿𝑿{\bm{X}} as a matrix in ℝn×dsuperscriptℝ𝑛𝑑\mathbb{R}^{n\times d}.
EncEnc\operatorname{Enc} has two components:

- 1.

An attention mechanism Attn that takes in the sequence 𝑿𝑿{\bm{X}} and returns sequence (𝒂1,…,𝒂n)subscript𝒂1…subscript𝒂𝑛({\bm{a}}_{1},...,{\bm{a}}_{n}) of the same length and dimensionality; and

- 2.

A two layer fully connected network O𝑂O that takes in a vector in ℝdsuperscriptℝ𝑑\mathbb{R}^{d} and returns a vector in ℝdsuperscriptℝ𝑑\mathbb{R}^{d}.

Then i𝑖i-th output vector of Enc⁡(𝑿)Enc𝑿\operatorname{Enc}({\bm{X}}) is computed as follows:

𝒛i=O​(𝒂i)+𝒂iwhere𝒂i=Attn​(𝑿)i+𝒙iformulae-sequencesubscript𝒛𝑖𝑂subscript𝒂𝑖subscript𝒂𝑖wheresubscript𝒂𝑖Attnsubscript𝑿𝑖subscript𝒙𝑖\displaystyle{\bm{z}}_{i}=O({\bm{a}}_{i})+{\bm{a}}_{i}\qquad\text{where}\qquad{\bm{a}}_{i}=\textsc{Attn}({\bm{X}})_{i}+{\bm{x}}_{i}

(1)

Now it remains to define Attn and O𝑂O which we do next.

As described in Sec. 2, an attention mechanism is parameterized by three functions: Q,K,V:ℝd→ℝm:𝑄𝐾𝑉→superscriptℝ𝑑superscriptℝ𝑚Q,K,V:\mathbb{R}^{d}\to\mathbb{R}^{m}.
In this paper, we assume that they are simply matrix products: Q​(𝒙)=𝒙​WQ𝑄𝒙𝒙subscript𝑊𝑄Q({\bm{x}})={\bm{x}}W_{Q},
K​(𝒙)=𝒙​WK𝐾𝒙𝒙subscript𝑊𝐾K({\bm{x}})={\bm{x}}W_{K}, and V​(𝒙)=𝒙​WV𝑉𝒙𝒙subscript𝑊𝑉V({\bm{x}})={\bm{x}}W_{V}, where WQ,WK,WV∈ℝd×msubscript𝑊𝑄subscript𝑊𝐾subscript𝑊𝑉superscriptℝ𝑑𝑚W_{Q},W_{K},W_{V}\in\mathbb{R}^{d\times m} and
WV∈ℝd×dsubscript𝑊𝑉superscriptℝ𝑑𝑑W_{V}\in\mathbb{R}^{d\times d}.
In reality a multi-headed attention is used, i.e. we have not only one,
but H𝐻H-sets of Query/Key/Value weight matrices, WQh,WVh,WKh​ for ​h=1,…,Hformulae-sequencesuperscriptsubscript𝑊𝑄ℎsuperscriptsubscript𝑊𝑉ℎsuperscriptsubscript𝑊𝐾ℎ for ℎ1…𝐻W_{Q}^{h},W_{V}^{h},W_{K}^{h}\text{ for }h=1,...,H.
Thus, for a directed graph D𝐷D over [n]delimited-[]𝑛[n], the ithsuperscript𝑖thi^{\text{th}} output vector of the generalized attention mechanism would be

AttnD​(𝑿)isubscriptAttn𝐷subscript𝑿𝑖\displaystyle\textsc{Attn}_{D}({\bm{X}})_{i}
=∑h=1Hσ​((𝒙i​WQh)​(𝑿N​(i)​WKh)T)⋅(𝑿N​(i)​WVh)absentsuperscriptsubscriptℎ1𝐻⋅𝜎subscript𝒙𝑖superscriptsubscript𝑊𝑄ℎsuperscriptsubscript𝑿𝑁𝑖superscriptsubscript𝑊𝐾ℎ𝑇subscript𝑿𝑁𝑖superscriptsubscript𝑊𝑉ℎ\displaystyle=\sum_{h=1}^{H}\sigma\left(({\bm{x}}_{i}W_{Q}^{h})({\bm{X}}_{N(i)}W_{K}^{h})^{T}\right)\cdot({\bm{X}}_{N(i)}W_{V}^{h})

(AT)

where N​(i)𝑁𝑖N(i) denote the out-neighbors set of node i𝑖i in D𝐷D.
In other words, the set of arcs (directed edges) in D𝐷D represents the set of inner products that our attention mechanism will consider.
Also recall that σ𝜎\sigma is a scoring function such as softmax or hardmax.

Lastly, we define the output fully connected network as follows:

O​(𝒂i)𝑂subscript𝒂𝑖\displaystyle O({\bm{a}}_{i})
=ReLU(𝒂iW1+b1)W2⋅+b2\displaystyle=\operatorname{ReLU}\left({\bm{a}}_{i}W_{1}+b_{1}\right)W_{2}\cdot+b_{2}

(FF)

Here W1∈ℝd×qsubscript𝑊1superscriptℝ𝑑𝑞W_{1}\in\mathbb{R}^{d\times q}, W2∈ℝq×dsubscript𝑊2superscriptℝ𝑞𝑑W_{2}\in\mathbb{R}^{q\times d}, b1∈ℝpsubscript𝑏1superscriptℝ𝑝b_{1}\in\mathbb{R}^{p}, and b2∈ℝdsubscript𝑏2superscriptℝ𝑑b_{2}\in\mathbb{R}^{d} are parameters of
output network O𝑂O.

Additional Notation We introduce a few pieces of additional notation that will be useful.
Let [a,b)δ={a,a+δ,…,a+⌊b−aδ⌋⋅δ}subscript𝑎𝑏𝛿𝑎𝑎𝛿…𝑎⋅𝑏𝑎𝛿𝛿[a,b)_{\delta}=\{a,a+\delta,\dots,a+\lfloor\frac{b-a}{\delta}\rfloor\cdot\delta\}. Therefore,
[0,1)δ={0,δ,2​δ,…,(1−δ)}subscript01𝛿0𝛿2𝛿…1𝛿[0,1)_{\delta}=\{0,\delta,2\delta,\dots,(1-\delta)\}.
We use 𝟏​[ℰ]1delimited-[]ℰ\mathbf{1}[\mathcal{E}] to denote the indicator variable; it is 111 if the event ℰℰ\mathcal{E} occurs and 00 otherwise.

### A.2 Proof

In this section, we will present the full proof of theorem 1.
The proof will contain three parts.
The first and the third part will largely follow standard techniques. The main innovation lies is in the second part.

#### A.2.1 Approximate ℱC​Dsubscriptℱ𝐶𝐷\mathcal{F}_{CD} by piece-wise constant functions

First, we consider a suitable partition of the region (0,1)01(0,1) into a
grid of granularity δ𝛿\delta, which we denote by Gδsubscript𝐺𝛿G_{\delta}. We do this using Lemma 8 from Yun19, which we restate for completeness:

###### Lemma 1 (Lemma 8 [Yun19]).

For any given f∈ℱC​D𝑓subscriptℱ𝐶𝐷f\in\mathcal{F}_{CD} and 1≤p≤∞1𝑝1\leq p\leq\infty, there exists a δ>0𝛿0\delta>0 such that
there exists a piece-wise constant function f¯¯𝑓\bar{f} with dp​(f,f¯)≤ϵ3subscript𝑑𝑝𝑓¯𝑓italic-ϵ3d_{p}(f,\bar{f})\leq\frac{\epsilon}{3}.
Concretely, f¯¯𝑓\bar{f} is defined as

f¯​(X)=∑P∈𝔾δf​(P)⋅𝟏​[‖ReLU⁡(X−P)‖∞≤δ]¯𝑓𝑋subscript𝑃subscript𝔾𝛿⋅𝑓𝑃1delimited-[]subscriptnormReLU𝑋𝑃𝛿\bar{f}(X)=\sum_{P\in\mathbb{G}_{\delta}}f(P)\cdot\mathbf{1}\left[\|\operatorname{ReLU}(X-P)\|_{\infty}\leq\delta\right]

Since transformers can learn a positional embedding E𝐸E, without any loss of generality,
we can consider the translated function. In particular, define

E=[000…0δ−dδ−dδ−d…δ−dδ−2​dδ−2​dδ−2​d…δ−2​d⋮δ−(n−1)​dδ−(n−1)​dδ−(n−1)​d…δ−(n−1)​d]𝐸matrix000…0superscript𝛿𝑑superscript𝛿𝑑superscript𝛿𝑑…superscript𝛿𝑑superscript𝛿2𝑑superscript𝛿2𝑑superscript𝛿2𝑑…superscript𝛿2𝑑⋮superscript𝛿𝑛1𝑑superscript𝛿𝑛1𝑑superscript𝛿𝑛1𝑑…superscript𝛿𝑛1𝑑E=\begin{bmatrix}0&0&0&\dots&0\\
\delta^{-d}&\delta^{-d}&\delta^{-d}&\dots&\delta^{-d}\\
\delta^{-2d}&\delta^{-2d}&\delta^{-2d}&\dots&\delta^{-2d}\\
\vdots\\
\delta^{-(n-1)d}&\delta^{-(n-1)d}&\delta^{-(n-1)d}&\dots&\delta^{-(n-1)d}\\
\end{bmatrix}

We will try to approximate g​(X)=f​(X−E)𝑔𝑋𝑓𝑋𝐸g(X)=f(X-E) where g𝑔g is defined on the domain
[0,1]d×[δ−d,δ−d+1]d×⋯×[δ−(n−1)​d,δ−(n−1)​d+1]dsuperscript01𝑑superscriptsuperscript𝛿𝑑superscript𝛿𝑑1𝑑⋯superscriptsuperscript𝛿𝑛1𝑑superscript𝛿𝑛1𝑑1𝑑[0,1]^{d}\times[\delta^{-d},\delta^{-d}+1]^{d}\times\dots\times[\delta^{-(n-1)d},\delta^{-(n-1)d}+1]^{d}. To do so, we will apply a suitable modification of Lemma 1,
which will consider the discretized grid

𝐆δE:=[0,1]δd×[δ−d,δ−d+1]δd×⋯×[δ−(n−1)​d,δ−(n−1)​d+1]δd.assignsubscriptsuperscript𝐆𝐸𝛿superscriptsubscript01𝛿𝑑superscriptsubscriptsuperscript𝛿𝑑superscript𝛿𝑑1𝛿𝑑⋯superscriptsubscriptsuperscript𝛿𝑛1𝑑superscript𝛿𝑛1𝑑1𝛿𝑑\mathbf{G}^{E}_{\delta}:=[0,1]_{\delta}^{d}\times[\delta^{-d},\delta^{-d}+1]_{\delta}^{d}\times\dots\times[\delta^{-(n-1)d},\delta^{-(n-1)d}+1]_{\delta}^{d}.

Therefore, it suffices to approximate a function f¯:𝐆δE→ℝn×d:¯𝑓→subscriptsuperscript𝐆𝐸𝛿superscriptℝ𝑛𝑑\bar{f}:\mathbf{G}^{E}_{\delta}\to\mathbb{R}^{n\times d}
defined as

f¯​(X)=∑P∈𝐆δEf​(P−E)⋅𝟏​[‖ReLU⁡(X−P)‖∞≤δ].¯𝑓𝑋subscript𝑃subscriptsuperscript𝐆𝐸𝛿⋅𝑓𝑃𝐸1delimited-[]subscriptnormReLU𝑋𝑃𝛿\bar{f}(X)=\sum_{P\in\mathbf{G}^{E}_{\delta}}f(P-E)\cdot\mathbf{1}\left[\|\operatorname{ReLU}(X-P)\|_{\infty}\leq\delta\right].

#### A.2.2 Contextual Mappings and Sparse Attention Mechanisms

Throughout this section, we will assume that we are given a function that has an extra global token
at index 00 and all vectors have an extra dimension appended to them. The latter assumption is
without loss of generality as we can use the Feed-Forward Network to append sparse dimensions.
In particular, we will associate X∈ℝ(n+1)×(d+1)𝑋superscriptℝ𝑛1𝑑1X\in\mathbb{R}^{(n+1)\times(d+1)} where we write X=(x0,x1,…,xn)𝑋subscript𝑥0subscript𝑥1…subscript𝑥𝑛X=(x_{0},x_{1},\dots,x_{n}).
Although our function is only defined for 𝐆δE⊂ℝn×dsubscriptsuperscript𝐆𝐸𝛿superscriptℝ𝑛𝑑\mathbf{G}^{E}_{\delta}\subset\mathbb{R}^{n\times d},
we can amend the function in a natural way by making it ignore the first column.
To avoid excessive clutter, we will assume that the function value is evaluated on the last n𝑛n columns.

The main idea in this section is the use of contextual mapping to enable Transformers
to compute any discretized function. A contextual mapping is an unique encoding of each
tuple (X,xi)𝑋subscript𝑥𝑖(X,x_{i}) where X∈𝐆δE𝑋subscriptsuperscript𝐆𝐸𝛿X\in\mathbf{G}^{E}_{\delta}, and each column xi∈[δ−(i−1)​d,δ−(i−1)​d+1)δdsubscript𝑥𝑖subscriptsuperscriptsuperscript𝛿𝑖1𝑑superscript𝛿𝑖1𝑑1𝑑𝛿x_{i}\in[\delta^{-(i-1)d},\delta^{-(i-1)d}+1)^{d}_{\delta} for all i∈[n]𝑖delimited-[]𝑛i\in[n].
We restate the definition adapted to our setting below

###### Definition 2 (Defn 3.1 [Yun19]).

(Contextual Mapping)

A contextual mapping is a function mapping q:𝐆δE→ℝn:𝑞→subscriptsuperscript𝐆𝐸𝛿superscriptℝ𝑛q:\mathbf{G}^{E}_{\delta}\to\mathbb{R}^{n} if it satisfies the following:

- 1.

For any P∈𝐆δE𝑃subscriptsuperscript𝐆𝐸𝛿P\in\mathbf{G}^{E}_{\delta}, q​(P)𝑞𝑃q(P) contains distinct entries.

- 2.

For any two P,P′∈𝐆δE𝑃superscript𝑃′subscriptsuperscript𝐆𝐸𝛿P,P^{\prime}\in\mathbf{G}^{E}_{\delta} with P≠P′𝑃superscript𝑃′P\neq P^{\prime}, all entries of q​(P)𝑞𝑃q(P) and q​(P′)𝑞superscript𝑃′q(P^{\prime})
are distinct.

The key technical novelty of the proof is computing a contextual mapping using only the
sparse attention mechanism. We create a
“selective shift” operator which only shifts entries of a vector that
lie in a certain range. We will use this shift operator strategically to ensure that
we attain a contextual mapping at the end of the process.
The lemma below, which is based on parts of the proof of Lemma 6 of [Yun19],
states that we can implement a suitable “selective” shift operator using a
sparse attention mechanism.

###### Lemma 2.

Given a function ψ:ℝ(n+1)×(d+1)×ℝ2→ℝ(n+1)×1:𝜓→superscriptℝ𝑛1𝑑1superscriptℝ2superscriptℝ𝑛11\psi:\mathbb{R}^{(n+1)\times(d+1)}\times\mathbb{R}^{2}\to\mathbb{R}^{(n+1)\times 1} and a vector u∈ℝd+1𝑢superscriptℝ𝑑1u\in\mathbb{R}^{d+1} and a sparse attention mechanism
based on the directed graph D𝐷D, we can implement a selective shift operator that receives as input
a matrix X∈ℝ(n+1)×(d+1)𝑋superscriptℝ𝑛1𝑑1X\in\mathbb{R}^{(n+1)\times(d+1)} and outputs X+ρ⋅ψu​(X,b1,b2)𝑋⋅𝜌subscript𝜓𝑢𝑋subscript𝑏1subscript𝑏2X+\rho\cdot\psi_{u}(X,b_{1},b_{2}) where

ψu​(Z;b1,b2)i={(maxj∈N​(i)⁡uT​Zj−minj∈N​(i)⁡uT​Zj)​e1 if ​b1≤uT​Zj≤b20 else. subscript𝜓𝑢subscript𝑍subscript𝑏1subscript𝑏2𝑖casessubscript𝑗𝑁𝑖superscript𝑢𝑇subscript𝑍𝑗subscript𝑗𝑁𝑖superscript𝑢𝑇subscript𝑍𝑗subscript𝑒1 if subscript𝑏1superscript𝑢𝑇subscript𝑍𝑗subscript𝑏20 else. \psi_{u}(Z;b_{1},b_{2})_{i}=\begin{cases}(\max_{j\in N(i)}u^{T}Z_{j}-\min_{j\in N(i)}u^{T}Z_{j})e_{1}&\text{ if }b_{1}\leq u^{T}Z_{j}\leq b_{2}\\
0&\text{ else. }\end{cases}

Note that e1∈Rd+1subscript𝑒1superscript𝑅𝑑1e_{1}\in R^{d+1} denotes (1,0,…,0)10…0(1,0,\dots,0).

###### Proof.

Consider the function , which can be implemented by a sparse attention mechanism :

ψ~​(X,b)i=σH​[(uT⋅Xi)T⋅(uT​XN​(i)−b​1N​(i)T)​e(1)​(uT​XN​(i))]~𝜓subscript𝑋𝑏𝑖subscript𝜎𝐻delimited-[]⋅superscript⋅superscript𝑢𝑇subscript𝑋𝑖𝑇superscript𝑢𝑇subscript𝑋𝑁𝑖𝑏superscriptsubscript1𝑁𝑖𝑇superscript𝑒1superscript𝑢𝑇subscript𝑋𝑁𝑖\tilde{\psi}(X,b)_{i}=\sigma_{H}\Big{[}(u^{T}\cdot X_{i})^{T}\cdot(u^{T}X_{N(i)}-b1_{N(i)}^{T})e^{(1)}(u^{T}X_{N(i)})\Big{]}

This is because the Key, Query and Value functions are simply affine transformations of X𝑋X.

Given any graph D𝐷D, the above function will evaluate to the following:

ψ~​(Z;b)i={(maxj∈N​(i)⁡uT​Zj)​e1 if ​uT​Zj>b(minj∈N​(i)⁡uT​Zj)​e1 if ​uT​Zj<b~𝜓subscript𝑍𝑏𝑖casessubscript𝑗𝑁𝑖superscript𝑢𝑇subscript𝑍𝑗subscript𝑒1 if superscript𝑢𝑇subscript𝑍𝑗𝑏subscript𝑗𝑁𝑖superscript𝑢𝑇subscript𝑍𝑗subscript𝑒1 if superscript𝑢𝑇subscript𝑍𝑗𝑏\tilde{\psi}(Z;b)_{i}=\begin{cases}(\max_{j\in N(i)}u^{T}Z_{j})e_{1}&\text{ if }u^{T}Z_{j}>b\\
(\min_{j\in N(i)}u^{T}Z_{j})e_{1}&\text{ if }u^{T}Z_{j}<b\\
\end{cases}

Therefore we can say that ψ~​(Z;bQ)−ψ~​(Z;bQ′)~𝜓𝑍subscript𝑏𝑄~𝜓𝑍subscript𝑏superscript𝑄′\tilde{\psi}(Z;b_{Q})-\tilde{\psi}(Z;b_{Q^{\prime}}) satisfies

ψ​(Z;b1,b2)i={(maxj∈N​(i)⁡uT​Zj−minj∈N​(i)⁡uT​Zj)​e1 if ​b1≤uT​Zj≤b20 else 𝜓subscript𝑍subscript𝑏1subscript𝑏2𝑖casessubscript𝑗𝑁𝑖superscript𝑢𝑇subscript𝑍𝑗subscript𝑗𝑁𝑖superscript𝑢𝑇subscript𝑍𝑗subscript𝑒1 if subscript𝑏1superscript𝑢𝑇subscript𝑍𝑗subscript𝑏20 else \psi(Z;b_{1},b_{2})_{i}=\begin{cases}(\max_{j\in N(i)}u^{T}Z_{j}-\min_{j\in N(i)}u^{T}Z_{j})e_{1}&\text{ if }b_{1}\leq u^{T}Z_{j}\leq b_{2}\\
0&\text{ else }\end{cases}

∎

The following lemma, which is the heart of the proof, uses the above selective shift operators
to construct contextual mappings.

###### Lemma 3.

There exists a function gc:ℝ(n+1)×(d+1)→ℝ(n+1):subscript𝑔𝑐→superscriptℝ𝑛1𝑑1superscriptℝ𝑛1g_{c}:\mathbb{R}^{(n+1)\times(d+1)}\to\mathbb{R}^{(n+1)} and
a unique vector u𝑢u, such that for all P∈𝐆δE𝑃subscriptsuperscript𝐆𝐸𝛿P\in\mathbf{G}^{E}_{\delta} gc​(P):=⟨u,g​(P)⟩assignsubscript𝑔𝑐𝑃𝑢𝑔𝑃g_{c}(P):=\left\langle u,g(P)\right\rangle
satisfies the property that gcsubscript𝑔𝑐g_{c} is a contextual mapping of P𝑃P.
Furthermore, gc∈𝒯D2,1,1subscript𝑔𝑐superscriptsubscript𝒯𝐷211g_{c}\in\mathcal{T}_{D}^{2,1,1} using a composition of sparse attention layers
as long as D𝐷D contains the star graph.

###### Proof.

Define u∈ℝd+1=[1,δ−1,δ−2,…,δ−d+1,δ−n​d]𝑢superscriptℝ𝑑11superscript𝛿1superscript𝛿2…superscript𝛿𝑑1superscript𝛿𝑛𝑑u\in\mathbb{R}^{d+1}=[1,\delta^{-1},\delta^{-2},\dots,\delta^{-d+1},\delta^{-nd}] and let
X0=(0,…,0,1)subscript𝑋00…01X_{0}=(0,\dots,0,1). We will assume that ⟨xi,x0⟩=0subscript𝑥𝑖subscript𝑥00\left\langle x_{i},x_{0}\right\rangle=0, by assuming that all the
columns x1,…,xnsubscript𝑥1…subscript𝑥𝑛x_{1},\dots,x_{n} are appended by 00.

To successfully encode the entire context in each token, we will interleave the shift operator
to target the original columns 1,…,n1…𝑛1,\dots,n and to target the global column 00. After a column i𝑖i is targeted, its inner product with u𝑢u will encode the entire context of the first i𝑖i columns.
Next, we will shift the global token to take this context into account. This can be subsequently used by the remaining columns.

For i∈{0,1,…,n}𝑖01…𝑛i\in\{0,1,\dots,n\}, we will use lisubscript𝑙𝑖l_{i} to denote the innerproducts ⟨u,xi⟩𝑢subscript𝑥𝑖\left\langle u,x_{i}\right\rangle at the beginning. For
fi=⟨u,xi⟩subscript𝑓𝑖𝑢subscript𝑥𝑖f_{i}=\left\langle u,x_{i}\right\rangle after the it​hsuperscript𝑖𝑡ℎi^{th} column has changed for i∈{1,…,n}𝑖1…𝑛i\in\{1,\dots,n\} and we will use
f0ksuperscriptsubscript𝑓0𝑘f_{0}^{k} to denote ⟨u,x0⟩𝑢subscript𝑥0\left\langle u,x_{0}\right\rangle after the kt​hsuperscript𝑘𝑡ℎk^{th} phase. We need to distinguish the global token
further as it’s inner product will change in each phase.
Initially, given X∈𝐆δE𝑋subscriptsuperscript𝐆𝐸𝛿X\in\mathbf{G}^{E}_{\delta}, the following are true:

δ−(i−1)​dsuperscript𝛿𝑖1𝑑\displaystyle\delta^{-(i-1)d}
≤⟨u,Xi⟩≤δ−i​d−δ for all ​i∈[n]formulae-sequenceabsent𝑢subscript𝑋𝑖superscript𝛿𝑖𝑑𝛿 for all 𝑖delimited-[]𝑛\displaystyle\leq\left\langle u,X_{i}\right\rangle\leq\delta^{-id}-\delta\qquad\text{ for all }i\in[n]

δ−(n+1)​dsuperscript𝛿𝑛1𝑑\displaystyle\delta^{-(n+1)d}
=⟨u,X0⟩absent𝑢subscript𝑋0\displaystyle=\left\langle u,X_{0}\right\rangle

Note that all lisubscript𝑙𝑖l_{i} ordered in distinct buckets l1<l2<⋯<ln<l0subscript𝑙1subscript𝑙2⋯subscript𝑙𝑛subscript𝑙0l_{1}<l_{2}<\dots<l_{n}<l_{0}.

We do this in phases indexed from i∈{1,…,n}𝑖1…𝑛i\in\{1,\dots,n\}. Each phase consists of two distinct parts:

 The low shift operation: These operation will be of the form

X←X+δ−d​ψ​(X,v−δ/2,v+δ/2)←𝑋𝑋superscript𝛿𝑑𝜓𝑋𝑣𝛿2𝑣𝛿2X\leftarrow X+\delta^{-d}\psi\left(X,v-\delta/2,v+\delta/2\right)

for values
v∈[δ−i​d),δ−(i+1)​d)δv\in[\delta^{-id}),\delta^{-(i+1)d})_{\delta}.
The range is chosen so that only lisubscript𝑙𝑖l_{i} will be in the range and no other ljsubscript𝑙𝑗l_{j} j≠i𝑗𝑖j\neq i
is in the range.
This will shift exactly the it​hsuperscript𝑖𝑡ℎi^{th} column xisubscript𝑥𝑖x_{i} so that the new inner product
fi=⟨u,xi⟩subscript𝑓𝑖𝑢subscript𝑥𝑖f_{i}=\left\langle u,x_{i}\right\rangle is substantially larger than lisubscript𝑙𝑖l_{i}. Furthermore, no
other column of X𝑋X will be affected.

 The high shift operation: 
These operation will be of the form

X←X+δ−n​d⋅ψ​(X,v−δ/2,v+δ/2)←𝑋𝑋⋅superscript𝛿𝑛𝑑𝜓𝑋𝑣𝛿2𝑣𝛿2X\leftarrow X+\delta^{-nd}\cdot\psi\left(X,v-\delta/2,v+\delta/2\right)

for values v∈[Si,Ti)δ𝑣subscriptsubscript𝑆𝑖subscript𝑇𝑖𝛿v\in[S_{i},T_{i})_{\delta}. The range [Si,Ti)δsubscriptsubscript𝑆𝑖subscript𝑇𝑖𝛿[S_{i},T_{i})_{\delta} is chosen to
only affect the column x0subscript𝑥0x_{0} (corresponding to the global token) and no other column. In particular, this
will shift the global token by a further δ−n​dsuperscript𝛿𝑛𝑑\delta^{-nd}. Let f~0isuperscriptsubscript~𝑓0𝑖\tilde{f}_{0}^{i} denote the value of
f~0i=⟨u,x0⟩subscriptsuperscript~𝑓𝑖0𝑢subscript𝑥0\tilde{f}^{i}_{0}=\left\langle u,x_{0}\right\rangle at the end of it​hsuperscript𝑖𝑡ℎi^{th} high operation.

Each phase interleaves a shift operation to column i𝑖i and updates the global token.
After each phase, the updated it​hsuperscript𝑖𝑡ℎi^{th} column fi=⟨u,xi⟩subscript𝑓𝑖𝑢subscript𝑥𝑖f_{i}=\left\langle u,x_{i}\right\rangle will contain a unique token
encoding the values of all the l1,…,lisubscript𝑙1…subscript𝑙𝑖l_{1},\dots,l_{i}. After the high update, f~0i=⟨u,x0⟩superscriptsubscript~𝑓0𝑖𝑢subscript𝑥0\tilde{f}_{0}^{i}=\left\langle u,x_{0}\right\rangle
will contain information about the first i𝑖i tokens.

Finally, we define the following constants for all k∈{0,1,…,n}𝑘01…𝑛k\in\{0,1,\dots,n\}.

Tksubscript𝑇𝑘\displaystyle T_{k}
=(δ−(n+1)​d+1)k⋅δ−n​d−∑t=2k(δ−(n+1)​d+1)k−t​(2​δ−n​d−d+δ−n​d+1)​δ−t​dabsent⋅superscriptsuperscript𝛿𝑛1𝑑1𝑘superscript𝛿𝑛𝑑superscriptsubscript𝑡2𝑘superscriptsuperscript𝛿𝑛1𝑑1𝑘𝑡2superscript𝛿𝑛𝑑𝑑superscript𝛿𝑛𝑑1superscript𝛿𝑡𝑑\displaystyle=(\delta^{-(n+1)d}+1)^{k}\cdot\delta^{-nd}-\sum_{t=2}^{k}(\delta^{-(n+1)d}+1)^{k-t}(2\delta^{-nd-d}+\delta^{-nd}+1)\delta^{-td}

−(δ−(n+1)​d+1)k−1​(δ−n​d−d+δ−n​d)​δ−d−δ−(k+1)​dsuperscriptsuperscript𝛿𝑛1𝑑1𝑘1superscript𝛿𝑛𝑑𝑑superscript𝛿𝑛𝑑superscript𝛿𝑑superscript𝛿𝑘1𝑑\displaystyle\qquad-(\delta^{-(n+1)d}+1)^{k-1}(\delta^{-nd-d}+\delta^{-nd})\delta^{-d}-\delta^{-(k+1)d}

(UP)

Sksubscript𝑆𝑘\displaystyle S_{k}
=(δ−(n+1)​d+1)k⋅δ−n​d−∑t=2k(δ−(n+1)​d+1)k−t​(2​δ−n​d−d+δ−n​d+1)​δ−(t−1)​dabsent⋅superscriptsuperscript𝛿𝑛1𝑑1𝑘superscript𝛿𝑛𝑑superscriptsubscript𝑡2𝑘superscriptsuperscript𝛿𝑛1𝑑1𝑘𝑡2superscript𝛿𝑛𝑑𝑑superscript𝛿𝑛𝑑1superscript𝛿𝑡1𝑑\displaystyle=(\delta^{-(n+1)d}+1)^{k}\cdot\delta^{-nd}-\sum_{t=2}^{k}(\delta^{-(n+1)d}+1)^{k-t}(2\delta^{-nd-d}+\delta^{-nd}+1)\delta^{-(t-1)d}

−(δ−(n+1)​d+1)k−1​(δ−n​d−d+δ−n​d)−δ−k​dsuperscriptsuperscript𝛿𝑛1𝑑1𝑘1superscript𝛿𝑛𝑑𝑑superscript𝛿𝑛𝑑superscript𝛿𝑘𝑑\displaystyle\qquad-(\delta^{-(n+1)d}+1)^{k-1}(\delta^{-nd-d}+\delta^{-nd})-\delta^{-kd}

(LP)

After each k𝑘k phases, we will maintain the following invariants:

- 1.

Sk<f~0k<Tksubscript𝑆𝑘subscriptsuperscript~𝑓𝑘0subscript𝑇𝑘S_{k}<\tilde{f}^{k}_{0}<T_{k} for all k∈{0,1,…,n}𝑘01…𝑛k\in\{0,1,\dots,n\}.

- 2.

Tk−1≤fk<Sksubscript𝑇𝑘1subscript𝑓𝑘subscript𝑆𝑘T_{k-1}\leq f_{k}<S_{k}

- 3.

The order of the inner products after kt​hsuperscript𝑘𝑡ℎk^{th} phase is

lk+1<lk+2​⋯<ln<f1<f2<⋯<fk<f~0k.subscript𝑙𝑘1subscript𝑙𝑘2⋯subscript𝑙𝑛subscript𝑓1subscript𝑓2⋯subscript𝑓𝑘superscriptsubscript~𝑓0𝑘l_{k+1}<l_{k+2}\dots<l_{n}<f_{1}<f_{2}<\dots<f_{k}<\tilde{f}_{0}^{k}.

##### Base case

The case k=0𝑘0k=0, is trivial as we simply set S0=δ−(n+1)​dsubscript𝑆0superscript𝛿𝑛1𝑑S_{0}=\delta^{-(n+1)d}, T0=δ−(n+1)⋅d+δsubscript𝑇0superscript𝛿⋅𝑛1𝑑𝛿T_{0}=\delta^{-(n+1)\cdot d}+\delta.

The first nontrivial case is k=1𝑘1k=1.

##### Inductive Step

First, in the low shift operation is performed in the range [δ−(k−1)​d,δ−k​d)δsubscriptsuperscript𝛿𝑘1𝑑superscript𝛿𝑘𝑑𝛿[\delta^{-(k-1)d},\delta^{-kd})_{\delta}
Due to the invariant, we know that there exists only one column xksubscript𝑥𝑘x_{k} that is affected by this shift.
In particular, for column k𝑘k, we will have maxj∈N​(k)⁡⟨u,xj⟩=⟨u,x0⟩=f~0k−1subscript𝑗𝑁𝑘𝑢subscript𝑥𝑗𝑢subscript𝑥0subscriptsuperscript~𝑓𝑘10\max_{j\in N(k)}\left\langle u,x_{j}\right\rangle=\left\langle u,x_{0}\right\rangle=\tilde{f}^{k-1}_{0}. The minimum is lksubscript𝑙𝑘l_{k}. Thus the update will be
fk=δ−d​(f~0k−1−lk)+lksubscript𝑓𝑘superscript𝛿𝑑superscriptsubscript~𝑓0𝑘1subscript𝑙𝑘subscript𝑙𝑘f_{k}=\delta^{-d}(\tilde{f}_{0}^{k-1}-l_{k})+l_{k}. Observe that for small enough δ𝛿\delta,
fk≥f~0k−1subscript𝑓𝑘superscriptsubscript~𝑓0𝑘1f_{k}\geq\tilde{f}_{0}^{k-1}. Hence the total ordering, after this operation is

lk+1<lk+2​⋯<ln<f1<f2<⋯<f~0k−1<fksubscript𝑙𝑘1subscript𝑙𝑘2⋯subscript𝑙𝑛subscript𝑓1subscript𝑓2⋯superscriptsubscript~𝑓0𝑘1subscript𝑓𝑘\displaystyle l_{k}+1<l_{k+2}\dots<l_{n}<f_{1}<f_{2}<\dots<\tilde{f}_{0}^{k-1}<f_{k}

(2)

Now when we operate a higher selective shift operator in the range [Sk−1,Tk−1)δsubscriptsubscript𝑆𝑘1subscript𝑇𝑘1𝛿[S_{k-1},T_{k-1})_{\delta}.
Since only global token’s innerproduct f~0k−1superscriptsubscript~𝑓0𝑘1\tilde{f}_{0}^{k-1} is in this range,
it will be the only column affected by the shift operator. The global token operates over the entire range, we know from Eq. 2 that, fk=maxi∈[n]⁡⟨u,xi⟩subscript𝑓𝑘subscript𝑖delimited-[]𝑛𝑢subscript𝑥𝑖f_{k}=\max_{i\in[n]}\left\langle u,x_{i}\right\rangle and lk+1=mini∈[n]⁡⟨u,xi⟩subscript𝑙𝑘1subscript𝑖delimited-[]𝑛𝑢subscript𝑥𝑖l_{k+1}=\min_{i\in[n]}\left\langle u,x_{i}\right\rangle.
The new value f~0k=δ−n​d⋅(fk−lk+1)+f~0k−1superscriptsubscript~𝑓0𝑘⋅superscript𝛿𝑛𝑑subscript𝑓𝑘subscript𝑙𝑘1superscriptsubscript~𝑓0𝑘1\tilde{f}_{0}^{k}=\delta^{-nd}\cdot(f_{k}-l_{k+1})+\tilde{f}_{0}^{k-1}.
Expanding and simplifying we get,

f~0ksuperscriptsubscript~𝑓0𝑘\displaystyle\tilde{f}_{0}^{k}
=δ−n​d⋅(fk−lk+1)+f~0k−1absent⋅superscript𝛿𝑛𝑑subscript𝑓𝑘subscript𝑙𝑘1superscriptsubscript~𝑓0𝑘1\displaystyle=\delta^{-nd}\cdot(f_{k}-l_{k+1})+\tilde{f}_{0}^{k-1}

=δ−n​d⋅(δ−d​(f~0k−1−lk)+lk−lk+1)+f~0k−1absent⋅superscript𝛿𝑛𝑑superscript𝛿𝑑superscriptsubscript~𝑓0𝑘1subscript𝑙𝑘subscript𝑙𝑘subscript𝑙𝑘1superscriptsubscript~𝑓0𝑘1\displaystyle=\delta^{-nd}\cdot(\delta^{-d}(\tilde{f}_{0}^{k-1}-l_{k})+l_{k}-l_{k+1})+\tilde{f}_{0}^{k-1}

=δ−(n+1)​d⋅(f~0k−1−lk)+δ−n​d​(lk−lk+1)+f~0k−1absent⋅superscript𝛿𝑛1𝑑superscriptsubscript~𝑓0𝑘1subscript𝑙𝑘superscript𝛿𝑛𝑑subscript𝑙𝑘subscript𝑙𝑘1superscriptsubscript~𝑓0𝑘1\displaystyle=\delta^{-(n+1)d}\cdot(\tilde{f}_{0}^{k-1}-l_{k})+\delta^{-nd}(l_{k}-l_{k+1})+\tilde{f}_{0}^{k-1}

=(δ−(n+1)​d+1)​f~0k−1−(δ−n​d−d+δ−n​d)​lk−lk+1absentsuperscript𝛿𝑛1𝑑1superscriptsubscript~𝑓0𝑘1superscript𝛿𝑛𝑑𝑑superscript𝛿𝑛𝑑subscript𝑙𝑘subscript𝑙𝑘1\displaystyle=(\delta^{-(n+1)d}+1)\tilde{f}_{0}^{k-1}-(\delta^{-nd-d}+\delta^{-nd})l_{k}-l_{k+1}

Expanding the above recursively, we get

=(δ−(n+1)​d+1)k⋅f~00−∑t=2k(δ−(n+1)​d+1)k−t​(2​δ−n​d−d+δ−n​d+1)​ltabsent⋅superscriptsuperscript𝛿𝑛1𝑑1𝑘superscriptsubscript~𝑓00superscriptsubscript𝑡2𝑘superscriptsuperscript𝛿𝑛1𝑑1𝑘𝑡2superscript𝛿𝑛𝑑𝑑superscript𝛿𝑛𝑑1subscript𝑙𝑡\displaystyle=(\delta^{-(n+1)d}+1)^{k}\cdot\tilde{f}_{0}^{0}-\sum_{t=2}^{k}(\delta^{-(n+1)d}+1)^{k-t}(2\delta^{-nd-d}+\delta^{-nd}+1)l_{t}

−(δ−(n+1)​d+1)k−1​(δ−n​d−d+δ−n​d)​l1−lk+1superscriptsuperscript𝛿𝑛1𝑑1𝑘1superscript𝛿𝑛𝑑𝑑superscript𝛿𝑛𝑑subscript𝑙1subscript𝑙𝑘1\displaystyle\qquad-(\delta^{-(n+1)d}+1)^{k-1}(\delta^{-nd-d}+\delta^{-nd})l_{1}-l_{k+1}

Since we know that f~00=δ−n​dsuperscriptsubscript~𝑓00superscript𝛿𝑛𝑑\tilde{f}_{0}^{0}=\delta^{-nd} and each li<δ−i​dsubscript𝑙𝑖superscript𝛿𝑖𝑑l_{i}<\delta^{-id}, we can substitute this to get Eq. UP
and we can get an lower-bound Eq. LP by using li≥δ−(i−1)​dsubscript𝑙𝑖superscript𝛿𝑖1𝑑l_{i}\geq\delta^{-(i-1)d}.

By construction, we know that Sk≤f~0k<Tksubscript𝑆𝑘superscriptsubscript~𝑓0𝑘subscript𝑇𝑘S_{k}\leq\tilde{f}_{0}^{k}<T_{k}. For sufficiently small δ𝛿\delta,
observe that Sk≤f~0k<Tksubscript𝑆𝑘superscriptsubscript~𝑓0𝑘subscript𝑇𝑘S_{k}\leq\tilde{f}_{0}^{k}<T_{k} all are essentially the dominant term ≈O​(δ−n​(k+1)​d−k​d)absent𝑂superscript𝛿𝑛𝑘1𝑑𝑘𝑑\approx O(\delta^{-n(k+1)d-kd}) and all the lower order terms do not matter. As a result it is
immediate to see that that fk>δ−d​(f~0k−1−lk)>Tk−1subscript𝑓𝑘superscript𝛿𝑑superscriptsubscript~𝑓0𝑘1subscript𝑙𝑘subscript𝑇𝑘1f_{k}>\delta^{-d}(\tilde{f}_{0}^{k-1}-l_{k})>T_{k-1} and hence we
can see that the invariant 2 is also satisfied. Since only column k𝑘k and the global token are affected,
we can see that invariant 3 is also satisfied.

After n𝑛n iterations, f~0nsubscriptsuperscript~𝑓𝑛0\tilde{f}^{n}_{0} contains a unique encoding for any P∈𝐆δE𝑃subscriptsuperscript𝐆𝐸𝛿P\in\mathbf{G}^{E}_{\delta}.
To ensure that all tokens are distinct, we will add an additional layer
X=X+δ−n2​d​ψ​(X,v−δ/2,v+δ/2)𝑋𝑋superscript𝛿superscript𝑛2𝑑𝜓𝑋𝑣𝛿2𝑣𝛿2X=X+\delta^{-n^{2}d}\psi(X,v-\delta/2,v+\delta/2) for all v∈[S1,Tn)δ𝑣subscriptsubscript𝑆1subscript𝑇𝑛𝛿v\in[S_{1},T_{n})_{\delta}.
This ensures that for all P,P′∈𝐆δE𝑃superscript𝑃′subscriptsuperscript𝐆𝐸𝛿P,P^{\prime}\in\mathbf{G}^{E}_{\delta}, each entry of q​(P)𝑞𝑃q(P) and q​(P′)𝑞superscript𝑃′q(P^{\prime}) are distinct.
∎

The previous lemma shows that we can compute a contextual mapping using only sparse transforms.
We now use the following lemma to show that we can use a contextual mapping and feed-forward layers
to accurately map to the desired output of the function f¯¯𝑓\bar{f}.

###### Lemma 4 (Lemma 7 [Yun19]).

Let gcsubscript𝑔𝑐g_{c} be the function in Lemma 3, we can construct
a function gv:ℝ(n+1)×(d+1)→ℝ(n+1)×d:subscript𝑔𝑣→superscriptℝ𝑛1𝑑1superscriptℝ𝑛1𝑑g_{v}:\mathbb{R}^{(n+1)\times(d+1)}\to\mathbb{R}^{(n+1)\times d} composed of
O​(n​δ−n​d)𝑂𝑛superscript𝛿𝑛𝑑O(n\delta^{-nd}) feed-forward layers (with hidden dimension q=1𝑞1q=1)
with activations in ΦΦ\Phi such that
gvsubscript𝑔𝑣g_{v} is defined as gv​(Z)=[gvt​k​n​(Z1),…,gvt​k​n​(Zn)]subscript𝑔𝑣𝑍superscriptsubscript𝑔𝑣𝑡𝑘𝑛subscript𝑍1…subscriptsuperscript𝑔𝑡𝑘𝑛𝑣subscript𝑍𝑛g_{v}(Z)=[g_{v}^{tkn}(Z_{1}),\dots,g^{tkn}_{v}(Z_{n})],
where for all j∈{1,…,n}𝑗1…𝑛j\in\{1,\dots,n\},

gvt​k​n​(gc​(L)j)=f​(L)jsuperscriptsubscript𝑔𝑣𝑡𝑘𝑛subscript𝑔𝑐subscript𝐿𝑗𝑓subscript𝐿𝑗g_{v}^{tkn}(g_{c}(L)_{j})=f(L)_{j}

#### A.2.3 Approximating modified Transformers by Transformers

The previous section assumed we used Transformers that used hardmax operator σHsubscript𝜎𝐻\sigma_{H} and
activations functions belonging to the set ΦΦ\Phi. This is without loss of generality as
following lemma shows.

###### Lemma 5 (Lemma 9 [Yun19]).

For each g∈𝒯¯2,1,1𝑔superscript¯𝒯211g\in\bar{\mathcal{T}}^{2,1,1} and 1≤p≤∞1𝑝1\leq p\leq\infty, ∃g∈𝒯2,1,4𝑔superscript𝒯214\exists g\in\mathcal{T}^{2,1,4} such that
dp​(g,g¯)≤ϵ/3subscript𝑑𝑝𝑔¯𝑔italic-ϵ3d_{p}(g,\bar{g})\leq\epsilon/3

Combining the above lemma with the Lemma 3, we get our main result:

###### Theorem 2.

Let 1≤p≤∞1𝑝1\leq p\leq\infty and ϵ>0italic-ϵ0\epsilon>0, there exists a transformer network
g∈𝒯D2,1,4𝑔superscriptsubscript𝒯𝐷214g\in\mathcal{T}_{D}^{2,1,4}
which achieves a ratio of dp​(f,g)≤ϵsubscript𝑑𝑝𝑓𝑔italic-ϵd_{p}(f,g)\leq\epsilon where D𝐷D is the sparse graph.

Since the sparsity graph associated with BigBird contains a star network, we know that it
can express any continuous function from a compact domain.

##### Contemporary work on Universal Approximability of Sparse Transformers

We would like to note that, contemporary work done by yun2020on, also parallelly explored the ability of sparse transformers with linear connections to capture sequence-to-sequence functions on the compact domain.

## Appendix B Turing Completeness

In this section, we will extend our results to the setting of Perez19. Our
exposition will largely use their proof structure but we will make a few changes.
We repeat some of the lemmas with the amendments to make the exposition
self-contained.

### B.1 Notation

##### Transformer Decoder

We need both an encoder and a decoder in the transformer for simulating a Turing machine.
We utilize the same notation used in Sec. A.1 for encoders.
The decoder is similar to an encoder but with additional attention to an external pair of key-value vectors (𝑲e∈ℝn×m,𝑽e∈ℝn×d)formulae-sequencesuperscript𝑲esuperscriptℝ𝑛𝑚superscript𝑽esuperscriptℝ𝑛𝑑({\bm{K}}^{\textbf{e}}\in\mathbb{R}^{n\times m},{\bm{V}}^{\textbf{e}}\in\mathbb{R}^{n\times d}), which usually come from the encoder stack.
A single layer of Transformer decoder is a parametric function DecDec\operatorname{Dec} receiving a sequence 𝒀j=(𝒚1,…,𝒚j)subscript𝒀𝑗subscript𝒚1…subscript𝒚𝑗{\bm{Y}}_{j}=({\bm{y}}_{1},\ldots,{\bm{y}}_{j}) of vectors in ℝdsuperscriptℝ𝑑\mathbb{R}^{d} plus the external (𝑲e,𝑽e)superscript𝑲esuperscript𝑽e({\bm{K}}^{\textbf{e}},{\bm{V}}^{\textbf{e}}) and returning a sequence of vectors 𝒁j=(𝒛1,…,𝒛j)subscript𝒁𝑗subscript𝒛1…subscript𝒛𝑗{\bm{Z}}_{j}=({\bm{z}}_{1},\ldots,{\bm{z}}_{j}) of the same length. Each 𝒛isubscript𝒛𝑖{\bm{z}}_{i} is a d𝑑d dimensional vector as well. DecDec\operatorname{Dec} has three components, one more than EncEnc\operatorname{Enc}:

- 1.

An attention mechanism Attn that takes in the sequence 𝒀jsubscript𝒀𝑗{\bm{Y}}_{j} and returns sequence (𝒑1,…,𝒑j)subscript𝒑1…subscript𝒑𝑗({\bm{p}}_{1},...,{\bm{p}}_{j}) of the same length and dimensionality;

- 2.

A cross-attention mechanism CrossAttn that takes in the sequence (𝒑1,…,𝒑j)subscript𝒑1…subscript𝒑𝑗({\bm{p}}_{1},...,{\bm{p}}_{j}) plus the external (𝑲e,𝑽e)superscript𝑲esuperscript𝑽e({\bm{K}}^{\textbf{e}},{\bm{V}}^{\textbf{e}}) and returns sequence (𝒂1,…,𝒂j)subscript𝒂1…subscript𝒂𝑗({\bm{a}}_{1},...,{\bm{a}}_{j}), with each 𝒂i∈ℝdsubscript𝒂𝑖superscriptℝ𝑑{\bm{a}}_{i}\in\mathbb{R}^{d}; and

- 3.

A two layer fully connected network O𝑂O that takes in a vector in ℝdsuperscriptℝ𝑑\mathbb{R}^{d} and returns a vector in ℝdsuperscriptℝ𝑑\mathbb{R}^{d}.

Then i𝑖i-th output vector of Dec⁡(𝒀j;𝑲e,𝑽e)Decsubscript𝒀𝑗superscript𝑲esuperscript𝑽e\operatorname{Dec}({\bm{Y}}_{j};{\bm{K}}^{\textbf{e}},{\bm{V}}^{\textbf{e}}) is computed as follows:

𝒛isubscript𝒛𝑖\displaystyle{\bm{z}}_{i}
=O​(𝒂i)+𝒂iabsent𝑂subscript𝒂𝑖subscript𝒂𝑖\displaystyle=O({\bm{a}}_{i})+{\bm{a}}_{i}

(3)

where
𝒂isubscript𝒂𝑖\displaystyle{\bm{a}}_{i}
=CrossAttn​(𝒑i,𝑲e,𝑽e)+𝒑iabsentCrossAttnsubscript𝒑𝑖superscript𝑲esuperscript𝑽esubscript𝒑𝑖\displaystyle=\textsc{CrossAttn}({\bm{p}}_{i},{\bm{K}}^{\textbf{e}},{\bm{V}}^{\textbf{e}})+{\bm{p}}_{i}

(4)

and
𝒑isubscript𝒑𝑖\displaystyle{\bm{p}}_{i}
=AttnD​(𝒀j)i+𝒚iabsentsubscriptAttn𝐷subscriptsubscript𝒀𝑗𝑖subscript𝒚𝑖\displaystyle=\textsc{Attn}_{D}({\bm{Y}}_{j})_{i}+{\bm{y}}_{i}

(5)

AttnDsubscriptAttn𝐷\textsc{Attn}_{D} and O𝑂O are as defined in Sec. A.1 and it remains to define CrossAttn.
The ithsuperscript𝑖thi^{\textrm{th}} output vector of multi-head cross-attention attention is given by

CrossAttn​(𝒀j)iCrossAttnsubscriptsubscript𝒀𝑗𝑖\displaystyle\textsc{CrossAttn}({\bm{Y}}_{j})_{i}
=∑h=1Hσ​((𝒚i​WQh)​(𝑲(e)​WKh)T)⋅(𝑽(e)​WVh)absentsuperscriptsubscriptℎ1𝐻⋅𝜎subscript𝒚𝑖superscriptsubscript𝑊𝑄ℎsuperscriptsuperscript𝑲𝑒superscriptsubscript𝑊𝐾ℎ𝑇superscript𝑽𝑒superscriptsubscript𝑊𝑉ℎ\displaystyle=\sum_{h=1}^{H}\sigma\left(({\bm{y}}_{i}W_{Q}^{h})({\bm{K}}^{(e)}W_{K}^{h})^{T}\right)\cdot({\bm{V}}^{(e)}W_{V}^{h})

(6)

where WQh,WKh,WVh∈ℝd×msuperscriptsubscript𝑊𝑄ℎsuperscriptsubscript𝑊𝐾ℎsuperscriptsubscript𝑊𝑉ℎsuperscriptℝ𝑑𝑚W_{Q}^{h},W_{K}^{h},W_{V}^{h}\in\mathbb{R}^{d\times m}, WVh∈ℝd×dsuperscriptsubscript𝑊𝑉ℎsuperscriptℝ𝑑𝑑W_{V}^{h}\in\mathbb{R}^{d\times d}, for all h=1,…​Hℎ1…𝐻h=1,\ldots H heads.

##### Turning Machine

We will use the same setup of Turning Machine that was used by Perez19 (see section B.4).
Given a Turing Machine M=(Q,Σ,δ,qi​n​i​t,F)𝑀𝑄Σ𝛿subscript𝑞𝑖𝑛𝑖𝑡𝐹M=(Q,\Sigma,\delta,q_{init},F), we use the following notation

q(j)superscript𝑞𝑗\displaystyle q^{(j)}
: state of Turing machine ​M​ at time ​j.:absent state of Turing machine 𝑀 at time 𝑗\displaystyle:\text{ state of Turing machine }M\text{ at time }j.

s(j)superscript𝑠𝑗\displaystyle s^{(j)}
: symbol under the head of ​M​ at time ​j.:absent symbol under the head of 𝑀 at time 𝑗\displaystyle:\text{ symbol under the head of }M\text{ at time }j.

v(j)superscript𝑣𝑗\displaystyle v^{(j)}
: symbol written by ​M​ at time ​j.:absent symbol written by 𝑀 at time 𝑗\displaystyle:\text{ symbol written by }M\text{ at time }j.

m(j)superscript𝑚𝑗\displaystyle m^{(j)}
: head direction in the transition of ​M​ at time ​j.:absent head direction in the transition of 𝑀 at time 𝑗\displaystyle:\text{ head direction in the transition of }M\text{ at time }j.

##### Vector representations

For a symbol s∈Σ𝑠Σs\in\Sigma, ⟦s⟧delimited-⟦⟧𝑠\llbracket\ s\ \rrbracket denotes its one-hot vector representation in ℚ|Σ|superscriptℚΣ\mathbb{Q}^{|\Sigma|}.
All the transformer intermediate vectors used in our simulations have dimension d=2​|Q|+4​|Σ|+16𝑑2𝑄4Σ16d=2|Q|+4|\Sigma|+16.
Note that we use five extra dimension as compared to Perez19.
We follow the convention used in Perez19 and write a a vector 𝒗∈ℚd𝒗superscriptℚ𝑑{\bm{v}}\in\mathbb{Q}^{d} arranged in four groups of values
as follows

𝒗=[𝒒1,𝒔1,x1,𝒒2,𝒔2,x2,x3,x4,x5,x6,𝒔3,x7,𝒔4,x8,x9,x10,x11,x12,x13,x14,x15,x16]𝒗[subscript𝒒1subscript𝒔1subscript𝑥1missing-subexpressionmissing-subexpressionmissing-subexpressionmissing-subexpressionsubscript𝒒2subscript𝒔2subscript𝑥2subscript𝑥3subscript𝑥4subscript𝑥5subscript𝑥6missing-subexpressionmissing-subexpressionmissing-subexpressionmissing-subexpressionsubscript𝒔3subscript𝑥7subscript𝒔4missing-subexpressionmissing-subexpressionmissing-subexpressionmissing-subexpressionsubscript𝑥8subscript𝑥9subscript𝑥10subscript𝑥11subscript𝑥12subscript𝑥13subscript𝑥14subscript𝑥15subscript𝑥16]\begin{array}[]{rcllr}{\bm{v}}&=&[&{\bm{q}}_{1},{\bm{s}}_{1},x_{1},\\
&&&{\bm{q}}_{2},{\bm{s}}_{2},x_{2},x_{3},x_{4},x_{5},x_{6},\\
&&&{\bm{s}}_{3},x_{7},{\bm{s}}_{4},\\
&&&x_{8},x_{9},x_{10},x_{11},x_{12},x_{13},x_{14},x_{15},x_{16}&]\end{array}

where 𝒒i∈ℚ|Q|subscript𝒒𝑖superscriptℚ𝑄{\bm{q}}_{i}\in\mathbb{Q}^{|Q|}, 𝒔i∈ℚ|Σ|subscript𝒔𝑖superscriptℚΣ{\bm{s}}_{i}\in\mathbb{Q}^{|\Sigma|}, and xi∈ℚsubscript𝑥𝑖ℚx_{i}\in\mathbb{Q}.

### B.2 Details of the Simulation

In this section, we give more details on the architecture of the encoder and decoder needed to implement our simulation strategy.

##### High Level Overview:

Given the Turing machine M𝑀M, we will show that a transformer with an appropriate encoder and decoder 𝒯Dsubscript𝒯𝐷\mathcal{T}_{D} can simulate each step of M𝑀M’s execution.
Our simulation strategy will mostly follow Perez19, except we will use a sparse attention mechanism.
The main idea is to maintain the current Turing machine state q(j)superscript𝑞𝑗q^{(j)} and symbol under the head s(j)superscript𝑠𝑗s^{(j)} as part of the decoder sequence 𝒀𝒀{\bm{Y}} for all time step j𝑗j so that we can always simulate the corresponding Turing machine transition δ​(q(j),s(j))=(q(j),v(j),m(j))𝛿superscript𝑞𝑗superscript𝑠𝑗superscript𝑞𝑗superscript𝑣𝑗superscript𝑚𝑗\delta(q^{(j)},s^{(j)})=(q^{(j)},v^{(j)},m^{(j)}).
The key difference will rise in Lemma B.4 of Perez19, where full attention is used to select the appropriate symbol from tape history in one step.
To accomplish the same task with sparse attention, we will exploit the associative property of max and break down the symbol selection over multiple steps.
Thus, unlike Perez19 one decoding step of our sparse transformer 𝒯Dsubscript𝒯𝐷\mathcal{T}_{D} does not correspond to one step of the Turing machine M𝑀M.
In particular, we will have two type of steps: compute step corresponding to update of M𝑀M’s state and intermediate steps corresponding to aggregating the max (which in turn is used for symbol selection).
Let i𝑖i denote the step of 𝒯Dsubscript𝒯𝐷\mathcal{T}_{D} and g​(i)𝑔𝑖g(i) denote the step of M𝑀M being simulated at step i𝑖i of the decoder.
At each decoding step we want to maintain the current Turing machine state qg​(i)superscript𝑞𝑔𝑖q^{g(i)} and symbol under the sg​(i)superscript𝑠𝑔𝑖s^{g(i)} in 𝒚isubscript𝒚𝑖{\bm{y}}_{i}.
For roughly O​(i)𝑂𝑖O(\sqrt{i}) intermediate steps the state will remain the same, while we aggregate information about relevant past output symbols through sparse attention.
To maintain the same state for intermediate steps, we introduce an extra switching layer (Sec. B.2.3).
Finally, at the next compute step we will make the transition to new state qg​(i)+1superscript𝑞𝑔𝑖1q^{g(i)+1}, new head movement mg​(i)superscript𝑚𝑔𝑖m^{g(i)}, and new output symbol vg​(i)superscript𝑣𝑔𝑖v^{g(i)} to be written.
Thereby we are able to completely simulate the given Turing machine M𝑀M.
As a result, we can prove the following main theorem:

###### Theorem 3.

There exists a sparse attention mechanism using O​(n)𝑂𝑛O(n) inner products such that the resulting class of Transformer Networks using this sparse attention mechanism is Turing Complete.

#### Encoder

As [Perez19], we use the same trivial single layer encoder where resulting 𝑲(e)superscript𝑲𝑒{\bm{K}}^{(e)} contains position embedding and 𝑽(e)superscript𝑽𝑒{\bm{V}}^{(e)} contains one-hot symbol representation.

#### Decoder

##### Sparse Self-Attention mechanism for Decoder

In this section, we will consider a particular instance of the sparse graph D𝐷D at decoder.
We define its edges to be given by the following relations:
∀j∈ℕ+,1≤k≤j+1formulae-sequencefor-all𝑗subscriptℕ1𝑘𝑗1\forall j\in\mathbb{N}_{+},1\leq k\leq j+1,

(j​(j+1)2+k,k​(k+1)2)​ and𝑗𝑗12𝑘𝑘𝑘12 and\displaystyle\left(\frac{j(j+1)}{2}+k,\frac{k(k+1)}{2}\right)\text{ and }

(j​(j+1)2+k,j​(j+1)2+k)​ if ​k>1​ else ​(j​(j+1)2+1,j​(j+1)2).𝑗𝑗12𝑘𝑗𝑗12𝑘 if 𝑘1 else 𝑗𝑗121𝑗𝑗12\displaystyle\left(\frac{j(j+1)}{2}+k,\frac{j(j+1)}{2}+k\right)\text{ if }k>1\text{ else }\left(\frac{j(j+1)}{2}+1,\frac{j(j+1)}{2}\right).

This graph can be seen as a special case of BigBird where first type
of edges are realizations of random and second type of edges correspond to locality.
Also note that this graph satisfies the left-to-right constraint of
decoder, i.e. no node attends to a node in the future.

##### Embeddings and positional encodings

Our construction needs a different positional encoding posDec:ℕ→ℚd:subscriptposDec→ℕsuperscriptℚ𝑑\operatorname{pos}_{\operatorname{Dec}}:\mathbb{N}\to\mathbb{Q}^{d} for decoder:

posDec⁡(i)=[0,…,0,0,…,0,0,…,0,1,g​(i)+1,1g​(i)+1,1(g​(i)+1)2,h​(i),0,0,0,0]subscriptposDec𝑖[0…0missing-subexpressionmissing-subexpressionmissing-subexpressionmissing-subexpression0…0missing-subexpressionmissing-subexpressionmissing-subexpressionmissing-subexpression0…0missing-subexpressionmissing-subexpressionmissing-subexpressionmissing-subexpression1𝑔𝑖11𝑔𝑖11superscript𝑔𝑖12ℎ𝑖0000]\begin{array}[]{rcllr}\operatorname{pos}_{\operatorname{Dec}}(i)&=&[&0,\ldots,0,\\
&&&0,\ldots,0,\\
&&&0,\ldots,0,\\
&&&1,g(i)+1,\frac{1}{g(i)+1},\frac{1}{(g(i)+1)^{2}},h(i),0,0,0,0&]\end{array}

where g​(i)=⌊−1+1+8​i2⌋𝑔𝑖118𝑖2g(i)=\left\lfloor\frac{-1+\sqrt{1+8i}}{2}\right\rfloor and h​(i)=g​(i+1)−g​(i)ℎ𝑖𝑔𝑖1𝑔𝑖h(i)=g(i+1)-g(i). Note that h​(i)ℎ𝑖h(i) reduces to a binary indicator variable 𝟏​{−1+1+8​i2=⌊−1+1+8​i2⌋}1118𝑖2118𝑖2\mathbf{1}\left\{\frac{-1+\sqrt{1+8i}}{2}=\left\lfloor\frac{-1+\sqrt{1+8i}}{2}\right\rfloor\right\}.

#### Induction Setup

We next show how to construct the decoder layers to produce the sequence of outputs 𝒚1,𝒚2,…subscript𝒚1subscript𝒚2…{\bm{y}}_{1},{\bm{y}}_{2},\ldots,
where 𝒚isubscript𝒚𝑖{\bm{y}}_{i} is given by:

𝒚i=[⟦qg​(i)⟧,⟦sg​(i)⟧,cg​(i),0,…,0,𝟎s,0,⟦w(i)⟧,0,0,0,0,0,u1(i),u2(i),u3(i),u4(i)]\begin{array}[]{rcllr}{{\bm{y}}}_{i}&=&[&\llbracket\ q^{g(i)}\ \rrbracket,\llbracket\ s^{g(i)}\ \rrbracket,c^{g(i)},\\
&&&0,\ldots,0,\\
&&&{\bm{0}}_{s},0,\llbracket\ w^{(i)}\ \rrbracket,\\
&&&0,0,0,0,0,u_{1}^{(i)},u_{2}^{(i)},u_{3}^{(i)},u_{4}^{(i)}&]\end{array}

That is, at step i𝑖i of our sparse decoder 𝒚isubscript𝒚𝑖{\bm{y}}_{i}, it will contain the information about the state of the turing machine M𝑀M at time g​(i)𝑔𝑖g(i), the symbol under the head of M𝑀M at time g​(i)𝑔𝑖g(i), and the current location of head of M𝑀M at time g​(i)𝑔𝑖g(i).
We also have a placeholder symbol w𝑤w and placeholder scalars u1,u2,u3subscript𝑢1subscript𝑢2subscript𝑢3u_{1},u_{2},u_{3}, whose role will be clear from our construction.

We consider as the starting vector for the decoder the vector

𝒚1=[⟦qinit⟧,⟦#⟧,0,0,…,0,0,…,0,0,…,0]\begin{array}[]{rcllr}{{\bm{y}}}_{1}&=&[&\llbracket\ q_{\text{init}}\ \rrbracket,\llbracket\ \#\ \rrbracket,0,\\
&&&0,\ldots,0,\\
&&&0,\ldots,0,\\
&&&0,\ldots,0&]\end{array}

We assume that the start head is at c(0)=0superscript𝑐00c^{(0)}=0, the initial state is q(0)=qinitsuperscript𝑞0subscript𝑞initq^{(0)}=q_{\text{init}}, and s(0)=#superscript𝑠0#s^{(0)}=\# as we initialize from clean tape.
We show the correctness of our construction by an inductive argument:
we describe the architecture piece by piece and at the same time will show for every r≥0𝑟0r\geq 0
, our architecture constructs 𝒚r+1subscript𝒚𝑟1{\bm{y}}_{r+1} from the previous vectors
(𝒚0,…,𝒚r)subscript𝒚0…subscript𝒚𝑟({\bm{y}}_{0},\ldots,{\bm{y}}_{r}).

Thus, assume that 𝒚1,…,𝒚rsubscript𝒚1…subscript𝒚𝑟{\bm{y}}_{1},\ldots,{\bm{y}}_{r} satisfy the properties stated above.
Since we are using positional encodings,
the actual input for the first layer of the decoder is the sequence

𝒚1+posDec⁡(1),𝒚2+posDec⁡(2),…,𝒚r+posDec⁡(r).subscript𝒚1subscriptposDec1subscript𝒚2subscriptposDec2…subscript𝒚𝑟subscriptposDec𝑟{\bm{y}}_{1}+\operatorname{pos}_{\operatorname{Dec}}(1),\ {\bm{y}}_{2}+\operatorname{pos}_{\operatorname{Dec}}(2),\ \ldots,\ {\bm{y}}_{r}+\operatorname{pos}_{\operatorname{Dec}}(r).

We denote by 𝒚¯isubscript¯𝒚𝑖\overline{{\bm{y}}}_{i} the vector 𝒚isubscript𝒚𝑖{\bm{y}}_{i} plus its positional encoding.
Thus we have ∀ 1≤i≤rfor-all1𝑖𝑟\forall\ 1\leq i\leq r that

𝒚¯i=[⟦qg​(i)⟧,⟦sg​(i)⟧,cg​(i),0,…,0,𝟎s,0,⟦w(i)⟧,1,g​(i)+1,1g​(i)+1,1(g​(i)+1)2,h​(i),u1(i),u2(i),u3(i),u4(i)]\begin{array}[]{rcllr}\overline{{\bm{y}}}_{i}&=&[&\llbracket\ q^{g(i)}\ \rrbracket,\llbracket\ s^{g(i)}\ \rrbracket,c^{g(i)},\\
&&&0,\ldots,0,\\
&&&{\bm{0}}_{s},0,\llbracket\ w^{(i)}\ \rrbracket,\\
&&&1,g(i)+1,\frac{1}{g(i)+1},\frac{1}{(g(i)+1)^{2}},h(i),u_{1}^{(i)},u_{2}^{(i)},u_{3}^{(i)},u_{4}^{(i)}&]\end{array}

#### B.2.1 Layer 1: Simulate Transition Function

In this layer, we use the cross-attention between encoder and decoder to access the
input string and a feed-forward network to simulate the transition function of M𝑀M.
The first self attention in Eq. 5 is not used in this layer and
we just produce the identity.
This identity function is achieved by setting all queries, keys, values to be 0
everywhere plus the residual connection.
Thus, we have 𝒑i1=𝒚¯isubscriptsuperscript𝒑1𝑖subscript¯𝒚𝑖{{\bm{p}}}^{1}_{i}=\overline{{\bm{y}}}_{i}.

Since 𝒑i1subscriptsuperscript𝒑1𝑖{\bm{p}}^{1}_{i} is of the form [¯,…,¯,1,g​(i)+1,¯,…,¯]¯absent…¯absent1𝑔𝑖1¯absent…¯absent[\underline{\phantom{A}},\ldots,\underline{\phantom{A}},1,g(i)+1,\underline{\phantom{A}},\ldots,\underline{\phantom{A}}], we know
by Lemma B.1 of Perez19 that if we use 𝒑i1subscriptsuperscript𝒑1𝑖{\bm{p}}^{1}_{i} to attend over the encoder we obtain

CrossAttn​(𝒑i1,𝑲e,𝑽e)=[0,…,0,0,…,0,⟦αg​(i)+1⟧,βg​(i)+1,𝟎s,0,…,0]\begin{array}[]{rcllr}\textsc{CrossAttn}({\bm{p}}^{1}_{i},{\bm{K}}^{\textbf{e}},{\bm{V}}^{\textbf{e}})&=&[&0,\ldots,0,\\
&&&0,\ldots,0,\\
&&&\llbracket\ \alpha^{g(i)+1}\ \rrbracket,\beta^{g(i)+1},{\bm{0}}_{s},\\
&&&0,\ldots,0&]\end{array}

where α𝛼\alpha and β𝛽\beta are as defined in Eq. (21) of [Perez19].
Thus in Eq. 4 we finally produce the vector 𝒂i1subscriptsuperscript𝒂1𝑖{\bm{a}}^{1}_{i} given by

𝒂i1=CrossAttn​(𝒑i1,𝑲e,𝑽e)+𝒑i1=[⟦qg​(i)⟧,⟦sg​(i)⟧,cg​(i),0,…,0,⟦αg​(i)+1⟧,βg​(i)+1,⟦w(i)⟧,1,g​(i)+1,1g​(i)+1,1(g​(i)+1)2,h​(i),u1(i),u2(i),u3(i),u4(i)]\begin{array}[]{rcllr}{\bm{a}}^{1}_{i}&=&&\textsc{CrossAttn}({\bm{p}}^{1}_{i},{\bm{K}}^{\textbf{e}},{\bm{V}}^{\textbf{e}})+{\bm{p}}^{1}_{i}\\
&=&[&\llbracket\ q^{g(i)}\ \rrbracket,\llbracket\ s^{g(i)}\ \rrbracket,c^{g(i)},\\
&&&0,\ldots,0,\\
&&&\llbracket\ \alpha^{g(i)+1}\ \rrbracket,\beta^{g(i)+1},\llbracket\ w^{(i)}\ \rrbracket,\\
&&&1,g(i)+1,\frac{1}{g(i)+1},\frac{1}{(g(i)+1)^{2}},h(i),u_{1}^{(i)},u_{2}^{(i)},u_{3}^{(i)},u_{4}^{(i)}&]\end{array}

(7)

As the final piece of the first decoder layer
we use a function O1​(⋅)subscript𝑂1⋅O_{1}(\cdot) (Eq. 3) that satisfies the following lemma.

###### Lemma 6 (Lemma B.2 [Perez19]).

There exists a two-layer feed-forward network O1:ℚd→ℚd:subscript𝑂1→superscriptℚ𝑑superscriptℚ𝑑O_{1}:\mathbb{Q}^{d}\to\mathbb{Q}^{d} such that with input vector 𝐚i1subscriptsuperscript𝐚1𝑖{\bm{a}}^{1}_{i} (Eq. 7) produces as output

O1​(𝒂i1)=[0,…,0,⟦qg​(i)+1⟧,⟦vg​(i)⟧,mg​(i),0,0,0,00,…,0,0,…,0]\begin{array}[]{rcllr}O_{1}({\bm{a}}^{1}_{i})&=&[&0,\ldots,0,\\
&&&\llbracket\ q^{g(i)+1}\ \rrbracket,\llbracket\ v^{g(i)}\ \rrbracket,m^{g(i)},0,0,0,0\\
&&&0,\ldots,0,\\
&&&0,\ldots,0&]\end{array}

That is, function O1​(⋅)subscript𝑂1⋅O_{1}(\cdot) simulates transition δ​(qg​(i),sg​(i))𝛿superscript𝑞𝑔𝑖superscript𝑠𝑔𝑖\delta(q^{g(i)},s^{g(i)})
to construct ⟦qg​(i)+1⟧delimited-⟦⟧superscript𝑞𝑔𝑖1\llbracket\ q^{g(i)+1}\ \rrbracket, ⟦vg​(i)⟧delimited-⟦⟧superscript𝑣𝑔𝑖\llbracket\ v^{g(i)}\ \rrbracket, and mg​(i)superscript𝑚𝑔𝑖m^{g(i)}
besides some other linear transformations.

Thus, finally the output of the first decoder layer is

𝒛i1=O1​(𝒂i1)+𝒂i1=[⟦qg​(i)⟧,⟦sg​(i)⟧,cg​(i),⟦qg​(i)+1⟧,⟦vg​(i)⟧,mg​(i),0,0,0,0,⟦αg​(i)+1⟧,βg​(i)+1,⟦w(i)⟧,1,g​(i)+1,1g​(i)+1,1(g​(i)+1)2,h​(i),u1(i),u2(i),u3(i),u4(i)]\begin{array}[]{rcllr}{\bm{z}}^{1}_{i}=O_{1}({\bm{a}}^{1}_{i})+{\bm{a}}^{1}_{i}&=&[&\llbracket\ q^{g(i)}\ \rrbracket,\llbracket\ s^{g(i)}\ \rrbracket,c^{g(i)},\\
&&&\llbracket\ q^{g(i)+1}\ \rrbracket,\llbracket\ v^{g(i)}\ \rrbracket,m^{g(i)},0,0,0,0,\\
&&&\llbracket\ \alpha^{g(i)+1}\ \rrbracket,\beta^{g(i)+1},\llbracket\ w^{(i)}\ \rrbracket,\\
&&&1,g(i)+1,\frac{1}{g(i)+1},\frac{1}{(g(i)+1)^{2}},h(i),u_{1}^{(i)},u_{2}^{(i)},u_{3}^{(i)},u_{4}^{(i)}&]\end{array}

#### B.2.2 Layer 2: Finding Head Node

In this layer, we only use the feed-forward network to evaluate the next location of the head.
The self-attention and cross-attention are set to be the identity function, so 𝒂i2=𝒑i2=𝒛i1superscriptsubscript𝒂𝑖2superscriptsubscript𝒑𝑖2superscriptsubscript𝒛𝑖1{\bm{a}}_{i}^{2}={\bm{p}}_{i}^{2}={\bm{z}}_{i}^{1}.
Recall that cg​(i)superscript𝑐𝑔𝑖c^{g(i)} is the cell to which M𝑀M is pointing to at time g​(i)𝑔𝑖g(i), and that it satisfies the following recursion cg​(i)+1=cg​(i)+mg​(i)superscript𝑐𝑔𝑖1superscript𝑐𝑔𝑖superscript𝑚𝑔𝑖c^{g(i)+1}=c^{g(i)}+m^{g(i)}, which can be expanded to see that
that cg​(i)+1=m(0)+m(1)+⋯+mg​(i)superscript𝑐𝑔𝑖1superscript𝑚0superscript𝑚1⋯superscript𝑚𝑔𝑖c^{g(i)+1}=m^{(0)}+m^{(1)}+\cdots+m^{g(i)}.
Its not difficult to see that a two layer network with non-linearity can compute cg​(i)+1/(g​(i)+1)superscript𝑐𝑔𝑖1𝑔𝑖1c^{g(i)+1}/(g(i)+1) and cg​(i)/(g​(i)+1)superscript𝑐𝑔𝑖𝑔𝑖1c^{g(i)}/(g(i)+1) from cg​(i)superscript𝑐𝑔𝑖c^{g(i)}, mg​(i)superscript𝑚𝑔𝑖m^{g(i)}, and 1/(g​(i)+1)1𝑔𝑖11/(g(i)+1) using the relation cg​(i)+1=cg​(i)+mg​(i)superscript𝑐𝑔𝑖1superscript𝑐𝑔𝑖superscript𝑚𝑔𝑖c^{g(i)+1}=c^{g(i)}+m^{g(i)}.
At the end of layer 2, we obtain

𝒛i2=O2​(𝒂i2)+𝒂i2=[⟦qg​(i)⟧,⟦sg​(i)⟧,cg​(i),⟦qg​(i)+1⟧,⟦vg​(i)⟧,cg​(i)+1,1g​(i)+1,1(g​(i)+1)2,cg​(i)+1g​(i)+1,cg​(i)g​(i)+1,⟦αg​(i)+1⟧,βg​(i)+1,⟦w(i)⟧,1,g​(i)+1,1g​(i)+1,1(g​(i)+1)2,h​(i),u1(i),u2(i),u3(i),u4(i)]\begin{array}[]{rcllr}{\bm{z}}^{2}_{i}\ =\ O_{2}({\bm{a}}^{2}_{i})+{\bm{a}}^{2}_{i}&=&[&\llbracket\ q^{g(i)}\ \rrbracket,\llbracket\ s^{g(i)}\ \rrbracket,c^{g(i)},\\
&&&\llbracket\ q^{g(i)+1}\ \rrbracket,\llbracket\ v^{g(i)}\ \rrbracket,c^{g(i)+1},\frac{1}{g(i)+1},\frac{1}{(g(i)+1)^{2}},\frac{c^{g(i)+1}}{g(i)+1},\frac{c^{g(i)}}{g(i)+1},\\
&&&\llbracket\ \alpha^{g(i)+1}\ \rrbracket,\beta^{g(i)+1},\llbracket\ w^{(i)}\ \rrbracket,\\
&&&1,g(i)+1,\frac{1}{g(i)+1},\frac{1}{(g(i)+1)^{2}},h(i),u_{1}^{(i)},u_{2}^{(i)},u_{3}^{(i)},u_{4}^{(i)}&]\end{array}

#### B.2.3 Layer 3: Distinguishing Node Type

This is an additional layer (not present in the work of [Perez19]), where we propagate computations
in our sparse graph.
In particular, we will use this layer to “compute” or accumulate state in intermediate nodes.
We make this clear below.
The self-attention and cross-attention are all set to be the identity function, so 𝒂i3=𝒑i3=𝒛i2superscriptsubscript𝒂𝑖3superscriptsubscript𝒑𝑖3superscriptsubscript𝒛𝑖2{\bm{a}}_{i}^{3}={\bm{p}}_{i}^{3}={\bm{z}}_{i}^{2}.
In this layer, we only use the dense attention layers to select the newly computed states or to continue
with previous states.
Using idea similar to Lemma B.6 of [Perez19], we can construct a dense network such that

O([𝒙,𝒚,𝒛,b]))={[𝟎,𝟎,𝟎,0]if ​b=1,[𝟎,𝒛−𝒚,−𝒛,0]if ​b=0.O([{\bm{x}},{\bm{y}},{\bm{z}},b]))=\begin{cases}[{\bm{0}},{\bm{0}},{\bm{0}},0]&\text{if }b=1,\\
[{\bm{0}},{\bm{z}}-{\bm{y}},-{\bm{z}},0]&\text{if }b=0.\end{cases}

The negatives are generated to offset results from skip connection.
We utilize such network to switch Turing machine state and position embedding for intermediate steps to the values received from previous time step and do nothing for compute nodes.
We use h​(i)ℎ𝑖h(i) as the flipping bit b𝑏b.
Thus, at end of layer 3, we obtain

𝒛i3=O3​(𝒂i3)+𝒂i3=[0,…,0,⟦q^(i)⟧,⟦v^(i)⟧,c^(i),1g​(i)+1,1(g​(i)+1)2,cg​(i)+1g​(i)+1,u^4(i),⟦α^(i)⟧,β^(i),𝟎s,1,u^1(i),u^2(i),u^3(i),h​(i),0,0,0,0]\begin{array}[]{rcllr}{\bm{z}}^{3}_{i}\ =\ O_{3}({\bm{a}}^{3}_{i})+{\bm{a}}^{3}_{i}&=&[&0,\ldots,0,\\
&&&\llbracket\ \hat{q}^{(i)}\ \rrbracket,\llbracket\ \hat{v}^{(i)}\ \rrbracket,\hat{c}^{(i)},\frac{1}{g(i)+1},\frac{1}{(g(i)+1)^{2}},\frac{c^{g(i)+1}}{g(i)+1},\hat{u}_{4}^{(i)},\\
&&&\llbracket\ \hat{\alpha}^{(i)}\ \rrbracket,\hat{\beta}^{(i)},{\bm{0}}_{s},\\
&&&1,\hat{u}_{1}^{(i)},\hat{u}_{2}^{(i)},\hat{u}_{3}^{(i)},h(i),0,0,0,0&]\end{array}

where we used h​(i)ℎ𝑖h(i) for selecting old states. In particular,

- •

We copy the input state and head position as is for intermediate nodes. We do not need to transition to next Turing machine states in these nodes.

q^(i)={qg​(i)+1if ​h​(i)=1qg​(i)if ​h​(i)=0superscript^𝑞𝑖casessuperscript𝑞𝑔𝑖1if ℎ𝑖1superscript𝑞𝑔𝑖if ℎ𝑖0\hat{q}^{(i)}=\begin{cases}q^{g(i)+1}&\text{if }h(i)=1\\
q^{g(i)}&\text{if }h(i)=0\end{cases} ,

v^(i)={vg​(i)if ​h​(i)=1w(i)if ​h​(i)=0superscript^𝑣𝑖casessuperscript𝑣𝑔𝑖if ℎ𝑖1superscript𝑤𝑖if ℎ𝑖0\hat{v}^{(i)}=\begin{cases}v^{g(i)}&\text{if }h(i)=1\\
w^{(i)}&\text{if }h(i)=0\end{cases} ,

c^(i)={cg​(i)+1if ​h​(i)=1cg​(i)if ​h​(i)=0superscript^𝑐𝑖casessuperscript𝑐𝑔𝑖1if ℎ𝑖1superscript𝑐𝑔𝑖if ℎ𝑖0\hat{c}^{(i)}=\begin{cases}c^{g(i)+1}&\text{if }h(i)=1\\
c^{g(i)}&\text{if }h(i)=0\end{cases} .

- •

To preserve the symbol under the head for intermediate nodes, we copy the previous symbol to α𝛼\alpha location and set β=g​(i)+1𝛽𝑔𝑖1\beta=g(i)+1, as the symbol at α𝛼\alpha location will be copied as the symbol under head for next transformer step by the final transformation layer if β=g​(i)+1𝛽𝑔𝑖1\beta=g(i)+1. Thus, we correctly preserve the previous symbol under head as Turing machine does not transition these nodes. For compute nodes, things happen as usual.

α^(i)={αg​(i)+1if ​h​(i)=1sg​(i)if ​h​(i)=0superscript^𝛼𝑖casessuperscript𝛼𝑔𝑖1if ℎ𝑖1superscript𝑠𝑔𝑖if ℎ𝑖0\hat{\alpha}^{(i)}=\begin{cases}\alpha^{g(i)+1}&\text{if }h(i)=1\\
s^{g(i)}&\text{if }h(i)=0\end{cases} ,

β^(i)={βg​(i)+1if ​h​(i)=1g​(i)+1if ​h​(i)=0superscript^𝛽𝑖casessuperscript𝛽𝑔𝑖1if ℎ𝑖1𝑔𝑖1if ℎ𝑖0\hat{\beta}^{(i)}=\begin{cases}\beta^{g(i)+1}&\text{if }h(i)=1\\
g(i)+1&\text{if }h(i)=0\end{cases} .

- •

Finally for the intermediate nodes, we copy the position embedding corresponding to current best symbol w𝑤w, which is stored in u1,u2,u3subscript𝑢1subscript𝑢2subscript𝑢3u_{1},u_{2},u_{3}. For compute node, we let the position embedding correspond to current Turing machine step.

u^1(i)={g​(i)+1if ​h​(i)=1u1(i)if ​h​(i)=0superscriptsubscript^𝑢1𝑖cases𝑔𝑖1if ℎ𝑖1superscriptsubscript𝑢1𝑖if ℎ𝑖0\hat{u}_{1}^{(i)}=\begin{cases}g(i)+1&\text{if }h(i)=1\\
u_{1}^{(i)}&\text{if }h(i)=0\end{cases} ,

u^2(i)={1(g​(i)+1)if ​h​(i)=1u2(i)if ​h​(i)=0superscriptsubscript^𝑢2𝑖cases1𝑔𝑖1if ℎ𝑖1superscriptsubscript𝑢2𝑖if ℎ𝑖0\hat{u}_{2}^{(i)}=\begin{cases}\frac{1}{(g(i)+1)}&\text{if }h(i)=1\\
u_{2}^{(i)}&\text{if }h(i)=0\end{cases} ,

u^3(i)={1(g​(i)+1)2if ​h​(i)=1u3(i)if ​h​(i)=0superscriptsubscript^𝑢3𝑖cases1superscript𝑔𝑖12if ℎ𝑖1superscriptsubscript𝑢3𝑖if ℎ𝑖0\hat{u}_{3}^{(i)}=\begin{cases}\frac{1}{(g(i)+1)^{2}}&\text{if }h(i)=1\\
u_{3}^{(i)}&\text{if }h(i)=0\end{cases} ,

u^4(i)={cg​(i)g​(i)+1if ​h​(i)=1u4(i)if ​h​(i)=0superscriptsubscript^𝑢4𝑖casessuperscript𝑐𝑔𝑖𝑔𝑖1if ℎ𝑖1superscriptsubscript𝑢4𝑖if ℎ𝑖0\hat{u}_{4}^{(i)}=\begin{cases}\frac{c^{g(i)}}{g(i)+1}&\text{if }h(i)=1\\
u_{4}^{(i)}&\text{if }h(i)=0\end{cases} .

For further simplification note that g​(i+1)=g​(i)𝑔𝑖1𝑔𝑖g(i+1)=g(i) if h​(i)=0ℎ𝑖0h(i)=0 else g​(i)+1𝑔𝑖1g(i)+1 when h​(i)=1ℎ𝑖1h(i)=1. With this fact, we can conclude that q^(i)=qg​(i+1)superscript^𝑞𝑖superscript𝑞𝑔𝑖1\hat{q}^{(i)}=q^{g(i+1)} and c^(i)=cg​(i+1)superscript^𝑐𝑖superscript𝑐𝑔𝑖1\hat{c}^{(i)}=c^{g(i+1)}. Thus, we can write,

𝒛i3=[0,…,0,⟦qg​(i+1)⟧,⟦v^(i)⟧,cg​(i+1),1g​(i)+1,1(g​(i)+1)2,cg​(i)+1g​(i)+1,u^4(i),⟦α^(i)⟧,β^(i),𝟎s,1,u^1(i),u^2(i),u^3(i),h​(i),0,0,0,0]\begin{array}[]{rcllr}{\bm{z}}^{3}_{i}&=&[&0,\ldots,0,\\
&&&\llbracket\ {q}^{g(i+1)}\ \rrbracket,\llbracket\ \hat{v}^{(i)}\ \rrbracket,{c}^{g(i+1)},\frac{1}{g(i)+1},\frac{1}{(g(i)+1)^{2}},\frac{c^{g(i)+1}}{g(i)+1},\hat{u}_{4}^{(i)},\\
&&&\llbracket\ \hat{\alpha}^{(i)}\ \rrbracket,\hat{\beta}^{(i)},{\bm{0}}_{s},\\
&&&1,\hat{u}_{1}^{(i)},\hat{u}_{2}^{(i)},\hat{u}_{3}^{(i)},h(i),0,0,0,0&]\end{array}

#### B.2.4 Layer 4: Finding next symbol on tape

To find the symbol on tape under next head position cg​(i)+1superscript𝑐𝑔𝑖1c^{g(i)+1}, we try to find what was written last at the location cg​(i)+1superscript𝑐𝑔𝑖1c^{g(i)+1}.
To facilitate this, following [Perez19], we define ℓ​(j)ℓ𝑗\ell(j) to be the last time (previous to j𝑗j) in which M𝑀M was pointing to position c(j)superscript𝑐𝑗c^{(j)},
or it is j−1𝑗1j-1 if this is the first time that M𝑀M is pointing to c(j)superscript𝑐𝑗c^{(j)}.
Recall j𝑗j is the Turing machine step counter, which is different from sparse transformer step i𝑖i. [Perez19] could utilize full attention mechanism to find vℓ​(j+1)superscript𝑣ℓ𝑗1v^{\ell(j+1)} at one go, but we have to do it over multiple steps owing to our sparse attention mechanism.

We use similar query, key, value functions as used for full attention by [Perez19] ∀ifor-all𝑖\forall i:

Q4​(𝒛i3)=[0,…,00,…,0,0,…,0,0,cg​(i)+1g​(i)+1,1g​(i)+1,13​(g​(i)+1)2,0,0,0,0,0]subscript𝑄4subscriptsuperscript𝒛3𝑖[0…0missing-subexpressionmissing-subexpressionmissing-subexpressionmissing-subexpression0…0missing-subexpressionmissing-subexpressionmissing-subexpressionmissing-subexpression0…0missing-subexpressionmissing-subexpressionmissing-subexpressionmissing-subexpression0superscript𝑐𝑔𝑖1𝑔𝑖11𝑔𝑖113superscript𝑔𝑖1200000]\begin{array}[]{rcllr}Q_{4}({\bm{z}}^{3}_{i})&=&[&0,\ldots,0\\
&&&0,\ldots,0,\\
&&&0,\ldots,0,\\
&&&0,\frac{c^{g(i)+1}}{g(i)+1},\frac{1}{g(i)+1},\frac{1}{3(g(i)+1)^{2}},0,0,0,0,0&]\\
\end{array}

K4​(𝒛i3)=[0,…,00,…,0,0,…,0,0,u^2(i),u^4(i),u^3(i),0,0,0,0,0]V4​(𝒛i3)=[0,…,0,0,…,0,𝟎s,0,⟦v^(i)⟧,0,0,0,0,0,u^1(i),u^2(i),u^3(i),u^4(i)]\begin{array}[]{rcllr}K_{4}({\bm{z}}^{3}_{i})&=&[&0,\ldots,0\\
&&&0,\ldots,0,\\
&&&0,\ldots,0,\\
&&&0,\hat{u}_{2}^{(i)},\hat{u}_{4}^{(i)},\hat{u}_{3}^{(i)},0,0,0,0,0&]\\
V_{4}({\bm{z}}^{3}_{i})&=&[&0,\ldots,0,\\
&&&0,\ldots,0,\\
&&&{\bm{0}}_{s},0,\llbracket\ \hat{v}^{(i)}\ \rrbracket,\\
&&&0,0,0,0,0,\hat{u}_{1}^{(i)},\hat{u}_{2}^{(i)},\hat{u}_{3}^{(i)},\hat{u}_{4}^{(i)}&]\end{array}

It is clear that the three functions are linear transformations and thus they can be defined by feed-forward networks. Notice that the query vector is always formed using current time step position embedding, whereas key and value vectors are formed using copied over entries for intermediate nodes and using current entries only for compute node.

Perez19 find the desired vl​(j+1)superscript𝑣𝑙𝑗1v^{l(j+1)} as vm​(j)superscript𝑣𝑚𝑗v^{m(j)} using full attention, where

m​(t)=arg​minm∈{0,…,t}⁡χtj=arg​minm∈{0,…,t}⁡|⟨Q4​(𝒛j3),K4​(𝒛m3)⟩|𝑚𝑡subscriptargmin𝑚0…𝑡superscriptsubscript𝜒𝑡𝑗subscriptargmin𝑚0…𝑡subscript𝑄4superscriptsubscript𝒛𝑗3subscript𝐾4superscriptsubscript𝒛𝑚3m(t)=\operatorname*{arg\,min}_{m\in\{0,...,t\}}\chi_{t}^{j}=\operatorname*{arg\,min}_{m\in\{0,...,t\}}|\langle Q_{4}({\bm{z}}_{j}^{3}),K_{4}({\bm{z}}_{m}^{3})\rangle|

Note the minimization is only over Turing machine steps, i.e. over compute nodes in our case.
We show below that we can estimates m​(j)𝑚𝑗m(j) by parts using sparse attention mechanism. The main idea is just to notice that minimization problem minm∈{0,…,t}⁡χtjsubscript𝑚0…𝑡superscriptsubscript𝜒𝑡𝑗\min_{m\in\{0,...,t\}}\chi_{t}^{j} can be expressed as min⁡{⋯​min⁡{min⁡{χ0j,χ1j},χ2j},…,χtj}⋯subscriptsuperscript𝜒𝑗0subscriptsuperscript𝜒𝑗1subscriptsuperscript𝜒𝑗2…subscriptsuperscript𝜒𝑗𝑡\min\{\cdots\min\{\min\{\chi^{j}_{0},\chi^{j}_{1}\},\chi^{j}_{2}\},...,\chi^{j}_{t}\} by the associativity property.

By definition of our graph D𝐷D, at every intermediate node i𝑖i of the form j​(j+1)/2+k𝑗𝑗12𝑘j(j+1)/2+k, i.e. where k>0𝑘0k>0, g​(i)=j𝑔𝑖𝑗g(i)=j and h​(i)=0ℎ𝑖0h(i)=0, we will attend over node k​(k+1)/2𝑘𝑘12k(k+1)/2 and best till now copied from i−1𝑖1i-1.
The node k​(k+1)/2𝑘𝑘12k(k+1)/2 is never an intermediate node as h​(k​(k+1)/2)=1ℎ𝑘𝑘121h(k(k+1)/2)=1 for all k𝑘k and in fact corresponds to Turing machine’s step k𝑘k.
This will help us select the key and value corresponding to min between node k​(k+1)/2𝑘𝑘12k(k+1)/2 and i−1𝑖1i-1.
In other words, at node i𝑖i of the form j​(j+1)/2+k𝑗𝑗12𝑘j(j+1)/2+k we would have evaluated m​(k)𝑚𝑘m(k) and corresponding value selected:

w(j​(j+1)/2+k+1)=v^m​(k−1)superscript𝑤𝑗𝑗12𝑘1superscript^𝑣𝑚𝑘1w^{(j(j+1)/2+k+1)}=\hat{v}^{m(k-1)}

and similarly for u𝑢u’s.
So after going through all the intermediate nodes, finally at the next compute node, i.e. when k=j+1𝑘𝑗1k=j+1, we will obtain the minimum value over all of 0,1,…,j01…𝑗0,1,...,j.
This implies at a compute node will be able to recover ℓ​(g​(i)+1)ℓ𝑔𝑖1\ell(g(i)+1) and its corresponding value as shown in Lemma B.4 of [Perez19].
Then we have that 𝒑i4subscriptsuperscript𝒑4𝑖{\bm{p}}^{4}_{i} is given by

𝒑i4=AttnD​(𝒁i3)+𝒛i3=[0,…,0,⟦qg​(i+1)⟧,⟦v^(i)⟧,cg​(i+1),0,cg​(i)+1g​(i)+1,u^4(i),⟦α^(i)⟧,β^(i),⟦w(i+1)⟧,1,u^1(i),u^2(i),u^3(i),h​(i),u1(i+1),u2(i+1),u3(i+1),u4(i+1)]\begin{array}[]{rcllr}{\bm{p}}^{4}_{i}&=&&\textsc{Attn}_{D}({\bm{Z}}_{i}^{3})+{\bm{z}}^{3}_{i}\\
&=&[&0,\ldots,0,\\
&&&\llbracket\ q^{g(i+1)}\ \rrbracket,\llbracket\ \hat{v}^{(i)}\ \rrbracket,c^{g(i+1)},0,\frac{c^{g(i)+1}}{g(i)+1},\hat{u}_{4}^{(i)},\\
&&&\llbracket\ \hat{\alpha}^{(i)}\ \rrbracket,\hat{\beta}^{(i)},\llbracket\ w^{(i+1)}\ \rrbracket,\\
&&&1,\hat{u}_{1}^{(i)},\hat{u}_{2}^{(i)},\hat{u}_{3}^{(i)},h(i),{u}_{1}^{(i+1)},{u}_{2}^{(i+1)},{u}_{3}^{(i+1)},{u}_{4}^{(i+1)}&]\end{array}

(8)

The cross-attention and feed-forward network are set to be identity, so 𝒛i4=𝒂i4=𝒑i4superscriptsubscript𝒛𝑖4superscriptsubscript𝒂𝑖4superscriptsubscript𝒑𝑖4{\bm{z}}_{i}^{4}={\bm{a}}_{i}^{4}={\bm{p}}_{i}^{4}.

#### B.2.5 Final transformation

We finish our construction by using the final transformation function F​(⋅)𝐹⋅F(\cdot) from the corresponding lemma from Perez19, with a slight modification.

###### Lemma 7 (Lemma B.5 [Perez19]).

There exists a function F:ℚd→ℚd:𝐹→superscriptℚ𝑑superscriptℚ𝑑F:\mathbb{Q}^{d}\to\mathbb{Q}^{d} defined by a feed-forward network such that

F​(𝒛r4)=[⟦qg​(r+1)⟧,⟦sg(r+1))⟧,cg​(r+1),0,…,0,𝟎s,0,⟦w(r+1)⟧,0,0,0,0,0,u1(r+1),u2(r+1),u3(r+1),u4(r+1)]=𝒚r+1\begin{array}[]{rcllr}F({\bm{z}}^{4}_{r})&=&[&\llbracket\ q^{g(r+1)}\ \rrbracket,\llbracket\ s^{g(r+1))}\ \rrbracket,c^{g(r+1)},\\
&&&0,\ldots,0,\\
&&&{\bm{0}}_{s},0,\llbracket\ w^{(r+1)}\ \rrbracket,\\
&&&0,0,0,0,0,{u}_{1}^{(r+1)},{u}_{2}^{(r+1)},{u}_{3}^{(r+1)},{u}_{4}^{(r+1)}]\\
&=&&{\bm{y}}_{r+1}\end{array}

The modification is to let w,u1,u2,u3𝑤subscript𝑢1subscript𝑢2subscript𝑢3w,u_{1},u_{2},u_{3} to pass through.
This yields the desired input to transformer at next time step for both intermediate and compute node, thereby concluding our induction.

## Appendix C Limitations

Finally, we show that sparse attention mechanisms can not universally replace dense attention mechanisms, i.e. there is no free lunch.
We demonstrate a natural task which can be solved by the full attention mechanism in O​(1)𝑂1O(1)-layers.
However, under standard complexity theoretic assumptions, we show that this problem will require
Ω~​(n)~Ω𝑛\tilde{\Omega}(n)-layers for any sparse attention layers with O~​(n)~𝑂𝑛\tilde{O}(n) edges (not just BigBird).
(We use the standard notation Ω~​(n)~Ω𝑛\tilde{\Omega}(n) to hide the dependence on poly-logarithmic factors. )

We consider the simple problem of finding the furthest vector for each vector
in the given sequence of length n𝑛n and dimension d∈Ω​(log2⁡n)𝑑Ωsuperscript2𝑛d\in\Omega(\log^{2}n). The assumption on
the dimension is mild , as in many situations the dimension d=768𝑑768d=768 is actually comparable to
the number of n𝑛n.

###### Task 1.

Given n𝑛n unit vectors {u1,…,un}subscript𝑢1…subscript𝑢𝑛\{u_{1},\dots,u_{n}\}, each in ℝdsuperscriptℝ𝑑\mathbb{R}^{d} where d=Θ​(log2⁡n)𝑑Θsuperscript2𝑛d=\Theta(\log^{2}n),
compute f​(u1,…,un)→(u1∗,…,un∗)→𝑓subscript𝑢1…subscript𝑢𝑛subscript𝑢superscript1…subscript𝑢superscript𝑛f(u_{1},\dots,u_{n})\to(u_{1^{*}},\dots,u_{n^{*}}) where for a fixed j∈[n]𝑗delimited-[]𝑛j\in[n], we define
j∗=arg​maxk⁡‖uk−uj‖22superscript𝑗subscriptargmax𝑘superscriptsubscriptnormsubscript𝑢𝑘subscript𝑢𝑗22j^{*}=\operatorname*{arg\,max}_{k}\|u_{k}-u_{j}\|_{2}^{2}.

Finding vectors that are furthest apart boils down to minimizing inner product search
in case of unit vectors. For a full-attention mechanism with appropriate query and keys,
this task is very easy as we can evaluate all pair-wise inner products.

The impossibility for sparse-attention follows from hardness results stemming from
Orthogonal Vector Conjecture (OVC) [abboud2015tight, abboud2014consequences, williams2005new, backurs2015edit], which is a widely used assumption in fine-grained complexity. Informally, it states that
one cannot determine if the minimum inner product among n𝑛n Boolean vectors is 00 in
subquadratic time.

###### Conjecture 1 (Orthogonal Vectors Conjecture).

For every ϵ>0italic-ϵ0\epsilon>0, there is a c≥1𝑐1c\geq 1 such that given n𝑛n Boolean vectors in
d𝑑d dimension, cannot determine if there is a pair of orthogonal vectors in O​(n2−ϵ)𝑂superscript𝑛2italic-ϵO(n^{2-\epsilon})
time on instances with d≥c​log⁡n𝑑𝑐𝑛d\geq c\log n.

Using 1, we show a reduction to show that a
transformer g∈𝒯DH=O​(d),m=O​(d),q=O​(d)𝑔superscriptsubscript𝒯𝐷formulae-sequence𝐻𝑂𝑑formulae-sequence𝑚𝑂𝑑𝑞𝑂𝑑g\in\mathcal{T}_{D}^{H=O(d),m=O(d),q=O(d)} for any sparse directed graph D𝐷D
which completes Task 111 must require a superlinear number of layers.

###### Proposition 2.

There exists a single layer full-attention network g∈𝒯H=1,m=2​d,q=0𝑔superscript𝒯formulae-sequence𝐻1formulae-sequence𝑚2𝑑𝑞0g\in\mathcal{T}^{H=1,m=2d,q=0} that can
evaluate Task 1, i.e. g​(u1,…,un)=[u1∗,…,un∗]𝑔subscript𝑢1…subscript𝑢𝑛subscript𝑢superscript1…subscript𝑢superscript𝑛g(u_{1},...,u_{n})=[u_{1^{*}},\dots,u_{n^{*}}], but for any sparse-attention network in 𝒯DH=O​(d),m=O​(d),q=O​(d)superscriptsubscript𝒯𝐷formulae-sequence𝐻𝑂𝑑formulae-sequence𝑚𝑂𝑑𝑞𝑂𝑑\mathcal{T}_{D}^{H=O(d),m=O(d),q=O(d)} with
graph D𝐷D having O~​(n)~𝑂𝑛\tilde{O}(n) edges (i.e. inner product evaluations), would require Ω~​(n1−o​(1))~Ωsuperscript𝑛1𝑜1\tilde{\Omega}(n^{1-o(1)}) layers.

###### Proof.

We will break this proof into two parts:

##### Part 1: The full attention mechanism can solve the problem in O​(1)𝑂1O(1) layer

We begin by providing an explicit construction of a single layer full self-attention that can evaluate Task 1.

Step 1
We embed each uisubscript𝑢𝑖u_{i} in the input into ℝ2​dsuperscriptℝ2𝑑\mathbb{R}^{2d} as follows:

xi:=E​(ui)=[ui;0]assignsubscript𝑥𝑖𝐸subscript𝑢𝑖subscript𝑢𝑖0x_{i}:=E(u_{i})=[u_{i};0]

(9)

Step 2
Construct query, key, value functions as follows:

Q​([a;b])𝑄𝑎𝑏\displaystyle Q([a;b])
=−aabsent𝑎\displaystyle=-a

(10)

K​([a;b])𝐾𝑎𝑏\displaystyle K([a;b])
=aabsent𝑎\displaystyle=a

V​([a;b])𝑉𝑎𝑏\displaystyle V([a;b])
=[0;a]absent0𝑎\displaystyle=[0;a]

Then Attn(Q(xi),K(X),V(X)=[0;uarg​maxj⁡⟨−ui,uj⟩]\mathrm{Attn}(Q(x_{i}),K(X),V(X)=[0;u_{\operatorname*{arg\,max}_{j}\langle-u_{i},u_{j}\rangle}]. Then,

ai=Attn​(Q​(xi),K​(X),V​(X))+xi=[ui;uarg​maxj⁡⟨−ui,uj⟩]=[ui;ui∗]subscript𝑎𝑖Attn𝑄subscript𝑥𝑖𝐾𝑋𝑉𝑋subscript𝑥𝑖subscript𝑢𝑖subscript𝑢subscriptargmax𝑗subscript𝑢𝑖subscript𝑢𝑗subscript𝑢𝑖subscript𝑢superscript𝑖a_{i}=\mathrm{Attn}(Q(x_{i}),K(X),V(X))+x_{i}=[u_{i};u_{\operatorname*{arg\,max}_{j}\langle-u_{i},u_{j}\rangle}]=[u_{i};u_{i^{*}}]

(11)

Step 3
Let O​(ai)=0𝑂subscript𝑎𝑖0O(a_{i})=0, then the output zi=[ui;ui∗]subscript𝑧𝑖subscript𝑢𝑖subscript𝑢superscript𝑖z_{i}=[u_{i};u_{i^{*}}] as desired.

To complete the argument, observe that it now only takes O​(n)𝑂𝑛O(n) inner products to check
if there is a pair of orthogonal vectors as we need only compare ⟨ui,ui∗⟩subscript𝑢𝑖subscript𝑢superscript𝑖\left\langle u_{i},u_{i^{*}}\right\rangle.

##### Part 2: Every Sparse Attention Mechanism will need Ω~​(n1−ϵ)~Ωsuperscript𝑛1italic-ϵ\tilde{\Omega}(n^{1-\epsilon}) layers

We prove by contradiction that it is impossible to solve Task 1 by any
g∈𝒯DH=O​(d),m=O​(d),q=O​(d)𝑔superscriptsubscript𝒯𝐷formulae-sequence𝐻𝑂𝑑formulae-sequence𝑚𝑂𝑑𝑞𝑂𝑑g\in\mathcal{T}_{D}^{H=O(d),m=O(d),q=O(d)} sparse-attention graph D𝐷D with O~​(n)~𝑂𝑛\tilde{O}(n) edges.

Suppose we can solve Task 1 using a network g∈𝒯DH=O​(d),m=O​(d),q=O​(d)𝑔superscriptsubscript𝒯𝐷formulae-sequence𝐻𝑂𝑑formulae-sequence𝑚𝑂𝑑𝑞𝑂𝑑g\in\mathcal{T}_{D}^{H=O(d),m=O(d),q=O(d)} that has
l𝑙l layers. Recall that all the computation we do in one layer is:

aisubscript𝑎𝑖\displaystyle a_{i}
=AttnD(Q(xi),K(XN​(i)),V(XN​(i))+xi\displaystyle=\textsc{Attn}_{D}(Q(x_{i}),K(X_{N(i)}),V(X_{N(i)})+x_{i}

(12)

xisubscript𝑥𝑖\displaystyle x_{i}
=O​(ai)+aiabsent𝑂subscript𝑎𝑖subscript𝑎𝑖\displaystyle=O(a_{i})+a_{i}

where AttnDsubscriptAttn𝐷\mathrm{Attn}_{D} is defined in eq. AT.

Thus, total computation per layer is O~​(n​d3)~𝑂𝑛superscript𝑑3\tilde{O}(nd^{3}) and consequently O~​(n​l​d3)~𝑂𝑛𝑙superscript𝑑3\tilde{O}(nld^{3}) for the
whole network consisting of l𝑙l layers.

We can use the result of Task 1 to solve the orthogonal vector (OV) problem (defined
in 1) in linear time. So in total, we will be able to solve any instance of OV in O~​(n​l​d3)~𝑂𝑛𝑙superscript𝑑3\tilde{O}(nld^{3}) time.

Now if l=O​(n1−ϵ)𝑙𝑂superscript𝑛1italic-ϵl=O(n^{1-\epsilon}) for any ϵ>0italic-ϵ0\epsilon>0 and d=Θ​(log2⁡n)𝑑Θsuperscript2𝑛d=\Theta(\log^{2}n), then it
appears that we are able to solve OV in O~​(n2−ϵ)~𝑂superscript𝑛2italic-ϵ\tilde{O}(n^{2-\epsilon}) which contradicts 1.
Therefore, we need at least Ω~​(n1−o​(1))~Ωsuperscript𝑛1𝑜1\tilde{\Omega}(n^{1-o(1)}) layers.
∎

## Appendix D Implementation details

We optimize the code for modern hardware. Hardware accelerators like GPUs and TPUs truly shine on
coalesced memory operations which load blocks of contiguous bytes at once. Thus, its not very efficient
to have small sporadic look-ups caused by a sliding window or random element queries. We alleviate this by
“blockifying” the lookups.

(a) Random Attention

(b) Window Attention

(c) Global Attention

(d) BigBird

##### GPU/TPU and Sparsity

Ideally, if the adjacency matrix A𝐴A described in Sec. 2 is sparse, one would hope this would be sufficient to speed up the implementation.
Unfortunately, it is well known [gray2017gpu, yao2019balanced], that such sparse multiplications cannot be efficiently implemented in GPUs. GPUs have thousands of cores performing operations in parallel. Thus, we cannot efficiently perform the sparse matrix multiplication mentioned in section Sec. 2.

As a result we propose to first blockify the attention pattern i.e. we pack sets of query and keys together and then define attention on these blocks.
It is easier to explain this process using the example
shown in Fig. 3. Suppose, there are 121212 query and 121212 key vectors to attend to. Using a block size of 222, we split the query matrix into 12/2=6122612/2=6 blocks and similarly the key matrix into 12/2=6122612/2=6 blocks. Then the three different building components of BigBird are defined on the block matrix. In particular the three different components are:

- 1.

Random attention: Each query block attends to r𝑟r random key blocks. In Fig. 3(a), r=1𝑟1r=1 with block size 222. This implies that each query block of size 222 randomly attends to a key block of size 222.

- 2.

Window local attention: While creating the block, we ensure that the number of query blocks and the number of key blocks are the same. This helps us in defining the block window attention. Every query block with index j𝑗j attends to key block with index j−(w−1)/2𝑗𝑤12j-(w-1)/2 to j+(w−1)/2𝑗𝑤12j+(w-1)/2, including key block j𝑗j. In Fig. 3(b), w=3𝑤3w=3 with block size 222. It means that each query block j𝑗j (size 222 queries) attends to key block j−1,j,j+1𝑗1𝑗𝑗1j-1,j,j+1.

- 3.

Global attention: Global attention remains the same as defined in Sec. 2, but we compute it in terms of blocks. In Fig. 3(c), g=1𝑔1g=1 with block size 222. For BigBird-itc this implies that one query and key block, attend to everyone.

The resulting overall attention matrix is shown in Fig. 3(d).
Unfortunately, simply trying to compute this attention score as multiplying arbitrary pairs of query and key vectors would require use of gather operation, which is inefficient.
Upon closer examination of window and global attention, we observe that we can compute these attention scores without using a gather operation.

(a) Full all pair attention can be obtained by direct matrix multiplication between the query and key matrix. Groupings just shown for guidance.

(b) Block diagonal attention can be computed by “blockifying” the query and key matrix

(c) Window local attention obtained by “blockifying” the query/key matrix, copying key matrix, and rolling the resulting key tensor (Obtaining rolled key-block tensor is illustrated in detail in Fig. 5). This ensures that every query attends to at least one block and at most two blocks of keys of size b𝑏b on each side.

(d) Window + Random attention obtained by following the procedure above along with gathering some random key blocks.

Recall, full dense attention scores can be calculated by simple matrix product of query and key matrix with a cost of O​(n2​d)𝑂superscript𝑛2𝑑O(n^{2}d), as illustrated in Fig. 4(a).
Now note that if we blockify the query and key matrix and multiply, then with only O​(n​b​d)𝑂𝑛𝑏𝑑O(nbd) cost we will obtain the block diagonal portion of the attention score, as depicted in Fig. 4(b).
To elaborate this lets assume that Q,K∈ℝn×d𝑄𝐾superscriptℝ𝑛𝑑Q,K\in\mathbb{R}^{n\times d} are the query and key matrix corresponding to n𝑛n tokens such that Qi.=xi​WQsubscript𝑄𝑖subscript𝑥𝑖subscript𝑊𝑄Q_{i.}=x_{i}W_{Q} and Ki.=xi​WKsubscript𝐾𝑖subscript𝑥𝑖subscript𝑊𝐾K_{i.}=x_{i}W_{K}.
We reshape n×d𝑛𝑑n\times d query matrix, Q𝑄Q, and key matrix, K𝐾K, along the sequence length to obtain ⌈n/b⌉×b×d𝑛𝑏𝑏𝑑\lceil n/b\rceil\times b\times d tensors Q′superscript𝑄′Q^{\prime} and K′superscript𝐾′K^{\prime} respectively.
Now we multiply the two tensors as

Aj​s​t=∑uQj​s​u′​Kj​t​u′,j=0,1,…,⌈n/b⌉formulae-sequencesubscript𝐴𝑗𝑠𝑡subscript𝑢subscriptsuperscript𝑄′𝑗𝑠𝑢subscriptsuperscript𝐾′𝑗𝑡𝑢𝑗01…𝑛𝑏A_{jst}=\sum_{u}Q^{\prime}_{jsu}K^{\prime}_{jtu},\qquad j=0,1,...,\lceil n/b\rceil

(13)

The resulting A𝐴A tensor of size ⌈n/b⌋×b×b\lceil n/b\rfloor\times b\times b can be reshaped to correspond to the block diagonal portion of the full attention pattern.
Now to extend the attention from block diagonal to a window, i.e. where query block with index j𝑗j attends to key block with index j−(w−1)/2𝑗𝑤12j-(w-1)/2 to j+(w−1)/2𝑗𝑤12j+(w-1)/2, we make w𝑤w copies of the reshaped key tensor K′superscript𝐾′K^{\prime}.
We “roll” each copy of key-block tensor incrementally along the first axis of length ⌈n/b⌉𝑛𝑏\lceil n/b\rceil as illustrated in Fig. 5.
Multiplying these w𝑤w rolled key-block tensors with the query-block tensor would yield the desired window attention scores (Fig. 4(c)).
Likewise the global component, we can always include the first g𝑔g blocks from key tensor corresponding to the global tokens.
Finally, for the random attention, which is very small (r=3𝑟3r=3 for all of our experiments), we resort to using gather ops (Fig. 4(d)).
Also note by design, each query block attends to exactly r𝑟r random blocks.

Thus, the result of all the three components is basically a compact dense tensor K′′superscript𝐾′′K^{\prime\prime} of size ⌈n/b⌉×(g+w+r)​b×d𝑛𝑏𝑔𝑤𝑟𝑏𝑑\lceil n/b\rceil\times(g+w+r)b\times d as shown in Fig. 6.
Computing the final attention score then just boils down to a dense tensor multiplication, at which TPU/GPU are very efficient.
Specifically, we need to multiply Q′superscript𝑄′Q^{\prime} (size: ⌈n/b⌉×b×d𝑛𝑏𝑏𝑑\lceil n/b\rceil\times b\times d) and K′′superscript𝐾′′K^{\prime\prime} (size: ⌈n/b⌉×(g+w+r)​b×d𝑛𝑏𝑔𝑤𝑟𝑏𝑑\lceil n/b\rceil\times(g+w+r)b\times d) with a cost of O​(n​(g+w+r)​b​d)𝑂𝑛𝑔𝑤𝑟𝑏𝑑O(n(g+w+r)bd) to yield the desired attention score tensor of size ⌈n/b⌉×b×(g+w+r)​b𝑛𝑏𝑏𝑔𝑤𝑟𝑏\lceil n/b\rceil\times b\times(g+w+r)b, which can be reshaped to obtain all the attention scores according to the BigBird pattern.

## Appendix E NLP experiments details

### E.1 MLM Pretraining

We use four publicly available datasets Books [zhu2015aligning], CC-News [guu2020realm], Stories [trinh2018simple] and Wikipedia to pretrain BigBird.
We borrow the sentencepiece vocabulary as RoBERTa (which is in turn borrowed from GPT2).
We split any document longer than 409640964096 into multiple documents and we join documents that were much smaller than 409640964096.
Following the original BERT training, we mask 15%percent1515\% of tokens in these four datasets, and train to predict the mask. We warm start from RoBERTa’s checkpoint.
We train two different models: BigBird-itc-base and BigBird-etc-base. The hyper-parameters for these two models are given in Tab. 8. In all experiments we use a learning
rate warmup over the first 10,000 steps, and linear
decay of the learning rate.

Similar to the norm, we trained a large version of model as well, which has 24 layers with 16 heads and hidden dimension of 1024.
Following the observation from RoBERTa, we pretrain on a larger batch size of 2048 for this size.
For BigBird-itc the block length was kept same as base size, but for BigBird-etc the block length was almost doubled to 169. All the remaining parameters were the same.

Parameter

BigBird-itc

BigBird-etc

Block length, b𝑏b

646464

84

##\# of global token, g𝑔g

2×b2𝑏2\times b

256256256

Window length, w𝑤w

3×b3𝑏3\times b

3×b3𝑏3\times b

##\# of random token, r𝑟r

3×b3𝑏3\times b

00

Max. sequence length

409640964096

409640964096

##\# of heads

121212

121212

##\# of hidden layers

121212

121212

Hidden layer size

768768768

768768768

Batch size

256256256

256256256

Loss

MLM

MLM

Activation layer

gelu

gelu

Dropout prob

0.10.10.1

0.10.10.1

Attention dropout prob

0.10.10.1

0.10.10.1

Optimizer

Adam

Adam

Learning rate

10−4superscript10410^{-4}

10−4superscript10410^{-4}

Compute resources

8×8888\times 8 TPUv3

8×8888\times 8 TPUv3

Dataset

##\# tokens

Avg. doc len.

Books [zhu2015aligning]

1.01.01.0B

373737K

CC-News [guu2020realm]

7.47.47.4B

561561561

Stories [trinh2018simple]

7.77.77.7B

8.28.28.2K

Wikipedia

3.13.13.1B

592592592

Model
Base
Large

RoBERTa (sqln: 512)
1.846
1.496

Longformer (sqln: 4096)
1.705
1.358

BigBird-itc (sqln: 4096)

1.678
1.456

BigBird-etc (sqln: 4096)

1.611
1.274

Instances

Instance Length

Dataset

Training
Dev

Median
Max

HotpotQA-distractor [yang2018hotpotqa]

904479044790447
740574057405

122712271227
356035603560

Natural Questions [kwiatkowski2019natural]

307373307373307373
783078307830

325832583258
779627796277962

TriviaQA [JoshiTriviaQA2017]

618886188861888
799379937993

4900
32755

WikiHop [welbl2018constructing]

437384373843738
512951295129

154115411541
203372033720337

Parameter

HotpotQA

NaturalQ

TriviaQA

WikiHop

Global token location

itc
etc

itc
etc

itc
etc

itc
etc

##\# of global token, g𝑔g

128128128
256256256

128128128
230230230

128128128
320320320

128128128
430430430

Window length, w𝑤w

192192192
252252252

192192192
252252252

192192192
252252252

192192192
252252252

##\# of random token, r𝑟r

192192192
00

192192192
00

192192192
00

192192192
00

Max. sequence length

409640964096
409640964096

409640964096
409640964096

409640964096
409640964096

409640964096
409640964096

##\# of heads

121212
121212

121212
121212

121212
121212

121212
121212

##\# of hidden layers

121212
121212

121212
121212

121212
121212

121212
121212

Hidden layer size

768768768
768768768

768768768
768768768

768768768
768768768

768768768
768768768

Batch size

323232
323232

128128128
128128128

323232
323232

646464
646464

Loss

cross-entropy

cross-entropy

cross-entropy

cross-entropy

golden spans

golden spans

noisy spans [clark2017simple]

ans choices

Compute resources

4×2424\times 2 TPUv3

4×8484\times 8 TPUv3

4×2424\times 2 TPUv3

4×4444\times 4 TPUv3

Parameter

HotpotQA

NaturalQ

TriviaQA

WikiHop

Global token location

etc

etc

etc

etc

##\# of global token, g𝑔g

256256256

230230230

320320320

430430430

Window length, w𝑤w

507507507

507507507

507507507

507507507

##\# of random token, r𝑟r

00

00

00

00

Max. sequence length

409640964096

409640964096

409640964096

409640964096

##\# of heads

161616

161616

161616

161616

##\# of hidden layers

242424

242424

242424

242424

Hidden layer size

102410241024

102410241024

102410241024

102410241024

Batch size

323232

646464

323232

646464

Loss

cross-entropy

cross-entropy

cross-entropy

cross-entropy

Num epochs

{5,9}59\{5,9\}

{3,5}35\{3,5\}

{3,5}35\{3,5\}

{5,10}510\{5,10\}

Optimizer

Adam

Adam

Adam

LAMB

Learning rate

3×10−53superscript1053\times 10^{-5}

{5,10}×10−5510superscript105\{5,10\}\times 10^{-5}

{3,5}×10−535superscript105\{3,5\}\times 10^{-5}

{2,5}×10−525superscript105\{2,5\}\times 10^{-5}

Compute resources

4×4444\times 4 TPUv3

4×8484\times 8 TPUv3

4×4444\times 4 TPUv3

4×8484\times 8 TPUv3

### E.2 Question Answering

The detailed statistics of the four datasets used are given in Tab. 11.
All the hyperparameters for BigBird, used for creating Tab. 2 are shown in Tab. 12 and those submitted to get Tab. 3
are shown in Tab. 13.
We use two types of regularization in training:

- •

We used a variant of contrastive predictive coding [oord2018representation] as a dual encoder model.

- •

We use position embedding for itc and relative position encoding [shaw2018self] for etc.

Next, we will mention the dataset/task specific part of the model.

##### HotpotQA

The data consists of each question with multiple evidence paragraphs.
We filtered 16 QA where the answer was not in the given evidences.
For BigBird-itc, we use first 128128128 global tokens.
For BigBird-etc, we have one global token for each question token, one for each evidence paragraph, and one for each sentence within the paragraph, for a maximum of 256256256 global token.
We use a dense layer on the output corresponding to global token of the evidence paragraph to predict whether its a supporting fact with a threshold over the output logits.
The answer type (yes/no/span) is predicted with a single dense layer from the global CLS token.
For span based answers, the spans are predicted with dense layers on the sequence with the distance between start and end positions to be no more than 30 words.
The spans are ranked by sum of start and end logits.

##### Natural Questions

Here also the data consists of question with supporting evidence, but in form of a single, potentially long, document and not multiple paragraphs.
We largely follow the setup of [alberti2019bert].
For documents, that are longer than
4096, a sliding window approach is used
with stride of 2048.
We use CLS token at the beginning, followed by the question followed by a separator token followed by the document as input.
For BigBird-itc, we make the first 128128128 tokens as global. For BigBird-etc, we make a global token for CLS, question, and one token for each of the paragraphs.
We train four predictors at the final layer to predict long answer start, long answer end, short answer start and short answer end respectively.
Instead of independently predicting the start and end of answers we first predict the start and then predict the best end location beyond the start.
For short answer, we limit the distance between start and end positions to be no more than 38 words.
The answer type (null, yes, no, short, long) is predicted from CLS token output embedding.
When the logit for a yes/no answer is higher than the logits for short, long or null answer, we replace the short answer with a corresponding yes/no text.

##### TriviaQA

The data consists of question-answer pairs with Wikipedia articles as the “noisy” supporting evidence.
We call them noisy because the given Wikipedia articles may or may not contain the answer.
Moreover, the answer entities is not annotated to appropriate span in the article, rather all occurrences found using fuzzy string matching are listed.
We use CLS token at the beginning, followed by the question followed by a separator token followed by the document as input.
For BigBird-itc, we make the first 128128128 tokens as global.
For BigBird-etc, we make a global token for CLS, question, and one token for each sentence up to a maximum of 320 global tokens.
Given the noisy nature of answer span, we follow clark2017simple for training.
We use a dense layer on the sequence to predict the answer span for each article independently, with the distance between start and end positions to be no more than 16 words.
For each article the span with maximum start logit + end logit is chosen.
Then we normalize over all the documents associated with that question.

##### WikiHop

For each question in WikiHop, we are given upto 797979 candidates, and 636363 supporting paragraphs.
In our BigBird-itc model, following beltagy2020longformer, we concatenate the answer and the question with special tokens,
[q] Question [/q] [ans] Ans1 [/ans] ……\ldots [ans] AnsN [/ans] along with the context.
As the start of the text, always contains questions followed by answers, we make the first 128128128 token attend globally.
In BigBird-etc model, we do not need to insert special [ans], [/ans] etc. as we design global tokens appropriately.
Along with global tokens for question, we have one per candidate answer up to a maximum of 430.
Further, we linked answer tokens to their mentions using relative position label.
Lastly, we use a dense layer that takes in the output vector corresponding to a candidate answer, and predicts a score for the current candidate to be the correct answer.
We apply this dense layer to each candidate independently and
the candidate with the best score is picked as our final answer.

It is worthwhile to note that explicitly designed attention connection in etc works slightly better, the random connection based itc is pretty competative.

### E.3 Relationship to Contemporary Work

##### Longformer

child2019generating introduced localized sliding window to reduce computation.
A more recent version, which includes localized sliding windows and global tokens was introduced independently by
Longofrmer[beltagy2020longformer]. Although BigBird contains additional random tokens, there are also differences in the way global and local tokens are realized. In particular even when there is no random token, as used to get SoTA in question answering, there are two key differences between Longformer and BigBird-etc (see [ainslie2020etc]):

- 1.

We use global-local attention with relative
position encodings enables it to better handle structured inputs

- 2.

Unlike Longformer, we train the global tokens using CPC loss and learn their use during finetuning.

### E.4 Classification

We try two types of classification task.

Parameter

IMDb
Arxiv
Patents
Hyperpartisan
Yelp-5

Batch size

64
64
64
32
32

Learning rate

1×10−51superscript1051\times 10^{-5}
3×10−53superscript1053\times 10^{-5}
5×10−55superscript1055\times 10^{-5}
5×10−65superscript1065\times 10^{-6}
2×10−52superscript1052\times 10^{-5}

Num epochs

40
10
3
15
2

TPUv3 slice

4×4444\times 4
4×4444\times 4
4×4444\times 4
4×2424\times 2
4×8484\times 8

##\# of heads

12
16

##\# of hidden layers

12
24

Hidden layer size

768
102410241024

Block length, b𝑏b

64

Global token location

itc

##\# of global token, g𝑔g

2×b2𝑏2\times b

Window length, w𝑤w

3×b3𝑏3\times b

##\# of random token, r𝑟r

3×b3𝑏3\times b

Max. sequence length

4096

Vocab size

503585035850358

Activation layer

gelu

Dropout prob

0.1

Attention dropout prob

0.1

Loss

cross-entropy

Optimizer

Adam

Model

IMDb [maas2011learning]

Yelp-5 [zhang2015character]

Arxiv [he2019long]

Patents [lee2020patent]

Hyperpartisan [kiesel2019semeval]

# Examples
25000
650000
30043
1890093
645

# Classes
2
5
11
663
2

Excess fraction
0.14
0.04
1.00
0.90
0.53

SoTA

[thongtan2019sentiment] 97.4

[abreu2019hierarchical] 73.28

[olson2019adapting] 87.96

[olson2019adapting] 69.01

[jiang2019team] 90.6

RoBERTa
95.0±0.2plus-or-minus95.00.295.0\pm 0.2
71.75
87.42
67.07
87.8±0.8plus-or-minus87.80.887.8\pm 0.8

BigBird
95.2±0.2plus-or-minus95.20.295.2\pm 0.2
72.16
92.31
69.30
92.2±1.7plus-or-minus92.21.7\mathbf{92.2\pm 1.7}

System
MNLI-(m/mm)
QQP
QNLI
SST-2
CoLA
STS-B
MRPC
RTE

392k
363k
108k
67k
8.5k
5.7k
3.5k
2.5k

BERT
84.6/83.4
71.2
90.5
93.5
52.1
85.8
88.9
66.4

XLNet
86.8/-
91.4
91.7
94.7
60.2
89.5
88.2
74.0

RoBERTa
87.6/-
91.9
92.8
94.8
63.6
91.2
90.2
78.7

BigBird
87.5/87.3
88.6
92.2
94.6
58.5
87.8
91.5
75.0

##### Document classification

We experiment on datasets of different lengths and contents,
as listed in Tab. 15.
In particular, we look at sentiment analysis (IMDb [maas2011learning] and Yelp-5 [zhang2015character]) task and topic assignment (Arxiv [he2019long], Patents [lee2020patent], and Hyperpartisan [kiesel2019semeval]) task.
Following BERT, we used one layer with cross entropy loss on top of the first [CLS] token from the BigBird encoder consuming 4096 tokens.
We report the results of document classification experiments in Tab. 15.
We compare against state-of-the-art (SoTA) methods for each dataset and plain RoBERTa model with 512 tokens truncation.
In all experiments we use a learning rate warmup over the first 10% steps, and linear decay of the learning rate and detail list of remaining hyperparameters are provided in
Tab. 14.
For better quantitative evaluation, we compute the fraction of the dataset that exceeds 512 tokens, i.e. the length at which the document are often truncated.
We see that gains of using BigBird are more significant when we have longer documents and fewer training examples.
For instance, using base sized model,
BigBird improves state-of-the-art for Arxiv dataset by about 𝟓%percent5\bm{5\%} points.
On Patents dataset, there is improvement over using simple BERT/RoBERTa, but given the large size of training data the improvement over SoTA (which is not BERT based) is not significant.
Note that this performance gain is not seen for much
smaller IMDb dataset.
Along with experimental setup detail, we present detailed results in Sec. E.4 which show competitive performance.

##### GLUE

The General Language Understanding Evaluation (GLUE) benchmark [wang2018glue], test language models on 8 different natural language understanding tasks.
We used the same training parameters as mentioned in https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.glue.md. Our model parameters are b=64,g=2×b,w=3×b,r=3×bformulae-sequence𝑏64formulae-sequence𝑔2𝑏formulae-sequence𝑤3𝑏𝑟3𝑏b=64,g=2\times b,w=3\times b,r=3\times b ( we used the BigBird-itc base model pretrained on MLM task).
We compare the performance of BigBird to BERT, XLNet [yang2019xlnet] and RoBERTa in Tab. 16. We find that even on task that have a much smaller context, our performance is competitive to full attention models.

### E.5 Summarization

As discussed in Sec. 4.1,
given the small length of output sequence, we used sparse BigBird attention only for encoder, while keeping the full attention for decoder.
The number of hidden layers, number of heads, and hidden dimension is same for encoder and decoder.
The hyperparameters are detailed in Tab. 17.
We summarize our result in Tab. 20. In all experiments, we use a learning
rate warmup over the first 10,000 steps, and square root
decay of the learning rate.

Parameter

Base: BigBird-RoBERTa

Large: BigBird-Pegasus

Block length, b𝑏b

646464

646464

Global token location

itc

itc

##\# of global token, g𝑔g

2×b2𝑏2\times b

2×b2𝑏2\times b

Window length, w𝑤w

3×b3𝑏3\times b

3×b3𝑏3\times b

##\# of random token, r𝑟r

3×b3𝑏3\times b

3×b3𝑏3\times b

Max. encoder sequence length

BBC-XSUM:   .1024

1024

CNN/DM:   .2048

2048

Others:   .3072

3072

Max. decoder sequence length

BBC-XSUM:   10.64

64

CNN/DM:   1.128

128

Others:   1.256

256

Beam size

5

5

Length penalty

BBC-XSUM:   100.7

0.7

Others:   100.8

0.8

##\# of heads

121212

161616

##\# of hidden layers

121212

161616

Hidden layer size

768768768

102410241024

Batch size

128128128

128128128

Loss

teacher forced

teacher forced

cross-entropy

cross-entropy

Activation layer

gelu

gelu

Dropout prob

0.10.10.1

0.10.10.1

Attention dropout prob

0.10.10.1

0.10.10.1

Optimizer

Adam

Adafactor

Learning rate

1×10−51superscript1051\times 10^{-5}

1×10−41superscript1041\times 10^{-4}

Compute resources

4×4444\times 4 TPUv3

4×8484\times 8 TPUv3

Instances

Input Length

Output Length

Dataset

Training
Dev
Test

Median
90%-ile

Median
90%-ile

Arxiv [cohan2018discourse]

203037
6436
6440

6151
14405

171
352

PubMed [cohan2018discourse]

119924
6633
6658

2715
6101

212
318

BigPatent [sharma2019bigpatent]

1207222
67068
67072

3082
7693

123
197

Following success of several recent works [rothe2019leveraging, liu2019roberta], we warm start our encoder-decoder BigBird transformer model with pretrained weights and the weights between encoder and decoder are shared.
In particular, the query/key/value matrix of self-attention and all the feedforward layers are shared between encoder and decoder.
The only variable that is initialized randomly is the encoder-decoder attention.
For base sized model, we utilize our MLM pretrained model on 4096 sequence length from Sec. E.1, which is in turn initialized using the public RoBERTa checkpoint.
For the large size model, we lift weight from the state-of-the-art Pegasus model [zhang2019pegasus], which is pretrained using an objective designed for summarization task.

To check if sparse attention causes significant degradation as compared to full attention, we further experiment on two shorter but popular datasets, where full attention can be used without significantly truncating the document.
The statistics of these two datasets are in Tab. 19.
We see that our performance is competitive, which shows that sparse attention can achieve similar performance to a full attention models.

Instances

Input Length

Output Length

Dataset

Training
Dev
Test

Median
90%-ile

Median
90%-ile

BBC XSum [narayan2018don]

204044
11332
11334

359
920

25
32

CNN/DailyMail [hermann2015teaching]

287113
13368
11490

777
1439

59
93

Model

BBC XSum

CNN/DailyMail

R-1
R-2
R-L

R1
R2
R-L

Prior Art

Lead

16.3016.3016.30
1.611.611.61
11.9511.9511.95

39.6039.6039.60
17.7017.7017.70
36.2036.2036.20

PtGen [see2017get]

29.7029.7029.70
9.219.219.21
23.2423.2423.24

39.5339.5339.53
17.2817.2817.28
36.3836.3836.38

ConvS2S [gehring2017convolutional]

31.8931.8931.89
11.5411.5411.54
25.7525.7525.75

−-
−-
−-

MMN [kim2018abstractive]

32.0032.0032.00
12.1012.1012.10
26.0026.0026.00

−-
−-
−-

Bottom-Up [gehrmann2018bottom]

−-
−-
−-

41.2241.2241.22
18.6818.6818.68
38.3438.3438.34

TransLM [khandelwal2019sample]

−-
−-
−-

39.6539.6539.65
17.7417.7417.74
36.8536.8536.85

UniLM [dong2019unified]

−-
−-
−-

43.4743.4743.47
20.3020.3020.30
40.6340.6340.63

Extr-Abst-BERT [liu2019text]

38.81
16.50
31.27

42.13
19.60
39.18

BART [lewis2019bart]

45.14
22.27
37.25

44.16
21.28
40.90

Base

Transformer [vaswani2017attention]

29.61
9.47
23.17

34.89
13.13
32.12

+ RoBERTa [rothe2019leveraging]

39.92
17.33
32.63

39.44
18.69
36.80

+ Pegasus [zhang2019pegasus]

39.79
16.58
31.70

41.79
18.81
38.93

BigBird-RoBERTa

39.52
17.22
32.30

39.25
18.46
36.61

Large

Pegasus (Reported) [zhang2019pegasus]

47.60
24.83
39.64

44.16
21.56
41.30

Pegasus (Re-eval)

47.37
24.31
39.23

44.15
21.56
41.05

BigBird-Pegasus

47.12
24.05
38.80

43.84
21.11
40.74

## Appendix F Genomics experiments details

In this section we provide details of the experimental setup for BigBird on genomics data.

### F.1 Pretraining

We try to keep the experimental setup as close to a typical NLP pipeline.
In this regard, we take human reference GRCh37777https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.39 and convert it into documents 𝒟𝒟\mathcal{D}. Each document d∈𝒟𝑑𝒟d\in\mathcal{D} is a sequence of sentences, where each sentence is a sequence of fragments of DNA. We construct the documents as follows:

- 1.

Start with empty document set D=∅𝐷D=\emptyset.

- 2.

For each chromosome C𝐶C, repeat the following procedure 10 times.

- (a)

Pick uniformly at random a starting point q𝑞q between base pairs 0 and 5000 from the 5’ end.

- (b)

Repeat until q>|C|𝑞𝐶q>|C|

- i.

Pick uniformly at random s𝑠s a number between 50 and 100 to denote number of sentences per document.

- ii.

Constructs a document d𝑑d containing s𝑠s sentences using consecutive base pairs (bps). The length of each sentence is chosen uniformly at random between 500-1000. Thus the resulting document has 25,0002500025,000 - 100,000100000100,000 bps.

- iii.

D=D​⋃d𝐷𝐷𝑑D=D\bigcup d

- iv.

q=q+|d|𝑞𝑞𝑑q=q+|d|

By this procedure we end-up with approximately 450​K450𝐾450K documents.

Next we run sentencepiece [kudo2018sentencepiece] tokenization on the resulting documents. In particular, using 5 characters as the building blocks (four for bases - A, T, C, G and one for missing symbol N), we construct a byte pair encoding table of size 32k, with each token representing 8.78 base pairs on average.

Using the above constructed documents, we construct a dataset for two pretraining tasks following devlin2018bert:

- •

Masked Language Model (MLM):
In order to train a deep bidirectional representation, BERT training introduces the MLM task, where we simply mask out 15% of the input tokens at random, and then predict those masked tokens.
We can simply replace such masked out of the tokens with a [MASK] placeholder, but it leads to a distribution mis-match for downstream tasks which will not have such placeholders. To mitigate with this issue, out of the 15% of the tokens selected for masking:

- –

80% of the tokens are actually replaced with the token [MASK].

- –

10% of the time tokens are replaced with a random token.

- –

10% of the time tokens are left unchanged, but are still predicted at output.

We run this entire sequence through the BigBird transformer encoder and then predict corresponding to the masked positions, based on the context provided by the other non-masked tokens in the sequence.

- •

Next Sentence Prediction (NSP):
In order to understand relationship between two sequences, BERT training introduces the NSP task, where we predict if a given pair of sequences are contiguous or not.
During training the model gets as input pairs of sequences separated by [SEP] token along with a [CLS] token at the start. Overall the input pattern is: [CLS] sequence A [SEP] sequence B [SEP]. For 50% of the time the second sequence comes from true sequence after the first one. Remaining 50% of the time it is a a random sequence from the full dataset. The model is then required to predict this relationship using the output corresponding to the [CLS] token, which is fed into a simple binary classification layer.

The sequence of steps is visually elaborated in Fig. 9.
The model is trained with both MLM and NSP together. Training hyperparameter is provided in second columns of Tab. 21. In all experiments we use a learning
rate warmup over the first 10,000 steps, and linear
decay of the learning rate.

We additionally performed a simple ablation study to validate the hypothesis, that similar to NLP, having a larger context improves performance.
We use MLM task described above to test how BigBird performed with sequences of different length.
Accuracy on MLM task with increasing sequence length is shown in Fig. 8.
Not only longer context improves final accuracy, it also leads to faster learning, as we have now more opportunities for masking.

### F.2 Promoter Region Prediction

The promoter region plays an important role in transcription initiation and thus its recognition is an important area of interest in the field of bioinformatics.
Following oubounyt2019deepromoter, we use datasets from Eukaryotic Promoter Database (EPDnew) [dreos2013epd], which contains
29,597 promoter region in the human genome.
Around the transcription start site (TSS), we extract a sequence of 8000 bp (-5000 +3000 bp) from the human reference genome GRCh37.
Since EPDnew uses newer GRCh38, we convert to GRCh37 coordinates using LiftOver [kent2002human].

Following oubounyt2019deepromoter for each promoter region example, a negative example (non-promoter sequences) with the same size of the positive one is constructed as follow:
The positive sequence is divided into 20 subsequences. Then, 12 subsequences are picked randomly and substituted randomly. The remaining 8 subsequences are conserved. This process is illustrated in Figure 1 of [oubounyt2019deepromoter]. Applying this process to the positive set results in new non-promoter sequences with conserved parts from promoter sequences (the unchanged subsequences, 8 subsequences out of 20). These parameters enable generating a negative set that has 32 and 40% of its sequences containing conserved portions of promoter sequences.

We prefix and append each example with [CLS] and [SEP] token respectively.
The output corresponding to the [CLS] token from BigBird transformer encoder is fed to a simple binary classification layer.
We fine-tune the pretrained BigBird from Sec. F.1 using hyper-parameters described in Tab. 21.
We note that high performance is not surprising due to the overlap in the nature of negative example generation and MLM pretraining.

### F.3 Chromatin-Profile Prediction

The first step of sequence-based algorithmic framework for predicting non-coding effects is to build a model to predict, large scale chromatic profile [zhou2015predicting].

Parameter

Pretraining

Promoter Region

Chromatin-Profile

Block length, b𝑏b

646464

646464

646464

Global token location

itc

itc

itc

##\# of global token, g𝑔g

2×b2𝑏2\times b

2×b2𝑏2\times b

2×b2𝑏2\times b

Window length, w𝑤w

3×b3𝑏3\times b

3×b3𝑏3\times b

3×b3𝑏3\times b

##\# of random token, r𝑟r

3×b3𝑏3\times b

3×b3𝑏3\times b

3×b3𝑏3\times b

Max. Sequence Length

409640964096

409640964096

409640964096

##\# of heads

121212

121212

121212

##\# of hidden layers

121212

121212

121212

Hidden layer size

768768768

768768768

768768768

Batch Size

256256256

256256256

256256256

Vocab Size

320003200032000

320003200032000

320003200032000

Loss

MLM+NSP

BCE

919 x +ve upweighted

BCE

Dropout prob

0.10.10.1

0.10.10.1

0.10.10.1

Optimizer

Adam

Adam

Adam

Learning rate

0.00010.00010.0001

0.00010.00010.0001

0.00010.00010.0001

##\# of steps

100000010000001000000

711

500000

Compute Resources

8×8888\times 8 TPUv3

8×8888\times 8 TPUv3

8×8888\times 8 TPUv3

In this paper, we use the dataset provided in zhou2015predicting888 http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz, to train BigBird to predict the chromatic profile.

Each training sample consists of a 8,000-bp sequence from the human GRCh37 reference genome centered on each 200-bp bin and is paired with a label vector for 919 chromatin features.
As before, we prefix and append each example with [CLS] and [SEP] token respectively.
The output corresponding to the [CLS] token from BigBird transformer encoder is fed to a linear layer with 919 heads. Thus we jointly predict the 919 independent binary classification problems.
We fine-tune the pretrained BigBird from Sec. F.1 using hyper-parameters described in Tab. 21.
As the data is highly imbalanced data (way more negative examples than positive examples), we upweighted loss function for positive examples by factor of 8.

We used training and testing split provided by zhou2015predicting using chromosomes and strictly non-overlapping. Chromosome 8 and 9 were excluded from training to test chromatin feature prediction performances, and the rest of the autosomes were used for training and validation. 4,000 samples on chromosome 7 spanning the genomic coordinates 30,508,751–35,296,850 were used as the validation set.

As the predicted probability for each sequence in DeepSea zhou2015predicting was computed as the ensemble average of the probability predictions for the forward and complementary sequence pairs, we also predict using an ensemble of two BigBird model trained independently.

Generated on Sun Mar 3 22:38:07 2024 by LaTeXML

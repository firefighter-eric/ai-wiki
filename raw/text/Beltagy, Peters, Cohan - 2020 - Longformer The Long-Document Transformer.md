# Beltagy, Peters, Cohan - 2020 - Longformer The Long-Document Transformer

- Source HTML: `raw/html/Beltagy, Peters, Cohan - 2020 - Longformer The Long-Document Transformer.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2004.05150
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Longformer: The Long-Document Transformer

Iz Beltagy
 Matthew E. Peters11footnotemark: 1
 Arman Cohan11footnotemark: 1
Allen Institute for Artificial Intelligence, Seattle, WA, USA
{{\{beltagy,matthewp,armanc}}\}@allenai.org

  Equal contribution.

###### Abstract

Transformer-based models are unable to process long sequences due to their self-attention operation, which scales quadratically with the sequence length.
To address this limitation, we introduce the Longformer with an attention mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or longer.
Longformer’s attention mechanism is a drop-in replacement for the standard self-attention and combines a local windowed attention with a task motivated global attention.
Following prior work on long-sequence transformers,
we evaluate Longformer on character-level language modeling
and achieve state-of-the-art results on text8 and enwik8.
In contrast to most prior work,
we also pretrain Longformer and finetune it on a variety of downstream tasks.
Our pretrained Longformer consistently outperforms RoBERTa on long document tasks
and sets new state-of-the-art results on WikiHop and
TriviaQA. We finally introduce the Longformer-Encoder-Decoder (LED), a Longformer variant for supporting long document generative sequence-to-sequence tasks, and demonstrate its effectiveness on the arXiv summarization dataset.111https://github.com/allenai/longformer

## 1 Introduction

Transformers Vaswani2017AttentionIA have achieved state-of-the-art results in a wide range of natural language tasks including generative language modeling transformerxl; gpt2 and discriminative language understanding bert.
This success is partly due to the self-attention component which enables the network to capture contextual information from the entire sequence. While powerful, the memory and computational requirements of self-attention grow quadratically with sequence length, making it infeasible (or very expensive) to process long sequences.

To address this limitation, we present Longformer, a modified Transformer architecture
with a self-attention operation that scales linearly with the sequence length, making it versatile for processing long documents (Fig 1).
This is an advantage for natural language tasks such as long document classification, question answering (QA), and coreference resolution, where existing approaches partition or shorten the long context into smaller sequences that fall within the typical 512 token limit of BERT-style pretrained models. Such partitioning could potentially result in loss of important cross-partition information, and to mitigate this problem, existing methods often rely on complex architectures to address such interactions. On the other hand, our proposed Longformer is able to build contextual representations of the entire context using multiple layers of attention, reducing the need for task-specific architectures.

Recent work has addressed the computational inefficiency of Transformers on long sequences (see Tab. 1).
However, they primarily focus on autoregressive language modeling (LM),
while the application of long document transformers to document-level NLP tasks in the transfer learning setting NIPS2015_5949; Peters2018DeepCW; Howard2018UniversalLM; bert has remained largely unexplored. We address this gap and show that Longformer’s attention mechanism can act as a drop-in replacement for the self-attention mechanism in pretrained Transformers, and leads to gains across a suite of document NLP tasks.

Longformer’s attention mechanism is a combination of a windowed local-context self-attention and an end task motivated global attention that encodes inductive bias about the task.
Through ablations and controlled trials we show both attention types are essential – the local attention is primarily used to build contextual representations, while the global attention allows Longformer to build full sequence representations for prediction.

We first evaluate Longformer on autoregressive character-level language modeling using a combination of windowed and a new dilated attention pattern, allowing the model to process sequences of up to 32K characters on modern GPUs. We achieve state-of-the-art results on text8 and enwik8 benchmark datasets, demonstrating the effectiveness of Longformer in long document modeling.

Then, to evaluate Longformer’s ability to replace the full self-attention operation of existing pretrained models, we pretrain
it with the masked language modeling (MLM) objective, continuing from the RoBERTa roberta released checkpoint.
After pretraining, we apply it to downstream language tasks through finetuning and demonstrate that Longformer consistently outperforms RoBERTa on a wide range of document-level natural language tasks including text classification, QA, and coreference resolution, achieving state-of-the-art results on two of these datasets.

We finally introduce a variant of Longformer which instead of an encoder-only Transformer architecture, it follows an encoder-decoder architecture similar to the original Transformer model Vaswani2017AttentionIA, and it is intended for sequence-to-sequence (seq2seq) learning Sutskever2014SequenceTS. We call this model Longformer-Encoder-Decoder (LED) that uses Longformer’s efficient attention pattern on the encoder network, allowing it to address long document seq2seq tasks such as summarization. We demonstrate the effectiveness of LED on the arXiv summarization dataset arxiv2018.

Model
attention
char-LM
other
pretrain

matrix

tasks

Transformer-XL transformerxl

ltr
yes
no
no

Adaptive Span adaptivespan

ltr
yes
no
no

Compressive compressive

ltr
yes
no
no

\hdashline[0.4pt/2pt]
Reformer reformer

sparse
yes
no
no

Sparse sparseOpenai

sparse
yes
no
no

Routing roy2020efficient

sparse
yes
no
no

\hdashline[0.4pt/2pt]
BP-Transformer BPTransformer

sparse
yes
MT
no

Blockwise blockbert

sparse
no
QA
yes

\hdashline[0.4pt/2pt]
Our Longformer
sparse
yes
multiple
yes

(a) Full n2superscript𝑛2n^{2} attention

(b) Sliding window attention

(c) Dilated sliding window

(d) Global+sliding window

## 2 Related Work

##### Long-Document Transformers

Tab. 1 summarizes recent prior work on long documents. Two types of self-attention approaches have been explored.
The first is a left-to-right (ltr) approach that processes the document in chunks moving from left-to-right.
While such models have been successful in autoregressive language modeling, they are unsuitable for transfer learning approaches with tasks that benefit from bidirectional context.

Our work falls within the other general approach that defines some form of sparse attention pattern and avoids computing
the full quadratic attention matrix multiplication.
The model with the most similar attention pattern to ours is Sparse Transformer sparseOpenai, which uses a form of dilated sliding window of blocks of size 8x8 provided by BlockSparse blocksparse.
Our implementation (§3) also includes a custom CUDA kernel, but it is more flexible and maintainable than BlockSparse which is implemented in C++, and designed for a specific version of TensorFlow.
We also introduce additional task motivated global attention patterns suitable for common NLP tasks
(§3) and show they are essential for good performance in the transfer learning setting.

A few models tried tasks other than autoregressive language modeling,
which is
a step forward because arguably focusing on
language modeling as the primary evaluation
has led to the development of models with limited applicability. BP-Transformer BPTransformer evaluated on
machine translation (MT), but didn’t explore the pretrain-finetune setting.
Blockwise attention blockbert pretrained their models and evaluated
on question answering (QA). However, the evaluation is
limited as it doesn’t include language modeling, and the QA datasets
are of relatively short documents,222
SQuAD contexts typically fit within the 512 limit,
and MRQA is constructed by dropping long-document examples.
 therefore the effectiveness of this model on long document tasks remains unexplored.

##### Task-specific Models for Long Documents

Many task-specific approaches have been developed to
workaround the 512 limit of pretrained transformer models like BERT.
The simplest approach just truncates the document, commonly used for classification truncateimdb.
Another approach chunks the document into chunks of
length 512 (could be overlapping), processes each chunk separately, then combines the activations with a task specific model joshi-etal-2019-bert.
A third approach popular for multihop and open domain QA tasks uses a two-stage model where the first stage retrieves relevant documents that are passed onto the second stage for answer extraction Clark2017SimpleAE; Chen2017ReadingWT.
All of these approaches suffer from information loss due to truncation or cascading errors from the two stage approach.
In contrast, Longformer can process long sequences without truncating or chunking, allowing us to adopt a much simpler approach that concatenates the available context and processes it in a single pass.

A few contemporaneous works333All were published on arXiv after Longformer. have explored similar ideas to Longformer using local + global attention in Transformers, and pre-training it for long document natural language tasks. In particular, ETC ainslie-etal-2020-etc uses a similar local + global attention instead of full self-attention to scale Transformers to long documents. Different from Longformer, ETC uses relative position embeddings (which we only used for the Autoregressive LM setting), introduces an additional training objective (CPC loss) for pre-training, and configures global attention in a slightly different way. It shows strong results on several tasks including reading comprehension and classification.
GMAT Gupta2020GMATGM uses a similar idea of few global locations in the input serving as global memory. BigBird Zaheer2020BigBT is an extension over ETC with evaluation on additional tasks, including summarization. Importantly, through theoretical analysis, BigBird shows that sparse Transformers are universal approximators of sequence functions and preserve these properties of the full self-attention.

## 3 Longformer

The original Transformer model has a self-attention component with O​(n2)𝑂superscript𝑛2O(n^{2}) time and memory complexity where n𝑛n is the input sequence length. To address this challenge, we sparsify the full self-attention matrix according to
an “attention pattern” specifying pairs of input locations attending to one another.
Unlike the full self-attention, our proposed attention pattern scales linearly with the input sequence, making it efficient for longer sequences.
This section discusses the design and implementation of this attention pattern.

### 3.1 Attention Pattern

##### Sliding Window

Given the importance of local context Kovaleva2019RevealingTD, our attention pattern employs a fixed-size window attention
surrounding each token.
Using multiple stacked layers of such windowed attention results in a large receptive field, where top layers have access to all input locations and have the capacity to build representations that incorporate information across the entire input, similar to CNNs Wu2019PayLA.
Given a fixed window size w𝑤w, each token attends to 12​w12𝑤\frac{1}{2}w tokens on each side (Fig. 2(b)).
The computation complexity of this pattern is O​(n×w)𝑂𝑛𝑤O(n\times w), which scales linearly with input sequence length n𝑛n. In a transformer with ℓℓ\ell layers, the receptive field size at the top layer is ℓ×wℓ𝑤\ell\times w (assuming w𝑤w is fixed for all layers).
Depending on the application, it might be helpful to use different values of w𝑤w for each layer to balance between efficiency and model representation capacity (§4.1).

##### Dilated Sliding Window

To further increase the receptive field without increasing computation, the sliding window can be “dilated”. This is analogous to dilated CNNs Oord2016WaveNetAG where the window has gaps of size dilation d𝑑d (Fig. 2(c)).
Assuming a fixed d𝑑d and w𝑤w for all layers, the receptive field
is ℓ×d×wℓ𝑑𝑤\ell\times d\times w, which can reach tens of thousands
of tokens even for small values of d𝑑d.

In multi-headed attention, each attention head computes a different attention score. We found settings with different dilation configurations per head improves performance by allowing some heads without dilation to focus on local context, while others with dilation focus on longer context.

##### Global Attention

In state-of-the-art BERT-style models for natural language tasks, the optimal input representation differs from language modeling and varies by task.
For masked language modeling (MLM), the model uses local context to predict the masked word, while for classification, the model aggregates the representation of the whole sequence into a special token ([CLS] in case of BERT).
For QA, the question and document are concatenated, allowing the model to compare the question with the document through self-attention.

In our case, the windowed and dilated attention are not flexible enough to learn task-specific representations.
Accordingly, we add “global attention” on few pre-selected input locations.
Importantly, we make this attention operation symmetric: that is, a token
with a global attention attends to all tokens across the sequence, and all tokens in the sequence attend to it.
Fig. 2(d) shows an example of a sliding window attention
with global attention at a few tokens at custom locations. For example for classification, global attention is used for the [CLS] token while in QA global attention is provided on all question tokens.
Since the number of such tokens is small relative to and independent of n𝑛n
the complexity of the combined local and global attention is still O​(n)𝑂𝑛O(n).
While specifying global attention is task specific, it is a easy way to add inductive bias to the model’s attention, and it is much simpler than existing task specific approaches that use complex architecture to combine information across smaller input chunks.

##### Linear Projections for Global Attention

Recall that given the linear projections Q𝑄Q, K𝐾K, V𝑉V, the Transformer
model Vaswani2017AttentionIA computes attention scores as follows:

Attention​(Q,K,V)=softmax⁡(Q​KTdk)​VAttention𝑄𝐾𝑉softmax𝑄superscript𝐾𝑇subscript𝑑𝑘𝑉\displaystyle\text{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^{T}}{\sqrt{d_{k}}}\right)V

(1)

We use two sets of projections, Qssubscript𝑄𝑠Q_{s}, Kssubscript𝐾𝑠K_{s}, Vssubscript𝑉𝑠V_{s} to compute attention scores of sliding window attention, and Qgsubscript𝑄𝑔Q_{g}, Kgsubscript𝐾𝑔K_{g}, Vgsubscript𝑉𝑔V_{g} to compute attention scores for the global attention.
The additional projections provide flexibility to model the different types of attention, which we show is critical for best performance on downstream tasks.
Qgsubscript𝑄𝑔Q_{g}, Kgsubscript𝐾𝑔K_{g}, Vgsubscript𝑉𝑔V_{g} are all initialized with values that match
Qssubscript𝑄𝑠Q_{s}, Kssubscript𝐾𝑠K_{s}, Vssubscript𝑉𝑠V_{s}.

### 3.2 Implementation

In regular transformers, attention scores are computed as in Eqn. 1.
The expensive operation is the matrix multiplication Q​KT𝑄superscript𝐾𝑇QK^{T} because both
Q𝑄Q and K𝐾K have n𝑛n (sequence length) projections.
For Longformer, the dilated sliding window attention computes only a fixed number of the diagonals
of Q​KT𝑄superscript𝐾𝑇QK^{T}. As shown in Fig. 1, this results in a linear increase
in memory usage compared to quadratic increase for full self-attention.
However, implementing it requires a form of banded matrix multiplication that is not supported in
existing deep learning libraries like PyTorch/Tensorflow.
Fig. 1 compares the performance of three different ways of implementing it:
loop is a memory efficient PyTorch implementation that supports dilation but is unusably slow and only used for testing;
chunks only supports the non-dilated case and is used for
the pretraining/finetuning setting;
and cuda is our fully functioning highly optimized custom CUDA kernel implemented using TVM tvm and used for the language modeling experiments (see Appendix LABEL:sec:tvm_details for more details).

## 4 Autoregressive Language Modeling

Autoregressive or left-to-right language modeling is loosely defined as estimating the probability distribution of an existing token/character given its previous tokens/characters in an input sequence.
This task is considered one of the fundamental tasks in natural language and recent prior work on modeling long sequences using transformers
has relied on this task as their primary evaluation transformerxl; compressive; adaptivespan.
Similarly, we develop and evaluate our model on autoregressive language modeling.

### 4.1 Attention Pattern

For autoregressive language modeling we use our dilated
sliding window attention.
Following adaptivespan we use differing window sizes across the layers. In particular, we use small window sizes for the lower layers and increase window sizes
as we move to higher layers. This allows the top layers to learn higher-level representation of the entire sequence while having the lower layers capture local information. In addition, it provides balance between efficiency (smaller window sizes are less computationally expensive due to fewer nonzero values)
and performance (larger window sizes have richer representation power and often result in performance improvements).

We do not use dilated sliding windows for lower layers to maximize their capacity to learn and utilize the immediate local context.
For the higher layers, we use a small amount of increasing dilation only on 2 heads.
This gives the model the ability to directly attend to distant tokens without sacrificing local context.

### 4.2 Experiment Setup

To compare to prior work we focus on character-level LM (text8 and enwik8; text8).

##### Training

Ideally, we would like to train our model on the largest window size and sequence length we can fit in a modern GPU memory. However, we found that
the model needs a large number of gradient updates to learn the local context
first, before learning to utilize longer context.
To accommodate this, we adopt a staged training procedure where we increase the attention window size and sequence length across multiple training phases. In particular, in the first phase we start with a short sequence length and window size, then on each subsequent phase, we double the window size and the sequence length, and halve the learning rate.
This makes training fast, while keeping the slow part
(longest sequences and window sizes) to the end.
We train the model over 5 total phases with starting sequence length of 2,048 and ending sequence length of 23,040
on the last phase (see Appendix B for detailed configurations of each phase, and for all other hyperparameters).

##### Evaluation

We evaluate with sequences of length 32,256. Following transformerxl, we split the dataset into overlapping sequences
of size 32,256 with a step of size 512, and report the performance on the last 512 tokens on the sequence.

#### 4.2.1 Results

Tab. 4.2.1 and 3 summarize evaluation results on text8 and enwik8 datasets.
We achieve a new state-of-the-art on both text8 and enwik8 using the small models with BPC of 1.10 and 1.00 on text8 and enwik8 respectively, demonstrating the effectiveness of our model.

For large models, given how expensive these experiments are,
and following recent work reformer; compressive,
we are only evaluating on enwik8.
Tab. 3 shows that Longformer outperforms the comparable Transformer-XL model,
matches the performance of the comparable Sparse Transformer sparseOpenai,
and matches or slightly underperforms recent models
that have more than twice the number of parameters.
It is worth noting that Adaptive Span adaptivespan
and Compressive Transformer compressive
are not good fit for the pretraining-finetuning paradigm as discussed in
§2.

Model
#Param
Dev
Test

Dataset text8

T12 AlRfou2018CharacterLevelLM

44M
-
1.18

Adaptive adaptivespan

38M
1.05
1.11

BP-Transformer BPTransformer

39M
-
1.11

Our Longformer
41M
1.04
1.10

\hdashline[0.2pt/0.2pt]
Dataset enwik8

T12 AlRfou2018CharacterLevelLM

44M
-
1.11

Transformer-XL transformerxl

41M
-
1.06

Reformer reformer

-
-
1.05

Adaptive adaptivespan

39M
1.04
1.02

BP-Transformer BPTransformer

38M
-
1.02

Our Longformer
41M
1.02
1.00

Model
#Param
Test BPC

Transformer-XL (18 layers)
88M
1.03

Sparse sparseOpenai

≈\approx100M
0.99

Transformer-XL (24 layers)
277M
0.99

Adaptive adaptivespan

209M
0.98

Compressive compressive

277M
0.97

Routing roy2020efficient

≈\approx223M
0.99

Our Longformer
102M
0.99

#### 4.2.2 Ablation Study

Model
Dev BPC

Decreasing w𝑤w (from 512 to 32)
1.24

Fixed w𝑤w (= 230)
1.23

Increasing w𝑤w (from 32 to 512)
1.21

No Dilation
1.21

Dilation on 2 heads
1.20

To show the importance of the design choices of our attention patterns, we tried different variants and report their controlled experiment results.
To make the ablation study more manageable, we train each configuration for 150K steps444
One caveat is that the ordering of end performance will not agree with that at step 150K.
However, this approximation saves the huge cost of running every experiment to completion.
with phase 1 configuration on a small model on text8, then report the BPC performance on the dev set.

The top of Tab. 4 demonstrates the impact of different ways of configuring the window sizes per layer. We observe that increasing the window size from the bottom
to the top layer leads to the best performance, arranging them in the reverse way leads to worse performance,
and using a fixed window size (the average of window sizes of the other configuration) leads to a performance that it is in between.
The bottom of Tab. 4 shows the impact of adding dilation.
Adding some dilation to two heads leads to some improvement compared with
no dilation at all.

## 5 Pretraining and Finetuning

Current state-of-the-art systems for many NLP tasks finetune a pretrained model with task supervision (e.g. BERT).
One of our main motivations is to develop such a model suitable for long document tasks.
To do so, we pretrained Longformer on a document corpus and finetune it for six tasks, including classification, QA and coreference resolution.
The resulting model can process sequences up to 4,096 tokens long (8 times longer than BERT)555Sequences up to 16K are possible on current GPUs..

We pretrain Longformer with masked language modeling (MLM), where the goal is to recover randomly masked tokens in a sequence.
Since MLM pretraining is expensive, we continue pretraining from the RoBERTa roberta released checkpoint, while only making the minimal changes necessary to support Longformer’s attention mechanism.
Note that our attention pattern can be plugged into any pretrained transformer model without the need to change the model architecture.

##### Attention Pattern

We use sliding window attention with window size of 512, therefore using the same amount of computation as RoBERTa.666Adding dilation on a few heads as in §4.1 hurt performance, likely because it is not compatible with the pretrained RoBERTa weights. Retraining such model from scratch might be needed to improve performance.

##### Position Embeddings

RoBERTa uses learned absolute position embeddings with the maximum
position being 512. To support longer documents, we add extra position embeddings to support up to position 4,096.
To leverage RoBERTa’s pretrained weights, instead of randomly initializing the new position embeddings, we initialize them by copying the 512 position
embeddings from RoBERTa multiple times as analysis of BERT’s attention heads shows a strong learned bias to attending to local context, including the previous or next token Clark2019WhatDB. Using the copy initialization preserves this local structure everywhere except at the partition boundaries.
Despite its simplicity, we found this to be a very effective (see Tab. 5), allowing Longformer pretraining to rapidly converge with a small number of gradient updates.

Model
base
large

RoBERTa (seqlen: 512)
1.846
1.496

Longformer (seqlen: 4,096)
10.299
8.738

+ copy position embeddings
1.957
1.597

+ 2K gradient updates
1.753
1.414

+ 65K gradient updates
1.705
1.358

Longformer (train extra pos. embed. only)
1.850
1.504

##### Continued MLM Pretraining

We pretrain Longformer using fairseq ott2019fairseq on a corpus of long documents that we compiled (see Appendix C for corpus details).
We train two model sizes, a base model and a large model.
Both models are trained for 65K gradient updates with sequences length 4,096, batch size 64 (218superscript2182^{18} tokens), maximum learning rate of 3e-5, linear warmup of 500 steps, followed by a power 3 polynomial decay. The rest of the hyperparameters are the same as RoBERTa.

Wordpieces
WH
TQA
HQA
ON
IMDB
HY

avg.
1,535
6,589
1,316
506
300
705

95th pctl.
3,627
17,126
1,889
1,147
705
1,975

QA
Coref.
Classification

Model
WikiHop
TriviaQA
HotpotQA
OntoNotes
IMDB
Hyperpartisan

RoBERTa-base
72.4
74.3
63.5
78.4
95.3
87.4

Longformer-base
75.0
75.2
64.4
78.6
95.7
94.8

Tab. 5 shows the BPC on the development
set of our training corpus. The first row shows a 1.846 BPC using
RoBERTa-base, which is comparable to the 1.880 BPC reported
on the RoBERTa paper on their corpus. This indicates
our training corpus is from a distribution close to that used to train RoBERTa.
The following two rows show the performance of Longformer before pretraining
with randomly initialized position embeddings and with
copied position embeddings. The significant difference indicates the importance of the copy initialization, and the relative small difference between the RoBERTa BPC and the initialized BPC indicates that our sliding window attention is working well with the RoBERTa weights.
The following two rows show the impact of continuing pretraining.
Traininig for 2K steps improves BPC from 1.957 to 1.753, which further decreases to 1.705 after 65K steps, demonstrating the model is learning to better utilize the sliding window attention and longer context.
Similar patterns are observed with RoBERTa-large and Longformer-large.

##### Frozen RoBERTa Weights

We also pretrained Longformer while freezing all RoBERTa weights, and only training the new position embeddings. The motivation for this configuration is to perfectly preserve
the RoBERTa performance on short documents.
This configuration has a BPC of 1.850 (down from 1.957 at initialization), but higher than 1.705 where all the weights
are trainable.

## 6 Tasks

We apply Longformer to multiple long document tasks, including QA, coreference resolution and classification. Tab. 6 shows the evaluation datasets have contexts significantly longer than 512 wordpieces.
Our primary goal is to evaluate whether our attention mechanism can act as a replacement for the standard self-attention mechanism in BERT style models, and to perform controlled trials against a strong baseline.
We are also interested in evaluating whether we can replace complicated task specific models necessitated by BERT’s limited context with simpler models that just concatenate all available context into a single sequence.

Our baseline is a RoBERTa based model that breaks the context into the longest possible segment, passes each individually through RoBERTa, and concatenates the activations for further processing.
For QA tasks, we also concatenate the question to each segment so that RoBERTa can condition it’s contextual representations of the context on the question.
The Longformer variant replaces the RoBERTa self-attention mechanism with our windowed attention used during pretraining, plus a task motivated global attention. The global attention uses additional linear projections (§3.1).

### 6.1 Question answering

We used three datasets: WikiHop Welbl2018ConstructingDF-Wikihop, TriviaQA (Joshi2017TriviaQAAL, Wikipedia setting), and HotpotQA, (Yang2018-HotpotQAAD, distractor setting).777We use the full version of TriviaQA and HotpotQA, not the simplified versions in MRQA mrqa.

For WikiHop and TriviaQA we follow the simple QA model of BERT bert, and concatenate question and documents into one long sequence, run it through Longformer, then have a dataset-specific prediction layer. WikiHop uses a classification layer for the candidate while TriviaQA uses the loss function of Clark2017SimpleAE to predict answer span. We include global attention to question tokens and answer candidates for WikiHop and to question tokens for TriviaQA.

HotpotQA is a multihop QA dataset that involves extracting answer spans and evidence sentences from 10 Wikipedia paragraphs, 2 of which are relevant and the rest are distractors. We use a two-stage model
that first selects the most relevant paragraphs then passes them to a second stage for answer extraction. Both stages concatenate question and context into one sequence, run it through Longformer, then use task-specific prediction layers.
We train the models in a multi-task way to predict relevant paragraphs, evidence sentences, answer spans and question types (yes/no/span) jointly.
Note that this model is simpler than recent SOTA models that include complex task-specific architectures (e.g., Tu2019SelectAA; Chen2019MultihopQA; Tu2020GraphSN; quark2020).
See Appendix D for further details about the models and hyperparameters.

### 6.2 Coreference Resolution

We use OntoNotes pradhan-etal-2012-conll, and the model from joshi-etal-2019-bert, a modification of the system from lee-etal-2018-higher to replace ELMo with BERT.
The Longformer system is a straightforward adaption of the baseline model by replacing RoBERTa with Longformer and extending the sequence length.
We didn’t use global attention for this task.

### 6.3 Document Classification

We evaluate on IMDB imdb and Hyperpartisan news detection hyperpartisan datasets.888For Hyperpartisan we split the training data into 80/10/10 train/dev/test sets, and report mean F1 across five seeds. IMDB is a standard sentiment classification datasets consisting of movie reviews. While most documents in this dataset are short, about 13.6% of them are larger than 512 wordpieces (Tab. 6).
Documents in Hyperpartisan are relatively long, and it is small with only 645 documents making it a good test for Longformer’s ability to adapt to limited data. We use global attention
on the [CLS] token.

### 6.4 Results

##### Main Result

Tab. 7 summarizes the results of all our finetuning experiments. We observe that Longformer consistently outperforms the RoBERTa baseline.
Its performance gain is especially obvious for tasks that require long context such as WikiHop and Hyperpartisan.
For TriviaQA, the improvement is more modest as the local context is often sufficient to answer the question.
In the case of HotpotQA, the supporting fact auxiliary supervision allows models to easily find relevant contexts and then focus on local context, leading to smaller gains. This is contrasted with WikiHop that only includes distant supervision of intermediate reasoning chains, where our approach excels by reasoning over the entire context.
On the IMDB and OntoNotes datasets the performance gains are smaller. For IMDB, the majority of the dataset consists of short documents and thus it is expected to see smaller improvements.
For OntoNotes, we found that the distance between any two mentions is typically quite small so that a baseline that processes smaller chunks separately is able to stitch together mentions into coreference chains without considering cross chunk interactions.

##### Longformer-large for QA

We also evaluate the performance of Longformer-large on long context QA tasks. Tab. 8
shows that our Longformer-large achieves new state-of-the-art
results999At submission time, May 2020. Later, BigBird Zaheer2020BigBT improved leaderboard results on these datasets. There are confounding factors such as using 16X more compute in BigBird’s pretraining compared with Longformer, potentially affecting the performance. on WikiHop and TriviaQA by large margins (3.6 and 4 points respectively), and for HotpotQA, it underperforms the current state-of-the-art hotpotqasota by a point.
Tab. LABEL:tab:hotpotqa shows the detailed results of HotpotQA compared with published and unpublished concurrent models. Longformer places second on the published leaderboard, outperforming all other published results except for HGN hotpotqasota. All published top performing models in this task Tu2019SelectAA; hotpotqasota; Shao2020IsGS use GNNs kipf2017semi or graph network of entities, which seem to encode an important inductive bias for the task and can potentially improve our results further.
Nevertheless, Longformer performs strongly outperforming all other methods including the recent non-GNN methods Gla2019SpanSP; Shao2020IsGS; quark2020.

Model
WikiHop
TriviaQA
HotpotQA

Current∗ SOTA
78.3
73.3
74.2

Longformer-large
81.9
77.3
73.2

Longformer-loop is a naive implementation that computes each diagonal separately in a loop.
It is memory efficient because it only computes the non-zero values, but
it is unusably slow. We only use it for testing because it is easy to implement but don’t use
it to run experiments. 
Longformer-chunks only supports the non-dilated case. It chunks Q𝑄Q and K𝐾K into
overlapping blocks of size w𝑤w and overlap of size 12​w12𝑤\frac{1}{2}w, multiplies the blocks,
then mask out the diagonals. This is very compute efficient because it uses
a single matrix multiplication operation from PyTorch, but it consumes 2x the amount of memory
a perfectly optimized implementation should consume because it computes some of the zero values.
Because of the compute efficiency, this implementation is most suitable for the
pretrain/finetune case. We didn’t find the increase in memory to be a problem for this setting. 
Longformer-cuda is a custom CUDA kernel that we implement using
TVM tvm. It is a fully functioning implementation of our attention (not limited as Longformer-chunks), it is the most memory efficient,
and it is as fast as the highly optimized full self-attention.101010It is worth noting that theoretically, a perfectly optimized Longformer-cuda should be faster than the n2superscript𝑛2n^{2} computation.
However, achieving this level of performance requires special knowledge of low-level GPU programming, similar to implementing a highly optimized matrix multiplication. Our current implementation is sufficiently fast and practical to use. We mainly use this implementation for the
autoregressive language modeling experiments because of the memory efficiency (allows the longest
sequences) and the support of dilation (needed for character-LM experiments).

##### Tensor Virtual Machine (TVM)

We build our custom CUDA kernel using TVM tvm, a deep learning compiler stack that compiles
high level description of a function into optimized device-specific code.
Using TVM, we describe our banded matrix multiplication in
high-level python constructs, then TVM generates the corresponding CUDA code
and compiles it for GPUs.

## Appendix B Character LM Hyperparameters

We evaluate on text8 and enwik8, both contain 100M
characters from Wikipedia split into 90M, 5M, 5M for train, dev, test.
Our model only specifies how the self-attention component works, and it is
agnostic to the other design choices for the transformer model.
Our implementation is based on the Transformer-XL transformerxl
code111111https://github.com/kimiyoung/transformer-xl
with the memory mechanism disabled.
We use relative position embeddings with sinusoidal weights as in transformerxl.
We use two different model sizes, a small (12 layers, 512 hidden size) model as in transformerxl,
and a large (30 layers, 512 hidden size) model as in sparseOpenai.
We employed mixed precision training (floating points 16 and 32) using apex121212https://github.com/NVIDIA/apex to reduce memory consumption and speed-up training. However, we kept the attention computation in fp32
to avoid numerical instability issues.131313We found that using fp16 in attention operation results in floating point overflow and NaNs in later stages of training.
We used gradient checkpointing gradckpt to reduce memory usage, and ran our experiments on 48GB RTX8000 GPUs.
All hyperparameters and stage configurations are listed in Tab. 12.
Our CUDA kernel supports the autoregressive mode where each token
attends to a window of previous tokens only. Our implementation
also includes a version of the relative position embedding
that is compatible with our dilated sliding window attention.

We ran the small model experiments on 4 RTX8000 GPUs for 16 days.
For the large model, we ran experiments on 8 RTX8000 GPUs for
13 days.
Most of our hyperparameter search is similar to the ablation in Tab. 4 where we run the configuration for 150K steps
on text8.
We experimented with absolute position embeddings and learned position embeddings,
dropout values of [0.1, 0.2] (small model) and [0.1, 0.4] (large model),
pre-layernorm and post-layernorm layernorm, learning rate (LR) of phase1 of values
[2.5e-5, 5e-4, 1e-4]
constant and cosine LR schedules,
and different configurations for dilation (on all heads, on 2 heads, no dilation).
Number of gradient updates/phase
reported in Tab. 12 is determined by running each phase until the validation BPC stops getting better.

Param
Value

Position Embeddings
Relative and Sinusoidal as in transformerxl

Small model config
12 layers, 8 heads, 512 hidden size as in transformerxl

Large model config
30 layers, 8 heads, 512 hidden size as in sparseOpenai

Optimizer
AdamW

Dropout
0.2 (small model), 0.4 (large model)

Gradient clipping
0.25

Weight Decay
0.01

Layernorm Location
pre-layernorm layernorm

Activation
GeLU

Number of phases
5

Phase 1 window sizes
32 (bottom layer) - 8,192 (top layer)

Phase 5 window sizes
512 (bottom layer) - (top layer)

Phase 1 sequence length
2,048

Phase 5 sequence length
23,040 (gpu memory limit)

Phase 1 LR
0.00025

Phase 5 LR
000015625

Batch size per phase
32, 32, 16, 16, 16

#Steps per phase (small)
430K, 50k, 50k, 35k, 5k

#Steps per phase (large)
350K, 25k, 10k, 5k, 5k

Warmup
10% of the phase steps with maximum 10K steps

LR scheduler
constant throughout each phase

Dilation (small model)
0 (layers 0-5), 1 (layers 6-7), 2 (layers 8-9), 3 (layers 10-11)

Dilation (large model)
0 (layers 0-14), 1 (layers 15-19), 2 (layers 20-24), 3 (layers 25-29)

Dilation heads
2 heads only

## Appendix C Pretraining Data

Source
Tokens
Avg doc len

Books Zhu2015AligningBA

0.5B
95.9K

English Wikipedia
2.1B
506

Realnews Zellers2019DefendingAN

1.8B
1.7K

Stories Trinh2018ASM

2.1B
7.8K

In order to allow the model to learn long dependencies in pretraining, we compiled a corpus of long documents. Some of these data sources were also included in the original RoBERTa pretraining including the Books corpus Zhu2015AligningBA plus English Wikipedia. We additionally included one third of a subset of the Realnews dataset Zellers2019DefendingAN with documents longer than 1,200 tokens as well as one third of the Stories Trinh2018ASM corpus. Our goal was to include a mix of long and short documents to both allow the model to learn longer dependencies while not to forget information from the original RoBERTa pretraining. The statistics of the pretraining data is shown in Tab. 13.

## Appendix D Task specific model details

All the QA and classification models are implemented using PyTorch-Lightning141414https://github.com/PyTorchLightning/pytorch-lightning. We use the official train/dev/test splits of all datasets except for the Hyperpartisan news which we randomely split into 80/10/10 for train/dev/test.

##### WikiHop

Instances in WikiHop consist of: a question, answer candidates (ranging from two candidates to 79 candidates), supporting contexts (ranging from three paragraphs to 63 paragraphs), and the correct answer. The dataset does not provide any intermediate annotation for the multihop reasoning chains, requiring models to instead infer them from the indirect answer supervision.

To prepare the data for input to Longformer and RoBERTa, we first tokenize the question, answer candidates, and support contexts using RoBERTa’s wordpiece tokenizer. Then we concatenate the question and answer candidates with special tokens as [q] question [/q] [ent] candidate1 [/ent] ... [ent] candidateN [/ent]. The contexts are also concatenated using RoBERTa’s document delimiter tokens as separators: </s> context1 </s> ... </s> contextM </s>. The special tokens [q], [/q], [ent], [/ent] were added to the RoBERTa vocabulary and randomly initialized before task finetuning.

After preparing the input data, we compute activations from the top layer of each model as follows.
We take the question and answer candidates and concatenate them to as much context as possible up to the model sequence length (512 for RoBERTa, 4,096 for Longformer), run the sequence through the model, collect the output activations, and repeat until all of the context is exhausted (for all models except Longformer-large, where we just include the first 4,096 length sequence due to memory requirements). Then all activations for all chunks are concatenated into one long sequence. In the case of Longformer, we use global attention to the entire question and answer candidate sequence.

For prediction, we attach a linear layer to each [ent] that outputs a single logit, average over all logits for each candidate across the chunks, apply a softmax and use the cross entropy loss with the correct answer candidate.

Training used the Adam optimizer with linear warmup over 200 gradient updates to a maximum LR, and linear decay over the remainder of training. We used gradient accumulation to effective batch size of 32 instances, checking the development accuracy every 250 gradient updates and reported the maximum development accuracy. Other hyperparameters (dropout, weight decay) were identical to RoBERTa pretraining.

In general, we ran minimal hyperparameter trials, but for fair comparison between Longformer and RoBERTa ran an identical hyperparameter search with Longformer-base and RoBERTa-base. This consisted of a grid search of LR in [2e-5, 3e-5, 5e-5] and number epochs in [5, 10, 15]. The best Longformer-base configuration used lr=3e-5, 15 epochs. We ran two hyperparameter trials for Longformer-large, lr=3e-5 and number epochs in [5, 15] (the 5 epoch model had higher dev accuracy of 77.6, and was the single model submitted to the public leaderboard for test set evaluation). All models were trained on a single RTX8000 GPU, with Longformer-base taking about a day for 5 epochs.

##### TriviaQA

TriviaQA has more than 100K question, answer, document triplets for training.
Documents are Wikipedia articles, and answers are named entities
mentioned in the article. The span that answers the question is not annotated,
but it is found using simple text matching.

Similar to WikiHop, we tokenize the question and the document
using RoBERTa’s tokenizer, then form the input as [s] question [/s]
document [/s]. We truncate the document at 4,096 wordpiece to avoid
it being very slow. Afterwards, we get the activations from RoBERTa
and Longformer similar to WikiHop (discussed above).
We use global attention on all question tokens.

For prediction, we add one layer that predicts the beginning and end of
the answer span. Because of the distant supervision nature of the training data (no gold answer spans), we use the loss function of Clark2017SimpleAE
which works like an OR that the model only needs to get
one answer span right, not all of them.

Hyperparameters of the best configuration are listed in Tab. 14. All other hyperparameters are similar to RoBERTa’s. For hyperparameter search, we only tuned LR for the RoBERTa
baseline and tried rates [3e-5, 5e-5, 1e-4], then used the best, which is 3e-5,
for all subsequent experiments with no further tuning.
We trained the Longformer-large with the best configuration once and submitted its
output to the leaderboard.
We ran our experiments on 32GB V100 GPUs.
Small model takes 1 day to train on 4 GPUs, while large model takes
1 day on 8 GPUs.

##### HotpotQA

HotpotQA dataset involves answering questions from a set of 10 paragraphs from 10 different Wikipedia articles where 2 paragraphs are relevant to the question and the rest are distractors. It includes 2 tasks of answer span extraction and evidence sentence identification.
Our model for HotpotQA combines both answer span extraction and evidence extraction in one joint model.
We found a higher performance using a two-stage Longformer model with similar setup that first identifies relevant paragraphs and then does find the final answer span and evidence.151515The final dev performance of the two stage model improves over a single stage model by about 4.2 points on joint-F1 metric This is largely because removing the distracting paragraphs first reduces the noise for the final evidence and span detection as also found to be important by recent state-of-the-art methods in this dataset hotpotqasota.
Similar to Wikihop and TriviaQA, to prepare the data for input to Longformer, we concatenate question and then all the 10 paragraphs in one long context. We particularly use the following input format with special tokens: “[CLS] [q] question [/q] ⟨⟨\langlet⟩⟩\rangle title1subscripttitle1\texttt{title}_{\texttt{1}} ⟨⟨\langle/t⟩⟩\rangle sent1,1subscriptsent1,1\texttt{sent}_{\texttt{1,1}} [s] sent1,2subscriptsent1,2\texttt{sent}_{\texttt{1,2}} [s] ... ⟨⟨\langlet⟩⟩\rangle title2subscripttitle2\texttt{title}_{\texttt{2}} ⟨⟨\langle/t⟩⟩\rangle sent2,1subscriptsent2,1\texttt{sent}_{\texttt{2,1}} [s] sent2,2subscriptsent2,2\texttt{sent}_{\texttt{2,2}} [s] ...” where [q], [/q], ⟨⟨\langlet⟩⟩\rangle, ⟨⟨\langle/t⟩⟩\rangle, [s], [p] are special tokens representing, question start and end, paragraph title start and end, and sentence, respectively. The special tokens were added to the Longformer vocabulary and randomly initialized before task finetuning. For Longformer, we use global attention to question tokens, paragraph title start tokens as well as sentence tokens. The model includes additional feedforward layers on top of paragraph title start tokens for prediction of relevant paragraphs, as well as sentence tokens for predicting evidence sentences. After training the first stage model, we predict relevant paragraph scores for both training and development set. We then keep up to 5 paragraphs whose raw score is higher than a pre-specified threshold (-3.0), and remove the other paragraphs from the context. We then train the second stage model on the resulting shortened context. For answer span extraction we use BERT’s QA model bert with addition of a question type (yes/no/span) classification head over the first special token ([CLS]). For evidence extraction we apply 2 layer feedforward networks on top of the representations corresponding to sentence and paragraph tokens to get the corresponding evidence prediction scores and use binary cross entropy loss to train the model. At inference time for evidence extraction, we use a constrained decoding strategy similar to quark2020 that ensures that the evidence sentences come from exactly two paragraphs which is the setup of this dataset. We combine span, question classification, sentence, and paragraphs losses and train the model in a multitask way using linear combination of losses. Our experiments are done on RTX8000 GPUs and training each epoch takes approximately half a day on 4 GPUs.
We trained the model using Adam optimizer with linear warmup (1000 steps) and linear decay. We used minimal hyperparameter tuning using LRs of 3e-5 and 5e-5 and epochs of 3 to 7 and found the model with LR of 3e-5 and 5 epochs to work best. We conduct the same hyperparameter search for the RoBERTa baseline as well. The rest of hyperparameters are reported in Tab 14.

Param
WikiHop
TriviaQA
HotpotQA

Epochs
15
5
5

LR
3e-5
3e-5
5e-5

Warmup steps
200
1000
1000

Batch size
32
32
32

Optimizer
Adam
Adam
Adam

##### Coreference model details

The coreference model is a straightforward adaptation of the coarse-to-fine BERT based model from joshi-etal-2019-bert. After preprocessing each document with the RoBERTa wordpiece tokenizer, it splits each document into non-overlapping segments up to the maximum sequence length, then concatenates the activations for the coarse-to-fine clustering stage that forms coreference clusters.
The maximum sequence length was 384 for RoBERTa-base, chosen after three trials from [256, 384, 512] using the default hyperparameters in the original implementation.161616https://github.com/mandarjoshi90/coref For Longformer-base the sequence length was 4,096. Similar to the original implementation, different learning rates were used for the pretrained RoBERTa parameters and the randomly initialized task parameters. Using a larger learning rate in the task parameters allows the optimizer to adjust them farther from their randomly initialized values without destroying the information in the pretrained RoBERTa parameters.

Hyperparameter searches were minimal and consisted of grid searches of RoBERTa LR in [1e-5, 2e-5, 3e-5] and task LR in [1e-4, 2e-4, 3e-4] for both RoBERTa and Longformer for a fair comparison. The best configuration for Longformer-base was RoBERTa lr=1e-5, task lr=1e-4. All other hyperparameters were the same as in the original implementation. Training takes about 10 hours on a single GPU.

Our implementation is a superhack that involves PyTorch and Tensorflow sharing a single process and GPU. To avoid re-implementing the complicated coarse-to-fine logic from Tensorflow in PyTorch (that involves a highly optimized custom GPU kernel originally released by lee-etal-2018-higher), we devised a system where the lower transformer portion of the model passes activations and gradients back and forth between PyTorch and Tensorflow. The input tensors are first run through the transformer in PyTorch, the activations are collected from the top layer, transferred from GPU to CPU then from CPU to Tensorflow and back to GPU to run the coarse-to-fine clustering and compute the loss. Then gradients are back propogated in Tensorflow to the top of the transformer and the process reversed to transfer them to PyTorch for back propogation through the remainder of the model. Separate optimizers are maintained with identical LR schedules for parameter updates. The overhead in this approach is minimal compared to the overall cost of running the model.

##### Text classification

For classification, following BERT, we used a simple binary cross entropy loss on top of a first [CLS] token with addition of global attention to [CLS]. We used Adam optimizer with batch sizes of 32 and linear warmup and decay with warmup steps equal to 0.1 of the total training steps. For both IMDB and Hyperpartisan news we did grid search of LRs [3e-5, 5e-5] and epochs [10, 15, 20] and found the model with [3e-5] and epochs 15 to work best. Experiments were done on a single RTX8000 GPU.

Generated on Sat Mar 2 11:53:35 2024 by LaTeXML

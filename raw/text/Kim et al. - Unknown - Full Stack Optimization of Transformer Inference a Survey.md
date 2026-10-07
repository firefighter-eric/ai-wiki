# Kim et al. - Unknown - Full Stack Optimization of Transformer Inference a Survey

- Source PDF: `raw/pdf/Kim et al. - Unknown - Full Stack Optimization of Transformer Inference a Survey.pdf`
- Source SHA256: `9d8a1b48d0d4a3777dd4ba22271951c3e019e422a1716f49264258a727829419`
- Source URL: https://arxiv.org/abs/2302.14017
- Generated from: `scripts/extract_pdf_text.py`

- Extraction: `pymupdf-pages-v2` (sorted text, page anchors; tables/formulas/figures require review)

## Extracted Text

<a id="source-section-0"></a>

<a id="page-1"></a>

### PDF 第 1 页

Full Stack Optimization of Transformer Inference: a Survey

                Sehoon Kim∗              Coleman Hooper∗           Thanakul Wattanawong
                 sehoonkim@berkeley.edu              chooper@berkeley.edu                 j.wat@berkeley.edu
                UC Berkeley                 UC Berkeley                 UC Berkeley

              Minwoo Kang                Ruohan Yan                 Hasan Genc
              minwoo_kang@berkeley.edu               yrh@berkeley.edu                 hngenc@berkeley.edu
                UC Berkeley                 UC Berkeley                 UC Berkeley

                  Grace Dinh                    Qijing Huang                  Kurt Keutzer
                    dinh@berkeley.edu               jennyhuang@nvidia.com               keutzer@berkeley.edu
                UC Berkeley                      NVIDIA                   UC Berkeley2023
              Michael W. Mahoney           Yakun Sophia Shao             Amir Gholami
               mmahoney@stat.berkeley.edu             ysshao@berkeley.edu                amirgh@berkeley.edu
                     ICSI, LBNL, UC Berkeley              UC Berkeley                          ICSI, UC BerkeleyFeb
27  ABSTRACT                                     1  INTRODUCTION
                                                               Deep learning models have scaled up to billions of parameters
           Recent advances in state-of-the-art neural network architecture de-
                                                                      and billions of Multiply-Accumulate (MAC) operations during both
           sign have been moving toward Transformer models. These models
                                                                                     training and inference. As a result, there has been a growing interest
           achieve superior accuracy across a wide range of applications in
                                                                                     in computing these models efficiently and in deploying these com-
          computer vision, natural language processing, and speech recog-
                                                                          pute and memory-intensive workloads on resource-constrained
            nition. This trend has been consistent over the past several years
                                                                      edge devices. These edge devices have tight energy and memory
           since Transformer models were originally introduced. However,[cs.CL]                                                                                constraints, and the corresponding applications that leverage deep
           the amount of compute and bandwidth required for inference of
                                                                                 learning models also often have real-time latency constraints.
           recent Transformer models is growing at a significant rate, and
                                                               CPUs and GPUs are both commonly used in general-performance
            this has made their deployment in latency-sensitive applications
                                                                     computing platforms, and they have the advantage of being both
           challenging. As such, there has been an increased focus on mak-
                                                                               ubiquitous and capable of supporting a wide variety of workloads
           ing Transformer models more efficient, with methods that range
                                                                      and operations. However, this flexibility comes at a cost of reduced
          from changing the architecture design, all the way to developing
                                                                                            efficiency. Deep learning models are composed of a small number
           dedicated domain-specific accelerators. In this work, we survey
                                                                                      of distinct operations that are repeated millions or billions of times,
             different approaches for efficient Transformer inference, including:
                                                                   and therefore they often do not require a high level of flexibility.
                 (i) analysis and profiling of the bottlenecks in existing Transformer
                                                                                   Additionally, while modern CPUs and GPUs can perform several
           architectures and their similarities and differences with previous
                                                                                operations in parallel, they lack the ability to leverage the massive
           convolutional models; (ii) implications of Transformer architec-
                                                                               data reuse opportunities in deep learning models.
           ture on hardware, including the impact of non-linear operations
                                                                    The combination of a need for fast, efficient computation, the use
          such as Layer Normalization, Softmax, and GELU, as well as linear
                                                                                     of a small number of distinct operations, and the opportunities for
           operations, on hardware design; (iii) approaches for optimizing a
                                                                                data reuse have all led to the use of hardware accelerators for deep
            fixed Transformer architecture; (iv) challenges in finding the right
                                                                                       learning. A multitude of enterprise deep learning accelerators, such
         mapping and scheduling of operations for Transformer models;arXiv:2302.14017v1                                                                           as [1, 4, 62, 91, 100, 115, 134, 137, 171, 196, 208], have been developed
         and (v) approaches for optimizing Transformer models by adapt-
                                                                   and integrated into commodity hardware by industry in the past
           ing the architecture using neural architecture search. Finally, we
                                                                              decade. This parallels many research accelerators developed in
          perform a case study by applying the surveyed optimizations on
                                                                         academia [34, 37, 39, 40, 59, 69, 70, 81, 169]. Together with hardware
         Gemmini, the open-source, full-stack deep neural network accel-
                                                                                    accelerator development, the software frameworks [3, 32, 98, 167]
            erator generator, and we show how each of these approaches can
                                                                   and compilers [33, 161, 185] for deploying various deep learning
            yield improvements, compared to previous benchmark results on
                                                                            algorithms have also enhanced and matured. These tools enable
          Gemmini. Among other things, we find that a full-stack co-design
                                                                             the execution of deep learning algorithms on accelerators, and
          approach with the aforementioned methods can result in up to
                                                                            they perform mapping optimizations to improve the performance
           88.7× speedup with a minimal performance degradation for Trans-
                                                                   and efficiency of the full deep learning pipeline. Nonetheless, the
          former inference.
                                                                                   fast-evolving deep learning algorithms still keep introducing new
                                                                 demands for hardware and software support, as well as their co-
                                                                                  optimization, to satisfy various deployment constraints.

              ∗Equal contribution.
<a id="page-2"></a>

### PDF 第 2 页

The recent rise in popularity of Transformers and large language         addressed properly, this can result in less than 1% hardware
models [22, 44, 52, 58, 86, 173–175, 177, 190, 198] for solving various           utilization (Sec. 3.4 and Fig. 14).
natural language processing (NLP) tasks presents a brand new set       • For Transformer accelerators, it is often better to have a larger
of challenges in the design of accelerators as well as frameworks.         accumulator size and smaller scratchpad size, while the op-
There has also been an increased focus on making Transformer          posite is often more optimal for CNN accelerators. Changing
inference more efficient, especially due to their growing size and           accelerator architecture to incorporate this observation can re-
run-time complexity. However, there is still a lack of understanding           sult in a 36% latency improvement over the baseline optimized
regarding the workload characteristics of Transformer architec-          for CNN benchmarks (Sec. 3.4.3).
tures, and thus of the design principles necessary for effectively       • Despite the fact that scheduling matmuls in Transformers only
running these models, when compared to the more well-known          requires 3 loops, as compared to 6 for convolutions in CNNs,
convolutional neural network (CNN) architectures. For instance,       we found that it is as challenging to find performant schedules
compared to the conventional CNN-focused design, Transformers           for Transformers as it is for CNNs. The selection of appropriate
are mostly composed of matrix multiplications (matmuls) together          scheduling decisions for Transformers involves a large number
with memory-intensive nonlinear operations. In addition, the com-          of decisions, with the best and worst solutions exhibiting perfor-
putational graph and dataflow of Transformer models are more        mance differences of up to four orders of magnitude (Sec. 5.5.1
complex than that of CNNs, with more types of operation nodes,        and Fig. 18, 19, 20).
as well as more dataflow splits and concatenations. All these chal-      • Fusing Batch Normalization with the neighboring convolution
lenges require us to undertake a comprehensive analysis of the           in CNN models is straightforward. However, when fusing Layer
current hardware and software solutions as well as the various         Normalization with the preceding matmul in the Transformer
design trade-offs for Transformer inference. Performing such an           architecture, constraints are imposed on the mapping, partic-
analysis will enable us to build a holistic and comprehensive under-          ularly related to tile sizes. This requires further consideration
standing of the requirements for efficiently running Transformers.          since the runtime cost due to the mapping constraints could out-
  The contribution of this work is two-fold: (1) to analyze the         weigh the gains from operation fusion in certain circumstances
run-time characteristics of Transformers and to survey different           (Sec. 5.5.2 and Fig. 21, 22).
approaches for efficient Transformer inference; and (2) to perform
a case study by applying the surveyed methodologies on Gem-
mini [70], the full-stack deep neural network (DNN) accelerator    2 TRANSFORMER MODEL ARCHITECTURE
generator. The longer-term goal of this work is to characterize dif-     AND PERFORMANCE BOTTLENECKS
ferent factors across the hardware and software stack in order to
                                                                        In this section, we start with a high level introduction to the building
optimize Transformer inference.
                                                                    blocks of the Transformer architecture. We first discuss the multi-
   Regarding our first contribution, this paper contains a survey and
                                                              head attention and feed-forward modules, the non-linear operations
analysis covering different hierarchies in end-to-end deep learning
                                                                used in Transformers, and the difference between Encoder/Decoder
inference, with a particular focus on Transformers. This includes:
                                                                 models, in Sec. 2.1. We then analyze the impact of these different
 • Analysis and profiling of the runtime characteristics and bot-      blocks on hardware performance using arithmetic, as well as the
    tlenecks of the Transformer architecture (Sec. 2).                    analytical modeling and direct profiling of each component, in
 • Hardware architectures for Transformer inference, including       Sec. 2.2.
    the impact of the non-linear operations of the Transformer
    architecture on their design (Sec 3).                          2.1  High-Level Overview of Transformer
 • Optimization strategies such as pruning and quantization for                                                       Architecture
    further improving the performance of a fixed Transformer ar-
                                          A Transformer architecture [217] typically consists of multiple    chitecture (Sec 4).
 • Mapping and scheduling of operations in the Transformer ar-     Transformer blocks, each of which includes a multi-head attention
                                                     (MHA) module and a feed-forward (FFN) module, and each of which    chitecture and the associated challenges (Sec. 5).
                                                                                        is followed by a Layer Normalization (LayerNorm) operation and • Designing and adapting Transformer architectures to be more
                                                               a residual connection. The detailed computations of MHA and   hardware efficient through an automated neural architecture
                                                     FFN are illustrated in Fig. 1, and the configuration parameters for    search process (Sec. 6).
                                                              Transformer architectures (along with the values used by BERT-
  Regarding our second contribution, our case study of applying                                                                Base and BERT-Large) are provided in Tab. 1. An input sequence to
the surveyed methodologies on deploying Transformers yields sev-                                                                   the Transformer block is composed of 𝑙tokens, each represented
eral key findings, including the following:                                                         by a vector of 𝑑dimension, forming a 𝑑× 𝑙matrix. A token is a
 • Gemmini, which was originally designed for CNN workloads,     segment of an input sequence. For example, when the input is a
    does not yield hardware accelerator architectures that are well-      sentence, a token may be a word or a sentence fragment.
    suited for Transformer inference. The primary bottleneck for       The MHA module (see Fig. 1, Left) first projects this sequence by
   running Transformers on CNN domain-specific accelerators is      multiplying it with three different weight matrices: 𝑊𝑄, 𝑊𝐾, and
    not necessarily linear operations, but rather it is the time spent     𝑊𝑉(the so-called query, key, and value matrices). This yields three
   on floating-point non-linear operations, as well as quantization       different activations, namely the query, key, and value activations.
   and dequantization operations. Unless those operations are     The query, key, and value activations are then split into ℎchunks,




                                                                                          2
<a id="page-3"></a>

### PDF 第 3 页

WQ
                 𝑑x 𝑑
                                𝑑/ℎx 𝑙
                  𝑑x 𝑙
                  WK                             𝑙x 𝑑/ℎ                                 W1
                 𝑑x 𝑑                          Transpose
   Encoder 𝑑x 𝑙                    𝑑/ℎx 𝑙                                                       𝑑𝐹𝐹𝑁x 𝑑
                                                                                                                        GELU    Input            𝑑x 𝑙                               Softmax   𝑙x 𝑙                   AttentionOutput    𝑑x 𝑙
                  WV                                                                          𝑑𝐹𝐹𝑁x 𝑙            𝑑𝐹𝐹𝑁x 𝑙
                 𝑑x 𝑑                              𝑙x 𝑙
                                𝑑/ℎx 𝑙                   𝑑/ℎx 𝑙
                  𝑑x 𝑙
                                                                                      W2           Norm + Add
                                                                          𝑑x 𝑑𝐹𝐹𝑁
                                                                                               𝑑x 𝑙  Encoder                                                                               Concatenate                                                  LayerNorm
                                                                           𝑑x 𝑙                                 Output
                                    WOut          Norm + Add
                          𝑑x 𝑑
                                                                                   Attention
                                                     LayerNorm                                                                            Output         Feed-Forward Network (FFN) Module                    𝑑x 𝑙    𝑑x 𝑙                                              𝑑x 𝑙


              Muti-Head Attention (MHA) Module

Figure 1: Map of the computations performed in (Left) the multi-head attention (MHA) module and (Right) the feed-forward network (FFN)
module in the Transformer encoder block.


Table 1: Configuration parameters for Transformer architectures.     Table 2: Linear operations in Transformer models. The last column
Parameters for BERT-Base, BERT-Large, and GPT-2 (smallest) are        is the matrix multiplication dimensions, i.e., 𝑚× 𝑛× 𝑘means the
given as examples. Note that GPT-2 has the same parameters as BERT-      input dimensions of 𝑚× 𝑛and 𝑛× 𝑘, and the output dimension of
Base. Sequence length can be any number, as long as it doesn’t exceed   𝑚× 𝑘. Note that act-to-act matmuls are both repeated ℎtimes in the
the maximum possible sequence length.                               multi-headed scheme. The entire computation graphs of MHA and
                                                        FFN are illustrated in detail in Fig. 1.
   Symbol      Parameter      BERT-Base   BERT-Large  GPT-2
   𝑁          # Layers           12           24         12                     Module       operation      matmul dim
     𝑑     Model dimension       768         1024       768               MHA     𝑊𝑄projection    𝑑× 𝑑× 𝑙
     ℎ      # Attention Heads       12           16         12                                𝑊𝐾projection    𝑑× 𝑑× 𝑙
    𝑑FFN     FFN dimension       3072         4096       3072                               𝑊𝑉projection    𝑑× 𝑑× 𝑙
         𝑙       Sequence length           -                 -              -                                    query × key     𝑙× 𝑑/ℎ× 𝑙
                                                                                                                      attn. score × value  𝑑/ℎ× 𝑙× 𝑙
                                                                                          𝑊out projection   𝑑× 𝑑× 𝑙
with each chunk having a hidden dimension of 𝑑/ℎ. These chunks                                                                        FFN    𝑊1 projection     𝑑FFN × 𝑑× 𝑙
are then forwarded to ℎdifferent attention heads, where the query                                                                      𝑊2 projection   𝑑× 𝑑FFN × 𝑙
and key chunks are multiplied along the hidden dimension, gen-
erating an activation matrix of size 𝑙× 𝑙. This activation matrix is
then passed through the Softmax operation (the output of which
                                                         The FFN module (see Fig. 1, RIght) is a relatively simple block
is often referred to as an attention score) and multiplied with the
                                                                         consisting of two linear layers. The input sequence is first projected
value chunk, resulting in an activation of hidden dimension 𝑑/ℎ.                                                           from the hidden dimension 𝑑to a higher FFN dimension 𝑑FFN via
Subsequently, all of the activations from the attention heads are                                                                    the first linear layer with the weight matrix 𝑊1. Subsequently, the
concatenated along the hidden dimension to generate a single ac-                                                                    projected sequence is projected back to the original dimension 𝑑
tivation of hidden dimension 𝑑, which is then projected into the                                                                through the second linear layer with the weight matrix 𝑊2. Gener-
same dimension by the last linear layer with the weight matrix𝑊out.                                                                                          ally, the dimension 𝑑FFN is chosen to be 4× larger than 𝑑, resulting
Finally, the output from the last linear layer in the MHA module is                                                                       in the 4:1 aspect rate of 𝑊1 and 𝑊2 (e.g., in BERT-Base [52]). In
passed through the LayerNorm operator before being added to a
                                                           between these two linear layers is a non-linear layer. Typically,
residual connection to get the MHA module output.
                                                  GELU [85] is used for this [22, 52, 143, 173, 174]. Tab. 2 summarizes
   In summary, an MHA module consists of six linear operations,
                                                                                          all types of linear layers in a Transformer block in both MHA and
four of which are identical weight-to-activation matmuls (i.e., the
                                                     FFN modules.
𝑊𝑄,𝑊𝐾,𝑊𝑉and𝑊out projections), and the remaining two of which
are activation-to-activation matmuls (i.e., query × key and attention        2.1.1  Nonlinear Operations. There are several nonlinear oper-
score × value). Throughout this paper, we refer to the first type of       ations such as Softmax, LayerNorm, and GELU that require special-
matmuls as projections and the second type of matmuls as activation-      ized support or off-chip computation. These nonlinear operations
to-activation matmuls (act-to-act matmuls for short), as they have      account for a relatively smaller portion of the overall operations,
different run-time behaviors.                              when inferring with Transformer networks, as compared to the




                                                                                           3
<a id="page-4"></a>

### PDF 第 4 页

(a) Softmax                     Sequence Length    (b) Layer Normalization             Sequence Length       (c) Batch Normalization              Height, Width
                      𝑒𝑥𝑖                                      𝑝𝑖𝑛−𝜇𝑡                                             𝑝𝑖𝑛−𝜇𝑐
   𝑆𝑜𝑓𝑡𝑚𝑎𝑥(𝑥𝑖) =                                    𝑝𝑜𝑢𝑡=        𝛾𝑒+𝛽𝑒                                    𝑝𝑜𝑢𝑡=         𝛾𝑐+𝛽𝑐
                 σ𝑗𝑒𝑥𝑗              …                         𝜎𝑡                     …                           𝜎𝑐                   …
                                                                      γ, β                                                            learned                                                                    during                                                                                           Length                             …                                                                 …                                                            training                                                    and applied                                                                       along                                                                                                                                            γ, β learned  Sum        of          exponential                                                                                                                                                                                                                                                                                                                                                                                                                                         Dimension                                                      the                                                      sequence                                                                      length                                                                                                                              training                                                                                                                    per channelterms         is       computed                  along
                                …                                                                     …                      …  …                                                           …  …                                                                                                                             during              Channel    …  …  …  …  the sequence length                                                                                                                         Sequence                               μ, σ computed during                Hidden                                   μ, σ learned during
                             …            inference across the               …                training per channel              …
                                                     hidden dimension

Figure 2: Diagrams outlining the Softmax, LayerNorm, and BatchNorm operations. Since they rely on runtime statistics, LayerNorm and Softmax
both require multiple passes over the input in order to compute the nonlinear operation. In the case of Softmax, a first pass over the inputs is
required to compute the denominator. For LayerNorm, three passes are required over the inputs: one to compute the mean; one to compute the
standard deviation; and one to apply the normalization. Unlike LayerNorm and Softmax, BatchNorm only uses statistics which are learned during
training, and therefore it only requires one pass over the inputs.



                                                                                                                          Output (1 token / iter)

        Output (𝒍tokens)                   Output (1 token / iter)


                                                                                                                Decoder Block
        Encoder Block                     Decoder Block                             Encoded Sequence
                                                                                                   (𝒍𝒆tokens)            …
      …              …

                                                                                                                                          Feed-Forward Network
                                Ne
       Feed-Forward Network                                                 Feed-Forward Network        Nd                  Encoder Block                                            Nd

                                        …                Ne
                                                                                                                                                            Cross-Attention
             Self-Attention                                  Self-Attention                              Encoder Block

                                                                                                                                                                     Self-Attention

         Input (𝒍tokens)                     Input (1 token / iter)      x 𝒍iterations          Input (𝒍𝒆tokens)

   (a) Encoder-Only            (b) Decoder-Only                (c) Encoder-Decoder             Input (1 token / iter)      x 𝒍𝒅iterations

Figure 3: Variants of Transformer networks. (a) An encoder-only model, which performs inference for all tokens in parallel. (b) A decoder-only
model, which performs inference in an auto-regressive manner. (c) An encoder-decoder model, which uses the output of the encoded sequence as
input to a cross-attention module.


linear operations (Sec. 2.2.2). However, they are more challenging      the normalization is actually applied, one division per input value
to compute on typical hardware than matmuls, and they can incur        is required.
significant overhead if not handled appropriately.                       Furthermore, the nonlinear operations entail challenges in op-
  The nonlinear operations present challenges in terms of efficient      eration fusing, which is a common technique to reduce interlayer
utilization of temporal memory as well as efficient computation.     communications by combining multiple operations into a single
This is because they require multiple passes over all input values,      operation (Sec. 5.2.1). Unlike Batch Normalization (BatchNorm) in
which requires those values to be held in temporal memory. As     many CNN architectures that can be seamlessly subsumed into pre-
depicted in Fig. 2 (a), the Softmax operation involves (1) exponen-     ceding or succeeding linear operations [97], LayerNorm requires
tial operations, (2) summing up the results across the sequence     computing the mean and variance of the inputs at runtime. There-
length dimension, and (3) normalizing the input by dividing it by       fore, to fuse this operation with the preceding matmul operation,
the summation result. It is also well known that the exponential      the entire output matrix must be accumulated in place across the
function is prone to numerical overflow, prompting the use of the      reduction dimension (i.e., the dimension in which the mean and
maximum subtraction trick [151] that transforms the expression      variance are computed) before writing out results. This leads to
exp(𝑥𝑖)/Í 𝑗exp(𝑥𝑗) into exp(𝑥𝑖−𝑥max)/Í 𝑗exp(𝑥𝑗−𝑥max), where       irregular tiling dimensions and lower data reuse. As a result, there
𝑥max is the maximum of the 𝑥𝑗’s. This, however, requires an addi-       is a nontrivial tradeoffbetween fusing these operations with pre-
tional pass over the inputs, resulting in a three-pass numerically      vious layers versus using better tiling dimensions for maximizing
stable implementation. Computing the LayerNorm function also       reuse. A detailed analysis of this tradeoffwill be provided later in
requires multiple passes over the entire input values across the       Sec. 5.5.2.
hidden dimension, as illustrated in Fig. 2 (b). In the first pass, the
mean must be computed. In the second pass, this is then used to
compute the standard deviation. Finally, in the third pass, where





                                                                                          4
<a id="page-5"></a>

### PDF 第 5 页

2.1.2  Encoder and Decoder Architectures. The Transformer      Consequently, these operations scale linearly with sequence length,
architecture was originally introduced as an encoder-decoder model      implying that more compute is required to process a token in a
for machine translation tasks [217]. In this setting, the encoder       larger time step than a token in a smaller time step. A key detail to
takes the entire source language sentence as input and passes it      note is that the full key and value activations must be present for the
through multiple Transformer encoder blocks, extracting the high-      input token to attend to all previously generated tokens. A common
level features of the input sentence. These extracted features are      optimization technique for token generation is to cache and reuse
then fed into the decoder, which is responsible for generating the      the intermediate key and value of the previously generated tokens
tokens in the target language, one after another. This is based on       in subsequent iterations, thus avoiding the need to recompute them
the source language features from the encoder as well as the tokens       for every iteration. Taken together, the end-to-end complexity of
it has previously generated [217]. In subsequent works, encoder-      generating the full sequence scales linearly for the projection layers
only and decoder-only architectures were introduced, taking only     and quadratically for the other two act-to-act matmuls. The end-to-
the encoder and the decoder components, respectively, from the     end computation graph of the Transformer decoder block is also
original encoder-decoder architecture [46, 174] (Fig. 3).                provided in Fig. 27 of Appendix A.6.

  Encoder Block.  In encoder-only Transformer models [52, 143,
                                                 Summary (Sec. 2.1. Transformer Overview)
240], the input sequence is passed through the repeated encoder
                                                                    Transformers are composed of several Transformer blocks,
blocks all at once. For this reason, the encoder-only structure is
                                                                  each of which has an MHA (multi-head attention module
suitable for natural language understanding tasks [52, 143], such as
                                                             and an FFN (feed-forward network) module (along with
sentiment analysis [202] or sentence similarity analysis [28, 53, 96],
                                                           LayerNorm and residual addition after each module). The
where the entire input sequences are fed into the model.
                                          MHA module contains projection layers as well as act-to-act
   In the encoder block, the inference is composed of matrix-matrix
                                                               matmuls and Softmax operations. The FFN module consists
multiplications as well as element-wise additions and nonlinear
                                                                              of two projection layers with a nonlinear function between
operations. The cost of the projection layers in the MHA module
                                                                    them. There are two types of Transformer blocks: encoderand FFN module scales linearly with the input sequence length 𝑙.
                                                                        blocks and decoder blocks. Encoder blocks process the en-
However, the act-to-act matmuls in the MHA module scale quadrat-
                                                                                       tire input sequence in parallel, making them suitable forically with sequence length (as demonstrated in query × key and
                                                                          natural language understanding tasks. Decoder blocks areattn. score × value rows in Tab. 2). In Sec. 2.2.2, we demonstrate via
                                                                             autoregressive, meaning that inference must be performed
profiling that this depends on the sequence length: with short se-
                                                                  once per generated output token, and are therefore typically
quence lengths, the projection layers dominate, making the overall
                                                                  used in generative tasks.complexity of the encoder block 𝑂(𝑙); with long sequence lengths,
however, the act-to-act matmuls dominate, making the overall com-
plexity 𝑂(𝑙2).
                                                         2.2  Model Analysis
  Decoder Block. In contrast to encoder-only models, the decoder-                                                                                  2.2.1  Workload Analysis. In order to evaluate bottlenecks in
only models [22, 173, 174] that consist of repeated decoder blocks                                                                    Transformer, we first modelled the number of floating-point opera-
are auto-regressive in nature. This means that the output at a given                                                                        tions (FLOPs) required to compute the Transformer encoder-only
time step is based on the outputs in the previous time steps. In                                                            and decoder-only models, as well as the arithmetic intensity of these
other words, the model predicts a token in a sentence based on                                                                   networks. Arithmetic intensity is the number of floating point oper-
the previous tokens it has generated so far, and the inference must                                                                        ations that can be performed per byte loaded from memory. It can
therefore be performed sequentially and iteratively, once for each                                                             be computed by dividing the total number of FLOPs by the total
output token. For instance, if the previously generated sequence                                                      number of bytes accessed (also referred to as MOPs, or memory
is “I am a”, the model takes this as input and may predict the next                                                                      operations) [227]:
token “student”. Then, in the next time step, the input to the model
                                                                                                   # FLOPsbecomes “I am a student”. Therefore, the decoder-only structure                                                                                 Arithmetic Intensity =                                                                                                                                                                                                                                .                (1)
is suitable for natural language generation tasks. It is important                                         # MOPs
to note that, in decoder-only models, the input prompt tokens                                                                    Here, we are assuming that the local memories are large enough to
can be consumed in parallel before the model begins to generate                                                                 hold both matrices entirely in memory for a given operation, and
subsequent tokens. For this work, we only consider open-ended                                                                          that the computed arithmetic intensity values therefore serve as an
generation (i.e., assuming no input prompt).                                                            upper bound for the achievable data reuse. We are also counting
   Unlike the encoder block, which operates on the entire input                                                                   the multiplication and addition from a MAC operation separately
sequence, the decoder block is inferred one token at a time. This                                                    when computing FLOPs.
results in a sequence length of one for each time step. In the case of
the projection layers, each token is independent of the previously        End-to-end FLOPs and MOPs. For the encoder analysis, we
generated token. Thus, the projection operations are solely applied      used the 12-layer BERT-Base model and the 24-layer BERT-Large
to the input token, resulting in a matrix-vector multiplication and      network (see Tab. 1 for model configurations); and for the decoder,
a constant cost. However, this does not hold for the act-to-act     we used the 12-layer GPT-2 model architecture which has the same
matmuls, as the input token is not independent of the previously     model configuration parameters as BERT-Base. For the purposes
generated tokens. Instead, it is required to attend to all of them.      of analysis, we ignored the maximum input sequence lengths of




                                                                                          5
<a id="page-6"></a>

### PDF 第 6 页

FLOPs of Transformer Models                                MOPs of Transformer Models
                                                                                                                                                             507.8               BERT-Base                                                                                             BERT-Base                                                               3942              500   4000                BERT-Large                                                                                               BERT-Large
               GPT-2                                                                        GPT-2
                                                                           400
   3000
 GFLOPs 2000                                                                                                                GMOPs 300                                                    214.3
                                                     1554                                                           1324                 200
                                                                  1012
   1000                                                                                                                                  97.3                                           673                                                                           100
                                                  505   428                                                                                                                               46.2                                 310                                                                                                                                                               29.9                                        214                                              194                                    92           22 73 22   4614945   97                                                                                                                                           1.12.9      3.28.7     11.2                                                                                                       0.10.411.1  0.20.622.5  0.41.2
            128      256      512      1024     2048     4096                      128      256      512      1024     2048     4096
                           Sequence Length                                                      Sequence Length

Figure 4: Plot of the FLOPs for the BERT-Base and BERT-Large en-     Figure 5: Plot of the MOPs for the BERT-Base and BERT-Large en-
coders and the GPT-2 decoder across different sequence lengths. The      coders and the GPT-2 decoder across different sequence lengths. The
FLOPs scales quadratically with sequence length due to quadratic     MOPs scale quadratically with sequence length for the encoder-only
scaling in the act-to-act matmuls as well as the Softmax function.     models due to quadratic scaling in the act-to-act matmuls as well
Additionally, inferring the BERT-Base encoder and the GPT-2 decoder      as the Softmax function. Additionally, the GPT-2 decoder requires a
(which have the same model architecture) requires a similar number     much greater number of MOPs than the BERT-Base encoder (which
of FLOPs for processing the same sequence length.                     have the same model architecture) for processing the same sequence
                                                                       length as it loads weights per every token generation.

512 for standard BERT models throughout this section. We then                     Arithemtic Intensity of Transformer Models
computed MOPs, the number of bytes that had to be accessed, when
                                                                                                                                          BERT-Baseinferring these models. We assumed 8-bit precision for all oper-         300                                                                                                               266                        BERT-Large
ations, meaning that loading one parameter or activation would                                                                  GPT-2
                                                                                                     239     231          235
require loading one byte. For the decoder model, we measured                           215
                                                                                                                       202the FLOPs and MOPs as the total amount of floating point oper-                                                                           200                                                                                                                                   179                                                                                                                                                                                                                                Intensity                                                                                           171
ations and memory operations needed to iteratively generate the                 160                                      156
full sequence of the given length. The FLOPs and MOPs for these                                                                      131
                                                                                                                                           117
networks for a range of sequence lengths are plotted in Fig. 4 and 5,
                                                                           100respectively. As one can see, FLOPs and MOPs scale super-linearly                       Arithmetic
for all models, especially in the long sequence length regime, due
to the quadratic complexity with respect to sequence length in the                                                                                              2         2         2         2         2         2
act-to-act matmuls.                                                               128      256      512      1024     2048     4096
                                                                                               Sequence Length
  End-to-end Arithmetic Intensity. We then modeled the arith-
metic intensity by dividing the number of FLOPs required when      Figure 6: Plot of the arithmetic intensity of the BERT-Base and
inferring these models by the number of MOPs. The arithmetic      BERT-Large encoders and the GPT-2 decoder across different sequence
intensity for BERT-Base, BERT-Large, and GPT-2 versus sequence       lengths. The arithmetic intensity initially increases since the larger
length is shown in Fig. 6. For both BERT-Base and BERT-Large, the      matrix dimensions allow for more computations to be performed per
arithmetic intensity initially increases with sequence length until      parameter loaded. However, at higher sequence lengths the arithmetic
512 and then decreases afterwards for larger sequence lengths. The       intensity decreases. This is because, for the long sequence length, the
reason for this is that, as will be analyzed in more detail in Sec. 2.2.2,       act-to-act matmuls, and Softmax computations of the MHA module
the FFN module that has higher arithmetic intensity than the MHA      begin to dominate. These have relatively lower arithmetic intensity
module (Tab. 3) dominates the total FLOPs for small sequences      compared to the projection layers in the FFN module.
(Fig. 7). However, this trend reverses for larger sequence lengths, as
the cost of act-to-act matmuls in the MHA module grow quadrati-
cally with the increase in sequence length, leading to a reduction      loads cannot be shared across tokens. This leads to performing
in arithmetic intensity for the end-to-end model inference.            roughly 2 operations per parameter loaded. It is important to note
   In comparison to encoder-only BERT inference, decoder-only      that GPT-2 has fewer FLOPs than BERT-Base and BERT-Large as
GPT-2 inference exhibits significantly lower arithmetic intensity.      the sequence length is increased. However, it is typically more
This is due to the fact that the decoder is composed solely of matrix-      challenging to run its inference efficiently due to its low arithmetic
vector operations, which limits the opportunities for data reuse.       intensity. This makes its performance memory bandwidth-bound,
That said, for a single matrix-vector operation, we perform roughly      as compared to encoder-only BERT models. This behavior is also
one multiplication and addition per parameter loaded since the      characterized in depth by [166].




                                                                                          6
<a id="page-7"></a>

### PDF 第 7 页

Table 3: Per-Layer FLOPs, memory operations (MOPs), and arithmetic intensity for the BERT-Base encoder with sequence lengths of 128, 512, and
4096 tokens. At low sequence lengths, the main contributors to both FLOPs and MOPs are the MHA and FFN projections. For longer sequence
lengths, the act-to-act matmuls consume a greater proportion of FLOPs, and these operations along with Softmax consume the majority of MOPs.
The act-to-act matmuls also have lower arithmetic intensity than the projection layers in the MHA and FFN for each sequence length.

     Sequence Length          Operator         FLOPs (× 109)  % of total FLOPs  MOPs (× 109)  % of total MOPs   Arithmetic Intensity

                MHA (projections)            7.25             32               0.04             27                192.00
              MHA (act-to-act matmuls)        0.60              3               0.006             7                  63.62
           128           FFN (projections)            14.50             65               0.07             49                211.86
                               Other                   0.08                0.3               0.02             18                  3.14
                                    Total                  22.42            100               0.14            100                159.68

                MHA (projections)           28.99             30               0.07             16                438.86
              MHA (act-to-act matmuls)        9.62             10               0.09             20                101.95
           512           FFN (projections)            57.98             60               0.10             25                558.54
                               Other                   0.42                0.4               0.16             37                  2.73
                                    Total                  97.02            100               0.42            100                 231.0

                MHA (projections)          231.93            18               0.33             3                 702.17
              MHA (act-to-act matmuls)       616.02            46               4.98             44                123.63
           4096          FFN (projections)           463.86            35               0.43             4                1068.52
                               Other                  11.85             1                5.47             49                  2.16
                                    Total                1323.66           100              11.22            100                117.96



Table 4: Per-Layer FLOPs, memory operations (MOPs), and arithmetic intensity for ResNet50. Convolutions consume the dominant proportion of
FLOPs, but BatchNorm, ReLU, and the other operations contribute a significant proportion of MOPs.

                      Operator     FLOPs (× 109)  % of total FLOPs  MOPs (× 109)  % of total MOPs   Arithmetic Intensity

                     Convolution          7.26             99               0.04             36                183.36
                   BatchNorm          0.03                0.5               0.03             31                  1.00
                    ReLU            0.008               0.1               0.02             15                  0.50
                       Other             0.01                0.1               0.02             18                  0.53
                      Total (Unfused)        7.31             100               0.11            100                 66.94
                       Total (Fused)          7.28             100               0.06            100                121.36


  Per-Layer FLOPs, MOPs, and Arithmetic Intensity. We then      a smaller number of heads (thus with a larger𝑑/ℎdimension) would
assessed the per-layer FLOPs, MOPs, and arithmetic intensity ver-     reduce the number of MOPs and improve the arithmetic intensity
sus sequence length for the BERT-Base encoder (Tab. 3). As shown       of the act-to-act attentions in the MHA module. This suggests that,
in Tab. 3, the proportion of FLOPs and MOPs consumed by the     when designing a Transformer architecture, the number of heads
act-to-act matmuls increases with sequence length, and these oper-     can entail a trade-offbetween accuracy versus performance metrics
ations have lower arithmetic intensity compared to the projection     on hardware.
layers in the FFN and MHA modules. This explains the decrease         Additionally, Tab. 3 illustrates that while the nonlinear opera-
in overall arithmetic intensity of encoder-only models for long      tions (classified as “Other” in the table) consume a small number
sequence lengths, as observed in Fig. 6.                                 of overall FLOPs, they consume a significant proportion of MOPs,
  The low arithmetic intensity of the act-to-act matmuls relative       especially for longer sequence lengths. Similar to the case of the
to the projection layers is because the 𝑑/ℎdimension in these two       act-to-act matmuls, the large number of MOPs in the Softmax op-
operations is small relative to the dimensions for the projection      erations for long sequence lengths is primarily due to several 𝑙×
layers (𝑑and 𝑑𝐹𝐹𝑁) and also relative to 𝑙, as the sequence length      𝑙matrices which must be either written out or loaded per atten-
is increased. Small matrix dimensions lead to lower arithmetic       tion head. This also indicates that the nonlinear activations, when
intensity, as there are fewer operations to perform per element in      handled poorly, can become a noticeable contributor to the overall
the matrix, leading to reduced reuse. The low arithmetic intensity is      performance, even though they might be overlooked due to their
further exacerbated with large activation sizes that must be loaded       insignificant contribution to total FLOPs. We provide a similar per-
and stored for the act-to-act matmuls. This activation size not only      layer analysis on the GPT-2 decoder in Tab. 11 of Appendix A.3,
grows quadratically with the sequence length 𝑙, but it is further     which demonstrates the significantly reduced arithmetic intensity
multiplied by the number of heads ℎsince each head has its own      across all layers, compared to the encoder-only model, resulting
activation (attention score) in the multi-head scheme. Therefore, as     from a large number of memory operations.
shown in Tab. 10 in Appendix A.3, a hypothetical BERT model with





                                                                                          7
<a id="page-8"></a>

### PDF 第 8 页

BERT-Base Latency Breakdown (CPU)                                 GPT-2 Latency Breakdown (CPU)
     100                                                                    100
   (%)                                                                       (%)
      Latency 75                                                                                                                                                              Latency 75
  of 50                                             of 50


      25                                                                     25         Percentage                                                                                                                                                                                                                                            Percentage

       0                                                                      0
             128     256     512     1024    2048    4096                          128     256     512     1024    2048    4096
                          Sequence Length                                                      Sequence Length
     MHA (act-to-act)      MHA (proj.)        FFN (proj.)        Other            MHA (act-to-act)      MHA (proj.)        FFN (proj.)        Other

Figure 7: Plot of the computation breakdown in the BERT-Base en-     Figure 8: Plot of the computation breakdown in the GPT-2 decoder
coder versus sequence length on a CPU. For smaller sequence lengths,      versus sequence length on a CPU. The projection layers in the MHA
the projection layers in the MHA and FFN modules dominate the     and FFN modules dominate latency for shorter sequence lengths, but
model latency. However, for longer sequence lengths the act-to-act       for longer sequence lengths the act-to-act matmuls become more sig-
matmuls begin to dominate.                                                  nificant. Note that nonlinear operations consume a more significant
                                                                         portion of latency than in encoder inference.

                                                                           Normalized Latency of Transformer Models (CPU)  Comparison with ResNet50. To provide a baseline in terms of
the FLOPs, MOPs, and arithmetic intensity for a typical CNN, we                      BERT-Base                                        2344
                                                                                                BERT-Large
also included a corresponding analysis of ResNet50 (architectural                                                                           2000       GPT-2
details can be found in Appedix A.2). Tab. 4 provides a breakdown of
the FLOPs, MOPs, and arithmetic intensity for ResNet50. Compared                                                                                                                                                                              Latency 1500to the BERT-Base encoder with a sequence length of 128 (Tab. 3),
ResNet50 without any operator fusion consumes 3.07 times fewer
FLOPs and 1.28 times fewer MOPs, resulting in lower end-to-end          1000                                               965arithmetic intensity than that of BERT-Base across all sequence                       Normalized
lengths in Tab. 3. The low arithmetic intensity is partially due to           500                                     427
                                                                                                                                             218                                                                                                                  201the nonlinear operations in ResNet50 that consume a negligible
                                                                                                                         9 26      2572      84                                                                                           1 3 45    2 6 93    3 10
proportion of FLOPs yet a significant proportion of MOPs, similar to
                                                                                   128      256      512      1024     2048     4096
the BERT-Base encoder. However, unlike the nonlinear operations                                 Sequence Length
in Transformers, these operations in ResNet50 can be fused with
the preceding matmuls in a straightforward manner for inference.     Figure 9: Plot of the normalized latency of the BERT-Base and BERT-
In particular, the ReLU operations can be applied directly to the      Large encoders and the GPT-2 decoder versus sequence length on a
accumulated outputs, and the BatchNorm operations can actually     CPU, normalized to the latency of BERT-Base with a sequence length
be folded into the prior convolutions. Fusing ReLU eliminates the       of 128. The latency scales quadratically with respect to sequence length
MOPs for this operation, and folding BatchNorm eliminates both       for both encoder-only and decoder-only networks. Additionally, for
the required FLOPs and MOPs for this operation. Broadly speaking,      encoder-only and decoder-only networks with the same model archi-
operation fusion refers to a methodology in which the output values        tecture, the latency is significantly longer for the decoder-only network
from one operation (e.g., a matmul or convolution) are directly used      due to its reduced arithmetic intensity.
as input to the subsequent operation (e.g., a ReLU or BatchNorm)
without first writing the output values to off-chip memory. Operator
                                                               modeling can provide a more accurate, non-ideal estimate by taking
fusion eliminates the need for unnecessary memory loads and stores
                                                                account of the hardware details.
for the nonlinear operations, and therefore it further improves
the end-to-end arithmetic intensity. As shown in Tab. 4, fusing
                                                                                     2.2.2  Profiling. To analyze the bottlenecks in Transformer work-
these operations with the prior convolutions improves the overall
                                                                      loads on commodity hardware, we profiled Transformer inference
arithmetic intensity for the ResNet-50 network from 66.9 to 121.4.
                                                          on an Intel Gold 6242 CPU. We profiled the workload latency break-
In Tab. 12 of Appendix A.4, we provide more detailed numbers for
                                                   down for both encoder-only BERT-Base and decoder-only GPT-2.
the FLOPs, MOPs, and arithmetic intensity of several convolutional
layers in ResNet50 as a reference.                                  Latency breakdown. Fig. 7 and 8 demonstrate how the latency
  Note that arithmetic intensity provides a rough estimate of how     breakdown changes with respect to sequence length on a CPU
much data reuse is possible for different models and operations       for BERT-Base and GPT-2, respectively. These breakdowns illus-
in the ideal case. Later in Sec. 3.3, we will discuss that analytical       trate that for short sequence lengths (e.g., 128-512), the majority




                                                                                          8
<a id="page-9"></a>

### PDF 第 9 页

of computations are in the projection layers of the FFN module,       3.1 then outlines the rationale of using domain specific acceler-
and that the majority of the MHA computation is in the projec-      ators for DNNs as well as the basic architectures and dataflows
tion layers. However, as sequence length increases, the act-to-act      that are used in most DNN accelerators. Sec. 3.2 then highlights
matmuls begin to dominate, as they both scale quadratically with       existing work on accelerating Transformers. Sec. 3.3 then provides
sequence length.                                                      analysis using an analytical model to assess how Transformers
                                                             run on a typical accelerator. Finally, Sec. 3.4 provides a case study
  End-to-end Latency. Fig. 9 shows the normalized latency for
                                                                              illustrating the process of building a typical accelerator for Trans-
different sequence lengths for BERT-Base, BERT-Large, and GPT-2.
                                                                     formers. Overall, this section gives relevant performance analysis
It is evident that the GPT-2 latency is far longer than the latency
                                                            and provides justification for the selected hardware decisions from
for either BERT-Base or BERT-Large for each sequence length,
                                                                  a full-stack perspective. Note that we are concerned here only with
even though BERT-Base and GPT-2 have largely the same model
                                                                              efficiently inferring DNNs. In particular, designing hardware for
configuration and end-to-end FLOPs (as was depicted in Fig. 4).
                                                                                 efficient model training is outside the scope of this paper.
This is mostly due to the lower arithmetic intensity of matrix-
vector operations, which was highlighted in Fig. 6. A model with
higher arithmetic intensity can run faster with the same (or possibly     3.1  Overview of Typical DNN Accelerators
even more) FLOPs than a model with lower arithmetic intensity.    A typical deep learning accelerator has a few key components, as
These observations confirm our findings that decoder inference      outlined in [27]:
is a memory-bound problem and not a compute-bound problem.
                                                             • Off-chip DRAM used for holding the weights and activationsWe revisit this issue at Sec. 4.3.3 to discuss some of the existing
methodologies to speed up the decoding process.                            of the full network, which needs to be large enough to hold all
                                                              model weights and activations;
                                                             • Smaller on-chip memory, referred to here as the global buffer,   Summary (Sec. 2.2. Model Analysis): Here are the high-
                                                               which needs to be large enough to hold a subset of the weights    level takeaways from this section.
                                                              and inputs in order to feed the processing elements (PEs);
     • Both FLOPs and normalized latency scale super-linearly
                                                             • An array of PEs, each of which has the capability to perform
       with sequence length for all Transformer models due
                                           MAC operations, and which often contains one or more small
        to the quadratic complexity of the act-to-act matmuls.
                                                                                  local memories called register files (RFs) that can store data with
       However, this trend is less obvious with small sequence
                                                                   lower per-access energy than the global buffer; and
        lengths, where the main contributor to the overall com-
                                                             • An internal network-on-chip (NoC) that transfers data between
        putation is the FFN, which scales linearly with sequence
                                                                         PEs.
        length, rather than the MHA module (Fig. 4 and 9).
     • For encoder-only models, arithmetic intensity initially              Fig. 10 shows the structure of a typical DNN accelerator. The
        increases as the sequence length increases. However, it          global buffer is designed to be able to hold a sufficient number of
       decreases for larger sequences since the MHA module          the weights and activations in order to allow for data reuse and
         (in particular, the act-to-act matmuls with lower arith-           limit the number of transfers to and from the off-chip DRAM. The
       metic intensity) becomes the dominant contributor to           local memories in the PEs are used to provide local data reuse in
         total compute (Fig. 6).                                        order to reduce global buffer accesses whenever possible. Without
     • The arithmetic intensity of decoder-only models is sig-          reuse, MAC operation requires loading three parameters, the two
        nificantly lower than that of encoder-only models, lead-          input values that are being multiplied as well as the current partial
       ing to significantly longer end-to-end latency for the       sum (which is the partially accumulated value for a given location
      same sequence length. This is due to the fact that de-          in the output matrix), and then storing the output value back to
       coder models involve matrix-vector operations with        memory. This is important because memory reads and writes are
        limited data reuse, making them memory bandwidth-         orders of magnitude more expensive from an energy perspective
      bound rather than compute-bound (Fig. 6 and 9).                 [87]. For example, for one particular technology, reads from a local
     • Matmuls consume over 99% of the FLOPs in both           buffer are roughly 6 times as expensive as a single MAC operation,
       encoder-only and decoder-only models, and nonlin-        and reads from external DRAM are roughly 200 times as expensive
       ear operations are a relatively small portion of over-           [206]. Leveraging reuse opportunities is therefore critical in order
          all FLOPs. However, the nonlinear operations have ex-          to reduce the number of expensive memory accesses performed.
       tremely low arithmetic intensity, especially for the large           To maximize data reuse, there are two broad classes of dataflows
       sequence length, due to the large volume of activations          that are widely adopted, which are referred to as temporal and
       they need to load and store.                                        spatial dataflows [27, 38, 206]. Temporal dataflows contain an array
                                                                       of centrally-controlled PEs that load data from the global buffer
                                                            and perform the requested ALU (Arithmetic Logic Unit) operations
3 HARDWARE DESIGN                                    before writing the data back to the global buffer. These PEs do not
So far, in Sec. 2, we have conducted an analysis of the run-time      contain local memories, and there is no communication or data
characteristics and bottlenecks of Transformer architectures. We     movement between PEs. Data reuse in this type of dataflow is only
now shift our focus to full-stack solutions for efficient Transformer       attainable through weight or partial sum reuse in the global buffer.
inference, beginning with the design of efficient hardware. Sec.     Examples of temporal dataflows include Single-Instruction Multiple




                                                                                          9
<a id="page-10"></a>

### PDF 第 10 页

activations for the full network, and an internal network-on-
                                   PE Array
                                                                      chip (NoC) can be used for transferring data between PEs.
                               …         DNN accelerators typically aim to leverage either temporal
                                                                         dataflows (by performing the same operation in parallel on
                       Local
    DRAM                                                                several datapoints) or spatial dataflows (where data can be
                Memory              …                                                                          transferred between PEs to leverage additional reuse op-
                                                                                  portunities). Spatial dataflow reuse schemes include weight
                …  …                      stationary dataflows, which hold weights in local memories
                                                                            in the PEs to improve reuse.


Figure 10: Basic structure of a DNN accelerator, assuming a spatial
dataflow between the Processing Elements (PEs) (inspired by [206]).      3.2  Adapting DNN Accelerators for
                                                   Transformers
                                                             There are several key considerations when designing DNN accel-Data (SIMD) and Single-Instruction Multiple Thread (SIMT) execu-
                                                                       erators for Transformers or adapting existing CNN accelerators.tion units. These type of units are commonly used in both CPUs
                                                    One difference between accelerators for CNNs and for Transform-and GPUs for vector processing. In temporal architectures, fully-
                                                                        ers is that due to differences in terms of arithmetic intensity andconnected and convolutional layers are both mapped as matrix-
                                                                matrix dimensions, these models have different optimal sizes formatrix multiplication operations.
                                                              each level of the memory hierarchy as well as different memory   In spatial dataflows, the PEs can communicate and data can be
                                                             bandwidth requirements.moved between PEs to leverage data reuse, without repeated reads
                                                               Another consideration is how the nonlinear functions are com-from the global buffer. The PEs themselves often contain RFs to
                                                               puted during inference, which imposes an additional challenge inhold weights or partial sums locally to improve data reuse, and
                                                            hardware design. These operations require either specialized sup-additional reuse can be attained through passing data between
                                                                      port for on-chip computation, or else they must be offloaded to theadjacent PEs. Spatial dataflows are commonly used in FPGA and
                                                         CPU. In Sec. 3.4, we will outline how the nonlinear operations canASIC-based accelerators, especially for convolutional networks [38].
                                                                      bottleneck inference even though they compose a small proportionThese dataflows allow for data reuse across multiple dimensions
                                                                       of model FLOPs, especially if they must be offloaded to the CPU.in order to drastically reduce the required memory accesses. In
                                                                      Several accelerators for Transformer inference contain specializedorder to maximize reuse in spatial dataflows, several different reuse
                                                                    post-processing units for nonlinear functions [107, 148, 166, 209].schemes have been employed:
                                                            However, adding an additional unit to support these operations
 • Weight stationary dataflows minimize the number of reads re-      also increases the area of the accelerator. This tradeoffbetween sup-
    quired for weight matrices by keeping weights in the local RFs      porting these operations on-chip and accelerator area will be also
    in PEs and streaming through inputs [72];                         explored in Sec. 3.4. Additionally, it can be challenging to design the
 • Output stationary dataflows minimize energy from reading and      hardware both to efficiently support the required nonlinear opera-
    writing partial sums by accumulating the outputs in the local       tions (e.g., Softmax and LayerNorm) and to support new nonlinear
   RFs in the PEs [59];                                              operations in future DNNs.
 • No local reuse dataflows have no RF in each PE, and use the        There are also considerations around datapath design, depend-
    area savings from having no RFs to allocate a larger global      ing on whether the accelerator is being designed for the MHA
    buffer [34]; and                                           module or for end-to-end Transformer inference. Accelerators spe-
 • Row stationary dataflows maximize reuse for both partial sums       cialized for the MHA module are designed to match the dataflow
   and weights by holding a row of the weights stationary in a       of the MHA module, where all the operations are “fused” together,
   row of the PEs, streaming in inputs, and streaming out partial      thus having less flexibility but better performance by reducing the
   sums [35].                                            number of required memory accesses [64, 79, 80, 223, 237, 253].
  Note that since DNNs consist of sequences of layers, it is also       Recall that operation fusion refers to using the output values from
possible to fuse operations in order to further leverage data reuse     one operation (e.g., a matmul) directly as input to the following
across multiple layers. We encourage the reader to refer to Section V      operation (e.g., a Softmax layer) without writing the intermedi-
of [206], Sections IV-A and IV-B of [49], and Sections III-A to III-C of      ate values to off-chip memory. Several accelerators for the MHA
[27] for more comprehensive surveys and comparisons of the basic     module develop dedicated datapaths with separate units for the
architecture of a DNN accelerator and typical accelerator dataflows.     query × key, Softmax, and attention score × value operations in
                                                                  order to better leverage operator-level fusion. In contrast, accel-
   Summary (Sec. 3.1. Accelerating Neural Networks):          erators for end-to-end Transformer inference typically employ a
    Typical DNN accelerators consist of on-chip memory for           similar structure to Gemmini [70] (which is outlined in more detail
    holding a subset of model weights and inputs and an array          in Sec. 3.4) where they are designed to be more flexible by per-
    of processing elements (PEs) which can perform MAC op-         forming individual operations separately in a more general matmul
    erations. Off-chip DRAM is used for holding weights and         engine [107, 148, 166, 209]. These accelerators also aim to fuse oper-
                                                                         ations whenever possible to improve performance (for example, by




                                                                                          10
<a id="page-11"></a>

### PDF 第 11 页

applying Softmax directly to the accumulated outputs of a matmul       Parameter    Default Value
before writing them out). However, the entire graph-level dataflow         L2 ↔DRAM BW   64 bytes/cycle                     DRAM
                                                                                                           Local Memory     16 bytes/cycle
is not hardcoded in hardware as in MHA-specific accelerators.          ↔L2 Bandwidth
   In both cases, there are dataflow considerations for nonlinear         L2 Size          2 MB
                                                                                                  Scratchpad Size   256 kB
function unit placement. This is because, as we have demonstrated          Accumulator Size  128 kB                                           PE Array (W x W)
in 2.2, non-linear operations generally have a high number of MOPs         PE Unit Width W   16
despite their small FLOPs count, and therefore the overall arithmetic         SFUClockWidthFrequencyS      161 GHz                L2         Scratchpad        Accumulator
intensity can be improved upon through operator fusion (as in the           Input Precision    8
ResNet50 case). In the case of accelerators for the MHA module,         Output Precision   32                                                 S-Wide Special
                                                                                                  Blue                                                                                              Arrows                                                                                                           –                                                                                                                                     External                                                                                                             Memory                                                                                                                                                                                    Trafficin order to leverage operator-level fusion in the MHA module,                                                                                        Function Unit (SFU)                                                                                           Green                                                                                                Arrows                                                                                                            –                                                                                                                                            Internal                                                                                                                             Data Movement
the Softmax unit must be placed appropriately such that it can
be computed after the query × key multiplication and before the      Figure 11: Diagram of the structure of the basic analytical perfor-
attention score × value multiplication. For example, [64] places the     mance model, as well as the parameters that were varied in this
Softmax unit in between specialized units for the query × key and       analysis.
attention score × value multiplications, and it computes LayerNorm
in a separate hardware module. Placing functional units to support
operator fusion provides higher efficiency, but this comes at a cost    DNN accelerators, with the notable assumption that the included
of less flexibility since the architecture now makes assumptions       special function unit (SFU) is able to compute all required nonlinear
about the operator-level dataflow.                                       operations, and thus none of these operations have to be computed
                                                                               off-chip. The model also assumes 𝑊-cycle latency for the PE array,
   Summary (Sec. 3.2. Adapting DNN Accelerators for       where𝑊is the width of the PE array, and 1-cycle latency per vector
   Transformers): Accelerators for Transformers and CNNs           for the SFU.
   have different optimal sizes for the memory hierarchy as                                                              Latency Breakdown and End-to-end Latency. One useful sce-
    well as different memory bandwidth requirements. Accel-                                                                      nario of analytical modeling is to obtain the estimated latency break-
    erators for the MHA module tend to design hardened data-                                                   down and end-to-end runtime latency. As an example, we applied
    paths to exploit operator fusion. End-to-end Transformer                                                                        analytic modeling to the BERT-Base and BERT-Large encoders as
    accelerators tend not to design their datapath around the                                                                      well as the GPT-2 decoder, under the assumption of square tiling for
    graph-level dataflow in the MHA module. Transformer ac-                                                                                          all matrix operations and no operation fusion (i.e., each operation
    celerators tend to incorporate a post-processing unit to                                                                      required inputs to be read from external memory and outputs to be
   compute nonlinear functions efficiently on-chip.                                                                      flushed out). In Appendix A.5, we provide the latency breakdowns
                                                                            for BERT-Base and GPT-2 (Fig. 30 and 31, respectively) as well as the
                                                                 end-to-end runtime latency of all models with different sequence
3.3  Analytical Modelling                                  lengths (Fig. 32). In general, the results of the analytical model
                                                       show similar trends in runtime latency scaling and breakdowns as
Analytic modeling is a useful tool for identifying bottlenecks when
                                                          compared with the profiling results on the CPU in Sec. 2.2.2, only
inferring DNN benchmarks, as it provides a quick estimate of run-                                                              with slight differences in details. Note that the analytical model
time behaviors on the target hardware platform. At design time, it
                                                        was designed assuming a hardware architecture that was different
can be quite difficult to analyze the runtime behaviors of benchmark
                                                           from the CPU architecture, and therefore the runtime behaviors
workloads as well as the potential impacts of hardware architectural
                                                           would not necessarily be identical for different hardware platforms.
changes on performance. This contrasts with the case in Sec. 2.2.2
                                                        More details can be found in Appendix A.5, including a comparison
where profiling can be conducted directly on actual hardware (e.g.,
                                                               with the analytic modeling results on ResNet50.
CPUs). In cases where profiling is difficult or infeasible, analytical
modeling can provide estimates to quickly guide design decisions.        Non-ideal Arithmetic Intensity. As with the analysis in Sec. 2.2,
   Here, we developed an analytical model to demonstrate how      arithmetic intensity provides a rough estimate of how much data
it can be useful in understanding the performance breakdown      reuse is possible for different operations in the ideal case. However,
of Transformer inference on hardware accelerators. Our analytic       in real-world scenarios, such as when tiling operations are required
model is based offof the Gemmini-driven architecture [70], which     due to the size of the matrices exceeding the capacity of the local
will be outlined in more detail in Section 3.4.1. Its structure is il-      scratchpad, the arithmetic intensity will be further reduced. In such
lustrated in Fig. 11, along with the tunable parameters. The model      a case, analytic modeling can provide a more accurate estimate,
includes local memories, a PE array for computing tiled matrix-     namely non-ideal arithmetic intensity, by taking into account the
matrix multiplications, and it relies on external memory for storing      hardware details. To take the tiling effect into account, we counted
all model parameters. The performance estimates assume that com-   DRAM to L2 memory traffic in our analytical modeling, but not
pute time and memory operation time can be overlapped perfectly,     L2 to Scratchpad, in order to avoid double counting. Furthermore,
and that the total for each operation is the maximum of these two.    we assume 32-bit output precision before the nonlinear operations,
Note that double buffering was assumed in the scratchpad to en-      since it is known that low input precision (e.g., 8-bit) to those op-
sure that compute could be overlapped with memory reads/writes      erations can result in a considerable accuracy degradation [111].
wherever possible. The model structure is comparable to typical     The non-ideal arithmetic intensities for different operations in the




                                                                                          11
<a id="page-12"></a>

### PDF 第 12 页

Table 5: Non-ideal arithmetic intensity for the BERT-Base encoder with sequence lengths of 128, 512, and 4096 tokens. The non-ideal arithmetic
intensity is lower than the ideal arithmetic intensities (provided in Tab. 3) due to using 32-bit output precision before nonlinear operations as well
as constraints from the memory sizes. Note that the differences in non-ideal arithmetic intensity between the 𝑊𝑄, 𝑊𝐾, 𝑊𝑉projections and the
𝑊out projection with the same operation dimensions are due to differences in output precision – 𝑊out is followed by nonlinear operations, and
therefore it uses 32-bit instead of 8-bit.

        Sequence Length  𝑊𝑄, 𝑊𝐾, 𝑊𝑉projections  Q × K   Attn. score × V  𝑊out projection  𝑊1 projection  𝑊2 projection    Total

              128                  170.670            25.400       63.750           128.000          130.723         186.182      106.110
              512                  341.333            29.882       102.300           204.800          211.862           409.6       111.122
              4096                   409.6             30.788       118.710           227.556          236.308         512.000       47.067

BERT-Base encoder are provided in Tab. 5 for sequence lengths of        CPU                 Gemmini
128, 512, and 4096.                                                                                     RoCC Cmd     Controller       Transposer   im2col                                                                       Core  Compared to the ideal arithmetic intensity that we have dis-                                                                                                           RoCC PTW    Dependency Mgmt
cussed in Sec. 2.2.1 (Fig. 6), which is 160, 231, and 118 for each                               DMA Engine              Spatial
                                                                                                       L1 I+D
sequence length, we observe significant reductions in the non-ideal                                                      Local TLB              Array
arithmetic intensity. This is due to the effects of tiling as well as the
                                                                                                Scratchpad
large 32-bit output activations which must be loaded and stored               L2                                                                                                                          Bank 0            ReLU       ++++++
before the nonlinear operations. The gap becomes even larger with                                                                                         Accumulator
                           …              Bitshi    SRAMa large sequence length (up to 2.5× reduction for sequence length
                                                                                                                          Bank K               Pooling   Matrix Scalar
4096), where the effect of loading and storing intermediate values       DRAM                                                 Engine     Multiplier
are more pronounced. This is different from the case of ResNet50,
whose non-ideal arithmetic intensity of 121.312 does not diverge a      Figure 12: Baseline accelerator’s hardware architectural overview.
lot from the ideal arithmetic intensity of 122.172. This also demon-
strates how even though the ideal arithmetic intensity of Trans-
formers was generally higher than that of ResNet50, the overall      ResNet50-like workloads, and we discuss changes we made to this
achieved arithmetic intensity is lower for Transformers across all       accelerator and it’s software stack to efficiently support transformer
sequence lengths.                                                workloads such as BERT. Several accelerators for end-to-end Trans-
                                                                 former inference employ a similar structure to Gemmini and to our
                                                                            analytical model and also contain specialized post-processing units   Summary (Sec. 3.3. Analytical Modeling): Analytical
                                                                           for nonlinear functions [107, 148, 166, 209].   modeling is a useful tool for identifying bottlenecks and
    runtime characteristics of DNN inference on a target hard-           3.4.1  Baseline Accelerator. We first generate a fairly typical
   ware platform. This technique can be especially useful dur-      CNN accelerator, which is illustrated in Fig. 12, using the Gemmini
    ing the design phase, where profiling on actual hardware          accelerator-generator. The accelerator performs matmuls using
   can be difficult, yet the analysis is necessary in order to         a 16×16 systolic array, which implements the weight-stationary
   make design decisions. We provided examples of using an-          dataflow. When performing convolutions, the dimensions of the
     alytic modeling to obtain latency breakdown and non-ideal         output-channels and input-channels are spatially unrolled. The 8-
    arithmetic intensity. In detail, we have demonstrated that            bit integer weights and inputs are stored in a 256 kB local scratchpad
    the non-ideal arithmetic intensity of the Transformer can        memory, and the 32-bit partial sums are stored in a dual-ported
    be further reduced (up to 2.5×) compared to the ideal case         64 kB accumulator SRAM which performs matrix additions. When
   when the hardware constraints and implementation details      DNN layers are too large to fit into the local scratchpad, they fall
    are taken into account.                                       back onto an external L2 cache and DRAM which are shared with
                                                   CPUs and other accelerators on the system-on-chip (SoC). A host
                                               CPU tiles such layers to compute the full outputs.
3.4  Case Study: Building a Transformer                                                                Although most of a CNN’s FLOPs are used to compute matmuls
     Accelerator                                               or convolutions, our baseline Gemmini-generated accelerator also
We now illustrate with a more “hands-on” example how architects      contains peripheral circuitry to execute ReLU and max-pool op-
familiar with mainstream accelerators for convolutional, vision-      erations, as well as integer-float multipliers to scale 32-bit partial
based workloads can design state-of-the-art transformer accelera-     sums to 8-bit inputs that can be fed into the next layer in a CNN.
tors. Although the analytical model in Sec. 3.3 presents ideal latency      Native support for these operations is important, as it eliminates
and runtime predictions for Transformer inferences, approaching      the need for costly transfers back and forth between DRAM or
the ideal performance and efficiency in real-world hardware accel-      outer caches (where the CPU can perform these operations) and
erators can take considerable engineering effort, which we explore      the local scratchpad (where Gemmini stores its matrix operands).
here. We start with a fairly typical CNN accelerator generated by          Finally, note that this baseline CNN accelerator does not include
the Gemmini [70] accelerator-generator, optimized primarily for     any Transformer-specific features. In particular, there is no sup-
                                                                    port for non-linear normalization operations such as LayerNorm




                                                                                          12
<a id="page-13"></a>

### PDF 第 13 页

module in Tab. 3. This suggests that the memory hierarchy and
               (kB) 64  66%  71%  69%                   memory bandwidth of our baseline CNN accelerator should be
               size                                                    re-tuned for more efficient Transformer inference.
            128  91%  91%  88%                               3.4.3  Memory Hierarchy. Transformer matmuls (in particular,
                                                                     the act-to-act matmuls) often have very different shapes and arith-
                                                                metic intensities than the convolutional layers in CNNs, as also
            256  91%  91%  90%                              illustrated in Tab. 3 and 4. As illustrated in Fig. 13, simply adjusting                                        Accumulator                                                     the sizes of the input/weight scratchpad and 32-bit partial accumu-
                  64    128   256                             lator significantly improves the performance of BERT’s matmul
               Input/weight scratchpad size (kB)                     operations. Larger accumulators enable higher output-reuse, which
                                                                                        is more suited for several of the matmuls in Transformers. The
Figure 13: The matmul utilization while performing a BERT-base                                                             query × key matmuls in particular have 𝑙× 𝑙output activation
inference on our baseline CNN accelerator, with different scratchpad
                                                                         matrices, which for long sequence lengths are much larger than the
and accumulator sizes.                                  𝑙× 𝑑/ℎinput query and key matrices. Increasing the accumulation
                                                                        buffer size therefore allows for improved output reuse with these
or Softmax. Neither is there support for GELU, which is a rela-      operations.
tively expensive non-linear activation function often implemented        Given this observation, we reduce the size of our baseline accel-
with costly lookup tables. Instead, this baseline design is a typical       erator’s shared input/weight scratchpad to 64 kB from 256kB, and
example of an accelerator designed and optimized for quantized     we increase the size of the partial-sum accumulator to 256 kB from
integer CNN inference. It achieves real-time or near-real-time per-     64kB. This involves no increase in the total SRAM capacity and
formance on end-to-end CNN workloads such as ResNet50 [82],      virtually no change to the total area of our accelerator. However,
SqueezeNet [93], or MobileNetV2 [188], but (we will see that) the      these changes yield a much more substantial 36% reduction in total
performance on Transformer workloads such as BERT is severely     matmul latency.
limited due to the need to perform operations such as GELU, Lay-       3.4.4  Hardware-Software Co-Design. As described in Sec. 3.3,
erNorm, and Softmax on the CPU.                              matmuls are the dominant kernel in Transformer workloads, but
                                                             even after maximizing matmul performance on our baseline CNN3.4.2  Performance Bottlenecks. Our baseline CNN accelerator
                                                                            accelerator, it still fails to achieve above 1% utilization. This is dueachieves far less than 1% utilization of its functional units when
                                                                       to the overhead of CPU-offloaded non-linear operations. Fig. 14performing BERT inferences. Although individual matmuls achieve
74% utilization, operations that the accelerator doesn’t support      demonstrates that this is because only 1% of time is actually spent on
                                                               matmuls. The rest is spent on floating-point non-linear activation,natively, such as LayerNorm, significantly reduce performance as
                                                                      normalizations, or on quantization and dequantization operations,they must be performed by the CPU instead. In fact, Fig. 14 shows
                                                                       since they are offloaded to the CPU.that 96% of execution is spent on non-matmul operations. Note
                                                          To alleviate the overhead of runtime quantization and dequan-that over 99% of FLOPs in our Transformer inference are MACs for
                                                                              tization, we switched our baseline Transformer workload from amatmuls, so the time consumed by each operation in the baseline
                                                                  naive BERT implementation, where only matmuls are quantized,accelerator’s run is far from the theoretical ideal, unless further
                                                                        to an integer-only BERT variant called I-BERT [111]. More detailsoptimizations are made.
                                                       on quantization and I-BERT will be revisited in Sec. 4.1 and 4.3,   Furthermore, our baseline accelerator offloads GELU and Soft-
                                                                  but the main idea of I-BERT is to replace floating-point nonlinearmax operations to the host CPU, which performs them with floating-
                                                                   operations such as GELU and Softmax with integer polynomialpoint units. As shown in Fig. 15, floating-point adders or multipliers
                                                                approximations such that they are both faster and cheaper to im-consume orders of magnitude more energy than the integer coun-
                                                              plement in specialized hardware accelerators.terparts. In our baseline CNN accelerator, matmuls are performed
                                                           To incorporate I-BERT, we added new integer implementationswith INT8 inputs, but these must be dequantized and requantized
                                                                          of I-BERT’s GELU, LayerNorm, and Softmax variants to our baselinein between matmul operations for floating-point activations to be
                                          CNN accelerator. The 32-bit matmul results resident in the accu-performed on the CPU, further contributing to the energy and
                                                               mulator are fed into a newly added “normalization unit” whichlatency overhead.
                                                           computes sums, sums-of-squares, maxes, and other reductions   Finally, a specialized hardware accelerator’s memory hierarchy
                                                         which are used by LayerNorm and Softmax. Multiple passes ofmust often be carefully tuned based on the workloads running on it.
                                                                 accumulator-reads are required to compute all the reductions inCNNs primarily perform convolutions,1 which have very high arith-
metic intensity, while Transformers primarily perform matmuls,      these operations. For example, a sum is computed first before a vari-
                                                               ance is computed using that sum. Afterwards, the matmul resultsoften with small and/or rectangular matrices, with significantly
                                                                       in the accumulators are read one final time to be fed into a set oflower arithmetic intensities and different optimal tiling strategies.
                                                                16 activation units which compute I-BERT’s GELU, LayerNorm, orFor example, we observe the low arithmetic intensities of the MHA
                                                              Softmax variants in parallel.
1Note that some CNN operations, such as “depthwise convolutions” in models such as       With these new features, overall end-to-end BERT inference per-
MobileNet [188], may also suffer from lower arithmetic intensities, but these operations
are found in only a subset of state-of-the-art CNNs, and often constitute only a small      formance improved by 39.6× over the baseline accelerator’s initial
portion of the total runtime of a vision model.                                performance. As Fig. 14 illustrates, the computational bottleneck




                                                                                          13
<a id="page-14"></a>

### PDF 第 14 页

100
               Matmul                                                                                       (%)
         Softmax                           GELU                                         Resadd                1%                                                                                                                                                                                                                                                                           Latency 75
              19%   10%                                  4%                                                                                LayerNorm        of 50                                                           7%  LayerNorm            4%                                           3%    Softmax
            11%                                   87%                                          25      Resadd                                Matmul+GELU                                                                                                                                                             Percentage                    49%
                                                                                                                   0
                                De/Quantization                                                                      128           256           512
                                                                                                                          Sequence Length
                                                                                               Matmul+GELU        LayerNorm        Resadd        Softmax

Figure 14: The time spent on different operations during a BERT inference with a sequence-length of 512, when running on (Left) the baseline
CNN accelerator, and (Middle) the accelerator after it has been extended with I-BERT’s hardware/software features for Transformers. Note that
with I-BERT support, quantization and dequantization operations are no longer required, because all operations happen in the integer format.
(Right) The time spent on different operations with different sequence lengths after the change. For all sequence lengths, the total execution time is
dominated by matmuls.


once again became the matmuls rather than normalization or acti-
                                                            To address these issues, we:
vation functions; and this trend persists across different sequence
                                                                • Reduced scratchpad capacity in favor of an increase inlengths. Quantization and dequantization no longer become nec-
                                                                     accumulator size, which enabled higher output reuseessary, since the non-linear floating-point operations are replaced
                                                                 and improved memory efficiency;with I-BERT’s integer polynomial approximations. Also note that
                                                                • Switched to I-BERT, an integer version of BERT that ap-GELU operations can now be trivially fused with the preceding mat-
                                                                         proximates floating point activations, eliminating quan-muls, so that they become one pipelined operation. When synthe-
                                                                                    tization overhead; andsized with the ASAP7 PDK [45], the new hardware units increased
                                                                • Added special normalizer units and activation units thatthe total area consumption of the accelerator by only 14%, and the
                                                                                 offload GELU, LayerNorm and Softmax computationsGELU, LayerNorm, and Softmax operations increased the power
                                                                  from the CPU.consumption of a BERT inference by only 9.3%.2
  To summarize, as shown in Sec. 3.3, the nonlinear operations         These changes mitigate the bottleneck on non-matmul op-
do not necessarily add much to the total FLOPs, area, or power           erations, and they trade a 14% area increase for a 39.6×
consumption of Transformer accelerators in the ideal case. How-         performance improvement. An important takeaway is that
ever, this may not be the case in practice, especially if the com-         even though the nonlinear operations in Transformers have
putation is offloaded to a CPU, leading to a non-trivial latency              little contribution to the overall FLOPs, area, or power, they
impact. We demonstrated that this can be addressed using the I-         can still have a non-trivial impact on latency.
BERT implementations of LayerNorm, Softmax, and GELU, which
only increases the area of a Transformer accelerator by 5-15%, and
adds 8% to the total latency.                             4 MODEL OPTIMIZATION
                                                           Given a DNN model that has already been designed and trained,
   Summary (Sec.  3.4. Accelerator for Transformers        one important question is whether it is still possible to algorith-
   Case Study): The baseline Gemmini accelerator designed          mically improve the efficiency of the model on the target hard-
    for CNN architectures uses 8-bit integer weights, and it        ware platform, through the adaptation of the model into a more
   has an accumulator for partial sum storage as well as a          hardware-friendly format. In this section, we discuss popular off-
    small scratchpad that overflows to an external L2 cache.          the-shelf model optimization methods, quantization and sparsity
   The performance of running Transformers on the baseline               (i.e., pruning), in Sec. 4.1 and 4.2, respectively. Then, in Sec. 4.3, we
    accelerator suffers for a number of reasons.                         outline Transformer-specific optimization methods to improve the
     • The bottleneck non-matmul operations running on the         performance of Transformer-specific features such as attentions
                                                           and nonlinear operations.     CPU takes 96% of total runtime;
     • Activation functions performed in floating point require
       repeated dequantization and requantization; and             4.1  Quantization
     • The lower arithmetic intensity nature of transformer      DNN models are typically trained using high-precision floating-
        inference is more sensitive to non-optimized memory          point computations. However, high-precision arithmetic is often
        hierarchy.                                                 unnecessary for inference. Quantization is a procedure for compress-
                                                                  ing DNN models by representing parameters and/or activations
                                                              with a lower-bit, typically (but not necessarily) fixed-point repre-
2Note that the ASAP7 PDK does not include energy models for SRAMs, so to derive
the total energy consumption of our accelerator, we used the SRAM energy estimates      sentation such as 8-bit integer (INT8), instead of 32-bit or 16-bit
in Accelergy’s [231] CACTI [159] plugin, and scaled them for 7nm.                     floating point (FP32 or FP16).




                                                                                          14
<a id="page-15"></a>

### PDF 第 15 页

Relative Energy Cost                Relative Area Cost
                                                               Operation:           Energy(pJ):                      Area(μm𝟐):
                                                     8b Add              0.03                          36
                                                     16b Add             0.05                          67
                                                     32b Add              0.1                           137
                                                     16b FP Add           0.4                           1360
                                                     32b FP Add           0.9                           4184
                                                    8b Mult              0.2                           282
                                                     32b Mult             3.1                           3495
                                                     16b FP Mult          1.1                           1640
                                                     32b FP Mult          3.7                           7700
                                                     32b SRAM Read (8kb)5.0                        N/A
                                                     32b DRAM Read     640                       N/A
                                                                                                    1    10   100  1000 10000            1    10   100  1000


Figure 15: (Left) Comparison between peak throughput for different bit-precision logic on Titan RTX and A100 GPU. (Right) Comparison of
the corresponding energy cost and relative area cost for different precision for 45nm technology [87]. As one can see, lower precision provides
exponentially better energy efficiency and higher throughput.


   Quantization offers multiple benefits for efficient DNN inference.      in Sec. 3.4.4 that integer-only quantization reduces the end-to-end
One obvious advantage of reduced precision is the reduction in      inference latency by 39.6× on Gemmini.
memory consumption. For example, quantizing model weights from         Quantization methods can broadly be categorized into uniform
FP32 to INT8 leads to a 4× smaller model size. This leads to reduced     and non-uniform quantization, depending on how they map the
off-chip storage and bandwidth without any modifications to a DNN       values. Uniform quantization splits the floating-point domain into
accelerator. Additionally, quantizing activations further allows for      evenly spaced intervals and maps each interval into a single fixed
reduced memory traffic and storage for intermediate partial results.      point value. This can be obtained from a simple arithmetic rule:
The memory hierarchy can also be restructured accounting for                         𝑄(𝑟) = Int(𝑟/𝑠) + 𝑍,                       (2)
the precision difference, either by allowing for greater local reuse
by storing a larger number of parameters (since each parameter     where 𝑄is the quantization operation, 𝑟is the floating point value,
now consumes less storage space), or else by using smaller internal     𝑆is a scaling factor, and 𝑍is a shift factor. Non-uniform quantiza-
buffers while maintaining the same amount of local data reuse.           tion, on the other hand, does not require the intervals to be evenly
  A second advantage of quantizing model weights and activa-      spaced. By assigning more quantization bins to important regions,
tions is the reduced size, latency, and energy consumption of the      generally resulting in improved compression rates, non-uniform
ALUs and the corresponding PEs. In general, floating point ALUs      quantization can more accurately capture the original data distri-
tend to be less efficient than integer ALUs in terms of area, latency,      bution in the floating point domain than uniform quantization.
and energy consumption. This is because floating-point PEs need      However, it is typically more challenging to efficiently deploy non-
to multiply mantissas, add exponents, and perform a left shift us-      uniformly quantized models on general computation hardware. As
ing the exponent to get the final result when performing a single      a result, uniform quantization is currently the de-facto method for
multiplication operation, whereas fixed-point PEs only require a        its simplicity and efficient mapping to hardware.
multiplication unit. For this reason, modern GPUs and TPUs often       While lower bit quantization can lead to a better compression
contain INT8 processing paths [43, 100], which can significantly       rate, reducing the precision too aggressively can significantly de-
benefit from quantization. For example, as illustrated in Fig. 15,      grade the model accuracy. It is therefore crucial to achieve a balance
performing INT8 addition can be ∼30× more energy efficient and     between performance gains through reduced precision and main-
∼120× more area efficient, as compared to the FP32 counterpart.        taining model accuracy. One promising strategy for alleviating this
  Another critical application for quantization is model deploy-      issue is mixed-precision quantization. It is known from previous
ment on integer-only hardware. Some edge processors for low-     work [55, 195, 224, 229] that different layers in a model exhibit differ-
cost and power-efficient embedded devices such as ARM Cortex-M      ent sensitivity to quantization, and that it is critical to assign higher
cores [12] and GAP-8 [66] do not include dedicated floating point       bit precision to the more sensitive layers. Notable works for quan-
units. When deploying models on these processors, not only must       tizing Transformers with mixed-precision include Q-BERT [195]
the model weights and activations be quantized, but also all compu-      that uses the Hessian information (i.e., curvature) as a proxy for
tations must be conducted using only integer arithmetic. Otherwise,       sensitivity, and HAT [222] that applies reinforcement learning (RL)
deployment is impossible or results in considerable overhead due to       to learn the appropriate bit precision per layer.
the need to process non-integer operations off-chip. This would lead        Another challenge with quantizing pre-trained Transformer
to additional latency and energy consumption for data transfer to a     models is the presence of outliers in activations [117]. Uniform
general-purpose host processor. This quantization technique of car-      quantization, which attempts to divide the range from the mini-
rying out the entire inference using integer arithmetic is known as    mum possible value to the maximum possible value into multiple
integer-only quantization [97, 110, 111, 132, 241]. We have discussed       bins, can result in significant performance degradation. This is be-
                                                                cause more values are mapped to the same quantized value (i.e.,




                                                                                          15
<a id="page-16"></a>

### PDF 第 16 页

resolution degradation) due to the outliers that extend the interval      order to store the data effectively without storing the null (i.e., zero)
of each quantization bin. While non-uniform quantization can be      parameters, a compressed memory format is necessary. Addition-
a solution to circumvent the outlier issue [246], a uniform quan-       ally, the computation units must be adjusted to be able to operate
tization scheme that assigns larger bit precisions to activations       directly on the compressed data. Otherwise, the parameters must
containing outliers has been proposed as well [51]. Furthermore,     be decompressed before computations and then re-compressed
the recently introduced FP8 precision [156], which provides extra      afterward, leading to additional overhead. For these reasons, com-
degrees of freedom in setting the exponent bit precision, has been      modity DNN accelerators might not efficiently exploit unstructured
found to be a suitable solution for quantizing models whose integer       sparsity patterns.
quantization results in reduced accuracy due to the presence of         Structured pruning circumvents these limitations by strictly re-
outliers [121].                                              moving structured sets of parameters. For instance, in Transform-
   For more information about this topic, see Section III-F of [27],       ers, rows and columns in linear layers, attention heads [155], or
Section IV-C-3 of [49], and Section 3.1 of [13], as well as [71] for a     even entire layers [63, 186] can be structurally pruned. Recent
more comprehensive survey of software-level approaches.           work has further integrated the structured pruning of different
                                                                        architectural components into a single framework (e.g., pruning
   Summary (Sec. 4.1. Quantization): Quantization is a way          attention heads in MHA modules and filters in FFN modules to-
    of compressing DNN models by reducing the precision of          gether) [88, 123, 145, 233]. Such structured pruning methodologies
   model parameters and/or activations. The immediate bene-         immediately lead to dense matmuls that are smaller than the orig-
      fit of quantization is reduced memory consumption, which            inal, eliminating the need for a compressed memory format or
    allows reduced off-chip storage and bandwidth, and a more          special hardware support to gain memory reduction and latency
    efficient memory hierarchy design. Furthermore, quantiza-         improvement. However, the compression rate might not be as good
    tion can reduce the size, latency, and energy consumption          as with unstructured pruning. It has been shown that a state-of-
    of the ALUs and the corresponding PE via low-bit preci-          the-art unstructured pruning method [120] can prune up to 90% of
    sion arithmetic. In some cases, quantization also makes it          the parameters in BERT [52] without any performance drop on the
    possible to deploy DNN models in integer-only hardware       MNLI benchmark [226], whereas the same performance can only be
    units, which otherwise may be impossible or may incur con-         achieved with a state-of-the-art structured pruning method [233]
    siderable overhead for offloading non-integer operations        by pruning up to 70% of the parameters.
    off-chip. While many DNN models are robust to a certain           While the aforementioned pruning methods belong to weight
    level of quantization noise, certain algorithm-level advance-          pruning, activation pruning (i.e., dynamic pruning) can also be ap-
   ments are necessary to prevent accuracy degradation with           plied to dynamically detect and zero out unimportant activations at
    lower-bit precision (e.g., INT4 or even less). In particular,          run-time. In Transformer inference, a popular branch of activation
    special considerations must be taken for quantizing pre-         pruning is token pruning [74, 108, 113, 150, 223], which detects and
    trained Transformers without accuracy degradation as they         drops less important tokens in each Transformer layer from the rest
    are known to have outlier activations.                               of the inference. The underlying rationale is that not all tokens (e.g.,
                                                            words in NLP tasks) are necessary to understand the meaning of the
                                                                     input sequence. By reducing the sequence length that Transformers
4.2  Sparsity                                              need to process, these methods have demonstrated a reduction of up
Another common avenue for reducing the overall number of compu-      to ∼30−50% in the total number of computations required, without
tations required for deep learning inference is through introducing      causing a noticeable drop in accuracy in NLP benchmarks [181, 220].
sparsity. Sparsity (also known as pruning) is a procedure of mak-     However, accelerating such dynamic sparsity patterns can be a chal-
ing DNN models sparse by removing those redundant/insensitive       lenge, as it requires detection logic to determine the location of
parameters. While it has been observed that having a dense model      nonzeros on-the-fly. Therefore, in many cases, dynamic sparsity
may be necessary to successfully train a model, it is also possible to      requires designing algorithm and hardware together.
remove many of the parameters after the model has been trained,         Regardless of the pruning methodologies used, the primary con-
without any quality degradation. It is known that training large      cern is determining which weights should be preserved and which
models and then compressing them via pruning achieves better      should be removed in order to improve the efficiency of the neural
accuracy then training a compressed model from scratch [133]. This     network without sacrificing its performance. Common methodolo-
may be due to the fact that having redundant parameters from the       gies for pruning Transformers include the following:
beginning of the training may make the loss landscape easier to
optimize [139]; or it may be related to the increase in the likelihood       • Magnitude pruning [68] is a technique that uses the absolute
of obtaining a “lottery ticket” [67].                                      value of each weight as a proxy for its importance. It prunes
   Broadly speaking, pruning can be categorized into two branches:         the weights with the smallest magnitudes during the training
unstructured pruning; and structured pruning. Unstructured prun-          process. The rationale behind this approach is that smaller
ing allows arbitrary patterns of sparsification for parameters and         weights contribute less to the model’s final outcome.
feature maps. It can, in theory, produce significant computational       • Movement pruning [124, 189] is a technique that takes into
savings without accuracy degradation [136]. However, unstructured         account the changes in weights during fine-tuning, assigning a
pruning can be challenging to leverage effectively in hardware. In           larger importance score to the weights that move further away
                                                                from zero as the fine-tuning process progresses. This technique




                                                                                          16
<a id="page-17"></a>

### PDF 第 17 页

has been found to be more effective than magnitude pruning for      We refer interested readers to Section V of [47] and Section III of
   models that are trained using the pre-training and fine-tuning       [27] for a more comprehensive overview of sparse encoding meth-
   scheme (e.g., BERT [52]), as it better captures the importance      ods. Additionally, a general overview of hardware architectures
    of weights as the fine-tuning process progresses.                    that leverage various sparsity patterns is provided in [47].
 • First-order pruning [155] uses gradients with respect to the loss
    that flow into the weights or a group of weights as a proxy       Summary (Sec. 4.2. Sparsity): Sparsity (or pruning) is an-
    for evaluating the importance of the model accuracy. This ap-          other widely-used method of reducing the inference cost of
   proach considers the gradient to be an indicator of the impact          overparameterized DNN models by removing redundant or
    of zeroing out a parameter on the loss. This scheme was further            less important weights and activations. Similar to quantiza-
   improved [88], where the product of weight magnitude and            tion, pruning helps to reduce off-chip memory consumption
    gradient was used as a proxy for importance, as it may be a        and the corresponding memory traffic, as well as energy
   more accurate estimate of the impact of zeroing out weights.          consumption and latency. Pruning can be broadly divided
 • Second-order pruning [120, 123, 245] uses the Hessian matrix of          into weight pruning and activation pruning. Weight prun-
    the weights or a group of weights with respect to the loss as a          ing can be further divided into unstructured pruning, which
   proxy importance metric. Compared to the first-order informa-          allows any sparsity pattern, and structured pruning, which
    tion, the second-order information is generally known to be a         imposes an additional constraint on the sparsity pattern.
   more accurate indicator of the effect of removing weights. How-        While structured pruning can provide benefits in terms of
    ever, due to the large size of the Hessian matrix, which grows        memory, energy consumption, and latency without addi-
    quadratically with the number of weights, it is necessary to           tional hardware support, it is known to achieve less com-
   employ an appropriate and scalable approximation, often with          pression rate than unstructured pruning. Activation prun-
    algorithms from randomized numerical linear algebra [57, 242].         ing prunes redundant activations during inference, which
                                                                can be especially effective for Transformer models. How-
                                                                                 ever, this requires support to dynamically detect and zero
  One of the main advantages of pruning is the reduction in mem-         out unimportant activations at run-time.
ory footprint. The gain in memory efficiency is straightforward
with structured pruning, which directly reduces the size and/or
number of matrix multiplications. In contrast, unstructured pruning     4.3  Transformer-specific Optimization
often requires the use of sparse encodings (also known as sparse        Methods
storage formats) to compress and store sparse data. These methods
                                                       The use of off-the-shelf optimization methods such as quantiza-
use less memory by employing metadata to encode the positions
                                                                       tion and pruning can lead to significant performance advantages.
of the nonzero entries in the matrices [27, 47]. Sparse encodings
                                                                     Nevertheless, there are other optimization strategies that are tai-
can reduce off-chip memory consumption and the corresponding
                                                                      lored specifically for the Transformer architecture, e.g., by taking
required memory traffic. They can also reduce the required storage
                                                              advantage of the features within it. Here, we review the signifi-
size on chip, thereby allowing for smaller buffers or, alternatively,
                                                                     cant Transformer-specific optimization techniques that can further
increased reuse. This is because, although the same amount of data
                                                                   optimize Transformer inference.
can be stored in a buffer, the encoded data corresponds to a greater
proportion of the full-sized input tensor.
  Pruning can also lead to reduced energy consumption and la-       4.3.1  Accelerating Attention. Several works aim to optimize
tency due to the elimination of unnecessary computations. Similar      the attention mechanism in the MHA module. Recall that the time
to what we described above, this is relatively straightforward to      spent performing the matrix multiplications in the MHA module
achieve through structured pruning, but unstructured pruning re-     grows quadratically with sequence length for long sequences, as
quires special techniques for identifying and bypassing calculations      outlined in Sec. 2.2.2. Therefore, for long sequences, computing
involving null elements [8, 9, 35, 164, 248, 251]. This can involve       attention becomes the dominant portion of the overall runtime. One
identifying and skipping individual elements or entire null vectors.    common route for more efficiently computing the attention net-
Some detection and skipping methods only save energy by not     work is token pruning. This involves removing unimportant tokens
performing the operation involving the null element. That is, the      so as to reduce the effective sequence lengths, as was discussed in
PE doesn’t have to be used for the null computation, in which case      Section 4.2. The need to efficiently identify and drop unimportant
it avoids energy consumption. Other methods additionally seek to      tokens on-the-fly has led to several hardware-software co-design ap-
reduce latency by assigning a different effectual computation to the      proaches. In SpAtten [223], tokens are ranked based on the amount
skipped PE, rather than having them idle for the ineffectual com-      of attention they are getting from other tokens in the input sen-
pute cycles. Furthermore, in order to maintain PE utilization with       tence, and the tokens that are receiving less attention are pruned
unstructured sparse matmuls, it may also be necessary to perform       out. This approach is based on the simple rationale that the more a
load balancing. Since the distribution of zeros can be unbalanced     word is attended, the more important it is in the inference process.
between PEs, some PEs may require a longer execution time than     To make this efficient, a top-𝑘hardware engine is employed to
others, resulting in idle waiting periods of the others. Several works        filter out the low-importance tokens based on their attention scores.
have used load balancing for accelerating neural networks with     DTA-Trans [237] takes a step further by introducing the two-tiered
unstructured sparsity [81, 119, 147].                              scheme where in the first round, it determines which tokens should




                                                                                          17
<a id="page-18"></a>

### PDF 第 18 页

be pruned, and in the second round, it further determines the bit-      thus reducing the number of passes required for the numerically sta-
precision to be allocated to each of the remaining tokens based on       ble Softmax computation from 3 to 2. I-BERT [111] provides a more
their significance.                                                    general approximation algorithm that approximates the nonlinear
  Another approach to accelerate attention is to leverage the dy-      operations with 2nd-order polynomials. This not only simplifies
namic sparsity patterns of the attention score activations [79, 80,      the operations, but it also enables them to be performed using
131, 147, 172, 194, 204, 253]. It is reasonable to assume that many      only integer arithmetic. SpAtten [223] takes a similar approach
combinations of query and key tokens are not semantically mean-      to use a 5th-order Taylor approximation for computing Softmax,
ingful, and thus the attention score associated with this combination      as described in [160]. I-ViT [132] further extends this idea to use
will be close to zero. By taking advantage of this sparsity, the in-      hardware-friendly bit shifting operation to efficiently compute the
ference accuracy can be preserved while the computational cost is      nonlinear operations for vision Transformer inference. While the
reduced by avoiding the associated act-to-act matmuls (i.e., query ×      major focus has been approximating the exponential operation for
key or attention score × value). However, this requires specialized      the Softmax, other works [19, 209, 225] have also exploited the log
hardware logic to detect and accelerate those dynamic sparsity pat-     sum-exp trick to avoid the division operation, another operation
terns on-the-fly. For instance, ELSA [80] proposes a datapath that       that can be complicated to implement in hardware [60].
approximates the angular similarity between key and query vectors,        Another widely-adopted approach is lookup tables, which store
thus allowing the prediction of whether their dot product is likely       pre-calculated output values for a given range of inputs. In this case,
to be zero. This approach enables the pruning of less important         if the input is given, the corresponding value stored in the lookup
key vectors with respect to a given query vector in advance. The       table is outputted, eliminating the need for evaluating the function.
Sanger framework [147] suggests quantizing the query and key     The use of lookup tables to accelerate the nonlinear function is
values prior to computing the attention score, as this will zero out     by no means a new concept, with its root predating the advent of
insignificant entries in the resulting attention score that would have      Transformer or DNN architectures [50, 212]. Recent approaches,
been close to zero if those values were not quantized. Similarly,      therefore, have more focused on reducing the size of the lookup
DOTA [172] proposes to approximate the attention score entries to       table to save area and latency. For instance, 𝐴3 [79] decomposes
be zeroed out by employing the matrix multiplication of low-rank      the exponential operation into a multiplication of two smaller-
(and hence smaller) projections of the query and key values as a      precision exponential operations, allowing one larger lookup table
proxy. LeOPArd [131] uses bit-serial computing for the query ×       to be replaced with two smaller ones. NN-LUT [244] approximates
key multiplications in order to terminate computation early if it      the nonlinear operation using a single-hidden layer network and
will not reach the pre-determined threshold.                             stores a numerical approximation of the network in a lookup table,
    It is worth noting that hardware support is essential for accel-      thereby avoiding the need for executing the network.
erating attention mechanisms, as it enables operations such as
top-𝑘[223, 237], angle approximation [80], clustering [204], and       4.3.3  Accelerating Decoding. As discussed in Sec. 2.2, Trans-
multi-precision computation [13, 147, 172, 253] that are necessary      former decoding for generative inference can entail a significant
to detect the dynamic sparsity patterns of attention scores. Further-      inference latency due to the low hardware utilization and arithmetic
more, specialized hardware support is needed to take advantage of       intensity. Due to the growing interest in generative tasks due to the
the (mostly unstructured) dynamic sparsity for skipping compu-      recent advancements of Large Language Models [2, 22, 44, 213], it
tations. For example, Sanger [147] uses load rebalancing through        is critical to optimize the latency for the decoding process. One av-
splitting and packing, and it is equipped with a custom datapath      enue to reduce inference latency is to skip unnecessary computation
that provides support for both sampled dense-dense matmuls and      through early exiting. This method dynamically adjusts the depth
sparse matmuls.                                                        of the decoder for each token generation by terminating the infer-
                                                                 ence at a mid-layer and making a prediction using the intermediate
                                                                hidden states, rather than waiting until the end layer. While being a
4.3.2  Nonlinear Operations. As discussed in Sec. 2.1.1, the Trans-     well-explored technique for encoder models [192, 234], CALM [191]
former architecture contains multiple nonlinear functions that pose      has only recently extended this methodology to decoder models. A
multiple challenges in efficient hardware design. Incorporating a      major challenge in decoding tasks is that, unlike in encoding tasks,
hardware module specialized for computing these operations may      the generation of one token relies on the activations of all previ-
be a viable solution. However, this may incur a considerable over-     ous tokens, due to the attention mechanism. If a previous token is
head for hardware design, particularly when targeting low-end      exited early, then there is nothing to attend for the skipped layers.
devices. Therefore, various solutions have been proposed to circum-     To address this issue, CALM proposes “state propagation,” which
vent this issue, without constructing a dedicated hardware module.      copies the activations of the final layer before exiting to all the
  One popular solution is function approximation [107, 111, 132,      skipped layers. This had a minimal impact on generation quality.
203, 223], which seeks to approximate the exact value of the non-        Another recent attempt is to collaboratively use multiple models
linear function, in order to obtain a good yet computationally ef-     with different sizes [31, 112]. The underlying motivation is that the
ficient approximation. For instance, Keller et al. [107] uses the      majority of simple word generation can be offloaded to a faster, less
Softermax [203] function, which uses a base-2 approximation that      accurate model with a smaller size. Once in a while, when the small
switches the base used in the exponential calculation of the Softmax     model is unable to predict a word accurately, it switches the control
operation from 𝑒to 2, allowing for simplified hardware implementa-      to the larger model for more accurate prediction. This approach not
tions. Softermax [203] also incorporates online normalization [157],      only enables the execution of the large model to be carried out less




                                                                                          18
<a id="page-19"></a>

### PDF 第 19 页

frequently, but it also enables its non-autoregressive (i.e., token-
                                                               methods should also be aware of the underlying hardware
level parallel) execution since it can consume all tokens generated
                                                                             architectures and datapaths. The use of separate datapaths
from the small model and process them in parallel, thereby utilizing
                                                                              for the MHA and FFN modules can have higher area over-
hardware more efficiently. Big Little Decoder [112] has shown ∼2×
                                                                        head, but can enable more aggressive optimization as com-
inference latency reduction across various models and generative
                                                                    pared to using a single datapath for both modules.
tasks without compromising generation quality.

4.3.4  Selecting Which Optimization Methods to Use. So far,    5 MAPPING TRANSFORMERS TO
we have discussed various optimization techniques that can be ap-
                                  HARDWAREplied to the Transformer architecture. It is important to note that a
significant portion of these optimization methods depends upon      In order to execute a Transformer block on a target hardware archi-
the underlying hardware support. Thus, when selecting which opti-      tecture, we must map it into hardware instructions that carry out
mization techniques to employ, it is essential to adopt a holistic view      both the required computation and communication. The choices
of both the hardware and software stack, taking into account the     made during the mapping process play a significant role in perfor-
characteristics of the underlying hardware. In particular, whether      mance. However, the size of the space of possible mappings makes
the accelerator supports MHA and FFN modules in the same datap-      finding the optimal mappings difficult, and it requires the use of
ath versus containing separate datapaths for each of these modules       carefully considered exploration, heuristic, or learning-based ap-
can have a significant impact on the optimizations that can be      proaches.
performed.                                                               In this section, we provide an introduction to the mapping prob-
   Accelerators with a unified datapath tend to pursue more general     lem in Sec. 5.1; and we discuss key mapping decisions for efficient
optimizations that can either be applied to both MHA and FFN mod-      execution of Transformers in Sec. 5.2. We overview the taxonomy of
ules, or at least those that do not require altering the datapath such       existing mapping techniques in Sec. 5.3 and similarly for techniques
that it can no longer compute the other modules. For example, sev-      to model the performance of different mappings in Sec. 5.4. Finally,
eral accelerators that support both MHA and FFN modules employ    we end with Transformer-specific considerations for mappers in
general static pruning methods for weight matrices [65, 166, 209],      Sec. 5.5.
but do not aim to exploit attention-specific pruning methods such
as dynamic sparsity. However, more exotic optimizations can be
pursued separately for the MHA and FFN modules, if they are com-     5.1  What are Mappings?
puted in separate datapaths or if the PEs can be reconfigured. For    A mapping or schedule is defined as a sequence of hardware instruc-
instance, FABNet [64] exploits static sparsity patterns that can only       tions to execute a set of operations on the specific target hardware
be applied to the FFN module by adopting separate datapaths for the       architecture. In the case of a systolic array accelerator such as Gem-
MHA and FFN modules. FTRANS [127] also applies different opti-      mini, such hardware instructions might include dense matmuls
mization methods for the MHA and FFN modules by incorporating     under a specific dataflow and load/stores to move data between
reconfigurable PEs that can handle both workloads without having       off-chip DRAM and local SRAMs. A mapping will list the complete
two separate datapaths. However, employing separate datapaths or      sequence of data and memory instructions, with the end goal of
reconfigurable PEs can incur an additional overhead, as compared      producing source code or compiled binaries that can be executed
to using a general, unified datapath. Consequently, there is a trade-     on hardware.
offto consider between the area overhead and the performance        For some operations, there may exist multiple valid mappings,
gain derived from the use of more aggressive optimizations.          where the validity refers to the guarantee of correctness from exe-
                                                                   cuting each mapping. Specifically, different sets of mapping deci-
   Summary (Sec. 4.3. Transformer-specific Optimiza-          sions applied to the same problem may result in valid yet different
    tions): While general off-the-shelf optimization methods         mappings. We refer to the total space of mapping decisions and
   can also benefit efficient Transformer inference, a great           their resulting mappings as the mapspace. Details about individual
    deal of research has been conducted to devise optimization        mapping decisions are discussed in the following Sec. 5.2.
    strategies that take advantage of Transformer-specific char-                 It is not surprising that two different valid mappings may ex-
    acteristics. One opportunity is to optimize the attention           hibit differences in end-to-end performance, when measured with
   mechanism in the MHA module, whose runtime cost in-          respect to latency, bandwidth, and energy consumption. Hence, it
    creases quadratically with sequence length. For instance,            is often the goal of a mapping or scheduling framework to obtain
   dynamic pruning has been widely applied to take advantage          Pareto-optimal, valid mappings for a given software operation and
    of the sparse nature of attention score activations. Addi-          desired performance metrics. For some operations, finding a good
    tionally, efficient computation of the nonlinear operations        mapping is unnecessary, either because the problem is trivial, as
   should also be taken into account. In order to reduce the          the mapspace is small, or because the operation itself is not a per-
   hardware costs associated with the implementation of ded-        formance bottleneck and does not warrant the effort of judicious
    icated hardware units for nonlinear operations, function          scheduling. However, in the case of core computational operators in
    approximation, and lookup table methods have been pro-        DNNs, including Transformers, the mapping problem is both chal-
   posed as viable alternatives. Finally, the model optimization          lenging due to large mapspace and rewarding due to the potential
                                                                    gains in overall model execution speedup.




                                                                                          19
<a id="page-20"></a>

### PDF 第 20 页

Conv in 6 nested loops:                Conv
                                                                                        for p in [0, P):
      Inputs          Weights          Outputs                for q in [0, Q):
                                                                                           for c in [0, C):
                                                                                            for k in [0, K):
  C                K                           for r in [0, R): S
 +                                                                                          for s in [0, S):
                   K                               Outputs[p][q][k]+=
                                                                                      Inputs[(p-1)*stride+r][(q-1)*stride+s]*Weights[r][s][c][k]      Stride        ∗       = Q x                             ...
  1)
 -          S                     Conv mapping:  (Q
    (P - 1) x Stride + R      C           P            ---------DRAM ---------------------
                    R                                                                          | for p1 in [0:4)                                                                           Loop Permutation                                                                                                                                                 |  for s0 in [0:3)
                                                                                                                                                 |    for c1 in [0:16) (Spatial-X)
                                                                                                 Spatial Mapping
    R, S: convolution kernel width and height                                ---------LocalBuffer---------------
                                                                                                                                                 |        for q0 in [0:28)
     P, Q: output width and height                                                                                                                                                 |         for p0 in [0:7)         Temporal Mapping
    C: input channel size                                                                                         |           for k0 in [0:128)
    K: output channel size                                                                                      |            for c0 in [0:8)            Tiling Factors


Figure 16: Visualization of mapping for the convolution operation in CNNs onto a typical DNN accelerator. Convolution is represented as a
six-nested loop excluding the batch dimension. Loop permutation concerns the order in which each loop level should be executed, with memory
accesses to and from either the accelerator’s local memory or off-chip DRAM. Spatio-temporal mapping determines which loop level should be
executed in parallel using accelerator hardware resources. Tiling factors are the loop bounds of each loop level, where each dimension can be
broken down with tiles into several sub-loops. As shown in the example, the input channel size dimension (𝐶) is tiled with a tiling factor of 8,
hence into two sub-loops with loop variables 𝑐0 and 𝑐1.


                Matmul                Matmul in 3 nested loops:

                                                                                       for m in [0, M):
         Inputs          Weights          Outputs           for n in [0, N):
                                                                                          for k in [0, K):
                                                                               Outputs[m][n]+=Inputs[m][k]*Weights[k][n]



             K  M                 = M         ∗
                                      Matmul mapping:

                                                                                  ---------DRAM ---------------------
         K            N           N                              | for n1 in [0:4)             Loop Permutation
                                                                                                                                                 |  for m1 in [0:4)
                                                                                                                                                 |    for n0 in [0:4) (Spatial-X)       Spatial Mapping
   M: number of rows in the output matrix
                                                                                 Temporal Mapping    K: reduction dimension size                                                    ---------LocalBuffer---------------
    N: number of columns in the output matrix                                            |     for m0 in [0:16)
                                                                                                                                                 |        for k0 in [0:64)               Tiling Factors



Figure 17: Visualization of mapping for the matmul operation in Transformer encoder/decoders onto a typical DNN accelerator. Matrix
multiplication is represented as a three-nested loop. Loop permutation concerns the order in which each loop level should be executed, with
memory accesses to and from either the accelerator’s local memory or off-chip DRAM. Spatio-temporal mapping determines which loop level
should be executed in parallel using accelerator hardware resources. Tiling factors are the loop bounds of each loop level, where each dimension
can be broken down with tiles into several sub-loops. As shown in the example, the output column dimension (𝑁) is tiled with a tiling factor of 4,
hence into two sub-loops with loop variables 𝑛0 and 𝑛1. As we will discuss in Sec. 5.5.1, even though matmuls have 3 nested loops instead of 6 as
in the convolutions, finding an optimal mapping could still be as challenging.




                                                                                          20
<a id="page-21"></a>

### PDF 第 21 页

Fig. 16 and 17 illustrate examples of key operators and their           retain accuracy even with extreme sparsity levels, but it is hard
possible mappings, for CNNs and Transformers, respectively. As          to accelerate. Nevertheless, the latter is going to become in-
shown in the example mappings, each level of the nested loops must          creasingly more important since it reduces the memory traffic,
be: (1) assigned to be executed either with data from DRAM or from        which is becoming a major bottleneck for power consumption.
local accelerator memory; (2) assigned to be executed spatially
                                                                                  5.2.2  Operation-level. The operation-level scheduling step de-(i.e., in parallel) or temporally (i.e., sequentially), if the accelerator
                                                            composes tensor operations into a set of tasks to be run on a givencontains spatially parallelized compute resources; and (3) assigned
                                                                            architecture. This consists of several different steps, each of whichto be executed as one loop or tiled into multiple subloops (and if so,
                                                                     presents a programmer with a decision problem. These include:with which tiling factors). In particular, for the case of Gemmini,
spatial mapping concerns the decision to assign which loop levels       • Dividing the operation into tiles that can fit onto different layers
to be executed on the N-by-N systolic array mesh of PEs.                    of the memory hierarchy; the dimensions of the tiles are a choice
                                                                                                   (e.g., tile sizes in Fig. 17).
5.2  What Are the Key Mapping Decisions?            • Determining the dataflow of the computation, i.e., the order that
                                                                         the tiles are executed in and the tensors that are held stationaryMapping occurs in two steps. First, the graph is transformed at a
                                                                      or moved across the processor. This can be encoded as a loopgraph level into a set of tensor operations. This may involve fusing
                                                                       ordering problem, with the innermost loops corresponding tosuccessive operations, sparsifying tensors, and deciding on appro-
                                                                       axes of tensors being held stationary (e.g., any loop permutationpriate quantization strategies. Then, each resulting tensor operation
                                                                            in Fig. 16).is scheduled in order to transform it into hardware instructions of
                                                             • Deciding which axes to parallelize, and which to run serially,the appropriate size.
                                                              which we refer to as spatio-temporal mapping.
5.2.1  Graph-level. Graph-level scheduling involves decisions that      • Deciding how to interleave communication and computation in
change the structure of the computational graph, rather than sim-         order to minimize latency. For instance, double-buffering may
ply the execution schedule of tensor operations represented by          divide the scratchpad into two halves, with one half being used
individual nodes within the graph. Typical changes include the         by the processor for computation while the other is loaded with
following:                                                              data from memory.
 • Layer fusion or operation fusion refers to combining multiple       • Mapping arithmetic instructions onto hardware instructions.
    layers (e.g., a matmul followed by a normalization layer) into a         For some architectures, this may be as simple as replacing a
    single tensor operation to be scheduled and run on the accel-        matmul operation of the appropriate size (achieved by a tiling)
    erator. This reduces interlayer communication, as the results         with a call to the appropriate ISA (Instruction Set Architecture)
    of one layer can remain on the chip as input without being           instruction. For others, it may involve selecting between differ-
    written to and later read from main memory, at the cost of          ent vector instructions, which may affect the decision of which
    intralayer communication. As we will see in Sec. 5.5, layer fu-         axes to vectorize, and the resulting spatio-temporal mapping.
    sion may not provide as much latency improvement, as with    A more complete description can be found in [128].
  CNN architectures, since static fusion opportunities are not as       The choice of points in the mapspace heavily affects performance,
    straightforward as fusing convolutions with BatchNorm layers.     by up to several orders of magnitude, as we will discuss in Sec. 5.5.1.
    For Transformers, it is possible to combine several operations      For this reason, the goal of a hardware mapper is to select a point
    in the same kernel, but this may increase intralayer commu-      in this space to minimize some cost such as energy, energy-delay
    nication to an extent that renders such approaches infeasible.      product (EDP), latency, etc., on a given hardware target. However,
    Furthermore, this can also be dependent on the target hard-      the size of the mapspace renders exploration difficult. For exam-
   ware platform.                                                            ple, considering only tiling, spatio-temporal mapping, and loop
 • Dynamic sparsification of tensors happens when the pruning      ordering (dataflow), the number of possible mappings for a BERT
    decisions are made based on the activation maps. Common       attention layer can exceed 1012. As a result, the design and selection
   methods for dynamic sparsification includes locality-sensitive      of mappers have been the subject of significant attention in both
    hashing to zero out dot products likely to be small. This can sig-      theory and practice.
    nificantly reduce the number of arithmetic operations required        Furthermore, the optimal mapping can significantly differ de-
    in the operation, as was also discussed in 4.2. Such optimiza-     pending on hardware architecture, and mappings that work well
    tions are heavily data-dependent as they require access to the       for one set of hardware parameters often perform poorly on others
    activations and, as a result, cannot always be estimated a pri-      [103]. This significantly increases the difficulty of mapping within a
    ori [103]. As a result, relatively few results on sparsity-aware      codesign context, as one must be computed for every pair of neural
   mapping exist, and those that do largely cover operation-level     network and hardware architecture.
   mappings for a given amount of sparsity.
 • Static sparsification of tensors happens when the pruning de-      Summary (Sec. 5.2. Key Mapping Decisions): Mapping
    cisions are independent of the activations and are determined         Transformers to hardware require decisions to be made
     statically. As was discussed in Sec 4.2, there are various meth-         both at the graph and operator levels. These decisions range
    ods used for static sparsification. In general, structured sparsity         from choosing simple numerical or categorical parameters
    results in high speedup, but it also often results in non-trivial          to structural modifications to the program being run. The
    accuracy degradation, whereas unstructured sparsity is able to




                                                                                          21
<a id="page-22"></a>

### PDF 第 22 页

model or to directly search for the solution using blackbox-tuning.
    space of decisions required is enormous, growing combina-
                                                           Although such approaches can potentially learn the scheduling
    torially with each possible decision, but selecting a good
                                                                   space accurately, their computational cost is significant due to both
    point in the space can significantly affect performance.
                                                                     the cost of evaluating enough schedules to learn a model as well as
                                                                     the cost of learning based algorithms. As a result, these approaches
                                                                         typically apply to existing hardware or analytical models where
5.3  Finding Performant Mappings                         large-scale measurement is feasible.
                                                                       Constrained-optimization approaches contrast with exhaustive
In order to deal with the size of the search space, many accelerator-
                                                                     search and learning-based algorithms, in that they formulate sched-
aware mapping techniques [84, 92, 102, 129, 163, 239] and fully-
                                                                      uling problems as a numerical optimization problem to determine
fledged compilers frameworks [3, 32, 33, 114, 161, 167, 185] have
                                                                       variable assignments subject to a given set of constraints and ob-
been developed. These are briefly discussed below.
                                                                            jective functions. Popular techniques, such as Mixed Integer Pro-
5.3.1  Mapping Strategies. To deal with the size of the search     gramming (MIP), have demonstrated their applicability to solve
space, mapping algorithms focus on a subspace of the mapspace,       large-scale and complex problems. In particular, polyhedral transfor-
only making decisions about how to perform a subset of steps      mation has leveraged constrained-optimization based approaches
required to map the network onto the architecture.                     for auto-vectorization and loop tiling [5, 6, 15, 21, 75, 116, 165].
                                                            These polyhedral optimizations focus on testing the feasibility of
  Graph-level Schedulers. Most of existing ML compilation frame-
                                                               a transform and offering information to guide iterative searches.
works (e.g., XLA [185], TensorRT [161], TVM [33], Glow [184] and
                                              On the other hand, [92, 129] leverage the regularities in the ML
CuDNN [41]) target graph-level optimizations such as operation
                                                                workloads and hardware to formulate the mapping as optimization
fusion, resource allocation, graph partitioning, graph rewriting, etc.
                                                                    problems, which can then be directly solved by off-the-shelf solvers.
A large number of operation fusion techniques [10, 247, 252] have
been developed to optimize the mapping for more data reuse across
                                                 Summary (Sec. 5.3. Finding Performant Mappings): ADNN layers. Among these, a few Transformer-specific operation
                                                                 comprehensive set of strategies has been developed to ad-fusion techniques have been proposed [42, 168]. In particular, [42]
                                                                          dress the challenge of mapping DNNs on accelerators anddecomposes the Softmax layer and dynamically fuses the GPU ker-
                                                                      general-purpose processors. The techniques originally de-nels for decomposed layers with the proceeding and succeeding
                                                                    veloped to target CNNs can be applied to Transformers asmatmul layers in the MHA block. Relatedly, [168] shows that fusing
                                                                      the key operations are also tensor algebra operations. AtLayerNorm layers and composing big matmul from small matmuls
                                                                       the graph level, operator fusion is an important optimiza-are beneficial to transform performance on GPUs. In [104], an op-
                                                                              tion technique that encodes a vast mapping space to decidetimized dataflow for DNN accelerators is introduced to efficiently
                                                    how the execution of layers are overlapped. At the opera-fuse the key matmul and Softmax layers in MHA. To learn about
                                                                              tion level, mapping strategies can be broadly categorized as
the operation fusion tradeoffs in the Gemmini accelerator, we have
                                                                               either search strategies - either random or feedback-drivenperformed a case study and included the analysis in Sec. 5.5.
                                                                                           - or optimization or heuristic strategies.
   Operation-level Mappers. In Sec. 5.2.2, we discussed that the
decisions around tiling, dataflow, and spatio-temporal mapping can
result in an enormous search space, and that selecting a good point     5.4  Performance Modeling of Mappings
in the space is key to achieving high efficiency and utilization in ML                                                             Performance models can provide the mappers with performance
accelerators. Within the scope of a given subspace, mappers can gen-                                                                feedback for different mappings without executing mappings on
erally be divided into three general categories, based on how they                                                                            real hardware or running simulations on accelerators under de-
make their decisions: brute-force search; feedback-based search;                                                                velopment. They can significantly reduce the evaluation costs for
and constrained optimization. Tab. 6 summarizes existing mappers                                                            mappers and be used as performance proxies to optimize mappings.
that leverage different techniques to navigate the mapspace.                                                                         Different performance models offer different levels of fidelity, run-
   Brute-force methods [29, 48, 163, 239] entail various sampling                                                                time costs, target workload scopes, and compatibility for various
strategies that either exhaustively explore or randomly sample a                                                        mapping algorithms. The selection of the performance model is
large number of points from the mapspace. To lower the exhaustive                                                               both mapper and target workload dependent.
search cost, mappers in this category typically rely on developer                                                                  For Transformers, the mappers can use domain-specific poly-
heuristics to prune the mapspace and lightweight performance                                                             nomial [92] and analytical models [122, 146, 153, 163] to provide
models to compare all valid mappings to find the best mapping in a                                                                                 fast comparisons among mappings. These models leverage known
reasonable amount of time. The disadvantages of this approach are                                                                             iteration space bounds in tensor algebra workloads, as well as stati-
two-fold: not only does a brute-force search tend to be exceedingly                                                                              cally analyzable data access patterns, to estimate performance. The
expensive, especially for more complex target workloads and hard-                                                              polynomial models expressed in mathematical forms can also be
ware architectures; but also this costly process repeats for any target                                                               used directly as the objectives in optimization-based mappers.
workload or accelerator architecture changes, without leveraging                                                              Another class of popular performance models involves data-
any prior knowledge.                                                                     driven ML models [33, 84, 106]. Instead of building the performance
   Feedback-driven approaches use ML algorithms or other statisti-                                                         model analytically to express known relations between mapping
cal methods [7, 33, 99, 179] either to improve the accuracy of the cost




                                                                                          22
<a id="page-23"></a>

### PDF 第 23 页

Search Strategy           Mappers

   Brute-force & Random Approaches:    Timeloop [163], dMazeRunner [48], Flexflow [99], Triton [214], Interstellar [239], Marvel [29]

      Feedback-based Approaches:      AutoTVM [33] (XGBoost), Ansor [249] (beam search), Halide [179] (beam search [7], OpenTuner [11,
                                             158]), FlexFlow [99] (MCMC), ConfuciuX [101] (RL), Gamma [102] (genetic algorithm), Mind
                                 Mapping [84] (gradient-based search)

 Constrained Optimization Approaches:   Polly+Pluto [20, 21, 75], Tensor Comprehension [216], Tiramisu [14], IOOpt [162], Analytical
                                         characterization [129], CoSA [92]

                            Table 6: State-of-the-art DNN schedulers for heterogeneous accelerators.



decisions and performance, these models use statistical techniques      executable code by representing these points as a series of rewrite
to iteratively fit a model to the mapping performance data collected       rules [158, 249].
over time. They typically require large amounts of data in order to
learn and provide accurate predictions. Once trained, they can be       Summary (Sec. 5.4. Performance Modeling of Map-
easily integrated with ML-based mappers [84, 199].                       pings): Performance estimation of Transformers running
  The major drawback of prior models is that the generated map-        on novel architecture is essential for finding an optimal
pings might not perform optimally (or even well) on the actual          algorithm, mapping, and hardware combination. There are
accelerator since the models can fail to capture the implementa-         various open-source performance models available to esti-
tion differences in the hardware accurately. Cycle-exact software         mate the mapping performance on hardware, ranging from
models based on real hardware implementation can achieve higher          domain-specific analytical models and data-driven ML mod-
fidelity [187, 218], as can FPGA emulation using platforms such as            els to cycle-exact models. The selection of the performance
Firesim [105], which can be used to model hardware in develop-        model for Transformer depends on the target workload
ment. However, such platforms require more than a set of problem            size, hardware complexity, and development stage. Addi-
dimensions and a description of mappings; they require a stream            tionally, there are many mature code generation tools one
of explicit instructions.                                              can leverage to optimize the Transformer design for off-the-
   Generating this stream of instructions requires that one account           shelf hardware.
for a large number of edge cases. For example, a simple tiling op-
eration for a matmul −representable as a single line of tile sizes
in Timeloop [163] −requires both the insertion of instructions     5.5  Transformer vs CNN Mapping
specifying memory movement between different levels of the mem-                                                                         Prior work discussed in Sec. 5.3 for finding good mapping strategies
ory hierarchy as well as the generation code for edge cases that                                                                        largely focuses on mapping CNNs onto accelerators or general-
appear when matrix dimensions are not evenly divisible by the tile                                                              purpose hardware. As with CNNs, the vast majority of cycles for
size. Furthermore, the codesign process requires this process to be                                                                Transformers are spent on matmuls from the MHA and FFN mod-
automatable. In other words, each mapping must be translatable to                                                                              ules. In essence, existing mappers for CNNs can easily extend to
code automatically.                                                                   scheduling Transformer matmuls. However, as we have discussed
  As a result, code generation [33, 126, 180] tools are used to actually                                                                        in Sec 3.4.2, Transformer blocks include LayerNorm and Softmax
implement mappings onto hardware (or simulators). Many of these                                                                      operations, which can be computationally non-trivial in certain
tools integrate not only a specification of the hardware backend but                                                                                  realistic scenarios (which was also observed by [42, 168]). In turn,
also mapping decision algorithms, often tuned for that hardware                                                                     the presence of these operations imposes constraints on scheduling
target. Such tools can also be useful for neural architecture search                                                                     the preceding and succeeding matmuls. This leads to a much more
(NAS) in order to obtain accurate performance numbers for a given                                                           complex problem for scheduling optimizations for Transformers
hardware architecture to guide automated DNN architecture design                                                                              overall. In this subsection:
(see Sec. 6.1 for details). However, these tools are difficult to adapt for
                                                             • We characterize the mapspace of Transformer blocks in com-a codesign framework where both the mapspace and the hardware
                                                                       parison to that of CNNs (Sec. 5.5.1).target can vary.
                                                             • We take a deeper dive into the issue of increased scheduling   In order to address this problem, user-schedulable languages such
                                                                    complexity of Transformer matrix operations due to the pres-as Halide [179], TVM [33], Rise/Elevate [78], and Exo [95] have
                                                                   ence of LayerNorm and Softmax (Sec. 5.5.2).been developed. These tools take as input a description of the com-
putation to be performed and a point in the mapspace. They are
generally defined by a set of rewrite rules, representing certain       5.5.1  Mapspace Characterization of Transformers. We em-
transformations such as splitting and rearranging loops, replacing       pirically characterize the search space of legal mappings for repre-
appropriate loops with ISA instructions, and fusing loops. These       sentative Transformer and CNN architectures. To do so, we chose
languages also allow the user to specify and customize the hard-    BERT [52] and ResNet50 [82], respectively. A total of 100K random
ware instruction set and seamlessly convert found schedules into       valid mappings were searched via the Timeloop mapper [163], and
                                                                    the estimated latency and energy were measured by the Timeloop




                                                                                          23
<a id="page-24"></a>

### PDF 第 24 页

2000                                                                 4000
                                                         BERT-Large MHA                       ResNet50 7x7 64
     1750                                             BERT-Base MHA            3500       ResNet50 1x1 512
                                                       BERT-Base FFN                        ResNet50 3x3 512
     1500                                                                 3000                                                         BERT-Large FFN                       ResNet50 1x1 2048
     1250                                                                 2500

     1000                                                                 2000    Count                                                                                                                  Count
      750                                                                 1500

      500                                                                 1000

      250                                                                  500

        0                                                                    0            100          101          102          103          104                       100         101         102         103         104
                       EDP Relative to Minimum                                          EDP Relative to Minimum

Figure 18: A comparison of the mapspace of (Left) BERT and (Right) ResNet50. Distributions of 100K randomly sampled valid mappings are
shown. Both distributions show a similar range of EDP of up to four degrees of magnitude difference with respect to the best (minimum) observed
value. Neither distribution is significantly more skewed towards lower or higher relative EDP. Overall, we find that mapspaces for BERT matmuls
and ResNet50 convolutions are similarly vast in size with no significant difference in the shapes of their distribution. This indicates that brute-force
or random search for BERT matmul scheduling is equally as challenging as in the case with ResNet50 operators.


    0.200                                                    which takes part in expanding the hidden dimension to four times
                BERT-Base MHA                                                         its value.
    0.175       BERT-Large FFN                                                        From ResNet50, we choose convolution operations of varying                ResNet50 7x7 64
    0.150       ResNet50 1x1 512                                          kernel sizes and shapes, spanning 1 × 1, 3 × 3, and 7 × 7 convolution
                ResNet50 3x3 512                                             kernels. Specifically, we use the 7×7 kernel convolution with output    0.125
                ResNet50 1x1 2048                                                                 channel size 64 and stride 2 in the conv1 layer of ResNet50, the 3×3
 CDF0.100                                                              kernel convolution with 512 channels and 1×1 kernel convolutions
    0.075                                                       with output channel sizes 512 and 2048 that belong to the final
                                                                    convolution layer conv5_x. These four particular convolutions vary    0.050
                                                                         in channel and kernel sizes and reasonably represent convolutions
    0.025                                                              found in ResNet50.
    0.000                                                  From our mapspace analysis, we observe that both BERT and            2       4       6       8      10      12      14
                                    Relative EDP                          ResNet50 mapspace have a similar range of potential energy-delay
                                                                 product values from randomly sampled mappings. Accounting for
Figure 19: Comparison of empirical cumulative distribution func-
                                                                          variations between operators for each architecture, the distribution
tions (CDFs) for Transformer matmuls and ResNet50 convolutional
                                                                       of EDP values for themselves are also largely similar, in that the
operations around the regime where near-optimal mappings are found.
                                                                   portion of Pareto-optimal mappings with lower EDP values are
The 10th percentile values for each relative EDP distribution are: 7.44
                                                                    small for both BERT and ResNet50 operators.for the 3 × 3, 512 convolution kernel; 12.06 for the 1 × 1, 2048 kernel;
                                                        As an alternative visualization, Fig. 19 compares the empirical
9.42 for the BERT MHA matmul; and 9.84 for the BERT FFN matmul.
                                                                cumulative distribution functions (CDFs) of the same set of 100K
Results show that the percentage of near-optimal mappings for BERT
                                                        random mappings. Here, we closely examine the difference in the
matmuls are similar if not smaller than that of ResNet convolution
                                                    CDFs near the regime where near-optimal mappings are found.
kernels. This indicates that the search problem of finding optimal
                                             We observe that the relative EDP value corresponding to the tenth
mappings can be as challenging for BERT matmuls.
                                                                       percentile is 7.44 for the 3 × 3, 512 convolution and 12.06 for the
                                                               1 × 1, 2048 convolution kernels. BERT matmuls for MHA and FFN
                                                                        projection had tenth percentile values of 9.42 and 9.84, respectively.model. The target spatial hardware architecture is the Gemmini
                                                                               Alternatively, we also examine the percentage of mappings withsystolic generator [70]. Both the BERT and ResNet50 models are
                                                                              relative EDP values less than 3 times the observed minimum. Thisassumed to have been 8-bit integer quantized. We assume an input
                                                                   percentage represents a rough upper bound on the number of map-sequence length of 512, which is a typical assumption for BERT-
                                                                   pings that can be safely labeled as near-optimal, and the differenceBase and BERT-Large models.
                                                                         in percentages signify relative difficulties in searching for optimal
                                                        mapping for different operators. We find 1.58% for the 3 × 3, 512
  EDP Distribution Analysis. Fig. 18 demonstrates the compar-                                                                   kernel and 2.62% for the 1 × 1, 2048 kernel with mappings within
ison between the mapspaces for BERT (Left) and ResNet50 (Right).
                                                                               this range of EDP. In the case of BERT, the MHA projection matmul
For the BERT MHA module, results in the figure correspond to map-
                                                                                         is shown to have 1.70% of mappings with relative EDP less than 3
pings for the matmul performing each of the query, key, and value
                                                               with 1.48% of mappings in this range for the FFN matmul.
projections. For the FFN, we take the matmul in the 𝑊1 projection,



                                                                                          24
<a id="page-25"></a>

### PDF 第 25 页

2000                                                                            5.5.2  Scheduling Complexities of LayerNorm and Softmax.
                                                    ResNet50 7x7 64         While we find that matmuls in Transformers are already non-trivial
    1750                                         BERT d=240 MHA
                                                                           targets for which to obtain efficient execution schedules to execute                                                 BERT d=120 FFN
    1500                                                 on DNN accelerators, the problem is further complicated by the
    1250                                                            presence of several non-linear operations, including LayerNorm and
                                                             Softmax that are interposed between different matrix operations.
    1000  Count                                              When pursuing more aggressive optimizations, an enticing strategy
     750                                                                              is to fuse relatively high-arithmetic-intensity matmuls with the
     500                                                                low-arithmetic-intensity normalization operations following them,
                                                             such as LayerNorm and Softmax. This can be especially enticing
     250                                                                       in handling quantized workloads, where partial sums awaiting
       0                                                           normalization are often of much higher bitwidth than the final           100         101         102         103         104
                      EDP Relative to Minimum                       normalized outputs. Architects familiar with CNN-type accelerators
                                                   may find this especially intuitive, since convolutions are often fused
Figure 20: Distributions of random, valid mappings for ResNet50                                                               with ReLU or max-pool operations.
operators, compared against distributions for Transformer MHA and                                                                              Similarly, for Transformer Encoders, we could overlap the ex-
FFN matrix multiplications, where the size of matmuls are tuned so                                                                   ecution of normalization operation and the previous matmul, yet
that total MACs are equivalent. The MHA matmul dimensions were                                                                             this is possible only with additional hardware support and appro-
calibrated to 240 × 240 × 512 and similarly for the FFN 𝑊1 projection       priate constraints on the matmul execution schedule. To enable
matmul, set to 480 × 120 × 512 for 𝑑FFN = 4 × 𝑑. Comparing against      complete latency-hiding of nonlinear operations, the tiling factor
Fig. 18, we see that the EDP distributions are still similar for the                                                                              size of either output dimension of the matmul must be maximized,
ResNet convolution and BERT matmuls. This implies that, even after                                                                 so that rows/columns are immediately ready and stored at the
accounting for differences in the total number of MACs, the mapspace                                                     Gemmini accumulator scratchpad for computing the mean and
of BERT matmuls exhibit as vast a range of relative EDPs as the                                                                   standard deviation. We refer to this alternate scheduling approach
mapspace of CNN convolution kernels.                                                                      as fusion-optimized scheduling.
                                                On the other hand, in memory-constrained edge devices, the
                                                                       strategy is (somewhat unintuitively) counter-productive. The nor-
                                                                   malization operations found in Transformers often require long   Overall, the analysis of EDP distribution from randomly sampled
                                                                     vectors of data to be resident in on-chip local memory before anyvalid mappings indicates that BERT matmuls, despite having fewer
                                                               normalized output element can be produced. Furthermore, whenloop levels for tiling and re-ordering compared to convolutions,
                                                                       fusing matmuls with these normalization operations, awkward mat-are as challenging to schedule as CNNs. As much as graph and
                                                       mul tile shapes are typically required. These awkward tile shapesoperator-level scheduling had a significant impact on end-to-end
                                                                    are often much larger in either dimension, as opposed to beingperformance and efficiency of CNN inference, the same impor-
                                                                     square-shaped, and such skewed tile shapes tend to yield far worsetance of appropriate scheduling also applies to Transformer matrix
                                                                     arithmetic intensity. This greatly reduces the performance of theoperations.
                                                                matmuls, and it may increase the total memory traffic, even account-
  EDP Distribution Analysis with Fixed Total Number of MACs.    ing for the high bitwidths of the unnormalized partial sums which
As an additional analysis on mapspace characterization, we further     must be sent to and from outer memory when fusion is not enabled.
force the total number of MACs to be fixed. This enables an even         In Fig. 21, we take a deeper look at the performance implications
fairer comparison between the distributions of mapping results for       of fusion-optimized scheduling for BERT matmuls. We consider the
Transformer and ResNet50 operators. We continue to assume the     BERT-Base encoder with hidden dimension 768 and 12 attention
Transformer input sequence length to be 512 and the feed-forward      heads. By default we assume a sequence length of 512. As the target
network expansion ratio of 4 times the hidden dimension size. To      hardware, we take the 16×16 weight-stationary systolic array Gem-
keep the number of MACs equal, we calculate the hidden dimen-     mini with custom hardware units for I-BERT implementations of
sion size that would yield the same total MACs as for the ResNet50      nonlinear operations (activation, normalization, etc.), as described
conv1 layer’s 7×7 kernel with output channel dimension 64. For       in Sec. 3.4.4. The total latency of each adjacent pair of matmul and
the matmul in the query projection of MHA, the corresponding     LayerNorm/Softmax operations is estimated via Timeloop[163].
hidden dimension size was 240. Similarly, for the matmul in 𝑊1      Opportunities for overlapping computations include: (1) the MHA
projection of the FFN block, the corresponding hidden dimension     query × key matmul and following Softmax; (2) MHA 𝑊out pro-
size was 120. To elucidate the comparison between synthetic BERT       jection and following LayerNorm; and (3) FFN 𝑊2 projection and
layers and actual ResNet50 convolutions, we plot the corresponding      following LayerNorm.
pairs mapping distributions in Fig. 20. Even after forcing equivalent      We compare two scheduling strategies. In the first strategy, we
numbers of MACs, we see that the range of relative EDP values are      use Gemmini’s default heuristic-based scheduler, which greedily
similar between BERT matmuls and ResNet50 convolutions. This     maximizes loop tile factors at the local SRAM level for each of
finding further accentuates how complex the scheduling problem      the three matmul dimensions. In this approach, we do not attempt
can be for matmuls found in Transformer models.                      to overlap the matmul computation with the following nonlinear
                                                                        operation, meaning that the matmul is scheduled independently as




                                                                                          25
<a id="page-26"></a>

### PDF 第 26 页

BERT-Base MHA Latency Breakdown                                         BERT-Base MHA Latency Breakdown
               (Sequence Length = 512, Varying Accumulator Size)                         (Sequence Length = 4096, Accumulator Size = 256kB)

                5                                                                                         250
                  1e6) 4                                                                                                                       1e6) 200

                3                                                                                         150                                    (Cycles,                                                                                                                                                                                                                                                  (Cycles,

                2                               Latency                                                                                                                                                                                                                                                          Latency 100
                1                                                                                          50

                0                                                                                           0
                    Non-Fused   Fusion-Optimized   Non-Fused   Fusion-Optimized                                          Non-Fused            Fusion-Optimized
                      (128kB)        (128kB)        (256kB)        (256kB)

      MHA (Q x K)      MHA (Softmax)      MHA (Wout proj.)      MHA (LN)         MHA (Q x K)      MHA (Softmax)      MHA (Wout proj.)      MHA (LN)
Figure 21: Impact of fusion-optimized scheduling for BERT MHA that enables latency hiding of LayerNorm and Softmax. Results are based on
the BERT-Base architecture and the Gemmini accelerator. (Left) Input sequence length is assumed to be 512, and the accumulator SRAM size is
increased from 128kB to 256kB. Hiding the Softmax latency improves combined matmul and Softmax latency by 78%. However, overlapping 𝑊out
projection with LayerNorm can either hurt or improve total latency, depending on the accumulator size. Overall, fusion-optimized scheduling for
both matmuls in MHA yields 23% and 52% latency improvements for accumulator sizes 128kB and 256kB, respectively. (Right) The input sequence
length is increased to 4096. Again, we see that overlapping the query × key matmul with Softmax improves latency by 22%. Overall, fusion of
both MHA matmuls with nonlinear operation yields a 21% latency improvement.

                 BERT-Base FFN Latency Breakdown                      regardless of accumulator size. In particular, we see that Softmax
         (Varying Sequence Length, Accumulator Size = 256kB)                                                                    latency is significant compared to the matmul, taking up around
        50                                                78% of the total cycles, and hiding this latency significantly re-
                                                               duces total latency. At the same time, the query × key matmul
       1e6) 40                                                            latency is relatively unchanged by the additional scheduling con-

        30                                                  (𝑑/𝑙= 64 for BERT-Base). On the other hand, the mapping con-              (Cycles,                                                                       straints, mainly because inner dimension of the matmul is small
                                                                            straints from fusion-optimized scheduling significantly harm the
        20
                                                                   execution latency of the 𝑊out projection matmul after fusing with            Latency
                                                                    the following LayerNorm, resulting in 83% worse latency than the        10
                                                                  non-fused schedule. However, once the accumulator size is doubled,
         0                                                           the performance hit on matmul scheduling is alleviated. Increased
             Non-Fused   Fusion-Optimized   Non-Fused   Fusion-Optimized
             (seqlen 512)     (seqlen 512)    (seqlen 4096)   (seqlen 4096)            accumulator SRAM size of 256kB allows more partial sums to be
                                                                      stored in the buffer instead of spilling to DRAM, thereby reducing
                  FFN (W2 proj.)        FFN (LayerNorm)
                                                                               total latency by 4%.
Figure 22: Impact of fusion-optimized scheduling for BERT FFN         In the right plot of Fig. 21, we further investigate the impact of
matmul that enables latency hiding of the LayerNorm operation.      sequence length on fusion-optimized scheduling for the MHA block.
Input sequence length is varied from 512 to 4096. We observe that      Here, the sequence length is increased from 512 to 4096, which
fusion-optimized scheduling hurts total latency by 27% in both cases.      impacts the ratio of cycles from matmuls, Softmax, and LayerNorm
This motivates the need to carefully evaluate the impact of chaining       in the MHA block. In particular, note that the size of the query × key
matmul and LayerNorm execution on systolic arrays since the impact     matmul and the Softmax computation depends quadratically on the
of mapping constraints may outweigh the gains from latency hiding      sequence length, while the other matmul and LayerNorm exhibit
of nonlinear operations.                                           a linear dependence. When fusing the query × key matmul with
                                                                    the subsequent Softmax, the mapping constraints worsen matmul
                                                             performance despite a larger (256kB) accumulator size. This is
if it were executed on its own. In Fig. 21, we denote this approach      because with increased dimensions of the query × key matmul and
as non-fused scheduling. The second strategy is the aforementioned      forced tiling factors, the scheduler can no longer avoid tiling at
fusion-optimized scheduling.                                        the DRAM level. However, by overlapping the Softmax operation
  The left plot of Fig. 21 summarizes how matmul and nonlin-     and thereby eliminating the need to load and store intermediate
ear operation fusion within the MHA block can be influenced by      activations (which quadratically scales with the sequence length),
the accumulator SRAM size. In this experiment, while the on-chip      the latency increase from the query × key matrix can be offset,
scratchpad for input activation and weights is held fixed at 256kB,      resulting in an overall 22% reduction in latency.
the output activation accumulator size is increased from 128kB to      On the other hand, Fig. 22 shows the results on matmul and Lay-
256kB. We note two findings: first, within MHA, fusing query ×     erNorm overlapping in the FFN 𝑊2 projection. Even with a larger
key matmuls with Softmax for each attention head reduces latency      accumulator size and in both sequence lengths, we consistently




                                                                                          26
<a id="page-27"></a>

### PDF 第 27 页

observe that fusion-optimized scheduling worsens total latency       will provide a general overview of NAS, and then Sec. 6.2 will ex-
by 27%. Together with previous findings, we see that latency im-      plore hardware-aware NAS methods. These two subsections will be
provements of fusion-optimized scheduling are dependent on the      mainly focused on NAS techniques for CNNs, as NAS was initially
accumulator SRAM size and sequence length. Furthermore, we find      introduced and extensively researched from the pre-Transformer
that, in the BERT-Base scale, it is consistently favorable to overlap       era. However, we believe it is helpful to provide a comprehensive
the MHA query × key with the ensuing Softmax but consistently     overview and background to understand NAS. In Sec. 6.3, NAS
disadvantageous to chain the FFN 𝑊2 projection matmul with Lay-     methods specific to Transformer architectures will be discussed.
erNorm. This is in contrast with previous studies on GPU kernel       Finally, in Sec. 6.4, a case study of applying NAS method in the
fusion for Transformers [42, 168], and it highlights how scheduling      scenario of optimizing Transformer inference on a target hardware
for Transformer matmuls becomes more complex when targeting       architecture will be provided.
different styles of custom hardware designs, including the Gemmini
accelerator.
                                                         6.1  Neural Architecture Search
   Summary (Sec. 5.5. Transformer vs. CNN Mapping):          Typically, DNN architectures are designed and trained to achieve
   Here are the high-level takeaways from this section.               the maximum accuracy for a given task, without necessarily consid-
    • Scheduling for Transformer matmuls is as challeng-          ering the target hardware or inference latency, memory, and power
       ing as scheduling CNN convolution operators. Both          requirements. However, often there exist several different varia-
      mapspaces have similar distributions of relative EDPs          tions of the DNN architecture which result in the same accuracy
      and similar percentages of near-optimal mappings.         but have better hardware performance.
       Brute-force or random scheduling is not simpler for           There is a rich literature in this area. Notable works here in-
       Transformer matmuls, despite them having fewer loop          clude MobileBERT [205], which is one of the earliest attempts, and
        levels than convolutions.                                which adopts the bottleneck structure to design a thinner version
    • The presence of nonlinear operations such as Layer-          of Transformer, as well as Lite Transformer [232], which proposes
     Norm and Softmax present additional complexities to          the Long-Short Range Attention, in which a group of heads are re-
       the scheduling problem for Transformer matmuls. La-          placed with convolution operations to capture short-range contexts
       tency of these nonlinear operations can be hidden by        more efficiently. SqueezeBERT [94] is another work that incorpo-
       fusing its computation with the preceding matmul.          rates grouped convolutions into the Transformer architecture to
       This requires additional hardware support, as noted         reduce the model size and latency. This approach is not limited to
        in Sec. 3.4.4, and it imposes constraints to the matmul        NLP, and similar models have been proposed in computer vision
       scheduling.                                               (CV) [24, 36, 130, 152] and speech recognition [23, 76, 109], to name
    • Whether this fusion-optimized scheduling yields end-           just a few.
       to-end latency improvements depends on the Trans-                 It is often very difficult to find these DNN architectures since
       former and underlying hardware parameters. In par-         the search space is exponentially large, even without considering
         ticular, we observe that: (1) size of the on-chip SRAM          the underlying hardware platform. Even for those with expertise in
        buffer for output activation; and (2) sequence length      DNN architecture design, the impact of an architectural change on
        matter.                                                     accuracy and runtime performance can be nontrivial to predict. As
    • We consistently observe that overlapping the execu-          such, automated NAS methods have been proposed to adapt a DNN
        tion of query × key matmul with Softmax in the MHA          architecture for a given constraint. However, it is critical to note that
       block reduces latency up to 78%, compared to execut-      NAS methods often require prohibitive amounts of compute and
       ing the two operations separately on a systolic array            trials before finding a candidate architecture. For instance, in one of
        accelerator. On the other hand, scheduling to overlap          the early NAS works [255], finding an optimized CNN took 22,400
       the FFN 𝑊2 projection with the following LayerNorm         GPU-hours. Moreover, NAS methods are not yet fully automated,
       hurts performance by 27%.                               and they often require hand-tuning the search space.
                                                                    Broadly speaking, a NAS framework consists of three main com-
                                                                   ponents: search space; search method; and evaluation method [18,
6  ADAPTING TRANSFORMER                               61]. The search space consists of a set of valid operations (e.g., con-
  ARCHITECTURE WITH NAS                            volution, pooling, activation, etc.) and their connectivity that define
                                                                        valid DNN architectures, from which a candidate model can be
So far, we have conducted an in-depth exploration of the full-stack
                                                               drawn. Prior knowledge and human intuition regarding good DNN
aspect of DNN inferencing, with a focus on the Transformer archi-
                                                                    designs is often necessary in order to restrict the search space and
tecture, from the hardware level to optimization and scheduling
                                                            improve the search efficiency. The search method defines how to
strategies to improve their inference performance. Another im-
                                                                     explore the search space. Exhaustive search is obviously intractable.
portant avenue in full stack optimization of DNNs is obviously
                                                                     Therefore, it is critical to have methods for quickly exploring the
to optimize DNN architecture itself and to tailor it for a specific
                                                                  search space and sampling candidate architectures. The evalua-
hardware platform.
                                                                             tion method is a way of assessing how well candidate architectures
   In this section, we will primarily focus on automated neural
                                                              perform on unseen data. The most naive method is to evaluate all
architecture search (NAS) as a method for designing DNNs. Sec. 6.1
                                                                   candidate architectures after the full training process is complete.




                                                                                          27
<a id="page-28"></a>

### PDF 第 28 页

in valid DNN architecture, and therefore it has been widely adopted                  Search Space
                                                                         in follow-up works [142, 182]
  Search              Conv 3x3       Sample                     Evaluation
 Method             Conv 1x1                            Method
                        ReLU                                                         6.1.2  Search Method. Since the NAS search space is usually too
                      …
                                             Candidate         Evaluate             large for an exhaustive search, efficient search methods are neces-
                                                Architectures                        sary to ensure overall performance. In early work on NAS, RL-based
                                                          methods were used as the search method [16, 170, 250, 254, 255]
                                                                                    (Fig. 24, a). At a high level, RL-based NAS frameworks contain the
                     Improve Search Quality                    Accuracy  HW       controller (i.e., RL agent) that takes an action of sampling DNN
                                                                            metrics
                                                                           architectures, whose evaluation accuracy after training is fed into
                                                                    the controller as a reward signal to refine its sampling policy. The
Figure 23: Illustration of the general structure of NAS frameworks.
                                                                        controller can be trained using different RL algorithms such as
Candidate DNN architectures are sampled from the search space
                                                                       policy gradient [254] or Q-learning [16].
according to the search method, and then they are evaluated. The
                                                  An alternative search strategy for NAS is evolutionary search [141,
evaluation result is then used by the search method to guide better
                                                                    182] (Fig. 24, b). Here one initializes a population of different DNN
exploration of architectures in the search space.
                                                                            architectures, which are then mutated (e.g., by adding, removing, or
                                                               changing layers), evaluated, and selected based on their validation
                                                                  accuracy in every evolution step. This generates a new population
However, this incurs a large overhead, and more efficient methods       for the subsequent evolution step. The search cost for evolutionary
of estimating performance are often used as a proxy for the final      search can be quite expensive, as it requires validating all DNNs
accuracy. Fig. 23 schematically shows these different components.      in the population for every evolution step. Therefore, it is often
   Below, we discuss each of these components in more detail. Note      coupled with various methods to reduce validation costs such as
that the main purpose of this section is not to conduct a thorough      weight sharing. These will be discussed in more detail in Sec. 6.1.3.
survey of existing works, but instead to provide a broader overview       The aforementioned methods can be regarded as a black-box
on various methodologies for improving NAS from a practitioner’s      optimization problem over a discrete search space. Due to the dis-
standpoint. We refer readers to [18, 61, 183, 193] for more compre-      crete nature of the search space with a large number of tunable
hensive survey on NAS.                                            knobs, the search cost can become prohibitively large. This is fur-
                                                                     ther exacerbated by the long evaluation time of a single RL or
                                                                      evolution iteration, which often requires training from scratch. For
6.1.1  Search Space. The search space for NAS defines a set of
                                                                        instance, RL-based NASNet [255] and evolutionary search-based
valid DNN architectures over which the NAS framework can search.
                                                        AmoebaNet [182] require a few thousands of GPU hours for end-to-
Designing a proper search space is critical, as its size and cover-
                                                            end search [183]. In contrast, DARTS [142] proposes the continuous
age can directly affect the final outcome of the NAS framework.
                                                                        relaxation of the search space, which allows them to efficiently
One naive principle of designing a search space is the layer-wise
                                                                    explore and optimize the search space through gradient-based op-
search [26, 77, 254] where each layer (or operation) can be searched
                                                                     timization methods (Fig. 24, c). In essence, DARTS introduces a
independently from other layers. For instance, in [254], the RNN
                                                                         trainable weight to allow for a weighted average of multiple oper-
controller model produces a description of individual layer in a
                                                                           ations, instead of requiring a selection of a single operation. This
sequence to construct a candidate DNN architecture.
                                                              weight can be trained alongside other model parameters during
  However, the layer-wise search space often suffers from the large
                                                                            training, and it can eventually converge to favor a particular oper-
search space size that grows exponentially with the depth of candi-
                                                                     ation over the others. This method reduces the search cost from
date architectures, and this could degrade the search efficiency and
                                                                 thousands of GPU hours in the preceding RL or evolutionary search
the final performance. The cell-wise search [54, 141, 170, 250, 255]
                                                               based methods to a few hours. Due to the search efficiency, the
can alleviate this shortcoming by searching cells (i.e., blocks or
                                                                       gradient based search has become a popular choice for many NAS
modules that consist of multiple layers) rather than an entire archi-
                                                            frameworks [219, 228].
tecture, which can later be stacked up repeatedly to compose an
architecture. This is motivated by many successful hand-designed
DNN architectures that consist of repeating cells or blocks of a       6.1.3  Weight Sharing and Supernetwork. One of the main
similar structure [82, 89]. NASNet [255] is one of the earliest works      challenges with NAS methods is the prohibitive training cost. To
that proposes to search two types of cells: the normal cell, which      address this, ENAS [170] proposed weight sharing. ENAS views a
stacks up multiple times without changing spatial resolution; and    DNN model as a directed acyclic graph, where the nodes represent
the reduction cell, which is inserted once every fixed number of      the computation with their own trainable weights and the edges
repeated normal cells, in order to reduce the spatial dimension to-      represent the information flow from one node to another. Then, an
wards the output layers. This significantly reduces the search time      individual candidate DNN can be regarded as a sub-network of a
by 7× compared to the previous layer-wise search method proposed       larger, over-parameterized super-network (supernet). This redefines
by the same authors [254]. Likewise, the cell-wise search space sub-    NAS as a process of searching for good sub-networks sampled from
stantially reduces the search space (as cells are much smaller than      the supernet whose weights are shared across all sub-networks.
the whole network) by imposing an additional structural constraint     Once the supernet is trained, its sub-networks can be sampled and




                                                                                          28
<a id="page-29"></a>

### PDF 第 29 页

Population                           Population
                                                                                                                                                                                                                                                      Accuracy
        Sample      Evaluate                                                                        3x3      1x1     3x3      1x1 ≈   3x3
                                      Mutate       Evaluate                Select
  Policy                                                                 FLOPs                         x0.6        x0.4    x0.9        x0.1


              Reward                                                                      Before Training    After Training
                                                                  Reinitialize Population


          (a) RL based NAS                 (b) Evolutionary Search based NAS                           (c) Gradient based NAS

Figure 24: Comparison of different NAS search methods. (a) RL-based methods employ a controller that samples architectures based on a
policy, which is reinforced by the evaluation results of the sampled architecture as a reward. (b) Evolutionary search-based methods initialize a
population, sample them based on the evaluation results, and then generate a next-round population by mutating the remaining architectures. (c)
Gradient-based methods (e.g., continuous relaxation) train weights along with model parameters that are multiplied to each operation choice.
After the training, the weights are converged to favor a particular operation over the others, thus approximating the sampled architecture.


evaluated without the need to train the models from scratch. This      to other tasks in a similar domain. This premise has been chal-
significantly reduces the overall search cost.                          lenged by some of the recent NAS work [26]. Supernet-based NAS
   This method, also known as the supernet-based NAS, was picked      algorithms can be a good alternative to avoiding the use of proxy
up by several subsequent algorithms [17, 25, 26, 77, 142, 228, 243].      tasks [17, 25, 26, 77, 142, 228, 243]. These algorithms require only
In particular, Single Path One-Shot NAS [77] constructs a supernet      a single iteration of supernet training, which can be performed
by stacking the choice blocks. The choice block consists of multiple       directly on large-scale datasets without prohibitive compute re-
operation choices (e.g., convolution with different kernel sizes or      quirements.
skip operation) from which a single operation can be selected at a
time. For every training step, a different sub-network is obtained       Summary (Sec. 6.1. NAS): Neural architecture search
and trained by uniformly sampling one operation for each choice        (NAS) is a promising alternative to hand-designing effi-
block, expecting all sub-networks with different permutations of           cient DNNs. NAS consists of: (1) a search space that defines
choices to be trained fully and equally. After training, an evolu-          valid candidate architectures; (2) a search method that de-
tionary algorithm is applied to search optimal sub-networks from           fines how to efficiently explore the search space; and (3) an
the supernet, without paying the expensive costs of from-scratch          evaluation method for evaluating the goodness of candidate
training.                                                                     architectures. Despite its potential, NAS presents its own
  However, the accuracy of sub-networks obtained from a fully-           set of challenges, which often necessitate manual tuning of
trained supernet is typically inferior to the same model architec-          the search space, and which can be prohibitively expensive
tures trained from scratch in a stand-alone fashion [17]. There-          in terms of time and resources. To address this, many recent
fore, the discovered sub-network architectures often need to be         advances in the NAS community have focused on improv-
re-trained. To address this, Once-For-All [25] proposes the progres-         ing search efficiency. Notable methodologies include: (1)
sive shrinking algorithm, and BigNAS [243] proposes the sandwich          the cell-based search that confines the search space size;
rule and in-place distillation. Both aim to train a supernet in a way            (2) the continuous relaxation of the search space that al-
that its sub-networks can achieve good accuracy (i.e., comparable         lows efficient gradient-based optimization methods; (3) the
accuracy to the from-scratch trained counterparts) without an addi-         weight sharing scheme across candidate architectures; and
tional training process. These methods can have a high value from            (4) faster evaluation methods for the candidate architecture
a practical standpoint as sub-networks can be sampled (e.g., via         performance.
evolutionary search) and immediately deployed.

6.1.4  Evaluation Method. One needs a metric to evaluate sam-     6.2  Hardware-aware NAS
pled architectures on a validation dataset to rank the “goodness” of     Hardware-aware NAS aims to optimize not only the accuracy of
candidate architectures. The early NAS algorithms [16, 254] fully    DNNs but also the various performance metrics (such as latency,
trained sampled architectures until convergence, which is not fea-     energy consumption, or memory usage) on target hardware plat-
sible for large datasets. A widely adopted strategy for applying      forms. One key question here is how to incorporate these metrics
NAS to larger-scale tasks is to discover an accurate cell architec-      into learning. It is often difficult to quickly measure the latency or
ture using a smaller dataset (e.g., CIFAR-10 in computer vision)     energy consumption of a candidate model. As such, most works
and then apply it to building a larger model for a larger dataset      in the literature only consider FLOPs or total number of parame-
(e.g., ImageNet) [140, 142, 149, 182, 210]. The premise here is that a        ters. However, as also discussed above in Sec. 2.2, FLOPs does not
DNN architecture optimized for one task can be transferred well




                                                                                          29
<a id="page-30"></a>

### PDF 第 30 页

necessarily have a correlation with latency or energy. Therefore,
                                                 Summary (Sec. 6.2. HW-Aware NAS): Hardware effi-
multiple hardware-aware NAS frameworks have been introduced
                                                                     ciency metrics can be coupled with NAS loss function to
to directly consider latency instead, or to use approximate metrics
                                                                            find an architecture that considers both accuracy as well
for measuring it (e.g., measuring latency of individual layers and
                                                                        as latency (or similar metrics). While directly measuring
accumulating them to approximate total latency, as opposed to mea-
                                                                   performance metrics on a real hardware environment is the
suring the end-to-end runtime). Here, we discuss popular strategies
                                                               most accurate method, it can be slow and poorly paralleliz-
to incorporate hardware performance into NAS frameworks. For a
                                                                                   able. Instead, the hardware performance can be estimated
more exhaustive survey on hardware-aware NAS techniques and
                                                                           in high accuracy using an operation-wise lookup table or
their algorithmic details, see [18].
                                                              by training a simple prediction model.
  The most straightforward way is to directly measure hardware
performance and bring it as an additional optimization objective
for NAS frameworks [138, 210]. For instance, MNasNet [210] ex-
tends the existing RL-based NAS framework to the multi-objective
                                                            Table 7: Summary of existing literature on Transformer-specific NAS
optimization setting. It aims to maximize accuracy, while limit-
                                                                        techniques. SPOS and OFA stand for Single Path One-Shot [77] and
ing latency on the target platform to less than a certain target
                                                                        Once-for-All [25], respectively.
latency. Instead of solving the multi-purpose optimization problem,
it combines the two objectives (accuracy and latency) into a single
                                                                                                 Search   Search     Weight
objective, by taking a weighted product. This modified objective is          Name       Domain
                                                                                               Space   Method     sharing
then provided as a reward for updating the controller. By directly
optimizing latency on the target platform, MNasNet finds DNN        Evolved Tfm. [200]                  Cell     EA        ×
                                                       HAT [222]                  Layer    EA     OFA [25]architectures that are ∼2× faster than MobileNetV2 [188] and NAS-                       NLP
                                                             NAS-BERT [235]                Layer    EA      SPOS [77]
Net [255] with a comparable ImageNet classification accuracy on a
                                                                            Primer [201]                     Cell     EA        ×
Pixel phone.
  Another notable work is MCUNet [138] that targets searching         Autoformer [230]               Layer    EA      SPOS [77]
DNNs for resource-constrained microcontrollers. Unlike GPUs or           GLiT [30]                   Layer    EA      SPOS [77]
                                                                   ViT-ResNAS [135]    CV      Layer    EA      SPOS [77]
mobile devices, microcontrollers for tiny IoT devices lack large
                                                                   NAS-ViT [73]                 Layer    EA    BigNAS [243]
memory and storage. As a result, it is critical to design a model that
                                                                      BurgerFormer [236]              Layer    EA      SPOS [77]
fits their tight memory budgets. MCUNet incorporates its supernet-
based NAS framework [17, 77] with TinyEngine, a lightweight
inference engine that the authors have developed as part of the
project. In this way, MCUNet samples sub-networks for every evolu-     6.3  Transformer-specific NAS
tionary step and feeds them to TinyEngine to optimize the memory      Early work on NAS focused on CNN models mostly for computer
scheduling and measure the optimal memory usage.                   vision applications. However, after the Transformer architecture
  However, due to the limited number of available devices, measur-     was introduced and matured enough to achieve state-of-the-art
ing hardware performance directly can be slow and not paralleliz-       results not just for NLP tasks but also for other tasks, several works
able [238]. Furthermore, it is not possible to pre-measure the hard-      started to explore NAS methods to find more efficient alternatives.
ware performance for all possible DNNs in the search space [228].     With the introduction and maturation of the Transformer architec-
To overcome this issue, some of the hardware-aware NAS methods       ture, which allowed for state-of-the-art results on a variety of tasks,
incorporate operation-wise lookup tables [219, 228, 238]. Rather      a number of recent works have begun to explore the use of NAS
than storing the end-to-end hardware performance, the lookup ta-     methods to find more efficient alternatives. As the Transformer
ble only contains pre-measured performance numbers of individual       architecture was initially developed for NLP tasks, the earliest NAS
operations which can be summed up to estimate the overall per-     works for Transformers were primarily in this domain.
formance of a given DNN. In FBNet [228], for instance, the latency        Evolved Transformer [200] was one of the earliest attempts to
number estimated from a lookup table is used as a regularizer term      apply NAS for searching better Transformer architectures, and
in its gradient based NAS framework to penalize operations that         it did so via an evolutionary search algorithm. Inspired by NAS-
would be slow on the target hardware device.                      Net [255], Evolved Transformer adopts the cell-wise search space
   Finally, some hardware-aware NAS frameworks rely on light-      to search two cell structures. Each of these cell structures can be
weight prediction models that can quickly predict hardware perfor-      stacked for multiple times to form the encoder and decoder of
mance numbers for a given DNN. For instance, ProxylessNAS [26]      the encoder-decoder Transformer architecture. The cell structure
has trained a model that takes as inputs a DNN configuration (e.g.,      contains a stack of multiple blocks, and each block has its own
operation types, input and output shapes, and other operation at-     hyperparameters such as operation type, normalization type, and
tributes like kernel sizes) and outputs the estimated latency on the      dimensions which can be searched. The main challenge here is
target hardware platform.                                              that NLP tasks require a much longer time to train and evaluate
                                                                                                (e.g., the popular WMT 2014 En-De translation benchmark contains
                                                                over 3 million sentence pairs). Furthermore, unlike the previous
                                                         works [140, 142, 149, 182, 210] for CNNs that found CIFAR-10 to
                                                              be a reasonable proxy for much larger ImageNet, these NLP tasks




                                                                                          30
<a id="page-31"></a>

### PDF 第 31 页

do not typically have good smaller proxy tasks. To address this,                     Table 8: NAS Search Space.
Evolved Transformer proposes to dynamically allocate resources
                                                                  Parameter     Range of Valuesto more promising architectures by early stopping those who fail
to achieve the hurdle fitness within a small number of steps.                  𝑁                  {3, 4, 5, 6}
  Due to the large computational cost of training Transformers on                   ℎ               {4, 6, 8, 10, 12}
NLP tasks, weight sharing and supernet based NAS have become                     𝑑        384 −768, step size=96
popular options. HAT [222] extends the Once-for-All [25] scheme                    𝑑FFN      768 −3072, step size=128
to Transformer architectures to train a single supernet from which
sub-networks with different depths, number of heads, and dimen-
sions can be sampled. Furthermore, HAT is hardware-aware, in     and AlphaNet [221] to apply the sandwich sampling rule to train a
that it directly optimizes for latency along with accuracy using a      supernet. GLiT [30] proposes a hierarchical NAS scheme for search-
multi-layer latency prediction model. HAT shares the benefits of      ing hybrid convolution-attention architectures. It determines the
Once-for-All, which allows sub-networks to be sampled through     number of convolutional and multi-head attention heads in each
evolutionary search and deployed immediately to target hardware       layer in the first stage of NAS, as well as the detailed hyperparame-
devices without retraining.                                                ters such as dimensions in the second stage.
  NAS-BERT [235] is another supernet based NAS for Transform-       One noticeable characteristic of most of the NAS methods intro-
ers that extends Single Path One-Shot [77]. Different from the     duced above for Transformer architectures (both for NLP and CV
aforementioned methods, NAS-BERT proposes a NAS method that       applications) is their use of supernet-based, weight-shared method-
can be applied at the pre-training stage of encoder-only BERT so as       ologies, which is summarized in Tab. 7. This is presumably due to
to be agnostic to downstream tasks. In order to avoid the prohibitive      the immense computational cost associated with training Trans-
cost of directly performing architecture search in a big supernet on      former architectures. The use of supernet-based NAS can limit the
the heavy pre-training task, NAS-BERT employs two novel tech-     range of architectures that can be discovered, due to the large con-
niques: (1) block-wise training, that splits the entire supernet into       straints it imposes on the search space, which may prevent the
multiple blocks of successive Transformer layers which are then      discovery of unique or innovative architectures. Therefore, there is
trained separately; and (2) progressive shrinking, that dynamically      a need to explore better ways to balance the flexibility and efficiency
prunes less promising sub-networks based on their validation loss.      of NAS techniques.
  Primer [201] searches for a more efficient decoder-only Trans-
former for auto-regressive language modeling. Unlike the majority       Summary (Sec. 6.3. Transformer-specific NAS): The
of NAS methods that view a model as a connection of multiple          existing NAS frameworks have been extended to design
operations selected from a NAS search space, Primer views it as a        more efficient Transformer architectures. Due to the large
single valid Tensorflow (TF) program comprised of fine-grained TF         computational cost for training Transformer models, which
primitive operations like addition, exponential, convolution, and             is even further exacerbated when combined with unsuper-
many others. Using evolutionary search, it targets to search a TF          vised pre-training methodologies, most of the existing meth-
program defining a decoder block that can be stacked multiple         ods heavily rely on the weight-sharing scheme followed
times to form an auto-regressive language model. The hope is that        by evolutionary search. A key challenge in Transformer-
this minimizes inductive bias when designing the search space, as           specific NAS is that existing work is primarily limited to
the possible set of operations and their connectivity are no longer         tuning relatively trivial hyperparameters such as hidden
pre-determined by human experts. In order to reduce the heavy         dimensions, depth, and number of heads. However, this
computational cost of auto-regressive pre-training, Primer brings             is likely to preclude the discovery of novel Transformer
the idea of hurdles of Evolved Transformer [200]. Additionally, it           variants.
uses relatively small LM1B dataset as a proxy task to discover model
architecture, which is then transferred to much larger target tasks
such as PG-19 [176] and C4 [178].                              6.4  Case Study: Running NAS and Co-design
  The Transformer architecture, initially developed for NLP tasks,       on the Transformer
has been adapted for use in the field of CV. Referred to as Vision
                                                           So far, we have discussed the general concept of NAS, its applica-
Transformers (ViTs) [56, 144, 215], these models have been demon-
                                                                          tion to hardware-aware scenarios, and its extension into the Trans-
strated to outperform popular CNN architectures in various CV
                                                                former architecture. Here, we conduct a case study to demonstrate
applications, thus driving research towards the development of NAS
                                                                    the performance gains of applying NAS to Transformer inference
techniques to automatically design better ViT architectures. How-
                                                       on Gemmini, with the goal of optimizing not only accuracy, but
ever, due to the architectural similarities, these works have much
                                                                         also hardware costs such as latency and energy.
in common with the NAS methodologies for NLP-targeted Trans-
formers. For instance, Autoformer [230] and ViT-ResNAS [135] are        6.4.1  Experiment Setup. As a baseline architecture, we use a 6-
extensions of Single Path One-Shot [77] to the ViT search space,      layer Transformer architecture with all other model configurations
including depth, hidden dimensions, and the number of heads of      remaining the same as BERT-Base or GPT-2 (see the details in Tab. 1).
each Transformer layer. Burgerformer [236] takes a step further to    We consider Language Modeling task, and we train a randomly
take into account the micro design, i.e., the type of operations, acti-       initialized model on the WikiText-2 [154] benchmark with 37k
vations, and normalization, as well. NASViT extends BigNAS [243]      training examples and 4k validation examples using a language




                                                                                          31
<a id="page-32"></a>

### PDF 第 32 页

modeling training objective. To evaluate the model performance, we     Table 9: Sample Architecture found using NAS with 3.6 × 109 EDP
measured perplexity on the validation examples, excluding empty     and 22.51 perplexity.
strings, where lower scores indicate better performance. The stand-
                                                               Parameter             Valuesalone baseline model was trained for 50 epochs with the Adam
optimizer and a linear learning rate scheduling with a peak learning             𝑁                     6
rate in the range {5, 2, 1, 0.5} × 10−5. The training examples are                ℎ                  [12, 6, 12, 8, 10, 6]
concatenated to reach a maximum sequence length of 512 and                 𝑑                    672
batched using a batch size of 16.                                              𝑑FFN       [1280, 1280, 2560, 768, 2048, 1024]
   For NAS, we adopt the BigNAS-style [243] strategy to train a
supernet, and then we used an evolutionary algorithm to search
sub-networks out of the fully trained supernet. The NAS search
space is comprised of various combinations of the number of layers      corresponds to the largest Transformer architecture in our search
𝑙, number of heads ℎ, hidden dimension 𝑑, and FFN dimension      space in Tab. 8.
𝑑FFN (see Tab. 8 for details). For supernet training, we use the same      We first present results from the evolutionary search process
training hyperparameters as the stand-alone training, except that      over EDP in Fig. 26. As can be seen in the plot, the NAS framework
in each training iteration, we sample four sub-networks: the largest      allows us to obtain multiple Transformer architectures with better
possible; the smallest possible; and two randomly sampled sub-     hardware cost to perplexity trade-offs. That is, it finds architec-
networks. The model parameter update is then performed using the      tures with similar or even better perplexity, as compared to the
sandwich rule, which involves taking the average of the gradients      baseline with smaller hardware costs. As an example, we select
collected from the backward paths of these four sub-networks.         the architecture with the lowest EDP while having less than +0.1
   For the evolutionary search, we initialize a population of 40      perplexity loss, whose EDP is 3.6 × 109 and perplexity is 22.51.
sub-networks and perform 40 rounds of evolution iterations. In     The architecture parameters are listed in Table 9. This architecture
each iteration, the validation perplexity and energy-delay-product       illustrates the importance of a diverse search space, as the number
(EDP) of each sub-network on the target hardware are collected,      of attention heads varies from 6 to 12 in each layer, and as the fully
and only the sub-networks that are Pareto-optimal are retained.     connected layer dimensions vary from 768 to 2560. By being able
Here, we use EDP as a single hardware cost metric, as it allows for       to change these parameters on a per-layer basis, one may discover
the conversion of a multi-objective optimization problem into a     more Pareto-optimal architectures compared to if these parameters
single-objective optimization problem, by combining both latency     were fixed for every layer.
and energy into one metric. The retained sub-networks are then         In Fig. 25, we separate out latency and energy, and substitute in
mutated with a mutation probability of 0.2 to refill the population     RTL values for the latency. As one can see, it is possible to attain a
for the next iteration. To measure the hardware cost, we use a      1.4× reduction in latency versus the baseline Transformer trained
lookup table-based method for quickly assessing the latency and     from scratch with 0.1 point perplexity degradation. If one could
energy consumption of each sub-network on the target hardware,       tolerate about one point degradation in perplexity, latency can be
instead of using RTL (Register Transfer Logic) simulation, which      reduced by 2.4×, and possibly even further with more advanced ar-
can be time-consuming. The entries in the lookup table are obtained       chitecture search techniques. With regards to energy, one can attain
from Timeloop [163] simulations, which provide simulated latency      a 1.6× improvement considering 0.1 point perplexity degradation,
and energy numbers for each operation. The end-to-end latency     and 4.4× if perplexity is allowed to increase by 1 point. Taking both
and energy are then estimated by summing the per-operation costs.      together, it is possible to reduce EDP by 2.2× with just 0.1 point per-
After the evolutionary search, the Pareto-optimal sub-networks       plexity degradation, and 10.6× with 1 point perplexity degradation.
are then evaluated with an RTL simulator to obtain a more precise      These examples illustrate the power of co-design in allowing practi-
estimation of the latency. For the energy measure, we continue to       tioners to choose a combination that best matches their needs. It is
use the numbers from Timeloop, as it is technically challenging to      important to note that this represents a single run of our co-design
measure the energy consumption via RTL.                         methodology on a specific hardware platform, and results may vary
   For the target hardware, we use Gemmini with the optimizations      depending on the target hardware and optimization goals.
applied in Sec. 3.4 with the dedicated normalization units for run-
ning non-linear operations on-chip. We configure Gemmini with a       Summary  (Sec.  6.4. NAS for Transformers Case
scratchpad size of 64 kB and accumulator size of 256 kB based on         Study): This case study used a supernet-based NAS to sam-
the insights in Sec. 3.4.3 to maximize the accumulator size.                  ple diverse architectures followed by an evolutionary search
                                                                           to discover Pareto-optimal architectures that trade-offbe-
                                                              tween perplexity and energy-delay-product, a measure of
6.4.2  Experiment Results. We show the NAS Pareto-frontier         runtime efficiency. Many discovered architectures have sig-
results for both latency and energy in Fig. 25 (blue curves) where           nificant latency and energy improvements compared to the
each point corresponds to a different Transformer architecture that          baseline trained from scratch when running on an opti-
has been found from the evolutionary search algorithm discussed         mized Gemmini hardware accelerator. The importance of
above. Additionally, we plot the baseline 6-layer Transformer model          using NAS to explore the search space is underscored by the
trained from scratch as a reference (× mark). All the EDP values           fact that many well-performing architectures use diverse
are normalized with the baseline EDP. Note that the baseline model




                                                                                          32
<a id="page-33"></a>

### PDF 第 33 页

NAS Results: Latency vs. Perplexity                                NAS Results: Latency vs. Energy
            (Scratchpad 64kB, Accumulator: 256kB)                                   (Scratchpad 64kB, Accumulator: 256kB)
                                  NAS                                                                 NAS    25.0                                                                                 25.0                                             Trained from scratch                                                                       Trained from scratch
    24.5                                                                                 24.5



                                       +1 Perplexity                                                                                                               +1 Perplexity     Perplexity24.023.5                                                                                 24.0                                                                                                                                                                                                                                                                                           Perplexity23.5

    23.0                                                                                 23.0
                                              +0.1 Perplexity                                                                        +0.1 Perplexity
    22.5                                                                                 22.5

            20      40      60      80     100     120                            0      20     40     60     80     100    120
                          Latency (109 Cycles)                                                                 Energy (10 3 J)

Figure 25: (Left) Latency-perplexity and (Right) Energy-perplexity plots of the Transformer architectures found via evolutionary search on our
optimal Gemmini hardware configuration. Similar to Fig. 26, lower perplexity indicates better performance, and we plot lines to illustrate +0.1 and
+1 point perplexity degradation.

          NAS Results: Latency vs. Normalized EDP                  limited understanding of the run-time characteristics and bottle-              (Scratchpad 64kB, Accumulator: 256kB)
                                                              necks of Transformer workloads, as well as the design principles
                                    NAS      25.0                                                         necessary for effectively running these models, in comparison to                                                Trained from scratch
      24.5                                      CNN architectures.
                                                                        In this paper, we have conducted a comprehensive analysis of
      24.0                                                              Transformer workloads, in order to better understand run-time
      23.5                               +1 Perplexity                    characteristics and identify performance bottlenecks of Transform-            Perplexity
                                                                          ers running on commodity hardware and accelerators (Sec. 2). Fur-
      23.0
                                               +0.1 Perplexity                thermore, we have performed an extensive survey of the current
      22.5                                                    hardware and software solutions, with the goal of identifying po-
      22.0                                                                    tential optimization opportunities in the full-stack deployment of
            0.0       0.2       0.4       0.6       0.8       1.0               Transformers. Specifically, our survey covered the following topics:                         EDP (Normalized)
                                                             • The design of hardware architectures for Transformer inference,
Figure 26: EDP-perplexity plots of the Transformer architectures          including the impact of the non-linear operations on designing
found via evolutionary search on our Gemmini hardware configu-         hardware accelerators (Sec. 3);
ration. Lower perplexity indicates better performance of the trained       • Optimization strategies such as pruning and quantization that
models. For better comparison, we additionally plot lines to illustrate         can be applied to a fixed Transformer architecture for better
+0.1 and +1 point perplexity degradation.                               performance (Sec. 4);
                                                             • Mapping and scheduling of operations in the Transformer ar-
                                                                            chitecture and the associated challenges (Sec. 5); and
                                                             • The use of automated NAS for designing more efficient Trans-    layer configurations. When trained on WikiText-2 language
                                                                 former architectures and adapting them to target hardware    modeling benchmark, these techniques found Transformer
                                                                                  (Sec. 6.1).    architectures with a 2.2× EDP reduction while tolerating a
    0.1 point perplexity degradation, and 10.6× with a 1 point        The key findings from this study include:
    degradation over the baseline.                                 • Despite the small FLOPs count, the nonlinear operations in
                                                                     Transformers can be highly influential on overall performance,
                                                                                                         if they are not taken into account properly when designing
7  CONCLUSION                                                domain-specific accelerators. The computation of nonlinear
The Transformer architecture [217] has revolutionized the field of          operations such as Softmax and LayerNorm also requires com-
natural language understanding [52, 125, 143, 173, 174, 177, 240],         puting runtime statistics, whereas the BatchNorm operations
which has been further accelerated with the recent development of          in CNNs can be absorbed into prior convolutional layers dur-
large language models with hundreds of billions of parameters [22,         ing inference.
44, 58, 86, 175, 190, 198]. This architecture has also been extended to       • Hardware design for CNNs may not necessarily be the same as
a wide range of fields, including computer vision [24, 36, 130, 152],          that for Transformers. For instance, increasing the accumulator
and speech recognition [23, 76, 109]. While Transformer models           sizes to enable higher output reuse yielded significant perfor-
have shown significant performance improvements, their growing        mance improvement in Gemmini for Transformer applications.
size and run-time complexity present a critical challenge in efficient       • It appears less complex to schedule matmuls in Transformers
inference. While DNN accelerators that enable fast and efficient         than convolutions in CNNs, due to the fact that scheduling
deep learning computation can be a viable solution, there is still        matmuls involves 3 loops, as compared to 6 for convolutions.




                                                                                          33
<a id="page-34"></a>

### PDF 第 34 页

However, we observed that scheduling matmuls involve similar       reflect the position or the policy of our sponsors, and no official
   amounts of decision points and a wide range of performance      endorsement should be inferred.
    outcomes, making it as challenging as scheduling convolutions.
 • Fusing LayerNorms with the preceding matmuls in the Trans-                                      REFERENCES
    former architecture imposes several constraints on the mapping,
                                                                                                   [1] Edge TPU. https://cloud.google.com/edge-tpu/. Accessed: 2018-12-05.
    particularly related to tile sizes. As a result, careful considera-           [2] Chatgpt: Optimizing language models for dialogue. https://openai.com/blog/
    tion must be taken when deciding whether to fuse operations,               chatgpt/, 2022.
                                                                                                   [3] Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jef-
    contrary to the common belief that operator fusion is gener-              frey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard,
    ally beneficial.                                                                                  et al. {TensorFlow}: a system for {Large-Scale} machine learning. In USENIX
                                                                               Symposium on Operating Systems Design and Implementation (OSDI), 2016.
   Finally, throughout the paper, we conducted case studies to quan-           [4] Dennis Abts, John Kim, Garrin Kimmell, Matthew Boyd, Kris Kang, Sahil Parmar,
tify the advantages of co-design and co-optimization techniques           Andrew Ling, Andrew Bitar, Ibrahim Ahmed, and Jonathan Ross. The groq
across the stack on full-stack Transformer inference. Overall, the re-             software-defined scale-out tensor streaming multiprocessor: From chips-to-
                                                                                      systems architectural overview. In IEEE Hot Chips Symposium, pages 1–69,
sult exhibited 88.7× EDP improvement without a noticeable perfor-              2022.
mance drop compared to a naive implementation without full-stack           [5] Aravind Acharya, Uday Bondhugula, and Albert Cohen. An approach for
                                                                                                 finding permutations quickly: Fusion and dimension matching. arXiv preprint
considerations.                                                                                 arXiv:1803.10726, 2018.
 • In Sec. 3.4, we applied hardware design techniques in order           [6] Aravind Acharya, Uday Bondhugula, and Albert Cohen.  Polyhedral auto-
                                                                                              transformation with no integer linear programming. In Proceedings of the ACM
    to avoid the high communication overhead associated with           SIGPLAN Conference on Programming Language Design and Implementation
    offloading unsupported operations to the host CPU. Gemmini               (PLDI), 2018.
   was originally designed for CNN workloads, and making it           [7] Andrew Adams, Karima Ma, Luke Anderson, Riyadh Baghdadi, Tzu-Mao Li,
                                                                                        Michael Gharbi, Benoit Steiner, Steven Johnson, Kayvon Fatahalian, Frédo Du-
    perform Softmax, LayerNorm, and GELU on-chip required addi-              rand, and Jonathan Ragan-Kelley. Learning to optimize halide with tree search
    tional changes. Our implementation of dedicated normalization            and random programs. ACM Transactions on Graphics (TOG), 2019.
    units to support Softmax and LayerNorm had an associated           [8] Vahideh Akhlaghi, Amir Yazdanbakhsh, Kambiz Samadi, Rajesh K. Gupta, and                                                                                  Hadi Esmaeilzadeh. Snapea: Predictive early activation for reducing computa-
   5−15% area cost and 8% latency increase. Nonetheless, this               tion in deep convolutional neural networks. In Proceedings of the 45th Annual
    extra overhead was compensated by the gain achieved by run-               International Symposium on Computer Architecture, page 662–673, 2018.
                                                                                                   [9] Jorge Albericio, Patrick Judd, Tayler Hetherington, Tor Aamodt, Natalie Enright
   ning the nonlinear operations on-chip using the polynomial                Jerger, and Andreas Moshovos. Cnvlutin: Ineffectual-neuron-free deep neural
    approximation proposed in I-BERT [111]. Combined with mem-            network computing. In Proceedings of the 43rd International Symposium on
   ory hierarchy re-balancing, this provided a net 39.6× latency            Computer Architecture, 2016.
                                                                                              [10] Manoj Alwani, Han Chen, Michael Ferdman, and Peter Milder. Fused-layer cnn
    reduction.                                                                                      accelerators. In Proceedings of the International Symposium on Microarchitecture
 • In Section 6.4, we ran NAS to search for Pareto-optimal Trans-            (MICRO), 2016.
   former architectures given the tradeoffbetween EDP and per-         [11] Jason Ansel, Shoaib Kamil, Kalyan Veeramachaneni, Jonathan Ragan-Kelley,                                                                                                           Jeffrey Bosboom, Una-May O’Reilly, and Saman Amarasinghe. Opentuner: An
    plexity on a popular language modeling task. We used Timeloop               extensible framework for program autotuning. In Proceedings of the International
    simulated numbers to estimate the cost of various architectures              Conference on Parallel Architectures and Compilation Techniques (PACT), 2014.
                                                                                              [12] ARM. Cortex-M, https://developer.arm.com/ip-products/processors/cortex-m,
    within a large search space and guide the automated NAS pro-              2020.
    cess. The total contribution as shown in Fig. 25 is 2.24× EDP         [13] Giorgos Armeniakos, Georgios Zervakis, Dimitrios Soudris, and Jörg Henkel.
    reduction without a noticeable perplexity drop, and 10.56× EDP            Hardware approximate techniques for deep neural network accelerators: A
                                                                                                  survey. ACM Comput. Surv., mar 2022. Just Accepted.
    reduction with 1 perplexity drop.                                              [14] R. Baghdadi, J. Ray, M. B. Romdhane, E. D. Sozzo, A. Akkas, Y. Zhang, P. Suriana,
  We anticipate that our in-depth analysis and results, along with                 S. Kamil, and S. Amarasinghe. Tiramisu: A polyhedral compiler for expressing                                                                                                            fast and portable code. In International Symposium on Code Generation and
the comprehensive survey presented in this paper will facilitate              Optimization (CGO), 2019.
further advencements in understanding Transformer inference and         [15] Riyadh Baghdadi, Ulysse Beaugnon, Albert Cohen, Tobias Grosser, Michael
                                                                                            Kruse, Chandan Reddy, Sven Verdoolaege, Adam Betts, Alastair F. Donald-
optimizing its inference efficiency from various angles. We believe               son, Jeroen Ketema, Javed Absar, Sven Van Haastregt, Alexey Kravets, Anton
that this will enable Transformers to reach their full potential and             Lokhmotov, Robert David, and Elnar Hajiyev. Pencil: A platform-neutral com-
expand their application to a much wider range of areas than what              pute intermediate language for accelerator programming. In Proceedings of the
                                                                                                    International Conference on Parallel Architectures and Compilation Techniques
they have achieved so far.                                                             (PACT), 2015.
                                                                                              [16] Bowen Baker, Otkrist Gupta, Nikhil Naik, and Ramesh Raskar.  Designing
8 ACKNOWLEDGEMENTS                                              neuralarXiv:1611.02167,network architectures2016.        using reinforcement learning.  arXiv preprint
We acknowledge gracious support from Meta and in particular         [17] Gabriel Bender, Pieter-Jan Kindermans, Barret Zoph, Vijay Vasudevan, and Quoc
                                                                                                         Le. Understanding and simplifying one-shot architecture search. In International
Michael Anderson, Satish Nadathur and Summer Deng, as well as               conference on machine learning, pages 550–559. PMLR, 2018.
Google Cloud, Google TRC team, and specifically Jonathan Caton,         [18] Hadjer Benmeziane, Kaoutar El Maghraoui, Hamza Ouarnoughi, Smail Niar,
Prof. David Patterson, and Jing Li. Prof. Keutzer’s lab is sponsored             Martin Wistuba, and Naigang Wang. A comprehensive survey on hardware-
                                                                                  aware neural architecture search. arXiv preprint arXiv:2101.09336, 2021.
by Intel corporation, Intel VLAB team, Intel One-API center of         [19] Pierre Blanchard, Desmond J Higham, and Nicholas J Higham.  Accurately
excellence, as well as funding through BDD and BAIR. Sehoon Kim            computing the log-sum-exp and softmax functions. IMA Journal of Numerical
would like to acknowledge the support from Korea Foundation for               Analysis, 41(4):2311–2330, 08 2020.
                                                                                              [20] Uday Bondhugula, Aravind Acharya, and Albert Cohen. The pluto+ algorithm:
Advanced Studies (KFAS). Amir Gholami was supported through         A practical approach for parallelization and locality optimization of affine loop
funding from Samsung SAIT. Michael W. Mahoney would also like                nests. ACM Transactions on Programming Languages and Systems (TOPLAS),
                                                                                                  2016.
to acknowledge a J. P. Morgan Chase Faculty Research Award as         [21] Uday Bondhugula, Albert Hartono, Jagannathan Ramanujam, and Ponnuswamy
well as the DOE, NSF, and ONR. Our conclusions do not necessarily             Sadayappan. A practical automatic polyhedral parallelizer and locality optimizer.




                                                                                          34
<a id="page-35"></a>

### PDF 第 35 页

In Proceedings of the ACM SIGPLAN Conference on Programming Language Design               Micro, 41(2):29–35, 2021.
    and Implementation (PLDI), 2008.                                                      [44] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gau-
[22] Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Ka-             rav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton,
     plan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry,              Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways.
    Amanda Askell, et al. Language models are few-shot learners. arXiv preprint              arXiv preprint arXiv:2204.02311, 2022.
     arXiv:2005.14165, 2020.                                                                [45]  L.T. Clark, V. Vashishtha, L. Shifren, A. Gujia, S. Sinha, B. Cline, C. Ramamurthya,
[23] Maxime Burchi and Valentin Vielzeuf. Efficient conformer: Progressive down-            and G. Yeric. ASAP7: A 7-nm FinFET Predictive Process Design Kit. Microelec-
     sampling and grouped attention for automatic speech recognition. In 2021 IEEE                tronics Journal, 2016.
     Automatic Speech Recognition and Understanding Workshop (ASRU), pages 8–15.         [46] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G Carbonell, Quoc Le, and Ruslan
     IEEE, 2021.                                                                            Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-
[24] Han Cai, Chuang Gan, and Song Han.  Efficientvit: Enhanced linear atten-              length context. In Proceedings of the 57th Annual Meeting of the Association for
     tion for high-resolution low-computation visual recognition. arXiv preprint             Computational Linguistics, pages 2978–2988, 2019.
     arXiv:2205.14756, 2022.                                                                [47] Shail Dave, Riyadh Baghdadi, Tony Nowatzki, Sasikanth Avancha, Aviral Shri-
[25] Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, and Song Han. Once-for-              vastava, and Baoxin Li. Hardware acceleration of sparse and irregular tensor
       all: Train one network and specialize it for efficient deployment. arXiv preprint             computations of ml models: A survey and insights. Proceedings of the IEEE,
     arXiv:1908.09791, 2019.                                                                   109(10):1706–1752, 2021.
[26] Han Cai, Ligeng Zhu, and Song Han. Proxylessnas: Direct neural architecture         [48] Shail Dave, Youngbin Kim, Sasikanth Avancha, Kyoungwoo Lee, and Aviral
     search on target task and hardware. arXiv preprint arXiv:1812.00332, 2018.                   Shrivastava.  DMazeRunner: Executing perfectly nested loops on dataflow
[27] Maurizio Capra, Beatrice Bussolino, Alberto Marchisio, Guido Masera, Maurizio                accelerators. ACM Transactions on Embedded Computing Systems, 2019.
     Martina, and Muhammad Shafique. Hardware and software optimizations for         [49] Lei Deng, Guoqi Li, Song Han, Luping Shi, and Yuan Xie. Model compres-
     accelerating deep neural networks: Survey of current trends, challenges, and              sion and hardware acceleration for neural networks: A comprehensive survey.
     the road ahead. IEEE Access, 8:225134–225180, 2020.                                        Proceedings of the IEEE, 108(4):485–532, 2020.
[28] Daniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, and Lucia Specia.         [50] Jérémie Detrey and Florent de Dinechin. A parameterized floating-point expo-
     Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual               nential function for fpgas. In Proceedings. 2005 IEEE International Conference on
     focused evaluation. arXiv preprint arXiv:1708.00055, 2017.                                 Field-Programmable Technology, 2005., pages 27–34. IEEE, 2005.
[29] Prasanth Chatarasi, Hyoukjun Kwon, Natesh Raina, Saurabh Malik, Vaisakh         [51] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Gpt3. int8
     Haridas, Angshuman Parashar, Michael Pellauer, Tushar Krishna, and Vivek                      (): 8-bit matrix multiplication for transformers at scale. In Advances in Neural
      Sarkar. Marvel: A data-centric compiler for dnn operators on spatial accelerators,              Information Processing Systems.
     2020.                                                                                  [52] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT:
[30] Boyu Chen, Peixia Li, Chuming Li, Baopu Li, Lei Bai, Chen Lin, Ming Sun, Junjie              Pre-training of deep bidirectional transformers for language understanding.
     Yan, and Wanli Ouyang. Glit: Neural architecture search for global and local              arXiv preprint arXiv:1810.04805, 2018.
    image transformer. In Proceedings of the IEEE/CVF International Conference on         [53] William B Dolan and Chris Brockett. Automatically constructing a corpus of
     Computer Vision, pages 12–21, 2021.                                                         sentential paraphrases. In Proceedings of the Third International Workshop on
[31] Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Lau-             Paraphrasing (IWP2005), 2005.
     rent Sifre, and John Jumper. Accelerating large language model decoding with         [54] Jin-Dong Dong, An-Chieh Cheng, Da-Cheng Juan, Wei Wei, and Min Sun. Dpp-
     speculative sampling. arXiv preprint arXiv:2302.01318, 2023.                                   net: Device-aware progressive search for pareto-optimal neural architectures.
[32] Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun              In Proceedings of the European Conference on Computer Vision (ECCV), pages
     Xiao, Bing Xu, Chiyuan Zhang, and Zheng Zhang. Mxnet: A flexible and efficient              517–531, 2018.
     machine learning library for heterogeneous distributed systems. arXiv preprint         [55] Zhen Dong, Zhewei Yao, Amir Gholami, Michael W Mahoney, and Kurt Keutzer.
     arXiv:1512.01274, 2015.                                             HAWQ: Hessian aware quantization of neural networks with mixed-precision.
[33] Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen              In Proceedings of the IEEE International Conference on Computer Vision, pages
     Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, et al. {TVM}: An              293–302, 2019.
     automated end-to-end optimizing compiler for deep learning. In 13th {USENIX}          [56] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xi-
    Symposium on Operating Systems Design and Implementation ({OSDI} 18), pages             aohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg
     578–594, 2018.                                                                            Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for
[34] Tianshi Chen, Zidong Du, Ninghui Sun, Jia Wang, Chengyong Wu, Yunji Chen,            image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.
     and Olivier Temam. Diannao: A small-footprint high-throughput accelerator for         [57] Petros Drineas and Michael W Mahoney. Lectures on randomized numerical
     ubiquitous machine-learning. In Proceedings of the 19th International Conference                linear algebra. In The Mathematics of Data, IAS/Park City Mathematics Series,
    on Architectural Support for Programming Languages and Operating Systems,             pages 1–48. AMS/IAS/SIAM, 2018.
    ASPLOS ’14, pages 269–284, New York, NY, USA, 2014. ACM.                          [58] Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin,
[35]  Y. Chen, J. Emer, and V. Sze. Eyeriss: A spatial architecture for energy-efficient            Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, et al.
     dataflow for convolutional neural networks. In 2016 ACM/IEEE 43rd Annual             Glam: Efficient scaling of language models with mixture-of-experts. In Interna-
      International Symposium on Computer Architecture (ISCA), pages 367–379, June                tional Conference on Machine Learning, pages 5547–5569. PMLR, 2022.
     2016.                                                                                  [59] Zidong Du, Robert Fasthuber, Tianshi Chen, Paolo Ienne, Ling Li, Tao Luo,
[36] Yinpeng Chen, Xiyang Dai, Dongdong Chen, Mengchen Liu, Xiaoyi Dong,             Xiaobing Feng, Yunji Chen, and Olivier Temam. Shidiannao: Shifting vision
    Lu Yuan, and Zicheng Liu. Mobile-former: Bridging mobilenet and transformer.             processing closer to the sensor. In 2015 ACM/IEEE 42nd Annual International
     In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recog-            Symposium on Computer Architecture (ISCA), pages 92–104, 2015.
      nition, pages 5270–5279, 2022.                                                         [60] Robert Eisele. The log-sum-exp trick in machine learning, 2016.
[37] Yu-Hsin Chen, Joel Emer, and Vivienne Sze. Eyeriss: A Spatial Architecture for         [61] Thomas Elsken, Jan Hendrik Metzen, and Frank Hutter. Neural architecture
     Energy-efficient Dataflow for Convolutional Neural Networks. In Proceedings               search: A survey. The Journal of Machine Learning Research, 20(1):1997–2017,
      of the International Symposium on Computer Architecture (ISCA), 2016.                       2019.
[38] Yu-Hsin Chen, Joel Emer, and Vivienne Sze. Using dataflow to optimize energy         [62] Hadi Esmaeilzadeh, Adrian Sampson, Luis Ceze, and Doug Burger. Neural
      efficiency of deep neural network accelerators. IEEE Micro, 37(3):12–21, 2017.               Acceleration for General-Purpose Approximate Programs. In Proceedings of the
[39] Yu-Hsin Chen, Tien-Ju Yang, Joel Emer, and Vivienne Sze. Eyeriss v2: A flexible               International Symposium on Microarchitecture (MICRO), 2012.
      accelerator for emerging deep neural networks on mobile devices. IEEE Journal         [63] Angela Fan, Edouard Grave, and Armand Joulin. Reducing transformer depth
     on Emerging and Selected Topics in Circuits and Systems, 2019.                         on demand with structured dropout. arXiv preprint arXiv:1909.11556, 2019.
[40] Yunji Chen, Tao Luo, Shaoli Liu, Shijin Zhang, Liqiang He, Jia Wang, Ling Li,         [64] Hongxiang Fan, Thomas Chau, Stylianos I. Venieris, Royson Lee, Alexandros
     Tianshi Chen, Zhiwei Xu, Ninghui Sun, and Olivier Temam. DaDianNao: A               Kouris, Wayne Luk, Nicholas D. Lane, and Mohamed S. Abdelfattah. Adaptable
     Machine-learning Supercomputer. In Proceedings of the International Symposium               butterfly accelerator for attention-based nns via hardware and algorithm co-
     on Microarchitecture (MICRO), 2014.                                                         design, 2022.
[41] Sharan Chetlur, CliffWoolley, Philippe Vandermersch, Jonathan Cohen, John         [65] Chao Fang, Aojun Zhou, and Zhongfeng Wang. An algorithm-hardware co-
     Tran, Bryan Catanzaro, and Evan Shelhamer. cudnn: Efficient primitives for             optimized framework for accelerating n:m sparse transformers. IEEE Transac-
     deep learning. arXiv preprint arXiv:1410.0759, 2014.                                             tions on Very Large Scale Integration (VLSI) Systems, pages 1–14, 2022.
[42] Jaewan Choi, Hailong Li, Byeongho Kim, Seunghwan Hwang, and Jung Ho Ahn.         [66] Eric Flamand, Davide Rossi, Francesco Conti, Igor Loi, Antonio Pullini, Florent
     Accelerating transformer networks through recomposing softmax layers. In             Rotenberg, and Luca Benini.  Gap-8: A risc-v soc for ai at the edge of the
      International Symposium on Workload Characterization (IISWC), 2021.                             iot. In 2018 IEEE 29th International Conference on Application-specific Systems,
[43] Jack Choquette, Wishwesh Gandhi, Olivier Giroux, Nick Stam, and Ronny               Architectures and Processors (ASAP), pages 1–4. IEEE, 2018.
     Krashinsky. Nvidia a100 tensor core gpu: Performance and innovation. IEEE




                                                                                         35
<a id="page-36"></a>

### PDF 第 36 页

[67] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding              Papers (ISSCC), pages 10–14, 2014.
      sparse, trainable neural networks. arXiv preprint arXiv:1803.03635, 2018.                [88] Lu Hou, Zhiqi Huang, Lifeng Shang, Xin Jiang, Xiao Chen, and Qun Liu. Dyn-
[68] Trevor Gale, Erich Elsen, and Sara Hooker. The state of sparsity in deep neural                abert: Dynamic bert with adaptive width and depth. Advances in Neural Infor-
     networks. arXiv preprint arXiv:1902.09574, 2019.                                       mation Processing Systems, 33:9782–9793, 2020.
[69] Mingyu Gao, Jing Pu, Xuan Yang, Mark Horowitz, and Christos Kozyrakis. Tetris:         [89] Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun
     Scalable and Efficient Neural Network Acceleration with 3D Memory. In Pro-           Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam.  Mobilenets:
     ceedings of the International Conference on Architectural Support for Programming                Efficient convolutional neural networks for mobile vision applications. arXiv
     Languages and Operation Systems (ASPLOS), 2017.                                            preprint arXiv:1704.04861, 2017.
[70] Hasan Genc, Seah Kim, Alon Amid, Ameer Haj-Ali, Vighnesh Iyer, Pranav         [90] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun
     Prakash, Jerry Zhao, Daniel Grubb, Harrison Liew, Howard Mao, Albert Ou,           Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam.  Mobilenets:
     Colin Schmidt, Samuel Steffl, John Wright, Ion Stoica, Jonathan Ragan-Kelley,                Efficient convolutional neural networks for mobile vision applications, 2017.
     Krste Asanovic, Borivoje Nikolic, and Yakun Sophia Shao. Gemmini: Enabling         [91]  J Hruska. New movidius myriad x vpu packs a custom neural compute engine,
     systematic deep-learning architecture evaluation via full-stack integration. In               2017.
     Proceedings of the 58th Annual Design Automation Conference (DAC), 2021.              [92] Qijing Huang, Minwoo Kang, Grace Dinh, Thomas Norell, Aravind Kalaiah,
[71] Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W. Mahoney, and             James Demmel, John Wawrzynek, and Yakun Sophia Shao. Cosa: Scheduling by
     Kurt Keutzer. A survey of quantization methods for efficient neural network              constrained optimization for spatial accelerators. In 2021 ACM/IEEE 48th Annual
      inference, 2021.                                                                                 International Symposium on Computer Architecture (ISCA), pages 554–566. IEEE,
[72] Vinayak Gokhale, Jonghoon Jin, Aysegul Dundar, Berin Martini, and Eugenio               2021.
      Culurciello. A 240 g-ops/s mobile coprocessor for deep neural networks. In         [93] Forrest N Iandola, Song Han, Matthew W Moskewicz, Khalid Ashraf, William J
     2014 IEEE Conference on Computer Vision and Pattern Recognition Workshops,               Dally, and Kurt Keutzer. SqueezeNet: Alexnet-level accuracy with 50x fewer
     pages 696–701, 2014.                                                                parameters and< 0.5 mb model size. arXiv preprint arXiv:1602.07360, 2016.
[73] Chengyue Gong, Dilin Wang, Meng Li, Xinlei Chen, Zhicheng Yan, Yuandong         [94] Forrest N Iandola, Albert E Shaw, Ravi Krishna, and Kurt W Keutzer. Squeezebert:
     Tian, Vikas Chandra, et al. Nasvit: Neural architecture search for efficient vision           What can computer vision teach nlp about efficient neural networks? arXiv
     transformers with gradient conflict aware supernet training. In International               preprint arXiv:2006.11316, 2020.
     Conference on Learning Representations, 2021.                                          [95] Yuka Ikarashi, Gilbert Louis Bernstein, Alex Reinking, Hasan Genc, and Jonathan
[74] Saurabh Goyal, Anamitra Roy Choudhury, Saurabh Raje, Venkatesan Chakar-              Ragan-Kelley. Exocompilation for productive programming of hardware accel-
     avarthy, Yogish Sabharwal, and Ashish Verma. Power-bert: Accelerating bert                erators. In Proceedings of the 43rd ACM SIGPLAN International Conference on
     inference via progressive word-vector elimination. In International Conference            Programming Language Design and Implementation, PLDI 2022, page 703–718,
     on Machine Learning, pages 3690–3699. PMLR, 2020.                          New York, NY, USA, 2022. Association for Computing Machinery.
[75] Tobias Grosser, Hongbin Zheng, Raghesh Aloor, Andreas Simbürger, Armin         [96] Shankar Iyer, Nikhil Dandekar, and Kornl Csernai. First quora dataset release:
     Größlinger, and Louis-Noël Pouchet. Polly-polyhedral optimization in llvm.             Question pairs.(2017). URL https://data. quora. com/First-Quora-Dataset-Release-
     In Proceedings of the First International Workshop on Polyhedral Compilation               Question-Pairs, 2017.
     Techniques (IMPACT), 2011.                                                            [97] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, An-
[76] Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui            drew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and
     Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, et al. Conformer:               training of neural networks for efficient integer-arithmetic-only inference. In
     Convolution-augmented transformer for speech recognition. arXiv preprint               Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,
     arXiv:2005.08100, 2020.                                                              pages 2704–2713, 2018.
[77] Zichao Guo, Xiangyu Zhang, Haoyuan Mu, Wen Heng, Zechun Liu, Yichen Wei,         [98] Yangqing Jia, Evan Shelhamer, JeffDonahue, Sergey Karayev, Jonathan Long,
    and Jian Sun. Single path one-shot neural architecture search with uniform             Ross B. Girshick, Sergio Guadarrama, and Trevor Darrell. Caffe: Convolutional
     sampling. In European Conference on Computer Vision, pages 544–560. Springer,              Architecture for Fast Feature Embedding. CoRR, abs/1408.5093, 2014.
     2020.                                                                                  [99] Zhihao Jia, Matei Zaharia, and Alex Aiken. Beyond data and model parallelism
[78] Bastian Hagedorn, Johannes Lenfers, Thomas Kundefinedhler, Xueying Qin,               for deep neural networks.  In Proceedings of Machine Learning and Systems
     Sergei Gorlatch, and Michel Steuwer. Achieving high-performance the func-              (MLSys), 2019.
      tional way: A functional pearl on expressing high-performance optimizations        [100] N. P. Jouppi, C. Young, N. Patil, D. Patterson, G. Agrawal, R. Bajwa, S. Bates,
     as rewrite strategies. Proc. ACM Program. Lang., 4(ICFP), aug 2020.                               S. Bhatia, N. Boden, A. Borchers, R. Boyle, P. Cantin, C. Chao, C. Clark, J. Coriell,
[79] Tae Jun Ham, S. J. Jung, Seonghak Kim, Young H. Oh, Yeonhong Park, Yongchan            M. Daley, M. Dau, J. Dean, B. Gelb, T. V. Ghaemmaghami, R. Gottipati, W. Gul-
     Song, Junghun Park, Sang-Hee Lee, K. Park, J. Lee, and Deog-Kyoon Jeong. A3:               land, R. Hagmann, C. R. Ho, D. Hogberg, J. Hu, R. Hundt, D. Hurt, J. Ibarz,
     Accelerating attention mechanisms in neural networks with approximation.             A. Jaffey, A. Jaworski, A. Kaplan, H. Khaitan, D. Killebrew, A. Koch, N. Kumar,
     2020 IEEE International Symposium on High Performance Computer Architecture                  S. Lacy, J. Laudon, J. Law, D. Le, C. Leary, Z. Liu, K. Lucke, A. Lundin, G. MacK-
     (HPCA), pages 328–341, 2020.                                                                ean, A. Maggiore, M. Mahony, K. Miller, R. Nagarajan, R. Narayanaswami, R. Ni,
[80] Tae Jun Ham, Yejin Lee, Seong Hoon Seo, Soosung Kim, Hyunji Choi, Sung Jun              K. Nix, T. Norrie, M. Omernick, N. Penukonda, A. Phelps, J. Ross, M. Ross,
     Jung, and Jae W. Lee. Elsa: Hardware-software co-design for efficient, light-             A. Salek, E. Samadiani, C. Severn, G. Sizikov, M. Snelham, J. Souter, D. Stein-
     weight self-attention mechanism in neural networks. In 2021 ACM/IEEE 48th               berg, A. Swing, M. Tan, G. Thorson, B. Tian, H. Toma, E. Tuttle, V. Vasudevan,
    Annual International Symposium on Computer Architecture (ISCA), pages 692–               R. Walter, W. Wang, E. Wilcox, and D. H. Yoon. In-datacenter performance
     705, 2021.                                                                                    analysis of a tensor processing unit. In 2017 ACM/IEEE 44th Annual International
[81] Song Han, Xingyu Liu, Huizi Mao, Jing Pu, Ardavan Pedram, Mark A. Horowitz,            Symposium on Computer Architecture (ISCA), pages 1–12, June 2017.
     and William J. Dally. Eie: Efficient inference engine on compressed deep neural        [101] Sheng-Chun Kao, Geonhwa Jeong, and Tushar Krishna.  ConfuciuX: Au-
     network. SIGARCH Comput. Archit. News, 44(3), June 2016.                           tonomous Hardware Resource Assignment for DNN Accelerators using Re-
[82] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning             inforcement Learning. In Proceedings of the International Symposium on Mi-
      for image recognition. In Proceedings of the IEEE conference on computer vision                croarchitecture (MICRO), 2020.
    and pattern recognition, pages 770–778, 2016.                                        [102] Sheng-Chun Kao and Tushar Krishna. GAMMA: Automating the HW Mapping
[83] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning               of DNN Models on Accelerators via Genetic Algorithm. In Proceedings of the
      for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern               International Conference on Computer-Aided Design (ICCAD), 2020.
     Recognition (CVPR), pages 770–778, 2016.                                            [103] Sheng-Chun Kao, Angshuman Parashar, Po-An Tsai, and Tushar Krishna. De-
[84] Kartik Hegde, Po-An Tsai, Sitao Huang, Vikas Chandra, Angshuman Parashar,              mystifying map space exploration for npus, 2022.
    and Christopher W Fletcher. Mind mappings: enabling efficient algorithm-        [104] Sheng-Chun Kao, Suvinay Subramanian, Gaurav Agrawal, and Tushar Krishna.
      accelerator mapping space search. In Proceedings of the International Conference          An optimized dataflow for mitigating attention performance bottlenecks. In Pro-
    on Architectural Support for Programming Languages and Operation Systems               ceedings of the International Conference on Architectural Support for Programming
     (ASPLOS), 2021.                                                                   Languages and Operation Systems (ASPLOS), 2022.
[85] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (GELUs). arXiv        [105]  S. Karandikar, H. Mao, D. Kim, D. Biancolin, A. Amid, D. Lee, N. Pemberton,
      preprint arXiv:1606.08415, 2016.                                                                E. Amaro, C. Schmidt, A. Chopra, Q. Huang, K. Kovacs, B. Nikolic, R. Katz,
[86] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya,                      J. Bachrach, and K. Asanovic. Firesim: Fpga-accelerated cycle-exact scale-out
     Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes             system simulation in the public cloud. In 2018 ACM/IEEE 45th Annual Interna-
     Welbl, Aidan Clark, et al. Training compute-optimal large language models.                tional Symposium on Computer Architecture (ISCA), pages 29–42, 2018.
     arXiv preprint arXiv:2203.15556, 2022.                                                [106] Sam Kaufman, Phitchaya Phothilimthana, Yanqi Zhou, Charith Mendis, Sudip
[87] Mark Horowitz. 1.1 computing’s energy problem (and what we can do about              Roy, Amit Sabne, and Mike Burrows. A learned performance model for tensor
        it). In 2014 IEEE International Solid-State Circuits Conference Digest of Technical              processing units. In Proceedings of Machine Learning and Systems (MLSys), 2021.





                                                                                         36
<a id="page-37"></a>

### PDF 第 37 页

[107] Ben Keller, Rangharajan Venkatesan, Steve Dai, Stephen G. Tell, Brian Zimmer,        [130] Yanyu Li, Geng Yuan, Yang Wen, Eric Hu, Georgios Evangelidis, Sergey Tulyakov,
      William J. Dally, C. Thomas Gray, and Brucek Khailany. A 17–95.6 tops/w             Yanzhi Wang, and Jian Ren. Efficientformer: Vision transformers at mobilenet
      deep learning inference accelerator with per-vector scaled 4-bit quantization for              speed. arXiv preprint arXiv:2206.01191, 2022.
      transformers in 5nm. In 2022 IEEE Symposium on VLSI Technology and Circuits        [131] Zheng Li, Soroush Ghodrati, Amir Yazdanbakhsh, Hadi Esmaeilzadeh, and
      (VLSI Technology and Circuits), pages 16–17, 2022.                                 Mingu Kang. Accelerating attention through gradient-based learned runtime
[108] Gyuwan Kim and Kyunghyun Cho. Length-adaptive transformer: Train once              pruning. In Proceedings of the 49th Annual International Symposium on Computer
      with length drop, use anytime with search. arXiv preprint arXiv:2010.07003,                Architecture, ISCA ’22, page 902–915, New York, NY, USA, 2022. Association for
      2020.                                                                      Computing Machinery.
[109] Sehoon Kim, Amir Gholami, Albert Shaw, Nicholas Lee, Karttikeya Man-        [132] Zhikai Li and Qingyi Gu. I-vit: Integer-only quantization for efficient vision
      galam, Jitendra Malik, Michael W Mahoney, and Kurt Keutzer. Squeezeformer:              transformer inference, 2022.
    An efficient transformer for automatic speech recognition.  arXiv preprint        [133] Zhuohan Li, Eric Wallace, Sheng Shen, Kevin Lin, Kurt Keutzer, Dan Klein, and
       arXiv:2206.00888, 2022.                                                               Joey Gonzalez. Train big, then compress: Rethinking model size for efficient
[110] Sehoon Kim, Amir Gholami, Zhewei Yao, Nicholas Lee, Patrick Wang, Aniruddha               training and inference of transformers. In International Conference on machine
      Nrusimha, Bohan Zhai, Tianren Gao, Michael W Mahoney, and Kurt Keutzer.               learning, pages 5958–5968. PMLR, 2020.
      Integer-only zero-shot quantization for efficient speech recognition. In ICASSP        [134] Heng Liao, Jiajin Tu, Jing Xia, and Xiping Zhou. Davinci: A scalable architecture
      2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing                for neural network computing. In IEEE Hot Chips Symposium, pages 1–44, 2019.
      (ICASSP), pages 4288–4292. IEEE, 2022.                                              [135] Yi-Lun Liao, Sertac Karaman, and Vivienne Sze. Searching for efficient multi-
[111] Sehoon Kim, Amir Gholami, Zhewei Yao, Michael W Mahoney, and Kurt Keutzer.              stage vision transformers. arXiv preprint arXiv:2109.00642, 2021.
       I-bert: Integer-only bert quantization. In International conference on machine        [136] Z. Liao, R. Couillet, and M. W. Mahoney. Sparse quantized spectral clustering.
       learning, pages 5506–5518. PMLR, 2021.                                                  Technical Report Preprint: arXiv:2010.01376, 2020.
[112] Sehoon Kim, Karttikeya Mangalam, Jitendra Malik, Michael W Mahoney, Amir        [137] Sean Lie. Cerebras architecture deep dive: First look inside the hw/sw co-design
      Gholami, and Kurt Keutzer.  Big little transformer decoder.  arXiv preprint                for deep learning: Cerebras systems. In IEEE Hot Chips Symposium, pages 1–34,
       arXiv:2302.07863, 2023.                                                                     2022.
[113] Sehoon Kim, Sheng Shen, David Thorsley, Amir Gholami, Woosuk Kwon, Joseph        [138]  Ji Lin, Wei-Ming Chen, Yujun Lin, Chuang Gan, Song Han, et al. Mcunet: Tiny
      Hassoun, and Kurt Keutzer. Learned token pruning for transformers. arXiv             deep learning on iot devices. Advances in Neural Information Processing Systems,
       preprint arXiv:2107.00910, 2021.                                                          33:11711–11722, 2020.
[114] Fredrik Kjolstad, Shoaib Kamil, Stephen Chou, David Lugato, and Saman Ama-        [139] Chaoyue Liu, Libin Zhu, and Mikhail Belkin. Loss landscapes and optimization
      rasinghe. The tensor algebra compiler.  In Proceedings of the International               in over-parameterized non-linear systems and neural networks. Applied and
      Conference on Object Oriented Programming Systems Languages and Applications.             Computational Harmonic Analysis, 59:85–116, 2022.
    ACM New York, NY, USA, 2017.                                                     [140] Chenxi Liu, Barret Zoph, Maxim Neumann, Jonathon Shlens, Wei Hua, Li-Jia Li,
[115] Simon Knowles. Graphcore. In IEEE Hot Chips Symposium, pages 1–25, 2021.                 Li Fei-Fei, Alan Yuille, Jonathan Huang, and Kevin Murphy. Progressive neural
[116] Martin Kong, Richard Veras, Kevin Stock, Franz Franchetti, Louis-Noël Pouchet,               architecture search. In Proceedings of the European conference on computer vision
     and Ponnuswamy Sadayappan. When polyhedral transformations meet simd             (ECCV), pages 19–34, 2018.
      code generation. In Proceedings of the ACM SIGPLAN Conference on Programming        [141] Hanxiao Liu, Karen Simonyan, Oriol Vinyals, Chrisantha Fernando, and Koray
      Language Design and Implementation (PLDI), 2013.                                     Kavukcuoglu. Hierarchical representations for efficient architecture search.
[117] Olga Kovaleva, Saurabh Kulshreshtha, Anna Rogers, and Anna Rumshisky.             arXiv preprint arXiv:1711.00436, 2017.
      Bert busters: Outlier dimensions that disrupt transformers.  arXiv preprint        [142] Hanxiao Liu, Karen Simonyan, and Yiming Yang. Darts: Differentiable architec-
       arXiv:2105.06990, 2021.                                                                     ture search. arXiv preprint arXiv:1806.09055, 2018.
[118] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification        [143] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen,
      with deep convolutional neural networks. Commun. ACM, 60(6), 2017.                Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. RoBERTa: A
[119] H. T. Kung, Bradley McDanel, and Sai Qian Zhang. Adaptive tiling: Applying              robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692,
       fixed-size systolic arrays to sparse convolutional neural networks. In 2018 24th               2019.
       International Conference on Pattern Recognition (ICPR), pages 1006–1011, 2018.        [144] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin,
[120] Eldar Kurtic, Daniel Campos, Tuan Nguyen, Elias Frantar, Mark Kurtz, Benjamin            and Baining Guo. Swin transformer: Hierarchical vision transformer using
      Fineran, Michael Goin, and Dan Alistarh. The optimal bert surgeon: Scalable               shifted windows. In Proceedings of the IEEE/CVF International Conference on
     and accurate second-order pruning for large language models. arXiv preprint            Computer Vision, pages 10012–10022, 2021.
       arXiv:2203.07259, 2022.                                                              [145] Zejian Liu, Fanrong Li, Gang Li, and Jian Cheng. Ebert: Efficient bert inference
[121] Andrey Kuzmin, Mart Van Baalen, Yuwei Ren, Markus Nagel, Jorn Peters, and             with dynamic structured pruning. In Findings of the Association for Computa-
     Tijmen Blankevoort. Fp8 quantization: The power of the exponent.  arXiv                tional Linguistics: ACL-IJCNLP 2021, pages 4814–4823, 2021.
       preprint arXiv:2208.09225, 2022.                                                      [146] Liqiang Lu, Naiqing Guan, Yuyue Wang, Liancheng Jia, Zizhang Luo, Jieming
[122] Hyoukjun Kwon, Prasanth Chatarasi, Vivek Sarkar, Tushar Krishna, Michael               Yin, Jason Cong, and Yun Liang.  Tenet: A framework for modeling tensor
       Pellauer, and Angshuman Parashar. Maestro: A data-centric approach to under-              dataflow based on relation-centric notation. In Proceedings of the International
      stand reuse, performance, and hardware cost of dnn mappings. In Proceedings            Symposium on Computer Architecture (ISCA), 2021.
       of the International Symposium on Microarchitecture (MICRO), 2020.                   [147] Liqiang Lu, Yicheng Jin, Hangrui Bi, Zizhang Luo, Peng Li, Tao Wang, and
[123] Woosuk Kwon, Sehoon Kim, Michael W Mahoney, Joseph Hassoun, Kurt Keutzer,           Yun Liang. Sanger: A co-design framework for enabling sparse attention using
     and Amir Gholami. A fast post-training pruning framework for transformers.              reconfigurable architecture. In MICRO-54: 54th Annual IEEE/ACM International
      arXiv preprint arXiv:2204.09656, 2022.                                             Symposium on Microarchitecture, MICRO ’21, page 977–991, New York, NY, USA,
[124] François Lagunas, Ella Charlaix, Victor Sanh, and Alexander M Rush. Block               2021. Association for Computing Machinery.
      pruning for faster transformers. arXiv preprint arXiv:2109.04838, 2021.                [148] Siyuan Lu, Meiqi Wang, Shuang Liang, Jun Lin, and Zhongfeng Wang. Hardware
[125] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush              Accelerator for Multi-Head Attention and Position-Wise Feed-Forward in the
      Sharma, and Radu Soricut. Albert: A lite bert for self-supervised learning of              Transformer. arXiv preprint arXiv:2009.08605, 2020.
      language representations. arXiv preprint arXiv:1909.11942, 2019.                      [149] Renqian Luo, Fei Tian, Tao Qin, Enhong Chen, and Tie-Yan Liu. Neural archi-
[126] Chris Lattner and Vikram Adve. Llvm: A compilation framework for lifelong pro-              tecture optimization. Advances in neural information processing systems, 31,
     gram analysis & transformation. In International Symposium on Code Generation               2018.
     and Optimization (CGO), 2004.                                                      [150] Dmitrii Marin, Jen-Hao Rick Chang, Anurag Ranjan, Anish Prabhu, Mohammad
[127] Bingbing Li, Santosh Pandey, Haowen Fang, Yanjun Lyv, Ji Li, Jieyang Chen,               Rastegari, and Oncel Tuzel. Token pooling in vision transformers. arXiv preprint
     Mimi Xie, Lipeng Wan, Hang Liu, and Caiwen Ding. Ftrans: Energy-efficient               arXiv:2110.03860, 2021.
      acceleration of transformers using fpga. In Proceedings of the ACM/IEEE In-        [151] James McCaffrey. The max trick when computing softmax, 2016.
       ternational Symposium on Low Power Electronics and Design, ISLPED ’20, page        [152] Sachin Mehta and Mohammad Rastegari.  Mobilevit: light-weight, general-
      175–180, New York, NY, USA, 2020. Association for Computing Machinery.                 purpose, and mobile-friendly vision transformer. arXiv preprint arXiv:2110.02178,
[128] Mingzhen Li, Yi Liu, Xiaoyan Liu, Qingxiao Sun, Xin You, Hailong Yang,              2021.
      Zhongzhi Luan, Lin Gan, Guangwen Yang, and Depei Qian. The deep learning        [153] Linyan Mei, Pouya Houshmand, Vikram Jain, Sebastian Giraldo, and Marian
      compiler: A comprehensive survey. IEEE Transactions on Parallel and Distributed                Verhelst. Zigzag: Enlarging joint architecture-mapping design space exploration
      Systems, 32(3):708–727, 2021.                                                                  for dnn accelerators. IEEE Transactions on Computers, 70(8), 2021.
[129] Rui Li, Yufan Xu, Aravind Sukumaran-Rajam, Atanas Rountev, and P Sadayap-        [154] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer
      pan. Analytical characterization and design space exploration for optimization               sentinel mixture models, 2016.
       of cnns. In Proceedings of the International Conference on Architectural Support        [155] Paul Michel, Omer Levy, and Graham Neubig. Are sixteen heads really better
       for Programming Languages and Operation Systems (ASPLOS), 2021.                       than one? arXiv preprint arXiv:1905.10650, 2019.




                                                                                          37
<a id="page-38"></a>

### PDF 第 38 页

[156] Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey,        [179] Jonathan Ragan-Kelley, Connelly Barnes, Andrew Adams, Sylvain Paris, Fredo
      Richard Grisenthwaite, Sangwon Ha, Alexander Heinecke, Patrick Judd, John             Durand, and Saman Amarasinghe. Halide: a language and compiler for optimiz-
      Kamalu, et al. Fp8 formats for deep learning. arXiv preprint arXiv:2209.05433,              ing parallelism, locality, and recomputation in image processing pipelines. Acm
      2022.                                                                                    Sigplan Notices, 2013.
[157] Maxim Milakov and Natalia Gimelshein. Online normalizer calculation for        [180] Jonathan Ragan-Kelley, Connelly Barnes, Andrew Adams, Sylvain Paris, Frédo
      softmax, 2018.                                                                     Durand, and Saman Amarasinghe. Halide: A language and compiler for optimiz-
[158] Ravi Teja Mullapudi, Andrew Adams, Dillon Sharlet, Jonathan Ragan-Kelley,              ing parallelism, locality, and recomputation in image processing pipelines. In
     and Kayvon Fatahalian. Automatically scheduling halide image processing               Proceedings of the ACM SIGPLAN Conference on Programming Language Design
       pipelines. ACM Transactions on Graphics (TOG), 2016.                                and Implementation (PLDI), 2013.
[159] Naveen Muralimanohar, Rajeev Balasubramonian, and Norman P Jouppi. Cacti        [181] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD:
        6.0: A tool to model large caches.                                                      100,000+ questions for machine comprehension of text.   arXiv preprint
[160] Peter Nilsson, Ateeq Ur Rahman Shaik, Rakesh Gangarajaiah, and Erik Hertz.               arXiv:1606.05250, 2016.
     Hardware implementation of the exponential function using taylor series. In        [182] Esteban Real, Alok Aggarwal, Yanping Huang, and Quoc V Le. Regularized
      2014 NORCHIP, pages 1–4, 2014.                                                         evolution for image classifier architecture search. In Proceedings of the aaai
[161] NVIDIA. TensorRT: https://developer.nvidia.com/tensorrt, 2018.                            conference on artificial intelligence, volume 33, pages 4780–4789, 2019.
[162] Auguste Olivry, Guillaume Iooss, Nicolas Tollenaere, Atanas Rountev, P Sa-        [183] Pengzhen Ren, Yun Xiao, Xiaojun Chang, Po-Yao Huang, Zhihui Li, Xiaojiang
      dayappan, and Fabrice Rastello. Ioopt: automatic derivation of i/o complexity             Chen, and Xin Wang. A comprehensive survey of neural architecture search:
     bounds for affine programs. In Proceedings of the ACM SIGPLAN Conference on              Challenges and solutions. ACM Computing Surveys (CSUR), 54(4):1–34, 2021.
     Programming Language Design and Implementation (PLDI), 2021.                     [184] Nadav Rotem, Jordan Fix, Saleem Abdulrasool, Garret Catron, Summer Deng,
[163] Angshuman Parashar, Priyanka Raina, Yakun Sophia Shao, Yu-Hsin Chen,          Roman Dzhabarov, Nick Gibson, James Hegeman, Meghan Lele, Roman Lev-
      Victor A Ying, Anurag Mukkara, Rangharajan Venkatesan, Brucek Khailany,               enstein, et al. Glow: Graph lowering compiler techniques for neural networks.
      Stephen W Keckler, and Joel Emer. Timeloop: A systematic approach to dnn              arXiv preprint arXiv:1805.00907, 2018.
      accelerator evaluation. In 2019 IEEE international symposium on performance        [185] Amit Sabne. Xla: Compiling machine learning for peak performance. 2020.
       analysis of systems and software (ISPASS), pages 304–315. IEEE, 2019.                 [186] Hassan Sajjad, Fahim Dalvi, Nadir Durrani, and Preslav Nakov. Poor man’s bert:
[164] Angshuman Parashar, Minsoo Rhu, Anurag Mukkara, Antonio Puglielli, Rang-             Smaller and faster transformer models. arXiv preprint arXiv:2004.03844, 2020.
      harajan Venkatesan, Brucek Khailany, Joel Emer, Stephen W. Keckler, and        [187] Ananda Samajdar, Jan Moritz Joseph, Yuhao Zhu, Paul Whatmough, Matthew
      William J. Dally. Scnn: An accelerator for compressed-sparse convolutional              Mattina, and Tushar Krishna. A systematic methodology for characterizing
      neural networks. In Proceedings of the 44th Annual International Symposium                 scalability of dnn accelerators using scale-sim. In Proceedings of the International
     on Computer Architecture, ISCA ’17, page 27–40, New York, NY, USA, 2017.            Symposium on Performance Analysis of Systems and Software (ISPASS), 2020.
      Association for Computing Machinery.                                              [188] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-
[165] Eunjung Park, John Cavazos, Louis-Noël Pouchet, Cédric Bastoul, Albert Cohen,             Chieh Chen. MobilenetV2: Inverted residuals and linear bottlenecks. In Proceed-
     and P Sadayappan. Predictive modeling in a polyhedral optimization space.               ings of the IEEE Conference on Computer Vision and Pattern Recognition, pages
       International journal of parallel programming, 2013.                                      4510–4520, 2018.
[166] Junki Park, Hyunsung Yoon, Daehyun Ahn, Jungwook Choi, and Jae-Joon Kim.        [189] Victor Sanh, Thomas Wolf, and Alexander Rush. Movement pruning: Adaptive
      Optimus: Optimized matrix multiplication structure for transformer neural net-              sparsity by fine-tuning. Advances in Neural Information Processing Systems,
     work accelerator. In I. Dhillon, D. Papailiopoulos, and V. Sze, editors, Proceedings              33:20378–20389, 2020.
       of Machine Learning and Systems, volume 2, pages 363–378. 2020.                     [190] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel
[167] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gre-             Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias
      gory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga,                Gallé, et al. Bloom: A 176b-parameter open-access multilingual language model.
       et al. Pytorch: An imperative style, high-performance deep learning library.             arXiv preprint arXiv:2211.05100, 2022.
      Advances in neural information processing systems, 32, 2019.                          [191] Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Q
[168] Suchita Pati, Shaizeen Aga, Nuwan Jayasena, and Matthew D Sinclair. Demysti-              Tran, Yi Tay, and Donald Metzler. Confident adaptive language modeling. arXiv
      fying bert: Implications for accelerator design. In International Symposium on               preprint arXiv:2207.07061, 2022.
      Workload Characterization (IISWC), 2021.                                            [192] Tal Schuster, Adam Fisch, Tommi Jaakkola, and Regina Barzilay.  Consis-
[169] Jing Pei, Lei Deng, Sen Song, Mingguo Zhao, Youhui Zhang, Shuang Wu, Guan-               tent accelerated inference via confident adaptive transformers. arXiv preprint
       rui Wang, Zhe Zou, Zhenzhi Wu, Wei He, et al. Towards artificial general               arXiv:2104.08803, 2021.
       intelligence with hybrid tianjic chip architecture. Nature, 572(7767):106–111,        [193] Lukas Sekanina. Neural architecture search and hardware accelerator co-search:
      2019.                                                   A survey. IEEE Access, 9:151337–151362, 2021.
[170] Hieu Pham, Melody Guan, Barret Zoph, Quoc Le, and Jeff Dean.  Efficient        [194] Guan Shen, Jieru Zhao, Quan Chen, Jingwen Leng, Chao Li, and Minyi Guo. Salo:
      neural architecture search via parameters sharing. In International conference          An efficient spatial accelerator enabling hybrid sparse attention mechanisms
      on machine learning, pages 4095–4104. PMLR, 2018.                                          for long sequences. In Proceedings of the 59th ACM/IEEE Design Automation
[171] Raghu Prabhakar and Sumti Jairath. Sambanova sn10 rdu: Accelerating software               Conference, DAC ’22, page 571–576, New York, NY, USA, 2022. Association for
       2.0 with dataflow. In IEEE Hot Chips Symposium, pages 1–37, 2021.                     Computing Machinery.
[172] Zheng Qu, Liu Liu, Fengbin Tu, Zhaodong Chen, Yufei Ding, and Yuan Xie.        [195] Sheng Shen, Zhen Dong, Jiayu Ye, Linjian Ma, Zhewei Yao, Amir Gholami,
      Dota: Detect and omit weak attentions for scalable transformer acceleration. In             Michael W Mahoney, and Kurt Keutzer. Q-BERT: Hessian based ultra low
      Proceedings of the 27th ACM International Conference on Architectural Support               precision quantization of bert. In AAAI, pages 8815–8821, 2020.
       for Programming Languages and Operating Systems, ASPLOS ’22, page 14–26,        [196] Frans Sijstermans. The NVIDIA Deep Learning Accelerator. In Hot Chips, 2018.
    New York, NY, USA, 2022. Association for Computing Machinery.                    [197] Karen Simonyan and Andrew Zisserman. Very Deep Convolutional Networks
[173] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving               for Large-scale Image Recognition. CoRR, abs/1408.1556, 2014.
      language understanding by generative pre-training, 2018.                            [198] Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Ra-
[174] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya               jbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay
       Sutskever. Language models are unsupervised multitask learners. OpenAI blog,               Korthikanti, et al. Using deepspeed and megatron to train megatron-turing nlg
        1(8):9, 2019.                                                                                530b, a large-scale generative language model. arXiv preprint arXiv:2201.11990,
[175] Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann,              2022.
      Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young,        [199] Jasper Snoek, Hugo Larochelle, and Ryan P Adams. Practical bayesian optimiza-
       et al. Scaling language models: Methods, analysis & insights from training               tion of machine learning algorithms. Advances in neural information processing
      gopher. arXiv preprint arXiv:2112.11446, 2021.                                               systems, 25, 2012.
[176] Jack W Rae, Anna Potapenko, Siddhant M Jayakumar, and Timothy P Lillicrap.        [200] David So, Quoc Le, and Chen Liang. The evolved transformer. In International
      Compressive transformers for long-range sequence modelling. arXiv preprint              Conference on Machine Learning, pages 5877–5886. PMLR, 2019.
       arXiv:1911.05507, 2019.                                                              [201] David So, Wojciech Mańke, Hanxiao Liu, Zihang Dai, Noam Shazeer, and Quoc V
[177] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang,               Le. Searching for efficient transformers for language modeling. Advances in
      Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits              Neural Information Processing Systems, 34:6010–6022, 2021.
      of transfer learning with a unified text-to-text transformer.  arXiv preprint        [202] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning,
       arXiv:1910.10683, 2019.                                                   Andrew Y Ng, and Christopher Potts. Recursive deep models for semantic
[178] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang,              compositionality over a sentiment treebank. In Proceedings of the 2013 conference
      Michael Matena, Yanqi Zhou, Wei Li, Peter J Liu, et al. Exploring the limits of             on empirical methods in natural language processing, pages 1631–1642, 2013.
       transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res.,        [203] Jacob R. Stevens, Rangharajan Venkatesan, Steve Dai, Brucek Khailany, and
       21(140):1–67, 2020.                                                       Anand Raghunathan. Softermax: Hardware/software co-design of an efficient




                                                                                          38
<a id="page-39"></a>

### PDF 第 39 页

softmax for transformers. In 2021 58th ACM/IEEE Design Automation Conference              IEEE, 2021.
      (DAC), pages 469–474, 2021.                                                         [224] Kuan Wang, Zhijian Liu, Yujun Lin, Ji Lin, and Song Han. HAQ: Hardware-
[204] Yang Sun, Wei Hu, Fang Liu, Min Jiang, FeiHu Huang, and Dian Xu. Speformer:            aware automated quantization. In Proceedings of the IEEE conference on computer
    An efficient hardware-software cooperative solution for sparse spectral trans-               vision and pattern recognition, 2019.
      former. In 2022 IEEE 9th International Conference on Cyber Security and Cloud        [225] Meiqi Wang, Siyuan Lu, Danyang Zhu, Jun Lin, and Zhongfeng Wang. A high-
      Computing (CSCloud)/2022 IEEE 8th International Conference on Edge Computing             speed and low-complexity architecture for softmax function in deep learning.
     and Scalable Cloud (EdgeCom), pages 180–185, 2022.                                       In 2018 IEEE Asia Pacific Conference on Circuits and Systems (APCCAS), pages
[205] Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, and Denny              223–226, 2018.
     Zhou. Mobilebert: a compact task-agnostic bert for resource-limited devices.        [226] Adina Williams, Nikita Nangia, and Samuel R Bowman. A broad-coverage
      arXiv preprint arXiv:2004.02984, 2020.                                                     challenge corpus for sentence understanding through inference. arXiv preprint
[206] Vivienne Sze, Yu-Hsin Chen, Tien-Ju Yang, and Joel Emer. Efficient processing               arXiv:1704.05426, 2017.
       of deep neural networks: A tutorial and survey, 2017.                                [227] Samuel Williams, Andrew Waterman, and David Patterson. Roofline: an insight-
[207] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir                ful visual performance model for multicore architectures. Communications of
      Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going               the ACM, 52(4):65–76, 2009.
      deeper with convolutions. In 2015 IEEE Conference on Computer Vision and        [228] Bichen Wu, Xiaoliang Dai, Peizhao Zhang, Yanghan Wang, Fei Sun, Yiming Wu,
       Pattern Recognition (CVPR), pages 1–9, 2015.                                       Yuandong Tian, Peter Vajda, Yangqing Jia, and Kurt Keutzer. Fbnet: Hardware-
[208] Emil Talpes, Debjit Das Sarma, Ganesh Venkataramanan, Peter Bannon, Bill            aware efficient convnet design via differentiable neural architecture search. In
     McGee, Benjamin Floering, Ankit Jalote, Christopher Hsiong, Sahil Arora,               Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recogni-
     Atchyuth Gorti, et al. Compute solution for tesla’s full self-driving computer.                 tion, pages 10734–10742, 2019.
      IEEE Micro, 40(2):25–35, 2020.                                                       [229] Bichen Wu, Yanghan Wang, Peizhao Zhang, Yuandong Tian, Peter Vajda, and
[209] Thierry Tambe, Coleman Hooper, Lillian Pentecost, Tianyu Jia, En-Yu Yang,             Kurt Keutzer. Mixed precision quantization of convnets via differentiable neural
     Marco Donato, Victor Sanh, Paul Whatmough, Alexander M. Rush, David Brooks,               architecture search. arXiv preprint arXiv:1812.00090, 2018.
     and Gu-Yeon Wei. EdgeBERT: Sentence-Level Energy Optimizations for Latency-        [230] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decom-
     Aware Multi-Task NLP Inference. page 830–844, 2021.                                      position transformers with auto-correlation for long-term series forecasting.
[210] Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew             Advances in Neural Information Processing Systems, 34:22419–22430, 2021.
     Howard, and Quoc V Le. Mnasnet: Platform-aware neural architecture search        [231] Yannan Nellie Wu, Joel S Emer, and Vivienne Sze. Accelergy: An architecture-
       for mobile. In Proceedings of the IEEE/CVF Conference on Computer Vision and                level energy estimation methodology for accelerator designs. In 2019 IEEE/ACM
       Pattern Recognition, pages 2820–2828, 2019.                                                   International Conference on Computer-Aided Design (ICCAD), pages 1–8. IEEE,
[211] Mingxing Tan and Quoc Le. EfficientNet: Rethinking model scaling for convo-              2019.
       lutional neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov,        [232] Zhanghao Wu, Zhijian Liu, Ji Lin, Yujun Lin, and Song Han. Lite transformer
       editors, Proceedings of the 36th International Conference on Machine Learning,             with long-short range attention. arXiv preprint arXiv:2004.11886, 2020.
     volume 97 of Proceedings of Machine Learning Research, pages 6105–6114. PMLR,        [233] Mengzhou Xia, Zexuan Zhong, and Danqi Chen. Structured pruning learns
      09–15 Jun 2019.                                                                compact and accurate models. arXiv preprint arXiv:2204.00408, 2022.
[212] James W Thomas, John P Okada, Peter Markstein, and Ren-Chang Li. The        [234]  Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, and Jimmy Lin. Deebert: Dynamic
      libm library and floatingpoint arithmetic in hp-ux for itanium-based systems.              early exiting for accelerating bert inference. arXiv preprint arXiv:2004.12993,
      Technical report, Technical report, Hewlett-Packard Company, Palo Alto, CA,              2020.
     USA, 2004.                                                                         [235]  Jin Xu, Xu Tan, Renqian Luo, Kaitao Song, Jian Li, Tao Qin, and Tie-Yan Liu. Nas-
[213] Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kul-                bert: task-agnostic and adaptive-size bert compression with neural architecture
      shreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al.              search.  In Proceedings of the 27th ACM SIGKDD Conference on Knowledge
      Lamda: Language models for dialog applications. arXiv preprint arXiv:2201.08239,              Discovery & Data Mining, pages 1933–1943, 2021.
      2022.                                                                               [236] Longxing Yang, Yu Hu, Shun Lu, Zihao Sun, Jilin Mei, Yinhe Han, and Xiaowei
[214] Philippe Tillet, HT Kung, and David Cox. Triton: an intermediate language                  Li. Searching for burgerformer with micro-meso-macro space design. In Inter-
     and compiler for tiled neural network computations. In Proceedings of the 3rd               national Conference on Machine Learning, pages 25055–25069. PMLR, 2022.
    ACM SIGPLAN International Workshop on Machine Learning and Programming        [237] Tao Yang, Hui Ma, Xiaoling Li, Fangxin Liu, Yilong Zhao, Zhezhi He, and Li Jiang.
      Languages, 2019.                                                                        Dtatrans: Leveraging dynamic token-based quantization with accuracy com-
[215] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre              pensation mechanism for efficient transformer architecture. IEEE Transactions
       Sablayrolles, and Hervé Jégou. Training data-efficient image transformers &             on Computer-Aided Design of Integrated Circuits and Systems, pages 1–1, 2022.
       distillation through attention. In International Conference on Machine Learning,        [238] Tien-Ju Yang, Andrew Howard, Bo Chen, Xiao Zhang, Alec Go, Mark Sandler,
      pages 10347–10357. PMLR, 2021.                                                     Vivienne Sze, and Hartwig Adam. Netadapt: Platform-aware neural network
[216] Nicolas Vasilache, Oleksandr Zinenko, Theodoros Theodoridis, Priya Goyal,              adaptation for mobile applications. In Proceedings of the European Conference
     Zachary DeVito, William S. Moses, Sven Verdoolaege, Andrew Adams, and             on Computer Vision (ECCV), pages 285–300, 2018.
      Albert Cohen. Tensor comprehensions: Framework-agnostic high-performance        [239] Xuan Yang, Mingyu Gao, Qiaoyi Liu, JeffSetter, Jing Pu, Ankita Nayak, Steven
     machine learning abstractions, 2018.                                                                Bell, Kaidi Cao, Heonjae Ha, Priyanka Raina, et al. Interstellar: Using halide’s
[217] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,              scheduling language to analyze dnn accelerators. In Proceedings of the Twenty-
     Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need.                Fifth International Conference on Architectural Support for Programming Lan-
      In Advances in neural information processing systems, pages 5998–6008, 2017.               guages and Operating Systems, pages 369–383, 2020.
[218] Rangharajan Venkatesan, Yakun Sophia Shao, Miaorong Wang, Jason Clemons,        [240] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov,
      Steve Dai, Matthew Fojtik, Ben Keller, Alicia Klinefelter, Nathaniel Pinckney,            and Quoc V Le. XLNet: Generalized autoregressive pretraining for language
      Priyanka Raina, et al. Magnet: A modular accelerator generator for neural              understanding. In Advances in neural information processing systems, pages
      networks. In Proceedings of the International Conference on Computer-Aided              5753–5763, 2019.
      Design (ICCAD), 2019.                                                              [241] Zhewei Yao, Zhen Dong, Zhangcheng Zheng, Amir Gholami, Jiali Yu, Eric Tan,
[219] Alvin Wan, Xiaoliang Dai, Peizhao Zhang, Zijian He, Yuandong Tian, Saining            Leyuan Wang, Qijing Huang, Yida Wang, Michael W Mahoney, and Kurt Keutzer.
       Xie, Bichen Wu, Matthew Yu, Tao Xu, Kan Chen, et al. Fbnetv2: Differentiable          HAWQV3: Dyadic neural network quantization. arXiv preprint arXiv:2011.10680,
      neural architecture search for spatial and channel dimensions. In Proceedings               2020.
       of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages        [242] Zhewei Yao, Amir Gholami, Kurt Keutzer, and Michael W. Mahoney. PyHessian:
      12965–12974, 2020.                                                                   Neural networks through the lens of the Hessian. arXiv preprint arXiv:1912.07145,
[220] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and               2019.
     Samuel R Bowman. GLUE: A multi-task benchmark and analysis platform        [243] Jiahui Yu, Pengchong Jin, Hanxiao Liu, Gabriel Bender, Pieter-Jan Kindermans,
       for natural language understanding. arXiv preprint arXiv:1804.07461, 2018.               Mingxing Tan, Thomas Huang, Xiaodan Song, Ruoming Pang, and Quoc Le.
[221] Dilin Wang, Chengyue Gong, Meng Li, Qiang Liu, and Vikas Chandra. Al-              Bignas: Scaling up neural architecture search with big single-stage models. In
      phanet: improved training of supernets with alpha-divergence. In International             European Conference on Computer Vision, pages 702–717. Springer, 2020.
      Conference on Machine Learning, pages 10760–10771. PMLR, 2021.                    [244] Joonsang Yu, Junki Park, Seongmin Park, Minsoo Kim, Sihwa Lee, Dong Hyun
[222] Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan,               Lee, and Jungwook Choi. Nn-lut: Neural approximation of non-linear operations
     and Song Han. Hat: Hardware-aware transformers for efficient natural language                for efficient transformer inference. In Proceedings of the 59th ACM/IEEE Design
      processing. arXiv preprint arXiv:2005.14187, 2020.                                     Automation Conference, DAC ’22, page 577–582, New York, NY, USA, 2022.
[223] Hanrui Wang, Zhekai Zhang, and Song Han. Spatten: Efficient sparse attention              Association for Computing Machinery.
      architecture with cascade token and head pruning. In 2021 IEEE International        [245] Shixing Yu, Zhewei Yao, Amir Gholami, Zhen Dong, Sehoon Kim, Michael W
     Symposium on High-Performance Computer Architecture (HPCA), pages 97–110.            Mahoney, and Kurt Keutzer. Hessian-aware pruning and optimal neural implant.




                                                                                          39
<a id="page-40"></a>

### PDF 第 40 页

In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer
       Vision, pages 3880–3891, 2022.
[246] Ali Hadi Zadeh and A. Moshovos. Gobo: Quantizing attention-based nlp models
       for low latency and energy efficient inference. In 53rd IEEE/ACM International
     Symposium on Microarchitecture (MICRO), 2020.
[247] Dan Zhang, Safeen Huda, Ebrahim Songhori, Kartik Prabhu, Quoc Le, Anna
      Goldie, and Azalia Mirhoseini. A full-stack search technique for domain opti-
     mized deep learning accelerators. In Proceedings of the 27th ACM International
      Conference on Architectural Support for Programming Languages and Operating
      Systems, pages 27–42, 2022.
[248] Shijin Zhang, Zidong Du, Lei Zhang, Huiying Lan, Shaoli Liu, Ling Li, Qi Guo,
      Tianshi Chen, and Yunji Chen. Cambricon-x: An accelerator for sparse neural
      networks. In 2016 49th Annual IEEE/ACM International Symposium on Microar-
       chitecture (MICRO), pages 1–12, 2016.
[249] Lianmin Zheng, Chengfan Jia, Minmin Sun, Zhao Wu, Cody Hao Yu, Ameer
       Haj-Ali, Yida Wang, Jun Yang, Danyang Zhuo, Koushik Sen, Joseph E. Gonzalez,
     and Ion Stoica. Ansor: Generating High-Performance Tensor Programs for Deep
      Learning. Technical report, arXiv. arXiv:2006.06762 [cs, stat] type: article.
[250] Zhao Zhong, Junjie Yan, Wei Wu, Jing Shao, and Cheng-Lin Liu.  Practical
      block-wise neural network architecture generation. In Proceedings of the IEEE
      conference on computer vision and pattern recognition, pages 2423–2432, 2018.
[251] Xuda Zhou, Zidong Du, Qi Guo, Shaoli Liu, Chengsi Liu, Chao Wang, Xuehai
      Zhou, Ling Li, Tianshi Chen, and Yunji Chen. Cambricon-s: Addressing irreg-
       ularity in sparse neural networks through a cooperative software/hardware
      approach. In 2018 51st Annual IEEE/ACM International Symposium on Microar-
       chitecture (MICRO), pages 15–28, 2018.
[252] Yanqi Zhou, Sudip Roy, Amirali Abdolrashidi, Daniel Wong, Peter Ma, Qiumin
      Xu, Hanxiao Liu, Phitchaya Phothilimtha, Shen Wang, Anna Goldie, et al. Trans-
       ferable graph optimizers for ml compilers. In Proceedings of the Conference on
      Neural Information Processing Systems (NeurIPS), 2020.
[253] Zhe Zhou, Junlin Liu, Zhenyu Gu, and Guangyu Sun. Energon: Towards efficient
       acceleration of transformers using dynamic sparse attention. IEEE Transactions
      on Computer-Aided Design of Integrated Circuits and Systems, pages 1–1, 2022.
[254] Barret Zoph and Quoc V Le. Neural architecture search with reinforcement
       learning. arXiv preprint arXiv:1611.01578, 2016.
[255] Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V Le. Learning
       transferable architectures for scalable image recognition. In Proceedings of the
      IEEE conference on computer vision and pattern recognition, pages 8697–8710,
      2018.





                                                                                          40
<a id="page-41"></a>

### PDF 第 41 页

A  APPENDIX
A.1  Decoder Model Architecture
Fig. 27 illustrates the computations performed in the MHA and FFN modules of the Transformer decoder. Compared to the Transformer
encoder, the main differences are (1) that the majority of the matmuls are matrix-vector operations and (2) that the keys and values from the
previous token generation iterations are cached.




                                                            WQ
                                                   𝑑x 𝑑         𝑑/ℎx 1                                                        Cached Key,
                                                    𝑑x 1                                                                            Value Tokens
                                                              WK                                     𝑙x 𝑑/ℎ          𝑙x 𝑑/ℎ                                Previous 𝑙-1
                                                   𝑑x 𝑑                                     Concatenate           Transpose                                   Key Tokens
                                                                         𝑑/ℎx 1                                          Decoder
                                                                                                                                              𝑙x 1                                  (𝑙-1 x d)
                                              Input   𝑑x 1        𝑑x 1
                                                                                                                               Softmax
                                                               WV                                                                                                              Previous 𝑙-1
                                                                                                                                                                                           Tokens                                                  𝑑x 𝑑                                                                                              𝑑/ℎx 𝑙   𝑙x 1                                           Value                                                                         𝑑/ℎx 1
                                                                                                                                                                                                                                                                    (𝑙-1                                                                                                                                                Concatenate                                                    𝑑x 1                                        𝑑/ℎx 1                                                 x d)
                                                                                                                                                                                                  Split Across h
                                                                                                                                                                        Attention Heads
                                                                                                    Split Across h
                                                                                      Attention Heads

                                                                                                                                         Concatenate

                                                                                                   WOut             Norm + Add
                                                             𝑑x 𝑑
                                                                                                                                 Attention
                                                                                                        LayerNorm
                                                                                                                     Output                                                                              𝑑x 𝑙                                                      𝑑x 1     𝑑x 1


                                              Muti-Head Attention (MHA) Module



                                                                      W1                             W2           Norm + Add
                                                                           𝑑                                                                     𝑑𝐹𝐹𝑁x                                                                                                                    𝑑𝐹𝐹𝑁                                                                                           𝑑𝐹𝐹𝑁x 1  𝑑x                                                    𝑑x 1                                                                                                                                              Encoder                                                            Attention                                                                                                                                      LayerNorm                                                                                    GELU                                                                                                                                               Output                                                       Output                                                                                                            𝑑x 1                                                                               1                                                                                    𝑑x                                                                                                             1                                                                       𝑑𝐹𝐹𝑁x


                                             Feed-Forward Network (FFN) Module

Figure 27: Map of the computations performed in the Transformer decoder. The decoder is primarily composed of matrix-vector operations. The
diagram displays the computation for one decoder block and for the 𝑙th iteration.


A.2  High-Level Overview of CNN Architecture
Convolutional Neural Networks (CNNs) are a class of neural networks which were popularized by the release of AlexNet in 2012 and have
seen widespread use across computer vision applications [90, 118, 197, 207, 211]. These networks leverage convolutions, which are operations
that apply a set of weights (also referred to as a filter or kernel) to groups of elements in the input. The CNN model used for baseline
comparisons in this paper is ResNet-50, which is a popular architecture for vision applications [83]. The ResNet-50 model architecture is
outlined in Fig. 28.



                                          Size: 112   Size: 56         Size: 28               Size: 14                Size: 7



                                               Conv1        Block 0       Block 1       Block 1       Block 2       Block 2       Block 3       Block 3        LayerFC



                                               X 3                 X 3                 X 5                 X 2

                                                           Kernel Size, # Output Channels

                                             Conv1         Block 0        Block 1        Block 2        Block 3

                                                      7x7, 64 / 2     1x1, 64         1x1, 128 *      1x1, 256 *      1x1, 512 *

                                                                      3x3, 64         3x3, 128       3x3, 256       3x3, 512

                                                                      1x1, 256       1x1, 512       1x1, 1024      1x1, 2048

Figure 28: Diagram of the ResNet-50 model architecture. The operations with a star beside them have a stride of 2 for the first block of that type.
The arrows correspond to residual additions. The dotted arrows correspond to additional 1×1 convolutional layers that project the previous input
to match the dimension of the output of the block. ReLU, BatchNorm, and Softmax layers are omitted for simplicity.





                                                                                          41
<a id="page-42"></a>

### PDF 第 42 页

(a) Convolution                                      (b) Average Pooling                          (c) Max Pooling

              Kernel slides over the input                                             Kernel averages values                            Kernel selects maximum value

                                                  1 ∗1 −1 ∗2 −1 ∗3 + 1 ∗4 = 0
               1   2                                                   1   2                                      1   2
                                     0                                                                 2.5                                         4
   1    -1       3   4                                        1/4  1/4      3   4                                      3   4

   -1   1                                                      1/4  1/4




   Kernel              Input            Output Channel           Kernel              Input            Output Channel                Input            Output Channel


                   Figure 29: Diagrams outlining the Convolution, Average Pooling, and Max Pooling operations.


  The basic convolution operation is outlined in Fig. 29, assuming two-dimensional inputs. In 2D, the convolution can be viewed as applying
a sliding kernel across the input matrix in order to produce the output matrix. When multiple kernels are applied to the image, each kernel
produces a separate output channel. The spacing between successive filter applications is termed the stride; for example, a stride of 2 means
that the kernel is only applied to every second set of input pixels.
  CNNs also contain several other operations, such as ReLU, Batch Normalization, and Pooling. ReLU is a nonlinear activation function that
can be expressed as 𝑅𝑒𝐿𝑈(𝑥) = 𝑚𝑎𝑥(0,𝑥). Batch Normalization (or BatchNorm) is also used in CNNs instead of LayerNorm. BatchNorm is
outlined graphically in Fig. 2. As opposed to LayerNorm, BatchNorm normalizes the data per channel in the input tensor and uses statistics
computed at training time. This means that the BatchNorm operation can be fused with a prior convolution without impacting the requisite
tiling dimensions; in fact, BatchNorm layers can be folded with convolutions to also eliminate the added FLOPs from these layers.
  CNNs also contain pooling layers for downsampling. Pooling layers are similar to CNNs in that they apply a filter element-wise to the
input. However, these pooling filters use fixed patterns, such as using a filter made up of equal values in the case of Average Pooling, or
selecting the maximum element in the group in the case of Max Pooling. These pooling operations are outlined graphically in Fig. 29. Note
that some networks use strided convolutions for downsampling instead of incorporating pooling layers [83]. Finally, CNNs also often use
one or more fully connected layers followed by the Softmax function for the output classifier [83, 90, 118, 197, 207, 211].

A.3  Additional Profiling Results
Tab. 10 provides per-layer breakdowns of the FLOPs, MOPs, and arithmetic intensity for a hypothetical BERT model with only 4 attention
heads for sequence lengths of 128, 512, and 4096. Tab. 11 provides per-layer breakdowns of the FLOPs, MOPs, and arithmetic intensity for
GPT-2 for sequence lengths of 128, 512, and 4096.


Table 10: Per-Layer FLOPs, memory operations (MOPs), and arithmetic intensity for the hypothetical BERT-Base encoder with 4 attention heads
and with sequence lengths of 128, 512, and 4096 tokens. The number of FLOPs consumed by each operation for each sequence length is similar to
the BERT-Base encoder with 12 attention heads (Tab. 3). However, the number of MOPs consumed by the activation-to-activation matmuls are
significantly lower for each sequence length relative to the BERT-Base encoder with 12 attention heads. This leads to greater arithmetic intensity
in the activation-to-activation matmuls and for end-to-end inference when using 4 attention heads rather than 12 attention heads.

     Sequence Length          Operator         FLOPs (× 109)  % of total FLOPs  MOPs (× 109)  % of total MOPs   Arithmetic Intensity

                MHA (projections)            7.25              0.32              0.04              0.28                192.00
              MHA (act-to-act matmuls)        0.60              0.03              0.01             0.047                95.69
            128           FFN (projections)            14.47              0.65              0.07              0.51                211.86
                                Other                   0.07              0.003              0.02              0.16                 3.30
                                     Total                  22.42             1                0.13             1                 167.14

                MHA (projections)           28.99              0.30              0.07              0.21                438.86
              MHA (act-to-act matmuls)        9.65              0.10              0.04              0.14                219.04
            512           FFN (projections)            57.98              0.60              0.10              0.32                558.54
                                Other                   0.32              0.003              0.10              0.33                 3.07
                                     Total                  96.94             1                0.32             1                 303.59

                MHA (projections)          231.93             0.18              0.33              0.07                702.17
              MHA (act-to-act matmuls)       617.63             0.47              1.76              0.37                350.61
           4096          FFN (projections)           463.86             0.35              0.43              0.09               1068.52
                                Other                   5.41              0.004              2.25              0.47                 2.40
                                     Total                1318.83            1                4.78             1                 276.00





                                                                                          42
<a id="page-43"></a>

### PDF 第 43 页

Table 11: Per-Layer FLOPs, MOPs, and arithmetic intensity for the GPT-2 decoder with sequence lengths of 128, 512, and 4096 tokens. The number
of FLOPs is similar to the BERT-Base encoder (provided in Tab. 3). However, the number of MOPs is much larger than in the BERT-Base encoder.
This results in lower arithmetic intensity in the GPT-2 decoder than the BERT-Base encoder.

     Sequence Length          Operator         FLOPs (× 109)  % of total FLOPs  MOPs (× 109)  % of total MOPs   Arithmetic Intensity

                MHA (projections)            7.25             33               3.63             33                  2.00
              MHA (act-to-act matmuls)        30               0.01              0.16             1                   1.92
            128           FFN (projections)            14.50             66               7.26             66                  2.00
                                Other                   0.07                0.3               0.03               0.3                  2.58
                                     Total                  22.12            100              11.08            100                   2.0

                MHA (projections)           28.99             32              14.53            32                  2.00
              MHA (act-to-act matmuls)        4.83              5                2.45             5                   2.00
            512           FFN (projections)            57.98             63              29.04            63                  2.00
                                Other                   0.35                0.4               0.14               0.3                  2.47
                                     Total                  92.15            100              46.17            100                  2.00

                MHA (projections)          231.93            23             116.27            23                  2.00
              MHA (act-to-act matmuls)       309.24            31             155.98            31                  1.98
           4096          FFN (projections)           463.86            46             232.31            46                    2.0
                                Other                   7.02                0.7               3.25               0.6                  2.16
                                     Total                1012.04           100             507.80           100                  1.99


A.4  Additional ResNet-50 Workload Analysis
Tab. 12 provides detailed analysis of the FLOPs, MOPs, and arithmetic intensity for several convolutional layers in ResNet-50.


          Table 12: FLOPs, memory operations (MOPs), and arithmetic intensity for different convolutional layers in ResNet-50.

                       Kernel Size   Output Channels   Output Size  FLOPs (× 109)  MOPs (× 109)   Arithmetic Intensity

                       1×1             64           56×56           0.31           0.0031             100.76
                       3×3             64           56×56           0.69           0.0013             527.55
                       1×1            256           56×56           0.31           0.0031             100.76
                       1×1            128           28×28           0.31           0.0017             181.14
                       3×3            128           28×28           0.92           0.0014             664.09
                       1×1            512           28×28           0.41           0.0023             181.14
                       1×1            256           14×14           0.51           0.0026             200.30
                       3×3            256           14×14           1.39           0.0041             335.00
                       1×1            1024          14×14           0.62           0.0031             200.30
                       1×1            512           7×7            0.21           0.0023              87.53
                       3×3            512           7×7            0.69           0.0072              95.96
                       1×1            2048          7×7            0.31           0.0035              87.53



A.5  Additional Analytical Modeling Results
  Latency Breakdown and End-to-end Runtime. We modeled the performance breakdown of the BERT-Base and BERT-Large encoders,
under the assumption of square tiling for all matrix operations, and no operation fusion (i.e., each operation required inputs to be read
from external memory and outputs to be flushed out). We also modeled the performance breakdown of the GPT-2 decoder under the
same assumption of no operation fusion. The latency breakdowns for BERT-Base and GPT-2 for different sequence lengths are provided in
Fig. 30 and 31, respectively. Fig. 32 shows the runtime latency of BERT-Base, BERT-Large, and GPT-2, normalized to the runtime latency
of BERT-Base with a sequence length of 128. The runtime scaling and breakdowns from the analytical model were similar to the trends
observed when profiling inference on the CPU in Section 2.2.2. However, for a fixed sequence length, the MHA computation takes up a
greater proportion of the computation on the CPU. Note that the analytical model was designed assuming a hardware architecture that
was different from the CPU architecture, and the relative breakdown between different operations would not necessarily be the same for
different hardware platforms.

  Comparison with ResNet-50. We also modeled ResNet-50 to provide a baseline for our analysis. We first compared the runtime between
BERT-Base and ResNet-50 without operation fusing. We found that the runtime of ResNet-50 was 1.28 times faster than the runtime of
BERT-Base with a sequence length of 128. As outlined in Section 2.2.1, ResNet-50 contains 3.07 times fewer FLOPs and 1.28 times fewer




                                                                                          43
<a id="page-44"></a>

### PDF 第 44 页

BERT-Base Latency Breakdown (Analytic Model)
                                       100
                                     (%)
                                                                                       Latency 75
                         of 50


                                        25                                                                                                                             Percentage

                                         0
                                              128     256     512     1024    2048    4096
                                                          Sequence Length
                                 MHA (act-to-act)      MHA (proj.)        FFN (proj.)        Other

Figure 30: Plot of the computation breakdown in the BERT-Base encoder versus sequence length using our analytical model. Proj. and act-to-act
refer to the projection operation (i.e., activation-to-weight matmul) and the activation-to-activation matmul, respectively. Other refers to the
non-matmul operations.

                                          GPT-2 Latency Breakdown (Analytic Model)
                                       100
                                     (%)
                                                                                       Latency 75
                         of 50


                                        25                                                                                                                             Percentage

                                         0
                                              128     256     512     1024    2048    4096
                                                          Sequence Length
                                 MHA (act-to-act)      MHA (proj.)        FFN (proj.)        Other

Figure 31: Plot of the computation breakdown in the GPT-2 decoder versus sequence length using our analytical model. Proj. and act-to-act
refer to the projection operation (i.e., activation-to-weight matmul) and the activation-to-activation matmul, respectively. Other refers to the
non-matmul operations.

                                    Normalized Latency of Transformer Models (Analytic Model)
                                                    BERT-Base                                          840
                                                     BERT-Large
                                     750       GPT-2
                                                                                   Latency
                                     500

                                                                                            352                                                                                                                       Normalized 250                                                        229
                                                                                  159
                                                                         75                82      80
                                                 1 3 18    2 7 37    5 14      1133      28
                                             128      256      512      1024     2048     4096
                                                           Sequence Length

Figure 32: Plot of the runtime latency of the BERT-Base and BERT-Large encoders and the GPT-2 decoder versus sequence length using our
analytical model, normalized to the runtime of BERT-Base with a sequence length of 128.





                                                                                          44
<a id="page-45"></a>

### PDF 第 45 页

MOPs than BERT-Base with a sequence length of 128. This shows how differences in FLOPs between two DNN models don’t necessarily
represent the relationship between the runtime latency of these two models.
   Additionally, we observe that, without operation fusion, nonlinear operations consume 32.4% of overall runtime latency, even though
convolutions consume 99.3% of FLOPs, as outlined in Tab. 4. We therefore assessed the latency of fusing the nonlinear operations with the
prior convolutional layers. We found that a 1.32 times speedup can be obtained by fusing BatchNorm and ReLU with the prior convolutional
layers, demonstrating how in the case of ResNet-50, the latencies from the nonlinear operations can be significantly reduced by fusing.
However, in Sec. 5.5.2, we demonstrate how operation fusion can be non-trivial for the Transformer architecture. For Transformers, fusing
LayerNorm or Softmax with the prior matmuls may require changes in tiling dimension changes which can actually increase runtime latency.

A.6  Acronyms and Abbreviations
Tab. 13 summarizes several acronyms/abbreviations used throughout this papers and their full names.

                           Table 13: Full names of the acronyms and abbreviations used in this paper.


                                                 Abbreviation               Full name

                                                          act-to-act          activation-to-activation
                                     ALU             arithmetic logic unit
                                                               attn.                   attention
                                      CDF       cumulative distribution function
                                  CNN        convolutional neural network
                                     CV              computer vision
                                 DNN           deep neural network
                                       EDP            energy-delay product
                                          FFN            feed-forward network
                                            FLOPs          floating-point operations
                                 MAC            multiply-accumulate
                                            matmul          matrix multiplication
                               MHA            multi-head attention
                                     MOPs         memory operations
                                     NAS          neural architecture search
                                        NLP         natural language processing
                                           PE              processing element
                                             RF                    register file
                                           RL            reinforcement learning
                                         RTL              register transfer logic





                                                                                          45

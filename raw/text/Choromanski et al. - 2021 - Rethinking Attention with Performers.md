# Choromanski et al. - 2021 - Rethinking Attention with Performers

- Source HTML: `raw/html/Choromanski et al. - 2021 - Rethinking Attention with Performers.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2009.14794
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Rethinking Attention with Performers

Krzysztof Choromanski∗1, Valerii Likhosherstov∗2, David Dohan∗1, Xingyou Song∗1 
Andreea Gane∗1, Tamas Sarlos∗1, Peter Hawkins∗1, Jared Davis∗3, Afroz Mohiuddin1 
Lukasz Kaiser1, David Belanger1, Lucy Colwell1,2, Adrian Weller2,4 
1Google 2University of Cambridge 3DeepMind 4Alan Turing Institute

###### Abstract

We introduce Performers, Transformer architectures which can estimate regular (softmax) full-rank-attention Transformers with provable accuracy, but using only linear (as opposed to quadratic) space and time complexity, without relying on any priors such as sparsity or low-rankness. To approximate softmax attention-kernels, Performers use a novel Fast Attention Via positive Orthogonal Random features approach (FAVOR+), which may be of independent interest for scalable kernel methods. FAVOR+ can also be used to efficiently model kernelizable attention mechanisms beyond softmax. This representational power is crucial to accurately compare softmax with other kernels for the first time on large-scale tasks, beyond the reach of regular Transformers, and investigate optimal attention-kernels. Performers are linear architectures fully compatible with regular Transformers and with strong theoretical guarantees: unbiased or nearly-unbiased estimation of the attention matrix, uniform convergence and low estimation variance. We tested Performers on a rich set of tasks stretching from pixel-prediction through text models to protein sequence modeling. We demonstrate competitive results with other examined efficient sparse and dense attention methods, showcasing effectiveness of the novel attention-learning paradigm leveraged by Performers.

## 1 Introduction and related work

Transformers (Vaswani et al., 2017; Dehghani et al., 2019) are powerful neural network architectures that have become SOTA in several areas of machine learning including natural language processing (NLP) (e.g. speech recognition (Luo et al., 2020)), neural machine translation (NMT) (Chen et al., 2018), document generation/summarization, time series prediction, generative modeling (e.g. image generation (Parmar et al., 2018)), music generation (Huang et al., 2019), and bioinformatics (Rives et al., 2019; Madani et al., 2020; Ingraham et al., 2019; Elnaggar et al., 2019; Du et al., 2020).

Transformers rely on a trainable attention mechanism that identifies complex dependencies between the elements of each input sequence. Unfortunately, the regular Transformer scales quadratically with the number of tokens L𝐿L in the input sequence, which is prohibitively expensive for large L𝐿L and precludes its usage in settings with limited computational resources even for moderate values of L𝐿L.
Several solutions have been proposed to address this issue (Beltagy et al., 2020; Gulati et al., 2020; Chan et al., 2020; Child et al., 2019; Bello et al., 2019). Most approaches restrict the attention mechanism to attend to local neighborhoods (Parmar et al., 2018) or incorporate structural priors on attention such as sparsity (Child et al., 2019), pooling-based compression (Rae et al., 2020) clustering/binning/convolution techniques (e.g. (Roy et al., 2020) which applies k𝑘k-means clustering to learn dynamic sparse attention regions, or (Kitaev et al., 2020), where locality sensitive hashing is used to group together tokens of similar embeddings), sliding windows (Beltagy et al., 2020), or truncated targeting (Chelba et al., 2020).
There is also a long line of research on using dense attention matrices, but defined by low-rank kernels
substituting softmax (Katharopoulos et al., 2020; Shen et al., 2018). Those methods critically rely on kernels admitting explicit representations as dot-products of finite positive-feature vectors.

The approaches above do not aim to approximate regular attention, but rather propose simpler and more tractable attention mechanisms, often by incorporating additional constraints (e.g. identical query and key sets as in (Kitaev et al., 2020)), or by trading regular with sparse attention using more layers (Child et al., 2019). Unfortunately, there is a lack of rigorous guarantees for the representation power produced by such methods, and sometimes the validity of sparsity patterns can only be verified empirically through trial and error by constructing special GPU operations (e.g. either writing C++ CUDA kernels (Child et al., 2019) or using TVMs (Beltagy et al., 2020)). Other techniques which aim to reduce Transformers’ space complexity include reversible residual layers allowing one-time activation storage in training (Kitaev et al., 2020) and shared attention weights (Xiao et al., 2019). These constraints may impede application to long-sequence problems, where approximations of the attention mechanism are not sufficient. Approximations based on truncated back-propagation (Dai et al., 2019) are also unable to capture long-distance correlations since the gradients are only propagated inside a localized window.
Other methods propose biased estimation of regular attention but only in the non-causal setting and with large mean squared error (Wang et al., 2020).

In response, we introduce the first Transformer architectures, Performers, capable of provably accurate and practical estimation of regular (softmax) full-rank attention, but of only linear space and time complexity and not relying on any priors such as sparsity or low-rankness. Performers use the Fast Attention Via positive Orthogonal Random features (FAVOR+) mechanism, leveraging new methods for approximating softmax and Gaussian kernels, which we propose. We believe these methods are of independent interest, contributing to the theory of scalable kernel methods.
Consequently, Performers are the first linear architectures fully compatible (via small amounts of fine-tuning) with regular Transformers, providing strong theoretical guarantees: unbiased or nearly-unbiased estimation of the attention matrix, uniform convergence and lower variance of the approximation.

FAVOR+ can be also applied to efficiently model other kernelizable attention mechanisms beyond softmax. This representational power is crucial to accurately compare softmax with other kernels for the first time on large-scale tasks, that are beyond the reach of regular Transformers, and find for them optimal attention-kernels.
FAVOR+ can also be applied beyond the Transformer scope as a more scalable replacement for regular attention, which itself has a wide variety of uses in computer vision (Fu et al., 2019), reinforcement learning (Zambaldi et al., 2019), training with softmax cross entropy loss, and even combinatorial optimization (Vinyals et al., 2015).

We test Performers on a rich set of tasks ranging from pixel-prediction through text models to protein sequence modeling. We demonstrate competitive results with other examined efficient sparse and dense attention methods, showcasing the effectiveness of the novel attention-learning paradigm leveraged by Performers. We emphasize that in principle, FAVOR+ can also be combined with other techniques, such as reversible layers (Kitaev et al., 2020) or cluster-based attention (Roy et al., 2020).

## 2 FAVOR+ Mechanism & Positive Orthogonal Random Features

Below we describe in detail the FAVOR+ mechanism - the backbone of the Performer′​ssuperscriptPerformer′s\mathrm{Performer^{\prime}s} architecture. We introduce a new method for estimating softmax (and Gaussian) kernels with positive orthogonal random features which FAVOR+ leverages for the robust and unbiased estimation of regular (softmax) attention and show how FAVOR+ can be applied for other attention-kernels.

### 2.1 Preliminaries - regular attention mechanism

Let L𝐿L be the size of an input sequence of tokens. Then regular dot-product attention (Vaswani et al., 2017) is a mapping which accepts matrices 𝐐,𝐊,𝐕∈ℝL×d𝐐𝐊𝐕superscriptℝ𝐿𝑑\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{L\times d} as input where d𝑑d is the hidden dimension (dimension of the latent representation). Matrices 𝐐,𝐊,𝐕𝐐𝐊𝐕\mathbf{Q},\mathbf{K},\mathbf{V} are intermediate representations of the input and their rows can be interpreted as queries, keys and values of the continuous dictionary data structure respectively. Bidirectional (or non-directional (Devlin et al., 2018)) dot-product attention has the following form, where 𝐀∈ℝL×L𝐀superscriptℝ𝐿𝐿\mathbf{A}\in\mathbb{R}^{L\times L} is the so-called attention matrix:

Att↔​(𝐐,𝐊,𝐕)=𝐃−1​𝐀𝐕,𝐀=exp⁡(𝐐𝐊⊤/d),𝐃=diag​(𝐀𝟏L).formulae-sequencesubscriptAtt↔𝐐𝐊𝐕superscript𝐃1𝐀𝐕formulae-sequence𝐀superscript𝐐𝐊top𝑑𝐃diagsubscript𝐀𝟏𝐿\mathrm{Att}_{\leftrightarrow}(\mathbf{Q},\mathbf{K},\mathbf{V})=\mathbf{D}^{-1}\mathbf{A}\mathbf{V},\quad\mathbf{A}=\exp(\mathbf{Q}\mathbf{K}^{\top}/\sqrt{d}),\quad\mathbf{D}=\mathrm{diag}(\mathbf{A}\mathbf{1}_{L}).

(1)

Here exp⁡(⋅)⋅\exp(\cdot) is applied elementwise, 𝟏Lsubscript1𝐿\mathbf{1}_{L} is the all-ones vector of length L𝐿L, and diag​(⋅)diag⋅\mathrm{diag}(\cdot) is a diagonal matrix with the input vector as the diagonal. Time and space complexity of computing (1) are O​(L2​d)𝑂superscript𝐿2𝑑O(L^{2}d) and O​(L2+L​d)𝑂superscript𝐿2𝐿𝑑O(L^{2}+Ld) respectively, because 𝐀𝐀\mathbf{A} has to be stored explicitly. Hence, in principle, dot-product attention of type (1) is incompatible with end-to-end processing of long sequences. Bidirectional attention is applied in encoder self-attention and encoder-decoder attention in Seq2Seq architectures.

Another important type of attention is unidirectional dot-product attention which has the form:

Att→​(𝐐,𝐊,𝐕)=𝐃~−1​𝐀~​𝐕,𝐀~=tril​(𝐀),𝐃~=diag​(𝐀~​𝟏L),formulae-sequencesubscriptAtt→𝐐𝐊𝐕superscript~𝐃1~𝐀𝐕formulae-sequence~𝐀tril𝐀~𝐃diag~𝐀subscript1𝐿\mathrm{Att}_{\to}(\mathbf{Q},\mathbf{K},\mathbf{V})=\widetilde{\mathbf{D}}^{-1}\widetilde{\mathbf{A}}\mathbf{V},\quad\widetilde{\mathbf{A}}=\mathrm{tril}(\mathbf{A}),\quad\widetilde{\mathbf{D}}=\mathrm{diag}(\widetilde{\mathbf{A}}\mathbf{1}_{L}),

(2)

where tril​(⋅)tril⋅\mathrm{tril}(\cdot) returns the lower-triangular part of the argument matrix including the diagonal. As discussed in (Vaswani et al., 2017), unidirectional attention is used for autoregressive generative modelling, e.g. as self-attention in generative Transformers as well as the decoder part of Seq2Seq Transformers.

We will show that attention matrix 𝐀𝐀\mathbf{A} can be approximated up to any precision in time O​(L​d2​log⁡(d))𝑂𝐿superscript𝑑2𝑑O(Ld^{2}\log(d)). For comparison, popular methods leveraging sparsity via Locality-Sensitive Hashing (LSH) techniques (Kitaev et al., 2020) have O​(L​d2​log⁡L)𝑂𝐿superscript𝑑2𝐿O(Ld^{2}\log L) time complexity.
In the main body of the paper we will describe FAVOR+ for bidirectional attention. Completely analogous results can be obtained for the unidirectional variant via the mechanism of prefix-sums (all details in the Appendix B.1).

### 2.2 Generalized Kernelizable Attention

FAVOR+ works for attention blocks using matrices 𝐀∈ℝL×L𝐀superscriptℝ𝐿𝐿\mathbf{A}\in\mathbb{R}^{L\times L} of the form 𝐀​(i,j)=K​(𝐪i⊤,𝐤j⊤)𝐀𝑖𝑗Ksuperscriptsubscript𝐪𝑖topsuperscriptsubscript𝐤𝑗top\mathbf{A}(i,j)=\mathrm{K}(\mathbf{q}_{i}^{\top},\mathbf{k}_{j}^{\top}), with 𝐪i/𝐤jsubscript𝐪𝑖subscript𝐤𝑗\mathbf{q}_{i}/\mathbf{k}_{j} standing for the it​h/jt​hsuperscript𝑖𝑡ℎsuperscript𝑗𝑡ℎi^{th}/j^{th} query/key row-vector in 𝐐/𝐊𝐐𝐊\mathbf{Q}/\mathbf{K} and kernel K:ℝd×ℝd→ℝ+:K→superscriptℝ𝑑superscriptℝ𝑑subscriptℝ\mathrm{K}:\mathbb{R}^{d}\times\mathbb{R}^{d}\rightarrow\mathbb{R}_{+} defined for the (usually randomized) mapping: ϕ:ℝd→ℝ+r:italic-ϕ→superscriptℝ𝑑superscriptsubscriptℝ𝑟\phi:\mathbb{R}^{d}\rightarrow\mathbb{R}_{+}^{r} (for some r>0𝑟0r>0) as:

K​(𝐱,𝐲)=𝔼​[ϕ​(𝐱)⊤​ϕ​(𝐲)].K𝐱𝐲𝔼delimited-[]italic-ϕsuperscript𝐱topitalic-ϕ𝐲\mathrm{K}(\mathbf{x},\mathbf{y})=\mathbb{E}[\phi(\mathbf{x})^{\top}\phi(\mathbf{y})].

(3)

We call ϕ​(𝐮)italic-ϕ𝐮\phi(\mathbf{u}) a random feature map for 𝐮∈ℝd𝐮superscriptℝ𝑑\mathbf{u}\in\mathbb{R}^{d}.
For 𝐐′,𝐊′∈ℝL×rsuperscript𝐐′superscript𝐊′superscriptℝ𝐿𝑟\mathbf{Q}^{\prime},\mathbf{K}^{\prime}\in\mathbb{R}^{L\times r} with rows given as ϕ​(𝐪i⊤)⊤italic-ϕsuperscriptsuperscriptsubscript𝐪𝑖toptop\phi(\mathbf{q}_{i}^{\top})^{\top} and ϕ​(𝐤i⊤)⊤italic-ϕsuperscriptsuperscriptsubscript𝐤𝑖toptop\phi(\mathbf{k}_{i}^{\top})^{\top} respectively,
Equation 3 leads directly to the efficient attention mechanism of the form:

Att↔^​(𝐐,𝐊,𝐕)=𝐃^−1​(𝐐′​((𝐊′)⊤​𝐕)),𝐃^=diag​(𝐐′​((𝐊′)⊤​𝟏L)).formulae-sequence^subscriptAtt↔𝐐𝐊𝐕superscript^𝐃1superscript𝐐′superscriptsuperscript𝐊′top𝐕^𝐃diagsuperscript𝐐′superscriptsuperscript𝐊′topsubscript1𝐿\widehat{\mathrm{Att}_{\leftrightarrow}}(\mathbf{Q},\mathbf{K},\mathbf{V})=\widehat{\mathbf{D}}^{-1}(\mathbf{Q}^{\prime}((\mathbf{K}^{\prime})^{\top}\mathbf{V})),\quad\quad\widehat{\mathbf{D}}=\mathrm{diag}(\mathbf{Q}^{\prime}((\mathbf{K}^{\prime})^{\top}\mathbf{1}_{L})).

(4)

Here Att↔^^subscriptAtt↔\widehat{\mathrm{Att}_{\leftrightarrow}} stands for the approximate attention and brackets indicate the order of computations. It is easy to see that such a mechanism is characterized by space complexity O​(L​r+L​d+r​d)𝑂𝐿𝑟𝐿𝑑𝑟𝑑O(Lr+Ld+rd) and time complexity O​(L​r​d)𝑂𝐿𝑟𝑑O(Lrd) as opposed to O​(L2+L​d)𝑂superscript𝐿2𝐿𝑑O(L^{2}+Ld) and O​(L2​d)𝑂superscript𝐿2𝑑O(L^{2}d) of the regular attention (see also Fig. 1).

The above scheme constitutes the FA-part of the FAVOR+ mechanism. The remaining OR+ part answers the following questions: (1) How expressive is the attention model defined in Equation 3, and in particular, can we use it in principle to approximate regular softmax attention ? (2) How do we implement it robustly in practice, and in particular, can we choose r≪Lmuch-less-than𝑟𝐿r\ll L for L≫dmuch-greater-than𝐿𝑑L\gg d to obtain desired space and time complexity gains? We answer these questions in the next sections.

### 2.3 How to and how not to approximate softmax-kernels for Attention

It turns out that by taking ϕitalic-ϕ\phi of the following form
for functions f1,…,fl:ℝ→ℝ:subscript𝑓1…subscript𝑓𝑙→ℝℝf_{1},...,f_{l}:\mathbb{R}\rightarrow\mathbb{R},
function g:ℝd→ℝ:𝑔→superscriptℝ𝑑ℝg:\mathbb{R}^{d}\rightarrow\mathbb{R} and deterministic vectors ωisubscript𝜔𝑖\omega_{i} or ω1,…,ωm​∼iid​𝒟subscript𝜔1…subscript𝜔𝑚iidsimilar-to𝒟\omega_{1},...,\omega_{m}\overset{\mathrm{iid}}{\sim}\mathcal{D} for some distribution 𝒟∈𝒫​(ℝd)𝒟𝒫superscriptℝ𝑑\mathcal{D}\in\mathcal{P}(\mathbb{R}^{d}):

ϕ​(𝐱)=h​(𝐱)m​(f1​(ω1⊤​𝐱),…,f1​(ωm⊤​𝐱),…,fl​(ω1⊤​𝐱),…,fl​(ωm⊤​𝐱)),italic-ϕ𝐱ℎ𝐱𝑚subscript𝑓1superscriptsubscript𝜔1top𝐱…subscript𝑓1superscriptsubscript𝜔𝑚top𝐱…subscript𝑓𝑙superscriptsubscript𝜔1top𝐱…subscript𝑓𝑙superscriptsubscript𝜔𝑚top𝐱\phi(\mathbf{x})=\frac{h(\mathbf{x})}{\sqrt{m}}(f_{1}(\omega_{1}^{\top}\mathbf{x}),...,f_{1}(\omega_{m}^{\top}\mathbf{x}),...,f_{l}(\omega_{1}^{\top}\mathbf{x}),...,f_{l}(\omega_{m}^{\top}\mathbf{x})),

(5)

we can model most kernels used in practice. Furthermore, in most cases 𝒟𝒟\mathcal{D} is isotropic (i.e. with pdf function constant on a sphere), usually Gaussian. For example, by taking h​(𝐱)=1ℎ𝐱1h(\mathbf{x})=1, l=1𝑙1l=1 and 𝒟=𝒩​(0,𝐈d)𝒟𝒩0subscript𝐈𝑑\mathcal{D}=\mathcal{N}(0,\mathbf{I}_{d}) we obtain estimators of the so-called PNG-kernels (Choromanski et al., 2017) (e.g. f1=sgnsubscript𝑓1sgnf_{1}=\mathrm{sgn} corresponds to the angular kernel).
Configurations: h​(𝐱)=1ℎ𝐱1h(\mathbf{x})=1, l=2𝑙2l=2, f1=sinsubscript𝑓1f_{1}=\sin, f2=cossubscript𝑓2f_{2}=\cos correspond to shift-invariant kernels, in particular 𝒟=𝒩​(0,𝐈d)𝒟𝒩0subscript𝐈𝑑\mathcal{D}=\mathcal{N}(0,\mathbf{I}_{d}) leads to the Gaussian kernel KgausssubscriptKgauss\mathrm{K}_{\mathrm{gauss}} (Rahimi & Recht, 2007). The softmax-kernel which defines regular attention matrix 𝐀𝐀\mathbf{A} is given as:

SM​(𝐱,𝐲)​=def​exp⁡(𝐱⊤​𝐲).SM𝐱𝐲defsuperscript𝐱top𝐲\mathrm{SM}(\mathbf{x},\mathbf{y})\overset{\mathrm{def}}{=}\exp(\mathbf{x}^{\top}\mathbf{y}).

(6)

In the above, without loss of generality, we omit d𝑑\sqrt{d}-renormalization since we can equivalently renormalize input keys and queries. Since: SM​(𝐱,𝐲)=exp⁡(‖𝐱‖22)​Kgauss​(𝐱,𝐲)​exp⁡(‖𝐲‖22)SM𝐱𝐲superscriptnorm𝐱22subscriptKgauss𝐱𝐲superscriptnorm𝐲22\mathrm{SM}(\mathbf{x},\mathbf{y})=\exp(\frac{\|\mathbf{x}\|^{2}}{2})\mathrm{K}_{\mathrm{gauss}}(\mathbf{x},\mathbf{y})\exp(\frac{\|\mathbf{y}\|^{2}}{2}), based on what we have said, we obtain random feature map unbiased approximation of SM​(𝐱,𝐲)SM𝐱𝐲\mathrm{SM}(\mathbf{x},\mathbf{y}) using trigonometric functions with: h​(𝐱)=exp⁡(‖𝐱‖22)ℎ𝐱superscriptnorm𝐱22h(\mathbf{x})=\exp(\frac{\|\mathbf{x}\|^{2}}{2}), l=2𝑙2l=2, f1=sinsubscript𝑓1f_{1}=\sin, f2=cossubscript𝑓2f_{2}=\cos. We call it SM^mtrig​(𝐱,𝐲)superscriptsubscript^SM𝑚trig𝐱𝐲\widehat{\mathrm{SM}}_{m}^{\mathrm{trig}}(\mathbf{x},\mathbf{y}).

There is however a caveat there. The attention module from (1) constructs for each token, a convex combination of value-vectors with coefficients given as corresponding renormalized kernel scores. That is why kernels producing non-negative scores are used. Applying random feature maps with potentially negative dimension-values (sin/cos\sin/\cos) leads to unstable behaviours, especially when kernel scores close to 00 (which is the case for many entries of 𝐀𝐀\mathbf{A} corresponding to low relevance tokens) are approximated by estimators with large variance in such regions. This results in abnormal behaviours, e.g. negative-diagonal-values renormalizers 𝐃−1superscript𝐃1\mathbf{D}^{-1}, and consequently either completely prevents training or leads to sub-optimal models.
We demonstrate empirically that this is what happens for SM^mtrigsuperscriptsubscript^SM𝑚trig\widehat{\mathrm{SM}}_{m}^{\mathrm{trig}} and provide detailed theoretical explanations showing that the variance of SM^mtrigsuperscriptsubscript^SM𝑚trig\widehat{\mathrm{SM}}_{m}^{\mathrm{trig}} is large as approximated values tend to 00 (see: Section 3). This is one of the main reasons why the robust random feature map mechanism for approximating regular softmax attention was never proposed.

We propose a robust mechanism in this paper. Furthermore, the variance of our new unbiased positive random feature map estimator tends to 00 as approximated values tend to 00 (see: Section 3).

###### Lemma 1 (Positive Random Features (PRFs) for Softmax).

For 𝐱,𝐲∈ℝd𝐱𝐲superscriptℝ𝑑\mathbf{x},\mathbf{y}\in\mathbb{R}^{d}, 𝐳=𝐱+𝐲𝐳𝐱𝐲\mathbf{z}=\mathbf{x}+\mathbf{y} we have:

SM​(𝐱,𝐲)=𝔼ω∼𝒩​(0,𝐈d)​[exp​(ω⊤​𝐱−‖𝐱‖22)​exp​(ω⊤​𝐲−‖𝐲‖22)]=Λ​𝔼ω∼𝒩​(0,𝐈d)​cosh⁡(ω⊤​𝐳),SM𝐱𝐲subscript𝔼similar-to𝜔𝒩0subscript𝐈𝑑delimited-[]expsuperscript𝜔top𝐱superscriptnorm𝐱22expsuperscript𝜔top𝐲superscriptnorm𝐲22Λsubscript𝔼similar-to𝜔𝒩0subscript𝐈𝑑superscript𝜔top𝐳\mathrm{SM}(\mathbf{x},\mathbf{y})=\mathbb{E}_{\omega\sim\mathcal{N}(0,\mathbf{I}_{d})}\!\Big{[}\mathrm{exp}\!\Big{(}\omega^{\top}\mathbf{x}-\frac{\|\mathbf{x}\|^{2}}{2}\Big{)}\mathrm{exp}\!\Big{(}\omega^{\top}\mathbf{y}-\frac{\|\mathbf{y}\|^{2}}{2}\Big{)}\Big{]}=\Lambda\mathbb{E}_{\omega\sim\mathcal{N}(0,\mathbf{I}_{d})}\cosh(\omega^{\top}\mathbf{z}),

(7)

where Λ=exp⁡(−‖𝐱‖2+‖𝐲‖22)Λsuperscriptnorm𝐱2superscriptnorm𝐲22\Lambda=\exp(-\frac{\|\mathbf{x}\|^{2}+\|\mathbf{y}\|^{2}}{2}) and cosh\cosh is hyperbolic cosine. Consequently, softmax-kernel admits a positive random feature map unbiased approximation with h​(𝐱)=exp⁡(−‖𝐱‖22)ℎ𝐱superscriptnorm𝐱22h(\mathbf{x})=\exp(-\frac{\|\mathbf{x}\|^{2}}{2}), l=1𝑙1l=1, f1=expsubscript𝑓1f_{1}=\exp and 𝒟=𝒩​(0,𝐈d)𝒟𝒩0subscript𝐈𝑑\mathcal{D}=\mathcal{N}(0,\mathbf{I}_{d}) or: h​(𝐱)=12​exp⁡(−‖𝐱‖22)ℎ𝐱12superscriptnorm𝐱22h(\mathbf{x})=\frac{1}{\sqrt{2}}\exp(-\frac{\|\mathbf{x}\|^{2}}{2}), l=2𝑙2l=2, f1​(u)=exp⁡(u)subscript𝑓1𝑢𝑢f_{1}(u)=\exp(u), f2​(u)=exp​(−u)subscript𝑓2𝑢exp𝑢f_{2}(u)=\mathrm{exp}(-u) and the same 𝒟𝒟\mathcal{D} (the latter for further variance reduction).
We call related estimators: SM^m+subscriptsuperscript^SM𝑚\widehat{\mathrm{SM}}^{+}_{m} and SM^mhyp+subscriptsuperscript^SMlimit-fromhyp𝑚\widehat{\mathrm{SM}}^{\mathrm{hyp+}}_{m}.

In Fig. 2 we visualize the advantages of positive versus standard trigonometric random features. In critical regions, where kernel values are small and need careful approximation, our method outperforms its counterpart. In Section 4 we further confirm our method’s advantages empirically, using positive features to efficiently train softmax-based linear Transformers.
If we replace in (7) ω𝜔\omega with d​ω‖ω‖𝑑𝜔norm𝜔\sqrt{d}\frac{\omega}{\|\omega\|}, we obtain the so-called regularized softmax-kernel SMREGSMREG\mathrm{SMREG} which we can approximate in a similar manner, simply changing 𝒟=𝒩​(0,𝐈d)𝒟𝒩0subscript𝐈𝑑\mathcal{D}=\mathcal{N}(0,\mathbf{I}_{d}) to 𝒟=Unif​(d​𝒮d−1)𝒟Unif𝑑superscript𝒮𝑑1\mathcal{D}=\mathrm{Unif}(\sqrt{d}\mathcal{S}^{d-1}), a distribution corresponding to Haar measure on the sphere of radius d𝑑\sqrt{d} in ℝdsuperscriptℝ𝑑\mathbb{R}^{d}, obtaining estimator SMREG^m+subscriptsuperscript^SMREG𝑚\widehat{\mathrm{SMREG}}^{+}_{m}. As we show in Section 3, such random features can also be used to accurately approximate regular softmax-kernel.

### 2.4 Orthogonal Random Features (ORFs)

The above constitutes the R+ part of the FAVOR+ method. It remains to explain the O-part. To further reduce the variance of the estimator (so that we can use an even smaller number of random features r𝑟r), we entangle different random samples ω1,…,ωmsubscript𝜔1…subscript𝜔𝑚\omega_{1},...,\omega_{m} to be exactly orthogonal. This can be done while maintaining unbiasedness whenever isotropic distributions 𝒟𝒟\mathcal{D} are used (i.e. in particular in all kernels we considered so far) by the standard Gram-Schmidt orthogonalization procedure (see (Choromanski et al., 2017) for details). ORFs is a well-known method, yet it turns out that it works particularly well with our introduced PRFs for softmax. This leads to the first theoretical results showing that ORFs can be applied to reduce the variance of softmax/Gaussian kernel estimators for any dimensionality d𝑑d rather than just asymptotically for large enough d𝑑d (as is the case for previous methods, see: next section) and leads to the first exponentially small bounds on large deviations probabilities that are strictly smaller than for non-orthogonal methods. Positivity of random features plays a key role in these bounds. The ORF mechanism requires m≤d𝑚𝑑m\leq d, but this will be the case in all our experiments. The pseudocode of the entire FAVOR+ algorithm is given in Appendix B.

Our theoretical results are tightly aligned with experiments. We show in Section 4 that PRFs+ORFs drastically improve accuracy of the approximation of the attention matrix and enable us to reduce r𝑟r which results in an accurate as well as space and time efficient mechanism which we call FAVOR+.

## 3 Theoretical results

We present here the theory of positive orthogonal random features for softmax-kernel estimation. All these results can be applied also to the Gaussian kernel, since as explained in the previous section, one can be obtained from the other by renormalization (see: Section 2.3).
All proofs and additional more general theoretical results with a discussion
are given in the Appendix.

###### Lemma 2 (positive (hyperbolic) versus trigonometric random features).

The following is true:

MSE​(SM^mtrig​(𝐱,𝐲))=12​m​exp⁡(‖𝐱+𝐲‖2)​SM−2​(𝐱,𝐲)​(1−exp​(−‖𝐱−𝐲‖2))2,MSE​(SM^m+​(𝐱,𝐲))=1m​exp⁡(‖𝐱+𝐲‖2)​SM2​(𝐱,𝐲)​(1−exp⁡(−‖𝐱+𝐲‖2)),MSE​(SM^mhyp+​(𝐱,𝐲))=12​(1−exp⁡(−‖𝐱+𝐲‖2))​MSE​(SM^m+​(𝐱,𝐲)),formulae-sequenceMSEsubscriptsuperscript^SMtrig𝑚𝐱𝐲12𝑚superscriptdelimited-∥∥𝐱𝐲2superscriptSM2𝐱𝐲superscript1expsuperscriptdelimited-∥∥𝐱𝐲22formulae-sequenceMSEsubscriptsuperscript^SM𝑚𝐱𝐲1𝑚superscriptdelimited-∥∥𝐱𝐲2superscriptSM2𝐱𝐲1superscriptdelimited-∥∥𝐱𝐲2MSEsubscriptsuperscript^SMlimit-fromhyp𝑚𝐱𝐲121superscriptdelimited-∥∥𝐱𝐲2MSEsubscriptsuperscript^SM𝑚𝐱𝐲\displaystyle\begin{split}\mathrm{MSE}(\widehat{\mathrm{SM}}^{\mathrm{trig}}_{m}(\mathbf{x},\mathbf{y}))=\frac{1}{2m}\exp(\|\mathbf{x}+\mathbf{y}\|^{2})\mathrm{SM}^{-2}(\mathbf{x},\mathbf{y})(1-\mathrm{exp}(-\|\mathbf{x}-\mathbf{y}\|^{2}))^{2},\\
\mathrm{MSE}(\widehat{\mathrm{SM}}^{\mathrm{+}}_{m}(\mathbf{x},\mathbf{y}))=\frac{1}{m}\exp(\|\mathbf{x}+\mathbf{y}\|^{2})\mathrm{SM}^{2}(\mathbf{x},\mathbf{y})(1-\exp(-\|\mathbf{x}+\mathbf{y}\|^{2})),\\
\mathrm{MSE}(\widehat{\mathrm{SM}}^{\mathrm{hyp}+}_{m}(\mathbf{x},\mathbf{y}))=\frac{1}{2}(1-\exp(-\|\mathbf{x}+\mathbf{y}\|^{2}))\mathrm{MSE}(\widehat{\mathrm{SM}}^{\mathrm{+}}_{m}(\mathbf{x},\mathbf{y})),\end{split}

(8)

for independent random samples ωisubscript𝜔𝑖\omega_{i}, and where MSEMSE\mathrm{MSE} stands for the mean squared error.

Thus, for SM​(𝐱,𝐲)→0→SM𝐱𝐲0\mathrm{SM}(\mathbf{x},\mathbf{y})\rightarrow 0 we have: MSE​(SM^mtrig​(𝐱,𝐲))→∞→MSEsubscriptsuperscript^SMtrig𝑚𝐱𝐲\mathrm{MSE}(\widehat{\mathrm{SM}}^{\mathrm{trig}}_{m}(\mathbf{x},\mathbf{y}))\rightarrow\infty and MSE​(SM^m+​(𝐱,𝐲))→0→MSEsubscriptsuperscript^SM𝑚𝐱𝐲0\mathrm{MSE}(\widehat{\mathrm{SM}}^{\mathrm{+}}_{m}(\mathbf{x},\mathbf{y}))\rightarrow 0. Furthermore, the hyperbolic estimator provides additional accuracy improvements that are strictly better than those from SM^2​m+​(𝐱,𝐲)superscriptsubscript^SM2𝑚𝐱𝐲\widehat{\mathrm{SM}}_{2m}^{\mathrm{+}}(\mathbf{x},\mathbf{y}) with twice as many random features. The next result shows that the regularized softmax-kernel is in practice an accurate proxy of the softmax-kernel in attention.

###### Theorem 1 (regularized versus softmax-kernel).

Assume that the L∞subscript𝐿L_{\infty}-norm of the attention matrix for the softmax-kernel satisfies: ‖𝐀‖∞≤Csubscriptnorm𝐀𝐶\|\mathbf{A}\|_{\infty}\leq C for some constant C≥1𝐶1C\geq 1. Denote by 𝐀regsuperscript𝐀reg\mathbf{A}^{\mathrm{reg}} the corresponding attention matrix for the regularized softmax-kernel. The following holds:

infi,j𝐀reg​(i,j)𝐀​(i,j)≥1−2d13+o​(1d13), and ​supi,j𝐀reg​(i,j)𝐀​(i,j)≤1.formulae-sequencesubscriptinfimum𝑖𝑗superscript𝐀reg𝑖𝑗𝐀𝑖𝑗12superscript𝑑13𝑜1superscript𝑑13 and subscriptsupremum𝑖𝑗superscript𝐀reg𝑖𝑗𝐀𝑖𝑗1\inf_{i,j}\frac{\mathbf{A}^{\mathrm{reg}}(i,j)}{\mathbf{A}(i,j)}\geq 1-\frac{2}{d^{\frac{1}{3}}}+o\left(\frac{1}{d^{\frac{1}{3}}}\right),\textrm{ and }\sup_{i,j}\frac{\mathbf{A}^{\mathrm{reg}}(i,j)}{\mathbf{A}(i,j)}\leq 1.

(9)

Furthermore, the latter holds for d≥2𝑑2d\geq 2 even if the L∞subscript𝐿L_{\infty}-norm condition is not satisfied, i.e. the regularized softmax-kernel is a universal lower bound for the softmax-kernel.

Consequently, positive random features for SMREGSMREG\mathrm{SMREG} can be used to approximate the softmax-kernel.
Our next result shows that orthogonality provably reduces mean squared error of the estimation with positive random features for any dimensionality d>0𝑑0d>0 and we explicitly provide the gap.

###### Theorem 2.

If SM^mort+​(𝐱,𝐲)superscriptsubscript^SM𝑚limit-fromort𝐱𝐲\widehat{\mathrm{SM}}_{m}^{\mathrm{ort+}}(\mathbf{x},\mathbf{y}) stands for the modification of
SM^m+​(𝐱,𝐲)superscriptsubscript^SM𝑚𝐱𝐲\widehat{\mathrm{SM}}_{m}^{+}(\mathbf{x},\mathbf{y}) with orthogonal random features (and thus for m≤d𝑚𝑑m\leq d), then the following holds for any d>0𝑑0d>0:

MSE​(SM^mort+​(𝐱,𝐲))≤MSE​(SM^m+​(𝐱,𝐲))−2​(m−1)m​(d+2)​(SM​(𝐱,𝐲)−exp⁡(−‖𝐱‖2+‖𝐲‖22))2.MSEsuperscriptsubscript^SM𝑚limit-fromort𝐱𝐲MSEsuperscriptsubscript^SM𝑚𝐱𝐲2𝑚1𝑚𝑑2superscriptSM𝐱𝐲superscriptnorm𝐱2superscriptnorm𝐲222\mathrm{MSE}(\widehat{\mathrm{SM}}_{m}^{\mathrm{ort+}}(\mathbf{x},\mathbf{y}))\leq\mathrm{MSE}(\widehat{\mathrm{SM}}_{m}^{+}(\mathbf{x},\mathbf{y}))-\frac{2(m-1)}{m(d+2)}\left(\mathrm{SM}(\mathbf{x},\mathbf{y})-\exp\left(-\frac{\|\mathbf{x}\|^{2}+\|\mathbf{y}\|^{2}}{2}\right)\right)^{2}.

(10)

Furthermore, completely analogous result holds for the regularized softmax-kernel SMREGSMREG\mathrm{SMREG}.

For the regularized softmax-kernel, orthogonal features provide additional concentration results - the first exponentially small bounds for probabilities of estimators’ tails that are strictly better than for non-orthogonal variants for every d>0𝑑0d>0. Our next result enables us to explicitly estimate the gap.

###### Theorem 3.

Let 𝐱,𝐲∈ℝd𝐱𝐲superscriptℝ𝑑\mathbf{x},\mathbf{y}\in\mathbb{R}^{d}. The following holds for any a>SMREG​(𝐱,𝐲)𝑎SMREG𝐱𝐲a>\mathrm{SMREG}(\mathbf{x},\mathbf{y}), θ>0𝜃0\theta>0 and m≤d𝑚𝑑m\leq d:

ℙ​[SMREG^m+​(𝐱,𝐲)>a]≤exp⁡(−θ​m​a)​MZ​(θ)m,ℙ​[SMREG^mort+​(𝐱,𝐲)>a]ℙdelimited-[]subscriptsuperscript^SMREG𝑚𝐱𝐲𝑎𝜃𝑚𝑎subscript𝑀𝑍superscript𝜃𝑚ℙdelimited-[]subscriptsuperscript^SMREGlimit-fromort𝑚𝐱𝐲𝑎\displaystyle\mathbb{P}[\widehat{\mathrm{SMREG}}^{+}_{m}(\mathbf{x},\mathbf{y})>a]\leq\exp(-\theta ma)M_{Z}(\theta)^{m},\quad\mathbb{P}[\widehat{\mathrm{SMREG}}^{\mathrm{ort+}}_{m}(\mathbf{x},\mathbf{y})>a]

≤exp⁡(−θ​m​a)​(MZ​(θ)m−exp⁡(−m2​(‖𝐱‖2+‖𝐲‖2))​θ4​m​(m−1)4​(d+2)​‖𝐱+𝐲‖4)absent𝜃𝑚𝑎subscript𝑀𝑍superscript𝜃𝑚𝑚2superscriptnorm𝐱2superscriptnorm𝐲2superscript𝜃4𝑚𝑚14𝑑2superscriptnorm𝐱𝐲4\displaystyle\leq\exp(-\theta ma)\biggl{(}M_{Z}(\theta)^{m}-\exp\left(-\frac{m}{2}(\|\mathbf{x}\|^{2}+\|\mathbf{y}\|^{2})\right)\frac{\theta^{4}m(m-1)}{4(d+2)}\|\mathbf{x}+\mathbf{y}\|^{4}\biggr{)}

where SMREG^mort+​(𝐱,𝐲)subscriptsuperscript^SMREGlimit-fromort𝑚𝐱𝐲\widehat{\mathrm{SMREG}}^{\mathrm{ort+}}_{m}(\mathbf{x},\mathbf{y}) stands for the modification of SMREG^m+​(𝐱,𝐲)subscriptsuperscript^SMREG𝑚𝐱𝐲\widehat{\mathrm{SMREG}}^{\mathrm{+}}_{m}(\mathbf{x},\mathbf{y}) with ORFs, X=Λ​exp⁡(d​ω⊤‖ω‖2​(𝐱+𝐲))𝑋Λ𝑑superscript𝜔topsubscriptnorm𝜔2𝐱𝐲X=\Lambda\exp(\sqrt{d}\frac{\omega^{\top}}{\|\omega\|_{2}}(\mathbf{x}+\mathbf{y})), ω∼𝒩​(0,𝐈d)similar-to𝜔𝒩0subscript𝐈𝑑\omega\sim\mathcal{N}(0,\mathbf{I}_{d}), ΛΛ\Lambda is as in Lemma 1 and
MZsubscript𝑀𝑍M_{Z} is the moment generating function of Z𝑍Z.

We see that ORFs provide exponentially small and sharper bounds for critical regions where the softmax-kernel is small.
Below we show that even for the SMtrigsuperscriptSMtrig\mathrm{SM}^{\mathrm{trig}} mechanism with ORFs, it suffices to take m=Θ​(d​log⁡(d))𝑚Θ𝑑𝑑m=\Theta(d\log(d)) random projections to accurately approximate the attention matrix (thus if not attention renormalization, PRFs would not be needed). In general, m𝑚m depends on the dimensionality d𝑑d of the embeddings, radius R𝑅R of the ball where all queries/keys live and precision parameter ϵitalic-ϵ\epsilon (see: Appendix F.6 for additional discussion), but does not depend on input sequence length L𝐿L.

###### Theorem 4 (uniform convergence for attention approximation).

Assume that L2subscript𝐿2L_{2}-norms of queries/keys are upper-bounded by R>0𝑅0R>0. Define l=R​d−14𝑙𝑅superscript𝑑14l=Rd^{-\frac{1}{4}} and take
h∗=exp⁡(l22)superscriptℎsuperscript𝑙22h^{*}=\exp(\frac{l^{2}}{2}).
Then for any ϵ>0italic-ϵ0\epsilon>0, δ=ϵ(h∗)2𝛿italic-ϵsuperscriptsuperscriptℎ2\delta=\frac{\epsilon}{(h^{*})^{2}}
and the number of random projections m=Θ​(dδ2​log⁡(4​d34​Rδ))𝑚Θ𝑑superscript𝛿24superscript𝑑34𝑅𝛿m=\Theta(\frac{d}{\delta^{2}}\log(\frac{4d^{\frac{3}{4}}R}{\delta})) the following holds for the attention approximation mechanism leveraging estimators SM^trigsuperscript^SMtrig\widehat{\mathrm{SM}}^{\mathrm{trig}} with ORFs:
‖𝐀^−𝐀‖∞≤ϵsubscriptnorm^𝐀𝐀italic-ϵ\|\widehat{\mathbf{A}}-\mathbf{A}\|_{\infty}\leq\epsilon with any constant probability, where 𝐀^^𝐀\widehat{\mathbf{A}} approximates the attention matrix 𝐀𝐀\mathbf{A}.

## 4 Experiments

We implemented our setup on top of pre-existing Transformer training code in Jax (Frostig et al., 2018) optimized with just-in-time (jax.jit) compilation, and complement our theory with empirical evidence to demonstrate the practicality of FAVOR+ in multiple settings. Unless explicitly stated, a Performer replaces only the attention component with our method, while all other components are exactly the same as for the regular Transformer. For shorthand notation, we denote unidirectional/causal modelling as (U) and bidirectional/masked language modelling as (B).

In terms of baselines, we use other Transformer models for comparison, although some of them are restricted to only one case - e.g. Reformer (Kitaev et al., 2020) is only (U), and Linformer (Wang et al., 2020) is only (B). Furthermore, we use PG-19 (Rae et al., 2020) as an alternative (B) pretraining benchmark, as it is made for long-length sequence training compared to the (now publicly unavailable) BookCorpus (Zhu et al., 2015) + Wikipedia dataset used in BERT (Devlin et al., 2018) and Linformer. All model and tokenization hyperparameters are shown in Appendix A.

### 4.1 Computational costs

We compared speed-wise the backward pass of the Transformer and the Performer in (B) setting, as it is one of the main computational bottlenecks during training, when using the regular default size (nh​e​a​d​s,nl​a​y​e​r​s,df​f,d)=(8,6,2048,512)subscript𝑛ℎ𝑒𝑎𝑑𝑠subscript𝑛𝑙𝑎𝑦𝑒𝑟𝑠subscript𝑑𝑓𝑓𝑑862048512(n_{heads},n_{layers},d_{ff},d)=(8,6,2048,512), where df​fsubscript𝑑𝑓𝑓d_{ff} denotes the width of the MLP layers. We observed (Fig. 3) that in terms of L𝐿L, the Performer reaches nearly linear time and sub-quadratic memory consumption (since the explicit O​(L2)𝑂superscript𝐿2O(L^{2}) attention matrix is not stored). In fact, the Performer achieves nearly optimal speedup and memory efficiency possible, depicted by the "X"-line when attention is replaced with the "identity function" simply returning the 𝐕𝐕\mathbf{V}-matrix. The combination of both memory and backward pass efficiencies for large L𝐿L allows respectively, large batch training and lower wall clock time per gradient step. Extensive additional results are demonstrated in Appendix E by varying layers, raw attention, and architecture sizes.

### 4.2 Softmax attention approximation error

We further examined the approximation error via FAVOR+ in Fig. 4.
We demonstrate that 1. Orthogonal features produce lower error than unstructured (IID) features, 2. Positive features produce lower error than trigonometric sin\sin/cos\cos features. These two empirically validate the PORF mechanism.

To further improve overall approximation of attention blocks across multiple iterations which further improves training, random samples should be periodically redrawn (Fig. 5, right). This is a cheap procedure, but can be further optimized (Appendix B.2).

### 4.3 Softmax approximation on Transformers

Even if the approximation of the attention mechanism is tight, small errors can easily propagate throughout multiple Transformer layers (e.g. MLPs, multiple heads), as we show in Fig. 14 (Appendix). In other words, the model’s Lipschitz constant can easily scale up small attention approximation error, which means that very tight approximations may sometimes be needed. Thus, when applying FAVOR(+)’s softmax approximations on a Transformer model (i.e. "Performer-X-SOFTMAX"), we demonstrate that:

1. Backwards compatibility with pretrained models is available as a benefit from softmax approximation, via small finetuning (required due to error propagation) even for trigonometric features (Fig. 5, left) on the LM1B dataset (Chelba et al., 2014).
However, when on larger dataset PG-19, 2. Positive (POS) softmax features (with redrawing) become crucial for achieving performance matching regular Transformers (Fig. 5, right).

### 4.4 Multiple layer training for proteins

We further benchmark the Performer on both (U) and (B) cases by training a 36-layer model using protein sequences from the Jan. 2019 release of TrEMBL (Consortium, 2019), similar to (Madani et al., 2020). In Fig. 6, the Reformer and Linformer significantly drop in accuracy on the protein dataset. Furthermore, the usefulness of generalized attention is evidenced by Performer-RELU (taking f=ReLU𝑓ReLUf=\mathrm{ReLU} in Equation 5) achieving the highest accuracy in both (U) and (B) cases. Our proposed softmax approximation is also shown to be tight, achieving the same accuracy as the exact-softmax Transformer and confirming our theoretical claims from Section 3.

### 4.5 Large length training - Common datasets

On the standard (U) ImageNet64 benchmark from (Parmar et al., 2018) with L=12288𝐿12288L=12288 which is unfeasible for regular Transformers, we set all models to use the same (nh​e​a​d​s,df​f,d)subscript𝑛ℎ𝑒𝑎𝑑𝑠subscript𝑑𝑓𝑓𝑑(n_{heads},d_{ff},d) but varying nl​a​y​e​r​ssubscript𝑛𝑙𝑎𝑦𝑒𝑟𝑠n_{layers}. Performer/6-layers matches the Reformer/12-layers, while the Performer/12-layers matches the Reformer/24-layers (Fig. 7: left). Depending on hardware (TPU or GPU), we also found that the Performer can be 2x faster than the Reformer via Jax optimizations for the (U) setting.

For a proof of principle study, we also create an initial protein benchmark for predicting interactions among groups of proteins by concatenating protein sequences to length L=8192𝐿8192L=8192 from TrEMBL, long enough to model protein interaction networks without the large sequence alignments required by existing methods (Cong et al., 2019). In this setting, a regular Transformer overloads memory even at a batch size of 111 per chip, by a wide margin. Thus as a baseline, we were forced to use a significantly smaller variant, reducing to (nh​e​a​d​s,nl​a​y​e​r​s,df​f,d)=(8,{1,2,3},256,256)subscript𝑛ℎ𝑒𝑎𝑑𝑠subscript𝑛𝑙𝑎𝑦𝑒𝑟𝑠subscript𝑑𝑓𝑓𝑑8123256256(n_{heads},n_{layers},d_{ff},d)=(8,\{1,2,3\},256,256). Meanwhile, the Performer trains efficiently at a batch size of 8 per chip using the standard (8,6,2048,512)862048512(8,6,2048,512) architecture. We see in Fig. 7 (right subfigure) that the smaller Transformer (nl​a​y​e​r=3subscript𝑛𝑙𝑎𝑦𝑒𝑟3n_{layer}=3) is quickly bounded at ≈19%absentpercent19\approx 19\%, while the Performer is able to train continuously to ≈24%absentpercent24\approx 24\%.

## 5 Conclusion

We presented PerformerPerformer\mathrm{Performer}, a new type of Transformer, relying on our Fast Attention Via positive Orthogonal Random features (FAVOR+) mechanism to significantly improve space and time complexity of regular Transformers. Our mechanism provides to our knowledge the first effective unbiased estimation of the original softmax-based Transformer with linear space and time complexity and opens new avenues in the research on Transformers and the role of non-sparsifying attention mechanisms.

## 6 Broader impact

We believe that the presented algorithm can be impactful in various ways:

Biology and Medicine: Our method has the potential to directly impact research on biological sequence analysis by enabling the Transformer to be applied to much longer sequences without constraints on the structure of the attention matrix. The initial application that we consider is the prediction of interactions between proteins on the proteome scale. Recently published approaches require large evolutionary sequence alignments, a bottleneck for applications to mammalian genomes (Cong et al., 2019). The potentially broad translational impact of applying these approaches to biological sequences was one of the main motivations of this work. We believe that modern bioinformatics can immensely benefit from new machine learning techniques with Transformers being among the most promising. Scaling up these methods to train faster more accurate language models opens the door to the ability to design sets of molecules with pre-specified interaction properties. These approaches could be used to augment existing physics-based design strategies that are of critical importance for example in the development of new nanoparticle vaccines (Marcandalli et al., 2019).

Environment: As we have shown, Performers with FAVOR+ are characterized by much lower compute costs and substantially lower space complexity which can be directly translated to CO2subscriptCO2\mathrm{CO}_{2} emission reduction (Strubell et al., 2019) and lower energy consumption (You et al., 2020), as regular Transformers require very large computational resources.

Research on Transformers: We believe that our results can shape research on efficient Transformers architectures, guiding the field towards methods with strong mathematical foundations. Our research may also hopefully extend Transformers also beyond their standard scope (e.g. by considering the Generalized Attention mechanism and connections with kernels). Exploring scalable Transformer architectures that can handle L𝐿L of the order of magnitude few thousands and more, preserving accuracy of the baseline at the same time, is a gateway to new breakthroughs in bio-informatics, e.g. language modeling for proteins, as we explained in the paper. Our presented method can be potentially a first step.

Backward Compatibility: Our Performer can be used on the top of a regular pre-trained Transformer as opposed to other Transformer variants. Even if up-training is not required, FAVOR+ can still be used for fast inference with no loss of accuracy. We think about this backward compatibility as a very important additional feature of the presented techniques that might be particularly attractive for practitioners.

Attention Beyond Transformers: Finally, FAVOR+ can be applied to approximate exact attention also outside the scope of Transformers. This opens a large volume of new potential applications including: hierarchical attention networks (HANS) (Yang et al., 2016), graph attention networks (Velickovic et al., 2018), image processing (Fu et al., 2019), and reinforcement learning/robotics (Tang et al., 2020).

## 7 Acknowledgements

We thank Nikita Kitaev and Wojciech Gajewski for multiple discussions on the Reformer, and also thank Aurko Roy and Ashish Vaswani for multiple discussions on the Routing Transformer. We further thank Joshua Meier, John Platt, and Tom Weingarten for many fruitful discussions on biological data and useful comments on this draft. We lastly thank Yi Tay and Mostafa Dehghani for discussions on comparing baselines.

Valerii Likhosherstov acknowledges support from the Cambridge Trust and DeepMind. Lucy Colwell acknowledges support from the Simons Foundation. Adrian Weller acknowledges support from a Turing AI Fellowship under grant EP/V025379/1, The Alan Turing Institute under EPSRC grant EP/N510129/1 and U/B/000074, and the Leverhulme Trust via CFI.

## References

- Bello et al. (2019)

Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens, and Quoc V. Le.

Attention augmented convolutional networks.

CoRR, abs/1904.09925, 2019.

URL http://arxiv.org/abs/1904.09925.

- Beltagy et al. (2020)

Iz Beltagy, Matthew E. Peters, and Arman Cohan.

Longformer: The long-document transformer.

CoRR, abs/2004.05150, 2020.

URL https://arxiv.org/abs/2004.05150.

- Chan et al. (2020)

William Chan, Chitwan Saharia, Geoffrey E. Hinton, Mohammad Norouzi, and
Navdeep Jaitly.

Imputer: Sequence modelling via imputation and dynamic programming.

CoRR, abs/2002.08926, 2020.

URL https://arxiv.org/abs/2002.08926.

- Chelba et al. (2014)

Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp
Koehn, and Tony Robinson.

One billion word benchmark for measuring progress in statistical
language modeling.

In INTERSPEECH 2014, 15th Annual Conference of the
International Speech Communication Association, Singapore, September 14-18,
2014, pp. 2635–2639, 2014.

- Chelba et al. (2020)

Ciprian Chelba, Mia Xu Chen, Ankur Bapna, and Noam Shazeer.

Faster transformer decoding: N-gram masked self-attention.

CoRR, abs/2001.04589, 2020.

URL https://arxiv.org/abs/2001.04589.

- Chen et al. (2018)

Mia Xu Chen, Orhan Firat, Ankur Bapna, Melvin Johnson, Wolfgang Macherey,
George F. Foster, Llion Jones, Mike Schuster, Noam Shazeer, Niki Parmar,
Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Zhifeng Chen, Yonghui Wu, and
Macduff Hughes.

The best of both worlds: Combining recent advances in neural machine
translation.

In Proceedings of the 56th Annual Meeting of the Association
for Computational Linguistics, ACL 2018, Melbourne, Australia, July 15-20,
2018, Volume 1: Long Papers, pp. 76–86. Association for Computational
Linguistics, 2018.

doi: 10.18653/v1/P18-1008.

URL https://www.aclweb.org/anthology/P18-1008/.

- Child et al. (2019)

Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever.

Generating long sequences with sparse transformers.

CoRR, abs/1904.10509, 2019.

URL http://arxiv.org/abs/1904.10509.

- Choromanski et al. (2018a)

Krzysztof Choromanski, Carlton Downey, and Byron Boots.

Initialization matters: Orthogonal predictive state recurrent neural
networks.

In 6th International Conference on Learning Representations,
ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track
Proceedings. OpenReview.net, 2018a.

URL https://openreview.net/forum?id=HJJ23bW0b.

- Choromanski et al. (2018b)

Krzysztof Choromanski, Mark Rowland, Tamás Sarlós, Vikas Sindhwani,
Richard E. Turner, and Adrian Weller.

The geometry of random features.

In International Conference on Artificial Intelligence and
Statistics, AISTATS 2018, 9-11 April 2018, Playa Blanca, Lanzarote, Canary
Islands, Spain, volume 84 of Proceedings of Machine Learning
Research, pp. 1–9. PMLR, 2018b.

URL http://proceedings.mlr.press/v84/choromanski18a.html.

- Choromanski et al. (2019a)

Krzysztof Choromanski, Aldo Pacchiano, Jeffrey Pennington, and Yunhao Tang.

KAMA-NNs: Low-dimensional rotation based neural networks.

In The 22nd International Conference on Artificial Intelligence
and Statistics, AISTATS 2019, 16-18 April 2019, Naha, Okinawa, Japan,
volume 89 of Proceedings of Machine Learning Research, pp. 236–245.
PMLR, 2019a.

URL http://proceedings.mlr.press/v89/choromanski19a.html.

- Choromanski et al. (2019b)

Krzysztof Choromanski, Mark Rowland, Wenyu Chen, and Adrian Weller.

Unifying orthogonal Monte Carlo methods.

In Proceedings of the 36th International Conference on Machine
Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA,
volume 97 of Proceedings of Machine Learning Research, pp. 1203–1212. PMLR, 2019b.

URL http://proceedings.mlr.press/v97/choromanski19a.html.

- Choromanski et al. (2017)

Krzysztof Marcin Choromanski, Mark Rowland, and Adrian Weller.

The unreasonable effectiveness of structured random orthogonal
embeddings.

In Advances in Neural Information Processing Systems 30: Annual
Conference on Neural Information Processing Systems 2017, 4-9 December 2017,
Long Beach, CA, USA, pp. 219–228, 2017.

- Clevert et al. (2016)

Djork-Arné Clevert, Thomas Unterthiner, and Sepp Hochreiter.

Fast and accurate deep network learning by exponential linear units
(elus).

In 4th International Conference on Learning Representations,
ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track
Proceedings, 2016.

URL http://arxiv.org/abs/1511.07289.

- Cong et al. (2019)

Qian Cong, Ivan Anishchenko, Sergey Ovchinnikov, and David Baker.

Protein interaction networks revealed by proteome coevolution.

Science, 365(6449):185–189, 2019.

- Consortium (2019)

UniProt Consortium.

Uniprot: a worldwide hub of protein knowledge.

Nucleic acids research, 47(D1):D506–D515,
2019.

- Cormen et al. (2009)

Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.

Introduction to Algorithms, 3rd Edition.

MIT Press, 2009.

ISBN 978-0-262-03384-8.

URL http://mitpress.mit.edu/books/introduction-algorithms.

- Dai et al. (2019)

Zihang Dai, Zhilin Yang, Yiming Yang, William W. Cohen, Jaime Carbonell,
Quoc V. Le, and Ruslan Salakhutdinov.

Transformer-XL: Language modeling with longer-term dependency,
2019.

URL https://openreview.net/forum?id=HJePno0cYm.

- Dehghani et al. (2019)

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz
Kaiser.

Universal transformers.

In 7th International Conference on Learning Representations,
ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019.

URL https://openreview.net/forum?id=HyzdRiR9Y7.

- Devlin et al. (2018)

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.

BERT: pre-training of deep bidirectional transformers for language
understanding.

CoRR, abs/1810.04805, 2018.

URL http://arxiv.org/abs/1810.04805.

- Du et al. (2020)

Yilun Du, Joshua Meier, Jerry Ma, Rob Fergus, and Alexander Rives.

Energy-based models for atomic-resolution protein conformations.

arXiv preprint arXiv:2004.13167, 2020.

- Elnaggar et al. (2019)

Ahmed Elnaggar, Michael Heinzinger, Christian Dallago, and Burkhard Rost.

End-to-end multitask learning, from protein language to protein
features without alignments.

bioRxiv, pp. 864405, 2019.

- Frostig et al. (2018)

Roy Frostig, Matthew Johnson, and Chris Leary.

Compiling machine learning programs via high-level tracing.

In Conference on Machine Learning and Systems 2018, 2018.

URL http://www.sysml.cc/doc/2018/146.pdf.

- Fu et al. (2019)

Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang, and Hanqing
Lu.

Dual attention network for scene segmentation.

In IEEE Conference on Computer Vision and Pattern
Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pp. 3146–3154, 2019.

- Gulati et al. (2020)

Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu,
Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, and Ruoming Pang.

Conformer: Convolution-augmented transformer for speech recognition,
2020.

- Huang et al. (2019)

Cheng-Zhi Anna Huang, Ashish Vaswani, Jakob Uszkoreit, Ian Simon, Curtis
Hawthorne, Noam Shazeer, Andrew M. Dai, Matthew D. Hoffman, Monica
Dinculescu, and Douglas Eck.

Music transformer: Generating music with long-term structure.

In 7th International Conference on Learning Representations,
ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019.

URL https://openreview.net/forum?id=rJe4ShAcF7.

- Ingraham et al. (2019)

John Ingraham, Vikas Garg, Regina Barzilay, and Tommi Jaakkola.

Generative models for graph-based protein design.

In Advances in Neural Information Processing Systems, pp. 15794–15805, 2019.

- Katharopoulos et al. (2020)

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François
Fleuret.

Transformers are rnns: Fast autoregressive transformers with linear
attention.

CoRR, abs/2006.16236, 2020.

URL https://arxiv.org/abs/2006.16236.

- Kitaev et al. (2020)

Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya.

Reformer: The efficient transformer.

In 8th International Conference on Learning Representations,
ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020.

URL https://openreview.net/forum?id=rkgNKkHtvB.

- Kovaleva et al. (2019)

Olga Kovaleva, Alexey Romanov, Anna Rogers, and Anna Rumshisky.

Revealing the dark secrets of bert.

arXiv preprint arXiv:1908.08593, 2019.

- Kudo & Richardson (2018)

Taku Kudo and John Richardson.

Sentencepiece: A simple and language independent subword tokenizer
and detokenizer for neural text processing.

CoRR, abs/1808.06226, 2018.

URL http://arxiv.org/abs/1808.06226.

- Ladner & Fischer (1980)

Richard E. Ladner and Michael J. Fischer.

Parallel prefix computation.

J. ACM, 27(4):831–838, October 1980.

ISSN 0004-5411.

doi: 10.1145/322217.322232.

URL https://doi.org/10.1145/322217.322232.

- Lin et al. (2020)

Han Lin, Haoxian Chen, Tianyi Zhang, Clément Laroche, and Krzysztof
Choromanski.

Demystifying orthogonal Monte Carlo and beyond.

CoRR, abs/2005.13590, 2020.

- Luo et al. (2020)

Haoneng Luo, Shiliang Zhang, Ming Lei, and Lei Xie.

Simplified self-attention for transformer-based end-to-end speech
recognition.

CoRR, abs/2005.10463, 2020.

URL https://arxiv.org/abs/2005.10463.

- Madani et al. (2020)

Ali Madani, Bryan McCann, Nikhil Naik, Nitish Shirish Keskar, Namrata Anand,
Raphael R. Eguchi, Po-Ssu Huang, and Richard Socher.

Progen: Language modeling for protein generation.

CoRR, abs/2004.03497, 2020.

URL https://arxiv.org/abs/2004.03497.

- Marcandalli et al. (2019)

Jessica Marcandalli, Brooke Fiala, Sebastian Ols, Michela Perotti, Willem
de van der Schueren, Joost Snijder, Edgar Hodge, Mark Benhaim, Rashmi
Ravichandran, Lauren Carter, et al.

Induction of potent neutralizing antibody responses by a designed
protein nanoparticle vaccine for respiratory syncytial virus.

Cell, 176(6):1420–1431, 2019.

- Nangia & Bowman (2018)

Nikita Nangia and Samuel R. Bowman.

Listops: A diagnostic dataset for latent tree learning.

In Proceedings of the 2018 Conference of the North American
Chapter of the Association for Computational Linguistics, NAACL-HLT 2018,
New Orleans, Louisiana, USA, June 2-4, 2018, Student Research Workshop, pp. 92–99, 2018.

doi: 10.18653/v1/n18-4013.

URL https://doi.org/10.18653/v1/n18-4013.

- Parmar et al. (2018)

Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer,
Alexander Ku, and Dustin Tran.

Image transformer.

In Proceedings of the 35th International Conference on Machine
Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15,
2018, volume 80 of Proceedings of Machine Learning Research, pp. 4052–4061. PMLR, 2018.

URL http://proceedings.mlr.press/v80/parmar18a.html.

- Rae et al. (2020)

Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, and
Timothy P. Lillicrap.

Compressive transformers for long-range sequence modelling.

In International Conference on Learning Representations, 2020.

URL https://openreview.net/forum?id=SylKikSYDH.

- Rahimi & Recht (2007)

Ali Rahimi and Benjamin Recht.

Random features for large-scale kernel machines.

In Advances in Neural Information Processing Systems 20,
Proceedings of the Twenty-First Annual Conference on Neural Information
Processing Systems, Vancouver, British Columbia, Canada, December 3-6, 2007,
pp. 1177–1184. Curran Associates, Inc., 2007.

URL
http://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines.

- Rives et al. (2019)

Alexander Rives, Siddharth Goyal, Joshua Meier, Demi Guo, Myle Ott, C. Zitnick,
Jerry Ma, and Rob Fergus.

Biological structure and function emerge from scaling unsupervised
learning to 250 million protein sequences.

bioArxiv, 04 2019.

doi: 10.1101/622803.

- Rowland et al. (2019)

Mark Rowland, Jiri Hron, Yunhao Tang, Krzysztof Choromanski, Tamás
Sarlós, and Adrian Weller.

Orthogonal estimation of Wasserstein distances.

In The 22nd International Conference on Artificial Intelligence
and Statistics, AISTATS 2019, 16-18 April 2019, Naha, Okinawa, Japan,
volume 89 of Proceedings of Machine Learning Research, pp. 186–195.
PMLR, 2019.

URL http://proceedings.mlr.press/v89/rowland19a.html.

- Roy et al. (2020)

Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier.

Efficient content-based sparse attention with routing transformers.

CoRR, abs/2003.05997, 2020.

URL https://arxiv.org/abs/2003.05997.

- Shen et al. (2018)

Zhuoran Shen, Mingyuan Zhang, Shuai Yi, Junjie Yan, and Haiyu Zhao.

Factorized attention: Self-attention with linear complexities.

CoRR, abs/1812.01243, 2018.

URL http://arxiv.org/abs/1812.01243.

- Strubell et al. (2019)

Emma Strubell, Ananya Ganesh, and Andrew McCallum.

Energy and policy considerations for deep learning in NLP.

CoRR, abs/1906.02243, 2019.

URL http://arxiv.org/abs/1906.02243.

- Tang et al. (2020)

Yujin Tang, Duong Nguyen, and David Ha.

Neuroevolution of self-interpretable agents.

CoRR, abs/2003.08165, 2020.

URL https://arxiv.org/abs/2003.08165.

- Tay et al. (2021)

Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham,
Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler.

Long range arena: A benchmark for efficient transformers.

2021.

- Tsai et al. (2019)

Yao-Hung Hubert Tsai, Shaojie Bai, Makoto Yamada, Louis-Philippe Morency, and
Ruslan Salakhutdinov.

Transformer dissection: An unified understanding for transformer’s
attention via the lens of kernel.

In Proceedings of the 2019 Conference on Empirical Methods in
Natural Language Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP), pp. 4335–4344, 2019.

- Vaswani et al. (2017)

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin.

Attention is all you need.

In Advances in Neural Information Processing Systems 30, pp. 5998–6008. Curran Associates, Inc., 2017.

URL
http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf.

- Velickovic et al. (2018)

Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro
Liò, and Yoshua Bengio.

Graph attention networks.

In 6th International Conference on Learning Representations,
ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track
Proceedings. OpenReview.net, 2018.

URL https://openreview.net/forum?id=rJXMpikCZ.

- Vig (2019)

Jesse Vig.

A multiscale visualization of attention in the transformer model.

arXiv preprint arXiv:1906.05714, 2019.

- Vig & Belinkov (2019)

Jesse Vig and Yonatan Belinkov.

Analyzing the structure of attention in a transformer language model.

CoRR, abs/1906.04284, 2019.

URL http://arxiv.org/abs/1906.04284.

- Vig et al. (2020)

Jesse Vig, Ali Madani, Lav R. Varshney, Caiming Xiong, Richard Socher, and
Nazneen Fatema Rajani.

Bertology meets biology: Interpreting attention in protein language
models.

CoRR, abs/2006.15222, 2020.

URL https://arxiv.org/abs/2006.15222.

- Vinyals et al. (2015)

Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly.

Pointer networks.

In Advances in Neural Information Processing Systems 28: Annual
Conference on Neural Information Processing Systems 2015, December 7-12,
2015, Montreal, Quebec, Canada, pp. 2692–2700, 2015.

- Wang et al. (2020)

Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma.

Linformer: Self-attention with linear complexity.

CoRR, abs/2006.04768, 2020.

URL https://arxiv.org/abs/2006.04768.

- Xiao et al. (2019)

Tong Xiao, Yinqiao Li, Jingbo Zhu, Zhengtao Yu, and Tongran Liu.

Sharing attention weights for fast transformer.

In Proceedings of the Twenty-Eighth International Joint
Conference on Artificial Intelligence, IJCAI 2019, Macao, China, August
10-16, 2019, pp. 5292–5298. ijcai.org, 2019.

doi: 10.24963/ijcai.2019/735.

URL https://doi.org/10.24963/ijcai.2019/735.

- Yang et al. (2016)

Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alexander J. Smola, and
Eduard H. Hovy.

Hierarchical attention networks for document classification.

In NAACL HLT 2016, The 2016 Conference of the North
American Chapter of the Association for Computational Linguistics: Human
Language Technologies, San Diego California, USA, June 12-17, 2016, pp. 1480–1489. The Association for Computational Linguistics, 2016.

doi: 10.18653/v1/n16-1174.

URL https://doi.org/10.18653/v1/n16-1174.

- You et al. (2020)

Haoran You, Chaojian Li, Pengfei Xu, Yonggan Fu, Yue Wang, Xiaohan Chen,
Richard G. Baraniuk, Zhangyang Wang, and Yingyan Lin.

Drawing early-bird tickets: Toward more efficient training of deep
networks.

In International Conference on Learning Representations, 2020.

URL https://openreview.net/forum?id=BJxsrgStvr.

- Yu et al. (2016)

Felix X. Yu, Ananda Theertha Suresh, Krzysztof Marcin Choromanski, Daniel N.
Holtmann-Rice, and Sanjiv Kumar.

Orthogonal random features.

In Advances in Neural Information Processing Systems 29: Annual
Conference on Neural Information Processing Systems 2016, December 5-10,
2016, Barcelona, Spain, pp. 1975–1983, 2016.

- Zambaldi et al. (2019)

Vinícius Flores Zambaldi, David Raposo, Adam Santoro, Victor Bapst,
Yujia Li, Igor Babuschkin, Karl Tuyls, David P. Reichert, Timothy P.
Lillicrap, Edward Lockhart, Murray Shanahan, Victoria Langston, Razvan
Pascanu, Matthew Botvinick, Oriol Vinyals, and Peter W. Battaglia.

Deep reinforcement learning with relational inductive biases.

In 7th International Conference on Learning Representations,
ICLR 2019, New Orleans, LA, USA, May 6-9, 2019, 2019.

- Zhu et al. (2015)

Yukun Zhu, Ryan Kiros, Richard S. Zemel, Ruslan Salakhutdinov, Raquel Urtasun,
Antonio Torralba, and Sanja Fidler.

Aligning books and movies: Towards story-like visual explanations by
watching movies and reading books.

In 2015 IEEE International Conference on Computer Vision,
ICCV 2015, Santiago, Chile, December 7-13, 2015, pp. 19–27, 2015.

doi: 10.1109/ICCV.2015.11.

URL https://doi.org/10.1109/ICCV.2015.11.

## APPENDIX: Rethinking Attention with Performers

## A Hyperparameters for experiments

This optimal setting (including comparisons to approximate softmax) we use for the Performer is specified in the Generalized Attention (Subsec. A.4), and unless specifically mentioned (e.g. using name "Performer-SOFTMAX"), "Performer" refers to using this generalized attention setting.

### A.1 Metrics

We report the following evaluation metrics:

- 1.

Accuracy: For unidirectional models, we measure the accuracy on next-token prediction, averaged across all sequence positions in the dataset. For bidirectional models, we mask each token with 15%percent1515\% probability (same as (Devlin et al., 2018)) and measure accuracy across the masked positions.

- 2.

Perplexity: For unidirectional models, we measure perplexity across all sequence positions in the dataset. For bidirectional models, similar to the accuracy case, we measure perplexity across the masked positions.

- 3.

Bits Per Dimension/Character (BPD/BPC): This calculated by loss divided by ln⁡(2)2\ln(2).

We used the full evaluation dataset for TrEMBL in the plots in the main section, while for other datasets such as ImageNet64 and PG-19 which have very large evaluation dataset sizes, we used random batches (>2048 samples) for plotting curves.

#### A.1.1 PG-19 Preprocessing

The PG-19 dataset (Rae et al., 2020) is presented as a challenging long range text modeling task. It consists of out-of-copyright Project Gutenberg books published before 1919. It does not have a fixed vocabulary size, instead opting for any tokenization which can model an arbitrary string of text. We use a unigram SentencePiece vocabulary (Kudo & Richardson, 2018) with 32768 tokens, which maintains whitespace and is completely invertible to the original book text. Perplexities are calculated as the average log-likelihood per token, multiplied by the ratio of the sentencepiece tokenization to number of tokens in the original dataset. The original dataset token count per split is: train=1973136207, validation=3007061, test=6966499. Our sentencepiece tokenization yields the following token counts per split: train=3084760726, valid=4656945, and test=10699704. This gives log likelihood multipliers of train=1.5634, valid=1.5487, test=1.5359 per split before computing perplexity, which is equal to exp⁡(log likelihood multiplier∗loss)log likelihood multiplierloss\exp(\text{log likelihood multiplier}*\text{loss}).

Preprocessing for TrEMBL is extensively explained in Appendix C.

### A.2 Training Hyperparameters

Unless specifically stated, all Performer + Transformer runs by default used 0.50.50.5 grad clip, 0.10.10.1 weight decay, 0.10.10.1 dropout, 10−3superscript10310^{-3} fixed learning rate with Adam hyperparameters (β1=0.9,β2=0.98,ϵ=10−9)formulae-sequencesubscript𝛽10.9formulae-sequencesubscript𝛽20.98italic-ϵsuperscript109(\beta_{1}=0.9,\beta_{2}=0.98,\epsilon=10^{-9}), with batch size maximized (until TPU memory overload) for a specific model.

All 36-layer protein experiments used the same amount of compute (i.e. 16x16 TPU-v2, 8GB per chip). For concatenated experiments, 16x16 TPU-v2’s were also used for the Performer, while 8x8’s were used for the 1-3 layer (d=256)𝑑256(d=256) Transformer models (using 16x16 did not make a difference in accuracy).

Note that Performers are using the same training hyperparameters as Transformers, yet achieving competitive results - this shows that FAVOR can act as a simple drop-in without needing much tuning.

### A.3 Approximate Softmax Attention Default Values

The optimal values, set to default parameters111https://github.com/google-research/google-research/blob/master/performer/fast_attention
, are: renormalize_attention = True, numerical stabilizer = 10−6superscript10610^{-6}, number of features = 256, ortho_features = True, ortho_scaling = 0.0.

### A.4 Generalized Attention Default Values

The optimal values, set to default parameters222https://github.com/google-research/google-research/blob/master/performer/fast_attention
, are: renormalize_attention = True, numerical stabilizer = 0.0, number of features = 256, kernel = ReLU, kernel_epsilon = 10−3superscript10310^{-3}.

### A.5 Reformer Default Values

For the Reformer, we used the same hyperparameters as mentioned for protein experiments, without gradient clipping, while using the defaults333https://github.com/google/trax/blob/master/trax/supervised/configs/reformer_imagenet64.gin (which instead use learning rate decay) for ImageNet-64. In both cases, the Reformer used the same default LSH attention parameters.

### A.6 Linformer Default Values

Using our standard pipeline as mentioned above, we replaced the attention function with the Linformer variant via Jax, with δ=10−6,k=600formulae-sequence𝛿superscript106𝑘600\delta=10^{-6},k=600 (same notation used in the paper (Wang et al., 2020)), where δ𝛿\delta is the exponent in a renormalization procedure using e−δsuperscript𝑒𝛿e^{-\delta} as a multiplier in order to approximate softmax, while k𝑘k is the dimension of the projections of the 𝐐𝐐\mathbf{Q} and 𝐊𝐊\mathbf{K} matrices. As a sanity check, we found that our Linformer implementation in Jax correctly approximated exact softmax’s output within 0.020.020.02 error for all entries.

Note that for rigorous comparisons, our Linformer hyperparameters are even stronger than the defaults found in (Wang et al., 2020), as:

- •

We use k=600𝑘600k=600, which is more than twice than the default k=256𝑘256k=256 from the paper, and also twice than our default m=256𝑚256m=256 number of features.

- •

We also use redrawing, which avoids "unlucky" projections on 𝐐𝐐\mathbf{Q} and 𝐊𝐊\mathbf{K}.

## B Main Algorithm: FAVOR+

We outline the main algorithm for FAVOR+ formally:

Input : 𝐐,𝐊,𝐕∈ℝL×d𝐐𝐊𝐕superscriptℝ𝐿𝑑\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{L\times d}, isBidirectionalisBidirectional\mathrm{isBidirectional} - binary flag.

Result: Att^↔​(𝐐,𝐊,𝐕)∈ℝL×Lsubscript^Att↔𝐐𝐊𝐕superscriptℝ𝐿𝐿\widehat{\mathrm{Att}}_{\leftrightarrow}(\mathbf{Q},\mathbf{K},\mathbf{V})\in\mathbb{R}^{L\times L} if isBidirectionalisBidirectional\mathrm{isBidirectional}, Att^→​(𝐐,𝐊,𝐕)∈ℝL×Lsubscript^Att→𝐐𝐊𝐕superscriptℝ𝐿𝐿\widehat{\mathrm{Att}}_{\to}(\mathbf{Q},\mathbf{K},\mathbf{V})\in\mathbb{R}^{L\times L} otherwise.

Compute 𝐐′superscript𝐐′\mathbf{Q}^{\prime} and 𝐊′superscript𝐊′\mathbf{K}^{\prime} as described in Section 2.2 and Section 2.3 and take 𝐂:=[𝐕𝟏L]assign𝐂matrix𝐕subscript1𝐿\mathbf{C}:=\begin{bmatrix}\mathbf{V}&\mathbf{1}_{L}\end{bmatrix};

if isBidirectionalisBidirectional\mathrm{isBidirectional} then

Buf1:=(𝐊′)⊤​𝐂∈ℝM×(d+1),Buf2:=𝐐′​Buf1∈ℝL×(d+1)formulae-sequenceassignsubscriptBuf1superscriptsuperscript𝐊′top𝐂superscriptℝ𝑀𝑑1assignsubscriptBuf2superscript𝐐′subscriptBuf1superscriptℝ𝐿𝑑1\mathrm{Buf}_{1}:=(\mathbf{K}^{\prime})^{\top}\mathbf{C}\in\mathbb{R}^{M\times(d+1)},\quad\mathrm{Buf}_{2}:=\mathbf{Q}^{\prime}\mathrm{Buf}_{1}\in\mathbb{R}^{L\times(d+1)};

else

Compute 𝐆𝐆\mathbf{G} and its prefix-sum tensor 𝐆PSsuperscript𝐆PS\mathbf{G}^{\mathrm{PS}} according to (11);

Buf2:=[𝐆1,:,:PS​𝐐1′…𝐆L,:,:PS​𝐐L′]⊤∈ℝL×(d+1)assignsubscriptBuf2superscriptmatrixsubscriptsuperscript𝐆PS1::subscriptsuperscript𝐐′1…subscriptsuperscript𝐆PS𝐿::subscriptsuperscript𝐐′𝐿topsuperscriptℝ𝐿𝑑1\mathrm{Buf}_{2}:=\begin{bmatrix}\mathbf{G}^{\mathrm{PS}}_{1,:,:}\mathbf{Q}^{\prime}_{1}&\dots&\mathbf{G}^{\mathrm{PS}}_{L,:,:}\mathbf{Q}^{\prime}_{L}\end{bmatrix}^{\top}\in\mathbb{R}^{L\times(d+1)};

end if

[Buf3buf4]:=Buf2,Buf3∈ℝL×d,buf4∈ℝLformulae-sequenceassignmatrixsubscriptBuf3subscriptbuf4subscriptBuf2formulae-sequencesubscriptBuf3superscriptℝ𝐿𝑑subscriptbuf4superscriptℝ𝐿\begin{bmatrix}\mathrm{Buf}_{3}&\mathrm{buf}_{4}\end{bmatrix}:=\mathrm{Buf}_{2},\quad\mathrm{Buf}_{3}\in\mathbb{R}^{L\times d},\quad\mathrm{buf}_{4}\in\mathbb{R}^{L};

return diag​(buf4)−1​Buf3diagsuperscriptsubscriptbuf41subscriptBuf3\mathrm{diag}(\mathrm{buf}_{4})^{-1}\mathrm{Buf}_{3};

### B.1 Unidirectional Case and Prefix Sums

We explain how our analysis from Section 2.2 can be extended to the unidirectional mechanism in this section.
Notice that this time attention matrix 𝐀𝐀\mathbf{A} is masked, i.e. all its entries not in the lower-triangular part (which contains the diagonal) are zeroed (see also Fig. 8).

For the unidirectional case, our analysis is similar as for the bidirectional case, but this time our goal is to compute tril​(𝐐′​(𝐊′)⊤)​𝐂trilsuperscript𝐐′superscriptsuperscript𝐊′top𝐂\mathrm{tril}(\mathbf{Q}^{\prime}(\mathbf{K}^{\prime})^{\top})\mathbf{C} without constructing and storing the L×L𝐿𝐿L\times L-sized matrix tril​(𝐐′​(𝐊′)⊤)trilsuperscript𝐐′superscriptsuperscript𝐊′top\mathrm{tril}(\mathbf{Q}^{\prime}(\mathbf{K}^{\prime})^{\top}) explicitly, where
𝐂=[V𝟏L]∈ℝL×(d+1)𝐂matrix𝑉subscript1𝐿superscriptℝ𝐿𝑑1\mathbf{C}~{}=~{}\begin{bmatrix}V&\mathbf{1}_{L}\end{bmatrix}\in\mathbb{R}^{L\times(d+1)}. In order to do so, observe that ∀1≤i≤Lfor-all1𝑖𝐿\forall 1\leq i\leq L:

[tril​(𝐐′​(𝐊′)⊤)​𝐂]i=𝐆i,:,:PS×𝐐i′,𝐆i,:,:PS=∑j=1i𝐆j,:,:,𝐆j,:,:=𝐊j′​𝐂j⊤∈ℝM×(d+1)formulae-sequencesubscriptdelimited-[]trilsuperscript𝐐′superscriptsuperscript𝐊′top𝐂𝑖subscriptsuperscript𝐆PS𝑖::subscriptsuperscript𝐐′𝑖formulae-sequencesubscriptsuperscript𝐆PS𝑖::superscriptsubscript𝑗1𝑖subscript𝐆𝑗::subscript𝐆𝑗::subscriptsuperscript𝐊′𝑗superscriptsubscript𝐂𝑗topsuperscriptℝ𝑀𝑑1[\mathrm{tril}(\mathbf{Q}^{\prime}(\mathbf{K}^{\prime})^{\top})\mathbf{C}]_{i}=\mathbf{G}^{\mathrm{PS}}_{i,:,:}\times\mathbf{Q}^{\prime}_{i},\quad\mathbf{G}^{\mathrm{PS}}_{i,:,:}=\sum_{j=1}^{i}\mathbf{G}_{j,:,:},\quad\mathbf{G}_{j,:,:}=\mathbf{K}^{\prime}_{j}\mathbf{C}_{j}^{\top}\in\mathbb{R}^{M\times(d+1)}

(11)

where 𝐆,𝐆PS∈ℝL×M×(d+1)𝐆superscript𝐆PSsuperscriptℝ𝐿𝑀𝑑1\mathbf{G},\mathbf{G}^{\mathrm{PS}}\in\mathbb{R}^{L\times M\times(d+1)} are 3d-tensors. Each slice 𝐆:,l,pPSsubscriptsuperscript𝐆PS:𝑙𝑝\mathbf{G}^{\mathrm{PS}}_{:,l,p} is therefore a result of a prefix-sum (or cumulative-sum) operation applied to 𝐆:,l,psubscript𝐆:𝑙𝑝\mathbf{G}_{:,l,p}: 𝐆i,l,pPS=∑j=1i𝐆i,l,psubscriptsuperscript𝐆PS𝑖𝑙𝑝superscriptsubscript𝑗1𝑖subscript𝐆𝑖𝑙𝑝\mathbf{G}^{\mathrm{PS}}_{i,l,p}=\sum_{j=1}^{i}\mathbf{G}_{i,l,p}. An efficient algorithm to compute the prefix-sum of L𝐿L elements takes O​(L)𝑂𝐿O(L) total steps and O​(log⁡L)𝑂𝐿O(\log L) time when computed in parallel (Ladner & Fischer, 1980; Cormen et al., 2009).
See Algorithm 1 for the whole approach.

### B.2 Orthogonal Random Features - Extensions

As mentioned in the main text, for isotropic ΩΩ\Omega (true for most practical applications, including regular attention), instead of sampling ωisubscript𝜔𝑖\omega_{i} independently, we can use orthogonal random features (ORF) (Yu et al., 2016; Choromanski et al., 2017; 2018b): these maintain the marginal distributions of samples ωisubscript𝜔𝑖\omega_{i} while enforcing that different samples are orthogonal. If we need m>d𝑚𝑑m>d, ORFs still can be used locally within each d×d𝑑𝑑d\times d block of 𝐖𝐖\mathbf{W} (Yu et al., 2016).

ORFs were introduced to reduce the variance of Monte Carlo estimators (Yu et al., 2016; Choromanski et al., 2017; 2018b; 2019a; Rowland et al., 2019; Choromanski et al., 2018a; 2019b) and we showed in the theoretical and experimental sections from the main body that they do indeed lead to more accurate approximations and substantially better downstream results. There exist several variants of the ORF-mechanism and in the main body we discussed only the base one (that we refer to here as regular). Below we briefly review the most efficient ORF mechanisms (based on their strengths and costs) to present the most complete picture.

(1) Regular ORFs [R-ORFs]: Applies Gaussian orthogonal matrices (Yu et al., 2016). Encodes matrix 𝐖𝐖\mathbf{W} of ω𝜔\omega-samples (with different rows corresponding to different samples) in O​(m​d)𝑂𝑚𝑑O(md) space. Provides algorithm for computing 𝐖𝐱𝐖𝐱\mathbf{Wx} in O​(m​d)𝑂𝑚𝑑O(md) time for any 𝐱∈ℝd𝐱superscriptℝ𝑑\mathbf{x}\in\mathbb{R}^{d}. Gives unbiased estimation. Requires one-time O​(m​d2)𝑂𝑚superscript𝑑2O(md^{2}) preprocessing (Gram-Schmidt orthogonalization).

(2) Hadamard/Givens ORFs [H/G-ORFs]: Applies random Hadamard (Choromanski et al., 2017) or Givens matrices (Choromanski et al., 2019b). Encodes matrix 𝐖𝐖\mathbf{W} in O​(m)𝑂𝑚O(m) or O​(m​log⁡(d))𝑂𝑚𝑑O(m\log(d)) space. Provides algorithm for computing 𝐖𝐱𝐖𝐱\mathbf{Wx} in O​(m​log⁡(d))𝑂𝑚𝑑O(m\log(d)) time for any 𝐱∈ℝd𝐱superscriptℝ𝑑\mathbf{x}\in\mathbb{R}^{d}. Gives small bias (tending to 00 with d→∞→𝑑d\rightarrow\infty).

### B.3 Time and Space Complexity - Detailed Analysis

We see that a variant of bidirectional FAVOR+ using iid samples or R-ORFs has O​(m​d+L​d+m​L)𝑂𝑚𝑑𝐿𝑑𝑚𝐿O(md+Ld+mL) space complexity as opposed to Θ​(L2+L​d)Θsuperscript𝐿2𝐿𝑑\Theta(L^{2}+Ld) space complexity of the baseline. Unidirectional FAVOR+ using fast prefix-sum pre-computation in parallel (Ladner & Fischer, 1980; Cormen et al., 2009) has O​(m​L​d)𝑂𝑚𝐿𝑑O(mLd) space complexity to store 𝐆PSsuperscript𝐆PS\mathbf{G}^{\textrm{PS}} which can be reduced to O​(m​d+L​d+m​L)𝑂𝑚𝑑𝐿𝑑𝑚𝐿O(md+Ld+mL) by running a simple (though non-parallel in L𝐿L) aggregation of 𝐆i,:,:PSsubscriptsuperscript𝐆PS𝑖::\mathbf{G}^{\textrm{PS}}_{i,:,:} without storing the whole tensor 𝐆PSsuperscript𝐆PS\mathbf{G}^{\textrm{PS}} in memory. From Subsec. B.2, we know that if instead we use G-ORFs, then space complexity is reduced to O​(m​log⁡(d)+L​d+m​L)𝑂𝑚𝑑𝐿𝑑𝑚𝐿O(m\log(d)+Ld+mL) and if the H-ORFs mechanism is used, then space is further reduced to O​(m+L​d+m​L)=O​(L​d+m​L)𝑂𝑚𝐿𝑑𝑚𝐿𝑂𝐿𝑑𝑚𝐿O(m+Ld+mL)=O(Ld+mL). Thus for m,d≪Lmuch-less-than𝑚𝑑𝐿m,d\ll L all our variants provide substantial space complexity improvements since they do not need to store the attention matrix explicitly.

The time complexity of Algorithm 1 is O​(L​m​d)𝑂𝐿𝑚𝑑O(Lmd) (note that constructing 𝐐′superscript𝐐′\mathbf{Q}^{\prime} and 𝐊′superscript𝐊′\mathbf{K}^{\prime} can be done in time O​(L​m​d)𝑂𝐿𝑚𝑑O(Lmd)). Note that the time complexity of our method is much lower than O​(L2​d)𝑂superscript𝐿2𝑑O(L^{2}d) of the baseline for L≫mmuch-greater-than𝐿𝑚L\gg m.

As explained in Subsec. B.2, the R-ORF mechanism incurs an extra one-time O​(m​d2)𝑂𝑚superscript𝑑2O(md^{2}) cost (negligible compared to the O​(L​m​d)𝑂𝐿𝑚𝑑O(Lmd) term for L≫dmuch-greater-than𝐿𝑑L\gg d). H-ORFs or G-ORFs do not have this cost, and when FAVOR+ uses them, computing 𝐐′superscript𝐐′\mathbf{Q}^{\prime} and 𝐊′superscript𝐊′\mathbf{K}^{\prime} can be conducted in time O​(L​log⁡(m)​d)𝑂𝐿𝑚𝑑O(L\log(m)d) as opposed to O​(L​m​d)𝑂𝐿𝑚𝑑O(Lmd) (see: Subsec. B.2). Thus even though H/G-ORFs do not change the asymptotic time complexity, they improve the constant factor from the leading term. This might play an important role in training very large models.

The number of random features m𝑚m allows a trade-off between computational complexity and the level of approximation: bigger m𝑚m results in higher computation costs, but also in a lower variance of the estimate of 𝐀𝐀\mathbf{A}. In the theoretical section from the main body we showed that in practice we can take M=Θ​(d​log⁡(d))𝑀Θ𝑑𝑑M=\Theta(d\log(d)).

Observe that the FAVOR+ algorithm is highly-parallelizable, and benefits from fast matrix multiplication and broadcasted operations on GPUs or TPUs.

## C Experimental Details for Protein Modeling Tasks

### C.1 TrEMBL Dataset

Dataset
Set Name
Count
Length Statistics

Min
Max
Mean
STD
Median

TrEMBL
Train
104,863,744
2
74,488
353.09
311.16
289.00

Valid
102,400
7
11,274
353.62
307.42
289.00

Test
1,033,216
8
32,278
353.96
312.23
289.00

OOD
29,696
24
4,208
330.96
269.86
200.00

TrEMBL

(concat)

Train
4,532,224
8,192
8,192
8,192
0
8,192

Valid
4,096

We used the TrEMBL dataset444https://www.uniprot.org/statistics/TrEMBL, which contains 139,394,261 sequences of which 106,030,080 are unique. While the training dataset appears smaller than the one used in Madani et al. (Madani et al., 2020), we argue that it includes most of the relevant sequences. Specifically, the TrEMBL dataset consists of the subset of UniProtKB sequences that have been computationally analyzed but not manually curated, and accounts for ≈99.5%absentpercent99.5\approx 99.5\% of the total number of sequences in the UniProtKB dataset555https://www.uniprot.org/uniprot/.

Following the methodology described in Madani et al. (Madani et al., 2020), we used both an OOD-Test set, where a selected subset of Pfam families are held-out for valuation, and an IID split, where the remaining protein sequences are split randomly into train, valid, and test tests. We held-out the following protein families (PF18369, PF04680, PF17988, PF12325, PF03272, PF03938, PF17724, PF10696, PF11968, PF04153, PF06173, PF12378, PF04420, PF10841, PF06917, PF03492, PF06905, PF15340, PF17055, PF05318), which resulted in 29,696 OOD sequences. We note that, due to deduplication and potential TrEMBL version mismatch, our OOD-Test set does not match exactly the one in Madani et al. (Madani et al., 2020). We also note that this OOD-Test selection methodology does not guarantee that the evaluation sequences are within a minimum distance from the sequences used during training. In future work, we will include rigorous distance based splits.

The statistics for the resulting dataset splits are reported in Table 1. In the standard sequence modeling task, given the length statistics that are reported in the table, we clip single sequences to maximum length L=1024𝐿1024L=1024, which results in few sequences being truncated significantly.

In the long sequence task, the training and validation sets are obtained by concatenating the sequences, separated by an end-of-sequence token, and grouping the resulting chain into non-overlapping sequences of length L=8192𝐿8192L=8192.

### C.2 Empirical Baseline

A random baseline, with uniform probability across all the vocabulary tokens at every position, has accuracy 5%percent55\% (when including only the 20 standard amino acids) and 4%percent44\% (when also including the 5 anomalous amino acids (Consortium, 2019)). However, the empirical frequencies of the various amino acids in our dataset may be far from uniform, so we also consider an empirical baseline where the amino acid probabilities are proportional to their empirical frequencies in the training set.

Figure 9 shows the estimated empirical distribution. We use both the standard and anomalous amino acids, and we crop sequences to length 1024 to match the data processing performed for the Transformer models. The figure shows only the 20 standard amino acids, colored by their class, for comparison with the visualization on the TrEMBL web page666https://www.uniprot.org/statistics/TrEMBL.

### C.3 Tabular Results

Table 2 contains the results on the single protein sequence modeling task (L=1024𝐿1024L=1024). We report accuracy and perplexity as defined in Appendix A:

Model Type
Set Name
Model
Accuracy
Perplexity

UNI
Test
Empirical Baseline
9.92
17.80

Transformer
30.80
9.37

Performer (generalized)
31.58
9.17

OOD
Empirical Baseline
 9.07
 17.93

Transformer
19.70
13.20

Performer (generalized)
18.44
13.63

BID
Test
Transformer
33.32
9.22

Performer (generalized)
36.09
8.36

Performer (softmax)
33.00
9.24

OOD
Transformer
25.07
12.09

Performer (generalized)
24.10
12.26

Performer (softmax)
23.48
12.41

### C.4 Attention Matrix Illustration

In this section we illustrate the attention matrices produced by a Performer model. We focus on the bidirectional case and choose one Performer model trained on the standard single-sequence TrEMBL task for over 500K steps. The same analysis can be applied to unidirectional Performers as well.

We note that while the Transformer model instantiates the attention matrix in order to compute the attention output that incorporates the (queries Q𝑄Q, keys K𝐾K, values V𝑉V) triplet (see Eq. 1 in the main paper), the FAVOR mechanism returns the attention output directly (see Algorithm 1). To account for this discrepancy, we extract the attention matrices by applying each attention mechanism twice: once on each original (Q,K,V)𝑄𝐾𝑉(Q,K,V) triple to obtain the attention output, and once on a modified (Q,K,V∘)𝑄𝐾superscript𝑉(Q,K,V^{\circ}) triple, where V∘superscript𝑉V^{\circ} contains one-hot indicators for each position index, to obtain the attention matrix. The choice of V∘superscript𝑉V^{\circ} ensures that the dimension of the attention output is equal to the sequence length, and that a non-zero output on a dimension i𝑖i can only arise from a non-zero attention weight to the it​hsuperscript𝑖𝑡ℎi^{th} sequence position. Indeed, in the Transformer case, when comparing the output of this procedure with the instantiated attention matrix, the outputs match.

Attention matrix example. We start by visualizing the attention matrix for an individual protein sequence. We use the BPT1_BOVIN protein sequence777https://www.uniprot.org/uniprot/P00974, one of the most extensively studied globular proteins, which contains 100 amino acids. In Figure 10, we show the attention matrices for the first 4 layers. Note that many heads show a diagonal pattern, where each node attends to its neighbors, and some heads show a vertical pattern, where each head attends to the same fixed positions. These patterns are consistent with the patterns found in Transformer models trained on natural language (Kovaleva et al., 2019). In Figure 12 we highlight these attention patterns by focusing on the first 25 tokens, and in Figure 11, we illustrate in more detail two attention heads.

Amino acid similarity. Furthermore, we analyze the amino-acid similarity matrix estimated from the attention matrices produced by the Performer model, as described in Vig et al. (Vig et al., 2020). We aggregate the attention matrix across 800 sequences. The resulting similarity matrix is illustrated in Figure 13. Note that the Performer recognizes highly similar amino acid pairs such as (D, E) and (F, Y).

## D Extended approximation and comparison results

### D.1 Backwards Compatibility - Error Propagation

Although mentioned previously (Sec. 4.2) that the Performer with additional finetuning is backwards compatible with the Transformer, we demonstrate below in Fig. 14 that error propagation due to non-attention components of the Transformer is one of the primary reasons that pretrained Transformer weights cannot be immediately used for inference on the corresponding Performer.

### D.2 Approximate Softmax - Extended Properties

We show the following properties of our softmax approximation, in Fig. 15:

Redrawing: While the benefits of redrawing features was shown in Subsec. 4.3 of the main body of the paper, we also demonstrate its benefits when there are multiple layers with large scale (16x16 TPU-v2) training.

Unidirectional: While we have shown on TrEMBL that Performer with generalized ReLU attention outperforms softmax, we also show that approximate softmax attention can still be a solid choice, for example on ImageNet64 (U). After 100K steps of training, the Performer-ReLU, Performer-Softmax, and Performer-Softmax (SMREG) variants achieve respectively, 3.67, 3.69, 3.67 BPD.

Instability of Trigonometric Features: We see the full view of the unstable training curve when using Trigonometric softmax.

### D.3 Generalized Attention

We investigated Generalized Attention mechanisms (mentioned in Sec. 2.2) on TrEMBL when L=512𝐿512L=512 for various kernel functions. This is similar to (Tsai et al., 2019) which also experiments with various attention kernels for natural language. Using hyperparameter sweeps across multiple variables in FAVOR, we compared several kernels and also renormalization on/off (Fig. 16 and Fig. 17), where RenormalizeRenormalize\mathrm{Renormalize} corresponds to applying 𝐃−1superscript𝐃1\mathbf{D}^{-1} operator in attention, as for the standard mechanism, though we noticed that disabling it does not necessarily hurt accuracy) to produce the best training configuration for the Performer. We note that the effective batch size slightly affects the rankings (as shown by the difference between 2x2 and 4x4 TPU runs) - we by default use the generalized ReLU kernel with other default hyperparameters shown in Appendix A, as we observed that they are empirically optimal for large batch size runs (i.e. 8x8 or 16x16 TPU’s).

### D.4 Comparison with Linear Transformer

We use the attention implementation of the Linear Transformer from (Katharopoulos et al., 2020), which mainly involves setting our feature map ϕ​(x)=elu​(x)+1italic-ϕ𝑥elu𝑥1\phi(x)=\text{elu}(x)+1, where elu​(x)elu𝑥\text{elu}(x) is the shifted-eLU function from (Clevert et al., 2016).

For the sake of fairness and to prevent confounding results, while (Katharopoulos et al., 2020) also uses the GeLU nonlinearity for the MLPs in the Linear Transformer, we instead use the original ReLU nonlinearity. We also used the exact same training hyperparameters as Performer-ReLU on our exact ProGen setting from Fig. 6. Ultimately, we empirically found that the Linear Transformer possessed numerical instability during training via unstable training curves, ultimately stopping training by producing exploding gradients (NaNs) (Fig. 18).

### D.5 Long Range Arena

Performers are compared against many additional (scalable and not scalable) methods not included in our paper: Local Attention, Sparse Attention, Longformer, Sinkhorn Transformer, Synthesizer, Big Bird and the aforementioned Linear Transformer on challenging long range context tasks in the Long Range Arena (Tay et al., 2021), with Fig. 19 displaying the original paper’s results. Performers obtain the largest LRA (Long Range Arena) score among all tested scalable Transformers methods (which we define by having speed of > 100 examples/sec).

Tasks used for comparison include: (1) a longer variation of the standard ListOps task proposed in (Nangia & Bowman, 2018), (2) byte-level text classification using real-world data, (3) byte-level document retrieval, (4) image classification on sequences of pixels, and (5) Pathfinder task (long-range spatial dependency problem). In the Long Range Arena paper, the authors found that all models do not learn anything on Path-X task (denoted by FAIL), contrary to the Pathfinder task, which shows that increasing the sequence length can cause seriously difficulties for model training.

## E Computation costs - Extended results

In this subsection, we empirically measure computational costs in terms wall clock time on forward and backward passes for three scenarios in Fig. 20:

- 1.

Performer, with varying number of layers. We show that our method can scale up to (but not necessarily limited to) even 20 layers.

- 2.

Attention time complexities when comparing standard attention (from Transformer) and FAVOR (from Performer). Note that the maximum memory size here is not reflective of the maximum memory size in an actual model (shown below), as this benchmark requires computing explicit tensors (causing memory increases) in Jax, while a model does not.

- 3.

Time complexities when comparing the Transformer and Performer models. "X" (OPT) denotes the maximum possible speedup achievable, when attention simply returns the 𝐕𝐕\mathbf{V}-vector, showing that the Performer is nearly optimal. We see that the maximum possible power of 2 length allowed on a V100 GPU (16GB) is 215=32768superscript215327682^{15}=32768 using regular dimensions.

Since some of the computational bottleneck in the Transformer may originate from the extra feed-forward layers (Kitaev et al., 2020), we also benchmark the “Small" version, i.e. (nh​e​a​d​s,nl​a​y​e​r​s,df​f,d)=(1,6,64,64)subscript𝑛ℎ𝑒𝑎𝑑𝑠subscript𝑛𝑙𝑎𝑦𝑒𝑟𝑠subscript𝑑𝑓𝑓𝑑166464(n_{heads},n_{layers},d_{ff},d)=(1,6,64,64) as well, when the attention component is the dominant source of computation and memory. We remind the reader that the “Regular" version consists of (nh​e​a​d​s,nl​a​y​e​r​s,df​f,d)=(8,6,2048,512)subscript𝑛ℎ𝑒𝑎𝑑𝑠subscript𝑛𝑙𝑎𝑦𝑒𝑟𝑠subscript𝑑𝑓𝑓𝑑862048512(n_{heads},n_{layers},d_{ff},d)=(8,6,2048,512).

## F Theoretical results

We provide here the proofs of all theoretical results presented in the paper.

### F.1 Proof of Lemma 1

###### Proof.

We first deduce that for any 𝒂,𝒃∈ℝd𝒂𝒃superscriptℝ𝑑{\bm{a}},{\bm{b}}\in\mathbb{R}^{d}

SM​(𝐱,𝐲)=exp⁡(𝒙⊤​𝒚)=exp⁡(−‖𝒙‖2/2)⋅exp⁡(‖𝒙+𝒚‖2/2)⋅exp⁡(−‖𝒚‖2/2).SM𝐱𝐲superscript𝒙top𝒚⋅superscriptnorm𝒙22superscriptnorm𝒙𝒚22superscriptnorm𝒚22\mathrm{SM}(\mathbf{x},\mathbf{y})=\exp({\bm{x}}^{\top}{\bm{y}})=\exp(-\|{\bm{x}}\|^{2}/2)\cdot\exp(\|{\bm{x}}+{\bm{y}}\|^{2}/2)\cdot\exp(-\|{\bm{y}}\|^{2}/2).

Next, let 𝒘∈ℝd𝒘superscriptℝ𝑑{\bm{w}}\in\mathbb{R}^{d}. We use the fact that

(2​π)−d/2​∫exp⁡(−‖𝒘−𝒄‖22/2)​𝑑𝒘=1superscript2𝜋𝑑2superscriptsubscriptnorm𝒘𝒄222differential-d𝒘1(2\pi)^{-d/2}\int\exp(-\|{\bm{w}}-{\bm{c}}\|_{2}^{2}/2)d{\bm{w}}=1

for any 𝒄∈ℝd𝒄superscriptℝ𝑑{\bm{c}}\in\mathbb{R}^{d} and derive:

exp(∥𝒙\displaystyle\exp(\|{\bm{x}}
+𝒚∥2/2)=(2π)−d/2exp(∥𝒙+𝒚∥2/2)∫exp(−∥𝒘−(𝒙+𝒚)∥2/2)d𝒘\displaystyle+{\bm{y}}\|^{2}/2)=(2\pi)^{-d/2}\exp(\|{\bm{x}}+{\bm{y}}\|^{2}/2)\int\exp(-\|{\bm{w}}-({\bm{x}}+{\bm{y}})\|^{2}/2)d{\bm{w}}

=(2​π)−d/2​∫exp⁡(−‖𝒘‖2/2+𝒘⊤​(𝒙+𝒚)−‖𝒙+𝒚‖2/2+‖𝒙+𝒚‖2/2)​𝑑𝒘absentsuperscript2𝜋𝑑2superscriptnorm𝒘22superscript𝒘top𝒙𝒚superscriptnorm𝒙𝒚22superscriptnorm𝒙𝒚22differential-d𝒘\displaystyle=(2\pi)^{-d/2}\int\exp(-\|{\bm{w}}\|^{2}/2+{\bm{w}}^{\top}({\bm{x}}+{\bm{y}})-\|{\bm{x}}+{\bm{y}}\|^{2}/2+\|{\bm{x}}+{\bm{y}}\|^{2}/2)d{\bm{w}}

=(2​π)−d/2​∫exp⁡(−‖𝒘‖2/2+𝒘⊤​(𝒙+𝒚))​𝑑𝒘absentsuperscript2𝜋𝑑2superscriptnorm𝒘22superscript𝒘top𝒙𝒚differential-d𝒘\displaystyle=(2\pi)^{-d/2}\int\exp(-\|{\bm{w}}\|^{2}/2+{\bm{w}}^{\top}({\bm{x}}+{\bm{y}}))d{\bm{w}}

=(2​π)−d/2​∫exp⁡(−‖𝒘‖2/2)⋅exp⁡(𝒘⊤​𝒙)⋅exp⁡(𝒘⊤​𝒚)​𝑑𝒘absentsuperscript2𝜋𝑑2⋅superscriptnorm𝒘22superscript𝒘top𝒙superscript𝒘top𝒚differential-d𝒘\displaystyle=(2\pi)^{-d/2}\int\exp(-\|{\bm{w}}\|^{2}/2)\cdot\exp({\bm{w}}^{\top}{\bm{x}})\cdot\exp({\bm{w}}^{\top}{\bm{y}})d{\bm{w}}

=𝔼ω∼𝒩​(𝟎d,𝐈d)​[exp⁡(ω⊤​𝒙)⋅exp⁡(ω⊤​𝒚)].absentsubscript𝔼similar-to𝜔𝒩subscript0𝑑subscript𝐈𝑑delimited-[]⋅superscript𝜔top𝒙superscript𝜔top𝒚\displaystyle=\mathbb{E}_{\omega\sim\mathcal{N}(\mathbf{0}_{d},\mathbf{I}_{d})}[\exp(\omega^{\top}{\bm{x}})\cdot\exp(\omega^{\top}{\bm{y}})].

That completes the proof of the first part of the lemma. An identity involving hyperbolic cosine function is implied by the fact that for every 𝐮∈ℝd𝐮superscriptℝ𝑑\mathbf{u}\in\mathbb{R}^{d} and ω∼𝒩​(0,𝐈d)similar-to𝜔𝒩0subscript𝐈𝑑\omega\sim\mathcal{N}(0,\mathbf{I}_{d}) the following is true:

𝔼​[exp⁡(ω⊤​𝐮)]=∑i=0∞𝔼​[(ω⊤​𝐮)2​i](2​i)!=12​∑i=0∞𝔼​[(ω⊤​𝐮)2​i]+𝔼​[(−ω⊤​𝐮)2​i](2​i)!.𝔼delimited-[]superscript𝜔top𝐮superscriptsubscript𝑖0𝔼delimited-[]superscriptsuperscript𝜔top𝐮2𝑖2𝑖12superscriptsubscript𝑖0𝔼delimited-[]superscriptsuperscript𝜔top𝐮2𝑖𝔼delimited-[]superscriptsuperscript𝜔top𝐮2𝑖2𝑖\mathbb{E}[\exp(\omega^{\top}\mathbf{u})]=\sum_{i=0}^{\infty}\frac{\mathbb{E}[(\omega^{\top}\mathbf{u})^{2i}]}{(2i)!}=\frac{1}{2}\sum_{i=0}^{\infty}\frac{\mathbb{E}[(\omega^{\top}\mathbf{u})^{2i}]+\mathbb{E}[(-\omega^{\top}\mathbf{u})^{2i}]}{(2i)!}.

(12)

The cancellation of the odd moments 𝔼​[(ω⊤​𝐮)2​i+1]𝔼delimited-[]superscriptsuperscript𝜔top𝐮2𝑖1\mathbb{E}[(\omega^{\top}\mathbf{u})^{2i+1}] follows directly from the fact that ω𝜔\omega is taken from the isotropic distribution (i.e. distribution with pdf function constant on each sphere).
That completes the proof.
∎

### F.2 Proof of Lemma 2

###### Proof.

Denote: 𝐳=𝐱+𝐲𝐳𝐱𝐲\mathbf{z}=\mathbf{x}+\mathbf{y} and Δ=𝐱−𝐲Δ𝐱𝐲\Delta=\mathbf{x}-\mathbf{y}.
Note that by using standard trigonometric identities (and the fact that the variance of the sum of independent random variables is the sum of variances of those random variables), we can get the following for ω∼𝒩​(0,𝐈d)similar-to𝜔𝒩0subscript𝐈𝑑\omega\sim\mathcal{N}(0,\mathbf{I}_{d}):

MSE​(SM^mtrig​(𝐱,𝐲))=1m​exp⁡(‖𝐱‖2+‖𝐲‖2)​Var​(cos⁡(ω⊤​Δ)).MSEsubscriptsuperscript^SMtrig𝑚𝐱𝐲1𝑚superscriptnorm𝐱2superscriptnorm𝐲2Varsuperscript𝜔topΔ\mathrm{MSE}(\widehat{\mathrm{SM}}^{\mathrm{trig}}_{m}(\mathbf{x},\mathbf{y}))=\frac{1}{m}\exp(\|\mathbf{x}\|^{2}+\|\mathbf{y}\|^{2})\mathrm{Var}(\cos(\omega^{\top}\Delta)).

(13)

Using the fact that (see: Lemma 1 in (Yu et al., 2016); note that in that lemma they use notation: z𝑧z for what we denote as: ‖Δ‖normΔ\|\Delta\|):

Var​(cos⁡(ω⊤​Δ))=12​(1−exp⁡(−‖Δ‖2))2,Varsuperscript𝜔topΔ12superscript1superscriptnormΔ22\mathrm{Var}(\cos(\omega^{\top}\Delta))=\frac{1}{2}(1-\exp(-\|\Delta\|^{2}))^{2},

(14)

we obtain:

MSE​(SM^mtrig​(𝐱,𝐲))=12​m​exp⁡(‖𝐱‖2+‖𝐲‖2)​(1−exp⁡(−‖Δ‖2))2=12​m​exp⁡(‖𝐳‖2)​SM−2​(𝐱,𝐲)​(1−exp⁡(−‖Δ‖2))2,MSEsubscriptsuperscript^SMtrig𝑚𝐱𝐲12𝑚superscriptdelimited-∥∥𝐱2superscriptdelimited-∥∥𝐲2superscript1superscriptdelimited-∥∥Δ2212𝑚superscriptdelimited-∥∥𝐳2superscriptSM2𝐱𝐲superscript1superscriptdelimited-∥∥Δ22\displaystyle\begin{split}\mathrm{MSE}(\widehat{\mathrm{SM}}^{\mathrm{trig}}_{m}(\mathbf{x},\mathbf{y}))=\frac{1}{2m}\exp(\|\mathbf{x}\|^{2}+\|\mathbf{y}\|^{2})(1-\exp(-\|\Delta\|^{2}))^{2}=\\
\frac{1}{2m}\exp(\|\mathbf{z}\|^{2})\mathrm{SM}^{-2}(\mathbf{x},\mathbf{y})(1-\exp(-\|\Delta\|^{2}))^{2},\end{split}

(15)

which completes the first part of the proof.
To obtain the formula for: MSE​(SM^m+​(𝐱,𝐲))MSEsubscriptsuperscript^SM𝑚𝐱𝐲\mathrm{MSE}(\widehat{\mathrm{SM}}^{\mathrm{+}}_{m}(\mathbf{x},\mathbf{y}))
notice first that:

𝔼ω∼𝒩​(0,𝐈d)​[exp⁡(ω⊤​𝐳)]=exp⁡(‖𝐳‖22).subscript𝔼similar-to𝜔𝒩0subscript𝐈𝑑delimited-[]superscript𝜔top𝐳superscriptnorm𝐳22\mathbb{E}_{\omega\sim\mathcal{N}(0,\mathbf{I}_{d})}[\exp(\omega^{\top}\mathbf{z})]=\exp(\frac{\|\mathbf{z}\|^{2}}{2}).

(16)

The above immediately follows from the fact that positive random feature maps provide unbiased estimation of the softmax-kernel, thus the following is true:

SM​(𝐱,𝐲)=exp⁡(−‖𝐱‖2+‖𝐲‖22)​𝔼ω∼𝒩​(0,𝐈d)​[exp⁡(ω⊤​𝐳)].SM𝐱𝐲superscriptnorm𝐱2superscriptnorm𝐲22subscript𝔼similar-to𝜔𝒩0subscript𝐈𝑑delimited-[]superscript𝜔top𝐳\mathrm{SM}(\mathbf{x},\mathbf{y})=\exp(-\frac{\|\mathbf{x}\|^{2}+\|\mathbf{y}\|^{2}}{2})\mathbb{E}_{\omega\sim\mathcal{N}(0,\mathbf{I}_{d})}[\exp(\omega^{\top}\mathbf{z})].

(17)

Therefore we obtain:

MSE​(SM^m+​(𝐱,𝐲))=1m​exp⁡(−(‖𝐱‖2+‖𝐲‖2))​Var​(exp⁡(ω⊤​𝐳))=1m​exp⁡(−(‖𝐱‖2+‖𝐲‖2))​(𝔼​[exp⁡(2​ω⊤​𝐳)]−(𝔼​[exp⁡(ω⊤​𝐳)])2)=1m​exp⁡(−(‖𝐱‖2+‖𝐲‖2))​(exp⁡(2​‖𝐳‖2)−exp⁡(‖𝐳‖2)),MSEsubscriptsuperscript^SM𝑚𝐱𝐲1𝑚superscriptdelimited-∥∥𝐱2superscriptdelimited-∥∥𝐲2Varsuperscript𝜔top𝐳1𝑚superscriptdelimited-∥∥𝐱2superscriptdelimited-∥∥𝐲2𝔼delimited-[]2superscript𝜔top𝐳superscript𝔼delimited-[]superscript𝜔top𝐳21𝑚superscriptdelimited-∥∥𝐱2superscriptdelimited-∥∥𝐲22superscriptdelimited-∥∥𝐳2superscriptdelimited-∥∥𝐳2\displaystyle\begin{split}\mathrm{MSE}(\widehat{\mathrm{SM}}^{\mathrm{+}}_{m}(\mathbf{x},\mathbf{y}))=\frac{1}{m}\exp(-(\|\mathbf{x}\|^{2}+\|\mathbf{y}\|^{2}))\mathrm{Var}(\exp(\omega^{\top}\mathbf{z}))=\\
\frac{1}{m}\exp(-(\|\mathbf{x}\|^{2}+\|\mathbf{y}\|^{2}))\left(\mathbb{E}[\exp(2\omega^{\top}\mathbf{z})]-(\mathbb{E}[\exp(\omega^{\top}\mathbf{z})])^{2}\right)=\\
\frac{1}{m}\exp(-(\|\mathbf{x}\|^{2}+\|\mathbf{y}\|^{2}))(\exp(2\|\mathbf{z}\|^{2})-\exp(\|\mathbf{z}\|^{2})),\end{split}

(18)

where the last equality follows from Equation 16.
Therefore we have:

MSE​(SM^m+​(𝐱,𝐲))=1m​exp⁡(−(‖𝐱‖2+‖𝐲‖2))​exp⁡(‖𝐳‖2)​(exp⁡(‖𝐳‖2)−1)=1m​exp⁡(‖𝐳‖2)​SM2​(𝐱,𝐲)​(1−exp⁡(−‖𝐳‖2)).MSEsubscriptsuperscript^SM𝑚𝐱𝐲1𝑚superscriptdelimited-∥∥𝐱2superscriptdelimited-∥∥𝐲2superscriptdelimited-∥∥𝐳2superscriptdelimited-∥∥𝐳211𝑚superscriptdelimited-∥∥𝐳2superscriptSM2𝐱𝐲1superscriptdelimited-∥∥𝐳2\displaystyle\begin{split}\mathrm{MSE}(\widehat{\mathrm{SM}}^{\mathrm{+}}_{m}(\mathbf{x},\mathbf{y}))=\frac{1}{m}\exp(-(\|\mathbf{x}\|^{2}+\|\mathbf{y}\|^{2}))\exp(\|\mathbf{z}\|^{2})(\exp(\|\mathbf{z}\|^{2})-1)=\\
\frac{1}{m}\exp(\|\mathbf{z}\|^{2})\mathrm{SM}^{2}(\mathbf{x},\mathbf{y})(1-\exp(-\|\mathbf{z}\|^{2})).\end{split}

(19)

Finally,

MSE(SM^mhyp+(𝐱,𝐲))=14​mexp(−‖𝐱‖2+‖𝐲‖22)2(Var(exp(ω⊤𝐳))+Var(exp(−ω⊤𝐳))+2Cov(exp(ω⊤𝐳)),exp(−ω⊤𝐳))))=14​mexp(−‖𝐱‖2+‖𝐲‖22)2(2Var(exp(ω⊤𝐳))+2Cov(exp(ω⊤𝐳)),exp(−ω⊤𝐳)))))=12​mexp(−(∥𝐱∥2+∥𝐲∥2))(Var​(exp⁡(ω⊤​𝐳))+1−(𝔼​[exp⁡(ω⊤​𝐳)])2)=12​m​exp⁡(−(‖𝐱‖2+‖𝐲‖2))(exp⁡(2​‖𝐳‖2)−exp⁡(‖𝐳‖2)+1−exp⁡(‖𝐳‖2))=12​m​exp⁡(−(‖𝐱‖2+‖𝐲‖2))​(exp⁡(‖𝐳‖2)−1)2=12​(1−exp⁡(−‖𝐳‖2))​MSE​(SM^m+​(𝐱,𝐲)).\displaystyle\begin{split}\mathrm{MSE}(\widehat{\mathrm{SM}}_{m}^{\mathrm{hyp+}}(\mathbf{x},\mathbf{y}))=\frac{1}{4m}\exp(-\frac{\|\mathbf{x}\|^{2}+\|\mathbf{y}\|^{2}}{2})^{2}(\mathrm{Var}(\exp(\omega^{\top}\mathbf{z}))+\mathrm{Var}(\exp(-\omega^{\top}\mathbf{z}))+\\
2\mathrm{Cov}(\exp(\omega^{\top}\mathbf{z})),\exp(-\omega^{\top}\mathbf{z}))))=\frac{1}{4m}\exp(-\frac{\|\mathbf{x}\|^{2}+\|\mathbf{y}\|^{2}}{2})^{2}(2\mathrm{Var}(\exp(\omega^{\top}\mathbf{z}))+\\
2\mathrm{Cov}(\exp(\omega^{\top}\mathbf{z})),\exp(-\omega^{\top}\mathbf{z})))))=\frac{1}{2m}\exp(-(\|\mathbf{x}\|^{2}+\|\mathbf{y}\|^{2}))\\
(\mathrm{Var}(\exp(\omega^{\top}\mathbf{z}))+1-(\mathbb{E}[\exp(\omega^{\top}\mathbf{z})])^{2})=\frac{1}{2m}\exp(-(\|\mathbf{x}\|^{2}+\|\mathbf{y}\|^{2}))\\
(\exp(2\|\mathbf{z}\|^{2})-\exp(\|\mathbf{z}\|^{2})+1-\exp(\|\mathbf{z}\|^{2}))=\frac{1}{2m}\exp(-(\|\mathbf{x}\|^{2}+\|\mathbf{y}\|^{2}))(\exp(\|\mathbf{z}\|^{2})-1)^{2}\\
=\frac{1}{2}(1-\exp(-\|\mathbf{z}\|^{2}))\mathrm{MSE}(\widehat{\mathrm{SM}}^{\mathrm{+}}_{m}(\mathbf{x},\mathbf{y})).\end{split}

(20)

In the chain of equalities above we used the fact that random variables exp​(ω⊤​𝐳)expsuperscript𝜔top𝐳\mathrm{exp}(\omega^{\top}\mathbf{z}) and
exp​(−ω⊤​𝐳)expsuperscript𝜔top𝐳\mathrm{exp}(-\omega^{\top}\mathbf{z}) have the same distribution. This is true since ω𝜔\omega and −ω𝜔-\omega have the same distribution (ω𝜔\omega is Gaussian).
That completes the proof.
∎

### F.3 Proof of Theorem 1

###### Proof.

Let 𝐱,𝐲∈ℝd𝐱𝐲superscriptℝ𝑑\mathbf{x},\mathbf{y}\in\mathbb{R}^{d} be respectively a query/key.
Note that from the definition of SMREG​(𝐱,𝐲)SMREG𝐱𝐲\mathrm{SMREG}(\mathbf{x},\mathbf{y}) we have
for 𝐳=𝐱+𝐲𝐳𝐱𝐲\mathbf{z}=\mathbf{x}+\mathbf{y}:

SMREG​(𝐱,𝐲)=exp⁡(−‖𝐱‖2+‖𝐲‖22)​∑k=0∞1(2​k)!​‖𝐳‖2​k​dk​𝔼ω∼𝒩​(0,𝐈d)​[(ω‖ω‖2​𝐞1)2​k],SMREG𝐱𝐲superscriptnorm𝐱2superscriptnorm𝐲22superscriptsubscript𝑘012𝑘superscriptnorm𝐳2𝑘superscript𝑑𝑘subscript𝔼similar-to𝜔𝒩0subscript𝐈𝑑delimited-[]superscript𝜔subscriptnorm𝜔2subscript𝐞12𝑘\mathrm{SMREG}(\mathbf{x},\mathbf{y})=\exp(-\frac{\|\mathbf{x}\|^{2}+\|\mathbf{y}\|^{2}}{2})\sum_{k=0}^{\infty}\frac{1}{(2k)!}\|\mathbf{z}\|^{2k}d^{k}\mathbb{E}_{\omega\sim\mathcal{N}(0,\mathbf{I}_{d})}[(\frac{\omega}{\|\omega\|_{2}}\mathbf{e}_{1})^{2k}],

(21)

where 𝐞1​=def​(1,0,…,0)⊤∈ℝdsubscript𝐞1defsuperscript10…0topsuperscriptℝ𝑑\mathbf{e}_{1}\overset{\mathrm{def}}{=}(1,0,...,0)^{\top}\in\mathbb{R}^{d}. To obtain the above we used the fact that 𝒩​(0,𝐈d)𝒩0subscript𝐈𝑑\mathcal{N}(0,\mathbf{I}_{d}) is isotropic (that in particular implies zeroing of the even terms in the Taylor expansion).

Let us denote: A​(k,d)​=def​𝔼ω∼𝒩​(0,𝐈d)​[(ω‖ω‖2​𝐞1)2​k]𝐴𝑘𝑑defsubscript𝔼similar-to𝜔𝒩0subscript𝐈𝑑delimited-[]superscript𝜔subscriptnorm𝜔2subscript𝐞12𝑘A(k,d)\overset{\mathrm{def}}{=}\mathbb{E}_{\omega\sim\mathcal{N}(0,\mathbf{I}_{d})}[(\frac{\omega}{\|\omega\|_{2}}\mathbf{e}_{1})^{2k}]. It turns out that:

A​(2​k,d)=(2​k−1)!!(d+2​k−2)​(d+2​k−4)⋅…⋅d.𝐴2𝑘𝑑double-factorial2𝑘1⋅𝑑2𝑘2𝑑2𝑘4…𝑑A(2k,d)=\frac{(2k-1)!!}{(d+2k-2)(d+2k-4)\cdot...\cdot d}.

(22)

The proof of that fact can be found in the supplement of (Choromanski et al., 2018b), yet we provide it below for completeness and the convenience of the Reader:

###### Lemma 3.

Expression A​(2​k,d)𝐴2𝑘𝑑A(2k,d) satisfies the following for k∈ℕ𝑘ℕk\in\mathbb{N} :

A​(2​k,d)=(2​k−1)!!(d+2​k−2)​(d+2​k−4)⋅…⋅d.𝐴2𝑘𝑑double-factorial2𝑘1⋅𝑑2𝑘2𝑑2𝑘4…𝑑A(2k,d)=\frac{(2k-1)!!}{(d+2k-2)(d+2k-4)\cdot...\cdot d}.

(23)

###### Proof.

Note first that for d≥2𝑑2d\geq 2 the density function pd​(θ)subscript𝑝𝑑𝜃p_{d}(\theta) of the angle between a vector 𝐫∈ℝd𝐫superscriptℝ𝑑\mathbf{r}\in\mathbb{R}^{d} chosen uniformly at random from the unit sphere and 𝐞1subscript𝐞1\mathbf{e}_{1} is given by the following formula:

pd​(θ)=sind−2⁡(θ)∫0πsind−2​(θ)⁡d​θ.subscript𝑝𝑑𝜃superscript𝑑2𝜃superscriptsubscript0𝜋superscript𝑑2𝜃𝑑𝜃p_{d}(\theta)=\frac{\sin^{d-2}(\theta)}{\int_{0}^{\pi}\sin^{d-2(\theta)}d\theta}.

(24)

Let us denote: F​(k,d)​=def​∫0πcosk⁡(θ)​sind⁡(θ)​𝑑θ𝐹𝑘𝑑defsuperscriptsubscript0𝜋superscript𝑘𝜃superscript𝑑𝜃differential-d𝜃F(k,d)\overset{\mathrm{def}}{=}\int_{0}^{\pi}\cos^{k}(\theta)\sin^{d}(\theta)d\theta.
Using partial integration, we get:

∫0πcosk⁡(θ)​sind⁡(θ)​𝑑θ=∫0πcosk−1⁡(θ)​sind⁡(θ)​(sin⁡(θ))′​𝑑θ=cosk−1(θ)sind+1(θ)|0π−∫0πsin(θ)((k−1)cosk−2(θ)(−sin(θ))sind(θ)+dcosk(θ)sind−1(θ))dθ.superscriptsubscript0𝜋superscript𝑘𝜃superscript𝑑𝜃differential-d𝜃superscriptsubscript0𝜋superscript𝑘1𝜃superscript𝑑𝜃superscript𝜃′differential-d𝜃evaluated-atsuperscript𝑘1𝜃superscript𝑑1𝜃0𝜋superscriptsubscript0𝜋𝜃𝑘1superscript𝑘2𝜃𝜃superscript𝑑𝜃𝑑superscript𝑘𝜃superscript𝑑1𝜃𝑑𝜃\displaystyle\begin{split}\int_{0}^{\pi}\cos^{k}(\theta)\sin^{d}(\theta)d\theta=\int_{0}^{\pi}\cos^{k-1}(\theta)\sin^{d}(\theta)(\sin(\theta))^{\prime}d\theta=\\
\cos^{k-1}(\theta)\sin^{d+1}(\theta)|^{\pi}_{0}-\int_{0}^{\pi}\sin(\theta)((k-1)\cos^{k-2}(\theta)(-\sin(\theta))\sin^{d}(\theta)+\\
d\cos^{k}(\theta)\sin^{d-1}(\theta))d\theta.\end{split}

(25)

Thus we conclude that: F​(k,d)=k−1d+1​F​(k−2,d+2)𝐹𝑘𝑑𝑘1𝑑1𝐹𝑘2𝑑2F(k,d)=\frac{k-1}{d+1}F(k-2,d+2).
Therefore we have:

F​(2​k,d)=(2​k−1)!!(d+1)​(d+3)⋅…⋅(d+2​k−1)​∫0πsind+2​k⁡(θ)​𝑑θ.𝐹2𝑘𝑑double-factorial2𝑘1⋅𝑑1𝑑3…𝑑2𝑘1superscriptsubscript0𝜋superscript𝑑2𝑘𝜃differential-d𝜃F(2k,d)=\frac{(2k-1)!!}{(d+1)(d+3)\cdot...\cdot(d+2k-1)}\int_{0}^{\pi}\sin^{d+2k}(\theta)d\theta.

(26)

We again conduct partial integration and get:

∫0πsind⁡(θ)​𝑑θ=−1d​sind−1⁡(θ)​cos⁡(θ)|0π+d−1d​∫0πsind−2⁡(θ)​𝑑θ=d−1d​∫0πsind−2⁡(θ)​𝑑θ.superscriptsubscript0𝜋superscript𝑑𝜃differential-d𝜃evaluated-at1𝑑superscript𝑑1𝜃𝜃0𝜋𝑑1𝑑superscriptsubscript0𝜋superscript𝑑2𝜃differential-d𝜃𝑑1𝑑superscriptsubscript0𝜋superscript𝑑2𝜃differential-d𝜃\displaystyle\begin{split}\int_{0}^{\pi}\sin^{d}(\theta)d\theta=-\frac{1}{d}\sin^{d-1}(\theta)\cos(\theta)|^{\pi}_{0}+\\
\frac{d-1}{d}\int_{0}^{\pi}\sin^{d-2}(\theta)d\theta=\frac{d-1}{d}\int_{0}^{\pi}\sin^{d-2}(\theta)d\theta.\end{split}

(27)

Therefore we conclude that:

A​(2​k,d)=1d−3d−2​d−5d−4⋅…​(2​k−1)!!(d−1)​(d+1)⋅…⋅(d+2​k−3)​d+2​k−3d+2​k−2​d+2​k−5d+2​k−4⋅….=(2​k−1)!!(d+2​k−2)​(d+2​k−4)⋅…⋅d,\displaystyle\begin{split}A(2k,d)=\frac{1}{\frac{d-3}{d-2}\frac{d-5}{d-4}\cdot...}\frac{(2k-1)!!}{(d-1)(d+1)\cdot...\cdot(d+2k-3)}\frac{d+2k-3}{d+2k-2}\frac{d+2k-5}{d+2k-4}\cdot....=\\
\frac{(2k-1)!!}{(d+2k-2)(d+2k-4)\cdot...\cdot d},\end{split}

(28)

which completes the proof.
∎

Applying the above lemma, we get:

SMREG​(𝐱,𝐲)=exp⁡(−‖𝐱‖2+‖𝐲‖22)​∑k=0∞1(2​k)!​‖𝐳‖2​k​dk​(2​k−1)!!(d+2​k−2)​(d+2​k−4)⋅…⋅d=exp⁡(−‖𝐱‖2+‖𝐲‖22)​∑k=0∞wkk!​f​(k,d),SMREG𝐱𝐲superscriptnorm𝐱2superscriptnorm𝐲22superscriptsubscript𝑘012𝑘superscriptdelimited-∥∥𝐳2𝑘superscript𝑑𝑘double-factorial2𝑘1⋅𝑑2𝑘2𝑑2𝑘4…𝑑superscriptnorm𝐱2superscriptnorm𝐲22superscriptsubscript𝑘0superscript𝑤𝑘𝑘𝑓𝑘𝑑\displaystyle\begin{split}\mathrm{SMREG}(\mathbf{x},\mathbf{y})=\exp(-\frac{\|\mathbf{x}\|^{2}+\|\mathbf{y}\|^{2}}{2})\sum_{k=0}^{\infty}\frac{1}{(2k)!}\|\mathbf{z}\|^{2k}d^{k}\frac{(2k-1)!!}{(d+2k-2)(d+2k-4)\cdot...\cdot d}\\
=\exp(-\frac{\|\mathbf{x}\|^{2}+\|\mathbf{y}\|^{2}}{2})\sum_{k=0}^{\infty}\frac{w^{k}}{k!}f(k,d),\end{split}

(29)

where w=‖𝐳‖22𝑤superscriptnorm𝐳22w=\frac{\|\mathbf{z}\|^{2}}{2}
and f​(k,d)=dk(d+2​k−2)​(d+2​k−4)⋅…⋅d𝑓𝑘𝑑superscript𝑑𝑘⋅𝑑2𝑘2𝑑2𝑘4…𝑑f(k,d)=\frac{d^{k}}{(d+2k-2)(d+2k-4)\cdot...\cdot d}.

Thus we obtain:

SMREG​(𝐱,𝐲)SM​(𝐱,𝐲)=e−w​∑k=0∞wkk!​f​(k,d).SMREG𝐱𝐲SM𝐱𝐲superscript𝑒𝑤superscriptsubscript𝑘0superscript𝑤𝑘𝑘𝑓𝑘𝑑\frac{\mathrm{SMREG}(\mathbf{x},\mathbf{y})}{\mathrm{SM}(\mathbf{x},\mathbf{y})}=e^{-w}\sum_{k=0}^{\infty}\frac{w^{k}}{k!}f(k,d).

(30)

Note first that for k≥1𝑘1k\geq 1 we have: f​(k,d)≤1𝑓𝑘𝑑1f(k,d)\leq 1, thus:

SMREG​(𝐱,𝐲)≤SM​(𝐱,𝐲).SMREG𝐱𝐲SM𝐱𝐲\mathrm{SMREG}(\mathbf{x},\mathbf{y})\leq\mathrm{SM}(\mathbf{x},\mathbf{y}).

(31)

We also have for l=d13𝑙superscript𝑑13l=d^{\frac{1}{3}}:

SMREG​(𝐱,𝐲)SM​(𝐱,𝐲)=e−w​∑k=0lwkk!​f​(k,d)+e−w​∑k=l+1∞wkk!​f​(k,d)≥f​(l,d)​e−w​∑k=0lwkk!+e−w​∑k=l+1∞wkk!​f​(k,d)≥f​(l,d)​(1−e−w​∑k=l+1∞wkk!)=f​(l,d)​(1−ℙ​[Po​(w)>l]),SMREG𝐱𝐲SM𝐱𝐲superscript𝑒𝑤superscriptsubscript𝑘0𝑙superscript𝑤𝑘𝑘𝑓𝑘𝑑superscript𝑒𝑤superscriptsubscript𝑘𝑙1superscript𝑤𝑘𝑘𝑓𝑘𝑑𝑓𝑙𝑑superscript𝑒𝑤superscriptsubscript𝑘0𝑙superscript𝑤𝑘𝑘superscript𝑒𝑤superscriptsubscript𝑘𝑙1superscript𝑤𝑘𝑘𝑓𝑘𝑑𝑓𝑙𝑑1superscript𝑒𝑤superscriptsubscript𝑘𝑙1superscript𝑤𝑘𝑘𝑓𝑙𝑑1ℙdelimited-[]Po𝑤𝑙\displaystyle\begin{split}\frac{\mathrm{SMREG}(\mathbf{x},\mathbf{y})}{\mathrm{SM}(\mathbf{x},\mathbf{y})}=e^{-w}\sum_{k=0}^{l}\frac{w^{k}}{k!}f(k,d)+e^{-w}\sum_{k=l+1}^{\infty}\frac{w^{k}}{k!}f(k,d)\geq\\
f(l,d)e^{-w}\sum_{k=0}^{l}\frac{w^{k}}{k!}+e^{-w}\sum_{k=l+1}^{\infty}\frac{w^{k}}{k!}f(k,d)\geq f(l,d)(1-e^{-w}\sum_{k=l+1}^{\infty}\frac{w^{k}}{k!})=\\
f(l,d)(1-\mathbb{P}[\mathrm{Po}(w)>l]),\end{split}

(32)

where Po​(w)Po𝑤\mathrm{Po}(w) stands for the random variable of Poisson distribution with parameter w𝑤w.
Therefore we get for t=ln⁡(lw)𝑡𝑙𝑤t=\ln(\frac{l}{w}):

SMREG​(𝐱,𝐲)SM​(𝐱,𝐲)≥(1−2​l−2d)l​(1−ℙ​[Po​(w)>l])≥exp⁡(l​ln⁡(1−2​l−2d))​(1−ℙ​[t​Po​(w)≥t​l])=exp⁡(l​∑i=1∞(−1)i​(2​l−2d)ii)​(1−ℙ​[exp⁡(t​Po​(w)−t​l)≥1])≥exp⁡(−2d13+o​(1d13))​(1−exp⁡(−t​l)​𝔼​[exp⁡(t​Po​(w))])=exp⁡(−2d13+o​(1d13))​(1−exp⁡(−w−l​(t−1))),SMREG𝐱𝐲SM𝐱𝐲superscript12𝑙2𝑑𝑙1ℙdelimited-[]Po𝑤𝑙𝑙12𝑙2𝑑1ℙdelimited-[]𝑡Po𝑤𝑡𝑙𝑙superscriptsubscript𝑖1superscript1𝑖superscript2𝑙2𝑑𝑖𝑖1ℙdelimited-[]𝑡Po𝑤𝑡𝑙12superscript𝑑13𝑜1superscript𝑑131𝑡𝑙𝔼delimited-[]𝑡Po𝑤2superscript𝑑13𝑜1superscript𝑑131𝑤𝑙𝑡1\displaystyle\begin{split}\frac{\mathrm{SMREG}(\mathbf{x},\mathbf{y})}{\mathrm{SM}(\mathbf{x},\mathbf{y})}\geq(1-\frac{2l-2}{d})^{l}(1-\mathbb{P}[\mathrm{Po}(w)>l])\geq\\
\exp(l\ln(1-\frac{2l-2}{d}))(1-\mathbb{P}[t\mathrm{Po}(w)\geq tl])=\\
\exp\left(l\sum_{i=1}^{\infty}(-1)^{i}\frac{(\frac{2l-2}{d})^{i}}{i}\right)(1-\mathbb{P}[\exp(t\mathrm{Po}(w)-tl)\geq 1])\geq\\
\exp(-\frac{2}{d^{\frac{1}{3}}}+o(\frac{1}{d^{\frac{1}{3}}}))(1-\exp(-tl)\mathbb{E}[\exp(t\mathrm{Po}(w))])=\\
\exp(-\frac{2}{d^{\frac{1}{3}}}+o(\frac{1}{d^{\frac{1}{3}}}))(1-\exp(-w-l(t-1))),\end{split}

(33)

where the last equality is implied by the formula for the Laplace Transform for the Poisson random variable:

𝔼​[exp⁡(t​Po​(w))]=exp⁡(w​(exp⁡(t)−1)).𝔼delimited-[]𝑡Po𝑤𝑤𝑡1\mathbb{E}[\exp(t\mathrm{Po}(w))]=\exp(w(\exp(t)-1)).

(34)

Notice that:
w=‖𝐳‖22=ln⁡(SM​(𝐱,𝐱))+ln⁡(SM​(𝐲,𝐲))+2​ln⁡(SM​(𝐱,𝐲))2≤2​ln⁡(C)𝑤superscriptnorm𝐳22SM𝐱𝐱SM𝐲𝐲2SM𝐱𝐲22𝐶w=\frac{\|\mathbf{z}\|^{2}}{2}=\frac{\ln(\mathrm{SM}(\mathbf{x},\mathbf{x}))+\ln(\mathrm{SM}(\mathbf{y},\mathbf{y}))+2\ln(\mathrm{SM}(\mathbf{x},\mathbf{y}))}{2}\leq 2\ln(C).
We conclude that:

SMREG​(𝐱,𝐲)SM​(𝐱,𝐲)≥(1−2d13+o​(1d13))​(1−C−2​(d132​e⋅ln⁡(C))−d13)=1−2d13+o​(1d13).SMREG𝐱𝐲SM𝐱𝐲12superscript𝑑13𝑜1superscript𝑑131superscript𝐶2superscriptsuperscript𝑑13⋅2𝑒𝐶superscript𝑑1312superscript𝑑13𝑜1superscript𝑑13\frac{\mathrm{SMREG}(\mathbf{x},\mathbf{y})}{\mathrm{SM}(\mathbf{x},\mathbf{y})}\geq(1-\frac{2}{d^{\frac{1}{3}}}+o(\frac{1}{d^{\frac{1}{3}}}))(1-C^{-2}(\frac{d^{\frac{1}{3}}}{2e\cdot\ln(C)})^{-d^{\frac{1}{3}}})=1-\frac{2}{d^{\frac{1}{3}}}+o(\frac{1}{d^{\frac{1}{3}}}).

(35)

That completes the proof.
∎

### F.4 Proofs of Theorem 2,Theorem 3 & Beautiful Functions

We will provide here much more general theoretical results which will imply Theorem 3 and Theorem 2. We need the following definition:

###### Definition 1.

We say that function F:ℝn→ℝ:𝐹→superscriptℝ𝑛ℝF:\mathbb{R}^{n}\rightarrow\mathbb{R} is beautiful if F𝐹F can be expressed as:

FΩ,g​(𝐳)=𝔼ω∼Ω​[g​(ω⊤​𝐳)],subscript𝐹Ω𝑔𝐳subscript𝔼similar-to𝜔Ωdelimited-[]𝑔superscript𝜔top𝐳F_{\Omega,g}(\mathbf{z})=\mathbb{E}_{\omega\sim\Omega}[g(\omega^{\top}\mathbf{z})],

(36)

for a probabilistic isotropic distribution ΩΩ\Omega, and where g:ℝ→ℝ:𝑔→ℝℝg:\mathbb{R}\rightarrow\mathbb{R} is an entire function
with non-negative power-series coefficients
(i.e. g​(x)=∑i=0∞ai​xi𝑔𝑥superscriptsubscript𝑖0subscript𝑎𝑖superscript𝑥𝑖g(x)=\sum_{i=0}^{\infty}a_{i}x^{i} for every x∈ℝ𝑥ℝx\in\mathbb{R} and with ai≥0subscript𝑎𝑖0a_{i}\geq 0 for i=0,1,…𝑖01…i=0,1,...).
In the formula above we assume that the expectation on the RHS exists.

Interestingly, beautiful functions can be used to define softmax and consequently, Gaussian kernels (both standard and regularized), leading to our PRF mechanism presented in the main body of the paper, as we explain below.

###### Remark 1.

If one takes Ω=𝒩​(0,𝐈d)Ω𝒩0subscript𝐈𝑑\Omega=\mathcal{N}(0,\mathbf{I}_{d})(note that 𝒩​(0,𝐈d)𝒩0subscript𝐈𝑑\mathcal{N}(0,\mathbf{I}_{d}) is isotropic) and g:x→exp⁡(x):𝑔→𝑥𝑥g:x\rightarrow\exp(x) (such g𝑔g is clearly entire with nonnegative power-series coefficient) then the following is true for 𝐳=𝐱+𝐲𝐳𝐱𝐲\mathbf{z}=\mathbf{x}+\mathbf{y}:

SM​(𝐱,𝐲)=exp⁡(−‖𝐱‖2+‖𝐲‖22)​FΩ,g​(𝐳).SM𝐱𝐲superscriptnorm𝐱2superscriptnorm𝐲22subscript𝐹Ω𝑔𝐳\mathrm{SM}(\mathbf{x},\mathbf{y})=\exp(-\frac{\|\mathbf{x}\|^{2}+\|\mathbf{y}\|^{2}}{2})F_{\Omega,g}(\mathbf{z}).

(37)

Similarly: SMREG​(𝐱,𝐲)=exp⁡(−‖𝐱‖2+‖𝐲‖22)​FΩreg,g​(𝐳)SMREG𝐱𝐲superscriptnorm𝐱2superscriptnorm𝐲22subscript𝐹subscriptΩreg𝑔𝐳\mathrm{SMREG}(\mathbf{x},\mathbf{y})=\exp(-\frac{\|\mathbf{x}\|^{2}+\|\mathbf{y}\|^{2}}{2})F_{\Omega_{\mathrm{reg}},g}(\mathbf{z}), where ΩregsubscriptΩreg\Omega_{\mathrm{reg}} stands for the distribution corresponding to Haar measure on the sphere of radius d𝑑\sqrt{d} (which is clearly isotropic).
Therefore general concentration results for Monte Carlo estimators of beautiful functions immediately imply corresponding results for the (standard and regularized) softmax (and thus also Gaussian) kernel.

We will consider two estimators of the beautiful functions from Definition 1 that directly lead (through Remark 1) to: PRF-based approximation of the softmax-kernel and its enhanced version with orthogonal features. Standard Monte Carlo estimator samples independently ω1iid,…,ωmiid​∼iid​Ωsuperscriptsubscript𝜔1iid…superscriptsubscript𝜔𝑚iidiidsimilar-toΩ\omega_{1}^{\mathrm{iid}},...,\omega_{m}^{\mathrm{iid}}\overset{\mathrm{iid}}{\sim}\Omega, where m𝑚m stands for the number of samples and then computes:

F^miid​(𝐳)​=def​1m​∑i=1mg​((ωiiid)⊤​𝐳).subscriptsuperscript^𝐹iid𝑚𝐳def1𝑚superscriptsubscript𝑖1𝑚𝑔superscriptsuperscriptsubscript𝜔𝑖iidtop𝐳\widehat{F}^{\mathrm{iid}}_{m}(\mathbf{z})\overset{\mathrm{def}}{=}\frac{1}{m}\sum_{i=1}^{m}g((\omega_{i}^{\mathrm{iid}})^{\top}\mathbf{z}).

(38)

Orthogonal Monte Carlo estimator samples ω1ort,…,ωmortsuperscriptsubscript𝜔1ort…superscriptsubscript𝜔𝑚ort\omega_{1}^{\mathrm{ort}},...,\omega_{m}^{\mathrm{ort}} (m≤d𝑚𝑑m\leq d) in such a way that marginally we have: ωiort∼Ωsimilar-tosuperscriptsubscript𝜔𝑖ortΩ\omega_{i}^{\mathrm{ort}}\sim\Omega, but (ωiort)⊤​ωjort=0superscriptsuperscriptsubscript𝜔𝑖orttopsuperscriptsubscript𝜔𝑗ort0(\omega_{i}^{\mathrm{ort}})^{\top}\omega_{j}^{\mathrm{ort}}=0 for i≠j𝑖𝑗i\neq j (such an orthogonal ensemble can be always created if ΩΩ\Omega is isotropic, as we already mentioned in the main body of the paper). We define:

F^mort​(𝐳)​=def​1m​∑i=1mg​((ωiort)⊤​𝐳).subscriptsuperscript^𝐹ort𝑚𝐳def1𝑚superscriptsubscript𝑖1𝑚𝑔superscriptsuperscriptsubscript𝜔𝑖orttop𝐳\widehat{F}^{\mathrm{ort}}_{m}(\mathbf{z})\overset{\mathrm{def}}{=}\frac{1}{m}\sum_{i=1}^{m}g((\omega_{i}^{\mathrm{ort}})^{\top}\mathbf{z}).

(39)

#### F.4.1 Orthogonality universally improves concentration

Denote by MZ​(θ)=𝔼​[eθ​Z]subscript𝑀𝑍𝜃𝔼delimited-[]superscript𝑒𝜃𝑍M_{Z}(\theta)=\mathbb{E}[e^{\theta Z}] a moment generating function of the random variable Z𝑍Z.
Note first that estimators of beautiful functions based on standard Monte Carlo procedure using independent vectors ωiiidsuperscriptsubscript𝜔𝑖iid\omega_{i}^{\mathrm{iid}} guarantee strong concentration bounds since
independent ωisubscript𝜔𝑖\omega_{i}s provide a way to obtain exponentially small upper bounds on failure probabilities through moment generating functions.
We summarize this classic observation which is a standard application of Markov’s Inequality below.

###### Lemma 4.

Consider an estimator F^miid​(𝐳)subscriptsuperscript^𝐹iid𝑚𝐳\widehat{F}^{\mathrm{iid}}_{m}(\mathbf{z}) of the beautiful function F𝐹F evaluated at 𝐳𝐳\mathbf{z}. Then the following holds for any a>F​(𝐳)𝑎𝐹𝐳a>F(\mathbf{z}), θ>0𝜃0\theta>0:

ℙ​[F^miid​(𝐳)>a]≤exp⁡(θ​m​a)​MX​(θ)m,ℙdelimited-[]subscriptsuperscript^𝐹iid𝑚𝐳𝑎𝜃𝑚𝑎subscript𝑀𝑋superscript𝜃𝑚\mathbb{P}[\widehat{F}^{\mathrm{iid}}_{m}(\mathbf{z})>a]\leq\exp(\theta ma)M_{X}(\theta)^{m},

(40)

where X=g​(𝐰⊤​𝐳)𝑋𝑔superscript𝐰top𝐳X=g(\mathbf{w}^{\top}\mathbf{z}), 𝐰∼𝒟similar-to𝐰𝒟\mathbf{w}\sim\mathcal{D}.

The above result provides us with exponentially small (in Legendre Transform) upper bounds on tail probabilities for the standard estimator.
Below we provide our two main theoretical results.

###### Theorem 5 (orthogonality provides smaller tails).

If FΩ,gsubscript𝐹Ω𝑔F_{\Omega,g} is a beautiful function then the following holds for m≤d𝑚𝑑m\leq d, X𝑋X as in Lemma 4 and any a>F​(𝐳)𝑎𝐹𝐳a>F(\mathbf{z}), θ>0𝜃0\theta>0:

ℙ[F^mort(𝐳))>a]≤exp(−θma)(MX(θ)m−θ4​m​(m−1)4​d2​(d+2)a0M−2a12∥𝐳∥4(𝔼∥ω∥2)2).\mathbb{P}[\widehat{F}^{\mathrm{ort}}_{m}(\mathbf{z}))>a]\leq\exp(-\theta ma)\left(M_{X}(\theta)^{m}-\frac{\theta^{4}m(m-1)}{4d^{2}(d+2)}a_{0}^{M-2}a_{1}^{2}\|\mathbf{z}\|^{4}(\mathbb{E}\|\omega\|^{2})^{2}\right).

(41)

This result shows that features obtained from the ensembles of pairwise orthogonal random vectors provide exponentially small bounds on tail probabilities and that these bounds are strictly better than for estimators using unstructured features. Furthermore, the result is universal, i.e. holds for any dimensionality d𝑑d, not just asymptotically for d𝑑d large enough.

We also obtain similar result regarding mean squared errors (MSEs) of the considered estimators:

###### Theorem 6.

If FΩ,gsubscript𝐹Ω𝑔F_{\Omega,g} is a beautiful function then the following holds for m≤d𝑚𝑑m\leq d:

MSE​(F^mort​(𝐳))≤MSE​(F^miid​(𝐳))−(1−1m)​2d+2​(FΩ,g​(𝐳)−a0)2.MSEsubscriptsuperscript^𝐹ort𝑚𝐳MSEsubscriptsuperscript^𝐹iid𝑚𝐳11𝑚2𝑑2superscriptsubscript𝐹Ω𝑔𝐳subscript𝑎02\mathrm{MSE}(\widehat{F}^{\mathrm{ort}}_{m}(\mathbf{z}))\leq\mathrm{MSE}(\widehat{F}^{\mathrm{iid}}_{m}(\mathbf{z}))-(1-\frac{1}{m})\frac{2}{d+2}\left(F_{\Omega,g}(\mathbf{z})-a_{0}\right)^{2}.

(42)

As before, an orthogonal estimator leads to better concentration results and as before, this is the case for any d>0𝑑0d>0, not only asymptotically for large enough d𝑑d.

Note that from what we have said above, Theorem 2 and Theorem 3 follow immediately from Theorem 6 and Theorem 5 respectively.

Thus in the remainder of this section we will prove Theorem 6 and Theorem 5.

#### F.4.2 Proof of Theorem 5

###### Proof.

Note that by the analogous application of Markov’s Inequality as in Lemma 4, we get:

ℙ[F^mort(𝐳))>a]≤𝔼​[eθ​(X1ort+…+Xmort)]eθ​m​a,\displaystyle\begin{split}\mathbb{P}[\widehat{F}^{\mathrm{ort}}_{m}(\mathbf{z}))>a]\leq\frac{\mathbb{E}[e^{\theta(X_{1}^{\mathrm{ort}}+...+X_{m}^{\mathrm{ort}})}]}{e^{\theta ma}},\end{split}

(43)

where we have:
Xiort=g​((ωiort)⊤​𝐳)superscriptsubscript𝑋𝑖ort𝑔superscriptsuperscriptsubscript𝜔𝑖orttop𝐳X_{i}^{\mathrm{ort}}=g((\omega_{i}^{\mathrm{ort}})^{\top}\mathbf{z}).
We see that it suffices to show that for any θ>0𝜃0\theta>0 the following holds:
𝔼​[eθ​(X1ort+…+Xmort)]<𝔼​[eθ​(X1iid+…+Xmiid)]𝔼delimited-[]superscript𝑒𝜃superscriptsubscript𝑋1ort…superscriptsubscript𝑋𝑚ort𝔼delimited-[]superscript𝑒𝜃superscriptsubscript𝑋1iid…superscriptsubscript𝑋𝑚iid\mathbb{E}[e^{\theta(X_{1}^{\mathrm{ort}}+...+X_{m}^{\mathrm{ort}})}]<\mathbb{E}[e^{\theta(X_{1}^{\mathrm{iid}}+...+X_{m}^{\mathrm{iid}})}].
We have:

𝔼​[eθ​(X1ort+…+Xmort)]=𝔼​[∑j=0∞(θ​∑i=1mXiort)jj!]=𝔼​[∑j=0∞θjj!​(∑i=1mXiort)j]=∑j=0∞θjj!​𝔼​[(∑i=1mXiort)j]=∑j=0∞θjj!​𝔼​[∑(j1,…,jm)∈𝒮j(jj1,…,jm)​(X1ort)j1⋅…⋅(Xmort)jm],𝔼delimited-[]superscript𝑒𝜃superscriptsubscript𝑋1ort…superscriptsubscript𝑋𝑚ort𝔼delimited-[]superscriptsubscript𝑗0superscript𝜃superscriptsubscript𝑖1𝑚superscriptsubscript𝑋𝑖ort𝑗𝑗𝔼delimited-[]superscriptsubscript𝑗0superscript𝜃𝑗𝑗superscriptsuperscriptsubscript𝑖1𝑚subscriptsuperscript𝑋ort𝑖𝑗superscriptsubscript𝑗0superscript𝜃𝑗𝑗𝔼delimited-[]superscriptsuperscriptsubscript𝑖1𝑚subscriptsuperscript𝑋ort𝑖𝑗superscriptsubscript𝑗0superscript𝜃𝑗𝑗𝔼delimited-[]subscriptsubscript𝑗1…subscript𝑗𝑚subscript𝒮𝑗⋅binomial𝑗subscript𝑗1…subscript𝑗𝑚superscriptsuperscriptsubscript𝑋1ortsubscript𝑗1…superscriptsuperscriptsubscript𝑋𝑚ortsubscript𝑗𝑚\displaystyle\begin{split}\mathbb{E}[e^{\theta(X_{1}^{\mathrm{ort}}+...+X_{m}^{\mathrm{ort}})}]=\mathbb{E}[\sum_{j=0}^{\infty}\frac{(\theta\sum_{i=1}^{m}X_{i}^{\mathrm{ort}})^{j}}{j!}]=\mathbb{E}[\sum_{j=0}^{\infty}\frac{\theta^{j}}{j!}(\sum_{i=1}^{m}X^{\mathrm{ort}}_{i})^{j}]=\\
\sum_{j=0}^{\infty}\frac{\theta^{j}}{j!}\mathbb{E}[(\sum_{i=1}^{m}X^{\mathrm{ort}}_{i})^{j}]=\sum_{j=0}^{\infty}\frac{\theta^{j}}{j!}\mathbb{E}[\sum_{(j_{1},...,j_{m})\in\mathcal{S}_{j}}\binom{j}{j_{1},\dots,j_{m}}(X_{1}^{\mathrm{ort}})^{j_{1}}\cdot...\cdot(X_{m}^{\mathrm{ort}})^{j_{m}}],\end{split}

(44)

where 𝒮j={(j1,…,jm)∈ℕ×…×ℕ:j1,…,jm≥0,j1+…+jm=j}subscript𝒮𝑗conditional-setsubscript𝑗1…subscript𝑗𝑚ℕ…ℕformulae-sequencesubscript𝑗1…subscript𝑗𝑚0subscript𝑗1…subscript𝑗𝑚𝑗\mathcal{S}_{j}=\{(j_{1},...,j_{m})\in\mathbb{N}\times...\times\mathbb{N}:j_{1},...,j_{m}\geq 0,j_{1}+...+j_{m}=j\}.

Thus we have:

𝔼​[eθ​(X1ort+…+Xmort)]=∑j=0∞θjj!​∑(j1,…,jm)∈𝒮j(jj1,…,jm)​𝔼​[(X1ort)j1⋅…⋅(Xmort)jm].𝔼delimited-[]superscript𝑒𝜃superscriptsubscript𝑋1ort…superscriptsubscript𝑋𝑚ortsuperscriptsubscript𝑗0superscript𝜃𝑗𝑗subscriptsubscript𝑗1…subscript𝑗𝑚subscript𝒮𝑗binomial𝑗subscript𝑗1…subscript𝑗𝑚𝔼delimited-[]⋅superscriptsuperscriptsubscript𝑋1ortsubscript𝑗1…superscriptsuperscriptsubscript𝑋𝑚ortsubscript𝑗𝑚\mathbb{E}[e^{\theta(X_{1}^{\mathrm{ort}}+...+X_{m}^{\mathrm{ort}})}]=\sum_{j=0}^{\infty}\frac{\theta^{j}}{j!}\sum_{(j_{1},...,j_{m})\in\mathcal{S}_{j}}\binom{j}{j_{1},\dots,j_{m}}\mathbb{E}[(X_{1}^{\mathrm{ort}})^{j_{1}}\cdot...\cdot(X_{m}^{\mathrm{ort}})^{j_{m}}].

(45)

Similarly, we get:

𝔼​[eθ​(X1iid+…+Xmiid)]=∑j=0∞θjj!​∑(j1,…,jm)∈𝒮j(jj1,…,jm)​𝔼​[(X1iid)j1⋅…⋅(Xmiid)jm].𝔼delimited-[]superscript𝑒𝜃superscriptsubscript𝑋1iid…superscriptsubscript𝑋𝑚iidsuperscriptsubscript𝑗0superscript𝜃𝑗𝑗subscriptsubscript𝑗1…subscript𝑗𝑚subscript𝒮𝑗binomial𝑗subscript𝑗1…subscript𝑗𝑚𝔼delimited-[]⋅superscriptsuperscriptsubscript𝑋1iidsubscript𝑗1…superscriptsuperscriptsubscript𝑋𝑚iidsubscript𝑗𝑚\mathbb{E}[e^{\theta(X_{1}^{\mathrm{iid}}+...+X_{m}^{\mathrm{iid}})}]=\sum_{j=0}^{\infty}\frac{\theta^{j}}{j!}\sum_{(j_{1},...,j_{m})\in\mathcal{S}_{j}}\binom{j}{j_{1},\dots,j_{m}}\mathbb{E}[(X_{1}^{\mathrm{iid}})^{j_{1}}\cdot...\cdot(X_{m}^{\mathrm{iid}})^{j_{m}}].

(46)

Therefore we get:

Δ=𝔼​[eθ​(X1iid+…+Xmiid)]−𝔼​[eθ​(X1ort+…+Xmort)]=∑j=0∞θjj!​∑(j1,…,jm)∈𝒮j(jj1,…,jm)​(𝔼​[(X1iid)j1⋅…⋅(Xmiid)jm]−𝔼​[(X1ort)j1⋅…⋅(Xmort)jm])Δ𝔼delimited-[]superscript𝑒𝜃superscriptsubscript𝑋1iid…superscriptsubscript𝑋𝑚iid𝔼delimited-[]superscript𝑒𝜃superscriptsubscript𝑋1ort…superscriptsubscript𝑋𝑚ortsuperscriptsubscript𝑗0superscript𝜃𝑗𝑗subscriptsubscript𝑗1…subscript𝑗𝑚subscript𝒮𝑗binomial𝑗subscript𝑗1…subscript𝑗𝑚𝔼delimited-[]⋅superscriptsuperscriptsubscript𝑋1iidsubscript𝑗1…superscriptsuperscriptsubscript𝑋𝑚iidsubscript𝑗𝑚𝔼delimited-[]⋅superscriptsuperscriptsubscript𝑋1ortsubscript𝑗1…superscriptsuperscriptsubscript𝑋𝑚ortsubscript𝑗𝑚\displaystyle\begin{split}\Delta=\mathbb{E}[e^{\theta(X_{1}^{\mathrm{iid}}+...+X_{m}^{\mathrm{iid}})}]-\mathbb{E}[e^{\theta(X_{1}^{\mathrm{ort}}+...+X_{m}^{\mathrm{ort}})}]\\
=\sum_{j=0}^{\infty}\frac{\theta^{j}}{j!}\sum_{(j_{1},...,j_{m})\in\mathcal{S}_{j}}\binom{j}{j_{1},\dots,j_{m}}\left(\mathbb{E}[(X_{1}^{\mathrm{iid}})^{j_{1}}\cdot...\cdot(X_{m}^{\mathrm{iid}})^{j_{m}}]-\mathbb{E}[(X_{1}^{\mathrm{ort}})^{j_{1}}\cdot...\cdot(X_{m}^{\mathrm{ort}})^{j_{m}}]\right)\end{split}

(47)

Note first that using the fact that f𝑓f is entire, we can rewrite each Xiortsuperscriptsubscript𝑋𝑖ortX_{i}^{\mathrm{ort}} as:

Xiort=∑s=0∞as​((ωiort)⊤​𝐳)s,superscriptsubscript𝑋𝑖ortsuperscriptsubscript𝑠0subscript𝑎𝑠superscriptsuperscriptsuperscriptsubscript𝜔𝑖orttop𝐳𝑠X_{i}^{\mathrm{ort}}=\sum_{s=0}^{\infty}a_{s}((\omega_{i}^{\mathrm{ort}})^{\top}\mathbf{z})^{s},

(48)

where f​(x)=∑s=0∞as​xs𝑓𝑥superscriptsubscript𝑠0subscript𝑎𝑠superscript𝑥𝑠f(x)=\sum_{s=0}^{\infty}a_{s}x^{s}
and a0,a1,…≥0subscript𝑎0subscript𝑎1…0a_{0},a_{1},...\geq 0.
Similarly,

Xiiid=∑s=0∞as​((ωiiid)⊤​𝐳)s.superscriptsubscript𝑋𝑖iidsuperscriptsubscript𝑠0subscript𝑎𝑠superscriptsuperscriptsuperscriptsubscript𝜔𝑖iidtop𝐳𝑠X_{i}^{\mathrm{iid}}=\sum_{s=0}^{\infty}a_{s}((\omega_{i}^{\mathrm{iid}})^{\top}\mathbf{z})^{s}.

(49)

By plugging in the above formulae for Xiortsuperscriptsubscript𝑋𝑖ortX_{i}^{\mathrm{ort}} and Xiiidsuperscriptsubscript𝑋𝑖iidX_{i}^{\mathrm{iid}} int the formula for ΔΔ\Delta and expanding power-expressions, we obtain:

Δ=∑j=0∞θjj!​∑(j1,…,jm)∈𝒮j(jj1,…,jm)​∑(d1,…,dm)∈𝒟​(j1,…,jm)c^j1,…,jm​(d1,…,dm)​Δ^​(d1,…,dm),Δsuperscriptsubscript𝑗0superscript𝜃𝑗𝑗subscriptsubscript𝑗1…subscript𝑗𝑚subscript𝒮𝑗binomial𝑗subscript𝑗1…subscript𝑗𝑚subscriptsubscript𝑑1…subscript𝑑𝑚𝒟subscript𝑗1…subscript𝑗𝑚subscript^𝑐subscript𝑗1…subscript𝑗𝑚subscript𝑑1…subscript𝑑𝑚^Δsubscript𝑑1…subscript𝑑𝑚\displaystyle\begin{split}\Delta=\sum_{j=0}^{\infty}\frac{\theta^{j}}{j!}\sum_{(j_{1},...,j_{m})\in\mathcal{S}_{j}}\binom{j}{j_{1},\dots,j_{m}}\sum_{(d_{1},...,d_{m})\in\mathcal{D}(j_{1},...,j_{m})}\widehat{c}_{j_{1},\dots,j_{m}}(d_{1},\dots,d_{m})\widehat{\Delta}(d_{1},...,d_{m}),\end{split}

(50)

for some ordered subsets of indices (with potentially repeating entries) 𝒟​(j1,…,jm)𝒟subscript𝑗1…subscript𝑗𝑚\mathcal{D}(j_{1},...,j_{m}) and some nonnegative c^j1,…,jm​(d1,…,dm)subscript^𝑐subscript𝑗1…subscript𝑗𝑚subscript𝑑1…subscript𝑑𝑚\widehat{c}_{j_{1},\dots,j_{m}}(d_{1},\dots,d_{m}) (exact formula for those can be given but we do not need it to complete the proof and since it is technical, it would unnecessarily complicate the proof so we skip it)
and Δ^​(d1,…,dm)^Δsubscript𝑑1…subscript𝑑𝑚\widehat{\Delta}(d_{1},...,d_{m}) defined as:

Δ^​(d1,…,dm)=𝔼​[((ω1iid)⊤​𝐳)d1⋅…⋅((ωmiid)⊤​𝐳)dm]−𝔼​[((ω1ort)⊤​𝐳)d1⋅…⋅((ωmort)⊤​𝐳)dm].^Δsubscript𝑑1…subscript𝑑𝑚𝔼delimited-[]⋅superscriptsuperscriptsuperscriptsubscript𝜔1iidtop𝐳subscript𝑑1…superscriptsuperscriptsuperscriptsubscript𝜔𝑚iidtop𝐳subscript𝑑𝑚𝔼delimited-[]⋅superscriptsuperscriptsuperscriptsubscript𝜔1orttop𝐳subscript𝑑1…superscriptsuperscriptsuperscriptsubscript𝜔𝑚orttop𝐳subscript𝑑𝑚\displaystyle\begin{split}\widehat{\Delta}(d_{1},...,d_{m})=\mathbb{E}[((\omega_{1}^{\mathrm{iid}})^{\top}\mathbf{z})^{d_{1}}\cdot...\cdot((\omega_{m}^{\mathrm{iid}})^{\top}\mathbf{z})^{d_{m}}]-\mathbb{E}[((\omega_{1}^{\mathrm{ort}})^{\top}\mathbf{z})^{d_{1}}\cdot...\cdot((\omega_{m}^{\mathrm{ort}})^{\top}\mathbf{z})^{d_{m}}].\end{split}

(51)

Our next goal is to re-write the formula for Δ^​(d1,…,dm)^Δsubscript𝑑1…subscript𝑑𝑚\widehat{\Delta}(d_{1},...,d_{m}). Denote:

Y=((ω1ort)⊤​𝐳)d1⋅…⋅((ωmort)⊤​𝐳)dm.𝑌⋅superscriptsuperscriptsuperscriptsubscript𝜔1orttop𝐳subscript𝑑1…superscriptsuperscriptsuperscriptsubscript𝜔𝑚orttop𝐳subscript𝑑𝑚Y=((\omega_{1}^{\mathrm{ort}})^{\top}\mathbf{z})^{d_{1}}\cdot...\cdot((\omega_{m}^{\mathrm{ort}})^{\top}\mathbf{z})^{d_{m}}.

(52)

Observe that Y𝑌Y has the same distribution as Y′superscript𝑌′Y^{\prime} defined as:

Y′=(𝐞1⊤​𝐠‖𝐠‖2​‖𝐳‖2)d1⋅…⋅(𝐞m⊤​𝐠‖𝐠‖2​‖𝐳‖2)dm⋅(‖ω1ort‖2)d1⋅…⋅(‖ωmort‖2)dm,superscript𝑌′⋅superscriptsuperscriptsubscript𝐞1top𝐠subscriptnorm𝐠2subscriptnorm𝐳2subscript𝑑1…superscriptsuperscriptsubscript𝐞𝑚top𝐠subscriptnorm𝐠2subscriptnorm𝐳2subscript𝑑𝑚superscriptsubscriptnormsuperscriptsubscript𝜔1ort2subscript𝑑1…superscriptsubscriptnormsuperscriptsubscript𝜔𝑚ort2subscript𝑑𝑚Y^{\prime}=(\mathbf{e}_{1}^{\top}\frac{\mathbf{g}}{\|\mathbf{g}\|_{2}}\|\mathbf{z}\|_{2})^{d_{1}}\cdot...\cdot(\mathbf{e}_{m}^{\top}\frac{\mathbf{g}}{\|\mathbf{g}\|_{2}}\|\mathbf{z}\|_{2})^{d_{m}}\cdot(\|\omega_{1}^{\mathrm{ort}}\|_{2})^{d_{1}}\cdot...\cdot(\|\omega_{m}^{\mathrm{ort}}\|_{2})^{d_{m}},

(53)

where 𝐠𝐠\mathbf{g} is a Gaussian vector taken from the 𝒩​(0,𝐈d)𝒩0subscript𝐈𝑑\mathcal{N}(0,\mathbf{I}_{d}) distribution, independently from: ‖ω1ort‖2,…,‖ωmort‖2subscriptnormsuperscriptsubscript𝜔1ort2…subscriptnormsuperscriptsubscript𝜔𝑚ort2\|\omega_{1}^{\mathrm{ort}}\|_{2},...,\|\omega_{m}^{\mathrm{ort}}\|_{2}.

This comes from the fact that for a fixed 𝐳𝐳\mathbf{z} one can think about the set:
ω1ort‖ω1ort‖2,…,ωmort‖ωmort‖2superscriptsubscript𝜔1ortsubscriptnormsuperscriptsubscript𝜔1ort2…superscriptsubscript𝜔𝑚ortsubscriptnormsuperscriptsubscript𝜔𝑚ort2\frac{\omega_{1}^{\mathrm{ort}}}{\|\omega_{1}^{\mathrm{ort}}\|_{2}},...,\frac{\omega_{m}^{\mathrm{ort}}}{\|\omega_{m}^{\mathrm{ort}}\|_{2}} as a random rotation of the system of m𝑚m canonical basis vectors: 𝐞1,…,𝐞msubscript𝐞1…subscript𝐞𝑚\mathbf{e}_{1},...,\mathbf{e}_{m}.
Thus instead of applying a random rotation to: 𝐞1,…,𝐞msubscript𝐞1…subscript𝐞𝑚\mathbf{e}_{1},...,\mathbf{e}_{m}, one can equivalently randomly rotate vector 𝐳𝐳\mathbf{z}. Randomly rotated vector 𝐳𝐳\mathbf{z} has the same distribution as: 𝐠‖𝐠‖2​‖𝐳‖2𝐠subscriptnorm𝐠2subscriptnorm𝐳2\frac{\mathbf{g}}{\|\mathbf{g}\|_{2}}\|\mathbf{z}\|_{2}.

Now note that lengths of vectors ω1ort,…,ωmortsuperscriptsubscript𝜔1ort…superscriptsubscript𝜔𝑚ort\omega_{1}^{\mathrm{ort}},...,\omega_{m}^{\mathrm{ort}} are chosen independently.

Therefore we obtain:

𝔼​[((ω1ort)⊤​𝐳)d1⋅…⋅((ωmort)⊤​𝐳)dm]=𝔼​[(‖ω1ort‖2)d1]⋅…⋅𝔼​[(‖ωmort‖2)dm]⋅𝔼​[(𝐞1⊤​𝐯)d1⋅…⋅(𝐞m⊤​𝐯)dm]​‖𝐳‖2d1+…+dm,𝔼delimited-[]⋅superscriptsuperscriptsuperscriptsubscript𝜔1orttop𝐳subscript𝑑1…superscriptsuperscriptsuperscriptsubscript𝜔𝑚orttop𝐳subscript𝑑𝑚⋅⋅𝔼delimited-[]superscriptsubscriptdelimited-∥∥superscriptsubscript𝜔1ort2subscript𝑑1…𝔼delimited-[]superscriptsubscriptdelimited-∥∥superscriptsubscript𝜔𝑚ort2subscript𝑑𝑚𝔼delimited-[]⋅superscriptsuperscriptsubscript𝐞1top𝐯subscript𝑑1…superscriptsuperscriptsubscript𝐞𝑚top𝐯subscript𝑑𝑚superscriptsubscriptdelimited-∥∥𝐳2subscript𝑑1…subscript𝑑𝑚\displaystyle\begin{split}\mathbb{E}[((\omega_{1}^{\mathrm{ort}})^{\top}\mathbf{z})^{d_{1}}\cdot...\cdot((\omega_{m}^{\mathrm{ort}})^{\top}\mathbf{z})^{d_{m}}]=\\
\mathbb{E}[(\|\omega_{1}^{\mathrm{ort}}\|_{2})^{d_{1}}]\cdot...\cdot\mathbb{E}[(\|\omega_{m}^{\mathrm{ort}}\|_{2})^{d_{m}}]\cdot\mathbb{E}[(\mathbf{e}_{1}^{\top}\mathbf{v})^{d_{1}}\cdot...\cdot(\mathbf{e}_{m}^{\top}\mathbf{v})^{d_{m}}]\|\mathbf{z}\|_{2}^{d_{1}+...+d_{m}},\end{split}

(54)

where 𝐯∼𝐠‖𝐠‖2similar-to𝐯𝐠subscriptnorm𝐠2\mathbf{v}\sim\frac{\mathbf{g}}{\|\mathbf{g}\|_{2}}.

Denote 𝐠=(g1,…,gd)⊤𝐠superscriptsubscript𝑔1…subscript𝑔𝑑top\mathbf{g}=(g_{1},...,g_{d})^{\top}.
Thus we obtain:

𝔼​[((ω1ort)⊤​𝐳)d1⋅…⋅((ωmort)⊤​𝐳)dm]=𝔼​[(‖ω1ort‖2)d1]⋅…⋅𝔼​[(‖ωmort‖2)dm]⋅‖𝐳‖2d1+…+dm​𝔼​[g1d1⋅…⁣⋅​gmdmg12+…+gd2d1+…+dm]𝔼delimited-[]⋅superscriptsuperscriptsuperscriptsubscript𝜔1orttop𝐳subscript𝑑1…superscriptsuperscriptsuperscriptsubscript𝜔𝑚orttop𝐳subscript𝑑𝑚⋅⋅𝔼delimited-[]superscriptsubscriptdelimited-∥∥superscriptsubscript𝜔1ort2subscript𝑑1…𝔼delimited-[]superscriptsubscriptdelimited-∥∥superscriptsubscript𝜔𝑚ort2subscript𝑑𝑚superscriptsubscriptdelimited-∥∥𝐳2subscript𝑑1…subscript𝑑𝑚𝔼delimited-[]superscriptsubscript𝑔1⋅subscript𝑑1…⋅superscriptsubscript𝑔𝑚subscript𝑑𝑚superscriptsuperscriptsubscript𝑔12…superscriptsubscript𝑔𝑑2subscript𝑑1…subscript𝑑𝑚\displaystyle\begin{split}\mathbb{E}[((\omega_{1}^{\mathrm{ort}})^{\top}\mathbf{z})^{d_{1}}\cdot...\cdot((\omega_{m}^{\mathrm{ort}})^{\top}\mathbf{z})^{d_{m}}]=\\
\mathbb{E}[(\|\omega_{1}^{\mathrm{ort}}\|_{2})^{d_{1}}]\cdot...\cdot\mathbb{E}[(\|\omega_{m}^{\mathrm{ort}}\|_{2})^{d_{m}}]\cdot\|\mathbf{z}\|_{2}^{d_{1}+...+d_{m}}\mathbb{E}[\frac{g_{1}^{d_{1}\cdot...\cdot}g_{m}^{d_{m}}}{\sqrt{g_{1}^{2}+...+g_{d}^{2}}^{d_{1}+...+d_{m}}}]\end{split}

(55)

Now let us focus on the second expression from the formula on Δ^​(d1,…,dm)^Δsubscript𝑑1…subscript𝑑𝑚\widehat{\Delta}(d_{1},...,d_{m}). We have:

𝔼​[((ω1iid)⊤​𝐳)d1⋅…⋅((ωmiid)⊤​𝐳)dm]=∏i=1m𝔼​[((ωiiid)⊤​𝐳)di]=𝔼​[(‖ω1iid‖2)d1]⋅…⋅𝔼​[(‖ωmiid‖2)dm]⋅‖𝐳‖2d1+…+dm⋅∏i=1m𝔼​[gidig12+…+gd2di],𝔼delimited-[]⋅superscriptsuperscriptsuperscriptsubscript𝜔1iidtop𝐳subscript𝑑1…superscriptsuperscriptsuperscriptsubscript𝜔𝑚iidtop𝐳subscript𝑑𝑚superscriptsubscriptproduct𝑖1𝑚𝔼delimited-[]superscriptsuperscriptsuperscriptsubscript𝜔𝑖iidtop𝐳subscript𝑑𝑖⋅⋅𝔼delimited-[]superscriptsubscriptdelimited-∥∥superscriptsubscript𝜔1iid2subscript𝑑1…𝔼delimited-[]superscriptsubscriptdelimited-∥∥superscriptsubscript𝜔𝑚iid2subscript𝑑𝑚superscriptsubscriptdelimited-∥∥𝐳2subscript𝑑1…subscript𝑑𝑚superscriptsubscriptproduct𝑖1𝑚𝔼delimited-[]superscriptsubscript𝑔𝑖subscript𝑑𝑖superscriptsuperscriptsubscript𝑔12…superscriptsubscript𝑔𝑑2subscript𝑑𝑖\displaystyle\begin{split}\mathbb{E}[((\omega_{1}^{\mathrm{iid}})^{\top}\mathbf{z})^{d_{1}}\cdot...\cdot((\omega_{m}^{\mathrm{iid}})^{\top}\mathbf{z})^{d_{m}}]=\prod_{i=1}^{m}\mathbb{E}[((\omega_{i}^{\mathrm{iid}})^{\top}\mathbf{z})^{d_{i}}]=\\
\mathbb{E}[(\|\omega_{1}^{\mathrm{iid}}\|_{2})^{d_{1}}]\cdot...\cdot\mathbb{E}[(\|\omega_{m}^{\mathrm{iid}}\|_{2})^{d_{m}}]\cdot\|\mathbf{z}\|_{2}^{d_{1}+...+d_{m}}\cdot\prod_{i=1}^{m}\mathbb{E}[\frac{g_{i}^{d_{i}}}{\sqrt{g_{1}^{2}+...+g_{d}^{2}}^{d_{i}}}],\end{split}

(56)

where the first equality comes from the fact that
different ωiiidsuperscriptsubscript𝜔𝑖iid\omega_{i}^{\mathrm{iid}}s are independent and the second one is implied by the analogous analysis to the one conducted above.

We will need the following lemma:

###### Lemma 5.

For every s∈ℕ+𝑠subscriptℕs\in\mathbb{N}_{+} such that s≤n𝑠𝑛s\leq n and every k1,…,ks∈ℕ+subscript𝑘1…subscript𝑘𝑠subscriptℕk_{1},...,k_{s}\in\mathbb{N}_{+} the following holds:

𝔼​[g1k1⋅…⋅gsksg12+…+gd2k1+…+ks]=∏i=1s𝔼​[giki]𝔼​[g12+…+gd2k1+…+ks].𝔼delimited-[]⋅superscriptsubscript𝑔1subscript𝑘1…superscriptsubscript𝑔𝑠subscript𝑘𝑠superscriptsuperscriptsubscript𝑔12…superscriptsubscript𝑔𝑑2subscript𝑘1…subscript𝑘𝑠superscriptsubscriptproduct𝑖1𝑠𝔼delimited-[]superscriptsubscript𝑔𝑖subscript𝑘𝑖𝔼delimited-[]superscriptsuperscriptsubscript𝑔12…superscriptsubscript𝑔𝑑2subscript𝑘1…subscript𝑘𝑠\mathbb{E}[\frac{g_{1}^{k_{1}}\cdot...\cdot g_{s}^{k_{s}}}{\sqrt{g_{1}^{2}+...+g_{d}^{2}}^{k_{1}+...+k_{s}}}]=\frac{\prod_{i=1}^{s}\mathbb{E}[g_{i}^{k_{i}}]}{\mathbb{E}[\sqrt{g_{1}^{2}+...+g_{d}^{2}}^{k_{1}+...+k_{s}}]}.

(57)

###### Proof.

Take 𝐫=𝐠‖𝐠‖2​‖𝐠~‖2𝐫𝐠subscriptnorm𝐠2subscriptnorm~𝐠2\mathbf{r}=\frac{\mathbf{g}}{\|\mathbf{g}\|_{2}}\|\tilde{\mathbf{g}}\|_{2}, where 𝐠~~𝐠\tilde{\mathbf{g}} is an independent copy of 𝐠𝐠\mathbf{g}. Note that 𝐫∼𝐠similar-to𝐫𝐠\mathbf{r}\sim\mathbf{g}.
We have:

𝔼​[r1k1]⋅…⋅𝔼​[rsks]=𝔼​[r1k1⋅…⋅rsks]=𝔼​[g1k1⋅…⋅gsksg12+…+gd2k1+…+ks]⋅𝔼​[‖𝐠~‖2k1+…+ks],⋅𝔼delimited-[]superscriptsubscript𝑟1subscript𝑘1…𝔼delimited-[]superscriptsubscript𝑟𝑠subscript𝑘𝑠𝔼delimited-[]⋅superscriptsubscript𝑟1subscript𝑘1…superscriptsubscript𝑟𝑠subscript𝑘𝑠⋅𝔼delimited-[]⋅superscriptsubscript𝑔1subscript𝑘1…superscriptsubscript𝑔𝑠subscript𝑘𝑠superscriptsuperscriptsubscript𝑔12…superscriptsubscript𝑔𝑑2subscript𝑘1…subscript𝑘𝑠𝔼delimited-[]superscriptsubscriptdelimited-∥∥~𝐠2subscript𝑘1…subscript𝑘𝑠\displaystyle\begin{split}\mathbb{E}[r_{1}^{k_{1}}]\cdot...\cdot\mathbb{E}[r_{s}^{k_{s}}]=\mathbb{E}[r_{1}^{k_{1}}\cdot...\cdot r_{s}^{k_{s}}]=\mathbb{E}[\frac{g_{1}^{k_{1}}\cdot...\cdot g_{s}^{k_{s}}}{\sqrt{g_{1}^{2}+...+g_{d}^{2}}^{k_{1}+...+k_{s}}}]\cdot\mathbb{E}[\|\tilde{\mathbf{g}}\|_{2}^{k_{1}+...+k_{s}}],\end{split}

(58)

where the first equality comes from the independence of different elements of 𝐳=(z1,…,zn)⊤𝐳superscriptsubscript𝑧1…subscript𝑧𝑛top\mathbf{z}=(z_{1},...,z_{n})^{\top}
and the second equality is implied by the fact that 𝐠~~𝐠\tilde{\mathbf{g}} is independent from 𝐠𝐠\mathbf{g}.

Therefore we have:

𝔼​[g1k1⋅…⋅gsksg12+…+gd2k1+…+ks]=𝔼​[r1k1]⋅…⋅𝔼​[rsks]𝔼​[‖𝐠~‖2k1+…+ks].𝔼delimited-[]⋅superscriptsubscript𝑔1subscript𝑘1…superscriptsubscript𝑔𝑠subscript𝑘𝑠superscriptsuperscriptsubscript𝑔12…superscriptsubscript𝑔𝑑2subscript𝑘1…subscript𝑘𝑠⋅𝔼delimited-[]superscriptsubscript𝑟1subscript𝑘1…𝔼delimited-[]superscriptsubscript𝑟𝑠subscript𝑘𝑠𝔼delimited-[]superscriptsubscriptnorm~𝐠2subscript𝑘1…subscript𝑘𝑠\mathbb{E}[\frac{g_{1}^{k_{1}}\cdot...\cdot g_{s}^{k_{s}}}{\sqrt{g_{1}^{2}+...+g_{d}^{2}}^{k_{1}+...+k_{s}}}]=\frac{\mathbb{E}[r_{1}^{k_{1}}]\cdot...\cdot\mathbb{E}[r_{s}^{k_{s}}]}{\mathbb{E}[\|\tilde{\mathbf{g}}\|_{2}^{k_{1}+...+k_{s}}]}.

(59)

That completes the proof since 𝐳∼𝐠similar-to𝐳𝐠\mathbf{z}\sim\mathbf{g} and 𝐠~∼𝐠similar-to~𝐠𝐠\tilde{\mathbf{g}}\sim\mathbf{g}.
∎

Note that by Lemma 5, we can rewrite the right expression from the formula on
Δ^​(d1,…,dm)^Δsubscript𝑑1…subscript𝑑𝑚\widehat{\Delta}(d_{1},...,d_{m})
as:

𝔼​[(‖ω1ort‖2)d1]⋅…⋅𝔼​[(‖ωmort‖2)dm]⋅‖𝐳‖2d1+…+dm​∏i=1m𝔼​[gidi]𝔼​[g12+…+gd2d1+…+dm].⋅⋅𝔼delimited-[]superscriptsubscriptnormsuperscriptsubscript𝜔1ort2subscript𝑑1…𝔼delimited-[]superscriptsubscriptnormsuperscriptsubscript𝜔𝑚ort2subscript𝑑𝑚superscriptsubscriptnorm𝐳2subscript𝑑1…subscript𝑑𝑚superscriptsubscriptproduct𝑖1𝑚𝔼delimited-[]superscriptsubscript𝑔𝑖subscript𝑑𝑖𝔼delimited-[]superscriptsuperscriptsubscript𝑔12…superscriptsubscript𝑔𝑑2subscript𝑑1…subscript𝑑𝑚\mathbb{E}[(\|\omega_{1}^{\mathrm{ort}}\|_{2})^{d_{1}}]\cdot...\cdot\mathbb{E}[(\|\omega_{m}^{\mathrm{ort}}\|_{2})^{d_{m}}]\cdot\\
\|\mathbf{z}\|_{2}^{d_{1}+...+d_{m}}\frac{\prod_{i=1}^{m}\mathbb{E}[g_{i}^{d_{i}}]}{\mathbb{E}[\sqrt{g_{1}^{2}+...+g_{d}^{2}}^{d_{1}+...+d_{m}}]}.

(60)

The left expression from the formula on
Δ^​(d1,…,dm)^Δsubscript𝑑1…subscript𝑑𝑚\widehat{\Delta}(d_{1},...,d_{m}) can be rewritten as:

L​(d1,…,dm)=𝔼​[(‖ω1iid‖2)d1]⋅…⋅𝔼​[(‖ωmiid‖2)dm]⋅‖𝐳‖2d1+…+dm∏i=1m𝔼​[gidi]𝔼​[g12+…+gd2d1]⋅…⋅𝔼​[g12+…+gd2dm].𝐿subscript𝑑1…subscript𝑑𝑚⋅⋅𝔼delimited-[]superscriptsubscriptdelimited-∥∥superscriptsubscript𝜔1iid2subscript𝑑1…𝔼delimited-[]superscriptsubscriptdelimited-∥∥superscriptsubscript𝜔𝑚iid2subscript𝑑𝑚superscriptsubscriptdelimited-∥∥𝐳2subscript𝑑1…subscript𝑑𝑚superscriptsubscriptproduct𝑖1𝑚𝔼delimited-[]superscriptsubscript𝑔𝑖subscript𝑑𝑖⋅𝔼delimited-[]superscriptsuperscriptsubscript𝑔12…superscriptsubscript𝑔𝑑2subscript𝑑1…𝔼delimited-[]superscriptsuperscriptsubscript𝑔12…superscriptsubscript𝑔𝑑2subscript𝑑𝑚\displaystyle\begin{split}L(d_{1},...,d_{m})=\mathbb{E}[(\|\omega_{1}^{\mathrm{iid}}\|_{2})^{d_{1}}]\cdot...\cdot\mathbb{E}[(\|\omega_{m}^{\mathrm{iid}}\|_{2})^{d_{m}}]\cdot\|\mathbf{z}\|_{2}^{d_{1}+...+d_{m}}\\
\frac{\prod_{i=1}^{m}\mathbb{E}[g_{i}^{d_{i}}]}{\mathbb{E}[\sqrt{g_{1}^{2}+...+g_{d}^{2}}^{d_{1}}]\cdot...\cdot\mathbb{E}[\sqrt{g_{1}^{2}+...+g_{d}^{2}}^{d_{m}}]}.\end{split}

(61)

Since marginal distributions of ωiortsuperscriptsubscript𝜔𝑖ort\omega_{i}^{\mathrm{ort}} and ωiiidsuperscriptsubscript𝜔𝑖iid\omega_{i}^{\mathrm{iid}} are the same, we can rewrite Δ^​(d1,…,dn)^Δsubscript𝑑1…subscript𝑑𝑛\widehat{\Delta}(d_{1},...,d_{n}) as:

Δ^​(d1,…,dm)=L​(d1,…,dm)​(1−τ​(d1,…,dm)),^Δsubscript𝑑1…subscript𝑑𝑚𝐿subscript𝑑1…subscript𝑑𝑚1𝜏subscript𝑑1…subscript𝑑𝑚\widehat{\Delta}(d_{1},...,d_{m})=L(d_{1},...,d_{m})(1-\tau(d_{1},...,d_{m})),

(62)

where τ​(d1,…,dm)𝜏subscript𝑑1…subscript𝑑𝑚\tau(d_{1},...,d_{m}) is defined as:

τ​(d1,…,dm)=𝔼​[g12+…+gd2d1]⋅…⋅𝔼​[g12+…+gd2dm]𝔼​[g12+…+gd2d1+…+dm]𝜏subscript𝑑1…subscript𝑑𝑚⋅𝔼delimited-[]superscriptsuperscriptsubscript𝑔12…superscriptsubscript𝑔𝑑2subscript𝑑1…𝔼delimited-[]superscriptsuperscriptsubscript𝑔12…superscriptsubscript𝑔𝑑2subscript𝑑𝑚𝔼delimited-[]superscriptsuperscriptsubscript𝑔12…superscriptsubscript𝑔𝑑2subscript𝑑1…subscript𝑑𝑚\tau(d_{1},...,d_{m})=\frac{\mathbb{E}[\sqrt{g_{1}^{2}+...+g_{d}^{2}}^{d_{1}}]\cdot...\cdot\mathbb{E}[\sqrt{g_{1}^{2}+...+g_{d}^{2}}^{d_{m}}]}{\mathbb{E}[\sqrt{g_{1}^{2}+...+g_{d}^{2}}^{d_{1}+...+d_{m}}]}

(63)

We need now few observations regarding Δ^​(d1,…,dm)^Δsubscript𝑑1…subscript𝑑𝑚\widehat{\Delta}(d_{1},...,d_{m}).
Note firsr that since odd moments of the Gaussian scalar distribution 𝒩​(0,1)𝒩01\mathcal{N}(0,1) are zero, Δ^​(d1,…,dm)^Δsubscript𝑑1…subscript𝑑𝑚\widehat{\Delta}(d_{1},...,d_{m}) is zero if at least of of disubscript𝑑𝑖d_{i} is odd. Furthermore, Δ​(d1,…,dm)^^Δsubscript𝑑1…subscript𝑑𝑚\widehat{\Delta(d_{1},...,d_{m})} is trivially zero if all but at most one disubscript𝑑𝑖d_{i} are zero.

With our new notation, ΔΔ\Delta can be rewritten as:

Δ=∑j=0∞θjj!​∑(j1,…,jm)∈𝒮j(jj1,…,jm)​∑(d1,…,dm)∈𝒟​(j1,…,jm)c^j1,…,jm​(d1,…,dm)Δsuperscriptsubscript𝑗0superscript𝜃𝑗𝑗subscriptsubscript𝑗1…subscript𝑗𝑚subscript𝒮𝑗binomial𝑗subscript𝑗1…subscript𝑗𝑚subscriptsubscript𝑑1…subscript𝑑𝑚𝒟subscript𝑗1…subscript𝑗𝑚subscript^𝑐subscript𝑗1…subscript𝑗𝑚subscript𝑑1…subscript𝑑𝑚\displaystyle\Delta=\sum_{j=0}^{\infty}\frac{\theta^{j}}{j!}\sum_{(j_{1},...,j_{m})\in\mathcal{S}_{j}}\binom{j}{j_{1},\dots,j_{m}}\sum_{(d_{1},...,d_{m})\in\mathcal{D}(j_{1},...,j_{m})}\widehat{c}_{j_{1},\dots,j_{m}}(d_{1},\dots,d_{m})

×L​(d1,…,dm)​(1−τ​(d1,…,dm)),absent𝐿subscript𝑑1…subscript𝑑𝑚1𝜏subscript𝑑1…subscript𝑑𝑚\displaystyle\times L(d_{1},...,d_{m})(1-\tau(d_{1},...,d_{m})),

Note also that we have:

eθ​(X1iid+…+Xmiid)=∑j=0∞θjj!​∑(j1,…,jm)∈𝒮j(jj1,…,jm)​∑(d1,…,dm)∈𝒟​(j1,…,jm)c^j1,…,jm​(d1,…,dm)superscript𝑒𝜃superscriptsubscript𝑋1iid…superscriptsubscript𝑋𝑚iidsuperscriptsubscript𝑗0superscript𝜃𝑗𝑗subscriptsubscript𝑗1…subscript𝑗𝑚subscript𝒮𝑗binomial𝑗subscript𝑗1…subscript𝑗𝑚subscriptsubscript𝑑1…subscript𝑑𝑚𝒟subscript𝑗1…subscript𝑗𝑚subscript^𝑐subscript𝑗1…subscript𝑗𝑚subscript𝑑1…subscript𝑑𝑚\displaystyle e^{\theta(X_{1}^{\mathrm{iid}}+...+X_{m}^{\mathrm{iid}})}=\sum_{j=0}^{\infty}\frac{\theta^{j}}{j!}\sum_{(j_{1},...,j_{m})\in\mathcal{S}_{j}}\binom{j}{j_{1},\dots,j_{m}}\sum_{(d_{1},...,d_{m})\in\mathcal{D}(j_{1},...,j_{m})}\widehat{c}_{j_{1},\dots,j_{m}}(d_{1},\dots,d_{m})

×L​(d1,…,dm).absent𝐿subscript𝑑1…subscript𝑑𝑚\displaystyle\times L(d_{1},...,d_{m}).

Therefore (see: our observations on Δ^​(d1,…,dm)^Δsubscript𝑑1…subscript𝑑𝑚\widehat{\Delta}(d_{1},...,d_{m})) to complete the proof it suffices to show that: τ​(d1,…,dm)≤dd+2𝜏subscript𝑑1…subscript𝑑𝑚𝑑𝑑2\tau(d_{1},...,d_{m})\leq\frac{d}{d+2} if at least two: disubscript𝑑𝑖d_{i}, djsubscript𝑑𝑗d_{j} for i≠j𝑖𝑗i\neq j are nonzero and all disubscript𝑑𝑖d_{i} are even.

###### Lemma 6.

The following holds if for some i≠j𝑖𝑗i\neq j we have: di,dj>0subscript𝑑𝑖subscript𝑑𝑗0d_{i},d_{j}>0 and all disubscript𝑑𝑖d_{i} are even:

τ​(d1,…,dm)≤dd+2.𝜏subscript𝑑1…subscript𝑑𝑚𝑑𝑑2\tau(d_{1},...,d_{m})\leq\frac{d}{d+2}.

(64)

###### Proof.

Note that τ​(d1,…,dm)𝜏subscript𝑑1…subscript𝑑𝑚\tau(d_{1},...,d_{m}) can be rewritten as:

τ​(d1,…,dm)=∏i=1mμd​(di)μd​(∑i=1mdi),𝜏subscript𝑑1…subscript𝑑𝑚superscriptsubscriptproduct𝑖1𝑚subscript𝜇𝑑subscript𝑑𝑖subscript𝜇𝑑superscriptsubscript𝑖1𝑚subscript𝑑𝑖\tau(d_{1},...,d_{m})=\frac{\prod_{i=1}^{m}\mu_{d}(d_{i})}{\mu_{d}(\sum_{i=1}^{m}d_{i})},

(65)

where μd​(j)subscript𝜇𝑑𝑗\mu_{d}(j) stands for the jt​hsuperscript𝑗𝑡ℎj^{th} moment of the χ𝜒\chi-distribution with d𝑑d degrees of freedom.
Note that μd​(j)=2j2​Γ​(d+j2)Γ​(d2)subscript𝜇𝑑𝑗superscript2𝑗2Γ𝑑𝑗2Γ𝑑2\mu_{d}(j)=2^{\frac{j}{2}}\frac{\Gamma(\frac{d+j}{2})}{\Gamma(\frac{d}{2})},
where ΓΓ\Gamma is the so-called Gamma-function.

Using the fact that: Γ​(n)=(n−1)!Γ𝑛𝑛1\Gamma(n)=(n-1)! and Γ​(n+12)=(2​n−1)!!2n​πΓ𝑛12double-factorial2𝑛1superscript2𝑛𝜋\Gamma(n+\frac{1}{2})=\frac{(2n-1)!!}{2^{n}}\sqrt{\pi} for n∈ℕ+𝑛subscriptℕn\in\mathbb{N}_{+}, it is easy to see
that for a fixed d𝑑d, the RHS of the Equality 65 is maximized when di=dj=2subscript𝑑𝑖subscript𝑑𝑗2d_{i}=d_{j}=2 and dk=0subscript𝑑𝑘0d_{k}=0 for some i≠j𝑖𝑗i\neq j and k∉{i,j}𝑘𝑖𝑗k\notin\{i,j\}. Furthermore, straightforward calculations show that in that case the value of the RHS from Equality 65 is dd+2𝑑𝑑2\frac{d}{d+2}. That completes the proof of the Lemma.
∎

By 𝒟′​(j1,…,jm)superscript𝒟′subscript𝑗1…subscript𝑗𝑚\mathcal{D}^{\prime}(j_{1},\dots,j_{m}) denote a subset of 𝒟​(j1,…,jm)𝒟subscript𝑗1…subscript𝑗𝑚\mathcal{D}(j_{1},\dots,j_{m}) formed by only keeping d1,…,dmsubscript𝑑1…subscript𝑑𝑚d_{1},\dots,d_{m} such that for some i≠j𝑖𝑗i\neq j, di,dj>0subscript𝑑𝑖subscript𝑑𝑗0d_{i},d_{j}>0 and all disubscript𝑑𝑖d_{i} are even. As we have shown above, Δ^​(d1,…,dm)=0^Δsubscript𝑑1…subscript𝑑𝑚0\widehat{\Delta}(d_{1},\dots,d_{m})=0 when (d1,…,dm)∉𝒟′​(j1,…,jm)subscript𝑑1…subscript𝑑𝑚superscript𝒟′subscript𝑗1…subscript𝑗𝑚(d_{1},\dots,d_{m})\notin\mathcal{D}^{\prime}(j_{1},\dots,j_{m}). Otherwise,

Δ^​(d1,…,dm)≥2d+2​Λ​(d1,…,dm)≥0.^Δsubscript𝑑1…subscript𝑑𝑚2𝑑2Λsubscript𝑑1…subscript𝑑𝑚0\widehat{\Delta}(d_{1},\dots,d_{m})\geq\frac{2}{d+2}\Lambda(d_{1},\dots,d_{m})\geq 0.

Hence, since all terms in the sum

Δ=∑j=0∞θjj!​∑(j1,…,jm)∈𝒮j(jj1,…,jm)​∑(d1,…,dm)∈𝒟​(j1,…,jm)c^j1,…,jm​(d1,…,dm)Δsuperscriptsubscript𝑗0superscript𝜃𝑗𝑗subscriptsubscript𝑗1…subscript𝑗𝑚subscript𝒮𝑗binomial𝑗subscript𝑗1…subscript𝑗𝑚subscriptsubscript𝑑1…subscript𝑑𝑚𝒟subscript𝑗1…subscript𝑗𝑚subscript^𝑐subscript𝑗1…subscript𝑗𝑚subscript𝑑1…subscript𝑑𝑚\displaystyle\Delta=\sum_{j=0}^{\infty}\frac{\theta^{j}}{j!}\sum_{(j_{1},\dots,j_{m})\in\mathcal{S}_{j}}\binom{j}{j_{1},\dots,j_{m}}\sum_{(d_{1},\dots,d_{m})\in\mathcal{D}(j_{1},\dots,j_{m})}\widehat{c}_{j_{1},\dots,j_{m}}(d_{1},\dots,d_{m})

(66)

×Δ^​(d1,…,dm).absent^Δsubscript𝑑1…subscript𝑑𝑚\displaystyle\times\widehat{\Delta}(d_{1},\dots,d_{m}).

(67)

are nonnegative, we’ll get a lower bound on ΔΔ\Delta by only taking a subset of these terms. For this subset, we take j=4𝑗4j=4, a subset of 𝒮4subscript𝒮4\mathcal{S}_{4} with only two nonzero jk1=jk2=2subscript𝑗subscript𝑘1subscript𝑗subscript𝑘22j_{k_{1}}=j_{k_{2}}=2 for some k1≠k2subscript𝑘1subscript𝑘2k_{1}\neq k_{2} (there are (m2)binomial𝑚2\binom{m}{2} combinations of such j1,…,jmsubscript𝑗1…subscript𝑗𝑚j_{1},\dots,j_{m}). Then, we take only those d1,…,dmsubscript𝑑1…subscript𝑑𝑚d_{1},\dots,d_{m} from 𝒟​(j1,…,jm)𝒟subscript𝑗1…subscript𝑗𝑚\mathcal{D}(j_{1},\dots,j_{m}) which correspond to s=1𝑠1s=1 in (49) for k1,k2subscript𝑘1subscript𝑘2k_{1},k_{2} and s=0𝑠0s=0 for all other k𝑘k’s. Hence, dk1=dk2=2subscript𝑑subscript𝑘1subscript𝑑subscript𝑘22d_{k_{1}}=d_{k_{2}}=2 and all other dksubscript𝑑𝑘d_{k}’s are zero and the corresponding weight from the second sum in (67) would be a12​a0m−2superscriptsubscript𝑎12superscriptsubscript𝑎0𝑚2a_{1}^{2}a_{0}^{m-2}. For d1,…,dmsubscript𝑑1…subscript𝑑𝑚d_{1},\dots,d_{m} in such set, we’ll have τ​(d1,…,dm)≤dd+2𝜏subscript𝑑1…subscript𝑑𝑚𝑑𝑑2\tau(d_{1},\dots,d_{m})\leq\frac{d}{d+2} by Lemma 6 and, hence, Δ^​(d1,…,dm)≥2d+2​Λ​(d1,…,dm)^Δsubscript𝑑1…subscript𝑑𝑚2𝑑2Λsubscript𝑑1…subscript𝑑𝑚\widehat{\Delta}(d_{1},\dots,d_{m})\geq\frac{2}{d+2}\Lambda(d_{1},\dots,d_{m}). As the result, we get the following lower bound on ΔΔ\Delta:

ΔΔ\displaystyle\Delta
≥2​θ44!​(d+2)​(m2)​(42,2,0,…,0)​a12​a0m−2​Λ​(2,2,0,…,0)absent2superscript𝜃44𝑑2binomial𝑚2binomial4220…0superscriptsubscript𝑎12superscriptsubscript𝑎0𝑚2Λ220…0\displaystyle\geq\frac{2\theta^{4}}{4!(d+2)}\binom{m}{2}\binom{4}{2,2,0,\dots,0}a_{1}^{2}a_{0}^{m-2}\Lambda(2,2,0,\dots,0)

=θ4​m​(m−1)4​(d+2)​a12​a0m−2​Λ​(2,2,0,…,0)absentsuperscript𝜃4𝑚𝑚14𝑑2superscriptsubscript𝑎12superscriptsubscript𝑎0𝑚2Λ220…0\displaystyle=\frac{\theta^{4}m(m-1)}{4(d+2)}a_{1}^{2}a_{0}^{m-2}\Lambda(2,2,0,\dots,0)

=θ4​m​(m−1)4​(d+2)​a12​a0m−2​‖𝐳‖4​(𝔼​‖𝝎‖2)2​(𝔼​(𝐠12))2(𝔼​‖𝐠‖2)2.absentsuperscript𝜃4𝑚𝑚14𝑑2superscriptsubscript𝑎12superscriptsubscript𝑎0𝑚2superscriptnorm𝐳4superscript𝔼superscriptnorm𝝎22superscript𝔼superscriptsubscript𝐠122superscript𝔼superscriptnorm𝐠22\displaystyle=\frac{\theta^{4}m(m-1)}{4(d+2)}a_{1}^{2}a_{0}^{m-2}\|\mathbf{z}\|^{4}\left(\mathbb{E}\|\bm{\omega}\|^{2}\right)^{2}\frac{(\mathbb{E}(\mathbf{g}_{1}^{2}))^{2}}{(\mathbb{E}\|\mathbf{g}\|^{2})^{2}}.

Since 𝐠∼𝒩​(0,1)dsimilar-to𝐠𝒩superscript01𝑑\mathbf{g}\sim\mathcal{N}(0,1)^{d}, 𝔼​𝐠12=1𝔼superscriptsubscript𝐠121\mathbb{E}\mathbf{g}_{1}^{2}=1 and 𝔼​‖𝐠‖2=d​𝔼​𝐠12=d𝔼superscriptnorm𝐠2𝑑𝔼superscriptsubscript𝐠12𝑑\mathbb{E}\|\mathbf{g}\|^{2}=d\mathbb{E}\mathbf{g}_{1}^{2}=d. This results in

Δ≥θ4​m​(m−1)4​d2​(d+2)​a12​a0m−2​‖𝐳‖4​(𝔼​‖ω‖2)2Δsuperscript𝜃4𝑚𝑚14superscript𝑑2𝑑2superscriptsubscript𝑎12superscriptsubscript𝑎0𝑚2superscriptnorm𝐳4superscript𝔼superscriptnorm𝜔22\Delta\geq\frac{\theta^{4}m(m-1)}{4d^{2}(d+2)}a_{1}^{2}a_{0}^{m-2}\|\mathbf{z}\|^{4}\left(\mathbb{E}\|\omega\|^{2}\right)^{2}

(68)

which concludes the proof.

∎

#### F.4.3 Proof of Theorem 6

###### Proof.

We will use the notation from the proof of Theorem 5.
Since both estimators: F^mort​(𝐳)subscriptsuperscript^𝐹ort𝑚𝐳\widehat{F}^{\mathrm{ort}}_{m}(\mathbf{z}) and
F^miid​(𝐳)subscriptsuperscript^𝐹iid𝑚𝐳\widehat{F}^{\mathrm{iid}}_{m}(\mathbf{z}) are unbiased, we have:
MSE​(F^mort​(𝐳))=Var​(F^mort​(𝐳))MSEsubscriptsuperscript^𝐹ort𝑚𝐳Varsubscriptsuperscript^𝐹ort𝑚𝐳\mathrm{MSE}(\widehat{F}^{\mathrm{ort}}_{m}(\mathbf{z}))=\mathrm{Var}(\widehat{F}^{\mathrm{ort}}_{m}(\mathbf{z})) and
MSE​(F^miid​(𝐳))=Var​(F^miid​(𝐳))MSEsubscriptsuperscript^𝐹iid𝑚𝐳Varsubscriptsuperscript^𝐹iid𝑚𝐳\mathrm{MSE}(\widehat{F}^{\mathrm{iid}}_{m}(\mathbf{z}))=\mathrm{Var}(\widehat{F}^{\mathrm{iid}}_{m}(\mathbf{z})).
We have:

Var​(F^miid​(𝐳))=𝔼​[(F^miid​(𝐳)−𝔼​[F^miid​(𝐳)])2]=𝔼​[(F^miid​(𝐳))2]−F2​(𝐳).Varsubscriptsuperscript^𝐹iid𝑚𝐳𝔼delimited-[]superscriptsubscriptsuperscript^𝐹iid𝑚𝐳𝔼delimited-[]subscriptsuperscript^𝐹iid𝑚𝐳2𝔼delimited-[]superscriptsubscriptsuperscript^𝐹iid𝑚𝐳2superscript𝐹2𝐳\displaystyle\begin{split}\mathrm{Var}(\widehat{F}^{\mathrm{iid}}_{m}(\mathbf{z}))=\mathbb{E}[(\widehat{F}^{\mathrm{iid}}_{m}(\mathbf{z})-\mathbb{E}[\widehat{F}^{\mathrm{iid}}_{m}(\mathbf{z})])^{2}]=\mathbb{E}[(\widehat{F}^{\mathrm{iid}}_{m}(\mathbf{z}))^{2}]-F^{2}(\mathbf{z}).\end{split}

(69)

Similarly,

Var​(F^mort​(𝐳))=𝔼​[(F^mort​(𝐳))2]−F2​(𝐳).Varsubscriptsuperscript^𝐹ort𝑚𝐳𝔼delimited-[]superscriptsubscriptsuperscript^𝐹ort𝑚𝐳2superscript𝐹2𝐳\displaystyle\begin{split}\mathrm{Var}(\widehat{F}^{\mathrm{ort}}_{m}(\mathbf{z}))=\mathbb{E}[(\widehat{F}^{\mathrm{ort}}_{m}(\mathbf{z}))^{2}]-F^{2}(\mathbf{z}).\end{split}

(70)

We have:

𝔼​[(F^miid​(𝐳))2]=1m2​∑i=1m𝔼​[(Xiiid)2]+1m2​∑i≠j𝔼​[Xiiid​Xjiid].𝔼delimited-[]superscriptsubscriptsuperscript^𝐹iid𝑚𝐳21superscript𝑚2superscriptsubscript𝑖1𝑚𝔼delimited-[]superscriptsuperscriptsubscript𝑋𝑖iid21superscript𝑚2subscript𝑖𝑗𝔼delimited-[]subscriptsuperscript𝑋iid𝑖subscriptsuperscript𝑋iid𝑗\displaystyle\begin{split}\mathbb{E}[(\widehat{F}^{\mathrm{iid}}_{m}(\mathbf{z}))^{2}]=\frac{1}{m^{2}}\sum_{i=1}^{m}\mathbb{E}[(X_{i}^{\mathrm{iid}})^{2}]+\frac{1}{m^{2}}\sum_{i\neq j}\mathbb{E}[X^{\mathrm{iid}}_{i}X^{\mathrm{iid}}_{j}].\end{split}

(71)

Similarly, we get:

𝔼​[(F^mort​(𝐳))2]=1m2​∑i=1m𝔼​[(Xiort)2]+1m2​∑i≠j𝔼​[Xiort​Xjort].𝔼delimited-[]superscriptsubscriptsuperscript^𝐹ort𝑚𝐳21superscript𝑚2superscriptsubscript𝑖1𝑚𝔼delimited-[]superscriptsuperscriptsubscript𝑋𝑖ort21superscript𝑚2subscript𝑖𝑗𝔼delimited-[]subscriptsuperscript𝑋ort𝑖subscriptsuperscript𝑋ort𝑗\displaystyle\begin{split}\mathbb{E}[(\widehat{F}^{\mathrm{ort}}_{m}(\mathbf{z}))^{2}]=\frac{1}{m^{2}}\sum_{i=1}^{m}\mathbb{E}[(X_{i}^{\mathrm{ort}})^{2}]+\frac{1}{m^{2}}\sum_{i\neq j}\mathbb{E}[X^{\mathrm{ort}}_{i}X^{\mathrm{ort}}_{j}].\end{split}

(72)

Therefore, since marginal distributions of Xiiidsuperscriptsubscript𝑋𝑖iidX_{i}^{\mathrm{iid}} and Xiortsuperscriptsubscript𝑋𝑖ortX_{i}^{\mathrm{ort}} are the same, we have:

MSE​(F^miid​(𝐳))−MSE​(F^mort​(𝐳))=(m2)⋅2⋅1m2​(𝔼​[X1iid​X2iid]−𝔼​[X1ort​X2ort])=(1−1m)​(𝔼​[X1iid​X2iid]−𝔼​[X1ort​X2ort])MSEsubscriptsuperscript^𝐹iid𝑚𝐳MSEsubscriptsuperscript^𝐹ort𝑚𝐳⋅binomial𝑚221superscript𝑚2𝔼delimited-[]subscriptsuperscript𝑋iid1subscriptsuperscript𝑋iid2𝔼delimited-[]subscriptsuperscript𝑋ort1subscriptsuperscript𝑋ort211𝑚𝔼delimited-[]subscriptsuperscript𝑋iid1subscriptsuperscript𝑋iid2𝔼delimited-[]subscriptsuperscript𝑋ort1subscriptsuperscript𝑋ort2\displaystyle\begin{split}\mathrm{MSE}(\widehat{F}^{\mathrm{iid}}_{m}(\mathbf{z}))-\mathrm{MSE}(\widehat{F}^{\mathrm{ort}}_{m}(\mathbf{z}))={m\choose 2}\cdot 2\cdot\frac{1}{m^{2}}(\mathbb{E}[X^{\mathrm{iid}}_{1}X^{\mathrm{iid}}_{2}]-\mathbb{E}[X^{\mathrm{ort}}_{1}X^{\mathrm{ort}}_{2}])\\
=(1-\frac{1}{m})(\mathbb{E}[X^{\mathrm{iid}}_{1}X^{\mathrm{iid}}_{2}]-\mathbb{E}[X^{\mathrm{ort}}_{1}X^{\mathrm{ort}}_{2}])\end{split}

(73)

Plugging in the formula for Xiortsubscriptsuperscript𝑋ort𝑖X^{\mathrm{ort}}_{i} and Xiiidsubscriptsuperscript𝑋iid𝑖X^{\mathrm{iid}}_{i} from Equation 48 and Equation 49, and using our analysis from the proof of Theorem 3 we obtain:

MSE(F^miid(𝐳))−MSE(F^mort(𝐳))=(1−1m)∑t,u=0∞atau∥𝐳∥2t+u𝔼[∥ω∥2t]𝔼[∥ω∥2u]⋅𝔼​[rt]​𝔼​[ru]𝔼​[g12+…+gd2t]​𝔼​[g12+…+gd2u]​(1−τ​(t,u)).MSEsubscriptsuperscript^𝐹iid𝑚𝐳MSEsubscriptsuperscript^𝐹ort𝑚𝐳11𝑚superscriptsubscript𝑡𝑢0⋅subscript𝑎𝑡subscript𝑎𝑢superscriptsubscriptdelimited-∥∥𝐳2𝑡𝑢𝔼delimited-[]superscriptsubscriptdelimited-∥∥𝜔2𝑡𝔼delimited-[]superscriptsubscriptdelimited-∥∥𝜔2𝑢𝔼delimited-[]superscript𝑟𝑡𝔼delimited-[]superscript𝑟𝑢𝔼delimited-[]superscriptsuperscriptsubscript𝑔12…superscriptsubscript𝑔𝑑2𝑡𝔼delimited-[]superscriptsuperscriptsubscript𝑔12…superscriptsubscript𝑔𝑑2𝑢1𝜏𝑡𝑢\displaystyle\begin{split}\mathrm{MSE}(\widehat{F}^{\mathrm{iid}}_{m}(\mathbf{z}))-\mathrm{MSE}(\widehat{F}^{\mathrm{ort}}_{m}(\mathbf{z}))=(1-\frac{1}{m})\sum_{t,u=0}^{\infty}a_{t}a_{u}\|\mathbf{z}\|_{2}^{t+u}\mathbb{E}[\|\omega\|_{2}^{t}]\mathbb{E}[\|\omega\|_{2}^{u}]\cdot\\
\frac{\mathbb{E}[r^{t}]\mathbb{E}[r^{u}]}{\mathbb{E}[\sqrt{g_{1}^{2}+...+g_{d}^{2}}^{t}]\mathbb{E}[\sqrt{g_{1}^{2}+...+g_{d}^{2}}^{u}]}(1-\tau(t,u)).\end{split}

(74)

for ω∼Ωsimilar-to𝜔Ω\omega\sim\Omega and r∼𝒩​(0,1)similar-to𝑟𝒩01r\sim\mathcal{N}(0,1).

Based on the definition of τ𝜏\tau (63), if t=0𝑡0t=0 or u=0𝑢0u=0, τ​(t,u)=1𝜏𝑡𝑢1\tau(t,u)=1 and the whole corresponding term in the sum (74) is zero. Also, if t𝑡t is odd, 𝔼​(rt)=0𝔼superscript𝑟𝑡0\mathbb{E}(r^{t})=0 and, again, the corresponding term in the sum (74) is zero. Same holds for u𝑢u from (74). Based on the analysis from Theorem 5’s proof and FΩ,g​(𝐳)subscript𝐹Ω𝑔𝐳F_{\Omega,g}(\mathbf{z})’s definition we have:

FΩ,g​(𝐳)=∑t=0∞at​‖𝐳‖2t​𝔼​[‖ω‖2t]⋅𝔼​[rt]𝔼​[g12+…+gd2t]=∑t=0∞a2​t​‖𝐳‖22​t​𝔼​[‖ω‖22​t]⋅𝔼​[r2​t]𝔼​[g12+…+gd22​t]subscript𝐹Ω𝑔𝐳superscriptsubscript𝑡0⋅subscript𝑎𝑡superscriptsubscriptnorm𝐳2𝑡𝔼delimited-[]superscriptsubscriptnorm𝜔2𝑡𝔼delimited-[]superscript𝑟𝑡𝔼delimited-[]superscriptsuperscriptsubscript𝑔12…superscriptsubscript𝑔𝑑2𝑡superscriptsubscript𝑡0⋅subscript𝑎2𝑡superscriptsubscriptnorm𝐳22𝑡𝔼delimited-[]superscriptsubscriptnorm𝜔22𝑡𝔼delimited-[]superscript𝑟2𝑡𝔼delimited-[]superscriptsuperscriptsubscript𝑔12…superscriptsubscript𝑔𝑑22𝑡F_{\Omega,g}(\mathbf{z})=\sum_{t=0}^{\infty}a_{t}\|\mathbf{z}\|_{2}^{t}\mathbb{E}[\|\omega\|_{2}^{t}]\cdot\frac{\mathbb{E}[r^{t}]}{\mathbb{E}[\sqrt{g_{1}^{2}+...+g_{d}^{2}}^{t}]}=\sum_{t=0}^{\infty}a_{2t}\|\mathbf{z}\|_{2}^{2t}\mathbb{E}[\|\omega\|_{2}^{2t}]\cdot\frac{\mathbb{E}[r^{2t}]}{\mathbb{E}[\sqrt{g_{1}^{2}+...+g_{d}^{2}}^{2t}]}

where in the second transition we use the fact that 𝔼​[rt]=0𝔼delimited-[]superscript𝑟𝑡0\mathbb{E}[r^{t}]=0 for odd t𝑡t.

Hence, we can rewrite (74) by excluding terms which are definitely zero and using Lemma 6:

MSE(F^miid(𝐳))−MSE(F^mort(𝐳))≥(1−1m)2d+2∑t,u=1∞a2​ta2​u∥𝐳∥22​t+2​u𝔼[∥ω∥22​t]𝔼[∥ω∥22​u]⋅𝔼​[r2​t]​𝔼​[r2​u]𝔼​[g12+…+gd22​t]​𝔼​[g12+…+gd22​u]=(1−1m)​2d+2​(∑t=1∞a2​t​‖𝐳‖22​t​𝔼​[‖ω‖22​t]⋅𝔼​[r2​t]𝔼​[g12+…+gd22​t])2=(1−1m)​2d+2​(FΩ,g​(𝐳)−a0)2.MSEsubscriptsuperscript^𝐹iid𝑚𝐳MSEsubscriptsuperscript^𝐹ort𝑚𝐳11𝑚2𝑑2superscriptsubscript𝑡𝑢1⋅subscript𝑎2𝑡subscript𝑎2𝑢superscriptsubscriptdelimited-∥∥𝐳22𝑡2𝑢𝔼delimited-[]superscriptsubscriptdelimited-∥∥𝜔22𝑡𝔼delimited-[]superscriptsubscriptdelimited-∥∥𝜔22𝑢𝔼delimited-[]superscript𝑟2𝑡𝔼delimited-[]superscript𝑟2𝑢𝔼delimited-[]superscriptsuperscriptsubscript𝑔12…superscriptsubscript𝑔𝑑22𝑡𝔼delimited-[]superscriptsuperscriptsubscript𝑔12…superscriptsubscript𝑔𝑑22𝑢11𝑚2𝑑2superscriptsuperscriptsubscript𝑡1⋅subscript𝑎2𝑡superscriptsubscriptdelimited-∥∥𝐳22𝑡𝔼delimited-[]superscriptsubscriptdelimited-∥∥𝜔22𝑡𝔼delimited-[]superscript𝑟2𝑡𝔼delimited-[]superscriptsuperscriptsubscript𝑔12…superscriptsubscript𝑔𝑑22𝑡211𝑚2𝑑2superscriptsubscript𝐹Ω𝑔𝐳subscript𝑎02\displaystyle\begin{split}\mathrm{MSE}(\widehat{F}^{\mathrm{iid}}_{m}(\mathbf{z}))-\mathrm{MSE}(\widehat{F}^{\mathrm{ort}}_{m}(\mathbf{z}))\geq(1-\frac{1}{m})\frac{2}{d+2}\sum_{t,u=1}^{\infty}a_{2t}a_{2u}\|\mathbf{z}\|_{2}^{2t+2u}\mathbb{E}[\|\omega\|_{2}^{2t}]\mathbb{E}[\|\omega\|_{2}^{2u}]\cdot\\
\frac{\mathbb{E}[r^{2t}]\mathbb{E}[r^{2u}]}{\mathbb{E}[\sqrt{g_{1}^{2}+...+g_{d}^{2}}^{2t}]\mathbb{E}[\sqrt{g_{1}^{2}+...+g_{d}^{2}}^{2u}]}\\
=(1-\frac{1}{m})\frac{2}{d+2}\left(\sum_{t=1}^{\infty}a_{2t}\|\mathbf{z}\|_{2}^{2t}\mathbb{E}[\|\omega\|_{2}^{2t}]\cdot\frac{\mathbb{E}[r^{2t}]}{\mathbb{E}[\sqrt{g_{1}^{2}+...+g_{d}^{2}}^{2t}]}\right)^{2}\\
=(1-\frac{1}{m})\frac{2}{d+2}\left(F_{\Omega,g}(\mathbf{z})-a_{0}\right)^{2}.\end{split}

(75)

That completes the proof.
∎

### F.5 Proof of Theorem 4

We showed in the main body of the paper that in contrast to other methods approximating the attention matrix 𝐀𝐀\mathbf{A}, our algorithm provides strong concentration guarantees. This is the case also for trigonometric random features, yet, as discussed in the main body of the paper, due to attention renormalization and higher variance of the estimation of small entries of the attention matrix, trigonometric mechanism is sub-optimal.
We show here that moptsubscript𝑚optm_{\mathrm{opt}}, the optimal number of random projections for the trigonometric orthogonal mechanism for accurate estimation of the attention matrix does not depend on L𝐿L but only on d𝑑d. In fact, we prove that if we take mopt=Θ​(d​log⁡(d))subscript𝑚optΘ𝑑𝑑m_{\mathrm{opt}}=\Theta(d\log(d)), then with O​(L​d2​log⁡(d))𝑂𝐿superscript𝑑2𝑑O(Ld^{2}\log(d))-time, we can approximate 𝐀𝐀\mathbf{A} up to any precision, regardless of the number of tokens L𝐿L. In order to provide those guarantees, we leverage recent research on the theory of negative dependence for ORFs (Lin et al., 2020).

We prove the more general version of Theorem 4 from the main body of the paper:

###### Theorem 7 (Uniform convergence for the trigonometric mechanism).

Define entries of the attention matrix 𝐀𝐀\mathbf{A} as follows: 𝐀i,j=g​(𝐪i⊤)​K​(1d14​𝐪i⊤,1d14​𝐤j⊤)​h​(𝐤j⊤)subscript𝐀𝑖𝑗𝑔superscriptsubscript𝐪𝑖topK1superscript𝑑14superscriptsubscript𝐪𝑖top1superscript𝑑14superscriptsubscript𝐤𝑗topℎsuperscriptsubscript𝐤𝑗top\mathbf{A}_{i,j}=g(\mathbf{q}_{i}^{\top})\mathrm{K}(\frac{1}{d^{\frac{1}{4}}}\mathbf{q}_{i}^{\top},\frac{1}{d^{\frac{1}{4}}}\mathbf{k}_{j}^{\top})h(\mathbf{k}_{j}^{\top}) for some
g,h:ℝd→ℝ:𝑔ℎ→superscriptℝ𝑑ℝg,h:\mathbb{R}^{d}\rightarrow\mathbb{R} and where KK\mathrm{K} is a radial basis function (RBF) kernel (Choromanski et al., 2018b) with corresponding spectral distribution ΩΩ\Omega (e.g. Gaussian kernel for which Ω=𝒩​(0,𝐈d)Ω𝒩0subscript𝐈𝑑\Omega=\mathcal{N}(0,\mathbf{I}_{d})). Assume that the rows of matrices 𝐐𝐐\mathbf{Q} and 𝐊𝐊\mathbf{K} are taken from a ball B​(R)𝐵𝑅B(R) of radius R𝑅R, centered at 00 (i.e. norms of queries and keys are upper-bounded by R𝑅R).
Define l=R​d−14𝑙𝑅superscript𝑑14l=Rd^{-\frac{1}{4}} and take g∗=max𝐱∈B​(l)⁡|g​(𝐱)|superscript𝑔subscript𝐱𝐵𝑙𝑔𝐱g^{*}=\max_{\mathbf{x}\in B(l)}|g(\mathbf{x})| and
h∗=max𝐱∈B​(l)⁡|h​(𝐱)|superscriptℎsubscript𝐱𝐵𝑙ℎ𝐱h^{*}=\max_{\mathbf{x}\in B(l)}|h(\mathbf{x})|.
Then for any ϵ>0italic-ϵ0\epsilon>0, δ=ϵg∗​h∗𝛿italic-ϵsuperscript𝑔superscriptℎ\delta=\frac{\epsilon}{g^{*}h^{*}}
and the number of random projections m=Ω​(dδ2​log⁡(4​σ​Rδ​d14))𝑚Ω𝑑superscript𝛿24𝜎𝑅𝛿superscript𝑑14m=\Omega(\frac{d}{\delta^{2}}\log(\frac{4\sigma R}{\delta d^{\frac{1}{4}}})) for σ=𝔼ω∼Ω​[ω⊤​ω]𝜎subscript𝔼similar-to𝜔Ωdelimited-[]superscript𝜔top𝜔\sigma=\mathbb{E}_{\omega\sim\Omega}[\omega^{\top}\omega] the following holds:
‖𝐀^−𝐀‖∞≤ϵsubscriptnorm^𝐀𝐀italic-ϵ\|\widehat{\mathbf{A}}-\mathbf{A}\|_{\infty}\leq\epsilon
with any constant probability,
where 𝐀^^𝐀\widehat{\mathbf{A}} approximates generalized attention matrix via orthogonal trigonometric random features.

The result holds in particular for regular softmax-attention for which KK\mathrm{K} is a Gaussian kernel and g​(𝐱)=h​(𝐱)=exp⁡(‖𝐱‖22)𝑔𝐱ℎ𝐱superscriptnorm𝐱22g(\mathbf{x})=h(\mathbf{x})=\exp(\frac{\|\mathbf{x}\|^{2}}{2}). In that case mopt=Ω​(dδ2​log⁡(4​d34​Rδ))subscript𝑚optΩ𝑑superscript𝛿24superscript𝑑34𝑅𝛿m_{\mathrm{opt}}=\Omega(\frac{d}{\delta^{2}}\log(\frac{4d^{\frac{3}{4}}R}{\delta})) since σ=d𝜎𝑑\sigma=d.

###### Proof.

Let 𝐃𝐐subscript𝐃𝐐\mathbf{D}_{\mathbf{Q}} be a diagonal matrix with entries of the form: g​(𝐪i⊤)𝑔superscriptsubscript𝐪𝑖topg(\mathbf{q}_{i}^{\top}) and let 𝐃𝐊subscript𝐃𝐊\mathbf{D}_{\mathbf{K}} be a diagonal matrix with entries of the form: h​(𝐤i⊤)ℎsuperscriptsubscript𝐤𝑖toph(\mathbf{k}_{i}^{\top}). Denote
𝐁=[K​(1d14​𝐪i⊤,1d14​𝐤j⊤)]i,j∈ℝL×L𝐁subscriptdelimited-[]K1superscript𝑑14superscriptsubscript𝐪𝑖top1superscript𝑑14superscriptsubscript𝐤𝑗top𝑖𝑗superscriptℝ𝐿𝐿\mathbf{B}=[\mathrm{K}(\frac{1}{d^{\frac{1}{4}}}\mathbf{q}_{i}^{\top},\frac{1}{d^{\frac{1}{4}}}\mathbf{k}_{j}^{\top})]_{i,j}\in\mathbb{R}^{L\times L}. Denote by 𝐀^^𝐀\widehat{\mathbf{A}} and approximation of the attention matrix obtained from trigonometric orthogonal random features and by 𝐁^^𝐁\widehat{\mathbf{B}} an approximation of matrix 𝐁𝐁\mathbf{B} that those random features provide.
We rely on Theorem 3 from (Lin et al., 2020).
Note that we can apply it in our case, since for RBF kernels the corresponding functions fisubscript𝑓𝑖f_{i} satisfy f1​(x)=sin⁡(x)subscript𝑓1𝑥𝑥f_{1}(x)=\sin(x), f2​(x)=cos⁡(x)subscript𝑓2𝑥𝑥f_{2}(x)=\cos(x) (thus in particular are bounded). Also, it is not hard to observe (see for instance analysis in Claim 1 from (Rahimi & Recht, 2007)) that we can take: Lf=1subscript𝐿𝑓1L_{f}=1 (for Lfsubscript𝐿𝑓L_{f} as in Theorem 3 from (Lin et al., 2020)).
Using Theorem 3 from (Lin et al., 2020), we conclude that:

‖𝐁^−𝐁‖∞≤δsubscriptnorm^𝐁𝐁𝛿\|\widehat{\mathbf{B}}-\mathbf{B}\|_{\infty}\leq\delta

(76)

with any constant probability as long as
m=Ω​(dδ2)​log⁡(σ⋅diam​(ℳ)δ)𝑚Ω𝑑superscript𝛿2⋅𝜎diamℳ𝛿m=\Omega(\frac{d}{\delta^{2}})\log(\frac{\sigma\cdot\mathrm{diam}(\mathcal{M})}{\delta}),
where σ=𝔼​[ω⊤​ω]𝜎𝔼delimited-[]superscript𝜔top𝜔\sigma=\mathbb{E}[\omega^{\top}\omega] and ℳℳ\mathcal{M} is the diameter of the smallest ball ℳℳ\mathcal{M} containing all vectors of the form 𝐳=𝐐id14−𝐊jd14𝐳subscript𝐐𝑖superscript𝑑14subscript𝐊𝑗superscript𝑑14\mathbf{z}=\frac{\mathbf{Q}_{i}}{d^{\frac{1}{4}}}-\frac{\mathbf{K}_{j}}{d^{\frac{1}{4}}}.
Since ‖𝐐i‖2,‖𝐊j‖2≤Rsubscriptnormsubscript𝐐𝑖2subscriptnormsubscript𝐊𝑗2𝑅\|\mathbf{Q}_{i}\|_{2},\|\mathbf{K}_{j}\|_{2}\leq R, we conclude that ‖𝐳‖2≤2​Rd14subscriptnorm𝐳22𝑅superscript𝑑14\|\mathbf{z}\|_{2}\leq\frac{2R}{d^{\frac{1}{4}}} and thus one can take diam​(ℳ)=4​Rd14diamℳ4𝑅superscript𝑑14\mathrm{diam}(\mathcal{M})=\frac{4R}{d^{\frac{1}{4}}}.
We have:

‖𝐀^−𝐀‖∞=‖𝐃𝐐​(𝐁^−𝐁)​𝐃𝐊‖∞≤‖𝐃𝐐‖∞​‖𝐁^−𝐁‖∞​‖𝐃𝐊‖∞≤δ​g∗​h∗subscriptnorm^𝐀𝐀subscriptnormsubscript𝐃𝐐^𝐁𝐁subscript𝐃𝐊subscriptnormsubscript𝐃𝐐subscriptnorm^𝐁𝐁subscriptnormsubscript𝐃𝐊𝛿superscript𝑔superscriptℎ\|\widehat{\mathbf{A}}-\mathbf{A}\|_{\infty}=\|\mathbf{D}_{\mathbf{Q}}(\widehat{\mathbf{B}}-\mathbf{B})\mathbf{D}_{\mathbf{K}}\|_{\infty}\leq\|\mathbf{D}_{\mathbf{Q}}\|_{\infty}\|\widehat{\mathbf{B}}-\mathbf{B}\|_{\infty}\|\mathbf{D}_{\mathbf{K}}\|_{\infty}\leq\delta g^{*}h^{*}

(77)

Taking δ=ϵg∗​h∗𝛿italic-ϵsuperscript𝑔superscriptℎ\delta=\frac{\epsilon}{g^{*}h^{*}} completes the proof.
∎

### F.6 Discussion of Theorem 4

As a consequence of Theorem 4, the number m𝑚m of random projections required to approximate the attention matrix within ϵitalic-ϵ\epsilon error is a function of data dimensionality d𝑑d, the parameter ϵitalic-ϵ\epsilon and the radius R𝑅R of the ball within which the queries and keys live:

m=Ψ​(ϵ,d,R).𝑚Ψitalic-ϵ𝑑𝑅m=\Psi(\epsilon,d,R).

The dependence on d𝑑d and ϵitalic-ϵ\epsilon is fairly easy to understand: with a larger dimensionality d𝑑d we need more random projeections (on the order of magnitude d​log⁡(d)𝑑𝑑d\log(d)) to get an approximation within ϵitalic-ϵ\epsilon error. The dependence on R𝑅R means that the length of queries and keys cannot grow at a fixed m𝑚m if we want to retain the quality of the approximation.
In particular, this means that FAVOR cannot approximate hard attention on sequences of unlimited length with a fixed m𝑚m. When the sequence length increases, even the standard attention requires longer and longer vectors to make the softmax concentrated enough to pick single elements. Nevertheless, as seen in our experiments, this limitation does not manifest itself in practice at the lengths we experimented with.

Generated on Tue Mar 19 04:37:20 2024 by LaTeXML

# Efficient Memory Management for Large Language Model Serving with PagedAttention

- Source HTML: `raw/html/Kwon et al. - 2023 - Efficient Memory Management for Large Language Model Serving with PagedAttention.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2309.06180
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Efficient Memory Management for Large Language Model Serving with PagedAttention

Woosuk Kwon1,∗1{}^{\text{1},*} Zhuohan Li1,∗1{}^{\text{1},*} Siyuan Zhuang11{}^{\text{1}} Ying Sheng1,212{}^{\text{1},\text{2}} Lianmin Zheng11{}^{\text{1}} Cody Hao Yu33{}^{\text{3}} 
Joseph E. Gonzalez11{}^{\text{1}} Hao Zhang44{}^{\text{4}} Ion Stoica11{}^{\text{1}}

11{}^{\text{1}}UC Berkeley  22{}^{\text{2}}Stanford University  33{}^{\text{3}}Independent Researcher  44{}^{\text{4}}UC San Diego

(2023)

###### Abstract.

High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time.
However, existing systems struggle because the key-value cache (KV cache) memory for each request is huge and grows and shrinks dynamically.
When managed inefficiently, this memory can be significantly wasted by fragmentation and redundant duplication, limiting the batch size.
To address this problem, we propose PagedAttention, an attention algorithm inspired by the classical virtual memory and paging techniques in operating systems.
On top of it, we build vLLM, an LLM serving system that achieves (1) near-zero waste in KV cache memory and (2) flexible sharing of KV cache within and across requests to further reduce memory usage.
Our evaluations show that vLLM improves the throughput of popular LLMs by 2-4×\times with the same level of latency compared to the state-of-the-art systems, such as FasterTransformer and Orca.
The improvement is more pronounced with longer sequences, larger models, and more complex decoding algorithms.
vLLM’s source code is publicly available at https://github.com/vllm-project/vllm.

## 1. Introduction

The emergence of large language models (LLMs) like GPT (OpenAI, 2023b; Brown et al., 2020) and PaLM (Chowdhery et al., 2022) have enabled new applications such as programming assistants (Github, 2022; Chen et al., 2021) and universal chatbots (OpenAI, 2022; Google, 2023) that are starting to profoundly impact our work and daily routines. Many cloud companies (OpenAI, 2020; Services, 2023) are racing to provide these applications as hosted services. However, running these applications is very expensive, requiring a large number of hardware accelerators such as GPUs. According to recent estimates, processing an LLM request can be 10×\times more expensive than a traditional keyword query (Reuters, 2023).
Given these high costs, increasing the throughput—and hence reducing the cost per request—of LLM serving systems is becoming more important.

At the core of LLMs lies an autoregressive Transformer model (Vaswani et al., 2017).
This model generates words (tokens), one at a time, based on the input (prompt) and the previous sequence of the output’s tokens it has generated so far.
For each request, this expensive process is repeated until the model outputs a termination token.
This sequential generation process makes the workload memory-bound, underutilizing the computation power of GPUs and limiting the serving throughput.

Improving the throughput is possible by batching multiple requests together.
However, to process many requests in a batch, the memory space for each request should be efficiently managed.
For example, Fig. 1 (left) illustrates the memory distribution for a 13B-parameter LLM on an NVIDIA A100 GPU with 40GB RAM.
Approximately 65% of the memory is allocated for the model weights, which remain static during serving.
Close to 30% of the memory is used to store the dynamic states of the requests.
For Transformers, these states consist of the key and value tensors associated with the attention mechanism, commonly referred to as KV cache (Pope et al., 2022), which represent the context from earlier tokens to generate new output tokens in sequence.
The remaining small percentage of memory is used for other data, including activations – the ephemeral tensors created when evaluating the LLM.
Since the model weights are constant and the activations only occupy a small fraction of the GPU memory, the way the KV cache is managed is critical in determining the maximum batch size.
When managed inefficiently, the KV cache memory can significantly limit the batch size and consequently the throughput of the LLM, as illustrated in Fig. 1 (right).

In this paper, we observe that existing LLM serving systems (Yu et al., 2022; NVIDIA, 2023a) fall short of managing the KV cache memory efficiently.
This is mainly because they store the KV cache of a request in contiguous memory space, as most deep learning frameworks (Paszke et al., 2019; Olston et al., 2017) require tensors to be stored in contiguous memory.
However, unlike the tensors in the traditional deep learning workloads, the KV cache has unique characteristics: it dynamically grows and shrinks over time as the model generates new tokens, and its lifetime and length are not known a priori.
These characteristics make the existing systems’ approach significantly inefficient in two ways:

First, the existing systems (Yu et al., 2022; NVIDIA, 2023a) suffer from internal and external memory fragmentation.
To store the KV cache of a request in contiguous space, they pre-allocate a contiguous chunk of memory with the request’s maximum length (e.g., 2048 tokens).
This can result in severe internal fragmentation, since the request’s actual length can be much shorter than its maximum length (e.g., Fig. 11).
Moreover, even if the actual length is known a priori, the pre-allocation is still inefficient: As the entire chunk is reserved during the request’s lifetime, other shorter requests cannot utilize any part of the chunk that is currently unused.
Besides, external memory fragmentation can also be significant, since the pre-allocated size can be different for each request.
Indeed, our profiling results in Fig. 2 show that only 20.4% - 38.2% of the KV cache memory is used to store the actual token states in the existing systems.

Second, the existing systems cannot exploit the opportunities for memory sharing.
LLM services often use advanced decoding algorithms, such as parallel sampling and beam search, that generate multiple outputs per request.
In these scenarios, the request consists of multiple sequences that can partially share their KV cache.
However, memory sharing is not possible in the existing systems because the KV cache of the sequences is stored in separate contiguous spaces.

To address the above limitations, we propose PagedAttention, an attention algorithm inspired by the operating system’s (OS) solution to memory fragmentation and sharing: virtual memory with paging.
PagedAttention divides the request’s KV cache into blocks, each of which can contain the attention keys and values of a fixed number of tokens.
In PagedAttention, the blocks for the KV cache are not necessarily stored in contiguous space.
Therefore, we can manage the KV cache in a more flexible way as in OS’s virtual memory: one can think of blocks as pages, tokens as bytes, and requests as processes.
This design alleviates internal fragmentation by using relatively small blocks and allocating them on demand.
Moreover, it eliminates external fragmentation as all blocks have the same size.
Finally, it enables memory sharing at the granularity of a block, across the different sequences associated with the same request or even across the different requests.

In this work, we build vLLM, a high-throughput distributed LLM serving engine on top of PagedAttention that achieves near-zero waste in KV cache memory.
vLLM uses block-level memory management and preemptive request scheduling that are co-designed with PagedAttention.
vLLM supports popular LLMs such as GPT (Brown et al., 2020), OPT (Zhang et al., 2022), and LLaMA (Touvron et al., 2023) with varying sizes, including the ones exceeding the memory capacity of a single GPU.
Our evaluations on various models and workloads show that vLLM improves the LLM serving throughput by 2-4×\times compared to the state-of-the-art systems (Yu et al., 2022; NVIDIA, 2023a), without affecting the model accuracy at all.
The improvements are more pronounced with longer sequences, larger models, and more complex decoding algorithms (§4.3).
In summary, we make the following contributions:

- •

We identify the challenges in memory allocation in serving LLMs and quantify their impact on serving performance.

- •

We propose PagedAttention, an attention algorithm that operates on KV cache stored in non-contiguous paged memory, which is inspired by the virtual memory and paging in OS.

- •

We design and implement vLLM, a distributed LLM serving engine built on top of PagedAttention.

- •

We evaluate vLLM on various scenarios and demonstrate that it substantially outperforms the previous state-of-the-art solutions such as FasterTransformer (NVIDIA, 2023a) and Orca (Yu et al., 2022).

## 2. Background

In this section, we describe the generation and serving procedures of typical LLMs and the iteration-level scheduling used in LLM serving.

### 2.1. Transformer-Based Large Language Models

The task of language modeling is to model the probability of a list of tokens (x1,…,xn).subscript𝑥1…subscript𝑥𝑛(x_{1},\ldots,x_{n}). Since language has a natural sequential ordering, it is common to factorize the joint probability over the whole sequence as the product of conditional probabilities (a.k.a. autoregressive decomposition (Bengio et al., 2000)):

(1)

P​(x)=P​(x1)⋅P​(x2∣x1)​⋯​P​(xn∣x1,…,xn−1).𝑃𝑥⋅𝑃subscript𝑥1𝑃conditionalsubscript𝑥2subscript𝑥1⋯𝑃conditionalsubscript𝑥𝑛subscript𝑥1…subscript𝑥𝑛1P(x)=P(x_{1})\cdot P(x_{2}\mid x_{1})\cdots P(x_{n}\mid x_{1},\ldots,x_{n-1}).

Transformers (Vaswani et al., 2017) have become the de facto standard architecture for modeling the probability above at a large scale. The most important component of a Transformer-based language model is its self-attention layers. For an input hidden state sequence (x1,…,xn)∈ℝn×dsubscript𝑥1…subscript𝑥𝑛superscriptℝ𝑛𝑑(x_{1},\ldots,x_{n})\in\mathbb{R}^{n\times d}, a self-attention layer first applies linear transformations on each position i𝑖i to get the query, key, and value vectors:

(2)

qi=Wq​xi,ki=Wk​xi,vi=Wv​xi.formulae-sequencesubscript𝑞𝑖subscript𝑊𝑞subscript𝑥𝑖formulae-sequencesubscript𝑘𝑖subscript𝑊𝑘subscript𝑥𝑖subscript𝑣𝑖subscript𝑊𝑣subscript𝑥𝑖q_{i}=W_{q}x_{i},\ k_{i}=W_{k}x_{i},\ v_{i}=W_{v}x_{i}.

Then, the self-attention layer computes the attention score ai​jsubscript𝑎𝑖𝑗a_{ij} by multiplying the query vector at one position with all the key vectors before it and compute the output oisubscript𝑜𝑖o_{i} as the weighted average over the value vectors:

(3)

ai​j=exp⁡(qi⊤​kj/d)∑t=1iexp⁡(qi⊤​kt/d),oi=∑j=1iai​j​vj.formulae-sequencesubscript𝑎𝑖𝑗superscriptsubscript𝑞𝑖topsubscript𝑘𝑗𝑑superscriptsubscript𝑡1𝑖superscriptsubscript𝑞𝑖topsubscript𝑘𝑡𝑑subscript𝑜𝑖superscriptsubscript𝑗1𝑖subscript𝑎𝑖𝑗subscript𝑣𝑗a_{ij}=\frac{\exp(q_{i}^{\top}k_{j}/\sqrt{d})}{\sum_{t=1}^{i}\exp(q_{i}^{\top}k_{t}/\sqrt{d})},\ o_{i}=\sum_{j=1}^{i}a_{ij}v_{j}.

Besides the computation in Eq. 4, all other components in the Transformer model, including the embedding layer, feed-forward layer, layer normalization (Ba et al., 2016), residual connection (He et al., 2016), output logit computation, and the query, key, and value transformation in Eq. 2, are all applied independently position-wise in a form of
yi=f​(xi).subscript𝑦𝑖𝑓subscript𝑥𝑖y_{i}=f(x_{i}).

### 2.2. LLM Service & Autoregressive Generation

Once trained, LLMs are often deployed as a conditional generation service (e.g., completion API (OpenAI, 2020) or chatbot (OpenAI, 2022; Google, 2023)). A request to an LLM service provides a list of input prompt tokens (x1,…,xn),subscript𝑥1…subscript𝑥𝑛(x_{1},\ldots,x_{n}), and the LLM service generates a list of output tokens (xn+1,…,xn+T)subscript𝑥𝑛1…subscript𝑥𝑛𝑇(x_{n+1},\ldots,x_{n+T}) according to Eq. 1. We refer to the concatenation of the prompt and output lists as sequence.

Due to the decomposition in Eq. 1, the LLM can only sample and generate new tokens one by one, and the generation process of each new token depends on all the previous tokens in that sequence, specifically their key and value vectors. In this sequential generation process, the key and value vectors of existing tokens are often cached for generating future tokens, known as KV cache. Note that the KV cache of one token depends on all its previous tokens. This means that the KV cache of the same token appearing at different positions in a sequence will be different.

Given a request prompt, the generation computation in the LLM service can be decomposed into two phases:

The prompt phase takes the whole user prompt (x1,…,xn)subscript𝑥1…subscript𝑥𝑛(x_{1},\ldots,x_{n}) as input and computes the probability of the first new token P​(xn+1∣x1,…,xn)𝑃conditionalsubscript𝑥𝑛1subscript𝑥1…subscript𝑥𝑛P(x_{n+1}\mid x_{1},\ldots,x_{n}). During this process, also generates the key vectors k1,…,knsubscript𝑘1…subscript𝑘𝑛k_{1},\ldots,k_{n} and value vectors v1,…,vnsubscript𝑣1…subscript𝑣𝑛v_{1},\ldots,v_{n}.
Since prompt tokens x1,…,xnsubscript𝑥1…subscript𝑥𝑛x_{1},\ldots,x_{n} are all known, the computation of the prompt phase can be parallelized using matrix-matrix multiplication operations.
Therefore, this phase can efficiently use the parallelism inherent in GPUs.

The autoregressive generation phase generates the remaining new tokens sequentially.
At iteration t𝑡t, the model takes one token xn+tsubscript𝑥𝑛𝑡x_{n+t} as input and computes the probability P​(xn+t+1∣x1,…,xn+t)𝑃conditionalsubscript𝑥𝑛𝑡1subscript𝑥1…subscript𝑥𝑛𝑡P(x_{n+t+1}\mid x_{1},\ldots,x_{n+t}) with the key vectors k1,…,kn+tsubscript𝑘1…subscript𝑘𝑛𝑡k_{1},\ldots,k_{n+t} and value vectors v1,…,vn+tsubscript𝑣1…subscript𝑣𝑛𝑡v_{1},\ldots,v_{n+t}. Note that the key and value vectors at positions 111 to n+t−1𝑛𝑡1n+t-1 are cached at previous iterations, only the new key and value vector kn+tsubscript𝑘𝑛𝑡k_{n+t} and vn+tsubscript𝑣𝑛𝑡v_{n+t} are computed at this iteration. This phase completes either when the sequence reaches a maximum length (specified by users or limited by LLMs) or when an end-of-sequence (¡eos¿) token is emitted.
The computation at different iterations cannot be parallelized due to the data dependency and often uses matrix-vector multiplication, which is less efficient.
As a result, this phase severely underutilizes GPU computation and becomes memory-bound, being responsible for most portion of the latency of a single request.

### 2.3. Batching Techniques for LLMs

The compute utilization in serving LLMs can be improved by batching multiple requests.
Because the requests share the same model weights, the overhead of moving weights is amortized across the requests in a batch, and can be overwhelmed by the computational overhead when the batch size is sufficiently large.
However, batching the requests to an LLM service is non-trivial for two reasons.
First, the requests may arrive at different times.
A naive batching strategy would either make earlier requests wait for later ones or delay the incoming requests until earlier ones finish, leading to significant queueing delays.
Second, the requests may have vastly different input and output lengths (Fig. 11).
A straightforward batching technique would pad the inputs and outputs of the requests to equalize their lengths, wasting GPU computation and memory.

To address this problem, fine-grained batching mechanisms, such as cellular batching (Gao et al., 2018) and iteration-level scheduling (Yu et al., 2022), have been proposed.
Unlike traditional methods that work at the request level, these techniques operate at the iteration level.
After each iteration, completed requests are removed from the batch, and new ones are added.
Therefore, a new request can be processed after waiting for a single iteration, not waiting for the entire batch to complete.
Moreover, with special GPU kernels, these techniques eliminate the need to pad the inputs and outputs.
By reducing the queueing delay and the inefficiencies from padding, the fine-grained batching mechanisms significantly increase the throughput of LLM serving.

## 3. Memory Challenges in LLM Serving

Although fine-grained batching reduces the waste of computing and enables requests to be batched in a more flexible way,
the number of requests that can be batched together is still constrained by GPU memory capacity, particularly the space allocated to store the KV cache.
In other words, the serving system’s throughput is memory-bound.
Overcoming this memory-bound requires addressing the following challenges in the memory management:

Large KV cache. The KV Cache size grows quickly with the number of requests. As an example, for the 13B parameter OPT model (Zhang et al., 2022), the KV cache of a single token demands 800 KB of space, calculated as 222 (key and value vectors) × 5120absent5120\times\ 5120 (hidden state size) × 40absent40\times\ 40 (number of layers) × 2absent2\times\ 2 (bytes per FP16). Since OPT can generate sequences up to 2048 tokens, the memory required to store the KV cache of one request can be as much as 1.6 GB. Concurrent GPUs have memory capacities in the tens of GBs. Even if all available memory was allocated to KV cache, only a few tens of requests could be accommodated. Moreover, inefficient memory management can further decrease the batch size, as shown in Fig. 2.
Additionally, given the current trends, the GPU’s computation speed grows faster than the memory capacity (Gholami et al., 2021). For example, from NVIDIA A100 to H100, The FLOPS increases by more than 2x, but the GPU memory stays at 80GB maximum. Therefore, we believe the memory will become an increasingly significant bottleneck.

Complex decoding algorithms.
LLM services offer a range of decoding algorithms for users to select from, each with varying implications for memory management complexity. For example, when users request multiple random samples from a single input prompt, a typical use case in program suggestion (Github, 2022), the KV cache of the prompt part, which accounts for 12% of the total KV cache memory in our experiment (§6.3), can be shared to minimize memory usage. On the other hand, the KV cache during the autoregressive generation phase should remain unshared due to the different sample results and their dependence on context and position.
The extent of KV cache sharing depends on the specific decoding algorithm employed. In more sophisticated algorithms like beam search (Sutskever et al., 2014), different request beams can share larger portions (up to 55% memory saving, see §6.3) of their KV cache, and the sharing pattern evolves as the decoding process advances.

Scheduling for unknown input & output lengths. The requests to an LLM service exhibit variability in their input and output lengths. This requires the memory management system to accommodate a wide range of prompt lengths. In addition, as the output length of a request grows at decoding, the memory required for its KV cache also expands and may exhaust available memory for incoming requests or ongoing generation for existing prompts. The system needs to make scheduling decisions, such as deleting or swapping out the KV cache of some requests from GPU memory.

### 3.1. Memory Management in Existing Systems

Since most operators in current deep learning frameworks (Paszke et al., 2019; Olston et al., 2017) require tensors to be stored in contiguous memory, previous LLM serving systems (Yu et al., 2022; NVIDIA, 2023a) also store the KV cache of one request as a contiguous tensor across the different positions. Due to the unpredictable output lengths from the LLM, they statically allocate a chunk of memory for a request based on the request’s maximum possible sequence length, irrespective of the actual input or eventual output length of the request.

Fig. 3 illustrates two requests: request A with 2048 maximum possible sequence length and request B with a maximum of 512. The chunk pre-allocation scheme in existing systems has three primary sources of memory wastes: reserved slots for future tokens, internal fragmentation due to over-provisioning for potential maximum sequence lengths, and external fragmentation from the memory allocator like the buddy allocator. The external fragmentation will never be used for generated tokens, which is known before serving a request. Internal fragmentation also remains unused, but this is only realized after a request has finished sampling. They are both pure memory waste.
Although the reserved memory is eventually used, reserving this space for the entire request’s duration, especially when the reserved space is large, occupies the space that could otherwise be used to process other requests. We visualize the average percentage of memory wastes in our experiments in Fig. 2, revealing that the actual effective memory in previous systems can be as low as 20.4%.

Although compaction (Wang et al., 2022b) has been proposed as a potential solution to fragmentation, performing compaction in a performance-sensitive LLM serving system is impractical due to the massive KV cache. Even with compaction, the pre-allocated chunk space for each request prevents memory sharing specific to decoding algorithms in existing memory management systems.

## 4. Method

In this work, we develop a new attention algorithm, PagedAttention, and build an LLM serving engine, vLLM, to tackle the challenges outlined in §3. The architecture of vLLM is shown in Fig. 4. vLLM adopts a centralized scheduler to coordinate the execution of distributed GPU workers. The KV cache manager effectively manages the KV cache in a paged fashion, enabled by PagedAttention. Specifically, the KV cache manager manages the physical KV cache memory on the GPU workers through the instructions sent by the centralized scheduler.

Next, We describe the PagedAttention algorithm in §4.1. With that, we show the design of the KV cache manager in §4.2 and how it facilitates PagedAttention in §4.3, respectively. Then, we show how this design facilitates effective memory management for various decoding methods (§4.4) and handles the variable length input and output sequences (§4.5). Finally, we show how the system design of vLLM works in a distributed setting (§4.6).

### 4.1. PagedAttention

To address the memory challenges in §3, we introduce PagedAttention, an attention algorithm inspired by the classic idea of paging (Kilburn et al., 1962) in operating systems. Unlike the traditional attention algorithms, PagedAttention allows storing continuous keys and values in non-contiguous memory space. Specifically, PagedAttention partitions the KV cache of each sequence into KV blocks. Each block contains the key and value vectors for a fixed number of tokens,111In Transformer, each token has a set of key and value vectors across layers and attention heads within a layer. All the key and value vectors can be managed together within a single KV block, or the key and value vectors at different heads and layers can each have a separate block and be managed in separate block tables. The two designs have no performance difference and we choose the second one for easy implementation. which we denote as KV block size (B𝐵B). Denote the key block Kj=(k(j−1)​B+1,…,kj​B)subscript𝐾𝑗subscript𝑘𝑗1𝐵1…subscript𝑘𝑗𝐵K_{j}=(k_{(j-1)B+1},\ldots,k_{jB}) and value block Vj=(v(j−1)​B+1,…,vj​B).subscript𝑉𝑗subscript𝑣𝑗1𝐵1…subscript𝑣𝑗𝐵V_{j}=(v_{(j-1)B+1},\ldots,v_{jB}). The attention computation in Eq. 4 can be transformed into the following block-wise computation:

(4)

Ai​j=exp⁡(qi⊤​Kj/d)∑t=1⌈i/B⌉exp⁡(qi⊤​Kt​𝟏/d),oi=∑j=1⌈i/B⌉Vj​Ai​j⊤,formulae-sequencesubscript𝐴𝑖𝑗superscriptsubscript𝑞𝑖topsubscript𝐾𝑗𝑑superscriptsubscript𝑡1𝑖𝐵superscriptsubscript𝑞𝑖topsubscript𝐾𝑡1𝑑subscript𝑜𝑖superscriptsubscript𝑗1𝑖𝐵subscript𝑉𝑗superscriptsubscript𝐴𝑖𝑗topA_{ij}=\frac{\exp(q_{i}^{\top}K_{j}/\sqrt{d})}{\sum_{t=1}^{\lceil i/B\rceil}\exp(q_{i}^{\top}K_{t}\mathbf{1}/\sqrt{d})},\ o_{i}=\sum_{j=1}^{\lceil i/B\rceil}V_{j}A_{ij}^{\top},

where Ai​j=(ai,(j−1)​B+1,…,ai,j​B)subscript𝐴𝑖𝑗subscript𝑎𝑖𝑗1𝐵1…subscript𝑎𝑖𝑗𝐵A_{ij}=(a_{i,(j-1)B+1},\ldots,a_{i,jB}) is the row vector of attention score on j𝑗j-th KV block.

During the attention computation, the PagedAttention kernel identifies and fetches different KV blocks separately. We show an example of PagedAttention in Fig. 5: The key and value vectors are spread across three blocks, and the three blocks are not contiguous on the physical memory. At each time, the kernel multiplies the query vector qisubscript𝑞𝑖q_{i} of the query token (“forth”) and the key vectors Kjsubscript𝐾𝑗K_{j} in a block (e.g., key vectors of “Four score and seven” for block 0) to compute the attention score Ai​j,subscript𝐴𝑖𝑗A_{ij}, and later multiplies Ai​jsubscript𝐴𝑖𝑗A_{ij} with the value vectors Vjsubscript𝑉𝑗V_{j} in a block to derive the final attention output oi.subscript𝑜𝑖o_{i}.

In summary, the PagedAttention algorithm allows the KV blocks to be stored in non-contiguous physical memory, which enables more flexible paged memory management in vLLM.

### 4.2. KV Cache Manager

The key idea behind vLLM’s memory manager is analogous to the virtual memory (Kilburn et al., 1962) in operating systems. OS partitions memory into fixed-sized pages and maps user programs’ logical pages to physical pages. Contiguous logical pages can correspond to non-contiguous physical memory pages, allowing user programs to access memory as though it were contiguous. Moreover, physical memory space needs not to be fully reserved in advance, enabling the OS to dynamically allocate physical pages as needed. vLLM uses the ideas behind virtual memory to manage the KV cache in an LLM service. Enabled by PagedAttention, we organize the KV cache as fixed-size KV blocks, like pages in virtual memory.

A request’s KV cache is represented as a series of logical KV blocks, filled from left to right as new tokens and their KV cache are generated.
The last KV block’s unfilled positions are reserved for future generations.
On GPU workers, a block engine allocates a contiguous chunk of GPU DRAM and divides it into physical KV blocks (this is also done on CPU RAM for swapping; see §4.5). The KV block manager also maintains block tables—the mapping between logical and physical KV blocks of each request. Each block table entry records the corresponding physical blocks of a logical block and the number of filled positions. Separating logical and physical KV blocks allows vLLM to dynamically grow the KV cache memory without reserving it for all positions in advance, which eliminates most memory waste in existing systems, as in Fig. 2.

### 4.3. Decoding with PagedAttention and vLLM

Next, we walk through an example, as in Fig. 6, to demonstrate how vLLM executes PagedAttention and manages the memory during the decoding process of a single input sequence: \small1⃝ As in OS’s virtual memory, vLLM does not require reserving the memory for the maximum possible generated sequence length initially.
Instead, it reserves only the necessary KV blocks to accommodate the KV cache generated during prompt computation. In this case, The prompt has 7 tokens, so vLLM maps the first 2 logical KV blocks (0 and 1) to 2 physical KV blocks (7 and 1, respectively). In the prefill step, vLLM generates the KV cache of the prompts and the first output token with a conventional self-attention algorithm (e.g., (Dao et al., 2022)). vLLM then stores the KV cache of the first 4 tokens in logical block 0 and the following 3 tokens in logical block 1. The remaining slot is reserved for the subsequent autoregressive generation phase.
\small2⃝ In the first autoregressive decoding step, vLLM generates the new token with the PagedAttention algorithm on physical blocks 7 and 1. Since one slot remains available in the last logical block, the newly generated KV cache is stored there, and the block table’s #filled record is updated. \small3⃝ At the second decoding step, as the last logical block is full, vLLM stores the newly generated KV cache in a new logical block; vLLM allocates a new physical block (physical block 3) for it and stores this mapping in the block table.

Globally, for each decoding iteration, vLLM first selects a set of candidate sequences for batching (more in §4.5), and allocates the physical blocks for the newly required logical blocks. Then, vLLM concatenates all the input tokens of the current iteration (i.e., all tokens for prompt phase requests and the latest tokens for generation phase requests) as one sequence and feeds it into the LLM. During LLM’s computation, vLLM uses the PagedAttention kernel to access the previous KV cache stored in the form of logical KV blocks and saves the newly generated KV cache into the physical KV blocks.
Storing multiple tokens within a KV block (block size ¿ 1) enables the PagedAttention kernel to process the KV cache across more positions in parallel, thus increasing the hardware utilization and reducing latency. However, a larger block size also increases memory fragmentation. We study the effect of block size in §7.2.

Again, vLLM dynamically assigns new physical blocks to logical blocks as more tokens and their KV cache are generated. As all the blocks are filled from left to right and a new physical block is only allocated when all previous blocks are full, vLLM limits all the memory wastes for a request within one block, so it can effectively utilize all the memory, as shown in Fig. 2. This allows more requests to fit into memory for batching—hence improving the throughput. Once a request finishes its generation, its KV blocks can be freed to store the KV cache of other requests.
In Fig. 7, we show an example of vLLM managing the memory for two sequences. The logical blocks of the two sequences are mapped to different physical blocks within the space reserved by the block engine in GPU workers. The neighboring logical blocks of both sequences do not need to be contiguous in physical GPU memory and the space of physical blocks can be effectively utilized by both sequences.

### 4.4. Application to Other Decoding Scenarios

§4.3 shows how PagedAttention and vLLM handle basic decoding algorithms, such as greedy decoding and sampling, that take one user prompt as input and generate a single output sequence. In many successful LLM applications (Github, 2022; OpenAI, 2020), an LLM service must offer more complex decoding scenarios that exhibit complex accessing patterns and more opportunities for memory sharing. We show the general applicability of vLLM on them in this section.

Parallel sampling.
In LLM-based program assistants (Chen et al., 2021; Github, 2022), an LLM generates multiple sampled outputs for a single input prompt; users can choose a favorite output from various candidates.
So far we have implicitly assumed that a request generates a single sequence. In the remainder of this paper, we assume the more general case in which a request generates multiple sequences.
In parallel sampling, one request includes multiple samples sharing the same input prompt, allowing the KV cache of the prompt to be shared as well. Via its PagedAttention and paged memory management, vLLM can realize this sharing easily and save memory.

Fig. 8 shows an example of parallel decoding for two outputs.
Since both outputs share the same prompt, we only reserve space for one copy of the prompt’s state at the prompt phase; the logical blocks for the prompts of both sequences are mapped to the same physical blocks: the logical block 0 and 1 of both sequences are mapped to physical blocks 7 and 1, respectively.
Since a single physical block can be mapped to multiple logical blocks, we introduce a reference count for each physical block. In this case, the reference counts for physical blocks 7 and 1 are both 2. At the generation phase, the two outputs sample different output tokens and need separate storage for KV cache.
vLLM implements a copy-on-write mechanism at the block granularity for the physical blocks that need modification by multiple sequences, similar to the copy-on-write technique in OS virtual memory (e.g., when forking a process).
Specifically, in Fig. 8, when sample A1 needs to write to its last logical block (logical block 1), vLLM recognizes that the reference count of the corresponding physical block (physical block 1) is greater than 1; it allocates a new physical block (physical block 3), instructs the block engine to copy the information from physical block 1, and decreases the reference count to 1. Next, when sample A2 writes to physical block 1, the reference count is already reduced to 1; thus A2 directly writes its newly generated KV cache to physical block 1.

In summary, vLLM enables the sharing of most of the space used to store the prompts’ KV cache across multiple output samples, with the exception of the final logical block, which is managed by a copy-on-write mechanism. By sharing physical blocks across multiple samples, memory usage can be greatly reduced, especially for long input prompts.

Beam search. In LLM tasks like machine translation (Wu et al., 2016), the users expect the top-k𝑘k most appropriate translations output by the LLM. Beam search (Sutskever et al., 2014) is widely used to decode the most probable output sequence from an LLM, as it mitigates the computational complexity of fully traversing the sample space.
The algorithm relies on the beam width parameter k𝑘k, which determines the number of top candidates retained at every step. During decoding, beam search expands each candidate sequence in the beam by considering all possible tokens, computes their respective probabilities using the LLM, and retains the top-k𝑘k most probable sequences out of k⋅|V|⋅𝑘𝑉k\cdot|V| candidates, where |V|𝑉|V| is the vocabulary size.

Unlike parallel decoding, beam search facilities sharing not only the initial prompt blocks but also other blocks across different candidates, and the sharing patterns dynamically change as the decoding process advances, similar to the process tree in the OS created by compound forks.
Fig. 9 shows how vLLM manages the KV blocks for a beam search example with k=4𝑘4k=4. Prior to the iteration illustrated as the dotted line, each candidate sequence has used 4 full logical blocks. All beam candidates share the first block 0 (i.e., prompt). Candidate 3 digresses from others from the second block. Candidates 0-2 share the first 3 blocks and diverge at the fourth block. At subsequent iterations, the top-4 probable candidates all originate from candidates 1 and 2. As the original candidates 0 and 3 are no longer among the top candidates, their logical blocks are freed, and the reference counts of corresponding physical blocks are reduced. vLLM frees all physical blocks whose reference counts reach 0 (blocks 2, 4, 5, 8). Then, vLLM allocates new physical blocks (blocks 9-12) to store the new KV cache from the new candidates. Now, all candidates share blocks 0, 1, 3; candidates 0 and 1 share block 6, and candidates 2 and 3 further share block 7.

Previous LLM serving systems require frequent memory copies of the KV cache across the beam candidates. For example, in the case shown in Fig. 9, after the dotted line, candidate 3 would need to copy a large portion of candidate 2’s KV cache to continue generation. This frequent memory copy overhead is significantly reduced by vLLM’s physical block sharing. In vLLM, most blocks of different beam candidates can be shared. The copy-on-write mechanism is applied only when the newly generated tokens are within an old shared block, as in parallel decoding. This involves only copying one block of data.

Shared prefix.
Commonly, the LLM user provides a (long) description of the task including instructions and example inputs and outputs, also known as system prompt (OpenAI, 2023a).
The description is concatenated with the actual task input to form the prompt of the request. The LLM generates outputs based on the full prompt. Fig. 10 shows an example. Moreover, the shared prefix can be further tuned, via prompt engineering, to improve the accuracy of the downstream tasks (Li and Liang, 2021; Lester et al., 2021).

For this type of application, many user prompts share a prefix, thus the LLM service provider can store the KV cache of the prefix in advance to reduce the redundant computation spent on the prefix. In vLLM, this can be conveniently achieved by reserving a set of physical blocks for a set of predefined shared prefixes by the LLM service provider, as how OS handles shared library across processes. A user input prompt with the shared prefix can simply map its logical blocks to the cached physical blocks (with the last block marked copy-on-write). The prompt phase computation only needs to execute on the user’s task input.

Mixed decoding methods.
The decoding methods discussed earlier exhibit diverse memory sharing and accessing patterns. Nonetheless, vLLM facilitates the simultaneous processing of requests with different decoding preferences, which existing systems cannot efficiently do.
This is because vLLM conceals the complex memory sharing between different sequences via a common mapping layer that translates logical blocks to physical blocks. The LLM and its execution kernel only see a list of physical block IDs for each sequence and do not need to handle sharing patterns across sequences. Compared to existing systems, this approach broadens the batching opportunities for requests with different sampling requirements, ultimately increasing the system’s overall throughput.

### 4.5. Scheduling and Preemption

When the request traffic surpasses the system’s capacity, vLLM must prioritize a subset of requests. In vLLM, we adopt the first-come-first-serve (FCFS) scheduling
policy for all requests, ensuring fairness and preventing starvation. When vLLM needs to preempt requests, it ensures
that the earliest arrived requests are served first and the
latest requests are preempted first.

LLM services face a unique challenge: the input prompts for an LLM can vary significantly in length, and the resulting output lengths are not known a priori, contingent on both the input prompt and the model. As the number of requests and their outputs grow, vLLM can run out of the GPU’s physical blocks to store the newly generated KV cache. There are two classic questions that vLLM needs to answer in this context: (1) Which blocks should it evict? (2) How to recover evicted blocks if needed again? Typically, eviction policies use heuristics to predict which block will be accessed furthest in the future and evict that block. Since in our case we know that all blocks of a sequence are accessed together, we implement an all-or-nothing eviction policy, i.e., either evict all or none of the blocks of a sequence. Furthermore, multiple sequences within one request (e.g., beam candidates in one beam search request) are gang-scheduled as a sequence group. The sequences within one sequence group are always preempted or rescheduled together due to potential memory sharing across those sequences.
To answer the second question of how to recover an evicted block, we consider two techniques:

Swapping. This is the classic technique used by most virtual memory implementations which copy the evicted pages to a swap space on the disk. In our case, we copy evicted blocks to the CPU memory. As shown in Fig. 4, besides the GPU block allocator, vLLM includes a CPU block allocator to manage the physical blocks swapped to CPU RAM. When vLLM exhausts free physical blocks for new tokens, it selects a set of sequences to evict and transfer their KV cache to the CPU. Once it preempts a sequence and evicts its blocks, vLLM stops accepting new requests until all preempted sequences are completed. Once a request completes, its blocks are freed from memory, and the blocks of a preempted sequence are brought back in to continue the processing of that sequence.
Note that with this design, the number of blocks swapped to the CPU RAM never exceeds the number of total physical blocks in the GPU RAM, so the swap space on the CPU RAM is bounded by the GPU memory allocated for the KV cache.

Recomputation. In this case, we simply recompute the KV cache when the preempted sequences are rescheduled. Note that recomputation latency can be significantly lower than the original latency, as the tokens generated at decoding can be concatenated with the original user prompt as a new prompt—their KV cache at all positions can be generated in one prompt phase iteration.

The performances of swapping and recomputation depend on the bandwidth between CPU RAM and GPU memory and the computation power of the GPU. We examine the speeds of swapping and recomputation in §7.3.

### 4.6. Distributed Execution

Many LLMs have parameter sizes exceeding the capacity of a single GPU (Brown et al., 2020; Chowdhery et al., 2022). Therefore, it is necessary to partition them across distributed GPUs and execute them in a model parallel fashion (Zheng et al., 2022; Li et al., 2023). This calls for a memory manager capable of handling distributed memory. vLLM is effective in distributed settings by supporting the widely used Megatron-LM style tensor model parallelism strategy on Transformers (Shoeybi et al., 2019).
This strategy adheres to an SPMD (Single Program Multiple Data) execution schedule, wherein the linear layers are partitioned to perform block-wise matrix multiplication, and the the GPUs constantly synchronize intermediate results via an all-reduce operation. Specifically, the attention operator is split on the attention head dimension, each SPMD process takes care of a subset of attention heads in multi-head attention.

We observe that even with model parallel execution, each model shard still processes the same set of input tokens, thus requiring the KV Cache for the same positions. Therefore, vLLM features a single KV cache manager within the centralized scheduler, as in Fig. 4. Different GPU workers share the manager, as well as the mapping from logical blocks to physical blocks.
This common mapping allows GPU workers to execute the model with the physical blocks provided by the scheduler for each input request.
Although each GPU worker has the same physical block IDs, a worker only stores a portion of the KV cache for its corresponding attention heads.

In each step, the scheduler first prepares the message with input token IDs for each request in the batch, as well as the block table for each request. Next, the scheduler broadcasts this control message to the GPU workers. Then, the GPU workers start to execute the model with the input token IDs. In the attention layers, the GPU workers read the KV cache according to the block table in the control message. During execution, the GPU workers synchronize the intermediate results with the all-reduce communication primitive without the coordination of the scheduler, as in (Shoeybi et al., 2019). In the end, the GPU workers send the sampled tokens of this iteration back to the scheduler. In summary, GPU workers do not need to synchronize on memory management as they only need to receive all the memory management information at the beginning of each decoding iteration along with the step inputs.

## 5. Implementation

Model size
13B
66B
175B

GPUs
A100

4×\timesA100

8×\timesA100-80GB

Total GPU memory
40 GB
160 GB
640 GB

Parameter size
26 GB
132 GB
346 GB

Memory for KV cache
12 GB
21 GB
264 GB

Max. # KV cache slots
15.7K
9.7K
60.1K

vLLM is an end-to-end serving system with a FastAPI (FastAPI, 2023) frontend and a GPU-based inference engine.
The frontend extends the OpenAI API (OpenAI, 2020) interface, allowing users to customize sampling parameters for each request, such as the maximum sequence length and the beam width k𝑘k.
The vLLM engine is written in 8.5K lines of Python and 2K lines of C++/CUDA code.
We develop control-related components including the scheduler and the block manager in Python while developing custom CUDA kernels for key operations such as PagedAttention.
For the model executor, we implement popular LLMs such as GPT (Brown et al., 2020), OPT (Zhang et al., 2022), and LLaMA (Touvron et al., 2023) using PyTorch (Paszke et al., 2019) and Transformers (Wolf et al., 2020).
We use NCCL (NVIDIA, 2023b) for tensor communication across the distributed GPU workers.

### 5.1. Kernel-level Optimization

Since PagedAttention introduces memory access patterns that are not efficiently supported by existing systems, we develop several GPU kernels for optimizing it.
(1) Fused reshape and block write.
In every Transformer layer, the new KV cache are split into blocks, reshaped to a memory layout optimized for block read, then saved at positions specified by the block table. To minimize kernel launch overheads, we fuse them into a single kernel.
(2) Fusing block read and attention.
We adapt the attention kernel in FasterTransformer (NVIDIA, 2023a) to read KV cache according to the block table and perform attention operations on the fly.
To ensure coalesced memory access, we assign a GPU warp to read each block.
Moreover, we add support for variable sequence lengths within a request batch.
(3) Fused block copy.
Block copy operations, issued by the copy-on-write mechanism, may operate on discontinuous blocks.
This can lead to numerous invocations of small data movements if we use the cudaMemcpyAsync API.
To mitigate the overhead, we implement a kernel that batches the copy operations for different blocks into a single kernel launch.

(a) ShareGPT

(b) Alpaca

### 5.2. Supporting Various Decoding Algorithms

vLLM implements various decoding algorithms using three key methods: fork, append, and free.
The fork method creates a new sequence from an existing one.
The append method appends a new token to the sequence.
Finally, the free method deletes the sequence.
For instance, in parallel sampling, vLLM creates multiple output sequences from the single input sequence using the fork method.
It then adds new tokens to these sequences in every iteration with append, and deletes sequences that meet a stopping condition using free.
The same strategy is also applied in beam search and prefix sharing by vLLM.
We believe future decoding algorithms can also be supported by combining these methods.

(a) ShareGPT

(b) Alpaca

## 6. Evaluation

In this section, we evaluate the performance of vLLM under a variety of workloads.

### 6.1. Experimental Setup

Model and server configurations.
We use OPT (Zhang et al., 2022) models with 13B, 66B, and 175B parameters and LLaMA (Touvron et al., 2023) with 13B parameters for our evaluation.
13B and 66B are popular sizes for LLMs as shown in an LLM leaderboard (ORG, 2023), while 175B is the size of the famous GPT-3 (Brown et al., 2020) model.
For all of our experiments, we use A2 instances with NVIDIA A100 GPUs on Google Cloud Platform.
The detailed model sizes and server configurations are shown in Table 1.

Workloads.
We synthesize workloads based on ShareGPT (Team, 2023) and Alpaca (Taori et al., 2023) datasets, which contain input and output texts of real LLM services.
The ShareGPT dataset is a collection of user-shared conversations with ChatGPT (OpenAI, 2022).
The Alpaca dataset is an instruction dataset generated by GPT-3.5 with self-instruct (Wang et al., 2022a).
We tokenize the datasets and use their input and output lengths to synthesize client requests.
As shown in Fig. 11, the ShareGPT dataset has 8.4×\times longer input prompts and 5.8×\times longer outputs on average than the Alpaca dataset, with higher variance.
Since these datasets do not include timestamps, we generate request arrival times using Poisson distribution with different request rates.

Baseline 1: FasterTransformer.
FasterTransformer (NVIDIA, 2023a) is a distributed inference engine highly optimized for latency.
As FasterTransformer does not have its own scheduler, we implement a custom scheduler with a dynamic batching mechanism similar to the existing serving systems such as Triton (NVIDIA, [n. d.]).
Specifically, we set a maximum batch size B𝐵B as large as possible for each experiment, according to the GPU memory capacity.
The scheduler takes up to B𝐵B number of earliest arrived requests and sends the batch to FasterTransformer for processing.

Baseline 2: Orca.
Orca (Yu et al., 2022) is a state-of-the-art LLM serving system optimized for throughput.
Since Orca is not publicly available for use, we implement our own version of Orca.
We assume Orca uses the buddy allocation algorithm to determine the memory address to store KV cache.
We implement three versions of Orca based on how much it over-reserves the space for request outputs:

- •

Orca (Oracle). We assume the system has the knowledge of the lengths of the outputs that will be actually generated for the requests. This shows the upper-bound performance of Orca, which is infeasible to achieve in practice.

- •

Orca (Pow2). We assume the system over-reserves the space for outputs by at most 2×\times. For example, if the true output length is 25, it reserves 32 positions for outputs.

- •

Orca (Max). We assume the system always reserves the space up to the maximum sequence length of the model, i.e., 2048 tokens.

Key metrics.
We focus on serving throughput.
Specifically, using the workloads with different request rates, we measure normalized latency of the systems, the mean of every request’s end-to-end latency divided by its output length, as in Orca (Yu et al., 2022).
A high-throughput serving system should retain low normalized latency against high request rates.
For most experiments, we evaluate the systems with 1-hour traces.
As an exception, we use 15-minute traces for the OPT-175B model due to the cost limit.

### 6.2. Basic Sampling

We evaluate the performance of vLLM with basic sampling (one sample per request) on three models and two datasets.
The first row of Fig. 12 shows the results on the ShareGPT dataset.
The curves illustrate that as the request rate increases, the latency initially increases at a gradual pace but then suddenly explodes.
This can be attributed to the fact that when the request rate surpasses the capacity of the serving system, the queue length continues to grow infinitely and so does the latency of the requests.

On the ShareGPT dataset, vLLM can sustain 1.7×1.7\times–2.7×2.7\times higher request rates compared to Orca (Oracle) and 2.7×2.7\times–8×8\times compared to Orca (Max), while maintaining similar latencies.
This is because vLLM’s PagedAttention can efficiently manage the memory usage and thus enable batching more requests than Orca.
For example, as shown in Fig. 13(a), for OPT-13B vLLM processes 2.2×2.2\times more requests at the same time than Orca (Oracle) and 4.3×4.3\times more requests than Orca (Max).
Compared to FasterTransformer, vLLM can sustain up to 22×22\times higher request rates, as FasterTransformer does not utilize a fine-grained scheduling mechanism and inefficiently manages the memory like Orca (Max).

The second row of Fig. 12 and Fig. 13(b) shows the results on the Alpaca dataset, which follows a similar trend to the ShareGPT dataset.
One exception is Fig. 12 (f), where vLLM’s advantage over Orca (Oracle) and Orca (Pow2) is less pronounced.
This is because the model and server configuration for OPT-175B (Table 1) allows for large GPU memory space available to store KV cache, while the Alpaca dataset has short sequences.
In this setup, Orca (Oracle) and Orca (Pow2) can also batch a large number of requests despite the inefficiencies in their memory management.
As a result, the performance of the systems becomes compute-bound rather than memory-bound.

(a) Parallel sampling

(b) Beam search

### 6.3. Parallel Sampling and Beam Search

We evaluate the effectiveness of memory sharing in PagedAttention with two popular sampling methods: parallel sampling and beam search.
In parallel sampling, all parallel sequences in a request can share the KV cache for the prompt.
As shown in the first row of Fig. 14, with a larger number of sequences to sample, vLLM brings more improvement over the Orca baselines.
Similarly, the second row of Fig. 14 shows the results for beam search with different beam widths.
Since beam search allows for more sharing, vLLM demonstrates even greater performance benefits.
The improvement of vLLM over Orca (Oracle) on OPT-13B and the Alpaca dataset goes from 1.3×1.3\times in basic sampling to 2.3×2.3\times in beam search with a width of 6.

Fig. 15 plots the amount of memory saving, computed by the number of blocks we saved by sharing divided by the number of total blocks without sharing. We show 6.1% - 9.8% memory saving on parallel sampling and 37.6% - 55.2% on beam search.
In the same experiments with the ShareGPT dataset, we saw 16.2% - 30.5% memory saving on parallel sampling and 44.3% - 66.3% on beam search.

### 6.4. Shared prefix

We explore the effectiveness of vLLM for the case a prefix is shared among different input prompts, as illustrated in Fig. 10.
For the model, we use LLaMA-13B (Touvron et al., 2023), which is multilingual.
For the workload, we use the WMT16 (Bojar et al., 2016) English-to-German translation dataset and synthesize two prefixes that include an instruction and a few translation examples.
The first prefix includes a single example (i.e., one-shot) while the other prefix includes 5 examples (i.e., few-shot).
As shown in Fig. 16 (a), vLLM achieves 1.67×1.67\times higher throughput than Orca (Oracle) when the one-shot prefix is shared.
Furthermore, when more examples are shared (Fig. 16 (b)), vLLM achieves 3.58×3.58\times higher throughput than Orca (Oracle).

### 6.5. Chatbot

A chatbot (OpenAI, 2022; Google, 2023; Chiang et al., 2023) is one of the most important applications of LLMs.
To implement a chatbot, we let the model generate a response by concatenating the chatting history and the last user query into a prompt.
We synthesize the chatting history and user query using the ShareGPT dataset.
Due to the limited context length of the OPT-13B model, we cut the prompt to the last 1024 tokens and let the model generate at most 1024 tokens.
We do not store the KV cache between different conversation rounds as doing this would occupy the space for other requests between the conversation rounds.

Fig. 17 shows that vLLM can sustain 2×2\times higher request rates compared to the three Orca baselines.
Since the ShareGPT dataset contains many long conversations, the input prompts for most requests have 1024 tokens.
Due to the buddy allocation algorithm, the Orca baselines reserve the space for 1024 tokens for the request outputs, regardless of how they predict the output lengths.
For this reason, the three Orca baselines behave similarly.
In contrast, vLLM can effectively handle the long prompts, as PagedAttention resolves the problem of memory fragmentation and reservation.

## 7. Ablation Studies

In this section, we study various aspects of vLLM and evaluate the design choices we make with ablation experiments.

### 7.1. Kernel Microbenchmark

The dynamic block mapping in PagedAttention affects the performance of the GPU operations involving the stored KV cache, i.e., block read/writes and attention.
Compared to the existing systems, our GPU kernels (§5) involve extra overheads of accessing the block table, executing extra branches, and handling variable sequence lengths.
As shown in Fig. 18(a), this leads to 20–26% higher attention kernel latency, compared to the highly-optimized FasterTransformer implementation.
We believe the overhead is small as it only affects the attention operator but not the other operators in the model, such as Linear.
Despite the overhead, PagedAttention makes vLLM significantly outperform FasterTransformer in end-to-end performance (§6).

(a) Latency of attention kernels.

(b) End-to-end latency with different block sizes.

### 7.2. Impact of Block Size

The choice of block size can have a substantial impact on the performance of vLLM.
If the block size is too small, vLLM may not fully utilize the GPU’s parallelism for reading and processing KV cache.
If the block size is too large, internal fragmentation increases and the probability of sharing decreases.

In Fig. 18(b), we evaluate the performance of vLLM with different block sizes, using the ShareGPT and Alpaca traces with basic sampling under fixed request rates.
In the ShareGPT trace, block sizes from 16 to 128 lead to the best performance.
In the Alpaca trace, while the block size 16 and 32 work well, larger block sizes significantly degrade the performance since the sequences become shorter than the block sizes.
In practice, we find that the block size 16 is large enough to efficiently utilize the GPU and small enough to avoid significant internal fragmentation in most workloads.
Accordingly, vLLM sets its default block size as 16.

(a) Microbenchmark

(b) End-to-end performance

### 7.3. Comparing Recomputation and Swapping

vLLM supports both recomputation and swapping as its recovery mechanisms.
To understand the tradeoffs between the two methods, we evaluate their end-to-end performance and microbenchmark their overheads, as presented in Fig. 19.
Our results reveal that swapping incurs excessive overhead with small block sizes.
This is because small block sizes often result in numerous small data transfers between CPU and GPU, which limits the effective PCIe bandwidth.
In contrast, the overhead of recomputation remains constant across different block sizes, as recomputation does not utilize the KV blocks.
Thus, recomputation is more efficient when the block size is small, while swapping is more efficient when the block size is large, though recomputation overhead is never higher than 20% of swapping’s latency.
For medium block sizes from 16 to 64, the two methods exhibit comparable end-to-end performance.

## 8. Discussion

Applying the virtual memory and paging technique to other GPU workloads.
The idea of virtual memory and paging is effective for managing the KV cache in LLM serving because the workload requires dynamic memory allocation (since the output length is not known a priori) and its performance is bound by the GPU memory capacity.
However, this does not generally hold for every GPU workload.
For example, in DNN training, the tensor shapes are typically static, and thus memory allocation can be optimized ahead of time.
For another example, in serving DNNs that are not LLMs, an increase in memory efficiency may not result in any performance improvement since the performance is primarily compute-bound.
In such scenarios, introducing the vLLM’s techniques may rather degrade the performance due to the extra overhead of memory indirection and non-contiguous block memory.
However, we would be excited to see vLLM’s techniques being applied to other workloads with similar properties to LLM serving.

LLM-specific optimizations in applying virtual memory and paging.
vLLM re-interprets and augments the idea of virtual memory and paging by leveraging the application-specific semantics.
One example is vLLM’s all-or-nothing swap-out policy, which exploits the fact that processing a request requires all of its corresponding token states to be stored in GPU memory.
Another example is the recomputation method to recover the evicted blocks, which is not feasible in OS.
Besides, vLLM mitigates the overhead of memory indirection in paging by fusing the GPU kernels for memory access operations with those for other operations such as attention.

## 9. Related Work

General model serving systems.
Model serving has been an active area of research in recent years, with numerous systems proposed to tackle diverse aspects of deep learning model deployment.
Clipper (Crankshaw et al., 2017), TensorFlow Serving (Olston et al., 2017), Nexus (Shen et al., 2019), InferLine (Crankshaw et al., 2020), and
Clockwork (Gujarati et al., 2020) are some earlier general model serving systems. They study batching, caching, placement, and scheduling for serving single or multiple models.
More recently, DVABatch (Cui et al., 2022) introduces multi-entry multi-exit batching. REEF (Han et al., 2022) and Shepherd (Zhang et al., 2023) propose preemption for serving. AlpaServe (Li et al., 2023) utilizes model parallelism for statistical multiplexing.
However, these general systems fail to take into account the auto-regressive property and token state of LLM inference, resulting in missed opportunities for optimization.

Specialized serving systems for transformers.
Due to the significance of the transformer architecture, numerous specialized serving systems for it have been developed. These systems utilize GPU kernel optimizations (Wang et al., 2021; Aminabadi et al., 2022; NVIDIA, 2023a; Ma et al., 2020),
advanced batching mechanisms (Fang et al., 2021; Yu et al., 2022),
model parallelism (Pope et al., 2022; Yu et al., 2022; Aminabadi et al., 2022), and parameter sharing (Zhou et al., 2022) for efficient serving.
Among them, Orca (Yu et al., 2022) is most relevant to our approach.

Comparison to Orca.
The iteration-level scheduling in Orca (Yu et al., 2022) and PagedAttention in vLLM are complementary techniques: While both systems aim to increase the GPU utilization and hence the throughput of LLM serving, Orca achieves it by scheduling and interleaving the requests so that more requests can be processed in parallel, while vLLM is doing so by increasing memory utilization so that the working sets of more requests fit into memory.
By reducing memory fragmentation and enabling sharing, vLLM runs more requests in a batch in parallel and achieves a 2-4×\times speedup compared to Orca.
Indeed, the fine-grained scheduling and interleaving of the requests like in Orca makes memory management more challenging, making the techniques proposed in vLLM even more crucial.

Memory optimizations.
The widening gap between the compute capability and memory capacity of accelerators has caused memory to become a bottleneck for both training and inference.
Swapping (Huang et al., 2020; Wang et al., 2018; Ren et al., 2021), recomputation (Chen et al., 2016; Jain et al., 2020) and their combination (Patil et al., 2022) have been utilized to reduce the peak memory of training.
Notably, FlexGen (Sheng et al., 2023) studies how to swap weights and token states for LLM inference with limited GPU memory, but it does not target the online serving settings.
OLLA (Steiner et al., 2022) optimizes the lifetime and location of tensors to reduce fragmentation, but it does not do fine-grained block-level management or online serving.
FlashAttention (Dao et al., 2022) applies tiling and kernel optimizations to reduce the peak memory of attention computation and reduce I/O costs.
This paper introduces a new idea of block-level memory management in the context of online serving.

## 10. Conclusion

This paper proposes PagedAttention, a new attention algorithm that allows attention keys and values to be stored in non-contiguous paged memory, and presents vLLM, a high-throughput LLM serving system with efficient memory management enabled by PagedAttention.
Inspired by operating systems, we demonstrate how established techniques, such as virtual memory and copy-on-write, can be adapted to efficiently manage KV cache and handle various decoding algorithms in LLM serving.
Our experiments show that vLLM achieves 2-4×\times throughput improvements over the state-of-the-art systems.

## Acknowledgement

We would like to thank Xiaoxuan Liu, Zhifeng Chen, Yanping Huang, anonymous SOSP reviewers, and our shepherd, Lidong Zhou, for their insightful feedback.
This research is partly supported by gifts from Andreessen Horowitz, Anyscale, Astronomer, Google, IBM, Intel, Lacework, Microsoft, Mohamed Bin Zayed University of Artificial Intelligence, Samsung SDS, Uber, and VMware.

## References

- (1)

- Aminabadi et al. (2022)

Reza Yazdani Aminabadi, Samyam Rajbhandari, Minjia Zhang, Ammar Ahmad Awan, Cheng Li, Du Li, Elton Zheng, Jeff Rasley, Shaden Smith, Olatunji Ruwase, et al. 2022.

DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale.

arXiv preprint arXiv:2207.00032 (2022).

- Ba et al. (2016)

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. 2016.

Layer normalization.

arXiv preprint arXiv:1607.06450 (2016).

- Bengio et al. (2000)

Yoshua Bengio, Réjean Ducharme, and Pascal Vincent. 2000.

A neural probabilistic language model.

Advances in neural information processing systems 13 (2000).

- Bojar et al. (2016)

Ond rej Bojar, Rajen Chatterjee, Christian Federmann, Yvette Graham, Barry Haddow, Matthias Huck, Antonio Jimeno Yepes, Philipp Koehn, Varvara Logacheva, Christof Monz, Matteo Negri, Aurelie Neveol, Mariana Neves, Martin Popel, Matt Post, Raphael Rubino, Carolina Scarton, Lucia Specia, Marco Turchi, Karin Verspoor, and Marcos Zampieri. 2016.

Findings of the 2016 Conference on Machine Translation. In Proceedings of the First Conference on Machine Translation. Association for Computational Linguistics, Berlin, Germany, 131–198.

http://www.aclweb.org/anthology/W/W16/W16-2301

- Brown et al. (2020)

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020.

Language models are few-shot learners.

Advances in neural information processing systems 33 (2020), 1877–1901.

- Chen et al. (2021)

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021.

Evaluating large language models trained on code.

arXiv preprint arXiv:2107.03374 (2021).

- Chen et al. (2016)

Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. 2016.

Training deep nets with sublinear memory cost.

arXiv preprint arXiv:1604.06174 (2016).

- Chiang et al. (2023)

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023.

Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality.

https://lmsys.org/blog/2023-03-30-vicuna/

- Chowdhery et al. (2022)

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022.

Palm: Scaling language modeling with pathways.

arXiv preprint arXiv:2204.02311 (2022).

- Crankshaw et al. (2020)

Daniel Crankshaw, Gur-Eyal Sela, Xiangxi Mo, Corey Zumar, Ion Stoica, Joseph Gonzalez, and Alexey Tumanov. 2020.

InferLine: latency-aware provisioning and scaling for prediction serving pipelines. In Proceedings of the 11th ACM Symposium on Cloud Computing. 477–491.

- Crankshaw et al. (2017)

Daniel Crankshaw, Xin Wang, Guilio Zhou, Michael J Franklin, Joseph E Gonzalez, and Ion Stoica. 2017.

Clipper: A Low-Latency Online Prediction Serving System. In 14th USENIX Symposium on Networked Systems Design and Implementation (NSDI 17). 613–627.

- Cui et al. (2022)

Weihao Cui, Han Zhao, Quan Chen, Hao Wei, Zirui Li, Deze Zeng, Chao Li, and Minyi Guo. 2022.

DVABatch: Diversity-aware Multi-Entry Multi-Exit Batching for Efficient Processing of DNN Services on GPUs. In 2022 USENIX Annual Technical Conference (USENIX ATC 22). 183–198.

- Dao et al. (2022)

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022.

Flashattention: Fast and memory-efficient exact attention with io-awareness.

Advances in Neural Information Processing Systems 35 (2022), 16344–16359.

- Fang et al. (2021)

Jiarui Fang, Yang Yu, Chengduo Zhao, and Jie Zhou. 2021.

TurboTransformers: an efficient GPU serving system for transformer models. In Proceedings of the 26th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming. 389–402.

- FastAPI (2023)

FastAPI. 2023.

FastAPI.

https://github.com/tiangolo/fastapi.

- Gao et al. (2018)

Pin Gao, Lingfan Yu, Yongwei Wu, and Jinyang Li. 2018.

Low latency rnn inference with cellular batching. In Proceedings of the Thirteenth EuroSys Conference. 1–15.

- Gholami et al. (2021)

Amir Gholami, Zhewei Yao, Sehoon Kim, Michael W Mahoney, and Kurt Keutzer. 2021.

Ai and memory wall.

RiseLab Medium Post 1 (2021), 6.

- Github (2022)

Github. 2022.

https://github.com/features/copilot

- Google (2023)

Google. 2023.

https://bard.google.com/

- Gujarati et al. (2020)

Arpan Gujarati, Reza Karimi, Safya Alzayat, Wei Hao, Antoine Kaufmann, Ymir Vigfusson, and Jonathan Mace. 2020.

Serving {{\{DNNs}}\} like Clockwork: Performance Predictability from the Bottom Up. In 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20). 443–462.

- Han et al. (2022)

Mingcong Han, Hanze Zhang, Rong Chen, and Haibo Chen. 2022.

Microsecond-scale Preemption for Concurrent {{\{GPU-accelerated}}\}{{\{DNN}}\} Inferences. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22). 539–558.

- He et al. (2016)

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016.

Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition. 770–778.

- Huang et al. (2020)

Chien-Chin Huang, Gu Jin, and Jinyang Li. 2020.

Swapadvisor: Pushing deep learning beyond the gpu memory limit via smart swapping. In Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems. 1341–1355.

- Jain et al. (2020)

Paras Jain, Ajay Jain, Aniruddha Nrusimha, Amir Gholami, Pieter Abbeel, Joseph Gonzalez, Kurt Keutzer, and Ion Stoica. 2020.

Checkmate: Breaking the memory wall with optimal tensor rematerialization.

Proceedings of Machine Learning and Systems 2 (2020), 497–511.

- Kilburn et al. (1962)

Tom Kilburn, David BG Edwards, Michael J Lanigan, and Frank H Sumner. 1962.

One-level storage system.

IRE Transactions on Electronic Computers 2 (1962), 223–235.

- Lester et al. (2021)

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021.

The power of scale for parameter-efficient prompt tuning.

arXiv preprint arXiv:2104.08691 (2021).

- Li and Liang (2021)

Xiang Lisa Li and Percy Liang. 2021.

Prefix-tuning: Optimizing continuous prompts for generation.

arXiv preprint arXiv:2101.00190 (2021).

- Li et al. (2023)

Zhuohan Li, Lianmin Zheng, Yinmin Zhong, Vincent Liu, Ying Sheng, Xin Jin, Yanping Huang, Zhifeng Chen, Hao Zhang, Joseph E Gonzalez, et al. 2023.

AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving.

arXiv preprint arXiv:2302.11665 (2023).

- Ma et al. (2020)

Lingxiao Ma, Zhiqiang Xie, Zhi Yang, Jilong Xue, Youshan Miao, Wei Cui, Wenxiang Hu, Fan Yang, Lintao Zhang, and Lidong Zhou. 2020.

Rammer: Enabling holistic deep learning compiler optimizations with rtasks. In Proceedings of the 14th USENIX Conference on Operating Systems Design and Implementation. 881–897.

- NVIDIA ([n. d.])

NVIDIA. [n. d.].

Triton Inference Server.

https://developer.nvidia.com/nvidia-triton-inference-server.

- NVIDIA (2023a)

NVIDIA. 2023a.

FasterTransformer.

https://github.com/NVIDIA/FasterTransformer.

- NVIDIA (2023b)

NVIDIA. 2023b.

NCCL: The NVIDIA Collective Communication Library.

https://developer.nvidia.com/nccl.

- Olston et al. (2017)

Christopher Olston, Noah Fiedel, Kiril Gorovoy, Jeremiah Harmsen, Li Lao, Fangwei Li, Vinu Rajashekhar, Sukriti Ramesh, and Jordan Soyke. 2017.

Tensorflow-serving: Flexible, high-performance ml serving.

arXiv preprint arXiv:1712.06139 (2017).

- OpenAI (2020)

OpenAI. 2020.

https://openai.com/blog/openai-api

- OpenAI (2022)

OpenAI. 2022.

https://openai.com/blog/chatgpt

- OpenAI (2023a)

OpenAI. 2023a.

https://openai.com/blog/custom-instructions-for-chatgpt

- OpenAI (2023b)

OpenAI. 2023b.

GPT-4 Technical Report.

arXiv:2303.08774 [cs.CL]

- ORG (2023)

LMSYS ORG. 2023.

Chatbot Arena Leaderboard Week 8: Introducing MT-Bench and Vicuna-33B.

https://lmsys.org/blog/2023-06-22-leaderboard/.

- Paszke et al. (2019)

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019.

Pytorch: An imperative style, high-performance deep learning library.

Advances in neural information processing systems 32 (2019).

- Patil et al. (2022)

Shishir G Patil, Paras Jain, Prabal Dutta, Ion Stoica, and Joseph Gonzalez. 2022.

POET: Training Neural Networks on Tiny Devices with Integrated Rematerialization and Paging. In International Conference on Machine Learning. PMLR, 17573–17583.

- Pope et al. (2022)

Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. 2022.

Efficiently Scaling Transformer Inference.

arXiv preprint arXiv:2211.05102 (2022).

- Ren et al. (2021)

Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, and Yuxiong He. 2021.

ZeRO-Offload: Democratizing Billion-Scale Model Training.. In USENIX Annual Technical Conference. 551–564.

- Reuters (2023)

Reuters. 2023.

https://www.reuters.com/technology/tech-giants-ai-like-bing-bard-poses-billion-dollar-search-problem-2023-02-22/

- Services (2023)

Amazon Web Services. 2023.

https://aws.amazon.com/bedrock/

- Shen et al. (2019)

Haichen Shen, Lequn Chen, Yuchen Jin, Liangyu Zhao, Bingyu Kong, Matthai Philipose, Arvind Krishnamurthy, and Ravi Sundaram. 2019.

Nexus: A GPU cluster engine for accelerating DNN-based video analysis. In Proceedings of the 27th ACM Symposium on Operating Systems Principles. 322–337.

- Sheng et al. (2023)

Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Daniel Y Fu, Zhiqiang Xie, Beidi Chen, Clark Barrett, Joseph E Gonzalez, et al. 2023.

High-throughput Generative Inference of Large Language Models with a Single GPU.

arXiv preprint arXiv:2303.06865 (2023).

- Shoeybi et al. (2019)

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. 2019.

Megatron-lm: Training multi-billion parameter language models using model parallelism.

arXiv preprint arXiv:1909.08053 (2019).

- Steiner et al. (2022)

Benoit Steiner, Mostafa Elhoushi, Jacob Kahn, and James Hegarty. 2022.

OLLA: Optimizing the Lifetime and Location of Arrays to Reduce the Memory Usage of Neural Networks.

(2022).

https://doi.org/10.48550/arXiv.2210.12924

- Sutskever et al. (2014)

Ilya Sutskever, Oriol Vinyals, and Quoc V Le. 2014.

Sequence to sequence learning with neural networks.

Advances in neural information processing systems 27 (2014).

- Taori et al. (2023)

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023.

Stanford Alpaca: An Instruction-following LLaMA model.

https://github.com/tatsu-lab/stanford_alpaca.

- Team (2023)

ShareGPT Team. 2023.

https://sharegpt.com/

- Touvron et al. (2023)

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023.

Llama: Open and efficient foundation language models.

arXiv preprint arXiv:2302.13971 (2023).

- Vaswani et al. (2017)

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017.

Attention is all you need.

Advances in neural information processing systems 30 (2017).

- Wang et al. (2022b)

Jing Wang, Youyou Lu, Qing Wang, Minhui Xie, Keji Huang, and Jiwu Shu. 2022b.

Pacman: An Efficient Compaction Approach for {{\{Log-Structured}}\}{{\{Key-Value}}\} Store on Persistent Memory. In 2022 USENIX Annual Technical Conference (USENIX ATC 22). 773–788.

- Wang et al. (2018)

Linnan Wang, Jinmian Ye, Yiyang Zhao, Wei Wu, Ang Li, Shuaiwen Leon Song, Zenglin Xu, and Tim Kraska. 2018.

Superneurons: Dynamic GPU memory management for training deep neural networks. In Proceedings of the 23rd ACM SIGPLAN symposium on principles and practice of parallel programming. 41–53.

- Wang et al. (2021)

Xiaohui Wang, Ying Xiong, Yang Wei, Mingxuan Wang, and Lei Li. 2021.

LightSeq: A High Performance Inference Library for Transformers. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Industry Papers. 113–120.

- Wang et al. (2022a)

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2022a.

Self-Instruct: Aligning Language Model with Self Generated Instructions.

arXiv preprint arXiv:2212.10560 (2022).

- Wolf et al. (2020)

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. 2020.

Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations. 38–45.

- Wu et al. (2016)

Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. 2016.

Google’s neural machine translation system: Bridging the gap between human and machine translation.

arXiv preprint arXiv:1609.08144 (2016).

- Yu et al. (2022)

Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. 2022.

Orca: A Distributed Serving System for {{\{Transformer-Based}}\} Generative Models. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22). 521–538.

- Zhang et al. (2023)

Hong Zhang, Yupeng Tang, Anurag Khandelwal, and Ion Stoica. 2023.

SHEPHERD: Serving DNNs in the Wild. In 20th USENIX Symposium on Networked Systems Design and Implementation (NSDI 23). USENIX Association, Boston, MA, 787–808.

https://www.usenix.org/conference/nsdi23/presentation/zhang-hong

- Zhang et al. (2022)

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022.

Opt: Open pre-trained transformer language models.

arXiv preprint arXiv:2205.01068 (2022).

- Zheng et al. (2022)

Lianmin Zheng, Zhuohan Li, Hao Zhang, Yonghao Zhuang, Zhifeng Chen, Yanping Huang, Yida Wang, Yuanzhong Xu, Danyang Zhuo, Eric P Xing, et al. 2022.

Alpa: Automating Inter-and Intra-Operator Parallelism for Distributed Deep Learning. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22). 559–578.

- Zhou et al. (2022)

Zhe Zhou, Xuechao Wei, Jiejing Zhang, and Guangyu Sun. 2022.

PetS: A Unified Framework for Parameter-Efficient Transformers Serving. In 2022 USENIX Annual Technical Conference (USENIX ATC 22). 489–504.

Generated on Wed Feb 28 06:44:45 2024 by LaTeXML

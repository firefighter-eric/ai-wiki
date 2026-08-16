# Motif 2 12.7B Technical Report

- Source HTML: `raw/html/Lim et al. - 2025 - Motif 2 12.7B Technical Report.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2511.07464
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Motif 2 12.7B technical report

Motif Technologies

###### Abstract

We introduce Motif-2-12.7B, a new open-weight foundation model that pushes the efficiency frontier of large language models by combining architectural innovation with system-level optimization. Designed for scalable language understanding and robust instruction generalization under constrained compute budgets, Motif-2-12.7B builds upon Motif-2.6B with the integration of Grouped Differential Attention (GDA), which improves representational efficiency by disentangling signal and noise-control attention pathways. The model is pre-trained on 5.5 trillion tokens spanning diverse linguistic, mathematical, scientific, and programming domains using a curriculum-driven data scheduler that gradually changes the data composition ratio. The training system leverages the MuonClip optimizer alongside custom high-performance kernels, including fused PolyNorm activations and the Parallel Muon algorithm, yielding significant throughput and memory efficiency gains in large-scale distributed environments. Post-training employs a three-stage supervised fine-tuning pipeline that successively enhances general instruction adherence, compositional understanding, and linguistic precision. Motif-2-12.7B demonstrates competitive performance across diverse benchmarks, showing that thoughtful architectural scaling and optimized training design can rival the capabilities of much larger models.
Hugging Face: https://huggingface.co/Motif-Technologies
Chat service: https://chat.motiftech.io/

## 1 Introduction

Recent years have seen rapid progress in large language models (LLMs), with both proprietary and open-weight systems such as GPT-5 gpt5systemcard, Claude-4 Anthropic2025Claude4SC, Grok-4 xAI_Grok4_2025, Gemini-2.5 comanici2025gemini, Qwen-3 yang2025qwen3, DeepSeek-r1 guo2025deepseek, and Kimi K2 team2025kimi achieving remarkable results across diverse domains, including mathematics, programming, STEM reasoning, medicine, and law. The scale and breadth of these systems have begun to redefine the boundaries of what language models can represent and reason about, signaling a shift from pattern recognition toward more structured and analytical forms of intelligence. Despite these advances, research on competitive capability under constrained model sizes and limited computational resources remains relatively sparse. Understanding how to push the limits of LLMs’ ability within efficiency constraints has emerged as an important challenge for the next generation of language models.

We introduce Motif-2-12.7B(-Base/Instruct), an open-weight foundational language model designed to support instruction following, multi-step reasoning, and general-purpose language understanding.
At its core is Grouped Differential Attention (GDA) lim2025grouped, a new mechanism that reorganizes attention computation to better capture multi-granular token interactions while preserving scalability.
Motif-2-12.7B achieves competitive results across a broad range of reasoning and instruction-following tasks, showing that architectural innovation and effective training strategies can lead to meaningful improvements even without relying on extremely large parameter sizes.
With its open availability and transparent training methodology, Motif-2-12.7B would serve as a strong baseline for future research on scalable attention and efficient reasoning.

Compared to its predecessor Motif-2.6B (lim2025motif), Motif-2-12.7B represents a significant evolution in scale and training methodology. It is obtained by upscaling Motif-2.6B and pre-training on a substantially larger and more diverse corpus, yielding broader linguistic coverage and deeper knowledge representation across general English, Question Answering, chat, multilingual, scientific, mathematical, and coding domains. Most importantly, we introduce Grouped Differential Attention (GDA), which enhances the model’s capacity to capture complex token dependencies and improves efficiency without additional computational cost. Complemented by strengthened data curation (rigorous filtering, deduplication, novel synthetic data generation, and domain-balanced sampling) and an expanded token budget, Motif-2-12.7B exhibits consistently stronger performance on standard evaluations while preserving continuity with Motif-2.6B.

Pre-training of Motif-2-12.7B-Base was conducted over a corpus total 5.5 trillion tokens, designed to support scalable reasoning and generalization. We employed a linear curriculum scheduler to modulate the data composition throughout training. Early phases consisted predominantly of general-domain English text, with a gradual and controlled increase in the proportion of STEM, mathematical, and programming data. This approach encouraged early convergence on core linguistic fluency while promoting progressive adaptation to more structured reasoning tasks. Optimization was carried out using the MuonClip optimizer (team2025kimi), which was chosen for its adaptive gradient scaling properties under large-batch conditions. To fully leverage hardware efficiency, we implemented a set of high-performance parallelism kernels for optimizing MuonClip’s update rule, significantly improves throughput and stability for long-sequence training. This combination of curriculum-aware data scheduling, specialized optimization, and systems-level acceleration enabled us to scale pre-training effectively without compromising convergence quality or compute efficiency.

Following pre-training, Motif-2-12.7B-Instruct further went through a post-training phase consisting of three-stage supervised fine-tuning (SFT) to enhance its general instruction-following and domain-specific abilities. The process began with large-scale alignment on diverse instruction datasets to establish broad conversational competence, then progressively incorporated curated and synthetic data emphasizing compositional reasoning, algorithmic thinking, mathematics, and code generation. In the final refinement stage, selective data pruning was applied to improve diversity and maintain linguistic fluency. Through this gradual and data-driven approach, Motif-2-12.7B-Instruct developed stronger generalization and more coherent reasoning behaviors across a wide range of tasks.

Organization of the technical report.
Section 2 introduces the architectural design of Motif-2-12.7B, detailing its scaling methodology and integration of key components.
Section 3 describes the pretraining data composition, training procedure, and evaluation results of the base model.
Section 4 presents system-level optimizations developed to improve large-scale training efficiency.
Section 5 outlines the three-stage supervised fine-tuning pipeline used to construct the Motif-2-12.7B-Instruct model, along with its corresponding evaluation results.
Finally, Section 6 concludes the report with a summary of findings and future directions.

## 2 Architecture

To expand model capacity while preserving the pretrained representation space, we employed a width expansion strategy that scales internal dimensions without altering the model’s learned function. Motif-2-12.7B was derived from Motif-2.6B through the Scaling Smart Hypercloning procedure samragh2024scaling, in which the model is scaled by an integer multiple through direct weight replication. This method enlarges the network while strictly preserving the original parameter topology and initialization statistics, thereby ensuring functional continuity between generations. Rather than re-initializing or retraining new components from scratch, hypercloning reproduces the pretrained activation tensors exactly, allowing the expanded model to inherit both the structural and representational properties of Motif-2.6B.

To improve representational disentanglement and enhance fine-grained control over information routing within attention, we incorporated the Grouped Differential Attention (GDA) mechanism lim2025grouped during the hypercloning stage. This mechanism enables different subsets of attention heads to specialize in either amplifying salient signals or suppressing residual noise, thereby improving the overall quality and efficiency of learned representations. During this stage, we integrated GDA directly into the cloned architecture, organizing the attention heads into asymmetric groups with a 4:1 ratio between signal-preserving and noise-control groups, respectively. Determining this ratio was a critical design choice, as the GDA mechanism requires balancing the proportion of signal and noise heads for optimal performance. Through preliminary experiments exploring various configurations, we empirically found that a 4:1 ratio provided the best balance—allocating more heads to salient information propagation while maintaining stability and regularization through the smaller noise-control group. This integration represents the first large-scale application of GDA within the Motif architecture, validating its scalability and effectiveness in production-scale training.

Following the width-preserving hypercloning and GDA integration, we expanded the model depth using the LLaMA-Pro framework wu2024llama. This step increased the number of layers in a structurally consistent manner while preserving RMS normalization zhang2019root, Rotary Positional Embedding su2024roformer, and cross-layer residual scaling inherited from Motif-2.6B. The activation PolyNorm zhuo2024polynomial and the tokenizer from Motif-2.6B were retained without modification. The resulting architecture, referred to “2” in the middle of its name ‘Motif-2-12.7B’, therefore represents a continuous and functionally extended successor of its predecessor Motif-2.6B. The detailed architectural configurations are summarized in Table 1.

Model size
12.7B

Hidden dimension
4,096

Number of layers
40

Feed-Forward dimension
16,384

Number of heads
40

Number of KV heads
16

Number of Noise heads
8

Attention Mechanism
Grouped Differential Attention

Activation Function
PolyNorm Activation

Max Sequence Length
32,768

Positional Embeddings
RoPE (θ=1,000,000\theta=1,000,000)

Vocab size
219,520

Tied word embedding
False

## 3 Pre-training

This section provides an overview of the pre-training data construction, describes the pre-training methodology, and presents benchmark evaluations of the base models.

### 3.1 Pre-training Data

We constructed a large-scale and high-quality pre-training corpus for Motif-2-12.7B by aggregating a broad spectrum of open-source and in-house datasets spanning diverse domains and modalities. The corpus encompasses general web data (e.g., Nemotron-CC su2024nemotron, Wikipedia, Stack Exchange tang2024txt360, and Facebook Recycling the Web nguyen2025recycling), multilingual and Korean-language resources (Fineweb2 penedo2025fineweb2pipelinescale, Translated DiverseQA su2024nemotron, and extended in-house Korean corpus), as well as specialized domains such as reasoning and knowledge-intensive tasks (DiverseQA su2024nemotron, Synthetic QA Dataset), scientific and mathematical content (FineWeb2 penedo2025fineweb2pipelinescale, OctoThinker MegaMath wang2025octothinker, Yulan Math hu2024yulan, arXiv), and code-centric corpora (OpenCoder-LLM Dataset huang2024opencoder, Yulan Code hu2024yulan).

The model was trained on 5.5 trillion tokens drawn from this corpus, encompassing general English, question answering, conversational, multilingual, scientific, mathematical, and programming domains. This large-scale training setup provides a balanced and robust foundation for advancing both linguistic and reasoning capabilities.

### 3.2 Pre-training Stage and Recipe

The pre-training of the Motif-2-12.7B series followed three core strategies designed to balance efficiency, diversity, and reasoning capability.

##### Dataset Mixture Scheduling

We adopted a progressive dataset mixture scheduling strategy, similar to that used in Motif-2.6B. The early training phase primarily consisted of English web-scale corpora, while reasoning, mathematical, and code-related datasets were introduced in small proportions. Their relative ratios were then gradually increased over the course of training. The dataset mixture was dynamically adjusted at each training step, analogous to how a learning-rate scheduler modulates optimization dynamics. This approach enabled a smooth transition from general linguistic learning to higher-order reasoning without destabilizing early-stage convergence.

##### Large-Batch Training

Training was conducted with large effective batch sizes, starting from 16 million tokens and progressively scaling up to 80 million tokens. We employed the Muon-Clip optimizer, originally introduced in the Kimi K2 team2025kimi, and further developed an internally optimized variant tailored for parallel and distributed training settings.
This enhanced Muon implementation features improved gradient stability, communication efficiency, and large-batch scalability, which together enable high-throughput optimization under multi-node environments. Details of the parallel and distributed optimization design are provided in a later section.

##### Reasoning Annealing Stage

In the final annealing phase, we gradually increased the proportion of reasoning data while capping it below 10% of total tokens to prevent overfitting and mode collapse toward narrow reasoning distributions. Notably, within this reasoning subset, mathematical data were weighted more heavily than code data, which is a deliberate choice aimed at strengthening symbolic abstraction and quantitative reasoning. Although this allocation is unconventional, we found it conducive to improved general reasoning robustness without degrading linguistic fluency.
We used a peak learning rate of 4×10−44\times 10^{-4}, which was decayed to approximately 10% of the value using the Warmup–Stable–Decay scheduler hu2024minicpmunveilingpotentialsmall. The annealing stage was conducted over roughly 1 trillion training tokens, during which the maximum sequence length was gradually increased from 4,096 to 16,384 tokens toward the end of training to enable smooth adaptation to longer-context understanding.

### 3.3 Pre-training Evaluation

##### Evaluation Setup

To evaluate the performance of the Motif-2-12.7B-Base, we conducted a comprehensive evaluation across multiple capability areas. We compared the models developed in this study against two leading open-source foundation models, Qwen3 and Gemma3 team2025gemma. These two models represent state-of-the-art open weight models similar scale and learning methods. To ensure fair and consistent comparison, we used the officially reported scores from each model’s technical report. All evaluations for Motif-2-12.7B-Base were conducted using greedy decoding.

General Knowledge
 We evaluate benchmarks such as MMLU hendrycks2009measuring, MMLU-Redux gema2025we, MMLU-Pro wang2024mmlu, BBH suzgun2023challenging, ARC-Challenge, and ARC-Easy clark2018think to measure the model’s breadth of factual knowledge, general reasoning ability. Motif-2-Base achieves solid performance on the MMLU and MMLU-Redux, surpassing Gemma-3 12B and approaching Qwen-3 14B/32B levels. Notably, it records the highest score on MMLU-Pro highlighting strong reasoning capability.

Math & Scientific Reasoning
 GSM8K cobbe2021training, MATH hendrycks2024measuring, GPQA, GPQA-Diamond rein2024gpqa, and SuperGPQA du2025supergpqa are used to assess mathematical reasoning, numerical precision, and scientific understanding requiring multi-step logical inference. Motif-2-Base exhibits clear strength, recording 94.9 on GSM8K and 73.6 on MATH, exceeding all open-weight baselines of comparable scale. It also attains competitive results on GPQA and SuperGPQA, showing superior performance among models of similar scale.

Code Tasks
 HumanEval chen2021evaluating, MBPP austin2021program, EvalPlus liu2023your, and CRUX-O gu2024cruxeval evaluate code generation, reasoning about programs, and functional correctness through test-based execution. Motif-2-Base leads with 65.9 on HumanEval and 81.5 on MBPP, establishing itself as one of the strongest code-capable models among non-proprietary peers.

Common-sense
 HellaSwag zellers2019hellaswag, BoolQ clark2019boolq, PIQA bisk2020piqa, SIQA sap2019socialiqa, WinoGrande sakaguchi2021winogrande, DROP dua2019drop, TriviaQA joshi2017triviaqa, and Natural Questions kwiatkowski2019natural
are employed to test everyday reasoning, contextual comprehension,
and the model’s ability to infer causal or pragmatic relationships in natural language. Although slightly lower than larger counterparts, the scores remain competitive and do not show significant degradation.

##### Evaluation Results

Across diverse benchmarks spanning knowledge, reasoning, coding, and commonsense understanding, Motif-2-12.7B-Base demonstrates strong overall competitiveness with leading open-source foundation models.
Overall, Motif-2-12.7B-Base outperforms both Gemma-3 12B and 27B, as well as Qwen models of comparable size (14B), while approaching the performance of much larger variants such as Qwen-3 32B. These results confirm that Motif-2-12.7B-Base stands as a high-performing and broadly capable foundation model among open-weight models.

Benchmark
Metric
Motif-2
Gemma3
Qwen2.5
Qwen3

12.7B
12B
27B
14B
32B
14B
32B
30B-A3B

General knowledge

MMLU
5-shot
78.1
74.5
78.6
79.66
83.32
81.05
83.61
81.38

MMLU-Redux
5-shot
78.68
-
-
76.64
81.97
79.88
83.41
81.17

MMLU-Pro
5-shot, CoT
66.38
45.3
52.2
51.16
55.1
61.03
65.54
61.49

BBH
3-shot, CoT
81.34
72.6
77.7
78.18
84.48
81.07
87.38
81.54

ARC-Easy
0-shot
84.1
88.3
89.0
-
-
-
-
-

ARC-Challenge
25-shot
69.6
68.9
70.6
-
-
-
-
-

Math & Scientific Reasoning

GSM8k
4-shot, CoT
93.85
-
-
90.22
92.87
92.49
93.40
91.81

8-shot, CoT
94.92
71.0
82.6
-
-
-
-
-

MATH
4-shot, CoT
73.62
43.3
50.0
55.64
57.7
62.02
61.62
59.04

GPQA
5-shot, CoT
42.18
-
-
32.83
47.97
39.9
49.49
43.94

GPQA-Diamond
5-shot, CoT
42.92
25.4
24.3
-
-
-
-
-

SuperGPQA
5-shot, CoT
32.68
-
-
30.68
33.55
34.27
39.78
35.72

Coding Tasks

HumanEval
0-shot
65.9
48.8
45.7
-
-
-
-
-

MBPP
3-shot
81.5
60.4
65.6
69.0
73.6
73.4
78.2
74.4

EvalPlus
0-shot
72.22
-
-
60.7
66.25
72.05
72.23
71.45

CRUX-O
1-shot
63.1
-
-
61.1
67.8
68.6
72.5
67.2

Common Sense

HellaSwag
10-shot
84.0
84.2
85.6
-
-
-
-
-

BoolQ
0-shot
78.5
78.8
82.4
-
-
-
-
-

PIQA
0-shot
81.6
81.8
83.3
-
-
-
-
-

SIQA
0-shot
53.8
53.4
54.9
-
-
-
-
-

DROP
1-shot
69.9
72.2
77.2
-
-
-
-
-

TriviaQA
5-shot
72.2
78.2
85.5
-
-
-
-
-

Natural Questions
5-shot
29.6
31.4
36.1
-
-
-
-
-

WinoGrande
5-shot
79.6
74.3
78.8
-
-
-
-
-

Average

71.53
63.87
67.96
-
-
-
-
-

69.42
-
-
62.35
67.69
67.81
71.54
68.10

## 4 Training System Optimization

In this section, we describe how training efficiency was significantly enhanced through the integration of novel kernel fusion and parallel muon techniques.

### 4.1 Infrastructure and Precision

We employed SkyPilot yang2023skypilot as the orchestration framework, enabling seamless multi-node and multi-cluster provisioning, and unified resource abstraction across multiple compute environments. For the core training stack, we adopted TorchTitan liang2024torchtitan, a PyTorch-native distributed training platform that provides modular 3D/4D parallelism (data, tensor, pipeline, and context), advanced optimizations such as FP8 all-gather and activation checkpointing.
Pretraining was performed on 400 H100 gpus for approximately 272K hours, using FP8 precision to maximize computational efficiency. For FP8 training, we applied row-wise scaling and keeps gradients in BF16 precision during the backward pass.

### 4.2 Kernel Fusion

During training, we implemented PolyNorm, a polynomial activation function introduced at Motif-2.6B (lim2025motif), as a fused CUDA kernel and further fused it with the subsequent elementwise multiplication in the FeedForward layer. PolyNorm is composed of lightweight sub-operations—elementwise scaling, reduction, and normalization—that are typically memory-bound and hence benefit from kernel fusion, which reduces redundant memory traffic and kernel launch overhead.

torch.compile111https://docs.pytorch.org/docs/stable/generated/torch.compile.html is a high-level API that enables graph-level optimization and automatic GPU kernel fusion with almost no code modification.
While it provides substantial performance gains, we observed that hand-written fused CUDA kernels can still deliver superior throughput.

Operation
Forward Speedup
Backward Speedup

vs Naive
vs torch.compile
vs Naive
vs torch.compile

PolyNorm

30.29
1.53
43.90
4.77

PolyNorm + Elementwise Multiplication

23.38
1.29
27.92
3.33

Table 3 compares our fused CUDA kernels against both the naive PyTorch implementation and the torch.compile-optimized version.
While torch.compile already provides substantial acceleration—yielding up to 12.9×12.9\times speedup over the naive baseline—our hand-written fused kernels achieve even higher throughput in both forward and backward passes.
This demonstrates that manual fusion can further exploit the remaining optimization headroom, particularly for normalization layers dominated by small, memory-bound operations.
The full implementation of Fused PolyNorm kernels are available at
https://huggingface.co/Motif-Technologies/activation.

### 4.3 Parallel Muon

Liu et al. introduced Distributed Muon liu2025muon, which applies a ZeRO-1–based parameter partitioning scheme to the Muon algorithm. Unlike element-wise optimizers such as AdamW, Muon requires access to the full gradient matrices to perform the Newton–Schulz iterations.
Distributed Muon addresses this by performing an all-gather to reconstruct the full matrix before each Newton–Schulz iteration, computing the update locally, and then discarding all shards except the one that belongs to each rank. While this approach resolves the full-matrix dependency of Muon, it does not parallelize the Newton–Schulz computation itself; as a result, the iteration steps are redundantly executed on all ranks.

There exist several ideas and implementations that aim to overcome the limitations of Distributed Muon. One notable approach is from Essential AI222https://www.essential.ai/research/infra,
where they propose employing All-to-All communication to re-shard gradients across ranks, thereby eliminating the redundant computations inherent in Distributed Muon.

We began developing our own implementation of Muon, extending Essential AI’s idea with pipelining and general sharding support (e.g., TP + HSDP), which we refer to as Parallel Muon.
Parallel Muon distributes computational workloads across shard ranks and executes them concurrently.
In addition, it schedules communication and computation operations to fully utilize the accelerators.
While some existing implementations of Muon support FSDP, none provide support for hybrid parallel configurations.

Require: DP sharded gradients 𝐆\mathbf{G}, DP sharded parameters 𝐏\mathbf{P}, rank 𝐫\mathbf{r}

1:// 1. Divide 𝐆\mathbf{G} into subsets and assign to each rank

2:𝐆′←assign​(𝐆,𝐫)\mathbf{G}^{\prime}\leftarrow\text{assign}(\mathbf{G},\mathbf{r})

3:// 2. All2All communicate to gather 𝐆′\mathbf{G}^{\prime} across DP shard group

4:// As a result, each rank get unsharded matrices of 𝐆′\mathbf{G}^{\prime}

5:𝐆full′←all2all_gather​(recv=​𝐆′,send=​𝐆∖𝐆′)\mathbf{G}^{\prime}_{\text{full}}\leftarrow\text{all2all\_gather}(\text{recv=}\mathbf{G^{\prime}},\text{send=}\mathbf{G}\setminus\mathbf{G}^{\prime})

6:// 3. Calculate Newton-Schulz of unsharded gradients assigned to rank 𝐫\mathbf{r}

7:𝐔full′←Newton-Schulz​(𝐆full′)\mathbf{U}^{\prime}_{\text{full}}\leftarrow\text{Newton-Schulz}(\mathbf{G}^{\prime}_{\text{full}})

8:// 4. All2All communicate to scatter 𝐔full′\mathbf{U}^{\prime}_{\text{full}} across DP shard group

9:// As a result, earch rank get sharded matrices of 𝐔full′\mathbf{U}^{\prime}_{\text{full}}

10:𝐔←all2all_scatter​(recv=​𝐔other′,send=​𝐔full′)\mathbf{U}\leftarrow\text{all2all\_scatter}(\text{recv=}\mathbf{U}^{\prime}_{\text{other}},\text{send=}\mathbf{U}^{\prime}_{\text{full}})

11:// 5. Apply DP sharded 𝐔\mathbf{U} to 𝐏\mathbf{P}

12:𝐏′←apply_update​(𝐏,𝐔)\mathbf{P^{\prime}}\leftarrow\text{apply\_update}(\mathbf{P},\mathbf{U})

13:return 𝐏′\mathbf{P^{\prime}}

Algorithm 1 describes the parallel version of Muon under Fully Sharded Data-Parallel (FSDP) training. Each rank first takes a subset of gradients and then participates in an All2All communication to gather the full unsharded gradients assigned to it. The Newton–Schulz iteration is then applied independently on each gathered subset to compute the update matrices. The resulting unsharded updates are scattered back to the corresponding ranks via another All2All operation, producing sharded update tensors. Finally, each rank applies its local update to the corresponding sharded parameters. This procedure enables concurrent computation across DP-shard ranks without additional memory overhead.

##### All-to-All Gather and Scatter

As discussed earlier in Essential AI’s idea, Parallel Muon also employs All-to-All communication to redistribute both sharded and unsharded gradients across ranks.
The All-to-All gather operation collects the sharded gradients so that each rank obtains its assigned unsharded gradients, while the All-to-All scatter operation redistributes the computed results back to all ranks, as illustrated in Figure 1.

##### Pipelined Execution

Although Algorithm 1 is sound on its own, pipelining its execution enables better performance through communication–computation overlap. As illustrated in Figure 2, we group the parameters into chunks and schedule their all-to-all gather, computation, and scatter phases in a pipelined fashion.
Specifically, we first issue the gather and computation operations for a few initial chunks to warm up the pipeline. Then, for each subsequent chunk, we overlap scatter of the previous results, gather of the next inputs, and the computation of the current chunk.
This scheduling can hide communication latency by maintaining a steady flow of computation and communication, thereby improving throughput and reducing idle time.

In addition to performance improvements, pipelining also helps reduce peak memory usage.
Since the full output matrices produced by each computation can only be deallocated after their corresponding scatter phase completes, a non-pipelined execution must keep multiple full matrices in memory simultaneously.
By overlapping computation and scatter, the pipelined schedule allows completed full matrices to be released earlier, thus lowering the overall memory footprint.

##### Chunk Size Trade-off

While pipelining improves both performance and memory efficiency, one important consideration is the choice of chunk size.
A smaller chunk size allows finer-grained overlap between computation and communication, which tends to improve performance and reduce memory usage.
However, all-to-all communication becomes less efficient when messages are too small, making larger chunks more favorable for bandwidth utilization.
In practice, the benefit from improved bandwidth utilization with larger chunks often outweighs the performance gain from finer overlap, resulting in an optimal chunk size that balances both effects.
We empirically find that a chunk size of 32 achieves the best overall trade-off between performance and memory usage.

##### Mitigating Rank Imbalance

To mitigate workload imbalance among ranks, we balanced the total computational load assigned to each rank.
Because the all-to-all communication in gather and scatter stages requires synchronization among all participating ranks, any imbalance in computation time would lead to idle waiting and reduced throughput.
We therefore sorted all gradients according to the number of floating-point operations (FLOPs) required for their respective Newton–Schulz computations and assigned them to ranks in a round-robin fashion.
This simple yet effective heuristic ensures that each rank receives a roughly equal total workload, reducing idle time and improving parallel efficiency.
While this strategy does not guarantee perfectly balanced execution, it provides a good practical trade-off between load balance and scheduling overhead.

##### Support for Hybrid Parallel Configurations

To support various strategies beyond FSDP, such as TP or TP + HSDP, Parallel Muon dynamically inspects the placement of gradient tensors at runtime, constructs corresponding sub–process groups for gather and scatter operations, and computes the shard mapping that defines which tensor slice is held by each rank.

Since Parallel Muon preserves the semantics of the original Newton–Schulz iterations, it directly leverages existing kernel optimizations, including Flash Muon lin2025flash. The full implementation of Parallel Muon is available at https://huggingface.co/Motif-Technologies/optimizer.

##### Parallel Muon Evaluation

We evaluate the performance of Parallel Muon under various configurations and compare it against the Distributed Muon baseline.
All experiments are conducted on a single node equipped with eight NVIDIA H200 GPUs, using Fully Sharded Data Parallel (FSDP) with eight ranks.
The model used for evaluation is Motif-2-12.7B, and all communication and GEMM operations are performed in BF16 precision.

We consider the following configurations:

- 1.

Distributed Muon – the baseline implementation, extended with Flash Muonlin2025flash for fair comparison with Parallel Muon.
Distributed Muon reconstructs the full parameter matrix via all-gather and redundantly executes the Newton–Schulz iterations on all ranks.

- 2.

Parallel Muon (non-pipelined) – our implementation of Parallel Muon without pipelining, where computation and communication are executed sequentially.

- 3.

Parallel Muon (pipelined) – Parallel Muon with pipelined gather–compute–scatter scheduling to enable communication–computation overlap.

- 4.

Parallel Muon (pipelined + FLOPs-sorted) – the full version, which additionally applies FLOPs-based gradient sorting and round-robin scheduling to balance computation across ranks.

We measure throughput, step time, and peak memory usage for each configuration to quantify the impact of pipelining and load balancing.

Table 4 summarizes the performance of Distributed Muon and the different Parallel Muon configurations.

First, simply adopting the non-pipelined Parallel Muon configuration (row 2) achieves a substantial performance of 571 Tera floating operations per second per GPU (TFLOPS/GPU), a 7.1×\times throughput improvement over the baseline Distributed Muon (80 TFLOPS/GPU). This significant gain is attributed to distributing the computational workload across the 8 GPUs.

Interestingly, introducing pipelining (row 3) initially results in a performance degradation, with throughput dropping by approximately 15.8% compared to the non-pipelined version. This is caused by workload imbalance, as the per-chunk all-to-all communication (for the small, size 32 chunks) acts as a synchronization barrier across ranks. This is strongly supported by the results from the fourth configuration: applying FLOPs-based parameter sorting (row 4) rectifies this imbalance, boosting performance significantly by over 21%.

It is also important to note that the throughput of the fully optimized (pipelined + sorted) version is only marginally higher—by about 2.1%—than that of the non-pipelined version. This relates to the chunk size trade-off; the non-pipelined approach effectively uses a single, large chunk (equal to the total number of parameters), thus benefiting from high bandwidth utilization on its single communication step while also avoiding the repeated synchronization overheads inherent to fine-grained pipelining.

Despite this small throughput difference, the primary benefit of the pipelined approach is its drastically lower memory footprint. The non-pipelined version’s peak memory usage is over 4.1 times that of the pipelined (non-sorted) configuration and over 3 times that of the fully optimized version, making pipelining a crucial optimization for memory efficiency.

Another open-sourced Muon implementation example333https://github.com/microsoft/dion/blob/main/dion/muon.py is provided in Dion ahn2025diondistributedorthonormalizedupdates. This work is also inspired by Essential AI’s approach, and its key idea is to avoid the full-matrix reconstruction required by Muon. However, this implementation does not support hybrid configurations such as FSDP2 combined with tensor parallelism (TP).

Configuration
Pipelining
Sort Param.
Chunk Size
Time (ms)
Peak Mem (MB)
TFLOPS per GPU

Distributed Muon

–
–
–
1574.67
832
80

Parallel Muon

X
X
–
221.27
11904
571

Parallel Muon

O
X
32
262.83
2894
481

Parallel Muon

O
O
32
216.46
3904
583

## 5 Post-training

### 5.1 Supervised Fine-tuning

Following the completion of pre-training, Motif-2-12.7B-Instruct went through a three-stage supervised fine-tuning (SFT) process designed to enhance its instruction-following, reasoning, and domain specialization capabilities.
Unlike Motif-2.6B, which primarily focused on general instruction adherence, each stage explicitly targeted multi-step reasoning, code synthesis, and mathematical problem-solving through carefully curated and synthesized datasets.
The three-stage SFT pipeline progressively refines the model’s capabilities—from large-scale general instruction learning (Stage 1), to reasoning- and domain-oriented enhancement through synthetic data (Stage 2), and finally to data-pruned targeted refinement (Stage 3).

#### 5.1.1 Stage 1: Large-Scale Supervised Fine-tuning

The first fine-tuning stage focused on establishing strong general instruction-following performance through large-scale training.
We trained on approximately 28 million samples, drawn from both open-source and proprietary datasets, with domain ratios adjusted to maintain a balanced representation across categories.
Training was conducted with sequence packing up to 16,384 tokens to enable efficient utilization of long-context information.
All parameters were unfrozen during this stage, allowing full adaptation of model representations to downstream supervision.

##### Training Setup

Training was performed using the Muon-Clip optimizer with a base learning rate of 2×10−52\times 10^{-5}, weight decay of 0.1, and a global batch size of 32 million tokens.
A cosine learning rate scheduler was employed with a warm-up phase corresponding to 5% of the total training steps, followed by smooth decay toward zero.
Sequence packing was enabled up to a maximum sequence length of 16,384 tokens to ensure efficient utilization of GPU memory and to improve long-context training stability.
All training was conducted in FP8 precision under Fully Sharded Data Parallel (FSDP) with the Parallel Muon optimization kernels as described in Section 4.3, thereby enabling stable and high-throughput large-batch training.

#### 5.1.2 Stage 2: Synthetic and Targeted Fine-tuning

To further enhance higher-order reasoning and domain-specific capabilities, the second fine-tuning phase employed a mixture of open and synthetic instruction datasets.
We first consolidated a diverse mixture of high-quality open instruction datasets spanning general conversation, STEM, mathematics, and code generation.
Each dataset was evaluated based on reasoning density, response completeness, and structural consistency.
The resulting dataset was designed to maintain domain balance while providing comprehensive coverage of general instruction-following and structured reasoning tasks.

Building upon this foundation, we introduced additional synthetic data generated from both internal and external models to strengthen compositional reasoning, algorithmic problem-solving, and multi-step instruction-following performance.
The synthetic datasets consisted of:

- •

Compositional Reasoning: Synthetic examples designed to enhance multi-step logical reasoning and structured task decomposition.

- •

Algorithmic and Mathematical Skills: Data emphasizing algorithmic reasoning, code generation, and quantitative problem-solving.

- •

Quality-Centric Selection: High-quality samples curated through multi-stage filtering and automatic quality scoring.

This stage adopted a lower learning rate and shorter training schedule relative to Stage 1, functioning as a form of curriculum continuation.
Through the combination of large-scale instruction data and carefully designed synthetic augmentation, Motif-2-12.7B-Instruct achieved significant improvements in reasoning quality while preserving conversational fluency.

#### 5.1.3 Stage 3: Data-Pruned Refinement

In the final stage, we conducted a focused fine-tuning process by selectively excluding a subset of the Stage 2 synthetic data identified as redundant or low-utility.
This data-pruned refinement stage aimed to reinforce domain diversity and reduce potential overfitting to synthetic distributions by re-training on a carefully filtered subset of Stage 2 data.
The pruning criteria emphasized diversity, reasoning coherence, and linguistic fidelity, resulting in a leaner but higher-quality dataset.
This stage served as a lightweight continuation of Stage 2, stabilizing the model’s reasoning and alignment behavior without altering its general conversational fluency.

### 5.2 Post-training Evaluation

##### Evaluation Setup

Following the same evaluation protocol as Motif-2-12.7B-Base, we conducted a comprehensive assessment of Motif-2-12.7B-Instruct to evaluate its generalization, reasoning, and instruction-following capabilities. To ensure comparability and fairness, we benchmarked our model against strong open-weight baselines, including the Qwen3 and Gemma3 series, using their officially reported scores from the respective technical reports. All evaluations for Motif-2-12.7B-Instruct were performed with a sampling temperature of 0.6 and a maximum sequence length of 32,768 tokens.

##### Benchmark Composition

Our evaluation suite spans a wide range of domains to measure both breadth and depth of model ability. The general task benchmarks include MMLU hendrycks2009measuring, MMLU-Redux gema2025we, GPQA-Diamond rein2024gpqa, LiveBench white2024livebench, and IFEval ifeval2024, which collectively test factual knowledge, reasoning, and alignment quality across diverse subjects.
The mathematics and text reasoning category comprises MATH, MATH-500 lightman2023let, AIME24, AIME25 AIME25, ZebraLogic lin2025zebralogic, and BBH suzgun2023challenging, emphasizing symbolic reasoning, structured problem-solving, and compositional understanding.
Finally, agentic and programming-oriented benchmarks, including BFCL v3 xie2023doremi, LiveCodeBench jain2024livecodebench, MBPP austin2021program, and HumanEval chen2021evaluating, assess multi-step reasoning, code synthesis, and executional accuracy in practical scenarios.
This diverse set of tasks allows us to holistically evaluate the instruction-following ability and analytical robustness of Motif-2-12.7B-Instruct across both linguistic and technical domains.

Benchmark
Metric
Motif-2
Qwen2.5
Qwen3

12.7B
72B
14B
14B
32B
32B
30B-A3B
30B-A3B

Instruct
Non-Think
Non-Think
Think
Non-Think
Think
Non-Think
Think

MMLU-Redux
0-shot
90.02
86.8
82
88.6
85.7
90.9
84.1
89.5

GPQA-Diamond
0-shot, CoT
63.6
49
54.8
64
54.6
68.4
54.8
65.8

LiveBench

2024-11-25

MATH-500
0-shot
96.8
83.6
90
96.8
88.6
97.2
89.8
98

AIME24
0-shot
72.3
18.9
31.7
79.3
31
81.4
32.8
80.4

AIME25
0-shot
63.6
15
23.3
70.4
20.2
72.9
21.6
70.9

ZebraLogic
-
69.5
26.6
33
88.5
29.2
88.8
33.2
89.5

BFCLv3
-
55.34
63.4
61.5
70.4
63
70.3
58.6
69.1

LiveCodeBench v5

(2024.10 - 2025.2)

IFEval
strict prompt
75.78
84.1
84.8
85.4
83.2
85
83.7
86.5

Average

67.08
50.95
54.97
77.82
54.66
79.55
54.78
78.66

Benchmark
Metric
Motif-2
Gemma3

12.7B
12B
27B

MMLU
0-shot
86.11
71.9
76.9

BBH
0-shot
85.78
85.7
87.6

GSM8k
0-shot, CoT
96.13
94.4
95.9

MATH
0-shot
97
83.8
89

MBPP
3-shot
91
73
74.4

IFEval
soft prompt
76.52
88.9
90.4

LiveCodeBench v5
0-shot
61.66
32
39

HumanEval
0-shot
93.2
85.4
87.8

Average

83.44
72.89
75.93

##### Evaluation Results

Detailed quantitative results are summarized in Table 5 and Table 6. Overall, Motif-2-12.7B-Instruct achieves performance comparable to or exceeding that of similarly sized or larger open-weight models, despite being trained on a substantially smaller dataset. Notably, without relying on computationally heavy and reasoning targeted reinforcement learning, the model demonstrates strong results on high-difficulty reasoning and code-oriented benchmarks such as MATH-500, AIME25, and LiveCodeBench, underscoring the effectiveness of its three-stage SFT curriculum.
While Qwen3 and Gemma3 models benefit from significantly larger training corpora, with Qwen3 trained on 36T tokens and Gemma3 on 12 to 14T tokens, our model attains similar or higher accuracy across several benchmarks using only 5.5T tokens, highlighting the data efficiency and architectural effectiveness of our approach. The integration of Grouped Differential Attention (GDA), curriculum-aware data scheduling, and high-throughput optimization via Parallel Muon collectively contribute to this efficiency, enabling more capable instruction models at a fraction of the compute cost.
These results illustrate that careful architectural design, optimization-aware training infrastructure, and balanced fine-tuning strategies can bridge much of the performance gap traditionally attributed to massive scale, reaffirming the viability of efficient frontier–oriented model design as a path toward high-quality open-weight systems.

## 6 Conclusion

Motif-2-12.7B demonstrates that carefully targeted architectural changes and systems-aware training can deliver strong performance without resorting to extreme parameter counts. By scaling Motif-2.6B with width-preserving hypercloning and Llama Pro depth scaling, integrating Grouped Differential Attention for finer-grained information routing, and coupling curriculum-aware pre-training with a three-stage SFT pipeline, we obtain a model that is both efficient and broadly capable. Our fused PolyNorm kernels and Parallel Muon further show how bespoke systems work unlocks practical throughput and memory gains at long context, making large-scale training more accessible. Looking ahead, we will release Motif-2-12.7B-Reasoning, a reinforcement-learning–enhanced variant that explicitly optimizes multi-step reasoning quality (with a focus on mathematics and code) while maintaining conversational fluency. We hope these open weights, transparent recipes, and forthcoming RL results provide a solid baseline for the community to study scalable attention, efficient training, and principled pathways to stronger reasoning.

## 7 Appendix

### 7.1 Contributions

All authors are alphabetically sorted by last name.

Technical and management leadership: Sungmin Lee, Junghwan Lim

Core contributors: Dongseok Kim, Taehyun Kim, Eunhwan Park, Jeesoo Lee, Jeongdoo Lee, Junhyeok Lee

Contributors: Wai Ting Cheung, Dahye Choi, Jaeheui Her, Jaeyeon Huh, Hanbin Jung, Changjin Kang, Beomgyu Kim, Minjae Kim, Taewhan Kim, Youngrok Kim, Hyukjin Kweon, Haesol Lee, Kungyu Lee, Dongpin Oh, Yeongjae Park, Bokki Ryu, Dongjoo Weon

Generated on Fri Dec 5 14:27:54 2025 by LaTeXML

# Hugging Face - 2026 - Trending Papers

- Source HTML: `raw/html/Hugging Face - 2026 - Trending Papers.html`
- Source SHA256: `0739791f3771d7503893760d53a0bb5ec6fd86738c4e3009f7a56c81f464c545`
- Source URL: https://huggingface.co/papers/trending
- Generated from: `scripts/fetch_web_text.py`
- Extraction: `structured-html-v2` (headings, links, MathML/TeX and tables; figures require visual review)

## Extracted Text

<a id="source-section-0"></a>

new


Get trending papers in your email inbox once a day!


Get trending papers in your email inbox!

[Subscribe](https://huggingface.co/login?next=%2Fpapers)


<a id="source-section-1"></a>

# Trending Papers


<a id="source-section-2"></a>

## by[[图片：无替代文本] AK](https://huggingface.co/akhaliq) and the research community


-

-

-


Trending Papers


Submitted by

[图片：无替代文本]

vvasilev


<a id="source-section-3"></a>

### [Kandinsky 6.0 Video: Foundation Models for Synchronized Video and Audio Generation](https://huggingface.co/papers/2610.05608)


We present Kandinsky 6.0 Video, a family of foundation diffusion models for synchronized text-to-audio-video generation, comprising Kandinsky 6.0 Video Lite (3B parameters) and Kandinsky 6.0 Video Pro (29B parameters). Both models generate 5-second video clips with synchronized 44 kHz audio, including lip-sync, in text-to-audio-video (T2AV) and image-to-audio-video (I2AV) modes; a built-in super-resolution model raises the output resolution to Full-HD (1920times1080). Building on the video generation capabilities of Kandinsky 5.0, Kandinsky 6.0 Video employs a dual-stream CrossDiT architecture that connects a pretrained video stream and a newly trained audio stream through bidirectional cross-attention for temporal and semantic alignment. Our continuous pretraining strategy first trains the audio stream from scratch on large-scale audio corpora and then trains both streams jointly on paired audio-video data while preserving unimodal fidelity; pretraining is followed by supervised fine-tuning, reinforcement-learning-based post-training, and distillation. In side-by-side human evaluation, Kandinsky 6.0 Video Pro clearly outperforms its predecessor, Kandinsky 5.0 Video Pro, and remains competitive with leading audio-video generation models, particularly in speech quality. To accelerate open research and deployment in multimedia generation, we release the code, model checkpoints, and diffusers integration under the MIT license.


[[图片：kandinskylab] Kandinsky Lab](https://huggingface.co/kandinskylab) · Published on Oct 4, 2026


[Upvote

117](https://huggingface.co/login?next=%2Fpapers%2F2610.05608)

[GitHub 131](https://github.com/kandinskylab/kandinsky-6) [arXiv Page](https://arxiv.org/abs/2610.05608)


Submitted by

[图片：无替代文本]

vvasilev


<a id="source-section-4"></a>

### [Kandinsky 6.0 Video: Foundation Models for Synchronized Video and Audio Generation](https://huggingface.co/papers/2610.05608)


We present Kandinsky 6.0 Video, a family of foundation diffusion models for synchronized text-to-audio-video generation, comprising Kandinsky 6.0 Video Lite (3B parameters) and Kandinsky 6.0 Video Pro (29B parameters). Both models generate 5-second video clips with synchronized 44 kHz audio, including lip-sync, in text-to-audio-video (T2AV) and image-to-audio-video (I2AV) modes; a built-in super-resolution model raises the output resolution to Full-HD (1920times1080). Building on the video generation capabilities of Kandinsky 5.0, Kandinsky 6.0 Video employs a dual-stream CrossDiT architecture that connects a pretrained video stream and a newly trained audio stream through bidirectional cross-attention for temporal and semantic alignment. Our continuous pretraining strategy first trains the audio stream from scratch on large-scale audio corpora and then trains both streams jointly on paired audio-video data while preserving unimodal fidelity; pretraining is followed by supervised fine-tuning, reinforcement-learning-based post-training, and distillation. In side-by-side human evaluation, Kandinsky 6.0 Video Pro clearly outperforms its predecessor, Kandinsky 5.0 Video Pro, and remains competitive with leading audio-video generation models, particularly in speech quality. To accelerate open research and deployment in multimedia generation, we release the code, model checkpoints, and diffusers integration under the MIT license.


[[图片：kandinskylab] Kandinsky Lab](https://huggingface.co/kandinskylab) · Oct 4, 2026


[Upvote

117](https://huggingface.co/login?next=%2Fpapers%2F2610.05608)


[GitHub 131](https://github.com/kandinskylab/kandinsky-6) [arXiv Page](https://arxiv.org/abs/2610.05608)


Submitted by

[图片：无替代文本]

bupalinyu


<a id="source-section-5"></a>

### [The Other Half of the Memory Wall: Serving 35B MoEs from SSD with Trained Routing Prediction](https://huggingface.co/papers/2609.18063)


Mixture-of-experts (MoE) inference on consumer hardware is bounded by weight memory: a 35B-class model is 19.5GB at 4-bit, and sparsity shrinks the compute per token, not the bytes that must be held. Naive offloading to SSD does not help on its own, because layer N+1's experts must be chosen before layer N's output exists, so the reads cannot start early enough to hide behind compute. We present Edge0, a streaming MoE inference engine that closes the gap with a prerouter: a per-layer head predicts the next layer's routing one token ahead, and the prediction is consumed as the routing itself, so the staged expert set equals the routed set and nothing is dropped. An unmerged recovery LoRA, trained on the student path, pays back the quality lost to int4 quantization and routing replacement. On a single 24GB machine, Edge0
serves a 35B MoE at 20tok/s inside 3GiB of peak active memory, within a few points of its fp16 teacher on average across five public benchmarks. An 8B tier runs on the same framework, and the framework, checkpoints, and adapters are open source.


[[图片：Edge0] Edge0](https://huggingface.co/Edge0) · Published on Sep 16, 2026


[Upvote

25](https://huggingface.co/login?next=%2Fpapers%2F2609.18063)

[GitHub 3.29k](https://github.com/Edge0-AI/edge0) [arXiv Page](https://arxiv.org/abs/2609.18063)


Submitted by

[图片：无替代文本]

bupalinyu


<a id="source-section-6"></a>

### [The Other Half of the Memory Wall: Serving 35B MoEs from SSD with Trained Routing Prediction](https://huggingface.co/papers/2609.18063)


Mixture-of-experts (MoE) inference on consumer hardware is bounded by weight memory: a 35B-class model is 19.5GB at 4-bit, and sparsity shrinks the compute per token, not the bytes that must be held. Naive offloading to SSD does not help on its own, because layer N+1's experts must be chosen before layer N's output exists, so the reads cannot start early enough to hide behind compute. We present Edge0, a streaming MoE inference engine that closes the gap with a prerouter: a per-layer head predicts the next layer's routing one token ahead, and the prediction is consumed as the routing itself, so the staged expert set equals the routed set and nothing is dropped. An unmerged recovery LoRA, trained on the student path, pays back the quality lost to int4 quantization and routing replacement. On a single 24GB machine, Edge0
serves a 35B MoE at 20tok/s inside 3GiB of peak active memory, within a few points of its fp16 teacher on average across five public benchmarks. An 8B tier runs on the same framework, and the framework, checkpoints, and adapters are open source.


[[图片：Edge0] Edge0](https://huggingface.co/Edge0) · Sep 16, 2026


[Upvote

25](https://huggingface.co/login?next=%2Fpapers%2F2609.18063)


[GitHub 3.29k](https://github.com/Edge0-AI/edge0) [arXiv Page](https://arxiv.org/abs/2609.18063)


[[图片：无替代文本]](https://huggingface.co/papers/2412.20138)


<a id="source-section-7"></a>

### [TradingAgents: Multi-Agents LLM Financial Trading Framework](https://huggingface.co/papers/2412.20138)


A multi-agent framework using large language models for stock trading simulates real-world trading firms, improving performance metrics like cumulative returns and Sharpe ratio.


-

-

-

- [图片：无替代文本]

- 4 authors

· Published on Dec 28, 2024


[Upvote

149](https://huggingface.co/login?next=%2Fpapers%2F2412.20138)

[GitHub 110k](https://github.com/tauricresearch/tradingagents) [arXiv Page](https://arxiv.org/abs/2412.20138)


[[图片：无替代文本]](https://huggingface.co/papers/2412.20138)


<a id="source-section-8"></a>

### [TradingAgents: Multi-Agents LLM Financial Trading Framework](https://huggingface.co/papers/2412.20138)


A multi-agent framework using large language models for stock trading simulates real-world trading firms, improving performance metrics like cumulative returns and Sharpe ratio.


-

-

-

- [图片：无替代文本]

- 4 authors

· Dec 28, 2024


[Upvote

149](https://huggingface.co/login?next=%2Fpapers%2F2412.20138)


[GitHub 110k](https://github.com/tauricresearch/tradingagents) [arXiv Page](https://arxiv.org/abs/2412.20138)


[[图片：无替代文本]](https://huggingface.co/papers/2609.05415)

Submitted by

[图片：无替代文本]

Linzhan


<a id="source-section-9"></a>

### [UniMate: One Unified Model to Animate Diverse Skeletons](https://huggingface.co/papers/2609.05415)


UniMate is a unified diffusion transformer that generates articulated motion for arbitrary skeletons from text and rigged 3D assets without per-skeleton retraining, using topology-aware attention and a large curated motion dataset.


[[图片：princetonu] Princeton University](https://huggingface.co/princetonu) · Published on Sep 4, 2026


[Upvote

22](https://huggingface.co/login?next=%2Fpapers%2F2609.05415)

[GitHub 1.54k](https://github.com/Friedrich-M/UniMate) [arXiv Page](https://arxiv.org/abs/2609.05415)


[[图片：无替代文本]](https://huggingface.co/papers/2609.05415)

Submitted by

[图片：无替代文本]

Linzhan


<a id="source-section-10"></a>

### [UniMate: One Unified Model to Animate Diverse Skeletons](https://huggingface.co/papers/2609.05415)


UniMate is a unified diffusion transformer that generates articulated motion for arbitrary skeletons from text and rigged 3D assets without per-skeleton retraining, using topology-aware attention and a large curated motion dataset.


[[图片：princetonu] Princeton University](https://huggingface.co/princetonu) · Sep 4, 2026


[Upvote

22](https://huggingface.co/login?next=%2Fpapers%2F2609.05415)


[GitHub 1.54k](https://github.com/Friedrich-M/UniMate) [arXiv Page](https://arxiv.org/abs/2609.05415)


[[图片：无替代文本]](https://huggingface.co/papers/2609.33325)

Submitted by

[图片：无替代文本]

PSRben


<a id="source-section-11"></a>

### [VisionHOPE: Visual Backbones as Self-Modifying Learning Systems](https://huggingface.co/papers/2609.33325)


Visual backbones have evolved from Convolutional Neural Networks (CNNs) with local aggregation to Vision Transformers (ViTs) with global interactions, State-Space Models (SSMs) with input-dependent state transitions, and Test-Time Training (TTT) layers that adapt an inner learner while processing an image. Across this progression, visual computation has become increasingly adaptive to each input, yet the rules governing that adaptation remain largely prescribed by the trained backbone. We introduce VisionHOPE, the first generic visual backbone formulated as a self-modifying learning system, in which what the model remembers and how it learns co-evolve within an image. Building on the self-referential construction of Nested Learning (NL), VisionHOPE realizes this co-evolution through five coupled memories that store content, generate key and value representations, and govern learning rate and retention. These memories evolve jointly as visual context accumulates along each scan. However, directly applying the unconstrained self-referential update to a visual backbone leads to instability. We therefore derive a stability-matched step-size control scheme that combines a soft cap on self-referential injection with a spectral clamp on the retained memory transition, and prove that the resulting memory dynamics are non-expansive along each scan. For two-dimensional feature maps, we adapt NL's chunk formulation by aligning chunks with image rows and columns across four directional scans. The proposed VisionHOPE achieves competitive results on ImageNet-1K, COCO, and ADE20K, establishing self-modifying learning systems as a practical foundation for general-purpose visual backbones. The code is available at https://github.com/PSRben/VisionHOPE.


[[图片：Mininglamp-2718] Mininglamp Technology](https://huggingface.co/Mininglamp-2718) · Published on Sep 27, 2026


[Upvote

323](https://huggingface.co/login?next=%2Fpapers%2F2609.33325)

[GitHub 843](https://github.com/PSRben/VisionHOPE) [arXiv Page](https://arxiv.org/abs/2609.33325)


[[图片：无替代文本]](https://huggingface.co/papers/2609.33325)

Submitted by

[图片：无替代文本]

PSRben


<a id="source-section-12"></a>

### [VisionHOPE: Visual Backbones as Self-Modifying Learning Systems](https://huggingface.co/papers/2609.33325)


Visual backbones have evolved from Convolutional Neural Networks (CNNs) with local aggregation to Vision Transformers (ViTs) with global interactions, State-Space Models (SSMs) with input-dependent state transitions, and Test-Time Training (TTT) layers that adapt an inner learner while processing an image. Across this progression, visual computation has become increasingly adaptive to each input, yet the rules governing that adaptation remain largely prescribed by the trained backbone. We introduce VisionHOPE, the first generic visual backbone formulated as a self-modifying learning system, in which what the model remembers and how it learns co-evolve within an image. Building on the self-referential construction of Nested Learning (NL), VisionHOPE realizes this co-evolution through five coupled memories that store content, generate key and value representations, and govern learning rate and retention. These memories evolve jointly as visual context accumulates along each scan. However, directly applying the unconstrained self-referential update to a visual backbone leads to instability. We therefore derive a stability-matched step-size control scheme that combines a soft cap on self-referential injection with a spectral clamp on the retained memory transition, and prove that the resulting memory dynamics are non-expansive along each scan. For two-dimensional feature maps, we adapt NL's chunk formulation by aligning chunks with image rows and columns across four directional scans. The proposed VisionHOPE achieves competitive results on ImageNet-1K, COCO, and ADE20K, establishing self-modifying learning systems as a practical foundation for general-purpose visual backbones. The code is available at https://github.com/PSRben/VisionHOPE.


[[图片：Mininglamp-2718] Mininglamp Technology](https://huggingface.co/Mininglamp-2718) · Sep 27, 2026


[Upvote

323](https://huggingface.co/login?next=%2Fpapers%2F2609.33325)


[GitHub 843](https://github.com/PSRben/VisionHOPE) [arXiv Page](https://arxiv.org/abs/2609.33325)


[[图片：无替代文本]](https://huggingface.co/papers/2309.06180)

Submitted by

[图片：无替代文本]

akhaliq


<a id="source-section-13"></a>

### [Efficient Memory Management for Large Language Model Serving with
PagedAttention](https://huggingface.co/papers/2309.06180)


PagedAttention algorithm and vLLM system enhance the throughput of large language models by efficiently managing memory and reducing waste in the key-value cache.


- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- 9 authors

· Published on Sep 12, 2023


[Upvote

76](https://huggingface.co/login?next=%2Fpapers%2F2309.06180)

[GitHub 86.1k](https://github.com/vllm-project/vllm) [arXiv Page](https://arxiv.org/abs/2309.06180)


[[图片：无替代文本]](https://huggingface.co/papers/2309.06180)

Submitted by

[图片：无替代文本]

akhaliq


<a id="source-section-14"></a>

### [Efficient Memory Management for Large Language Model Serving with
PagedAttention](https://huggingface.co/papers/2309.06180)


PagedAttention algorithm and vLLM system enhance the throughput of large language models by efficiently managing memory and reducing waste in the key-value cache.


- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- 9 authors

· Sep 12, 2023


[Upvote

76](https://huggingface.co/login?next=%2Fpapers%2F2309.06180)


[GitHub 86.1k](https://github.com/vllm-project/vllm) [arXiv Page](https://arxiv.org/abs/2309.06180)


[[图片：无替代文本]](https://huggingface.co/papers/2407.16741)

Submitted by

[图片：无替代文本]

akhaliq


<a id="source-section-15"></a>

### [OpenDevin: An Open Platform for AI Software Developers as Generalist
Agents](https://huggingface.co/papers/2407.16741)


OpenDevin is a platform for developing AI agents that interact with the world by writing code, using command lines, and browsing the web, with support for multiple agents and evaluation benchmarks.


- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- 24 authors

· Published on Jul 23, 2024


[Upvote

90](https://huggingface.co/login?next=%2Fpapers%2F2407.16741)

[GitHub 90.1k](https://github.com/opendevin/opendevin) [arXiv Page](https://arxiv.org/abs/2407.16741)


[[图片：无替代文本]](https://huggingface.co/papers/2407.16741)

Submitted by

[图片：无替代文本]

akhaliq


<a id="source-section-16"></a>

### [OpenDevin: An Open Platform for AI Software Developers as Generalist
Agents](https://huggingface.co/papers/2407.16741)


OpenDevin is a platform for developing AI agents that interact with the world by writing code, using command lines, and browsing the web, with support for multiple agents and evaluation benchmarks.


- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- 24 authors

· Jul 23, 2024


[Upvote

90](https://huggingface.co/login?next=%2Fpapers%2F2407.16741)


[GitHub 90.1k](https://github.com/opendevin/opendevin) [arXiv Page](https://arxiv.org/abs/2407.16741)


[[图片：无替代文本]](https://huggingface.co/papers/2510.22200)

Submitted by

[图片：无替代文本]

taesiri


<a id="source-section-17"></a>

### [LongCat-Video Technical Report](https://huggingface.co/papers/2510.22200)


LongCat-Video, a 13.6B parameter video generation model based on the Diffusion Transformer framework, excels in efficient and high-quality long video generation across multiple tasks using unified architecture, coarse-to-fine generation, and block sparse attention.


[[图片：meituan-longcat] LongCat](https://huggingface.co/meituan-longcat) · Published on Oct 25, 2025


[Upvote

43](https://huggingface.co/login?next=%2Fpapers%2F2510.22200)

[GitHub 8.99k](https://github.com/meituan-longcat/LongCat-Video) [arXiv Page](https://arxiv.org/abs/2510.22200)


[[图片：无替代文本]](https://huggingface.co/papers/2510.22200)

Submitted by

[图片：无替代文本]

taesiri


<a id="source-section-18"></a>

### [LongCat-Video Technical Report](https://huggingface.co/papers/2510.22200)


LongCat-Video, a 13.6B parameter video generation model based on the Diffusion Transformer framework, excels in efficient and high-quality long video generation across multiple tasks using unified architecture, coarse-to-fine generation, and block sparse attention.


[[图片：meituan-longcat] LongCat](https://huggingface.co/meituan-longcat) · Oct 25, 2025


[Upvote

43](https://huggingface.co/login?next=%2Fpapers%2F2510.22200)


[GitHub 8.99k](https://github.com/meituan-longcat/LongCat-Video) [arXiv Page](https://arxiv.org/abs/2510.22200)


[[图片：无替代文本]](https://huggingface.co/papers/2508.02739)


<a id="source-section-19"></a>

### [Kronos: A Foundation Model for the Language of Financial Markets](https://huggingface.co/papers/2508.02739)


Kronos, a specialized pre-training framework for financial K-line data, outperforms existing models in forecasting and synthetic data generation through a unique tokenizer and autoregressive pre-training on a large dataset.


-

-

-

-

-

- 7 authors

· Published on Aug 2, 2025


[Upvote

58](https://huggingface.co/login?next=%2Fpapers%2F2508.02739)

[GitHub 40.1k](https://github.com/shiyu-coder/Kronos) [arXiv Page](https://arxiv.org/abs/2508.02739)


[[图片：无替代文本]](https://huggingface.co/papers/2508.02739)


<a id="source-section-20"></a>

### [Kronos: A Foundation Model for the Language of Financial Markets](https://huggingface.co/papers/2508.02739)


Kronos, a specialized pre-training framework for financial K-line data, outperforms existing models in forecasting and synthetic data generation through a unique tokenizer and autoregressive pre-training on a large dataset.


-

-

-

-

-

- 7 authors

· Aug 2, 2025


[Upvote

58](https://huggingface.co/login?next=%2Fpapers%2F2508.02739)


[GitHub 40.1k](https://github.com/shiyu-coder/Kronos) [arXiv Page](https://arxiv.org/abs/2508.02739)


[[图片：无替代文本]](https://huggingface.co/papers/2609.33439)

Submitted by

[图片：无替代文本]

LivXue


<a id="source-section-21"></a>

### [Raven: The Harness of Harnesses for Composable Agentic Intelligence](https://huggingface.co/papers/2609.33439)


As large language models advance, AI agents are moving beyond isolated, domain-specific tasks toward long-horizon, cross-domain workflows. This transition exposes two challenges: increasing harness complexity makes manual design difficult to scale, while tighter coupling to specific domains limits the generality of a single harness. The central question thus shifts from how to engineer a stronger harness for one domain to how to autonomously construct specialized harnesses, improve them through experience, and orchestrate them across domains. We introduce Raven, The Harness of Harnesses, an open-source multi-agent ecosystem that automatically constructs and evolves modular harnesses for specific models and domains, treating each executable model--harness pair as a composable unit of intelligence. To support an All-Domain Collaboration Network, its Host Agent decomposes goals, matches subtasks to specialized agents, coordinates execution dependencies, and integrates results, while a host archive and EverOS preserve experience across tasks and Skill Forge makes that experience available as reusable procedures. Our theory establishes sufficient conditions for such composition to expand reliable task coverage beyond that of the available individual agents under a shared resource budget. On complex and long-horizon tasks, Raven significantly outperforms the state-of-the-art agent systems, pushing the frontier of composable agentic intelligence.


[[图片：EverMindAI] EverMind](https://huggingface.co/EverMindAI) · Published on Sep 27, 2026


[Upvote

566](https://huggingface.co/login?next=%2Fpapers%2F2609.33439)

[GitHub 5.24k](https://github.com/EverMind-AI/Raven) [arXiv Page](https://arxiv.org/abs/2609.33439)


[[图片：无替代文本]](https://huggingface.co/papers/2609.33439)

Submitted by

[图片：无替代文本]

LivXue


<a id="source-section-22"></a>

### [Raven: The Harness of Harnesses for Composable Agentic Intelligence](https://huggingface.co/papers/2609.33439)


As large language models advance, AI agents are moving beyond isolated, domain-specific tasks toward long-horizon, cross-domain workflows. This transition exposes two challenges: increasing harness complexity makes manual design difficult to scale, while tighter coupling to specific domains limits the generality of a single harness. The central question thus shifts from how to engineer a stronger harness for one domain to how to autonomously construct specialized harnesses, improve them through experience, and orchestrate them across domains. We introduce Raven, The Harness of Harnesses, an open-source multi-agent ecosystem that automatically constructs and evolves modular harnesses for specific models and domains, treating each executable model--harness pair as a composable unit of intelligence. To support an All-Domain Collaboration Network, its Host Agent decomposes goals, matches subtasks to specialized agents, coordinates execution dependencies, and integrates results, while a host archive and EverOS preserve experience across tasks and Skill Forge makes that experience available as reusable procedures. Our theory establishes sufficient conditions for such composition to expand reliable task coverage beyond that of the available individual agents under a shared resource budget. On complex and long-horizon tasks, Raven significantly outperforms the state-of-the-art agent systems, pushing the frontier of composable agentic intelligence.


[[图片：EverMindAI] EverMind](https://huggingface.co/EverMindAI) · Sep 27, 2026


[Upvote

566](https://huggingface.co/login?next=%2Fpapers%2F2609.33439)


[GitHub 5.24k](https://github.com/EverMind-AI/Raven) [arXiv Page](https://arxiv.org/abs/2609.33439)


Submitted by

[图片：无替代文本]

ruihong04


<a id="source-section-23"></a>

### [4DCodeBench: Benchmarking Agents on Inverse Graphics of Dynamic Scenes](https://huggingface.co/papers/2610.03715)


We introduce 4DCodeBench, a benchmark for 4D inverse graphics through code generation, in which agents reconstruct dynamic scenes from video as executable graphics programs. To accomplish this, agents must translate visual observations into compact representations of scene structure and dynamics, by implementing abstractions such as physical simulations to reproduce complex behavior. To evaluate this capability, we curate a set of real-world videos and construct synthetic scenes spanning diverse physical phenomena, including deformation, fluid flow, and fracture. We perform extensive benchmarking of frontier models, finding that strong static reconstruction capabilities do not yet translate into reliable reconstruction of complex dynamics. 4DCodeBench provides a testbed for tracking progress toward agents that can interpret the dynamics of the world through code. Our benchmark is available at https://github.com/4DCodeBench/4DCodeBench


[[图片：4DCodeBench] 4DCodeBench](https://huggingface.co/4DCodeBench) · Published on Oct 2, 2026


[Upvote

24](https://huggingface.co/login?next=%2Fpapers%2F2610.03715)

[GitHub 82](https://github.com/4DCodeBench/4DCodeBench) [arXiv Page](https://arxiv.org/abs/2610.03715)


Submitted by

[图片：无替代文本]

ruihong04


<a id="source-section-24"></a>

### [4DCodeBench: Benchmarking Agents on Inverse Graphics of Dynamic Scenes](https://huggingface.co/papers/2610.03715)


We introduce 4DCodeBench, a benchmark for 4D inverse graphics through code generation, in which agents reconstruct dynamic scenes from video as executable graphics programs. To accomplish this, agents must translate visual observations into compact representations of scene structure and dynamics, by implementing abstractions such as physical simulations to reproduce complex behavior. To evaluate this capability, we curate a set of real-world videos and construct synthetic scenes spanning diverse physical phenomena, including deformation, fluid flow, and fracture. We perform extensive benchmarking of frontier models, finding that strong static reconstruction capabilities do not yet translate into reliable reconstruction of complex dynamics. 4DCodeBench provides a testbed for tracking progress toward agents that can interpret the dynamics of the world through code. Our benchmark is available at https://github.com/4DCodeBench/4DCodeBench


[[图片：4DCodeBench] 4DCodeBench](https://huggingface.co/4DCodeBench) · Oct 2, 2026


[Upvote

24](https://huggingface.co/login?next=%2Fpapers%2F2610.03715)


[GitHub 82](https://github.com/4DCodeBench/4DCodeBench) [arXiv Page](https://arxiv.org/abs/2610.03715)


[[图片：无替代文本]](https://huggingface.co/papers/2609.37725)

Submitted by

[图片：无替代文本]

rulins


<a id="source-section-25"></a>

### [Context Language Models](https://huggingface.co/papers/2609.37725)


We introduce Context Language Models (CLMs), language models that natively manage their own context. We implement this by treating the context as a file and allowing the model to make unrestricted updates to this file. This allows the model to learn what is most important to maintain in context, and naturally extends to multi-agent systems where multiple agent contexts coexist as files. Building CLMs zero-shot with existing models outperforms SOTA context management strategies across a variety of tasks: 11.4% higher accuracy with 21.5% fewer FLOPs on BrowseComp-Plus, 5% higher scores with 59% fewer FLOPs on 12-hour EdgeBench, and 65% greater improvement with the same compute on a 24-hour multi-repository agent-swarm task. Moreover, by shifting context management from external harness control to intrinsic model behavior, CLMs naturally enable both in-context and parametric learning of context-management strategies. We show that CLMs can be steered with natural-language instructions evolved through a standard skill-optimization loop, improving held-out accuracy by up to 35.9 points on a context-management task while reducing compute. We also introduce an online reinforcement learning method for CLMs, improving Qwen3.5-9B performance on BrowseComp-Plus by 47.6% while using 12% fewer FLOPs. Finally, we co-design Suffix Cache Reuse for CLM serving, further reducing server-side compute by 35% relative to standard SGLang at matched performance.


[[图片：meta] Meta](https://huggingface.co/meta) · Published on Sep 29, 2026


[Upvote

41](https://huggingface.co/login?next=%2Fpapers%2F2609.37725)

[GitHub 587](https://github.com/facebookresearch/context-language-models) [arXiv Page](https://arxiv.org/abs/2609.37725)


[[图片：无替代文本]](https://huggingface.co/papers/2609.37725)

Submitted by

[图片：无替代文本]

rulins


<a id="source-section-26"></a>

### [Context Language Models](https://huggingface.co/papers/2609.37725)


We introduce Context Language Models (CLMs), language models that natively manage their own context. We implement this by treating the context as a file and allowing the model to make unrestricted updates to this file. This allows the model to learn what is most important to maintain in context, and naturally extends to multi-agent systems where multiple agent contexts coexist as files. Building CLMs zero-shot with existing models outperforms SOTA context management strategies across a variety of tasks: 11.4% higher accuracy with 21.5% fewer FLOPs on BrowseComp-Plus, 5% higher scores with 59% fewer FLOPs on 12-hour EdgeBench, and 65% greater improvement with the same compute on a 24-hour multi-repository agent-swarm task. Moreover, by shifting context management from external harness control to intrinsic model behavior, CLMs naturally enable both in-context and parametric learning of context-management strategies. We show that CLMs can be steered with natural-language instructions evolved through a standard skill-optimization loop, improving held-out accuracy by up to 35.9 points on a context-management task while reducing compute. We also introduce an online reinforcement learning method for CLMs, improving Qwen3.5-9B performance on BrowseComp-Plus by 47.6% while using 12% fewer FLOPs. Finally, we co-design Suffix Cache Reuse for CLM serving, further reducing server-side compute by 35% relative to standard SGLang at matched performance.


[[图片：meta] Meta](https://huggingface.co/meta) · Sep 29, 2026


[Upvote

41](https://huggingface.co/login?next=%2Fpapers%2F2609.37725)


[GitHub 587](https://github.com/facebookresearch/context-language-models) [arXiv Page](https://arxiv.org/abs/2609.37725)


[[图片：无替代文本]](https://huggingface.co/papers/2504.19413)

Submitted by

[图片：无替代文本]

akhaliq


<a id="source-section-27"></a>

### [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://huggingface.co/papers/2504.19413)


Mem0, a memory-centric architecture with graph-based memory, enhances long-term conversational coherence in LLMs by efficiently extracting, consolidating, and retrieving information, outperforming existing memory systems in terms of accuracy and computational efficiency.


-

-

-

- [图片：无替代文本]

- [图片：无替代文本]

- 5 authors

· Published on Apr 28, 2025


[Upvote

73](https://huggingface.co/login?next=%2Fpapers%2F2504.19413)

[GitHub 66.7k](https://github.com/mem0ai/mem0) [arXiv Page](https://arxiv.org/abs/2504.19413)


[[图片：无替代文本]](https://huggingface.co/papers/2504.19413)

Submitted by

[图片：无替代文本]

akhaliq


<a id="source-section-28"></a>

### [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://huggingface.co/papers/2504.19413)


Mem0, a memory-centric architecture with graph-based memory, enhances long-term conversational coherence in LLMs by efficiently extracting, consolidating, and retrieving information, outperforming existing memory systems in terms of accuracy and computational efficiency.


-

-

-

- [图片：无替代文本]

- [图片：无替代文本]

- 5 authors

· Apr 28, 2025


[Upvote

73](https://huggingface.co/login?next=%2Fpapers%2F2504.19413)


[GitHub 66.7k](https://github.com/mem0ai/mem0) [arXiv Page](https://arxiv.org/abs/2504.19413)


[[图片：无替代文本]](https://huggingface.co/papers/2610.05416)

Submitted by

[图片：无替代文本]

FrancisRing


<a id="source-section-29"></a>

### [Prism: Dynamic Sparse Attention for Native 2K Joint Video-Audio Generation Model Training](https://huggingface.co/papers/2610.05416)


Natively training joint video-audio generation models at higher resolutions empowers them to learn richer visual details and sharper motion dynamics. However, full attention incurs quadratic cost and, as resolution increases, spreads attention over increasingly redundant tokens, diluting learning signals for informative content and disrupting pretrained priors. Existing sparse attention methods either target training-free acceleration or overlook the unique structure of joint video-audio data, where cross-modal interactions are inherently concentrated around sound-producing regions. To address this, we propose Prism, a dynamic sparse attention framework for natively training joint video-audio generation models at 2K. In particular, Prism organizes the token sequence into spatiotemporal macro-zones, enabling the attention structure to adapt to local content. For each zone, it estimates local information structure via video feature variance along the channel and feature norms from the audio-to-video cross-attention, jointly capturing how visual content varies directionally and how strongly audio influences each visual region. Based on these signals, Prism dynamically assigns a tailored block shape to each zone, applying finer partitioning along axes of rapid visual content variation and strong audio-visual coupling. This encourages tokens within each block to remain semantically coherent, allowing block-level features to capture both visual content and joint video-audio interaction patterns. Prism further adopts a hybrid block selection strategy to dynamically determine per-query sparsity. Experiments show that Prism achieves 2.5times training speedup compared to full attention, while surpassing it in generation quality.


[[图片：Tencent-Hunyuan] Tencent Hunyuan](https://huggingface.co/Tencent-Hunyuan) · Published on Oct 4, 2026


[Upvote

8](https://huggingface.co/login?next=%2Fpapers%2F2610.05416)

[GitHub 41](https://github.com/Tencent-Hunyuan/Prism) [arXiv Page](https://arxiv.org/abs/2610.05416)


[[图片：无替代文本]](https://huggingface.co/papers/2610.05416)

Submitted by

[图片：无替代文本]

FrancisRing


<a id="source-section-30"></a>

### [Prism: Dynamic Sparse Attention for Native 2K Joint Video-Audio Generation Model Training](https://huggingface.co/papers/2610.05416)


Natively training joint video-audio generation models at higher resolutions empowers them to learn richer visual details and sharper motion dynamics. However, full attention incurs quadratic cost and, as resolution increases, spreads attention over increasingly redundant tokens, diluting learning signals for informative content and disrupting pretrained priors. Existing sparse attention methods either target training-free acceleration or overlook the unique structure of joint video-audio data, where cross-modal interactions are inherently concentrated around sound-producing regions. To address this, we propose Prism, a dynamic sparse attention framework for natively training joint video-audio generation models at 2K. In particular, Prism organizes the token sequence into spatiotemporal macro-zones, enabling the attention structure to adapt to local content. For each zone, it estimates local information structure via video feature variance along the channel and feature norms from the audio-to-video cross-attention, jointly capturing how visual content varies directionally and how strongly audio influences each visual region. Based on these signals, Prism dynamically assigns a tailored block shape to each zone, applying finer partitioning along axes of rapid visual content variation and strong audio-visual coupling. This encourages tokens within each block to remain semantically coherent, allowing block-level features to capture both visual content and joint video-audio interaction patterns. Prism further adopts a hybrid block selection strategy to dynamically determine per-query sparsity. Experiments show that Prism achieves 2.5times training speedup compared to full attention, while surpassing it in generation quality.


[[图片：Tencent-Hunyuan] Tencent Hunyuan](https://huggingface.co/Tencent-Hunyuan) · Oct 4, 2026


[Upvote

8](https://huggingface.co/login?next=%2Fpapers%2F2610.05416)


[GitHub 41](https://github.com/Tencent-Hunyuan/Prism) [arXiv Page](https://arxiv.org/abs/2610.05416)


[[图片：无替代文本]](https://huggingface.co/papers/2609.08183)

Submitted by

[图片：无替代文本]

JarvisPei


<a id="source-section-31"></a>

### [NeoHorse-1: Towards Recursive Self-Improvement via Agentic Post-Training with Routing Harness](https://huggingface.co/papers/2609.08183)


NeoHorse-1 uses agentic post-training with intelligent routing, structured feedback loops, and curriculum-based distillation to improve model capabilities across agent benchmarks.


[[图片：TokenRhythm] TokenRhythm](https://huggingface.co/TokenRhythm) · Published on Sep 8, 2026


[Upvote

327](https://huggingface.co/login?next=%2Fpapers%2F2609.08183)

[GitHub 1.66k](https://github.com/TokenRhythm/NeoHorse) [arXiv Page](https://arxiv.org/abs/2609.08183)


[[图片：无替代文本]](https://huggingface.co/papers/2609.08183)

Submitted by

[图片：无替代文本]

JarvisPei


<a id="source-section-32"></a>

### [NeoHorse-1: Towards Recursive Self-Improvement via Agentic Post-Training with Routing Harness](https://huggingface.co/papers/2609.08183)


NeoHorse-1 uses agentic post-training with intelligent routing, structured feedback loops, and curriculum-based distillation to improve model capabilities across agent benchmarks.


[[图片：TokenRhythm] TokenRhythm](https://huggingface.co/TokenRhythm) · Sep 8, 2026


[Upvote

327](https://huggingface.co/login?next=%2Fpapers%2F2609.08183)


[GitHub 1.66k](https://github.com/TokenRhythm/NeoHorse) [arXiv Page](https://arxiv.org/abs/2609.08183)


[[图片：无替代文本]](https://huggingface.co/papers/2606.03264)

Submitted by

[图片：无替代文本]

ChengCui


<a id="source-section-33"></a>

### [PaddleOCR-VL-1.6: Expanding the Frontier of Document Parsing with Under-Optimized Region Refinement and Progressive Post-Training](https://huggingface.co/papers/2606.03264)


PaddleOCR-VL-1.6 enhances document parsing performance through targeted data optimization and progressive post-training techniques, achieving state-of-the-art results on OmniDocBench v1.6.


[[图片：PaddlePaddle] PaddlePaddle](https://huggingface.co/PaddlePaddle) · Published on Jun 2, 2026


[Upvote

26](https://huggingface.co/login?next=%2Fpapers%2F2606.03264)

[GitHub 90.7k](https://github.com/PaddlePaddle/PaddleOCR) [arXiv Page](https://arxiv.org/abs/2606.03264)


[[图片：无替代文本]](https://huggingface.co/papers/2606.03264)

Submitted by

[图片：无替代文本]

ChengCui


<a id="source-section-34"></a>

### [PaddleOCR-VL-1.6: Expanding the Frontier of Document Parsing with Under-Optimized Region Refinement and Progressive Post-Training](https://huggingface.co/papers/2606.03264)


PaddleOCR-VL-1.6 enhances document parsing performance through targeted data optimization and progressive post-training techniques, achieving state-of-the-art results on OmniDocBench v1.6.


[[图片：PaddlePaddle] PaddlePaddle](https://huggingface.co/PaddlePaddle) · Jun 2, 2026


[Upvote

26](https://huggingface.co/login?next=%2Fpapers%2F2606.03264)


[GitHub 90.7k](https://github.com/PaddlePaddle/PaddleOCR) [arXiv Page](https://arxiv.org/abs/2606.03264)


[[图片：无替代文本]](https://huggingface.co/papers/2609.33757)

Submitted by

[图片：无替代文本]

a43992899


<a id="source-section-35"></a>

### [YuE2: Unifying Symbolic and Audio Music Generation at Frontier Quality](https://huggingface.co/papers/2609.33757)


Symbolic models make melody, harmony, rhythm, and form explicit but typically stop before a finished recording; audio models produce complete songs while leaving composition implicit. We introduce YuE2, which unifies symbolic and audio music generation at frontier quality through symbolic planning. A single AR-NAR Mixture-of-Transformers (MoT) first writes a readable score specifying melody and harmony, expands it into semantic music tokens, and realizes it as full-song audio. In comparisons using the same checkpoint, experts prefer symbolic planning for overall quality and musicality, with 49.3% of overall preferences versus 34.6% without planning. Experts also favor the unified model over a separate language model and diffusion Transformer. On WildSongBench, YuE2 scores 6.73 on SongBench Global Avg, exceeding all evaluated public baselines. Selecting from eight candidates (best-of-8), YuE2 reaches 6.96, the highest observed mean among all evaluated systems. Expert listening further establishes its competitiveness with proprietary song generators, favoring best-of-8 over Suno v4.5 and yielding nearly balanced preferences against Suno v5. To learn this generation process from recordings without aligned scores, we introduce MERT2 and SheetSage2 to supply semantic and symbolic supervision. MERT2 sets a new state of the art in music representation learning, surpassing previous best results on 14 of 15 MARBLE metrics; SheetSage2 leads 12 of 15 benchmark-metric pairs in our lead-sheet transcription comparison. The same checkpoint follows score edits while largely preserving unedited musical content and generates zero-shot covers without cover-specific training. Its readable score also enables agentic music editing, with external language models translating user feedback into revisions of the composition.


[[图片：m-a-p] Multimodal Art Projection](https://huggingface.co/m-a-p) · Published on Sep 27, 2026


[Upvote

246](https://huggingface.co/login?next=%2Fpapers%2F2609.33757)

[GitHub 10.9k](https://github.com/multimodal-art-projection/YuE) [arXiv Page](https://arxiv.org/abs/2609.33757)


[[图片：无替代文本]](https://huggingface.co/papers/2609.33757)

Submitted by

[图片：无替代文本]

a43992899


<a id="source-section-36"></a>

### [YuE2: Unifying Symbolic and Audio Music Generation at Frontier Quality](https://huggingface.co/papers/2609.33757)


Symbolic models make melody, harmony, rhythm, and form explicit but typically stop before a finished recording; audio models produce complete songs while leaving composition implicit. We introduce YuE2, which unifies symbolic and audio music generation at frontier quality through symbolic planning. A single AR-NAR Mixture-of-Transformers (MoT) first writes a readable score specifying melody and harmony, expands it into semantic music tokens, and realizes it as full-song audio. In comparisons using the same checkpoint, experts prefer symbolic planning for overall quality and musicality, with 49.3% of overall preferences versus 34.6% without planning. Experts also favor the unified model over a separate language model and diffusion Transformer. On WildSongBench, YuE2 scores 6.73 on SongBench Global Avg, exceeding all evaluated public baselines. Selecting from eight candidates (best-of-8), YuE2 reaches 6.96, the highest observed mean among all evaluated systems. Expert listening further establishes its competitiveness with proprietary song generators, favoring best-of-8 over Suno v4.5 and yielding nearly balanced preferences against Suno v5. To learn this generation process from recordings without aligned scores, we introduce MERT2 and SheetSage2 to supply semantic and symbolic supervision. MERT2 sets a new state of the art in music representation learning, surpassing previous best results on 14 of 15 MARBLE metrics; SheetSage2 leads 12 of 15 benchmark-metric pairs in our lead-sheet transcription comparison. The same checkpoint follows score edits while largely preserving unedited musical content and generates zero-shot covers without cover-specific training. Its readable score also enables agentic music editing, with external language models translating user feedback into revisions of the composition.


[[图片：m-a-p] Multimodal Art Projection](https://huggingface.co/m-a-p) · Sep 27, 2026


[Upvote

246](https://huggingface.co/login?next=%2Fpapers%2F2609.33757)


[GitHub 10.9k](https://github.com/multimodal-art-projection/YuE) [arXiv Page](https://arxiv.org/abs/2609.33757)


[[图片：无替代文本]](https://huggingface.co/papers/2503.11576)

Submitted by

[图片：无替代文本]

andito


<a id="source-section-37"></a>

### [SmolDocling: An ultra-compact vision-language model for end-to-end
multi-modal document conversion](https://huggingface.co/papers/2503.11576)


SmolDocling is a compact vision-language model that performs end-to-end document conversion with robust performance across various document types using 256M parameters and a new markup format.


[[图片：ibm-granite] IBM Granite](https://huggingface.co/ibm-granite) · Published on Mar 14, 2025


[Upvote

177](https://huggingface.co/login?next=%2Fpapers%2F2503.11576)

[GitHub 68.5k](https://github.com/docling-project/docling) [arXiv Page](https://arxiv.org/abs/2503.11576)


[[图片：无替代文本]](https://huggingface.co/papers/2503.11576)

Submitted by

[图片：无替代文本]

andito


<a id="source-section-38"></a>

### [SmolDocling: An ultra-compact vision-language model for end-to-end
multi-modal document conversion](https://huggingface.co/papers/2503.11576)


SmolDocling is a compact vision-language model that performs end-to-end document conversion with robust performance across various document types using 256M parameters and a new markup format.


[[图片：ibm-granite] IBM Granite](https://huggingface.co/ibm-granite) · Mar 14, 2025


[Upvote

177](https://huggingface.co/login?next=%2Fpapers%2F2503.11576)


[GitHub 68.5k](https://github.com/docling-project/docling) [arXiv Page](https://arxiv.org/abs/2503.11576)


[[图片：无替代文本]](https://huggingface.co/papers/2609.24972)

Submitted by

[图片：无替代文本]

richardxp888


<a id="source-section-39"></a>

### [RRSI: Regularized Recursive Self-Improvement of Agent Harnesses](https://huggingface.co/papers/2609.24972)


An LLM agent's capability is largely magnified by its harness, namely the prompts, control flow, tooling, memory, and context management surrounding the frozen backbone model. Recent methods increasingly automate this process by iteratively proposing and selecting component-wise edits of an agent harness, practically establishing a form of recursive self-improvement (RSI) at the agent-system level. However, such recursive evolution may overfit by memorizing the training tasks, showing large in-distribution gains that shrink or even vanish on out-of-distribution benchmarks. We introduce Regularized Recursive Self-Improvement of Agent Harnesses (RRSI), which incorporates the principles of regularizations into harness self-improvement by constraining the evolution candidate proposal and selection. The proposer operates with a temporally annealed budget, limiting how many edits a candidate can bundle, and it encourages unexplored trajectories based on evolution history. The selector is equipped with a critic and a pruner: the critic screens benchmark-specific proposals, while the pruner, removes changes that are too small, too expensive, or no longer useful. Together these constraints favor reusable agent mechanisms over benchmark-specific ones or even noises. Across eight benchmarks spanning coding, agentic workspace and engineering design tasks, RRSI gains up to 14.1 points on the split it evolves against and up to 4.7 points on the five out-of-distribution benchmarks, while producing a harness that runs on 30% fewer policy tokens than the unregularized evolution. Code is available at https://github.com/google-research/rrsi and project page is https://regularized-rsi.com/.


[[图片：google] Google](https://huggingface.co/google) · Published on Sep 21, 2026


[Upvote

222](https://huggingface.co/login?next=%2Fpapers%2F2609.24972)

[GitHub 1.28k](https://github.com/google-research/rrsi) [arXiv Page](https://arxiv.org/abs/2609.24972)


[[图片：无替代文本]](https://huggingface.co/papers/2609.24972)

Submitted by

[图片：无替代文本]

richardxp888


<a id="source-section-40"></a>

### [RRSI: Regularized Recursive Self-Improvement of Agent Harnesses](https://huggingface.co/papers/2609.24972)


An LLM agent's capability is largely magnified by its harness, namely the prompts, control flow, tooling, memory, and context management surrounding the frozen backbone model. Recent methods increasingly automate this process by iteratively proposing and selecting component-wise edits of an agent harness, practically establishing a form of recursive self-improvement (RSI) at the agent-system level. However, such recursive evolution may overfit by memorizing the training tasks, showing large in-distribution gains that shrink or even vanish on out-of-distribution benchmarks. We introduce Regularized Recursive Self-Improvement of Agent Harnesses (RRSI), which incorporates the principles of regularizations into harness self-improvement by constraining the evolution candidate proposal and selection. The proposer operates with a temporally annealed budget, limiting how many edits a candidate can bundle, and it encourages unexplored trajectories based on evolution history. The selector is equipped with a critic and a pruner: the critic screens benchmark-specific proposals, while the pruner, removes changes that are too small, too expensive, or no longer useful. Together these constraints favor reusable agent mechanisms over benchmark-specific ones or even noises. Across eight benchmarks spanning coding, agentic workspace and engineering design tasks, RRSI gains up to 14.1 points on the split it evolves against and up to 4.7 points on the five out-of-distribution benchmarks, while producing a harness that runs on 30% fewer policy tokens than the unregularized evolution. Code is available at https://github.com/google-research/rrsi and project page is https://regularized-rsi.com/.


[[图片：google] Google](https://huggingface.co/google) · Sep 21, 2026


[Upvote

222](https://huggingface.co/login?next=%2Fpapers%2F2609.24972)


[GitHub 1.28k](https://github.com/google-research/rrsi) [arXiv Page](https://arxiv.org/abs/2609.24972)


[[图片：无替代文本]](https://huggingface.co/papers/2509.22186)

Submitted by

[图片：无替代文本]

taesiri


<a id="source-section-41"></a>

### [MinerU2.5: A Decoupled Vision-Language Model for Efficient
High-Resolution Document Parsing](https://huggingface.co/papers/2509.22186)


MinerU2.5, a 1.2B-parameter document parsing vision-language model, achieves state-of-the-art recognition accuracy with computational efficiency through a coarse-to-fine parsing strategy.


- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- 61 authors

· Published on Sep 26, 2025


[Upvote

180](https://huggingface.co/login?next=%2Fpapers%2F2509.22186)

[GitHub 81.2k](https://github.com/opendatalab/MinerU) [arXiv Page](https://arxiv.org/abs/2509.22186)


[[图片：无替代文本]](https://huggingface.co/papers/2509.22186)

Submitted by

[图片：无替代文本]

taesiri


<a id="source-section-42"></a>

### [MinerU2.5: A Decoupled Vision-Language Model for Efficient
High-Resolution Document Parsing](https://huggingface.co/papers/2509.22186)


MinerU2.5, a 1.2B-parameter document parsing vision-language model, achieves state-of-the-art recognition accuracy with computational efficiency through a coarse-to-fine parsing strategy.


- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- 61 authors

· Sep 26, 2025


[Upvote

180](https://huggingface.co/login?next=%2Fpapers%2F2509.22186)


[GitHub 81.2k](https://github.com/opendatalab/MinerU) [arXiv Page](https://arxiv.org/abs/2509.22186)


[[图片：无替代文本]](https://huggingface.co/papers/2608.23283)

Submitted by

[图片：无替代文本]

oriuta


<a id="source-section-43"></a>

### [Apodex 1.1: Scaling Agentic Intelligence for Complex Work](https://huggingface.co/papers/2608.23283)


Apodex 1.1 improves sustained, verifiable progress on complex real-world tasks by scaling executable environments and training agents to coordinate long-horizon work with state maintenance and recovery.


[[图片：apodex] Apodex](https://huggingface.co/apodex) · Published on Aug 24, 2026


[Upvote

212](https://huggingface.co/login?next=%2Fpapers%2F2608.23283)

[GitHub 5.14k](https://github.com/ApodexAI/FrontierAgent) [arXiv Page](https://arxiv.org/abs/2608.23283)


[[图片：无替代文本]](https://huggingface.co/papers/2608.23283)

Submitted by

[图片：无替代文本]

oriuta


<a id="source-section-44"></a>

### [Apodex 1.1: Scaling Agentic Intelligence for Complex Work](https://huggingface.co/papers/2608.23283)


Apodex 1.1 improves sustained, verifiable progress on complex real-world tasks by scaling executable environments and training agents to coordinate long-horizon work with state maintenance and recovery.


[[图片：apodex] Apodex](https://huggingface.co/apodex) · Aug 24, 2026


[Upvote

212](https://huggingface.co/login?next=%2Fpapers%2F2608.23283)


[GitHub 5.14k](https://github.com/ApodexAI/FrontierAgent) [arXiv Page](https://arxiv.org/abs/2608.23283)


Submitted by

[图片：无替代文本]

Lanxingxuan


<a id="source-section-45"></a>

### [OneStreamer: Unifying Perception, Memory, and Proactive Response in Streaming Video Interaction](https://huggingface.co/papers/2610.01762)


Streaming video LLMs must retain evidence before its relevance to future tasks is known and respond when sufficient evidence becomes available. The challenge is to form reusable factual memory without compromising real-time perception. We introduce OneStreamer, which jointly learns query-independent evidence recording and task response through a shared proactive generation process. Its Proactive Hierarchical Caption Memory (PHCM) produces time-grounded local-detail captions and summaries of completed events. Streaming caption targets supervise the interpretation of observed video prefixes during training. At inference, model-generated records complement a recent visual window, providing reusable factual context without revisiting historical visual features. Proactive State Transition Learning (PSTL) reduces the dominance of repeated waiting states by preserving supervision at all output anchors and selecting representative state-change and state-persistence tokens. We further develop a streaming data synthesis pipeline that aligns output content and timing with available evidence. Combining the resulting streaming captions and QA with cleaned open-source data yields OneStreamer-1M, a broad-coverage streaming video interaction dataset with over one million records spanning diverse tasks. Our 4B model achieves the best results among the compared methods across all eight evaluated streaming video understanding benchmarks. Ablations show that retaining generated captions improves historical QA without degrading real-time perception. PSTL also outperforms dense state supervision while supervising only 27.5% of annotated state tokens. Together, these results support proactive generation as a shared learning interface connecting perception, memory formation, and timely response in streaming video interaction.


[[图片：NJU] Nanjing University](https://huggingface.co/NJU) · Published on Oct 1, 2026


[Upvote

230](https://huggingface.co/login?next=%2Fpapers%2F2610.01762)

[GitHub 157](https://github.com/MCG-NJU/OneStreamer) [arXiv Page](https://arxiv.org/abs/2610.01762)


Submitted by

[图片：无替代文本]

Lanxingxuan


<a id="source-section-46"></a>

### [OneStreamer: Unifying Perception, Memory, and Proactive Response in Streaming Video Interaction](https://huggingface.co/papers/2610.01762)


Streaming video LLMs must retain evidence before its relevance to future tasks is known and respond when sufficient evidence becomes available. The challenge is to form reusable factual memory without compromising real-time perception. We introduce OneStreamer, which jointly learns query-independent evidence recording and task response through a shared proactive generation process. Its Proactive Hierarchical Caption Memory (PHCM) produces time-grounded local-detail captions and summaries of completed events. Streaming caption targets supervise the interpretation of observed video prefixes during training. At inference, model-generated records complement a recent visual window, providing reusable factual context without revisiting historical visual features. Proactive State Transition Learning (PSTL) reduces the dominance of repeated waiting states by preserving supervision at all output anchors and selecting representative state-change and state-persistence tokens. We further develop a streaming data synthesis pipeline that aligns output content and timing with available evidence. Combining the resulting streaming captions and QA with cleaned open-source data yields OneStreamer-1M, a broad-coverage streaming video interaction dataset with over one million records spanning diverse tasks. Our 4B model achieves the best results among the compared methods across all eight evaluated streaming video understanding benchmarks. Ablations show that retaining generated captions improves historical QA without degrading real-time perception. PSTL also outperforms dense state supervision while supervising only 27.5% of annotated state tokens. Together, these results support proactive generation as a shared learning interface connecting perception, memory formation, and timely response in streaming video interaction.


[[图片：NJU] Nanjing University](https://huggingface.co/NJU) · Oct 1, 2026


[Upvote

230](https://huggingface.co/login?next=%2Fpapers%2F2610.01762)


[GitHub 157](https://github.com/MCG-NJU/OneStreamer) [arXiv Page](https://arxiv.org/abs/2610.01762)


[[图片：无替代文本]](https://huggingface.co/papers/2605.03042)

Submitted by

[图片：无替代文本]

RuofengYang


<a id="source-section-47"></a>

### [ARIS: Autonomous Research via Adversarial Multi-Agent Collaboration](https://huggingface.co/papers/2605.03042)


ARIS is an open-source research harness that uses cross-model adversarial collaboration to ensure reliable long-term research outcomes through coordinated execution, orchestration, and assurance layers.


[[图片：SJTU] Shanghai Jiao Tong University](https://huggingface.co/SJTU) · Published on May 4, 2026


[Upvote

154](https://huggingface.co/login?next=%2Fpapers%2F2605.03042)

[GitHub 17.1k](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) [arXiv Page](https://arxiv.org/abs/2605.03042)


[[图片：无替代文本]](https://huggingface.co/papers/2605.03042)

Submitted by

[图片：无替代文本]

RuofengYang


<a id="source-section-48"></a>

### [ARIS: Autonomous Research via Adversarial Multi-Agent Collaboration](https://huggingface.co/papers/2605.03042)


ARIS is an open-source research harness that uses cross-model adversarial collaboration to ensure reliable long-term research outcomes through coordinated execution, orchestration, and assurance layers.


[[图片：SJTU] Shanghai Jiao Tong University](https://huggingface.co/SJTU) · May 4, 2026


[Upvote

154](https://huggingface.co/login?next=%2Fpapers%2F2605.03042)


[GitHub 17.1k](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) [arXiv Page](https://arxiv.org/abs/2605.03042)


Submitted by

[图片：无替代文本]

CongWei1230


<a id="source-section-49"></a>

### [PixelUMM: Encoder-Free Unified Image and Video Understanding and Generation](https://huggingface.co/papers/2609.38597)


Unified Multimodal Models (UMMs) often rely on separate visual representations for understanding and generation, increasing visual context length and complicating integration with established vision-language pretraining pipelines. Recent advances in pixel-space modeling offer an encoder-free alternative, but extending this paradigm from images to videos is non-trivial: video understanding and generation adopt different temporal representations, leaving the design of a unified visual interface an open question. We present PixelUMM, an encoder-free model for unified image and video understanding and generation directly in pixel space. PixelUMM represents images as spatial patches and videos as spatiotemporal tubelets, connecting raw pixels to a shared multimodal backbone through single-layer linear projections. Its Mixture-of-Transformers architecture combines shared attention with task-specific parameters and extends clean-pixel prediction to video generation, jointly supporting autoregressive text prediction and pixel-space flow matching. Experiments show that PixelUMM achieves competitive performance across image and video understanding and generation tasks. We further conduct empirical studies of key design choices, including decoder design and spatial-temporal patch size, providing insights for future pixel-space unified multimodal models.


[[图片：nvidia] NVIDIA](https://huggingface.co/nvidia) · Published on Sep 29, 2026


[Upvote

32](https://huggingface.co/login?next=%2Fpapers%2F2609.38597)

[GitHub 160](https://github.com/nv-tlabs/PixelUMM) [arXiv Page](https://arxiv.org/abs/2609.38597)


Submitted by

[图片：无替代文本]

CongWei1230


<a id="source-section-50"></a>

### [PixelUMM: Encoder-Free Unified Image and Video Understanding and Generation](https://huggingface.co/papers/2609.38597)


Unified Multimodal Models (UMMs) often rely on separate visual representations for understanding and generation, increasing visual context length and complicating integration with established vision-language pretraining pipelines. Recent advances in pixel-space modeling offer an encoder-free alternative, but extending this paradigm from images to videos is non-trivial: video understanding and generation adopt different temporal representations, leaving the design of a unified visual interface an open question. We present PixelUMM, an encoder-free model for unified image and video understanding and generation directly in pixel space. PixelUMM represents images as spatial patches and videos as spatiotemporal tubelets, connecting raw pixels to a shared multimodal backbone through single-layer linear projections. Its Mixture-of-Transformers architecture combines shared attention with task-specific parameters and extends clean-pixel prediction to video generation, jointly supporting autoregressive text prediction and pixel-space flow matching. Experiments show that PixelUMM achieves competitive performance across image and video understanding and generation tasks. We further conduct empirical studies of key design choices, including decoder design and spatial-temporal patch size, providing insights for future pixel-space unified multimodal models.


[[图片：nvidia] NVIDIA](https://huggingface.co/nvidia) · Sep 29, 2026


[Upvote

32](https://huggingface.co/login?next=%2Fpapers%2F2609.38597)


[GitHub 160](https://github.com/nv-tlabs/PixelUMM) [arXiv Page](https://arxiv.org/abs/2609.38597)


[[图片：无替代文本]](https://huggingface.co/papers/2006.15704)


<a id="source-section-51"></a>

### [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://huggingface.co/papers/2006.15704)


The PyTorch distributed data parallel module optimizes large-scale model training using techniques like gradient bucketing, computation-communication overlap, and selective synchronization to achieve near-linear scalability.


-

-

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- 11 authors

· Published on Jun 28, 2020


[Upvote

13](https://huggingface.co/login?next=%2Fpapers%2F2006.15704)

[GitHub 104k](https://github.com/pytorch/pytorch) [arXiv Page](https://arxiv.org/abs/2006.15704)


[[图片：无替代文本]](https://huggingface.co/papers/2006.15704)


<a id="source-section-52"></a>

### [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://huggingface.co/papers/2006.15704)


The PyTorch distributed data parallel module optimizes large-scale model training using techniques like gradient bucketing, computation-communication overlap, and selective synchronization to achieve near-linear scalability.


-

-

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- 11 authors

· Jun 28, 2020


[Upvote

13](https://huggingface.co/login?next=%2Fpapers%2F2006.15704)


[GitHub 104k](https://github.com/pytorch/pytorch) [arXiv Page](https://arxiv.org/abs/2006.15704)


[[图片：无替代文本]](https://huggingface.co/papers/2605.23904)

Submitted by

[图片：无替代文本]

taesiri


<a id="source-section-53"></a>

### [SkillOpt: Executive Strategy for Self-Evolving Agent Skills](https://huggingface.co/papers/2605.23904)


SkillOpt introduces a systematic text-space optimizer for agent skills that trains skills as external agent state with stable updates and zero deployment inference overhead, achieving superior performance across multiple benchmarks and execution environments.


[[图片：MicrosoftResearch] Microsoft Research](https://huggingface.co/MicrosoftResearch) · Published on May 22, 2026


[Upvote

266](https://huggingface.co/login?next=%2Fpapers%2F2605.23904)

[GitHub 18.1k](https://github.com/microsoft/SkillOpt) [arXiv Page](https://arxiv.org/abs/2605.23904)


[[图片：无替代文本]](https://huggingface.co/papers/2605.23904)

Submitted by

[图片：无替代文本]

taesiri


<a id="source-section-54"></a>

### [SkillOpt: Executive Strategy for Self-Evolving Agent Skills](https://huggingface.co/papers/2605.23904)


SkillOpt introduces a systematic text-space optimizer for agent skills that trains skills as external agent state with stable updates and zero deployment inference overhead, achieving superior performance across multiple benchmarks and execution environments.


[[图片：MicrosoftResearch] Microsoft Research](https://huggingface.co/MicrosoftResearch) · May 22, 2026


[Upvote

266](https://huggingface.co/login?next=%2Fpapers%2F2605.23904)


[GitHub 18.1k](https://github.com/microsoft/SkillOpt) [arXiv Page](https://arxiv.org/abs/2605.23904)


[[图片：无替代文本]](https://huggingface.co/papers/2609.20800)

Submitted by

[图片：无替代文本]

taesiri


<a id="source-section-55"></a>

### [JEPA-Anything: Learning Predictive Models across Different Worlds](https://huggingface.co/papers/2609.20800)


World modeling enables intelligence to anticipate consequences, guide interventions, and learn from interaction. Yet predictive models remain domain-specific: can a common learning principle support world modeling across radically different systems? We introduce JEPA-Anything, a domain-agnostic framework based on orthogonal predictive factorization (OPF). Extending joint-embedding predictive architectures, OPF decomposes latent targets into complementary factors, learns them through dedicated pathways, and recombines them within a shared predictive design. We evaluate JEPA-Anything across seven domains: vision, biology, clinical trajectories, control, molecular dynamics, physical fields, and weather. Experiments span representation learning, intervention prediction, out-of-distribution generalization, and long-horizon dynamics, including 10 matched dynamics tasks, forecasting of over 1,000 clinical events, and 100-step molecular rollouts across four systems. Against matched JEPA baselines, JEPA-Anything improves reported metrics on all 10 dynamics tasks and reduces single-intervention prediction error on Interventional Pong by 34.8%. It achieves the lowest one-step and 100-step molecular errors among compared methods in all four systems. Beyond prediction, a factor-nominated biological intervention receives experimental support in cell co-cultures, patient-derived organoids, tumor fragments, and mice; latent orbital modes recover the Keplerian scaling exponent with a fitted slope of -1.4991. These results support a common factorized predictive principle across heterogeneous worlds, connecting world modeling with intervention and experimentally grounded scientific discovery. Code: https://github.com/Gen-Verse/JEPA-Anything


-

-

-

-

-

- 13 authors

· Published on Sep 17, 2026


[Upvote

77](https://huggingface.co/login?next=%2Fpapers%2F2609.20800)

[GitHub 261](https://github.com/Gen-Verse/JEPA-Anything) [arXiv Page](https://arxiv.org/abs/2609.20800)


[[图片：无替代文本]](https://huggingface.co/papers/2609.20800)

Submitted by

[图片：无替代文本]

taesiri


<a id="source-section-56"></a>

### [JEPA-Anything: Learning Predictive Models across Different Worlds](https://huggingface.co/papers/2609.20800)


World modeling enables intelligence to anticipate consequences, guide interventions, and learn from interaction. Yet predictive models remain domain-specific: can a common learning principle support world modeling across radically different systems? We introduce JEPA-Anything, a domain-agnostic framework based on orthogonal predictive factorization (OPF). Extending joint-embedding predictive architectures, OPF decomposes latent targets into complementary factors, learns them through dedicated pathways, and recombines them within a shared predictive design. We evaluate JEPA-Anything across seven domains: vision, biology, clinical trajectories, control, molecular dynamics, physical fields, and weather. Experiments span representation learning, intervention prediction, out-of-distribution generalization, and long-horizon dynamics, including 10 matched dynamics tasks, forecasting of over 1,000 clinical events, and 100-step molecular rollouts across four systems. Against matched JEPA baselines, JEPA-Anything improves reported metrics on all 10 dynamics tasks and reduces single-intervention prediction error on Interventional Pong by 34.8%. It achieves the lowest one-step and 100-step molecular errors among compared methods in all four systems. Beyond prediction, a factor-nominated biological intervention receives experimental support in cell co-cultures, patient-derived organoids, tumor fragments, and mice; latent orbital modes recover the Keplerian scaling exponent with a fitted slope of -1.4991. These results support a common factorized predictive principle across heterogeneous worlds, connecting world modeling with intervention and experimentally grounded scientific discovery. Code: https://github.com/Gen-Verse/JEPA-Anything


-

-

-

-

-

- 13 authors

· Sep 17, 2026


[Upvote

77](https://huggingface.co/login?next=%2Fpapers%2F2609.20800)


[GitHub 261](https://github.com/Gen-Verse/JEPA-Anything) [arXiv Page](https://arxiv.org/abs/2609.20800)


[[图片：无替代文本]](https://huggingface.co/papers/2608.16157)

Submitted by

[图片：无替代文本]

andy-yang


<a id="source-section-57"></a>

### [FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution](https://huggingface.co/papers/2608.16157)


FreeToken is an edge-native Mixture-of-Experts serving system that dynamically maps computation and model state onto heterogeneous local hardware to run large open-weight models on personal machines.


[[图片：UCBerkeley] University of California, Berkeley](https://huggingface.co/UCBerkeley) · Published on Aug 17, 2026


[Upvote

113](https://huggingface.co/login?next=%2Fpapers%2F2608.16157)

[GitHub 14.2k](https://github.com/FlashML-org/FreeToken) [arXiv Page](https://arxiv.org/abs/2608.16157)


[[图片：无替代文本]](https://huggingface.co/papers/2608.16157)

Submitted by

[图片：无替代文本]

andy-yang


<a id="source-section-58"></a>

### [FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution](https://huggingface.co/papers/2608.16157)


FreeToken is an edge-native Mixture-of-Experts serving system that dynamically maps computation and model state onto heterogeneous local hardware to run large open-weight models on personal machines.


[[图片：UCBerkeley] University of California, Berkeley](https://huggingface.co/UCBerkeley) · Aug 17, 2026


[Upvote

113](https://huggingface.co/login?next=%2Fpapers%2F2608.16157)


[GitHub 14.2k](https://github.com/FlashML-org/FreeToken) [arXiv Page](https://arxiv.org/abs/2608.16157)


[[图片：无替代文本]](https://huggingface.co/papers/2407.17789)

Submitted by

[图片：无替代文本]

akhaliq


<a id="source-section-59"></a>

### [Very Large-Scale Multi-Agent Simulation in AgentScope](https://huggingface.co/papers/2407.17789)


Enhancements to the AgentScope platform improve scalability, efficiency, and ease of use for large-scale multi-agent simulations through distributed mechanisms, flexible environments, and user-friendly tools.


-

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- 8 authors

· Published on Jul 25, 2024


[Upvote

47](https://huggingface.co/login?next=%2Fpapers%2F2407.17789)

[GitHub 32.8k](https://github.com/modelscope/agentscope) [arXiv Page](https://arxiv.org/abs/2407.17789)


[[图片：无替代文本]](https://huggingface.co/papers/2407.17789)

Submitted by

[图片：无替代文本]

akhaliq


<a id="source-section-60"></a>

### [Very Large-Scale Multi-Agent Simulation in AgentScope](https://huggingface.co/papers/2407.17789)


Enhancements to the AgentScope platform improve scalability, efficiency, and ease of use for large-scale multi-agent simulations through distributed mechanisms, flexible environments, and user-friendly tools.


-

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- 8 authors

· Jul 25, 2024


[Upvote

47](https://huggingface.co/login?next=%2Fpapers%2F2407.17789)


[GitHub 32.8k](https://github.com/modelscope/agentscope) [arXiv Page](https://arxiv.org/abs/2407.17789)


[[图片：无替代文本]](https://huggingface.co/papers/2508.16279)

Submitted by

[图片：无替代文本]

taesiri


<a id="source-section-61"></a>

### [AgentScope 1.0: A Developer-Centric Framework for Building Agentic
Applications](https://huggingface.co/papers/2508.16279)


AgentScope enhances agentic applications by providing flexible tool-based interactions, unified interfaces, and advanced infrastructure based on the ReAct paradigm, supporting efficient and safe development and deployment.


-

-

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- 23 authors

· Published on Aug 22, 2025


[Upvote

70](https://huggingface.co/login?next=%2Fpapers%2F2508.16279)

[GitHub 32.8k](https://github.com/agentscope-ai/agentscope) [arXiv Page](https://arxiv.org/abs/2508.16279)


[[图片：无替代文本]](https://huggingface.co/papers/2508.16279)

Submitted by

[图片：无替代文本]

taesiri


<a id="source-section-62"></a>

### [AgentScope 1.0: A Developer-Centric Framework for Building Agentic
Applications](https://huggingface.co/papers/2508.16279)


AgentScope enhances agentic applications by providing flexible tool-based interactions, unified interfaces, and advanced infrastructure based on the ReAct paradigm, supporting efficient and safe development and deployment.


-

-

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- 23 authors

· Aug 22, 2025


[Upvote

70](https://huggingface.co/login?next=%2Fpapers%2F2508.16279)


[GitHub 32.8k](https://github.com/agentscope-ai/agentscope) [arXiv Page](https://arxiv.org/abs/2508.16279)


[[图片：无替代文本]](https://huggingface.co/papers/2509.19296)

Submitted by

[图片：无替代文本]

taesiri


<a id="source-section-63"></a>

### [Lyra: Generative 3D Scene Reconstruction via Video Diffusion Model
Self-Distillation](https://huggingface.co/papers/2509.19296)


A self-distillation framework converts implicit 3D knowledge from video diffusion models into an explicit 3D Gaussian Splatting representation, enabling 3D scene generation from text or images.


- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- 13 authors

· Published on Sep 23, 2025


[Upvote

32](https://huggingface.co/login?next=%2Fpapers%2F2509.19296)

[GitHub 2.63k](https://github.com/nv-tlabs/lyra) [arXiv Page](https://arxiv.org/abs/2509.19296)


[[图片：无替代文本]](https://huggingface.co/papers/2509.19296)

Submitted by

[图片：无替代文本]

taesiri


<a id="source-section-64"></a>

### [Lyra: Generative 3D Scene Reconstruction via Video Diffusion Model
Self-Distillation](https://huggingface.co/papers/2509.19296)


A self-distillation framework converts implicit 3D knowledge from video diffusion models into an explicit 3D Gaussian Splatting representation, enabling 3D scene generation from text or images.


- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- 13 authors

· Sep 23, 2025


[Upvote

32](https://huggingface.co/login?next=%2Fpapers%2F2509.19296)


[GitHub 2.63k](https://github.com/nv-tlabs/lyra) [arXiv Page](https://arxiv.org/abs/2509.19296)


[[图片：无替代文本]](https://huggingface.co/papers/2610.05162)

Submitted by

[图片：无替代文本]

Qing145


<a id="source-section-65"></a>

### [Memadapter: Counterfactual Adaptation Against Memory-induced Sycophancy](https://huggingface.co/papers/2610.05162)


Long-term memory enables LLM-based agents to retain and reuse information across tasks and sessions, supporting personalization and long-horizon interactions. However, persistent memories can also induce sycophancy, causing agents to over-align with users' historical beliefs even when they are inaccurate, outdated, or inconsistent with objective evidence. Existing mitigation methods assume that memory-induced sycophancy originates from biased or incorrect memories and attempt to reduce this risk by filtering such memories at different stages of the memory pipeline. However, in the real world, objective and correct memories can still induce sycophancy, and the same memory can warrant different influence across different contexts. To this end, we propose MemAdapter, a novel framework that adaptively integrates retrieved memories to support objective and reliable reasoning. Specifically, MemAdapter consists of three components: (i) Counterfactual Induction, which leverages counterfactual reasoning to uncover the potential risk of retrieved memories; (ii) Context-Aware Reflection, which calibrates the inferential influence of each retrieved memory in light of the current task via self-reflection; and (iii) Evidence-Based Reasoning, which grounds the final response in appropriate evidence while preserving the legitimate influence of memory. Extensive experiments on three benchmarks demonstrate that MemAdapter consistently improves memory reliability across diverse scenarios. Our code is available at https://github.com/DEEP-JLU/MemAdapter.


-

-

-

-

-

- 7 authors

· Published on Oct 4, 2026


[Upvote

33](https://huggingface.co/login?next=%2Fpapers%2F2610.05162)

[GitHub 21](https://github.com/DEEP-JLU/MemAdapter) [arXiv Page](https://arxiv.org/abs/2610.05162)


[[图片：无替代文本]](https://huggingface.co/papers/2610.05162)

Submitted by

[图片：无替代文本]

Qing145


<a id="source-section-66"></a>

### [Memadapter: Counterfactual Adaptation Against Memory-induced Sycophancy](https://huggingface.co/papers/2610.05162)


Long-term memory enables LLM-based agents to retain and reuse information across tasks and sessions, supporting personalization and long-horizon interactions. However, persistent memories can also induce sycophancy, causing agents to over-align with users' historical beliefs even when they are inaccurate, outdated, or inconsistent with objective evidence. Existing mitigation methods assume that memory-induced sycophancy originates from biased or incorrect memories and attempt to reduce this risk by filtering such memories at different stages of the memory pipeline. However, in the real world, objective and correct memories can still induce sycophancy, and the same memory can warrant different influence across different contexts. To this end, we propose MemAdapter, a novel framework that adaptively integrates retrieved memories to support objective and reliable reasoning. Specifically, MemAdapter consists of three components: (i) Counterfactual Induction, which leverages counterfactual reasoning to uncover the potential risk of retrieved memories; (ii) Context-Aware Reflection, which calibrates the inferential influence of each retrieved memory in light of the current task via self-reflection; and (iii) Evidence-Based Reasoning, which grounds the final response in appropriate evidence while preserving the legitimate influence of memory. Extensive experiments on three benchmarks demonstrate that MemAdapter consistently improves memory reliability across diverse scenarios. Our code is available at https://github.com/DEEP-JLU/MemAdapter.


-

-

-

-

-

- 7 authors

· Oct 4, 2026


[Upvote

33](https://huggingface.co/login?next=%2Fpapers%2F2610.05162)


[GitHub 21](https://github.com/DEEP-JLU/MemAdapter) [arXiv Page](https://arxiv.org/abs/2610.05162)


[[图片：无替代文本]](https://huggingface.co/papers/2501.13956)


<a id="source-section-67"></a>

### [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://huggingface.co/papers/2501.13956)


Zep, a memory layer service, outperforms MemGPT in the DMR benchmark and LongMemEval by excelling in dynamic knowledge integration and temporal reasoning, critical for enterprise use cases.


-

-

-

- [图片：无替代文本]

- [图片：无替代文本]

- 5 authors

· Published on Jan 20, 2025


[Upvote

17](https://huggingface.co/login?next=%2Fpapers%2F2501.13956)

[GitHub 31.5k](https://github.com/getzep/graphiti) [arXiv Page](https://arxiv.org/abs/2501.13956)


[[图片：无替代文本]](https://huggingface.co/papers/2501.13956)


<a id="source-section-68"></a>

### [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://huggingface.co/papers/2501.13956)


Zep, a memory layer service, outperforms MemGPT in the DMR benchmark and LongMemEval by excelling in dynamic knowledge integration and temporal reasoning, critical for enterprise use cases.


-

-

-

- [图片：无替代文本]

- [图片：无替代文本]

- 5 authors

· Jan 20, 2025


[Upvote

17](https://huggingface.co/login?next=%2Fpapers%2F2501.13956)


[GitHub 31.5k](https://github.com/getzep/graphiti) [arXiv Page](https://arxiv.org/abs/2501.13956)


[[图片：无替代文本]](https://huggingface.co/papers/2609.38839)

Submitted by

[图片：无替代文本]

YINBO0927


<a id="source-section-69"></a>

### [FrameMorrow: Future-guided Frame Selection with Prospective Tokens for Long-Horizon Video Generation](https://huggingface.co/papers/2609.38839)


Long-horizon video generation requires models to effectively leverage an increasingly long generation history. As the generated history grows, retaining all previous content becomes increasingly expensive and redundant, making effective historical selection essential. Existing approaches often determine historical relevance based on the current content. However, information relevant to the present is not necessarily useful for future generation, while seemingly less relevant history may become important later. Our key insight is that historical information should be selected according to its relevance to future information needs. Capturing these needs does not require generating the full future; instead, a compact representation of what becomes important next is sufficient to guide historical selection. Building on this insight, we propose FrameMorrow, a prospective frame selector that predicts a small set of prospective tokens representing future information needs and uses them to identify relevant information from history. FrameMorrow selects explicit historical frames rather than model-specific internal states, enabling plug-and-play integration across diverse generators, including closed-source models, with little additional inference cost. We evaluate FrameMorrow across five benchmarks and 11 generative models spanning long-video generation, interactive generation, and action-conditioned world models. Extensive experiments demonstrate consistent improvements in long-range consistency, visual quality, and action alignment across diverse generation settings.


[[图片：NationalUniversityofSingapore] National University of Singapore](https://huggingface.co/NationalUniversityofSingapore) · Published on Sep 30, 2026


[Upvote

98](https://huggingface.co/login?next=%2Fpapers%2F2609.38839)

[GitHub 26](https://github.com/YinBo0927/FrameMorrow) [arXiv Page](https://arxiv.org/abs/2609.38839)


[[图片：无替代文本]](https://huggingface.co/papers/2609.38839)

Submitted by

[图片：无替代文本]

YINBO0927


<a id="source-section-70"></a>

### [FrameMorrow: Future-guided Frame Selection with Prospective Tokens for Long-Horizon Video Generation](https://huggingface.co/papers/2609.38839)


Long-horizon video generation requires models to effectively leverage an increasingly long generation history. As the generated history grows, retaining all previous content becomes increasingly expensive and redundant, making effective historical selection essential. Existing approaches often determine historical relevance based on the current content. However, information relevant to the present is not necessarily useful for future generation, while seemingly less relevant history may become important later. Our key insight is that historical information should be selected according to its relevance to future information needs. Capturing these needs does not require generating the full future; instead, a compact representation of what becomes important next is sufficient to guide historical selection. Building on this insight, we propose FrameMorrow, a prospective frame selector that predicts a small set of prospective tokens representing future information needs and uses them to identify relevant information from history. FrameMorrow selects explicit historical frames rather than model-specific internal states, enabling plug-and-play integration across diverse generators, including closed-source models, with little additional inference cost. We evaluate FrameMorrow across five benchmarks and 11 generative models spanning long-video generation, interactive generation, and action-conditioned world models. Extensive experiments demonstrate consistent improvements in long-range consistency, visual quality, and action alignment across diverse generation settings.


[[图片：NationalUniversityofSingapore] National University of Singapore](https://huggingface.co/NationalUniversityofSingapore) · Sep 30, 2026


[Upvote

98](https://huggingface.co/login?next=%2Fpapers%2F2609.38839)


[GitHub 26](https://github.com/YinBo0927/FrameMorrow) [arXiv Page](https://arxiv.org/abs/2609.38839)


[[图片：无替代文本]](https://huggingface.co/papers/2406.11927)

Submitted by

[图片：无替代文本]

bdqnghi


<a id="source-section-71"></a>

### [REPOEXEC: Evaluate Code Generation with a Repository-Level Executable
Benchmark](https://huggingface.co/papers/2406.11927)


RepoExec is a benchmark for evaluating repository-level code generation focusing on executability, functional correctness, and dependency integration.


- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- 3 authors

· Published on Jun 17, 2024


[Upvote

12](https://huggingface.co/login?next=%2Fpapers%2F2406.11927)

[GitHub 164](https://github.com/FSoft-AI4Code/RepoExec) [arXiv Page](https://arxiv.org/abs/2406.11927)


[[图片：无替代文本]](https://huggingface.co/papers/2406.11927)

Submitted by

[图片：无替代文本]

bdqnghi


<a id="source-section-72"></a>

### [REPOEXEC: Evaluate Code Generation with a Repository-Level Executable
Benchmark](https://huggingface.co/papers/2406.11927)


RepoExec is a benchmark for evaluating repository-level code generation focusing on executability, functional correctness, and dependency integration.


- [图片：无替代文本]

- [图片：无替代文本]

- [图片：无替代文本]

- 3 authors

· Jun 17, 2024


[Upvote

12](https://huggingface.co/login?next=%2Fpapers%2F2406.11927)


[GitHub 164](https://github.com/FSoft-AI4Code/RepoExec) [arXiv Page](https://arxiv.org/abs/2406.11927)


[[图片：无替代文本]](https://huggingface.co/papers/2606.23050)

Submitted by

[图片：无替代文本]

taesiri


<a id="source-section-73"></a>

### [Unlimited OCR Works](https://huggingface.co/papers/2606.23050)


Unlimited OCR introduces Reference Sliding Window Attention to eliminate growing memory consumption during long-sequence OCR tasks, enabling efficient transcription of multiple pages in a single forward pass.


[[图片：baidu] BAIDU](https://huggingface.co/baidu) · Published on Jun 22, 2026


[Upvote

91](https://huggingface.co/login?next=%2Fpapers%2F2606.23050)

[GitHub 26.7k](https://github.com/baidu/Unlimited-OCR) [arXiv Page](https://arxiv.org/abs/2606.23050)


[[图片：无替代文本]](https://huggingface.co/papers/2606.23050)

Submitted by

[图片：无替代文本]

taesiri


<a id="source-section-74"></a>

### [Unlimited OCR Works](https://huggingface.co/papers/2606.23050)


Unlimited OCR introduces Reference Sliding Window Attention to eliminate growing memory consumption during long-sequence OCR tasks, enabling efficient transcription of multiple pages in a single forward pass.


[[图片：baidu] BAIDU](https://huggingface.co/baidu) · Jun 22, 2026


[Upvote

91](https://huggingface.co/login?next=%2Fpapers%2F2606.23050)


[GitHub 26.7k](https://github.com/baidu/Unlimited-OCR) [arXiv Page](https://arxiv.org/abs/2606.23050)


[[图片：无替代文本]](https://huggingface.co/papers/2604.09557)

Submitted by

[图片：无替代文本]

talor-abr


<a id="source-section-75"></a>

### [SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding](https://huggingface.co/papers/2604.09557)


Speculative Decoding evaluation requires diverse workloads to accurately measure performance, which existing benchmarks lack, prompting the introduction of SPEED-Bench for standardized assessment across semantic domains and serving regimes.


[[图片：nvidia] NVIDIA](https://huggingface.co/nvidia) · Published on Feb 10, 2026


[Upvote

16](https://huggingface.co/login?next=%2Fpapers%2F2604.09557)

[GitHub 5.23k](https://github.com/NVIDIA/Model-Optimizer) [arXiv Page](https://arxiv.org/abs/2604.09557)


[[图片：无替代文本]](https://huggingface.co/papers/2604.09557)

Submitted by

[图片：无替代文本]

talor-abr


<a id="source-section-76"></a>

### [SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding](https://huggingface.co/papers/2604.09557)


Speculative Decoding evaluation requires diverse workloads to accurately measure performance, which existing benchmarks lack, prompting the introduction of SPEED-Bench for standardized assessment across semantic domains and serving regimes.


[[图片：nvidia] NVIDIA](https://huggingface.co/nvidia) · Feb 10, 2026


[Upvote

16](https://huggingface.co/login?next=%2Fpapers%2F2604.09557)


[GitHub 5.23k](https://github.com/NVIDIA/Model-Optimizer) [arXiv Page](https://arxiv.org/abs/2604.09557)


[[图片：无替代文本]](https://huggingface.co/papers/1910.03771)


<a id="source-section-77"></a>

### [HuggingFace's Transformers: State-of-the-art Natural Language Processing](https://huggingface.co/papers/1910.03771)


Transformers library provides state-of-the-art Transformer architectures and pretrained models for natural language processing tasks with a unified API and emphasis on extensibility and robust deployment.


[[图片：huggingface] Hugging Face](https://huggingface.co/huggingface) · Published on Oct 9, 2019


[Upvote

33](https://huggingface.co/login?next=%2Fpapers%2F1910.03771)

[GitHub 167k](https://github.com/huggingface/transformers) [arXiv Page](https://arxiv.org/abs/1910.03771)


[[图片：无替代文本]](https://huggingface.co/papers/1910.03771)


<a id="source-section-78"></a>

### [HuggingFace's Transformers: State-of-the-art Natural Language Processing](https://huggingface.co/papers/1910.03771)


Transformers library provides state-of-the-art Transformer architectures and pretrained models for natural language processing tasks with a unified API and emphasis on extensibility and robust deployment.


[[图片：huggingface] Hugging Face](https://huggingface.co/huggingface) · Oct 9, 2019


[Upvote

33](https://huggingface.co/login?next=%2Fpapers%2F1910.03771)


[GitHub 167k](https://github.com/huggingface/transformers) [arXiv Page](https://arxiv.org/abs/1910.03771)


[[图片：无替代文本]](https://huggingface.co/papers/2608.23552)

Submitted by

[图片：无替代文本]

milkkarten


<a id="source-section-79"></a>

### [Prime Agent: A Self-Improving RLM Harness](https://huggingface.co/papers/2608.23552)


Prime Agent is an open-source harness that uses recursive subagents, persistent computation, and agent-to-agent coordination to extend language models' long-horizon capabilities across coding and reasoning tasks.


[[图片：PrimeIntellect] Prime Intellect](https://huggingface.co/PrimeIntellect) · Published on Aug 24, 2026


[Upvote

57](https://huggingface.co/login?next=%2Fpapers%2F2608.23552)

[GitHub 21.6k](https://github.com/PrimeIntellect-ai/prime-agent) [arXiv Page](https://arxiv.org/abs/2608.23552)


[[图片：无替代文本]](https://huggingface.co/papers/2608.23552)

Submitted by

[图片：无替代文本]

milkkarten


<a id="source-section-80"></a>

### [Prime Agent: A Self-Improving RLM Harness](https://huggingface.co/papers/2608.23552)


Prime Agent is an open-source harness that uses recursive subagents, persistent computation, and agent-to-agent coordination to extend language models' long-horizon capabilities across coding and reasoning tasks.


[[图片：PrimeIntellect] Prime Intellect](https://huggingface.co/PrimeIntellect) · Aug 24, 2026


[Upvote

57](https://huggingface.co/login?next=%2Fpapers%2F2608.23552)


[GitHub 21.6k](https://github.com/PrimeIntellect-ai/prime-agent) [arXiv Page](https://arxiv.org/abs/2608.23552)


[[图片：无替代文本]](https://huggingface.co/papers/2605.31264)

Submitted by

[图片：无替代文本]

jasonrqh


<a id="source-section-81"></a>

### [COLLEAGUE.SKILL: Automated AI Skill Generation via Expert Knowledge Distillation](https://huggingface.co/papers/2605.31264)


Person-grounded AI skills are automatically distilled from heterogeneous traces into inspectable, correctable packages that capture both capabilities and behavioral patterns.


[[图片：ShanghaiAiLab] shanghai ailab](https://huggingface.co/ShanghaiAiLab) · Published on May 29, 2026


[Upvote

131](https://huggingface.co/login?next=%2Fpapers%2F2605.31264)

[GitHub 25.3k](https://github.com/titanwings/colleague-skill) [arXiv Page](https://arxiv.org/abs/2605.31264)


[[图片：无替代文本]](https://huggingface.co/papers/2605.31264)

Submitted by

[图片：无替代文本]

jasonrqh


<a id="source-section-82"></a>

### [COLLEAGUE.SKILL: Automated AI Skill Generation via Expert Knowledge Distillation](https://huggingface.co/papers/2605.31264)


Person-grounded AI skills are automatically distilled from heterogeneous traces into inspectable, correctable packages that capture both capabilities and behavioral patterns.


[[图片：ShanghaiAiLab] shanghai ailab](https://huggingface.co/ShanghaiAiLab) · May 29, 2026


[Upvote

131](https://huggingface.co/login?next=%2Fpapers%2F2605.31264)


[GitHub 25.3k](https://github.com/titanwings/colleague-skill) [arXiv Page](https://arxiv.org/abs/2605.31264)


[[图片：无替代文本]](https://huggingface.co/papers/2609.38426)

Submitted by

[图片：无替代文本]

Gtime666


<a id="source-section-83"></a>

### [LoopVL: Recurrent Visual Intelligence](https://huggingface.co/papers/2609.38426)


We introduce LoopVL to study whether Loop Transformers can be effectively extended to vision- language models. LoopVL combines Module-Loop and Model-Loop computation to iteratively update a unified vision-language state through shared modules. We train LoopVL from scratch through language pre-training, multimodal training, and post-training. LoopVL outperforms a range of similarly sized and larger non-recurrent models on multimodal understanding and visual reasoning benchmarks. We also observe Visual Aha Moments in LoopVL, characterized by pronounced shifts in visual attention across loops. LoopVL provides practical evidence for recurrent vision-language modeling and offers an intuitive perspective on how shared parameters can support deeper multimodal computation over continuously evolving visual-language states.


[[图片：RUC] Renmin University of China](https://huggingface.co/RUC) · Published on Sep 29, 2026


[Upvote

469](https://huggingface.co/login?next=%2Fpapers%2F2609.38426)

[GitHub 128](https://github.com/Tier-Flow/LoopVL) [arXiv Page](https://arxiv.org/abs/2609.38426)


[[图片：无替代文本]](https://huggingface.co/papers/2609.38426)

Submitted by

[图片：无替代文本]

Gtime666


<a id="source-section-84"></a>

### [LoopVL: Recurrent Visual Intelligence](https://huggingface.co/papers/2609.38426)


We introduce LoopVL to study whether Loop Transformers can be effectively extended to vision- language models. LoopVL combines Module-Loop and Model-Loop computation to iteratively update a unified vision-language state through shared modules. We train LoopVL from scratch through language pre-training, multimodal training, and post-training. LoopVL outperforms a range of similarly sized and larger non-recurrent models on multimodal understanding and visual reasoning benchmarks. We also observe Visual Aha Moments in LoopVL, characterized by pronounced shifts in visual attention across loops. LoopVL provides practical evidence for recurrent vision-language modeling and offers an intuitive perspective on how shared parameters can support deeper multimodal computation over continuously evolving visual-language states.


[[图片：RUC] Renmin University of China](https://huggingface.co/RUC) · Sep 29, 2026


[Upvote

469](https://huggingface.co/login?next=%2Fpapers%2F2609.38426)


[GitHub 128](https://github.com/Tier-Flow/LoopVL) [arXiv Page](https://arxiv.org/abs/2609.38426)


[[图片：无替代文本]](https://huggingface.co/papers/2609.12552)

Submitted by

[图片：无替代文本]

nielsr


<a id="source-section-85"></a>

### [RelateAnything: Real-Time Open-Vocabulary Relation Prediction From Any Inputs](https://huggingface.co/papers/2609.12552)


RelateAnything is a lightweight open-vocabulary relation prediction model that accepts arbitrary predicate vocabularies and region sources at inference, trained on a large geometrically verified dataset with positive-unlabeled supervision and evaluated on a new cross-dataset benchmark.


- [图片：无替代文本]

- 1 authors

· Published on Sep 11, 2026


[Upvote

8](https://huggingface.co/login?next=%2Fpapers%2F2609.12552)

[GitHub 920](https://github.com/Maelic/RelateAnything) [arXiv Page](https://arxiv.org/abs/2609.12552)


[[图片：无替代文本]](https://huggingface.co/papers/2609.12552)

Submitted by

[图片：无替代文本]

nielsr


<a id="source-section-86"></a>

### [RelateAnything: Real-Time Open-Vocabulary Relation Prediction From Any Inputs](https://huggingface.co/papers/2609.12552)


RelateAnything is a lightweight open-vocabulary relation prediction model that accepts arbitrary predicate vocabularies and region sources at inference, trained on a large geometrically verified dataset with positive-unlabeled supervision and evaluated on a new cross-dataset benchmark.


- [图片：无替代文本]

- 1 authors

· Sep 11, 2026


[Upvote

8](https://huggingface.co/login?next=%2Fpapers%2F2609.12552)


[GitHub 920](https://github.com/Maelic/RelateAnything) [arXiv Page](https://arxiv.org/abs/2609.12552)


[[图片：无替代文本]](https://huggingface.co/papers/2310.10688)


<a id="source-section-87"></a>

### [A decoder-only foundation model for time-series forecasting](https://huggingface.co/papers/2310.10688)


A large language model adapted for time-series forecasting achieves near-optimal zero-shot performance on diverse datasets across different time scales and granularities.


-

-

-

-

- 4 authors

· Published on Oct 14, 2023


[Upvote

47](https://huggingface.co/login?next=%2Fpapers%2F2310.10688)

[GitHub 34.1k](https://github.com/google-research/timesfm) [arXiv Page](https://arxiv.org/abs/2310.10688)


[[图片：无替代文本]](https://huggingface.co/papers/2310.10688)


<a id="source-section-88"></a>

### [A decoder-only foundation model for time-series forecasting](https://huggingface.co/papers/2310.10688)


A large language model adapted for time-series forecasting achieves near-optimal zero-shot performance on diverse datasets across different time scales and granularities.


-

-

-

-

- 4 authors

· Oct 14, 2023


[Upvote

47](https://huggingface.co/login?next=%2Fpapers%2F2310.10688)


[GitHub 34.1k](https://github.com/google-research/timesfm) [arXiv Page](https://arxiv.org/abs/2310.10688)


[[图片：无替代文本]](https://huggingface.co/papers/2610.02193)

Submitted by

[图片：无替代文本]

rhfeiyang


<a id="source-section-89"></a>

### [Hierarchical Continuous Diffusion Language Models](https://huggingface.co/papers/2610.02193)


Discrete diffusion language models offer a compelling alternative to autoregressive generation for tasks demanding bidirectional reasoning and global constraint satisfaction. Yet they share a structural bottleneck: when decoding in parallel, each token is sampled independently from its marginal, severing the statistical dependencies among the tokens decoded together. Continuous diffusion language models avoid this by denoising a shared continuous state, but their denoiser sees only that state, so nothing ties it to a valid token configuration until it is finally decoded. To address this, we propose Hierarchical Continuous Diffusion Language Models (HC-DLM), which couple discrete token generation with a continuous latent trajectory in a single, principled denoising process, whose training objective is derived from a variational bound on the token likelihood. In contrast to recent methods that attach continuous context to a self-contained discrete chain, HC-DLM makes the latent the only persistent generative state: tokens are read out from it at every step and feed back as a scaffold for the next latent update. On structured reasoning (Sudoku), mathematical planning (Countdown) and language modeling (LM1B), HC-DLM improves over discrete and continuous diffusion baselines at matched model size, in puzzle accuracy on Sudoku and Countdown and in generative perplexity on LM1B. Project page: https://hc-dlm.github.io/.


[[图片：UIUC-CS] University of Illinois at Urbana-Champaign](https://huggingface.co/UIUC-CS) · Published on Oct 1, 2026


[Upvote

89](https://huggingface.co/login?next=%2Fpapers%2F2610.02193)

[GitHub 57](https://github.com/rhfeiyang/HC-DLM) [arXiv Page](https://arxiv.org/abs/2610.02193)


[[图片：无替代文本]](https://huggingface.co/papers/2610.02193)

Submitted by

[图片：无替代文本]

rhfeiyang


<a id="source-section-90"></a>

### [Hierarchical Continuous Diffusion Language Models](https://huggingface.co/papers/2610.02193)


Discrete diffusion language models offer a compelling alternative to autoregressive generation for tasks demanding bidirectional reasoning and global constraint satisfaction. Yet they share a structural bottleneck: when decoding in parallel, each token is sampled independently from its marginal, severing the statistical dependencies among the tokens decoded together. Continuous diffusion language models avoid this by denoising a shared continuous state, but their denoiser sees only that state, so nothing ties it to a valid token configuration until it is finally decoded. To address this, we propose Hierarchical Continuous Diffusion Language Models (HC-DLM), which couple discrete token generation with a continuous latent trajectory in a single, principled denoising process, whose training objective is derived from a variational bound on the token likelihood. In contrast to recent methods that attach continuous context to a self-contained discrete chain, HC-DLM makes the latent the only persistent generative state: tokens are read out from it at every step and feed back as a scaffold for the next latent update. On structured reasoning (Sudoku), mathematical planning (Countdown) and language modeling (LM1B), HC-DLM improves over discrete and continuous diffusion baselines at matched model size, in puzzle accuracy on Sudoku and Countdown and in generative perplexity on LM1B. Project page: https://hc-dlm.github.io/.


[[图片：UIUC-CS] University of Illinois at Urbana-Champaign](https://huggingface.co/UIUC-CS) · Oct 1, 2026


[Upvote

89](https://huggingface.co/login?next=%2Fpapers%2F2610.02193)


[GitHub 57](https://github.com/rhfeiyang/HC-DLM) [arXiv Page](https://arxiv.org/abs/2610.02193)


Submitted by

[图片：无替代文本]

Sensen02


<a id="source-section-91"></a>

### [SoL-Pi: Recursively Scaling Auto-Research Loops for Efficient Agent Harness](https://huggingface.co/papers/2609.20519)


As coding agents move from supervised code completion to unattended, around-the-clock exploration, their work expands from isolated predictions into long trajectories of reasoning, tool use, and feedback. Token efficiency therefore becomes important for scaling recursive self-improvement. We take an RSI-inspired approach at the harness layer, scaling auto-research loops across increasingly numerous and diverse environments for harness rollouts. At this scale, the process yields reusable improvements that transfer beyond their development setting, moving automated harness discovery toward production-level outcomes. Four mechanisms survive selection and form SoL-Pi, spanning action execution, context compaction, observation handling, and delegated reading. On the 51-task EdgeBench evaluation, SoL-Pi achieves performance comparable to Pi across GPT-5.6 Sol and Opus 5 while reducing recorded token traffic by 44.7-49.0% and API cost by about one third. In other words, estimated hourly savings are \8.75-13.50 relative to native Codex and Claude Code harnesses, and \4.36-5.71 relative to Pi.


[[图片：nvidia] NVIDIA](https://huggingface.co/nvidia) · Published on Sep 17, 2026


[Upvote

139](https://huggingface.co/login?next=%2Fpapers%2F2609.20519)

[GitHub 3.35k](https://github.com/NVlabs/SoL-Pi) [arXiv Page](https://arxiv.org/abs/2609.20519)


Submitted by

[图片：无替代文本]

Sensen02


<a id="source-section-92"></a>

### [SoL-Pi: Recursively Scaling Auto-Research Loops for Efficient Agent Harness](https://huggingface.co/papers/2609.20519)


As coding agents move from supervised code completion to unattended, around-the-clock exploration, their work expands from isolated predictions into long trajectories of reasoning, tool use, and feedback. Token efficiency therefore becomes important for scaling recursive self-improvement. We take an RSI-inspired approach at the harness layer, scaling auto-research loops across increasingly numerous and diverse environments for harness rollouts. At this scale, the process yields reusable improvements that transfer beyond their development setting, moving automated harness discovery toward production-level outcomes. Four mechanisms survive selection and form SoL-Pi, spanning action execution, context compaction, observation handling, and delegated reading. On the 51-task EdgeBench evaluation, SoL-Pi achieves performance comparable to Pi across GPT-5.6 Sol and Opus 5 while reducing recorded token traffic by 44.7-49.0% and API cost by about one third. In other words, estimated hourly savings are \8.75-13.50 relative to native Codex and Claude Code harnesses, and \4.36-5.71 relative to Pi.


[[图片：nvidia] NVIDIA](https://huggingface.co/nvidia) · Sep 17, 2026


[Upvote

139](https://huggingface.co/login?next=%2Fpapers%2F2609.20519)


[GitHub 3.35k](https://github.com/NVlabs/SoL-Pi) [arXiv Page](https://arxiv.org/abs/2609.20519)


[[图片：无替代文本]](https://huggingface.co/papers/2609.08977)

Submitted by

[图片：无替代文本]

taesiri


<a id="source-section-93"></a>

### [Omni Interaction Agent Technical Report](https://huggingface.co/papers/2609.08977)


Gander is an end-to-end framework that integrates continuous multi-modal streaming, real-time full-duplex interaction, and agentic reasoning through a Cerebellum-Brain architecture and a chunk-level token stream design.


[[图片：Tencent-Hunyuan] Tencent Hunyuan](https://huggingface.co/Tencent-Hunyuan) · Published on Sep 8, 2026


[Upvote

96](https://huggingface.co/login?next=%2Fpapers%2F2609.08977)

[GitHub 457](https://github.com/Omni-Interaction-Gander/Omni-Interaction-Agent) [arXiv Page](https://arxiv.org/abs/2609.08977)


[[图片：无替代文本]](https://huggingface.co/papers/2609.08977)

Submitted by

[图片：无替代文本]

taesiri


<a id="source-section-94"></a>

### [Omni Interaction Agent Technical Report](https://huggingface.co/papers/2609.08977)


Gander is an end-to-end framework that integrates continuous multi-modal streaming, real-time full-duplex interaction, and agentic reasoning through a Cerebellum-Brain architecture and a chunk-level token stream design.


[[图片：Tencent-Hunyuan] Tencent Hunyuan](https://huggingface.co/Tencent-Hunyuan) · Sep 8, 2026


[Upvote

96](https://huggingface.co/login?next=%2Fpapers%2F2609.08977)


[GitHub 457](https://github.com/Omni-Interaction-Gander/Omni-Interaction-Agent) [arXiv Page](https://arxiv.org/abs/2609.08977)


[[图片：无替代文本]](https://huggingface.co/papers/2403.08299)


<a id="source-section-95"></a>

### [AutoDev: Automated AI-Driven Development](https://huggingface.co/papers/2403.08299)


AutoDev is an AI-driven software development framework that automates complex engineering tasks within a secure Docker environment, achieving high performance in code and test generation.


-

-

-

-

-

- 5 authors

· Published on Mar 13, 2024


[Upvote

21](https://huggingface.co/login?next=%2Fpapers%2F2403.08299)

[GitHub 25.3k](https://github.com/vxcontrol/pentagi) [arXiv Page](https://arxiv.org/abs/2403.08299)


[[图片：无替代文本]](https://huggingface.co/papers/2403.08299)


<a id="source-section-96"></a>

### [AutoDev: Automated AI-Driven Development](https://huggingface.co/papers/2403.08299)


AutoDev is an AI-driven software development framework that automates complex engineering tasks within a secure Docker environment, achieving high performance in code and test generation.


-

-

-

-

-

- 5 authors

· Mar 13, 2024


[Upvote

21](https://huggingface.co/login?next=%2Fpapers%2F2403.08299)


[GitHub 25.3k](https://github.com/vxcontrol/pentagi) [arXiv Page](https://arxiv.org/abs/2403.08299)


Submitted by

[图片：无替代文本]

WenyiWU0111


<a id="source-section-97"></a>

### [RSIGame: Autonomous Agentic Game Development with Recursive Self-improvement](https://huggingface.co/papers/2609.39045)


Recent advances in large language models have made automatic game generation increasingly feasible, yet reliably improving generated games beyond a playable version remains challenging. Naive iterative refinement can easily overfit a small set of test cases, producing fragile games with unresolved bugs, missing behaviors, and poor generalization to broader player interactions. We introduce RSIGame, an autonomous agentic game development framework with recursive self-improvement. RSIGame organizes development into complementary local and global loops. Concretely, a local explore-diagnose-improve loop broadly explores the executable game, diagnoses and prioritizes discovered issues, and performs evidence-grounded revision, where an evolving checklist continually accumulates new testing and improvement guidance. A global loop tracks overall quality, preserves the best checkpoint, and detects saturation or regression over long-horizon development. Beyond test-time improvement, RSIGame further internalizes successful development experience into the generator through training. Across 140 GameCraft-Bench tasks, two game engines, and five generators, RSIGame consistently improves game quality under matched development budgets. Notably, experience internalization enables Qwen3.8-27B to reach 61.38 on Godot and 58.53 on Phaser, exceeding GPT-5.5 one-shot scores while reducing Qwen's generation tokens by 11 times.


[[图片：RSIGame] RSIGame](https://huggingface.co/RSIGame) · Published on Sep 30, 2026


[Upvote

91](https://huggingface.co/login?next=%2Fpapers%2F2609.39045)

[GitHub 125](https://github.com/WenyiWU0111/RSIGame) [arXiv Page](https://arxiv.org/abs/2609.39045)


Submitted by

[图片：无替代文本]

WenyiWU0111


<a id="source-section-98"></a>

### [RSIGame: Autonomous Agentic Game Development with Recursive Self-improvement](https://huggingface.co/papers/2609.39045)


Recent advances in large language models have made automatic game generation increasingly feasible, yet reliably improving generated games beyond a playable version remains challenging. Naive iterative refinement can easily overfit a small set of test cases, producing fragile games with unresolved bugs, missing behaviors, and poor generalization to broader player interactions. We introduce RSIGame, an autonomous agentic game development framework with recursive self-improvement. RSIGame organizes development into complementary local and global loops. Concretely, a local explore-diagnose-improve loop broadly explores the executable game, diagnoses and prioritizes discovered issues, and performs evidence-grounded revision, where an evolving checklist continually accumulates new testing and improvement guidance. A global loop tracks overall quality, preserves the best checkpoint, and detects saturation or regression over long-horizon development. Beyond test-time improvement, RSIGame further internalizes successful development experience into the generator through training. Across 140 GameCraft-Bench tasks, two game engines, and five generators, RSIGame consistently improves game quality under matched development budgets. Notably, experience internalization enables Qwen3.8-27B to reach 61.38 on Godot and 58.53 on Phaser, exceeding GPT-5.5 one-shot scores while reducing Qwen's generation tokens by 11 times.


[[图片：RSIGame] RSIGame](https://huggingface.co/RSIGame) · Sep 30, 2026


[Upvote

91](https://huggingface.co/login?next=%2Fpapers%2F2609.39045)


[GitHub 125](https://github.com/WenyiWU0111/RSIGame) [arXiv Page](https://arxiv.org/abs/2609.39045)


[[图片：无替代文本]](https://huggingface.co/papers/2609.25001)

Submitted by

[图片：无替代文本]

taesiri


<a id="source-section-99"></a>

### [GameHorizon Suite: Multi-Horizon Data and Evaluation in Gameplay](https://huggingface.co/papers/2609.25001)


Modern video games provide a measurable testbed for AI models, combining abilities of visual understanding, instruction decomposition, goal planning, and precise action control over multiple temporal horizons. Existing datasets and benchmarks, however, either cover a narrow range of games, lack language instructions, or rely on high-variance online rollouts. To address these challenges, we introduce GameHorizon, a unified data and evaluation suite that measures gameplay capabilities at different horizons for diverse model families. GameHorizon Suite consists of three components. First, GameHorizon-Annotator is a scalable and automated annotation pipeline for multi-horizon instructions. Second, utilizing the pipeline, we construct GameHorizon-Data, the first large-scale AAA gameplay dataset with temporally aligned videos, player actions, and multi-horizon instructions. It comprises 5,000 hours of recordings from 21 games, collected by 100 human expert players. Third, we build GameHorizon-Bench with reproducible offline and stepwise online testing. The offline track enables reproducible evaluation using thousands of standardized questions organized into three primary tasks and a series of diagnostic variants, while the online track tests whether offline scores reflect actual gameplay capabilities and localizes failures to specific steps within long-horizon gameplay. Based on our GameHorizon Suite, we evaluate 47 models through more than one million model invocations, revealing a meaningful hierarchy of task difficulty and pronounced differences in model capabilities. Our work can provide a standardized yardstick for evaluating gameplay capabilities across horizons and model families. We will release our dataset, annotator, and benchmark to facilitate future research.


[[图片：tencent] Tencent](https://huggingface.co/tencent) · Published on Sep 21, 2026


[Upvote

132](https://huggingface.co/login?next=%2Fpapers%2F2609.25001)

[GitHub 561](https://github.com/TencentARC/GameHorizon) [arXiv Page](https://arxiv.org/abs/2609.25001)


[[图片：无替代文本]](https://huggingface.co/papers/2609.25001)

Submitted by

[图片：无替代文本]

taesiri


<a id="source-section-100"></a>

### [GameHorizon Suite: Multi-Horizon Data and Evaluation in Gameplay](https://huggingface.co/papers/2609.25001)


Modern video games provide a measurable testbed for AI models, combining abilities of visual understanding, instruction decomposition, goal planning, and precise action control over multiple temporal horizons. Existing datasets and benchmarks, however, either cover a narrow range of games, lack language instructions, or rely on high-variance online rollouts. To address these challenges, we introduce GameHorizon, a unified data and evaluation suite that measures gameplay capabilities at different horizons for diverse model families. GameHorizon Suite consists of three components. First, GameHorizon-Annotator is a scalable and automated annotation pipeline for multi-horizon instructions. Second, utilizing the pipeline, we construct GameHorizon-Data, the first large-scale AAA gameplay dataset with temporally aligned videos, player actions, and multi-horizon instructions. It comprises 5,000 hours of recordings from 21 games, collected by 100 human expert players. Third, we build GameHorizon-Bench with reproducible offline and stepwise online testing. The offline track enables reproducible evaluation using thousands of standardized questions organized into three primary tasks and a series of diagnostic variants, while the online track tests whether offline scores reflect actual gameplay capabilities and localizes failures to specific steps within long-horizon gameplay. Based on our GameHorizon Suite, we evaluate 47 models through more than one million model invocations, revealing a meaningful hierarchy of task difficulty and pronounced differences in model capabilities. Our work can provide a standardized yardstick for evaluating gameplay capabilities across horizons and model families. We will release our dataset, annotator, and benchmark to facilitate future research.


[[图片：tencent] Tencent](https://huggingface.co/tencent) · Sep 21, 2026


[Upvote

132](https://huggingface.co/login?next=%2Fpapers%2F2609.25001)


[GitHub 561](https://github.com/TencentARC/GameHorizon) [arXiv Page](https://arxiv.org/abs/2609.25001)


[[图片：无替代文本]](https://huggingface.co/papers/2606.03748)

Submitted by

[图片：无替代文本]

nielsr


<a id="source-section-101"></a>

### [Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models](https://huggingface.co/papers/2606.03748)


YOLO26 addresses real-time vision challenges through a unified model family with NMS-free inference, improved training strategies, and multi-task capabilities spanning detection, segmentation, and pose estimation.


[[图片：Ultralytics] Ultralytics](https://huggingface.co/Ultralytics) · Published on Jun 2, 2026


[Upvote

23](https://huggingface.co/login?next=%2Fpapers%2F2606.03748)

[GitHub 62.2k](https://github.com/ultralytics/ultralytics) [arXiv Page](https://arxiv.org/abs/2606.03748)


[[图片：无替代文本]](https://huggingface.co/papers/2606.03748)

Submitted by

[图片：无替代文本]

nielsr


<a id="source-section-102"></a>

### [Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models](https://huggingface.co/papers/2606.03748)


YOLO26 addresses real-time vision challenges through a unified model family with NMS-free inference, improved training strategies, and multi-task capabilities spanning detection, segmentation, and pose estimation.


[[图片：Ultralytics] Ultralytics](https://huggingface.co/Ultralytics) · Jun 2, 2026


[Upvote

23](https://huggingface.co/login?next=%2Fpapers%2F2606.03748)


[GitHub 62.2k](https://github.com/ultralytics/ultralytics) [arXiv Page](https://arxiv.org/abs/2606.03748)

# Google Developers - 2026 - DiffusionGemma The Developer Guide

- Source HTML: `raw/html/Google Developers - 2026 - DiffusionGemma The Developer Guide.html`
- Source URL: https://developers.googleblog.com/en/diffusiongemma-the-developer-guide/
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

Community/Events

Learn

Blog

YouTube

- Community/Events

- Learn

- Blog

- YouTube

# DiffusionGemma: The Developer Guide

JUNE 10, 2026

Omar Sanseviero
 
 Member of the Technical Staff

Ian Ballantyne
 
 Senior Developer Relations Engineer
 
 
 Google DeepMind

Share

- Facebook

- Twitter

- LinkedIn

- Mail

Following our announcement in our launch blog post, we are sharing this developer guide to help you understand, serve and customize this experimental model.

Built on the Gemma 4 backbone, DiffusionGemma introduces several milestones for developer workflows:

- Compute-bound parallel generation: Bypasses memory-bandwidth limitations by shifting the bottleneck to compute, delivering up to 4x faster token generation on GPUs (up to 700+ tokens per second on NVIDIA GeForce RTX 5090 and 1000+ tokens per second on a single NVIDIA H100).

- Bidirectional context & self-correction: Uses bidirectional attention to evaluate the entire text block simultaneously during generation, enabling real-time error correction and parallel context propagation.

- Developer-friendly sizes: Designed as a 26B Mixture of Experts (MoE) model that activates only 3.8B parameters during inference, allowing quantized deployment within 18 GB VRAM limits.

## The Architecture

For developers building with traditional LLMs on GPUs, the primary bottleneck is memory bandwidth. Autoregressive language models must repeatedly load model weights from memory to generate text one token at a time. DiffusionGemma bypasses this limitation by shifting the bottleneck from memory bandwidth to compute, generating and refining a 256-token canvas in parallel. By providing the GPU with a large parallel workload, it utilizes tensor cores that would otherwise sit idle during local serving.

- Uniform State Diffusion: Instead of predicting tokens sequentially, DiffusionGemma starts with a canvas of random placeholder tokens and iteratively refines them in parallel. Over multiple denoising passes, highly confident tokens help resolve adjacent positions, causing the entire sequence to snap into focus.

- Block Autoregressive Diffusion for Variable Length Generation: For sequences longer than 256 tokens, once a 256-token block is fully denoised, the model processes and commits it to the KV cache. The model then transitions to the next block, initializing a fresh 256-token canvas conditioned on the previously committed history. This combines parallel block speed with the sequential stability of autoregressive models.

## Showcase: Solving Sudoku with Parallel Denoising

Traditional autoregressive models struggle with strict, multivariable constrained problems like Sudoku. Because they generate text strictly from left to right, they cannot evaluate future placeholders or backtrack.

To demonstrate customization of DiffusionGemma, we are releasing a fine-tuning recipe and results using Hackable Diffusion, a modular JAX research toolbox. This training setup focuses on a classic multi-variable grid task: the Sudoku Solver.

### Why Sudoku is Interesting for Diffusion

In an 81-character Sudoku string representation (where empty cells are marked with periods), every digit is bound by strict intersecting horizontal, vertical, and 9x9 grid constraints.

Bidirectional Context Propagation: Unlike autoregressive models, DiffusionGemma’s denoising step allows every canvas query to attend to all positions in parallel. Information flows symmetrically across the board, resolving global dependencies in each step.

- Error Correction via Re-Noising: Under Uniform State Diffusion, the model evaluates the entire board simultaneously. If confidence drops, the sampler replaces digits with random ones, allowing for continuous self-correction.

- Efficient Early Stopping: Fine-tuning on Sudoku shows that adapters enhance early stopping. The SFT-tuned model stabilizes faster than the base model, allowing the engine to halt sooner, reducing latency and compute costs.

Sorry, your browser doesn't support playback for this video

Left: DiffusionGemma generating Sudoku output. The base model is unable to solve the Sudoku after 48 steps. Right: Fine-tuned (SFT) DiffusionGemma solves the puzzle after 12 steps. It is able to complete early thanks to adaptive stopping.

The Performance Impact: While the base DiffusionGemma model is not specifically trained to solve Sudoku puzzles (~0% success rate), applying the simple JAX SFT recipe on a Sudoku dataset raises correctness to 80% success, while decreasing the overall inference step count.

## Block Autoregressive Denoising

To enable block autoregressive denoising, DiffusionGemma alternates between incremental prefill and denoising during inference:

- Prefill / Incremental Prefill (Causal): Uses causal attention to ingest the prompt context and write to the KV cache. This runs once to prefill the initial context and then once per block to append each finalized 256-token canvas to the KV cache before proceeding to denoising the next canvas.

- Denoising (Bidirectional): Uses bidirectional attention to iteratively denoise the canvas. Query tokens at any position on the canvas can attend to all other canvas tokens (as well as KV cache), letting the model process context bidirectionally.

This architectural choice makes the following possible:

- Global Context Awareness: Unlike autoregressive (AR) models that only "look backward," the Denoiser's bidirectional attention allows every token on the canvas to attend to every other token. This makes the model much more effective at solving non-sequential problems, such as Sudoku, where a digit in the first cell must respect constraints in the last cell.

- Self-Correction: Because the model iteratively refines the whole canvas, it can "fix" earlier mistakes. If a token's confidence drops during a pass, the sampler can re-noise and replace it. This is a capability AR models lack since they are "stuck" with a token once it is generated, especially during long output sequences.

- Efficient Long-Context Scaling: The "block-autoregressive" approach allows the model to handle long sequences. It combines the parallel speed of diffusion for blocks with the proven sequential stability of AR models for long-form text.

- Simplified Deployment: Using the same architecture as the Gemma 4 26B A4B model means developers only need to implement a denoising step, making it easier to integrate into existing serving frameworks like vLLM.

Sorry, your browser doesn't support playback for this video

## Serving DiffusionGemma

To serve this experimental architecture efficiently, we worked with the vLLM team to implement DiffusionGemma into vLLM. This integration allows the engine to run the iterative parallel denoising loops efficiently across batched request streams.

Developers can deploy DiffusionGemma out of the box using vLLM's standard OpenAI-compatible local server.

```
vllm serve google/diffusiongemma-26B-A4B-it \
 --max-model-len 262144 \
 --max-num-seqs 4 \
 --gpu-memory-utilization 0.85 \
 --attention-backend TRITON_ATTN \
 --generation-config vllm \
 --hf-overrides '{"diffusion_sampler": "entropy_bound", "diffusion_entropy_bound": 0.1}' \
 --diffusion-config '{"canvas_length": 256}' \
 --enable-chunked-prefill
```

Shell

Copied

## Getting Started Today

Ready to explore the frontier of non-autoregressive text generation? Take a look at the following resources to find out more:

- Download the Weights: Access the weights of the experimental model (released under the Apache 2.0 license) directly on Hugging Face.

- Integrate & Learn: Review the Visual Guide to DiffusionGemma to understand the mechanics of text-based diffusion models. Read more about DiffusionGemma in the Gemma documentation.

- Use Your Favorite Inference Frameworks: Run the model efficiently using vLLM, Hugging Face Transformers, SGLang, and MLX.

- Adapt & Fine-Tune: For rapid experimentation, we are releasing the official training recipes using Hackable Diffusion You can also explore efficient fine-tuning using Unsloth or NVIDIA NeMo.

- Deploy Your Way: Instantly deploy on Google Cloud using Model Garden or via NVIDIA NIM. The model is optimized natively across the hardware stack from consumer RTX 4090 and 5090 cards to enterprise Hopper and Blackwell servers.

posted in:

- AI

- Announcements

- Explore

Previous

Next

Related Posts

AI
 
 
 Announcements
 
 Explore

Gemma 4 12B: The Developer Guide

JUNE 3, 2026

AI
 
 Cloud
 
 
 Announcements
 
 Learn

Introducing the Google Colab CLI

JUNE 5, 2026

- Connect

- Blog

- Bluesky

- Instagram

- LinkedIn

- X (Twitter)

- YouTube

- Programs

- Google Developer Program

- Google Developer Groups

- Google Developer Experts

- Accelerators

- Women Techmakers

- Google Cloud & NVIDIA

- Developer consoles

- Google API Console

- Google Cloud Platform Console

- Google Play Console

- Firebase Console

- Actions on Google Console

- Cast SDK Developer Console

- Chrome Web Store Dashboard

- Google Home Developer Console

- Android

- Chrome

- Firebase

- Google Cloud Platform

- All products

- Manage cookies

- Terms

- Privacy

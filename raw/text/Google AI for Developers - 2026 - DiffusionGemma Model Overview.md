# Google AI for Developers - 2026 - DiffusionGemma Model Overview

- Source HTML: `raw/html/Google AI for Developers - 2026 - DiffusionGemma Model Overview.html`
- Source URL: https://ai.google.dev/gemma/docs/diffusiongemma
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

- Gemini

- Imagen

- Veo

- Gemma

- Build with Gemini

- Customize Gemma open models

- Run on-device

- Build responsibly

/

- English

- Deutsch

- Español – América Latina

- Français

- Indonesia

- Italiano

- Polski

- Português – Brasil

- Shqip

- Tiếng Việt

- Türkçe

- Русский

- עברית

- العربيّة

- فارسی

- हिंदी

- বাংলা

- ภาษาไทย

- 中文 – 简体

- 中文 – 繁體

- 日本語

- 한국어

- Gemma

- Models

- More

- Gemma

- Docs

- Solutions

- More

- Code assistance

- More

- Community

- More

- Overview

- Get started

- Releases

- Overview

- Gemma 4 model card

- Gemma 3 model card

- Gemma 2 model card

- Gemma 1 model card

- Overview

- Model card

- Overview

- Model card

- Diffusion Explained

- Generate Output

- Overview

- Model card

- Formatting and best practices

- Function calling with Hugging Face Transformers

- Full function calling sequence with FunctionGemma

- Fine-tune FunctionGemma

- Overview

- Model card

- Generate embeddings with Sentence Transformers

- Fine-tune EmbeddingGemma

- Overview

- v2 model card

- v1 model card

- Generate output with Keras

- Fine-tune with JAX and Flax

- Prompt and system instructions

- Overview

- ShieldGemma 2 Model card

- ShieldGemma 1 Model card

- Overview

- Prompt Formatting

- Legacy Gemma setup [Gemma 1, 2, and 3]

- Legacy Prompt and system instructions [Gemma 1, 2, and 3]

- LM Studio

- Ollama

- LiteRT-LM

- Llama.cpp

- MediaPipe LLM Inference API

- MLX

- Gemma library

- Hugging Face Transformers

- Keras

- Unsloth

- Gemini API

- Cloud GKE

- Cloud Run

- Vertex AI

- vLLM

- Overview

- Hugging Face Transformers

- Basic and multi-turn chat

- Function calling

- Overview

- Image understanding

- Video understanding

- Audio data

- Thinking

- Overview

- Tune using Hugging Face Transformers and QLoRA

- Vision Tune using Hugging Face Transformers and QLoRA

- Full model fine-tune using Hugging Face Transformers

- Tune using Gemma library

- Convert Hugging Face Safetensors to MediaPipe Task

- Web

- Mobile

- Google Cloud

- LangChain

- Overview

- Inference using JAX and Flax

- Fine-tune using JAX and Flax

- Model card

- DataGemma

- Gemma Scope

- Gemma-APS

- Gemmaverse

- Discord

- Terms of use

- Gemma 4 license

- Prohibited use

- Intended use statement

- Gemini

- About

- Docs

- API reference

- Pricing

- Imagen

- About

- Docs

- Pricing

- Veo

- About

- Docs

- Pricing

- Gemma

- About

- Docs

- Gemmaverse

- Build with Gemini

- Gemini API

- Google AI Studio

- Customize Gemma open models

- Gemma open models

- Multi-framework with Keras

- Fine-tune in Colab

- Run on-device

- Google AI Edge

- Gemini Nano on Android

- Chrome built-in web APIs

- Build responsibly

- Responsible GenAI Toolkit

- Secure AI Framework

- Android Studio

- Chrome DevTools

- Colab

- Firebase

- Google Cloud

- JetBrains

- Jules

- VS Code

- Google AI Forum

- Gemini for Research

Gemma 4 released with text, audio and image input and long up to 256K context window! Learn more

- Home

# DiffusionGemma model overview

DiffusionGemma is an experimental open model that explores text diffusion, an
exceptionally fast approach to text generation. Based on the 26B (4B active)
Mixture-of-Experts (MoE) Gemma 4 architecture, DiffusionGemma generates tokens
using discrete diffusion. This open-weights model is multimodal, handling text,
image, and video inputs to generate text output.

Built on a MoE foundation, DiffusionGemma is designed to improve generation
speed (tokens per second) while remaining deployable across various hardware
environments. DiffusionGemma builds upon the architectural and capability
advancements of Gemma 4, introducing several core features:

- Discrete Text Diffusion: Shifts away from traditional causal token
generation to block-autoregressive multi-canvas sampling. The model
generates text by iteratively denoising blocks of tokens (a "canvas") in
parallel to dramatically boost decoding speeds.

- Multimodal Processing: Natively accepts text, images (with variable
aspect ratio and resolution support), and video inputs. (Note: Audio input
is not supported).

- Encoder-Decoder Architecture: Utilizes an autoregressive encoder to
process and cache prompt context, paired with denoising that applies
bi-directional attention over the generation canvas.

- Mixture-of-Experts (MoE) Efficiency: Leverages a sparse MoE design based
on the 26B (4B active) MoE variant, offering deep reasoning capabilities
with minimal overhead. When quantized, it fits within the 18GB VRAM limits
of consumer GPUs, ideal for local execution.

- Thinking Mode: Built-in configurable reasoning channels allow the model
to think step-by-step before emitting a final answer.

## Tradeoff with traditional models

While traditional language models are highly efficient for large-scale cloud
deployments because they can batch thousands of requests, running them locally
for a single user leaves hardware underutilized. DiffusionGemma solves this by
generating an entire 256-token block simultaneously rather than one token at a
time, maximizing local hardware performance.

However, this approach is strictly aimed at consumer-facing, low-concurrency
local use; because its parallel decoding offers diminishing returns under
high-QPS cloud workloads, the throughput advantage is strongest at low-to-medium
batch sizes on a single accelerator.

## Recommended Serving Configuration

For optimal latency and quality, we recommend deploying with the following
default parameters for the Diffusion Sampling Settings:

Parameter
Recommended Value
Function
Rationale

Maximum Number of Denoising Steps
48
Upper bound on number of denoising steps per canvas.
A safe limit on the number of denoising steps. Denoising will stop in fewer steps when adaptive stopping is enabled, typically 12-16 steps depending on the task.

Temperature Schedule
Linear 0.8 -> 0.4
Temperature scaling schedule that starts high and reduces as a function of denoising steps.
High temperature (0.8) encourages early exploration; low temperature (0.4) locks in final tokens.

Adaptive Early Stopping
Entropy threshold: 0.005
Halts execution early ifA) the average model entropy over the canvas is below the threshold, andB) if two consecutive denoiser predictions remain identical.
Simpler prompts and structured tasks like code require fewer denoising steps, enabling dynamic tokens-per-second speeds based on task complexity.

Token selection
Entropy bound: 0.1
At each step, the sampler selects the lowest-entropy tokens such that their mutual information bound stays below entropy bound. The sampler fully renoises the non-selected tokens.
Ensures only tokens that the model is relatively certain about are selected to refine the canvas, leaving other tokens to be refined in later denoising steps.

Get it on Hugging Face
Get it on Kaggle
Access it on Vertex

Access the experimental model weights (released under the Apache 2.0 license),
allowing you to deploy it in your own projects and applications.

Learn more about DiffusionGemma architecture
Try DiffusionGemma

Fine-tune DiffusionGemma
Deploy DiffusionGemma

Except as otherwise noted, the content of this page is licensed under the Creative Commons Attribution 4.0 License, and code samples are licensed under the Apache 2.0 License. For details, see the Google Developers Site Policies. Java is a registered trademark of Oracle and/or its affiliates.

Last updated 2026-06-10 UTC.

Need to tell us more?
 
 

 
 
 
 
 [[["Easy to understand","easyToUnderstand","thumb-up"],["Solved my problem","solvedMyProblem","thumb-up"],["Other","otherUp","thumb-up"]],[["Missing the information I need","missingTheInformationINeed","thumb-down"],["Too complicated / too many steps","tooComplicatedTooManySteps","thumb-down"],["Out of date","outOfDate","thumb-down"],["Samples / code issue","samplesCodeIssue","thumb-down"],["Other","otherDown","thumb-down"]],["Last updated 2026-06-10 UTC."],[],[]]

- Terms

- Privacy

- Manage cookies

- English

- Deutsch

- Español – América Latina

- Français

- Indonesia

- Italiano

- Polski

- Português – Brasil

- Shqip

- Tiếng Việt

- Türkçe

- Русский

- עברית

- العربيّة

- فارسی

- हिंदी

- বাংলা

- ภาษาไทย

- 中文 – 简体

- 中文 – 繁體

- 日本語

- 한국어

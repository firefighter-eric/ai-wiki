# NVIDIA - 2026 - Run DiffusionGemma on NVIDIA for Developer-Ready High-Throughput Text Generation

- Source HTML: `raw/html/NVIDIA - 2026 - Run DiffusionGemma on NVIDIA for Developer-Ready High-Throughput Text Generation.html`
- Source URL: https://developer.nvidia.com/blog/run-diffusiongemma-on-nvidia-for-developer-ready-high-throughput-text-generation/
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

DEVELOPER

- Home

- Blog

- Forums

- Docs

- Downloads

- Training

- Join

Technical Blog

Subscribe

Agentic AI / Generative AI

# Run DiffusionGemma on NVIDIA for Developer-Ready, High-Throughput Text Generation

Jun 10, 2026

By Anu Srivastava

Discuss (0)

- L

- T

- F

- R

- E

AI-Generated Summary

- DiffusionGemma, developed by Google DeepMind and optimized for NVIDIA hardware, generates tokens in parallel using diffusion-based denoising, enabling much faster and more scalable real-time AI applications compared to traditional token-by-token models.

- The model supports both text and image modalities, is built on the Gemma 4 26B A4B MoE architecture, and achieves high performance on various NVIDIA platforms, including H100 GPUs, DGX Spark, DGX Station, and RTX systems, with support for context lengths up to 256K tokens.

- Developers can access DiffusionGemma via Hugging Face and NVIDIAs GPU-accelerated endpoints, deploy it in production using NVIDIA NIM with OpenAI-compatible APIs, and fine-tune it for specific applications through the NVIDIA NeMo Framework.

AI-generated content may summarize information incompletely. Verify important information. Learn more

Developers building real-time AI—such as chat assistants, copilots, and agentic workflows—are often constrained by token-by-token generation speed. This limits responsiveness, increases serving costs, and makes fluid, interactive experiences difficult to achieve.

DiffusionGemma, created by Google DeepMind and optimized to run efficiently across NVIDIA platforms, introduces a new approach to text generation, producing tokens in parallel rather than one at a time, enabling faster, higher-throughput AI applications. The model uses diffusion-based denoising to generate 256 tokens in parallel per step, delivering up to 1,000 tokens/sec on a single NVIDIA H100 Tensor Core GPU, up to 150 tokens/sec on NVIDIA DGX Spark, and the fastest local performance on NVIDIA DGX Station.

For enterprise developers, this speed translates into lower serving costs, higher concurrency, and more responsive user experiences without sacrificing model quality. DiffusionGemma is built on the Gemma 4 26B A4B MoE architecture and optimized for low-latency, memory-bound inference.

Model name DiffusionGemma

Supported modalities Text, image

Total parameters 25.2B

Active parameters 3.8B

Context length Up to 256K tokens

Precision format BF16, NVFP4

In addition to NVIDIA data center GPUs, developers can enjoy optimal performance on a variety of client GPUs and systems.

PlatformBest ForKey highlightsGetting started

NVIDIA DGX SparkPersonal AI supercomputer for local AI development, autonomous agents, AI research, and prototypingNVIDIA GB10 Grace Blackwell Superchip, 128 GB unified memory, 1 PFLOP of FP4 AI compute, and a preinstalled NVIDIA AI software stack for fully local OpenClaw workflowsDGX Spark playbooks for vLLM and Unsloth; deployment guides; NVIDIA NeMo Automodel fine-tuning guide; vLLM on DGX Spark guide

NVIDIA DGX StationDeskside AI supercomputer for building, running, and scaling AI workloadsNVIDIA GB300 Grace Blackwell Ultra Superchip, NVIDIA AI software stack, 748 GB coherent memory, up to 20 PFLOPS of FP4 compute, and support for models up to 1T parameters. Frontier AI development, inference, and agents at your desk.DGX Station playbooks; vLLM on DGX Station guide

NVIDIA RTX + NVIDIA RTX PRODesktop AI apps, Windows development, and local inferenceOptimized local inference performance across desktop and workstation environments for creators and professionalsRTX blog; vLLM on RTX guide

## Build and prototype on NVIDIA

Access DiffusionGemma through Hugging Face Transformers for initial testing and prototyping on NVIDIA GeForce RTX 5090 or DGX Spark. For higher throughput or concurrent multi-user serving on DGX Spark, DGX Station, and RTX PRO, use vLLM by following our playbooks in Table 2.

With Day 0 support across NVIDIA hardware and software—from local prototyping to production deployment—developers can quickly move from experimentation to real-world applications. NVIDIA GPU-accelerated endpoints

Start building with DiffusionGemma with free access for prototyping to GPU-accelerated endpoints on build.nvidia.com as part of the NVIDIA Developer Program. The browser experience can also be connected to custom data sources.

BF16 and NVFP4

The model is available today on Hugging Face with BF16 checkpoints, and an NVFP4 quantized checkpoint for DiffusionGemma is also available using NVIDIA Model Optimizer.

## Enterprise deployments with NVIDIA NIM

NVIDIA NIM makes it simple to deploy DiffusionGemma from development into production. NIM packages the model as an optimized, containerized inference microservice — with performance tuning, standardized APIs, and the flexibility to run on-premises, in the cloud, or across hybrid environments. NIM exposes a standard OpenAI-compatible API for sending inference requests to the server.

- Download the container.

- Start the NIM server.

```
$ export NIM_IMAGE_PATH = “nvcr.io/nim/google/diffusiongemma-26b-a4b-it:latest”
$ docker run --gpus=all \ 
 -e NGC_API_KEY=$NGC_API_KEY \ 
 -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \ 
 -p 8000:8000 \ 
 ${NIM_IMAGE_PATH}
```

- Make a test request and read the full NIM documentation.

```
from openai import OpenAI 
client = OpenAI( 
 base_url="http://localhost:8000/v1", 
 api_key="not-required" 
) 
response = client.chat.completions.create( 
 model="google/diffusiongemma-26b-a4b-it”,
 messages=[ 
 {"role": "user", "content": "Write a poem about text diffusion"} 
 ], 
 max_tokens=256 
) 
print(response.choices[0].message.content)
```

## Adapt to specific use cases

Fine-tuning is available through the NVIDIA NeMo Framework for developers looking to adapt the model to specific tasks or domains.

NVIDIA is an active contributor to the open-source ecosystem and has released several hundred projects under open-source licenses. NVIDIA is committed to open models such as DiffusionGemma that promote AI transparency and enable users to share their work in AI safety and resilience.

Check out DiffusionGemma on Hugging Face or test for free using NVIDIA APIs at build.nvidia.com.

Discuss (0)

## Tags

Agentic AI / Generative AI | Developer Tools & Techniques | General | DGX | H100 | NIM | RTX GPU | Beginner Technical | News | DGX Spark | Text Generation

## About the Authors

About Anu Srivastava
 
 
 
 Anu Srivastava is a senior technical marketing manager who focuses on NVIDIA’s lighthouse AI model collaborations. She works with key partners and foundations to enable NVIDIA accelerated platform support for the open source developer ecosystem. Prior to NVIDIA, she worked at Google for over a decade in various engineering and management roles and holds a degree in computer science from the University of Texas at Austin.

View all posts by Anu Srivastava

## Comments

## Related posts

### NVIDIA Collaborates with Hugging Face to Simplify Generative AI Model Deployments

### Mistral Large and Mixtral 8x22B LLMs Now Powered by NVIDIA NIM and NVIDIA API

### Setting New Records at Data Center Scale Using NVIDIA H100 GPUs and NVIDIA Quantum-2 InfiniBand

### Bringing Generative AI to Life with NVIDIA Jetson

### New AI Technologies Introduced at GTC 2020 Keynote

## Related posts

### Model Quantization: Turn FP8 Checkpoints into High-Performance Inference Engines with NVIDIA TensorRT

### Accelerating Federated Learning Research with AI Agents and NVIDIA FLARE Auto-FL

### Evaluate Clinical ASR Models Faster with Agent Skills and NVIDIA Nemotron Speech

### Train Models Faster with JAX and MaxText Using NVFP4 on NVIDIA Blackwell

### Deploy Self-Evolving Agents for Faster, More Secure Research with a Hermes Agent and NVIDIA NemoClaw

- L

- T

- F

- R

- E

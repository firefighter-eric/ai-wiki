# NVIDIA - 2026 - Run DiffusionGemma on NVIDIA for Developer-Ready High-Throughput Text Generation

- Source HTML: `raw/html/NVIDIA - 2026 - Run DiffusionGemma on NVIDIA for Developer-Ready High-Throughput Text Generation.html`
- Source SHA256: `c5ca208797f900ed0cec247b2e1f5ea7f8f051b543353c29d8e672145a219d95`
- Source URL: https://developer.nvidia.com/blog/run-diffusiongemma-on-nvidia-for-developer-ready-high-throughput-text-generation/
- Generated from: `scripts/fetch_web_text.py`
- Extraction: `structured-html-v2` (headings, links, MathML/TeX and tables; figures require visual review)

## Extracted Text

<a id="source-section-0"></a>

[Agentic AI / Generative AI](https://developer.nvidia.com/blog/category/generative-ai/)


<a id="source-section-1"></a>

# Run DiffusionGemma on NVIDIA for Developer-Ready, High-Throughput Text Generation


[图片：Decorative image.]


Jun 10, 2026


By [Anu Srivastava](https://developer.nvidia.com/blog/author/anusrivastav/)


[Discuss (0)](https://developer.nvidia.com/blog/run-diffusiongemma-on-nvidia-for-developer-ready-high-throughput-text-generation/#entry-content-comments)


- [L](https://www.linkedin.com/sharing/share-offsite/?url=https%3A%2F%2Fdeveloper.nvidia.com%2Fblog%2Frun-diffusiongemma-on-nvidia-for-developer-ready-high-throughput-text-generation%2F)


- [T](https://twitter.com/intent/tweet?text=Run%C2%A0DiffusionGemma%C2%A0on%C2%A0NVIDIA%C2%A0for+Developer-Ready%2C+High-Throughput+Text+Generation+%7C+NVIDIA+Technical+Blog+https%3A%2F%2Fdeveloper.nvidia.com%2Fblog%2Frun-diffusiongemma-on-nvidia-for-developer-ready-high-throughput-text-generation%2F)


- [F](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fdeveloper.nvidia.com%2Fblog%2Frun-diffusiongemma-on-nvidia-for-developer-ready-high-throughput-text-generation%2F)


- [R](https://www.reddit.com/submit?url=https%3A%2F%2Fdeveloper.nvidia.com%2Fblog%2Frun-diffusiongemma-on-nvidia-for-developer-ready-high-throughput-text-generation%2F&title=Run%C2%A0DiffusionGemma%C2%A0on%C2%A0NVIDIA%C2%A0for+Developer-Ready%2C+High-Throughput+Text+Generation+%7C+NVIDIA+Technical+Blog)


- E


AI-Generated Summary


Like


Dislike


- DiffusionGemma, developed by Google DeepMind and optimized for NVIDIA hardware, generates tokens in parallel using diffusion-based denoising, enabling much faster and more scalable real-time AI applications compared to traditional token-by-token models.

- The model supports both text and image modalities, is built on the Gemma 4 26B A4B MoE architecture, and achieves high performance on various NVIDIA platforms, including H100 GPUs, DGX Spark, DGX Station, and RTX systems, with support for context lengths up to 256K tokens.

- Developers can access DiffusionGemma via Hugging Face and NVIDIAs GPU-accelerated endpoints, deploy it in production using NVIDIA NIM with OpenAI-compatible APIs, and fine-tune it for specific applications through the NVIDIA NeMo Framework.


AI-generated content may summarize information incompletely. Verify important information. [Learn more](https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/)


Developers building real-time AI—such as chat assistants, copilots, and agentic workflows—are often constrained by token-by-token generation speed. This limits responsiveness, increases serving costs, and makes fluid, interactive experiences difficult to achieve.


DiffusionGemma, created by Google DeepMind and optimized to run efficiently across NVIDIA platforms, introduces a new approach to text generation, producing tokens in parallel rather than one at a time, enabling faster, higher-throughput AI applications. The model uses diffusion-based denoising to generate 256 tokens in parallel per step, delivering up to 1,000 tokens/sec on a single NVIDIA H100 Tensor Core GPU, up to 150 tokens/sec on [NVIDIA DGX Spark,](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) and the fastest local performance on [NVIDIA DGX Station](https://www.nvidia.com/en-us/products/workstations/dgx-station/).


For enterprise developers, this speed translates into lower serving costs, higher concurrency, and more responsive user experiences without sacrificing model quality. DiffusionGemma is built on the Gemma 4 26B A4B MoE architecture and optimized for low-latency, memory-bound inference.


| 列 1 | 列 2 |
| --- | --- |
| **Model name** | **DiffusionGemma** |
| Supported modalities | Text, image |
| Total parameters | 25.2B |
| Active parameters | 3.8B |
| Context length | Up to 256K tokens |
| Precision format | BF16, NVFP4 |


Table 1. Overview of the DiffusionGemma, summarizing modalities, parameter sizes, and supported context length


In addition to NVIDIA data center GPUs, developers can enjoy optimal performance on a variety of client GPUs and systems.


| **Platform** | **Best For** | **Key highlights** | **Getting started** |
| --- | --- | --- | --- |
| **NVIDIA DGX Spark** | Personal AI supercomputer for local AI development, autonomous agents, AI research, and prototyping | NVIDIA GB10 Grace Blackwell Superchip, 128 GB unified memory, 1 PFLOP of FP4 AI compute, and a preinstalled NVIDIA AI software stack for fully local OpenClaw workflows | [DGX Spark playbooks](https://build.nvidia.com/spark) for vLLM and Unsloth; deployment guides; NVIDIA NeMo Automodel fine-tuning guide; [vLLM on DGX Spark guide](https://build.nvidia.com/spark/vllm) |
| **NVIDIA DGX Station** | Deskside AI supercomputer for building, running, and scaling AI workloads | NVIDIA GB300 Grace Blackwell Ultra Superchip, NVIDIA AI software stack, 748 GB coherent memory, up to 20 PFLOPS of FP4 compute, and support for models up to 1T parameters. Frontier AI development, inference, and agents at your desk. | [DGX Station playbooks](https://build.nvidia.com/station); [vLLM on DGX Station guide](https://build.nvidia.com/station/vllm) |
| **NVIDIA RTX + NVIDIA RTX PRO** | Desktop AI apps, Windows development, and local inference | Optimized local inference performance across desktop and workstation environments for creators and professionals | [RTX blog](https://blogs.nvidia.com/blog/rtx-ai-garage-local-gemma-diffusion); [vLLM on RTX guide](https://build.nvidia.com/rtx/vllm) |


Table 2. Comparison of local deployment options across NVIDIA platforms, highlighting primary use cases, key capabilities, and recommended getting‑started resources for DGX Spark, DGX Station, and RTX + RTX PRO systems


<a id="source-section-2"></a>

## **Build and prototype on NVIDIA**


Access DiffusionGemma through Hugging Face Transformers for initial testing and prototyping on NVIDIA GeForce RTX 5090 or DGX Spark. For higher throughput or concurrent multi-user serving on DGX Spark, DGX Station, and RTX PRO, use vLLM by following our playbooks in Table 2.


With Day 0 support across NVIDIA hardware and software—from local prototyping to production deployment—developers can quickly move from experimentation to real-world applications.

**NVIDIA GPU-accelerated endpoints**


Start building with DiffusionGemma with free access for prototyping to GPU-accelerated endpoints on [build.nvidia.com](https://build.nvidia.com/google/diffusiongemma-26b-a4b-it) as part of the [NVIDIA Developer Program](https://developer.nvidia.com/developer-program). The browser experience can also be connected to custom data sources.


**BF16 and NVFP4**


The model is available today on [Hugging Face](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4) with BF16 checkpoints, and an NVFP4 quantized checkpoint for [DiffusionGemma](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4) is also available using [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).


<a id="source-section-3"></a>

## Enterprise deployments with NVIDIA NIM


[NVIDIA NIM](https://docs.nvidia.com/nim/index.html) makes it simple to deploy DiffusionGemma from development into production. NIM packages the model as an optimized, containerized inference microservice — with performance tuning, standardized APIs, and the flexibility to run on-premises, in the cloud, or across hybrid environments. NIM exposes a standard OpenAI-compatible API for sending inference requests to the server.


- Download the [container](https://catalog.ngc.nvidia.com/orgs/nim/teams/google/containers/diffusiongemma-26b-a4b-it?version=latest).


- Start the NIM server.


```
$ export NIM_IMAGE_PATH = “nvcr.io/nim/google/diffusiongemma-26b-a4b-it:latest”
$ docker run --gpus=all \ 
  -e NGC_API_KEY=$NGC_API_KEY \ 
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \ 
  -p 8000:8000 \ 
 ${NIM_IMAGE_PATH}
```


- Make a test request and read the full [NIM documentation](https://docs.nvidia.com/nim/vision-language-models/1.7.0/examples/diffusiongemma-26b-a4b-it/api.html).


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


<a id="source-section-4"></a>

## Adapt to specific use cases


Fine-tuning is available through the  [NVIDIA NeMo Framework](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/model-coverage/dllm/google/diffusiongemma4.md) for developers looking to adapt the model to specific tasks or domains.


NVIDIA is an active contributor to the open-source ecosystem and has released several hundred [projects under open-source licenses](https://developer.nvidia.com/open-source). NVIDIA is committed to open models such as DiffusionGemma that promote AI transparency and enable users to share their work in AI safety and resilience.


Check out DiffusionGemma on [Hugging Face](https://huggingface.co/google/diffusiongemma-26B-A4B-it) or test for free using NVIDIA APIs at [build.nvidia.com](https://build.nvidia.com/google/diffusiongemma-26b-a4b-it).


[Discuss (0)](https://developer.nvidia.com/blog/run-diffusiongemma-on-nvidia-for-developer-ready-high-throughput-text-generation/#entry-content-comments)


<a id="source-section-5"></a>

## Tags


[Agentic AI / Generative AI](https://developer.nvidia.com/blog/category/generative-ai/) | [Developer Tools & Techniques](https://developer.nvidia.com/blog/category/development/) | [General](https://developer.nvidia.com/blog/recent-posts/?industry=General) | [DGX](https://developer.nvidia.com/blog/recent-posts/?products=DGX) | [H100](https://developer.nvidia.com/blog/recent-posts/?products=H100) | [NIM](https://developer.nvidia.com/blog/recent-posts/?products=NIM) | [RTX GPU](https://developer.nvidia.com/blog/recent-posts/?products=RTX+GPU) | [Beginner Technical](https://developer.nvidia.com/blog/recent-posts/?learning_levels=Beginner+Technical) | [News](https://developer.nvidia.com/blog/recent-posts/?content_types=News) | [DGX Spark](https://developer.nvidia.com/blog/tag/dgx-spark/) | [Text Generation](https://developer.nvidia.com/blog/tag/generative-ai-text/)


<a id="source-section-6"></a>

## About the Authors


[图片：Avatar photo]


**
About Anu Srivastava
**


Anu Srivastava is a senior technical marketing manager who focuses on NVIDIA’s lighthouse AI model collaborations. She works with key partners and foundations to enable NVIDIA accelerated platform support for the open source developer ecosystem. Prior to NVIDIA, she worked at Google for over a decade in various engineering and management roles and holds a degree in computer science from the University of Texas at Austin.


[View all posts by Anu Srivastava](https://developer.nvidia.com/blog/author/anusrivastav/)


<a id="source-section-7"></a>

## Comments

# HiCache System Design and Optimization

- Source HTML: `raw/html/SGLang Project - 2026 - HiCache System Design and Optimization.html`
- Source URL: https://docs.sglang.io/docs/advanced_features/hicache_design
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

## Documentation Index

Fetch the complete documentation index at: /llms.txt

Use this file to discover all available pages before exploring further.

Skip to main content

SGLang Documentation home page

Search...

Navigation

Hierarchical KV Caching (HiCache)

HiCache System Design and Optimization

### Basic Usage

Basic Usage

OpenAI-Compatible APIs

Anthropic-Compatible API

Ollama-Compatible API

Offline Engine API

SGLang Native APIs

Sampling Parameters

### Advanced Features

Advanced Features

Server Arguments

Session-Aware Radix Cache

Hyperparameter Tuning

Attention Backend

HiSparse: Hierarchical Sparse Attention

Speculative Decoding

Adaptive Speculative Decoding

Structured Outputs

Structured Outputs For Reasoning Models

Tool Parser

Reasoning Parser

Quantization

Quantized KV Cache

DP, DPA and SGLang DP Router

Expert Parallelism

LoRA Serving

PD Disaggregation

EPD Disaggregation

Pipeline Parallelism for Long Context

Hierarchical KV Caching (HiCache)

SGLang HiCache Best Practices

HiCache System Design and Optimization

Runtime Attach/Detach HiCache Storage Backend (No Restart)

Query VLM with Offline Engine

DP for Multi-Modal Encoder in SGLang

Cuda Graph for Multi-Modal Encoder in SGLang

Breakable CUDA Graph

Piecewise CUDA Graph

SGLang Model Gateway

llm-d

Deterministic Inference

Observability

Model Loading

Loading Models from Object Storage

Checkpoint Engine Integration

SGLang for RL Systems

### Supported Models

Supported models

Text Generation

Retrieval and Ranking

Specialized Models

Extending SGLang

### Developer Guide

Developer Guide

Contribution Guide

Development

Benchmarking

Evaluating New Models with SGLang

MSProbe Debugging Guide

### References

References

Troubleshooting and Frequently Asked Questions

Environment Variables

Production Metrics

Production Request Tracing

Deployment

Custom Chat Template

Frontend Language

Cookbook

Post-Training Integration

Nightly precision regression

## On this page

- Why and What is HiCache?

- System Design

- Overall Architecture

- HiRadixTree: Metadata Organization in HiCache

- Overall Workflow

- Local Match

- Prefetch from L3

- Data Write-back

- Multi-Rank Synchronization

- Data Transfer Optimization

- Integration with PD-Disaggregation Deployment Mode

- Unified Interfaces and Rich L3 Storage Backends

- Related Parameters

Hierarchical KV Caching (HiCache)

# HiCache System Design and Optimization

Copy pageCopy page

This document provides a comprehensive overview of SGLang HiCache, covering its system architecture, workflow and key components. It also details configuration parameters, optimization techniques, and integration with various L3 storage backends, serving as a complete reference for users and developers to understand and tune HiCache for efficient LLM inference.

​

- best_effort: Terminates immediately when GPU can execute prefill computation, with no waiting time, suitable for scenarios extremely sensitive to latency.

- wait_complete: Must wait for all prefetch operations to complete, suitable for scenarios requiring high cache hit rates.

- timeout: Terminates after specified time or when complete, balancing latency and cache hit rate needs.

- prefetch_timeout_base: the base timeout, representing overhead unrelated to the number of tokens (e.g., scheduling and synchronization). Default: 2 seconds.

- prefetch_timeout_per_ki_token: the incremental timeout per thousand tokens. Default: 0.1 seconds per 1024 tokens.

- prefetch_timeout_max: the upper bound applied to the linear timeout, preventing very long prompts from waiting unboundedly. Default: 30 seconds.

Example

```
timeout = min(
 prefetch_timeout_max,
 prefetch_timeout_base + prefetch_timeout_per_ki_token * num_token_to_fetch / 1024,
)
```

​

- write_through: Every access is immediately written back to the next level. When bandwidth is sufficient, this strategy provides the strongest caching benefit.

- write_through_selective: Data is written back only after the access frequency exceeds a threshold. This strategy backs up only hot data, reducing I/O overhead.

- write_back: Data is written back to the next level only when it is evicted from the upper level. This strategy alleviates storage pressure and is suitable for scenarios where storage capacity is limited but memory utilization must be maximized.

​

- Compute-Transfer Overlap: During the prefill phase, when transferring data from CPU to GPU, HiCache overlaps layers by concurrently loading the KV cache of layer N+1 while computing layer N. This effectively hides data transfer latency.

- GPU-assisted I/O Kernels: On top of cudaMemcpyAsync, HiCache implements a set of GPU-assisted I/O kernels specifically optimized for KV cache transfers between CPU and GPU. Compared to the baseline approach, these kernels achieve up to 3x higher transfer speed.

​

- Mooncake: Mooncake is a high-performance caching system for LLM inference that leverages RDMA and multi-NIC resources to enable zero-copy, ultra-fast data transfers. Try Mooncake here.

- DeepSeek 3FS (HF3FS): HF3FS is a Kubernetes-native distributed storage solution with operator-based deployment. Try HF3FS here.

- NIXL: NIXL provides a unified API for accessing various storage plugins, including but not limited to DeepSeek’s 3FS, GPU Direct Storage (GDS) and Amazon S3-compatible object storage. Try NIXL here.

- AIBrix KVCache: AIBrix KVCache is a production-ready KVCache Offloading Framework, which enables efficient memory tiering and low-overhead cross-engine reuse. Try AIBrix KVCache here.

- HiCacheFile: A simple file-based storage backend for demonstration purposes.

​

- --enable-hierarchical-cache: Enable hierarchical cache functionality. This is required to use HiCache.

- --hicache-ratio HICACHE_RATIO: The ratio of the size of host KV cache memory pool to the size of device pool. For example, a value of 2 means the host memory pool is twice as large as the device memory pool. The value of this parameter must be greater than 1, as the current implementation requires the host memory allocated for the KV cache to be larger than the device memory allocated for the KV cache.

- --hicache-size HICACHE_SIZE: The size of host KV cache memory pool in gigabytes. This parameter overrides hicache-ratio if set. For example, --hicache-size 30 allocates 30GB (1GB = 1e9 bytes) for the host memory pool for each rank. If there are 8 ranks, then the total memory size is 240GB. Just like hicache-ratio, the value of this parameter must be larger than the size of device memory allocated for KV cache.

- --page-size PAGE_SIZE: The number of tokens per page. This parameter determines the granularity of KV cache storage and retrieval. Larger page sizes reduce metadata overhead and improve I/O efficiency for storage backends, but may lower the cache hit rate when only part of a page matches the stored KV cache. For workloads with long common prefixes, larger pages can improve performance, while workloads with more diverse prefixes may benefit from smaller pages. See Data Transfer Optimization for how page granularity affects I/O performance.

- --hicache-storage-prefetch-policy {best_effort,wait_complete,timeout}: Controls when prefetching from storage should stop. See Prefetch from L3 for details.

- best_effort: Prefetch as much as possible without blocking

- wait_complete: Wait for prefetch to complete before proceeding

- timeout: Terminates after specified time or when complete (Recommended for production environments, as setting an appropriate timeout helps the system meet required SLOs)

- --hicache-write-policy {write_back,write_through,write_through_selective}: Controls how data is written from faster to slower memory tiers. See Data Write-back for details.

- write_through: Immediately writes data to all tiers (strongest caching benefits)

- write_through_selective: Uses hit-count tracking to back up only frequently accessed data

- write_back: Writes data back to slower tiers only when eviction is needed (reduces I/O load)

- --hicache-io-backend {direct,kernel}: Choose the I/O backend for KV cache transfer between CPU and GPU. See Data Transfer Optimization for details.

- direct: Standard CUDA memory copy operations

- kernel: GPU-assisted I/O kernels (recommended for better performance)

- --hicache-mem-layout {layer_first,page_first,page_first_direct}: Memory layout for the host memory pool. See Data Transfer Optimization for details.

- layer_first: Compatible with GPU computation kernels (default for GPU memory)

- page_first: Optimized for I/O efficiency

- page_first_direct: Groups all tokens of a given layer within a page, allowing transfers from L2 to GPU to be aggregated at the page-layer level

- --hicache-storage-backend {file,mooncake,hf3fs,nixl,aibrix,dynamic}: Choose the storage backend for the L3 tier. Built-in backends: file, mooncake, hf3fs, nixl, aibrix. For dynamic backend, use —hicache-storage-backend-extra-config to specify: backend_name (custom name), module_path (Python module path), class_name (backend class name). See Unified Interfaces and Rich L3 Storage Backends for available backends.

- --enable-lmcache: Using LMCache as an alternative hierarchical cache solution.

- --lmcache-config-file: Path to the LMCache YAML configuration file.

- --hicache-storage-backend-extra-config HICACHE_STORAGE_BACKEND_EXTRA_CONFIG: the extra config can be either

- a JSON string containing extra configuration for the storage backend, e.g., --hicache-storage-backend-extra-config '{"prefetch_threshold":512, "prefetch_timeout_base": 0.5, "prefetch_timeout_per_ki_token": 0.25}' , or

- a TOML or JSON or YAML file specifying the extra configuration for the storage backend (to differentiate from the JSON string input, prepend a @ in front of the file name), e.g., --hicache-storage-backend-extra-config "@config.toml" where config.toml is the config file containing the complex configurations. This can be useful when the configuration consists of many or complex key-value pairs (for instance, it is preferred to use a config file for NIXL backend as its configurations can be complex).

Was this page helpful?

YesNo

SGLang HiCache Best Practices

Previous

Runtime Attach/Detach HiCache Storage Backend (No Restart)

Next

⌘I

githubxlinkedinslackdiscord

Powered byThis documentation is built and hosted on Mintlify, a developer documentation platform

Assistant

Responses are generated using AI and may contain mistakes.

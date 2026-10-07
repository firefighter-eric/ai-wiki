# vLLM Project - 2026 - vLLM V1 Guide

- Source HTML: `raw/html/vLLM Project - 2026 - vLLM V1 Guide.html`
- Source SHA256: `171338224217d7c87f932b4a37c089a76a1bc85b78d9cd78ddb62e4e44842a35`
- Source URL: https://docs.vllm.ai/en/stable/usage/v1_guide/
- Generated from: `scripts/fetch_web_text.py`
- Extraction: `structured-html-v2` (headings, links, MathML/TeX and tables; figures require visual review)

## Extracted Text

<a id="source-section-0"></a>

<a id="source-section-1"></a>

# vLLM V1[¶](https://docs.vllm.ai/en/stable/usage/v1_guide/#vllm-v1)


Announcement


We have fully deprecated V0. Please read [RFC #18571](https://github.com/vllm-project/vllm/issues/18571) for more details.


If you have a use case that works on V0 Engine but not V1, please share it on [GitHub](https://github.com/vllm-project/vllm) or in the [vLLM Slack](https://inviter.co/vllm-slack).


vLLM V0 successfully supported a wide range of models and hardware, but as new features were developed independently, the system grew increasingly complex. This complexity made it harder to integrate new capabilities and introduced technical debt, revealing the need for a more streamlined and unified design.


Building on V0’s success, vLLM V1 retains the stable and proven components from V0 (such as the models, GPU kernels, and utilities). At the same time, it significantly re-architects the core systems, covering the scheduler, KV cache manager, worker, sampler, and API server, to provide a cohesive, maintainable framework that better accommodates continued growth and innovation.


Specifically, V1 aims to:


- Provide a **simple, modular, and easy-to-hack codebase**.

- Ensure **high performance** with near-zero CPU overhead.

- **Combine key optimizations** into a unified architecture.

- Require **zero configs** by enabling features/optimizations by default.


We see significant performance improvements from upgrading to V1 core engine, in particular for long context scenarios. Please see performance benchmark (To be added).


For more details, check out the vLLM V1 blog post [vLLM V1: A Major Upgrade to vLLM’s Core Architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html) (published Jan 27, 2025).


This living user guide outlines a few known **important changes and limitations** introduced by vLLM V1. The team has been working actively to bring V1 as the default engine, therefore this guide will be updated constantly as more features get supported on vLLM V1.


<a id="source-section-2"></a>

## Differences from V0[¶](https://docs.vllm.ai/en/stable/usage/v1_guide/#differences-from-v0)


This section lists some differences in behavior between V0 and V1.


<a id="source-section-3"></a>

### Chunked Prefill[¶](https://docs.vllm.ai/en/stable/usage/v1_guide/#chunked-prefill)


Chunked prefill is enabled by default whenever possible, unlike in V0 where it was conditionally enabled based on model characteristics.


<a id="source-section-4"></a>

### CUDA Graphs[¶](https://docs.vllm.ai/en/stable/usage/v1_guide/#cuda-graphs)


CUDA graph capture takes up more memory in V1 than in V0.


<a id="source-section-5"></a>

### Semantic Changes to Logprobs[¶](https://docs.vllm.ai/en/stable/usage/v1_guide/#semantic-changes-to-logprobs)


<a id="source-section-6"></a>

#### Logprobs Calculation[¶](https://docs.vllm.ai/en/stable/usage/v1_guide/#logprobs-calculation)


By default, logprobs in V1 are now returned immediately once computed from the model’s raw output (i.e. before applying any logits post-processing such as temperature scaling or penalty adjustments). As a result, the returned logprobs do not reflect the final adjusted probabilities used during sampling.


You can adjust this behavior by setting the `--logprobs-mode` flag. Four modes are supported: `raw_logprobs` (default), `processed_logprobs`, `raw_logits`, `processed_logits`. Raw means the values before applying any logit processors, like bad words. Processed means the values after applying all processors, including temperature and top_k/top_p.


<a id="source-section-7"></a>

#### Prompt Logprobs with Prefix Caching[¶](https://docs.vllm.ai/en/stable/usage/v1_guide/#prompt-logprobs-with-prefix-caching)


While V1 supports passing prompt logprobs with prefix caching enabled, it no longer caches the logprobs. For a request requiring prompt logprobs, the engine will ignore the prefix cache and recompute the prefill of full prompt to generate the logprobs.


<a id="source-section-8"></a>

## Feature Support[¶](https://docs.vllm.ai/en/stable/usage/v1_guide/#feature-support)


For each item, its support in vLLM V1 falls into one of the following states:


- **🟢 Functional**: Fully operational with optimizations comparable to or better than V0.

- **🟡 In Progress**: Planned to be in vLLM V1, with open PRs/RFCs.

- **🔴 Removed**: Dropped from vLLM V1. Will only consider re-introducing if there is strong demand.


Note


vLLM V1’s unified scheduler treats both prompt and output tokens the same way by using a simple dictionary (e.g., `{request_id: num_tokens}`) to dynamically allocate a fixed token budget per request, enabling features like chunked prefills, prefix caching, and speculative decoding without a strict separation between prefill and decode phases.


The V1 scheduler supports multiple scheduling policies, including First-Come, First-Served (FCFS) and priority-based scheduling (where requests are processed based on assigned priority, with FCFS as a tie-breaker), configurable via the `--scheduling-policy` argument.


<a id="source-section-9"></a>

### Hardware[¶](https://docs.vllm.ai/en/stable/usage/v1_guide/#hardware)


| Hardware | Status |
| --- | --- |
| **NVIDIA** | 🟢 |
| **AMD** | 🟢 |
| **INTEL GPU** | 🟢 |
| **TPU** | 🟢 |
| **CPU** | 🟢 |


Note


More hardware platforms may be supported via plugins, e.g.:


- [vllm-ascend](https://github.com/vllm-project/vllm-ascend)

- [vllm-spyre](https://github.com/vllm-project/vllm-spyre)

- [vllm-gaudi](https://github.com/vllm-project/vllm-gaudi)

- [vllm-openvino](https://github.com/vllm-project/vllm-openvino)


Please check their corresponding repositories for more details.


<a id="source-section-10"></a>

### Models[¶](https://docs.vllm.ai/en/stable/usage/v1_guide/#models)


| Model Type | Status |
| --- | --- |
| **Decoder-only Models** | 🟢 |
| **Encoder-Decoder Models** | 🟢 (Whisper), 🔴 (Others) |
| **Pooling Models** | 🟢 |
| **Mamba Models** | 🟢 |
| **Multimodal Models** | 🟢 |


See below for the status of models that are not yet supported or have more features planned in V1.


<a id="source-section-11"></a>

#### Pooling Models[¶](https://docs.vllm.ai/en/stable/usage/v1_guide/#pooling-models)


Now fully supported, with prefix caching and chunked prefill newly available for last-pooling models.


We are working on enabling prefix caching and chunked prefill for more categories of pooling models.


<a id="source-section-12"></a>

#### Mamba Models[¶](https://docs.vllm.ai/en/stable/usage/v1_guide/#mamba-models)


Models using selective state-space mechanisms instead of standard transformer attention are supported. Models that use Mamba-2 and Mamba-1 layers (e.g., `Mamba2ForCausalLM`, `MambaForCausalLM`, `FalconMambaForCausalLM`) are supported.


Hybrid models that combine Mamba-2 and Mamba-1 layers with standard attention layers are also supported (e.g., [`Zamba2ForCausalLM`](https://docs.vllm.ai/en/stable/api/vllm/model_executor/models/zamba2/#vllm.model_executor.models.zamba2.Zamba2ForCausalLM), `NemotronHForCausalLM`, `FalconH1ForCausalLM` and `GraniteMoeHybridForCausalLM`, `JambaForCausalLM`, `Plamo2ForCausalLM`).


Hybrid models with mechanisms different to Mamba are also supported (e.g, `Lfm2ForCausalLM`).


Please note that prefix caching is not yet supported for any of the above models.


<a id="source-section-13"></a>

#### Encoder-Decoder Models[¶](https://docs.vllm.ai/en/stable/usage/v1_guide/#encoder-decoder-models)


Whisper is supported natively. Other encoder-decoder models are supported via the plugin system:


- **BART**: `BartForConditionalGeneration` is supported via the official [bart-plugin](https://github.com/vllm-project/bart-plugin).

- **Florence-2**: `Florence2ForConditionalGeneration` is supported via the official [bart-plugin](https://github.com/vllm-project/bart-plugin).


For other encoder-decoder models (e.g., `MllamaForConditionalGeneration`), we recommend following a similar pattern by implementing support through the [plugin system](https://docs.vllm.ai/en/stable/design/plugin_system/).


<a id="source-section-14"></a>

### Features[¶](https://docs.vllm.ai/en/stable/usage/v1_guide/#features)


| Feature | Status |
| --- | --- |
| **Prefix Caching** | 🟢 Functional |
| **Chunked Prefill** | 🟢 Functional |
| **LoRA** | 🟢 Functional |
| **Logprobs Calculation** | 🟢 Functional |
| **FP8 KV Cache** | 🟢 Functional |
| **Spec Decode** | 🟢 Functional |
| **Prompt Logprobs with Prefix Caching** | 🟢 Functional |
| **Structured Output Alternative Backends** | 🟢 Functional |
| **Concurrent Partial Prefills** | 🟡 [In Progress](https://github.com/vllm-project/vllm/issues/14003) |
| **best_of** | 🔴 [Removed](https://github.com/vllm-project/vllm/issues/13361) |
| **Per-Request Logits Processors** | 🔴 [Removed](https://github.com/vllm-project/vllm/pull/13360) |
| **GPU <> CPU KV Cache Swapping** | 🔴 Removed |
| **Request-level Structured Output Backend** | 🔴 Removed |


Note


vLLM V1’s unified scheduler treats both prompt and output tokens the same way by using a simple dictionary (e.g., `{request_id: num_tokens}`) to dynamically allocate a fixed token budget per request, enabling features like chunked prefills, prefix caching, and speculative decoding without a strict separation between prefill and decode phases.


<a id="source-section-15"></a>

#### Removed Features[¶](https://docs.vllm.ai/en/stable/usage/v1_guide/#removed-features)


As part of the major architectural rework in vLLM V1, several legacy features have been removed.


<a id="source-section-16"></a>

##### Sampling features[¶](https://docs.vllm.ai/en/stable/usage/v1_guide/#sampling-features)


- **best_of**: This feature has been removed due to limited usage. See details at [RFC #13361](https://github.com/vllm-project/vllm/issues/13361).

- **Per-Request Logits Processors**: In V0, users could pass custom processing functions to adjust logits on a per-request basis. In vLLM V1, this feature has been removed. Instead, we now support **global logits processors** which are set at startup time, see [RFC #17799](https://github.com/vllm-project/vllm/issues/17799).


<a id="source-section-17"></a>

##### KV Cache features[¶](https://docs.vllm.ai/en/stable/usage/v1_guide/#kv-cache-features)


- **GPU <> CPU KV Cache Swapping**: with the new simplified core architecture, vLLM V1 no longer requires KV cache swapping to handle request preemptions.


<a id="source-section-18"></a>

##### Structured Output features[¶](https://docs.vllm.ai/en/stable/usage/v1_guide/#structured-output-features)


- **Request-level Structured Output Backend**: Removed; alternative backends (outlines, guidance) with fallbacks are supported now.

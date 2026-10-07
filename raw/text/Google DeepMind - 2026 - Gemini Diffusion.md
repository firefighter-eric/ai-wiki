# Google DeepMind - 2026 - Gemini Diffusion

- Source HTML: `raw/html/Google DeepMind - 2026 - Gemini Diffusion.html`
- Source SHA256: `c0904b3f7cc76c59a69c8a0515fd4773c29be833b74182e6c4345ca122169a6e`
- Source URL: https://deepmind.google/models/gemini-diffusion/
- Generated from: `scripts/fetch_web_text.py`
- Extraction: `structured-html-v2` (headings, links, MathML/TeX and tables; figures require visual review)

## Extracted Text

<a id="source-section-0"></a>

Gemini Diffusion

<a id="source-section-1"></a>

# Our state-of-the-art, experimental text diffusion model


Large-language models are the foundation of generative AI today. Weâre using a technique called diffusion to explore a new kind of language model that gives users greater control, creativity, and speed in text generation.


- [Overview](https://deepmind.google/models/gemini-diffusion/#overview)

- [Capabilities](https://deepmind.google/models/gemini-diffusion/#capabilities)

- [Performance](https://deepmind.google/models/gemini-diffusion/#performance)


<a id="source-section-2"></a>

### What is a diffusion model?


Traditional autoregressive language models generate text one word â or token â at a time. This sequential process can be slow, and limit the quality and coherence of the output.


Diffusion models work differently. Instead of predicting text directly, they learn to generate outputs by refining noise, step-by-step. This means they can iterate on a solution very quickly and error correct during the generation process. This helps them excel at tasks like editing, including in the context of math and code.


Your browser does not support the video tag.


<a id="source-section-3"></a>

### Capabilities


<a id="source-section-4"></a>

### Rapid response


Generates content significantly faster than even our fastest model so far.


<a id="source-section-5"></a>

### More coherent text


Generates entire blocks of tokens at once, meaning it responds more coherently to a userâs prompt than autoregressive models.


<a id="source-section-6"></a>

### Iterative refinement


Corrects errors during generation for more consistent outputs.


<a id="source-section-7"></a>

### Performance


<a id="source-section-8"></a>

#### Gemini Diffusionâs external benchmark performance is comparable to much larger models, whilst also being faster.


| Benchmark | Gemini Diffusion | Gemini 2.0 Flash-Lite |
| --- | --- | --- |
| Code LiveCodeBench (v6) | 30.9% | 28.5% |
| Code BigCodeBench | 45.4% | 45.8% |
| Code LBPP (v2) | 56.8% | 56.0% |
| Code SWE-Bench Verified* | 22.9% | 28.5% |
| Code HumanEval | 89.6% | 90.2% |
| Code MBPP | 76.0% | 75.8% |
| Science GPQA Diamond | 40.4% | 56.5% |
| Mathematics AIME 2025 | 23.3% | 20.0% |
| Reasoning BIG-Bench Extra Hard | 15.0% | 21.0% |
| Multilingual Global MMLU (Lite) | 69.1% | 79.0% |


Methodology


All scores are pass @1 (no majority voting). The Gemini 2.0 Flash-Lite experiments are run with the AI Studio API for the model-id gemini-2.0-flash-lite with the default sampling settings.


* Non-agentic evaluation (single turn edit only), max prompt length of 32K.


<a id="source-section-9"></a>

### Gemini Diffusion speed


Sampling speed excluding overhead


1479 tokens / sec


Overhead


0.84 sec


Average sampling speed across reported evals.


Try Gemini Diffusion

<a id="source-section-10"></a>

## Gemini Diffusion is currently available as an experimental demo to help develop and refine future models.

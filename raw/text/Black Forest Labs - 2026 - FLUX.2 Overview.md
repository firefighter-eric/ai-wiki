# Black Forest Labs - 2026 - FLUX.2 Overview

- Source HTML: `raw/html/Black Forest Labs - 2026 - FLUX.2 Overview.html`
- Source SHA256: `1d4e8f82dc37a3fea72180be79e60a1d85899da2ebb54e0fd46398d20dba702f`
- Source URL: https://docs.bfl.ai/flux_2/flux2_overview
- Generated from: `scripts/fetch_web_text.py`
- Extraction: `structured-html-v2` (headings, links, MathML/TeX and tables; figures require visual review)

## Extracted Text

<a id="source-section-0"></a>

[Skip to main content](https://docs.bfl.ai/flux_2/flux2_overview#content-area)

🚀 FLUX.2 [klein] — Sub-second generation. Open weights, Apache 2.0, API from $0.014/image. [Learn more →](https://docs.bfl.ai/flux_2/flux2_overview#flux-2-%5Bklein%5D-models)


[Black Forest Labs home page [图片：light logo] [图片：dark logo]](https://docs.bfl.ai/)


[Documentation](https://docs.bfl.ai/quick_start/introduction)[Prompting Guide](https://docs.bfl.ai/guides/prompting_guide_flux2)[API Reference](https://docs.bfl.ai/api-reference/get-the-users-credits)[Release Notes](https://docs.bfl.ai/release-notes)


- [Documentation](https://docs.bfl.ai/quick_start/introduction)

- [Prompting Guide](https://docs.bfl.ai/guides/prompting_summary)

- [BFL Homepage](https://bfl.ai)

- [Help Center](https://help.bfl.ai)


FLUX.2


<a id="source-section-1"></a>

# Overview


FLUX.2 model family overview — from sub-second generation to highest quality, with multi-reference editing, color control, and up to 4MP output.


[图片：Black Forest]
**FLUX.2** spans the full spectrum of image generation—from **sub-second inference** with [klein] to **highest quality** with [max]. Generate photorealistic images with precise control over colors, poses, and composition, or edit existing images by referencing up to 10 sources simultaneously.
Choose **[klein]** for real-time, high-volume generation, **[pro]** for production at scale, **[flex]** for fine-grained control, or **[max]** for maximum quality and grounding search.


**Want to try first?** Test FLUX.2 [max], [pro], and [flex] in our [playground](https://playground.bfl.ai). [klein] is available via our [API](https://docs.bfl.ai/flux_2/flux2_text_to_image) and on [Hugging Face](https://huggingface.co/black-forest-labs).


<a id="source-section-2"></a>

## [​](https://docs.bfl.ai/flux_2/flux2_overview#what-can-you-do)

What Can You Do?


- Multi-Reference

- Photorealism & Detail

- Grounding Search

- Typography & Text

- Exact Color Control

- Structured Prompting


Combine elements from multiple images while maintaining identity across complex scenes. Create ad variants with consistent faces, product mockups in any context, or fashion editorials where models stay consistent.

[图片：无替代文本]


[图片：无替代文本]


[图片：无替代文本]


Generate photorealistic images with enhanced detail, texture, and lighting. FLUX.2 produces images that merge seamlessly with real photography—ideal for e-commerce and product marketing.

[图片：无替代文本]


[图片：无替代文本]


[图片：无替代文本]


[图片：无替代文本]


Generate images grounded in real-time information with FLUX.2 [max]. It searches the web when needed, so you can create visuals of yesterday’s football game, the weather in real-time of any cities, or re-create historical events.

[图片：无替代文本]


[图片：无替代文本]


[图片：无替代文本]


[图片：无替代文本]


Reliable text rendering for infographics, UI mockups, and marketing materials.

[图片：无替代文本]


[图片：无替代文本]


[图片：无替代文本]


[图片：无替代文本]


Specify brand colors via hex codes with precision matching. No approximation—get the exact colors you need.**Example**: Gradient colors with hex codes**Prompt**: `A vase on a table in living room, the color of the vase is a gradient of color, starting with color #02eb3c and finishing with color #edfa3c. The flowers inside the vase have the color #ff0088`

[图片：无替代文本]

**Example**: Multiple hex colors for product design**Prompt**: `Luxury eyeshadow palette with 6 pans: top row #B76E79, #E8D5B7, #8B4789; bottom row #CD7F32, #F8F6F0, #800020`

[图片：无替代文本]


Use structured prompts for precise control over generation. Perfect for production workflows and automation.

Example: Structured Prompting


```
{
  "subject": "Mona Lisa painting by Leonardo da Vinci",
  "background": "museum gallery wall, ornate gold frame",
  "lighting": "soft gallery lighting, warm spotlights",
  "style": "digital art, high contrast",
  "camera_angle": "eye level view",
  "composition": "centered, portrait orientation"
}
```


[图片：无替代文本]


[图片：无替代文本]


<a id="source-section-3"></a>

## [​](https://docs.bfl.ai/flux_2/flux2_overview#which-model-to-choose)

Which Model to Choose?


| | **[klein]** | **[max]** | **[pro]** | **[flex]** | **[dev]** |
| --- | --- | --- | --- | --- | --- |
| **Best for** | Real-time, high-volume | Highest quality, final assets | Production at scale | Quality with control | Local development |
| **Multi-reference** | Up to 4 | Up to 8 (API), 10 (playground) | Up to 8 (API), 10 (playground) | Up to 8 (API), 10 (playground) | Recommended max 6 |
| **Controls** | Standard | Standard | Standard | Adjustable steps & guidance | Full customization |
| **Grounding search** | No | Yes | No | No | No |
| **Pricing** | from $0.014 / image | from $0.07 / MP | from $0.03 / MP | $0.06 / MP | Free (non-commercial) |


**FLUX.2 [klein]** delivers sub-second inference with open weights. 4B runs on consumer GPUs (~13GB VRAM). Apache 2.0 for 4B, FLUX NCL for 9B. See [model details below](https://docs.bfl.ai/flux_2/flux2_overview#flux2-klein-models).


**FLUX.2 [max]** includes **grounding search**: when prompted, it performs web searches to access real-time information to visualize trending products, current events, or the latest styles without manually sourcing reference material.


<a id="source-section-4"></a>

## [​](https://docs.bfl.ai/flux_2/flux2_overview#compare-flux-2-models)

Compare FLUX.2 Models


<a id="source-section-5"></a>

### [​](https://docs.bfl.ai/flux_2/flux2_overview#at-a-glance)

At a Glance


<a id="source-section-6"></a>

## [klein]


**Sub-second inference.** Our fastest models with open weights. Runs on consumer GPUs (~13GB VRAM). From $0.014/image via API, or run locally with Apache 2.0 (4B) / FLUX NCL (9B).


<a id="source-section-7"></a>

## [max]


**Maximum performance.** Highest editing consistency across tasks. Vast world knowledge. Strongest prompt following and faithful style representation.


<a id="source-section-8"></a>

## [pro]


**Top performance at affordable price.** The high quality, production-grade image editing and generation model.


<a id="source-section-9"></a>

## [flex]


**Specialized for typography.** Best for text rendering and preserving small details.


<a id="source-section-10"></a>

### [​](https://docs.bfl.ai/flux_2/flux2_overview#use-cases)

Use Cases


| **Use Case** | **FLUX.2 [klein]** | **FLUX.2 [max]** | **FLUX.2 [pro]** | **FLUX.2 [flex]** |
| --- | --- | --- | --- | --- |
| **Product Marketing** | Bulk catalog generation, A/B testing variants | Highest quality hero shots for marketplaces | Create ads at scale for social campaigns | Text overlay while preserving details |
| **Movie Making** | Rapid storyboarding, concept exploration | Top quality cinematic pre-visualization | Rapid ideation and static movie banners | Intros, credits, static advertising |
| **Creative Platforms** | Cost-efficient generation for all tiers | Premium model for highest-tier subs | High quality backbone at scale | Specialized text placement |
| **E-commerce** | High-volume product variations, thumbnails | Premium product photography | Production-grade catalog images | Price tags, labels, descriptions |
| **Editorial & Fashion** | Rapid mood boards, style exploration | Final hero images | Campaign imagery at scale | Text-heavy layouts |


<a id="source-section-11"></a>

## [​](https://docs.bfl.ai/flux_2/flux2_overview#flux-2-klein-models)

FLUX.2 [klein] Models


**FLUX.2 [klein]** is our fastest model family, delivering state-of-the-art quality with **sub-second inference**. Unifying generation and editing in a single compact architecture, [klein] is built for applications requiring real-time image generation—and runs on consumer hardware with as little as **13GB VRAM**.


[图片：FLUX.2 [klein] editing examples]


[图片：FLUX.2 [klein] photorealistic examples]


[图片：FLUX.2 [klein] diverse output examples]


**Open weights available**: [klein] 4B is fully open under **Apache 2.0**. [klein] 9B is available under the **FLUX Non-Commercial License**. Download from [Hugging Face](https://huggingface.co/black-forest-labs).


<a id="source-section-12"></a>

### [​](https://docs.bfl.ai/flux_2/flux2_overview#api-models)

API Models


| | **[klein] 4B** | **[klein] 9B** |
| --- | --- | --- |
| **Best for** | High volume, local deployment | Balanced quality and speed |
| **Architecture** | 4B flow model | 9B flow model + 8B Qwen3 text embedder |
| **Inference steps** | 4 (step-distilled) | 4 (step-distilled) |
| **VRAM** | ~13GB | ~24GB |
| **Speed** | Sub-second | Sub-second |
| **API Pricing** | $0.014 +$0.001/MP | $0.015 +$0.002/MP |
| **License** | Apache 2.0 | FLUX Non-Commercial License |


<a id="source-section-13"></a>

### [​](https://docs.bfl.ai/flux_2/flux2_overview#open-weights-community)

Open Weights (Community)


The **Base** variants are undistilled foundation models with full training signal—ideal for fine-tuning, LoRA training, research, and custom pipelines. Higher output diversity than distilled models.


| | **[klein] Base 4B** | **[klein] Base 9B** |
| --- | --- | --- |
| **Best for** | Fine-tuning, research, custom pipelines | Maximum quality, research |
| **Output diversity** | High | Highest |
| **Step-distilled** | No (full capacity) | No (full capacity) |
| **License** | Apache 2.0 | FLUX Non-Commercial License |
| **Availability** | [Hugging Face](https://huggingface.co/black-forest-labs) | [Hugging Face](https://huggingface.co/black-forest-labs) |


Base models are available as open weights for local development and research. They are not offered on the public API.


FLUX.2 [klein] does not include prompt upsampling. Write detailed, descriptive prompts for best results. See our [prompting guide](https://docs.bfl.ai/guides/prompting_guide_flux2_klein) for techniques.


<a id="source-section-14"></a>

## [​](https://docs.bfl.ai/flux_2/flux2_overview#preview-endpoints)

Preview Endpoints


Preview endpoints are where our latest improvements land first. They reflect our most recent advances in quality and speed.


| Endpoint | Description |
| --- | --- |
| `flux-2-pro-preview` | Our latest FLUX.2 [pro] model. |
| `flux-2-pro` | A fixed snapshot of FLUX.2 [pro]. This endpoint will not change, making it suitable for workflows that require reproducibility. |
| `flux-2-klein-9b-preview` | Our latest FLUX.2 [klein] 9B model with KV caching for improved performance. |
| `flux-2-klein-9b` | A fixed snapshot of FLUX.2 [klein] 9B. Choose this when you need reproducibility. |


**Which endpoint should I use?** For most use cases, the preview endpoints (`flux-2-pro-preview`, `flux-2-klein-9b-preview`) give you the best results. Choose the non-preview endpoints when you need a pinned model — for example, if your workflow depends on consistent outputs across runs or you have compliance requirements around model stability.


The `flux-2-pro` and `flux-2-klein-9b` endpoints are unchanged. If you are already using them, no action is required.


Both preview and non-preview endpoints share the same API contract — the request and response format is identical. Only the underlying model weights differ.


<a id="source-section-15"></a>

## [​](https://docs.bfl.ai/flux_2/flux2_overview#technical-specifications)

Technical Specifications


<a id="source-section-16"></a>

## Resolution


- **Output**: Up to 4MP


- **Input**: 64x64 minimum


- Any aspect ratio


<a id="source-section-17"></a>

## Multi-Reference


- Up to 10 input images ([klein]: 4)


- Character consistency


- Style transfer


<a id="source-section-18"></a>

## Advanced Controls


- Pose guidance


- Hex color matching


- Structured prompting


- Grounding search ([max] only)


<a id="source-section-19"></a>

## [​](https://docs.bfl.ai/flux_2/flux2_overview#getting-started)

Getting Started


<a id="source-section-20"></a>

## Try in Playground


Test FLUX.2 [max], [pro], and [flex] in your browser. No setup required.


<a id="source-section-21"></a>

## Download [klein] Weights


Get [klein] weights from Hugging Face for local inference.


<a id="source-section-22"></a>

## Text-to-Image API


Generate images from text prompts.


<a id="source-section-23"></a>

## Image Editing API


Edit images with multi-reference support.


<a id="source-section-24"></a>

## [klein] Prompting Guide


Master narrative prompting for best [klein] results.


<a id="source-section-25"></a>

## Local Development


Download [dev] weights for local inference.


Was this page helpful?


[Credits & Billing](https://docs.bfl.ai/account_management/credits_billing)[FLUX.2 Image Editing](https://docs.bfl.ai/flux_2/flux2_image_editing)


⌘I

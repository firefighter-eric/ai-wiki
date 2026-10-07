# Gabeur et al. - 2026 - Image Generators are Generalist Vision Learners

- Source HTML: `raw/html/Gabeur et al. - 2026 - Image Generators are Generalist Vision Learners.html`
- Source SHA256: `28dc99c8ff19054a08c40a2c8278d675ab7c2568aef8bf08e3fe39dc26cb996a`
- Source URL: https://ar5iv.labs.arxiv.org/html/2604.20329
- Generated from: `scripts/fetch_web_text.py`
- Extraction: `structured-html-v2` (headings, links, MathML/TeX and tables; figures require visual review)

## Extracted Text

<a id="source-section-0"></a>

\uselogo\reportnumber


0001\correspondingauthorvision-banana@google.com


<a id="source-section-1"></a>

# Image Generators are Generalist Vision Learners


Valentin Gabeur


Shangbang Long


Songyou Peng


Paul Voigtlaender


Shuyang Sun


Yanan Bao


Karen Truong


Zhicheng Wang


Wenlei Zhou


Jonathan T. Barron


Kyle Genova


Nithish Kannen


Sherry Ben


Yandong Li


Mandy Guo


Suhas Yogin


Yiming Gu


Huizhong Chen


Oliver Wang


Saining Xie


Howard Zhou


Kaiming He


Thomas Funkhouser


Jean-Baptiste Alayrac


Radu Soricut


<a id="source-section-2"></a>

###### Abstract


Recent works show that image and video generators exhibit zero-shot visual understanding behaviors, in a way reminiscent of how Large Language Models (LLMs) develop emergent capabilities of language understanding and reasoning from generative pretraining.
While it has long been conjectured that the ability to create visual content implies an ability to understand it, there has been limited evidence that generative vision models have developed strong understanding capabilities.
In this work, we demonstrate that image generation training serves a role similar to LLM pretraining, and lets models learn powerful and general visual representations that enable state-of-the-art performance on various vision tasks.
We introduce Vision Banana, a generalist model built by instruction-tuning Nano Banana Pro (NBP) on a mixture of its original training data alongside a small amount of vision task data.
By parameterizing the output space of vision tasks as RGB images, we seamlessly reframe perception as image generation.
Our generalist model, Vision Banana, achieves state-of-the-art results on a variety of vision tasks involving both 2D and 3D understanding, beating or rivaling zero-shot domain-specialists, including Segment Anything Model 3 on segmentation tasks, and the Depth Anything series on metric depth estimation.
We show that these results can be achieved with lightweight instruction-tuning without sacrificing the base model’s image generation capabilities.
The superior results suggest that image generation pretraining is a generalist vision learner.
It also shows that image generation serves as a unified and universal interface for vision tasks, similar to text generation’s role in language understanding and reasoning.
We could be witnessing a major paradigm shift for computer vision, where generative vision pretraining takes a central role in building Foundational Vision Models for both generation and understanding.
Project Page: [vision-banana.github.io](https://ar5iv.labs.arxiv.org/html/vision-banana.github.io)


<a id="source-section-3"></a>

## 1 Introduction


In recent years, advanced image and video generation models (Google, [2025a](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib27), [b](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib28); Black Forest Labs, [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib5); ByteDance, [2026](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib8); Luma, [2026](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib47); OpenAI, [2026](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib49)) have demonstrated unprecedented generation capabilities, synthesizing highly complex, high-fidelity visual context with precise semantic control. This remarkable capability for visual creation suggests that these models possess a deep, internalized comprehension of the visual world’s underlying structures, semantics, and relationships.
However, leading methods on visual representation learning in general do not belong to the family of generative modeling.
Instead, they include supervised discriminative learning (Krizhevsky et al., [2012](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib40); Dehghani et al., [2023](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib22); Dosovitskiy et al., [2020](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib23)),
contrastive learning (Chen et al., [2020b](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib16); He et al., [2020](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib32); Chen et al., [2020c](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib17); Zhai et al., [2023](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib78); Tschannen et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib63); Radford et al., [2021](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib56)),
bootstrapping (Caron et al., [2021](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib14); Grill et al., [2020](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib29)),
auto-encoding (He et al., [2022](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib33); Bao et al., [2021](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib3); Chen et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib18)) among others, and their combinations (Oquab et al., [2023](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib50); Siméoni et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib62); Zhou et al., [2021](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib82); Cao et al., [2026](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib12)).
Early efforts in generative vision pretraining (Chen et al., [2020a](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib15); Bai et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib2)) have shown promising scaling behaviors but their effectiveness has lagged behind non-generative models.


In this paper, we investigate whether visual generative models are secretly generalist vision learners, i.e., whether models trained for image and video generation develop internal representations that are suitable for visual understanding tasks.
To achieve this, we finetune a pretrained image generator with a small amount of computer vision data (depth estimation, surface normal estimation, segmentation, etc.).
We then evaluate the resulting model on a wide variety of vision benchmarks.
If the finetuned model performs at or near SOTA on these benchmarks, while retaining its image generation capabilities, then there is strong evidence that the image generator was indeed a foundation model for visual understanding – i.e., a generalist vision learner.


This is not the first paper to use image and video generators as base models for visual understanding.
Previous research observes that state-of-the-art image and video generators can generate visual content that look like RGB visualizations of computer vision outputs for tasks such as segmentation, depth estimation, and surface normal estimation (Zuo et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib84); Wiedemer et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib73)). However, those methods do not provide state-of-the-art results on modern benchmarks.
This is partially because these models don’t strictly follow the prompts to produce vision outputs in the desired formats that can be decoded back to vision outputs for computing quantitative metrics.
Other researchers (He et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib30), [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib31); Ke et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib37); Ye et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib75); Yu et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib77); Zhao et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib81); Wang et al., [2026b](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib68)) adapt the generation architectures by adding specialized modules and performing full-finetuning to achieve SOTA-level results on specific target tasks.
Although these methods successfully leverage the understanding capabilities of the pre-trained features, they sacrifice the model’s generality across other understanding and generation tasks.


[图片：Refer to caption]


Figure 1: We demonstrate the hidden visual understanding capabilities of image generators by instruction-tuning Nano Banana Pro.
The instruction-tuned model, Vision Banana [图片：Refer to caption] , can produce visualizations in a precise format that can enable evaluation on established benchmarks.


We take an approach motivated by recent advancements in large language models (LLMs). In natural language processing (NLP), generative pretraining (Brown et al., [2020](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib7); Chowdhery et al., [2023](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib19)) is performed to produce base models, often referred to as LLMs, that are good at generating text, whereas instruction-tuning (Ouyang et al., [2022](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib51); Wei et al., [2021](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib72)) guides them to follow specific tasks and produce text in requested formats and stay on the task.
Analogously, we position a visual generative model as a “base” model and perform instruction-tuning to align the model to produce visual output in desired formats, in accordance with the prompts, as illustrated in Fig. [1](https://ar5iv.labs.arxiv.org/html/2604.20329#S1.F1).
Specifically, the model is instructed to produce RGB images that can be decoded to computer vision outputs.
Such instruction prompts and decodable visualization schemes are designed to bridge and calibrate the visual generations to formats where measurable metrics for benchmarking can be applied.
For example, by prompting the model to “Segment the skateboard category in pure yellow (<255, 255, 0>)”, we can easily parse the mask for skateboard by clustering pixels whose values are close to <255, 255, 0>.
This strategy has three main advantages.
First, it supports a wide variety of tasks with a single unified model – after instruction tuning, the weights are shared among all tasks, and only the prompt changes.
Second, it requires relatively little new training data, since the instruction tuning is solely teaching the model how to format computer vision outputs as RGB.
Third, it helps the model retain its original image generation capabilities, since the outputs are simply new RGB images.


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| Capabilities | Benchmarks and Metrics | Vision Banana [图片：[Uncaptioned image]] | Best Counterpart |
| 2D Understanding [rowspan=4] | Referring segmentation: RefCOCOg UMD val (cIoU $\uparrow$) | 0.738 | 0.734 (SAM3 Agent) |
| Referring segmentation: ReasonSeg val (gIoU $\uparrow$) | 0.793 | 0.770 (SAM3 Agent) | |
| Semantic segmentation: Cityscapes val (mIoU $\uparrow$) | 0.699 | 0.652 (SAM3) | |
| Instance segmentation: SA-Co/Gold ($pmF_{1}$ $\uparrow$) | 0.540* | 0.552 (DINO-X) | |
| 3D Understanding [rowspan=2] | Metric depth estimation: average of 4 datasets ($\delta_{1}$$\uparrow$) | 0.929 | 0.918 (Depth Anything 3) |
| Surface normal estimation: average of 4 datasets (mean angle error $\downarrow$) | 18.928 | 19.642 (Lotus-2) | |
| Visual Generation [rowspan=2] | Text-to-image: GenAI-Bench (win rate against the other $\uparrow$) | 53.5% | $46.5\%$ (Nano Banana Pro) |
| Image editing: ImgEdit (win rate against the other $\uparrow$) | $47.8\%$ | 52.2% (Nano Banana Pro) | |


*We evaluate on a randomly sampled subset of 500 queries from SA-Co/Gold to save compute.


Table 1: The instruction-tuned Vision Banana model surpasses or rivals SOTA specialists across visual generation and understanding.
For 2D visual understanding, it beats the highly specialized Segment Anything Model 3 (Carion et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib13)) on 3 segmentation datasets, and is on par with DINO-X (Ren et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib58)) on instance segmentation.
For 3D visual understanding, it surpasses the best metric depth estimation expert, Depth Anything 3 (Lin et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib43)), and the best surface normal estimation specialist, Lotus-2 (He et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib31)).
In visual generation, Vision Banana inherits its capabilities from Nano Banana Pro and is on par with it on text-to-image and image editing.


We present Vision Banana [图片：[Uncaptioned image]] , a generalist vision model trained by performing a lightweight instruction-tuning to Nano Banana Pro on a mixture of its original image generation data and our additional vision task data. During evaluation across several benchmarks, we find that Visual Banana excels at both visual understanding and generation, as summarized in Tab. [1](https://ar5iv.labs.arxiv.org/html/2604.20329#S1.T1). On the understanding side, Vision Banana surpasses or matches state-of-the-art results on both 2D and 3D tasks. For example, it beats the highly specialized segmentation model, SAM 3 (Carion et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib13)), on various segmentation tasks, and the 3D expert, Depth Anything 3 (Lin et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib43)), on metric depth estimation. On the generation side, it performs on par with its base model on image generation and editing benchmarks. On GenAI-Bench (Li et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib42)), Vision Banana scores a $53.5\%$ win rate against its base model. On ImgEdit (Ye et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib76)) for image editing, Vision Banana’s win rate is $47.8\%$. Since these results are achieved with a single unified model built by a lightweight instruction-tuning on its base model, there is strong evidence that Nano Banana Pro already possessed internal representations for visual understanding, which only needed to be unlocked with instruction tuning.


The implications of this study are two-fold.
First, it suggests that image generators are indeed generalist vision learners under the hood, with
generative vision pretraining playing a foundational role similar to language model pretraining.
Second, it suggests that image generation can serve as a universal interface for unified visual understanding, mirroring the role of text generation in language understanding and reasoning.
We could be witnessing a major paradigm shift for computer vision, where generative vision pretraining takes a central role in building Foundational Vision Models for both generation and understanding.


<a id="source-section-4"></a>

## 2 Method


<a id="source-section-5"></a>

#### Instruction-tuning Nano Banana Pro


Recent image and video generators have demonstrated zero-shot capabilities in generating visualizations of visual understanding tasks (Wiedemer et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib73); Zuo et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib84)).
To rigorously investigate and benchmark these capabilities, we need to align the models to generate visualizations that can be decoded back to visual task outputs for quantitative evaluation.
For example, in metric depth estimation, a generated depth heatmap must be invertible back to physical depth values for quantitative assessment.
Therefore, we create Vision Banana by instruction-tuning our base model, Nano Banana Pro, on a selection of vision tasks formatted in such invertible manners.
Specifically, we mix vision task data into Nano Banana Pro’s own training mixture at a very low ratio.
This process allows us to align the model’s emergent generative representations into measurable physical geometry and semantic labels, allowing our single generalist model to be evaluated alongside task-specific specialists.


Mixing the vision data at a low ratio serves as a lightweight instruction-tuning strategy, ensuring that our vision task alignment does not degrade the model’s original generative priors.
We validate the preservation of image generation capabilities by benchmarking Vision Banana against the base Nano Banana Pro on two tasks: text-to-image generation (GenAI-Bench (Li et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib42))) and image editing (ImgEdit (Ye et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib76))).
In human evaluations, we obtain win rates of 53.5% and 47.8% respectively, indicating that Vision Banana successfully maintains the generative power of the base model.
Qualitative comparisons in Fig. [9](https://ar5iv.labs.arxiv.org/html/2604.20329#Sx2.F9) (text-to-image generation) and Fig. [10](https://ar5iv.labs.arxiv.org/html/2604.20329#Sx2.F10) (image editing) further confirm that the outputs are highly similar between Vision Banana and Nano Banana Pro.
These results verify that Vision Banana does not forget its generative nature.


<a id="source-section-6"></a>

#### Vision Tasks and Data.


We evaluate our framework on two fundamental categories of visual understanding: $2$D scene understanding and $3$D structure inference.
The $2$D suite consists of referring expression, semantic, and instance segmentation, which collectively test the model’s capability to ground natural language and segment the corresponding objects.
For $3$D understanding, we focus on monocular metric depth and surface normal estimation, which demand geometric reasoning and internal knowledge about object scales.
To collect data for instruction tuning, we utilize in-house model annotations for web-crawled 2D images, as well as synthetic data from rendering engines for 3D tasks.
Crucially, no training data from our evaluation benchmarks is included in the instruction-tuning mixture, ensuring that our results reflect true generalist capability.


<a id="source-section-7"></a>

## 3 Vision Banana - Generalist Vision Model from Image Generator


In this section, we present qualitative and quantitative assessments compared to task-specific specialist models.
Built upon an image generator,
Vision Banana achieves SOTA-level results across a broad range of visual understanding tasks, without specialized architectures or custom training losses.


<a id="source-section-8"></a>

### 3.1 2D Semantic Understanding


Table 2: Vision-Banana [图片：[Uncaptioned image]] compared with state-of-the-art methods on various segmentation datasets.
We mainly compare with other methods that have not been trained on in-domain data, i.e., the training splits of these benchmarks.
We denote them as “Zero-Shot Transfer” in the table.
The usage of this term follows Segment Anything (Kirillov et al., [2023](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib38)) and CLIP (Radford et al., [2021](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib56)).
Non zero-shot transfer methods are marked in gray.
∗: On SA-Co/Gold, we evaluate our method on $500$ randomly sampled queries.
On ReasonSeg, methods are paired with multimodal LLMs for reasoning. We use Gemini 2.5 Pro in our case.
References of previous methods are in the main text.


| 列 1 | 列 2 |
| --- | --- |
| Model | mIoU |
| Non Zero-Shot Transfer [colspan=2] | |
| SegMan-L | 0.842 |
| Zero-Shot Transfer [colspan=2] | |
| APE-D | 0.442 |
| OpenSeeD | 0.478 |
| X-Decoder | 0.520 |
| SAM 3 | 0.652 |
| Vision Banana | 0.699 |


(a) Cityscapes val.


| 列 1 | 列 2 |
| --- | --- |
| Model | $pmF_{1}$ |
| Non Zero-Shot Transfer [colspan=2] | |
| SAM 3 | 0.661 |
| Zero-Shot Transfer [colspan=2] | |
| APE-D | 0.369 |
| OWLv2 | 0.420 |
| Gemini 2.5 | 0.461 |
| DINO-X | 0.552 |
| Vision Banana | 0.540∗ |


(b) SA-Co/Gold.


| 列 1 | 列 2 |
| --- | --- |
| Model | cIoU |
| Non Zero-Shot Transfer [colspan=2] | |
| HyperSeg + Phi2 2.7B | 0.794 |
| X-SAM + Phi3 3.8B | 0.838 |
| Zero-Shot Transfer [colspan=2] | |
| HybridGL | 0.513 |
| Kang et al. + LLaVA-1.5 13B | 0.677 |
| SAM 3 + Gemini 2.5 Pro | 0.734 |
| Vision Banana | 0.738 |


(c) RefCOCOg val (U).


| 列 1 | 列 2 | 列 3 |
| --- | --- | --- |
| Model | MLLM | gIoU |
| Non Zero-Shot Transfer [colspan=2] | | |
| X-SAM | Phi-3-3.8B | 0.566 |
| LISA-13B-LLaVA1.5 | - | 0.650 |
| Zero-Shot Transfer [colspan=2] | | |
| SegZero | Qwen2.5-VL 7B | 0.626 |
| RSVP | GPT-4o | 0.647 |
| SAM 3 Agent | Gemini 2.5 Pro | 0.770 |
| Vision Banana | Gemini 2.5 Pro | 0.793 |


(d) ReasonSeg val.


Image segmentation stands as a cornerstone of visual understanding, traditionally requiring complex, task-specific models to parse pixels into semantic categories or object instances. Current leading methods such as the Segment Anything series (Kirillov et al., [2023](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib38); Ravi et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib57); Carion et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib13)) tackle this through heavy architectural specialization and large volumes of expensive, human-annotated mask data. Vision Banana challenges this prevailing paradigm by demonstrating that SOTA segmentation can naturally emerge from an image generation model. Rather than training on vast amounts of meticulously crafted segmentation examples, we tap into the rich representations learned by the base image generation model. By instructing the model to generate multi-colored images of segmentation masks, we obtain dense segmentation maps from which individual masks can be decoded, therefore enabling segmentation through image generation. As detailed in Tab. [2(d)](https://ar5iv.labs.arxiv.org/html/2604.20329#S3.T2.st4), this elegant generative approach outperforms highly tuned specialist models, achieving SOTA zero-shot transfer performance on three of the four evaluated datasets.


<a id="source-section-9"></a>

#### Semantic Segmentation.


Historically, the task “semantic segmentation” is to classify each pixel into one of the predefined categories, without distinguishing instances of the same category. For example, Cityscapes (Cordts et al., [2016](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib20)) has $19$ classes, including road, person, truck, vegetation, sky, and more.
It is worth noting that advanced tasks of “instance / referral expression” segmentation are also semantic.
Here, we use “semantic segmentation” strictly to refer to the traditional, instance-agnostic, category-level classification task.
In our case, this nature of the classical semantic segmentation task can be specified via the prompt, and we train the model to follow such instructions.
We prompt the model to generate a visualization image where each pixel is colored according to its class, as shown in Fig. [2](https://ar5iv.labs.arxiv.org/html/2604.20329#S3.F2).
Note that the class can be any text string, not limited to a fixed set.
The color for each class is specified in the prompt.
We can use natural language to describe the correspondence.
We can also use a mapping notation such as JSON.
The color can be represented as hex numbers or RGB value tuples.
In evaluation, we assign pixels to classes by matching its color according to the prompt.


We compare Vision Banana with existing methods (Fu et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib25); Shen et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib60); Zhang et al., [2023](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib80); Zou et al., [2023](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib83); Carion et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib13)) in Tab. [2(a)](https://ar5iv.labs.arxiv.org/html/2604.20329#S3.T2.st1).
On Cityscapes (Cordts et al., [2016](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib20)), Vision Banana surpasses SAM 3 by $4.7$ points in mIoU and is the best open vocabulary model, narrowing the gap with
closed-set models such as SegMan (Fu et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib25)).


[图片：Refer to caption]


(a) Example 1


[图片：Refer to caption]


“Generate a semantic segmentation visualization image, using this color mapping: {"cat": "red", "lock": "pink", "exit sign": "light purple", "background": yellow}.”


[图片：Refer to caption]


“Generate a visualization image of semantic segmentation, using this color mapping: {"cat ears": <255, 165, 0>, "exit sign": <0, 0, 255>, "background":<125,0, 125>}”


[图片：Refer to caption]


(b) Example 2


[图片：Refer to caption]


“This image is a per-pixel class labeling of the input. The macaron cakes are represented by (255, 255, 0). The round plates are represented by (255, 192, 128). The slice cakes are depicted in (64, 192, 64). The flowers are shown in (128, 0, 64). The tongs are (255, 0, 192).”


[图片：Refer to caption]


“Generate a semantic segmentation visualization of the input. The menu is #80C000. The dessert is #800000. The patterns on the wall is #40FFC0”


Figure 2: Vision Banana can perform semantic segmentation, following the instruction prompts. It handles various prompting styles.
It can also segment anything specified via text prompts, from single-word nouns to phrases.
It produces segmentation mask at fine-grained granularity, for example, the cats’ whiskers in Example 1 (middle).


[图片：Refer to caption]


(a) Example 1


[图片：Refer to caption]


“Generate an instance segmentation visualization of this image. Each piece of garlic is colored differently.”


[图片：Refer to caption]


“Generate an instance segmentation visualization of this image. Each piece of beef is colored differently.”


[图片：Refer to caption]


(b) Example 2


[图片：Refer to caption]


“Generate an instance segmentation visualization of this image. Each price tag is colored differently. ”


[图片：Refer to caption]


“This image is a segmentation task derived from the input. The "crescent"-shaped croissant instances are each represented by a unique, solid color. Background is RGB(88, 50, 82).”


[图片：Refer to caption]


(c) Example 3


[图片：Refer to caption]


“This image shows segmentation masks for the basketballs from the input image. The background is set to #10aa05. Each basketball instance is represented by a solid circular mask, and a different colora is used for each mask.”


[图片：Refer to caption]


“This image shows segmentation masks for the balls from the input image. The background is set to white color. Each ball is represented by a different color.”


Figure 3: Vision Banana can perform instance segmentation, one class at a time. It renders different instances with different colors.
It can understand the nuanced concept in language as well.


<a id="source-section-10"></a>

#### Instance Segmentation.


Unlike semantic segmentation, instance segmentation requires the model to distinguish between individual objects that belong to the same class.
For example, if an image contains five dogs, we expect the model to produce an individual mask for each distinct animal.
This poses a unique challenge for Vision Banana: since the number of instances is unknown in advance, we cannot assign the colors in the prompt.
To address this, we adopt a per-class inference strategy. For each inference, we instruct Vision Banana to produce segmentation masks for only one class, allowing the model to dynamically assign colors to different instances.
Examples of instance segmentation are shown in Fig. [3](https://ar5iv.labs.arxiv.org/html/2604.20329#S3.F3).
During evaluation, we simply cluster pixels that have similar colors by thresholding.


[图片：Refer to caption]


(a) Input image


[图片：Refer to caption]


A segmentation map image. The area that corresponds to the man in pink t shirt is rendered solid white; the other man is rendered in green.


[图片：Refer to caption]


(b) Input image


[图片：Refer to caption]


A segmentation map image. The stretching cat is rendered in green, the cat that is cleaning itself is in cyan.


[图片：Refer to caption]


(c) Input image


[图片：Refer to caption]


This image shows segmentation masks from the given image. The background is black color. The game control device is represented by a solid yellow.


[图片：Refer to caption]


(d) Input image


[图片：Refer to caption]


This image shows segmentation masks from the given image. The background is black color. The chef’s names in both Chinese and English are rendered as cyan color.


Figure 4: Vision Banana can understand natural language prompts and reason about them, including but not limited to: (a) description of objects’ appearances (“man in pink t shirt”); (b) description of actions (“stretching” and “cleaning”); (c) objects that have uncommon usage (toaster as a game controller); (d) multilingual text content (text on the menu in Chinese and English). This requires strong and comprehensive visual understanding capability.


We evaluated our model on SA-Co/Gold (Carion et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib13)) using the $pmF_{1}$ metric, summarizing results in Tab. [2(b)](https://ar5iv.labs.arxiv.org/html/2604.20329#S3.T2.st2).
Compared with SAM 3, Vision Banana still lags behind in instance segmentation on SA-Co/Gold, highlighting some challenges in this task. Note that unlike SAM 3, we did not include SA-Co in our model’s training data.
Under the zero-shot transfer setting,
Vision Banana surpasses many existing methods, including Gemini 2.5 (Gemini Team, [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib26)) ($pmF_{1}=0.461$), APE-D (Shen et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib60)) ($pmF_{1}=0.369$), and OWLv2 (Minderer et al., [2023](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib48)) ($pmF_{1}=0.420$).
Vision Banana is on par with DINO-X (Ren et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib58)) ($pmF_{1}$ score $0.540$ v.s. $0.552$).


<a id="source-section-11"></a>

#### Referring Expression Segmentation.


Unlike traditional fixed-class segmentation, referring expression segmentation is based on free-form text queries. This requires models to comprehend and reason the nuanced natural language expressions, as well as understand complex relationships among objects.
Vision Banana is a natural fit for this task and establishes a new state-of-the-art. As summarized in Tab. [2(c)](https://ar5iv.labs.arxiv.org/html/2604.20329#S3.T2.st3) and Tab. [2(d)](https://ar5iv.labs.arxiv.org/html/2604.20329#S3.T2.st4), it achieves a cIoU of $0.738$ on RefCOCOg UMD (Kazemzadeh et al., [2014](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib36)) and an IoU of $0.793$ on ReasonSeg (Lai et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib41)), outperforming SAM 3 Agent (Carion et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib13)) and other previous works under the zero-shot transfer setting, including HybridGL (Liu and Li, [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib44)), Kang et al. ([2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib35)), SegZero (Liu et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib45)), and RSVP (Lu et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib46)).
On RefCOCOg, there is still some gap from methods that are trained on the training split, including HyperSeg (Wei et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib71)) and X-SAM (Wang et al., [2026a](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib66)).
On ReasonSeg, Vision Banana + Gemini 2.5 Pro even beats methods that are not zero-shot, such as X-SAM (Wang et al., [2026a](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib66)) and LISA (Lai et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib41)).
Fig. [4](https://ar5iv.labs.arxiv.org/html/2604.20329#S3.F4) shows qualitative examples of various referring expression segmentation.
This success highlights a key advantage of our approach: the multimodal intelligence inherited from generative pre-training allows Vision Banana to reason about “what” to segment more effectively than discriminative models.


Interestingly, Vision Banana also demonstrates a similar mastery of referring expressions on semantic and instance segmentation, despite not being explicitly trained to condition these specific tasks on free-form text queries.
For example, in Fig. [2(b)](https://ar5iv.labs.arxiv.org/html/2604.20329#S3.F2.sf2) (right), the model understands what “patterns on the wall” is referring to.
In Fig. [3(b)](https://ar5iv.labs.arxiv.org/html/2604.20329#S3.F3.sf2) (right), the model successfully distinguishes crescent-shaped croissants from other variations of croissants.
These observations suggest robust cross-task transfers in Vision Banana.


<a id="source-section-12"></a>

### 3.2 3D Understanding from Monocular Images


Vision Banana demonstrates a strong ability to infer 3D structures from 2D monocular images. We evaluate this capability on two classical tasks: monocular metric depth estimation and surface normal estimation.
As summarized in Tab. [1](https://ar5iv.labs.arxiv.org/html/2604.20329#S1.T1), Vision Banana achieves SOTA performance on both tasks, surpassing specialists such as Depth Anything V3 (Lin et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib43)) and Lotus-2 (He et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib31)).


<a id="source-section-13"></a>

#### Metric Depth Estimation.


The goal of depth estimation is to produce a depth map from a monocular image, where each pixel’s value represents the physical metric distance from the camera plane to the observed object (Eigen et al., [2014](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib24)). This is a fundamental computer vision task that benefits a wide range of applications such as robotics, augmented/virtual reality, and autonomous driving.
However, depth estimation is inherently ill-posed, as 2D projections inherently discard critical 3D geometric information.
Furthermore, monocular depth estimation is particularly challenging due to the absence of parallax cues available in multi-view setups, even when camera intrinsic parameters are known.


In the deep-learning era, the research community has largely framed depth estimation as a dense per-pixel supervised regression problem, employing specialized architectures and domain-specific loss functions. Most recent SOTA methods rely on camera intrinsics during training, inference, or both (Yang et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib74); Bochkovskii et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib6); Wang et al., [2025b](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib69), [c](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib70); He et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib31), [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib30); Hu et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib34); Cai et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib11); Lin et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib43); Piccinelli et al., [2025b](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib53), [a](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib52)).
While using intrinsics mitigates the inherent ambiguity of depth estimation, it also necessitates specialized model designs.
In contrast, our work is predicated on the hypothesis that the mode-seeking nature of generative modeling naturally resolves training target ambiguities, thereby eliminating the need for such specialized techniques.
Furthermore, the broad world knowledge acquired during pretraining endows the model with stronger priors on object sizes and distances compared to narrowly targeted models.
To enable Nano Banana Pro to estimate depth in metric units, we instruct the model to output a carefully constructed false-color visualization of depth values.


To visualize depth maps as RGB images, we establish a mapping between unbounded depth values in $[0,\infty)$ and bounded RGB values in $[0,1]^{3}$.
Because the utility of accurate metric depth for nearby image content is generally higher than that of distant content (e.g., graspable objects matter more for robotics tasks, stereo/monodepth benchmarks usually measure accuracy terms of disparity or relative/log-depth) we “curve” metric depth prior to RGB encoding.
Specifically, this is achieved by first applying the power transform of Barron ([2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib4)) to warp the depth values, and then using those curved distances to produce a false-color visualization.
We constrain the power transform to $\lambda<-1$ and rescale it to map metric distances $d\in[0,\infty)$ to normalized distances in $[0,1)$:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>f(d,\lambda,c)=1-\mathopen{}\left(1-\nicefrac{{d}}{{\lambda c}}\mathclose{}\right)^{\lambda+1}<br>$$ | | (1) |


In all experiments, we set the shape parameter to $\lambda=-3$ and the scale parameter to $c=\nicefrac{{10}}{{3}}$. These curved and normalized distances $f(d,\lambda,c)$ are then used to interpolate along a piecewise-linear function that follows the edges of the RGB cube, traversing along its edges from black to white, similarly to the first iteration of a 3D Hilbert curve. A visualization of this process is provided in Fig. [5](https://ar5iv.labs.arxiv.org/html/2604.20329#S3.F5.5).


[图片：Refer to caption]


Figure 5: A visualization of our bijection between scalar metric distances $d\geq 0$ and RGB color values in $[0,1]^{3}$, which is achieved by curving metric depth with a power transform, and then interpolating along the edges of the color cube according to that curved metric depth. The metric depth values (in meters) corresponding to various RGB colors are overlaid.


This mapping from normalized distance to RGB color can be inverted by simply projecting the RGB values onto the nearest line segment and then inverting the linear interpolation along the cube’s edges. Because both the false-color visualization and the power transform are strictly invertible, their composition forms a bijection between metric depth in $[0,\infty]$ and RGB space in $[0,1]^{3}$. During training, we apply this mapping to ground-truth metric depths to generate RGB training targets. At inference, we apply the inverse mapping to decode the model’s generated RGB images back into metric depth, enabling direction evaluation on standard depth benchmarks. To enhance the model’s robustness across diverse color representations, we augment our training data with alternative color maps, such as Plasma, Inferno, Viridis, and grayscale.


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 | 列 6 | 列 7 | 列 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [colspan=2] | DepthLM-7B<br><br>(Cai et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib11)) | Depth Any. v3<br><br>(Lin et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib43)) | Depth Pro<br><br>(Bochkovskii et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib6)) | UniK3D<br><br>(Piccinelli et al., [2025a](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib52)) | MoGe-2<br><br>(Wang et al., [2025c](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib70)) | Vision<br><br>Banana<br> [图片：[Uncaptioned image]] | |
| Camera<br><br>Intrinsics [rowspan=2] | Inference | ✓ | ✓ | | | | |
| Training | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| Average | $\delta_{1}\uparrow$ | partial* | partial* | 0.715 | 0.823 | 0.802 | 0.882 |
| Benchmarks | AbsRel $\downarrow$ | | | | 0.156 | 0.144 | 0.116 |
| NYU | $\delta_{1}\uparrow$ | 0.915 | 0.963 | 0.961 | 0.965 | 0.961 | 0.948 |
| (Silberman et al., [2012](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib61)) | AbsRel $\downarrow$ | | 0.07 | | 0.074 | 0.0733 | 0.081 |
| iBims1 | $\delta_{1}\uparrow$ | 0.92 | | 0.913 | 0.919 | 0.830 | 0.934 |
| (Koch et al., [2018](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib39)) | AbsRel $\downarrow$ | | | | 0.104 | 0.136 | 0.078 |
| ETH3D | $\delta_{1}\uparrow$ | 0.718 | 0.917 | 0.415 | 0.687 | 0.908 | 0.935 |
| (Schops et al., [2019](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib59)) | AbsRel $\downarrow$ | | 0.104 | 0.327 | 0.236 | 0.104 | 0.103 |
| DIODE-Indoor | $\delta_{1}\uparrow$ | | 0.838 | 0.671 | 0.713 | 0.664 | 0.917 |
| (Vasiljevic et al., [2019](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib65)) | AbsRel $\downarrow$ | | 0.123 | 0.199 | 0.161 | 0.175 | 0.108 |
| KITTI | $\delta_{1}\uparrow$ | | 0.953 | 0.843$\ddagger$ | 0.812 | 0.629 | 0.915 |
| (Uhrig et al., [2017](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib64)) | AbsRel $\downarrow$ | | 0.086 | 0.121$\ddagger$ | 0.174 | 0.181 | 0.107 |
| nuScenes | $\delta_{1}\uparrow$ | 0.865$\dagger$ | | 0.491 | 0.840 | 0.820 | 0.643 |
| (Caesar et al., [2020](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib10)) | AbsRel $\downarrow$ | | | 0.287 | 0.189 | 0.195 | 0.219 |


* The average $\delta_{1}$ of DepthLM-7B on the 4 datasets it evaluated on (NYU + iBims1 + ETH3D + nuScenes) is $0.855$; our average $\delta_{1}$ on the same 4 datasets is $0.865$. The average $\delta_{1}$ of Depth-Anything V3 on the 4 datasets it evaluated on (NYU + ETH3D + DIODE + KITTI) is $0.918$; our average $\delta_{1}$ on the same 4 datasets is $0.929$.

$\dagger$ DepthLM is trained on nuScenes so it’s not zero-shot.

$\ddagger$ Numbers reported by Depth-Anything V3 (Lin et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib43)).


Table 3: Monocular metric depth estimation under the zero-shot transfer setting. Vision Banana achieves superior results on public datasets without using camera intrinsics in neither training of inference.
Metrics marked with $\uparrow$ are better if higher; metrics marked with $\downarrow$ are better if lower.


Tab. [3](https://ar5iv.labs.arxiv.org/html/2604.20329#S3.T3) presents the empirical results of Vision Banana compared to specialist models across six major academic benchmarks. Vision Banana achieves an average $\delta_{1}$ accuracy of 0.882, outperforming Unik3D (Piccinelli et al., [2025a](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib52)) by nearly 6 points, while achieving a 20% lower absolute relative error (AbsRel) compared to MoGe-2 (Wang et al., [2025c](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib70)).
Notably, Vision Banana outperforms Depth Anything V3 (Lin et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib43)) on average across the four datasets (NYU, ETH3D, DIODE, KITTI) on which it was evaluated ($0.929$ v.s. $0.918$), demonstrating robust performance in both near-field and distant scenes.
Our model is trained entirely on synthetic depth data created from simulation engines — we use zero real-world depth data, and exclude training data from any of the depth datasets we evaluate on.
Note that this result is achieved without relying on camera parameters (neither intrinsics nor extrinsics) during both training or inference.
By leveraging the immense geometric priors embedded in its foundation model, Vision Banana infers absolute scale solely from visual cues and object relationships, enabling zero-shot generalization to any arbitrary input image.


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| [图片：Refer to caption] | [图片：Refer to caption] | [图片：Refer to caption] | [图片：Refer to caption] |
| [图片：Refer to caption] | [图片：Refer to caption] | [图片：Refer to caption] | [图片：Refer to caption] |
| [图片：Refer to caption] | [图片：Refer to caption] | [图片：Refer to caption] | [图片：Refer to caption] |
| [图片：Refer to caption] | [图片：Refer to caption] | [图片：Refer to caption] | [图片：Refer to caption] |
| [图片：Refer to caption] | [图片：Refer to caption] | [图片：Refer to caption] | [图片：Refer to caption] |
| Input image | Generated depth image | Vis. view 1 | Vis. view 2 |


Figure 6:
Demonstration of Vision Banana’s metric depth estimation capabilities.
The two columns to the left are the input images and the depth visualization image generated by Vision Banana.
The depth images are then decoded back to metric depth values. Combining them with the camera intrinsics, we can reconstruct the 3D scene accurately. The two columns on the right are random views of the reconstructed scenes.
Note that camera intrinsics are not needed in predicting the depth itself.
Samples taken from NYU v2 (Silberman et al., [2012](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib61)) and ETH 3D (Schops et al., [2019](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib59)).


[图片：Refer to caption]


(a) Photo taken at Kinkaku-Ji


[图片：Refer to caption]


(b) Vision Banana estimated depth


[图片：Refer to caption]


(c) Measurement from Google Maps


Figure 7: Vision Banana depth estimation in the wild. (a) Author of this paper takes a picture near Kinkaku-Ji with a consumer cell-phone. (b) Vision Banana generates a depth estimation image. The depth value at the position marked by a green star is decoded to be $13.71$ meters. (c) Author then measures the actual distance using Google Map, which turns out to be 12.87 meters. The AbsRel error at this point is around $0.065$.


Qualitative inspections further validate the model’s capabilities. As illustrated in Fig. [6](https://ar5iv.labs.arxiv.org/html/2604.20329#S3.F6), Vision Banana generates highly precise depth maps that preserve crisp geometric details, even in cluttered environments like classrooms.
When these 2D predictions are unprojected into 3D point clouds, they exhibit global consistency across diverse scenes, maintaining accurate planar surfaces and correct geometry.
In addition to common academic benchmarks, we also conducted a “vibe test” using a casual smartphone photograph, as shown in Fig. [7](https://ar5iv.labs.arxiv.org/html/2604.20329#S3.F7). Crossed validated by depth measured on Google Maps, Vision Banana successfully produced an accurate depth estimation on this photo captured by a consumer device unseen during training.


<a id="source-section-14"></a>

#### Surface Normal Estimation.


Surface normal estimation represents another critical vision task. Surface normals, which are unit vectors $(x,y,z)$ with values ranging from $-1.0$ to $1.0$, serve as a critical proxy for local geometry and scene structures.
Unlike the complex color mapping required for metric depth, the visualization of surface normals is intrinsically aligned with the RGB color space, allowing straightforward integration into our model.


We specifically utilize a camera-space normal formulation using the standard right-handed coordinate system (+x right, +y up, +z pointing out of the image plane). In this representation, the directional vector components map directly to RGB channels:


- •


Facing Left $(-1,0,0)$: Encoded as Pinkish Red.


- •


Facing Up $(0,1,0)$: Encoded as Light Green.


- •


Facing the Camera $(0,0,1)$: Encoded as Light Blue/Purple.


Table [4](https://ar5iv.labs.arxiv.org/html/2604.20329#S3.T4) compares Vision Banana against SOTA specialist methods on four public benchmarks. When averaged across the three indoor datasets, Vision Banana achieves the lowest mean and median angular errors. It also demonstrates competitive accuracy on outdoor scenes.


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 | 列 6 | 列 7 | 列 8 | 列 9 | 列 10 | 列 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Methods [rowspan=3] | Indoor [colspan=2] | Indoor [colspan=6] | Outdoor [colspan=2] | | | | | | | |
| Average [colspan=2] | NYUv2<br><br>(Silberman et al., [2012](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib61)) [colspan=2] | DIODE-indoor<br><br>(Vasiljevic et al., [2019](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib65)) [colspan=2] | ScanNet<br><br>(Dai et al., [2017](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib21)) [colspan=2] | VKitti<br><br>(Cabon et al., [2020](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib9)) [colspan=2] | | | | | | |
| mean $\downarrow$ | median $\downarrow$ | mean $\downarrow$ | median $\downarrow$ | mean $\downarrow$ | median $\downarrow$ | mean $\downarrow$ | median $\downarrow$ | mean $\downarrow$ | median $\downarrow$ | |
| Marigold (Ke et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib37)) | 19.606 | 11.828 | 20.864 | 11.134 | 16.671 | 12.084 | 21.284 | 12.268 | – | – |
| DSINE (Bae and Davison, [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib1)) | 17.017 | 10.190 | 16.4 | 8.4 | 18.453 | 13.871 | 16.2 | 8.3 | 28.9 | 9.9 |
| StableNormal (Ye et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib75)) | 17.168 | 10.028 | 19.707 | 10.527 | 13.701 | 9.46 | 18.098 | 10.097 | – | – |
| Lotus-2-Normal (He et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib31)) | 16.558 | – | 16.9 | N/A | 18.575 | N/A | 14.2 | N/A | 28.894 | 9.677 |
| Vision Banana [图片：[Uncaptioned image]] | 15.549 | 9.300 | 17.778 | 8.876 | 13.818 | 11.556 | 15.052 | 7.468 | 29.063 | 10.699 |


Table 4: Surface normal estimation results. Vision Banana achieves the lowest mean and median angle errors on the indoor datasets on average, and is on par with previous SOTA on outdoor scenes.


| 列 1 | 列 2 | 列 3 |
| --- | --- | --- |
| [图片：Refer to caption] | [图片：Refer to caption] | [图片：Refer to caption] |
| [图片：Refer to caption] | [图片：Refer to caption] | [图片：Refer to caption] |
| [图片：Refer to caption] | [图片：Refer to caption] | [图片：Refer to caption] |
| Input image | Lotus-2-Normal | Vision Banana [图片：Refer to caption] |


Figure 8:
Comparison with SOTA surface normal estimation method Lotus-2 (He et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib31)).
Results of Lotus-2 are obtained using its Hugging-Face demo: [https://huggingface.co/spaces/haodongli/Lotus-2_Normal](https://huggingface.co/spaces/haodongli/Lotus-2_Normal).
Vision Banana can produce surface normal map with much higher visual quality and better fine-grained details.
Zoom-in for the details.


Figure [8](https://ar5iv.labs.arxiv.org/html/2604.20329#S3.F8) visually compares the output from Vision Banana with the leading external method, Lotus-2 (He et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib31)). Vision Banana consistently produces surface normal maps with significantly higher fidelity and finer granular details.
The bottom row of Fig. [8](https://ar5iv.labs.arxiv.org/html/2604.20329#S3.F8) highlights a sample from Virtual KITTI 2 (Cabon et al., [2020](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib9)). Although Vision Banana registers slightly higher quantitative errors on this benchmark compared to Lotus-2, it yields demonstrably superior visual quality. Also note that Lotus-2 is trained on Virtual KITTI 2 for surface normal estimation, whereas Vision Banana maintains a strict zero-shot transfer protocol, having never seen the training sets of any evaluated benchmarks.


<a id="source-section-15"></a>

## 4 Discussion


<a id="source-section-16"></a>

#### Image Generators are Generalist Vision Learners.


Generative pretraining (Radford et al., [2018](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib54), [2019](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib55); Brown et al., [2020](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib7)) has fundamentally transformed language understanding and reasoning.
In the meantime, recent observations of emergent vision capabilities (Wiedemer et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib73); Zuo et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib84)) have ignited speculation that computer vision is approaching a similar paradigm shift.
By instruction-tuning a leading image generator, Nano Banana Pro, into a state-of-the-art visual generation and understanding model, we confirm that this shift is already underway.
Models pretrained on large-scale image generation naturally acquire robust visual understanding capabilities.
These generative priors surpass the specialized architectures and dedicated training paradigms traditionally employed by specialist vision models.
We are witnessing a paradigm shift for computer vision that will be fueled by generative vision pretraining, which we believe paves the way for true Foundational Vision Models and Artificial General Intelligence from Vision (AGI-V).


<a id="source-section-17"></a>

#### Image Generation as a Universal Interface.


As a byproduct of this study, we show that image generation can serve as the universal interface for computer vision, analogous to how text generation acts as the unifying interface for many tasks embedded in natural language, including language understanding, generation, reasoning, math, coding, agentic tasks, etc..
By representing vision task outputs as RGB images, we can use natural language prompts to seamlessly instruct the model. While we are not the first to encode vision outputs as RGB (Ke et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib37); Zhao et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib81)), we demonstrate that when combined with powerful pretrained visual generators, this simple design is sufficient to outperform modern domain-specific specialist models.


In addition to the unification of vision task outputs as RGB images, generative modeling naturally provides a workaround for the ambiguity in vision tasks where a single input can correspond to several modes of the output distribution.
In order to prevent the collapse of the output to a blurry mean, expert discriminative models (Carion et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib13); Lin et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib43)) usually resort to custom architectures and training losses.
For example, the Segment Anything models (Kirillov et al., [2023](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib38); Ravi et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib57); Carion et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib13)) return several segmentation masks but only apply the loss to a single one.
Generative models, however, inherently learn the full data distribution, gracefully managing ambiguity by design.
By eliminating the need for bespoke architectural designs, this formulation could lead to a truly unified “omni” multimodal model.


<a id="source-section-18"></a>

#### Future Work.


While Vision Banana achieves SOTA results on fundamental tasks for 2D semantic understanding and 3D understanding from monocular images, several exciting avenues remain for future exploration.
First, scaling the diversity of instruction-tuned tasks may unlock further emergent cross-task generalization, similar to behaviors observed in LLMs Wei et al. ([2021](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib72)).
Second, our current evaluation focuses on monocular image inputs.
In the future, we can extend this framework to process multi-view inputs (Wang et al., [2025a](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib67)) and video inputs (Zhang et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib79)).
Similarly, investigating whether video generators yield even richer, temporally-aware visual representations presents a highly promising research direction.
Another important next step is exploring the synergistic integration of foundational vision models with large language models to enhance cross-modality reasoning.
Finally, utilizing image generators like Nano Banana Pro currently incurs a significantly higher computational overhead than running lightweight specialist models.
Developing acceleration and cost-reduction strategies will be an essential hurdle to overcome for the widespread deployment of generative vision framework.


<a id="source-section-19"></a>

## Acknowledgment


We thank
Xi Chen, Fei Xia,
Kaushik Shivakumar, Abhishek Sinha, Phillip Lippe, Yilin Gao, Javier Rey, Sanghyun Woo,
Renshen Wang, Wentao Yuan, Keran Rong, Rundi Wu,
Manoj Kumar, Manli Shu, Francesco Piccinno, Ishita Dasgupta,
Benigno Uria, Miki Rubinstein, Aäron van den Oord, Jon Shlens
for their helpful discussions, advice, and technical guidance.


<a id="source-section-20"></a>

## References


- Bae and Davison (2024)

G. Bae and A. J. Davison.


Rethinking inductive biases for surface normal estimation.


In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9535–9545, 2024.


- Bai et al. (2024)

Y. Bai, X. Geng, K. Mangalam, A. Bar, A. L. Yuille, T. Darrell, J. Malik, and A. A. Efros.


Sequential modeling enables scalable learning for large vision models.


In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22861–22872, 2024.


- Bao et al. (2021)

H. Bao, L. Dong, S. Piao, and F. Wei.


Beit: Bert pre-training of image transformers.


arXiv preprint arXiv:2106.08254, 2021.


- Barron (2025)

J. T. Barron.


A power transform.


arXiv preprint arXiv:2502.10647, 2025.


- Black Forest Labs (2025)

Black Forest Labs.


FLUX.2: Frontier Visual Intelligence.


[https://bfl.ai/blog/flux-2](https://bfl.ai/blog/flux-2), 2025.


- Bochkovskii et al. (2024)

A. Bochkovskii, A. Delaunoy, H. Germain, M. Santos, Y. Zhou, S. R. Richter, and V. Koltun.


Depth pro: Sharp monocular metric depth in less than a second.


arXiv preprint arXiv:2410.02073, 2024.


- Brown et al. (2020)

T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al.


Language models are few-shot learners.


Advances in neural information processing systems, 33:1877–1901, 2020.


- ByteDance (2026)

ByteDance.


Seedance 2.0.


[https://seed.bytedance.com/en/seedance2_0/](https://seed.bytedance.com/en/seedance2_0/), 2026.


Accessed: 2026-03-18.


- Cabon et al. (2020)

Y. Cabon, N. Murray, and M. Humenberger.


Virtual kitti 2.


arXiv preprint arXiv:2001.10773, 2020.


- Caesar et al. (2020)

H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom.


nuscenes: A multimodal dataset for autonomous driving.


In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020.


- Cai et al. (2025)

Z. Cai, C.-F. Yeh, H. Xu, Z. Liu, G. Meyer, X. Lei, C. Zhao, S.-W. Li, V. Chandra, and Y. Shi.


Depthlm: Metric depth from vision language models.


arXiv preprint arXiv:2509.25413, 2025.


- Cao et al. (2026)

B. Cao, K. Chen, K.-K. Maninis, K. Chen, A. Karpur, Y. Xia, S. Dua, T. Dabral, G. Han, B. Han, et al.


Tipsv2: Advancing vision-language pretraining with enhanced patch-text alignment.


arXiv preprint arXiv:2604.12012, 2026.


- Carion et al. (2025)

N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali, K. V. Alwala, H. Khedr, A. Huang, et al.


Sam 3: Segment anything with concepts.


arXiv preprint arXiv:2511.16719, 2025.


- Caron et al. (2021)

M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin.


Emerging properties in self-supervised vision transformers.


In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.


- Chen et al. (2020a)

M. Chen, A. Radford, R. Child, J. Wu, H. Jun, D. Luan, and I. Sutskever.


Generative pretraining from pixels.


In International conference on machine learning, pages 1691–1703. PMLR, 2020a.


- Chen et al. (2020b)

T. Chen, S. Kornblith, M. Norouzi, and G. Hinton.


A simple framework for contrastive learning of visual representations.


In International conference on machine learning, pages 1597–1607. PmLR, 2020b.


- Chen et al. (2020c)

X. Chen, H. Fan, R. Girshick, and K. He.


Improved baselines with momentum contrastive learning.


arXiv preprint arXiv:2003.04297, 2020c.


- Chen et al. (2024)

X. Chen, Z. Liu, S. Xie, and K. He.


Deconstructing denoising diffusion models for self-supervised learning.


arXiv preprint arXiv:2401.14404, 2024.


- Chowdhery et al. (2023)

A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, et al.


Palm: Scaling language modeling with pathways.


Journal of machine learning research, 24(240):1–113, 2023.


- Cordts et al. (2016)

M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele.


The cityscapes dataset for semantic urban scene understanding.


In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016.


- Dai et al. (2017)

A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner.


Scannet: Richly-annotated 3d reconstructions of indoor scenes.


In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017.


- Dehghani et al. (2023)

M. Dehghani, J. Djolonga, B. Mustafa, P. Padlewski, J. Heek, J. Gilmer, A. P. Steiner, M. Caron, R. Geirhos, I. Alabdulmohsin, et al.


Scaling vision transformers to 22 billion parameters.


In International conference on machine learning, pages 7480–7512. PMLR, 2023.


- Dosovitskiy et al. (2020)

A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al.


An image is worth 16x16 words: Transformers for image recognition at scale.


arXiv preprint arXiv:2010.11929, 2020.


- Eigen et al. (2014)

D. Eigen, C. Puhrsch, and R. Fergus.


Depth map prediction from a single image using a multi-scale deep network.


In Advances in Neural Information Processing Systems (NeurIPS), volume 27, 2014.


- Fu et al. (2025)

Y. Fu, M. Lou, and Y. Yu.


Segman: Omni-scale context modeling with state space models and local attention for semantic segmentation.


In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19077–19087, 2025.


- Gemini Team (2025)

Gemini Team.


Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities.


arXiv preprint, 2025.


- Google (2025a)

Google.


Introducing nano banana pro.


[https://blog.google/innovation-and-ai/products/nano-banana-pro/](https://blog.google/innovation-and-ai/products/nano-banana-pro/), 2025a.


Accessed: 2026-03-15.


- Google (2025b)

Google.


Veo 3 announcement.


[https://blog.google/innovation-and-ai/products/generative-media-models-io-2025/](https://blog.google/innovation-and-ai/products/generative-media-models-io-2025/), 2025b.


Accessed: 2026-03-15.


- Grill et al. (2020)

J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. Richemond, E. Buchatskaya, C. Doersch, B. Avila Pires, Z. Guo, M. Gheshlaghi Azar, et al.


Bootstrap your own latent-a new approach to self-supervised learning.


Advances in neural information processing systems, 33:21271–21284, 2020.


- He et al. (2024)

J. He, H. Li, W. Yin, Y. Liang, L. Li, K. Zhou, H. Zhang, B. Liu, and Y.-C. Chen.


Lotus: Diffusion-based visual foundation model for high-quality dense prediction.


arXiv preprint arXiv:2409.18124, 2024.


- He et al. (2025)

J. He, H. Li, M. Sheng, and Y.-C. Chen.


Lotus-2: Advancing geometric dense prediction with powerful image generative model.


arXiv preprint arXiv:2512.01030, 2025.


- He et al. (2020)

K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick.


Momentum contrast for unsupervised visual representation learning.


In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020.


- He et al. (2022)

K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick.


Masked autoencoders are scalable vision learners.


In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000–16009, 2022.


- Hu et al. (2024)

M. Hu, W. Yin, C. Zhang, Z. Cai, X. Long, H. Chen, K. Wang, G. Yu, C. Shen, and S. Shen.


Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation.


IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.


- Kang et al. (2025)

S. Kang, J. Kim, J. Kim, and S. J. Hwang.


Your large vision-language model only needs a few attention heads for visual grounding.


In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9339–9350, 2025.


- Kazemzadeh et al. (2014)

S. Kazemzadeh, V. Ordonez, M. Matten, and T. Berg.


Referitgame: Referring to objects in photographs of natural scenes.


In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787–798, 2014.


- Ke et al. (2024)

B. Ke, A. Obukhov, S. Huang, N. Metzger, R. C. Daudt, and K. Schindler.


Repurposing diffusion-based image generators for monocular depth estimation.


In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9492–9502, 2024.


- Kirillov et al. (2023)

A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, and R. Girshick.


Segment anything.


In IEEE/CVF International Conference on Computer Vision (ICCV), pages 4015–4026, 2023.


- Koch et al. (2018)

T. Koch, L. Liebel, F. Fraundorfer, and M. Korner.


Evaluation of cnn-based single-image depth estimation methods.


In Proceedings of the European Conference on Computer Vision (ECCV) Workshops, pages 0–0, 2018.


- Krizhevsky et al. (2012)

A. Krizhevsky, I. Sutskever, and G. E. Hinton.


Imagenet classification with deep convolutional neural networks.


Advances in neural information processing systems, 25, 2012.


- Lai et al. (2024)

X. Lai, Z. Tian, Y. Chen, Y. Li, Y. Yuan, S. Liu, and J. Jia.


Lisa: Reasoning segmentation via large language model.


In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9579–9589, 2024.


- Li et al. (2024)

B. Li, Z. Lin, D. Pathak, J. Li, Y. Fei, K. Wu, T. Ling, X. Xia, P. Zhang, G. Neubig, et al.


Genai-bench: Evaluating and improving compositional text-to-visual generation.


arXiv preprint arXiv:2406.13743, 2024.


- Lin et al. (2025)

H. Lin, S. Chen, J. Liew, D. Y. Chen, Z. Li, G. Shi, J. Feng, and B. Kang.


Depth anything 3: Recovering the visual space from any views.


arXiv preprint arXiv:2511.10647, 2025.


- Liu and Li (2025)

T. Liu and S. Li.


Hybrid global-local representation with augmented spatial guidance for zero-shot referring image segmentation.


In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 29634–29643, 2025.


- Liu et al. (2025)

Y. Liu, B. Peng, Z. Zhong, Z. Yue, F. Lu, B. Yu, and J. Jia.


Seg-zero: Reasoning-chain guided segmentation via cognitive reinforcement.


arXiv preprint arXiv:2503.06520, 2025.


- Lu et al. (2025)

Y. Lu, J. Cao, Y. Wu, B. Li, L. Tang, Y. Ji, C. Wu, J. Wu, and W. Zhu.


Rsvp: Reasoning segmentation via visual prompting and multi-modal chain-of-thought.


In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14699–14716, 2025.


- Luma (2026)

Luma.


UNI-1.


[https://lumalabs.ai/uni-1/](https://lumalabs.ai/uni-1/), 2026.


Accessed: 2026-03-19.


- Minderer et al. (2023)

M. Minderer, A. Gritsenko, and N. Houlsby.


Scaling open-vocabulary object detection.


Advances in Neural Information Processing Systems, 36:72983–73007, 2023.


- OpenAI (2026)

OpenAI.


GPT-Image-1.5.


[https://openai.com/index/new-chatgpt-images-is-here/](https://openai.com/index/new-chatgpt-images-is-here/), 2026.


Accessed: 2026-03-19.


- Oquab et al. (2023)

M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al.


Dinov2: Learning robust visual features without supervision.


arXiv preprint arXiv:2304.07193, 2023.


- Ouyang et al. (2022)

L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al.


Training language models to follow instructions with human feedback.


Advances in neural information processing systems, 35:27730–27744, 2022.


- Piccinelli et al. (2025a)

L. Piccinelli, C. Sakaridis, M. Segu, Y.-H. Yang, S. Li, W. Abbeloos, and L. Van Gool.


Unik3d: Universal camera monocular 3d estimation.


In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1028–1039, 2025a.


- Piccinelli et al. (2025b)

L. Piccinelli, C. Sakaridis, Y.-H. Yang, M. Segu, S. Li, W. Abbeloos, and L. Van Gool.


Unidepthv2: Universal monocular metric depth estimation made simpler.


IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025b.


- Radford et al. (2018)

A. Radford, K. Narasimhan, T. Salimans, I. Sutskever, et al.


Improving language understanding by generative pre-training.


2018.


- Radford et al. (2019)

A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al.


Language models are unsupervised multitask learners.


OpenAI blog, 1(8):9, 2019.


- Radford et al. (2021)

A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al.


Learning transferable visual models from natural language supervision.


In International conference on machine learning, pages 8748–8763. PmLR, 2021.


- Ravi et al. (2024)

N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. Dollár, and C. Feichtenhofer.


Sam 2: Segment anything in images and videos.


arXiv preprint arXiv:2408.00714, 2024.


- Ren et al. (2024)

T. Ren, Y. Chen, Q. Jiang, Z. Zeng, Y. Xiong, W. Liu, Z. Ma, J. Shen, Y. Gao, X. Jiang, et al.


Dino-x: A unified vision model for open-world object detection and understanding.


arXiv preprint arXiv:2411.14347, 2024.


- Schops et al. (2019)

T. Schops, T. Sattler, and M. Pollefeys.


Bad slam: Bundle adjusted direct rgb-d slam.


In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 134–144, 2019.


- Shen et al. (2024)

Y. Shen, C. Fu, P. Chen, M. Zhang, K. Li, X. Sun, Y. Wu, S. Lin, and R. Ji.


Aligning and prompting everything all at once for universal visual perception.


In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13193–13203, 2024.


- Silberman et al. (2012)

N. Silberman, D. Hoiem, P. Kohli, and R. Fergus.


Indoor segmentation and support inference from rgbd images.


In European conference on computer vision, pages 746–760. Springer, 2012.


- Siméoni et al. (2025)

O. Siméoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, et al.


Dinov3.


arXiv preprint arXiv:2508.10104, 2025.


- Tschannen et al. (2025)

M. Tschannen, A. Gritsenko, X. Wang, M. F. Naeem, I. Alabdulmohsin, N. Parthasarathy, T. Evans, L. Beyer, Y. Xia, B. Mustafa, et al.


Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features.


arXiv preprint arXiv:2502.14786, 2025.


- Uhrig et al. (2017)

J. Uhrig, N. Schneider, L. Schneider, U. Franke, T. Brox, and A. Geiger.


Sparsity invariant cnns.


In International Conference on 3D Vision (3DV), 2017.


- Vasiljevic et al. (2019)

I. Vasiljevic, N. Kolkin, S. Zhang, R. Luo, H. Wang, F. Z. Dai, A. F. Daniele, M. Mostajabi, S. Basart, M. R. Walter, and G. Shakhnarovich.


DIODE: A Dense Indoor and Outdoor DEpth Dataset.


CoRR, abs/1908.00463, 2019.


URL [http://arxiv.org/abs/1908.00463](http://arxiv.org/abs/1908.00463).


- Wang et al. (2026a)

H. Wang, L. Qiao, Z. Jie, Z. Huang, C. Feng, Q. Zheng, L. Ma, X. Lan, and X. Liang.


X-sam: From segment anything to any segmentation.


In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 26187–26196, 2026a.


- Wang et al. (2025a)

J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny.


Vggt: Visual geometry grounded transformer.


In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025a.


- Wang et al. (2026b)

L. Wang, A. Zanfir, E. G. Bazavan, M. Andriluka, and C. Sminchisescu.


Thfm: A unified video foundation model for 4d human perception and beyond.


arXiv preprint arXiv:2603.25892, 2026b.


- Wang et al. (2025b)

R. Wang, S. Xu, C. Dai, J. Xiang, Y. Deng, X. Tong, and J. Yang.


Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision.


In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025b.


- Wang et al. (2025c)

R. Wang, S. Xu, Y. Dong, Y. Deng, J. Xiang, Z. Lv, G. Sun, X. Tong, and J. Yang.


Moge-2: Accurate monocular geometry with metric scale and sharp details.


arXiv preprint arXiv:2507.02546, 2025c.


- Wei et al. (2024)

C. Wei, Y. Zhong, H. Tan, Y. Liu, Z. Zhao, J. Hu, and Y. Yang.


Hyperseg: Towards universal visual segmentation with large language model.


arXiv preprint arXiv:2411.17606, 2024.


- Wei et al. (2021)

J. Wei, M. Bosma, V. Y. Zhao, K. Guu, A. W. Yu, B. Lester, N. Du, A. M. Dai, and Q. V. Le.


Finetuned language models are zero-shot learners.


arXiv preprint arXiv:2109.01652, 2021.


- Wiedemer et al. (2025)

T. Wiedemer, Y. Li, P. Vicol, S. S. Gu, N. Matarese, K. Swersky, B. Kim, P. Jaini, and R. Geirhos.


Video models are zero-shot learners and reasoners.


arXiv preprint arXiv:2509.20328, 2025.


- Yang et al. (2024)

L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao.


Depth anything v2.


Advances in Neural Information Processing Systems, 37:21875–21911, 2024.


- Ye et al. (2024)

C. Ye, L. Qiu, X. Gu, Q. Zuo, Y. Wu, Z. Dong, L. Bo, Y. Xiu, and X. Han.


Stablenormal: Reducing diffusion variance for stable and sharp normal.


ACM Transactions on Graphics (ToG), 43(6):1–18, 2024.


- Ye et al. (2025)

Y. Ye, X. He, Z. Li, B. Lin, S. Yuan, Z. Yan, B. Hou, and L. Yuan.


Imgedit: A unified image editing dataset and benchmark.


arXiv preprint arXiv:2505.20275, 2025.


- Yu et al. (2024)

Q. Yu, P.-T. Jiang, H. Zhang, J. Chen, B. Li, L. Zhang, and H. Lu.


High-precision dichotomous image segmentation via probing diffusion capacity.


arXiv preprint arXiv:2410.10105, 2024.


- Zhai et al. (2023)

X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer.


Sigmoid loss for language image pre-training.


In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.


- Zhang et al. (2025)

C. Zhang, G. L. Moing, S. Koppula, I. Rocco, L. Momeni, J. Xie, S. Sun, R. Sukthankar, J. K. Barral, R. Hadsell, et al.


Efficiently reconstructing dynamic scenes one d4rt at a time.


arXiv preprint arXiv:2512.08924, 2025.


- Zhang et al. (2023)

H. Zhang, F. Li, X. Zou, S. Liu, C. Li, J. Yang, and L. Zhang.


A simple framework for open-vocabulary segmentation and detection.


In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1020–1031, 2023.


- Zhao et al. (2025)

C. Zhao, Y. Sun, M. Liu, H. Zheng, M. Zhu, Z. Zhao, H. Chen, T. He, and C. Shen.


Diception: A generalist diffusion model for visual perceptual tasks.


arXiv preprint arXiv:2502.17157, 2025.


- Zhou et al. (2021)

J. Zhou, C. Wei, H. Wang, W. Shen, C. Xie, A. Yuille, and T. Kong.


ibot: Image bert pre-training with online tokenizer.


arXiv preprint arXiv:2111.07832, 2021.


- Zou et al. (2023)

X. Zou, Z.-Y. Dou, J. Yang, Z. Gan, L. Li, C. Li, X. Dai, H. Behl, J. Wang, L. Yuan, et al.


Generalized decoding for pixel, image, and language.


In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15116–15127, 2023.


- Zuo et al. (2025)

J. Zuo, H. Deng, H. Zhou, J. Zhu, Y. Zhang, Y. Zhang, Y. Yan, K. Huang, W. Chen, Y. Deng, R. Jin, N. Sang, and C. Gao.


Is nano banana pro a low-level vision all-rounder? A comprehensive evaluation on 14 tasks and 40 datasets.


arXiv preprint, 2025.


<a id="source-section-21"></a>

## Appendix - Additional Demonstrations


[图片：Refer to caption]


A ghostly ship sailing on a fog-shrouded, moonlit sea.


[图片：Refer to caption]


[图片：Refer to caption]


A lantern casting dim light in a haunted forest.


[图片：Refer to caption]


[图片：Refer to caption]


A yellow taxi waiting outside a modern glass building.


[图片：Refer to caption]


[图片：Refer to caption]


A samurai with a silk sash in a cherry blossom garden.


[图片：Refer to caption]


Figure 9: Comparing Vision Banana (left) and Nano Banana Pro (right) on text-to-image generation. Prompts sampled from GenAI-Bench (Li et al., [2024](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib42)).
Results verify that Vision Banana does not forget its generative features during the instruction-tuning.


| 列 1 | 列 2 | 列 3 |
| --- | --- | --- |
| Original image | Vision Banana [图片：Refer to caption] | Nano Banana Pro |
| [图片：Refer to caption] | [图片：Refer to caption] | [图片：Refer to caption] |
| Change the grassy hills in the picture to a beach with ocean waves. [colspan=3] | | |
| [图片：Refer to caption] | [图片：Refer to caption] | [图片：Refer to caption] |
| Remove the plant from the shelf, and resize the picture frame to be larger. [colspan=3] | | |
| [图片：Refer to caption] | [图片：Refer to caption] | [图片：Refer to caption] |
| Change the vehicle’s color to red. [colspan=3] | | |
| [图片：Refer to caption] | [图片：Refer to caption] | [图片：Refer to caption] |
| Change the background of the suit from a blank wall to a luxurious office setting that includes<br><br>a wooden desk and a large window showing a cityscape. [colspan=3] | | |


Figure 10: Comparing Vision Banana (left) and Nano Banana Pro (right) on image-editing. Prompts sampled from ImgEdit (Ye et al., [2025](https://ar5iv.labs.arxiv.org/html/2604.20329#bib.bib76)).

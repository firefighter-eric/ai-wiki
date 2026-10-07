# MosaicML - 2023 - MPT-7B

- Source HTML: `raw/html/MosaicML - 2023 - MPT-7B.html`
- Source SHA256: `a1ac9e60855b725ac468bb1398501e1005265de487119a53eb71d60f090c4cc7`
- Source URL: https://www.databricks.com/blog/mpt-7b
- Generated from: `scripts/fetch_web_text.py`
- Extraction: `structured-html-v2` (headings, links, MathML/TeX and tables; figures require visual review)

## Extracted Text

<a id="source-section-0"></a>

[图片：Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs]


Introducing MPT-7B, the first entry in our MosaicML Foundation Series. MPT-7B is a transformer trained from scratch on 1T tokens of text and code. It is open source, available for commercial use, and matches the quality of LLaMA-7B. MPT-7B was trained on the MosaicML platform in 9.5 days with zero human intervention at a cost of ~$200k.


Large language models (LLMs) are changing the world, but for those outside well-resourced industry labs, it can be extremely difficult to train and deploy these models. This has led to a flurry of activity centered on open-source LLMs, such as the LLaMA series from Meta, the Pythia series from EleutherAI, the StableLM series from StabilityAI, and the OpenLLaMA model from Berkeley AI Research.


Today, we at MosaicML are releasing a new model series called [MPT (MosaicML Pretrained Transformer)](https://github.com/mosaicml/llm-foundry) to address the limitations of the above models and finally provide a commercially-usable, open-source model that matches (and - in many ways - surpasses) LLaMA-7B. Now you can train, finetune, and deploy your own private MPT models, either starting from one of our checkpoints or training from scratch. For inspiration, we are also releasing three finetuned models in addition to the base MPT-7B: MPT-7B-Instruct, MPT-7B-Chat, and MPT-7B-StoryWriter-65k+, the last of which uses a context length of 65k tokens!


Our MPT model series is:


- **Licensed for commercial use** (unlike LLaMA).


- **Trained on a large amount of data** (1T tokens like LLaMA vs. 300B for Pythia, 300B for OpenLLaMA, and 800B for StableLM).


- **Prepared to handle extremely long inputs** thanks to [ALiBi](https://arxiv.org/abs/2108.12409) (we trained on up to 65k inputs and can handle up to 84k vs. 2k-4k for other open source models).


- **Optimized for fast training and inference** (via [FlashAttention](https://arxiv.org/abs/2205.14135) and [FasterTransformer](https://github.com/NVIDIA/FasterTransformer))


- **Equipped with highly efficient open-source training code.**


We rigorously evaluated MPT on a range of benchmarks, and MPT met the high quality bar set by LLaMA-7B.


Today, we are releasing the [base MPT](https://huggingface.co/mosaicml/mpt-7b) model and three other finetuned variants that demonstrate the many ways of building on this base model:


<a id="source-section-1"></a>

## MPT-7B Base:


[**MPT-7B Base**](https://huggingface.co/mosaicml/mpt-7b) is a decoder-style transformer with 6.7B parameters. It was trained on 1T tokens of text and code that was curated by MosaicML's data team. This base model includes FlashAttention for fast training and inference and ALiBi for finetuning and extrapolation to long context lengths.


- **License: Apache-2.0**


- **HuggingFace Link: [https://huggingface.co/mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b)**


<a id="source-section-2"></a>

## MPT-7B-StoryWriter-65k+


[**MPT-7B-StoryWriter-65k+**](https://huggingface.co/mosaicml/mpt-7b-storywriter) is a model designed to read and write stories with super long context lengths. It was built by finetuning MPT-7B with **a context length of 65k tokens** on a filtered fiction subset of the [books3 dataset](https://huggingface.co/datasets/the_pile_books3). At inference time, thanks to ALiBi, MPT-7B-StoryWriter-65k+ can extrapolate even beyond 65k tokens, and we have demonstrated **generations as long as 84k tokens** on a single node of A100-80GB GPUs.


- **License: Apache-2.0**


- **HuggingFace Link: [https://huggingface.co/mosaicml/mpt-7b-storywriter](https://huggingface.co/mosaicml/mpt-7b-storywriter)**


<a id="source-section-3"></a>

## MPT-7B-Instruct


[**MPT-7B-Instruct**](https://huggingface.co/mosaicml/mpt-7b-instruct) is a model for short-form instruction following. Built by finetuning MPT-7B on a [dataset](https://huggingface.co/datasets/sam-mosaic/dolly_hhrlhf) we also release, derived from [Databricks Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) and [Anthropic's Helpful and Harmless](https://huggingface.co/datasets/Anthropic/hh-rlhf) datasets.


- **License: CC-By-SA-3.0**


- **HuggingFace Link: [https://huggingface.co/mosaicml/mpt-7b-instruct](https://huggingface.co/mosaicml/mpt-7b-instruct)**


<a id="source-section-4"></a>

## MPT-7B-Chat


[**MPT-7B-Chat**](https://huggingface.co/mosaicml/mpt-7b-chat) is a chatbot-like model for dialogue generation. Built by finetuning MPT-7B on the ShareGPT-Vicuna, [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3), [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca), [Helpful and Harmless](https://huggingface.co/datasets/Anthropic/hh-rlhf), and [Evol-Instruct](https://huggingface.co/datasets/victor123/evol_instruct_70k) datasets.


- **License: CC-By-NC-SA-4.0 (non-commercial use only)**


- **HuggingFace Link: [https://huggingface.co/mosaicml/mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat)**


We hope businesses and the open-source community will build on this effort: alongside the model checkpoints, we have open-sourced the entire codebase for pretraining, finetuning, and evaluating MPT [via our new MosaicML LLM Foundry!](https://github.com/mosaicml/llm-foundry)


This release is more than just a model checkpoint: it's an entire framework for building great LLMs with MosaicML's usual emphasis on efficiency, ease-of-use, and rigorous attention to detail. These models were built by MosaicML's NLP team on the [MosaicML platform](https://www.mosaicml.com/training) with the exact same tools our customers use (just ask our customers, like [Replit](https://huggingface.co/replit/replit-code-v1-3b)!).


e trained MPT-7B with ZERO human intervention from start to finish: over 9.5 days on 440 GPUs, the MosaicML platform detected and addressed 4 hardware failures and resumed the training run automatically, and - due to architecture and optimization improvements we made - there were no catastrophic loss spikes. Check out our [**empty training logbook for MPT-7B!**](https://www.mosaicml.com/blog/mpt-7b#training-stability)


<a id="source-section-5"></a>

## Training and Deploying Your Own Custom MPT


If you'd like to start [building](https://www.mosaicml.com/blog/train-custom-gpt-diffusion-models) and deploying your own custom MPT models on the MosaicML platform, [sign up here](https://forms.mosaicml.com/demo?utm_source=mosaicml-blog&utm_medium=mosaicml.com&utm_campaign=mpt-7b) to get started.


For more engineering details on data, training, and inference, skip ahead to [the section below](https://www.mosaicml.com/blog/mpt-7b#building-with-mosaicml-platform).


For more information about our four new models, read on!


<a id="source-section-6"></a>

## Introducing the Mosaic Pretrained Transformers (MPT)


MPT models are GPT-style decoder-only transformers with several improvements: performance-optimized layer implementations, architecture changes that provide greater training stability, and the elimination of context length limits by replacing positional embeddings with ALiBi. Thanks to these modifications, customers can train MPT models with efficiency [(40-60% MFU)](https://github.com/mosaicml/llm-foundry/tree/main/scripts/train/benchmarking) without diverging from loss spikes and can [serve MPT models](https://www.mosaicml.com/inference) with both standard HuggingFace pipelines and [FasterTransformer](https://github.com/NVIDIA/FasterTransformer).


<a id="source-section-7"></a>

## MPT-7B (Base Model)


[MPT-7B](https://huggingface.co/mosaicml/mpt-7b) matches the quality of LLaMA-7B and outperforms other open source 7B - 20B models on standard academic tasks. To evaluate model quality, we compiled 11 open-source benchmarks commonly used for in-context learning (ICL) and formatted and evaluated them in an industry-standard manner. We also added our own self-curated Jeopardy benchmark to evaluate the model's ability to produce factually correct answers to challenging questions.


See **Table 1** for a comparison of zero-shot performance between MPT and other models:


[[图片：Zero-shot Accuracy]](https://www.databricks.com/sites/default/files/inline-images/mpt-commercially-usable-llms-img-2.jpg)


Table 1 - Zero-shot accuracy of MPT-7B vs. LLaMA-7B vs. other open source models on academic tasks. MPT-7B and LLaMA-7B have similar quality across all tasks, and each model scores highest (indicated in red) on 6 out of 12 tasks. Both models outperform other open source language models, even models with much larger parameter counts. All data was measured using the MosaicML LLM Foundry's in-context-learning (ICL) evaluation framework on checkpoints for each model. To ensure fair comparisons between models, no prompt strings or prompt tuning was used.


To ensure apples-to-apples comparisons, we fully re-evaluated each model: the model checkpoint was run through our open source [LLM Foundry eval framework](https://github.com/mosaicml/llm-foundry/tree/main/scripts/eval) with the same (empty) prompt strings and no model-specific prompt tuning. For full details on the evaluation, see the [**Appendix**](https://www.mosaicml.com/blog/mpt-7b#appendix-eval). In previous benchmarks, our setup is [8x faster than other eval frameworks](https://www.mosaicml.com/blog/llm-evaluation-for-icl) on a single GPU and seamlessly achieves linear scaling with multiple GPUs. Built-in support for FSDP makes it possible to evaluate large models and use larger batch sizes for further acceleration.


We invite the community to use our evaluation suite for their own model evaluations and to submit pull requests with additional datasets and ICL task types so we can ensure the most rigorous possible evaluation.


<a id="source-section-8"></a>

## MPT-7B-StoryWriter-65k+


Most open-source language models can only handle sequences with up to a few thousand tokens (see **Figure 1**). But with the MosaicML platform and a single node of 8xA100-80GB, you can easily finetune MPT-7B to handle context lengths up to 65k! The ability to handle such extreme context length adaptation comes from [ALiBi](https://arxiv.org/abs/2108.12409), one of the key architectural choices in MPT-7B.


To show off this capability and to get you thinking about what you could do with a 65k context window, we are releasing MPT-7B-StoryWriter-65k+. StoryWriter was finetuned from MPT-7B for 2500 steps on 65k-token excerpts of fiction books contained in the books3 corpus. Like pretraining, this finetuning process used a next-token-prediction objective. Once we prepared the data, all that was needed for training was Composer with FSDP, activation checkpointing, and a microbatch size of 1.


As it turns out, the full text of The Great Gatsby weighs in at just under 68k tokens. So, naturally, we had StoryWriter read The Great Gatsby and generate an epilogue. One of the epilogues we generated is in **Figure 2**. StoryWriter took in The Great Gatsby in about 20 seconds (about 150k words-per-minute). Due to the long sequence length, its "typing" speed is slower than our other MPT-7B models, about 105 words-per-minute.


Even though StoryWriter was fine-tuned with a 65k context length, ALiBi makes it possible for the model to extrapolate to even longer inputs than it was trained on: 68k tokens in the case of The Great Gatsby, and up to 84k tokens in our testing.


[[图片：Training Context Length]](https://www.databricks.com/sites/default/files/inline-images/mpt-commercially-usable-llms-img-3.jpg)


Figure 1 - Training context length of MPT-7B-StoryWriter-65k+ vs. other models.


The longest context length of any other open-source model is 4k. GPT-4 has a context length of 8k, and another variant of the model has a context length of 32k.


[[图片：MPT-7B-StoryWriter-65k]](https://www.databricks.com/sites/default/files/inline-images/mpt-commercially-usable-llms-img-4.jpg)


Figure 2 - MPT-7B-StoryWriter-65k+ writes an epilogue to The Great Gatsby.


The epilogue results from providing the entire text of The Great Gatsby (about 68k tokens) as input to the model followed by the word "Epilogue" and allowing the model to continue generating from there.


<a id="source-section-9"></a>

## MPT-7B-Instruct


[[图片：An interaction with MPT-7B-Instruct]](https://www.databricks.com/sites/default/files/inline-images/mpt-commercially-usable-llms-img-5.jpg)


Figure 3 - An interaction with MPT-7B-Instruct.


The model properly converts content formatted as YAML into the same content formatted as JSON.


LLM pretraining teaches the model to continue generating text based on the input it was provided. But in practice, we expect LLMs to treat the input as instructions to follow. Instruction finetuning is the process of training LLMs to perform instruction-following in this way. By reducing the reliance on clever prompt engineering, instruction finetuning makes LLMs more accessible, intuitive, and immediately usable. The progress of instruction finetuning has been driven by open-source datasets like FLAN, Alpaca, and the Dolly-15k dataset.


We created a commercially-usable instruction-following variant of our model called [MPT-7B-Instruct](https://huggingface.co/mosaicml/mpt-7b-instruct). We liked the commercial license of Dolly, but wanted more data, so we augmented Dolly with a subset of Anthropic's Helpful & Harmless dataset, quadrupling the dataset size while maintaining a commercial license.


This new aggregate dataset, released [here](https://huggingface.co/datasets/mosaicml/dolly_hhrlhf), was used to finetune MPT-7B, resulting in MPT-7B-Instruct, which is commercially usable. Anecdotally, we find MPT-7B-Instruct to be an effective instruction-follower. (See **Figure 3** for an example interaction.) With its extensive training on 1 trillion tokens, MPT-7B-Instruct should be competitive with the larger dolly-v2-12b, whose base model, Pythia-12B, was only trained on 300 billion tokens.


We are releasing the [code](https://github.com/mosaicml/llm-foundry/tree/main/llmfoundry), [weights](https://huggingface.co/mosaicml/mpt-7b-instruct), and an [online demo](https://huggingface.co/spaces/mosaicml/mpt-7b-instruct) of MPT-7B-Instruct. We hope that the small size, competitive performance, and commercial license of MPT-7B-Instruct will make it immediately valuable to the community.


<a id="source-section-10"></a>

## MPT-7B-Chat


[[图片：An interaction with MPT-7B-Chat]](https://www.databricks.com/sites/default/files/inline-images/mpt-commercially-usable-llms-img-6.jpg)


Figure 4 - An interaction with MPT-7B-Chat.


A multi-turn conversation with the chat model in which it suggests high-level approaches to solving a problem (using AI to protect endangered wildlife) and then proposes an implementation of one of them in Python using Keras.


We have also developed [MPT-7B-Chat](https://huggingface.co/mosaicml/mpt-7b-chat), a conversational version of MPT-7B. MPT-7B-Chat has been finetuned using ShareGPT-Vicuna, [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3), [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca), [Helpful and Harmless](https://huggingface.co/datasets/Anthropic/hh-rlhf), and [Evol-Instruct](https://huggingface.co/datasets/victor123/evol_instruct_70k), ensuring that it is well-equipped for a wide array of conversational tasks and applications. It uses the ChatML format, which provides a convenient and standardized way to pass the model system messages and helps prevent malicious prompt injection.


While [MPT-7B-Instruct](https://huggingface.co/mosaicml/mpt-7b-instruct) focuses on delivering a more natural and intuitive interface for instruction-following, MPT-7B-Chat aims to provide seamless, engaging multi-turn interactions for users. (See **Figure 4** for an example interaction.)


As with MPT-7B and MPT-7B-Instruct, we are releasing the [code](https://github.com/mosaicml/llm-foundry/tree/main/llmfoundry), [weights](https://huggingface.co/mosaicml/mpt-7b-chat), and an [online demo](https://huggingface.co/spaces/mosaicml/mpt-7b-chat) for MPT-7B-Chat.


<a id="source-section-11"></a>

## How we built these models on the MosaicML platform


The models released today were built by the MosaicML NLP team, but the tools we used are the exact same ones available to every customer of MosaicML.


Think of MPT-7B as a demonstration – our small team was able to build these models in only a few weeks, including the data preparation, training, finetuning, and deployment (and writing this blog!). Let's take a look at the process of building MPT-7B with MosaicML:


<a id="source-section-12"></a>

## Data


We wanted MPT-7B to be a high-quality standalone model and a useful jumping off point for diverse downstream uses. Accordingly, our pretraining data came from a MosaicML-curated mix of sources, which we summarize in **Table 2** and describe in detail in the [**Appendix**](https://www.mosaicml.com/blog/mpt-7b#appendix-data). Text was tokenized using the [EleutherAI GPT-NeoX-20B tokenizer](https://huggingface.co/docs/transformers/model_doc/gpt_neox) and the model was pretrained on 1 trillion tokens. This dataset emphasizes English natural language text and diversity for future uses (e.g., code or scientific models), and includes elements of the recently-released [RedPajama dataset](https://github.com/togethercomputer/RedPajama-Data) so that the web crawl and Wikipedia portions of the dataset contain up-to-date information from 2023.


[[图片：Data mix for MPT-7B pretraining]](https://www.databricks.com/sites/default/files/inline-images/mpt-commercially-usable-llms-img-7.jpg)


Table 2 - Data mix for MPT-7B pretraining.


A mix of data from ten different open-source text corpora. Text was tokenized using the EleutherAI GPT-NeoX-20B tokenizer, and the model was pre-trained on 1T tokens sampled according to this mix.


<a id="source-section-13"></a>

## Tokenizer


We used [EleutherAI's GPT-NeoX 20B tokenizer](https://aclanthology.org/2022.bigscience-1.9/). This BPE tokenizer has a number of desirable characteristics, most of which are relevant for tokenizing code:


- Trained on a diverse mix of data that includes code ([The Pile](https://pile.eleuther.ai/))


- Applies consistent space delimitation, unlike the GPT2 tokenizer which tokenizes inconsistently depending on the presence of prefix spaces


- Contains tokens for repeated space characters, which allows superior compression of text with large amounts of repeated space characters.


The tokenizer has a vocabulary size of 50257, but we set the model vocabulary size to 50432. The reasons for this were twofold: First, to make it a multiple of 128 (as in [Shoeybi et al.](https://arxiv.org/abs/1909.08053)), which [we found improved MFU by up to four percentage points in initial experiments](https://twitter.com/karpathy/status/1621578354024677377?s=20). Second, to leave tokens available that can be used in subsequent [UL2 training](https://arxiv.org/abs/2205.05131).


<a id="source-section-14"></a>

## Efficient Data Streaming


We leveraged [MosaicML's StreamingDataset](https://github.com/mosaicml/streaming) to host our data in a standard cloud object store and efficiently stream it to our compute cluster during training. StreamingDataset provides a number of advantages:


- Obviates the need to download the whole dataset before starting training.


- Allows instant resumption of training from any point in the dataset. A paused run can be resumed without fast-forwarding the dataloader from the start.


- Is fully deterministic. Samples are read in the same order regardless of the number of GPUs, nodes, or CPU workers.


- Allows arbitrary mixing of data sources in: simply enumerate the your data sources and desired proportions of the total training data, and StreamingDataset handles the rest. This made it extremely easy to run preparatory experiments on different data mixes.


Check out the [StreamingDataset blog](https://www.mosaicml.com/blog/mosaicml-streamingdataset) for more details!


<a id="source-section-15"></a>

## Training Compute


All MPT-7B models were trained on the [MosaicML platform](https://www.mosaicml.com/training) with the following tools:


- **Compute:** A100-40GB and A100-80GB GPUs from Oracle Cloud


- **Orchestration and Fault Tolerance:** [MCLI](https://docs.mosaicml.com/projects/mcli/en/latest/) and [MosaicML platform](https://www.mosaicml.com/training)


- **Data:** OCI Object Storage and [StreamingDataset](https://github.com/mosaicml/streaming)


- **Training software:** Composer, PyTorch FSDP, and [LLM Foundry](https://github.com/mosaicml/llm-foundry)


As shown in **Table 3**, nearly all of the training budget was spent on the base MPT-7B model, which took ~9.5 days to train on 440xA100-40GB GPUs, and cost ~$200k. The finetuned models took much less compute and were much cheaper – ranging between a few hundred and few thousand dollars each.


[[图片：Training details for each of our MPT-7B models]](https://www.databricks.com/sites/default/files/inline-images/mpt-commercially-usable-llms-img-8.jpg)


Table 3 - Training details for each of our MPT-7B models.


Time to Train' is the total runtime from job start to finish, including checkpointing, periodic evaluation, restarts, etc. 'Cost' is computed with pricing of $2/A100-40GB/hr and $2.50/A100-80GB/hr for reserved GPUs on the MosaicML platform.


Each of these training recipes can be fully customized. For example, if you'd like to start from our open source MPT-7B and finetune it on proprietary data with a long context length, you can do that today on the MosaicML platform.


As another example, to train a new model from scratch on a custom domain (e.g. on [biomedical text](https://crfm.stanford.edu/2022/12/15/biomedlm.html) or [code](https://huggingface.co/replit/replit-code-v1-3b)), simply reserve short-term large blocks of compute with MosaicML's **hero cluster** offering. Just pick the desired model size and token budget, upload your data to an object store like S3, and launch an MCLI job. You will have your very own custom LLM in just days!


Check out [our earlier LLM blog post](https://www.mosaicml.com/blog/gpt-3-quality-for-500k) for guidance on the times and costs to train different LLMs. Find the latest throughput data for specific model configurations here. In line with our previous work, all MPT-7B models were trained with Pytorch FullyShardedDataParallelism (FSDP) and without tensor- or pipeline- parallelism.


<a id="source-section-16"></a>

## Training Stability


As many teams have documented, training LLMs with billions of parameters on hundreds-to-thousands of GPUs is incredibly challenging. Hardware will fail frequently and in creative and unexpected ways. Loss spikes will derail training. Teams must "babysit" the training run 24/7 in case of failures and apply manual interventions when things go wrong. Check out the [OPT logbook](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf) for a candid example of the many perils awaiting anyone training an LLM.


At MosaicML, our research and engineering teams have worked tirelessly over the last 6 months to eliminate these issues. As a result, our MPT-7B training logbook (**Figure 5**) is very boring! We trained MPT-7B on 1 trillion tokens from start to finish with no human intervention. No loss spikes, no mid-stream learning rate changes, no data skipping, automatic handling of dead GPUs, etc.


[[图片：The (very uneventful) MPT-7B Training Logbook]](https://www.databricks.com/sites/default/files/inline-images/mpt-commercially-usable-llms-img-9.jpg)


Figure 5 - The (very uneventful) MPT-7B Training Logbook.


MPT-7B was trained on 1T tokens over the course of 9.5 days on 440xA100-40GB. During that time the training job encountered 4 hardware failures, all of which were detected by the MosaicML platform. The run was automatically paused and resumed upon each failure, and no human intervention was required.


[[图片：The loss curve over time, highlighting hardware failures and automatic recoveries]](https://www.databricks.com/sites/default/files/inline-images/mpt-commercially-usable-llms-img-10.jpg)


Figure 6 - The loss curve over time, highlighting hardware failures and automatic recoveries.


If hardware failures occur while a job is running, the MosaicML platform automatically detects the failure, pauses the job, cordons any broken nodes, and resumes the job. During the MPT-7B training run, we encountered 4 such failures, and each time the job was automatically resumed


How did we do this? First, we addressed convergence stability with architecture and optimization improvements. Our MPT models use [ALiBi](https://arxiv.org/abs/2108.12409) rather than positional embeddings, which we found to improve resilience to loss spikes. We also train our MPT models with [the Lion optimizer](https://arxiv.org/abs/2302.06675) rather than AdamW, which provides stable update magnitudes and cuts optimizer state memory in half.


Second, we used the MosaicML platform's NodeDoctor feature to monitor for and resolve hardware failures and the JobMonitor feature to resume runs after these failures were resolved. These features enabled us to train MPT-7B with no human intervention from start to finish despite 4 hardware failures during the run. See **Figure 6** for a closeup view of what autoresumption looks like on the MosaicML platform.


<a id="source-section-17"></a>

## Inference


MPT is designed to be fast, easy, and cheap to deploy for inference. To begin with, all MPT models are subclassed from the [HuggingFace PretrainedModel base class](https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/model#transformers.PreTrainedModel), which means that they are fully compatible with the HuggingFace ecosystem. You can upload MPT models to the HuggingFace Hub, generate outputs with standard pipelines like `model.generate(...)`, build HuggingFace Spaces ([see some of ours here!](https://huggingface.co/mosaicml#spaces)), and more.


What about performance? With MPT's optimized layers (including [FlashAttention](https://arxiv.org/abs/2205.14135) and low precision layernorm), the out-of-the-box performance of MPT-7B when using `model.generate(...)` is **1.5x-2x faster** than other 7B models like LLaMa-7B. This makes it easy to build fast and flexible inference pipelines with just HuggingFace and PyTorch.


But what if you **really** need the best performance? In that case, directly port MPT weights to FasterTransformer or ONNX. Check out the [LLM Foundry's inference folder](https://github.com/mosaicml/llm-foundry/tree/main/scripts/inference) for scripts and instructions.


Finally, for the best hosting experience, deploy your MPT models directly on [MosaicML's Inference service](https://www.mosaicml.com/inference). Start with our managed endpoints for models like MPT-7B-Instruct, and/or deploy your own custom model endpoints for optimal cost and data privacy.


<a id="source-section-18"></a>

## What's Next?


This MPT-7B release is the culmination of two years of work at MosaicML building and battle-testing open-source software ([Composer](https://github.com/mosaicml/composer), [StreamingDataset](https://github.com/mosaicml/streaming), [LLM Foundry](https://github.com/mosaicml/llm-foundry)) and proprietary infrastructure (MosaicML [Training](https://www.mosaicml.com/training) and [Inference](https://www.mosaicml.com/inference)) that makes it possible for customers to train LLMs on any compute provider, with any data source, with efficiency, privacy and cost transparency - and to have things go right the first time.


We believe MPT, the MosaicML LLM Foundry, and the MosaicML platform are the best starting point for building custom LLMs for private, commercial, and community use, whether you want to finetune our checkpoints or train your own from scratch. We look forward to seeing how the community builds on these tools and artifacts.


Importantly, today's MPT-7B models are just the beginning! To help our customers address more challenging tasks and continually improve their products, MosaicML will continue to produce foundation models of higher and higher quality. Exciting follow-on models are already training. Expect to hear more about them soon!


<a id="source-section-19"></a>

## Acknowledgements


We are grateful to our friends at AI2 for helping us to curate our pretraining dataset, choose a great tokenizer, and for many other helpful conversations along the way ⚔️


<a id="source-section-20"></a>

## Appendix


<a id="source-section-21"></a>

### Data


<a id="source-section-22"></a>

### mC4


[Multilingual C4 (mC4) 3.1.0](https://huggingface.co/datasets/allenai/c4/blob/mC4_3.1.0/README.md) is an update of the original [mC4](https://arxiv.org/abs/2010.11934) by [Chung et al.](https://openreview.net/forum?id=kXwdL1cWOAi), which contains sources through August 2022. We selected the English subset, and then applied the following filtering criteria to each document:


- The most common character must be alphabetic.


- ≥ 92% of characters must be alphanumeric.


- If the document is > 500 words, the most common word cannot constitute > 7.5% of the total word count; If the document is ≤ 500 words, the most common word cannot constitute > 30% of the total word count.


- The document must be ≥ 200 words and ≤ 50000 words.


The first three filtering criteria were used to improve sample quality, and the final filtering criterion (documents must be ≥200 words and ≤50000 words) was used to increase the mean sequence length of the pretraining data.


mC4 was released as part of the continued effort from [Dodge et al.](https://arxiv.org/abs/2104.08758).


<a id="source-section-23"></a>

### C4


[Colossal Cleaned Common Crawl (C4)](https://www.databricks.com/blog/mpt-7b#https:/arxiv.org/abs/1910.10683) is an English Common Crawl corpus introduced by Raffel et al.. We applied [Abbas et al.'s Semantic Deduplication](https://arxiv.org/abs/2303.09540) process to remove the 20% most similar documents within C4, as internal experiments showed that this is a [Pareto improvement](https://assets-global.website-files.com/61fd4eb76a8d78bc0676b47d/6206b5e01213364eee74d69b_Mosaic_Methodology_Blog_Post.pdf) for models trained on C4.


<a id="source-section-24"></a>

### RedPajama


We included a number of subsets of the [RedPajama dataset](https://github.com/togethercomputer/RedPajama-Data), which is [Together](https://www.together.xyz/)'s attempt to replicate [LLaMA](https://arxiv.org/abs/2302.13971)'s training data. Specifically, we used the CommonCrawl, arXiv, Wikipedia, Books, and StackExchange subsets.


<a id="source-section-25"></a>

### The Stack


We wanted our model to be capable of code generation, so we turned to [The Stack](https://huggingface.co/datasets/bigcode/the-stack), a 6.4TB corpus of code data. We used [The Stack Dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup), a variant of the stack that has been approximately deduplicated (via MinHashLSH) to 2.9TB. We selected a subset of 18 of The Stack's 358 programming languages in order to reduce dataset size and increase relevance:


- C


- C-Sharp


- C++


- Common Lisp


- F-Sharp


- Fortran


- Go


- Haskell


- Java


- Ocaml


- Perl


- Python


- Ruby


- Rust


- Scala


- Scheme


- Shell


- Tex


We chose to have code constitute 10% of the pretraining tokens, as internal experiments showed that we could train on up to 20% code (and 80% natural language) with no negative impact on natural language evaluation.


We also extracted the Markdown component of The Stack Dedup and treated this as an independent pretraining data subset (i.e. not counted towards the 10% code tokens). Our motivation for this is that markup language documents are predominantly natural language, and as such should count towards our natural language token budget.


<a id="source-section-26"></a>

## Semantic Scholar ORC


The Semantic Scholar Open Research Corpus (S2ORC) is a corpus of English-language academic papers, which we consider to be a high-quality data source. The following quality filtering criteria were applied:


- The paper is open access.


- The paper has a title and abstract.


- The paper is in English (as assessed using [cld3](https://github.com/google/cld3)).


- The paper has at least 500 words and 5 paragraphs.


- The paper was published after 1970 and before 2022-12-01.


- The most frequent word in the paper consists of alpha characters only, and it appears in less than 7.5% of the document.


This yielded 9.9M papers. Instructions to obtain the latest dataset version are available [here](https://api.semanticscholar.org/), and the original publication is [here](https://aclanthology.org/2020.acl-main.447/). The filtered version of the dataset was kindly provided to us by AI2.


<a id="source-section-27"></a>

## Evaluation Tasks


**Lambada:** 5153 samples of text curated from the books corpus. Consists of a several hundred word paragraph in which the model is expected to predict the next word.


**PIQA:** 1838 samples of physical intuitive binary multiple choice questions, e.g. "Question: How can I easily carry clothes on hangers when I move?", "Answer: "Take a couple of empty heavy duty clothes hangers, then hook several hangers of clothes on Those hangers and carry them all at once."


**COPA:** 100 sentences of the form XYZ therefore/because TUV. Framed as binary multiple choice questions where the model has a choice of two possible ways to follow the therefore/because. e.g. {"query": "The woman was in a bad mood, therefore", "gold": 1, "choices": ["she engaged in small talk with her friend.", "she told her friend to leave her alone."]}


**BoolQ:** 3270 yes/no questions based on some passage which contains relevant information. Question topics range from pop culture to science, law, history, etc. e.g. {"query": "Passage: Kermit the Frog is a Muppet character and Jim Henson's most well-known creation. Introduced in 1955, Kermit serves as the straight man protagonist of numerous Muppet productions, most notably Sesame Street and The Muppet Show, as well as in other television series, films, specials, and public service announcements through the years. Henson originally performed Kermit until his death in 1990; Steve Whitmire performed Kermit from that time up until his dismissal from the role in 2016. Kermit is currently performed by Matt Vogel. He was also voiced by Frank Welker in Muppet Babies and occasionally in other animation projects, and is voiced by Matt Danner in the 2018 reboot of Muppet Babies.\nQuestion: has kermit the frog been on sesame street?\n", "choices": ["no", "yes"], "gold": 1}


**Arc-Challenge:** 1172 challenging four-choice multiple choice questions about science


**Arc-Easy:** 2376 easy four choice multiple choice science questions


**HellaSwag:** 10042 four choice multiple choice questions in which a real life scenario is presented and the model must choose the most likely conclusion to the scenario.


**Jeopardy:** 2117 Jeopardy questions from five categories: science, world history, us history, word origins, and literature. The model must provide the exact correct answer


**MMLU:** 14,042 multiple choice questions from 57 diverse academic categories


**TriviaQA:** 11313 free response pop culture trivia questions


**Winograd:** 273 schema questions where the model must resolve which referent of a pronoun is most likely.


**Winogrande:** 1,267 schema questions where the model must resolve which ambiguous sentence is more logically likely (both versions of the sentence are syntactically valid)


<a id="source-section-28"></a>

## MPT Hugging Face Spaces Privacy Policy


Please see our [MPT Hugging Face Spaces Privacy Policy](https://gist.github.com/samhavens/c29c68cdcd420a9aa0202d0839876dac).


<a id="source-section-29"></a>

## Recommended for you


[June 22, 2023

<a id="source-section-30"></a>

#### MPT-30B: Raising the bar for open-source foundation models](https://www.databricks.com/blog/mpt-30b?itm_source=www&itm_category=home&itm_page=home&itm_offer=mpt-30b)


[July 18, 2023

<a id="source-section-31"></a>

#### Announcing MPT-7B-8K: 8K Context Length for Document Understanding](https://www.databricks.com/blog/long-context-mpt-7b-8k?itm_source=www&itm_category=home&itm_page=home&itm_offer=long-context-mpt-7b-8k)


[March 9, 2023

<a id="source-section-32"></a>

#### MosaicBERT: Pretraining BERT from Scratch for $20](https://www.databricks.com/blog/mosaicbert?itm_source=www&itm_category=home&itm_page=home&itm_offer=mosaicbert)

# Qwen Team - 2025 - Qwen2.5-Omni See Hear Talk Write Do It All

- Source HTML: `raw/html/Qwen Team - 2025 - Qwen2.5-Omni See Hear Talk Write Do It All.html`
- Source SHA256: `ff1d58e12c72c31d07187a0725b82c1d5fde4de7cf85519068573094f0d620f6`
- Source URL: https://qwenlm.github.io/blog/qwen2.5-omni/
- Generated from: `scripts/fetch_web_text.py`
- Extraction: `structured-html-v2` (headings, links, MathML/TeX and tables; figures require visual review)

## Extracted Text

<a id="source-section-0"></a>

[图片：无替代文本]


[QWEN CHAT](https://chat.qwenlm.ai)
[HUGGING FACE](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)
[MODELSCOPE](https://modelscope.cn/models/Qwen/Qwen2.5-Omni-7B)
[DASHSCOPE](https://help.aliyun.com/zh/model-studio/user-guide/qwen-omni)
[GITHUB](https://github.com/QwenLM/Qwen2.5-Omni)
[PAPER](https://github.com/QwenLM/Qwen2.5-Omni/blob/main/assets/Qwen2.5_Omni.pdf)
[DEMO](https://huggingface.co/spaces/Qwen/Qwen2.5-Omni-7B-Demo)
[DISCORD](https://discord.com/invite/yPEP2vHTu4)


We release **Qwen2.5-Omni**, the new flagship end-to-end multimodal model in the Qwen series. Designed for comprehensive multimodal perception, it seamlessly processes diverse inputs including text, images, audio, and video, while delivering real-time streaming responses through both text generation and natural speech synthesis. To try the latest model, feel free to visit [Qwen Chat](https://chat.qwenlm.ai) and choose Qwen2.5-Omni-7B. The model is now openly available on [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Omni-7B), [ModelScope](https://modelscope.cn/models/Qwen/Qwen2.5-Omni-7B), [DashScope](https://help.aliyun.com/zh/model-studio/user-guide/qwen-omni),and [GitHub](https://github.com/QwenLM/Qwen2.5-Omni), with technical documentation available in our [Paper](https://github.com/QwenLM/Qwen2.5-Omni/assets/Qwen2.5_Omni.pdf). Experience interactive capabilities through our [Demo](https://huggingface.co/spaces/Qwen/Qwen2.5-Omni-7B-Demo) or join our [Discord](https://discord.gg/yPEP2vHTu4) for discussions.


Key Features:


- **Omni and Novel Architecture**: We propose Thinker-Talker architecture, an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously
generating text and natural speech responses in a streaming manner. We prpose a novel position embedding, named TMRoPE (Time-aligned
Multimodal RoPE), to synchronize the timestamps of video inputs with audio.

- **Real-Time Voice and Video Chat**: Architecture Designed for fully real-time interactions, supporting chunked input and immediate output.

- **Natural and Robust Speech Generation**: Surpassing many existing streaming and non-streaming alternatives, demonstrating superior robustness and naturalness in speech generation.

- **Strong Performance Across Modalities**: Exhibiting exceptional performance across all modalities when benchmarked against similarly sized single-modality models. Qwen2.5-Omni outperforms the similarly sized Qwen2-Audio in audio capabilities and achieves comparable performance to Qwen2.5-VL-7B.

- **Excellent End-to-End Speech Instruction Following**: Qwen2.5-Omni shows performance in end-to-end speech instruction following that rivals its effectiveness with text inputs, evidenced by benchmarks such as MMLU and GSM8K.


<a id="source-section-1"></a>

## Architecture


Qwen2.5-Omni employs Thinker-Talker architecture. Thinker functions like a brain, responsible for processing and understanding inputs from text, audio and video modalities, generating high-level representations and corresponding text. Talker operates like a human mouth, taking in the high-level representations and text produced by the Thinker in a streaming manner, and outputting discrete tokens of speech fluidly. Thinker is a Transformer decoder, accompanied by encoders for audio and image that facilitate information extraction. In contrast, Talker is designed as a dual-track autoregressive Transformer Decoder architecture. During both training and inference, Talker directly receives high-dimensional representations from Thinker and shares all of Thinker’s historical context information. Consequently, the entire architecture operates as a cohesive single model, enabling end-to-end training and inference.


[图片：无替代文本]


<a id="source-section-2"></a>

## Performance


We conducted a comprehensive evaluation of Qwen2.5-Omni, which demonstrates strong performance across all modalities when compared to similarly sized single-modality models and closed-source models like Qwen2.5-VL-7B, Qwen2-Audio, and Gemini-1.5-pro. In tasks requiring the integration of multiple modalities, such as OmniBench, Qwen2.5-Omni achieves state-of-the-art performance. Furthermore, in single-modality tasks, it excels in areas including speech recognition (Common Voice), translation (CoVoST2), audio understanding (MMAU), image reasoning (MMMU, MMStar), video understanding (MVBench), and speech generation (Seed-tts-eval and subjective naturalness).


[图片：无替代文本]


<a id="source-section-3"></a>

## What’s Next


We are eager to hear your feedback and see the innovative applications you create with Qwen2.5-Omni. In the near future, our goal is to enhance our model’s ability to follow voice commands and improve audio-visual collaborative understanding. Additionally, we strive to integrate more modalities towards an omni-model!

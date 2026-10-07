# Wiki Index

## TL;DR（快速导读）

这里按研究问题组织 AI 论文、模型和方法。想理解一个方向，先看[主题综述](#topics)；查一个具体方法，进[概念与模型](#concepts)；选方案看[横向比较](#comparisons)；找版本关系看[时间线](#timelines)；沿团队找资料看[作者与机构](#authors)。[单篇解读](#summary)保存来源的解释与证据，原文链接在对应页面里。

现有 288 篇来源摘要已核证本页核心主张，20 个主题已完成证据综合；这不表示逐项复现实验或审阅所有附录。以后新增的“待精读”摘要和“待建设”主题会继续明确标注成熟度。先用每页开头的 TL;DR 判断是否相关，再读例子、方法和条件。

本页是 LLM Wiki 的主导航入口。查询和维护时，优先先读本页，再进入具体页面。

## 使用规则

- `raw/pdf/` 是原始资料层，不在本页登记。
- `wiki/` 中每个实质内容页都应在本页登记。
- 每条记录保持一句话摘要，优先写“这页解决什么问题”。
- 当前仓库已进入持续维护阶段；正式 topic 必须通过精修 summary 证据门禁，待建设页面会在索引中显式标注。

## Summary

- 当前已接入 288 篇 summary 页，包含论文、官方文档与入口快照，按主题分组如下。

### 论文发现与阅读入口

- [arXiv - 2026 - Artificial Intelligence Recent Submissions](./wiki/summaries/arXiv%20-%202026%20-%20Artificial%20Intelligence%20Recent%20Submissions.md)：精修摘要；arXiv 近期提交目录按学科和日期组织新稿，适合已有研究方向后扩大阅读候选。
- [Hugging Face - 2026 - Trending Papers](./wiki/summaries/Hugging%20Face%20-%202026%20-%20Trending%20Papers.md)：精修摘要；Hugging Face Trending 适合发现社区关注的论文；投票热度、论文日期和研究质量要分别判断。
- [Hugging Face - 2026 - Paper Pages](./wiki/summaries/Hugging%20Face%20-%202026%20-%20Paper%20Pages.md)：精修摘要；Hugging Face 论文页用 arXiv ID 连接论文与模型、数据和演示，方便从一篇论文继续寻找相关资源。

### 知识库维护与文档处理

- [Anthropic - 2025 - Effective Context Engineering for AI Agents](./wiki/summaries/Anthropic%20-%202025%20-%20Effective%20Context%20Engineering%20for%20AI%20Agents.md)：精修摘要；长任务要按问题读取材料，并把已确认的结论与待办写成笔记，让有限上下文保存真正需要的状态。
- [Anthropic - 2024 - Introducing Contextual Retrieval](./wiki/summaries/Anthropic%20-%202024%20-%20Introducing%20Contextual%20Retrieval.md)：精修摘要；Contextual Retrieval 在片段进入检索索引前补上整篇文档的背景，减少断章取义的匹配。
- [Microsoft - 2026 - GraphRAG Query Engine](./wiki/summaries/Microsoft%20-%202026%20-%20GraphRAG%20Query%20Engine.md)：精修摘要；GraphRAG 按问题范围选择查询路径：具体实体问题与全库综合问题需要不同的材料组织方式。
- [Microsoft - 2026 - GraphRAG Indexing Methods](./wiki/summaries/Microsoft%20-%202026%20-%20GraphRAG%20Indexing%20Methods.md)：精修摘要；GraphRAG 的两种索引路线在关系描述、噪声和建库成本之间取舍，生成更多连接并不等于知识更准确。
- [Docling Project - 2026 - Document Processing Overview](./wiki/summaries/Docling%20Project%20-%202026%20-%20Document%20Processing%20Overview.md)：精修摘要；Docling 把文档解析为结构化表示；检查论文时，要分别验收文字、阅读顺序、表格与公式。

### Attention / Transformer

- [Vaswani et al. - 2017 - Attention is all you need](./wiki/summaries/Vaswani%20et%20al.%20-%202017%20-%20Attention%20is%20all%20you%20need.md)：精修摘要；Transformer 用注意力与位置编码处理序列，让各位置直接获取其他位置的信息，取代循环计算作为主要结构。
- [Shazeer - 2019 - Fast Transformer Decoding One Write-Head is All You Need](./wiki/summaries/Shazeer%20-%202019%20-%20Fast%20Transformer%20Decoding%20One%20Write-Head%20is%20All%20You%20Need.md)：精修摘要；MQA 让多个查询头共享同一组键和值，减少逐词元生成时反复读取的缓存。
- [Kitaev, Kaiser, Levskaya - 2020 - Reformer The Efficient Transformer](./wiki/summaries/Kitaev,%20Kaiser,%20Levskaya%20-%202020%20-%20Reformer%20The%20Efficient%20Transformer.md)：精修摘要；Reformer 用哈希分桶筛选相似位置，并用可逆层节省训练存储，研究长序列的低成本处理。
- [Beltagy, Peters, Cohan - 2020 - Longformer The Long-Document Transformer](./wiki/summaries/Beltagy,%20Peters,%20Cohan%20-%202020%20-%20Longformer%20The%20Long-Document%20Transformer.md)：精修摘要；Longformer 让大多数位置只看附近内容，少数任务关键位置看全篇，以较少连接处理长文。
- [Wang et al. - 2020 - Linformer Self-Attention with Linear Complexity](./wiki/summaries/Wang%20et%20al.%20-%202020%20-%20Linformer%20Self-Attention%20with%20Linear%20Complexity.md)：精修摘要；Linformer 先在序列维压缩键和值，再计算注意力，以低秩近似降低长序列成本。
- [Zaheer et al. - 2020 - Big bird Transformers for longer sequences](./wiki/summaries/Zaheer%20et%20al.%20-%202020%20-%20Big%20bird%20Transformers%20for%20longer%20sequences.md)：精修摘要；BigBird 混合局部、全局和随机连接，用较少注意力关系处理长序列，并分析表达能力。
- [Choromanski et al. - 2021 - Rethinking Attention with Performers](./wiki/summaries/Choromanski%20et%20al.%20-%202021%20-%20Rethinking%20Attention%20with%20Performers.md)：精修摘要；Performer 用随机特征近似标准注意力，希望降低长序列的计算与存储成本；近似误差是比较时的关键。
- [Xiong et al. - 2021 - Nyströmformer A Nystrom-Based Algorithm for Approximating Self-Attention](./wiki/summaries/Xiong%20et%20al.%20-%202021%20-%20Nystr%C3%B6mformer%20A%20Nystrom-Based%20Algorithm%20for%20Approximating%20Self-Attention.md)：精修摘要；Nyströmformer 用少量代表位置重建注意力的近似关系，减少长序列的计算。
- [Dao et al. - 2022 - FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness](./wiki/summaries/Dao%20et%20al.%20-%202022%20-%20FlashAttention%20Fast%20and%20Memory-Efficient%20Exact%20Attention%20with%20IO-Awareness.md)：精修摘要；FlashAttention 保持标准注意力计算的含义，通过分块和融合计算减少显存读写，让执行更省内存、更快。
- [Ainslie et al. - 2023 - GQA Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](./wiki/summaries/Ainslie%20et%20al.%20-%202023%20-%20GQA%20Training%20Generalized%20Multi-Query%20Transformer%20Models%20from%20Multi-Head%20Checkpoints.md)：精修摘要；GQA 让多个查询头共用一组键和值，在标准多头注意力与单组共享之间调节缓存成本。

### LLM 推理与服务系统

- [Kwon et al. - 2023 - Efficient Memory Management for Large Language Model Serving with PagedAttention](./wiki/summaries/Kwon%20et%20al.%20-%202023%20-%20Efficient%20Memory%20Management%20for%20Large%20Language%20Model%20Serving%20with%20PagedAttention.md)：精修摘要；PagedAttention 将逐渐增长的键值缓存切成小块，按需分配并允许共享，减少语言模型服务的内存浪费。
- [Zheng et al. - 2024 - SGLang Efficient Execution of Structured Language Model Programs](./wiki/summaries/Zheng%20et%20al.%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.md)：精修摘要；SGLang 将多次模型调用组织成程序，并用运行时复用共同前缀、加速约束输出与执行。
- [SGLang Team - 2024 - SGLang v0.4 Zero-Overhead Batch Scheduler Cache-Aware Load Balancer Faster Structured Outputs](./wiki/summaries/SGLang%20Team%20-%202024%20-%20SGLang%20v0.4%20Zero-Overhead%20Batch%20Scheduler%20Cache-Aware%20Load%20Balancer%20Faster%20Structured%20Outputs.md)：精修摘要；SGLang v0.4 分别优化调度、请求路由、多 GPU 数据流和结构化输出，各项收益对应不同瓶颈。
- [SGLang Project - 2026 - HiCache System Design and Optimization](./wiki/summaries/SGLang%20Project%20-%202026%20-%20HiCache%20System%20Design%20and%20Optimization.md)：精修摘要；HiCache 用 GPU、主机内存和远端存储三级缓存保存重复前缀，在容量、读写速度与命中率之间取舍。
- [vLLM Project - 2026 - Architecture Overview](./wiki/summaries/vLLM%20Project%20-%202026%20-%20Architecture%20Overview.md)：精修摘要；vLLM 架构文档解释请求怎样经过接口服务、引擎调度和 GPU 执行，适合定位不同层的瓶颈。
- [vLLM Project - 2026 - vLLM V1 Guide](./wiki/summaries/vLLM%20Project%20-%202026%20-%20vLLM%20V1%20Guide.md)：精修摘要；vLLM V1 指南说明按统一词元预算调度请求，并记录重构后的功能与兼容边界。
- [vLLM Project - 2026 - Automatic Prefix Caching](./wiki/summaries/vLLM%20Project%20-%202026%20-%20Automatic%20Prefix%20Caching.md)：精修摘要；vLLM 前缀缓存复用内容与执行条件一致的完整缓存块，跳过重复提示计算；它不直接加速后续逐词生成。

### LLM预训练

- [Bai et al. - 2023 - Qwen Technical Report](./wiki/summaries/Bai%20et%20al.%20-%202023%20-%20Qwen%20Technical%20Report.md)：精修摘要；初代 Qwen 报告同时介绍基础、聊天及专门模型，并说明它们如何训练成能遵循指令和使用工具的系统。
- [Qwen Team - 2024 - Introducing Qwen1.5](./wiki/summaries/Qwen%20Team%20-%202024%20-%20Introducing%20Qwen1.5.md)：精修摘要；Qwen1.5 发布资料扩展模型规模与部署支持，关注开发者如何下载、量化和运行不同成员。
- [Qwen Team - 2024 - Hello Qwen2](./wiki/summaries/Qwen%20Team%20-%202024%20-%20Hello%20Qwen2.md)：精修摘要；Qwen2 发布页介绍语言覆盖、长上下文及代码数学能力的更新，适合辨认家族代际变化。
- [Qwen Team - 2024 - Qwen2.5-LLM Extending the boundary of LLMs](./wiki/summaries/Qwen%20Team%20-%202024%20-%20Qwen2.5-LLM%20Extending%20the%20boundary%20of%20LLMs.md)：精修摘要；Qwen2.5 发布页介绍多个模型尺寸，以及知识、代码、数学、结构化输出和长文本方面的更新。
- [Brown et al. - 2020 - Language models are few-shot learners](./wiki/summaries/Brown%20et%20al.%20-%202020%20-%20Language%20models%20are%20few-shot%20learners.md)：精修摘要；GPT-3 展示了上下文少样本学习：给模型几组任务示例，再让它回答新问题，过程中不必为这个任务更新模型参数。
- [Chen et al. - 2024 - A Survey on Large Language Models for Critical Societal Domains Finance, Healthcare, and Law](./wiki/summaries/Chen%20et%20al.%20-%202024%20-%20A%20Survey%20on%20Large%20Language%20Models%20for%20Critical%20Societal%20Domains%20Finance,%20Healthcare,%20and%20Law.md)：精修摘要；这篇综述讨论 LLM 在金融、医疗和法律中的应用，重点是专业数据、可靠性与合规约束，适合了解任务和研究缺口。
- [Chowdhery et al. - 2022 - PaLM Scaling Language Modeling with Pathways](./wiki/summaries/Chowdhery%20et%20al.%20-%202022%20-%20PaLM%20Scaling%20Language%20Modeling%20with%20Pathways.md)：精修摘要；PaLM 用大规模密集 Transformer 和 Pathways 训练系统研究语言模型的规模化收益；它同时是一份模型与训练系统报告。
- [Dubey et al. - 2024 - The Llama 3 Herd of Models](./wiki/summaries/Dubey%20et%20al.%20-%202024%20-%20The%20Llama%203%20Herd%20of%20Models.md)：精修摘要；Llama 3 报告描述语言模型家族的预训练、后训练和评测，涉及多语言、编程、推理与工具使用，需要分开看模型能力和开放条件。
- [Touvron et al. - 2023 - LLaMA Open and Efficient Foundation Language Models](./wiki/summaries/Touvron%20et%20al.%20-%202023%20-%20LLaMA%20Open%20and%20Efficient%20Foundation%20Language%20Models.md)：精修摘要；LLaMA 通过更多公开数据和更长训练，让较小模型在给定推理预算下获得较强语言能力。
- [Touvron et al. - 2023 - Llama 2 Open Foundation and Fine-Tuned Chat Models](./wiki/summaries/Touvron%20et%20al.%20-%202023%20-%20Llama%202%20Open%20Foundation%20and%20Fine-Tuned%20Chat%20Models.md)：精修摘要；Llama 2 同时提供基础模型与聊天模型，后者通过专门后训练改善指令遵循和对话行为。
- [Roziere et al. - 2023 - Code Llama Open Foundation Models for Code](./wiki/summaries/Roziere%20et%20al.%20-%202023%20-%20Code%20Llama%20Open%20Foundation%20Models%20for%20Code.md)：精修摘要；Code Llama 在 Llama 2 基础上继续学习代码，并加入长上下文和补全中间代码的训练目标。
- [Scao et al. - 2022 - BLOOM A 176B-Parameter Open-Access Multilingual Language Model](./wiki/summaries/Scao%20et%20al.%20-%202022%20-%20BLOOM%20A%20176B-Parameter%20Open-Access%20Multilingual%20Language%20Model.md)：精修摘要；BLOOM 由国际协作建设多语言大模型，阅读重点包括语言覆盖、数据、公开材料与发布条件。
- [MosaicML - 2023 - MPT-7B](./wiki/summaries/MosaicML%20-%202023%20-%20MPT-7B.md)：精修摘要；MPT-7B 发布资料介绍面向开发者的语言模型底座，关注训练配方、长上下文与部署使用。
- [Jiang et al. - 2023 - Mistral 7B](./wiki/summaries/Jiang%20et%20al.%20-%202023%20-%20Mistral%207B.md)：精修摘要；Mistral 7B 将查询分组共享与滑动窗口注意力结合，用较紧凑的模型研究语言能力和推理效率。
- [Jiang et al. - 2024 - Mixtral of Experts](./wiki/summaries/Jiang%20et%20al.%20-%202024%20-%20Mixtral%20of%20Experts.md)：精修摘要；Mixtral 为每个输入选择部分专家，扩大总容量，同时控制每次实际参与计算的规模。
- [Team, Google - 2024 - Gemma Open Models Based on Gemini Research and Technology](./wiki/summaries/Team,%20Google%20-%202024%20-%20Gemma%20Open%20Models%20Based%20on%20Gemini%20Research%20and%20Technology.md)：精修摘要；初代 Gemma 报告介绍从 Google 研究经验发展出的较小开放模型，适合建立家族的训练与使用背景。
- [Team, Google DeepMind - 2024 - Gemma 2 Improving Open Language Models at a Practical Size](./wiki/summaries/Team,%20Google%20DeepMind%20-%202024%20-%20Gemma%202%20Improving%20Open%20Language%20Models%20at%20a%20Practical%20Size.md)：精修摘要；Gemma 2 报告讨论实用模型规模下的能力与部署取舍，帮助辨认相对初代的改进。
- [Google DeepMind - 2026 - Gemma 4 Model Card](./wiki/summaries/Google%20DeepMind%20-%202026%20-%20Gemma%204%20Model%20Card.md)：精修摘要；Gemma 4 模型卡用于核对不同规模与架构的规格，以及长上下文、多模态和工具使用的支持范围。
- [Google - 2026 - Gemma 4 Byte for Byte Most Capable Open Models](./wiki/summaries/Google%20-%202026%20-%20Gemma%204%20Byte%20for%20Byte%20Most%20Capable%20Open%20Models.md)：精修摘要；Gemma 4 发布博客介绍开放模型家族的定位，包括推理、多模态和代理工作流；它主要提供产品层面的入口。
- [Lozhkov et al. - 2024 - StarCoder 2 and The Stack v2 The Next Generation](./wiki/summaries/Lozhkov%20et%20al.%20-%202024%20-%20StarCoder%202%20and%20The%20Stack%20v2%20The%20Next%20Generation.md)：精修摘要；StarCoder2 与 The Stack v2 报告介绍代码模型及其训练数据，重点关注数据质量、授权处理与不同模型规模。
- [DBRX：Databricks 官方模型发布说明](./wiki/summaries/Databricks%20-%202024%20-%20DBRX%20A%20Highly%20Efficient%20Open%20LLM.md)：精修摘要；DBRX 用更细的专家划分控制每个 token 的计算：总参数 132B、激活约 36B，每次选择 16 个专家中的 4 个。正确来源是 Databricks 官方发布说明，旧附件已确认误配。
- [Mehta et al. - 2024 - OpenELM An Efficient Language Model Family with Open Training and Inference Framework](./wiki/summaries/Mehta%20et%20al.%20-%202024%20-%20OpenELM%20An%20Efficient%20Language%20Model%20Family%20with%20Open%20Training%20and%20Inference%20Framework.md)：精修摘要；OpenELM 将小模型、端侧效率与公开训练推理框架放在一起，适合研究设备约束下的语言模型。
- [Abdin et al. - 2024 - Phi-3 Technical Report A Highly Capable Language Model Locally on Your Phone](./wiki/summaries/Abdin%20et%20al.%20-%202024%20-%20Phi-3%20Technical%20Report%20A%20Highly%20Capable%20Language%20Model%20Locally%20on%20Your%20Phone.md)：精修摘要；Phi-3 研究怎样用较小语言模型提供实用能力，重点是训练数据质量与本地部署成本。
- [Ai2 - 2024 - OLMo 2 The Best Fully Open Language Model to Date](./wiki/summaries/Ai2%20-%202024%20-%20OLMo%202%20The%20Best%20Fully%20Open%20Language%20Model%20to%20Date.md)：精修摘要；OLMo 2 的价值在于把权重、训练数据、代码与评测一起公开，并展示稳定训练和后期数据课程如何改善 7B/13B 模型。性能结论应按发布时的英语基准理解。
- [TII - 2024 - Falcon 3](./wiki/summaries/TII%20-%202024%20-%20Falcon%203.md)：精修摘要；Falcon 3 发布资料用于追踪 Falcon 家族的后续模型与开发者入口，具体能力应按成员检查。
- [Zeng et al. - 2022 - GLM-130B An Open Bilingual Pre-trained Model](./wiki/summaries/Zeng%20et%20al.%20-%202022%20-%20GLM-130B%20An%20Open%20Bilingual%20Pre-trained%20Model.md)：精修摘要；GLM-130B 将 GLM 的填空式训练路线扩展到大型中英双语模型，是追踪后续家族的基础资料。
- [Hoffmann et al. - 2022 - Training Compute-Optimal Large Language Models](./wiki/summaries/Hoffmann%20et%20al.%20-%202022%20-%20Training%20Compute-Optimal%20Large%20Language%20Models.md)：精修摘要；Chinchilla 研究固定训练预算下参数量和数据量的搭配，发现一味增大模型而不给足训练数据会浪费计算。
- [Hu et al. - 2021 - LoRA Low-Rank Adaptation of Large Language Models](./wiki/summaries/Hu%20et%20al.%20-%202021%20-%20LoRA%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models.md)：精修摘要；LoRA 冻结大模型原有权重，只训练小规模低秩增量，让同一个底座能更便宜地适配不同任务。
- [Inan et al. - 2023 - Llama Guard LLM-based Input-Output Safeguard for Human-AI Conversations](./wiki/summaries/Inan%20et%20al.%20-%202023%20-%20Llama%20Guard%20LLM-based%20Input-Output%20Safeguard%20for%20Human-AI%20Conversations.md)：精修摘要；Llama Guard 用独立模型检查用户输入和模型回答的风险，适合放在对话系统的审核环节。
- [Iyer et al. - 2022 - OPT-IML Scaling Language Model Instruction Meta Learning through the Lens of Generalization](./wiki/summaries/Iyer%20et%20al.%20-%202022%20-%20OPT-IML%20Scaling%20Language%20Model%20Instruction%20Meta%20Learning%20through%20the%20Lens%20of%20Generalization.md)：精修摘要；OPT-IML 系统研究指令微调的规模、任务多样性和数据分配，重点是模型能否迁移到没见过的任务。
- [Rothe, Narayan, Severyn - 2020 - Leveraging pre-trained checkpoints for sequence generation tasks](./wiki/summaries/Rothe,%20Narayan,%20Severyn%20-%202020%20-%20Leveraging%20pre-trained%20checkpoints%20for%20sequence%20generation%20tasks.md)：精修摘要；这篇工作把已有预训练检查点用于序列生成，研究怎样复用编码器和解码器，减少从头训练的成本。
- [Sun et al. - 2023 - A Comparative Study between Full-Parameter and LoRA-based Fine-Tuning on Chinese Instruction Data for Instruction Fo](./wiki/summaries/Sun%20et%20al.%20-%202023%20-%20A%20Comparative%20Study%20between%20Full-Parameter%20and%20LoRA-based%20Fine-Tuning%20on%20Chinese%20Instruction%20Data%20for%20Instruction%20Fo.md)：精修摘要；这篇中文指令微调实验比较 LoRA 与全参数微调，关注节省训练成本之后，任务效果有怎样的变化。
- [Team, Meta - 2024 - The Llama 3 Herd of Models](./wiki/summaries/Team,%20Meta%20-%202024%20-%20The%20Llama%203%20Herd%20of%20Models.md)：精修摘要；同源归档；这份 Llama 3 报告归档与库中另一份可能属于同一工作，阅读重点仍是数据、训练、后训练和具体版本评测。
- [Unknown - 2024 - DeepSeek-V3 Technical Report](./wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-V3%20Technical%20Report.md)：精修摘要；DeepSeek-V3 的 MLA 将键和值压缩到较小的联合表示，降低生成时缓存的内存与读写压力。
- [DeepSeek AI - 2026 - DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence](./wiki/summaries/DeepSeek%20AI%20-%202026%20-%20DeepSeek-V4%20Towards%20Highly%20Efficient%20Million-Token%20Context%20Intelligence.md)：精修摘要；DeepSeek-V4 报告把长上下文、代理任务和专家模型效率一起设计，重点检查注意力与缓存怎样承担更长输入。
- [Wei et al. - 2021 - Finetuned Language Models Are Zero-Shot Learners](./wiki/summaries/Wei%20et%20al.%20-%202021%20-%20Finetuned%20Language%20Models%20Are%20Zero-Shot%20Learners.md)：精修摘要；FLAN 通过多任务自然语言指令微调，改善模型在未见任务上的零样本表现，展示了指令数据的迁移价值。
- [Kimi Team et al. - 2025 - Kimi k1.5 Scaling Reinforcement Learning with LLMs](./wiki/summaries/Kimi%20Team%20et%20al.%20-%202025%20-%20Kimi%20k1.5%20Scaling%20Reinforcement%20Learning%20with%20LLMs.md)：精修摘要；Kimi k1.5 把强化学习、长上下文与多模态推理一起研究，是理解 Kimi 推理训练路线的资料。
- [Kimi Team - 2026 - Kimi K3 Open Frontier Intelligence](./wiki/summaries/Kimi%20Team%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence.md)：精修摘要；Kimi K3 报告同时调整序列、层间和专家信息流，并结合多模态训练与后训练处理长程任务。
- [Kimi - 2026 - Kimi K3 Open Frontier Intelligence Release](./wiki/summaries/Kimi%20-%202026%20-%20Kimi%20K3%20Open%20Frontier%20Intelligence%20Release.md)：精修摘要；Kimi K3 发布页提供可用入口与长任务案例，也记录思考历史、主动执行和用户体验方面的限制。
- [Moonshot AI - 2026 - Kimi K3 Model Repository](./wiki/summaries/Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20Model%20Repository.md)：精修摘要；Kimi K3 模型仓库提供结构、部署入口和消息协议，其中多轮工具调用需要保留完整助手历史。
- [Moonshot AI - 2026 - Kimi K3 License](./wiki/summaries/Moonshot%20AI%20-%202026%20-%20Kimi%20K3%20License.md)：精修摘要；这份 Kimi K3 许可快照包含使用授权与商业条件；下载到权重后，仍需核对对应版本的条款。
- [Kimi - 2026 - Kimi API Model Selection](./wiki/summaries/Kimi%20-%202026%20-%20Kimi%20API%20Model%20Selection.md)：精修摘要；这份 Kimi API 快照帮助辨认 K3 与 K2.6 的模式、上下文和思考预算，适合核对调用接口。
- [Qwen Team - 2025 - Qwen3 Think Deeper Act Faster](./wiki/summaries/Qwen%20Team%20-%202025%20-%20Qwen3%20Think%20Deeper%20Act%20Faster.md)：精修摘要；Qwen3 的混合思考模式让同一模型在深入推理与快速回答之间切换，思考预算成为使用条件之一。
- [Qwen Team - 2026 - Qwen3.5 Towards Native Multimodal Agents](./wiki/summaries/Qwen%20Team%20-%202026%20-%20Qwen3.5%20Towards%20Native%20Multimodal%20Agents.md)：精修摘要；这份 Qwen3.5 官方索引快照将视觉语言与多模态代理作为家族方向，主要提供研究入口。
- [Yuan et al. - 2023 - Scaling Relationship on Learning Mathematical Reasoning with Large Language Models](./wiki/summaries/Yuan%20et%20al.%20-%202023%20-%20Scaling%20Relationship%20on%20Learning%20Mathematical%20Reasoning%20with%20Large%20Language%20Models.md)：精修摘要；这篇数学推理缩放研究比较预训练损失、监督数据和增广数据的影响，发现参数量本身不是充分的能力指标。
- [Zhang et al. - 2022 - OPT Open Pre-trained Transformer Language Models](./wiki/summaries/Zhang%20et%20al.%20-%202022%20-%20OPT%20Open%20Pre-trained%20Transformer%20Language%20Models.md)：精修摘要；OPT 开放不同规模的解码器语言模型与研究材料，为研究大规模语言模型提供更可访问的训练和评测入口。

### 优化器与训练稳定性

- [Kingma and Ba - 2015 - Adam: A Method for Stochastic Optimization](./wiki/summaries/Kingma%20and%20Ba%20-%202015%20-%20Adam%20A%20Method%20for%20Stochastic%20Optimization.md)：精修摘要；Adam 记录梯度的平均趋势与平方大小，为每个参数调整更新尺度；它是一阶优化器。
- [Loshchilov and Hutter - 2019 - Decoupled Weight Decay Regularization](./wiki/summaries/Loshchilov%20and%20Hutter%20-%202019%20-%20Decoupled%20Weight%20Decay%20Regularization.md)：精修摘要；AdamW 将权重缩小操作与自适应梯度更新分开，解决 Adam 中 L2 惩罚与权重衰减不等价的问题。
- [Keller Jordan - 2024 - Muon: An Optimizer for Hidden Layers in Neural Networks](./wiki/summaries/Keller%20Jordan%20-%202024%20-%20Muon%20An%20Optimizer%20for%20Hidden%20Layers%20in%20Neural%20Networks.md)：精修摘要；Muon 对矩阵参数的动量更新做近似正交化，尝试让不同更新方向获得更均衡的尺度。
- [Liu et al. - 2025 - Muon is Scalable for LLM Training](./wiki/summaries/Liu%20et%20al.%20-%202025%20-%20Muon%20is%20Scalable%20for%20LLM%20Training.md)：精修摘要；Moonlight 报告研究把 Muon 用于大规模语言模型训练，强调权重衰减和随矩阵形状调整更新尺度。
- [Kimi Team - 2025 - Kimi K2: Open Agentic Intelligence](./wiki/summaries/Kimi%20Team%20-%202025%20-%20Kimi%20K2%20Open%20Agentic%20Intelligence.md)：精修摘要；Kimi K2 报告使用 MuonClip 限制过大的注意力分数，研究把矩阵优化器扩展到大型专家模型时的稳定性。
- [OLMo Team - 2025 - 2 OLMo 2 Furious](./wiki/summaries/Team%20OLMo%20-%202025%20-%202%20OLMo%202%20Furious.md)：精修摘要；这份 OLMo 2 报告沿用 AdamW，并把数值稳定项和权重衰减范围作为训练稳定性的实验对象。
- [Lim et al. - 2025 - Motif-2-12.7B Technical Report](./wiki/summaries/Lim%20et%20al.%20-%202025%20-%20Motif%202%2012.7B%20Technical%20Report.md)：精修摘要；Motif-2 报告将矩阵正交化任务分给不同设备并行执行，研究分布式 Muon 的通信与重复计算成本。

### 文本扩散语言模型

- [Google DeepMind - 2026 - DiffusionGemma 26B A4B IT Model Card](./wiki/summaries/Google%20DeepMind%20-%202026%20-%20DiffusionGemma%2026B%20A4B%20IT%20Model%20Card.md)：精修摘要；DiffusionGemma 模型卡介绍一个从噪声逐步修正文本块的开放模型，并明确讨论速度收益与质量限制。
- [Google - 2026 - DiffusionGemma 4x Faster Text Generation](./wiki/summaries/Google%20-%202026%20-%20DiffusionGemma%204x%20Faster%20Text%20Generation.md)：精修摘要；DiffusionGemma 发布博客介绍文本扩散的速度实验，主要针对本地、低并发的交互场景。
- [Google Developers - 2026 - DiffusionGemma The Developer Guide](./wiki/summaries/Google%20Developers%20-%202026%20-%20DiffusionGemma%20The%20Developer%20Guide.md)：精修摘要；DiffusionGemma 开发指南解释如何先读取提示，再按文本块并行去噪，并把完成的块接回上下文。
- [Google AI for Developers - 2026 - DiffusionGemma Model Overview](./wiki/summaries/Google%20AI%20for%20Developers%20-%202026%20-%20DiffusionGemma%20Model%20Overview.md)：精修摘要；DiffusionGemma 官方概览说明输入输出、底座和采样配置，适合先确认模型是否符合自己的部署任务。
- [Google DeepMind - 2026 - Gemini Diffusion](./wiki/summaries/Google%20DeepMind%20-%202026%20-%20Gemini%20Diffusion.md)：精修摘要；Gemini Diffusion 实验页展示从噪声反复修正文本的路线，用来理解快速生成与编辑的研究动机。
- [NVIDIA - 2026 - Run DiffusionGemma on NVIDIA for Developer-Ready High-Throughput Text Generation](./wiki/summaries/NVIDIA%20-%202026%20-%20Run%20DiffusionGemma%20on%20NVIDIA%20for%20Developer-Ready%20High-Throughput%20Text%20Generation.md)：精修摘要；NVIDIA 的 DiffusionGemma 部署文章解释并行去噪如何使用 GPU 计算，并介绍平台与精度配置。
- [Maarten Grootendorst - 2026 - A Visual Guide to DiffusionGemma](./wiki/summaries/Maarten%20Grootendorst%20-%202026%20-%20A%20Visual%20Guide%20to%20DiffusionGemma.md)：精修摘要；这篇视觉指南用图解解释 DiffusionGemma 的分块去噪、采样与速度动机，适合先建立直觉。

### 多模态与 Omni

- [Qwen Team - 2024 - Qwen2-VL](./wiki/summaries/Qwen%20Team%20-%202024%20-%20Qwen2-VL.md)：精修摘要；Qwen2-VL 将语言模型接到图片和视频输入，也覆盖文档读取与视觉代理任务。
- [Bai et al. - 2025 - Qwen2.5-VL Technical Report](./wiki/summaries/Bai%20et%20al.%20-%202025%20-%20Qwen2.5-VL%20Technical%20Report.md)：精修摘要；Qwen2.5-VL 处理图片、文档和视频，也研究把视觉理解用于界面操作；关键是保留分辨率、时间和结构信息。
- [Qwen Team - 2025 - Qwen2.5-Omni See Hear Talk Write Do It All](./wiki/summaries/Qwen%20Team%20-%202025%20-%20Qwen2.5-Omni%20See%20Hear%20Talk%20Write%20Do%20It%20All.md)：精修摘要；Qwen2.5-Omni 接收文字、图片、音频和视频，并生成文字或语音，研究流式多模态交互。
- [Qwen Team - 2026 - Qwen3.5-Omni Scaling Up Toward Native Omni-Modal AGI](./wiki/summaries/Qwen%20Team%20-%202026%20-%20Qwen3.5-Omni%20Scaling%20Up%20Toward%20Native%20Omni-Modal%20AGI.md)：精修摘要；这份 Qwen3.5-Omni 索引快照介绍多模态家族的扩展方向，关注长上下文和音视频理解。

### 扩散模型与文生图

- [Rombach et al. - 2022 - High-Resolution Image Synthesis with Latent Diffusion Models](./wiki/summaries/Rombach%20et%20al.%20-%202022%20-%20High-Resolution%20Image%20Synthesis%20with%20Latent%20Diffusion%20Models.md)：精修摘要；潜空间扩散先把图像压缩成表示，再在较小表示中去噪，降低图像生成的计算成本。
- [Stability AI - 2022 - Stable Diffusion Launch Announcement](./wiki/summaries/Stability%20AI%20-%202022%20-%20Stable%20Diffusion%20Launch%20Announcement.md)：精修摘要；Stable Diffusion 发布公告记录潜空间扩散模型进入公开权重与代码阶段，适合了解早期使用入口。
- [Black Forest Labs - 2026 - FLUX.2 Overview](./wiki/summaries/Black%20Forest%20Labs%20-%202026%20-%20FLUX.2%20Overview.md)：精修摘要；FLUX.2 官方概览将图像生成、编辑和多参考图控制组织成一个家族，供读者按创作需求辨认不同入口。
- [Gabeur et al. - 2026 - Image Generators are Generalist Vision Learners](./wiki/summaries/Gabeur%20et%20al.%20-%202026%20-%20Image%20Generators%20are%20Generalist%20Vision%20Learners.md)：精修摘要；Vision Banana 把分割、深度等视觉任务的输出表示成图像，研究图像生成模型能否兼做视觉理解。
- [Qwen Team - 2025 - Qwen-Image Crafting with Native Text Rendering](./wiki/summaries/Qwen%20Team%20-%202025%20-%20Qwen-Image%20Crafting%20with%20Native%20Text%20Rendering.md)：精修摘要；Qwen-Image 以文字渲染和图像编辑为重点，适合研究生成图中的中英文文字怎样更可控。
- [Wang et al. - 2025 - AlphaVAE Unified End-to-End RGBA Image Reconstruction and Generation with Alpha-Aware Representation Learning](./wiki/summaries/Wang%20et%20al.%20-%202025%20-%20AlphaVAE%20Unified%20End-to-End%20RGBA%20Image%20Reconstruction%20and%20Generation%20with%20Alpha-Aware%20Representation%20Learning.md)：精修摘要；AlphaVAE 联合编码颜色与透明度，为透明图像重建和生成提供统一潜表示及评测。
- [Yin et al. - 2025 - Qwen-Image-Layered Towards Inherent Editability via Layer Decomposition](./wiki/summaries/Yin%20et%20al.%20-%202025%20-%20Qwen-Image-Layered%20Towards%20Inherent%20Editability%20via%20Layer%20Decomposition.md)：精修摘要；Qwen-Image-Layered 把一张图片拆成多个语义图层，让移动、改色等操作尽量只影响目标层。
- [Yang et al. - 2025 - Generative Image Layer Decomposition with Visual Effects](./wiki/summaries/Yang%20et%20al.%20-%202025%20-%20Generative%20Image%20Layer%20Decomposition%20with%20Visual%20Effects.md)：精修摘要；LayerDecomp 将图像拆为干净背景和带透明效果的前景，帮助移动物体时保留阴影与反射。
- [Luo et al. - 2024 - IntrinsicDiffusion Joint Intrinsic Layers from Latent Diffusion Models](./wiki/summaries/Luo%20et%20al.%20-%202024%20-%20IntrinsicDiffusion%20Joint%20Intrinsic%20Layers%20from%20Latent%20Diffusion%20Models.md)：精修摘要；IntrinsicDiffusion 将图像分解为材质颜色、照明和几何等内在因素，研究可控的物理属性编辑。
- [Khan et al. - 2026 - Step-by-step Layered Design Generation](./wiki/summaries/Khan%20et%20al.%20-%202026%20-%20Step-by-step%20Layered%20Design%20Generation.md)：精修摘要；SLEDGE 把设计过程拆成逐步叠加的图层更新，研究如何按连续指令修改画布。
- [Maruani et al. - 2026 - Illustrator's Depth Monocular Layer Index Prediction for Image Decomposition](./wiki/summaries/Maruani%20et%20al.%20-%202026%20-%20Illustrator's%20Depth%20Monocular%20Layer%20Index%20Prediction%20for%20Image%20Decomposition.md)：精修摘要；Illustrator's Depth 为像素预测有序图层编号，帮助把插画拆成可以重新排列和编辑的层。

### LLM RL

- [Baker et al. - Unknown - Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation](./wiki/summaries/Baker%20et%20al.%20-%20Unknown%20-%20Monitoring%20Reasoning%20Models%20for%20Misbehavior%20and%20the%20Risks%20of%20Promoting%20Obfuscation.md)：精修摘要；论文研究如何通过另一个模型观察推理过程来发现奖励作弊，也提醒：直接惩罚可见坏念头，可能让模型把意图藏起来。
- [Hu et al. - 2024 - MiniCPM Unveiling the Potential of Small Language Models with Scalable Training Strategies](./wiki/summaries/Hu%20et%20al.%20-%202024%20-%20MiniCPM%20Unveiling%20the%20Potential%20of%20Small%20Language%20Models%20with%20Scalable%20Training%20Strategies.md)：精修摘要；MiniCPM 研究如何把小语言模型训练得更充分，用规模实验和学习率安排提高有限参数预算下的能力。
- [Jiang et al. - 2025 - MME-CoT Benchmarking Chain-of-Thought in Large Multimodal Models for Reasoning Quality, Robustness, and Efficiency](./wiki/summaries/Jiang%20et%20al.%20-%202025%20-%20MME-CoT%20Benchmarking%20Chain-of-Thought%20in%20Large%20Multimodal%20Models%20for%20Reasoning%20Quality,%20Robustness,%20and%20Efficiency.md)：精修摘要；MME-CoT 分别评估多模态模型推理过程的质量、稳健性和效率，研究“让模型多想几步”是否总有帮助。
- [Jiang, Lu - Unknown - InfiniteYou Flexible Photo Recrafting While Preserving Your Identity](./wiki/summaries/Jiang,%20Lu%20-%20Unknown%20-%20InfiniteYou%20Flexible%20Photo%20Recrafting%20While%20Preserving%20Your%20Identity.md)：精修摘要；InfiniteYou 研究保持人物身份的图像再创作，让同一个人能出现在不同场景或风格中，同时关注文本匹配与画面质量。
- [OpenAI et al. - 2019 - Dota 2 with Large Scale Deep Reinforcement Learning](./wiki/summaries/OpenAI%20et%20al.%20-%202019%20-%20Dota%202%20with%20Large%20Scale%20Deep%20Reinforcement%20Learning.md)：精修摘要；OpenAI Five 把已有强化学习方法扩大到复杂的 Dota 2 环境，研究长时间决策、部分可见信息与大规模训练。
- [Ouyang et al. - 2022 - Training language models to follow instructions with human feedback](./wiki/summaries/Ouyang%20et%20al.%20-%202022%20-%20Training%20language%20models%20to%20follow%20instructions%20with%20human%20feedback.md)：精修摘要；InstructGPT 先学习示范，再学习人类偏好并做强化学习，研究让语言模型更符合用户意图。
- [Rafailov et al. - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model](./wiki/summaries/Rafailov%20et%20al.%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)：精修摘要；DPO 直接利用“偏好回答与不偏好回答”的成对数据训练语言模型，简化显式奖励模型和在线强化学习的部分流程。
- [Rafailov, Mitchell, Jul - 2023 - Direct Preference Optimization Your Language Model is Secretly a Reward Model](./wiki/summaries/Rafailov,%20Mitchell,%20Jul%20-%202023%20-%20Direct%20Preference%20Optimization%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.md)：精修摘要；同源归档；本页是 DPO 的另一份归档：核心仍是用成对偏好直接优化模型，阅读前应与同 arXiv ID 的来源核对版本。
- [Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model](./wiki/summaries/Hong%20et%20al.%20-%202024%20-%20ORPO%20Monolithic%20Preference%20Optimization%20without%20Reference%20Model.md)：精修摘要；ORPO 把示范学习和偏好学习放进同一训练目标，并尝试省去单独的参考模型。
- [Ethayarajh et al. - 2024 - KTO Model Alignment as Prospect Theoretic Optimization](./wiki/summaries/Ethayarajh%20et%20al.%20-%202024%20-%20KTO%20Model%20Alignment%20as%20Prospect%20Theoretic%20Optimization.md)：精修摘要；KTO 使用“这个回答好或不好”的反馈训练模型，适合研究缺少成对偏好数据时怎样做对齐。
- [Shao et al. - 2024 - DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models](./wiki/summaries/Shao%20et%20al.%20-%202024%20-%20DeepSeekMath%20Pushing%20the%20Limits%20of%20Mathematical%20Reasoning%20in%20Open%20Language%20Models.md)：精修摘要；DeepSeekMath 同时研究数学语料和强化学习，其中 GRPO 用同题多份回答的相对分数估计更新参照。
- [DeepSeek-R1：奖励驱动推理与多阶段训练（2025）](./wiki/summaries/Unknown%20-%202024%20-%20DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.md)：精修摘要；DeepSeek-R1 区分纯强化学习探索的 R1-Zero 与加入冷启动、多阶段训练的 R1，并将推理能力蒸馏到较小模型。
- [DeepSeek AI - 2025 - DeepSeek-R1-0528 Release](./wiki/summaries/DeepSeek%20AI%20-%202025%20-%20DeepSeek-R1-0528%20Release.md)：精修摘要；R1-0528 是 R1 的更新发布页，说明推理模型怎样继续改善交互、结构化输出和工具调用。
- [DeepSeek AI - 2025 - DeepSeek-V3.2 Release](./wiki/summaries/DeepSeek%20AI%20-%202025%20-%20DeepSeek-V3.2%20Release.md)：精修摘要；DeepSeek-V3.2 的发布页强调把思考过程接进工具使用，使模型能在推理和执行之间持续推进任务。
- [Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale](./wiki/summaries/Yu%20et%20al.%20-%202025%20-%20DAPO%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.md)：精修摘要；DAPO 将长推理强化学习当作系统问题，分别处理探索收缩、采样、损失权重和过长回答。
- [Yang et al. - 2026 - Learning beyond Teacher Generalized On-Policy Distillation with Reward Extrapolation](./wiki/summaries/Yang%20et%20al.%20-%202026%20-%20Learning%20beyond%20Teacher%20Generalized%20On-Policy%20Distillation%20with%20Reward%20Extrapolation.md)：精修摘要；G-OPD 让学生在自己的回答上学习教师分布，并调整奖励与约束的相对权重，研究更一般的在线蒸馏。
- [Smock, Pesala, Abraham - 2023 - Aligning Benchmark Datasets for Table Structure Recognition](./wiki/summaries/Smock,%20Pesala,%20Abraham%20-%202023%20-%20Aligning%20Benchmark%20Datasets%20for%20Table%20Structure%20Recognition.md)：精修摘要；这篇工作对齐表格基准中的错误与不一致标注，说明评测数据的处理方式也会影响模型比较。
- [Tay et al. - 2020 - Efficient Transformers A Survey](./wiki/summaries/Tay%20et%20al.%20-%202020%20-%20Efficient%20Transformers%20A%20Survey.md)：精修摘要；《Efficient Transformers》按不同计算与内存瓶颈整理高效 Transformer 方法，帮助分辨各类“更快注意力”到底改了什么。
- [Hermes 3：开放模型的指令与行为训练（2024）](./wiki/summaries/Teknium,%20Quesnelle,%20Guang%20-%20Unknown%20-%20arXiv%202408%20.%2011857v1%20cs%20.%20CL%2015%20Aug%202024.md)：精修摘要；本地原文是 Hermes 3 技术报告，讨论指令与工具使用模型；旧标题只是 arXiv 页眉，需要按实际报告内容阅读。
- [Wang et al. - 2025 - VRAG-RL Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning wit](./wiki/summaries/Wang%20et%20al.%20-%202025%20-%20VRAG-RL%20Empower%20Vision-Perception-Based%20RAG%20for%20Visually%20Rich%20Information%20Understanding%20via%20Iterative%20Reasoning%20wit.md)：精修摘要；VRAG-RL 研究用强化学习改善视觉检索增强的迭代推理，让模型围绕视觉证据选择下一步，而非固定一次检索后回答。
- [Wu et al. - 2021 - Recursively Summarizing Books with Human Feedback](./wiki/summaries/Wu%20et%20al.%20-%202021%20-%20Recursively%20Summarizing%20Books%20with%20Human%20Feedback.md)：精修摘要；这篇整本书摘要工作把长任务递归拆成小摘要，再用人类反馈改善各层结果，研究超长材料怎样逐步压缩。
- [Wu et al. - 2025 - On the Generalization of SFT A Reinforcement Learning Perspective with Reward Rectification](./wiki/summaries/Wu%20et%20al.%20-%202025%20-%20On%20the%20Generalization%20of%20SFT%20A%20Reinforcement%20Learning%20Perspective%20with%20Reward%20Rectification.md)：精修摘要；DFT 从强化学习视角分析监督微调的泛化，并根据词元概率调整训练目标，探索对标准 SFT 的简洁改进。
- [Yang et al. - 2021 - Robust Transformer Modeling for Table-Text Encoding](./wiki/summaries/Yang%20et%20al.%20-%202021%20-%20Robust%20Transformer%20Modeling%20for%20Table-Text%20Encoding.md)：精修摘要；这篇表格文本模型研究减少行列顺序带来的虚假偏差，让表示更稳健地利用表格结构与文字关系。
- [Yao et al. - 2024 - MiniCPM-V A GPT-4V Level MLLM on Your Phone](./wiki/summaries/Yao%20et%20al.%20-%202024%20-%20MiniCPM-V%20A%20GPT-4V%20Level%20MLLM%20on%20Your%20Phone.md)：精修摘要；MiniCPM-V 面向更轻量的视觉语言部署，研究怎样在有限模型规模下提供图像理解能力，成本和质量都需按设备测试。

### 传统NLP

- [Sutskever, Vinyals, Le - 2014 - Sequence to Sequence Learning with Neural Networks](./wiki/summaries/Sutskever,%20Vinyals,%20Le%20-%202014%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.md)：精修摘要；早期 Seq2Seq 用编码器读取输入序列，再用解码器逐步生成输出，为翻译等任务建立统一接口。
- [Bekoulis et al. - 2018 - Joint entity recognition and relation extraction as a multi-head selection problem](./wiki/summaries/Bekoulis%20et%20al.%20-%202018%20-%20Joint%20entity%20recognition%20and%20relation%20extraction%20as%20a%20multi-head%20selection%20problem.md)：精修摘要；这篇信息抽取方法把实体识别和关系抽取一起建模，让词语可以选择多个关系对象，减少对外部语法工具的依赖。
- [Bommasani et al. - 2021 - On the Opportunities and Risks of Foundation Models](./wiki/summaries/Bommasani%20et%20al.%20-%202021%20-%20On%20the%20Opportunities%20and%20Risks%20of%20Foundation%20Models.md)：精修摘要；这份报告把基础模型看作可适配多种任务的共同底座，讨论其技术机会与社会风险；它适合建立问题地图。
- [Conneau - 2021 - Larger-Scale Transformers for Multilingual Masked Language Modeling](./wiki/summaries/Conneau%20-%202021%20-%20Larger-Scale%20Transformers%20for%20Multilingual%20Masked%20Language%20Modeling.md)：精修摘要；XLM-RXL 和 XLM-RXXL 研究扩大多语言遮挡语言模型的收益：增加容量能改善部分跨语言理解任务，但评测条件仍重要。
- [Corporation - 2022 - NVIDIA DGX A100 The Universal System for AI Infrastructure](./wiki/summaries/Corporation%20-%202022%20-%20NVIDIA%20DGX%20A100%20The%20Universal%20System%20for%20AI%20Infrastructure.md)：精修摘要；DGX A100 是把 GPU、互连和软件组合成 AI 计算平台的产品资料，适合理解系统组成；功能说明与实测性能需要区分。
- [Dean, Scientist, Deepmind - Unknown - Important Trends in AI How Did We Get Here , What Can We Do Now and How Can We Shape AI ’ s Fut](./wiki/summaries/Dean,%20Scientist,%20Deepmind%20-%20Unknown%20-%20Important%20Trends%20in%20AI%20How%20Did%20We%20Get%20Here%20,%20What%20Can%20We%20Do%20Now%20and%20How%20Can%20We%20Shape%20AI%20’%20s%20Fut.md)：精修摘要；Jeff Dean 的演讲串起神经网络、规模化训练和计算硬件的变化，适合了解 AI 进步由哪些因素共同推动。
- [Dozat, Manning - 2017 - Deep biaffine attention for neural dependency parsing](./wiki/summaries/Dozat,%20Manning%20-%202017%20-%20Deep%20biaffine%20attention%20for%20neural%20dependency%20parsing.md)：精修摘要；双仿射依存分析器给词语之间的语法连接打分，再预测连接类型，用较简洁的结构构建句法树。
- [Kong, Kim, Bae - 2020 - HiFi-GAN Generative adversarial networks for efficient and high fidelity speech synthesis](./wiki/summaries/Kong,%20Kim,%20Bae%20-%202020%20-%20HiFi-GAN%20Generative%20adversarial%20networks%20for%20efficient%20and%20high%20fidelity%20speech%20synthesis.md)：精修摘要；HiFi-GAN 用生成对抗网络把声学特征转成波形，结合不同尺度和周期的判别器，追求语音质量与生成效率。
- [Krishnamoorthi - 2018 - Quantizing deep convolutional networks for efficient inference A whitepaper](./wiki/summaries/Krishnamoorthi%20-%202018%20-%20Quantizing%20deep%20convolutional%20networks%20for%20efficient%20inference%20A%20whitepaper.md)：精修摘要；这份量化白皮书比较卷积网络的整数推理方案，包括训练后量化和量化感知训练，重点是精度与部署成本的折中。
- [Kumar et al. - 2019 - MelGAN Generative adversarial networks for conditional waveform synthesis](./wiki/summaries/Kumar%20et%20al.%20-%202019%20-%20MelGAN%20Generative%20adversarial%20networks%20for%20conditional%20waveform%20synthesis.md)：精修摘要；MelGAN 用对抗训练从梅尔频谱生成声音波形，探索比逐点自回归生成更高效的声码器路线。
- [Liang et al. - 2022 - Holistic Evaluation of Language Models](./wiki/summaries/Liang%20et%20al.%20-%202022%20-%20Holistic%20Evaluation%20of%20Language%20Models.md)：精修摘要；HELM 主张从准确率、稳健性、公平性和效率等多个维度评估语言模型，用统一场景呈现能力与代价。
- [Liu - 2019 - Fine-tune BERT for Extractive Summarization](./wiki/summaries/Liu%20-%202019%20-%20Fine-tune%20BERT%20for%20Extractive%20Summarization.md)：精修摘要；BERTSUM 用 BERT 编码文档并选择重要句子生成抽取式摘要，输出主要来自原文，而不是自由改写。
- [Liu et al. - 2019 - RoBERTa A Robustly Optimized BERT Pretraining Approach](./wiki/summaries/Liu%20et%20al.%20-%202019%20-%20RoBERTa%20A%20Robustly%20Optimized%20BERT%20Pretraining%20Approach.md)：精修摘要；RoBERTa 重新检查 BERT 的训练配方，说明训练量、数据和设置的变化本身就能带来显著效果差异。
- [8-bit Inference with TensorRT：INT8 校准与推理（2017）](./wiki/summaries/Migacz%20-%202017%20-%20Intro.md)：精修摘要；这是 NVIDIA 2017 年的 TensorRT INT8 推理讲稿，重点是量化范围与精度的取舍，适合作为历史方法参考。
- [Pradhan, Moschitti, Uryupina - 2012 - CoNLL-2012 Shared Task Modeling Multilingual Unrestricted Coreference in OntoNotes](./wiki/summaries/Pradhan,%20Moschitti,%20Uryupina%20-%202012%20-%20CoNLL-2012%20Shared%20Task%20Modeling%20Multilingual%20Unrestricted%20Coreference%20in%20OntoNotes.md)：精修摘要；CoNLL-2012 在 OntoNotes 上评估英文、中文和阿拉伯文共指消解，为跨语言指代研究提供统一任务与数据。
- [Sakata et al. - 2019 - FAQ retrieval using query-question similarity and BERT-based query-answer relevance](./wiki/summaries/Sakata%20et%20al.%20-%202019%20-%20FAQ%20retrieval%20using%20query-question%20similarity%20and%20BERT-based%20query-answer%20relevance.md)：精修摘要；FAQ 检索既要比较用户问题与已有问题，也要判断候选答案能否解决当前提问。
- [A Deep Look into Neural Ranking Models for Information Retrieval（Guo 等，2019）](./wiki/summaries/Mitra,%20Craswell%20-%202019%20-%20A%20Deep%20Look%20into%20Neural%20Ranking%20Models%20for%20Information%20Retrieval.md)：精修摘要；这篇神经排序综述解释搜索系统怎样表示问题和文档、让两者交互，以及如何评价相关性与效率。
- [Nogueira, Cho - 2019 - Passage Re-ranking with BERT](./wiki/summaries/Nogueira,%20Cho%20-%202019%20-%20Passage%20Re-ranking%20with%20BERT.md)：精修摘要；BERT 重排序将问题和候选片段一起输入模型，直接判断相关性，适合在初步召回之后精细筛选。
- [Khattab, Zaharia - 2020 - ColBERT Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](./wiki/summaries/Khattab,%20Zaharia%20-%202020%20-%20ColBERT%20Efficient%20and%20Effective%20Passage%20Search%20via%20Contextualized%20Late%20Interaction%20over%20BERT.md)：精修摘要；ColBERT 先独立编码问题和文档，再比较细粒度词元表示，在检索成本与匹配精细度之间折中。
- [Pretrained Transformers for Text Ranking: BERT and Beyond（Lin、Nogueira、Yates）](./wiki/summaries/Nogueira%20et%20al.%20-%202020%20-%20Pretrained%20Transformers%20for%20Text%20Ranking%20BERT%20and%20Beyond.md)：精修摘要；这篇 Transformer 排序综述同时整理重排序和向量检索，帮助理解速度、存储与精细匹配的取舍。
- [Schick et al. - 2023 - Toolformer Language Models Can Teach Themselves to Use Tools](./wiki/summaries/Schick%20et%20al.%20-%202023%20-%20Toolformer%20Language%20Models%20Can%20Teach%20Themselves%20to%20Use%20Tools.md)：精修摘要；Toolformer 用少量示范和自监督筛选，让语言模型学习何时调用工具、传什么参数以及如何使用结果。
- [Sciavolino et al. - 2021 - Simple Entity-Centric Questions Challenge Dense Retrievers](./wiki/summaries/Sciavolino%20et%20al.%20-%202021%20-%20Simple%20Entity-Centric%20Questions%20Challenge%20Dense%20Retrievers.md)：精修摘要；EntityQuestions 发现一些看似简单、围绕具体实体的问题会难倒稠密检索器，提醒检索能力受训练分布影响。
- [Shen et al. - 2018 - Natural TTS Synthesis by Conditioning Wavenet on MEL Spectrogram Predictions](./wiki/summaries/Shen%20et%20al.%20-%202018%20-%20Natural%20TTS%20Synthesis%20by%20Conditioning%20Wavenet%20on%20MEL%20Spectrogram%20Predictions.md)：精修摘要；Tacotron 2 先把文字变成梅尔频谱，再用 WaveNet 生成声音波形，把文字到语音分成两个可理解的阶段。
- [Wang et al. - 2017 - Tacotron Towards end-To-end speech synthesis](./wiki/summaries/Wang%20et%20al.%20-%202017%20-%20Tacotron%20Towards%20end-To-end%20speech%20synthesis.md)：精修摘要；Tacotron 从字符直接预测语音声学表示，减少传统文字转语音系统中大量手工模块，是端到端 TTS 的代表路线。
- [Wang et al. - 2022 - HPT Hierarchy-aware Prompt Tuning for Hierarchical Text Classification](./wiki/summaries/Wang%20et%20al.%20-%202022%20-%20HPT%20Hierarchy-aware%20Prompt%20Tuning%20for%20Hierarchical%20Text%20Classification.md)：精修摘要；HPT 把标签层级纳入提示微调，让分类模型利用父子类别关系，研究预训练目标与层级分类之间的衔接。
- [Wang et al. - 2024 - UniMERNet A Universal Network for Real-World Mathematical Expression Recognition](./wiki/summaries/Wang%20et%20al.%20-%202024%20-%20UniMERNet%20A%20Universal%20Network%20for%20Real-World%20Mathematical%20Expression%20Recognition.md)：精修摘要；UniMERNet 面向真实场景的数学表达识别，配合多样训练和测试数据，研究复杂公式的转写能力。
- [Wei, Zou - 2019 - EDA Easy data augmentation techniques for boosting performance on text classification tasks](./wiki/summaries/Wei,%20Zou%20-%202019%20-%20EDA%20Easy%20data%20augmentation%20techniques%20for%20boosting%20performance%20on%20text%20classification%20tasks.md)：精修摘要；EDA 用同义替换、随机插入、交换和删除扩充文本分类数据，方法简单，适合研究小数据场景的增强。
- [Xue et al. - 2021 - mT5 A Massively Multilingual Pre-trained Text-to-Text Transformer](./wiki/summaries/Xue%20et%20al.%20-%202021%20-%20mT5%20A%20Massively%20Multilingual%20Pre-trained%20Text-to-Text%20Transformer.md)：精修摘要；mT5 把 T5 的文本到文本接口扩展到多语言预训练，让同一框架处理不同语言的理解和生成任务。
- [FastPitch：可控音高的并行文字转语音（2020 预印本）](./wiki/summaries/Ła%20-%20Unknown%20-%20FASTPITCH%20PARALLEL%20TEXT-TO-SPEECH%20WITH%20PITCH%20PREDICTION%20Adrian%20Ła´%20ncucki%20NVIDIA%20Corporation.md)：精修摘要；FastPitch 并行预测语音并显式建模音高，使合成过程更快，也提供调整语音表达的控制量。

### 传统CV

- [Ahead - 2024 - Leopold Aschenbrenner S I T U AT I O N A L AWA R E N E S S The Decade Ahead](./wiki/summaries/Ahead%20-%202024%20-%20Leopold%20Aschenbrenner%20S%20I%20T%20U%20AT%20I%20O%20N%20A%20L%20AWA%20R%20E%20N%20E%20S%20S%20The%20Decade%20Ahead.md)：精修摘要；这是 Leopold Aschenbrenner 在 2024 年提出的 AI 发展情景与政策主张，适合研究其论证链和假设，不应当作已验证的时间表。
- [Alabdulmohsin et al. - 2023 - Getting ViT in Shape Scaling Laws for Compute-Optimal Model Design](./wiki/summaries/Alabdulmohsin%20et%20al.%20-%202023%20-%20Getting%20ViT%20in%20Shape%20Scaling%20Laws%20for%20Compute-Optimal%20Model%20Design.md)：精修摘要；SoViT 研究同样的训练预算该怎样分配给 ViT 的宽度和深度；模型的形状也影响效率，参数总量不能解释一切。
- [Blecher et al. - 2023 - Nougat Neural Optical Understanding for Academic Documents](./wiki/summaries/Blecher%20et%20al.%20-%202023%20-%20Nougat%20Neural%20Optical%20Understanding%20for%20Academic%20Documents.md)：精修摘要；Nougat 把学术 PDF 页面转成带结构的标记文本，重点是保留论文中的数学表达；它提供转写材料，阅读者仍需核对。
- [Bubeck et al. - 2023 - Sparks of Artificial General Intelligence Early experiments with GPT-4](./wiki/summaries/Bubeck%20et%20al.%20-%202023%20-%20Sparks%20of%20Artificial%20General%20Intelligence%20Early%20experiments%20with%20GPT-4.md)：精修摘要；《Sparks of AGI》记录早期 GPT-4 在多类任务上的探索性实验；它能展示能力现象，不能单独证明模型具有通用智能。
- [Carion et al. - 2020 - End-to-End Object Detection with Transformers](./wiki/summaries/Carion%20et%20al.%20-%202020%20-%20End-to-End%20Object%20Detection%20with%20Transformers.md)：精修摘要；DETR 把目标检测写成一组对象的预测，利用匹配训练减少手工设计的候选框和去重步骤，展示了端到端检测路线。
- [Ren et al. - 2015 - Faster R-CNN Towards Real-Time Object Detection with Region Proposal Networks](./wiki/summaries/Ren%20et%20al.%20-%202015%20-%20Faster%20R-CNN%20Towards%20Real-Time%20Object%20Detection%20with%20Region%20Proposal%20Networks.md)：精修摘要；Faster R-CNN 用可学习网络产生候选区域，并与后续分类定位共享特征，减少两阶段检测的额外开销。
- [Chen et al. - 2020 - What comprises a good talking-head video generation A Survey and Benchmark](./wiki/summaries/Chen%20et%20al.%20-%202020%20-%20What%20comprises%20a%20good%20talking-head%20video%20generation%20A%20Survey%20and%20Benchmark.md)：精修摘要；这份说话人视频综述与基准把评测拆成可重复的流程，提醒口型、画质和动作自然度需要分别观察。
- [Sentence-BERT：孪生编码器句向量（Reimers 与 Gurevych，2019）](./wiki/summaries/Devlin,%20Liu%20-%202014%20-%20Sentence-BERT%20Sentence%20Embeddings%20using%20Siamese%20BERT-Networks.md)：精修摘要；Sentence-BERT 把句子各自编码成向量，再比较向量相似度，避免为每一对句子都运行一次完整 BERT。
- [Ding et al. - 2024 - Using the divergent association task to measure divergent thinking in Chinese elementary school students](./wiki/summaries/Ding%20et%20al.%20-%202024%20-%20Using%20the%20divergent%20association%20task%20to%20measure%20divergent%20thinking%20in%20Chinese%20elementary%20school%20students.md)：精修摘要；这篇研究用词语之间的语义距离评估中国小学生的发散联想表现；量化联想差异与全面判断创造力仍是两件事。
- [Dong et al. - 2019 - Unified language model pre-training for natural language understanding and generation](./wiki/summaries/Dong%20et%20al.%20-%202019%20-%20Unified%20language%20model%20pre-training%20for%20natural%20language%20understanding%20and%20generation.md)：精修摘要；UniLM 用不同注意力遮挡方式，让共享 Transformer 学习单向、双向和序列到序列任务，连接语言理解与生成。
- [Dosovitskiy et al. - 2020 - An Image is Worth 16x16 Words Transformers for Image Recognition at Scale](./wiki/summaries/Dosovitskiy%20et%20al.%20-%202020%20-%20An%20Image%20is%20Worth%2016x16%20Words%20Transformers%20for%20Image%20Recognition%20at%20Scale.md)：精修摘要；ViT 把图像切成小块，将图块当成序列输入 Transformer，说明图像分类也可以通过序列建模来完成。
- [Dumas, Organisciak, Doherty - 2020 - Measuring Divergent Thinking Originality With Human Raters and Text-Mining Models A Psychometric Co](./wiki/summaries/Dumas,%20Organisciak,%20Doherty%20-%202020%20-%20Measuring%20Divergent%20Thinking%20Originality%20With%20Human%20Raters%20and%20Text-Mining%20Models%20A%20Psychometric%20Co.md)：精修摘要；这篇心理测量研究比较人工评分与文本挖掘对发散思维原创性的评价，重点是自动分数是否测到了同一种能力。
- [Fedus, Zoph, Shazeer - 2022 - Switch Transformers Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](./wiki/summaries/Fedus,%20Zoph,%20Shazeer%20-%202022%20-%20Switch%20Transformers%20Scaling%20to%20Trillion%20Parameter%20Models%20with%20Simple%20and%20Efficient%20Sparsity.md)：精修摘要；Switch Transformer 让每个输入只路由到少数专家，扩大模型总容量，同时控制每个词元实际参与的计算。
- [Gordon, Duh, Andrews - 2020 - Compressing BERT Studying the Effects of Weight Pruning on Transfer Learning](./wiki/summaries/Gordon,%20Duh,%20Andrews%20-%202020%20-%20Compressing%20BERT%20Studying%20the%20Effects%20of%20Weight%20Pruning%20on%20Transfer%20Learning.md)：精修摘要；这篇 BERT 剪枝研究发现，压掉权重对预训练和下游迁移的影响会随剪枝程度变化，不能只用模型压缩率判断效果。
- [Herzig et al. - 2020 - TaPas Weakly Supervised Table Parsing via Pre-training](./wiki/summaries/Herzig%20et%20al.%20-%202020%20-%20TaPas%20Weakly%20Supervised%20Table%20Parsing%20via%20Pre-training.md)：精修摘要；TAPAS 回答表格问题时预测相关单元格和聚合操作，利用答案等弱监督信号训练，减少完整查询程序标注的需求。
- [Khan et al. - 2021 - Transformers in Vision A Survey](./wiki/summaries/Khan%20et%20al.%20-%202021%20-%20Transformers%20in%20Vision%20A%20Survey.md)：精修摘要；这篇综述整理 Transformer 在视觉任务中的用法，帮助比较全局关系建模、图像表示和计算成本。
- [Kim et al. - 2021 - I-BERT Integer-only BERT Quantization](./wiki/summaries/Kim%20et%20al.%20-%202021%20-%20I-BERT%20Integer-only%20BERT%20Quantization.md)：精修摘要；I-BERT 研究只用整数运算运行 BERT，不仅量化权重，也处理非线性运算，以适应高效推理硬件。
- [Kim et al. - Unknown - Full Stack Optimization of Transformer Inference a Survey](./wiki/summaries/Kim%20et%20al.%20-%20Unknown%20-%20Full%20Stack%20Optimization%20of%20Transformer%20Inference%20a%20Survey.md)：精修摘要；这篇综述从模型、软件到硬件整理 Transformer 推理优化，说明速度问题需要沿整条执行链定位。
- [Li et al. - 2021 - TrOCR Transformer-based Optical Character Recognition with Pre-trained Models](./wiki/summaries/Li%20et%20al.%20-%202021%20-%20TrOCR%20Transformer-based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.md)：精修摘要；TrOCR 将预训练图像 Transformer 与文本 Transformer 组合，直接把文字图像解码成文本，研究端到端文字识别。
- [Li et al. - 2022 - DiT Self-supervised Pre-training for Document Image Transformer](./wiki/summaries/Li%20et%20al.%20-%202022%20-%20DiT%20Self-supervised%20Pre-training%20for%20Document%20Image%20Transformer.md)：精修摘要；文档 DiT 在大量未标注文档图像上自监督预训练，为版面分析等文档视觉任务提供表示底座。
- [Li et al. - 2023 - TrOCR Transformer-Based Optical Character Recognition with Pre-trained Models](./wiki/summaries/Li%20et%20al.%20-%202023%20-%20TrOCR%20Transformer-Based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.md)：精修摘要；同源归档；这份 TrOCR 来源讨论利用预训练编码器和解码器识别文字，与库中另一份 TrOCR 来源可能是同一工作的不同版本。
- [Li et al. - 2025 - dots.ocr Multilingual Document Layout Parsing in a Single Vision-Language Model](./wiki/summaries/Li%20et%20al.%20-%202025%20-%20dots.ocr%20Multilingual%20Document%20Layout%20Parsing%20in%20a%20Single%20Vision-Language%20Model.md)：精修摘要；dots.ocr 用同一视觉语言模型理解文档版面、识别内容和组织阅读关系，研究减少多阶段误差累积。
- [Liang et al. - 2021 - R-Drop Regularized Dropout for Neural Networks arXiv 2106 . 14448v2 cs . LG 29 Oct 2021](./wiki/summaries/Liang%20et%20al.%20-%202021%20-%20R-Drop%20Regularized%20Dropout%20for%20Neural%20Networks%20arXiv%202106%20.%2014448v2%20cs%20.%20LG%2029%20Oct%202021.md)：精修摘要；R-Drop 让同一输入经过两次不同 dropout 后，输出分布仍保持一致，用额外一致性约束改善训练。
- [Lin et al. - 2021 - A Survey of Transformers](./wiki/summaries/Lin%20et%20al.%20-%202021%20-%20A%20Survey%20of%20Transformers.md)：精修摘要；这篇 Transformer 综述整理架构改进、预训练和应用，适合建立方法分类，再沿具体论文深入阅读。
- [Lin et al. - 2021 - Real-Time High-Resolution Background Matting](./wiki/summaries/Lin%20et%20al.%20-%202021%20-%20Real-Time%20High-Resolution%20Background%20Matting.md)：精修摘要；Background Matting 使用额外拍摄的干净背景，估计前景与透明度，以支持高分辨率的人像背景替换。
- [Lin et al. - 2022 - Robust High-Resolution Video Matting with Temporal Guidance](./wiki/summaries/Lin%20et%20al.%20-%202022%20-%20Robust%20High-Resolution%20Video%20Matting%20with%20Temporal%20Guidance.md)：精修摘要；Robust Video Matting 利用视频前后帧的时间信息进行人像抠图，减少逐帧独立处理导致的边缘闪烁。
- [Lipman et al. - 2024 - Flow Matching Guide and Code](./wiki/summaries/Lipman%20et%20al.%20-%202024%20-%20Flow%20Matching%20Guide%20and%20Code.md)：精修摘要；Flow Matching 指南介绍通过学习向量场进行生成建模的训练与采样方式，适合从统一框架理解图像、音频等生成路线。
- [Liu et al. - 2021 - Fast, Effective, and Self-Supervised Transforming Masked Language Models into Universal Lexical and Sentence Encoder](./wiki/summaries/Liu%20et%20al.%20-%202021%20-%20Fast,%20Effective,%20and%20Self-Supervised%20Transforming%20Masked%20Language%20Models%20into%20Universal%20Lexical%20and%20Sentence%20Encoder.md)：精修摘要；这篇工作研究把遮挡语言模型转成通用词语与句子编码器，尽量利用已有模型与语料，减少新增标注数据。
- [Long et al. - 2024 - LORE Logical Location Regression Network for Table Structure Recognition with Pre-training](./wiki/summaries/Long%20et%20al.%20-%202024%20-%20LORE%20Logical%20Location%20Regression%20Network%20for%20Table%20Structure%20Recognition%20with%20Pre-training.md)：精修摘要；这份 LORE 来源研究直接预测单元格的逻辑行列位置，并结合预训练改进表格结构识别。
- [Lu et al. - 2024 - Large Language Model for Table Processing A Survey](./wiki/summaries/Lu%20et%20al.%20-%202024%20-%20Large%20Language%20Model%20for%20Table%20Processing%20A%20Survey.md)：精修摘要；这篇综述整理 LLM 处理表格的任务和方法，重点是二维结构怎样输入模型，以及模型怎样查询、理解和操作它。
- [Instant Neural Graphics Primitives：多分辨率哈希编码（2022）](./wiki/summaries/Müller%20et%20al.%20-%202021%20-%20Real-time%20neural%20radiance%20caching%20for%20path%20tracing.md)：精修摘要；现有原文实际是 Instant-NGP：用多分辨率哈希特征和小网络加速神经图形表示。旧文件名误写成了神经辐射缓存论文。
- [Nassar et al. - 2025 - SmolDocling An ultra-compact vision-language model for arXiv 2503 . 11576v1 cs . CV 14 Mar 2025](./wiki/summaries/Nassar%20et%20al.%20-%202025%20-%20SmolDocling%20An%20ultra-compact%20vision-language%20model%20for%20arXiv%202503%20.%2011576v1%20cs%20.%20CV%2014%20Mar%202025.md)：精修摘要；SmolDocling 用较小的视觉语言模型读取整页文档，输出包含位置和页面元素的 DocTags，研究紧凑的端到端转换。
- [Nvidia - 2022 - Nvidia Ada Gpu Architecture](./wiki/summaries/Nvidia%20-%202022%20-%20Nvidia%20Ada%20Gpu%20Architecture.md)：精修摘要；这份 Ada GPU 架构白皮书介绍 NVIDIA 的图形、AI 与计算硬件设计，适合理解部件职责和历史产品能力。
- [Oğuz et al. - 2021 - Domain-matched Pre-training Tasks for Dense Retrieval](./wiki/summaries/Oğuz%20et%20al.%20-%202021%20-%20Domain-matched%20Pre-training%20Tasks%20for%20Dense%20Retrieval.md)：精修摘要；这篇检索论文研究预训练任务是否匹配实际搜索：学习语言本身不一定足以学好问题与证据的对应。
- [Peng et al. - 2023 - Kosmos-2 Grounding Multimodal Large Language Models to the World](./wiki/summaries/Peng%20et%20al.%20-%202023%20-%20Kosmos-2%20Grounding%20Multimodal%20Large%20Language%20Models%20to%20the%20World.md)：精修摘要；Kosmos-2 把语言中的对象描述与图像中的位置绑定起来，使模型生成文字时也能指出“说的是哪里”。
- [Pfitzmann et al. - 2022 - DocLayNet A Large Human-Annotated Dataset for Document-Layout Segmentation](./wiki/summaries/Pfitzmann%20et%20al.%20-%202022%20-%20DocLayNet%20A%20Large%20Human-Annotated%20Dataset%20for%20Document-Layout%20Segmentation.md)：精修摘要；DocLayNet 提供更丰富的文档版面人工标注，缓解只用学术论文训练的版面模型难以适应其他文档的问题。
- [PaddlePaddle Team et al. - 2025 - PaddleOCR 3.0 Technical Report](./wiki/summaries/PaddlePaddle%20Team%20et%20al.%20-%202025%20-%20PaddleOCR%203.0%20Technical%20Report.md)：精修摘要；PaddleOCR 3.0 将文字识别、文档结构恢复和信息抽取组织成工具链，适合按处理环节选择组件。
- [Poznanski, Wilhelm - Unknown - olmOCR Unlocking Trillions of Tokens in PDFs with Vision Language Models](./wiki/summaries/Poznanski,%20Wilhelm%20-%20Unknown%20-%20olmOCR%20Unlocking%20Trillions%20of%20Tokens%20in%20PDFs%20with%20Vision%20Language%20Models.md)：精修摘要；olmOCR 把 PDF 解析组织成可批处理的数据工具链，重点是自然阅读顺序和结构保留，适合研究大规模文档转换。
- [Wei, Sun, Li - 2025 - DeepSeek-OCR Contexts Optical Compression](./wiki/summaries/Wei,%20Sun,%20Li%20-%202025%20-%20DeepSeek-OCR%20Contexts%20Optical%20Compression.md)：精修摘要；DeepSeek-OCR 将文档图像编码为较少视觉词元，研究视觉压缩能否节省长文本上下文。
- [Wei, Sun, Li - 2026 - DeepSeek-OCR 2 Visual Causal Flow](./wiki/summaries/Wei,%20Sun,%20Li%20-%202026%20-%20DeepSeek-OCR%202%20Visual%20Causal%20Flow.md)：精修摘要；DeepSeek-OCR 2 研究怎样按语义组织视觉词元的顺序，再交给语言模型读取复杂页面。
- [Duan et al. - 2026 - GLM-OCR Technical Report](./wiki/summaries/Duan%20et%20al.%20-%202026%20-%20GLM-OCR%20Technical%20Report.md)：精修摘要；GLM-OCR 先分析页面区域，再识别文字、公式和表格，研究紧凑模型与完整文档处理流程的配合。
- [Prajwal et al. - 2020 - A Lip Sync Expert Is All You Need for Speech to Lip Generation in the Wild](./wiki/summaries/Prajwal%20et%20al.%20-%202020%20-%20A%20Lip%20Sync%20Expert%20Is%20All%20You%20Need%20for%20Speech%20to%20Lip%20Generation%20in%20the%20Wild.md)：精修摘要；Wav2Lip 用口型同步判别器指导视频中的嘴部生成，目标是让不同人物的讲话视频与目标音频对齐。
- [Redmon et al. - 2015 - You Only Look Once Unified Real-Time Object Detection](./wiki/summaries/Redmon%20et%20al.%20-%202015%20-%20You%20Only%20Look%20Once%20Unified%20Real-Time%20Object%20Detection.md)：精修摘要；YOLOv1 一次前向计算就预测目标框和类别，把检测组织成单阶段回归任务。
- [Redmon, Farhadi - 2016 - YOLO9000 Better Faster Stronger](./wiki/summaries/Redmon,%20Farhadi%20-%202016%20-%20YOLO9000%20Better%20Faster%20Stronger.md)：精修摘要；YOLOv2 改善候选框和多尺度训练，YOLO9000 进一步结合分类与检测数据来扩大可识别类别。
- [Redmon, Farhadi - 2018 - YOLOv3 An Incremental Improvement](./wiki/summaries/Redmon,%20Farhadi%20-%202018%20-%20YOLOv3%20An%20Incremental%20Improvement.md)：精修摘要；YOLOv3 结合更强特征提取、多尺度预测与目标性判断，研究稳定的实时检测。
- [Bochkovskiy, Wang, Liao - 2020 - YOLOv4 Optimal Speed and Accuracy of Object Detection](./wiki/summaries/Bochkovskiy,%20Wang,%20Liao%20-%202020%20-%20YOLOv4%20Optimal%20Speed%20and%20Accuracy%20of%20Object%20Detection.md)：精修摘要；YOLOv4 通过组合网络结构、数据增强和训练技巧，提高单阶段目标检测的实用性。
- [Seed - Unknown - Seed1.5-VL Technical Report](./wiki/summaries/Seed%20-%20Unknown%20-%20Seed1.5-VL%20Technical%20Report.md)：精修摘要；Seed1.5-VL 将视觉编码器与专家混合语言模型结合，用于多模态理解和推理；报告的比较需保留模型与评测条件。
- [Smock, Pesala, Abraham - 2022 - GriTS Grid table similarity metric for table structure recognition](./wiki/summaries/Smock,%20Pesala,%20Abraham%20-%202022%20-%20GriTS%20Grid%20table%20similarity%20metric%20for%20table%20structure%20recognition.md)：精修摘要；GriTS 直接以表格网格比较预测与真实结构，研究比单纯比较标记字符串更贴近表格形态的指标。
- [Team, Deepmind - 2025 - Gemma 3 Technical Report](./wiki/summaries/Team,%20Deepmind%20-%202025%20-%20Gemma%203%20Technical%20Report.md)：精修摘要；Gemma 3 用局部与全局注意力降低长上下文缓存成本；4B、12B、27B 支持图文与 128K 上下文，1B 是 32K 的文本模型。
- [Tewari et al. - 2020 - State of the Art on Neural Rendering](./wiki/summaries/Tewari%20et%20al.%20-%202020%20-%20State%20of%20the%20Art%20on%20Neural%20Rendering.md)：精修摘要；这篇神经渲染综述讨论学习模型如何参与生成图像与视频，连接传统场景表示、渲染过程和数据驱动方法。
- [Tian et al. - 2024 - SpreadsheetLLM Encoding Spreadsheets for Large Language Models](./wiki/summaries/Tian%20et%20al.%20-%202024%20-%20SpreadsheetLLM%20Encoding%20Spreadsheets%20for%20Large%20Language%20Models.md)：精修摘要；SpreadsheetLLM 研究怎样压缩并编码电子表格，保留单元格地址、布局和格式，让 LLM 更有效地处理二维信息。
- [Tschannen et al. - 2025 - SigLIP 2 Multilingual Vision-Language Encoders with Improved Semantic Understanding , Localization , and Dens](./wiki/summaries/Tschannen%20et%20al.%20-%202025%20-%20SigLIP%202%20Multilingual%20Vision-Language%20Encoders%20with%20Improved%20Semantic%20Understanding%20,%20Localization%20,%20and%20Dens.md)：精修摘要；SigLIP 2 在图文训练中结合多种学习信号和数据整理，改进多语言、定位与密集视觉特征，适合研究视觉编码器。
- [Ultralytics - 2026 - Ultralytics YOLO Docs Home](./wiki/summaries/Ultralytics%20-%202026%20-%20Ultralytics%20YOLO%20Docs%20Home.md)：精修摘要；这份 Ultralytics 首页快照用于辨认当时的 YOLO 产品版本与官方入口，不能代替原论文的机制说明。
- [Vlasov, Mosig, Nichol - 2019 - Dialogue Transformers](./wiki/summaries/Vlasov,%20Mosig,%20Nichol%20-%202019%20-%20Dialogue%20Transformers.md)：精修摘要；Dialogue Transformers 用注意力读取历史对话轮次，为对话系统选择下一步行动，研究哪些历史信息真正相关。
- [Wang - Unknown - PIKE-RAG sPecIalized KnowledgE and Rationale Augmented Generation](./wiki/summaries/Wang%20-%20Unknown%20-%20PIKE-RAG%20sPecIalized%20KnowledgE%20and%20Rationale%20Augmented%20Generation.md)：精修摘要；PIKE-RAG 面向专业语料，把知识提炼与推理过程结合到检索增强中，尝试解决仅找相似片段仍无法回答的复杂问题。
- [Wang et al. - 2021 - LayoutReader Pre-training of Text and Layout for Reading Order Detection](./wiki/summaries/Wang%20et%20al.%20-%202021%20-%20LayoutReader%20Pre-training%20of%20Text%20and%20Layout%20for%20Reading%20Order%20Detection.md)：精修摘要；LayoutReader 利用文字和版面判断阅读顺序，并从 Word 文件的结构信息自动构造训练数据。
- [Wang et al. - 2022 - OFA Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](./wiki/summaries/Wang%20et%20al.%20-%202022%20-%20OFA%20Unifying%20Architectures,%20Tasks,%20and%20Modalities%20Through%20a%20Simple%20Sequence-to-Sequence%20Learning%20Framework.md)：精修摘要；OFA 把多种视觉与语言任务写成统一序列到序列接口，通过指令与输出序列表达不同任务。
- [DocLLM：布局感知文档语言模型（2024）](./wiki/summaries/Wang%20et%20al.%20-%202023%20-%20DocLLM%20A%20layout-aware%20generative%20language%20model%20for%20multimodal%20document%20understanding.md)：精修摘要；DocLLM 在语言模型中加入文字位置关系，帮助理解发票、表单等版面承载重要语义的文档。
- [Wang et al. - 2024 - YOLOv10 Real-Time End-to-End Object Detection](./wiki/summaries/Wang%20et%20al.%20-%202024%20-%20YOLOv10%20Real-Time%20End-to-End%20Object%20Detection.md)：精修摘要；YOLOv10 用两种样本分配协同训练，推理采用一对一预测，研究省去 NMS 后处理的目标检测。
- [Wu et al. - 2020 - CorefQA Coreference Resolution as Query-based Span Prediction](./wiki/summaries/Wu%20et%20al.%20-%202020%20-%20CorefQA%20Coreference%20Resolution%20as%20Query-based%20Span%20Prediction.md)：精修摘要；CorefQA 把共指消解改写成基于提及生成问题、在文档里找答案跨度的任务，利用问答式接口寻找指代对象。
- [Xing et al. - 2023 - LORE Logical Location Regression Network for Table Structure Recognition](./wiki/summaries/Xing%20et%20al.%20-%202023%20-%20LORE%20Logical%20Location%20Regression%20Network%20for%20Table%20Structure%20Recognition.md)：精修摘要；LORE 直接预测单元格的逻辑行列位置，尝试用结构回归恢复表格，减少额外规则和冗长序列解码。
- [Yang et al. - 2022 - Prompt Tuning for Generative Multimodal Pretrained Models](./wiki/summaries/Yang%20et%20al.%20-%202022%20-%20Prompt%20Tuning%20for%20Generative%20Multimodal%20Pretrained%20Models.md)：精修摘要；这篇工作将提示微调用于生成式多模态预训练模型，研究只训练少量提示参数能否适配理解和生成任务。
- [Yao et al. - 2021 - NLP From Scratch Without Large-Scale Pretraining A Simple and Efficient Framework](./wiki/summaries/Yao%20et%20al.%20-%202021%20-%20NLP%20From%20Scratch%20Without%20Large-Scale%20Pretraining%20A%20Simple%20and%20Efficient%20Framework.md)：精修摘要；TLM 用任务数据去大语料中找相关子集，再从头联合训练任务目标与语言目标，研究替代昂贵通用预训练的路径。
- [Zhai et al. - 2022 - Scaling Vision Transformers](./wiki/summaries/Zhai%20et%20al.%20-%202022%20-%20Scaling%20Vision%20Transformers.md)：精修摘要；这篇 ViT 规模化研究同时扩大模型与数据，分析视觉 Transformer 的增长规律，帮助设计更有效的训练投入。
- [Zhao et al. - 2023 - DETRs Beat YOLOs on Real-time Object Detection](./wiki/summaries/Zhao%20et%20al.%20-%202023%20-%20DETRs%20Beat%20YOLOs%20on%20Real-time%20Object%20Detection.md)：精修摘要；RT-DETR 研究实时端到端检测，针对传统检测后处理与 Transformer 计算成本做设计，比较速度和精度的整体取舍。
- [Zong, Song, Liu - 2024 - DETRs with Collaborative Hybrid Assignments Training](./wiki/summaries/Zong,%20Song,%20Liu%20-%202024%20-%20DETRs%20with%20Collaborative%20Hybrid%20Assignments%20Training.md)：精修摘要；Co-DETR 在训练时加入协同的混合分配，缓解 DETR 一对一匹配正样本稀疏的问题，改善特征学习。
- [Chen et al. - 2025 - A Comprehensive Survey of YOLO From YOLOv1 to YOLO11 and Beyond](./wiki/summaries/Chen%20et%20al.%20-%202025%20-%20A%20Comprehensive%20Survey%20of%20YOLO%20From%20YOLOv1%20to%20YOLO11%20and%20Beyond.md)：精修摘要；这篇 YOLO 综述按特征提取、特征融合、预测与训练等环节解释版本差异，适合建立家族阅读地图。

### Slide  理解与生成

- [AV-HuBERT：遮挡多模态聚类的音视频语音表示（2022）](./wiki/summaries/Ai%20-%202022%20-%20Learning%20Audio-Visual%20Speech%20Representation%20by%20Masked%20Multimodal%20Cluster%20Prediction.md)：精修摘要；AV-HuBERT 同时看嘴唇运动和听声音，用遮挡后的预测任务学习语音表示，为音视频语音识别提供预训练基础。
- [Robust Self-Supervised Audio-Visual Speech Recognition：抗噪音视频识别](./wiki/summaries/Ai%20-%20Unknown%20-%20Robust%20Self-Supervised%20Audio-Visual%20Speech%20Recognition.md)：精修摘要；这篇工作用自监督音视频预训练改善嘈杂环境下的语音识别：声音不清楚时，口型帮助模型判断目标说话人在说什么。
- [Allen, Science - 2018 - Higher-order Coreference Resolution with Coarse-to-fine Inference](./wiki/summaries/Allen,%20Science%20-%202018%20-%20Higher-order%20Coreference%20Resolution%20with%20Coarse-to-fine%20Inference.md)：精修摘要；这篇共指消解方法反复利用可能的前文指代更新文本片段表示，并先粗筛候选，控制推理成本。
- [data2vec：跨语音、视觉与语言的自监督框架（2022）](./wiki/summaries/Baevski%20et%20al.%20-%202021%20-%20data2vec%20A%20General%20Framework%20for%20Self-supervised%20Learning%20in%20Speech%20,%20Vision%20and%20Language.md)：精修摘要；data2vec 用同一种自监督思路处理语音、图像和文本：遮住输入的一部分，预测教师模型看到完整输入时产生的表示。
- [UniLMv2：Pseudo-Masked Language Models（2020）](./wiki/summaries/Bao%20et%20al.%20-%202019%20-%20LMv2%20Pseudo-Masked%20Language%20Models%20for%20Unified%20Language%20Model%20Pre-Training.md)：精修摘要；PMLM 用普通遮挡和伪遮挡结合的预训练方式，让同一语言模型同时学习理解上下文和逐步生成被遮住的内容。
- [Chen, He - 2021 - Exploring Simple Siamese Representation Learning](./wiki/summaries/Chen,%20He%20-%202021%20-%20Exploring%20Simple%20Siamese%20Representation%20Learning.md)：精修摘要；SimSiam 用一张图的两种增强视图学习表示，不需要负样本或动量编码器；停止梯度是其避免表示坍塌的关键设计。
- [Conneau et al. - 2020 - Unsupervised cross-lingual representation learning at scale](./wiki/summaries/Conneau%20et%20al.%20-%202020%20-%20Unsupervised%20cross-lingual%20representation%20learning%20at%20scale.md)：精修摘要；XLM-R 在大规模多语言文本上预训练同一个编码器，用共享表示支持跨语言理解，说明数据规模与语言覆盖的重要性。
- [Devlin et al. - 2019 - BERT Pre-training of deep bidirectional transformers for language understanding](./wiki/summaries/Devlin%20et%20al.%20-%202019%20-%20BERT%20Pre-training%20of%20deep%20bidirectional%20transformers%20for%20language%20understanding.md)：精修摘要；BERT 通过同时利用词语左右两边的上下文进行预训练，为分类、问答和抽取提供可微调的语言表示。
- [Fang et al. - 2021 - Injecting Semantic Concepts into End-to-End Image Captioning](./wiki/summaries/Fang%20et%20al.%20-%202021%20-%20Injecting%20Semantic%20Concepts%20into%20End-to-End%20Image%20Captioning.md)：精修摘要；这篇图像描述方法向端到端生成过程注入语义概念，帮助模型把视觉内容组织成文字，研究不用独立检测器的描述路线。
- [Gao, Yao, Chen - 2021 - SimCSE Simple Contrastive Learning of Sentence Embeddings](./wiki/summaries/Gao,%20Yao,%20Chen%20-%202021%20-%20SimCSE%20Simple%20Contrastive%20Learning%20of%20Sentence%20Embeddings.md)：精修摘要；SimCSE 用对比学习得到句向量：无监督版本把同一句话的两次 dropout 表示拉近，有监督版本利用推断数据构造正负例。
- [Giorgi et al. - 2021 - DeCLUTR Deep contrastive learning for unsupervised textual representations](./wiki/summaries/Giorgi%20et%20al.%20-%202021%20-%20DeCLUTR%20Deep%20contrastive%20learning%20for%20unsupervised%20textual%20representations.md)：精修摘要；DeCLUTR 从未标注文本构造对比学习任务，让句子表示更适合聚类和检索，减少对人工语义配对数据的依赖。
- [He, Girshick, Dollar - 2019 - Rethinking imageNet pre-training](./wiki/summaries/He,%20Girshick,%20Dollar%20-%202019%20-%20Rethinking%20imageNet%20pre-training.md)：精修摘要；这篇检测研究说明，在合适数据和更长训练下，从随机初始化训练也能得到有竞争力的结果，重新审视 ImageNet 预训练的必要性。
- [Hoppe, Toussaint - 2020 - Qgraph-bounded Q-learning Stabilizing Model-Free Off-Policy Deep Reinforcement Learning](./wiki/summaries/Hoppe,%20Toussaint%20-%202020%20-%20Qgraph-bounded%20Q-learning%20Stabilizing%20Model-Free%20Off-Policy%20Deep%20Reinforcement%20Learning.md)：精修摘要；Qgraph 方法把经验回放中的转移组成图，利用可计算的 Q 值下界稳定离策略训练；本条归档存在 PDF 与 HTML 内容不一致。
- [Hsu et al. - 2021 - HuBERT Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](./wiki/summaries/Hsu%20et%20al.%20-%202021%20-%20HuBERT%20Self-Supervised%20Speech%20Representation%20Learning%20by%20Masked%20Prediction%20of%20Hidden%20Units.md)：精修摘要；HuBERT 先把音频片段聚类成离散目标，再遮住部分输入做预测，用未标注语音学习可迁移的表示。
- [Huang et al. - 2022 - LayoutLMv3 Pre-training for Document AI with Unified Text and Image Masking](./wiki/summaries/Huang%20et%20al.%20-%202022%20-%20LayoutLMv3%20Pre-training%20for%20Document%20AI%20with%20Unified%20Text%20and%20Image%20Masking.md)：精修摘要；LayoutLMv3 同时遮挡文档里的文字和图像内容，并学习两者对应关系，让模型理解文字在页面上的位置与结构。
- [Jiang, Wang - 2022 - Deep Continuous Prompt for Contrastive Learning of Sentence Embeddings](./wiki/summaries/Jiang,%20Wang%20-%202022%20-%20Deep%20Continuous%20Prompt%20for%20Contrastive%20Learning%20of%20Sentence%20Embeddings.md)：精修摘要；这篇句向量方法冻结语言模型，只训练少量深层连续提示，再用对比学习适配句子表示，降低全参数微调成本。
- [Joshi et al. - 2020 - Spanbert Improving pre-training by representing and predicting spans](./wiki/summaries/Joshi%20et%20al.%20-%202020%20-%20Spanbert%20Improving%20pre-training%20by%20representing%20and%20predicting%20spans.md)：精修摘要；SpanBERT 把连续文本片段一起遮住，再让边界表示预测被遮片段，强化对实体、答案跨度等连续内容的建模。
- [Karpukhin et al. - 2020 - Dense passage retrieval for open-domain question answering](./wiki/summaries/Karpukhin%20et%20al.%20-%202020%20-%20Dense%20passage%20retrieval%20for%20open-domain%20question%20answering.md)：精修摘要；DPR 分别把问题和文章片段编码成向量，便于提前建库，再按语义相似度找到回答证据。
- [Karras, Härkönen - 2021 - Alias-Free Generative Adversarial Networks](./wiki/summaries/Karras,%20Härkönen%20-%202021%20-%20Alias-Free%20Generative%20Adversarial%20Networks.md)：精修摘要；StyleGAN3 的无混叠设计针对生成视频中纹理像粘在像素坐标上的问题，让细节随对象运动更一致。
- [Kwon, Engineering, Jaechul - Unknown - CLIPstyler Image Style Transfer with a Single Text Condition](./wiki/summaries/Kwon,%20Engineering,%20Jaechul%20-%20Unknown%20-%20CLIPstyler%20Image%20Style%20Transfer%20with%20a%20Single%20Text%20Condition.md)：精修摘要；CLIPstyler 用一句风格描述指导图像风格迁移，减少必须提供参考风格图片的限制。
- [Lee et al. - 2022 - Multimodal Lecture Presentations Dataset Understanding Multimodality in Educational Slides](./wiki/summaries/Lee%20et%20al.%20-%202022%20-%20Multimodal%20Lecture%20Presentations%20Dataset%20Understanding%20Multimodality%20in%20Educational%20Slides.md)：精修摘要；这份教育幻灯片数据集把页面、图示和讲解语音放在一起，研究教学材料中不同模态怎样共同传递知识。
- [Li et al. - 2019 - On the Sentence Embeddings from Pre-trained Language Models](./wiki/summaries/Li%20et%20al.%20-%202019%20-%20On%20the%20Sentence%20Embeddings%20from%20Pre-trained%20Language%20Models.md)：精修摘要；这篇工作研究为什么直接取 BERT 句向量常不能很好表示语义，并分析怎样更充分利用预训练表示。
- [Li, Fan, Ai - Unknown - Scaling Language-Image Pre-training via Masking](./wiki/summaries/Li,%20Fan,%20Ai%20-%20Unknown%20-%20Scaling%20Language-Image%20Pre-training%20via%20Masking.md)：精修摘要；FLIP 在训练图文模型时移除大量图块，把节省的计算用于更多样本，研究精度与训练成本的折中。
- [Liao et al. - 2023 - DocTr Document Transformer for Structured Information Extraction in Documents](./wiki/summaries/Liao%20et%20al.%20-%202023%20-%20DocTr%20Document%20Transformer%20for%20Structured%20Information%20Extraction%20in%20Documents.md)：精修摘要；DocTr 把文档中的实体表示成锚点词与位置框，再建模实体关系，探索结构化信息抽取的新接口。
- [Liu et al. - 2021 - Trans-Encoder Unsupervised sentence-pair modelling through self- and mutual-distillations](./wiki/summaries/Liu%20et%20al.%20-%202021%20-%20Trans-Encoder%20Unsupervised%20sentence-pair%20modelling%20through%20self-%20and%20mutual-distillations.md)：精修摘要；Trans-Encoder 在双塔和交叉编码器之间进行自蒸馏与相互蒸馏，尝试兼顾句子匹配的速度和质量。
- [Liu, Lapata - 2020 - Text summarization with pretrained encoders](./wiki/summaries/Liu,%20Lapata%20-%202020%20-%20Text%20summarization%20with%20pretrained%20encoders.md)：精修摘要；这篇摘要工作使用预训练编码器建模文档，同时研究抽取式和生成式摘要，说明两类输出需要不同的训练与接口。
- [Lv et al. - 2023 - Kosmos-2.5 A Multimodal Literate Model](./wiki/summaries/Lv%20et%20al.%20-%202023%20-%20Kosmos-2.5%20A%20Multimodal%20Literate%20Model.md)：精修摘要；Kosmos-2.5 读取文字密集图像，既能输出带位置的文本块，也能输出保留结构与样式的 Markdown。
- [Lysak et al. - 2023 - Optimized Table Tokenization for Table Structure Recognition](./wiki/summaries/Lysak%20et%20al.%20-%202023%20-%20Optimized%20Table%20Tokenization%20for%20Table%20Structure%20Recognition.md)：精修摘要；这篇研究优化表格结构的词元表示，关注同一张表怎样编码成更适合模型生成的序列。
- [Mikolov et al. - 2013 - Efficient estimation of word representations in vector space](./wiki/summaries/Mikolov%20et%20al.%20-%202013%20-%20Efficient%20estimation%20of%20word%20representations%20in%20vector%20space.md)：精修摘要；word2vec 用高效的词预测任务学习词向量，使词语能在连续空间中比较，成为后续语义表示方法的重要基础。
- [Mokady, Hertz, Bermano - 2021 - ClipCap CLIP Prefix for Image Captioning](./wiki/summaries/Mokady,%20Hertz,%20Bermano%20-%202021%20-%20ClipCap%20CLIP%20Prefix%20for%20Image%20Captioning.md)：精修摘要；ClipCap 把 CLIP 图像表示映射为语言模型的前缀，让已有视觉和语言模型组合完成图像描述。
- [Ouyang et al. - Unknown - OmniDocBench Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations](./wiki/summaries/Ouyang%20et%20al.%20-%20Unknown%20-%20OmniDocBench%20Benchmarking%20Diverse%20PDF%20Document%20Parsing%20with%20Comprehensive%20Annotations.md)：精修摘要；OmniDocBench 用多来源 PDF 和细致标注评估文档解析，帮助发现正文、公式、表格等不同内容的识别短板。
- [Pang et al. - 2022 - Long Document Summarization with Top-down and Bottom-up Inference](./wiki/summaries/Pang%20et%20al.%20-%202022%20-%20Long%20Document%20Summarization%20with%20Top-down%20and%20Bottom-up%20Inference.md)：精修摘要；这篇长文摘要方法把局部到全局和全局到局部的信息推断结合起来，试图在有限成本下保留长文关键内容。
- [Polyak et al. - 2021 - Speech resynthesis from discrete disentangled self-supervised representations](./wiki/summaries/Polyak%20et%20al.%20-%202021%20-%20Speech%20resynthesis%20from%20discrete%20disentangled%20self-supervised%20representations.md)：精修摘要；这篇语音重合成方法分别表示内容、韵律和说话人身份，研究怎样以低码率特征控制合成声音。
- [Qiu et al. - 2020 - Pre-trained models for natural language processing A survey](./wiki/summaries/Qiu%20et%20al.%20-%202020%20-%20Pre-trained%20models%20for%20natural%20language%20processing%20A%20survey.md)：精修摘要；这篇 NLP 预训练综述整理表示方式、训练目标和适配方法，帮助理解不同语言模型为何适合不同任务。
- [Radford et al. - 2021 - Learning Transferable Visual Models From Natural Language Supervision](./wiki/summaries/Radford%20et%20al.%20-%202021%20-%20Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision.md)：精修摘要；CLIP 用大量图文配对学习共同表示，通过比较图片与文字描述，实现灵活的视觉分类和检索。
- [Raffel et al. - 2020 - Exploring the limits of transfer learning with a unified text-to-text transformer](./wiki/summaries/Raffel%20et%20al.%20-%202020%20-%20Exploring%20the%20limits%20of%20transfer%20learning%20with%20a%20unified%20text-to-text%20transformer.md)：精修摘要；T5 把翻译、分类、摘要等任务统一成“输入文本、输出文本”，用同一接口系统比较预训练与迁移方法。
- [Smock, Pesala, Abraham - 2022 - PubTables-1M Towards comprehensive table extraction from unstructured documents](./wiki/summaries/Smock,%20Pesala,%20Abraham%20-%202022%20-%20PubTables-1M%20Towards%20comprehensive%20table%20extraction%20from%20unstructured%20documents.md)：精修摘要；PubTables-1M 提供大规模表格抽取标注，重点是完整、清楚的单元格结构，支撑检测和表格恢复研究。
- [Soares et al. - 2020 - Matching the blanks Distributional similarity for relation learning](./wiki/summaries/Soares%20et%20al.%20-%202020%20-%20Matching%20the%20blanks%20Distributional%20similarity%20for%20relation%20learning.md)：精修摘要；Matching the Blanks 用文本中的实体对和上下文学习关系表示，探索减少人工关系标签依赖的方法。
- [AI Index Report 2025：研究、能力与社会趋势](./wiki/summaries/StandfordUniversity%20-%202023%20-%20Artificial%20Intelligence%20Index%20Report%20Introduction%20to%20the%20AI%20Index%20Report%202023%20GP-003.md)：精修摘要；本地原始文件实际是《AI Index Report 2025》，不是归档名中的 2023 版；它汇总多方面统计，阅读时要保留每项数据的年份和口径。
- [Su et al. - 2021 - Whitening Sentence Representations for Better Semantics and Faster Retrieval](./wiki/summaries/Su%20et%20al.%20-%202021%20-%20Whitening%20Sentence%20Representations%20for%20Better%20Semantics%20and%20Faster%20Retrieval.md)：精修摘要；句向量白化通过后处理改善表示空间分布，探索不用复杂新模型就改善语义比较与检索的方式。
- [Sun et al. - 2019 - How to Fine-Tune BERT for Text Classification](./wiki/summaries/Sun%20et%20al.%20-%202019%20-%20How%20to%20Fine-Tune%20BERT%20for%20Text%20Classification.md)：精修摘要；这篇研究系统比较 BERT 文本分类的微调方式，帮助理解训练配置和数据条件怎样影响结果。
- [SlideVQA：多页幻灯片视觉问答（2023）](./wiki/summaries/Tanaka%20et%20al.%20-%20Unknown%20-%20Images.md)：精修摘要；SlideVQA 要求模型跨多页幻灯片寻找证据并回答问题，涵盖多跳和数值推理；旧文件名“Images”没有表达真实主题。
- [Tewari et al. - 2021 - Advances in neural rendering](./wiki/summaries/Tewari%20et%20al.%20-%202021%20-%20Advances%20in%20neural%20rendering.md)：精修摘要；《Advances in Neural Rendering》继续整理学习式场景表示与图像合成进展，重点是表示、渲染和可控性之间的关系。
- [Wang et al. - 2022 - DAMO-NLP at SemEval-2022 Task 11 A Knowledge-based System for Multilingual Named Entity Recognition](./wiki/summaries/Wang%20et%20al.%20-%202022%20-%20DAMO-NLP%20at%20SemEval-2022%20Task%2011%20A%20Knowledge-based%20System%20for%20Multilingual%20Named%20Entity%20Recognition.md)：精修摘要；DAMO-NLP 为短文本实体识别补充 Wikipedia 知识上下文，帮助分辨缺少上下文的多语言实体。
- [Wang et al. - 2024 - CDM A Reliable Metric for Fair and Accurate Formula Recognition Evaluation](./wiki/summaries/Wang%20et%20al.%20-%202024%20-%20CDM%20A%20Reliable%20Metric%20for%20Fair%20and%20Accurate%20Formula%20Recognition%20Evaluation.md)：精修摘要；CDM 研究公式识别的评价方式，避免仅用字符串差异惩罚形式不同但内容相近的公式表达。
- [Xiao et al. - 2023 - Florence-2 Advancing a Unified Representation for a Variety of Vision Tasks](./wiki/summaries/Xiao%20et%20al.%20-%202023%20-%20Florence-2%20Advancing%20a%20Unified%20Representation%20for%20a%20Variety%20of%20Vision%20Tasks.md)：精修摘要；Florence-2 用文字提示指定视觉任务，再生成对应描述或位置等输出，把多种视觉能力放进统一接口。
- [Xu et al. - 2016 - Review on knowledge graph techniques](./wiki/summaries/Xu%20et%20al.%20-%202016%20-%20Review%20on%20knowledge%20graph%20techniques.md)：精修摘要；这篇知识图谱综述整理定义、构建和应用，把实体与关系组织为可连接的知识结构，适合建立技术地图。
- [Xu, Choi - 2020 - Revealing the Myth of Higher-Order Inference in Coreference Resolution](./wiki/summaries/Xu,%20Choi%20-%202020%20-%20Revealing%20the%20Myth%20of%20Higher-Order%20Inference%20in%20Coreference%20Resolution.md)：精修摘要；这篇共指研究重新检验高阶推理的收益，比较多种方法，提醒复杂推理模块是否有效需要受控实验支持。
- [Yan et al. - 2021 - ConSERT A contrastive framework for self-supervised sentence representation transfer](./wiki/summaries/Yan%20et%20al.%20-%202021%20-%20ConSERT%20A%20contrastive%20framework%20for%20self-supervised%20sentence%20representation%20transfer.md)：精修摘要；ConSERT 用自监督对比学习改善句子表示，针对原始 BERT 向量直接做语义相似度时的不足。
- [Zhang et al. - 2022 - Tip-Adapter Training-Free Adaption of CLIP for Few-Shot Classification](./wiki/summaries/Zhang%20et%20al.%20-%202022%20-%20Tip-Adapter%20Training-Free%20Adaption%20of%20CLIP%20for%20Few-Shot%20Classification.md)：精修摘要；Tip-Adapter 利用少量标注样本的特征缓存辅助 CLIP 分类，研究无需完整重新训练的少样本适配方式。
- [Zhang et al. - 2025 - SlideAudit A Dataset and Taxonomy for Automated Evaluation of Presentation Slides](./wiki/summaries/Zhang%20et%20al.%20-%202025%20-%20SlideAudit%20A%20Dataset%20and%20Taxonomy%20for%20Automated%20Evaluation%20of%20Presentation%20Slides.md)：精修摘要；SlideAudit 用专家整理的设计缺陷分类和标注幻灯片，研究如何自动发现演示页面中的具体问题。
- [Zhang, Li, Zhang - 2020 - Efficient Second-Order TreeCRF for Neural Dependency Parsing](./wiki/summaries/Zhang,%20Li,%20Zhang%20-%202020%20-%20Efficient%20Second-Order%20TreeCRF%20for%20Neural%20Dependency%20Parsing.md)：精修摘要；这篇依存分析方法把二阶结构信息纳入 TreeCRF，在计算效率和全局句法结构之间寻找平衡。
- [Zheng et al. - 2025 - PPTAgent Generating and Evaluating Presentations Beyond Text-to-Slides](./wiki/summaries/Zheng%20et%20al.%20-%202025%20-%20PPTAgent%20Generating%20and%20Evaluating%20Presentations%20Beyond%20Text-to-Slides.md)：精修摘要；PPTAgent 把演示文稿生成组织为分阶段编辑流程，同时评估内容、视觉效果和跨页结构，超出只把文字放进幻灯片。
- [Zhou et al. - 2021 - Pose-Controllable Talking Face Generation by Implicitly Modularized Audio-Visual Representation](./wiki/summaries/Zhou%20et%20al.%20-%202021%20-%20Pose-Controllable%20Talking%20Face%20Generation%20by%20Implicitly%20Modularized%20Audio-Visual%20Representation.md)：精修摘要；这篇说话人生成方法把音频驱动的口型与头部姿态控制分开考虑，研究保持同步时怎样控制动作。
- [Zuo et al. - 2022 - MoEBERT from BERT to Mixture-of-Experts via Importance-Guided Adaptation](./wiki/summaries/Zuo%20et%20al.%20-%202022%20-%20MoEBERT%20from%20BERT%20to%20Mixture-of-Experts%20via%20Importance-Guided%20Adaptation.md)：精修摘要；MoEBERT 按重要性将 BERT 适配为专家混合形式，探索保留模型容量同时减少每次输入实际计算的路径。
### 视频生成 / 音视频生成

- [OpenAI - 2025 - Sora 2 is here](./wiki/summaries/OpenAI%20-%202025%20-%20Sora%202%20is%20here.md)：精修摘要；Sora 2 的历史贡献是把视频、同步对白和音效纳入同一创作入口，并加入人物参考与多镜头控制。本文区分2025年的发布声明与页面后来新增的2026-04-26产品停用公告，避免把历史入口当作当前服务。
- [Alibaba Cloud - 2025 - Alibaba Unveils Wan2.6 Series Enabling Everyone to Star in Videos](./wiki/summaries/Alibaba%20Cloud%20-%202025%20-%20Alibaba%20Unveils%20Wan2.6%20Series%20Enabling%20Everyone%20to%20Star%20in%20Videos.md)：精修摘要；Wan2.6 的发布稿介绍参考视频、文字和图片驱动的视频生成；重点是把同一角色带入新的场景与镜头。
- [Google DeepMind - 2026 - Veo](./wiki/summaries/Google%20DeepMind%20-%202026%20-%20Veo.md)：精修摘要；Veo 官方页面介绍带音频的视频生成及参考控制；它展示产品能力，具体质量仍要通过对应任务核对。
- [Kuaishou Technology - 2026 - Kling VIDEO 3.0 Omni Model User Guide](./wiki/summaries/Kuaishou%20Technology%20-%202026%20-%20Kling%20VIDEO%203.0%20Omni%20Model%20User%20Guide.md)：精修摘要；Kling VIDEO 3.0 Omni 指南介绍角色参考、声音绑定和多镜头控制，适合按创作步骤了解功能。
- [Vidu - 2026 - Pricing](./wiki/summaries/Vidu%20-%202026%20-%20Pricing.md)：精修摘要；Vidu 价格页同时列出模型与任务支持，可作为归档时的能力矩阵入口，价格和限制需看具体版本。
- [Team Seedance et al. - 2026 - Seedance 2.0 Advancing Video Generation for World Complexity](./wiki/summaries/Team%20Seedance%20et%20al.%20-%202026%20-%20Seedance%202.0%20Advancing%20Video%20Generation%20for%20World%20Complexity.md)：精修摘要；Seedance 2.0 将文字、图片、音频和视频作为创作参考，研究可控的视频与声音联合生成。

### 经典 CNN / 视觉 Backbone

- [Simonyan, Zisserman - 2014 - Very Deep Convolutional Networks for Large-Scale Image Recognition](./wiki/summaries/Simonyan,%20Zisserman%20-%202014%20-%20Very%20Deep%20Convolutional%20Networks%20for%20Large-Scale%20Image%20Recognition.md)：精修摘要；VGG 重复堆叠小卷积，系统研究更深网络如何改善图像识别，并提供规则的特征提取结构。
- [Szegedy et al. - 2014 - Going Deeper with Convolutions](./wiki/summaries/Szegedy%20et%20al.%20-%202014%20-%20Going%20Deeper%20with%20Convolutions.md)：精修摘要；Inception 在同一模块中组合不同尺度的分支，并用小投影控制计算，研究预算内的多尺度视觉特征。
- [He et al. - 2015 - Deep Residual Learning for Image Recognition](./wiki/summaries/He%20et%20al.%20-%202015%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition.md)：精修摘要；ResNet 让网络层学习对输入的增量修正，用跳跃连接改善很深网络的训练。
- [Huang et al. - 2016 - Densely Connected Convolutional Networks](./wiki/summaries/Huang%20et%20al.%20-%202016%20-%20Densely%20Connected%20Convolutional%20Networks.md)：精修摘要；DenseNet 将前面各层的特征直接拼接给后面的层，研究怎样复用特征并改善信息传播。
- [Xie et al. - 2016 - Aggregated Residual Transformations for Deep Neural Networks](./wiki/summaries/Xie%20et%20al.%20-%202016%20-%20Aggregated%20Residual%20Transformations%20for%20Deep%20Neural%20Networks.md)：精修摘要；ResNeXt 将统一的多分支变换放进残差块，研究分支数量怎样成为深度和宽度之外的容量选择。
- [Howard et al. - 2017 - MobileNets Efficient Convolutional Neural Networks for Mobile Vision Applications](./wiki/summaries/Howard%20et%20al.%20-%202017%20-%20MobileNets%20Efficient%20Convolutional%20Neural%20Networks%20for%20Mobile%20Vision%20Applications.md)：精修摘要；MobileNet 用深度可分离卷积减少计算，并提供可调节的模型宽度与输入分辨率，适合研究移动端视觉成本。
- [Liu et al. - 2022 - A ConvNet for the 2020s](./wiki/summaries/Liu%20et%20al.%20-%202022%20-%20A%20ConvNet%20for%20the%202020s.md)：精修摘要；ConvNeXt 逐步调整 ResNet 的结构与训练方式，研究纯卷积模型在现代视觉任务中仍能达到什么水平。

## Topics

- [注意力机制 Attention](./wiki/topics/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%20Attention.md)：正式综述；“高效注意力”可能改连接、近似矩阵、压缩缓存或优化显存读写。先分清瓶颈，才能公平比较方法的质量与速度。
- [BERT类双向Transformer语言模型](./wiki/topics/BERT%E7%B1%BB%E5%8F%8C%E5%90%91Transformer%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)：正式综述；BERT 类编码器提供上下文表示，后续工作分别改训练配方、跨度目标、句向量空间和多语言迁移。先确认要的是 token 标签、句子相似度还是 query–document 匹配，再选模型。
- [搜索排序](./wiki/topics/%E6%90%9C%E7%B4%A2%E6%8E%92%E5%BA%8F.md)：正式综述；搜索排序在候选材料中判断哪些更相关，核心取舍是词面、语义交互与计算成本。召回正确和排在前面是不同问题。
- [传统 NLP](./wiki/topics/传统%20NLP.md)：正式综述；传统 NLP 的价值在任务结构：词、句子、跨度、依存树和检索候选需要不同预测接口。共享预训练表示并没有消除这些差别，理解它们有助于判断何时用生成模型、编码器或专门推断。
- [传统 CV](./wiki/topics/传统%20CV.md)：正式综述；视觉方法可以按三层阅读：如何学习表示、如何定义任务输出、如何在真实设备运行。CNN、ViT、文档模型与生成式感知改变的是不同层，不能用一个“统一视觉”的口号合并。
- [OCR](./wiki/topics/OCR.md)：正式综述；文字识别、整页阅读顺序、表格结构和文档问答是不同层次。选 OCR 方案前，先确定要纯文本、坐标、Markdown 还是可保留结构的标签，再按文档类型和错误成本比较。
- [经典 CNN 架构](./wiki/topics/经典%20CNN%20架构.md)：正式综述；经典 CNN 的演进分别改进深度、连接、多分支与效率。VGG、ResNet、DenseNet 和 MobileNet 改的是不同设计维度。
- [目标检测](./wiki/topics/目标检测.md)：正式综述；检测路线的差别在复杂度放在哪里：候选框、密集预测、集合匹配或训练辅助。比较前统一 AP 定义、输入尺寸、硬件、精度和后处理；NMS-free 表示接口变化，不等于所有场景更优。
- [LLM 预训练](./wiki/topics/LLM%20预训练.md)：正式综述；预训练比较要同时看目标、数据、参数激活、计算预算和训练公开程度。模型最终的聊天、推理或 agent 成绩还受后训练与运行系统影响，不能全部归因于预训练。
- [文本扩散语言模型](./wiki/topics/%E6%96%87%E6%9C%AC%E6%89%A9%E6%95%A3%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)：正式综述；DiffusionGemma 在一个文本块内多次并行修正，块与块之间仍按顺序生成。收益是特定硬件和请求规模下的延迟折中；输出 token/s、计算量、首字等待和任务质量需要分别比较。
- [LLM RL](./wiki/topics/LLM%20RL.md)：正式综述；先看训练信号来自哪里：人类偏好、可验证结果，还是 teacher 概率。再看是否需要当前策略采样、reference、critic 和环境。DPO、GRPO、DAPO 与 OPD 不能只按“是否叫 RL”来比较。
- [DeepSeek 系列](./wiki/topics/DeepSeek%20系列.md)：正式综述；DeepSeek 的效率架构、推理后训练和文档视觉压缩是不同技术线。V3/V4 改计算与缓存，Math/R1 改奖励驱动行为，OCR 研究页面转写与压缩；它们的成绩需要分别看实验条件。
- [AI 智能问答与智能客服](./wiki/topics/AI%20%E6%99%BA%E8%83%BD%E9%97%AE%E7%AD%94%E4%B8%8E%E6%99%BA%E8%83%BD%E5%AE%A2%E6%9C%8D.md)：正式综述；客服要完成三个不同的判断：找到适用依据、生成符合条件的答复、确认是否可以执行操作。FAQ、RAG、工具和人工接管分别处理其中一部分；回答流畅不能替代政策核对和任务完成。
- [Slide 理解与生成](./wiki/topics/Slide%20理解与生成.md)：正式综述；读懂幻灯片、从多页找到答案、检查设计缺陷和生成可编辑演示，是四个不同任务。应把内容正确、页面设计和跨页叙事分别验收，单个自动总分无法覆盖它们。
- [Scaling 与 compute-optimal training](./wiki/topics/Scaling%20与%20compute-optimal%20training.md)：正式综述；Chinchilla 研究固定训练 FLOPs 下怎样分配参数和 token；部署还要计算长期推理成本。幂律是实验区间内的拟合，不能当作所有模型、数据和任务通用的预算公式。
- [指令对齐与 post-training](./wiki/topics/指令对齐与%20post-training.md)：正式综述；后训练分别塑造任务接口、输出分布和偏好行为。示范、偏好对、可验证奖励与 teacher 分布提供不同信息；改善人工偏好不表示事实、安全和执行能力都同时得到保证。
- [Qwen 系列](./wiki/topics/Qwen%20系列.md)：正式综述；Qwen 从语言模型扩展出视觉理解、音视频交互和图像生成等分支。先确认任务的输入输出，再看代际如何改变训练和能力。
- [扩散模型与文生图](./wiki/topics/%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%96%87%E7%94%9F%E5%9B%BE.md)：正式综述；扩散与潜空间生成连接了图像训练、采样和编辑。画质、文字遵循、参考控制与可部署成本是不同维度，应分别比较。
- [图像分层 layered](./wiki/topics/%E5%9B%BE%E5%83%8F%E5%88%86%E5%B1%82%20layered.md)：正式综述；图像分层把单张画面拆成可独立编辑和合成的对象，难点包括透明边缘、遮挡补全与图层关系，远超简单抠出一个轮廓。
- [视频生成](./wiki/topics/视频生成.md)：正式综述；视频系统正在增加参考控制、续写、编辑和原生音频，但每项能力要单独核验。产品页支持接口说明，技术报告支持受限实验；逼真样例不能证明可靠的世界模型或稳定生产流程。

## Concepts

- [神经渲染](wiki/concepts/%E7%A5%9E%E7%BB%8F%E6%B8%B2%E6%9F%93.md)：神经渲染把学习表示与相机、几何或渲染过程连接起来，目标常是生成条件视图。它与物体识别或普通文生图的约束不同，先看场景输入和能控制哪些变量。

- [语音表示、识别与合成](wiki/concepts/%E8%AF%AD%E9%9F%B3%E8%A1%A8%E7%A4%BA%E3%80%81%E8%AF%86%E5%88%AB%E4%B8%8E%E5%90%88%E6%88%90.md)：语音识别把音频变文字，TTS 把文字变音频，声码器把声学表示变波形。自监督表示和音视频输入可以帮助其中部分阶段；它们的质量与训练条件不能互相替代。

- [LLM Wiki 文档处理流程](./wiki/concepts/LLM%20Wiki%20文档处理流程.md)：这页说明怎样把一篇资料读懂并用进知识库：先保存原文，按问题阅读和核证，再更新摘要与受影响页面。
- [GPT-3](./wiki/concepts/GPT-3.md)：GPT-3 用大规模自回归预训练展示上下文少样本学习：提示中给例子，模型便可尝试新任务，无需当场更新参数。
- [PaLM](./wiki/concepts/PaLM.md)：PaLM 用大规模密集 Transformer 与 Pathways 训练系统研究语言能力的规模化，包括少样本和多步推理表现。
- [Chinchilla](./wiki/concepts/Chinchilla.md)：Chinchilla 研究固定训练预算怎样在模型大小与数据量之间分配，提醒更大的模型也需要足够训练。
- [Qwen](./wiki/concepts/Qwen.md)：Qwen 家族分为语言、视觉理解、音视频和图像生成等支线；先按输入输出找分支，再比较代际变化。
- [Qwen1.5](./wiki/concepts/Qwen1.5.md)：Qwen1.5 扩展了模型尺寸和长上下文等配置，标志 Qwen 从单个模型报告走向更完整的开放家族。
- [Qwen2](./wiki/concepts/Qwen2.md)：Qwen2 将多语言、长上下文与密集或专家混合结构组织成模型家族，是理解 Qwen 后续演进的一环。
- [Qwen2.5](./wiki/concepts/Qwen2.5.md)：Qwen2.5 是语言模型家族的一次多尺寸升级，资料涉及预训练、指令能力和任务表现，适用性要按具体版本理解。
- [Qwen3](./wiki/concepts/Qwen3.md)：Qwen3 引入可切换的思考方式并扩展专家混合与代理能力，关注任务质量与推理预算怎样平衡。
- [Qwen3.5](./wiki/concepts/Qwen3.5.md)：本库的 Qwen3.5 资料将多模态与代理执行放到主干模型中，阅读重点是训练机制和实际任务接口。
- [Llama 家族](./wiki/concepts/Llama%20家族.md)：Llama 家族按初代、Llama 2、Llama 3 及代码和安全分支阅读，便于分清通用底座、聊天适配与专门任务。
- [LLaMA（初代）](./wiki/concepts/LLaMA%20初代.md)：初代 LLaMA 是 2023 年的开放权重语言模型，研究重点是数据与训练配置如何支持不同尺寸的基础模型。
- [Llama 2](./wiki/concepts/Llama%202.md)：Llama 2 同时提供基础与对话模型，连接预训练语言能力和聊天后训练；两种版本适用方式不同。
- [Code Llama](./wiki/concepts/Code%20Llama.md)：Code Llama 是 Llama 的代码专门分支，用于代码生成与补全；比较时需明确规模、版本和编程任务。
- [Llama 3](./wiki/concepts/Llama%203.md)：Llama 3 报告涉及多语言、代码、推理和工具使用，读者应按规模、基础或指令版本与具体任务理解结果。
- [BLOOM](./wiki/concepts/BLOOM.md)：BLOOM 是 BigScience 协作训练的多语言开放模型，阅读时重点看语言覆盖、训练组织和发布条件。
- [MPT](./wiki/concepts/MPT.md)：MPT 是 MosaicML 的开放语言模型系列，资料同时涉及模型、训练效率和发布方式，使用前要区分版本与许可。
- [Mistral 7B](./wiki/concepts/Mistral%207B.md)：Mistral 7B 用较紧凑的密集模型与注意力设计提供开放语言能力，适合研究单位参数效果和部署成本。
- [Mixtral](./wiki/concepts/Mixtral.md)：Mixtral 将语言模型组织为专家混合结构，每次只计算部分专家，以更高总容量控制单次计算。
- [Gemma](./wiki/concepts/Gemma.md)：Gemma 是 Google 的开放模型家族入口，按代际理解尺寸、视觉能力、上下文和生成机制的变化更容易读。
- [Gemma 2](./wiki/concepts/Gemma%202.md)：Gemma 2 是 Gemma 的后续开放模型代际，重点是有限尺寸下的模型结构、训练和能力表现。
- [Gemma 4](./wiki/concepts/Gemma%204.md)：本库的 Gemma 4 资料描述密集与专家混合、多模态和代理工作流的开放家族，是后续 DiffusionGemma 的底座来源。
- [DiffusionGemma](./wiki/concepts/DiffusionGemma.md)：DiffusionGemma 以反复修正一块文本的方式生成内容，是基于 Gemma 的实验性离散扩散模型；它与逐词续写接口不同。
- [DeepSeek](./wiki/concepts/DeepSeek.md)：DeepSeek 家族可沿三条线阅读：通用模型的高效训练、R1 的推理后训练，以及长上下文和 OCR 的信息压缩。
- [DeepSeek-V3](./wiki/concepts/DeepSeek-V3.md)：DeepSeek-V3 是高效专家混合语言模型，重点技术包括稀疏计算与训练安排；它与 R1 的推理后训练应分开阅读。
- [DeepSeek-V4](./wiki/concepts/DeepSeek-V4.md)：本库的 V4 预览资料关注百万词元上下文、缓存压缩和代理工作流，读者应保留版本及开发者报告的边界。
- [DeepSeek-OCR](./wiki/concepts/DeepSeek-OCR.md)：DeepSeek-OCR 研究将文档视觉信息压缩成较少词元再转写，关注识别质量与长文处理成本的关系。
- [Compressed Sparse Attention](./wiki/concepts/Compressed%20Sparse%20Attention.md)：CSA 先压缩长上下文的键值缓存，再从压缩块中选择相关部分做注意力，目标是减少长文本推理负担。
- [Heavily Compressed Attention](./wiki/concepts/Heavily%20Compressed%20Attention.md)：HCA 对长上下文缓存采用更大的压缩跨度，进一步减少缓存和计算；代价与细节保留需结合 V4 设定理解。
- [Manifold-Constrained Hyper-Connections](./wiki/concepts/Manifold-Constrained%20Hyper-Connections.md)：mHC 约束多路残差连接中的映射，目标是让深层网络传递信息更稳定；它调整层间连接，而非文本中的注意力范围。
- [Muon](./wiki/concepts/Muon.md)：Muon 对二维权重的动量更新做矩阵级变换，尝试改善更新几何；实际训练常与其他参数组的 AdamW 配合。
- [StarCoder2](./wiki/concepts/StarCoder2.md)：StarCoder2 是开放代码模型路线，理解它应同时看代码数据、模型规模、任务评测和发布条件。
- [DBRX](./wiki/concepts/DBRX.md)：DBRX 是 Databricks 的开放专家混合模型，了解它应同时看稀疏计算、任务能力与实际部署条件。
- [OpenELM](./wiki/concepts/OpenELM.md)：OpenELM 是面向较小模型效率的开放模型路线，关注层间参数分配、训练与部署，而不只追求总规模。
- [Phi-3](./wiki/concepts/Phi-3.md)：Phi-3 以较小模型与高质量训练数据探索能力密度，适合研究本地部署与任务质量的折中。
- [OLMo 2](./wiki/concepts/OLMo%202.md)：OLMo 2 强调模型与训练研究的开放透明度，阅读时可同时关注数据、训练流程和具体能力。
- [Falcon 3](./wiki/concepts/Falcon%203.md)：Falcon 3 是 Falcon 开放模型家族的后续节点，阅读重点是各尺寸、任务能力和发布条件。
- [GLM](./wiki/concepts/GLM.md)：GLM 家族从通用语言预训练发展到对话和后续模型，阅读时要区分研究方法、具体代际和产品名称。
- [Kimi](./wiki/concepts/Kimi.md)：Kimi 家族入口连接长上下文、推理训练与后续开放模型，开放性和部署条件要逐代核对。
- [Kimi K3](./wiki/concepts/Kimi%20K3.md)：本库的 K3 报告把多模态、稀疏模型、长上下文和长程代理训练放在同一系统中，阅读应拆成各个技术环节。
- [Kimi Delta Attention](./wiki/concepts/Kimi%20Delta%20Attention.md)：KDA 用固定大小状态随序列更新，控制随文本增长的缓存开销；K3 将它与全局注意力配合使用。
- [Attention Residuals](./wiki/concepts/Attention%20Residuals.md)：AttnRes 让网络当前层有选择地读取之前各层的表示，而不只是把历史信息不断相加，调整的是深度方向的信息传递。
- [Stable LatentMoE](./wiki/concepts/Stable%20LatentMoE.md)：Stable LatentMoE 让共享专家在完整宽度、路由专家在较窄潜空间处理信息，降低大型专家系统的计算与通信负担。
- [Quantile Balancing](./wiki/concepts/Quantile%20Balancing.md)：Quantile Balancing 根据路由分数差的分位数调整专家分发偏置，帮助大型专家系统控制负载，不直接增加辅助训练损失。
- [MoonViT-V2](./wiki/concepts/MoonViT-V2.md)：K3 报告中的 MoonViT-V2 与语言模型联合训练视觉表示，关注视觉塔如何进入统一多模态训练目标。
- [MoonEP](./wiki/concepts/MoonEP.md)：MoonEP 为大型专家混合训练组织专家放置与执行，减少负载不均和通信浪费，关注整套分布式系统效率。
- [DeepSeek-R1](./wiki/concepts/DeepSeek-R1.md)：DeepSeek-R1 通过强化学习与多阶段训练增强推理，并提供蒸馏模型；应区分 R1-Zero、R1 和后续版本。
- [InstructGPT](./wiki/concepts/InstructGPT.md)：InstructGPT 用示范、偏好和强化学习让语言模型更符合用户意图，说明知识能力与交互行为需要分别训练。
- [DPO](./wiki/concepts/DPO.md)：DPO 利用成对的好坏回答直接训练模型偏好，简化独立奖励模型与在线策略优化的部分流程。
- [RLHF](./wiki/concepts/RLHF.md)：RLHF 通过人类示范和偏好建立奖励信号，再优化模型行为，使回答更符合期望；数据和奖励设计是关键。
- [ORPO](./wiki/concepts/ORPO.md)：ORPO 将监督学习与偏好目标放进同一训练阶段，减少独立参考模型的流程，适合比较后训练配方的简化。
- [KTO](./wiki/concepts/KTO.md)：KTO 使用单个回答的好坏反馈进行偏好训练，不强制每条样本都有一对回答，适合研究不同反馈接口。
- [GRPO](./wiki/concepts/GRPO.md)：GRPO 对同一问题采样多份回答，利用组内相对奖励调整策略，省去单独价值模型的部分资源成本。
- [DAPO](./wiki/concepts/DAPO.md)：DAPO 将推理强化学习中的优化与工程技巧组织成训练方案，关注长回答、样本筛选和训练稳定性。
- [OPD](./wiki/concepts/OPD.md)：在线策略蒸馏让学生先按自己的当前模型生成，再由教师在这些轨迹上提供分布监督，缓解只模仿教师轨迹的偏差。
- [Instruction Tuning](./wiki/concepts/Instruction%20Tuning.md)：指令微调把任务说明与参考回答作为训练样本，让预训练模型学会按要求完成任务，是常见的后训练阶段。
- [LoRA](./wiki/concepts/LoRA.md)：LoRA 冻结大模型原权重，只训练两个较小矩阵表示的低秩增量，以降低任务适配的训练与存储成本。
- [MoE](./wiki/concepts/MoE.md)：专家混合让路由器为不同输入选择部分子网络，扩大总参数容量，同时控制每个输入实际使用的计算。
- [BERT](./wiki/concepts/BERT.md)：BERT 同时利用左右上下文学习文本表示，适合分类、抽取和问答等理解任务；具体任务通常还需微调。
- [RoBERTa](./wiki/concepts/RoBERTa.md)：RoBERTa 系统重做 BERT 的训练配置，说明更多数据与更充分训练可能比新增结构更关键。
- [SpanBERT](./wiki/concepts/SpanBERT.md)：SpanBERT 以连续片段为遮挡单位，用边界表示学习片段内容，适合研究实体、答案跨度与指代表示。
- [Seq2Seq](./wiki/concepts/Seq2Seq.md)：序列到序列模型把可变长度输入映射成可变长度输出，是翻译、摘要、语音和多模态任务的共同接口。
- [T5](./wiki/concepts/T5.md)：T5 将 NLP 任务统一为文本到文本，分类也输出标签文字，便于比较预训练与迁移策略。
- [FLAN](./wiki/concepts/FLAN.md)：FLAN 用多任务指令微调改善未见任务的零样本表现，核心是让模型学习如何按照自然语言任务说明工作。
- [OPT](./wiki/concepts/OPT.md)：OPT 提供一组开放研究的解码器语言模型，帮助检查大规模预训练与适配；权重、材料和许可要分别看。
- [OPT-IML](./wiki/concepts/OPT-IML.md)：OPT-IML 研究任务数量、指令形式与数据分配怎样影响未见任务泛化，让指令微调的选择可比较。
- [Switch Transformer](./wiki/concepts/Switch%20Transformer.md)：Switch Transformer 简化专家路由，让每个输入只使用少量专家，研究扩大容量时如何控制计算和训练不稳定。
- [mT5](./wiki/concepts/mT5.md)：mT5 将 T5 的文本到文本预训练扩展到多语言，支持不同语言的理解和生成，需要关注语言不平衡。
- [XLM-R](./wiki/concepts/XLM-R.md)：XLM-R 在多语言语料上用遮挡预测训练共享编码器，支持跨语言理解，语言覆盖和训练比例影响迁移。
- [Transformer](./wiki/concepts/Transformer.md)：Transformer 通过注意力在序列元素之间交换信息，并结合逐位置变换建模，是语言及多模态模型的常用架构。
- [FlashAttention](./wiki/concepts/FlashAttention.md)：FlashAttention 保留标准注意力的数学结果，通过分块与减少显存读写加快执行，改变的是算子实现。
- [vLLM](./wiki/concepts/vLLM.md)：vLLM 是语言模型推理服务系统，优化缓存、调度和执行以支持高吞吐；应区分早期论文机制与后续版本功能。
- [SGLang](./wiki/concepts/SGLang.md)：SGLang 优化由多次模型调用、提示状态、分支和约束输出组成的程序，前端表达流程，运行时负责高效执行。
- [PagedAttention](./wiki/concepts/PagedAttention.md)：PagedAttention 用分页式映射组织生成过程的键值缓存，减少连续显存预留和碎片，是 vLLM 的早期关键机制。
- [RadixAttention](./wiki/concepts/RadixAttention.md)：RadixAttention 用基数树保存并匹配请求前缀的缓存，让共享提示的多次调用少做重复计算，是运行时复用机制。
- [Grouped-Query Attention](./wiki/concepts/Grouped-Query%20Attention.md)：GQA 让一组查询头共享键和值，折中普通多头注意力与完全共享的 MQA，减少解码缓存。
- [ViT](./wiki/concepts/ViT.md)：ViT 把图片切成图块序列交给 Transformer，在大规模训练下用于视觉任务，改变了图像的建模接口。
- [VGG](./wiki/concepts/VGG.md)：VGG 用规则堆叠的小卷积核增加网络深度，形成容易理解的视觉骨干，但计算和参数成本也需考虑。
- [GoogLeNet](./wiki/concepts/GoogLeNet.md)：GoogLeNet 的 Inception 模块用多种尺度的并行分支处理图像，再合并信息，在预算内扩展表示能力。
- [ResNet](./wiki/concepts/ResNet.md)：ResNet 用残差连接让深层网络学习对输入的修正，改善深层训练，成为许多视觉模型的基础骨干。
- [DenseNet](./wiki/concepts/DenseNet.md)：DenseNet 把前面层的特征直接传给后面层，促进特征复用；它使用拼接，区别于 ResNet 的残差相加。
- [ResNeXt](./wiki/concepts/ResNeXt.md)：ResNeXt 在残差模块中加入多组相似分支，把分支数量作为容量维度，探索深度和宽度之外的结构选择。
- [MobileNet](./wiki/concepts/MobileNet.md)：MobileNet 通过深度可分离卷积及规模调节面向移动端效率，关注准确率、延迟与模型大小的折中。
- [ConvNeXt](./wiki/concepts/ConvNeXt.md)：ConvNeXt 重新设计纯卷积骨干的结构和训练细节，研究在视觉 Transformer 时代卷积模型能做到什么。
- [Faster R-CNN](./wiki/concepts/Faster%20R-CNN.md)：Faster R-CNN 先提出候选区域，再分类和调整框，是经典两阶段检测方案，将候选生成纳入神经网络。
- [YOLO](./wiki/concepts/YOLO.md)：YOLO 将对象类别和位置在检测网络中共同预测，形成实时检测家族；每代结构与后处理需分别阅读。
- [DETR](./wiki/concepts/DETR.md)：DETR 直接预测一组对象，并用集合匹配训练检测模型，减少候选框和去重的手工环节。
- [CLIP](./wiki/concepts/CLIP.md)：CLIP 把图片和文字映射到可比较的表示空间，用自然语言描述进行图像分类或图文检索。
- [Toolformer](./wiki/concepts/Toolformer.md)：Toolformer 研究语言模型如何从训练信号中学会何时调用工具、怎样组织输入并利用结果，是工具使用的早期方法。
- [Llama Guard](./wiki/concepts/Llama%20Guard.md)：Llama Guard 对对话输入或输出进行安全分类，作为系统的一道检查环节；分类器本身也有误判和漏判。
- [Gemma 3](./wiki/concepts/Gemma%203.md)：Gemma 3 的报告扩展了视觉、多语言和长上下文能力，并调整注意力安排以控制缓存成本；具体尺寸需分别看。
- [MiniCPM](./wiki/concepts/MiniCPM.md)：MiniCPM 研究小语言模型的训练上限，通过更精细的规模实验和训练计划提高有限参数下的能力。
- [MiniCPM-V](./wiki/concepts/MiniCPM-V.md)：MiniCPM-V 面向较轻量的视觉语言部署，将图像理解接到语言接口，关注设备成本与任务效果。
- [Qwen2-VL](./wiki/concepts/Qwen2-VL.md)：Qwen2-VL 面向图片和视频理解，处理不同分辨率与时间信息，把视觉内容接到语言问答接口。
- [Qwen2.5-VL](./wiki/concepts/Qwen2.5-VL.md)：Qwen2.5-VL 扩展视觉理解、文档与视频等任务，阅读时应分别核对文字、空间和时间信息的处理。
- [Qwen2.5-Omni](./wiki/concepts/Qwen2.5-Omni.md)：Qwen2.5-Omni 将文本、图像、音视频输入与文字和语音输出放进统一交互模型，重点是跨模态与流式处理。
- [Qwen3.5-Omni](./wiki/concepts/Qwen3.5-Omni.md)：本库的 Qwen3.5-Omni 快照延续端到端音视频交互路线，应按当时资料核对支持模态和具体接口。
- [Sora 2](./wiki/concepts/Sora%202.md)：本库的 Sora 2 资料关注视频、原生音频和场景一致性，创作时应按镜头与声音分别验证结果。
- [Veo 3.1](./wiki/concepts/Veo%203.1.md)：本库的 Veo 3.1 快照连接视频、音频和参考控制与创作产品，理解能力应按具体输入与工作流程。
- [Kling VIDEO 3.0 Omni](./wiki/concepts/Kling%20VIDEO%203.0%20Omni.md)：本库的 Kling 3.0 Omni 资料关注参考资产、角色一致性、语音与分镜控制，适合按创作环节理解能力。
- [Wan2.6](./wiki/concepts/Wan2.6.md)：本库的 Wan2.6 资料覆盖参考视频、多镜头与音画同步等生成能力，适合沿叙事和控制流程理解。
- [Vidu Q2-Pro](./wiki/concepts/Vidu%20Q2-Pro.md)：本库的 Vidu Q2-Pro 资料关注参考图驱动、扩展和相关视频接口，选择用法时要按接口与输出要求核对。
- [Seedance 2.0](./wiki/concepts/Seedance%202.0.md)：本库的 Seedance 2.0 资料关注多参考、编辑、续写与音视频生成，适合从完整创作流程理解功能。
- [Stable Diffusion](./wiki/concepts/Stable%20Diffusion.md)：Stable Diffusion 在压缩后的潜空间进行图像扩散，将生成模型推向可本地运行和适配的开放生态。
- [FLUX.2](./wiki/concepts/FLUX.2.md)：本库的 FLUX.2 资料关注图像生成与编辑、多参考控制和产品分层，具体能力应按型号与快照核对。
- [Vision Banana](./wiki/concepts/Vision%20Banana.md)：Vision Banana 用图像生成模型适配视觉感知任务，将分割、深度等结果表达为图像，探索统一视觉输出。
- [Qwen-Image](./wiki/concepts/Qwen-Image.md)：Qwen-Image 是 Qwen 的图像生成与编辑支线，关注视觉内容与文字渲染，需与图像理解模型区分。
- [Qwen-Image-Layered](./wiki/concepts/Qwen-Image-Layered.md)：Qwen-Image-Layered 将图像分成可独立处理的 RGBA 图层，关注编辑时保留内容与合成关系。
- [AlphaVAE](./wiki/concepts/AlphaVAE.md)：AlphaVAE 为带透明度的图像学习压缩表示，帮助生成模型同时处理颜色和透明边缘，是透明图像生成的底层组件。
- [RGBA 图层图像](./wiki/concepts/RGBA%20%E5%9B%BE%E5%B1%82%E5%9B%BE%E5%83%8F.md)：RGBA 在颜色之外保存透明度，图层表示再把对象分开，让局部编辑、移动与重新合成更可控。
- [Florence-2](./wiki/concepts/Florence-2.md)：Florence-2 用文字提示指定视觉任务，再输出描述、定位等结果，研究统一的视觉理解接口。
- [dots.ocr](./wiki/concepts/dots.ocr.md)：dots.ocr 将页面布局、内容识别和阅读关系组织到文档视觉语言模型中，目标是更完整的页面解析。
- [Kosmos-2](./wiki/concepts/Kosmos-2.md)：Kosmos-2 把文字描述与图像对象的位置连接起来，让模型能说明“这段话指向哪个对象”。
- [Kosmos-2.5](./wiki/concepts/Kosmos-2.5.md)：Kosmos-2.5 为文字密集图像生成带位置的文本或结构化 Markdown，帮助保留页面内容与组织。
- [OFA](./wiki/concepts/OFA.md)：OFA 用统一序列到序列框架表达多种视觉与语言任务，让任务指令决定输入如何转成输出。
- [data2vec](./wiki/concepts/data2vec.md)：data2vec 让学生从部分可见输入预测教师的完整上下文表示，用相近学习目标处理语音、图像和文字。
- [HuBERT](./wiki/concepts/HuBERT.md)：HuBERT 利用语音聚类产生的离散单元做遮挡预测，从未标注音频学习语音表示，之后可适配识别等任务。
- [Dense Retrieval](./wiki/concepts/Dense%20Retrieval.md)：稠密检索将查询与文档变成向量，按相似度寻找候选，能捕捉部分语义关系，但也会漏掉精确实体。
- [DPR](./wiki/concepts/DPR.md)：DPR 分别把问题和文本段落编码成向量，再用向量相似度快速召回候选，是开放域问答的检索组件。
- [ColBERT](./wiki/concepts/ColBERT.md)：ColBERT 提前编码文档的多个词元，在查询时进行较细粒度的相似度交互，折中搜索效果与计算成本。
- [SimCSE](./wiki/concepts/SimCSE.md)：SimCSE 用对比学习适配句向量，无监督版本把同一句话的不同 dropout 表示视为正例，方法简洁。
- [Sentence-BERT](./wiki/concepts/Sentence-BERT.md)：Sentence-BERT 把句子独立编码成向量，再比较相似度，支持预计算文档表示和高效语义检索。
- [Prompt Tuning](./wiki/concepts/Prompt%20Tuning.md)：提示微调冻结模型主体，只训练连续提示向量，为具体任务提供低参数适配；这些向量不是手写提示词。
- [Tip-Adapter](./wiki/concepts/Tip-Adapter.md)：Tip-Adapter 用少量任务示例的特征与标签缓存辅助 CLIP 分类，减少重新训练的成本；还需区分可训练变体。
- [PubTables-1M](./wiki/concepts/PubTables-1M.md)：PubTables-1M 提供大规模表格抽取与结构标注，帮助模型学习单元格、行列和合并关系。
- [DocLayNet](./wiki/concepts/DocLayNet.md)：DocLayNet 是文档版面标注数据集，提供不同页面类型的区域标签，用于训练和评估版面分析。
- [LayoutLMv3](./wiki/concepts/LayoutLMv3.md)：LayoutLMv3 联合学习文档文字、图像和版面，用统一遮挡与对齐任务预训练，为文档理解提供表示。
- [DocLLM](./wiki/concepts/DocLLM.md)：DocLLM 将文字与页面位置关系一起纳入语言模型，帮助理解字段、表单和票据，而不只读取文字内容。
- [GLM-OCR](./wiki/concepts/GLM-OCR.md)：GLM-OCR 是面向文字和文档理解的专门模型，适用性要按页面类型、输出结构与部署成本评估。
- [PaddleOCR](./wiki/concepts/PaddleOCR.md)：PaddleOCR 是 OCR 与文档处理工具链，既包含文字识别，也涉及版面、结构和相关应用模块。
- [TrOCR](./wiki/concepts/TrOCR.md)：TrOCR 用预训练图像编码器和文本解码器，将文字图像直接生成字符序列，重点是识别环节。

## Authors

- [Qwen Team](./wiki/authors/Qwen%20Team.md)：这里集中阅读 Qwen 团队已收录的模型发布与报告，先按 VL、Omni、Image 和语言主干选方向，再核对代际。
- [Junyang Lin](./wiki/authors/Junyang%20Lin.md)：这里按已收录来源阅读 Junyang Lin 相关的 Qwen 与多模态工作，重点是不同任务如何共享模型和接口。
- [Jingren Zhou](./wiki/authors/Jingren%20Zhou.md)：这里汇集 Jingren Zhou 相关的 Qwen 与多模态来源，按语言、视觉和图像分支定位具体研究。
- [Jinze Bai](./wiki/authors/Jinze%20Bai.md)：这里连接 Jinze Bai 相关的 Qwen 报告与家族页面，阅读时可先看具体代际，再回到训练与能力证据。
- [Shuai Bai](./wiki/authors/Shuai%20Bai.md)：这里按已收录来源阅读 Shuai Bai 相关的 Qwen 与统一多模态工作，先分清输入输出，再看共享建模方式。
- [Hugo Touvron](./wiki/authors/Hugo%20Touvron.md)：这里按已收录来源阅读 Hugo Touvron 相关的 LLaMA 研究，重点是基础模型训练与后续对话适配的区别。
- [Joseph Redmon](./wiki/authors/Joseph%20Redmon.md)：这里集中阅读 Joseph Redmon 相关的 YOLO 早期工作，了解统一检测、速度与精度的设计取舍。
- [Ali Farhadi](./wiki/authors/Ali%20Farhadi.md)：这里集中阅读 Ali Farhadi 相关的 YOLO 早期检测资料，重点是对象检测怎样向实时统一预测发展。
- [Brandon Smock](./wiki/authors/Brandon%20Smock.md)：这里整理 Brandon Smock 相关的表格研究，可从数据标注、结构恢复与 GriTS 评测三个问题开始读。
- [Rohith Pesala](./wiki/authors/Rohith%20Pesala.md)：这里连接 Rohith Pesala 相关的表格研究，重点是训练数据、网格结构和模型评价怎样对应。
- [Robin Abraham](./wiki/authors/Robin%20Abraham.md)：这里连接 Robin Abraham 相关的表格研究，可先看标注数据、结构比较与评测一致性。
- [Qwen Team - Alibaba](./wiki/authors/Qwen%20Team%20-%20Alibaba.md)：这里沿 Qwen 团队资料阅读语言、视觉、音视频与图像生成分支，家族综述和时间线帮助辨认版本。
- [Meta AI](./wiki/authors/Meta%20AI.md)：这里连接 Meta AI 的 Llama、代码模型与安全分类分支，可按基础模型、任务适配和系统检查阅读。
- [OpenAI](./wiki/authors/OpenAI.md)：这里连接 OpenAI 相关的语言模型、反馈训练、长任务摘要与视频生成来源，按具体任务进入原报告。
- [Microsoft Research](./wiki/authors/Microsoft%20Research.md)：这里连接 Microsoft Research 相关的 LoRA、Kosmos、Florence 和文档理解工作，分别看适配、定位与统一视觉任务。
- [Google Research](./wiki/authors/Google%20Research.md)：这里连接 Google Research 相关的 Transformer、BERT、T5 与 PaLM 等来源，可沿架构、理解和生成路线阅读。
- [DeepMind](./wiki/authors/DeepMind.md)：这里连接 DeepMind 相关的训练预算、Gemma 开放模型和视觉表示资料，按研究问题选择入口更容易阅读。
- [Kuaishou Technology](./wiki/authors/Kuaishou%20Technology.md)：这里从 Kling 已收录资料了解快手的视频生成路线，重点是角色、参考、分镜和音画控制。
- [Alibaba Group](./wiki/authors/Alibaba%20Group.md)：这里按 Qwen 与 Wan 等已收录来源了解阿里相关模型，分别进入语言、多模态和视频生成方向。
- [ShengShu Technology](./wiki/authors/ShengShu%20Technology.md)：这里从 Vidu 已收录资料了解生数科技的视频生成路线，阅读参考驱动、续写和具体视频接口。
- [DeepSeek](./wiki/authors/DeepSeek.md)：这里集中阅读 DeepSeek 的语言底座、推理训练、长上下文和 OCR 资料，各分支的实验和版本应分别核对。
- [Moonshot AI](./wiki/authors/Moonshot%20AI.md)：这里沿 Kimi 的长上下文、推理强化学习和后续专家混合系统阅读，模型能力与执行基础设施需要分开看。
- [MiniCPM - ModelBest](./wiki/authors/MiniCPM%20-%20ModelBest.md)：这里集中阅读 MiniCPM 与 MiniCPM-V 的已收录资料，研究小模型训练和轻量多模态部署的取舍。
- [Stability AI](./wiki/authors/Stability%20AI.md)：这里沿 Stable Diffusion 的已收录来源了解潜空间生成与开放图像生态，训练方法和具体发布条件分别核对。
- [Black Forest Labs](./wiki/authors/Black%20Forest%20Labs.md)：这里集中阅读 Black Forest Labs 的图像生成资料，沿 FLUX 路线了解参考控制、编辑与模型发布。
- [ByteDance Seed](./wiki/authors/ByteDance%20Seed.md)：这里按已收录来源阅读 ByteDance Seed 的多模态与生成媒体研究，视频创作可先进入 Seedance 资料。

## Comparisons

- [AI 能力评测：任务、过程与预测](wiki/comparisons/AI%20%E8%83%BD%E5%8A%9B%E8%AF%84%E6%B5%8B%EF%BC%9A%E4%BB%BB%E5%8A%A1%E3%80%81%E8%BF%87%E7%A8%8B%E4%B8%8E%E9%A2%84%E6%B5%8B.md)：基准成绩、探索案例、过程监控、社会趋势与未来预测是不同证据。先判断材料属于哪一种，再看数据、评价和外推条件；多篇材料不能自动拼成 AGI 已实现的证明。

- [推理优化：量化、缓存与硬件](wiki/comparisons/%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96%EF%BC%9A%E9%87%8F%E5%8C%96%E3%80%81%E7%BC%93%E5%AD%98%E4%B8%8E%E7%A1%AC%E4%BB%B6.md)：先定位瓶颈再选优化：量化减表示成本，剪枝改有效权重，缓存复用已有计算，调度提高资源利用率。它们可以配合，但速度收益不能简单相乘。

- [文档与表格的输入输出接口](wiki/comparisons/%E6%96%87%E6%A1%A3%E4%B8%8E%E8%A1%A8%E6%A0%BC%E7%9A%84%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%E6%8E%A5%E5%8F%A3.md)：页面转写、表格结构恢复、单表问答和电子表格压缩需要不同表示。先决定保留哪些行列、坐标、样式与运算，再选模型；结构合法和答案正确要分别验收。

- [句向量、稠密召回与交互排序](wiki/comparisons/%E5%8F%A5%E5%90%91%E9%87%8F%E3%80%81%E7%A8%A0%E5%AF%86%E5%8F%AC%E5%9B%9E%E4%B8%8E%E4%BA%A4%E4%BA%92%E6%8E%92%E5%BA%8F.md)：句子相似度、找到相关段落和给候选排序是三个目标。双塔便于离线索引，交互模型能细看词间关系；训练与几何修正则决定向量是否适合对应任务。

- [LLM Wiki 与检索和文档解析方法](./wiki/comparisons/LLM%20Wiki%20与检索和文档解析方法.md)：本库继续把可读、可修订的 wiki 作为知识产物，再用按需读取、证据定位、查询分层与评测改善执行。额外工具要用真实问题验证收益。
- [arXiv 与 Hugging Face 论文发现入口](./wiki/comparisons/arXiv%20与%20Hugging%20Face%20论文发现入口.md)：方向还不明确时，可用 Hugging Face 发现社区关注的候选；有具体问题后，用 arXiv 分类、关键词和引用继续扩展。热度只是筛选线索。
- [RLHF vs DPO vs ORPO vs KTO](./wiki/comparisons/RLHF%20vs%20DPO%20vs%20ORPO%20vs%20KTO.md)：四条路线都利用反馈改变模型，但数据与流程不同：RLHF 有奖励和在线优化，DPO 用成对偏好，ORPO 合并训练，KTO 接受单项好坏反馈。
- [开放模型家族与中国重要家族对照](./wiki/comparisons/%E5%BC%80%E6%94%BE%E6%A8%A1%E5%9E%8B%E5%AE%B6%E6%97%8F%E4%B8%8E%E4%B8%AD%E5%9B%BD%E9%87%8D%E8%A6%81%E5%AE%B6%E6%97%8F%E5%AF%B9%E7%85%A7.md)：比较开放模型先区分权重可得、许可限制与训练透明度，再看模型分支、任务和部署成本；家族标签会随代际改变。
- [SGLang 与 vLLM 架构对比](./wiki/comparisons/SGLang%20%E4%B8%8E%20vLLM%20%E6%9E%B6%E6%9E%84%E5%AF%B9%E6%AF%94.md)：vLLM 早期从缓存分页与高吞吐服务切入，SGLang 从多调用程序与前缀复用切入；后续能力已交叉，选择需看版本与工作负载。
- [Muon 与 AdamW](./wiki/comparisons/Muon%20%E4%B8%8E%20AdamW.md)：AdamW 按参数元素自适应更新，Muon 对二维矩阵更新做整体变换。Muon 的计算效率报告需保留训练条件，实际方案常是混合分组。

## Timelines

- [Qwen 系列演进](./wiki/timelines/Qwen%20系列演进.md)：这条时间线把 Qwen 的语言主干、VL、Omni 与 Image 分支放在一起，帮助找版本关系；技术判断再进入主题与单篇来源。

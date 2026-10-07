# Jiang et al. - 2025 - MME-CoT Benchmarking Chain-of-Thought in Large Multimodal Models for Reasoning Quality, Robustness, and Efficiency

- Source PDF: `raw/pdf/Jiang et al. - 2025 - MME-CoT Benchmarking Chain-of-Thought in Large Multimodal Models for Reasoning Quality, Robustness, and Efficiency.pdf`
- Source SHA256: `69cc977d44ab2e286a4e565b6d240707465439854a436f355a6e9679936398bd`
- Source URL: https://arxiv.org/abs/2502.09621
- Generated from: `scripts/extract_pdf_text.py`

- Extraction: `pymupdf-pages-v2` (sorted text, page anchors; tables/formulas/figures require review)

## Extracted Text

<a id="source-section-0"></a>

<a id="page-1"></a>

### PDF 第 1 页

MME-CoT: Benchmarking Chain-of-Thought in Large Multimodal Models
                       for Reasoning Quality, Robustness, and Efficiency



                 Dongzhi Jiang∗1  , Renrui Zhang∗† 1  , Ziyu Guo 2  , Yanwei Li‡ 3  , Yu Qi‡ 4  , Xinyan Chen‡ 1
                          Liuhui Wang‡ 5  , Jianhan Jin‡ 6  , Claire Guo‡ 7  , Shen Yan 3  , Bo Zhang 8
                                   Chaoyou Fu 6  , Peng Gao 8  , Hongsheng Li 1

                                      1 CUHK MMLab   2 CUHK MiuLar Lab   3 ByteDance   4 NEU   5 UPenn
                                                 6 NJU   7 CUHK (Shenzhen)   8 Shanghai AI Laboratory
                              {dzjiang,renruizhang}@link.cuhk.edu.hk2025                            ∗Core contribution                                                                       † Project lead   ‡ Equal contribution
Feb                                       Project Page: https://mmecot.github.io/
13                    Abstract                                                             Precision
             Answering  questions with Chain-of-Thought                                                     92.0
             (CoT) has significantly enhanced the reasoning ca-                                                                                                                          Quality
                  pabilities of Large Language Models (LLMs), yet            Reflection                              85.4
                                                                                                                                                                                        51.2    Recall                     its impact on Large Multimodal Models (LMMs)            Quality     100.0                   80.2              50.5
                     still lacks a systematic assessment and in-depth                                                         79.5         49.249.3
                 investigation. In this paper, we introduce MME-                                                         77.3    44.2                                                                                                                                          72.2[cs.CV]
            CoT, a specialized benchmark evaluating the CoT                                                 61.7     73.6 41.1
               reasoning performance of LMMs, spanning six                                               60.6                                                                                                                                                                                                                                                                                                                                                                              Efficiency                           -2.9 -6.5
              domains: math, science, OCR, logic, space-time,                                                83.7            -3.1
                                                                                                                                                              -2.0
             and general scenes. As the first comprehensive                                    90.6          -0.4          -1.7
                                                                                                                          92.0                              -1.0
               study in this area, we propose a thorough evalu-                                  92.2               0.0                       2.9
                                                                              Relevance    92.9                ation suite incorporating three novel metrics that                                                                               Stability
                                                                              Rate                                   2.4                assess the reasoning quality, robustness, and effi-
               ciency at a fine-grained level. Leveraging curated                                                                                                                                                                            5.1     Robustness
                high-quality data and a unique evaluation strategy,
           we conduct an in-depth analysis of state-of-the-                                                                                                                  Efficacy
                  art LMMs, uncovering several key insights: 1)
             Models with reflection mechanism demonstrate a               Qwen2-VL-72B            GPT-4o            Virgo-72B
                superior CoT quality, with Kimi k1.5 outperform-
                                                                                  InternVL2-5-78B-MPO       Kimi k1.5         QVQ-72B               ing GPT-4o and demonstrating the highest quality
                  results; 2) CoT prompting often degrades LMMarXiv:2502.09621v1                                                             Figure 1: Chain-of-Thought Performance of Leading
              performance on perception-heavy tasks, suggest-
                                         LMMs in MME-CoT. Our evaluation suite assesses LMMs
               ing a potentially harmful overthinking behavior;
                                                                    using three novel metrics that yield six distinct scores. Re-
             and 3) Although the CoT quality is high, LMMs
                                                                                   sults reveal that current open-source models, including those
              with reflection exhibit significant inefficiency in
                                                                  with reflection capabilities, still lag behind closed-source
              both normal response and self-correction phases.
                                                              models like GPT-4o and Kimi k1.5 in key aspects of chain-
          We hope MME-CoT serves as a foundation for
                                                                      of-thought reasoning.
              advancing multimodal reasoning in LMMs.

          1. Introduction                                   by the recent OpenAI o1 (OpenAI, 2024a) and DeepSeek-
                                                  R1 (Guo et al., 2025a). By engaging in a more deliberate,
        The emergence of Chain-of-Thought (CoT) (Wei et al.,    stepwise reasoning process before reaching a final answer,
          2022) in Large Language Models (LLMs) has demonstrated     this methodology presents an effective solution in tackling
          promising advances in reasoning capabilities, exemplified    complex scenarios.

                                                         1
<a id="page-2"></a>

### PDF 第 2 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency

                           CoT Evaluation Suite

                  Quality                           Robustness                            Efficiency
      Precision                                       Perception Task     Reasoning Task          Question: How many cars are presented in all images?
                                                                  No Need for Reasoning            Need Reasoning
                 Model Output             Judgment                                          Relevance Rate             Step Judgment

                                                                                                                                                     are 2                                                                                                                                                           cars in                 1. ABCD is a square       ✓                                                                                     There                                                                                                                                                             Relevant                                                                                                                                               the first                                                                                                                                                         image.
                 2.[ABCD] = 484          ✗
                 3.[AEH]                         = [BEG]                                                                                                                                                   building                                                                                                                                                            is new                                 = 85      ✓                                                                                     The                 4.[Shadow]                            = [ABCD]–                                                                                                                                               and                                                                                                                                                   fancy, ...                                                                                                                                                                               Irrelevant                 [AEH]-[BEG]= 314         ✗                                                                                     The people are ...
                                                                                                        Question + Please answer directly.
      Recall                                        ……                ……          Reflection Quality         Reflection Step
          Key Step Annotation              Judgment                        (Direct Answer)                           (Direct Answer)                      Alternatively, maybe       Judgment

                                                                                                                                                                              Invalid                          ✗             Stability            Efficacy                     Isurecan...if thatBut helpsI'm not.
          1. [ABCD] = 625
          2. [AEH] = [BEG] = 85    Match    ✓                             Question + Please think step by step.                                   Let’s double-check
                                                                                                                                               with                                                                                                                                                    a new                                                                                                                                                                    Valid          3. [Shadow] = 455        Prediction    ✗                                                  ……                                                                    ……                                                                                                                                               perspective:                                                                                                                                                            ...
                                                                                      (CoT                                                                                                  Answer)                      (CoT                                                                                                                                     Answer)



                               Dataset Overview

                 Math                                General Scenes                              Space-Time

                                                                              Question:                                                                              Key Step Annotation                            Key Step Annotation                                                                                What emotion
                                                                                                                                               is                                                                                                           expressed                                                                                                                               in                                                                                             Key                                                                                                                 Caption:                                   Key                                              Caption:                                                                                                             the                                                                                                        artwork                                                                                                                               in                                                                                                                         Question:                                                                                                                                                   1.                                                                                                           The                                                                                                                                                      elderly                                                                                         man…                                                                                                                         Key                                                                                                                                                      Step                                                                                                                                                  Annotation                                                        1.                                 AB                                     =                                                   8                                                                                                             the                                                                                                                       picture?                                                                                                          How                                                                                                                                many                                                                                                                                                                                                         different                                                                                                                                                                                         tones                                                        2.                                          The                                                          distance                                                          between                                                                                           A:                                                                                                                                   fear.                                                                                                                                               Key                                                                                                                                                                             Caption:                                                                                                                                                                                of                                                                                                                                                                        blue                                                                                                                                                                                   are                                                                                                                                                                                 accenting                                                                                   …                                                                                             Key                                                                                                                  Conclusion:                               AB                                             and                                                       A'B’                                                                                    is                                                                         1.      Question:                                                                                                       B:                                                                                                   awe.                                                                                                                                                                                                                                    1.                                                                                                                                                                                                        In                                                                                                                                                                                                                  the                                                                                                                                                                                                                                                                                                 first                                                                                                                                                                                                                                  image...                                                                                                                                  A:                                                                                                                      None                                                                                                                                                                           of                                                                                                                                                                        the choices                                                                                                                                                   1.                                                                                                           The                                                                                                                   young                                                                                                                        woman's         ...The              water                      surface                          width                                  A'B'                                                                                                                                                                                                                                    2.                                                                                                                                                                                                        In                                                                                                                                                                                                                  the                                                                                                                                                                                                 second                                                                                                                                                                    image…                                                                                                C:                                                 …                                   Key                                              Conclusion:                                                                                                                                               attentive                                                                                                                                  gaze                                                                                                                                                                     suggests...                                                                                                                                                   provided         in           the               bridge                     hole                                    is:
                                                        1.                                          The                                                          radius                                                                  of                                                                  the                                                                                                                                                   2.                                                                                                           The                                                                                                                           answer                                                                                                                                                                                                 is B.                                                                                                                                                   B:                                                                                                                                                                    three                                                                                                                                                C:                                                                                                                                                  five…       A:          √15               meters                       B:                      2√15                              meters                                                                                                                                                Key                                                                                                                                                                               Conclusion:                                                                       Answer:                                                               B                                                     semicircle                                                                                        is...    …                                                                                                                                                                                                                                     1.                                                                                                                                                                                                        In                                                                                                                                                                                                                                                             total,                                                                                                                                                                                                                                there                                                                                                                                                                                                                                 are five …
                                                        2.                                          The                                                   width                                                        A’B’                              …                                                                                                              Answer:                                                                                  C
      Answer: B

                     Logic                        OCR                                        Science
                                                              Question:
                                                                  Which of the
                  Question:                                               following figure
                  …Your                                   task      Key Step Annotation           does …                                                                 Key Step Annotation
                                  is to select
                          the                               correct                                    Key Caption: /            Answer: D                                                                                                                         Question:                 Key Caption: /                       shape                           from
                            six options                                 Key Step Annotation                             A conducting sphere
                           (labeled A to         Key Conclusion:                                                                                               holding a charge of +10          Key Conclusion:
                                                         1. The third column                 Key Caption:            Key Conclusion:                𝜇C…Which diagram                  1. The electric field...
        F) to fill the empty box...                        is the combination…                          1. Option B: This figure           1. Options B, C, and E are              shows the electric…                   2. The electric field
                                                         2. The third…                                        is …                             consistent with molecular...                                                           lines outside...
      Answer: C                                                                                                Answer: D


Figure 2: An Overview of MME-CoT. Our benchmark contains a comprehensive CoT evaluation suite with three novel
aspects and a meticulously curated dataset encompassing six categories.


In parallel, the multimodal extensions of LLMs, termed     ciently systematic and thorough, limiting our understanding
Large Multimodal Models (LMMs), have demonstrated re-    of multimodal reasoning and its further development.
markable proficiency across diverse visual domains, e.g.,
                                                To bridge this gap, we propose MME-CoT, a comprehen-
general image recognition (Zhang et al., 2023; Zhu et al.,
                                                                 sive and specialized benchmark for evaluating the CoT rea-
2023; OpenAI, 2023; Zhang et al., 2024a), temporal video
                                                        soning skills within LMMs (Figure 2). Our benchmark
understanding (Li et al., 2023; Chen et al., 2023), and 3D
                                                          spans six fundamental domains: math, science, OCR, logic,
geometry perception (Guo et al., 2024b; Xu et al., 2023;
                                                              space-time, and general scenes, encompassing a broad range
Guo et al., 2023; Jia et al., 2024). However, to what ex-
                                                            of CoT-relevant scenarios. Unlike the simplistic metrics
tent and how much CoT reasoning can benefit multimodal
                                                       used in previous studies, MME-CoT introduces a rigorous
challenges still remains an open question. Although some
                                                              evaluation framework that delves into the fine-grained CoT
previous efforts (Zhang et al., 2024c; Yu et al., 2023; Zhang
                                                          process of LMMs, assessing reasoning quality, robustness,
et al., 2024d; Guo et al., 2025b) have been made to evaluate
                                                    and efficiency.  Specifically, we address three critical re-
the CoT capabilities of LMMs, their examination is insuffi-
                                                           search questions as follows:


                                                2
<a id="page-3"></a>

### PDF 第 3 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency

  1. Is each intermediate CoT step logically valid and   when applying CoT on the perception tasks. This signifi-
     faithful without hallucination? The outcome-oriented    cantly impedes the applicability of models using CoT rea-
     evaluation paradigm, where most current benchmark    soning as a default practice. Moreover, for CoT efficiency,
     adapts, omits the scenario where the model reaches the   we notice that not all steps within the long CoT are related to
     correct answer through flawed logic or random guess.    answering the question, and the model could be distracted
    This causes an illusion of inflated reasoning capabili-   by the image content, especially when handling general
      ties in the model. To delve into the reasoning process,    scenes, space-time, and OCR tasks. Around 30% to 40% of
   we introduce two interpretable metrics to evaluate the     reflection steps fail to help answer questions, pointing out
    Quality of CoT: 1) Recall, which quantifies reason-     critical issues of current models’ reflection capabilities.
     ing informativeness by measuring the proportion of
                                                 The contributions of this paper are summarized as follows:
     ground-truth solution steps appearing in the response;
     2) Precision, which measures faithfulness by evaluat-
                                                                            • The MME-CoT benchmark is curated, covering a com-     ing how many of the generated steps are accurate.
                                                               prehensive scope of six multimodal reasoning scenar-
  2. Does CoT interfere with perception tasks, and to what           ios. The data collection and annotation process under-
     extent does it enhance reasoning tasks? While exist-        goes rigorous human verification, aiming to provide
     ing studies primarily focus on the performance im-         the community with a high-quality evaluation dataset
    provements CoT brings to reasoning tasks, they often          for multimodal reasoning.
     overlook whether CoT could inadvertently disrupt the
                                                                            • We identify critical issues in existing benchmarks, and
    model’s ability to solve perception tasks that require
                                                                introduce a thorough evaluation suite specialized for
    minimal reasoning. To this end, we present the first in-
                                                            multimodal CoT reasoning, which meticulously exam-
     vestigation into the Robustness of CoT in LMMs. Our
                                                                    ines the reasoning quality, robustness, and efficiency.
    benchmark incorporates two task categories (percep-
     tion and reasoning), and employs two distinct prompt-                                                                            • We conduct extensive experiments and analysis on
     ing strategies (‘direct answer’ and ‘step-by-step’) to as-                                                                           state-of-the-art LMMs with reasoning capabilities. We
     sess two metrics: 1) Stability, which examines whether                                                       summarize our observations and insights, hoping to
   CoT negatively impacts the model’s performance on                                                                       inspire future advancements of reasoning performance.
     direct perception tasks; 2) Efficacy, which measures
     the extent to which CoT enhances the model’s perfor-
    mance on complex reasoning tasks.                    2. Dataset Curation

                                                                    2.1. Data Composition and Categorization.
  3. How can we assess the efficiency of CoT in a long
    reasoning process? Recent o1-like models have dis-   MME-CoT composes 6 major domains with 17 subcate-
     tinguished themselves by employing excessively long     gories, as visualized in Fig. 3. Different from textual reason-
   CoT and reflection steps. This raises a critical trade-off    ing questions, the extra visual input significantly enriches
     question: does this approach strike an optimal balance    the scope of the visual reasoning questions. With the image
    between accuracy and computational cost? To investi-    input, the model needs to frequently visit the image for rel-
     gate this, we present the first study on the Efficiency    evant information according to current reasoning progress.
     of CoT in LMMs. We evaluate efficiency using two    Describing the image area of interest becomes a crucial part
    key metrics: 1) Relevance Rate, which assesses the    of the CoT process. Thus, in addition to complex prob-
     proportion of generated content that contributes to an-   lems demanding rigorous logic, commonsense scenarios
    swering the question.  2) Reflection Quality, which    also pose a challenging reasoning problem, as shown in
     analyzes whether each reflection step drives the ques-    the general scenes in Fig. 2. To maintain focus on the rea-
     tion towards correctness.                              soning process, we exclude questions that require complex
                                                           domain-specific theorems or specialized knowledge.

Through our systematic evaluation and analysis, we discover    In addition, to evaluate CoT robustness detailed in Sec-
that the fine-grained reflection capability greatly enhances     tion 3.2, we incorporate a variety of perception tasks along
the CoT quality, e.g., QVQ achieves F1 Score of 62.0%,    with the reasoning tasks in the benchmark. The reasoning
largely surpassing Qwen2-VL-72B by 6.8%. Kimi k1.5    tasks contain questions that demand multi-step logical in-
beats GPT-4o and achieves the best quality. As for the     ference, while the perception tasks consist of questions that
robustness, we surprisingly find that most models are in-    primarily test visual recognition abilities or require very
terfered with by CoT on the perception tasks, implying a    minimal reasoning.  Existing benchmarks often conflate
harmful overthinking behavior. The worst case happens    these two types of tasks, with perception and reasoning
in InternVL2.5-8B, where we witness a 6.8% degradation    questions frequently appearing within the same categories.

                                                3
<a id="page-4"></a>

### PDF 第 4 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency



                                                                                                                                           Statistic                     Number

                                                                                                                                    questions                                                                                                                                                           1,130
                                                5.1%                                                                                               Total- Reasoning                                                                                                                                             questions                                                                                                                                         837                                                                                                                                                       (74.1%)                          Search                                                                                       Chemistry
                                                                                                                                 Multiple-choice                                                                                                                                                   questions                                                                                                                                            431                         1.2% 6.7%SenseCommon 4.7%Details
                                                                                                                          Free-form questions              406          Diagram                                                                                                                                                                   - Perception                                                                                                                                             questions                                                                                                                                         293                                                                                                                                                       (25.9%)                                                            Details            Spatial                           Physics              3.9%        General                                                                                                                                 Multiple-choice                                                                                                                                                   questions                                                                                                                                            275                                      Scenes                        16.4%              21.8%                                                        26.3%
   Doc                 11.4%  Science                                    General   Space-                              Free-form questions              18
                                                                    Time                              Total                                                                                                                     key                                                                                                                                              step annotation                                                                                                                                                           3,865                                             3.7%                                                                     Scenes                       25.2%     Biology   11.2%     16.4%OCR                                                                      33.5%                                                                                                                                                                   - Total                                                                                                                                          inference                                                                                                                                              conclusions         2,667                                                              41.6%                                                                        Application                                     Space-                                                                   2.7%
                                                                                                                                                                   -                                                                                                                    Average                                                                                                                                             inference                                                                                                                                                 conclusions                                                                                                                                                                        3.2
                                Time8.2%                      Math                                                                                                                                                                   -                                                                                                                                       Total                                                                                                                     image                                                                                                                                               captions                                                                                                                                                           1,198                                            Algebra                                                 OCR       Temporal     Temporal 3.2%                                Logic                                           4.3%      Common                             9.0%    29.9%                                                                                                                                                                   -                                                                                                                    Average                                                                                                                       image                                                                                                                                                  captions                                                                                                                                                                        1.4                                                                   24.9%                                                                        7.2%          5.0%                                            Sense                                                                                                                      Reference                                                                                                                     image                                                                                                                                               caption                                                                                                                                              item                                                                                                                                                           1,579                                                     Coordinate                      4.7%       Spatial                                         19.8%                             4.3%  3.3%         Plane         3.6%                                                Doc
                                                                                                                                        of                                                                                                                               unique                                                                                                                                 images                                                                                                                                                           2,380                                                        7.8%                Number                      Transformation                                                                                                 Average reference caption                1.9
                                                                                                 Number                                                                                                                                        of                                                                                                                               unique                                                                                                                                                 questions                                                                                                                                            808                                           Attribution     Geometry        15.9%Geometry                             Diagram17.1%                                                                                                 Number                                                                                                                                        of                                                                                                                               unique                                                                                                                                    answers                                                    Solid                                                                                                                       271
                                                                                         Maximum question length             477
                                                                                         Maximum answer length              15
          Reasoning Task                        Perception Task                     AverageAverage questionanswer lengthlength                41.21.2

     Figure 3: Category and Subcategory Distribution of MME-CoT.       Table 1: Key Statistics of MME-COT.


To address this, we implement a two-stage classification ap-    annotators are required to provide all possible methods. For
proach combining both model-based and human assessment.    reference captions, we also ask annotators to verify and
Initially, we leverage LMMs to guide the preliminary cate-    correct the details.
gorization by comparing their performance with and without
CoT prompting. We employ GPT-4o (OpenAI, 2024b) and                                                        3. CoT Evaluation Strategy
Qwen2-VL-7B (Wang et al., 2024b) to answer questions us-
ing both direct and CoT approaches. Superior performance    Existing benchmarks only focus on evaluating the final an-
with CoT indicates a reasoning-dominant subcategory, while    swer of the questions, leaving the whole chain of thoughts
comparable or inferior CoT performance suggests either     unvisited. We argue that the CoT process reflects reason-
perception-focused content or insufficient model reasoning    ing capability from multiple aspects, serving as a crucial
capabilities. The results are shown in Appendix B.2. Sub-   medium to understand LMM’s thinking pattern and defi-
sequently, expert annotators review individual questions to    ciency. Here, we present the first holistic CoT evaluation
finalize their classification. In total, MME-CoT contains     suite to facilitate a comprehensive understanding of the
1,130 questions with 3,865 key step annotation. The detailed   LMMs’ reasoning abilities. We detail the evaluation of cor-
statistics of data compositions are shown in Table 1. Please     rectness in Section 3.1, stability and efficacy in Section 3.2,
refer to Appendix B for more details about the distribution    and reflection quality in Section 3.3.
of data sources.
                                                                    3.1. CoT Quality Evaluation
2.2. Data Annotation and Review
                                                            Existing methods typically rely on state-of-the-art LLMs
To facilitate CoT evaluation, we provide key steps anno-    or LMMs to directly evaluate Chain-of-Thought reasoning
tation and reference image captions for all the reasoning    based on self-defined criteria, using only the final answer
questions. Key steps are defined as those that must be done    as a reference (Hao et al., 2024; Zhang et al., 2024c). We
to reach the correct answer. For efficient annotation, we     identify two primary issues with the strategy. First, the scor-
first employ GPT-4o to generate the answer rationale and    ing process only attends to the logical validity of each step,
image captions. For the rationale, we provide both ques-    omitting the helpfulness evaluation. Second, there is a large
tions and ground truth answers to the model, which yields    number of complex visual reasoning questions that even the
more accurate rationales compared to question-only prompt-    scoring model cannot solve. It is unreasonable for the scor-
ing. Annotators are then asked to provide key intermediate    ing model to judge another model’s reasoning process on
steps with the help of GPT-4o’s responses. For cases where    these questions without knowing the ground truth solution
GPT-4o fails to generate reasonable rationales, annotators    process. Therefore, building upon our annotated key steps
develop solutions independently. The intermediate steps fall    and reference image captions, we leverage two interpretable
into two categories: inference conclusion and image caption.    metrics to evaluate the CoT correctness: recall and preci-
Note that the final answer is also included as a concluding    sion (Figure 5). The two metrics respectively attend to the
inference. All the steps are reduced to the simplest form,   two aspects of the CoT correctness: informativeness and
retaining only core conclusions and relevant visual element    accuracy. We denote the key steps as S = C ∪I, where
descriptions. Notably, for problems with multiple solutions,   C = {c1, ..., cM} includes M key inference conclusions


                                                4
<a id="page-5"></a>

### PDF 第 5 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency


                                                                                                                                             Question

           Donna has a laser at C. She points the laser beam at the point E. The beam reflects off of DF at E and then off of FH at G, as shown, arriving at point B on AD. If
          DE=EF=1 m, what is the length of BD, in meters?


                      Background Information                    Image Caption                             Logical Inference


                                                    Model Response                                                            Step Partition

            To find the length of BD, we can use the properties of right triangles                 Step1 [background information]
                                                                                                  To find the length of BD, we can use the properties of right
            and the given information.                                                             triangles and the given information.
            1. **Identify the Right Triangles:**
              - Triangle CDE is a right triangle with angle CDE = 45°. Triangle EFG
                                                                                                  Step2 [image description]
            is a right triangle with angle EFG = 45°. Triangle EGB is a right                      Triangle CDE is a right triangle with angle CDE = 45°;
            triangle with angle EGB = 45°.                                                         triangle EFG is a right triangle with angle EFG = 45°;
            2. **Use the 45-45-90 Triangle Theorem:**                                              triangle EGB is a right triangle with angle EGB = 45°
              - In a 45-45-90 triangle, the sides are in the ratio 1:1:√2.
              - Since DE = EF = 1 m, the hypotenuse EG is 1√2 = √2 m.                              Step3 [background information]
            3. **Calculate EG:**                                                                   Use the 45-45-90 Triangle Theorem: In a 45-45-90
              - EG = DE + EF = 1 + 1 = 2 m.                                                        triangle, the sides are in the ratio 1:1:sqrt(2)
            4. **Use the Pythagorean Theorem in Triangle EGB:**
              - EG² = EB² + BG².                                                                   Step4 [logical inference]
              - 2² = EB² + BG².                                                                    Since DE = EF = 1 m, the hypotenuse EG is sqrt(2) m.
              - 4 = EB² + BG².
            5. **Calculate BD:**                                                                   Step5 [logical inference]
              - Since BD = EB + BG, we need to find EB and BG.                                     EG = DE + EF = 1 + 1 = 2 m.
              - From the Pythagorean theorem, we have EB² + BG² = 4.
              - Since EB = BG (both are the hypotenuses of the 45-45-90 triangles),                Step6 [logical inference]
            we can write 2EB² = 4.                                                                 4 = EB^2 + BG^2
              - Solving for EB, we get EB² = 2.
                                                                                                   Step7 [logical inference]
              - Therefore, EB = √2.                                                                EB = sqrt(2)
            6. **Calculate BD:**
              - BD = EB + BG = √2 + √2 = 2√2.                                                      Step8 [logical inference]
                                                                                                   BD = sqrt(2) + sqrt(2) = 2sqrt(2)

Figure 4: Illustration of Step Partition. We instruct GPT-4o to divide each step into three categories: image caption,
background information, or logical inference. The step partition result is later used to perform step-wise reasoning evaluation.
We focus on evaluating the image caption and logical inference steps, which are the keys to visual reasoning.


and I = {i1, ..., iN} includes N key image captions.        image caption, and background information. The logical
                                                            inference step draws an intermediate or final conclusionRecall.  We prompt GPT-4o (OpenAI, 2024) to determine
                                                       based on the previously obtained information. The imagewhether each key step occurs in the model’s CoT response.
                                                            caption step depicts elements of interest in the image. TheThen we calculate the ratio of the matched key steps Smatched
                                                     background information step states external knowledge oragainst all the annotated key steps:
                                                           question information. Visual reasoning can be primarily
                      Skmatched                         characterized as an interleaved sequence of image captions
            k0 = arg max               ,                 (1)    and logical inferences, so we focus on measuring precision                      k      |Sk|
                                                                for these two key step types. We assess the correctness of
                 Ck0matched               Ik0matched             logical inference steps (CP) and image caption steps (IP)
      RecallC =               ,   RecallI =               ,    (2)                    |Ck0|                   |Ik0|             using two criteria: 1. If the step exists in S, the step is cor-
                                                                       rect. 2. If the step is logically correct or faithfully depicts
                     Sk0matched                              the image based on the annotations, the step is also correct.
             Recall =               .                      (3)
                         |Sk0|                            Thus, we compute precision as:

where Sk denotes the kth method of the problem. Intuitively,
recall measures how many informative steps are reached                CPcorrect              IPcorrect
by the model. From another perspective, this metric also        PrecisionC =             ,   PrecisionI =             ,   (4)                                                                         |CP|                   |IP|
strictly examines the process’s rigorousness toward reaching
                                                               CPcorrect ∪IPcorrectthe correct answer, eliminating the probability of random                                                                           Precision =                               (5)
guessing. For questions with multiple methods, we compute                             |CP ∪IP|
the recall on the most matched method.


Precision.  We first instruct GPT-4o to partition the predic-     Intuitively, precision evaluates the faithfulness of each step,
tion into a sequence of steps P, as shown in Fig. 4. Each step    considering all the possible reasoning output. Finally, we
is categorized into one of three classes: logical inference,    calculate the F1 score as the metric of CoT quality.

                                                5
<a id="page-6"></a>

### PDF 第 6 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency


                                                      Question                                                   Key Step Annotation

                     Square A B C D has side length 22. Points G and H lie                                          1. GH is equal to 12.
                   on \overline{A B} so that A H = B G = 5. Points E                                             2.  The area of ABCD is 484.
                    and F lie outside square A B C D so that E F G H is a                                           3.  The area of EFGH is 144.
                        square. Compute the area of hexagon A E F B C D.                                                                                                                             4.  The area of AEH and BEG is 30.
                                                                                                                             5.  The area of AEFB equals the sum of areas of
                                                                                ABCD, EFGH, AEH, and BFG.
                                            Model Response                                            6.  The area of AEFB is 688.
                                     ...


                                                                                                                                           Eval Prompt

                                                                                                      Determine if each key step appear in the             Step Partition              Judge the correctness
                                                                                                                                          prediction.                     Recall                                                                       Precision
                                                           Prompt                                                                 Prompt

                                                                                                                                            Eval Result

  ✗     Step1 [logical inference]
           The area of hexagon AEFBCD can be broken down into:                                 ✓  1. GH is equal to 12. [Match]           a square ABCD, a square EFGH, and four triangles
           AHE, BGF, CGF, and DHE.
                                      ✗  2. The area of ABCD is 484. [Missing] ✓     Step2 [logical inference]
           The side length of EFGH is 12.                ✓  3. The area of EFGH is 144. [Match]
           Step3 [logical inference] ✓
           The area of square EFGH is 144 (12 × 12).
                                      ✗  4. The area of AEH and BEG is 30. [Conflict]
           Step4 [background information]
           Area of triangle = (base × height) ÷ 2               ✗  5. The area of AEFB equals the sum of areas of ABCD,
                 [logical                          inference]  ✗     Step5           Each triangle                         (AHE,                               BGF, CGF, DHE) has area of 72.                     EFGH, AEH, and BFG. [Conflict]

            Step6 [logical inference]                     ✗  6. The area of AEFB is 688. [Conflict]  ✗      Total area of hexagon AEFBCD = 216.

                     Precision = 2/5                                                 Recall = 2/6

Figure 5: Illustration of CoT Quality Evaluation. For recall, we prompt GPT-4o to match each key step annotation in the
prediction. For precision, GPT-4o is instructed to split the prediction into steps and determine the correctness of all the
image caption and logical inference steps.


3.2. CoT Robustness Evaluation                               direct prompt asks the model to directly provide the final an-
                                                             swer, while the CoT prompt instructs the model to perform
Here, we perform the first investigation on the robustness
                                                             step-by-step reasoning and finally give the answer. To di-
of CoT in visual reasoning. The effectiveness of CoT on
                                                                   rectly compare the performance difference caused by these
reasoning tasks has been verified in many works (Wei et al.,
                                                   two prompts, we conduct the direct evaluation, which only
2022; OpenAI, 2024a). However, how CoT impacts visual
                                                           judges the correctness of the final answer, i.e., accuracy. We
perception tasks or tasks requiring minimal reasoning still
                                                                  instruct GPT-4o mini (OpenAI, 2024) to extract the final
remains unknown. Despite the neglect, this question bears
                                                         answer, and then compare it with the ground truth answer,
great importance. In real-world applications, what task is
                                                            following the two-step procedure introduced in (Zhang et al.,
given is unknown in advance. Whether the model should
                                                          2024c).
perform CoT to solve the task is difficult to determine. In
fact, there exists no golden standard to determine which     Stability.  We define the performance difference of the two
question can benefit from CoT so far (Sprague et al., 2024).   prompts on the perception tasks P as the stability score:
Instead of trying to define this criterion, we examine the
                                                                                   Stability = AccPCOT −AccPDIR.            (6)performance of CoT across all kinds of tasks, both reason-
ing and perception. We argue that an ideal CoT process
                                                                     Intuitively, applying the CoT prompt to perception tasksshould assist in reasoning and not interfere with pure per-
                                                        should not degrade performance compared with the directception. Therefore, it can be applied for any tasks. Based
                                                        prompt. Thus, a model with stable CoT should be not lesson this, we propose to evaluate two metrics of CoT: stability
                                                          than 0. Otherwise, the model’s thinking process demon-and efficacy (Figure 6). We leverage two kinds of prompts:
                                                                    strates inconsistency and harm. The overthinking processthe direct prompt (DIR) and the CoT prompt (COT). The
                                                        pushes over the original correct judgment.

                                                6
<a id="page-7"></a>

### PDF 第 7 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency


                                                       Task Type                                                                                     Model Response
                                                                                                                                        ...

                                                                                                                                               with                                                                                                                                                    center                                                                                                                                                            O, and                                                                                                                                                                  there                                                                                                                                                                         are                                                                                                                                                                            points                                                                                                                                                                                    A,                                                                                                                                                                                       B,                                                                                                                                                                                          C,                                                                                                                                                                                             D,                                                                                                                                                                                                and E connected                                                                                                                                                                                                                in some                                                                                                                                                                                                                        way.                                                                                                                                                                                                                             The                                                                                                                                                                                                                                problem            Perception Task           Reasoning Task                                        SocirclegivesI'veme got this geometry problem here. Let me try to understand what's going on. There's a                                                                                                                                                 that                                                                                                                                                       T equals                                                                                                                                                               80,                                                                                                                                                                    and the                                                                                                                                                                           minor                                                                                                                                                                                  arc                                                                                                                                                                                      AB                                                                                                                                                                                         is                                                                                                                                                                                            T/4                                                                                                                                                                                               degrees.                                                                                                                                                                                                        Also, there                                                                                                                                                                                                                     are                                                                                                                                                                                                                        angles                                                                                                                                                                                                                                OAC                                                                                                                                                                                                                                    and
                                                                                                                                        OBD given as 10 degrees and 5 degrees, respectively. I need to find the measure of angle AEB.

                                                                                                                                        First, I need to interpret the diagram based on the labels. It seems like E is outside the
                                                                                                                                        circle, and lines EC and ED intersect the circle at points C and D, respectively. Points A and B
                                                                                                                                        are on the circle, and O is the center.

                                                                                                                                        Given that T = 80, so T/4 = 20 degrees. That means the minor arc AB is 20 degrees.

                                                                                                                                        Now, since O is the center, OA and OB are radii of the circle, so they are equal in length.
                                                                                                                                        ...

          Question: The Sphinx is in what type      Question: The motion of the charged
                      of environment?                           point particle is                                     Relevance Rate
          GT Answer: Rainforest            GT Answer: Path a                                                                                                                                                       Eval Result
                                                                                                             Step Partition
                                                                                                                                                                                                           Step1                                                                                                                                                                                                                 [logical inference]
                                                                                                                                                                                                            ...                                                            Direct Answer                           ✓
                                   Please answer directly.
                                                                                                         Judge the relevance.                                                                                                        Direct
                                                                                        Prompt                                                               ✗      Step2...   [Image Caption]                                                                                                      Relevance
                 Rainforest                      Path a                                  Rate
           ✓            ✓                            Prompt    Tip:Relevance Definition: reach a                                  ...
                                                                                                                                              conclusion                                                                                                                                                           or                                                                                                                                                                                                            Step7                                                                                                                                                                                                                  [logical inference]                                                                                                                                                                     description helpful to       ✗                                                                                                                                                                                                            ...                                                                                                                               answer the                                                                                                                                                                question.

   Compare     Stability                Efficacy       Compare
          Does CoT interfere Perception Tasks?    Does CoT help Reasoning Tasks?                                 Reflection Quality

                                                                                                                                                                                                                                                                        Eval Result

                                                                                                                                                                                                                 Step1                                                                                                    Find Reflection Step            ✓  ...ReflectionWait, I can
                                                                                                                                                                                                             or cosines in triangle                                           CoT Answer                                                                                                     sinesAEB,        use the law of
                                                                                                                                                                                                      ...
                                                                                                         Judge the validity.                                 Please answer step by step.
                                                                                                                                                                                                      Reflection                                                                                                                                                                                                                 Step2                                                                                    ✗                                                                       CoT Prompt                      Reflection                                                                                                                                                                                                        I can look                                                                                                                                                                                                                  at the angles formed
               The Sphinx is...                To solve ...                            QualityPrompt     Tip:                                                     byED, thebut intersectionit is too complicatedof EC and
                 The answer is                   will move                                            Reflection Indicators:                                 ...
                    Desert                     along Path d                                                 Alternatively,Wait,                                                                                                                                                                                                                 Step3                                                                                                                                                             Perhaps, Let          ✗ ReflectionI can use the                                                                                                                                                                                                                    fact that the                                                                                                   me double-check
                   Extract Final Answer                   Extract Final Answer                                       Validity Definition: corrects                           measurementioned... But no tangent is
                                                                                                                                                  the mistake or verifies with new                            ...
                 Desert                       Path d                                                        insights
               ✗                 ✗

Figure 6: Illustration of CoT Robustness Evaluation.    Figure 7: Illustration of CoT Efficiency Evaluation. For
We compare the performance of applying CoT prompt and    relevance rate, we partition the prediction into steps and
direct prompt on two types of tasks: perception and reason-    determine if it is relevant by GPT-4o. For reflection quality,
ing. The stability score measures whether CoT interferes   we prompt GPT-4o to identify the reflection steps by com-
with perception, while the efficacy score assesses the perfor-   mon indicators and judge the validity of the reflection. The
mance gain of CoT on reasoning tasks.                         definitions of relevance and validity are included.


                                                                in the image for answering the question, but it still gener-Efficacy.   Similarly, the performance difference of the two
                                                                ates a detailed description of other objects. This irrelevantprompts on the reasoning tasks R is defined as the score:
                                                            information provides no helpful information to work out the
              Efficacy = AccRCOT −AccRDIR.            (7)    answer. In the meantime, this extra content slows down the
                                                           generation speed. Similar to the calculation of precision,
Intuitively, CoT facilitates stepwise thinking and therefore   we employ the same method to partition the prediction into
benefits answering reasoning tasks. The difference reflects     steps. Then, we instruct GPT-4o to determine all the rel-
how much CoT can enhance reasoning.                      evant steps Prelevant. The step is considered relevant only
                                               when the majority of its content works towards solving the
3.3. CoT Efficiency Evaluation                              question. We first compute the raw relevance rate and then
                                                        apply a scaling factor to amplify the differences between
Models like o1 generate extremely long thinking processes    models. Let rx denote the raw relevance rate:
with reflection and verification of current steps and out-
                                                         CPrelevant        IPrelevantcomes. We perform the first exhaustive analysis of the CoT                                                                    rC =              ,   rI =              ,          (8)
efficiency of visual reasoning with two carefully designed                      |CP|            |IP|
                                                                                                                 |Prelevant|metrics (Figure 7):                                                           r =               .                    (9)
                                                                                    |P|

Relevance Rate.  Although the long reasoning content
                                                        Then, the final relevance rate Relevance Ratex is defined as:allows for deeper thinking, it may also introduce a large
amount of irrelevant information. As shown in the bottom                             rx −α                                                               Relevance Ratex =            ,  x ∈C, I, ∅    (10)
left of Fig. 7, the model has identified the critical element                           1 −α

                                                7
<a id="page-8"></a>

### PDF 第 8 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency

Table 2: Evaluation Results of Three Aspects of CoT in MME-COT. We mark the highest score of each metric in red .
∗denotes unreliable results due to the refusal to answer directly.


                                   CoT Quality                                      CoT Robustness                                CoT Efficiency
 Model              F1                                                     Avg.          CoT      Direct           CoT      Direct   Avg.  Relevance                     Reflection
                                 Precision Image Conclusion Recall Image Conclusion         Stability                         Efficacy                                   Image Conclusion
                       Score                                                     Score           Perception Perception         Reasoning Reasoning Score    Rate                        Quality

                                                                                     Open-source LMMs

 Mulberry               27.4    59.1    74.1     53.8     17.8   26.5     17.1     3.5*    4.4*      42.3       37.9      2.6*      18.6       16.0     89.5     79.0     50.8     95.4      100

 LLaVA-OV-7B          30.9    50.9    47.2     43.5     22.2   24.4     23.2      -3.4     -3.8      46.1       49.8       -3.0      16.4       19.4     91.5     83.0     72.1     93.6      100

 LLaVA-CoT            34.9    53.9    75.6     46.2     25.8   35.8     24.4     0.4*    1.4*      51.5       50.2      -0.6*     24.4       25.0     94.0     88.1     69.2     96.2      100

 LLaVA-OV-72B        36.3    57.3    43.4     50.6     26.6   29.5     27.4      -0.2     0.3       61.1       60.8       -0.6      27.6       28.2     95.4     90.8     83.7     98.3      100

 MiniCPM-V-2.6        39.8    57.3    63.4     45.4     30.5   47.5     26.7      -3.5     -4.8      59.4       64.2       -2.2      26.2       28.3     92.8     85.7     74.6     97.6      100

 InternVL2.5-8B         41.1    60.0    52.4     50.8     31.3   40.4     30.6      -3.0     -6.8      57.3       64.2       0.9      30.3       29.4     98.4     96.8     93.0     98.9      100

 Qwen2-VL-7B          42.1    61.6    61.0     49.3     32.0   46.6     30.5      -4.0     -3.1      60.1       63.1       -4.8      26.0       30.8     94.9     89.8     80.3     98.8      100

 InternVL2.5-8B-MPO   43.0    60.4    60.8     49.9     33.4   44.9     31.8      0.6     0.3       62.5       62.1       0.9      28.8       27.9     94.7     89.3     84.0     96.4      100
 InternVL2.5-78B-MPO  52.7    73.6    68.4     63.0     41.1   53.6     39.1      0.2     -2.0      68.3       70.3       2.4      38.0       35.6     95.3     90.6     82.9     98.2      100

 Qwen2-VL-72B         56.2    77.3    67.2     70.3     44.2   57.1     42.2      -2.1     -6.5      68.9       75.4       2.4      38.6       36.2     96.5     92.9     86.0     98.7      100

 Virgo-72B              60.8    79.5    71.6     72.7     49.2   60.5     47.7     -2.3*   -1.7*      74.1       75.8      -2.9*     41.8       44.7     75.3     90.6     79.8     95.6       60.6

 QVQ-72B              62.0    80.2    73.9     77.5     50.5   60.1     48.9     -1.8*   -3.1*      69.6       72.7      -0.4*     41.0       41.3     67.9     83.7     63.9     95.1       61.7

                                                                                         Closed-source LMMs

 GPT-4o                64.0    85.4    73.3     81.4     51.2   64.3     49.9      2.1     -1.0      71.0       72.0       5.1      40.6       35.5     96.0     92.0     82.4     99.1      100

 Kimi k1.5              64.2    92.0    78.1     89.8     49.3   62.9     47.9     1.4*    2.9*      65.7       62.9      0.0*      40.0       40.0     82.2     92.2     82.2     97.2       72.2


where x = ∅corresponds to the overall relevance rate, and     4.1. Experiment Setup
we take α as 0.8.
                                                     Evaluation Models. We select top-performing LMMs
                                                                for comprehensive CoT evaluation. We test earlier mod-
Reflection Quality.  The superior reasoning ability could     els such as LLaVA-OneVision (7B, 72B) (Li et al., 2024a),
be largely attributed to the reflection and verification pro-   Qwen2-VL (7B, 72B) (Qwen Team, 2024), MiniCPM-V-
cess. However, our analysis reveals that not all reflective    2.6 (Yao et al., 2024b), and InternVL2.5 (8B) (Chen et al.,
steps contribute meaningfully to finding correct answers.    2024b), which are not trained for the reasoning capabil-
We identify distinct failure patterns in the reflection process.     ity. We also include GPT-4o (OpenAI, 2024b) as a strong
Some reflective steps mislead the reasoning by introduc-    baseline model. Besides, we test recent models targeting
ing new errors or incorrect assumptions, while others are    reasoning, including LLaVA-CoT (11B) (Xu et al., 2024),
redundant, simply echoing previous conclusions without    Mulberry (8B) (Yao et al., 2024a), InternVL2.5-MPO (8B,
contributing new insights. To account for failure reflection   78B) (Wang et al., 2024c).  Finally, we evaluate LMMs
scenarios, we propose to measure the validity of the reflec-    with reflection capabilities, including both closed-source
tion. We define a valid reflection as either correctly pointing    models like Kimi k1.5 (Team et al., 2025) and open-source
out the previous mistakes or verifying the previous conclu-    implementations such as QVQ-72B (Team, 2024) and Virgo-
sion with a new insight. Otherwise, the reflection only slows   72B (Du et al., 2025).
down the reasoning. To instruct GPT-4o to determine all the
                                                     Note that we sample 150 questions from MME-COT to eval-valid reflection steps R, we list a set of common indicators
                                                            uate Kimi k1.5, due to the access limitations. The sampleof the start of the reflection, such as “Wait” and “Alterna-
                                                        comprises 115 reasoning and 35 perception questions.tively”, and illustrate the definition of valid reflection. For
all the valid reflection steps Rvalid, the reflection quality is
computed as:


                                         |Rvalid|
              Reflection Quality =            .          (11)
                                 |R|                 Implementation Details.  We define the CoT prompt as:
                                                        Please generate a step-by-step answer, include all your in-
                                                            termediate reasoning process, and provide the final answer
                                                                  at the end. and the direct prompt as: Please directly provide4. Experiments
                                                               the final answer without any other output. We only calculate
In this section, we conduct a systematic evaluation of state-    recall of image observation and logical inference on ques-
of-the-art models on MME-COT. We first detail the experi-    tions where key inference conclusion or image observation
ment setup in Section 4.1. Then in Section 4.2, we report     exists. We employ GPT-4o mini for the direct evaluation
the quantitative results and provide valuable insights derived    and GPT-4o for all other criteria. For hyperparameters, we
from our analysis.                                          follow the settings in VLMEvalKit (Duan et al., 2024).

                                                8
<a id="page-9"></a>

### PDF 第 9 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency

Table 3: Evaluation Results of Three Aspects of CoT in Each Category in MME-COT. Best performance is marked in
 red . ∗denotes unreliable results due to the refusal to answer directly.


                              General Scenes               Space-Time              OCR                Math            Science           Logic
 Model
                        Quality Robustness Efficiency Quality Robustness Efficiency Quality Robustness Efficiency Quality Efficiency Quality Efficiency Quality Efficiency

 Mulberry               33.9      4.3       76.0     18.2      1.0       38.4     26.7      6.6       26.4     29.1     87.9     29.1     91.9     13.9     99.1

 LLaVA-OV-7B          41.8      -6.2       81.8     23.8      -6.7       24.8     44.1      -0.2       42.7     27.4     97.3     28.5     95.1     12.2     98.0
 LLaVA-CoT            38.2      -2.2       89.9     33.6      2.8       68.9     37.4      0.0       77.8     35.3     91.0     36.4     93.4     14.9     97.1

 LLaVA-OV-72B         41.8      -2.3       98.9     29.0      -0.9       43.6     40.8      -1.7       84.2     38.4     98.7     35.4     95.7     18.4     82.3

 MiniCPM-V-2.6         47.1      3.2       87.7     49.3     -14.4      71.1     63.7      -4.9       62.0     32.9     95.2     29.5     90.4     16.9     93.7

 InternVL2.5-8B         43.8      -6.4       87.1     50.7      -8.9       99.1     44.7      -4.1       98.9     40.9     98.0     40.8     97.1     19.5     96.8

 Qwen2-VL-7B          46.7      -3.4       79.3     51.7     -11.8      73.0     65.9      0.9       86.2     34.0     97.9     34.6     95.0     18.4     76.7

 InternVL2.5-8B-MPO    47.2      2.9       94.3     51.8      -0.2       74.6     59.6      -1.0       81.5     37.4     93.4     39.0     95.6     20.9     79.9

 InternVL2.5-78B-MPO  47.9      0.0       89.3     55.5      -2.3       91.9     72.2      2.2       73.1     50.6     95.1     48.5     97.7     24.2     87.2

 Qwen2-VL-72B         51.9      -2.9       88.9     59.7      -5.3       86.7     77.6      2.5       81.7     49.6     97.8     53.6     99.0     40.0     88.0

 Virgo-72B              60.5      0.5       91.0     59.6      -3.8       86.0     79.9      -1.0       82.1     59.6     90.3     55.5     98.7     39.6     88.2

 QVQ-72B               62.6      -1.5       86.9     58.2      -2.5       57.7     76.9      -1.4       52.6     61.4     92.7     57.7     95.9     44.6     94.9
 GPT4o                 62.3      -1.7       96.2     66.3      5.5       64.7     83.3      -1.0       82.1     60.8     98.8     64.1     97.4     27.2     92.0



4.2. Quantitative Results                                   achieves the highest robustness score. Among open-source
                                                       models, only InternVL2.5-MPO, in both its 8B and 78B
We conduct extensive experiments on various LMMs with
                                                                     variants, attains a positive robustness score. Finally, for CoT
our proposed CoT evaluation suite. The main results are
                                                                     efficiency, InternVL2.5-8B obtains the maximum relevance
presented in Table 2 and Table 3. We begin by analyzing
                                                             of 98.4%, suggesting its consistent focus on questions.
the overall performance and then highlight key findings.
                                                Now, we summarize our key observations as follows:

Overall Results.  In Table 2, we present the overall perfor-                                                  Models with reflection largely benefit CoT quality.  As
mance of three CoT evaluation perspectives with specific                                                shown in Table 2, the F1 scores of the two models with
metrics. To provide a comprehensive understanding, we                                                                  reflection capability most closely approach GPT-4o. After
report precision, recall, and relevance for both logical infer-                                                                  specifically fine-tuning for the reasoning capabilities from
ence and image caption steps. For robustness, we provide                                               Qwen2-VL-72B, QVQ surpasses its base model by 5.8%.
the direct evaluation result on the perception and reasoning                                                          Notably, although QVQ generates longer CoT sequences
tasks, with either CoT or direct prompt. We employ the                                                            than Qwen2-VL-72B, QVQ’s precision still exceeds Qwen2-
average value of the stability and efficacy as the final robust-                                             VL-72B by 2.9%, indicating superior accuracy in each rea-
ness metric. Notably, we define the reflection quality as 100                                                         soning step. Kimi k1.5 also surpasses the previous state-of-
on models incapable of reflection.                                                                    the-art model GPT-4o, obtaining the highest CoT quality.
For CoT quality, Kimi k1.5 achieves the highest F1 score.
Open-source models with larger sizes consistently demon-   Long CoT does not necessarily cover key steps.  Despite
strate better performance, highlighting the scalability of    high precision in long CoT models, the informativeness of
LMMs.  Notably, Qwen2-VL-72B outperforms all other    each step is not guaranteed. We observe that the recall trend
open-source models without reflection, even surpassing   among GPT-4o, QVQ, and Virgo does not align with their
InternVL2.5-78B-MPO, which is specifically enhanced for   CoT Rea. performance (i.e., their final answer accuracy on
reasoning.  Analysis reveals that GPT-4o achieves supe-    the reasoning tasks under the CoT prompt). Specifically,
rior performance across all recall metrics, while Kimi k1.5    while both Virgo and QVQ outperform GPT-4o in direct
demonstrates the highest scores in precision evaluations.    evaluation, they lag behind in recall. This suggests that
For CoT robustness, Mulberry obtains the highest average    long CoT models sometimes reach correct answers while
score. However, when we look into its output, we find it    skipping intermediate steps, which contradicts the principle
still generates lengthy rationales despite receiving a direct    of stepwise reasoning and warrants further investigation.
prompt. Even worse, the direct prompt seems to be an
out-of-distribution input for Mulberry, frequently leading   CoT impairs perception task performance in most mod-
to nonsensical outputs. Further analysis of other models’     els.  Surprisingly, most models exhibit negative stability
predictions reveals that LLaVA-CoT, Virgo, QVQ, and Kimi     scores, indicating that CoT interferes with perception tasks.
k1.5 similarly neglect the direct prompt, instead generating   The most significant degradation occurs in InternVL2.5-
extended rationales before answering. Consequently, their    8B, where performance drops by 6.8%. This reveals in-
robustness scores may be misleading. Once again, GPT-4o    consistency and potential overthinking in current models,

                                                9
<a id="page-10"></a>

### PDF 第 10 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency

presenting a significant barrier to adopting CoT as the de-
fault answering strategy. Among models that provide direct                                                                           Interference
answers, only LLaVA-OV-72B and InternVL2.5-8B-MPO                                                                            Repetition1.8%
achieve a modest positive score of 0.3%.                                                                                            4.9%

                                                                               Ineffective
More parameters enable models to grasp reasoning bet-                                              Incompleteness                                                                         Reflection
ter.  We find that models with larger parameter counts tend                                                  17.3%
                                                                  76.0%
to achieve higher efficacy scores. This pattern is evident
across LLaVA-OV, InternVL2.5-MPO, and Qwen2-VL. For
instance, while Qwen2-VL-7B shows a 4.8% decrease in
performance when applying CoT to reasoning tasks, its
larger counterpart, Qwen2-VL-72B, demonstrates a 2.4%
improvement. This discrepancy suggests that models with
                                                         Figure 8: Distribution of Reflection Error Types. We
more parameters could better grasp the reasoning ability
                                                                identify four types of error: ineffective reflection, incom-
under the same training paradigm.
                                                                 pleteness, repetition, and interference.

Long CoT models may be more susceptible to distraction.
                                                            ductive reflection. These patterns are illustrated in Fig. 10Long CoT models may demonstrate lower relevance scores
                                                     and their distribution is shown in Fig. 8.compared to other models. They frequently generate content
unrelated to solving the given question, corresponding to   The four major error types are:
their relatively low recall scores compared to direct evalu-
ation, like QVQ. Although a few models with short CoT,
                                                                            • Ineffective Reflection. The model arrives at an incor-
like Mulberry and LLaVA-OV-7B, also obtain a low rel-
                                                                          rect conclusion and, upon reflecting, continues to make
evance rate, we find that it is because these models may
                                                                     incorrect adjustments. This is the most common error
keep repeating words when dealing with specific type of
                                                               type and is also witnessed most frequently.
questions, resulting in irrelevant judgment. The fine-grained
metric reveals that models tend to lose focus when describ-        • Incompleteness. The model proposes new analytical
ing images, often producing exhaustive captions regardless        approaches but does not execute them, only stopping
of their relevance to the question. From Table 3, we find          at the initial thought. The reflection slows down the
that this phenomenon prevails in general scenes, space-time,         inference process without bringing any gain.
and OCR tasks. This behavior can significantly slow infer-
ence by generating substantial irrelevant content. Teaching        • Repetition. The model restates previous content or
long CoT models to focus on question-critical elements        methods without introducing new insights, leading to
represents a promising direction for future research.                  inefficient reasoning.

                                                                            • Interference. The model initially reaches a correct
Reflection often fails to help.  While reflection is a key                                                               conclusion but, through reflection, introduces errors.
feature of long CoT models for answer verification, both
QVQ and Virgo achieve reflection quality scores of only
                                                       Understanding and mitigating these errors is crucial forabout 60%, indicating that approximately 40% of reflection
                                                      improving the reliability of LMM reflection mechanisms.attempts fail to contribute meaningfully to answer accuracy.
                                                The analysis provides the opportunity to focus on solvingEven for the closed-source model Kimi k1.5, over 25% re-
                                                                   specific error types to enhance the overall reflection quality.flection steps are also invalid. This substantial failure rate
compromises efficiency by potentially introducing unneces-
sary or distracting steps before reaching correct solutions.    5. Conclusion
Future research should explore methods to reduce these in-
                                                             In this paper, we have introduced MME-CoT, a comprehen-effective reflections to improve both efficiency and quality.
                                                                 sive benchmark designed to evaluate Chain-of-Thought rea-
                                                          soning in Large Multimodal Models. Our dataset comprises
4.3. Error Analysis
                                                                six categories to cover most scenarios of visual reasoning
In this section, we analyze error patterns in the LMM reflec-    tasks. To gain a thorough understanding of the reasoning
tion process. An effective reflection should either correct    process, we design a novel CoT evaluation suite with three
previous mistakes or validate correct conclusions through    metrics. Our systematic evaluation obtains useful insights
new insights. We examined 200 model predictions from     into the issues within the current state-of-the-art Large Mul-
QVQ and identified four distinct error types that hinder pro-    timodal Models. We identify critical flaws in all the tested

                                                10
<a id="page-11"></a>

### PDF 第 11 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency

open-source models. As the field continues to evolve, MME-   Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R.,
CoT stands as a valuable tool for measuring progress and      Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: In-
identifying areas for improvement in the development of       centivizing reasoning capability in llms via reinforcement
more sophisticated multimodal AI systems.                       learning. arXiv preprint arXiv:2501.12948, 2025a.

                                                  Guo, Z., Zhang, R., Zhu, X., Tang, Y., Ma, X., Han, J.,
Impact Statement                                    Chen, K., Gao, P., Li, X., Li, H., et al.  Point-bind &
                                                                 point-llm: Aligning point cloud with multi-modality for
This paper presents work whose goal is to advance the field
                                                    3d understanding, generation, and instruction following.
of Computer Vision and Machine Learning. There are many
                                                          arXiv preprint arXiv:2309.00615, 2023.
potential societal consequences of our work, none of which
we feel must be specifically highlighted here.              Guo, Z., Zhang, R., Chen, H., Gao, J., Gao, P., Li, H.,
                                                        and Heng, P.-A. Sciverse. https://sciverse-cuhk.github.io,
References                                             2024a. URL https://sciverse-cuhk.github.
                                                   io/.
Chen, G., Zheng, Y.-D., Wang, J., Xu, J., Huang, Y., Pan,
                                                  Guo, Z., Zhang, R., Zhu, X., Tong, C., Gao, P., Li, C.,   J., Wang, Y., Wang, Y., Qiao, Y., Lu, T., et al. Videollm:
                                                      and Heng, P.-A. Sam2point: Segment any 3d as videos  Modeling video sequence with large language models.
                                                                  in zero-shot and promptable manners.  arXiv preprint  arXiv preprint arXiv:2305.13292, 2023.
                                                           arXiv:2408.16768, 2024b.
Chen, Q., Qin, L., Zhang, J., Chen, Z., Xu, X., and Che, W.
                                                  Guo, Z., Zhang, R., Tong, C., Zhao, Z., Gao, P., Li, H.,
  M3cot: A novel benchmark for multi-domain multi-step
                                                      and Heng, P.-A. Can we generate images with cot? let’s
  multi-modal chain-of-thought. In Proc. of ACL, 2024a.
                                                                    verify and reinforce image generation step by step. arXiv
                                                                   preprint arXiv:2501.13926, 2025b.Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z., Cui, E.,
  Zhu, J., Ye, S., Tian, H., Liu, Z., et al. Expanding per-   Hao, S., Gu, Y., Luo, H., Liu, T., Shao, X., Wang, X.,
  formance boundaries of open-source multimodal models      Xie, S., Ma, H., Samavedhi, A., Gao, Q., et al. Llm
  with model, data, and test-time scaling. arXiv preprint       reasoners: New evaluation, library, and analysis of step-
  arXiv:2412.05271, 2024b.                                  by-step reasoning with large language models.  arXiv
                                                                   preprint arXiv:2404.05221, 2024.
Chen, Z., Wang, W., Tian, H., Ye, S., Gao, Z., Cui, E.,
  Tong, W., Hu, K., Luo, J., Ma, Z., et al. How far are    He, C., Luo, R., Bai, Y., Hu, S., Thai, Z. L., Shen, J., Hu, J.,
  we to gpt-4v?  closing the gap to commercial multi-      Han, X., Huang, Y., Zhang, Y., Liu, J., Qi, L., Liu, Z., and
  modal models with open-source suites. arXiv preprint      Sun, M. Olympiadbench: A challenging benchmark for
  arXiv:2404.16821, 2024c.                                promoting agi with olympiad-level bilingual multimodal
                                                                         scientific problems, 2024.
Du, Y., Liu, Z., Li, Y., Zhao, W. X., Huo, Y., Wang, B.,
                                                                         Jia, Y., Liu, J., Chen, S., Gu, C., Wang, Z., Luo, L., Lee,  Chen, W., Liu, Z., Wang, Z., and Wen, J.-R.  Virgo:
                                                                      L., Wang, P., Wang, Z., Zhang, R., et al. Lift3d foun- A preliminary exploration on reproducing o1-like mllm.
                                                              dation policy:  Lifting 2d large-scale pretrained mod-  arXiv preprint arXiv:2501.01904, 2025.
                                                                      els for robust 3d robotic manipulation. arXiv preprint
Duan, H., Yang,  J., Qiao, Y., Fang, X., Chen, L., Liu,      arXiv:2411.18623, 2024.
   Y., Dong, X., Zang, Y., Zhang,  P., Wang,  J., et al.
                                                               Jiang, D., Zhang, R., Guo, Z., Wu, Y., Lei, J., Qiu, P.,
  Vlmevalkit: An open-source toolkit for evaluating large
                                                         Lu, P., Chen, Z., Song, G., Gao, P., et al. Mmsearch:
  multi-modality models. In Proceedings of the 32nd ACM
                                                     Benchmarking the potential of large models as multi-
   International Conference on Multimedia, pp. 11198–
                                                     modal search engines. arXiv preprint arXiv:2409.12959,
  11201, 2024.
                                                           2024.
Gao, P., Zhang, R., Liu, C., Qiu, L., Huang, S., Lin, W.,    Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang,
  Zhao, S., Geng, S., Lin, Z., Jin, P., et al.  Sphinx-x:      H., Zhang, K., Li, Y., Liu, Z., and Li, C.   Llava-
  Scaling data and parameters for a family of multi-modal       onevision: Easy visual task transfer.  arXiv preprint
   large language models. ICML 2024, 2024.                   arXiv:2408.03326, 2024a.

Golovneva, O., Chen, M., Poff, S., Corredor, M., Zettle-    Li, F., Zhang, R., Zhang, H., Zhang, Y., Li, B., Li, W.,
  moyer,  L., Fazel-Zarandi, M., and Celikyilmaz, A.     Ma, Z., and Li, C. Llava-next-interleave: Tackling multi-
  Roscoe: A suite of metrics for scoring step-by-step rea-      image, video, and 3d in large multimodal models. arXiv
  soning. arXiv preprint arXiv:2212.07919, 2022.               preprint arXiv:2407.07895, 2024b.

                                                11
<a id="page-12"></a>

### PDF 第 12 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency

Li,  J.,  Li, D., Xiong, C., and Hoi, S.   Blip:  Boot-    Sprague, Z., Yin, F., Rodriguez, J. D., Jiang, D., Wadhwa,
  strapping language-image pre-training for unified vision-      M., Singhal, P., Zhao, X., Ye, X., Mahowald, K., and
  language understanding and generation. In International       Durrett, G. To cot or not to cot? chain-of-thought helps
  Conference on Machine Learning, pp. 12888–12900.      mainly on math and symbolic reasoning. arXiv preprint
  PMLR, 2022.                                             arXiv:2409.12183, 2024.

Li, K., He, Y., Wang, Y., Li, Y., Wang, W., Luo, P., Wang,   Team, K., Du, A., Gao, B., Xing, B., Jiang, C., Chen, C.,
   Y., Wang, L., and Qiao, Y. Videochat: Chat-centric video       Li, C., Xiao, C., Du, C., Liao, C., et al. Kimi k1. 5:
  understanding. arXiv preprint arXiv:2305.06355, 2023.       Scaling reinforcement learning with llms. arXiv preprint
                                                           arXiv:2501.12599, 2025.
Lin, Z., Liu, C., Zhang, R., Gao, P., Qiu, L., Xiao, H., Qiu,
                                                  Team, Q. Qvq: To see the world with wisdom, December  H., Lin, C., Shao, W., Chen, K., et al.  Sphinx: The
                                                            2024. URL https://qwenlm.github.io/blog/   joint mixing of weights, tasks, and visual embeddings for
                                             qvq-72b-preview/.  multi-modal large language models. ECCV 2024, 2023.

                                                        Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux,
Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction
                                                         M.-A., Lacroix, T., Rozi`ere, B., Goyal, N., Hambro, E.,
  tuning. In NeurIPS, 2023.
                                                             Azhar, F., et al. Llama: Open and efficient foundation lan-
Lu, P., Bansal, H., Xia, T., Liu, J., yue Li, C., Hajishirzi,      guage models. arXiv preprint arXiv:2302.13971, 2023.
  H., Cheng, H., Chang, K.-W., Galley, M., and Gao, J.
                                                 Wang, F., Fu, X., Huang, J. Y., Li, Z., Liu, Q., Liu, X., Ma,
  Mathvista: Evaluating math reasoning in visual contexts
                                               M. D., Xu, N., Zhou, W., Zhang, K., et al. Muirbench: A
  with gpt-4v, bard, and other large multimodal models.
                                                         comprehensive benchmark for robust multi-image under-
  ArXiv, abs/2310.02255, 2023.
                                                                 standing. arXiv preprint arXiv:2406.09411, 2024a.

OpenAI.       GPT-4V(ision)  system   card,   2023.                                                 Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen,
 URL      https://openai.com/research/                                                                  K., Liu, X., Wang, J., Ge, W., et al. Qwen2-vl: Enhancing
  gpt-4v-system-card.                                                            vision-language model’s perception of the world at any
                                                                    resolution. arXiv preprint arXiv:2409.12191, 2024b.
OpenAI.   Gpt-4o mini:  advancing  cost-efficient  in-
   telligence.      https://openai.com/index/   Wang, W., Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z.,
  gpt-4o-mini-advancing-cost-efficient-intelligence/,Zhu, J., Zhu, X., Lu, L., Qiao, Y., and Dai, J. Enhancing
  2024.                                                        the reasoning ability of multimodal large language mod-
                                                                      els via mixed preference optimization.  arXiv preprint
OpenAI.  Introducing openai o1, 2024., 2024a. URL                                                           arXiv:2411.10442, 2024c.
  https://openai.com/o1/.
                                                Wang, W., Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z.,
OpenAI.   Hello gpt-4o.  https://openai.com/      Zhu, J., Zhu, X., Lu, L., Qiao, Y., et al.  Enhancing
  index/hello-gpt-4o/, 2024b.                          the reasoning ability of multimodal large language mod-
                                                                      els via mixed preference optimization.  arXiv preprint
Peng, T., Li, M., Zhou, H., Xia, R., Zhang, R., Bai, L.,
                                                           arXiv:2411.10442, 2024d.
  Mao, S., Wang, B., He, C., Zhou, A., et al. Chimera:
  Improving generalist model with domain-specific experts.   Wang, Z., Xia, M., He, L., Chen, H., Liu, Y., Zhu, R., Liang,
  arXiv preprint arXiv:2412.05983, 2024.                      K., Wu, X., Liu, H., Malladi, S., Chevalier, A., Arora,
                                                                         S., and Chen, D.  Charxiv: Charting gaps in realistic
Prasad, A., Saha, S., Zhou, X., and Bansal, M. Receval:                                                                chart understanding in multimodal llms. arXiv preprint
  Evaluating reasoning chains via correctness and informa-                                                           arXiv:2406.18521, 2024e.
   tiveness. arXiv preprint arXiv:2304.10703, 2023.
                                                        Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi,
Qwen Team. Qwen2-vl. 2024.                                       E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting
                                                                            elicits reasoning in large language models. Advances in
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh,                                                               neural information processing systems, 35:24824–24837,
  G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P.,                                                          2022.
  Clark, J., Krueger, G., and Sutskever, I. Learning trans-
  ferable visual models from natural language supervi-   Xu, G., Jin, P., Li, H., Song, Y., Sun, L., and Yuan, L.
   sion. In International Conference on Machine Learning,      Llava-cot: Let vision language models reason step-by-
  2021.  URL https://api.semanticscholar.       step, 2024. URL https://arxiv.org/abs/2411.
  org/CorpusID:231591445.                      10440.

                                                12
<a id="page-13"></a>

### PDF 第 13 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency

Xu, R., Wang, X., Wang, T., Chen, Y., Pang, J., and Lin,       fine-tuning of large language models with zero-initialized
  D. Pointllm: Empowering large language models to un-       attention.  In The Twelfth International Conference on
  derstand point clouds. arXiv preprint arXiv:2308.16911,      Learning Representations, 2024b.  URL https://
  2023.                                       openreview.net/forum?id=d4UiXAHN2W.

Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C.,    Zhang, R., Jiang, D., Zhang, Y., Lin, H., Guo, Z., Qiu, P.,
   Li, C., Li, C., Liu, D., Huang, F., Dong, G., Wei, H., Lin,      Zhou, A., Lu, P., Chang, K.-W., Gao, P., et al. Mathverse:
  H., Tang, J., Wang, J., Yang, J., Tu, J., Zhang, J., Ma, J.,     Does your multi-modal llm truly see the diagrams in
  Xu, J., Zhou, J., Bai, J., He, J., Lin, J., Dang, K., Lu, K.,       visual math problems? ECCV 2024, 2024c.
  Chen, K., Yang, K., Li, M., Xue, M., Ni, N., Zhang, P.,
                                                     Zhang, R., Wei, X., Jiang, D., Zhang, Y., Guo, Z., Tong,  Wang, P., Peng, R., Men, R., Gao, R., Lin, R., Wang, S.,
                                                                  C., Liu, J., Zhou, A., Wei, B., Zhang, S., et al. Mavis:  Bai, S., Tan, S., Zhu, T., Li, T., Liu, T., Ge, W., Deng,
                                                        Mathematical visual instruction tuning. arXiv preprint  X., Zhou, X., Ren, X., Zhang, X., Wei, X., Ren, X., Fan,
                                                           arXiv:2407.08739, 2024d.   Y., Yao, Y., Zhang, Y., Wan, Y., Chu, Y., Liu, Y., Cui, Z.,
  Zhang, Z., and Fan, Z. Qwen2 technical report. arXiv                                                     Zhang, Y., Bai, H., Zhang, R., Gu, J., Zhai, S., Susskind,
   preprint arXiv:2407.10671, 2024.                                                                                       J., and Jaitly, N. How far are we from intelligent visual
                                                             deductive reasoning? In COLM, 2024e.Yao, H., Huang, J., Wu, W., Zhang, J., Wang, Y., Liu, S.,
  Wang, Y., Song, Y., Feng, H., Shen, L., et al. Mulberry:                                                     Zhu, D., Chen, J., Shen, X., Li, X., and Elhoseiny, M.
  Empowering mllm with o1-like reasoning and reflection      Minigpt-4: Enhancing vision-language understanding
  via collective monte carlo tree search.  arXiv preprint                                                          with advanced large language models.  arXiv preprint
  arXiv:2412.18319, 2024a.                                                           arXiv:2304.10592, 2023.

Yao, Y., Yu, T., Zhang, A., Wang, C., Cui, J., Zhu, H., Cai, T.,
   Li, H., Zhao, W., He, Z., et al. Minicpm-v: A gpt-4v level
  mllm on your phone. arXiv preprint arXiv:2408.01800,
  2024b.

Ying, K., Meng, F., Wang, J., Li, Z., Lin, H., Yang, Y.,
  Zhang, H., Zhang, W., Lin, Y., Liu, S., et al. Mmt-
  bench: A comprehensive multimodal benchmark for eval-
  uating large vision-language models towards multitask
   agi. arXiv preprint arXiv:2404.16006, 2024.

Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang,
  X., and Wang, L.  Mm-vet:  Evaluating large multi-
  modal models for integrated capabilities. arXiv preprint
  arXiv:2308.02490, 2023.

Yue, X., Zheng, T., Ni, Y., Wang, Y., Zhang, K., Tong,
   S., Sun, Y., Yu, B., Zhang, G., Sun, H., Su, Y., Chen,
  W., and Neubig, G. Mmmu-pro: A more robust multi-
   discipline multimodal understanding benchmark, 2024.
  URL https://arxiv.org/abs/2409.02813.

Zhang, H., Li, H., Li, F., Ren, T., Zou, X., Liu, S., Huang,
   S., Gao, J., Zhang, L., Li, C., et al. Llava-grounding:
  Grounded visual chat with large multimodal models.
  arXiv preprint arXiv:2312.02949, 2023.

Zhang, R., Han, J., Liu, C., Zhou, A., Lu, P., Qiao, Y., Li,
  H., and Gao, P. Llama-adapter: Efficient fine-tuning of
   large language models with zero-initialized attention. In
  ICLR 2024, 2024a.

Zhang, R., Han, J., Zhou, A., Hu, X., Yan, S., Lu, P., Li,
  H., Gao, P., and Qiao, Y.  LLaMA-adapter: Efficient

                                                13
<a id="page-14"></a>

### PDF 第 14 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency

Appendix Overview

   • Section A: Related Work.

   • Section B: More Dataset Details.

   • Section C: Error Analysis.

   • Section D: More Qualitative Examples.

   • Section E: Evaluation Prompts.


A. Related Work

A.1. Large Multimodal Models

The field of multimodal (Radford et al., 2021; Li et al., 2022; OpenAI, 2023; 2024b) AI has experienced extraordinary
growth, particularly through the development of Large Multimodal Models (LMMs) (Liu et al., 2023; Zhu et al., 2023; Lin
et al., 2023; Qwen Team, 2024). These models build upon the achievements of Large Language Models (LLMs) (Touvron
et al., 2023; Yang et al., 2024) and advanced vision models (Radford et al., 2021), expanding their capabilities to process
multiple kinds of visual input (Li et al., 2024b; Guo et al., 2023; Li et al., 2023).

Closed-source models, such as OpenAI’s GPT-4o (OpenAI, 2024b), have demonstrated exceptional capabilities in visual
understanding and reasoning. However, their closed-source nature creates barriers to widespread adoption and further
development by the broader research community. In response, significant progress has been made in developing open-source
alternatives. Early approaches like LLaVA (Liu et al., 2023), LLaMA-Adapter (Zhang et al., 2024b), and MiniGPT-4 (Zhu
et al., 2023) established a foundation by combining frozen CLIP models for image encoding with LLMs, enabling multimodal
instruction tuning. Subsequent developments through projects such as InternVL2 (Chen et al., 2024c), Qwen2-VL (Qwen
Team, 2024), SPHINX (Gao et al., 2024; Lin et al., 2023), and MiniCPM-V (Yao et al., 2024b) have expanded these
capabilities by incorporating more diverse visual instruction datasets and broadening application scenarios.

Recently, with the introduction of o1 (OpenAI, 2024a), the field of LMMs has also focused on enhancing the reasoning
capability. (Wang et al., 2024d) introduces mixed preference optimization with automatically constructed data. (Yao et al.,
2024a) proposes to leverage collective knowledge from multiple models to identify effective reasoning paths. Besides,
several works (Team, 2024; Du et al., 2025) have demonstrated the ability to replicate behaviors similar to o1 models,
particularly regarding multi-step CoT reasoning with iterative self-reflection and verification processes.


A.2. Reasoning Evaluation

Several methods have been developed to evaluate reasoning in natural language processing, including ROSCOE (Golovneva
et al., 2022) and ReCEval (Prasad et al., 2023), which assess reasoning chains across multiple dimensions such as correctness
and informativeness. However, these approaches are limited to text-only scenarios and do not address the unique challenges
present in visual reasoning tasks. Furthermore, the emergence of long chain-of-thought (CoT) reasoning has introduced
additional considerations, such as output efficiency and reflection quality, which existing evaluation methods do not
adequately address.

On the other hand, various multimodal benchmarks have been developed to assess reasoning abilities across specific
domains. Current exploration of visual reasoning predominantly focuses on the mathematics (Zhang et al., 2024d; Peng
et al., 2024) domains. MathVista (Lu et al., 2023) provides a comprehensive collection of mathematical problems that assess
mathematical and logical reasoning abilities. Building on this, MathVerse (Zhang et al., 2024c) introduces a new benchmark
by eliminating redundant textual information to evaluate whether LMMs can accurately interpret graphical representations.
OlympiadBench (He et al., 2024) further raises the complexity bar by incorporating challenging Olympiad-level mathematics
and physics problems. Despite these advances in specialized domains, broader applications such as general-scene reasoning
remain relatively unexplored. Recent developments have begun to expand beyond purely scientific reasoning. For instance,
M³CoT (Chen et al., 2024a) and SciVerse (Guo et al., 2024a) incorporate commonsense tasks alongside scientific reasoning
and knowledge-based assessment in the multimodal benchmark. However, most existing benchmarks focus solely on
evaluating final answers while overlooking the intermediate steps, thus providing limited insights into the process through
which models arrive at their conclusions.

                                                14
<a id="page-15"></a>

### PDF 第 15 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency

B. More Dataset Details

B.1. Data Source Distribution

We visualize the data source distributions in our benchmark, which consists of 15 sets, including MathVerse (Zhang et al.,
2024c), MMMUPro (Yue et al., 2024), OlympiadBench (He et al., 2024), MMT-Bench (Ying et al., 2024), MuirBench (Wang
et al., 2024a), ml-rpm-bench (Zhang et al., 2024e), MMSearch (Jiang et al., 2024), CharXiv (Wang et al., 2024e), and
SciVerse (Guo et al., 2024a).

                                                           6.9%                     1.9%                                                                                                  olympiad      mathverse10.1%                                                             sciverse                                      6.7%                                                                                                 mmmupro                                       mmmupro                                          5.0%        Science                                                               Math    olympiad                                                           charxiv          18.7%              10.2%                                                     2.8%                 22.1%                                                                                  mmsearch 0.9%
                                                       mmt
                                 OCR                 Space-
                                                                              Time     6.5%                               muir     18.6%               14.8%
                                                                           General                                                                                                               Logic                              12.7%                                                                             Scenes                                                                                                6.6%      muir
                                                                    19.2%                                                   8.3%                                                2.2%
                                   mmt                                                                     mlrpm                                         muir                  6.6%                                         5.3%                                               mmt                                                                         13.9%


                                Figure 9: Data Source Distribution of MME-CoT.





                                                15
<a id="page-16"></a>

### PDF 第 16 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency

B.2. Preliminary Categorization Result

Table 4: Accuracy of MMT-Bench for different subcategories. ACT: Action Understanding; AUT: Attribute Similarity;
CNT: Cartoon Understanding; CIM: Counting; DOC: Diagram Understanding; EMO: Difference Spotting; HAL: Geographic
Understanding; IIT: Image-Text Matching; IRT: Ordering; IQT: Scene Understanding; MEM: Visual Grounding; MIA:
Visual Retrieval; OCR: Object Recognition; PLP: Physical Layout Prediction; RRE: Relationship Extraction; TMP:
Temporal Reasoning; VCP: Visual Comprehension; VCR: Visual Coherence Reasoning; VGR: Visual Generation; VIL:
Visual Identification; VPU: Visual Prediction Understanding; VRE: Visual Reasoning Evaluation.

 File Name                ACT   AUT   CNT   CIM   DOC   EMO   HAL   IIT   IRT   IQT   MEM   MIA   OCR   PLP   RRE   TMP   VCP   VCR   VGR   VIL   VPU   VRE

 GPT4o-cot            0.60 0.60 0.44 0.67 0.79 0.30 0.71 0.50 0.63 0.10 0.85 0.60 0.77 0.36 0.76 0.48 0.86 0.80 0.49 0.48 0.82 0.85
 GPT4-direct           0.53 0.60 0.44 0.67 0.81 0.23 0.69 0.33 0.66 0.25 0.80 0.43 0.78 0.42 0.78 0.36 0.89 0.85 0.41 0.37 0.85 0.85
 Qwen2-VL-7B-cot     0.53 0.61 0.34 0.65 0.77 0.53 0.74 0.40 0.31 0.20 0.78 0.58 0.60 0.43 0.69 0.43 0.85 0.90 0.54 0.35 0.79 0.81
 Qwen2-VL-7B-direct  0.49 0.67 0.40 0.78 0.75 0.52 0.73 0.43 0.31 0.10 0.78 0.55 0.60 0.54 0.69 0.40 0.85 0.85 0.67 0.38 0.85 0.82


Table 5: Accuracy of MUIRBench for different subcategories. AU: Action Understanding; AS: Attribute Similarity;
CU: Cartoon Understanding; CO: Counting; DU: Diagram Understanding; DS: Difference Spotting; GU: Geographic
Understanding; ITM: Image-Text Matching; OR: Ordering; SU: Scene Understanding; VG: Visual Grounding; VR: Visual
Retrieval.


 File Name        AU    AS    CU   CO   DU    DS   GU    ITM   OR    SU   VG   VR

 GPT4o-cot            0.48     0.57     0.55     0.75     0.82     0.64     0.59     0.82     0.38     0.88     0.56     0.70
 GPT4o-direct         0.45     0.62     0.59     0.50     0.88     0.62     0.55     0.86     0.33     0.74     0.38     0.77
 Qwen2-VL-7B-cot    0.38     0.51     0.42     0.43     0.43     0.27     0.21     0.55     0.13     0.69     0.37     0.28
 Qwen2-VL-7B-direct  0.39     0.47     0.44     0.41     0.40     0.33     0.25     0.51     0.13     0.67     0.31     0.20


              Table 6: Accuracy of OlympiadBench for the mathematics and physics subcategories.


                                             File Name            Mathematics   Physics

                                   GPT4o-cot                 0.25        0.04
                                      GPT4o-direct              0.07        0.03
                                  Qwen2-VL-7B-cot         0.05        0.01
                                    Qwen2-VL-7B-direct       0.07        0.01





                                                16
<a id="page-17"></a>

### PDF 第 17 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency

C. Error Analysis

We showcase the examples of the identified error types of reflection in Fig. 10.


                             Ineffective Reflection                                                           Repetition

                                                                                             Question                                                                                                            Question




       Question: The blue car is moving forward. What is the correct order of images according to the                    Question: Conversation: 2021/05/31 People1: This left picture was taken on New Year‘s Day.
       given context?                                                                       And the right picture was taken today. I’ve gained a lot of weight and I‘m so annoyed. How
      A. the second image, the first image, the third image                                               many kilograms does people1 gain per month on average?
       B. None of the choices provided                                                                      A. 21.4   B. 10.7   C. None of the choices provided   D. 1.07             Answer: D
      C. the third image, the first image, the second image
      D. the third image, the second image, the first image.                      Answer: A                                                                                          Model Response
                                                                                                                                Reflection
                                                                            Model Response                                ...
                                                                                                                             So,                                                                                                                                average                                                                                                                                        monthly                                                                                                                                                 gain                                                                                                                                                      =                                                                                                                                                       total                                                                                                                                                             weight                                                                                                                                                                     gain                                                                                                                                                                          /                                                                                                                                                                           number                                                                                                                                                                                   of                                                                                                                                                                                     months.
                                                                                                                                                                                                                                              Reasoning                                                                                                                                                                                                                                                                           First                                                                                                                                                                                                                               Time                Reflection                                                                                 Wrong                                                                                              Answer
                                                                                                                             So,                                                                                                                                total                                                                                                                                       days                                                                                                                                            = 31                                                                                                                                                (January)                                                                                                                                                           +                                                                                                                                                             28                                                                                                                                                               (February)                                                                                                                                                                           +                                                                                                                                                                             31 (March)                                                                                                                                                                                        + 30                                                                                                                                                                                            (April)                                                                                                                                                                                                     +                                                                                                                                                                                                       31                                                                                                                                                                                                          (May)
            ...                                                                                                                             =                                                                                                                               151                                                                                                                                  days.
            Alternatively,                           perhaps                                   the street                                              is not                                                     horizontal                                                                or                                                                   vertical,                                                                             but                                                                                 diagonal.                                                                                                                             Now,                                                                                                                                  to                                                                                                                                     find                                                                                                                                          the                                                                                                                                             average                                                                                                                                                     monthly                                                                                                                                                             weight                                                                                                                                                                     gain,                                                                                                                                                                          perhaps                                                                                                                                                                                   I need                                                                                                                                                                                          to                                                                                                                                                                                            consider                                                                                                                                                                                                      the
            Looking                    back at                            the images,                                        the street                                                   appears                                                           to be                                                                 horizontal                                                                            in                                                                               the                                                                                   aerial                                                                                          view,                                                                                                                            average                                                                                                                                     per                                                                                                                                        month                                                                                                                                              based                                                                                                                                                     on                                                                                                                                                        the                                                                                                                                                           number                                                                                                                                                                   of                                                                                                                                                                      days.
            with the yellow car moving from left to right. Therefore, the sequence should be
            first                  image                        (left),                                second image                                             (middle),                                                       third                                                             image                                                                   (right).                                                                            Since none                                                                                       of the                               But                                                                                                                                 that                                                                                                                                     seems                                                                                                                                           unnecessary.                                                                                                                                                        Since                                                                                                                                                               the                                                                                                                                                                  weight                                                                                                                                                                          was                                                                                                                                                                             measured                                                                                                                                                                                       at the beginning and
            options                    match                          this sequence,                                         the                                             answer must                                                         be B.                                                               none                                                                    of the choices                                                                                   provided.                                                                                                                             the                                                                                                                                 end,                                                                                                                                     using                                                                                                                                           months as the                                                                                                                                                          time                                                                                                                                                               unit                                                                                                                                                                   should                                                                                                                                                                           be                                                                                                                                                                             sufficient.
            Alternatively, perhaps the question expects a different perspective. Wait, maybe the                            Alternatively, maybe the problem wants the average daily weight gain, and then
            street                   is                      not                          horizontal                                     but                                         vertical.                                                   If the                                                          street                                                                 is                                                                    vertical,                                                                              with                                                                                   the                                                                                       yellow                                                                                              car                           convert                                                                                                                                     it                                                                                                                                        to                                                                                                                                          monthly                                                                                                                                                  gain.                                                                                                                                                                                                                           Repeated Reasoning            moving                   from                        top                            to                               bottom,                                       then                                            the sequence                                                         would                                                               be                                                                  first                                                                        image                                                                              (top),                                                                                     third                                                                                           image                                                                                                                             6.4                                                                                                                                 kg /                                                                                                                                      151                                                                                                                                          days                                                                                                                                               =                                                                                                                                                approximately                                                                                                                                                              0.0424 kg per                                                                                                                                                                             day.
            (middle),                      second                             image                                   (bottom).                                                                                                                            0.0424                                                                                                                                   kg/day                                                                                                                                           * 30                                                                                                                                               days/month                                                                                                                                                           = approximately                                                                                                                                                                           1.27                                                                                                                                                                               kg/month.                                                                                                                Still                                                                           Wrong                                                                                       Answer                                                                                                           after                                                                                                                      Reflection
            Again, none of the options match this sequence.                                                                 That's roughly 1.27 kg per month, which is close to my earlier calculation of
            ...                                                                                                             1.28 kg per month.
                                                                                                                             ...



                            Incompleteness                                                                  Interference

                                                                                             Question                                                                                                             Question

                                                                                                                                      Question: As shown in the figure, the straight line a parallel b,
                                                                                                                                            then the degree of angle 1 is ().
                                                                                                                                  A:75°     B:105°     C:140°     D:145°"
       Question: Suppose there are n plates equally spaced around a circular table. Ross wishes to place
      an identical gift on each of k plates, so that no two neighbouring plates have gifts. Let f(n, k)                                                                                   Answer: B
        represent the number of ways in which he can place the gifts. For example, f(6,3)=2, as shown
       below.                                                                                                                                                                      Model Response
       Determine the smallest possible value of n+k among all possible ordered pairs of integers (n, k) for                           Reflection
      which f(n, k) is a positive multiple of 2009.                                                                      Answer: 54                           ...

                                                                                                                          If angle 1 is 140°, then perhaps there's a triangle involved where the sum of angles
                                                                                                                          is 180°. Let me see: if I have a triangle with angles 40°, 35°, and angle 3 (let's
                                                                            Model Response                             say), then angle 3 would be 180° - 40° - 35° = 105°.
               Reflection                                                                                                                                                                                                           Correct Answer
           ...                                                                                                            If angle 1 is corresponding to angle 3, then it should also be 105°,
                                                                                                                                                                                         Wrong Answer           Let’s                 consider                          that f(n,k)                                      is a                                           combinatorial                                                         number,                                                                  and                                                                     since                                                                           it‘s                                                                                related                                                                                         to
           binomial                    coefficients,                                  perhaps                                           I                                            can use Lucas’                                                           theorem                                                                    to                                                                      determine                                                                                when                                                                                     f(n,k)                                                                                             is                            but according to my earlier calculation, angle 1 is 140°.
           divisible                     by                         a prime                                p. Lucas‘                                          theorem                                                  states                                                         that                                                              C(m,p^e)                                                                        ≡ C(m_p,                                                                                e_p)                                                                                      mod                                                                                          p,                                                                                            where                         Hmm,                                                                                                                               there's                                                                                                                                       inconsistency                                                                                                                                                     here.                                                                                                                                                           Maybe                                                                                                                                                                  I need                                                                                                                                                                        to                                                                                                                                                                           approach                                                                                                                                                                                    this                                                                                                                                                                                         differently.           m_p               and e_p                       are                           the                               digits                                      in base                                               p of                                                    m and                                                          e.
                                                                                                                          If                                                                                                                             angle                                                                                                                                   between                                                                                                                                           lines                                                                                                                                                 "a"                                                                                                                                                     and                                                                                                                                                         "b"                                                                                                                                                             is                                                                                                                                                                180°                                                                                                                                                                      -                                                                                                                                                                       40°                                                                                                                                                                            =                                                                                                                                                                             140°, then                                                                                                                                                                                        angle                                                                                                                                                                                              1, being           However,                    applying                             Lucas'                                    theorem                                            directly                                                      to f(n,k)                                                                is complicated                                                                               due                                                                                   to                                                                                      the                                                                                         circular
                                                                                                                          corresponding                                                                                                                                        to                                                                                                                                           that angle,                                                                                                                                                       should                                                                                                                                                              also                                                                                                                                                                   be                                                                                                                                                                      140°.           adjacency                     constraints.
           ...                                                                           Give Up Without Trying                               ...



                                 Figure 10: Examples of Reflection Error Types.





                                                17
<a id="page-18"></a>

### PDF 第 18 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency

D. More Qualitative Examples


                                           Precision & Recall

                                                                        Question:                    Key Steps Annotation:
                                                                            Find the length of DE.                  Key Caption:
                                                                                                                                                   1.  The image shows two triangle FDE and CAB,
                                                                                                             FE=3, DE=x, CB=15, AB=45
                                                                         Choices: A. 3  B. 4  C. 6 D. 9          Key1.  TriangleConclusion:ABC is similar to triangle DEF
                                                                                                                                                   2. AB/DE = BC/FE
                                                                                                                                                   3.  The length of DE is 9, which corresponds to
                                                                                                                                    choice D
                                                                  Answer: D. 9                                  4.  The correct answer is D



                                   GPT4o:

                                          Prediction:

                                                   To find the length of DE in the image provided, we need to carefully analyze the two given triangles:
                                                    triangle ABC and triangle DEF. The problem involves using the concept of **similar triangles** since the
                                                    triangles share the same angles, which makes their corresponding sides proportional.  ### Step 1: Confirm
                                                   the similarity of the triangles From the diagram: 1. Both triangle ABC and triangle DEF have a 90 degrees
                                                    angle (right triangles). 2. The pink arcs indicate that one of the other angles in triangle ABC is equal to
                                                   one of the angles in triangle DEF. 3. Since the angles of a triangle add up to 180degrees, the third angles
                                                    must also be equal. This means triangle ABC sim triangle DEF (the two triangles are similar).  ### Step 2:
                                                   Use  the  property  of  similar  triangles  For  similar  triangles,  the  sides  are  proportional.  That  is:
                                                   \[ \frac{AB}{DE} = \frac{BC}{EF} = \frac{AC}{DF}. \] From the diagram: - In triangle ABC :   -  AB = 45   -
                                                   BC = 15 - In triangle DEF:    - EF = 3   - DF = x We must find DE, so we will use the side ratio:
                                                   \[ \frac{AB}{DE} = \frac{BC}{EF}. \]  ### Step 3: Substitute known values Substituting the values into the
                                                    proportion: \[ \frac{45}{DE} = \frac{15}{3}. \] Simplify the ratio on the right-hand side: \[ \frac{45}{DE}
                                                   = 5. \]  ### Step 4: Solve for DE Rearrange the equation to solve for DE: \[ DE = \frac{45}{5}. \] Simplify:
                                                   \[ DE = 9. \]  ### Final Answer: The length of  DE is: 9.


                                          Step Partition & Judgment:

                                                          step_type: Background Information
                                                          conclulsion: To find the value of DE in the image provided, we need to carefully analyze the two given
                                                          triangles: triangle ABC and triangle DEF. The problem involves using the concept of similar triangles
                                                          since the triangles share the same angles, which makes their corresponding sides proportional.
                         ✓   step_type:conclusion:ImageBoth Descriptiontriangle ABC and triangle DEF have a 90° angle (right triangles).
                         ✓   step_type:conclusion:ImageThe pinkDescriptionarcs indicate that one of the other angles in triangle ABC is equal to one of the
                                                          angles in triangle DEF.

                                                          step_type: Logical Inference
                         ✓   premise:that one ofBoththetriangleother anglesABC andintriangletriangleDEFABChaveis equala 90° toangleone (rightof the triangles).angles in triangleThe pinkDEF.arcs indicate
                                                          conclusion: Since the angles of a triangle add up to 180°, the third angles must also be equal. This
                                                          means triangle ABC is similar to triangle DEF (the two triangles are similar).

                                                          step_type: Background Information
                                                          conclusion: For similar triangles, the sides are proportional. That is: AB/DE = BC/EF = AC/DF.
                         ✓   step_type:conclusion:ImageIn triangleDescriptionABC: AB = 45, BC = 15.

                         ✓   step_type:conclusion:ImageIn triangleDescriptionDEF: EF = 3, DF = x.


                                                                   For similar                                                                               triangles,                                                                                          the                                                                                              sides are                                                                                                        proportional.                                                                                                                      That                                                                                                                           is: AB/DE = BC/EF = AC/DF. In                         ✓   step_type:premise:triangle ABC:Logical Inference                                                                        AB = 45,                                                                                 BC = 15. In                                                                                             triangle                                                                                                      DEF:                                                                                                           EF = 3, DF =                                                                                                                        x.
                                                          conclusion: We must find DE, so we will use the side ratio: AB/DE = BC/EF.
                         ✓   step_type:premise: AB/DELogical= BC/EF.InferenceAB = 45, BC = 15, EF = 3.
                                                          conclusion: Substituting the values into the proportion: 45/DE = 15/3.


                                                                   45/DE                                                                         = 15/3.                         ✓   step_type:premise:conclusion:Logical Inference                                                                      Simplify                                                                               the ratio on the right-hand side: 45/DE = 5.


                                                                   45/DE                                                                         = 5.                         ✓   step_type:premise:conclusion:Logical Inference                                                                      Rearrange                                                                                the equation to solve for DE: DE = 45/5.
                         ✓   step_type:premise: DELogical= 45/5.Inference
                                                          conclusion: Simplify: DE = 9.
                         ✓   step_type:premise: DELogical= 9.   Inference
                                                          conclusion: The length of DE is: 9.


                                                                   Precision= 11/11



                                          Key Steps Annotation Matching:

                                                          Key Caption:
                         ✓   - TheFE=3,imageDE=x,showsCB=15,twoAB=45triangle FDE and CAB,

                                                          Key Conclusion:
                         ✓   -  Triangle ABC is similar to triangle DEF
                         ✓   -  AB/DE = BC/FE
                         ✓   - ThechoicelengthD   of DE is 9, which corresponds to
                         ✓   - The correct answer is D
                                                                     Recall= 5/5


                             Figure 11: Examples of Precision and Recall Evaluation.



                                                18
<a id="page-19"></a>

### PDF 第 19 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency



Qwen2-7b:


    Prediction:

     To find the length of DE, we can use the property of similar triangles. Since triangles DEF and ABC are
     similar, the corresponding sides are proportional. We can set up the following proportion:  \[ \frac{DE}{AB}
     = \frac{EF}{BC} \]   Given that EF = 3, BC = 15, and AB = 45, we can substitute these values into the
     proportion:    \[  \frac{DE}{45}  =  \frac{3}{15}  \]    Simplifying  the  right  side  of  the  equation:
     \[ \frac{DE}{45} = \frac{1}{5} \]  To find DE, we can cross-multiply:  \[ 5 \cdot DE = 45 \cdot 1 \]  \[ 5
     \cdot DE = 45 \]  \[ DE = \frac{45}{5} \]  \[ DE = 9 \]  Therefore, the length of DE is 9. The correct
     answer is D: 9.


    Step Partition & Judgment:

            step_type: background information
            conclusion: To find the length of DE, we can use the property of similar triangles.

   ✓   step_type:premise: triangleslogical DEFinferenceand ABC are similar.
            conclusion: The corresponding sides are proportional.
   ✓   step_type:premise: thelogicalcorrespondinginferencesides are proportional.
            conclusion: We can set up the following proportion: DE/AB = EF/BC.

            step_type: background information
            conclusion: Given that EF = 3, BC = 15, and AB = 45.
   ✓   step_type:premise: DE/ABlogical= EF/BCinferenceand EF = 3, BC = 15, AB = 45.
            conclusion: Substitute these values into the proportion: DE/45 = 3/15.

            step_type: logical inference
   ✓   premise: DE/45 = 3/15.
            conclusion: Simplifying the right side of the equation: DE/45 = 1/5.
   ✓   step_type:premise: DE/45logical= 1/5.inference
            conclusion: To find DE, we can cross-multiply: 5 * DE = 45 * 1.

            step_type: logical inference
   ✓   premise: 5 * DE = 45.
            conclusion: DE = 45/5.

            step_type: logical inference
   ✓   premise: DE = 45/5.
            conclusion: DE = 9.

            step_type: logical inference
   ✓   premise: DE = 9.
            conclusion: The length of DE is 9. The correct answer is D: 9.

                                 Precision= 8/8




    Key Steps Annotation Matching:

             Key Caption:
   ✓    - TheFE=3,imageDE=x,showsCB=15,twoAB=45triangle FDE and CAB,

             Key Conclusion:
   ✓    -  Triangle ABC is similar to triangle DEF
   ✓    -  AB/DE = BC/FE
   ✓    - The length of DE is 9, which corresponds to
                choice D
   ✓    - The correct answer is D
                                  Recall= 5/5


                    Figure 12: Examples of Precision and Recall Evaluation.




                                        19
<a id="page-20"></a>

### PDF 第 20 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency



                        QVQ-72B-Preview:

                               Prediction:

                                      So, I've got this geometry problem here, and I need to find the length of DE. Let's see what information is
                                      given.  There are two triangles: one is ADE, and the other is ABC. From the diagram, it looks like triangle
                                      ADE is similar to triangle ABC because they have corresponding angles that are equal. Specifically, angle D
                                      in triangle ADE corresponds to angle A in triangle ABC, angle E in triangle ADE corresponds to angle B in
                                      triangle ABC, and angle F in triangle ADE corresponds to angle C in triangle ABC.  Wait, actually, looking
                                      closer,  triangle  ADE  is  similar  to  triangle  ABC,  with  angle  D  corresponding  to  angle  A,  angle  E
                                      corresponding to angle B, and angle F corresponding to angle C. But in the diagram, it's labeled as points
                                      D, E, F for triangle ADE, and points A, B, C for triangle ABC.  Hmm, maybe I need to be careful with the
                                      labeling. Let's look at the coordinates or the labels again.  In triangle ADE, points are D, E, F, and in
                                      triangle ABC, points are A, B, C. There's a side labeled FE with length 3, and sides BC and DE are to be
                                      compared.  Wait, no. In triangle ADE, sides are DE, DF, and EF, with EF being 3 units. In triangle ABC,
                                      sides are AB, BC, and AC, with BC being 15 units and AB being 45 units.  There's also an angle in triangle
                                      ABC that's 90 degrees, specifically angle C is 90 degrees, since there's a red square at point C, which
                                      typically denotes a right angle.  So, triangle ABC is a right-angled triangle with right angle at C, AB is
                                      45 units, and BC is 15 units.   Now, since triangle ADE is similar to triangle ABC, their corresponding
                                      sides are proportional.  First, I need to identify which sides correspond to each other.  Given that angle
                                      F in triangle ADE corresponds to angle C in triangle ABC, and angle C is the right angle, so angle F must
                                      also be the right angle in triangle ADE.  Wait, but in triangle ADE, angles D, E, and F are all acute in
                                      the diagram, so maybe that's not the case.   Actually, perhaps I misread the diagram. Let's assume that
                                      triangle ADE is similar to triangle ABC, with angle at A being common to both.  But in the diagram, it
                                      seems like triangle ADE is positioned with point A at the origin, and triangle ABC is another triangle with
                                      point A shared.  Wait, perhaps point A is the common vertex.  Let me try to rephrase the problem based on
                                      the diagram description.  Points D, E, F form a triangle with EF = 3 units.  Points A, B, C form another
                                      triangle with BC = 15 units and AB = 45 units, and angle at C is 90 degrees.  There's a line from A to C,
                                      and another line from A to E.   Wait, maybe I need to consider the similarity ratio between the two
                                      triangles.  Since triangle ADE is similar to triangle ABC, the ratio of their corresponding sides should be
                                      equal.  Let’s denote the lengths:  Let DE = x (which is what we need to find)  EF = 3 units  In triangle
                                      ABC, BC = 15 units, AB = 45 units, and angle C = 90 degrees.   Wait, but in triangle ADE, which sides
                                      correspond to which sides in triangle ABC?  If angle at D corresponds to angle at A, angle at E corresponds
                                      to angle at B, and angle at F corresponds to angle at C, then:  DE corresponds to AB, DF corresponds to AC,
                                      and EF corresponds to BC.  Therefore, the ratio of corresponding sides is DE / AB = DF / AC = EF / BC.  But
                                      we don't know the length of DF or AC.  Wait, perhaps I need to find another way.  Since triangle ABC is a
                                      right-angled triangle with right angle at C, and sides BC = 15 and AB = 45, I can find AC using the
                                      Pythagorean theorem.  In triangle ABC:  AB² = AC² + BC²  45² = AC² + 15²  2025 = AC² + 225  AC² = 2025 -
                                      225 = 1800  AC = sqrt(1800) = sqrt(100*18) = 10*sqrt(18) = 10*3*sqrt(2) = 30*sqrt(2)  So, AC = 30*sqrt(2)
                                      Now, since triangle ADE is similar to triangle ABC, with DE corresponding to AB, DF corresponding to AC,
                                      and EF corresponding to BC.  Therefore, the ratio DE / AB = DF / AC = EF / BC  We can write:  x / 45 = DF /
                                      (30*sqrt(2)) = 3 / 15  Simplify 3 / 15: that's 1 / 5  So, x / 45 = 1 / 5  Therefore, x = 45 * (1 / 5) = 9
                                      So, DE = 9 units  Therefore, the correct choice is D:9  **Final Answer**: D.

                               Step Partition & Judgment:

                                             step_type: background information
                                             conclusion: there are two triangles: one is ADE, and the other is ABC.
                   ✓   step_type:conclusion:imagetriangledescriptionADE is similar to triangle ABC because they have corresponding angles that are
                                             equal.
                   ✓   step_type:conclusion:imagein triangledescriptionADE, points are D, E, F, and in triangle ABC, points are A, B, C.
                   ✓   step_type:conclusion:imagethere'sdescriptiona side labeled FE with length 3.
                   ✓   step_type:conclusion:imagein triangledescriptionABC, sides are AB, BC, and AC, with BC being 15 units and AB being 45 units.
                   ✓   step_type:conclusion:imagethere'sdescriptionan angle in triangle ABC that's 90 degrees, specifically angle C is 90 degrees.


                                                      triangle ABC is a right-angled triangle with right angle at C, AB is 45 units, and BC is                   ✓   step_type:premise:15 units.  logical inference
                                             conclusion: since triangle ADE is similar to triangle ABC, their corresponding sides are
                                             proportional.

                  ×   step_type:premise: givenlogicalthatinferenceangle F in triangle ADE corresponds to angle C in triangle ABC, and angle C is
                                             the right angle.
                                             conclusion: angle F must also be the right angle in triangle ADE.

                   ✓   step_type:premise: trianglelogicalADEinferenceis similar to triangle ABC, with angle at A being common to both.
                                             conclusion: points D, E, F form a triangle with EF = 3 units.

                   ✓   step_type:premise: pointslogicalA, B,inferenceC form another triangle with BC = 15 units and AB = 45 units, and angle at C
                                             is 90 degrees.
                                             conclusion: there's a line from A to C, and another line from A to E.

                                             step_type: logical inference
                   ✓   premise:should besinceequal.triangle ADE is similar to triangle ABC, the ratio of their corresponding sides
                                             conclusion: let DE = x (which is what we need to find), EF = 3 units.

                   ✓   step_type:premise: inlogicaltriangleinferenceABC, BC = 15 units, AB = 45 units, and angle C = 90 degrees.
                                             conclusion: the ratio of corresponding sides is DE / AB = DF / AC = EF / BC.
                   ✓   step_type:premise: welogicaldon't knowinferencethe length of DF or AC.
                                             conclusion: perhaps I need to find another way.


                                                      triangle ABC is a right-angled triangle with right angle at C, and sides BC = 15 and AB =                   ✓   step_type:premise:45.        logical inference
                                             conclusion: I can find AC using the Pythagorean theorem.
                   ✓   step_type:premise: inlogicaltriangleinferenceABC: AB² = AC² + BC², 45² = AC² + 15².
                                             conclusion: AC² = 2025 - 225 = 1800, AC = sqrt(1800) = 30*sqrt(2).


                                                      since                                                            triangle                                                                     ADE                                                                        is similar to                                                                                       triangle                                                                                                ABC, with DE corresponding to AB, DF                   ✓   step_type:premise:correspondinglogical inference                                                           to                                                              AC, and                                                                     EF corresponding                                                                                      to                                                                                          BC.
                                             conclusion: the ratio DE / AB = DF / AC = EF / BC.


                                                     x /                                                          45 = DF /                                                                    (30*sqrt(2))                                                                                =                                                                                  3                                                                                    /                                                                                       15.                   ✓   step_type:premise:conclusion:logical inference                                                         simplify                                                                 3                                                                   / 15: that's                                                                                1                                                                                  /                                                                                    5.

                   ✓   step_type:premise: x /logical45 = 1inference/ 5.
                                             conclusion: x = 45 * (1 / 5) = 9.
                   ✓   step_type:premise: x =logical9.    inference
                                             conclusion: DE = 9 units.
                   ✓   step_type:premise: DElogical= 9 units.inference
                                             conclusion: the correct choice is D:9.

                                                        Precision= 18/19


                    Figure 13: Examples of Precision and Recall Evaluation.




                                        20
<a id="page-21"></a>

### PDF 第 21 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency


QVQ-72B-Preview:

     Annotation Steps Matching:

             Key Caption:
   ✓    - The image shows two triangle FDE and CAB,               FE=3, DE=x, CB=15, AB=45

             Key Conclusion:
   ×    -  Triangle ABC is similar to triangle DEF

             -  AB/DE = BC/FE   ✓
   ✓    - The length of DE is 9, which corresponds to
               choice D
   ✓    - The correct answer is D

                                  Recall= 4/5



                          Figure 14: Examples of Precision and Recall Evaluation.





                                              21
<a id="page-22"></a>

### PDF 第 22 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency


                                          Precision & Recall

                                                                       Question:                       Key Steps Annotation:

                                                                         Between                                                                                          Object                                                                                                                 1:                                                                                                         the                                                                                                                  object                                                                                                               within                                                                                                                          the
                                                                                                      Key                                                                                                                             Caption:                                                                              orange                                                                                                rectangle                                                                                     and                                                                                                      Object                                                                                                                              2:                                                                                                                    the                                                                                                                              object
                                                                                                                                                      1.                                                                                                                                    Object                                                                                                                              1                                                                                                                                                    (within                                                                                                                                                           the                                                                                                                                                 orange                                                                                                                                                                                rectangle)                                                                                  within                                                                                             the                                                                                               blue                                                                                                              rectangle,                                                                                               which                                                                                                     one                                                                                                                                                                   is
                                                                                                                         seems                                                                                                                                                                       like                                                                                                                                      a fruit                                                                                                                                 and                                                                                                                                                     nut                                                                                                                                                                     salad                                                                                            healthier?
                                                                                                                                                      2.                                                                                                                                    Object                                                                                                                              2                                                                                                                                                    (within                                                                                                                                                           the                                                                                                                                                           blue                                                                                                                                                                             rectangle)
                                                                                                                         seems like a frilled or roasted chicken
                                                                        Choices: A. Object 1  B. Object 2           Key Conclusion:
                                                                                      C. Both objects are equally              1.  Compared with Object 2, Object 1 seems
                                                                                               healthy                            more healthier because of the fruit and
                                                                              D. It's impossible to                       nut. Or Object 1 has less calories and fat
                                                                                          determine healthiness                  compared with Object 2.
                                                                                          based on appearance alone              2.  The correct answer is A.

                                                                 Answer:  A. Object 1



                                  GPT4o:

                                         Prediction:

                                                  To determine which object (Object 1 or Object 2) is healthier based on their appearance in the image, let's
                                                   follow  a  systematic  approach  and  analyze  the  information  step  by  step.  ###  Step  1:  **Understand  the
                                                   context** The focus of the question is to assess healthiness specifically based on appearance. This means
                                                  we cannot consider factors like portion size, preparation method, or nutritional values that aren't visible
                                                  in the image. ### Step 2: **Examine Object 1 (orange rectangle)** - Object 1 is a bowl of what appears to
                                                  be a seafood dish, possibly containing shellfish (such as mussels, clams, or shrimp), and some visible
                                                   vegetables like tomatoes. - Observations:  - Seafood is generally considered healthy as it can be low in
                                                   saturated fat and rich in nutrients like omega-3 fatty acids, vitamins, and minerals.  - The inclusion of
                                                   vegetables (tomatoes in this case) adds to its nutritional value, providing vitamins and fiber.  - However,
                                                  we cannot tell the exact cooking method used. If it’s fried or cooked with excessive oil, cream, or butter,
                                                  it might reduce its healthiness. ### Step 3: **Examine Object 2 (blue rectangle)** - Object 2 appears to be
                                                  a plate of fried or sautéed meat or poultry. - Observations:  - From the appearance, it seems that the food
                                                  in Object 2 is fried or cooked in oil, which may increase its saturated fat content.   - There are no
                                                   visible vegetables or fruits in the dish, which makes it likely less balanced in terms of nutrients
                                                   compared to Object 1.  - Additionally, fried foods are generally considered less healthy when consumed in
                                                   excess due to their high calorie and fat content. ### Step 4: **Compare the two objects**- Based on
                                                   appearance, Object 1 likely contains lean proteins from seafood and includes vegetables, which makes it a
                                                   more nutrient-dense option. - Object 2 seems heavier on oils or fats due to the fried presentation and
                                                   lacks visible vegetables or a balance of food groups. - We are relying on visible attributes, and thus,
                                                   there is some uncertainty since we don’t have information about portion size, seasoning (e.g., salt), or
                                                   preparation (e.g., sauces). ### Step 5: **Consider the limitations of appearance** - It’s important to note
                                                   that  determining  healthiness  solely  based  on  appearance  has  limitations.  Cooking  methods,  hidden
                                                   ingredients, and the full nutritional profile are not always evident. ### Final Answer:**A. Object 1**
                                                   appears healthier based on visible factors, such as the inclusion of seafood and vegetables. However, this
                                                  is a visual assessment and may not reflect exact healthiness. If the ambiguity of appearance is prioritized,
                                                   the  answer  could  alternatively  be  **D.  It's  impossible  to  determine  healthiness  based  on  appearance
                                                   alone.**

                                         Step Partition & Judgment:

                                                         step_type: Background Information
                                                          conclusion: The focus of the question is to assess healthiness specifically based on appearance. This
                                                         means we cannot consider factors like portion size, preparation method, or nutritional values that
                                                         aren't visible in the image.


                                                                     Object                                                                            1 is a                                                                                   bowl                                                                                        of what                                                                                                appears                                                                                                        to be a                                                                                                                seafood dish,                                                                                                                              possibly                                                                                                                                       containing shellfish                        ×   step_type:conclusion:(such as mussels,Image Description                                                                           clams,                                                                                  or                                                                                     shrimp),                                                                                              and                                                                                                  some visible                                                                                                               vegetables                                                                                                                          like                                                                                                                               tomatoes.

                                                         step_type: Logical Inference
                         ✓   premise:nutrientsSeafoodlike omega-3is generallyfatty acids,consideredvitamins,healthyandasminerals.it can be low in saturated fat and rich in
                                                          conclusion: Seafood is generally considered healthy.


                                                                  The                                                                      inclusion of vegetables (tomatoes in this case) adds to its nutritional value, providing                         ✓   step_type:premise:vitamins andLogical Inference                                                                      fiber.
                                                          conclusion: The inclusion of vegetables adds to its nutritional value.


                                                                  We                                                                     cannot                                                                            tell                                                                                 the                                                                                     exact                                                                                           cooking                                                                                                   method used. If it’s fried or cooked with excessive oil,                         ✓   step_type:premise:cream, or  Logical Inference                                                                   butter,                                                                           it                                                                              might                                                                                    reduce                                                                                           its                                                                                               healthiness.
                                                          conclusion: The cooking method might reduce its healthiness.

                        ×   step_type:conclusion:ImageObjectDescription2 appears to be a plate of fried or sautéed meat or poultry.

                                                         step_type: Logical Inference
                         ✓   premise:increase itsFromsaturatedthe appearance,fat content.it seems that the food in Object 2 is fried or cooked in oil, which may
                                                          conclusion: The food in Object 2 may have increased saturated fat content.

                                                         step_type: Logical Inference
                         ✓   premise:terms of nutrientsThere are comparedno visibletovegetablesObject 1. or fruits in the dish, which makes it likely less balanced in
                                                          conclusion: Object 2 is likely less balanced in terms of nutrients compared to Object 1.


                                                                  Fried                                                                        foods                                                                              are generally considered less healthy when consumed in excess due to their high                         ✓   step_type:premise:calorie andLogical Inference                                                                     fat                                                                         content.
                                                          Conclusion: Fried foods are generally considered less healthy.

                                                         step_type: Logical Inference
                         ✓   premise:vegetables,Basedwhichon makesappearance,it a moreObjectnutrient-dense1 likely containsoption.lean proteins from seafood and includes
                                                          conclusion: Object 1 is a more nutrient-dense option.

                                                         step_type: Logical Inference
                         ✓   premise:vegetablesObjector a balance2 seems ofheavierfood groups.on oils or fats due to the fried presentation and lacks visible
                                                          conclusion: Object 2 seems heavier on oils or fats and lacks balance.

                                                         step_type: Background Information
                                                          conclusion: We are relying on visible attributes, and thus, there is some uncertainty since we don’t
                                                         have information about portion size, seasoning (e.g., salt), or preparation (e.g., sauces).

                                                         step_type: Background Information
                                                          conclusion: It’s important to note that determining healthiness solely based on appearance has
                                                          limitations. Cooking methods, hidden ingredients, and the full nutritional profile are not always
                                                         evident.


                                                                  Based on visible factors, such as the inclusion of seafood and vegetables, Object 1 appears                         ✓   step_type:premise:healthier. Logical Inference
                                                          conclusion: A. Object 1 appears healthier based on visible factors.

                                                                    Precision= 9/11


                                         Key Steps Annotation Matching:

                                               Key Caption:
                         ✓   -  Object 1 (within the orange rectangle) seems like a fruit and nut salad.
                        ×   - Object 2 (within the blue rectangle) seems like a frilled or roasted chicken

                                               Key Conclusion:
                         ✓   -  Comparedhas less withcaloriesObjectand2,fatObjectcompared1 seemswithmoreObjecthealthier2.     because of the fruit and nut. Or Object 1
                         ✓   - The correct answer is A.
                                                                      Recall= 3/4

                    Figure 15: Examples of Precision and Recall Evaluation.




                                        22
<a id="page-23"></a>

### PDF 第 23 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency



      Qwen2-7b:

           Prediction:

              To determine which object is healthier between Object 1 (the object within the orange rectangle) and Object
              2 (the object within the blue rectangle), let's analyze each object based on common health criteria:  1.
              **Object 1 (the object within the orange rectangle)**:    - This appears to be a bowl of yogurt topped with
              various fruits, nuts, and possibly granola.     - Yogurt is a good source of protein and probiotics.     -
              Fruits provide vitamins, minerals, and fiber.     - Nuts and granola add healthy fats and fiber.     -
              Overall, this looks like a balanced and nutritious meal.   2. **Object 2 (the object within the blue
              rectangle)**:    - This appears to be a plate of fried food, possibly pancakes or waffles.    - Fried foods
              are typically high in calories, saturated fats, and sodium.     - They may also be low in fiber and other
              essential nutrients.    - This does not appear to be a healthy choice compared to the yogurt bowl.  Based
              on this analysis, **Object 1 (the yogurt bowl with fruits and granola)** is healthier than **Object 2 (the
              fried food)**.  **Final Answer: A. Object 1**

           Step Partition & Judgment:
                     step_type: background information
                     conclusion: To determine which object is healthier between Object 1 (the object within the orange
                     rectangle) and Object 2 (the object within the blue rectangle), let's analyze each object based on
                     common health criteria.

       ✓    step_type:conclusion:imageObjectdescription1 (the object within the orange rectangle) appears to be a bowl of yogurt topped
                     with various fruits, nuts, and possibly granola.

                     step_type: background information
                     conclusion: Yogurt is a good source of protein and probiotics.

                     step_type: background information
                     conclusion: Fruits provide vitamins, minerals, and fiber.

                     step_type: background information
                     conclusion: Nuts and granola add healthy fats and fiber.

                     step_type: logical inference
       ✓    premise: yogurt is a good source of protein and probiotics. Fruits provide vitamins, minerals, and
                     fiber. Nuts and granola add healthy fats and fiber.
                     conclusion: Overall, this looks like a balanced and nutritious meal.

       ×    step_type:conclusion:imageObjectdescription2 (the object within the blue rectangle) appears to be a plate of fried food,
                     possibly pancakes or waffles.

                     step_type: background information
                     conclusion: Fried foods are typically high in calories, saturated fats, and sodium.

                     step_type: background information
                     conclusion: They may also be low in fiber and other essential nutrients.

                     step_type: logical inference
       ✓    premise:in fiber andfriedotherfoodsessentialare typicallynutrients.high in calories, saturated fats, and sodium. They may also be low
                     conclusion: This does not appear to be a healthy choice compared to the yogurt bowl.

                     step_type: logical inference
       ✓    premise:choice comparedoverall,tothisthe yogurtlooks likebowl.a balanced and nutritious meal. This does not appear to be a healthy
                     conclusion: Based on this analysis, Object 1 (the yogurt bowl with fruits and granola) is healthier
                     than Object 2 (the fried food).

                     step_type: logical inference
       ✓    premise: based on this analysis, Object 1 (the yogurt bowl with fruits and granola) is healthier than
                     Object 2 (the fried food).
                     conclusion: Final Answer: A. Object 1

                                     Precision= 5/6





           Key Steps Annotation Matching:

                Key Caption:
       ✓    -  Object 1 (within the orange rectangle) seems like a fruit and nut salad.
      ×    - Object 2 (within the blue rectangle) seems like a frilled or roasted chicken

                Key Conclusion:

       ✓    -  Comparedhas less withcaloriesObjectand2,fatObjectcompared1 seemswithmoreObjecthealthier2.     because of the fruit and nut. Or Object 1
       ✓    - The correct answer is A.
                                      Recall= 3/4



                    Figure 16: Examples of Precision and Recall Evaluation.



                                        23
<a id="page-24"></a>

### PDF 第 24 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency



     QVQ-72B-Preview:


          Prediction:

            Alright, I've got this question here. There are two objects, each within a colored rectangle: one in orange
            and one in blue. The task is to determine which one is healthier. The options are A. Object 1 (orange
            rectangle), B. Object 2 (blue rectangle), C. Both are equally healthy, or D. It's impossible to determine
            based on appearance alone.   First, I need to understand what's in each rectangle.   Object 1 is in the
            orange rectangle. From the description, it sounds like a breakfast setup. There's a plate with what appears
            to be yogurt topped with fruits like bananas, strawberries, and possibly blueberries, along with some
            granola. There's also a glass of iced tea, a mug of coffee, and some roasted potatoes in the background.
            Object 2 is in the blue rectangle. It's a plate of roasted potatoes.  So, the comparison is between a fruit
            and granola yogurt bowl and a plate of roasted potatoes.  Now, I need to assess which of these is healthier.
            Let's consider the nutrients each provides.  The yogurt bowl with fruits and granola:  - Yogurt is a good
            source of protein and calcium.  - Bananas provide potassium and vitamin C.  - Strawberries and blueberries
            are rich in antioxidants and vitamin C.  - Granola can vary, but it often contains oats, which are high in
            fiber, and possibly nuts or seeds for healthy fats and additional protein.   The roasted potatoes:   -
            Potatoes are a source of carbohydrates, vitamin C, and potassium.  - How they're roasted matters: if it's
            with minimal oil and seasoning, they can be quite healthy. However, if they're deeply fried or cooked with
            a lot of fat, they might be higher in unhealthy fats.  Given that, the yogurt bowl seems to offer a more
            balanced meal with a combination of protein, carbohydrates, fiber, and vitamins from the fruits and granola.
            The roasted potatoes, while they can be part of a healthy diet, especially if prepared healthily, don't
            provide as diverse a range of nutrients as the yogurt bowl.  Moreover, depending on the preparation, the
            potatoes might be higher in calories from fat compared to the yogurt bowl.   But, to be fair, if the
            potatoes are roasted with very little oil, they could still be quite healthy.  However, in general, the
            yogurt bowl with fruits and granola is likely to be a healthier choice due to its nutrient density and
            diversity.  Therefore, my answer would be A. Object 1.  But I should consider if there's any way that the
            potatoes could be healthier in some scenarios.   For example, if the granola is high in sugar or if the
            portion sizes are not considered, the potatoes might be a better option in those specific cases.   But,
            based on the visual information provided, the yogurt bowl looks well-balanced and nutritious.  So, I think
            A. Object 1 is the correct answer.  **Final Answer** A.

          Step Partition & Judgment:

                   step_type: background information
                   conclusion: There are two objects, each within a colored rectangle: one in orange and one in blue. The
                   task is to determine which one is healthier. The options are A. Object 1 (orange rectangle), B. Object
                  2 (blue rectangle), C. Both are equally healthy, or D. It's impossible to determine based on appearance
                   alone.

                   step_type: image description
      ✓   conclusion: Object 1 is in the orange rectangle. There's a plate with what appears to be yogurt topped
                   with fruits like bananas, strawberries, and possibly blueberries, along with some granola. There's also
                  a glass of iced tea, a mug of coffee, and some roasted potatoes in the background.
      ✓   step_type:conclusion:imageObjectdescription2 is in the blue rectangle. It's a plate of roasted potatoes.

                   step_type: logical inference
      ✓   premise:conclusion:comparisonThe yogurtbetweenbowl aseemsfruittoandoffergranolaa moreyogurtbalancedbowlmealand awithplatea combinationof roasted potatoes.of protein,
                   carbohydrates, fiber, and vitamins from the fruits and granola.

                   step_type: logical inference
      ✓   premise:healthily,thedon'troastedprovidepotatoes,as diversewhileatheyrangecanofbenutrientspart of asa healthythe yogurtdiet,bowl.especially if prepared
                   conclusion: The yogurt bowl is likely to be a healthier choice due to its nutrient density and
                   diversity.


                   step_type: logical inference
      ✓   premise:conclusion:basedA. onObjectthe visual1 is theinformationcorrect answer.provided, the yogurt bowl looks well-balanced and nutritious.

                                         precision: 5/5





          Key Steps Annotation Matching:

                Key Caption:
      ✓    -  Object 1 (within the orange rectangle) seems like a fruit and nut salad.
      ×    - Object 2 (within the blue rectangle) seems like a frilled or roasted chicken

                Key Conclusion:
      ✓    -  Compared with Object 2, Object 1 seems more healthier because of the fruit and nut. Or Object 1
                      has less calories and fat compared with Object 2.
      ✓    - The correct answer is A.
                                               recall: 3/4



                    Figure 17: Examples of Precision and Recall Evaluation.




                                        24
<a id="page-25"></a>

### PDF 第 25 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency

                   Relevance Rate

                                                      Key Steps Annotation:                              Question:

                      How many puppies in the images are            Key Caption:
                                      sticking out their tongues?                                 1.  In the first image, there is one puppy,
                                                                                         while its tongue is not sticking out.
                                                                                                     2.  In the second image, there is one
                              Choices: A. Zero  B. Two  C. Three                  puppy, while its tongue is not
                                      D. None of the choices provided                                                                                                      sticking out.
                                           E. Four                                                                    Key Conclusion:
                                                                                                     1.  no puppies are sticking out their
                           Answer:  A. Zero                                       tongues.
                                                                                                     2.  The answer is A.


GPT4o:
                                                                                                                 irrelevant information
     Prediction:

      Let's carefully analyze the images step by step: ### 1. **Image 1 Analysis:** - The puppy is sitting on an
      orange blanket. - The position of the mouth and tongue of the puppy is visible. - Upon closer inspection,
      **the puppy is not sticking out its tongue.** ### 2. **Image 2 Analysis:** - The puppy is sitting on a
      purple background. - The position of the mouth and tongue is also visible here. - Upon careful observation,
      **the puppy is again not sticking out its tongue.** ### Intermediate Observations: - In both images, the
      puppies are not sticking out their tongues. ### Final Answer: **A. Zero**

     Judgments:
   ×   step_type:conclusion:imageThe puppydescriptionis sitting on an orange blanket.
    ✓   step_type:conclusion:imageThe positiondescriptionof the mouth and tongue of the puppy is visible.
    ✓   step_type:conclusion:logicalThe puppyinferenceis not sticking out its tongue.

   ×   step_type:conclusion:imageThe puppydescriptionis sitting on a purple background.
    ✓   step_type:conclusion:imageThe positiondescriptionof the mouth and tongue is also visible here.

    ✓   step_type:conclusion:logicalThe puppyinferenceis again not sticking out its tongue.

    ✓   step_type:conclusion:logicalIn bothinferenceimages, the puppies are not sticking out their tongues.
    ✓   step_type:conclusion:logicalFinal answerinferenceis A. Zero.

                    Raw Relevance Rate = 6/8



                            Figure 18: Examples of Relevance Rate Evaluation.





                                             25
<a id="page-26"></a>

### PDF 第 26 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency


Qwen2-7b:
                                                                                                                 irrelevant information
     Prediction:

      To determine how many puppies are sticking out their tongues in each image, let's analyze each image step
      by step:  1. **First Image**:    - There is one puppy in the image.    - The puppy is not sticking out its
      tongue.  2. **Second Image**:    - There is one puppy in the image.    - The puppy is not sticking out its
      tongue.  Since neither puppy in either image is sticking out its tongue, the correct answer is:  A. Zero

     Judgments:
    ✓   step_type:conclusion:backgroundto determineinformationhow many puppies are sticking out their tongues in each image, let's
             analyze each image step by step.

   ✓   step_type:conclusion:imagetheredescriptionis one puppy in the first image.
    ✓   step_type:conclusion:imagethe puppydescriptionin the first image is not sticking out its tongue.

   ✓   step_type:conclusion:imagetheredescriptionis one puppy in the second image.
   ✓   step_type:conclusion:imagethe puppydescriptionin the second image is not sticking out its tongue.
    ✓   step_type:conclusion:logicalsince neitherinferencepuppy in either image is sticking out its tongue, the correct answer is:
             A. Zero.

                    Raw Relevance Rate = 6/6


                            Figure 19: Examples of Relevance Rate Evaluation.





                                             26
<a id="page-27"></a>

### PDF 第 27 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency


QVQ-72B-Preview:
                                                                                                               irrelevant information
     Prediction:

      Alright, I've got this question here. It says, "How many puppies are sticking out their tongues in each
      image?" And there are two images provided. Okay, so I need to look at both images and count how many
      puppies are sticking out their tongues in each one. First, let's look at the first image. There's a corgi
      puppy sitting on an orange blanket against a wooden background with some decorative metalwork. The puppy
      has a tri-color coat—white, black, and tan—and it's wearing a black collar. Its tongue is not sticking out.
      So, in this image, the number of puppies with their tongues out is zero. Now, moving on to the second image.
      This one features another corgi puppy, this time against a purple fabric background. This puppy also has a
      tri-color coat and is looking directly at the camera. I need to check if its tongue is sticking out. Hmm,
      from what I can see, the puppy's mouth is closed, and there's no tongue visible. So, again, zero puppies
      with their tongues out in this image. Wait a minute, the question says "each image," and there are two
      images. So, for each image, I need to count the number of puppies sticking out their tongues. In both
      images, there's only one puppy, and neither of them is sticking out their tongue. So, for both images, the
      count is zero. But let's double-check. Maybe there's something I'm missing. In the first image, the puppy
      is clearly not showing its tongue. In the second image, similarly, no tongue is visible. So, I think it's
      safe to say that in each image, there are zero puppies sticking out their tongues. Looking back at the
      options provided: A . Zero, B. Two, C. Three, D. None of the choices provided, E. Four. Since in each image,
      the number is zero, and there are two images, but the question specifies "in each image," so the answer
      should be A. Zero. Wait, but maybe there's a trick here. Maybe "each image" refers to all images combined.
      But no, the question specifically says "in each image," so I think it's asking for the number per image,
      not in total across all images. So, my final answer is A. Zero. Final Answer: A. 0


     Judgments:
    ✓   step_type:conclusion:backgroundThe questioninformationasks how many puppies are sticking out their tongues in each image.

   ×   step_type:conclusion:imageThe firstdescriptionimage shows a corgi puppy on an orange blanket with a tri-color coat and a
             black collar.
    ✓   step_type: logical inference
             conclusion: In the first image, the puppy's tongue is not sticking out, so the count is zero.

             step_type: image description
   ×   conclusion: The second image shows a corgi puppy against a purple fabric background with a tri-
             color coat.
    ✓   step_type: logical inference             conclusion: In the second image, the puppy's mouth is closed, so the count is zero.
    ✓   step_type:conclusion:logicalFor eachinferenceimage, the number of puppies sticking out their tongues is zero.

    ✓   step_type:conclusion:logicalDouble-checkinginferenceconfirms zero puppies sticking out their tongues in both images.

    ✓   step_type:conclusion:backgroundReviewing informationthe answer choices: A. Zero, B. Two, C. Three, D. None of the choices
             provided, E. Four.
    ✓   step_type:conclusion:logicalThe answerinferenceis A. Zero, as the count is zero in each image.

    ✓   step_type:conclusion:logicalConsideringinferenceif 'each image' refers to all images combined, but it specifies per image.

    ✓   step_type:conclusion:logicalFinal answerinferenceis A. Zero.

                     Raw Relevance Rate = 9/11


                            Figure 20: Examples of Relevance Rate Evaluation.





                                             27
<a id="page-28"></a>

### PDF 第 28 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency

                    Reflection Quality

                             Question:                        Key Steps Annotation:
                                 In the diagram, ACDF is a rectangle            Key Conclusion:
                                with AC=200 and CD=50. Also,                     1.  join BE
                                      triangle FBD and triangle AEC are                   2. BAFE is a rectangle
                                congruent triangles which are right-                  3. BCDE is a rectangle
                                 angled at B and E, respectively. What                4.  1/4 of rectangle ABEF is shaded and
                                              is the area of the shaded region?                    1/4 of rectangle BCDE is shaded
                                                                                               5.  The total shaded area is 1/4(10000) =
                                                                             2500                          Answer:  2500
                                                                                               6.  The answer is 2500



QVQ-72B-Preview:
                                                                                      redundant reflection
    Prediction:

     …
     Now, triangle FBD has a right angle at B, so B is somewhere on FD, I think. Wait, no, FD is the bottom side
     of the rectangle. But if FBD is a triangle with vertices at F, B, and D, and it's right-angled at B, then B
     must be one of the endpoints, but that doesn't make sense. Wait, maybe B is not on FD. Similarly, triangle
     AEC is right-angled at E, and E is on FD, perhaps. Wait, maybe I need to think differently. Since AC = 200
     and CD = 50, and the triangles are congruent right-angled triangles, perhaps the legs of these triangles
     are equal in length. Let’s denote the legs of the right-angled triangles. Let’s say, for triangle FBD, FB
     and BD are the legs, and for triangle AEC, AE and EC are the legs. Given that the triangles are congruent,
     FB = EC and BD = AE. But since both triangles are right-angled and congruent, their corresponding sides
     must be equal. Wait, but I need to find the area of the shaded region, which is the overlapping area of
     these two triangles. Maybe I should find the area of one triangle and then double it, and then subtract the
     non-overlapping parts. Alternatively, perhaps I can find the area of the intersection directly. Let me try
     to find the equations of the lines that form the triangles to find their intersection points.
     …


    Judgments:

            …
   ✓   conclusion: B is not on FD, reconsidering the position of B
   ×   conclusion: The legs of the triangles are equal, reconsidering the length of the sides
   ×   conclusion:calculating itThe goal is to find the area of the shaded region, reconsidering the method of
            …


                          Figure 21: Examples of Reflection Quality Evaluation.





                                             28
<a id="page-29"></a>

### PDF 第 29 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency

E. Detailed Evaluation Setup

E.1. CoT Quality Evaluation Prompts

    Recall Evaluation Prompt

   You are an expert system to verify solutions to image-based problems. Your task is to match the ground truth middle
    steps with the provided solution.

   INPUT FORMAT:
    1. Problem: The original question/task
    2. A Solution of a model
    3. Ground Truth: Essential steps required for a correct answer

  MATCHING PROCESS:

   You need to match each ground truth middle step with the solution:

   Match Criteria:
    - The middle step should exactly match in the content or is directly entailed by a certain content in the solution
    - All the details must be matched, including the specific value and content
    - You should judge all the middle steps for whether there is a match in the solution

  OUTPUT FORMAT:

   [
     {
       "step_index": \textless integer\textgreater,
       "judgment": "Matched" | "Unmatched"
     }
   ]

  ADDITIONAL RULES:
    1. Only output the JSON array with no additional information.
    2. Judge each ground truth middle step in order without omitting any step.

   Here are the problem, answer, solution, and ground truth middle steps:

   [Problem]

    {question}

   [Answer]

   {answer}

    [Solution]

    {solution}

   [Ground Truth Information]

    {gt annotation}





                                                29
<a id="page-30"></a>

### PDF 第 30 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency

Precision Evaluation Prompt

# Task Overview
Given a solution with multiple reasoning steps for an image-based problem, reformat it into well-structured steps
and evaluate their correctness.

# Step 1: Reformatting the Solution
Convert the unstructured solution into distinct reasoning steps while:
- Preserving all original content and order
- Not adding new interpretations
- Not omitting any steps

## Step Types
1. Logical Inference Steps
- Contains exactly one logical deduction
- Must produce a new derived conclusion
- Cannot be just a summary or observation

2. Image Observation Steps
- Pure visual observations
- Only includes directly visible elements
- No inferences or assumptions

3. Background Information Steps
- External knowledge or question context
- No inference process involved

## Step Requirements
- Each step must be atomic (one conclusion per step)
- No content duplication across steps
- Initial analysis counts as background information
- Final answer determination counts as logical inference

# Step 2: Evaluating Correctness
Evaluate each step against:

## Ground Truth Matching
For image observations:
- Key elements must match ground truth observations

For logical inferences:
- Conclusion must EXACTLY match or be DIRECTLY entailed by ground truth

## Reasonableness Check (if no direct match)
Step must:
- Premises must not contradict any ground truth or correct answer
- Logic is valid
- Conclusion must not contradict any ground truth
- Conclusion must support or be neutral to correct answer

## Judgement Categories
- ”Match”: Aligns with ground truth
- ”Reasonable”: Valid but not in ground truth



                                             30
<a id="page-31"></a>

### PDF 第 31 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency


    - ”Wrong”: Invalid or contradictory
    - ”N/A”: For background information steps

   # Output Requirements
    1. The output format must be in valid JSON format without any other content.
    2. For highly repetitive patterns, output it as a single step.
    3. Output maximum 40 steps. Always include the final step that contains the answer.

   Here is the json output format:
   ## Output Format

   [
     {
       "step_type": "image observation|logical inference|background information",
       "premise": "Evidence (only for logical inference)",
       "conclusion": "Step result",
       "judgment": "Match|Reasonable|Wrong|N/A"
     }
   ]

   Here is the problem, and the solution that needs to be reformatted to steps:

   [Problem]

    {question}

    [Solution]

    {solution}

    [Correct Answer]

   {answer}

   [Ground Truth Information]

    {gt annotation}



E.2. CoT Efficiency Prompt

   Relevance Rate Evaluation Prompt

   # Task Overview Given a solution with multiple reasoning steps for an image-based problem, evaluate the relevance
    to get a solution (ignore correct or wrong) of each step.

   # Step 1: Reformatting the Solution Convert the unstructured solution into distinct reasoning steps while:
    - Preserving all original content and order
    - Not adding new interpretations
    - Not omitting any steps

   ## Step Types
    1. Logical Inference Steps
    - Contains exactly one logical deduction



                                                31
<a id="page-32"></a>

### PDF 第 32 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency


- Must produce a new derived conclusion
- Cannot be just a summary or observation
2. Image Description Steps
- Pure visual observations
- Only includes directly visible elements
- No inferences or assumptions
3. Background Information Steps
- External knowledge or question context
- No inference process involved

## Step Requirements - Each step must be atomic (one conclusion per step)
- No content duplication across steps
- Initial analysis counts as background information
- Final answer determination counts as logical inference

# Step 2: Evaluating Relevancy
A relevant step is considered as: 75% content of the step must be related to trying to get a solution (ignore correct or
wrong) to the question.

IMPORTANT NOTE:
Evaluate relevancy independent of correctness. As long as the step is trying to get to a solution, it is considered
relevant. Logical fallacy, knowledge mistake, inconsistent with previous steps, or other mistakes do not affect
relevance. A logically wrong step can be relevant if the reasoning attempts to address the question.

The following behaviour is considered as relevant:
 i. The step is planning, summarizing, thinking, verifying, calculating, or confirming an intermediate/final conclusion
helpful to get a solution.
 ii. The step is summarizing or reflecting on previously reached conclusion relevant to get a solution.
 iii. Repeating the information in the question or give the final answer.
 iv. A relevant image depiction should be in one of following situation:
1. help to obtain a conclusion helpful to solve the question later;
2. help to identify certain patterns in the image later;
3. directly contributes to the answer
v. Depicting or analyzing the options of the question is also relevant.
vi. Repeating previous relevant steps are also considered relevant.

The following behaviour is considered as irrelevant:
 i. Depicting image information that does not related to what is asking in the question. Example: The question asks
how many cars are present in all the images. If the step focuses on other visual elements like the road or building,
the step is considered as irrelevant.
 ii. Self-thought not related to what the question is asking.
 iii. Other information that is tangential for answering the question.

# Output Format

[
  {
    "step_type": "image observation|logical inference|background information",
    "conclusion": "A brief summary of step result",
    "relevant": "Yes|No"
  }
]

# Output Rules



                                             32
<a id="page-33"></a>

### PDF 第 33 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency


Direct JSON output without any other output
Output at most 40 steps

Here is the problem, and the solution that needs to be reformatted to steps:
[Problem]

{question}

[Solution]

{solution}




Reflection Quality Evaluation Prompt

Here´s a refined prompt that improves clarity and structure:

# Task
Evaluate reflection steps in image-based problem solutions, where reflections are self-corrections or reconsideration
of previous statements.

# Reflection Step Identification
Reflections typically begin with phrases like:
- ”But xxx”
- ”Alternatively, xxx”
- ”Maybe I should”
- ”Let me double-check”
- ”Wait xxx”
- ”Perhaps xxx”
 It will throw a doubt of its previously reached conclusion or raise a new thought.

# Evaluation Criteria
Correct reflections must:
1. Reach accurate conclusions aligned with ground truth
2. Use new insights to find the mistake of the previous conclusion or verify its correctness.

Invalid reflections include:
1. Repetition - Restating previous content or method without new insights
2. Wrong Conclusion - Reaching incorrect conclusions vs ground truth
3. Incompleteness - Proposing but not executing new analysis methods
4. Other - Additional error types

# Input Format

[Problem]

{question}

[Solution]

{solution}




                                             33
<a id="page-34"></a>

### PDF 第 34 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency


   [Ground Truth]

    {gt annotation}

   # Output Requirements
    1. The output format must be in valid JSON format without any other content.
    2. Output maximum 30 reflection steps.

   Here is the json output format:
   ## Output Format

   [
     {
       "conclusion": "One-sentence summary of reflection outcome",
       "judgment": "Correct|Wrong",
       "error_type": "N/A|Repetition|Wrong Conclusion|Incompleteness|Other"
     }
   ]

   # Rules
    1. Preserve original content and order
    2. No new interpretations
    3. Include ALL reflection steps
    4. Empty list if no reflections found
    5. Direct JSON output without any other output



E.3. Direct Evaluation Prompt

   Answer Extraction Prompt

   You are an AI assistant who will help me to extract an answer of a question. You are provided with a question and a
    response, and you need to find the final answer of the question.

    Extract Rule:
    [Multiple choice question]
    1. The answer could be answering the option letter or the value. You should directly output the choice letter of the
    answer.
    2. You should output a single uppercase character in A, B, C, D, E, F, G, H, I (if they are valid options), and Z.
    3. If the meaning of all options are significantly different from the final answer, output Z.

   [Non Multiple choice question]
    1. Output the final value of the answer.  It could be hidden inside the last step of calculation or inference. Pay
    attention to what the question is asking for to extract the value of the answer.
    2. The final answer could also be a short phrase or sentence.
    3. If the response doesn’t give a final answer, output Z.

   Output Format: Directly output the extracted answer of the response.

    {In Context Examples}

    Question: {question}
   Answer: {response}




                                                34
<a id="page-35"></a>

### PDF 第 35 页

MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency


Your output:


Answer Scoring Prompt

You are an AI assistant who will help me to judge whether two answers are consistent.

Input Illustration: [Standard Answer] is the standard answer to the question. [Model Answer] is the answer extracted
from a model’s output to this question.
Task Illustration: Determine whether [Standard Answer] and [Model Answer] are consistent.

Consistent Criteria:
[Multiple-Choice questions]
1. If the [Model Answer] is the option letter, then it must completely matches the [Standard Answer].
2. If the [Model Answer] is not an option letter, then the [Model Answer] must completely match the option content
of [Standard Answer].
[Nan-Multiple-Choice questions]
1. The [Model Answer] and [Standard Answer] should exactly match.
2. If the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.

Output Format: 1. If they are consistent, output 1; if they are different, output 0.
2. DIRECTLY output 1 or 0 without any other content.
{In Context Examples}

Question: {question}
[Model Answer]: {extract answer}
[Standard Answer]: {gt answer}
Your output:





                                             35

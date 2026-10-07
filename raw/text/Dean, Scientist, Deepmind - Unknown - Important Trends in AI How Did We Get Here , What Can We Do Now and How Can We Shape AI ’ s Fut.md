# Dean, Scientist, Deepmind - Unknown - Important Trends in AI How Did We Get Here , What Can We Do Now and How Can We Shape AI ’ s Fut

- Source PDF: `raw/pdf/Dean, Scientist, Deepmind - Unknown - Important Trends in AI How Did We Get Here , What Can We Do Now and How Can We Shape AI ’ s Fut.pdf`
- Source SHA256: `5b543c26ae88a2adba3d009b8fd7aaba18245f5a3360e7b8e23220a013abcc7e`
- Generated from: `scripts/extract_pdf_text.py`

- Extraction: `pymupdf-pages-v2` (sorted text, page anchors; tables/formulas/figures require review)

## Extracted Text

<a id="source-section-0"></a>

<a id="page-1"></a>

### PDF 第 1 页

Important Trends in AI:
How Did We Get Here,
What Can We Do Now and How
Can We Shape AI’s Future?

 Jeff Dean, Chief Scientist, Google Research & Google DeepMind

@jeffdean.bsky.social and @JeffDean
 ai.google/research/people/jeff

Presenting the work of many people at Google and elsewhere
<a id="page-2"></a>

### PDF 第 2 页

Some observations

    In recent years, ML has completely changed our expectations of
   what is possible with computers

    Increasing scale (compute, data, model size) delivers better results

    Algorithmic and model architecture improvements have provided
   massive improvements as well


   The kinds of computations we want to run and the hardware on
   which we run them is changing dramatically
<a id="page-3"></a>

### PDF 第 3 页

Fifteen Years of Machine Learning Advances

                     or

  How Did Today’s Models Come To Be?
<a id="page-4"></a>

### PDF 第 4 页

Key Building Block from Last Century: Neural Networks





                        weights

                                                 weights





Key building block: neural networks, made up of artificial neurons, loosely designed to
                       mimic how real neurons behave
<a id="page-5"></a>

### PDF 第 5 页

Key Building Block from Last Century: Backpropagation





                          weights

                                                   weights
                               Backpropgation of
                                     errors gives an
                                algorithm for how to
                             update the weights of
                            whole neural network
                              based on errors
                                observed at the
                               outputs of the model



Key building block: backpropagation of errors (using chain rule) gives effective algorithm
     for updating the weights of a neural network to minimize errors on training data
<a id="page-6"></a>

### PDF 第 6 页

2012: Scale Matters





   Training a very large neural network (60X bigger than previous largest neural network) using
                      16,000 CPU cores gives major advances in quality
              (~70% relative improvement in ImageNet 22K state-of-the-art)

Le et al., ICML 2012, arxiv.org/abs/1112.6209
<a id="page-7"></a>

### PDF 第 7 页

2012: Distributed Training on Many Computers





                Model parallelism                 Data parallelism

    Combining model parallelism and data parallelism for neural network training across
  thousands of computers enables training of much larger (50-100X) neural networks than
                                   previously possible


Large Scale Distributed Deep Networks, Dean et al., NeurIPS 2012,
research.google.com/archive/large_deep_networks_nips2012.pdf
<a id="page-8"></a>

### PDF 第 8 页

2013: Distributed Representations of Words Are Powerful

   Word2Vec





 Distributed representations of words are powerful:
 (1) Nearby words in high dimensional space are related
      cat, puma, tiger, … are all nearby

 (2) Directions are meaningful
     king – queen ~= man – woman

ICLR 2013 workshop, arxiv.org/abs/1310.4546        Appeared in NeurIPS 2013, arxiv.org/abs/1310.4546
<a id="page-9"></a>

### PDF 第 9 页

2014: Models that Map One Sequence to Another are Powerful

Sequence to Sequence





    Use a neural encoder over an input sequence to generate state, use that to
           initialize state of a neural decoder. Scale up LSTMs and this works.


Appeared in NeurIPS 2014, arxiv.org/abs/1409.3215
<a id="page-10"></a>

### PDF 第 10 页

2015: Specialized Hardware for Neural Network Inference


                                                                        about 1.2                     1.21042
                                      reduced                                                                   × about 0.6                   × 0.61127
                                           precision                     NOT
                                                                        about 0.7                  0.73989343                                       ok



                                    handful of speciﬁc
 Tensor Processing Unit (TPU)                                        operations       ×     =
  v1: 2015, 92 teraops (inference only)

                                        Specialization is much more efficient:
                               Compared to contemporary CPUs & GPUs:
                                TPU v1 is 15X-30X faster
                                TPU v1 is 30X-80X more energy efficient



Appeared in ISCA, 2017, arxiv.org/abs/1704.04760. Now most cited paper in ISCA’s 50 year history
<a id="page-11"></a>

### PDF 第 11 页

2016: Specialized Supercomputers for Neural Network Training





    Connect thousands of chips together (TPU pods) with custom high-speed networks
                            to enable faster neural network training


TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for
Embeddings, Jouppi et al., ISCA 2023, arxiv.org/abs/2304.01433
<a id="page-12"></a>

### PDF 第 12 页

Continual Hardware Performance Scaling





                                                                 11           1126         42522
                                                                   petaflops      petaflops      petaflops


blog.google/products/google-cloud/ironwood-tpu-age-of-inference/
<a id="page-13"></a>

### PDF 第 13 页

Continual Hardware Improvements in Energy Efficiency





                                                             ~30X energy
                                                                               efficiency
                                                              improvement
                                                                              vs. TPU v2





                           Peak FP8 flops delivered per watt of thermal design power per chip package


blog.google/products/google-cloud/ironwood-tpu-age-of-inference/
<a id="page-14"></a>

### PDF 第 14 页

Open source tools enable the whole community





                                                                        pytorch.org
                tensorflow.org





                                         github.com/jax-ml/jax
<a id="page-15"></a>

### PDF 第 15 页

2017: Transformer Model Architecture: Attention





 Don’t try to force state into single recurrent distributed representation.
        Instead, save all past representations and attend to them.



Attention is All You Need, Vaswani et al., NeurIPS 2017, arxiv.org/abs/1706.03762
<a id="page-16"></a>

### PDF 第 16 页

2017: Transformer Model Architecture: Attention





                                                                                                                                      Figure from Scaling Laws for Neural Language Models,
                                                                                                                          Kaplan et al., arxiv.org/abs/2001.08361

 Higher accuracy w/ 10X-100X less compute and 10X smaller models!



Attention is All You Need, Vaswani et al., NeurIPS 2017, arxiv.org/abs/1706.03762
<a id="page-17"></a>

### PDF 第 17 页

2018: Language Modeling At Scale With Self-Supervised Data


   There’s lots of text in the world! Self-supervised learning on this text can
   provide very large amounts of training data with the “right” answer known (“wrong
   guess” is used to provide gradient descent loss training signal)



                                                     Self-supervised learning
                                           on text with large models
                                                               is one of the major
                                                reasons chat/language
                                             models have gotten so
                                           good





Language Models are Few-Shot Learners, Brown et al., NeurIPS, 2020, arxiv.org/abs/2005.14165
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al., ACL 2019, arxiv.org/abs/1810.04805
<a id="page-18"></a>

### PDF 第 18 页

2018: Language Modeling At Scale With Self-Supervised Data


   There’s lots of text in the world! Self-supervised learning on this text can
   provide very large amounts of training data with the “right” answer known (“wrong
   guess” is used to provide gradient descent loss training signal)

   Different kinds of training objectives:
   Autoregressive (look at prefix, predict next word):                                                     Self-supervised learning
          Zürich is ______                           on text with large models
          Zürich is the _______                                   is one of the major
          Zürich is the largest _______                      reasons chat/language
                                             models have gotten so
   Fill-in-the-Blank (e.g. look in both directions, BERT):
                                           good
          Zürich ____ the largest ____ in ______.
          Zürich is the ______ city ____ Switzerland.
       ….

Language Models are Few-Shot Learners, Brown et al., NeurIPS, 2020, arxiv.org/abs/2005.14165
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al., ACL 2019, arxiv.org/abs/1810.04805
<a id="page-19"></a>

### PDF 第 19 页

2021: Transformers for Vision





                                                                                    Visualization of
                                                                                  attention mechanism




An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, Alexey Dosovitskiy et al., ICLR 2021,
arxiv.org/abs/2010.11929
<a id="page-20"></a>

### PDF 第 20 页

2017: Sparse Models (e.g. Mixture of Experts) Outperform
  Dense Models





                                                                                                     (A)

                                                                                             or   (B)




    Give model much larger capacity w/ lots of experts but only activate a few chosen experts per token:
                        (A) ~8X reduction in training compute cost for ~same accuracy, or
                        (B) major accuracy improvements for same training compute cost

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton and Jeff Dean.
ICLR 2017, arxiv.org/abs/1701.06538
<a id="page-21"></a>

### PDF 第 21 页

Continued Research on Sparse Models


Gemini 1.5 Pro/Gemini 2.0/Gemini 2.5 use mixture-of-expert (MoE) architectures, building on a long line
of Google research efforts on sparse models:

 ●   2017: Shazeer et al., Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.
     ICLR 2017. arxiv.org/abs/1701.06538
 ●   2020: Lepikhin et al., GShard: Scaling giant models with conditional computation and automatic sharding.
     ICLR 2020. arxiv.org/abs/2006.16668
 ●   2021: Carlos Riquelme et al., Scaling vision with sparse mixture of experts, NeurIPS 2021.
      arxiv.org/abs/2106.05974
 ●   2021: Fedus et al., Switch transformers: Scaling to trillion parameter models with simple and efficient
      sparsity. JMLR 2022. arxiv.org/abs/2101.03961
 ●   2022: Clark et al., Unified scaling laws for routed language models, ICML 2022. arxiv.org/abs/2202.01169
 ●   2022: Zoph et al., Designing effective sparse expert models. arxiv.org/abs/2202.08906
 ●   2023: Puigcerver et al., From Sparse to Soft Mixtures of Experts. arxiv.org/abs/2308.00951
 ●   2023: Obando-Cero et al., Mixtures of Experts Unlock Parameter Scaling for Deep RL.
      arxiv.org/abs/2402.08609
 ●   2024: Raposo et al., Mixture-of-Depths: Dynamically allocating compute in transformer-based language
     models. arxiv.org/abs/2404.02258
 ●   2024: Douillard et al., DiPaCo: Distributed Path Composition. arxiv.org/abs/2403.10616
<a id="page-22"></a>

### PDF 第 22 页

2018: Software abstractions for Distributed ML Computations

  Example: Pathways





                                                  Region A                                  Region B


                                          Building 1               Building 2                           Building 1


               …





          Scalable software can simplify running large-scale computations

Pathways: Asynchronous Distributed Dataflow for ML, Barham et al., MLSys 2022: arxiv.org/abs/2203.12533
<a id="page-23"></a>

### PDF 第 23 页

2018: Software abstractions for Distributed ML Computations


                      With JAX+Pathways, entire training process
                           Client
                         driven by a single Python process on one host



                                                  Region A                                  Region B


                                          Building 1               Building 2                           Building 1


               …





          Scalable software can simplify running large-scale computations

Pathways: Asynchronous Distributed Dataflow for ML, Barham et al., MLSys 2022: arxiv.org/abs/2203.12533
<a id="page-24"></a>

### PDF 第 24 页

Pathways: Now Available for Cloud Customers





    Pathways: Enables a single JAX client can see and use many devices (e.g. 1 to 100,000
    chips), even though these are distributed across many hosts and even many TPU pods


Pathways: Asynchronous Distributed Dataflow for ML, Barham et al., MLSys 2022: arxiv.org/abs/2203.12533
<a id="page-25"></a>

### PDF 第 25 页

2022: “Thinking longer” at inference time is very useful

“Chain of Thought prompting” is one such technique





Chain of Thought Prompting Elicits Reasoning in Large Language Models, Jason Wei, Xuezhi Wang, Dale
Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou, 2022, arxiv.org/abs/2201.11903
<a id="page-26"></a>

### PDF 第 26 页

2022: “Thinking longer” at inference time is very useful

“Chain of Thought prompting” is one such technique

                                                                                                                                                                                                                                   (%age)
                                                                                                                                                       rate
                                                                                                                                                                                             Solve


                                                                             Model scale
                                                                                                         (billions of parameters)

Prompting model to “show its work” improves accuracy on reasoning tasks
dramatically

Chain of Thought Prompting Elicits Reasoning in Large Language Models, Jason Wei, Xuezhi Wang, Dale
Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou, 2022, arxiv.org/abs/2201.11903
<a id="page-27"></a>

### PDF 第 27 页

2014: Distillation: Use Powerful “Teacher” Models to Make
  Smaller, Cheaper “Student” Models





   “performed the Concerto for “         __?__
   Real next word:                             “Violin”





 Distillation: Use large high quality model as “teacher” when training smaller
 “student” model

Rejected from NeurIPS 2014. Published in workshop & put on Arxiv: arxiv.org/abs/1503.02531. 24,000+ citations.
<a id="page-28"></a>

### PDF 第 28 页

2014: Distillation: Use Powerful “Teacher” Models to Make
  Smaller, Cheaper “Student” Models



                                                        Gives much richer signal for
                                                                     training: try to get student to
                                                    match “soft probability
                                                                   distribution” of large model

   “performed the Concerto for “         __?__
   Real next word:                             “Violin”

   Teacher model says:                       “Violin: 0.4, Piano: 0.2, Trumpet: 0.01, Airplane: 0.00000001”




 Distillation: Use large high quality model as “teacher” when training smaller
 “student” model

Rejected from NeurIPS 2014. Published in workshop & put on Arxiv: arxiv.org/abs/1503.02531. 24,000+ citations.
<a id="page-29"></a>

### PDF 第 29 页

2014: Distillation: Use Powerful “Teacher” Models to Make
  Smaller, Cheaper “Student” Models





   “performed the Concerto for “         __?__
   Real next word:                             “Violin”

   Teacher model says:                       “Violin: 0.4, Piano: 0.2, Trumpet: 0.01, Airplane: 0.00000001”




 Distillation: Use large high quality model as “teacher” when training smaller
 “student” model

Rejected from NeurIPS 2014. Published in workshop & put on Arxiv: arxiv.org/abs/1503.02531. 24,000+ citations.
<a id="page-30"></a>

### PDF 第 30 页

2022: Many Different Parallelism Schemes During Inference





                                          Right choices for how to distribute
                                           inference computation heavily influenced
                                     by things like batch size or latency
                                             constraints





Efficiently Scaling Transformer Inference, Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James
Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, Jeff Dean, arxiv.org/abs/2211.05102
<a id="page-31"></a>

### PDF 第 31 页

2022: Many Different Parallelism Schemes During Inference





Efficiently Scaling Transformer Inference, Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James
Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, Jeff Dean, arxiv.org/abs/2211.05102
<a id="page-32"></a>

### PDF 第 32 页

2023: Speculative Decoding

  Use small “drafter” model to predict next K tokens
    ●  Then predict next K tokens in one shot with large model (more efficient: batch size K not 1)
   ●  Advance generation by as many tokens as match in prefix of size K
   ●  Guaranteed identical output distribution



           Larger, slower model




                                                            vs

           Larger, slower model


          Faster model (drafter)




Fast Inference from Transformers via Speculative Decoding, Yaniv Leviathan, Matan Kalman & Yossi Matias,
ICML ‘23, arxiv.org/abs/2211.17192
<a id="page-33"></a>

### PDF 第 33 页

Innovations at Many Levels



                                                                                       Inference-time
                              Chain-of-Thought       Speculative DecodingInference algorithms                                                     compute scaling


                                Unsupervised and       Asynchronous
                                                                                              Distillation  SFT + RLxFTraining algorithms           Self-Supervised Learning       Training


Model architecture         Word2Vec     Seq2Seq     Transformers  MoEs      Visual Transformers



                                                                             PathwaysSoftware abstractions        DistBelief



Hardware                TPUv1 → TPUv2 → TPUv3 → TPUv4 → TPUv5p → Trillium → Ironwood
<a id="page-34"></a>

### PDF 第 34 页

Gemini:

Putting These Advances Together
<a id="page-35"></a>

### PDF 第 35 页

Project started in Feb 2023
   Many collaborators from Google DeepMind, Google Research, and rest of Google

    Goal: Train the world’s best multimodal models and use them all across Google
    Gemini 1.0: Dec 2023
    Gemini 1.5: Feb 2024 (demonstrated 10M token context window, Flash model)
    Gemini 2.0: Dec 2024 (2.0 Flash as good as 1.5 Pro, multimodal live streaming, …)
    Gemini 2.0 Thinking: Jan 2025 (2.0 Flash Experimental Thinking)
    Gemini 2.5: Mar 2025 (2.5 Pro released), Apr 2025 (“2.5 Flash coming soon”)

https://blog.google/technology/ai/google-gemini-ai                                               https://g.co/gemini

Gemini: A Family of Highly Capable Multimodal Models, by the Gemini Team, arxiv.org/abs/2312.11805
Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, by the Gemini Team, arxiv.org/abs/2403.05530
<a id="page-36"></a>

### PDF 第 36 页

Gemini - Multimodal from the start




   Gemini - multimodal from the start





Gemini: A Family of Highly Capable Multimodal Models, by the Gemini Team, arxiv.org/abs/2312.11805
<a id="page-37"></a>

### PDF 第 37 页

Gemini 1.5

   Increased context length
    Models can now handle up to 10 million
    tokens, with external APIs now offering up
     to 2 million tokens for text and/or video.

   Clearer context
    The information within the context window
      is clearer, reducing hallucinations &
    enabling in-context learning.





Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, by the Gemini Team, arxiv.org/abs/2403.05530
<a id="page-38"></a>

### PDF 第 38 页

Gemini 2.0

 (Like 1.0 and 1.5 and 2.5) Builds on many of
 the innovations I just described:

  ●  TPUs
  ●  Cross-datacenter training
  ●  Pathways
  ●  JAX
  ●   Distributed representations of words
  ●  Transformers
  ●  Sparse Mixture of Experts
  ●   Distillation
  ●  + … many more innovations …





blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024
<a id="page-39"></a>

### PDF 第 39 页

Gemini 2.5 Pro
Our most capable model (for now!)





blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/
<a id="page-40"></a>

### PDF 第 40 页

Gemini 2.5 Pro
 Our most capable model (for now!)                      Leaderboard positions
                                                     ●  #1 LMSYS
                                                     ●  # LiveBench
                                                     ●  #1 Humanity’s Last Exam
                                                     ●  #1 SEAL
                                                     ●  #1 Artificial Analysis
                                                     ●  #1 Aider Polyglot
                                                     ●  #1 MathArena.ai
                                                     ●  #1 Mensa IQ test
                                                     ●  #1 Fiction.LiveBench
                                                     ●  #1 SimpleBench
                                                     ●  #1 Kagi leaderboard
                                                     ●  #2 WebDev Arena
                                                     ●  #4 LiveCodeBench
                                                     ●  #4 NYT Connections
                                                     ●  #2 Creative Writing
                                                     ●  #4 Vectara
                                                     ●  # 1 Perfect Information Game



blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/
<a id="page-41"></a>

### PDF 第 41 页

Users Generally Enjoying Capabilities of Gemini 2.5 Pro
<a id="page-42"></a>

### PDF 第 42 页

Long context abilities are very helpful (especially for code)
<a id="page-43"></a>

### PDF 第 43 页

Pushing the Pareto Frontier of Optimal Quality/Price
<a id="page-44"></a>

### PDF 第 44 页

Organizing a Large-Scale Scientific Effort
              Like Gemini
<a id="page-45"></a>

### PDF 第 45 页

Many Contributors in Many Different Areas
<a id="page-46"></a>

### PDF 第 46 页

Many Contributors in Many Different Areas
<a id="page-47"></a>

### PDF 第 47 页

Gemini Structure & Ways of Working

Overall Leads        Program Management    Product Management

Model Development Areas          Capabilities

          Pre-training                            Safety                Code

         Post-training                           Vision                 Agents

      On-device Models                      Audio               Internationalization
…                 …

Core Areas

           Data                            Evals

         Infrastructure                  Codebase

           Serving                  Longer-term Research
…
<a id="page-48"></a>

### PDF 第 48 页

Gemini Structure & Ways of Working

 Many people in many locations:
 ~⅓ in San Francisco Bay Area
 ~⅓ in London
 ~⅓ in many other places:
     NYC, Paris, Boston, Zürich, Bangalore, Tel Aviv, Seattle, …



 Time zones are annoying!
  ●  “Golden Hours” between California/West Coast and London/Europe
      are important
<a id="page-49"></a>

### PDF 第 49 页

Gemini Structure & Ways of Working

 Lots and lots of large and small discussions and information sharing conducted via
 Google Chat Spaces (I’m in 200+ such spaces)

 RFCs (Request for Comment): semi-formal way of getting feedback, knowing what
 others are working on, etc.

 Leaderboards and common baselines enable data-driven decision making about how
 to improve
  ●  Multiple rounds of experimentation.
  ●  Many experiments at small scale
  ●  Advance smaller number of successful experiments to next scale
  ●  Every so often (every few weeks), incorporate successful experiments
     demonstrated at largest experimental scale into new candidate baseline
  ●  Repeat
<a id="page-50"></a>

### PDF 第 50 页

Training at Scale:
   Silent Data Corruption errors (SDCs)

Despite best efforts, given the scale of ML systems
 and the size of ML training jobs, hardware errors
can occur, and sometimes incorrect computations
  from one buggy chip can spread and infect the
                entire training system
<a id="page-51"></a>

### PDF 第 51 页

Silent data corruption




           Non-deterministically produce incorrect
            results, silently


          Challenging problem when running largely
          independent computation


            Multiplicatively worse at scale with
          synchronous stochastic gradient descent


        Can quickly spread results across
          thousands of components across ML
         supercomputer




Cores that Don't Count, Peter H. Hochschild, Paul Jack Turner, Jeffrey C. Mogul, Rama Krishna Govindaraju, Parthasarathy
Ranganathan, David E Culler, Amin Vahdat, HotOS 2021, research.google/pubs/cores-that-dont-count/
<a id="page-52"></a>

### PDF 第 52 页

Metrics anomaly: anomaly due to SDC





                            Anomaly due to SDC

Norm
Gradient





                                              Time
<a id="page-53"></a>

### PDF 第 53 页

Metrics anomaly: expected anomaly (no SDC)





                     Anomaly with NO SDC

Norm
Gradient





                                              Time
<a id="page-54"></a>

### PDF 第 54 页

SDC with no metrics anomaly




Norm
Gradient
                          SDC detected with NO anomaly
                                        The step replay shows different values,
                                           but both values are in the normal range.

                                              Time
<a id="page-55"></a>

### PDF 第 55 页

ML Controller transparently handles Silent Data Corruption
(SDC)





            Synchronous training worker        SDC checker         Hot spare





                                       Defective machine           SDC checker             SDC Checker
         Normal training                                      causes SDC                   automatically             moves training to
                state                                                                                   identifies SDC                hot spare and
                                                                                          sends defective
                                                                                       machine for repair
<a id="page-56"></a>

### PDF 第 56 页

What Can These Models Do?
<a id="page-57"></a>

### PDF 第 57 页

[本页未提取到文本；需要图像检查或 OCR。]
<a id="page-58"></a>

### PDF 第 58 页

Example
       In-context learning: Kalamang translation





First part of
  chapter 1
<a id="page-59"></a>

### PDF 第 59 页

Example
       In-context learning: Kalamang translation
Kalamang is only spoken by ~130 people in eastern Indonesian Papua
<a id="page-60"></a>

### PDF 第 60 页

In-context learning: Kalamang translation





      With in-context info, model can translate as effectively as a human learner
           who has spent months on the same language materials
<a id="page-61"></a>

### PDF 第 61 页

Example

Video of bookshelf
 -> JSON
<a id="page-62"></a>

### PDF 第 62 页

“The killer app of
Gemini 1.5 Pro is video.”

    Simon Willison


            …
<a id="page-63"></a>

### PDF 第 63 页

Example

Video understanding
& summarization
<a id="page-64"></a>

### PDF 第 64 页

In a table, please write
the sport, the
teams/athletes involved,
the year and a short
description of why each
of these moments in
sports are so iconic.
<a id="page-65"></a>

### PDF 第 65 页

[本页未提取到文本；需要图像检查或 OCR。]
<a id="page-66"></a>

### PDF 第 66 页

Example         Digitization of historical data





https://climatelabbook.substack.com/p/data-rescue-with-ai
<a id="page-67"></a>

### PDF 第 67 页

Gemini 2.5 Pro example:
Code Generation via High Level Language
<a id="page-68"></a>

### PDF 第 68 页

[本页未提取到文本；需要图像检查或 OCR。]
<a id="page-69"></a>

### PDF 第 69 页

[本页未提取到文本；需要图像检查或 OCR。]
<a id="page-70"></a>

### PDF 第 70 页

Inference time compute gives us another
dimension of compute for quality scaling
<a id="page-71"></a>

### PDF 第 71 页

deepmind.google/technologies/gemini/flash-thinking/
<a id="page-72"></a>

### PDF 第 72 页

deepmind.google/technologies/gemini/flash-thinking/
<a id="page-73"></a>

### PDF 第 73 页

Now That We Have These Powerful
  Models, What Will This Mean?
<a id="page-74"></a>

### PDF 第 74 页

Shaping AI's Impact on Billions of Lives




●  Form team of senior computer scientists + rising stars in AI
     ○   From academia, big tech and startups
●  Propose what impact could be given directed research &
    policy efforts on AI for public good
                                                                                                                       Mariano-Florentino Cuéllar    Jeff Dean       John Hennessy
     ○   Rather than predict societal impact of AI given a laissez faire approach
●  Aim to shape AI’s upsides and dampen AI’s downsides
     ○   For high, middle, and low income nations
●  Audience: AI practitioners + policymakers + public
●  Approach: Interview 24 experts in 7 ﬁelds
     ○   Employment, Education, Healthcare, Information, Media, Governance,
                                                                                                                                  Finale Doshi-Velez    Andy Konwinski      Sanmi Koyejo
          and Science
     ○    e.g. Barack Obama, Sal Khan, John Jumper, Neal Stephenson, Dario
          Amodei, Bob Wachter, …
●  Uncovered 5 guidelines for AI for public good


                                                                                                                      74
                                                                                                              Pelonomi Moiloa    Emma Pierson      David Patterson
<a id="page-75"></a>

### PDF 第 75 页

Shaping AI's Impact on Billions of Lives





                                                                                                                          Mariano-Florentino Cuéllar    Jeff Dean       John Hennessy





                                                                                                                                    Finale Doshi-Velez    Andy Konwinski      Sanmi Koyejo




  “Shaping AI's Impact on Billions of Lives,” by Mariano-Florentino (Tino) Cuéllar, Jeff Dean, Finale
Doshi-Velez, John Hennessy, Andy Konwinski, Sanmi Koyejo, Pelonomi Moiloa, Emma Pierson, and
                             David Patterson, December, 2024                                                                    75
                  See ShapingAI.com and arxiv.org/abs/2412.02730                                 Pelonomi Moiloa    Emma Pierson      David Patterson
<a id="page-76"></a>

### PDF 第 76 页

Humans and AI systems working as a team can
  do more than either on their own



    ●  AI focused on human productivity produce
       more positive beneﬁts than those focused
       on human labor replacement
          ○   Increases human employability
          ○   Bonus: People can also be safeguards if AI veers
                   off course in areas not well trained
          ○   Bonus: People and AIs tend to make different
                 mistakes, so collaboration of experts with AI can
                 also improve results
    ●  Productivity focus helps both AI and people
        succeed



Shaping AI's Impact on Billions of Lives, see ShapingAI.com and arxiv.org/abs/2412.02730
<a id="page-77"></a>

### PDF 第 77 页

To increase employment, aim for productivity
  improvements in ﬁelds that create more jobs



     ●  Despite tremendous productivity gains in computing and
         passenger jets, the US in 2020 had 8 times more
         commercial airline pilots and 11 times more
         programmers than in 1970

     ●  Demand for passenger travel and programming was
            elastic ⇒ more jobs
            ○   Goods with elastic demand are those where a decrease in price
                     results in a large increase in the quantity acquired

     ●  US agriculture demand is inelastic, so productivity gains
     ⇒ fewer jobs
            ○   From 20% of US workforce to 2% in one lifetime (1940 to 2020)


Shaping AI's Impact on Billions of Lives, see ShapingAI.com and arxiv.org/abs/2412.02730
<a id="page-78"></a>

### PDF 第 78 页

What could be impact in next 5 years of near
  term AI by following the guidelines?



   ●  To give concrete targets for improving AI’s
       impact, propose
       milestoneskilometerstones per ﬁeld


   ●  Rather than recognize past achievements,
        offer signiﬁcant inducement prizes that try
       to stimulate progress on these milestones
        ○   E.g., XPRIZE, Netﬂix, Kaggle, …




Shaping AI's Impact on Billions of Lives, see ShapingAI.com and arxiv.org/abs/2412.02730
<a id="page-79"></a>

### PDF 第 79 页

Education AI Milestone: Worldwide Tutor



   ●  A tutoring tool to accelerate general education
         for every child
         ○   In their language
         ○   In their culture
         ○   In their best learning style
   ●  To help teachers with challenge of supporting a
       range of student capability
         ○   Keeping high-achieving students engaged while
                supporting those who struggle
   ●   E.g., Rising Academies* in Africa
         ○   Improves student outcomes by one grade level relative
                 to students without it


* Henkel, Owen, Hannah Horne-Robinson, Nessie Kozhakhmetova, and Amanda Lee. “Eﬀective and Scalable Math Support: Experimental Evidence on the                                                                                                                         79
Impact of an AI-Math Tutor in Ghana.” In International Conference on Artiﬁcial Intelligence in Education, pp. 373-381. Cham: Springer Nature Switzerland, 2024.
<a id="page-80"></a>

### PDF 第 80 页

Healthcare AI Milestone: Broad Medical AI



     ●  Learns from many data modalities
            ○   Images, laboratory results, health records, genomics,
                  medical research, …
     ●  Can help carry out diverse set of tasks
            ○   Bedside decision support
            ○    Interacting with patients after leaving hospital
            ○   Drafting radiology reports that describe both abnormalities
                 and relevant normal ﬁndings
                 ■   While taking into account the patient’s history
     ●  Can explain recommendations using written or
         spoken text and images
     ●  Milestone requires deﬁning metrics and benchmarks
           to measure progress




Shaping AI's Impact on Billions of Lives, see ShapingAI.com and arxiv.org/abs/2412.02730
<a id="page-81"></a>

### PDF 第 81 页

Information AI Milestone:
  Civic Discourse Platform



    ●  Mediates conversations or attitudes to enhance
         public understanding and civic discourse
          ○  Move communities from polarization to pluralism
    ●   AI system makes suggestions on how to rephrase
       comments and questions more diplomatically*
    ●   AI system to hold discussions with conspiracy
         theorists**
    ●   AI systems could help bring consensus on diﬃcult
        issues across whole populations***

* Argyle, Lisa, et al. “Leveraging AI for democratic discourse: Chat interventions can improve online political conversations at scale.” Proc. National Academy of
Sciences, vol. 120, no. 41, 2023.
** Costello, Thomas, Gordon Pennycook, and David Rand. “Durably reducing conspiracy beliefs through dialogues with AI.” Science, vol. 385, no. 6714, 2024, p.
Eadq1814.
*** Tsai, Lily and Alex Pentland. “Rediscovering the Pleasures of Pluralism: The Potential of Digitally Mediated Civic Participation,” The Digitalist Papers, 2024.
<a id="page-82"></a>

### PDF 第 82 页

Science



    ●  Advances in science via AI could be one of
          largest impacts for public good
    ●  Many examples:
           ○   AlphaFold for protein folding
           ○   Black hole visualization
           ○   Flood forecasting
           ○   Materials discovery
           ○   Neural net-based weather prediction
           ○   Airplane contrail reduction to reduce CO2e
           ○   Controlling plasma for nuclear fusion
           ○  …
    ●  Most ﬁelds of science excited about AI




Shaping AI's Impact on Billions of Lives, see ShapingAI.com and arxiv.org/abs/2412.02730
<a id="page-83"></a>

### PDF 第 83 页

Science AI Milestone:
   Scientist’s AI Aide/Collaborator



    ●  Accelerate pace of science by improving the
          productivity of scientists
           ○   Help suggest interesting hypotheses and automate
                 experiments
           ○   Identify important new relevant research, ideally
                customized to individual to summarize what is new
               compared to what the scientist already knew

     Early example: Google’s Co-Scientist work*
     ●   Multi-agent scientiﬁc discovery system, showing
           inference time compute scaling leads to better rated
          hypotheses

* research.google/blog/accelerating-scientiﬁc-breakthroughs-with-an-ai-co-scientist/

Shaping AI's Impact on Billions of Lives, see ShapingAI.com and arxiv.org/abs/2412.02730
<a id="page-84"></a>

### PDF 第 84 页

Conclusions


●  AI models and products are becoming incredibly
   powerful and useful tools
     ○   Further research and innovation will continue this trend

●  Will have dramatic impact in many diverse areas:
   ○  Healthcare, education, scientiﬁc research, media
         creation, misinformation, …

●  Potentially makes deep expertise more available to
   many more people

●  Done well, our AI-assisted future is bright!

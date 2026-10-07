# Seedance 2.0: Advancing Video Generation for World Complexity

- Source PDF: `raw/pdf/Team Seedance et al. - 2026 - Seedance 2.0 Advancing Video Generation for World Complexity.pdf`
- Source SHA256: `b8a169f36db23966158c7db305986fe89b2f3d945e45987290942842daedcac9`
- Source URL: https://arxiv.org/abs/2604.14148
- Generated from: `scripts/extract_pdf_text.py`

- Extraction: `pymupdf-pages-v2` (sorted text, page anchors; tables/formulas/figures require review)

## Extracted Text

<a id="source-section-0"></a>

<a id="page-1"></a>

### PDF 第 1 页

Seedance 2.0:
        Advancing Video Generation for World Complexity


                                  ByteDance Seed

             1  Introduction

            Video generation models have become a core technology for modern digital content infrastructure and generative
           AI ecosystems, with rapid, widespread adoption across professional content production and consumer-facing
              creative scenarios. Through years of iterative research and engineering optimization, the ByteDance Seed2026
           team has built full-stack generative media technologies — including previous Seedance video generation
                series [6, 15, 16, 23], Seedream image generation and editing series [5, 7, 13, 17, 19, 20], Seed-VL multimodal
             vision-language models [3, 4, 9] for cross-modal semantic understanding, and other key components [21, 22],Apr
            which are now widely integrated into our large-scale product ecosystem, supporting video generation services
               for billion-level daily active users.15
             In this work, we push the frontier of video generation technologies with a notable paradigm shift: from
             generating short video clips with limited controllability to robust, highly controllable video synthesis natively
             supporting diverse control signals. This industry-wide trend has driven the development of Seedance 2.0, our
          new model designed specifically to deliver enhanced generation quality with rich multi-modal controllability
               for large-scale creative engine platforms.[cs.CV]
            Seedance 2.0 is a new native multi-modal audio-video generation model, officially released in China in
              early February 2026. Compared with its predecessors, Seedance 1.0 and 1.5 Pro [6, 16], Seedance 2.0 adopts
            a unified, highly efficient, and large-scale architecture for multi-modal audio-video joint generation. This
             allows it to support four input modalities: text, image, audio, and video, by integrating one of the most
            comprehensive suites of multi-modal content reference and editing capabilities available in the industry to
              date. It delivers substantial, well-rounded improvements across all key sub-dimensions of video and audio
              generation. In both expert evaluations and public user tests, the model has demonstrated performance on par
            with the leading levels in the field.

           Within its multi-modal framework, Seedance 2.0 is equipped with a full set of multi-modal reference and
              editing capabilities, supporting both standalone and combinatorial tasks, including subject control, motion
             manipulation, style transfer, special effects design and creative content generation, as well as video extension.
             This suite of capabilities renders the model applicable to a diverse array of creative production scenarios. The
           model achieves more accurate compliance with complex instructions and more stable motion performance,
             enabling it to accommodate more sophisticated production workflows. Coupled with its native, professionalarXiv:2604.14148v1       multi-shot narrative capability, vivid details of motion and facial expressions, and improved cross-frame
              consistency, the model reaches a competitive usability rate in real-world production settings.

            Seedance 2.0 supports direct generation of audio-video content with durations ranging from 4 to 15 seconds,
            with native output resolutions of 480p and 720p. For multi-modal inputs as reference, its current open
             platform supports up to 3 video clips, 9 images, and 3 audio clips. In addition, we provide Seedance 2.0 Fast
              version, an accelerated variant of Seedance 2.0 designed to boost generation speed for low-latency scenarios.
            Seedance 2.0 has delivered significant improvements to its foundational generation capabilities and multi-modal
             generation performance, bringing an enhanced creative experience for end users. Its key model capabilities
              are highlighted as follows.

             • Generation of Real-world Complexity. Seedance 2.0 achieves remarkable improvements in generation
                   quality, particularly in human motion modeling, which delivers significantly enhanced naturalness, temporal
                coherence, and physical plausibility compared to previous versions. It can synthesize temporally precise,



                                                           1
<a id="page-2"></a>

### PDF 第 2 页

complex interaction scenes with high fidelity, while adhering to real-world motion laws throughout the
   generation process, thereby mitigating the artifacts common in recent video generation models. For detailed
   close-up shots, the generated frames exhibit highly realistic details and rigorous consistency—whether
    for subtle changes in light refraction or natural, fluid interactions between characters and the environ-
   ment—closely matching the visual fidelity of real-world live-action footage. With robust motion stability
   and physics compliance, the model delivers favorable performance in multi-subject interaction and complex
   motion scenarios, achieving a usability rate that is clearly higher than recent commercial models.

 • Strong multimodal capability. First, Seedance 2.0 accepts a comprehensive multimodal reference input,
   allowing you to combine text, image, video, and audio sources. The model can accurately interpret
   multimodal input content and generate output referencing user-specified elements that include frame
   composition, cinematographic design, motion rhythm, and acoustic characteristics in accordance with user
    instructions. It can also directly reference text-based storyboards, enhancing the flexibility of conventional
   video generation workflows and expanding the creative freedom. Second, Seedance 2.0 features substantially
   improved controllability in video generation. It delivers strong instruction-following performance, accurately
   generating specified content, and maintaining consistent subject identity preservation even when processing
   complex scripts with extensive character interactions and fine-grained action descriptions. Meanwhile,
   the model exhibits fundamental directorial and cinematographic reasoning capabilities, enabling it to
   autonomously plan shot sequencing and design visual presentation templates. In addition, Seedance 2.0
   introduces new video editing capabilities, which enable targeted modifications to specified clips, characters,
   actions, or plot elements. It also provides video continuation functionality, which generates consecutive
   shots aligned with user prompts, supporting both de novo video generation and seamless extension of
    existing footage.

 • High-Fidelity Audio-Video Generation. Seedance 2.0 has binaural audio capability with synchronized high-
    fidelity immersive sound generation. It is equipped with an upgraded audio generation module integrated
   with binaural audio technology, which enables high-fidelity, immersive sound generation. The model
   supports simultaneous multi-track output of audio content including background audio, ambient sound
    effects, and character narration, with precise temporal alignment to the visual rhythm of generated footage.
  The audio content generated by the model features highly natural sound design, faithfully reproducing
   subtle natural ambient sounds to enhance scene immersion. Coupled with strict audio-visual temporal
    control, the model ensures tight synchronization between audio tracks and visual actions, providing robust
   support for professional-grade audio-visual content creation.

 • Applications in Productivity Scenarios.  It exhibits strong cross-scene adaptability, which reduces the
    barriers to professional content production. In response to the diverse demands of video content production,
   Seedance 2.0 demonstrates high cross-scene adaptability. It delivers high-quality generation results across
   a wide range of use cases, including commercial advertising, cinematic and television visual effects, game
   animation, and commentary videos. By replacing complex visual effects production and live-action shooting
   workflows with AI generation, Seedance 2.0 can significantly reduce production costs and shorten the
   production cycle of professional audio-video content, helping creators and enterprises realize their creative
   concepts.

From the audio-video synchronous generation achieved by Seedance 1.5 [16] to the unified multimodal audio-
video joint generation framework established by Seedance 2.0, the Seedance series has consistently been built
around a unified architecture, with a core commitment to high-fidelity reconstruction of real-world complexity.

We acknowledge that Seedance 2.0 remains imperfect, with room for improvement in its generation outputs.
Moving forward, we will continue to explore deep alignment between generative models and the physical
world, advance accurate modeling of real-world dynamics, deepen our understanding of physical and semantic
rules, and enable the technology to better serve every creator.

Safety is a core consideration in our work. Throughout the model iteration lifecycle, we have implemented a
structured safety assessment framework and made continuous efforts to evaluate and mitigate potential risks,
with the aim of supporting responsible, compliant, and ethically aligned development.




                                               2
<a id="page-3"></a>

### PDF 第 3 页

We invite readers to explore the capabilities of Seedance 2.0.
     Seedance 2.0 is now accessible on Doubao1, Jimeng2 and Volcano Engine, under the model id:
     doubao-seedance-2-0-260128.
    The model can be accessed at https://www.volcengine.com/experience/ark?mode=vision&
     modelId=doubao-seedance-2-0-260128&tab=GenVideo.
    More details are available on the official page: https://seed.bytedance.com/seedance2_0.


2  Evaluation

2.1  Overview

To objectively and comprehensively assess the overall capabilities of Seedance 2.0 in multi-modal scenarios, our
team collaborated with experts from the media industry to establish a comprehensive evaluation benchmark
and corresponding evaluation protocols. The benchmark covers audio-video generation, reference-based
generation, and video editing scenarios. This evaluation focuses on the model’s performance across core
dimensions: multi-modal reference-based generation, complex audio-video instruction following, complex
motion stability, professional cinematographic language expression, audio and video expressiveness, and
audio-visual integrated alignment. As shown in Figure 1, Seedance 2.0 achieves the highest scores across all
evaluated dimensions in Text-to-Video (T2V), Image-to-Video (I2V), and Reference-to-Video (R2V) tasks,
demonstrating comprehensive leading performance over current competing models.


                    Text-to-Video Evaluation                                     lmage-to-Video Evaluation                                    Multimodal Task Evaluation
                               Motion Quality                                                               Motion Quality                                                       Prompt Following




    Audio Prompt                                         Video Prompt          Audio Prompt                                         Video Prompt
      Following                                                 Following               Following                                                 Following

                                                                                                                                                      Reference                                                 Multimodal
                                                                                                                                                     Alignment                                            Task Following





     Audio-Visual
        Sync                                                 Aesthetic               Audio-VisualSync                                                 PreservationImage



                                                                                                                                                                          Editing Consistency                               Motion Quality
                           Audio Expressiveness                                                        Audio Expressiveness

  Seedance 2.0   Seedance 1.5 pro   Sora 2 Pro   Veo 3.1    Kling 3.0   Kling 2.6    Seedance 2.0   Seedance 1.5 pro   Wan 2.6   Veo 3.1    Kling 3.0    Kling 2.6    Seedance 2.0            Vidu Q2 Pro             Kling 3.0              Kling O1

Figure 1 Overall performance comparison across T2V, I2V, and R2V tasks. Seedance 2.0 achieves comprehensive
leading performance over all competing models across every evaluated dimension in all three generation tasks.


Text-to-Video and Image-to-Video Evaluation. In video generation tasks, Seedance 2.0 delivers competitive
leading performance in the industry. Marked improvements have been achieved in motion stability, instruction
following capability, and visual aesthetics. The model effectively mitigates common structural inaccuracies
and visual artifacts, generating smooth and nuanced complex motions. It can accurately render high-tension
large-scale movements and subtle micro-expressions, while supporting professional-level combined camera
movements and narrative rhythm control. For long scripts and open-ended instructions, the model can
respond appropriately and deliver reasonable generation outputs. Meanwhile, the generated videos exhibit
notable cinematic aesthetics, with well-rendered object textures, lighting and composition, as well as costume,
makeup and prop design. In the audio domain, Seedance 2.0 maintains strong performance with substantial
improvements in audio expressiveness. Its dual-channel audio output presents rich and nuanced layers, and
can generate sound effects or melodies that align well with the context described in the prompts. Compared
with the previous version, the model delivers an enhanced audio-visual integrated experience, with tighter
alignment between dialogue lines, sound effects, background audio, and visual content. Meanwhile, the

   1https://www.doubao.com/chat/create-video
   2https://jimeng.jianying.com/ai-tool/video/generate



                                               3
<a id="page-4"></a>

### PDF 第 4 页

instruction following accuracy for Chinese dialects, traditional opera, and singing scenarios is significantly
improved.

Multi-modal Reference-Based Generation Evaluation. Seedance 2.0 achieves competitive leading comprehensive
performance in reference-based generation tasks. The model supports a more comprehensive range of reference-
based tasks, covering multiple creative scenarios including multi-modal reference-based generation, video
editing, and video continuation. Meanwhile, it demonstrates advantages in the depth of understanding and
response accuracy for reference content.  Specifically, in video editing tasks, Seedance 2.0 delivers more
complete instruction following and more photorealistic visual outputs compared with peer models. In terms of
generation consistency, the model achieves favorable performance in subject appearance and voice restoration,
with particularly notable advantages in maintaining reference consistency for action logic, special effect styles,
and plot narrative. Despite these strengths, there is still room for optimization in multi-subject consistency,
text restoration accuracy, and the performance of complex editing tasks.

We evaluate Seedance 2.0 on three generation tasks—T2V (Text-to-Video), I2V (Image-to-Video), and R2V
(Reference-to-Video)—against current commercial video generation models. Seedance 2.0 ranks first across all
video and audio dimensions on every task.

On video, the main advances are: (1) Stability—fewer deformations and structural issues; complex actions are
fluid; multilingual text generation and preservation are relatively strong. (2) Vividness—sports, combat, and
other high-amplitude actions carry strong momentum; facial expressions and gaze are emotionally engaging; the
model produces professional-level camera movements, dynamic editing, and narrative pacing. (3) Instruction
following—long-script prompts are executed with reasonable precision, multi-shot and multi-angle instructions
are followed accurately, and open-ended prompts receive appropriate creative interpretation; multiple art
styles are supported; in I2V, special art styles from the reference image are preserved while adapting subject
motion to match. (4) Visual realism—object materials look authentic, lighting, composition, and character
texture improve notably, and costume and prop design is polished.

On audio:  (1) Expressiveness—audio is detailed and layered, with dual-channel support; melodies and
tones match the prompt context, and the audio dimension adds to the emotional impact of the video. (2)
Audio-visual sync—lip movements match the visual, dialogue, sound effects, and background audio align well,
and beat-matching between audio and video is strong. (3) Audio instruction following—Chinese dialects
(Sichuan, Northeastern, Cantonese), opera, and singing improve markedly over the previous version; singing,
rap, and instrumental performance are consistent, with melodies adapted to the prompt context.

Areas for improvement remain: minor deformation artifacts, motion plausibility in edge cases, high-frequency
visual noise, audio distortion and noise, and lip-sync errors in multi-speaker scenes. The following subsections
present detailed results for each task.

2.2  Evaluation Framework

To prepare the model for production deployment, we upgraded our evaluation framework to SeedVideoBench
2.0. The new version adds multimodal generation, narrative quality, and multilingual coverage to the evaluation
scope, and refines how audio expressiveness is assessed. We also brought in expert evaluators from advertising
and game production to provide subjective ratings, with a focus on narrative and aesthetic quality.

2.2.1  SeedVideoBench 2.0

Two main changes define the new framework.  First, a multimodal task evaluation system that formally
defines multimodal task following and generation consistency, while also covering baseline generation quality
(prompt following, motion quality) in multimodal settings. Second, we split evaluation into objective and
subjective tracks. Objective metrics like motion stability—use automated pipelines. Subjective metrics like
aesthetics—go through blind expert review.  Separately, we ran a realism study: evaluators tried to tell
Seedance 2.0 outputs apart from real video clips. The results fed back into our aesthetic tuning process.

The multimodal task evaluation module and the narrative assessment module saw the largest changes from the
previous version. Multimodal Task Following measures instruction-following accuracy across reference, editing,



                                               4
<a id="page-5"></a>

### PDF 第 5 页

and extension scenarios, broken into dozens of fine-grained task types (subject identity, motion, style, etc.).
Most models have limited multimodal coverage, which forces users to probe capability boundaries through
trial and error; these metrics make the boundaries explicit. Specifically, the evaluation covers four task groups:

 • Reference tasks: subject, motion, visual-effects, and style reference generation.

 • Editing tasks: subject, style, scene, and audio content editing.

 • Extension tasks: plot continuation and seamless extension, both forward and backward on the timeline.

 • Combination tasks: paired evaluations that match real workflows—e.g., swapping a video subject with a
   reference image (reference + editing combined).

Consistency captures how closely generated content matches the reference input (reference alignment) and
how well non-edited regions survive during editing (editing consistency). We built specialized datasets covering
subject, motion, scene, style, and audio, with sample distributions tuned to minimize variance at small
evaluation budgets.

On the video metrics side, SeedVideoBench 1.5 already tracked vividness; version 2.0 adds finer narrative
quality metrics alongside the existing vividness and aesthetics dimensions. Unlike motion quality, which can
be measured more objectively, narrative quality is inherently subjective—it asks whether the overall narrative
reads as coherent, whether character performances and visual effects carry emotional weight, and whether the
aesthetic choices fit the content. We evaluate it along three sub-dimensions:

 • Cinematographic language: does the camerawork support the story? We assess shot logic and expressiveness,
   looking for problems like redundant coverage, axis-crossing (180-degree rule violations), mismatched shot
    sizes, and uneven pacing.

 • Plot design: can the model take a vague or brief prompt and produce something both coherent and
   engaging?

 • Stylistic aesthetics: do the visuals have a considered look? This covers lighting, framing, composition, and
   color grading, plus whether characters, costumes, props, and sets hold together aesthetically.

Unless otherwise noted, all evaluation results reported in the following sections are obtained using Seed-
VideoBench 2.0.





                                               5
<a id="page-6"></a>

### PDF 第 6 页

2.2.2  Arena.AI Results





                                     (a) Text-to-Video generation leaderboard.





                                     (b) Image-to-Video generation leaderboard.


                 Figure 2 Leaderboards on Arena.AI (Accessed: April 8, 2026, Eastern Time).


Arena (formerly LMArena) [2], created by researchers from UC Berkeley, is a community-powered platform
that evaluates AI models through real-world user preferences. For video generation, users are presented with
outputs from two anonymous models side-by-side and vote for the one they prefer, producing an Elo-style
leaderboard that reflects genuine human judgment at scale. Unlike automated benchmarks that rely on
metrics such as FVD or CLIPScore, Arena captures holistic human preferences encompassing visual quality,
motion realism, temporal coherence, and prompt adherence in a single unified ranking.

As shown in Figure 2, our Dreamina Seedance 2.0 720p ranks #1 on both the Text-to-Video and Image-to-
Video leaderboards, with Elo scores of 1450 (±15) and 1449 (±11) respectively. On T2V (Figure 2a), it leads
the second-place veo-3.1-audio-1080p by 79 points; on I2V (Figure 2b), it leads grok-imagine-video-720p by
29 points. Notably, the model achieves this at 720p resolution, outperforming competitors that operate at
1080p, which suggests that our improvements in motion dynamics and visual coherence are more perceptually
significant than resolution alone. The Rank Spread of 1↔1 on both leaderboards indicates consistently
top-ranked performance across evaluation dimensions. These results complement our SeedVideoBench2.0
findings, demonstrating that the gains observed in objective metrics translate directly into stronger human
preference.



                                               6
<a id="page-7"></a>

### PDF 第 7 页

2.3  Text-to-Video Evaluation on SeedVideoBench 2.0

2.3.1  Overall Results
               Table 1 T2V overall evaluation results across six dimensions (Rating from 1–5).
                             Video Prompt              Audio   Audio-Visual  Audio Prompt    Model            Motion                  Aesthetics                                  Following                 Quality      Sync        Following

    Kling 2.6 [10]        2.72         2.39           3.21        2.46         2.67            2.00
    Kling 3.0 [12]        3.10         2.78           3.36       2.74         2.78            2.54
    Sora2 Pro [14]       2.69         2.81           2.82        2.76         2.65            2.92
    Veo3.1 [8]           2.73         2.59           2.88        2.62         2.54            2.24
    Seedance 1.5 [16]    2.39         2.59           3.19        2.88         2.91            2.69
    Seedance 2.0         3.75         3.43            3.67        3.63          3.75             3.56

            Table 2 T2V usability, satisfaction, and delight rates across six evaluation dimensions.
                     Motion   Video Prompt              Audio   Audio-Visual  Audio Prompt                                                Aesthetics                       Quality     Following                 Quality      Sync        Following

                                         Usability Rate (score ≥3)

    Kling 2.6 [10]     70.55%     41.72%       90.80%    45.98%     58.04%        29.02%
    Kling 3.0 [12]     82.82%     58.90%       91.10%    66.52%     66.96%        54.02%
    Sora2 Pro [14]     65.08%     62.54%       63.17%    66.82%     59.35%        63.08%
    Veo3.1 [8]         67.13%     53.63%       74.39%    60.51%     54.36%        37.44%
    Seedance 1.5 [16]  46.93%     54.29%       96.93%    82.59%     69.64%        56.70%
    Seedance 2.0       97.55%      84.97%       96.32%    93.75%     93.30%        83.93%

                                       Satisfaction Rate (score ≥4)

    Kling 2.6 [10]      3.99%      9.20%       29.75%    2.75%     18.75%        8.04%
    Kling 3.0 [12]     28.22%     21.47%       43.56%    9.59%     15.18%        17.86%
    Sora2 Pro [14]     6.98%      22.54%       20.00%    9.81%     19.16%        31.78%
    Veo3.1 [8]         6.57%      12.11%       13.84%    4.10%     13.33%        14.87%
    Seedance 1.5 [16]   1.23%      12.27%       21.78%    5.36%     25.45%        25.45%
    Seedance 2.0       67.18%       51.23%        61.66%    62.05%     68.30%        57.94%

                                         Delight Rate (score = 5)

    Kling 2.6 [10]      0.00%      0.31%       1.53%     0.00%      1.34%         0.45%
    Kling 3.0 [12]      0.61%      2.76%       1.53%     0.00%      1.34%         2.23%
    Sora2 Pro [14]     0.00%      2.86%       0.63%     0.00%      0.47%        11.68%
    Veo3.1 [8]         0.00%      0.69%       0.35%     0.00%      0.00%         0.51%
    Seedance 1.5 [16]   0.00%      0.31%       0.00%     0.00%      1.79%         1.79%
    Seedance 2.0       10.43%      8.28%        9.20%     6.70%      13.84%         26.92%

Table 1 summarizes the T2V results. Seedance 2.0 ranks first on all six dimensions, the only model above
3.4 on every dimension and improving over Seedance 1.5 [16] by an average of 0.86 points, with the largest
gain on motion quality (+1.36). On both motion quality and audio-visual sync, Seedance 2.0 reaches 3.75, at
least 0.65 points ahead of the runner-up. The audio dimensions are where competitors struggle most—most
stay below 2.9—while Seedance 2.0 exceeds 3.5 on all three. Among competitors, Kling 3.0 [12] is the most
balanced overall, Sora 2 Pro [14] stands out on prompt following, and Veo 3.1 [8] is weaker on audio.

The usability breakdown in Table 2 sharpens this picture. Seedance 2.0 is the only model with usability
above 83% on all dimensions, reaching 97.55% on motion quality. More tellingly, Seedance 2.0 exceeds 51%
satisfaction on every dimension—the majority of its outputs score 4 or above—while no competitor exceeds
44% on any single dimension. The gap is most pronounced on audio: audio quality satisfaction is 62.05% vs.
below 10% for all competitors, and audio-visual sync reaches 68.30% vs. a competitor high of 25.45%. At the
delight level (score of 5), only Seedance 2.0 produces any delight-rated audio quality outputs (6.70%), and its
audio prompt following delight rate of 26.92% far exceeds the next-best Sora 2 Pro (11.68%).


                                               7
<a id="page-8"></a>

### PDF 第 8 页

Ad Scene T2V Comparison                                                           Fiction Scene T2V Comparison

                                       Motion                                                                                            Motion
                                                   3.71
                                                                                                                                                                              3.67




 Audio Adherence                                                   Video Adherence               Audio Adherence                                                   Video Adherence
                                                                                 3.50                                                                                                                                                                                                           3.53
                         3.10                                                                                                                                                      3.15





                                                                                                                                                      3.15
                      3.44                                                        3.58                                                                                                                        3.87
Audio-Video Sync                                                         Aesthetics                     Audio-Video Sync                                                         Aesthetics





                                                                                                                                                                              3.54
                                                   3.56

                                   Audio Quality                                                                                    Audio Quality


Kling 2.6        Kling 3.0      Sora2 Pro     Veo 3.1     Seedance 1.5     Seedance 2.0                   Kling 2.6        Kling 3.0      Sora2 Pro     Veo 3.1     Seedance 1.5     Seedance 2.0

                  (a) Ad Scene                                            (b) Fiction Scene


               PGC Scene T2V Comparison                                        Consumer Effects Scene T2V Comparison

                                       Motion                                                                                            Motion
                                                   3.97                                                                                                                       3.46




 Audio Adherence                                                   Video Adherence               Audio Adherence                                                   Video Adherence
                         3.53                                               3.34                                                                 3.42                                                       3.46





                                                                                                                                                                                                      2.89
                      3.84                                                         4.13                                                         3.58
Audio-Video Sync                                                         Aesthetics                     Audio-Video Sync                                                         Aesthetics



                                                                                                                                                                              3.08
                                                   3.75


                                   Audio Quality                                                                                    Audio Quality


Kling 2.6        Kling 3.0      Sora2 Pro     Veo 3.1     Seedance 1.5     Seedance 2.0                   Kling 2.6        Kling 3.0      Sora2 Pro     Veo 3.1     Seedance 1.5     Seedance 2.0

                   (c) PGC Scene                                      (d) Consumer Effects Scene


                      Social Scene T2V Comparison                                                   Basic Scene T2V Comparison

                                       Motion                                                                                            Motion

                                                                                                                                                                              3.78
                                                   3.55




 Audio Adherence                                                   Video Adherence               Audio Adherence                                                   Video Adherence
                     3.77
                                                                                                                                                 3.63                                                     3.47                                                                             3.32





                                                                             3.32
                     3.70                                                                                                                      3.83                                                         3.73
Audio-Video Sync                                                         Aesthetics                     Audio-Video Sync                                                         Aesthetics




                                                   3.67                                                                                                                       3.63

                                   Audio Quality                                                                                    Audio Quality


Kling 2.6        Kling 3.0      Sora2 Pro     Veo 3.1     Seedance 1.5     Seedance 2.0                   Kling 2.6        Kling 3.0      Sora2 Pro     Veo 3.1     Seedance 1.5     Seedance 2.0

                  (e) Social Scene                                                      (f) Basic Scene

                     Figure 3 T2V performance comparison across six scenarios.
                                            8
<a id="page-9"></a>

### PDF 第 9 页

2.3.2  Motion Quality

Table 3 T2V detailed motion evaluation results across fine-grained categories. Rating from 1 to 5, with higher scores
indicating better performance.


 Category                       Kling 2.6 [10]   Kling 3.0 [12]  Sora2 Pro [14]   Veo3.1 [8]  Seedance 1.5 [16]  Seedance 2.0

  Holidays / Festivals                 2.57           2.57            2.69          3.00            2.71               3.29
 Consumer Visual Effects            2.64           3.00            2.77          2.82            2.43                 3.71
  Counter-Reality Instructions        2.86           3.00            2.67          3.00            3.00                 3.71
 Cinematic Visual Effects            3.00           3.00            2.57          2.82            2.57               3.79
 Same-Type Interaction              2.57           3.29            2.64          2.62            2.29               3.79
 Cross-Type Interaction              2.79           3.14            2.79          2.77            2.43               3.57
 Group Coordinated Motion         2.71            3.29            2.57          2.33            2.57               3.29
 Advanced Camera Movement       3.00           3.23            2.58          2.80            2.31                3.77
  Special Camera Shots               2.85           3.08            2.75          2.64            2.08               3.92
  Editing Rhythm                    2.86           3.14            2.93          2.69            2.43                4.21
 Combined Shot Instructions         2.86           3.00            2.29          2.67            2.29               3.86
  Physical Feedback                  2.69           3.00            2.69          2.83            1.69              3.46
  Physical Phenomena                2.46           3.08            2.77          2.58            2.23               3.38
  Natural Phenomena                 2.56           3.11            2.78          2.57            2.00               3.78
 Text Overlay                       2.46           2.85            2.64          2.67            2.38               3.69
  Short Text                          3.00           3.14            3.00          2.67            2.57                 3.71
  Creative Text                       2.43           2.86            2.86          2.83            2.57               3.57
 Long Script                         3.00           3.29            2.57          2.83            2.57               3.57
  Abstract Challenges                 3.00           3.57            3.33          2.57            2.57             4.00
  Multi-Entity Feature Match         3.00           3.43            3.00          2.50            2.29              4.43
 Knowledge Assessment              2.62           3.38            2.62          2.91            2.62               3.69
 Compound Multi-Instructions       2.57           2.86            2.29          2.40            2.57                 3.71
  Surreal Motion                      2.43           2.86            2.00          2.43            2.43                 3.71
  Intense Sports Motion              2.43           2.86            2.21          2.43            2.00               3.79
  Fine Hand Motion                  2.64           3.00            2.57          2.69            2.36                 3.71
 Anthropomorphic Motion           2.57           2.71            2.43          2.71            2.29               3.29
 Emotion & Expression              2.86           3.64            2.93          3.00            2.64             4.00
  Visual Style                         2.71           3.14            2.77          2.62            2.50             4.00
  Lighting & Color Tone              2.71           3.29            2.92          2.93            2.36                 3.71
 Framing / Composition             3.13           3.38            3.00          3.25            2.63               4.25


Seedance 2.0 leads on motion stability, editing rhythm, and multi-entity interaction, with far fewer cases
of subject deformation and physically implausible motion than Seedance 1.5. Kling 3.0 ranks second but is
limited on high-difficulty actions and complex camera work. Veo 3.1 and Sora 2 Pro handle basic motion
reasonably well, yet fall short on fine-grained dynamics and long-take stability. The fine-grained breakdown
in Table 3 confirms this: Seedance 2.0 ranks first on 29 of 30 categories (tying with Kling 3.0 only on group
coordinated motion), scoring 3.29–4.43. Multi-entity feature match (4.43), framing/composition (4.25), and
editing rhythm (4.21) all exceed 4.0. Seedance 1.5 scored low on physical feedback (1.69), natural phenomena
(2.00), and intense sports motion (2.00); Seedance 2.0 improves by over 1.5 points on each. Kling 3.0 does
well on emotion & expression (3.64), abstract challenges (3.57), and multi-entity feature match (3.43), but
drops on intense sports motion (2.86), surreal motion (2.86), and special camera shots (3.08). Sora 2 Pro
scores lowest on surreal motion (2.00) and intense sports motion (2.21); Veo 3.1 is weakest on multi-entity
feature match (2.50) and group coordinated motion (2.33).

2.3.3  Video Prompt Following

Compared to Seedance 1.5, version 2.0 improves on text rendering, physical phenomena, intent comprehension,
and style following, with more precise action following and reasonable creative interpretation beyond the
core instructions. Kling 3.0 carries over strengths from Kling 2.6 on emotional expression and physical
feedback, but stays weak on text following. Veo 3.1’s main shortcoming is poor text generation and instruction
response. Sora 2 Pro handles most categories well with strong creative interpretation, leading on abstract
challenges, but scores lowest on surreal motion, placing it in the second tier. Table 4 shows Seedance 2.0


                                               9
<a id="page-10"></a>

### PDF 第 10 页

Table 4 T2V detailed Video Prompt Following evaluation results across fine-grained categories. Rating from 1 to 5,
with higher scores indicating better performance.


 Category                       Kling 2.6 [10]   Kling 3.0 [12]  Sora2 Pro [14]   Veo3.1 [8]  Seedance 1.5 [16]  Seedance 2.0

  Holidays / Festivals                 2.14           2.50            3.08          2.57            2.50               3.57
 Consumer Visual Effects            2.14           2.93            3.00          2.55            2.79               3.36
  Counter-Reality Instructions        2.43           3.00            3.17          3.40            3.00              4.29
 Cinematic Visual Effects            2.71           3.00            2.50          3.09            3.00              3.64
 Same-Type Interaction              2.29           3.07            2.36          2.62            2.36              3.64
 Cross-Type Interaction              2.64           2.79            2.57          2.69            2.79              3.50
 Group Coordinated Motion         2.86           3.14            2.86          2.50            2.57               3.86
 Advanced Camera Movement       2.69           2.85            2.50          2.40            2.77              3.46
  Special Camera Shots               2.00           2.62            2.75          2.27            2.46              3.00
  Editing Rhythm                    2.57           2.50            2.93          2.62            2.43                3.14
 Combined Shot Instructions         2.71           2.57            2.29          2.33            2.43               3.29
  Physical Feedback                  3.08           3.23            2.77          2.75            2.46               3.62
  Physical Phenomena                2.23           2.77            2.77          2.33            1.92                3.31
  Natural Phenomena                 2.33           3.11            3.22          2.71            2.56               3.89
 Text Overlay                       1.85           2.00            2.91          2.17            2.15                3.31
  Short Text                          1.86           2.29            3.33          2.17            2.00               3.57
  Creative Text                       1.71           2.00            3.00          1.67            1.86              3.43
 Long Script                         2.00           2.86            3.00          2.67            2.43               3.29
  Abstract Challenges                 2.00           3.14              4.17           2.86            2.86              3.86
  Multi-Entity Feature Match         2.14           3.14            3.00          2.17            2.43               3.86
 Knowledge Assessment              2.23           2.54            2.77          3.00            3.00               3.23
 Compound Multi-Instructions       2.71           3.14            2.14          2.40            2.57                 3.71
  Surreal Motion                      2.43              2.71            1.86          2.00            2.14                 2.71
  Intense Sports Motion              2.64           2.79            2.36          2.43            2.71                3.21
  Fine Hand Motion                  2.29           2.93            2.86          2.38            2.57              3.50
 Anthropomorphic Motion           2.00           2.14            2.86          3.00            2.57              2.86
 Emotion & Expression              2.64           3.43            3.21          3.40            2.93             4.00
  Visual Style                         2.36           2.21            2.62          2.54            2.14               2.93
  Lighting & Color Tone              2.50           2.79            3.00          2.79            2.50                3.21
 Framing / Composition             2.88           3.25            3.50          3.00            2.63              3.13



first on 27 of 30 categories, scoring 2.71–4.29. The largest gains over Seedance 1.5 appear on text-related
categories—creative text (1.86 →3.43), short text (2.00 →3.57), text overlay (2.15 →3.31)—and on physical
phenomena (1.92 →3.31) and natural phenomena (2.56 →3.89). Counter-reality instructions (4.29) and
emotion & expression (4.00) are its two highest categories. Sora 2 Pro leads on abstract challenges (4.17
vs. 3.86) and framing/composition (3.50 vs. 3.13), but drops to 1.86 on surreal motion. Veo 3.1 leads on
anthropomorphic motion (3.00) but scores below 2.2 on text overlay (2.17), short text (2.17), and creative
text (1.67). Kling 3.0 scores 3.43 on emotion & expression and 3.23 on physical feedback, but falls to 2.00 on
text overlay and creative text.

2.3.4  Video Aesthetics

Aesthetics is the most competitive dimension. Seedance 2.0 leads on visual effects, scene design, lighting
and color, and realistic detail. Kling 3.0 is relatively strong on stylization and color expression, while other
competitors are weaker on photorealism and fine detail. In Table 5, Seedance 2.0 ranks first or tied for
first on 28 of 30 categories, scoring 2.79–4.14. Its highest scores are visual style (4.14), long script (4.14),
framing/composition (4.13), and four categories at 4.00 (cinematic visual effects, editing rhythm, natural
phenomena, multi-entity feature match). Seedance 2.0 does not lead on consumer visual effects (Seedance
1.5: 3.00 vs. 2.79) or surreal motion (Kling 3.0: 3.86 vs. 3.57), and ties on anthropomorphic motion (3.71,
three-way) and advanced camera movement (3.54, tied with Kling 2.6). Kling 3.0 scores above 3.5 on 13
categories, with its best on surreal motion (3.86), same-type interaction (3.79), and framing/composition
(3.75). Sora 2 Pro and Veo 3.1 lag: Sora 2 Pro drops below 2.5 on holidays (2.38), consumer visual effects
(2.38), and natural phenomena (2.33); Veo 3.1 scores lowest on holidays (2.36), consumer visual effects (2.45),
and multi-entity feature match (2.50).


                                              10
<a id="page-11"></a>

### PDF 第 11 页

Table 5 T2V detailed aesthetics evaluation results across fine-grained categories. Rating from 1 to 5, with higher
scores indicating better performance.


 Category                       Kling 2.6 [10]   Kling 3.0 [12]  Sora2 Pro [14]   Veo3.1 [8]  Seedance 1.5 [16]  Seedance 2.0

 Holidays / Festivals                 2.71           2.71            2.38          2.36            3.00              3.00
 Consumer Visual Effects            2.71           2.64            2.38          2.45            3.00              2.79
 Counter-Reality Instructions        3.43           3.57            3.00          3.20            3.29               3.86
 Cinematic Visual Effects            3.64           3.64            3.00          2.73            3.14             4.00
 Same-Type Interaction              3.29           3.79            2.79          2.85            3.29               3.86
 Cross-Type Interaction              3.36           3.14            3.14          3.15            3.36              3.43
 Group Coordinated Motion         2.86           3.00            2.43          2.50            3.00               3.29
 Advanced Camera Movement        3.54           3.38            2.83          3.00            2.92              3.54
 Special Camera Shots               3.08           3.46            2.67          3.09            3.15               3.85
 Editing Rhythm                    3.21           3.57            3.29          3.08            3.29             4.00
 Combined Shot Instructions         3.29             3.57            3.00          2.67            3.29               3.57
 Physical Feedback                  3.31           3.31            2.46          3.33            3.23              3.54
 Physical Phenomena                3.00           3.23            2.77          2.92            3.00              3.54
 Natural Phenomena                 3.33           3.67            2.33          2.57            3.00             4.00
 Text Overlay                       2.92           3.15            3.00          2.75            3.15                3.31
 Short Text                          3.43           3.14            3.00          3.00            3.00               3.86
 Creative Text                       2.57           3.00            2.86          3.17            3.00               3.57
 Long Script                         3.29           3.57            2.86          3.17            3.14               4.14
 Abstract Challenges                 3.29           3.57            3.17          2.57            3.29               3.86
 Multi-Entity Feature Match         3.71           3.43            3.14          2.50            3.43             4.00
 Knowledge Assessment              3.23           3.31            2.62          3.09            3.31              3.54
 Compound Multi-Instructions       3.29           3.57            2.86          2.80            3.29               3.86
 Surreal Motion                      3.14            3.86            2.86          2.57            3.57              3.57
 Intense Sports Motion              3.07           3.21            2.64          2.71            3.00               3.79
 Fine Hand Motion                   3.43           2.93            2.57          3.00            3.21              3.43
 Anthropomorphic Motion              3.71              3.71            3.29          3.29            3.43                 3.71
 Emotion & Expression              3.21           3.71            2.93          3.10            3.29               3.86
 Visual Style                         3.29           3.50            3.15          2.92            3.21               4.14
 Lighting & Color Tone              3.00           3.50            2.62          2.79            3.36               3.86
 Framing / Composition             3.38           3.75            3.38          2.88            3.25                4.13


2.3.5  Audio Quality

Table 6 T2V detailed audio quality evaluation results across fine-grained categories. Rating from 1 to 5, with higher
scores indicating better performance.


 Category                        Kling 2.6 [10]   Kling 3.0 [12]  Sora2 Pro [14]   Veo3.1 [8]  Seedance 1.5 [16]  Seedance 2.0

 Chinese Dialect / Accent             2.05           2.41            2.29          2.10            2.32               2.82
 Chinese Multi-Person Dialogue       2.36           2.93            2.79          2.20            3.00                 3.71
 Chinese Variety Show Voice          2.14           2.57            2.71          2.14            2.57                3.14
 Chinese Opera                       2.13           2.88            2.17          2.00            2.50               3.75
 English                              3.08           3.17            2.82          3.10            3.00                4.17
 Minority Languages                  2.03           2.59            2.85          2.89            3.09               3.82
 Singing / Rap                        3.14           2.71            3.67          3.00            2.71                 3.71
 Spatial Scene                        2.57           3.14            2.71          2.67            2.86              3.43
 Off-Screen Voice                     2.29           3.00            3.00          2.50            3.00               3.29
 Non-Verbal Voice                    2.44           2.44            2.78          2.56            2.67               3.56
 Voice + Action Interaction           2.71           3.14            3.17          2.67            3.00             4.00
 Object Interaction Sound            2.59           2.47            2.65          2.76            3.06               3.76
 Animal Sound                       2.36           2.57            2.54          2.57            2.79               3.57
 Ambient / Background Sound        2.78           2.33            2.44          2.63            3.00               3.78
 Special Effects (ASMR, etc.)         2.59           2.76            3.12          2.79            3.00                 3.71
 Instruments & Audio                2.79           3.00            2.78          2.89            2.95               3.68
 Dual-Channel Audio                 3.00           3.00            2.57          2.50            3.14              3.43


Moving to audio, Seedance 2.0 improves over Seedance 1.5 on vocal and singing expressiveness, BGM-to-visual


                                              11
<a id="page-12"></a>

### PDF 第 12 页

matching, and audio layering. Kling 3.0 regresses from Kling 2.6 on singing/rap and ambient background
sound. Sora 2 Pro produces vivid audio with strength on singing, though limited to a narrow set of categories.
Competitors broadly have muddy audio, noticeable noise, and weak layering, especially on complex sound
effects and vocal clarity. Per Table 6, Seedance 2.0 ranks first on all 17 categories, scoring 2.82–4.17. English
(4.17), voice + action interaction (4.00), minority languages (3.82), and ambient/background sound (3.78)
are strongest. The improvement over Seedance 1.5 is largest on Chinese opera (2.50 →3.75), English (3.00
→4.17), and singing/rap (2.71 →3.71). Kling 3.0 drops below Kling 2.6 on singing/rap (2.71 vs. 3.14) and
ambient/background sound (2.33 vs. 2.78), despite improving elsewhere. Sora 2 Pro scores 3.67 on singing/rap
(second only to Seedance 2.0) and 3.17 on voice + action interaction, but falls to 2.17 on Chinese opera and
2.44 on ambient sound. No competitor exceeds 3.2 on any category apart from Sora 2 Pro’s singing/rap (3.67).

2.3.6  Audio-Visual Sync

Table 7 T2V detailed audio-visual synchronization evaluation results across fine-grained categories. Rating from 1 to
5, with higher scores indicating better performance.


 Category                        Kling 2.6 [10]   Kling 3.0 [12]  Sora2 Pro [14]   Veo3.1 [8]  Seedance 1.5 [16]  Seedance 2.0

  Chinese Dialect / Accent             2.68           3.14            2.67          2.50            3.00              3.64
  Chinese Multi-Person Dialogue       2.36           2.93            2.64          2.10            2.36               3.86
  Chinese Variety Show Voice          2.29           2.71            2.29          2.43            2.86                3.14
  Chinese Opera                       2.38           2.63            2.50          2.71            2.63               3.63
  English                              2.83           3.00            3.00          2.40            3.50                4.17
  Minority Languages                  2.88           2.97            2.53          2.68            2.74               3.88
  Singing / Rap                        3.00           2.57            3.50          3.00            3.29               4.14
  Spatial Scene                        2.71           3.14            2.43          1.67            3.00               3.86
  Off-Screen Voice                     2.29           2.43            2.33          2.33             2.86               2.86
 Non-Verbal Voice                    2.67           2.56            2.44          2.56            2.89             4.00
  Voice + Action Interaction           2.64           2.36            2.42          2.33            3.14                 3.71
 Object Interaction Sound            2.65           2.53            2.53          2.82            2.65               3.82
 Animal Sound                       2.21           2.64            2.77          2.36            2.79               3.93
 Ambient / Background Sound        2.67           2.89            2.89          2.38            3.00               3.56
  Special Effects (ASMR, etc.)         2.65           2.53            2.53          2.86            3.18               3.53
  Instruments & Audio                3.00           2.79            2.72          2.78            2.89               3.63
 Dual-Channel Audio                 2.86           3.00            3.43          2.17            3.29             4.00


Lip synchronization and action-audio alignment are both strong for Seedance 2.0, with very few cases of delay
or misalignment. Competitors commonly produce lip-speech mismatches and action-sound offsets, worse in
fast dialogue and complex action scenes. In Table 7, Seedance 2.0 ranks first on 16 of 17 categories (tying with
Seedance 1.5 on off-screen voice at 2.86), scoring 2.86–4.17. English (4.17), singing/rap (4.14), dual-channel
audio (4.00), and non-verbal voice (4.00) are strongest. The largest gains over Seedance 1.5 are Chinese
multi-person dialogue (2.36 →3.86), object interaction sound (2.65 →3.82), and animal sound (2.79 →3.93).
No competitor reaches 3.5 on any category apart from Sora 2 Pro on singing/rap (3.50) and Seedance 1.5
on English (3.50). Veo 3.1 is weakest, dropping to 1.67 on spatial scene and 2.10 on Chinese multi-person
dialogue. Kling 3.0 regresses from Kling 2.6 on singing/rap (2.57 vs. 3.00) and instruments & audio (2.79 vs.
3.00), while improving on Chinese dialect (3.14 vs. 2.68).

2.3.7  Audio Prompt Following

Audio prompt following is the dimension where competitors score lowest. Seedance 2.0 is strong on complex
audio instructions involving multilingual dialogue, dialect-specific speech, and diverse sound profiles such
as animal vocalizations. Sora 2 Pro has an edge on instrument and natural sound effect following, while
other models are weaker at generating specific timbres and language-accurate audio. Table 8 shows Seedance
2.0 first on 16 of 17 categories (tying with Kling 3.0 on off-screen voice at 3.14), scoring 2.91–4.25. English
(4.25), instruments & audio (3.89), ambient/background sound (3.89), voice + action interaction (3.86), and
animal sound (3.86) are strongest. The largest gains over Seedance 1.5 are Chinese opera (1.75 →3.50),
singing/rap (2.14 →3.71), and animal sound (2.50 →3.86). Chinese dialect drops below 2.0 for five of six



                                              12
<a id="page-13"></a>

### PDF 第 13 页

Table 8 T2V detailed Audio Prompt Following evaluation results across fine-grained categories. Rating from 1 to 5,
with higher scores indicating better performance.


 Category                        Kling 2.6 [10]   Kling 3.0 [12]  Sora2 Pro [14]   Veo3.1 [8]  Seedance 1.5 [16]  Seedance 2.0

 Chinese Dialect / Accent             1.23           1.86            1.86          1.20            1.82                2.91
 Chinese Multi-Person Dialogue       2.00           2.79            2.21          1.80            2.36               3.29
 Chinese Variety Show Voice          2.00           2.29            2.71          1.57            2.43               3.29
 Chinese Opera                       1.63           2.13            2.33          1.29            1.75              3.50
 English                              2.25           3.33            3.64          3.00            3.25               4.25
 Minority Languages                  1.03           2.18            3.12          2.61            3.26               3.74
 Singing / Rap                        3.14           3.43            3.67          2.17            2.14                 3.71
  Spatial Scene                        2.71           2.86            2.71          2.50            2.86               3.29
 Off-Screen Voice                     2.57             3.14            3.00          1.83            2.71                3.14
 Non-Verbal Voice                    2.00           2.00            2.89          2.33            3.00               3.67
 Voice + Action Interaction           2.57           3.21            3.50          2.33            3.21               3.86
 Object Interaction Sound            2.41           2.00            2.53          2.53            2.53               3.29
 Animal Sound                       1.64           2.64            2.77          2.21            2.50               3.86
 Ambient / Background Sound        2.44           2.67            3.22          2.38            2.78               3.89
  Special Effects (ASMR, etc.)         2.18           2.35            3.29          2.14            2.94               3.47
 Instruments & Audio                2.58           3.11            3.61          3.06            2.79               3.89
 Dual-Channel Audio                 2.71           2.57            2.86          2.17            2.43               3.29


models, and Chinese opera below 2.4 for five of six. Sora 2 Pro is the strongest competitor, scoring 3.67 on
singing/rap, 3.64 on English, 3.61 on instruments & audio, and 3.50 on voice + action interaction, but falls to
1.86 on Chinese dialect. Kling 3.0 regresses from Kling 2.6 on object interaction sound (2.00 vs. 2.41) and
dual-channel audio (2.57 vs. 2.71). Veo 3.1 scores below 2.0 on Chinese dialect (1.20), Chinese variety show
voice (1.57), Chinese opera (1.29), and off-screen voice (1.83).

2.4  Image-to-Video Evaluation on SeedVideoBench 2.0

2.4.1  Overall Results

         Table 9 I2V overall evaluation results across video and audio dimensions (Rating from 1 to 5).

                                     Video                                   Audio
        Model         Motion   Video Prompt     Image         Quality &     Audio-Visual  Audio Prompt
                        Quality     Following     Preservation   Expressiveness      Sync        Following

 Wan 2.6 [1]              2.32         2.74            2.61             2.20             2.18            2.55
 Kling 2.6 [10]            2.52         2.55            2.98             2.21             2.27            2.21
 Veo3.1 [8]                2.65         2.87            2.69             2.68             2.69            2.79
 Seedance 1.5 Pro [16]    2.53         2.77            2.92             3.07             2.95            3.10
 Kling 3.0 [12]            2.80         2.78            3.18             2.89             2.83            2.85
 Seedance 2.0              3.35          3.46              3.31               3.61              3.54             3.70



Table 9 summarizes the I2V results. Seedance 2.0 ranks first on all six dimensions, scoring 3.31–3.70, while
no competitor exceeds 3.18. The three video dimensions show 3.35 (motion quality), 3.46 (video prompt
following), and 3.31 (image preservation). Source image preservation is the tightest race—Kling 3.0 trails by
only 0.13—while motion quality shows a 0.55-point gap to the runner-up. The three audio dimensions are
where competitors fall furthest behind: Seedance 2.0 scores 3.61, 3.54, and 3.70, while Kling 2.6 (2.21) and
Wan 2.6 [1] (2.18–2.55) sit well below 3.0. Seedance 1.5 Pro is second on audio (3.07, 2.95, 3.10) but still
trails by 0.54–0.60. Audio prompt following (3.70) is Seedance 2.0’s highest I2V score. A two-tier pattern is
clear: Seedance 2.0 leads on both video and audio, while competitors are weaker on audio than on video.

Table 10 breaks this down further. Seedance 2.0 is the only model with usability above 87% on all six
dimensions. On motion quality, its 43.88% satisfaction rate is over 3× the runner-up Kling 3.0 (12.00%); on
video prompt following, 47.48% vs. Veo 3.1’s 20.54%. Source image preservation is again closest: 91.37%


                                              13
<a id="page-14"></a>

### PDF 第 14 页

Table 10 I2V usability and satisfaction rates across video and audio dimensions.

                                     Video                                   Audio
        Model         Motion   Video Prompt     Image         Quality &     Audio-Visual  Audio Prompt
                        Quality     Following     Preservation   Expressiveness      Sync        Following

                                              Usability Rate (score ≥3)

 Wan 2.6 [1]           32.71%     59.02%        59.77%        27.03%        25.68%        51.80%
 Kling 2.6 [10]         48.72%     48.72%        80.59%        27.19%        31.14%        30.26%
 Veo3.1 [8]            59.69%     65.50%        67.05%        60.75%        59.35%        56.54%
 Seedance 1.5 Pro [16]  51.44%     58.99%        80.58%        93.99%        80.26%        70.39%
 Kling 3.0 [12]         68.00%     60.00%        90.55%        77.39%        68.70%        66.52%
 Seedance 2.0          87.05%      88.85%         91.37%         97.42%          91.85%         92.27%

                                           Satisfaction Rate (score ≥4)

 Wan 2.6 [1]           2.63%      16.54%        7.52%         0.45%         1.80%         9.91%
 Kling 2.6 [10]          5.86%      9.52%        19.05%         1.32%         2.19%         5.70%
 Veo3.1 [8]             7.36%      20.54%        6.20%         7.48%         10.28%        24.30%
 Seedance 1.5 Pro [16]   3.60%      18.71%        12.59%        13.30%        15.45%        37.77%
 Kling 3.0 [12]         12.00%     18.91%        27.27%        12.61%        15.22%        20.87%
 Seedance 2.0          43.88%      47.48%        38.85%         57.08%         54.94%        63.52%



usability vs. Kling 3.0’s 90.55%, though the satisfaction gap is wider (38.85% vs. 27.27%). The audio contrast
is sharper. On audio quality, Seedance 2.0 reaches 97.42% usability and 57.08% satisfaction; Kling 2.6 and
Wan 2.6 have usability below 28%, meaning most of their audio is rated unacceptable. On audio prompt
following, Seedance 2.0’s 63.52% satisfaction is 1.7× Seedance 1.5 Pro’s 37.77% and over 10× Kling 2.6’s
5.70%.

Beyond the scores, human evaluation of generated videos surfaces additional patterns. Seedance 2.0 produces
dynamic motion with a clear sense of momentum—combat and dance sequences mix slow-motion highlights
with fast action in ways competitors do not, and facial expressions and gaze are more vivid than Seedance 1.5
Pro. Camera work tracks subject motion closely with varied angles and smooth push/pull transitions. First-
and third-person game-following perspectives with handheld breathing effects are new to this version and add
immersion. The model handles special art styles (felt, oil painting, Chinese gongbi) without breaking visual
coherence, matching subject motion to the referenced style. Realistic and 3D visual effects render fluidly. On
the audio side, dialogue voices carry emotional nuance in both Chinese and non-Chinese languages. Voice,
sound effects, and audio are well-layered—outputs sound like composed audio rather than isolated tracks
stacked together. Common Chinese dialects (Sichuan, Northeastern, Cantonese) come through accurately.
Singing, rap, and instrumental audio across languages are strong, with melodies that fit the prompt context.

2.4.2  Detailed Visual Evaluation Results

Table 11 Fine-grained I2V visual evaluation on prompt abstraction tasks. Rating from 1 to 5. MQ means motion
quality, IP means image preservation, and VPF means video prompt following.

                            UGC Creative / Portrait   Script-Controlled (15s)
                     Model                         MQ     IP     VPF    MQ     IP     VPF

                Kling 2.6 [10]          2.33   3.00      2.33      2.47   2.73     2.13
            Wan 2.6 [1]            2.60   3.00      3.13      2.14   2.36     2.50
              Veo 3.1 [8]             2.53   2.87      3.87      2.58   2.33     2.33
                Seedance 1.5 Pro [16]   2.87   3.07      3.13      2.80   2.87     2.53
                Kling 3.0 [12]          3.07   3.27      2.73      3.07   2.87     3.00
                Seedance 2.0           3.40   3.40      3.53      3.40   3.13      3.87




                                              14
<a id="page-15"></a>

### PDF 第 15 页

Prompt Abstraction.  This category tests UGC-style creative prompts and script-controlled generation.
Seedance 2.0 leads on all three metrics for script-controlled (15s) generation and on MQ and IP for UGC
creative/portrait (Table 11). The gap is largest on script-controlled (15s) generation: VPF 3.87 vs. Kling
3.0’s 3.00 and Wan 2.6’s 2.50. Veo 3.1 scores 3.87 on VPF for UGC creative/portrait—the best in that
sub-category—but its MQ (2.53) and IP (2.87) are much lower. Kling 2.6 falls below 2.5 on MQ for both
sub-categories, and Wan 2.6 drops to 2.14 on script-controlled MQ. Open-ended and script-based prompts
remain difficult for most models; only Seedance 2.0 and Kling 3.0 consistently exceed 3.0 on MQ.

       Table 12 Fine-grained I2V visual evaluation on complex instruction following. Rating from 1 to 5.

                           New Entity       Compound         Degree
                                           (Size Rel.)           Multi-Instr.         Adverbs                 Model
                      MQ     IP   VPF  MQ     IP   VPF  MQ     IP   VPF

            Kling 2.6 [10]          2.50   2.88   3.13   2.38   2.88   2.00   2.47   2.87   2.33
        Wan 2.6 [1]            2.63   2.63   2.75   2.29   2.43   2.57   2.38   3.00   2.77
          Veo 3.1 [8]             2.50   2.88   3.38   2.57   2.29   2.29   2.79   2.64   2.79
           Seedance 1.5 Pro [16]   3.25   2.75   2.63   2.13   2.63   2.25   2.80   3.00   3.07
            Kling 3.0 [12]          2.75   3.00   3.00   2.88   3.25   2.50   2.60   3.07   2.67
           Seedance 2.0             3.75   3.25   3.88   4.00   3.38   3.75   3.20   3.40   3.40


Complex Instructions.  Compound multi-instruction is where Seedance 2.0 pulls furthest ahead: MQ 4.00
and VPF 3.75, outscoring Kling 3.0 by over 1.0 point on MQ (Table 12). Seedance 1.5 Pro scored 2.13/2.25
on MQ/VPF for this sub-category, so complex instruction handling improved by nearly 2 points in the 2.0
generation. Veo 3.1 reaches 3.38 on VPF for new entity tasks—close to Seedance 2.0’s 3.88—but its MQ
(2.50) lags, meaning it follows the instruction but produces weaker motion. Degree adverbs is the tightest
sub-category: Seedance 2.0 scores 3.20/3.40/3.40 with Seedance 1.5 Pro not far behind at 2.80/3.00/3.07.

          Table 13 Fine-grained I2V visual evaluation on complex camera work. Rating from 1 to 5.

                                Combined        Adv. Camera       Difficult Shots &
                                    Shot Instr.        Movement        Special Tech.
                 Model
                      MQ     IP   VPF  MQ     IP   VPF  MQ     IP   VPF

            Kling 2.6 [10]          2.43   2.93   2.43   2.29   2.57   2.86   2.50   3.25   3.38
        Wan 2.6 [1]            2.33   2.93   2.93   2.14   2.14   2.71   2.25   2.50   3.00
          Veo 3.1 [8]             2.80   3.00   3.00   2.67   2.50   2.33   2.75   2.38   2.75
           Seedance 1.5 Pro [16]   2.67   3.07   3.07   2.43   2.71   2.57   2.50   2.63   3.00
            Kling 3.0 [12]          3.20   3.27   2.93    2.71   2.71   2.43   2.63   3.25   2.38
           Seedance 2.0            3.47   3.27   3.53    2.71   3.00   3.14   3.50   3.88   3.00


Complex Camera.  Seedance 2.0 leads on MQ for all three camera sub-categories and on VPF for combined
shot instructions and advanced camera movement (Table 13). On difficult shots & special techniques, it
scores 3.88 on IP—the highest single value in this table—while Kling 3.0 reaches 3.25. Kling 2.6 scores 3.38
on VPF for difficult shots—the best in that sub-category—handling special techniques better than general
camera work. Advanced camera movement is the hardest sub-category: Seedance 2.0 and Kling 3.0 tie on MQ
(2.71), and no model exceeds 3.14 on any metric. Camera flexibility is an area where all models have room to
improve; advanced movement scores stay below 3.2 across the board.


Complex Motion.  Seedance 2.0’s strongest results here are sports (MQ 3.73, VPF 3.93) and micro-expression
& emotion (VPF 4.00), as shown in Table 14. Combat visual effects shows the widest gap: Seedance 2.0
scores MQ 3.63 vs. Kling 3.0’s 2.25 and Seedance 1.5 Pro’s 2.25—a 1.38-point difference. Expression and
gaze vividness improved substantially over Seedance 1.5 Pro (micro-expression MQ: 2.88 →3.63). Kling 3.0


                                              15
<a id="page-16"></a>

### PDF 第 16 页

Table 14 Fine-grained I2V visual evaluation on complex motion. Rating from 1 to 5.

                            Sports           Fine Motion     Micro-Expr. & Emotion  Combat Visual Effects
        Model               MQ     IP   VPF  MQ     IP   VPF  MQ     IP     VPF   MQ     IP     VPF

  Kling 2.6 [10]          3.00   3.07   2.53   2.40   3.27   2.47   2.88   3.38     2.63     2.63   2.75     2.50
 Wan 2.6 [1]            2.00   2.64   2.29   2.57   2.86   3.00   2.38   2.38     3.13     2.14   1.86     2.14
 Veo 3.1 [8]             2.54   2.92   2.69   2.57   3.00   2.79   3.00   3.00     3.00     2.50   2.75     2.88
 Seedance 1.5 Pro [16]   2.40   3.07   2.53   2.33   3.20   2.73   2.88   3.13     3.13     2.25   2.63     2.50
  Kling 3.0 [12]          2.71   3.29   3.00   2.80   3.47   3.00   3.13   3.50     2.88     2.25   2.88     2.13
 Seedance 2.0             3.73   3.47   3.93   3.33   3.47   3.53   3.63   3.63     4.00     3.63   3.25      3.13



is competitive on fine motion IP (3.47, tying with Seedance 2.0) and micro-expression IP (3.50), preserving
image identity well even when its motion quality lags. Wan 2.6 scores 1.86 on combat IP—the lowest value
across all visual tables.

           Table 15 Fine-grained I2V visual evaluation on complex interaction. Rating from 1 to 5.

                           Group Motion    Same-Type Interaction  Cross-Type Interaction
            Model                   MQ     IP   VPF  MQ     IP     VPF   MQ     IP     VPF

       Kling 2.6 [10]           2.38   2.50   2.75   2.91   3.36     2.45     2.88   3.13     2.75
     Wan 2.6 [1]             2.13   2.25   2.50   2.45   2.82     2.91     2.38   3.13     2.88
      Veo 3.1 [8]             2.63   2.50   2.75   2.40   2.50     2.90     3.00   2.88     3.00
       Seedance 1.5 Pro [16]   2.50   2.63   2.88   2.45   2.91     3.18     2.88   3.25      3.13
       Kling 3.0 [12]           2.50   3.00   2.50   2.73   3.27     2.55     3.38   3.13     3.63
       Seedance 2.0           3.00   3.00   2.88   3.64   3.82       3.91      3.50   3.25     4.00



Complex Interaction.  Same-type interaction (MQ 3.64, IP 3.82, VPF 3.91) and cross-type interaction (VPF
4.00) are Seedance 2.0’s strongest results in Table 15. Group motion is hard for everyone—Seedance 2.0
scores 3.00/3.00/2.88, and most competitors hover near 2.5. Kling 3.0 scores 3.63 on cross-type VPF, close to
Seedance 2.0’s 4.00, and 3.38 on MQ, handling inter-species or human-object interactions reasonably well.
Kling 2.6 scores 3.36 on same-type IP but lags on MQ (2.91) and VPF (2.45).

            Table 16 Fine-grained I2V visual evaluation on creative generation. Rating from 1 to 5.

                            Counter-           Design          Visual Effects                                                                                     Holidays                                Reality           Instructions      (Transformation)        Model
                MQ     IP   VPF  MQ     IP   VPF  MQ     IP   VPF  MQ     IP   VPF

  Kling 2.6 [10]          2.36   2.71   2.57   2.20   2.80   2.20   2.43   2.57   2.71   2.43   2.86   2.71
  Wan 2.6 [1]            2.08   2.08   2.54   2.47   2.60   2.53   2.13   2.50   2.75   2.00   2.25   2.75
  Veo 3.1 [8]             2.43   2.57   2.86   2.67   2.60   2.93   3.00   2.50   3.38   2.50   2.00   2.38
  Seedance 1.5 Pro [16]   2.27   2.80   2.40   2.33   2.60   2.20   2.75   2.88   3.25   2.38   2.88   2.38
  Kling 3.0 [12]          2.27   3.07   2.47   2.60   2.87   2.33   2.63   3.13   2.50   2.71   2.86   3.00
  Seedance 2.0            2.93   3.07   3.07    3.13   2.87   2.87   3.50   3.38   3.50   3.00   3.00   3.38


Creative.  Seedance 2.0 leads on MQ for all four creative sub-categories (Table 16), with visual effects
(transformation) as its best at 3.50/3.38/3.50. Realistic and 3D visual effects render fluidly, and the model
preserves special art styles (felt, oil painting, Chinese gongbi) while matching motion to the style. Veo 3.1 is
close on VPF for visual effects (3.38) and design instructions (2.93), but its MQ trails. Holidays is weak for
most models—Veo 3.1 drops to 2.00 on IP, Wan 2.6 to 2.00/2.25. No model besides Seedance 2.0 exceeds 3.50
on any metric in this category.


                                              16
<a id="page-17"></a>

### PDF 第 17 页

Table 17 Fine-grained I2V visual evaluation on physical laws. Rating from 1 to 5.

                            Natural Phenomena   Physical Phen. (Prof.)   Physical Feedback (Daily)
           Model                  MQ     IP    VPF   MQ     IP    VPF   MQ     IP      VPF

     Kling 2.6 [10]          2.67   3.11    2.67    2.57   3.50    2.50    2.29   2.86       2.50
    Wan 2.6 [1]            2.38   2.00    2.88    2.38   2.77    2.54    2.27   2.80       2.67
     Veo 3.1 [8]             2.67   2.56    2.78    2.79   3.00    3.07    2.46   2.92       2.69
     Seedance 1.5 Pro [16]   2.33   2.89    2.56    2.14   3.07    2.36    2.13   2.93       2.80
     Kling 3.0 [12]          3.00   3.56    3.00    2.64   3.36    2.93    2.73   3.13       2.73
     Seedance 2.0            3.33   3.44    3.33     3.14   3.36     3.36     2.87   3.07       2.93



Physical Laws.  Seedance 2.0 leads on MQ across all three physical laws sub-categories (Table 17), scoring
2.87–3.33. Kling 3.0 outscores Seedance 2.0 on IP for natural phenomena (3.56 vs. 3.44) and ties on professional
phenomena (3.36)—one of the few areas where a competitor beats Seedance 2.0 on a specific metric. Kling
2.6 scores 3.50 on professional phenomena IP, its highest value in any visual table, though its MQ (2.57) and
VPF (2.50) stay low. Physical laws is difficult across the board: Seedance 1.5 Pro scores below 2.4 on MQ for
all three sub-categories, and motion stability during physics simulations remains a challenge for every model.

         Table 18 Fine-grained I2V visual evaluation on complex reference images. Rating from 1 to 5.

                                          High Information    Multi-Ethnicity /
                                                Density           Skin Tone                         Model
                            MQ     IP   VPF  MQ     IP   VPF

                     Kling 2.6 [10]          2.67   3.07   3.00   2.60   3.07   2.73
               Wan 2.6 [1]            2.20   2.53   3.00   2.57   3.00   2.71
                   Veo 3.1 [8]             2.67   2.80   2.87   2.82   2.45   2.64
                    Seedance 1.5 Pro [16]   2.67   3.00   3.13   2.73   3.00   3.00
                     Kling 3.0 [12]          2.87   3.33   3.13   3.21   3.43   2.93
                    Seedance 2.0           3.40   3.40   3.73   3.47   3.33   3.53



Complex Reference.  Seedance 2.0 leads on MQ and VPF for both complex reference sub-categories, and on
IP for high information density (Table 18). High information density VPF (3.73) outscores Kling 3.0 (3.13)
by 0.60 points. Kling 3.0 outscores Seedance 2.0 on multi-ethnicity IP (3.43 vs. 3.33), one of the few metrics
where a competitor takes the lead. The visual average confirms the overall ranking: Seedance 2.0 leads on
MQ (3.35), IP (3.31), and VPF (3.46). Kling 3.0 is second on IP (3.18) and third on VPF (2.78), ahead of
Seedance 1.5 Pro (2.77). Wan 2.6 ranks last on MQ (2.32) and IP (2.61).

2.4.3  Detailed Audio Evaluation Results

Chinese Voice.  Seedance 2.0 leads on AQ for all four Chinese voice sub-categories in Table 19, scoring
3.13–3.92. Chinese dialogue carries emotional nuance, and common dialects and accents come through clearly.
Chinese conversation is its strongest (AQ 3.92, APF 4.08), with a 0.83-point AQ lead over Kling 3.0 and a
1.67-point lead over Seedance 1.5 Pro. Variety show voice reaches 4.00 on both AVS and APF, the highest
sync score in this table. Chinese opera is a weak spot for all models on prompt following—Seedance 2.0 scores
only 2.50 on APF, though its AQ (3.75) is best. Kling 3.0 is the closest competitor on dialect/lip sync (AQ
3.17 vs. 3.23) and opera sync (AVS 3.00 vs. 3.38). Veo 3.1, Wan 2.6, and Kling 2.6 score below 2.5 on AQ for
most sub-categories, with Kling 2.6 at 2.00 on dialect—the lowest AQ in this table.


Non-Chinese Voice.  Seedance 2.0 scores at least 3.50 on AQ for all six non-Chinese languages in Table 20,
peaking on Spanish (AQ 4.14, AVS 4.14) and English (AQ 4.00, APF 4.20). English prompt following reaches
4.20—the highest APF in this table. Indonesian APF (4.14) is also strong, providing the second-highest


                                              17
<a id="page-18"></a>

### PDF 第 18 页

Table 19  Fine-grained I2V audio evaluation on Chinese voice. AQ = Audio Quality & Expressiveness, AVS =
Audio-Visual Sync, APF = Audio Prompt Following. Rating from 1 to 5.

                        Chinese Dialect /       Chinese         Variety Show         Chinese
                               Lip Sync         Conversation          Voice            Opera        Model
                 AQ  AVS  APF  AQ  AVS  APF  AQ  AVS  APF  AQ  AVS  APF

  Kling 2.6 [10]          2.00   2.25   2.25   2.33   2.08   2.33   2.13   2.13   2.25   2.50   2.38   2.13
 Wan 2.6 [1]            2.46   2.15   2.92   2.33   2.08   2.75   2.43   3.00   3.00   2.38   2.38   2.13
  Veo 3.1 [8]             2.09   2.45   2.18   2.20   2.50   2.10   2.14   2.29   1.86   2.25   2.75   1.75
  Seedance 1.5 Pro [16]   2.92   3.00   2.92   2.25   2.83   3.08   3.00   2.75   3.13   3.00   2.88   2.38
  Kling 3.0 [12]          3.17   3.08   3.08   3.09   2.73   3.00   2.75   2.88   3.13   2.88   3.00   1.88
  Seedance 2.0            3.23   3.46    3.31   3.92   3.42   4.08   3.13   4.00   4.00   3.75   3.38   2.50

            Table 20 Fine-grained I2V audio evaluation on non-Chinese voice. Rating from 1 to 5.

                               English          Japanese          Korean           Indonesian        Portuguese          Spanish
        Model
                 AQ  AVS  APF  AQ  AVS  APF  AQ  AVS  APF  AQ  AVS  APF  AQ  AVS  APF  AQ  AVS  APF

 Kling 2.6 [10]           2.40   2.33   2.40   1.88   2.25   1.00   1.57   2.29   1.00   2.29   2.00   1.29   2.38   2.38   1.38   2.29   2.29   1.57
 Wan 2.6 [1]             2.33   2.40   2.40   2.25   2.25   2.13   2.13   2.25   2.75   2.29   1.71   1.71   2.13   2.13   2.38   2.20   2.20   2.20
 Veo 3.1 [8]             3.21   2.71   2.93   3.00   2.57   2.71   2.88   2.75   3.13   2.83   2.33   3.00   2.86   2.86   3.00   3.00   3.00   3.83
 Seedance 1.5 Pro [16]   3.33   3.13   3.40   3.13   3.13   3.63   3.13   3.00   3.50   3.14   3.14   3.00   2.88   3.13   4.00   3.29   3.14   3.43
 Kling 3.0 [12]           3.07   3.00   2.80   2.88   2.88   2.50   2.63   2.88   3.25   3.00   3.57   1.71   2.88   2.88   3.00   3.00   3.00   3.00
 Seedance 2.0           4.00   3.93   4.20   4.00   3.63   3.13   3.75   3.38   3.38    3.71    3.71    4.14   3.50   3.63   3.63   4.14   4.14   4.00


prompt following score. Seedance 1.5 Pro is second overall, with competitive scores on Portuguese APF (4.00)
and Japanese APF (3.63), occasionally matching or exceeding Seedance 2.0 on prompt following. Veo 3.1
scores 3.83 on Spanish APF, its single best result across all audio tables, but its AQ and AVS stay around
2.7–3.0. Wan 2.6 and Kling 2.6 fall far behind: Kling 2.6 scores 1.00 on APF for both Japanese and Korean,
and Wan 2.6 drops to 1.71 on Indonesian AVS and APF.

           Table 21 Fine-grained I2V audio evaluation on composite voice tasks. Rating from 1 to 5.

                             Singing /          Off-Screen           Spatial          Non-Verbal
                         Rap              Voice            Scene             Voice        Model
                 AQ  AVS  APF  AQ  AVS  APF  AQ  AVS  APF  AQ  AVS  APF

  Kling 2.6 [10]          2.67   2.33   3.22   2.50   2.38   2.50   2.20   2.20   2.40   2.23   2.38   2.46
 Wan 2.6 [1]            2.50   2.30   3.00   2.43   2.14   2.57   2.20   2.30   2.40   2.08   2.15   2.54
  Veo 3.1 [8]             3.30   3.20   3.80   3.00   2.67   1.83   2.88   3.00   2.50   2.45   2.73   2.73
  Seedance 1.5 Pro [16]   2.80   2.90   2.90   3.13   2.50   2.88   3.10   3.00   2.40   2.85   2.85   3.23
  Kling 3.0 [12]          3.30   3.10   3.30   2.88   2.50   2.88   2.90   2.80   2.30   2.69   2.62   3.15
  Seedance 2.0            3.90   3.60   4.10   3.75    3.75   3.88   3.30   3.20   3.30   3.54   3.54   3.54



Voice Composite.  Seedance 2.0 leads on all four composite voice sub-categories in Table 21. Singing, rap, and
instrumental audio across languages perform well, with melodies matched to the prompt context. Singing/rap
scores 4.10 on APF—the model generates lyrics and melodies that match the prompted style. Off-screen voice
scores 3.75/3.75/3.88, with audio-visual rhythm staying tight even when the speaker is not visible. Veo 3.1 is
competitive on singing APF (3.80), its best result across all composite tasks, but drops to 1.83 on off-screen
voice APF. Kling 3.0 scores 3.30 on singing AQ but only 2.30 on spatial scene APF. Audio-visual sync on
off-screen narration is a pain point for most competitors—Kling 3.0 and Seedance 1.5 Pro both score 2.50 on
AVS.


Sound Effects.  Seedance 2.0 leads on AVS for all five sound effects sub-categories and on AQ for four of five
in Table 22; Seedance 1.5 Pro edges ahead on background/ambient AQ (3.30 vs. 3.20). Voice, sound effects,


                                              18
<a id="page-19"></a>

### PDF 第 19 页

Table 22 Fine-grained I2V audio evaluation on sound effects. Rating from 1 to 5.

                           Dialogue-       Object-Physical        Animal        Background /      Special Effects
        Model              Interaction           Events           Sound           Ambient         (ASMR, etc.)
                 AQ  AVS  APF  AQ  AVS  APF  AQ  AVS  APF  AQ  AVS  APF  AQ  AVS  APF

 Kling 2.6 [10]          2.00   2.00   2.00   2.46   2.38   2.62   2.00   1.89   0.56   1.60   1.90   2.30   2.25   2.44   2.38
 Wan 2.6 [1]            1.86   2.14   2.86   1.92   2.23   2.23   2.00   1.89   2.89   2.22   2.33   2.89   2.00   2.00   2.33
 Veo 3.1 [8]             2.50   2.50   2.75   2.79   2.57   3.14   2.33   2.44   2.78   2.50   2.50   3.20   2.86   2.71   3.14
 Seedance 1.5 Pro [16]   3.00   2.75   3.13   2.86   2.86   3.07   3.11   3.00   3.11   3.30   3.00   3.60   3.13   3.19   2.94
 Kling 3.0 [12]          2.88   2.25   3.00   2.77   2.69   3.00   2.78   2.78   3.00   2.80   2.90   3.30   2.75   2.75   2.63
 Seedance 2.0            3.50   3.00   3.88   3.57   3.50   3.86   3.56   3.22   3.44   3.20   3.50   4.00   3.44   3.56    3.81


and audio are well-layered—outputs sound like composed audio rather than isolated tracks stacked on top of
each other. Background/ambient sound reaches 4.00 on APF—the model matches BGM and environmental
audio to the video rhythm. Object-physical events scores 3.86 on APF, with action sounds synchronized to
on-screen motion. Seedance 1.5 Pro is competitive on background sound (APF 3.60) and animal sound (AQ
3.11), placing it second on several sub-categories. Kling 2.6 scores 0.56 on animal sound APF—nearly zero
prompt following—the lowest value across all audio tables. Wan 2.6 also struggles, scoring 2.00 on AQ for
animal sound and dropping to 1.92 on object-physical events AQ.

    Table 23 Fine-grained I2V audio evaluation on instruments, dual-channel audio, and UGC creative tasks.

                                     Instruments       Dual-Channel    UGC Creative /
                          & Audio            Audio              Portrait                 Model
                        AQ  AVS  APF  AQ  AVS  APF  AQ  AVS  APF

           Kling 2.6 [10]          2.50   2.86   2.57   2.00   2.07   2.07   2.27   2.36   2.45
        Wan 2.6 [1]            2.29   2.36   2.64   2.08   1.92   2.38   2.09   2.45   2.73
          Veo 3.1 [8]             2.87   3.07   3.00   2.43   2.71   2.36   2.64   2.55   3.00
           Seedance 1.5 Pro [16]   3.13   3.07   2.73   3.13   2.93   2.87   3.00   2.64   3.45
           Kling 3.0 [12]          3.00   2.67   3.00   2.67   2.80   2.47   2.91   2.91   3.00
           Seedance 2.0            3.60   3.27   3.80   3.47   3.53    3.27   3.64   3.64   3.82



Other Audio.  In Table 23, Seedance 2.0 scores 3.80 on instruments & audio APF, generating instrument
sounds and melodies that match the prompt. Dual-channel AVS reaches 3.53, with stereo separation that
tracks the visual scene. UGC creative/portrait scores 3.64/3.64/3.82, its strongest sub-category in this
table. Seedance 1.5 Pro is second on UGC APF (3.45) and instruments AQ (3.13). Dual-channel audio is
weak across the board for competitors: Wan 2.6 scores 1.92 on AVS and Kling 2.6 scores 2.07, both below
usability. The audio average row confirms the overall ranking: Seedance 2.0 (3.61/3.54/3.70), Seedance 1.5
Pro (3.07/2.95/3.10), Kling 3.0 (2.89/2.83/2.85), Veo 3.1 (2.68/2.69/2.79), Kling 2.6 (2.21/2.27/2.21), Wan
2.6 (2.20/2.18/2.55).

2.5  Reference-to-Video Evaluation on SeedVideoBench 2.0

2.5.1  Quantitative Results

Table 24 Reference-to-video (R2V) evaluation results. Multimodal Task Following and Prompt Following are rated
1–3, other dimensions are rated 1–5.

                           Multimodal Task     Editing     Reference   Motion   Prompt
             Model
                                Following      Consistency  Alignment   Quality   Following

        Vidu Q2 Pro [18]         2.13             2.29          1.79       2.38       2.08
         Kling O1 [11]             2.30             2.89          2.32       2.30       1.95
         Kling 3.0 [12]             2.32             3.37          2.37       2.36       1.95
         Seedance 2.0              2.50              3.54          3.03        3.24        2.52


                                              19
<a id="page-20"></a>

### PDF 第 20 页

Seedance 2.0 leads all evaluated models on R2V, outperforming Kling 3 Omni [12], Kling O1 [11], and Vidu
Q2 Pro [18] across every dimension. It supports more multi-modal task types with higher accuracy—beyond
subject, style, motion reference, and video editing supported by competitors, Seedance 2.0 also handles
creative and visual-effects reference, video continuation and extension, and combined tasks such as motion
reference + subject reference, with fewer issues of missing or confused input materials. Reference alignment is
best across subject appearance and voice, motion, and style, with clear advantages in subject identity and
style preservation. Motion quality—vividness, physical plausibility, and structural stability—is uniformly
stronger, with stability as the largest lead. Prompt following is also strongest, particularly on long-text and
complex narratives, visual effects, dialogue, text rendering, and open-ended instructions. Table 24 quantifies
this: Seedance 2.0 ranks first on all five dimensions, scoring 2.50 and 2.52 on multimodal task following and
prompt following (1–3 scale), and 3.54, 3.03, 3.24 on editing consistency, reference alignment, and motion
quality (1–5 scale). The gaps are smallest on editing consistency (Kling 3.0 trails by 0.17) and largest on
motion quality (0.86–0.94 behind across all competitors) and reference alignment (0.66–1.24 behind). Vidu
Q2 Pro scores lowest on three of five dimensions. Kling 3.0 and Kling O1 trade second and third depending
on the dimension, but neither approaches Seedance 2.0 on motion quality or prompt following.

          Table 25 R2V multi-modal task support across models. ✓= supported, ✗= not supported.

 Task             Input Modality                                   Seedance 2.0   Kling 3 Omni [12]  Vidu Q2 Pro [18]   Kling O1 [11]
                 Image Reference                           ✓         ✓          ✓         ✓
                 Video Reference                           ✓         ✓          ✓            ✗
  Subject
                  Audio-Visual Reference                       ✓         ✓          ✓            ✗
  Reference
                 Audio + Image Reference                      ✓         ✓          ✓            ✗
                 Video Motion Reference                       ✓         ✓          ✓         ✓
 Motion                 Video Motion Reference + Image Reference           ✓         ✓          ✓         ✓  Reference
                 Video Motion Reference + First Frame              ✓         ✓          ✓         ✓
                   Visual Effects / Creative Reference                ✓            ✗              ✗            ✗
  Visual Effects /
                   Visual Effects / Creative Reference + Image Reference     ✓            ✗              ✗            ✗
  Creative Ref.
                   Visual Effects / Creative Reference + First Frame        ✓            ✗              ✗            ✗
                    Style Image Reference                        ✓            ✗          ✓         ✓
  Style              Style Image + Subject Image Reference              ✓            ✗          ✓         ✓
  Reference         Style Video Reference                        ✓            ✗          ✓         ✓
                    Style Video + Subject Image Reference              ✓            ✗          ✓         ✓
                 Video Instruction Editing                      ✓         ✓          ✓         ✓
 Video                 Video Reference Image Editing                   ✓         ✓          ✓         ✓  Editing
                  Continuation                             ✓            ✗              ✗            ✗
  Continuation     Continuation + Subject Image Reference             ✓            ✗              ✗            ✗
 / Extension      Extension                               ✓            ✗              ✗            ✗
                  Extension + Subject Image Reference               ✓            ✗              ✗            ✗


Table 25 compares multi-modal task support. Seedance 2.0 supports 20 of 22 input modalities—the broadest
of any model. The two unsupported tasks (subject audio-visual + audio reference, video audio editing)
are unsupported by every model. Three task groups are exclusive to Seedance 2.0: all three visual effects
/ creative reference variants and all four continuation / extension variants, totaling 7 tasks no competitor
can handle. Kling 3 Omni supports 9 of 22, lacking style reference, visual effects / creative reference, and
continuation / extension. Vidu Q2 Pro supports 13 of 22, covering style reference but missing visual effects /
creative reference and continuation / extension. Kling O1 is the most limited at 10 of 22, additionally lacking
video-based subject reference and audio-visual inputs.

Seedance 2.0 achieves the best overall subject reference quality in both appearance and voice across all
multi-modal competitors, with a clear lead in appearance consistency. As shown in Table 26, on image-based
subject reference, Seedance 2.0 scores 2.80 on task following (1–3 scale) with 100% 2-point rate and 80%
3-point rate, ahead of Kling O1 (2.71, 73.68%), Vidu Q2 Pro (2.58, 69.70%), and Kling 3 Omni (2.50, 50%).
The reference alignment gap is wider: Seedance 2.0 scores 3.18 vs. Kling O1 at 2.71 and Vidu Q2 Pro at
1.91, a 1.27-point deficit for the latter. On video-based subject reference, Seedance 2.0 scores 2.95 on task
following with 95% of outputs reaching 3 points, and 3.35 on reference alignment—Vidu Q2 Pro trails at 2.00,
a 1.35-point gap. For first-video reference, Sora 2 leads on task following (3.00, with 100% 3-point rate), while
Seedance 2.0 (2.89) and Kling 3 Omni (2.91) are close behind; on reference alignment, Seedance 2.0 and Sora


                                              20
<a id="page-21"></a>

### PDF 第 21 页

Table 26 R2V subject reference evaluation results. “–” denotes unsupported tasks. Task Fol. (Multimodal Task
Following) is rated 1–3, Ref. Align. (Reference Alignment) is rated 1–5.

                      Subject Ref. Image     Subject Ref. Video    Subject Ref. First Video  Subject Ref. Image & Audio
      Model                  Task Fol.   Ref. Align.  Task Fol.   Ref. Align.  Task Fol.    Ref. Align.   Task Fol.      Ref. Align.

 Veo 3.1 [8]            –          –          –          –          –           –           –            –
  Sora 2 [14]            –          –          –          –         3.00          3.27         –            –
 Wan 2.6 [1]            –          –          –          –          2.68         2.42         –            –
  Kling O1 [11]          2.71        2.71        –          –          –           –           –            –
 Vidu Q2 Pro [18]      2.58        1.91        2.50        2.00        –           –           –            –
  Kling 3 Omni [12]     2.50        2.55        2.67        2.50        2.91         2.82         2.11           2.05
 Seedance 2.0           2.80          3.18         2.95         3.35        2.89          3.27          2.29             2.37


2 tie at 3.27. Image & audio combined reference is supported only by Seedance 2.0 and Kling 3 Omni, with
Seedance 2.0 scoring 2.29 vs. 2.11 on task following—the low absolute scores for both models indicate that
joint image-audio conditioning remains a difficult problem.

Table 27 R2V motion and style reference evaluation results. “–” denotes unsupported tasks. Task Fol. (Multimodal
Task Following) is rated 1–3, Ref. Align. (Reference Alignment) and First Frame Pres. are rated 1–5.

                                      Motion Reference                 Style Reference
              Model                          Task Fol.   Ref. Align.   First Frame Pres.  Task Fol.   Ref. Align.

         Veo 3.1 [8]            –          –             –             –          –
          Sora 2 [14]            –          –             –             –          –
       Wan 2.6 [1]            –          –             –             –          –
          Kling O1 [11]          2.19        1.68            3.46           1.96        1.84
         Vidu Q2 Pro [18]      1.92        1.14            2.57            2.15        1.85
          Kling 3 Omni [12]     2.20        1.97              4.31            –          –
          Seedance 2.0           2.60         2.64            2.71             2.57         2.37


Seedance 2.0 produces the best reference alignment for body motion, visual effects, and creative sequences,
while competitors frequently fail to reproduce referenced effects or capture complete body movements. Table 27
shows Seedance 2.0 scoring 2.60 on motion reference task following, ahead of Kling 3 Omni (2.20), Kling O1
(2.19), and Vidu Q2 Pro (1.92). On reference alignment, the gap widens: Seedance 2.0 scores 2.64, while
all competitors fall below 2.0—Vidu Q2 Pro scores only 1.14, meaning most outputs bear little resemblance
to the reference motion. One exception: on first-frame preservation, Kling 3 Omni scores 4.31, well above
Seedance 2.0’s 2.71. Kling 3 Omni tends to keep the first frame nearly unchanged but produces weaker
subsequent motion, while Seedance 2.0 generates more dynamic video at the cost of lower first-frame fidelity.
On style reference, Seedance 2.0 leads with 2.57 on task following (60% 3-point rate) vs. Vidu Q2 Pro (2.15,
33.33%) and Kling O1 (1.96, 10.71%). Kling 3 Omni does not support style reference at all. Reference
alignment follows the same order: 2.37, 1.85, 1.84. When combining style and subject reference, Seedance 2.0
produces more accurate responses and better generation quality; competitors frequently misinterpret the task
as reference-image editing or produce artifacts, with Kling 3 Omni exhibiting this issue most often.

Video editing is the most competitive R2V task. In Table 28, Kling O1 slightly leads on task following (2.29
vs. Seedance 2.0’s 2.20), and Kling 3 Omni is close at 2.24—all three within 0.09 points. However, Seedance
2.0 pulls ahead on reference alignment (3.79 vs. Kling O1’s 3.03) and editing consistency (3.75 vs. Kling 3
Omni’s 3.09). Seedance 2.0 responds better to long-text and multi-edit instructions and handles complex
editing tasks more completely, while also producing more accurate results for well-known IP references. All
models share common failure modes: unresponsive edits and unintended modifications to non-edit regions.

Video continuation is currently supported only by Seedance 2.0, which scores 2.88 on task following and 3.18
on reference alignment. It handles complex narratives and long-text continuation prompts well, though issues
remain with color consistency, multi-subject omission, and subject duplication.

For video extension, Seedance 2.0 and Veo 3.1 are the only two models evaluated, but they differ in scope:


                                              21
<a id="page-22"></a>

### PDF 第 22 页

Table 28 R2V video editing, continuation, and extension evaluation results. “–” denotes unsupported tasks. Task Fol.
(Multimodal Task Following) is rated 1–3, Ref. Align. (Reference Alignment), Edit. Consist. (Editing Consistency),
and other dimensions are rated 1–5.

                              Video Editing              Video Continuation      Video Extension
      Model                  Task Fol.   Ref. Align.   Edit. Consist.  Task Fol.   Ref. Align.  Task Fol.   Ref. Align.

 Veo 3.1 [8]            –          –           –           –          –           2.78        3.44
 Sora 2 [14]            –          –           –           –          –          –          –
 Wan 2.6 [1]            –          –           –           –          –          –          –
 Kling O1 [11]           2.29        3.03          2.78          –          –          –          –
 Vidu Q2 Pro [18]      2.02        2.58          2.22          –          –          –          –
 Kling 3 Omni [12]     2.24        2.71          3.09          –          –          –          –
 Seedance 2.0          2.20         3.79           3.75          2.88          3.18        1.93        3.28


Seedance 2.0 accepts arbitrary uploaded videos for extension and supports combining extension with subject
image input, while Veo 3.1 can only extend videos it generated itself. Despite broader input support, Seedance
2.0’s extension quality trails Veo 3.1 notably—Veo 3.1 scores 2.78 on task following (88.89% 3-point rate)
vs. Seedance 2.0’s 1.93 (31.82%), and 3.44 vs. 3.28 on reference alignment, making extension Seedance 2.0’s
weakest R2V task.

2.6  Visualization Results

Leveraging significant advancements in foundational capabilities and Multimodal Task Performance, Seedance
2.0 will deliver an entirely new creative experience for users. It is capable of synthesizing temporally precise
and complex interactive scenes with high fidelity. As shown in the first and second row of Figure 4, the
generation process maintains exceptional motion quality by strictly adhering to real-world physical laws of
motion, avoiding physical anomalies commonly observed in earlier AI-generated videos. For example, in the
skating scenario, the model effectively renders a series of highly demanding maneuvers—such as synchronized
take-offs, mid-air rotations, and precise landings. In response to the diversified demands of video content
production, Seedance 2.0 demonstrates exceptional scenario adaptability in the third row of Figure 4. Whether
applied to commercial advertising, television visual effects, video game animation, or explainer videos, the
model consistently delivers high-quality generation results and robust multimodal task performance. By
replacing complex visual effects pipelines and live-action workflows with AI-driven generation, Seedance 2.0
significantly reduces the production costs of professional audio and video content, shortens production cycles,
and empowers both creators and enterprises to more effectively realize their creative visions.


 T2V




    竞技级双⼈花样滑冰现场。开场低机位跟随冰⼑滑⾏，冰屑与反光细节清晰。进⼊旋转段，男选⼿轴线微偏出现失误，旋转节奏短暂塌陷。⼥选⼿迅速调整重⼼，眼神冷静并示意“Stay with me”，主动引导男选⼿重新对⻬节奏。随后⽆缝衔接托举动
    作，线条⼲净稳定。⾼潮为同步跳跃组合，空中姿态笔直，落冰果断，⾳画完美对⻬。⼥选⼿⾝着深蓝花滑裙，男选⼿为竞技运动装。整体呈现从紧张失误到冷静完成⽐赛的完整叙事，体现顶级双⼈花样滑冰中的技术能⼒与⼼理强度。

 T2V





    武侠⻛格视听⼤⽚，⽵林⾥⽩⾐剑客与蓑⾐⼑客对峙。镜头在两⼈之间缓慢推移，焦点在⾬滴和剑柄之间切换，⽓氛压抑到极点，只能听⻅⾬声。突然⼀道惊雷闪过，两⼈同时冲锋，侧拍镜头极速平移，捕捉泥浆⻜溅的脚步。双兵相接瞬间画⾯切
    换为极慢动作，清晰展示⼑剑震⻜⾬⽔形成的圆环激波，以及被剑⽓切断的⽵叶。随后恢复常速两⼈背对背落地，蓑⾐⼑客的⽃笠裂开，画⾯戛然⽽⽌。


 I2V




      The figure in the painting looks guilty — eyes darting left and right, then peeks out beyond the picture frame. Quickly reaches a hand out of the frame, picks up a cola, takes a sip, and breaks into a deeply satisfied expression. Just then, the sound
         of footsteps approaches. The figure hurriedly puts the cola back in its place. At that moment, a Western cowboy walks up, picks up the cola from the glass, and walks away with it. The closing shot pushes in to a top-lit close-up of the cola against
      a pure black background. At the bottom of the frame, stylized artistic subtitles appear alongside a voiceover: "宜⼝可乐，不可不尝！"
              Figure 4 Visualization of text-to-video (T2V) and image-to-video (I2V) generation.



                                              22
<a id="page-23"></a>

### PDF 第 23 页

References

 [1] Alibaba Group. Wan2.6. https://wan.video/introduction/wan2.6, 2025.

 [2] Arena AI. Arena ai leaderboard. https://arena.ai/leaderboard.

 [3] ByteDance Seed Team.    Seed2.0 Model  Card:  Towards  Intelligence  Frontier  for Real-World Com-
     plexity. https://lf3-static.bytednsdoc.com/obj/eden-cn/lapzild-tss/ljhwZthlaukjlkulzlp/seed2/0214/
    Seed2.0%20Model%20Card.pdf, 2026.

 [4] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan
     Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683,
     2025.

 [5] Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao,
    Liyang Liu, et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025.

 [6] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang
     Li, Xiaojie Li, et al.  Seedance 1.0: Exploring the boundaries of video generation models.  arXiv preprint
     arXiv:2506.09113, 2025.

 [7] Lixue Gong, Xiaoxia Hou, Fanshi Li, Liang Li, Xiaochen Lian, Fei Liu, Liyang Liu, Wei Liu, Wei Lu, Yichun
     Shi, et al. Seedream 2.0: A native chinese-english bilingual image generation foundation model. arXiv preprint
     arXiv:2503.07703, 2025.

 [8] Google DeepMind. Veo 3.1. https://deepmind.google/models/veo, 2025.

 [9] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang,
     Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025.

[10] Kuaishou Technology. Kling video 2.6. https://kling.ai, 2025.

[11] Kuaishou Technology. Kling o1. https://kling.ai, 2025.

[12] Kuaishou Technology. Kling 3.0. https://kling.ai, 2026.

[13] Chao Liao, Liyang Liu, Xun Wang, Zhengxiong Luo, Xinyu Zhang, Wenliang Zhao, Jie Wu, Liang Li, Zhi Tian,
    and Weilin Huang. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint
     arXiv:2505.05472, 2025.

[14] OpenAI. Sora 2. https://openai.com/index/sora-2/, 2025.

[15] Team Seawead, Ceyuan Yang, Zhijie Lin, Yang Zhao, Shanchuan Lin, Zhibei Ma, Haoyuan Guo, Hao Chen,
    Lu Qi, Sen Wang, et al. Seaweed-7b: Cost-effective training of video generation foundation model. arXiv preprint
     arXiv:2504.08685, 2025.

[16] Team Seedance, Heyi Chen, Siyan Chen, Xin Chen, Yanfei Chen, Ying Chen, Zhuo Chen, Feng Cheng, Tianheng
    Cheng, Xinqi Cheng, et al. Seedance 1.5 pro: A native audio-visual joint generation foundation model. arXiv
     preprint arXiv:2512.13507, 2025.

[17] Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou,
    Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv
     preprint arXiv:2509.20427, 2025.

[18] ShengShu Technology. Vidu q2 pro. https://www.vidu.com, 2026.

[19] Yichun Shi, Peng Wang, and Weilin Huang. Seededit: Align image re-generation to image editing. arXiv preprint
     arXiv:2411.06686, 2024.

[20] Peng Wang, Yichun Shi, Xiaochen Lian, Zhonghua Zhai, Xin Xia, Xuefeng Xiao, Weilin Huang, and Jianchao
    Yang. Seededit 3.0: Fast and high-quality generative image editing. arXiv preprint arXiv:2506.05083, 2025.

[21] Jie Wu, Yu Gao, Zilyu Ye, Ming Li, Liang Li, Hanzhong Guo, Jie Liu, Zeyue Xue, Xiaoxia Hou, Wei Liu, et al.
    Rewarddance: Reward scaling in visual generation. arXiv preprint arXiv:2509.08826, 2025.

[22] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo,
    Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.




                                              23
<a id="page-24"></a>

### PDF 第 24 页

[23] Yan Zeng, Guoqiang Wei, Jiani Zheng, Jiaxin Zou, Yang Wei, Yuchen Zhang, and Hang Li. Make pixels dance:
    High-dynamic video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
     Recognition, 2024.





                                              24
<a id="page-25"></a>

### PDF 第 25 页

3  Contributions and Acknowledgments

All authors of Seedance-2.0 are listed in alphabetical order by their last names.

Team Seedance            Haifeng Li                    Dongliang Wang
                               Huixia Li                    Feng Wang
                                  Jiashi Li                         Hulin Wang
De Chen                    Ming Li                    Ke Wang
Liyang Chen                    Xiaojie Li                      Qingyi Wang
Xin Chen                      Xingxing Li                     Rui Wang
Ying Chen                Yameng Li                    Shuai Wang
Zhuo Chen                     Yiying Li                        Shulei Wang
Zhuowei Chen              Yu Li                       Weichen Wang
Feng Cheng                Yueyan Li                 Xuanda Wang
Tianheng Cheng             Chao Liang                  Yanhui Wang
Yufeng Cheng              Han Liang                  Yue Wang
Mojie Chi                     Jianzhong Liang              Yuping Wang
Xuyan Chi                    Ying Liang                 Yuxuan Wang
Jian Cong               Wang Liao                          Zijie Wang
Qinpeng Cui                         J. H. Lien                       Ziyu Wang
Fei Ding                    Shanchuan Lin               Guoqiang Wei
Qide Dong                      Xi Lin                    Meng Wei
Yujiao Du                   Feng Ling                       Di Wu
Haojie Duanmu              Yue Ling                   Guohong Wu
Junliang Fan                Fangfang Liu                  Hanjie Wu
Jiarui Fang                    Jiawei Liu                  Huachao Wu
Jing Fang                     Jihao Liu                       Jian Wu
Zetao Fang                    Jingtuo Liu                       Jie Wu
Chengjian Feng              Shu Liu                      Ruolan Wu
Yu Gao                       Sichao Liu                     Shaojin Wu
Diandian Gu                Wei Liu                      Xiaohu Wu
Dong Guo                  Xue Liu                       Xinglong Wu
Hanzhong Guo                 Zuxi Liu                     Yonghui Wu
Qiushan Guo                      Ruijie Lu                        Ruiqi Xia
Boyang Hao                 Lecheng Lyu                   Xin Xia
Hongxiang Hao                 Jingting Ma                 Xuefeng Xiao
Haoxun He                    Tianxiang Ma               Shuang Xu
Jiaao He                    Xiaonan Nie                Bangbang Yang
Qian He                       Jingzhe Ning                     Jiaqi Yang
Tuyen Hoang                    Junjie Pan                    Runkai Yang
Heng Hu                      Xitong Pan                 Tao Yang
Ruoqing Hu                  Ronggui Peng                Yihang Yang
Yuxiang Hu                  Xueqiong Qu                   Zhixian Yang
Jiancheng Huang              Yuxi Ren                      Ziyan Yang
Weilin Huang                Yuchen Shen                  Fulong Ye
Zhaoyang Huang            Guang Shi                    Bingqian Yi
Zhongyi Huang                   Lei Shi                       Xing Yin
Jishuo Jin                     Yinglong Song                Yongbin You
Ming Jing                  Fan Sun                        Linxiao Yuan
Ashley Kim                          Li Sun                     Weihong Zeng
Shanshan Lao                  Renfei Sun                    Xuejiao Zeng
Yichong Leng                Wenjing Tang              Yan Zeng
Bingchuan Li                Boyang Tao                   Siyu Zhai
Gen Li                              Zirui Tao                   Zhonghua Zhai


                                              25
<a id="page-26"></a>

### PDF 第 26 页

Bowen Zhang                 Xiaoying Zhang               Yian Zhao
Chenlin Zhang                Xinyan Zhang               Guangcong Zheng
Heng Zhang                     Xinyi Zhang                    Jianbin Zheng
Jun Zhang                       Yichi Zhang                  Xiaozheng Zheng
Manlin Zhang                  Zixiang Zhang                Zerong Zheng
Peiyuan Zhang               Haiyu Zhao                 Kuan Zhu
Shuo Zhang                  Huating Zhao                   Feilong Zuo
Xiaohe Zhang                 Liming Zhao





                                              26

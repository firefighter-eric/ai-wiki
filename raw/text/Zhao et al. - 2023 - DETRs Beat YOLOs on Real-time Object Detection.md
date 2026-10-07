# Zhao et al. - 2023 - DETRs Beat YOLOs on Real-time Object Detection

- Source PDF: `raw/pdf/Zhao et al. - 2023 - DETRs Beat YOLOs on Real-time Object Detection.pdf`
- Source SHA256: `01018b33f15723f057b1b65c4ae6958e8cc99b75acf1ca434467c9f5f0782f15`
- Generated from: `scripts/extract_pdf_text.py`

- Extraction: `pymupdf-pages-v2` (sorted text, page anchors; tables/formulas/figures require review)

## Extracted Text

<a id="source-section-0"></a>

<a id="page-1"></a>

### PDF 第 1 页

DETRs Beat YOLOs on Real-time Object Detection


              Yian Zhao1,2†  Wenyu Lv1†‡   Shangliang Xu1   Jinman Wei1   Guanzhong Wang1
                                 Qingqing Dang1   Yi Liu1   Jie Chen2

                   1Baidu Inc, Beijing, China   2School of Electronic and Computer Engineering, Peking University, Shenzhen, China

                                   zhaoyian@stu.pku.edu.cn  lvwenyu01@baidu.com   jiechen2019@pku.edu.cn

                        Abstract                                MS COCO Object Detection

                                                                                                               R101
          The YOLO series has become the most popular frame-        54                               X                                                                                                   R502024                                                                                               L             X        work for real-time object detection due to its reasonable                                                                                         L                                                                                            X
          trade-off between speed and accuracy. However, we observe
                                                                         52        R50        L            L
          that the speed and accuracy of YOLOs are negatively af-                      Scaled                           XApr          fected by the NMS. Recently, end-to-end Transformer-based      (%)           M  M                                                                         50
3   detectors (DETRs) have provided an alternative to eliminat-    AP          M                  L
         ing NMS. Nevertheless, the high computational cost limits
          their practicality and hinders them from fully exploiting the         48                                                                                                        COCO
        advantage of excluding NMS. In this paper, we propose the               R18                                YOLOv5
        Real-Time DEtection TRansformer (RT-DETR), the first         46  Scaled              M         PP-YOLOE
                                                                                               S    S                            YOLOv6-v3.0
          real-time end-to-end object detector to our best knowledge                                                                                                                 YOLOv7[cs.CV]   that addresses the above dilemma. We build RT-DETR in         44                                     YOLOv8
                                                                                              S                               RT-DETR (Ours)        two steps, drawing on the advanced DETR: first we focus
        on maintaining accuracy while improving speed, followed                                                                                 4      8      12     16     20     24
        by maintaining speed while improving accuracy.  Specifi-                   End-to-end Latency T4 TensorRT FP16 (ms)
           cally, we design an efficient hybrid encoder to expeditiously                                                                             Figure 1. Compared to previously advanced real-time object detec-
         process multi-scale features by decoupling intra-scale inter-       tors, our RT-DETR achieves state-of-the-art performance.
         action and cross-scale fusion to improve speed. Then, we
         propose the uncertainty-minimal query selection to provide      to their reasonable trade-off between speed and accuracy.
         high-quality initial queries to the decoder, thereby improv-    However, these detectors typically require Non-Maximum
         ing accuracy. In addition, RT-DETR supports flexible speed     Suppression (NMS) for post-processing, which not only
         tuning by adjusting the number of decoder layers to adapt     slows down the inference speed but also introduces hyperpa-
          to various scenarios without retraining. Our RT-DETR-R50     rameters that cause instability in both the speed and accuracy.
              / R101 achieves 53.1% / 54.3% AP on COCO and 108 / 74     Moreover, considering that different scenarios place different
       FPS on T4 GPU, outperforming previously advanced YOLOs     emphasis on recall and accuracy, it is necessary to carefully
          in both speed and accuracy. Furthermore, RT-DETR-R50      select the appropriate NMS thresholds, which hinders thearXiv:2304.08069v3   outperforms DINO-R50 by 2.2% AP in accuracy and about     development of real-time detectors.
        21 times in FPS. After pre-training with Objects365, RT-        Recently,  the  end-to-end  Transformer-based  detec-
       DETR-R50 / R101 achieves 55.3% / 56.2% AP. The project      tors (DETRs) [4, 17, 23, 27, 36, 39, 44, 45] have received ex-
         page: https://zhao-yian.github.io/RTDETR.                      tensive attention from the academia due to their streamlined
                                                                           architecture and elimination of hand-crafted components.
         1. Introduction                                      However, their high computational cost prevents them from
                                                                 meeting real-time detection requirements, so the NMS-free
         Real-time object detection is an important area of research                                                                             architecture does not demonstrate an inference speed advan-
        and has a wide range of applications, such as object track-                                                                             tage. This inspires us to explore whether DETRs can be
         ing [43], video surveillance [28], and autonomous driv-                                                                    extended to real-time scenarios and outperform the advanced
         ing [2], etc. Existing real-time detectors generally adopt                                            YOLO detectors in both speed and accuracy, eliminating the
         the CNN-based architecture, the most famous of which is                                                                     delay caused by NMS for real-time object detection.
         the YOLO detectors [1, 10–12, 15, 16, 25, 30, 38, 40] due                                                              To achieve the above goal, we rethink DETRs and conduct
                 Corresponding author. †Equal contribution. ‡ Project leader.          detailed analysis of key components to reduce unnecessary
<a id="page-2"></a>

### PDF 第 2 页

computational redundancy and further improve accuracy.      The main contributions are summarized as: (i). We pro-
For the former, we observe that although the introduction     pose the first real-time end-to-end object detector called RT-
of multi-scale features is beneficial in accelerating the train-    DETR, which not only outperforms the previously advanced
ing convergence [45], it leads to a significant increase in   YOLO detectors in both speed and accuracy but also elimi-
the length of the sequence feed into the encoder. The high     nates the negative impact caused by NMS post-processing
computational cost caused by the interaction of multi-scale     on real-time object detection; (ii). We quantitatively analyze
features makes the Transformer encoder the computational     the impact of NMS on the speed and accuracy of YOLO
bottleneck. Therefore, implementing the real-time DETR      detectors, and establish an end-to-end speed benchmark to
requires a redesign of the encoder. And for the latter, pre-      test the end-to-end inference speed of real-time detectors;
vious works [42, 44, 45] show that the hard-to-optimize ob-       (iii). The proposed RT-DETR supports flexible speed tuning
ject queries hinder the performance of DETRs and propose     by adjusting the number of decoder layers to accommodate
the query selection schemes to replace the vanilla learnable     various scenarios without retraining.
embeddings with encoder features. However, we observe
that the current query selection directly adopt classification     2. Related Work
scores for selection, ignoring the fact that the detector are
required to simultaneously model the category and location     2.1. Real-time Object Detectors
of objects, both of which determine the quality of the fea-                                         YOLOv1 [31] is the first CNN-based one-stage object de-
tures. This inevitably results in encoder features with low                                                                   tector to achieve true real-time object detection. Through
localization confidence being selected as initial queries, thus                                                              years of continuous development, the YOLO detectors have
leading to a considerable level of uncertainty and hurting the                                                         outperformed other one-stage object detectors [21, 24] and
performance of DETRs. We view query initialization as a                                                 become the synonymous with the real-time object detec-
breakthrough to further improve performance.                                                                              tor. YOLO detectors can be classified into two categories:
   In this paper, we propose the Real-Time DEtection     anchor-based [1, 11, 15, 25, 29, 30, 37, 38] and anchor-
TRansformer (RT-DETR), the first real-time end-to-end ob-     free [10, 12, 16, 40], which achieve a reasonable trade-off
ject detector to our best knowledge. To expeditiously process     between speed and accuracy and are widely used in vari-
multi-scale features, we design an efficient hybrid encoder to     ous practical scenarios. These advanced real-time detectors
replace the vanilla Transformer encoder, which significantly     produce numerous overlapping boxes and require NMS post-
improves inference speed by decoupling the intra-scale in-     processing, which slows down their speed.
teraction and cross-scale fusion of features with different
scales. To avoid encoder features with low localization con-     2.2. End-to-end Object Detectors
fidence being selected as object queries, we propose the                                                           End-to-end object detectors are well-known for their stream-
uncertainty-minimal query selection, which provides high-                                                                lined pipelines. Carion et al. [4] first propose the end-to-
quality initial queries to the decoder by explicitly optimizing                                                       end detector based on Transformer called DETR, which has
the uncertainty, thereby increasing the accuracy. Further-                                                                   attracted extensive attention due to its distinctive features.
more, RT-DETR supports flexible speed tuning to accommo-                                                                     Particularly, DETR eliminates the hand-crafted anchor and
date various real-time scenarios without retraining, thanks                                   NMS components. Instead, it employs bipartite matching
to the multi-layer decoder architecture of DETR.                                                     and directly predicts the one-to-one object set. Despite its
  RT-DETR achieves an ideal trade-off between the speed     obvious advantages, DETR suffers from several problems:
and accuracy. Specifically, RT-DETR-R50 achieves 53.1%     slow training convergence, high computational cost, and
AP on COCO val2017 and 108 FPS on T4 GPU, while RT-     hard-to-optimize queries. Many DETR variants have been
DETR-R101 achieves 54.3% AP and 74 FPS, outperforming     proposed to address these issues. Accelerating convergence.
L and X models of previously advanced YOLO detectors in    Deformable-DETR [45] accelerates training convergence
both speed and accuracy, Figure 1. We also develop scaled     with multi-scale features by enhancing the efficiency of the
RT-DETRs by scaling the encoder and decoder with smaller      attention mechanism. DAB-DETR [23] and DN-DETR [17]
backbones, which outperform the lighter YOLO detectors (S      further improve performance by introducing the iterative
and M models). Furthermore, RT-DETR-R50 outperforms     refinement scheme and denoising training. Group-DETR [5]
DINO-Deformable-DETR-R50 by 2.2% AP (53.1% AP vs     introduces group-wise one-to-many assignment. Reduc-
50.9% AP) in accuracy and by about 21 times in FPS (108     ing computational cost. Efficient DETR [42] and Sparse
FPS vs 5 FPS), significantly improves accuracy and speed   DETR [33] reduce the computational cost by reducing the
of DETRs.  After pre-training with Objects365 [35], RT-    number of encoder and decoder layers or the number of
DETR-R50 / R101 achieves 55.3% / 56.2% AP, resulting in     updated queries. Lite DETR [18] enhances the efficiency
surprising performance improvements. More experimental     of encoder by reducing the update frequency of low-level
results are provided in the Appendix.                            features in an interleaved way. Optimizing query initial-
<a id="page-3"></a>

### PDF 第 3 页

10k                                              IoU thr.   AP NMS   Conf thr.  AP NMS
                         YOLOv5 (anchor-based)                                                               (Conf=0.001)  (%)   (ms)     (IoU=0.7)  (%)   (ms)
                         YOLOv8 (anchor-free)
    8k                                                              0.5      52.1   2.24      0.001    52.9   2.36
boxes 6k                                                              0.6      52.6   2.29       0.01     52.4   1.73
of                                                                 0.8      52.8   2.46       0.05     51.2   1.06

    4k                                                                Table 1. The effect of IoU threshold and confidence threshold onNumber                                                                 accuracy and NMS execution time.
    2k                                                        the NMS operation under different hyperparameters. Note
                                                                    that the NMS operation we adopt refers to the TensorRT
                                            efficientNMSPlugin, which involves multiple ker-
     0
           0.001   0.005    0.01    0.05     0.1     0.25           nels, including EfficientNMSFilter, RadixSort,
                      Confidence threshold               EfficientNMS, etc., and we only report the execution
Figure 2. The number of boxes at different confidence thresholds.     time of the EfficientNMS kernel. We test the speed
                                                   on T4 GPU with TensorRT FP16, and the input and pre-ization. Conditional DETR [27] and Anchor DETR [39]
                                                            processing remain consistent. The hyperparameters anddecrease the optimization difficulty of the queries. Zhu et
                                                               the corresponding results are shown in  Table 1. Fromal. [45] propose the query selection for two-stage DETR, and
                                                               the results, we can conclude that the execution time of theDINO [44] suggests the mixed query selection to help better
                                           EfficientNMS kernel increases as the confidence thresh-initialize queries. Current DETRs are still computationally
                                                            old decreases or the IoU threshold increases. The reason isintensive and are not designed to detect in real time. Our
                                                                    that the high confidence threshold directly filters out moreRT-DETR vigorously explores computational cost reduction
                                                                prediction boxes, whereas the high IoU threshold filters outand attempts to optimize query initialization, outperforming
                                                          fewer prediction boxes in each round of screening. We alsostate-of-the-art real-time detectors.
                                                                visualize the predictions of YOLOv8 with different NMS
3. End-to-end Speed of Detectors                       thresholds in Appendix. The results show that inappropriate
                                                           confidence thresholds lead to significant false positives or
3.1. Analysis of NMS                                           false negatives by the detector. With a confidence threshold
                                                             of 0.001 and an IoU threshold of 0.7, YOLOv8 achieves
NMS is a widely used post-processing algorithm in object
                                                               the best AP results, but the corresponding NMS time is at
detection, employed to eliminate overlapping output boxes.
                                                         a higher level. Considering that YOLO detectors typically
Two thresholds are required in NMS: confidence threshold
                                                                 report the model speed and exclude the NMS time, thus an
and IoU threshold. Specifically, the boxes with scores be-
                                                            end-to-end speed benchmark needs to be established.
low the confidence threshold are directly filtered out, and
whenever the IoU of any two boxes exceeds the IoU thresh-     3.2. End-to-end Speed Benchmark
old, the box with the lower score will be discarded. This
                                                  To enable a fair comparison of the end-to-end speed of var-process is performed iteratively until all boxes of every cate-
                                                             ious real-time detectors, we establish an end-to-end speedgory have been processed. Thus, the execution time of NMS
                                                       benchmark. Considering that the execution time of NMS isprimarily depends on the number of boxes and two thresh-
                                                             influenced by the input, it is necessary to choose a bench-olds. To verify this observation, we leverage YOLOv5 [11]
                                                    mark dataset and calculate the average execution time across(anchor-based) and YOLOv8 [12] (anchor-free) for analysis.
                                                             multiple images. We choose COCO val2017 [20] as the  We first count the number of boxes remaining after fil-
                                                    benchmark dataset and append the NMS post-processingtering the output boxes with different confidence thresholds
on the same input. We sample values from 0.001 to 0.25     plugin of TensorRT for YOLO detectors as mentioned above.
                                                                    Specifically, we test the average inference time of the de-as confidence thresholds to count the number of remaining
                                                                   tector according to the NMS thresholds of the correspond-boxes of the two detectors and plot them on a bar graph,
                                                            ing accuracy taken on the benchmark dataset, excludingwhich intuitively reflects that NMS is sensitive to its hyper-
                                           I/O and MemoryCopy operations. We utilize the bench-parameters, Figure 2. As the confidence threshold increases,
                                                   mark to test the end-to-end speed of anchor-based detectorsmore prediction boxes are filtered out, and the number of
                                          YOLOv5 [11] and YOLOv7 [38], as well as anchor-free de-remaining boxes that need to calculate IoU decreases, thus
                                                                   tectors PP-YOLOE [40], YOLOv6 [16] and YOLOv8 [12]reducing the execution time of NMS.
   Furthermore, we use YOLOv8 to evaluate the accuracy         https://github.com/NVIDIA/TensorRT/tree/release/8.6/
on the COCO val2017 and test the execution time of     plugin/efficientNMSPlugin
<a id="page-4"></a>

### PDF 第 4 页

Concat                                                      CSF            CCFF                                 MSE


                        SSE                                                          SSE               AIFI                                              Concat       Concat

      A    Intra-scale    B    Cross-scale   C   Decoupled    D   Enhanced     E

Figure 3. The encoder structure for each variant. SSE represents the single-scale Transformer encoder, MSE represents the multi-scale
Transformer encoder, and CSF represents cross-scale fusion. AIFI and CCFF are the two modules designed into our hybrid encoder.

on T4 GPU with TensorRT FP16.  According to the re-     neous intra-scale and cross-scale feature interaction is ineffi-
sults (cf. Table 2), we conclude that anchor-free detectors      cient, Figure 3. Specially, we use DINO-Deformable-R50
outperform anchor-based detectors with equivalent accu-     with the smaller size data reader and lighter decoder used in
racy for YOLO detectors because the former require less    RT-DETR for experiments and first remove the multi-scale
NMS time than the latter. The reason is that anchor-based     Transformer encoder in DINO-Deformable-R50 as variant A.
detectors produce more prediction boxes than anchor-free     Then, different types of the encoder are inserted to produce a
detectors (three times more in our tested detectors).               series of variants based on A, elaborated as follows (Detailed
                                                                  indicators of each variant are referred to in Table 3):
4. The Real-time DETR                                        • A →B: Variant B inserts a single-scale Transformer en-
                                                            coder into A, which uses one layer of Transformer block.
4.1. Model Overview
                                                   The multi-scale features share the encoder for intra-scale
RT-DETR consists of a backbone, an efficient hybrid en-       feature interaction and then concatenate as output.
coder, and a Transformer decoder with auxiliary prediction      • B →C: Variant C introduces cross-scale feature fusion
heads. The overview of RT-DETR is illustrated in Figure 4.      based on B and feeds the concatenated features into the
Specifically, we feed the features from the last three stages        multi-scale Transformer encoder to perform simultaneous
of the backbone {S3, S4, S5} into the encoder. The effi-        intra-scale and cross-scale feature interaction.
cient hybrid encoder transforms multi-scale features into a      • C →D: Variant D decouples intra-scale interaction and
sequence of image features through intra-scale feature inter-        cross-scale fusion by utilizing the single-scale Transformer
action and cross-scale feature fusion (cf. Sec. 4.2). Subse-      encoder for the former and a PANet-style [22] structure
quently, the uncertainty-minimal query selection is employed        for the latter.
to select a fixed number of encoder features to serve as ini-      • D →E: Variant E enhances the intra-scale interaction and
tial object queries for the decoder (cf. Sec. 4.3). Finally, the        cross-scale fusion based on D, adopting an efficient hybrid
decoder with auxiliary prediction heads iteratively optimizes       encoder designed by us.
object queries to generate categories and boxes.             Hybrid design. Based on the above analysis, we rethink
                                                               the structure of the encoder and propose an efficient hybrid
4.2. Efficient Hybrid Encoder
                                                           encoder, consisting of two modules, namely the Attention-
Computational bottleneck analysis. The introduction of     based Intra-scale Feature Interaction (AIFI) and the CNN-
multi-scale features accelerates training convergence and im-     based Cross-scale Feature Fusion (CCFF). Specifically, AIFI
proves performance [45]. However, although the deformable      further reduces the computational cost based on variant D
attention reduces the computational cost, the sharply in-    by performing the intra-scale interaction only on S5 with
creased sequence length still causes the encoder to become     the single-scale Transformer encoder. The reason is that
the computational bottleneck. As reported in Lin et al. [19],     applying the self-attention operation to high-level features
the encoder accounts for 49% of the GFLOPs but contributes     with richer semantic concepts captures the connection be-
only 11% of the AP in Deformable-DETR. To overcome this     tween conceptual entities, which facilitates the localization
bottleneck, we first analyze the computational redundancy     and recognition of objects by subsequent modules. How-
present in the multi-scale Transformer encoder. Intuitively,      ever, the intra-scale interactions of lower-level features are
high-level features that contain rich semantic information     unnecessary due to the lack of semantic concepts and the
about objects are extracted from low-level features, making it      risk of duplication and confusion with high-level feature in-
redundant to perform feature interaction on the concatenated      teractions. To verify this opinion, we perform the intra-scale
multi-scale features. Therefore, we design a set of variants      interaction only on S5 in variant D, and the experimental
with different types of the encoder to prove that the simulta-      results are reported in Table 3 (see row DS5). Compared to
<a id="page-5"></a>

### PDF 第 5 页

Efficient Hybrid Encoder                                              Conv1x1 s1       Conv3x3 s2
                                                                                    BN           BN
                                                                                                        SiLU            SiLU                                  F5                                                                                                Selection
                                                     Fusion
                                                                                                                     Query             Head
                         AIFI                &
                                                     Fusion    C
                                                                                                                                                                                                 Decoder
                         S5                                                                                                    Position Embedding
                       S4                         Fusion                                                                                                                Image Feature                     S3                                                                                                                                                                                                                                                                                                        Uncertainty-minimal                                                                                                                            Object Query

Figure 4. Overview of RT-DETR. We feed the features from the last three stages of the backbone into the encoder. The efficient hybrid
encoder transforms multi-scale features into a sequence of image features through the Attention-based Intra-scale Feature Interaction (AIFI)
and the CNN-based Cross-scale Feature Fusion (CCFF). Then, the uncertainty-minimal query selection selects a fixed number of encoder
features to serve as initial object queries for the decoder. Finally, the decoder with auxiliary prediction heads iteratively optimizes object
queries to generate categories and boxes.
                    c               Fusion         The confidence score represents the likelihood that the fea-  c           1x1 Conv                                        ture includes foreground objects. Nevertheless, the detector
                                  c                are required to simultaneously model the category and loca-    C                             F             tion of objects, both of which determine the quality of the
  c               N×                            features. Hence, the performance score of the feature is a la-                    c              1x1 Conv    RepBlock                          tent variable that is jointly correlated with both classification
                                                     and localization. Based on the analysis, the current query
   C Concatenate     Element-wise add  F Flatten          selection lead to a considerable level of uncertainty in the
                                                                selected features, resulting in sub-optimal initialization for
              Figure 5. The fusion block in CCFF.                                                                the decoder and hindering the performance of the detector.
D, DS5 not only significantly reduces latency (35% faster),       To address this problem, we propose the uncertainty mini-
but also improves accuracy (0.4% AP higher). CCFF is opti-    mal query selection scheme, which explicitly constructs and
mized based on the cross-scale fusion module, which inserts     optimizes the epistemic uncertainty to model the joint latent
several fusion blocks consisting of convolutional layers into      variable of encoder features, thereby providing high-quality
the fusion path. The role of the fusion block is to fuse two      queries for the decoder. Specifically, the feature uncertainty
adjacent scale features into a new feature, and its structure is    U is defined as the discrepancy between the predicted dis-
illustrated in Figure 5. The fusion block contains two 1 × 1      tributions of localization P and classification C in Eq. (2).
convolutions to adjust the number of channels, N RepBlocks    To minimize the uncertainty of the queries, we integrate
composed of RepConv [8] are used for feature fusion, and      the uncertainty into the loss function for the gradient-based
the two-path outputs are fused by element-wise add. We     optimization in Eq. (3).
formulate the calculation of the hybrid encoder as:
                                                       U(X)ˆ = ∥P(X)ˆ −C(X)∥,ˆ  Xˆ ∈RD         (2)
     Q = K = V = Flatten(S5),
        F5 = Reshape(AIFI(Q, K, V)),         (1)         L(X,ˆ ˆY, Y) = Lbox(ˆb, b) + Lcls(U(X),ˆ   ˆc, c)   (3)
     O = CCFF({S3, S4, F5}),                                                    where ˆY and Y denote the prediction and ground truth,
where Reshape represents restoring the shape of the flat-     ˆY = {ˆc, ˆb}, ˆc and ˆb represent the category and bounding
tened feature to the same shape as S5.                     box respectively, Xˆ represent the encoder feature.
                                                            Effectiveness analysis. To analyze the effectiveness of the
4.3. Uncertainty-minimal Query Selection
                                                            uncertainty-minimal query selection, we visualize the clas-
To reduce the difficulty of optimizing object queries in      sification scores and IoU scores of the selected features on
DETR, several subsequent works [42, 44, 45] propose query   COCO val2017, Figure 6. We draw the scatterplot with
selection schemes, which have in common that they use the      classification scores greater than 0.5. The purple and green
confidence score to select the top K features from the en-     dots represent the selected features from the model trained
coder to initialize object queries (or just position queries).     with uncertainty-minimal query selection and vanilla query
<a id="page-6"></a>

### PDF 第 6 页

The width and depth of the decoder can be controlled by
                                                           manipulating the number of object queries and decoder lay-
                                                                        ers. Furthermore, the speed of RT-DETR supports flexible     1.0
                 Uncertainty-minimal                                 adjustment by adjusting the number of decoder layers. We
                   Vanilla                                          observe that removing a few decoder layers at the end has
     0.9                                                 minimal effect on accuracy, but greatly enhances inference
                                                        speed (cf. Sec. 5.4). We compare the RT-DETR equipped
  score 0.8                                                    withmodelsResNet50of YOLOanddetectors.ResNet101Lighter[13,RT-DETRs14] to the canL andbe de-X
                                                           signed by applying other smaller (e.g., ResNet18/34) or
                                                               scalable (e.g., CSPResNet [40]) backbones with scaled en-     0.7
                                                           coder and decoder. We compare the scaled RT-DETRs with      Classification
                                                                the lighter (S and M) YOLO detectors in Appendix, which
     0.6                                                     outperform all S and M models in both speed and accuracy.

                                                          5. Experiments
     0.5
         0.0       0.2       0.4       0.6       0.8       1.0               5.1. Comparison with SOTA
                        IoU score
Figure 6. Classification and IoU scores of the selected encoder     Table 2 compares RT-DETR with current real-time (YOLOs)
features. Purple and Green dots represent the selected features     and end-to-end (DETRs) detectors, where only the L and
from model trained with uncertainty-minimal query selection and   X models of the YOLO detector are compared, and the S
vanilla query selection, respectively.                          and M models are compared in Appendix. Our RT-DETR
                                                     and YOLO detectors share a common input size of (640,selection, respectively. The closer the dot is to the top right
                                                                640), and other DETRs use an input size of (800, 1333). Theof the figure, the higher the quality of the corresponding
                                              FPS is reported on T4 GPU with TensorRT FP16, and forfeature, i.e., the more likely the predicted category and box
                                      YOLO detectors using official pre-trained models accordingare to describe the true object. The top and right density
                                                                    to the end-to-end speed benchmark proposed in Sec. 3.2. Ourcurves reflect the number of dots for two types.
                                               RT-DETR-R50 achieves 53.1% AP and 108 FPS, while RT-
  The most striking feature of the scatterplot is that the pur-
                                               DETR-R101 achieves 54.3% AP and 74 FPS, outperforming
ple dots are concentrated in the top right of the figure, while
                                                                       state-of-the-art YOLO detectors of similar scale and DETRs
the green dots are concentrated in the bottom right. This
                                                         with the same backbone in both speed and accuracy. The
shows that uncertainty-minimal query selection produces
                                                            experimental settings are shown in Appendix.
more high-quality encoder features. Furthermore, we per-
                                                Comparison with real-time detectors.  We compare
form quantitative analysis on two query selection schemes.
                                                               the end-to-end speed (cf.  Sec. 3.2) and accuracy of RT-
There are 138% more purple dots than green dots, i.e., more
                                       DETR with YOLO detectors. We compare RT-DETR with
green dots with a classification score less than or equal to
                                         YOLOv5 [11], PP-YOLOE [40], YOLOv6v3.0 [16] (here-
0.5, which can be considered low-quality features. And there
                                                                    inafter  referred  to  as YOLOv6), YOLOv7  [38] and
are 120% more purple dots than green dots with both scores
                                         YOLOv8 [12]. Compared to YOLOv5-L / PP-YOLOE-L /
greater than 0.5. The same conclusion can be drawn from
                                              YOLOv6-L, RT-DETR-R50 improves accuracy by 4.1% /
the density curves, where the gap between purple and green
                                                 1.7% / 0.3% AP, increases FPS by 100.0% / 14.9% / 9.1%,
is most evident in the top right of the figure. Quantitative re-
                                                     and reduces the number of parameters by 8.7% / 19.2%
sults further demonstrate that the uncertainty-minimal query
                                                                                            / 28.8%. Compared to YOLOv5-X / PP-YOLOE-X, RT-
selection provides more features with accurate classifica-
                                             DETR-R101 improves accuracy by 3.6% / 2.0%, increases
tion and precise location for queries, thereby improving the
                                              FPS by 72.1%  / 23.3%, and reduces the number of pa-
accuracy of the detector (cf. Sec. 5.3).
                                                           rameters by 11.6% / 22.4%. Compared to YOLOv7-L /
                                              YOLOv8-L, RT-DETR-R50 improves accuracy by 1.9% /
4.4. Scaled RT-DETR
                                                 0.2% AP and increases FPS by 96.4% / 52.1%. Compared
Since real-time detectors typically provide models at differ-     to YOLOv7-X / YOLOv8-X, RT-DETR-R101 improves ac-
ent scales to accommodate different scenarios, RT-DETR     curacy by 1.4% / 0.4% AP and increases FPS by 64.4% /
also supports flexible scaling. Specifically, for the hybrid     48.0%. This shows that our RT-DETR achieves state-of-the-
encoder, we control the width by adjusting the embedding      art real-time detection performance.
dimension and the number of channels, and the depth by    Comparison with end-to-end detectors. We also compare
adjusting the number of Transformer layers and RepBlocks.    RT-DETR with existing DETRs using the same backbone.
<a id="page-7"></a>

### PDF 第 7 页

Model                      Backbone  #Epochs  #Params (M)  GFLOPs  FPSbs=1  APval  APval50   APval75   APvalS   APvalM   APvalL

 Real-time Object Detectors
 YOLOv5-L [11]                       -            300         46         109       54       49.0    67.3        -          -          -          -
 YOLOv5-X [11]                      -            300         86         205       43       50.7    68.9        -          -          -          -
 PPYOLOE-L [40]                    -            300         52         110       94       51.4    68.9    55.6    31.4    55.3    66.1
 PPYOLOE-X [40]                    -            300         98         206       60       52.3    69.9    56.5    33.3    56.3    66.4
 YOLOv6-L [16]                       -            300         59         150       99       52.8    70.3    57.7    34.4    58.1    70.1
 YOLOv7-L [38]                       -            300         36         104       55       51.2    69.7    55.5    35.2    55.9    66.7
 YOLOv7-X [38]                      -            300         71         189       45       52.9    71.1    57.4    36.9    57.7    68.6
 YOLOv8-L [12]                       -                    -          43         165       71       52.9    69.8    57.5    35.3    58.3    69.8
 YOLOv8-X [12]                      -                    -          68         257       50       53.9    71.0    58.7    35.7    59.3    70.7

 End-to-end Object Detectors
 DETR-DC5 [4]              R50         500         41         187            -       43.3    63.1    45.9    22.5    47.3    61.1
 DETR-DC5 [4]               R101        500         60         253            -       44.9    64.7    47.7    23.7    49.5    62.3
 Anchor-DETR-DC5 [39]       R50          50         39         172            -       44.2    64.7    47.5    24.7    48.2    60.6
 Anchor-DETR-DC5 [39]        R101         50               -                -             -       45.1    65.7    48.8    25.8    49.4    61.6
 Conditional-DETR-DC5 [27]    R50         108         44         195            -       45.1    65.4    48.5    25.3    49.0    62.2
 Conditional-DETR-DC5 [27]    R101        108         63         262            -       45.9    66.8    49.5    27.2    50.3    63.3
 Efficient-DETR [42]           R50          36         35         210            -       45.1    63.1    49.1    28.3    48.4    59.0
 Efficient-DETR [42]           R101         36         54         289            -       45.7    64.1    49.5    28.2    49.1    60.2
 SMCA-DETR [9]            R50         108         40         152            -       45.6    65.5    49.1    25.9    49.3    62.6
 SMCA-DETR [9]             R101        108         58         218            -       46.3    66.6    50.2    27.2    50.5    63.2
 Deformable-DETR [45]        R50          50         40         173            -       46.2    65.2    50.0    28.8    49.2    61.7
 DAB-Deformable-DETR [23]    R50          50         48         195            -       46.9    66.0    50.8    30.1    50.4    62.5
 DAB-Deformable-DETR++ [23]  R50          50         47               -             -       48.7    67.2    53.0    31.4    51.6    63.9
 DN-Deformable-DETR [17]     R50          50         48         195            -       48.6    67.4    52.7    31.0    52.0    63.7
 DN-Deformable-DETR++ [17]   R50          50         47               -             -       49.5    67.6    53.8    31.3    52.6    65.4
 DINO-Deformable-DETR [44]   R50          36         47         279        5       50.9    69.0    55.3    34.6    54.1    64.6

 Real-time End-to-end Object Detector (ours)
 RT-DETR                  R50          72         42         136       108      53.1    71.3    57.7    34.8    58.0    70.0
 RT-DETR                   R101         72         76         259       74       54.3    72.7    58.6    36.0    58.8    72.1

Table 2. Comparison with SOTA (only L and X models of YOLO detectors, see Appendix for the comparison with S and M models). We
do not test the speed of other DETRs, except for DINO-Deformable-DETR [44] for comparison, as they are not real-time detectors. Our
RT-DETR outperforms the state-of-the-art YOLO detectors and DETRs in both speed and accuracy.

We test the speed of DINO-Deformable-DETR [44] accord-     a 0.8% AP improvement over C, but reduces latency by 8%,
ing to the settings of the corresponding accuracy taken on     suggesting that decoupling intra-scale interaction and cross-
COCO val2017 for comparison, i.e., the speed is tested      scale fusion not only reduces computational cost but also
with TensorRT FP16 and the input size is (800, 1333). Ta-     improves accuracy. Compared to variant D, DS5 reduces the
ble 2 shows that RT-DETR outperforms all DETRs with the      latency by 35% but delivers 0.4% AP improvement, demon-
same backbone in both speed and accuracy. Compared to      strating that intra-scale interactions of lower-level features
DINO-Deformable-DETR-R50, RT-DETR-R50 improves     are not required. Finally, variant E delivers 1.5% AP im-
the accuracy by 2.2% AP and the speed by 21 times (108     provement over D. Despite a 20% increase in the number
FPS vs 5 FPS), both of which are significantly improved.       of parameters, the latency is reduced by 24%, making the
                                                         encoder more efficient. This shows that our hybrid encoder
5.2. Ablation Study on Hybrid Encoder                  achieves a better trade-off between speed and accuracy.

We  evaluate  the  indicators  of  the  variants  designed                                                               5.3. Ablation Study on Query Selection
in Sec. 4.2, including AP (trained with 1× configuration), the
number of parameters, and the latency, Table 3. Compared   We conduct an ablation study on uncertainty-minimal query
to baseline A, variant B improves accuracy by 1.9% AP and      selection, and the results are reported on RT-DETR-R50 with
increases the latency by 54%. This proves that the intra-   1× configuration, Table 4. The query selection in RT-DETR
scale feature interaction is significant, but the single-scale      selects the top K (K = 300) encoder features according
Transformer encoder is computationally expensive. Variant      to the classification scores as the content queries, and the
C delivers a 0.7% AP improvement over B and increases      prediction boxes corresponding to the selected features are
the latency by 20%. This shows that the cross-scale feature     used as initial position queries. We compare the encoder
fusion is also necessary but the multi-scale Transformer en-     features selected by the two query selection schemes on
coder requires higher computational cost. Variant D delivers   COCO val2017 and calculate the proportions of classi-
<a id="page-8"></a>

### PDF 第 8 页

AP      #Params      Latency                       AP(%)               Latency
   Variant                                        ID
               (%)         (M)            (ms)                  Det4    Det5    Det6    Det7       (ms)

    A         43.0         31             7.2            7         -           -           -      52.6        9.6
    B          44.9         32            11.1            6         -           -      53.1     52.6        9.3
    C          45.6         32            13.3            5         -      52.9     53.0     52.5        8.8
    D         46.4         35            12.2            4     52.7     52.7     52.7     52.1        8.3
    DS5         46.8         35             7.9            3     52.4     52.3     52.4     51.5        7.9
     E          47.9         42             9.3            2     51.6     51.3     51.3     50.6        7.5
                                                       1     49.6     48.8     49.1     48.3        7.0
Table 3. The indicators of the set of variants illustrated in Figure 3.

                                                                Table 5. Results of the ablation study on decoder. ID indicates
                 AP   Propcls↑   Propboth↑   Query selection                                            decoder layer index. Detk represents detector with k decoder layers.
                      (%)     (%)       (%)                                                                    All results are reported on RT-DETR-R50 with 6× configuration.

        Vanilla          47.9     0.35        0.30
 Uncertainty-minimal   48.7     0.82        0.67          than the highest APvalS  in the L model (YOLOv8-L) and RT-
                                               DETR-R101 is 0.9% AP lower than the highest APvalS  in the
Table 4.  Results of the ablation study on uncertainty-minimal   X model (YOLOv7-X). We hope that this problem will be
query selection. Propcls and Propboth represent the proportion of                                                           addressed in future work.
classification score and both scores greater than 0.5 respectively.
                                                           Discussion. Existing large DETR models [3, 6, 32, 41, 44,
fication scores greater than 0.5 and both classification and                                                          46] have demonstrated impressive performance on COCO
IoU scores greater than 0.5, respectively. The results show                                           test-dev [20] leaderboard. The proposed RT-DETR at
that the encoder features selected by uncertainty-minimal                                                                   different scales preserves decoders homogeneous to other
query selection not only increase the proportion of high clas-                                               DETRs, which makes it possible to distill our lightweight
sification scores (0.82% vs 0.35%) but also provide more                                                                 detector with high accuracy pre-trained large DETR models.
high-quality features (0.67% vs 0.30%). We also evaluate                                       We believe that this is one of the advantages of RT-DETR
the accuracy of the detectors trained with the two query selec-                                                          over other real-time detectors and could be an interesting
tion schemes on COCO val2017, where the uncertainty-                                                                  direction for future exploration.
minimal query selection achieves an improvement of 0.8%
AP (48.7% AP vs 47.9% AP).
                                                          7. Conclusion
5.4. Ablation Study on Decoder

Table 5 shows the inference latency and accuracy of each de-     In this work, we propose a real-time end-to-end detector,
coder layer of RT-DETR-R50 trained with different numbers      called RT-DETR, which successfully extends DETR to the
of decoder layers. When the number of decoder layers is set      real-time detection scenario and achieves state-of-the-art per-
to 6, the RT-DETR-R50 achieves the best accuracy 53.1%     formance. RT-DETR includes two key enhancements: an
AP. Furthermore, we observe that the difference in accuracy      efficient hybrid encoder that expeditiously processes multi-
between adjacent decoder layers gradually decreases as the      scale features, and the uncertainty-minimal query selection
index of the decoder layer increases. Taking the column RT-     that improves the quality of initial object queries. Further-
DETR-R50-Det6 as an example, using 5-th decoder layer     more, RT-DETR supports flexible speed tuning without re-
for inference only loses 0.1% AP (53.1% AP vs 53.0% AP)      training and eliminates the inconvenience caused by two
in accuracy, while reducing latency by 0.5 ms (9.3 ms vs 8.8   NMS thresholds, facilitating its practical application. RT-
ms). Therefore, RT-DETR supports flexible speed tuning by    DETR, along with its model scaling strategy, broadens the
adjusting the number of decoder layers without retraining,     technical approach to real-time object detection, offering
thus improving its practicality.                          new possibilities beyond YOLO for diverse real-time scenar-
                                                                         ios. We hope that RT-DETR can be put into practice.
6. Limitation and Discussion
                                                   Acknowledgements.  This work was supported in part
Limitation. Although the proposed RT-DETR outperforms    by  the  National Key R&D Program  of China  (No.
the state-of-the-art real-time detectors and end-to-end detec-    2022ZD0118201), Natural Science Foundation of China (No.
tors with similar size in both speed and accuracy, it shares the     61972217, 32071459, 62176249, 62006133, 62271465),
same limitation as the other DETRs, i.e., the performance on     and the Shenzhen Medical Research Funds in China (No.
small objects is still inferior than the strong real-time detec-    B2302037). Thanks to Chang Liu, Zhennan Wang and Ke-
tors. According to Table 2, RT-DETR-R50 is 0.5% AP lower     han Li for helpful suggestions on writing and presentation.
<a id="page-9"></a>

### PDF 第 9 页

References                                                               tion with convolutional neural networks. In Proceedings of
                                                                           the IEEE/CVF Conference on Computer Vision and Pattern
 [1] Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark                                                                         Recognition, pages 558–567, 2019. 6, 1
     Liao. Yolov4: Optimal speed and accuracy of object detection.
                                                                     [15] Xin Huang, Xinxin Wang, Wenyu Lv, Xiaying Bai, Xiang
     arXiv preprint arXiv:2004.10934, 2020. 1, 2
                                                                 Long, Kaipeng Deng, Qingqing Dang, Shumin Han, Qiwen
 [2] Daniel Bogdoll, Maximilian Nitsche, and J Marius Z¨ollner.                                                                          Liu, Xiaoguang Hu, et al.  Pp-yolov2: A practical object
    Anomaly detection in autonomous driving: A survey. In Pro-                                                                                  detector. arXiv preprint arXiv:2104.10419, 2021. 1, 2
     ceedings of the IEEE/CVF Conference on Computer Vision
                                                                     [16] Chuyi Li, Lulu Li, Yifei Geng, Hongliang Jiang, Meng
    and Pattern Recognition, pages 4488–4499, 2022. 1
                                                                   Cheng, Bo Zhang, Zaidan Ke, Xiaoming Xu, and Xiangxiang
 [3] Yuxuan Cai, Yizhuang Zhou, Qi Han, Jianjian Sun, Xiang-
                                                                Chu. Yolov6 v3.0: A full-scale reloading. arXiv preprint
    wen Kong, Jun Li, and Xiangyu Zhang. Reversible column
                                                                      arXiv:2301.05586, 2023. 1, 2, 3, 6, 7
     networks. In International Conference on Learning Repre-
                                                                     [17] Feng Li, Hao Zhang, Shilong Liu, Jian Guo, Lionel M Ni, and
      sentations, 2022. 8
                                                                      Lei Zhang. Dn-detr: Accelerate detr training by introducing
 [4] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas
                                                                      query denoising. In Proceedings of the IEEE/CVF Conference
     Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-
                                                             on Computer Vision and Pattern Recognition, pages 13619–
     end object detection with transformers. In European Confer-
                                                                    13627, 2022. 1, 2, 7
     ence on Computer Vision, pages 213–229. Springer, 2020. 1,
                                                                     [18] Feng Li, Ailing Zeng, Shilong Liu, Hao Zhang, Hongyang      2, 7
                                                                                Li, Lei Zhang, and Lionel M Ni. Lite detr: An interleaved
 [5] Qiang Chen, Xiaokang Chen, Gang Zeng, and Jingdong Wang.
                                                                           multi-scale encoder for efficient detr.  In Proceedings of
    Group detr: Fast training convergence with decoupled one-
                                                                           the IEEE/CVF Conference on Computer Vision and Pattern
     to-many label assignment. arXiv preprint arXiv:2207.13085,
                                                                         Recognition, pages 18558–18567, 2023. 2
     2022. 2
                                                                     [19] Junyu Lin, Xiaofeng Mao, Yuefeng Chen, Lei Xu, Yuan [6] Qiang Chen, Jian Wang, Chuchu Han, Shan Zhang, Zex-
                                                                 He, and Hui Xue. Dˆ 2etr: Decoder-only detr with com-     ian Li, Xiaokang Chen, Jiahui Chen, Xiaodi Wang, Shum-
                                                                             putationally efficient cross-scale attention. arXiv preprint     ing Han, Gang Zhang, et al. Group detr v2: Strong object
                                                                      arXiv:2203.00860, 2022. 4     detector with encoder-decoder pretraining. arXiv preprint
     arXiv:2211.03594, 2022. 8                                      [20] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays,
                                                                               Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence [7] Cheng Cui, Ruoyu Guo, Yuning Du, Dongliang He, Fu Li,
                                                                              Zitnick. Microsoft coco: Common objects in context.  In    Zewu Wu, Qiwen Liu, Shilei Wen, Jizhou Huang, Xiaoguang
                                                              European Conference on Computer Vision, pages 740–755.    Hu, Dianhai Yu, Errui Ding, and Yanjun Ma. Beyond self-
                                                                             Springer, 2014. 3, 8, 1     supervision: A simple yet effective network distillation alter-
     native to improve backbones. CoRR, abs/2103.05959, 2021.      [21] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and
    1                                                                           Piotr Doll´ar. Focal loss for dense object detection. In Proceed-
                                                                            ings of the IEEE/CVF International Conference on Computer [8] Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han,
                                                                               Vision, pages 2980–2988, 2017. 2    Guiguang Ding, and Jian Sun. Repvgg: Making vgg-style
     convnets great again. In Proceedings of the IEEE/CVF Con-      [22] Shu Liu, Lu Qi, Haifang Qin, Jianping Shi, and Jiaya Jia. Path
     ference on Computer Vision and Pattern Recognition, pages           aggregation network for instance segmentation. In Proceed-
    13733–13742, 2021. 5                                               ings of the IEEE/CVF Conference on Computer Vision and
 [9] Peng Gao, Minghang Zheng, Xiaogang Wang, Jifeng Dai,           Pattern Recognition, pages 8759–8768, 2018. 4
     and Hongsheng Li. Fast convergence of detr with spatially      [23] Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi,
     modulated co-attention.  In Proceedings of the IEEE/CVF         Hang Su, Jun Zhu, and Lei Zhang. Dab-detr: Dynamic anchor
     International Conference on Computer Vision, pages 3621–          boxes are better queries for detr. In International Conference
     3630, 2021. 7                                              on Learning Representations, 2021. 1, 2, 7
[10] Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian      [24] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian
     Sun. Yolox: Exceeding yolo series in 2021. arXiv preprint           Szegedy, Scott Reed, Cheng-Yang Fu, and Alexander C Berg.
     arXiv:2107.08430, 2021. 1, 2                                       Ssd: Single shot multibox detector. In European Conference
[11] Jocher Glenn. Yolov5 release v7.0. https://github.         on Computer Vision, pages 21–37. Springer, 2016. 2
    com/ultralytics/yolov5/tree/v7.0, 2022. 2, 3,      [25] Xiang Long, Kaipeng Deng, Guanzhong Wang, Yang Zhang,
      6, 7                                                          Qingqing Dang, Yuan Gao, Hui Shen, Jianguo Ren, Shumin
[12] Jocher Glenn.   Yolov8.  https://github.com/         Han, Errui Ding, et al.  Pp-yolo: An effective and effi-
    ultralytics/ultralytics/tree/main, 2023.  1,            cient implementation of object detector.  arXiv preprint
      2, 3, 6, 7                                                         arXiv:2007.12099, 2020. 1, 2
[13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.      [26] Ilya Loshchilov and Frank Hutter.  Decoupled weight de-
    Deep residual learning for image recognition. In Proceedings          cay regularization. In International Conference on Learning
      of the IEEE/CVF Conference on Computer Vision and Pattern           Representations, 2018. 1
     Recognition, pages 770–778, 2016. 6, 1                          [27] Depu Meng, Xiaokang Chen,  Zejia Fan, Gang Zeng,
[14] Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Jun-         Houqiang Li, Yuhui Yuan, Lei Sun, and Jingdong Wang. Con-
    yuan Xie, and Mu Li. Bag of tricks for image classifica-            ditional detr for fast training convergence. In Proceedings of
<a id="page-10"></a>

### PDF 第 10 页

the IEEE/CVF International Conference on Computer Vision,          Dang, Shengyu Wei, Yuning Du, et al. Pp-yoloe: An evolved
     pages 3651–3660, 2021. 1, 3, 7                                       version of yolo. arXiv preprint arXiv:2203.16250, 2022. 1, 2,
[28] Rashmika  Nawaratne, Damminda Alahakoon,  Daswin             3, 6, 7
    De Silva, and Xinghuo Yu. Spatiotemporal anomaly detection      [41] Jianwei Yang, Chunyuan Li, Xiyang Dai, and Jianfeng Gao.
     using deep learning for real-time video surveillance. IEEE           Focal modulation networks. Advances in Neural Information
     Transactions on Industrial Informatics, 16(1):393–402, 2019.          Processing Systems, 35:4203–4217, 2022. 8
    1                                                               [42] Zhuyu Yao, Jiangbo Ai, Boxun Li, and Chi Zhang. Efficient
[29] Joseph Redmon and Ali Farhadi. Yolo9000: better, faster,            detr: improving end-to-end object detector with dense prior.
      stronger. In Proceedings of the IEEE/CVF Conference on           arXiv preprint arXiv:2104.01318, 2021. 2, 5, 7
    Computer Vision and Pattern Recognition, pages 7263–7271,      [43] Fangao Zeng, Bin Dong, Yuang Zhang, Tiancai Wang, Xi-
     2017. 2                                                    angyu Zhang, and Yichen Wei. Motr: End-to-end multiple-
[30] Joseph Redmon and Ali Farhadi. Yolov3: An incremental            object tracking with transformer. In European Conference on
     improvement. arXiv preprint arXiv:1804.02767, 2018. 1, 2          Computer Vision, pages 659–675. Springer, 2022. 1
[31] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali      [44] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun
     Farhadi. You only look once: Unified, real-time object de-          Zhu, Lionel Ni, and Heung-Yeung Shum. Dino: Detr with
      tection.  In Proceedings of the IEEE/CVF Conference on          improved denoising anchor boxes for end-to-end object de-
    Computer Vision and Pattern Recognition, pages 779–788,             tection. In International Conference on Learning Representa-
     2016. 2                                                                      tions, 2022. 1, 2, 3, 5, 7, 8
[32] Tianhe Ren, Jianwei Yang, Shilong Liu, Ailing Zeng, Feng Li,      [45] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang,
    Hao Zhang, Hongyang Li, Zhaoyang Zeng, and Lei Zhang.         and Jifeng Dai. Deformable detr: Deformable transformers
   A strong and reproducible object detector with only public            for end-to-end object detection. In International Conference
      datasets. arXiv preprint arXiv:2304.13027, 2023. 8                on Learning Representations, 2020. 1, 2, 3, 4, 5, 7
[33] Byungseok Roh, JaeWoong Shin, Wuhyun Shin, and Saehoon      [46] Zhuofan Zong, Guanglu Song, and Yu Liu. Detrs with col-
    Kim. Sparse detr: Efficient end-to-end object detection with            laborative hybrid assignments training. In Proceedings of
     learnable sparsity. In International Conference on Learning            the IEEE/CVF International Conference on Computer Vision,
     Representations, 2021. 2                                         pages 6748–6758, 2023. 8
[34] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, San-
     jeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy,
     Aditya Khosla, Michael Bernstein, et al.  Imagenet large
     scale visual recognition challenge. International Journal of
    Computer Vision, 115:211–252, 2015. 1
[35] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang
     Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A
      large-scale, high-quality dataset for object detection. In Pro-
     ceedings of the IEEE/CVF International Conference on Com-
     puter Vision, pages 8430–8439, 2019. 2, 1
[36] Peize Sun, Rufeng Zhang, Yi Jiang, Tao Kong, Chenfeng
    Xu, Wei Zhan, Masayoshi Tomizuka, Lei Li, Zehuan Yuan,
    Changhu Wang, et al.  Sparse r-cnn: End-to-end object
     detection with learnable proposals.  In Proceedings of the
    IEEE/CVF Conference on Computer Vision and Pattern
     Recognition, pages 14454–14463, 2021. 1
[37] Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark
     Liao.  Scaled-yolov4: Scaling cross stage partial network.
     In Proceedings of the IEEE/CVF Conference on Computer
     Vision and Pattern Recognition, pages 13029–13038, 2021. 2
[38] Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark
     Liao. Yolov7: Trainable bag-of-freebies sets new state-of-
      the-art for real-time object detectors.  In Proceedings of
     the IEEE/CVF Conference on Computer Vision and Pattern
     Recognition, pages 7464–7475, 2023. 1, 2, 3, 6, 7
[39] Yingming Wang, Xiangyu Zhang, Tong Yang, and Jian Sun.
    Anchor detr: Query design for transformer-based detector. In
     Proceedings of the AAAI Conference on Artificial Intelligence,
     pages 2567–2575, 2022. 1, 3, 7
[40] Shangliang Xu, Xinxin Wang, Wenyu Lv, Qinyao Chang,
    Cheng Cui, Kaipeng Deng, Guanzhong Wang, Qingqing
<a id="page-11"></a>

### PDF 第 11 页

Appendix of “DETRs Beat YOLOs on Real-time Object Detection”


1. Experimental Settings                                 Item                        Value

Dataset and  metric.  We  conduct  experiments on                                                                     optimizer               AdamW
COCO [20] and Objects365 [35], where RT-DETR is trained
                                                                  base learning rate               1e-4
on COCO train2017 and validated on COCO val2017
                                                                        learning rate of backbone        1e-5dataset. We report the standard COCO metrics, including
AP (averaged over uniformly sampled IoU thresholds rang-             freezing BN                   True
ing from 0.50-0.95 with a step size of 0.05), AP50, AP75, as               linear warm-up start factor       0.001
well as AP at different scales: APS, APM, APL.                          linear warm-up steps           2000
Implementation details. We use ResNet [13, 14] pretrained            weight decay                  0.0001
on ImageNet [7, 34] as the backbone and the learning rate               clip gradient norm               0.1
strategy of the backbone follows [4]. In the hybrid encoder,                                                     ema decay                     0.9999
AIFI consists of 1 Transformer layer and the fusion block in
                                                          number of AIFI layers         1
CCFF consists of 3 RepBlocks. We leverage the uncertainty-
                                                          number of RepBlocks          3minimal query selection to select top 300 encoder features
to initialize object queries of the decoder. The training           embedding dim               256
strategy and hyperparameters of the decoder almost follow            feedforward dim              1024
DINO [44]. We train RT-DETR with the AdamW [26] op-            nheads                      8
timizer using four NVIDIA Tesla V100 GPUs with a batch           number of feature scales        3
size of 16 and apply the exponential moving average (EMA)                                                          number of decoder layers       6
with ema decay = 0.9999. The 1× configuration means
                                                          number of queries             300
that the total epoch is 12, and the final reported results adopt
                                                                 decoder npoints               4the 6× configuration. The data augmentation applied during
training includes random {color distort, expand, crop, flip,              class cost weight                 2.0
resize} operations, following [40]. The main hyperparame-        α in class cost                  0.25
ters of RT-DETR are listed in Table A (refer to RT-DETR-          γ in class cost                   2.0
R50 for detailed configuration).                                                           bbox cost weight                 5.0
                                                   GIoU cost weight                2.0
2. Comparison with Lighter YOLO Detectors                class loss weight                 1.0
                                              α in class loss                  0.75
To adapt to diverse real-time detection scenarios, we develop
                                                         γ in class loss                   2.0lighter scaled RT-DETRs by scaling the encoder and decoder
with ResNet50/34/18 [13]. Specifically, we halve the number           bbox loss weight                 5.0
of channels in the RepBlock, while leaving other components          GIoU loss weight                2.0
unchanged, and obtain a set of RT-DETRs by adjusting the             denoising number             200
number of decoder layers during inference. We compare              label noise ratio                  0.5
the scaled RT-DETRs with the S and M models of YOLO                                                           box noise scale                  1.0
detectors in Table B. The number of decoder layers used
by scaled RT-DETR-R50/34/18 during training is 6/4/3 re-
                                                                           Table A. Main hyperparameters of RT-DETR.
spectively, and Deck indicates that k decoder layers are used
during inference. Our RT-DETR-R50-Dec2−5 outperform     3. Large-scale Pre-training for RT-DETR
all M models of YOLO detectors in both speed and accuracy,
while RT-DETR-R18-Dec2 outperforms all S models. Com-   We pre-train RT-DETR on the larger Objects365[35] dataset
pared to the state-of-the-art M model (YOLOv8-M [12]),    and then fine-tune it on COCO to achieve higher perfor-
RT-DETR-R50-Dec5 improves accuracy by 0.9% AP and     mance. As shown in Table C, we perform experiments on
increases FPS by 36%. Compared to the state-of-the-art S    RT-DETR-R18/50/101 respectively. All three models are
model (YOLOv6-S [16]), RT-DETR-R18-Dec2 improves ac-     pre-trained on Objects365 for 12 epochs, and RT-DETR-R18
curacy by 0.5% AP and increases FPS by 18%. This shows       is fine-tuned on COCO for 60 epochs, while RT-DETR-R50
that RT-DETR is able to outperform the lighter YOLO de-    and RT-DETR-R101 are fine-tuned for 24 epochs. Experi-
tectors in both speed and accuracy by simple scaling.           mental results show that RT-DETR-R18/50/101 is improved
<a id="page-12"></a>

### PDF 第 12 页

Model                   #Epochs  #Params (M)  GFLOPs  FPSbs=1   APval   APval50   APval75   APvalS    APvalM   APvalL

 S and M models of YOLO Detectors
  YOLOv5-S[11]               300           7.2          16.5       74       37.4    56.8        -          -          -          -
  YOLOv5-M[11]              300          21.2          49.0       64       45.4    64.1        -          -          -          -
  PPYOLOE-S[40]             300           7.9          17.4       218      43.0    59.6    47.1    25.9    47.4    58.6
  PPYOLOE-M[40]             300          23.4          49.9       131      48.9    65.8    53.7    30.8    53.4    65.3
  YOLOv6-S[16]               300          18.5          45.3       201      45.0    61.8    48.9    24.3    50.2    62.7
  YOLOv6-M[16]              300          34.9          85.8       121      50.0    66.9    54.6    30.6    55.4    67.3
  YOLOv8-S[12]                        -           11.2          28.6       136      44.9    61.8    48.6    25.7    49.9    61.0
  YOLOv8-M[12]                       -           25.9          78.9       97       50.2    67.2    54.6    32.0    55.7    66.4

  Scaled RT-DETRs
  Scaled RT-DETR-R50-Dec2     72          36†          98.4       154      50.3    68.4    54.5    32.2    55.2    67.5
  Scaled RT-DETR-R50-Dec3     72          36†         100.1      145      51.3    69.6    55.4    33.6    56.1    68.6
  Scaled RT-DETR-R50-Dec4     72          36†         101.8      137      51.8    70.0    55.9    33.7    56.4    69.4
  Scaled RT-DETR-R50-Dec5     72          36†         103.5      132      52.1    70.5    56.2    34.3    56.9    69.9
  Scaled RT-DETR-R50-Dec6     72          36          105.2      125      52.2    70.6    56.4    34.4    57.0    70.0

  Scaled RT-DETR-R34-Dec2     72          31†          89.3       185      47.4    64.7    51.3    28.9    51.0    64.2
  Scaled RT-DETR-R34-Dec3     72          31†          91.0       172      48.5    66.2    52.3    30.2    51.9    66.2
  Scaled RT-DETR-R34-Dec4     72          31          92.7       161      48.9    66.8    52.9    30.6    52.4    66.3

  Scaled RT-DETR-R18-Dec2     72          20†          59.0       238      45.5    62.5    49.4    27.8    48.7    61.7
  Scaled RT-DETR-R18-Dec3     72          20          60.7       217      46.5    63.8    50.4    28.4    49.8    63.0


Table B. Comparison with S and M models of YOLO detectors. The FPS of YOLO detectors are reported on T4 GPU with TensorRT FP16
using official pre-trained models according to the proposed end-to-end speed benchmark. † denotes the number of parameters during the
training, not inference.

    Model          #Epochs  #Params (M)  GFLOPs  FPSbs=1     APval     APval50   APval75   APvalS    APvalM   APvalL

    RT-DETR-R18      60          20          61       217     49.2 (↑2.7)    66.6    53.5    33.2    52.3    64.8
    RT-DETR-R50      24          42         136       108     55.3 (↑2.2)    73.4    60.1    37.9    59.9    71.8
    RT-DETR-R101     24          76         259       74      56.2 (↑1.9)    74.6    61.3    38.3    60.5    73.5

                         Table C. Fine-tuning results on COCO val2017 with pre-training on Objects365.


by 2.7%/2.2%/1.9% AP on COCO val2017. The surpris-     are filtered out and the number of false negatives increases.
ing improvement further demonstrates the potential of RT-    However, using a lower confidence threshold, e.g., 0.001,
DETR and provides the strongest real-time object detector      results in a large number of redundant boxes and increases
for various real-time scenarios in the industry.                   the number of false positives. The higher the IoU threshold,
                                                                 the fewer overlapping boxes are filtered out in each round of
4. Visualization of Predictions with Different     screening, and the number of false positives increases (the
    Post-processing Thresholds                          position marked by the red circle in Figure A). Nevertheless,
                                                           adopting a lower IoU threshold will result in true positives
To intuitively demonstrate the impact of post-processing     being deleted if there are overlapping or mutually occluding
on the detector, we visualize the predictions produced by     objects in the input. The confidence threshold is relatively
YOLOv8 [12] and RT-DETR using different post-processing      straightforward to process predicted boxes and therefore easy
thresholds, as shown in Figure A and Figure B, respectively.     to set, whereas the IoU threshold is difficult to set accurately.
We show the predictions for two randomly selected samples     Considering that different scenarios place different emphasis
from COCO val2017 by setting different NMS thresholds    on recall and accuracy, e.g., the general detection scenario
for YOLOv8-L and score thresholds for RT-DETR-R50.         requires the lower confidence threshold and the higher IoU
   There are two NMS thresholds: confidence threshold and      threshold to increase the recall, while the dedicated detection
IoU threshold, both of which affect the detection results. The     scenario requires the higher confidence threshold and the
higher the confidence threshold, the more prediction boxes     lower IoU threshold to increase the accuracy, it is neces-
<a id="page-13"></a>

### PDF 第 13 页

Conf_thr = 0.001               Conf_thr = 0.001                Conf_thr = 0.25
           IoU_thr = 0.3                  IoU_thr = 0.7                  IoU_thr = 0.7

                        Figure A. Visualization of YOLOv8-L [12] predictions with different NMS thresholds.





          Score_thr = 0.001                 Score_thr = 0.3                  Score_thr = 0.5

                        Figure B. Visualization of RT-DETR-R50 predictions with different score thresholds.


sary to carefully select the appropriate NMS thresholds for     post-processing threshold in RT-DETR is straightforward
different scenarios.                                       and does not affect the inference speed, enhancing the adapt-
  RT-DETR utilizes bipartite matching to predict the one-      ability of real-time detectors across various scenarios.
to-one object set, eliminating the need for suppressing over-
lapping boxes. Instead, it directly filters out low-confidence     5. Visualization of RT-DETR Predictions
boxes with a score threshold.  Similar to the confidence
threshold used in NMS, the score threshold can be adjusted   We select several samples from the COCO val2017 to
in different scenarios based on the specific emphasis to     showcase the detection performance of RT-DETR in com-
achieve optimal detection performance. Thus, setting the     plex scenarios and challenging conditions (refer to Figure C
<a id="page-14"></a>

### PDF 第 14 页

Figure C. Visualization of RT-DETR-R101 predictions in complex scenarios (score threshold=0.5).





Figure D. Visualization of RT-DETR-R101 predictions under difficult conditions, including motion blur, rotation, and occlusion (score
threshold=0.5).


and Figure D). In complex scenarios, RT-DETR demon-
strates its capability to detect diverse objects, even when
they are small or densely packed, e.g., cups, wine glasses,
and individuals. Moreover, RT-DETR successfully detects
objects under various difficult conditions, including motion
blur, rotation, and occlusion. These predictions substantiate
the excellent detection performance of RT-DETR.

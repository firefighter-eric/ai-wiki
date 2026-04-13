# Wang et al. - 2024 - YOLOv10 Real-Time End-to-End Object Detection

- Source HTML: `raw/html/Wang et al. - 2024 - YOLOv10 Real-Time End-to-End Object Detection.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2405.14458
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# YOLOv10: Real-Time End-to-End Object Detection

Ao Wang Hui Chen∗  Lihao Liu Kai Chen Zijia Lin 
Jungong Han Guiguang Ding 
Tsinghua University

Corresponding Author.

###### Abstract

Over the past years, YOLOs have emerged as the predominant paradigm in the field of real-time object detection owing to their effective balance between computational cost and detection performance. Researchers have explored the architectural designs, optimization objectives, data augmentation strategies, and others for YOLOs, achieving notable progress. However, the reliance on the non-maximum suppression (NMS) for post-processing hampers the end-to-end deployment of YOLOs and adversely impacts the inference latency. Besides, the design of various components in YOLOs lacks the comprehensive and thorough inspection, resulting in noticeable computational redundancy and limiting the model’s capability. It renders the suboptimal efficiency, along with considerable potential for performance improvements. In this work, we aim to further advance the performance-efficiency boundary of YOLOs from both the post-processing and the model architecture. To this end, we first present the consistent dual assignments for NMS-free training of YOLOs, which brings the competitive performance and low inference latency simultaneously. Moreover, we introduce the holistic efficiency-accuracy driven model design strategy for YOLOs. We comprehensively optimize various components of YOLOs from both the efficiency and accuracy perspectives, which greatly reduces the computational overhead and enhances the capability. The outcome of our effort is a new generation of YOLO series for real-time end-to-end object detection, dubbed YOLOv10. Extensive experiments show that YOLOv10 achieves the state-of-the-art performance and efficiency across various model scales. For example, our YOLOv10-S is 1.8×\times faster than RT-DETR-R18 under the similar AP on COCO, meanwhile enjoying 2.8×\times smaller number of parameters and FLOPs. Compared with YOLOv9-C, YOLOv10-B has 46% less latency and 25% fewer parameters for the same performance. Code: https://github.com/THU-MIG/yolov10.

## 1 Introduction

Real-time object detection has always been a focal point of research in the area of computer vision, which aims to accurately predict the categories and positions of objects in an image under low latency. It is widely adopted in various practical applications, including autonomous driving [3], robot navigation [11], and object tracking [66], etc. In recent years, researchers have concentrated on devising CNN-based object detectors to achieve real-time detection [18, 22, 43, 44, 45, 51, 12]. Among them, YOLOs have gained increasing popularity due to their adept balance between performance and efficiency [2, 19, 27, 19, 20, 59, 54, 64, 7, 65, 16, 27]. The detection pipeline of YOLOs consists of two parts: the model forward process and the NMS post-processing. However, both of them still have deficiencies, resulting in suboptimal accuracy-latency boundaries.

Specifically, YOLOs usually employ one-to-many label assignment strategy during training, whereby one ground-truth object corresponds to multiple positive samples. Despite yielding superior performance, this approach necessitates NMS to select the best positive prediction during inference. This slows down the inference speed and renders the performance sensitive to the hyperparameters of NMS, thereby preventing YOLOs from achieving optimal end-to-end deployment [71]. One line to tackle this issue is to adopt the recently introduced end-to-end DETR architectures [4, 74, 67, 28, 34, 40, 61]. For example, RT-DETR [71] presents an efficient hybrid encoder and uncertainty-minimal query selection, propelling DETRs into the realm of real-time applications. Nevertheless, the inherent complexity in deploying DETRs impedes its ability to attain the optimal balance between accuracy and speed. Another line is to explore end-to-end detection for CNN-based detectors, which typically leverages one-to-one assignment strategies to suppress the redundant predictions [5, 49, 60, 73, 16]. However, they usually introduce additional inference overhead or achieve suboptimal performance.

Furthermore, the model architecture design remains a fundamental challenge for YOLOs, which exhibits an important impact on the accuracy and speed [45, 16, 65, 7]. To achieve more efficient and effective model architectures, researchers have explored different design strategies. Various primary computational units are presented for the backbone to enhance the feature extraction ability, including DarkNet [43, 44, 45], CSPNet [2], EfficientRep [27] and ELAN [56, 58], etc. For the neck, PAN [35], BiC [27], GD [54] and RepGFPN [65], etc., are explored to enhance the multi-scale feature fusion. Besides, model scaling strategies [56, 55] and re-parameterization [10, 27] techniques are also investigated. While these efforts have achieved notable advancements, a comprehensive inspection for various components in YOLOs from both the efficiency and accuracy perspectives is still lacking. As a result, there still exists considerable computational redundancy within YOLOs, leading to inefficient parameter utilization and suboptimal efficiency. Besides, the resulting constrained model capability also leads to inferior performance, leaving ample room for accuracy improvements.

In this work, we aim to address these issues and further advance the accuracy-speed boundaries of YOLOs. We target both the post-processing and the model architecture throughout the detection pipeline. To this end, we first tackle the problem of redundant predictions in the post-processing by presenting a consistent dual assignments strategy for NMS-free YOLOs with the dual label assignments and consistent matching metric. It allows the model to enjoy rich and harmonious supervision during training while eliminating the need for NMS during inference, leading to competitive performance with high efficiency. Secondly, we propose the holistic efficiency-accuracy driven model design strategy for the model architecture by performing the comprehensive inspection for various components in YOLOs. For efficiency, we propose the lightweight classification head, spatial-channel decoupled downsampling, and rank-guided block design, to reduce the manifested computational redundancy and achieve more efficient architecture. For accuracy, we explore the large-kernel convolution and present the effective partial self-attention module to enhance the model capability, harnessing the potential for performance improvements under low cost.

Based on these approaches, we succeed in achieving a new family of real-time end-to-end detectors with different model scales, i.e., YOLOv10-N / S / M / B / L / X. Extensive experiments on standard benchmarks for object detection, i.e., COCO [33], demonstrate that our YOLOv10 can significantly outperform previous state-of-the-art models in terms of computation-accuracy trade-offs across various model scales. As shown in Fig. 1,
our YOLOv10-S / X are 1.8×\times / 1.3×\times faster than RT-DETR-R18 / R101, respectively, under the similar performance. Compared with YOLOv9-C, YOLOv10-B achieves a 46% reduction in latency with the same performance. Moreover, YOLOv10 exhibits highly efficient parameter utilization. Our YOLOv10-L / X outperforms YOLOv8-L / X by 0.3 AP and 0.5 AP, with 1.8×\times and 2.3×\times smaller number of parameters, respectively. YOLOv10-M achieves the similar AP compared with YOLOv9-M / YOLO-MS, with 23% / 31% fewer parameters, respectively. We hope that our work can inspire further studies and advancements in the field.

## 2 Related Work

Real-time object detectors. Real-time object detection aims to classify and locate objects under low latency, which is crucial for real-world applications. Over the past years, substantial efforts have been directed towards developing efficient detectors [18, 51, 43, 32, 72, 69, 30, 29, 39]. Particularly, the YOLO series [43, 44, 45, 2, 19, 27, 56, 20, 59] stand out as the mainstream ones. YOLOv1, YOLOv2, and YOLOv3 identify the typical detection architecture consisting of three parts, i.e., backbone, neck, and head [43, 44, 45]. YOLOv4 [2] and YOLOv5 [19] introduce the CSPNet [57] design to replace DarkNet [42], coupled with data augmentation strategies, enhanced PAN, and a greater variety of model scales, etc. YOLOv6 [27] presents BiC and SimCSPSPPF for neck and backbone, respectively, with anchor-aided training and self-distillation strategy. YOLOv7 [56] introduces E-ELAN for rich gradient flow path and explores several trainable bag-of-freebies methods. YOLOv8 [20] presents C2f building block for effective feature extraction and fusion. Gold-YOLO [54] provides the advanced GD mechanism to boost the multi-scale feature fusion capability. YOLOv9 [59] proposes GELAN to improve the architecture and PGI to augment the training process.

End-to-end object detectors. End-to-end object detection has emerged as a paradigm shift from traditional pipelines, offering streamlined architectures [48]. DETR [4] introduces the transformer architecture and adopts Hungarian loss to achieve one-to-one matching prediction, thereby eliminating hand-crafted components and post-processing. Since then, various DETR variants have been proposed to enhance its performance and efficiency [40, 61, 50, 28, 34]. Deformable-DETR [74] leverages multi-scale deformable attention module to accelerate the convergence speed. DINO [67] integrates contrastive denoising, mix query selection, and look forward twice scheme into DETRs. RT-DETR [71] further designs the efficient hybrid encoder and proposes the uncertainty-minimal query selection to improve both the accuracy and latency. Another line to achieve end-to-end object detection is based CNN detectors. Learnable NMS [23] and relation networks [25] present another network to remove duplicated predictions for detectors. OneNet [49] and DeFCN [60] propose one-to-one matching strategies to enable end-to-end object detection with fully convolutional networks. FCOSpsspss{}_{\text{pss}} [73] introduces a positive sample selector to choose the optimal sample for prediction.

## 3 Methodology

### 3.1 Consistent Dual Assignments for NMS-free Training

During training, YOLOs [20, 59, 27, 64] usually leverage TAL [14] to allocate multiple positive samples for each instance. The adoption of one-to-many assignment yields plentiful supervisory signals, facilitating the optimization and achieving superior performance. However, it necessitates YOLOs to rely on the NMS post-processing, which causes the suboptimal inference efficiency for deployment. While previous works [49, 60, 73, 5] explore one-to-one matching to suppress the redundant predictions, they usually introduce additional inference overhead or yield suboptimal performance. In this work, we present a NMS-free training strategy for YOLOs with dual label assignments and consistent matching metric, achieving both high efficiency and competitive performance.

Dual label assignments.
Unlike one-to-many assignment, one-to-one matching assigns only one prediction to each ground truth, avoiding the NMS post-processing. However, it leads to weak supervision, which causes suboptimal accuracy and convergence speed [75]. Fortunately, this deficiency can be compensated for by the one-to-many assignment [5]. To achieve this, we introduce dual label assignments for YOLOs to combine the best of both strategies. Specifically, as shown in Fig. 2.(a), we incorporate another one-to-one head for YOLOs. It retains the identical structure and adopts the same optimization objectives as the original one-to-many branch but leverages the one-to-one matching to obtain label assignments. During training, two heads are jointly optimized with the model, allowing the backbone and neck to enjoy the rich supervision provided by the one-to-many assignment. During inference, we discard the one-to-many head and utilize the one-to-one head to make predictions. This enables YOLOs for the end-to-end deployment without incurring any additional inference cost. Besides, in the one-to-one matching, we adopt the top one selection, which achieves the same performance as Hungarian matching [4] with less extra training time.

Consistent matching metric.
During assignments, both one-to-one and one-to-many approaches leverage a metric to quantitatively assess the level of concordance between predictions and instances. To achieve prediction aware matching for both branches, we employ a uniform matching metric, i.e.,

m​(α,β)=s⋅pα⋅IoU​(b^,b)β,𝑚𝛼𝛽⋅𝑠superscript𝑝𝛼IoUsuperscript^𝑏𝑏𝛽m(\alpha,\beta)=s\cdot p^{\alpha}\cdot\text{IoU}(\hat{b},b)^{\beta},

(1)

where p𝑝p is the classification score, b^^𝑏\hat{b} and b𝑏b denote the bounding box of prediction and instance, respectively. s𝑠s represents the spatial prior indicating whether the anchor point of prediction is within the instance [20, 59, 27, 64]. α𝛼\alpha and β𝛽\beta are two important hyperparameters that balance the impact of the semantic prediction task and the location regression task. We denote the one-to-many and one-to-one metrics as mo​2​msubscript𝑚𝑜2𝑚m_{o2m}=m​(αo​2​m,βo​2​m)𝑚subscript𝛼𝑜2𝑚subscript𝛽𝑜2𝑚m(\alpha_{o2m},\beta_{o2m}) and mo​2​osubscript𝑚𝑜2𝑜m_{o2o}=m​(αo​2​o,βo​2​o)𝑚subscript𝛼𝑜2𝑜subscript𝛽𝑜2𝑜m(\alpha_{o2o},\beta_{o2o}), respectively. These metrics influence the label assignments and supervision information for the two heads.

In dual label assignments, the one-to-many branch provides much richer supervisory signals than one-to-one branch. Intuitively, if we can harmonize the supervision of the one-to-one head with that of one-to-many head, we can optimize the one-to-one head towards the direction of one-to-many head’s optimization. As a result, the one-to-one head can provide improved quality of samples during inference, leading to better performance. To this end, we first analyze the supervision gap between the two heads. Due to the randomness during training, we initiate our examination in the beginning with two heads initialized with the same values and producing the same predictions, i.e., one-to-one head and one-to-many head generate the same p𝑝p and IoU for each prediction-instance pair. We note that the regression targets of two branches do not conflict, as matched predictions share the same targets and unmatched predictions are ignored. The supervision gap thus lies in the different classification targets. Given an instance, we denote its largest IoU with predictions as u∗superscript𝑢u^{*}, and the largest one-to-many and one-to-one matching scores as mo​2​m∗superscriptsubscript𝑚𝑜2𝑚m_{o2m}^{*} and mo​2​o∗superscriptsubscript𝑚𝑜2𝑜m_{o2o}^{*}, respectively. Suppose that one-to-many branch yields the positive samples ΩΩ\Omega and one-to-one branch selects i𝑖i-th prediction with the metric mo​2​o,isubscript𝑚𝑜2𝑜𝑖m_{o2o,i}=mo​2​o∗superscriptsubscript𝑚𝑜2𝑜m_{o2o}^{*}, we can then derive the classification target to​2​m,jsubscript𝑡𝑜2𝑚𝑗t_{o2m,j}=u∗⋅mo​2​m,jmo​2​m∗≤u∗⋅superscript𝑢subscript𝑚𝑜2𝑚𝑗superscriptsubscript𝑚𝑜2𝑚superscript𝑢u^{*}\cdot\frac{m_{o2m,j}}{m_{o2m}^{*}}\leq u^{*} for j∈Ω𝑗Ωj\in\Omega and to​2​o,isubscript𝑡𝑜2𝑜𝑖t_{o2o,i}=u∗⋅mo​2​o,imo​2​o∗⋅superscript𝑢subscript𝑚𝑜2𝑜𝑖superscriptsubscript𝑚𝑜2𝑜u^{*}\cdot\frac{m_{o2o,i}}{m_{o2o}^{*}}=u∗superscript𝑢u^{*} for task aligned loss as in [20, 59, 27, 64, 14]. The supervision gap between two branches can thus be derived by the 1-Wasserstein distance [41] of different classification objectives, i.e.,

A=to​2​o,i−𝕀​(i∈Ω)​to​2​m,i+∑k∈Ω\{i}to​2​m,k,𝐴subscript𝑡𝑜2𝑜𝑖𝕀𝑖Ωsubscript𝑡𝑜2𝑚𝑖subscript𝑘\Ω𝑖subscript𝑡𝑜2𝑚𝑘A=t_{o2o,i}-\mathbb{I}(i\in\Omega)t_{o2m,i}+\sum\nolimits_{k\in\Omega\backslash\{i\}}t_{o2m,k},

(2)

We can observe that the gap decreases as to​2​m,isubscript𝑡𝑜2𝑚𝑖t_{o2m,i} increases, i.e., i𝑖i ranks higher within ΩΩ\Omega. It reaches the minimum when to​2​m,isubscript𝑡𝑜2𝑚𝑖t_{o2m,i}=u∗superscript𝑢u^{*}, i.e., i𝑖i is the best positive sample in ΩΩ\Omega, as shown in Fig. 2.(a). To achieve this, we present the consistent matching metric, i.e., αo​2​osubscript𝛼𝑜2𝑜\alpha_{o2o}=r⋅αo​2​m⋅𝑟subscript𝛼𝑜2𝑚r\cdot\alpha_{o2m} and βo​2​osubscript𝛽𝑜2𝑜\beta_{o2o}=r⋅βo​2​m⋅𝑟subscript𝛽𝑜2𝑚r\cdot\beta_{o2m}, which implies mo​2​osubscript𝑚𝑜2𝑜m_{o2o}=mo​2​mrsuperscriptsubscript𝑚𝑜2𝑚𝑟m_{o2m}^{r}. Therefore, the best positive sample for one-to-many head is also the best for one-to-one head. Consequently, both heads can be optimized consistently and harmoniously. For simplicity, we take r𝑟r=1, by default, i.e., αo​2​osubscript𝛼𝑜2𝑜\alpha_{o2o}=αo​2​msubscript𝛼𝑜2𝑚\alpha_{o2m} and βo​2​osubscript𝛽𝑜2𝑜\beta_{o2o}=βo​2​msubscript𝛽𝑜2𝑚\beta_{o2m}. To verify the improved supervision alignment, we count the number of one-to-one matching pairs within the top-1 / 5 / 10 of the one-to-many results after training. As shown in Fig. 2.(b), the alignment is improved under the consistent matching metric. For a more comprehensive understanding of the mathematical proof, please refer to the appendix.

### 3.2 Holistic Efficiency-Accuracy Driven Model Design

In addition to the post-processing, the model architectures of YOLOs also pose great challenges to the efficiency-accuracy trade-offs [45, 7, 27]. Although previous works explore various design strategies, the comprehensive inspection for various components in YOLOs is still lacking. Consequently, the model architecture exhibits non-negligible computational redundancy and constrained capability, which impedes its potential for achieving high efficiency and performance. Here, we aim to holistically perform model designs for YOLOs from both efficiency and accuracy perspectives.

Efficiency driven model design. The components in YOLO consist of the stem, downsampling layers, stages with basic building blocks, and the head. The stem incurs few computational cost and we thus perform efficiency driven model design for other three parts.

(1) Lightweight classification head.
The classification and regression heads usually share the same architecture in YOLOs. However, they exhibit notable disparities in computational overhead. For example, the FLOPs and parameter count of the classification head (5.95G/1.51M) are 2.5×\times and 2.4×\times those of the regression head (2.34G/0.64M) in YOLOv8-S, respectively. However, after analyzing the impact of classification error and the regression error (seeing Tab. 9), we find that the regression head undertakes more significance for the performance of YOLOs. Consequently, we can reduce the overhead of classification head without worrying about hurting the performance greatly. Therefore, we simply adopt a lightweight architecture for the classification head, which consists of two depthwise separable convolutions [24, 8] with the kernel size of 3×\times3 followed by a 1×\times1 convolution.

(2) Spatial-channel decoupled downsampling.
YOLOs typically leverage regular 3×\times3 standard convolutions with stride of 2, achieving spatial downsampling (from H×W𝐻𝑊H\times W to H2×W2𝐻2𝑊2\frac{H}{2}\times\frac{W}{2}) and channel transformation (from C𝐶C to 2​C2𝐶2C) simultaneously. This introduces non-negligible computational cost of 𝒪​(92​H​W​C2)𝒪92𝐻𝑊superscript𝐶2\mathcal{O}(\frac{9}{2}HWC^{2}) and parameter count of 𝒪​(18​C2)𝒪18superscript𝐶2\mathcal{O}(18C^{2}). Instead, we propose to decouple the spatial reduction and channel increase operations, enabling more efficient downsampling. Specifically, we firstly leverage the pointwise convolution to modulate the channel dimension and then utilize the depthwise convolution to perform spatial downsampling. This reduces the computational cost to 𝒪​(2​H​W​C2+92​H​W​C)𝒪2𝐻𝑊superscript𝐶292𝐻𝑊𝐶\mathcal{O}(2HWC^{2}+\frac{9}{2}HWC) and the parameter count to 𝒪​(2​C2+18​C)𝒪2superscript𝐶218𝐶\mathcal{O}(2C^{2}+18C). Meanwhile, it maximizes information retention during downsampling, leading to competitive performance with latency reduction.

(3) Rank-guided block design.
YOLOs usually employ the same basic building block for all stages [27, 59], e.g., the bottleneck block in YOLOv8 [20]. To thoroughly examine such homogeneous design for YOLOs, we utilize the intrinsic rank [31, 15] to analyze the redundancy111A lower rank implies greater redundancy, while a higher rank signifies more condensed information. of each stage. Specifically, we calculate the numerical rank of the last convolution in the last basic block in each stage, which counts the number of singular values larger than a threshold. Fig. 3.(a) presents the results of YOLOv8, indicating that deep stages and large models are prone to exhibit more redundancy. This observation suggests that simply applying the same block design for all stages is suboptimal for the best capacity-efficiency trade-off. To tackle this, we propose a rank-guided block design scheme which aims to decrease the complexity of stages that are shown to be redundant using compact architecture design. We first present a compact inverted block (CIB) structure, which adopts the cheap depthwise convolutions for spatial mixing and cost-effective pointwise convolutions for channel mixing, as shown in Fig. 3.(b). It can serve as the efficient basic building block, e.g., embedded in the ELAN structure [58, 20] (Fig. 3.(b)). Then, we advocate a rank-guided block allocation strategy to achieve the best efficiency while maintaining competitive capacity. Specifically, given a model, we sort its all stages based on their intrinsic ranks in ascending order. We further inspect the performance variation of replacing the basic block in the leading stage with CIB. If there is no performance degradation compared with the given model, we proceed with the replacement of the next stage and halt the process otherwise. Consequently, we can implement adaptive compact block designs across stages and model scales, achieving higher efficiency without compromising performance. Due to the page limit, we provide the details of the algorithm in the appendix.

Accuracy driven model design.

We further explore the large-kernel convolution and self-attention for accuracy driven design, aiming to boost the performance under minimal cost.

(1) Large-kernel convolution.
Employing large-kernel depthwise convolution is an effective way to enlarge the receptive field and enhance the model’s capability [9, 38, 37]. However, simply leveraging them in all stages may introduce contamination in shallow features used for detecting small objects, while also introducing significant I/O overhead and latency in high-resolution stages [7]. Therefore, we propose to leverage the large-kernel depthwise convolutions in CIB within the deep stages. Specifically, we increase the kernel size of the second 3×\times3 depthwise convolution in the CIB to 7×\times7, following [37]. Additionally, we employ the structural reparameterization technique [10, 9, 53] to bring another 3×\times3 depthwise convolution branch to alleviate the optimization issue without inference overhead. Furthermore, as the model size increases, its receptive field naturally expands, with the benefit of using large-kernel convolutions diminishing. Therefore, we only adopt large-kernel convolution for small model scales.

(2) Partial self-attention (PSA). Self-attention [52] is widely employed in various visual tasks due to its remarkable global modeling capability [36, 13, 70]. However, it exhibits high computational complexity and memory footprint. To address this, in light of the prevalent attention head redundancy [63], we present an efficient partial self-attention (PSA) module design, as shown in Fig. 3.(c). Specifically, we evenly partition the features across channels into two parts after the 1×\times1 convolution. We only feed one part into the NPSAsubscript𝑁PSAN_{\text{PSA}} blocks comprised of multi-head self-attention module (MHSA) and feed-forward network (FFN). Two parts are then concatenated and fused by a 1×\times1 convolution. Besides, we follow [21] to assign the dimensions of the query and key to half of that of the value in MHSA and replace the LayerNorm [1] with BatchNorm [26] for fast inference. Furthermore, PSA is only placed after the Stage 4 with the lowest resolution, avoiding the excessive overhead from the quadratic computational complexity of self-attention. In this way, the global representation learning ability can be incorporated into YOLOs with low computational costs, which well enhances the model’s capability and leads to improved performance.

Model
#Param.(M)
FLOPs(G)

APval(%)

Latency(ms)

Latencyf(ms)

YOLOv6-3.0-N [27]

4.7
11.4
37.0
2.69
1.76

Gold-YOLO-N [54]

5.6
12.1
39.6
2.92
1.82

YOLOv8-N [20]

3.2
8.7
37.3
6.16
1.77

YOLOv10-N (Ours)
2.3
6.7
38.5 / 39.5†
1.84
1.79

YOLOv6-3.0-S [27]

18.5
45.3
44.3
3.42
2.35

Gold-YOLO-S [54]

21.5
46.0
45.4
3.82
2.73

YOLO-MS-XS [7]

4.5
17.4
43.4
8.23
2.80

YOLO-MS-S [7]

8.1
31.2
46.2
10.12
4.83

YOLOv8-S [20]

11.2
28.6
44.9
7.07
2.33

YOLOv9-S [59]

7.1
26.4
46.7
-
-

RT-DETR-R18 [71]

20.0
60.0
46.5
4.58
4.49

YOLOv10-S (Ours)
7.2
21.6
46.3 / 46.8†
2.49
2.39

YOLOv6-3.0-M [27]

34.9
85.8
49.1
5.63
4.56

Gold-YOLO-M [54]

41.3
87.5
49.8
6.38
5.45

YOLO-MS [7]

22.2
80.2
51.0
12.41
7.30

YOLOv8-M [20]

25.9
78.9
50.6
9.50
5.09

YOLOv9-M [59]

20.0
76.3
51.1
-
-

RT-DETR-R34 [71]

31.0
92.0
48.9
6.32
6.21

RT-DETR-R50m [71]

36.0
100.0
51.3
6.90
6.84

YOLOv10-M (Ours)
15.4
59.1
51.1 / 51.3†
4.74
4.63

YOLOv6-3.0-L [27]

59.6
150.7
51.8
9.02
7.90

Gold-YOLO-L [54]

75.1
151.7
51.8
10.65
9.78

YOLOv9-C [59]

25.3
102.1
52.5
10.57
6.13

YOLOv10-B (Ours)
19.1
92.0
52.5 / 52.7†
5.74
5.67

YOLOv8-L [20]

43.7
165.2
52.9
12.39
8.06

RT-DETR-R50 [71]

42.0
136.0
53.1
9.20
9.07

YOLOv10-L (Ours)
24.4
120.3
53.2 / 53.4†
7.28
7.21

YOLOv8-X [20]

68.2
257.8
53.9
16.86
12.83

RT-DETR-R101 [71]

76.0
259.0
54.3
13.71
13.58

YOLOv10-X (Ours)
29.5
160.4
54.4 / 54.4†
10.70
10.60

## 4 Experiments

### 4.1 Implementation Details

We select YOLOv8 [20] as our baseline model, due to its commendable latency-accuracy balance and its availability in various model sizes. We employ the consistent dual assignments for NMS-free training and perform holistic efficiency-accuracy driven model design based on it, which brings our YOLOv10 models. YOLOv10 has the same variants as YOLOv8, i.e., N / S / M / L / X. Besides, we derive a new variant YOLOv10-B, by simply increasing the width scale factor of YOLOv10-M. We verify the proposed detector on COCO [33] under the same training-from-scratch setting [20, 59, 56]. Moreover, the latencies of all models are tested on T4 GPU with TensorRT FP16, following [71].

### 4.2 Comparison with state-of-the-arts

As shown in Tab. 1, our YOLOv10 achieves the state-of-the-art performance and end-to-end latency across various model scales. We first compare YOLOv10 with our baseline models, i.e., YOLOv8. On N / S / M / L / X five variants, our YOLOv10 achieves 1.2% / 1.4% / 0.5% / 0.3% / 0.5% AP improvements, with 28% / 36% / 41% / 44% / 57% fewer parameters, 23% / 24% / 25% / 27% / 38% less calculations, and 70% / 65% / 50% / 41% / 37% lower latencies. Compared with other YOLOs, YOLOv10 also exhibits superior trade-offs between accuracy and computational cost. Specifically, for lightweight and small models, YOLOv10-N / S outperforms YOLOv6-3.0-N / S by 1.5 AP and 2.0 AP, with 51% / 61% fewer parameters and 41% / 52% less computations, respectively. For medium models, compared with YOLOv9-C / YOLO-MS, YOLOv10-B / M enjoys the 46% / 62% latency reduction under the same or better performance, respectively. For large models, compared with Gold-YOLO-L, our YOLOv10-L shows 68% fewer parameters and 32% lower latency, along with a significant improvement of 1.4% AP. Furthermore, compared with RT-DETR, YOLOv10 obtains significant performance and latency improvements. Notably, YOLOv10-S / X achieves 1.8×\times and 1.3×\times faster inference speed than RT-DETR-R18 / R101, respectively, under the similar performance. These results well demonstrate the superiority of YOLOv10 as the real-time end-to-end detector.

We also compare YOLOv10 with other YOLOs using the original one-to-many training approach. We consider the performance and the latency of model forward process (Latencyf) in this situation, following [56, 20, 54]. As shown in Tab. 1, YOLOv10 also exhibits the state-of-the-art performance and efficiency across different model scales, indicating the effectiveness of our architectural designs.

### 4.3 Model Analyses

Ablation study. We present the ablation results based on YOLOv10-S and YOLOv10-M in Tab. 2. It can be observed that our NMS-free training with consistent dual assignments significantly reduces the end-to-end latency of YOLOv10-S by 4.63ms, while maintaining competitive performance of 44.3% AP. Moreover, our efficiency driven model design leads to the reduction of 11.8 M parameters and 20.8 GFlOPs, with a considerable latency reduction of 0.65ms for YOLOv10-M, well showing its effectiveness. Furthermore, our accuracy driven model design achieves the notable improvements of 1.8 AP and 0.7 AP for YOLOv10-S and YOLOv10-M, alone with only 0.18ms and 0.17ms latency overhead, respectively, which well demonstrates its superiority.

#
Model
NMS-free.
Efficiency.
Accuracy.
#Param.(M)
FLOPs(G)

APval(%)

Latency(ms)

1
YOLOv10-S

11.2
28.6
44.9
7.07

2
✓

11.2
28.6
44.3
2.44

3
✓
✓

6.2
20.8
44.5
2.31

4
✓
✓
✓
7.2
21.6
46.3
2.49

5
YOLOv10-M

25.9
78.9
50.6
9.50

6
✓

25.9
78.9
50.3
5.22

7
✓
✓

14.1
58.1
50.4
4.57

8
✓
✓
✓
15.4
59.1
51.1
4.74

Table 3: Dual assign.

o2m
o2o
AP
Latency

✓

44.9
7.07

✓
43.4
2.44

✓
✓
44.3
2.44

Table 4: Matching metric.

αo​2​osubscript𝛼𝑜2𝑜\alpha_{o2o}
βo​2​osubscript𝛽𝑜2𝑜\beta_{o2o}

APval

αo​2​osubscript𝛼𝑜2𝑜\alpha_{o2o}
βo​2​osubscript𝛽𝑜2𝑜\beta_{o2o}

APval

0.5
2.0
42.7
0.25
3.0
44.3

0.5
4.0
44.2
0.25
6.0
43.5

0.5
6.0
44.3
1.0
6.0
43.9

0.5
8.0
44.0
1.0
12.0
44.3

Table 5: Efficiency. for YOLOv10-S/M.

#
Model
#Param
FLOPs

APval

Latency

1
base.
11.2/25.9
28.6/78.9
44.3/50.3
2.44/5.22

2
+cls.
9.9/23.2
23.5/67.7
44.2/50.2
2.39/5.07

3
+downs.
8.0/19.7
22.2/65.0
44.4/50.4
2.36/4.97

4
+block.
6.2/14.1
20.8/58.1
44.5/50.4
2.31/4.57

Analyses for NMS-free training.

- •

Dual label assignments. We present dual label assignments for NMS-free YOLOs, which can bring both rich supervision of one-to-many (o2m) branch during training and high efficiency of one-to-one (o2o) branch during inference. We verify its benefit based on YOLOv8-S, i.e., #1 in Tab. 2. Specifically, we introduce baselines for training with only o2m branch and only o2o branch, respectively. As shown in Tab. 5, our dual label assignments achieve the best AP-latency trade-off.

- •

Consistent matching metric. We introduce consistent matching metric to make the one-to-one head more harmonious with the one-to-many head. We verify its benefit based on YOLOv8-S, i.e., #1 in Tab. 2, under different αo​2​osubscript𝛼𝑜2𝑜\alpha_{o2o} and βo​2​osubscript𝛽𝑜2𝑜\beta_{o2o}. As shown in Tab. 5, the proposed consistent matching metric, i.e., αo​2​osubscript𝛼𝑜2𝑜\alpha_{o2o}=r⋅αo​2​m⋅𝑟subscript𝛼𝑜2𝑚r\cdot\alpha_{o2m} and βo​2​osubscript𝛽𝑜2𝑜\beta_{o2o}=r⋅βo​2​m⋅𝑟subscript𝛽𝑜2𝑚r\cdot\beta_{o2m}, can achieve the optimal performance, where αo​2​msubscript𝛼𝑜2𝑚\alpha_{o2m}=0.50.50.5 and βo​2​msubscript𝛽𝑜2𝑚\beta_{o2m}=6.06.06.0 in the one-to-many head [20]. Such an improvement can be attributed to the reduction of the supervision gap (Eq. 2), which provides improved supervision alignment between two branches. Moreover, the proposed consistent matching metric eliminates the need for exhaustive hyper-parameter tuning, which is appealing in practical scenarios.

Table 6: cls. results.

base.
+cls.

APval

44.3
44.2

APw/o​cv​a​lsubscriptsuperscriptabsent𝑣𝑎𝑙𝑤𝑜𝑐{}^{val}_{w/o\ c}

59.9
59.9

APw/o​rv​a​lsubscriptsuperscriptabsent𝑣𝑎𝑙𝑤𝑜𝑟{}^{val}_{w/o\ r}

64.5
64.2

Table 7: Results of d.s.

Model

APval

Latency

base.
43.7
2.33

ours
44.4
2.36

Table 8: Results of CIB.

Model

APval

Latency

IRB
43.7
2.30

IRB-DW
44.2
2.30

ours
44.5
2.31

Table 9: Rank-guided.

Stages with CIB

APval

empty
44.4

8
44.5

8,4,
44.5

8,4,7
44.3

Table 10: Accuracy. for S/M.

#
Model

APval

Latency

1
base.
44.5/50.4
2.31/4.57

2
+L.k.
44.9/-
2.34/-

3
+PSA
46.3/51.1
2.49/4.74

Table 11: L.k. results.

Model

APval

Latency

k.s.=5
44.7
2.32

k.s.=7
44.9
2.34

k.s.=9
44.9
2.37

w/o rep.
44.8
2.34

Table 12: L.k. usage.

w/o L.k.
w/ L.k.

N
36.3
36.6

S
44.5
44.9

M
50.4
50.4

Table 13: PSA results.

Model

APval

Latency

PSA
46.3
2.49

Trans.
46.0
2.54

NPSAsubscript𝑁PSAN_{\text{PSA}} = 1

46.3
2.49

NPSAsubscript𝑁PSAN_{\text{PSA}} = 2

46.5
2.59

Analyses for efficiency driven model design. We conduct experiments to gradually incorporate the efficiency driven design elements based on YOLOv10-S/M. Our baseline is the YOLOv10-S/M model without efficiency-accuracy driven model design, i.e., #2/#6 in Tab. 2. As shown in Tab. 5, each design component, including lightweight classification head, spatial-channel decoupled downsampling, and rank-guided block design, contributes to the reduction of parameters count, FLOPs, and latency. Importantly, these improvements are achieved while maintaining competitive performance.

- •

Lightweight classification head. We analyze the impact of category and localization errors of predictions on the performance, based on the YOLOv10-S of #1 and #2 in Tab. 5, like [6]. Specifically, we match the predictions to the instances by the one-to-one assignment. Then, we substitute the predicted category score with instance labels, resulting in APw/o​cv​a​lsubscriptsuperscriptabsent𝑣𝑎𝑙𝑤𝑜𝑐{}^{val}_{w/o\ c} with no classification errors. Similarly, we replace the predicted locations with those of instances, yielding APw/o​rv​a​lsubscriptsuperscriptabsent𝑣𝑎𝑙𝑤𝑜𝑟{}^{val}_{w/o\ r} with no regression errors. As shown in Tab. 9, APw/o​rv​a​lsubscriptsuperscriptabsent𝑣𝑎𝑙𝑤𝑜𝑟{}^{val}_{w/o\ r} is much higher than APw/o​cv​a​lsubscriptsuperscriptabsent𝑣𝑎𝑙𝑤𝑜𝑐{}^{val}_{w/o\ c}, revealing that eliminating the regression errors achieves greater improvement. The performance bottleneck thus lies more in the regression task. Therefore, adopting the lightweight classification head can allow higher efficiency without compromising the performance.

- •

Spatial-channel decoupled downsampling. We decouple the downsampling operations for efficiency, where the channel dimensions are first increased by pointwise convolution (PW) and the resolution is then reduced by depthwise convolution (DW) for maximal information retention. We compare it with the baseline way of spatial reduction by DW followed by channel modulation by PW, based on the YOLOv10-S of #3 in Tab. 5. As shown in Tab. 9, our downsampling strategy achieves the 0.7% AP improvement by enjoying less information loss during downsampling.

- •

Compact inverted block (CIB). We introduce CIB as the compact basic building block. We verify its effectiveness based on the YOLOv10-S of #4 in the Tab. 5. Specifically, we introduce the inverted residual block [46] (IRB) as the baseline, which achieves the suboptimal 43.7% AP, as shown in Tab. 9. We then append a 3×\times3 depthwise convolution (DW) after it, denoted as “IRB-DW”, which brings 0.5% AP improvement. Compared with “IRB-DW”, our CIB further achieves 0.3% AP improvement by prepending another DW with minimal overhead, indicating its superiority.

- •

Rank-guided block design. We introduce the rank-guided block design to adaptively integrate compact block design for improving the model efficiency. We verify its benefit based on the YOLOv10-S of #3 in the Tab. 5. The stages sorted in ascending order based on the intrinsic ranks are Stage 8-4-7-3-5-1-6-2, like in Fig. 3.(a). As shown in Tab. 9, when gradually replacing the bottleneck block in each stage with the efficient CIB, we observe the performance degradation starting from Stage 7. In the Stage 8 and 4 with lower intrinsic ranks and more redundancy, we can thus adopt the efficient block design without compromising the performance. These results indicate that rank-guided block design can serve as an effective strategy for higher model efficiency.

Analyses for accuracy driven model design. We present the results of gradually integrating the accuracy driven design elements based on YOLOv10-S/M. Our baseline is the YOLOv10-S/M model after incorporating efficiency driven design, i.e., #3/#7 in Tab. 2. As shown in Tab. 13, the adoption of large-kernel convolution and PSA module leads to the considerable performance improvements of 0.4% AP and 1.4% AP for YOLOv10-S under minimal latency increase of 0.03ms and 0.15ms, respectively. Note that large-kernel convolution is not employed for YOLOv10-M (see Tab. 13).

- •

Large-kernel convolution. We first investigate the effect of different kernel sizes based on the YOLOv10-S of #2 in Tab. 13. As shown in Tab. 13, the performance improves as the kernel size increases and stagnates around the kernel size of 7×\times7, indicating the benefit of large perception field. Besides, removing the reparameterization branch during training achieves 0.1% AP degradation, showing its effectiveness for optimization. Moreover, we inspect the benefit of large-kernel convolution across model scales based on YOLOv10-N / S / M. As shown in Tab. 13, it brings no improvements for large models, i.e., YOLOv10-M, due to its inherent extensive receptive field. We thus only adopt large-kernel convolutions for small models, i.e., YOLOv10-N / S.

- •

Partial self-attention (PSA). We introduce PSA to enhance the performance by incorporating the global modeling ability under minimal cost. We first verify its effectiveness based on the YOLOv10-S of #3 in Tab. 13. Specifically, we introduce the transformer block, i.e., MHSA followed by FFN, as the baseline, denoted as “Trans.”. As shown in Tab. 13, compared with it, PSA brings 0.3% AP improvement with 0.05ms latency reduction. The performance enhancement may be attributed to the alleviation of optimization problem [62, 9] in self-attention, by mitigating the redundancy in attention heads. Moreover, we investigate the impact of different NPSAsubscript𝑁PSAN_{\text{PSA}}. As shown in Tab. 13, increasing NPSAsubscript𝑁PSAN_{\text{PSA}} to 2 obtains 0.2% AP improvement but with 0.1ms latency overhead. Therefore, we set NPSAsubscript𝑁PSAN_{\text{PSA}} to 1, by default, to enhance the model capability while maintaining high efficiency.

## 5 Conclusion

In this paper, we target both the post-processing and model architecture throughout the detection pipeline of YOLOs. For the post-processing, we propose the consistent dual assignments for NMS-free training, achieving efficient end-to-end detection. For the model architecture, we introduce the holistic efficiency-accuracy driven model design strategy, improving the performance-efficiency trade-offs. These bring our YOLOv10, a new real-time end-to-end object detector. Extensive experiments show that YOLOv10 achieves the state-of-the-art performance and latency compared with other advanced detectors, well demonstrating its superiority.

## References

- [1]

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton.

Layer normalization.

arXiv preprint arXiv:1607.06450, 2016.

- [2]

Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao.

Yolov4: Optimal speed and accuracy of object detection, 2020.

- [3]

Daniel Bogdoll, Maximilian Nitsche, and J Marius Zöllner.

Anomaly detection in autonomous driving: A survey.

In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 4488–4499, 2022.

- [4]

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander
Kirillov, and Sergey Zagoruyko.

End-to-end object detection with transformers.

In European conference on computer vision, pages 213–229.
Springer, 2020.

- [5]

Yiqun Chen, Qiang Chen, Qinghao Hu, and Jian Cheng.

Date: Dual assignment for end-to-end fully convolutional object
detection.

arXiv preprint arXiv:2211.13859, 2022.

- [6]

Yiqun Chen, Qiang Chen, Peize Sun, Shoufa Chen, Jingdong Wang, and Jian Cheng.

Enhancing your trained detrs with box refinement.

arXiv preprint arXiv:2307.11828, 2023.

- [7]

Yuming Chen, Xinbin Yuan, Ruiqi Wu, Jiabao Wang, Qibin Hou, and Ming-Ming
Cheng.

Yolo-ms: rethinking multi-scale representation learning for real-time
object detection.

arXiv preprint arXiv:2308.05480, 2023.

- [8]

François Chollet.

Xception: Deep learning with depthwise separable convolutions.

In Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 1251–1258, 2017.

- [9]

Xiaohan Ding, Xiangyu Zhang, Jungong Han, and Guiguang Ding.

Scaling up your kernels to 31x31: Revisiting large kernel design in
cnns.

In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 11963–11975, 2022.

- [10]

Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, and Jian
Sun.

Repvgg: Making vgg-style convnets great again.

In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 13733–13742, 2021.

- [11]

Douglas Henke Dos Reis, Daniel Welfer, Marco Antonio De Souza Leite Cuadros,
and Daniel Fernando Tello Gamarra.

Mobile robot navigation using an object recognition software with
rgbd images and the yolo algorithm.

Applied Artificial Intelligence, 33(14):1290–1305, 2019.

- [12]

Kaiwen Duan, Song Bai, Lingxi Xie, Honggang Qi, Qingming Huang, and Qi Tian.

Centernet: Keypoint triplets for object detection.

In Proceedings of the IEEE/CVF international conference on
computer vision, pages 6569–6578, 2019.

- [13]

Patrick Esser, Robin Rombach, and Bjorn Ommer.

Taming transformers for high-resolution image synthesis.

In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 12873–12883, 2021.

- [14]

Chengjian Feng, Yujie Zhong, Yu Gao, Matthew R Scott, and Weilin Huang.

Tood: Task-aligned one-stage object detection.

In 2021 IEEE/CVF International Conference on Computer Vision
(ICCV), pages 3490–3499. IEEE Computer Society, 2021.

- [15]

Ruili Feng, Kecheng Zheng, Yukun Huang, Deli Zhao, Michael Jordan, and
Zheng-Jun Zha.

Rank diminishing in deep neural networks.

Advances in Neural Information Processing Systems,
35:33054–33065, 2022.

- [16]

Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun.

Yolox: Exceeding yolo series in 2021.

arXiv preprint arXiv:2107.08430, 2021.

- [17]

Golnaz Ghiasi, Yin Cui, Aravind Srinivas, Rui Qian, Tsung-Yi Lin, Ekin D Cubuk,
Quoc V Le, and Barret Zoph.

Simple copy-paste is a strong data augmentation method for instance
segmentation.

In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 2918–2928, 2021.

- [18]

Ross Girshick.

Fast r-cnn.

In Proceedings of the IEEE international conference on computer
vision, pages 1440–1448, 2015.

- [19]

Jocher Glenn.

Yolov5 release v7.0.

https://github.com/ultralytics/yolov5/tree/v7.0, 2022.

- [20]

Jocher Glenn.

Yolov8.

https://github.com/ultralytics/ultralytics/tree/main,
2023.

- [21]

Benjamin Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin,
Hervé Jégou, and Matthijs Douze.

Levit: a vision transformer in convnet’s clothing for faster
inference.

In Proceedings of the IEEE/CVF international conference on
computer vision, pages 12259–12269, 2021.

- [22]

Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick.

Mask r-cnn.

In Proceedings of the IEEE international conference on computer
vision, pages 2961–2969, 2017.

- [23]

Jan Hosang, Rodrigo Benenson, and Bernt Schiele.

Learning non-maximum suppression.

In Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 4507–4515, 2017.

- [24]

Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang,
Tobias Weyand, Marco Andreetto, and Hartwig Adam.

Mobilenets: Efficient convolutional neural networks for mobile vision
applications.

arXiv preprint arXiv:1704.04861, 2017.

- [25]

Han Hu, Jiayuan Gu, Zheng Zhang, Jifeng Dai, and Yichen Wei.

Relation networks for object detection.

In Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 3588–3597, 2018.

- [26]

Sergey Ioffe and Christian Szegedy.

Batch normalization: Accelerating deep network training by reducing
internal covariate shift.

In International conference on machine learning, pages
448–456. pmlr, 2015.

- [27]

Chuyi Li, Lulu Li, Yifei Geng, Hongliang Jiang, Meng Cheng, Bo Zhang, Zaidan
Ke, Xiaoming Xu, and Xiangxiang Chu.

Yolov6 v3.0: A full-scale reloading.

arXiv preprint arXiv:2301.05586, 2023.

- [28]

Feng Li, Hao Zhang, Shilong Liu, Jian Guo, Lionel M Ni, and Lei Zhang.

Dn-detr: Accelerate detr training by introducing query denoising.

In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 13619–13627, 2022.

- [29]

Xiang Li, Wenhai Wang, Xiaolin Hu, Jun Li, Jinhui Tang, and Jian Yang.

Generalized focal loss v2: Learning reliable localization quality
estimation for dense object detection.

In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 11632–11641, 2021.

- [30]

Xiang Li, Wenhai Wang, Lijun Wu, Shuo Chen, Xiaolin Hu, Jun Li, Jinhui Tang,
and Jian Yang.

Generalized focal loss: Learning qualified and distributed bounding
boxes for dense object detection.

Advances in Neural Information Processing Systems,
33:21002–21012, 2020.

- [31]

Ming Lin, Hesen Chen, Xiuyu Sun, Qi Qian, Hao Li, and Rong Jin.

Neural architecture design for gpu-efficient networks.

arXiv preprint arXiv:2006.14090, 2020.

- [32]

Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár.

Focal loss for dense object detection.

In Proceedings of the IEEE international conference on computer
vision, pages 2980–2988, 2017.

- [33]

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva
Ramanan, Piotr Dollár, and C Lawrence Zitnick.

Microsoft coco: Common objects in context.

In Computer Vision–ECCV 2014: 13th European Conference, Zurich,
Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755.
Springer, 2014.

- [34]

Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, and
Lei Zhang.

Dab-detr: Dynamic anchor boxes are better queries for detr.

arXiv preprint arXiv:2201.12329, 2022.

- [35]

Shu Liu, Lu Qi, Haifang Qin, Jianping Shi, and Jiaya Jia.

Path aggregation network for instance segmentation.

In Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 8759–8768, 2018.

- [36]

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and
Baining Guo.

Swin transformer: Hierarchical vision transformer using shifted
windows.

In Proceedings of the IEEE/CVF international conference on
computer vision, pages 10012–10022, 2021.

- [37]

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell,
and Saining Xie.

A convnet for the 2020s.

In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 11976–11986, 2022.

- [38]

Wenjie Luo, Yujia Li, Raquel Urtasun, and Richard Zemel.

Understanding the effective receptive field in deep convolutional
neural networks.

Advances in neural information processing systems, 29, 2016.

- [39]

Chengqi Lyu, Wenwei Zhang, Haian Huang, Yue Zhou, Yudong Wang, Yanyi Liu,
Shilong Zhang, and Kai Chen.

Rtmdet: An empirical study of designing real-time object detectors.

arXiv preprint arXiv:2212.07784, 2022.

- [40]

Depu Meng, Xiaokang Chen, Zejia Fan, Gang Zeng, Houqiang Li, Yuhui Yuan, Lei
Sun, and Jingdong Wang.

Conditional detr for fast training convergence.

In Proceedings of the IEEE/CVF international conference on
computer vision, pages 3651–3660, 2021.

- [41]

Victor M Panaretos and Yoav Zemel.

Statistical aspects of wasserstein distances.

Annual review of statistics and its application, 6:405–431,
2019.

- [42]

Joseph Redmon.

Darknet: Open source neural networks in c.

http://pjreddie.com/darknet/, 2013–2016.

- [43]

Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi.

You only look once: Unified, real-time object detection.

In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), June 2016.

- [44]

Joseph Redmon and Ali Farhadi.

Yolo9000: Better, faster, stronger.

In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), July 2017.

- [45]

Joseph Redmon and Ali Farhadi.

Yolov3: An incremental improvement, 2018.

- [46]

Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh
Chen.

Mobilenetv2: Inverted residuals and linear bottlenecks.

In Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 4510–4520, 2018.

- [47]

Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing
Li, and Jian Sun.

Objects365: A large-scale, high-quality dataset for object detection.

In Proceedings of the IEEE/CVF international conference on
computer vision, pages 8430–8439, 2019.

- [48]

Russell Stewart, Mykhaylo Andriluka, and Andrew Y Ng.

End-to-end people detection in crowded scenes.

In Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 2325–2333, 2016.

- [49]

Peize Sun, Yi Jiang, Enze Xie, Wenqi Shao, Zehuan Yuan, Changhu Wang, and Ping
Luo.

What makes for end-to-end object detection?

In International Conference on Machine Learning, pages
9934–9944. PMLR, 2021.

- [50]

Peize Sun, Rufeng Zhang, Yi Jiang, Tao Kong, Chenfeng Xu, Wei Zhan, Masayoshi
Tomizuka, Lei Li, Zehuan Yuan, Changhu Wang, et al.

Sparse r-cnn: End-to-end object detection with learnable proposals.

In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 14454–14463, 2021.

- [51]

Zhi Tian, Chunhua Shen, Hao Chen, and Tong He.

Fcos: A simple and strong anchor-free object detector.

IEEE Transactions on Pattern Analysis and Machine Intelligence,
44(4):1922–1933, 2020.

- [52]

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.

Attention is all you need.

Advances in neural information processing systems, 30, 2017.

- [53]

Ao Wang, Hui Chen, Zijia Lin, Hengjun Pu, and Guiguang Ding.

Repvit: Revisiting mobile cnn from vit perspective.

arXiv preprint arXiv:2307.09283, 2023.

- [54]

Chengcheng Wang, Wei He, Ying Nie, Jianyuan Guo, Chuanjian Liu, Yunhe Wang, and
Kai Han.

Gold-yolo: Efficient object detector via gather-and-distribute
mechanism.

Advances in Neural Information Processing Systems, 36, 2024.

- [55]

Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao.

Scaled-yolov4: Scaling cross stage partial network.

In Proceedings of the IEEE/cvf conference on computer vision and
pattern recognition, pages 13029–13038, 2021.

- [56]

Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao.

Yolov7: Trainable bag-of-freebies sets new state-of-the-art for
real-time object detectors.

In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 7464–7475, 2023.

- [57]

Chien-Yao Wang, Hong-Yuan Mark Liao, Yueh-Hua Wu, Ping-Yang Chen, Jun-Wei
Hsieh, and I-Hau Yeh.

Cspnet: A new backbone that can enhance learning capability of cnn.

In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition workshops, pages 390–391, 2020.

- [58]

Chien-Yao Wang, Hong-Yuan Mark Liao, and I-Hau Yeh.

Designing network design strategies through gradient path analysis.

arXiv preprint arXiv:2211.04800, 2022.

- [59]

Chien-Yao Wang, I-Hau Yeh, and Hong-Yuan Mark Liao.

Yolov9: Learning what you want to learn using programmable gradient
information.

arXiv preprint arXiv:2402.13616, 2024.

- [60]

Jianfeng Wang, Lin Song, Zeming Li, Hongbin Sun, Jian Sun, and Nanning Zheng.

End-to-end object detection with fully convolutional network.

In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 15849–15858, 2021.

- [61]

Yingming Wang, Xiangyu Zhang, Tong Yang, and Jian Sun.

Anchor detr: Query design for transformer-based detector.

In Proceedings of the AAAI conference on artificial
intelligence, volume 36, pages 2567–2575, 2022.

- [62]

Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, and Lei
Zhang.

Cvt: Introducing convolutions to vision transformers.

In Proceedings of the IEEE/CVF international conference on
computer vision, pages 22–31, 2021.

- [63]

Haiyang Xu, Zhichao Zhou, Dongliang He, Fu Li, and Jingdong Wang.

Vision transformer with attention map hallucination and ffn
compaction.

arXiv preprint arXiv:2306.10875, 2023.

- [64]

Shangliang Xu, Xinxin Wang, Wenyu Lv, Qinyao Chang, Cheng Cui, Kaipeng Deng,
Guanzhong Wang, Qingqing Dang, Shengyu Wei, Yuning Du, et al.

Pp-yoloe: An evolved version of yolo.

arXiv preprint arXiv:2203.16250, 2022.

- [65]

Xianzhe Xu, Yiqi Jiang, Weihua Chen, Yilun Huang, Yuan Zhang, and Xiuyu Sun.

Damo-yolo: A report on real-time object detection design.

arXiv preprint arXiv:2211.15444, 2022.

- [66]

Fangao Zeng, Bin Dong, Yuang Zhang, Tiancai Wang, Xiangyu Zhang, and Yichen
Wei.

Motr: End-to-end multiple-object tracking with transformer.

In European Conference on Computer Vision, pages 659–675.
Springer, 2022.

- [67]

Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M Ni, and
Heung-Yeung Shum.

Dino: Detr with improved denoising anchor boxes for end-to-end object
detection.

arXiv preprint arXiv:2203.03605, 2022.

- [68]

Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz.

mixup: Beyond empirical risk minimization.

arXiv preprint arXiv:1710.09412, 2017.

- [69]

Shifeng Zhang, Cheng Chi, Yongqiang Yao, Zhen Lei, and Stan Z Li.

Bridging the gap between anchor-based and anchor-free detection via
adaptive training sample selection.

In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 9759–9768, 2020.

- [70]

Wenqiang Zhang, Zilong Huang, Guozhong Luo, Tao Chen, Xinggang Wang, Wenyu Liu,
Gang Yu, and Chunhua Shen.

Topformer: Token pyramid transformer for mobile semantic
segmentation.

In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 12083–12093, 2022.

- [71]

Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang, Qingqing Dang,
Yi Liu, and Jie Chen.

Detrs beat yolos on real-time object detection.

arXiv preprint arXiv:2304.08069, 2023.

- [72]

Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, and Dongwei Ren.

Distance-iou loss: Faster and better learning for bounding box
regression.

In Proceedings of the AAAI conference on artificial
intelligence, volume 34, pages 12993–13000, 2020.

- [73]

Qiang Zhou and Chaohui Yu.

Object detection made simpler by eliminating heuristic nms.

IEEE Transactions on Multimedia, 2023.

- [74]

Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai.

Deformable detr: Deformable transformers for end-to-end object
detection.

arXiv preprint arXiv:2010.04159, 2020.

- [75]

Zhuofan Zong, Guanglu Song, and Yu Liu.

Detrs with collaborative hybrid assignments training.

In Proceedings of the IEEE/CVF international conference on
computer vision, pages 6748–6758, 2023.

## Appendix A Appendix

### A.1 Implementation Details

Following [20, 56, 59], all YOLOv10 models are trained from scratch using the SGD optimizer for 500 epochs. The SGD momentum and weight decay are set to 0.937 and 5×\times10-4, respectively. The initial learning rate is 1×\times10-2 and it decays linearly to 1×\times10-4. For data augmentation, we adopt the Mosaic [2, 19], Mixup [68] and copy-paste augmentation [17], etc., like [20, 59]. Tab. 14 presents the detailed hyper-parameters. All models are trained on 8 NVIDIA 3090 GPUs. Besides, we increase the width scale factor of YOLOv10-M to 1.0 to obtain YOLOv10-B. For PSA, we employ it after the SPPF module [20] and adopt the expansion factor of 2 for FFN. For CIB, we also adopt the expansion ratio of 2 for the inverted bottleneck block structure. Following [59, 56], we report the standard mean average precision (AP) across different object scales and IoU thresholds on the COCO dataset [33].

Moreover, we follow [71] to establish the end-to-end speed benchmark. Since the execution time of NMS is affected by the input, we thus measure the latency on the COCO val set, like [71]. We adopt the same NMS hyperparameters used by the detectors during their validation. The TensorRT efficientNMSPlugin is appended for post-processing and the I/O overhead is omitted. We report the average latency across all images.

hyper-parameter
YOLOv10-N/S/M/B/L/X

epochs
500

optimizer
SGD

momentum
0.937

weight decay

5×\times10-4

warm-up epochs
3

warm-up momentum
0.8

warm-up bias learning rate
0.1

initial learning rate

10-2

final learning rate

10-4

learning rate schedule
linear decay

box loss gain
7.5

class loss gain
0.5

DFL loss gain
1.5

HSV saturation augmentation
0.7

HSV value augmentation
0.4

HSV hue augmentation
0.015

translation augmentation
0.1

scale augmentation
0.5/0.5/0.9/0.9/0.9/0.9

mosaic augmentation
1.0

Mixup augmentation
0.0/0.0/0.1/0.1/0.15/0.15

copy-paste augmentation
0.0/0.0/0.1/0.1/0.3/0.3

close mosaic epochs
10

### A.2 Details of Consistent Matching Metric

We provide the detailed derivation of consistent matching metric here.

As mentioned in the paper, we suppose that the one-to-many positive samples is ΩΩ\Omega and the one-to-one branch selects i𝑖i-th prediction. We can then leverage the normalized metric [14] to obtain the classification target for task alignment learning [20, 14, 59, 27, 64], i.e., to​2​m,j=u∗⋅mo​2​m,jmo​2​m∗≤u∗subscript𝑡𝑜2𝑚𝑗⋅superscript𝑢subscript𝑚𝑜2𝑚𝑗superscriptsubscript𝑚𝑜2𝑚superscript𝑢t_{o2m,j}=u^{*}\cdot\frac{m_{o2m,j}}{m_{o2m}^{*}}\leq u^{*} for j∈Ω𝑗Ωj\in\Omega and to​2​o,i=u∗⋅mo​2​o,imo​2​o∗=u∗subscript𝑡𝑜2𝑜𝑖⋅superscript𝑢subscript𝑚𝑜2𝑜𝑖superscriptsubscript𝑚𝑜2𝑜superscript𝑢t_{o2o,i}=u^{*}\cdot\frac{m_{o2o,i}}{m_{o2o}^{*}}=u^{*}. We can thus derive the supervision gap between two branches by the 1-Wasserstein distance [41] of the different classification targets, i.e.,

A𝐴\displaystyle A
=|(1−to​2​o,i)−(1−𝕀​(i∈Ω)​to​2​m,i)|+∑k∈Ω\{i}|1−(1−to​2​m,k)|absent1subscript𝑡𝑜2𝑜𝑖1𝕀𝑖Ωsubscript𝑡𝑜2𝑚𝑖subscript𝑘\Ω𝑖11subscript𝑡𝑜2𝑚𝑘\displaystyle=|(1-t_{o2o,i})-(1-\mathbb{I}(i\in\Omega)t_{o2m,i})|+\sum\nolimits_{k\in\Omega\backslash\{i\}}|1-(1-t_{o2m,k})|

(3)

=|to​2​o,i−𝕀​(i∈Ω)​to​2​m,i|+∑k∈Ω\{i}to​2​m,kabsentsubscript𝑡𝑜2𝑜𝑖𝕀𝑖Ωsubscript𝑡𝑜2𝑚𝑖subscript𝑘\Ω𝑖subscript𝑡𝑜2𝑚𝑘\displaystyle=|t_{o2o,i}-\mathbb{I}(i\in\Omega)t_{o2m,i}|+\sum\nolimits_{k\in\Omega\backslash\{i\}}t_{o2m,k}

=to​2​o,i−𝕀​(i∈Ω)​to​2​m,i+∑k∈Ω\{i}to​2​m,k,absentsubscript𝑡𝑜2𝑜𝑖𝕀𝑖Ωsubscript𝑡𝑜2𝑚𝑖subscript𝑘\Ω𝑖subscript𝑡𝑜2𝑚𝑘\displaystyle=t_{o2o,i}-\mathbb{I}(i\in\Omega)t_{o2m,i}+\sum\nolimits_{k\in\Omega\backslash\{i\}}t_{o2m,k},

where 𝕀​(⋅)𝕀⋅\mathbb{I}(\cdot) is the indicator function. We denote the classification targets of the predictions in ΩΩ\Omega as {t^1,t^2,…,t^|Ω|}subscript^𝑡1subscript^𝑡2…subscript^𝑡Ω\{\hat{t}_{1},\hat{t}_{2},...,\hat{t}_{|\Omega|}\} in descending order, with t^1≥t^2≥…≥t^|Ω|subscript^𝑡1subscript^𝑡2…subscript^𝑡Ω\hat{t}_{1}\geq\hat{t}_{2}\geq...\geq\hat{t}_{|\Omega|}. We can then replace to​2​o,isubscript𝑡𝑜2𝑜𝑖t_{o2o,i} with u∗superscript𝑢u^{*} and obtain:

A𝐴\displaystyle A
=u∗−𝕀​(i∈Ω)​to​2​m,i+∑k∈Ω\{i}to​2​m,kabsentsuperscript𝑢𝕀𝑖Ωsubscript𝑡𝑜2𝑚𝑖subscript𝑘\Ω𝑖subscript𝑡𝑜2𝑚𝑘\displaystyle=u^{*}-\mathbb{I}(i\in\Omega)t_{o2m,i}+\sum\nolimits_{k\in\Omega\backslash\{i\}}t_{o2m,k}

(4)

=u∗+∑k∈Ωto​2​m,k−2⋅𝕀​(i∈Ω)​to​2​m,iabsentsuperscript𝑢subscript𝑘Ωsubscript𝑡𝑜2𝑚𝑘⋅2𝕀𝑖Ωsubscript𝑡𝑜2𝑚𝑖\displaystyle=u^{*}+\sum\nolimits_{k\in\Omega}t_{o2m,k}-2\cdot\mathbb{I}(i\in\Omega)t_{o2m,i}

=u∗+∑k=1|Ω|t^k−2⋅𝕀​(i∈Ω)​to​2​m,iabsentsuperscript𝑢superscriptsubscript𝑘1Ωsubscript^𝑡𝑘⋅2𝕀𝑖Ωsubscript𝑡𝑜2𝑚𝑖\displaystyle=u^{*}+\sum\nolimits_{k=1}^{|\Omega|}\hat{t}_{k}-2\cdot\mathbb{I}(i\in\Omega)t_{o2m,i}

We further discuss the supervision gap in two scenarios, i.e.,

- 1.

Supposing i∉Ω𝑖Ωi\not\in\Omega, we can obtain:

A=u∗+∑k=1|Ω|t^k𝐴superscript𝑢superscriptsubscript𝑘1Ωsubscript^𝑡𝑘A=u^{*}+\sum\nolimits_{k=1}^{|\Omega|}\hat{t}_{k}

(5)

- 2.

Supposing i∈Ω𝑖Ωi\in\Omega, we denote to​2​m,i=t^nsubscript𝑡𝑜2𝑚𝑖subscript^𝑡𝑛t_{o2m,i}=\hat{t}_{n} and obtain:

A=u∗+∑k=1|Ω|t^k−2⋅t^n𝐴superscript𝑢superscriptsubscript𝑘1Ωsubscript^𝑡𝑘⋅2subscript^𝑡𝑛A=u^{*}+\sum\nolimits_{k=1}^{|\Omega|}\hat{t}_{k}-2\cdot\hat{t}_{n}

(6)

Due to t^n≥0subscript^𝑡𝑛0\hat{t}_{n}\geq 0, the second case can lead to smaller supervision gap. Besides, we can observe that A𝐴A decreases as t^nsubscript^𝑡𝑛\hat{t}_{n} increases, indicating that n𝑛n decreases and the ranking of i𝑖i within ΩΩ\Omega improves. Due to t^n≤t^1subscript^𝑡𝑛subscript^𝑡1\hat{t}_{n}\leq\hat{t}_{1}, A𝐴A thus achieves the minimum when t^n=t^1subscript^𝑡𝑛subscript^𝑡1\hat{t}_{n}=\hat{t}_{1}, i.e., i𝑖i is the best positive sample in ΩΩ\Omega with mo​2​m,i=mo​2​m∗subscript𝑚𝑜2𝑚𝑖superscriptsubscript𝑚𝑜2𝑚m_{o2m,i}=m_{o2m}^{*} and to​2​m,i=u∗⋅mo​2​m,imo​2​m∗=u∗subscript𝑡𝑜2𝑚𝑖⋅superscript𝑢subscript𝑚𝑜2𝑚𝑖superscriptsubscript𝑚𝑜2𝑚superscript𝑢t_{o2m,i}=u^{*}\cdot\frac{m_{o2m,i}}{m_{o2m}^{*}}=u^{*}.

Furthermore, we prove that we can achieve the minimized supervision gap by the consistent matching metric. We suppose αo​2​m>0subscript𝛼𝑜2𝑚0\alpha_{o2m}>0 and βo​2​m>0subscript𝛽𝑜2𝑚0\beta_{o2m}>0, which are common in [20, 59, 27, 14, 64]. Similarly, we assume αo​2​o>0subscript𝛼𝑜2𝑜0\alpha_{o2o}>0 and βo​2​o>0subscript𝛽𝑜2𝑜0\beta_{o2o}>0. We can obtain r1=αo​2​oαo​2​m>0subscript𝑟1subscript𝛼𝑜2𝑜subscript𝛼𝑜2𝑚0r_{1}=\frac{\alpha_{o2o}}{\alpha_{o2m}}>0 and r2=βo​2​oβo​2​m>0subscript𝑟2subscript𝛽𝑜2𝑜subscript𝛽𝑜2𝑚0r_{2}=\frac{\beta_{o2o}}{\beta_{o2m}}>0, and then derive mo​2​osubscript𝑚𝑜2𝑜m_{o2o} by

mo​2​osubscript𝑚𝑜2𝑜\displaystyle m_{o2o}
=s⋅pαo​2​o⋅IoU​(b^,b)βo​2​oabsent⋅𝑠superscript𝑝subscript𝛼𝑜2𝑜IoUsuperscript^𝑏𝑏subscript𝛽𝑜2𝑜\displaystyle=s\cdot p^{\alpha_{o2o}}\cdot\text{IoU}(\hat{b},b)^{\beta_{o2o}}

(7)

=s⋅pr1⋅αo​2​m⋅IoU​(b^,b)r2⋅βo​2​mabsent⋅𝑠superscript𝑝⋅subscript𝑟1subscript𝛼𝑜2𝑚IoUsuperscript^𝑏𝑏⋅subscript𝑟2subscript𝛽𝑜2𝑚\displaystyle=s\cdot p^{r_{1}\cdot\alpha_{o2m}}\cdot\text{IoU}(\hat{b},b)^{r_{2}\cdot\beta_{o2m}}

=s⋅(pαo​2​m⋅IoU​(b^,b)βo​2​m)r1⋅IoU​(b^,b)(r2−r1)⋅βo​2​mabsent⋅𝑠superscript⋅superscript𝑝subscript𝛼𝑜2𝑚IoUsuperscript^𝑏𝑏subscript𝛽𝑜2𝑚subscript𝑟1IoUsuperscript^𝑏𝑏⋅subscript𝑟2subscript𝑟1subscript𝛽𝑜2𝑚\displaystyle=s\cdot(p^{\alpha_{o2m}}\cdot\text{IoU}(\hat{b},b)^{\beta_{o2m}})^{r_{1}}\cdot\text{IoU}(\hat{b},b)^{(r_{2}-r_{1})\cdot\beta_{o2m}}

=mo​2​mr1⋅IoU​(b^,b)(r2−r1)⋅βo​2​mabsent⋅superscriptsubscript𝑚𝑜2𝑚subscript𝑟1IoUsuperscript^𝑏𝑏⋅subscript𝑟2subscript𝑟1subscript𝛽𝑜2𝑚\displaystyle=m_{o2m}^{r_{1}}\cdot\text{IoU}(\hat{b},b)^{(r_{2}-r_{1})\cdot\beta_{o2m}}

To achieve mo​2​m,i=mo​2​m∗subscript𝑚𝑜2𝑚𝑖superscriptsubscript𝑚𝑜2𝑚m_{o2m,i}=m_{o2m}^{*} and mo​2​o,i=mo​2​o∗subscript𝑚𝑜2𝑜𝑖superscriptsubscript𝑚𝑜2𝑜m_{o2o,i}=m_{o2o}^{*}, we can make mo​2​osubscript𝑚𝑜2𝑜m_{o2o} monotonically increase with mo​2​msubscript𝑚𝑜2𝑚m_{o2m} by assigning (r2−r1)=0subscript𝑟2subscript𝑟10(r_{2}-r_{1})=0, i.e.,

mo​2​osubscript𝑚𝑜2𝑜\displaystyle m_{o2o}
=mo​2​mr1⋅IoU​(b^,b)0⋅βo​2​mabsent⋅superscriptsubscript𝑚𝑜2𝑚subscript𝑟1IoUsuperscript^𝑏𝑏⋅0subscript𝛽𝑜2𝑚\displaystyle=m_{o2m}^{r_{1}}\cdot\text{IoU}(\hat{b},b)^{0\cdot\beta_{o2m}}

(8)

=mo​2​mr1absentsuperscriptsubscript𝑚𝑜2𝑚subscript𝑟1\displaystyle=m_{o2m}^{r_{1}}

Supposing r1=r2=rsubscript𝑟1subscript𝑟2𝑟r_{1}=r_{2}=r, we can thus derive the consistent matching metric, i.e., αo​2​o=r⋅αo​2​msubscript𝛼𝑜2𝑜⋅𝑟subscript𝛼𝑜2𝑚\alpha_{o2o}=r\cdot\alpha_{o2m} and βo​2​o=r⋅βo​2​msubscript𝛽𝑜2𝑜⋅𝑟subscript𝛽𝑜2𝑚\beta_{o2o}=r\cdot\beta_{o2m}. By simply taking r=1𝑟1r=1, we obtain αo​2​o=αo​2​msubscript𝛼𝑜2𝑜subscript𝛼𝑜2𝑚\alpha_{o2o}=\alpha_{o2m} and βo​2​o=βo​2​msubscript𝛽𝑜2𝑜subscript𝛽𝑜2𝑚\beta_{o2o}=\beta_{o2m}.

### A.3 Details of Rank-Guided Block Design

We present the details of the algorithm of rank-guided block design in Algorithm 1. Besides, to calculate the numerical rank of the convolution, we reshape its weight to the shape of (Cosubscript𝐶𝑜C_{o}, K2×Cisuperscript𝐾2subscript𝐶𝑖K^{2}\times C_{i}), where Cosubscript𝐶𝑜C_{o} and Cisubscript𝐶𝑖C_{i} denote the number of output and input channels, and K𝐾K means the kernel size, respectively.

Input: Intrinsic ranks R𝑅R for all stages S𝑆S; Original Network ΘΘ\Theta; CIB θc​i​bsubscript𝜃𝑐𝑖𝑏\theta_{cib};

Output: New network Θ∗superscriptΘ\Theta^{*} with CIB for certain stages.

1
t←0←𝑡0t\leftarrow 0;

2
Θ0←Θ←subscriptΘ0Θ\Theta_{0}\leftarrow\Theta; Θ∗←Θ0←superscriptΘsubscriptΘ0\Theta^{*}\leftarrow\Theta_{0};

a​p0←AP​(T​(Θ0))←𝑎subscript𝑝0APTsubscriptΘ0ap_{0}\leftarrow\text{AP}(\text{T}(\Theta_{0})) ;

// T:training the network; AP:evaluating the AP performance.

3
while S≠∅𝑆S\neq\emptyset do

4 
𝒔t←argmins∈SR←subscript𝒔𝑡subscriptargmin𝑠𝑆𝑅\boldsymbol{s}_{t}\leftarrow\operatorname*{argmin}_{s\in S}R;

Θt+1←Replace​(Θt,θc​i​b,𝒔t)←subscriptΘ𝑡1ReplacesubscriptΘ𝑡subscript𝜃𝑐𝑖𝑏subscript𝒔𝑡\Theta_{t+1}\leftarrow\text{Replace}(\Theta_{t},\theta_{cib},\boldsymbol{s}_{t}) ;

// Replace the block in Stage 𝒔tsubscript𝒔𝑡\boldsymbol{s}_{t} of ΘtsubscriptΘ𝑡\Theta_{t} with CIB θc​i​bsubscript𝜃𝑐𝑖𝑏\theta_{cib}.

5 
a​pt+1←AP​(T​(Θt+1))←𝑎subscript𝑝𝑡1APTsubscriptΘ𝑡1ap_{t+1}\leftarrow\text{AP}(\text{T}(\Theta_{t+1}));

6 
if a​pt+1≥a​p0𝑎subscript𝑝𝑡1𝑎subscript𝑝0ap_{t+1}\geq ap_{0} then

7 
Θ∗←Θt+1←superscriptΘsubscriptΘ𝑡1\Theta^{*}\leftarrow\Theta_{t+1};
S←S∖{𝒔t}←𝑆𝑆subscript𝒔𝑡S\leftarrow S\setminus\{\boldsymbol{s}_{t}\};

8 else

9 
return Θ∗superscriptΘ\Theta^{*};

10

11 end if

12

13 end while

14return Θ∗superscriptΘ\Theta^{*};

### A.4 More Results on COCO

We report the detailed performance of YOLOv10 on COCO, including AP50v​a​lsubscriptsuperscriptabsent𝑣𝑎𝑙50{}^{val}_{50} and AP75v​a​lsubscriptsuperscriptabsent𝑣𝑎𝑙75{}^{val}_{75} at different IoU thresholds, as well as APs​m​a​l​lv​a​lsubscriptsuperscriptabsent𝑣𝑎𝑙𝑠𝑚𝑎𝑙𝑙{}^{val}_{small}, APm​e​d​i​u​mv​a​lsubscriptsuperscriptabsent𝑣𝑎𝑙𝑚𝑒𝑑𝑖𝑢𝑚{}^{val}_{medium}, and APl​a​r​g​ev​a​lsubscriptsuperscriptabsent𝑣𝑎𝑙𝑙𝑎𝑟𝑔𝑒{}^{val}_{large} across different scales, in Tab. 15.

### A.5 More Analyses for Holistic Efficiency-Accuracy Driven Model Design

We note that reducing the latency of YOLOv10-S (#2 in Tab. 2) is particularly challenging due to its small model scale. However, as shown in Tab. 2, our efficiency driven model design still achieves a 5.3% reduction in latency without compromising performance. This provides substantial support for the further accuracy driven model design. YOLOv10-S achieves a better latency-accuracy trade-off with our holistic efficiency-accuracy driven model design, showing a 2.0% AP improvement with only 0.05ms latency overhead. Besides, for YOLOv10-M (#6 in Tab. 2), which has a larger model scale and more redundancy, our efficiency driven model design results in a considerable 12.5% latency reduction, as shown in Tab. 2. When combined with accuracy driven model design, we observe a notable 0.8% AP improvement for YOLOv10-M, along with a favorable latency reduction of 0.48ms. These results well demonstrate the effectiveness of our design strategy across different model scales.

Model

APval(%)

APv​a​l50superscriptsubscriptabsent50𝑣𝑎𝑙{}_{50}^{val}(%)

APv​a​l75superscriptsubscriptabsent75𝑣𝑎𝑙{}_{75}^{val}(%)

APv​a​ls​m​a​l​lsuperscriptsubscriptabsent𝑠𝑚𝑎𝑙𝑙𝑣𝑎𝑙{}_{small}^{val}(%)

APv​a​lm​e​d​i​u​msuperscriptsubscriptabsent𝑚𝑒𝑑𝑖𝑢𝑚𝑣𝑎𝑙{}_{medium}^{val}(%)

APv​a​ll​a​r​g​esuperscriptsubscriptabsent𝑙𝑎𝑟𝑔𝑒𝑣𝑎𝑙{}_{large}^{val}(%)

YOLOv10-N
38.5
53.8
41.7
18.9
42.4
54.6

YOLOv10-S
46.3
63.0
50.4
26.8
51.0
63.8

YOLOv10-M
51.1
68.1
55.8
33.8
56.5
67.0

YOLOv10-B
52.5
69.6
57.2
35.1
57.8
68.5

YOLOv10-L
53.2
70.1
58.1
35.8
58.5
69.4

YOLOv10-X
54.4
71.3
59.3
37.0
59.8
70.9

### A.6 Visualization Results

Fig. 4 presents the visualization results of our YOLOv10 in the complex and challenging scenarios. It can be observed that YOLOv10 can achieve precise detection under various difficult conditions, such as low light, rotation, etc. It also demonstrates a strong capability in detecting diverse and densely packed objects, such as bottle, cup, and person. These results indicate its superior performance.

### A.7 Contribution, Limitation, and Broader Impact

Contribution. In summary, our contributions are three folds as follows:

- 1.

We present a novel consistent dual assignments strategy for NMS-free YOLOs. A dual label assignments way is designed to provide rich supervision by one-to-many branch during training and high efficiency by one-to-one branch during inference. Besides, to ensure the harmonious supervision between two branches, we innovatively propose the consistent matching metric, which can well reduce the theoretical supervision gap and lead to improved performance.

- 2.

We propose a holistic efficiency-accuracy driven model design strategy for the model architecture of YOLOs. We present novel lightweight classification head, spatial-channel decoupled downsampling, and rank-guided block design, which greatly reduce the computational redundancy and achieve high efficiency. We further introduce the large-kernel convolution and innovative partial self-attention module, which effectively enhance the performance under low cost.

- 3.

Based on the above approaches, we introduce YOLOv10, a new real-time end-to-end object detector. Extensive experiments demonstrate that our YOLOv10 achieves the state-of-the-art performance and efficiency trade-offs compared with other advanced detectors.

Limitation. Due to the limited computational resources, we do not investigate the pretraining of YOLOv10 on large-scale datasets, e.g., Objects365 [47]. Besides, although we can achieve competitive end-to-end performance using the one-to-one head under NMS-free training, there still exists a performance gap compared with the original one-to-many training using NMS, especially noticeable in small models. For example, in YOLOv10-N and YOLOv10-S, the performance of one-to-many training with NMS outperforms that of NMS-free training by 1.0% AP and 0.5% AP, respectively. We will explore ways to further reduce the gap and achieve higher performance for YOLOv10 in the future work.

Broader impact. The YOLOs can be widely applied in various real-world applications, including medical image analyses and autonomous driving, etc. We hope that our YOLOv10 can assist in these fields and improve the efficiency. However, we acknowledge the potential for malicious use of our models. We will make every effort to prevent this.

Generated on Wed Jun 5 16:56:58 2024 by LaTeXML

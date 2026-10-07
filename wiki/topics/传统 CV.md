---
type: topic
status: formal
review_scope: evidence_synthesis
reviewed: 2026-10-07
---
# 传统 CV

## TL;DR（快速导读）

视觉方法可以按三层阅读：如何学习表示、如何定义任务输出、如何在真实设备运行。CNN、ViT、文档模型与生成式感知改变的是不同层，不能用一个“统一视觉”的口号合并。

阅读重点：先按问题选择路线，再核对比较条件与证据边界。

## 先用一个问题理解

图像分类输出类别，检测还输出位置，OCR 输出文字，表格解析还要输出结构。它们都处理图像，但监督与评价不同；下文用任务接口连接各条研究主线。

## 页面状态

正式 topic；2026-10-07 复核核心来源并补充方法比较。正文区分论文实验、作者报告和本文综合判断；开放问题表示研究证据的边界。

## 主题定义

本页讨论的是**不以通用多模态 LLM 为中心组织**的视觉研究主线。这里的“传统 CV”不是怀旧式标签，也不等于“卷积时代残余方法”；更准确地说，它指的是这样一类研究共同体：问题定义首先来自视觉任务本身，模型接口围绕分类、检测、OCR、文档解析、版面建模等任务目标展开，而不是先假设一个统一的对话式多模态代理，再把视觉能力嵌进去。

因此，本页的边界需要收紧到三个层面。第一，它讨论的是**视觉任务如何形成基础表征、任务接口与文档理解能力**，而不是所有非 LLM 论文的杂项汇编。第二，它把目标检测与 OCR 都视为已经足够成熟、足够独立的子主线，因此这里只保留它们在整体视觉谱系中的位置，不再在本页内部展开其细部综述。第三，文档 AI 与版面建模仍暂时留在本页，不是因为它们与通用视觉完全同质，而是因为当前知识库证据更支持它们作为“视觉结构化理解”路线的一部分，而不是已经可以稳定拆成更细的多个正式 topic。

从现有 summary 出发，本页最稳妥的定位是：它描述的是**视觉研究从经典 CNN backbone、任务专用 pipeline，走向统一 Transformer 表征，再进一步走向 layout-aware、generative 接口**的过渡地带。这个判断目前主要由经典 CNN 主线、`ViT`、`LayoutLMv3`、`TrOCR` 与 `Vision Banana` 这几类来源共同支撑。

## 核心问题

- **视觉基础架构是否必须以卷积归纳偏置为核心**，还是可以被 patch 化、token 化的统一 Transformer 表示改写。
- **经典 CNN 为什么并未随着 ViT 出现而立即失去解释力**，而是继续以 `ResNet / ConvNeXt / MobileNet` 等形式维持 backbone 地位。
- **文档与 OCR 为什么没有停留在“检测 + 识别 + 规则后处理”**，而是逐步转向预训练与生成式接口。
- **通用视觉、文档视觉、OCR、版面理解之间究竟共享多少表示层**，又在哪些任务边界上仍然必须分开讨论。
- **视觉研究的“统一化”究竟指什么**：是统一 backbone、统一预训练目标，还是统一为自然语言驱动的生成接口。
- **图像生成预训练能否反过来成为视觉理解底座**，即感知任务是否可以被稳定改写为可解码的图像生成任务。

## 主线脉络 / 方法分层

从当前证据看，本主题不宜按“模型家族名”来分，而应按**视觉对象和任务接口被如何重写**来分层。这样做的原因是：`ViT`、`LayoutLMv3`、`TrOCR` 虽然都与 Transformer 有关，但它们解决的并不是同一个问题。

- **经典卷积 backbone 层**：在当前知识库里，`VGG -> GoogLeNet -> ResNet -> DenseNet / ResNeXt -> MobileNet -> ConvNeXt` 已经足以构成一条独立主线。它们共同说明，传统视觉并不是一个被 `ViT` 直接替换的静态旧时代，而是一条围绕深度、连接方式、多分支、效率与现代化设计持续演化的 backbone 谱系。由于这条线已经具备独立综述价值，细节移交给 [经典 CNN 架构](./经典%20CNN%20架构.md)，本页只保留其在整体视觉史中的位置。
- **视觉表征基础转向层**：`ViT` 的意义不只是提出一个新分类器，而是把“图像可以被切成 token 序列，并直接送入 Transformer”变成可行命题。它真正改写的是视觉基础表征的组织方式，把“视觉模型是否必须显式保留 CNN 式局部归纳偏置”从默认前提改成开放问题。也正因为如此，`ViT` 在本页里不是一篇普通分类论文，而是后续文档视觉、跨模态视觉与视觉 Transformer 家族的共同起点之一。
- **文档多模态建模层**：`LayoutLMv3` 代表的是另一类问题设定。它不是要证明 Transformer 能否处理图像，而是要证明**文字、版面和图像区域可以在统一预训练目标下共同学习**。这条线的价值在于，它把文档 AI 从“先 OCR，再喂给下游模型”的松耦合流程，推进到 layout-aware 的统一表示学习框架。这里的核心对象不再是自然图像，而是带有强空间结构约束的视觉文档。
- **识别到生成接口层**：`TrOCR` 的关键不是把 OCR 精度再抬高一点，而是把文本识别从 `CNN/RNN + LM` 组合系统，改写成**图像编码器加文本生成器的端到端序列生成问题**。这意味着 OCR 在接口层上开始向生成模型靠拢，其输出不再只是中间模块结果，而是可以被更大生成式工作流吸收的自然语言序列。
- **视觉感知生成化层**：`Vision Banana` 把这一趋势从 OCR 扩展到更一般的 2D / 3D 视觉任务。它将 semantic segmentation、referring segmentation、metric depth、surface normal 等输出编码成 `RGB` 图像，让同一个图像生成模型通过 prompt 产生可解码的任务答案。对传统 CV 而言，这不是简单的多任务模型，而是在挑战“感知任务必须由专门判别式架构完成”的默认前提。
- **专门任务向独立 topic 外溢层**：目标检测与 OCR 在当前知识库里都已经形成独立主线。前者围绕 proposal、set prediction 与实时化接口展开，后者围绕生成式识别、阅读顺序与结构化转写展开。因此本页只保留它们作为传统视觉谱系关键支柱的定位，而不再承担细部综述；否则会削弱本页围绕“视觉表征统一化”和“文档视觉生成化”的主线。

把这几层放在一起，当前可以得到一个比原页更稳的综述判断：**传统 CV 在本库中的主线，不是“旧方法大全”，而是视觉 backbone 与任务接口如何从经典卷积谱系，转向更统一的 token 表示、layout-aware 预训练和生成式解码。** 其中经典 CNN 主线解决的是视觉 backbone 如何持续演化，`ViT` 改写基础表征范式，`LayoutLMv3` 解决结构化文档问题，`TrOCR` 解决识别接口生成化问题，`Vision Banana` 则把生成式图像预训练推向通用感知任务，而检测已经外溢成独立主题。

### 表征训练、感知输出与图像合成的边界

[SimSiam](../summaries/Chen%2C%20He%20-%202021%20-%20Exploring%20Simple%20Siamese%20Representation%20Learning.md)说明不靠显式负样本也能学习图像表示，但依赖 stop-gradient 等训练设计；它既不是无需防止坍塌，也不是任意结构都有效。[ViT](../summaries/Dosovitskiy%20et%20al.%20-%202020%20-%20An%20Image%20is%20Worth%2016x16%20Words%20Transformers%20for%20Image%20Recognition%20at%20Scale.md)依赖数据与预训练规模，[视觉 Transformer 综述](../summaries/Khan%20et%20al.%20-%202021%20-%20Transformers%20in%20Vision%20A%20Survey.md)组织已有路线，二者不能直接证明 Transformer 在所有小数据任务占优。[FLIP](../summaries/Li%2C%20Fan%2C%20Ai%20-%20Unknown%20-%20Scaling%20Language-Image%20Pre-training%20via%20Masking.md)在图文预训练中遮蔽图像 patches 来节省训练计算，其结果仍需记录 masking、数据和微调协议。

模型学得表示后，还要规定输出。检测给框，文档模型给字段或文本，matting 给 alpha；它们的错误代价不同。[Background Matting V2](../summaries/Lin%20et%20al.%20-%202021%20-%20Real-Time%20High-Resolution%20Background%20Matting.md)使用额外干净背景输入，[Robust Video Matting](../summaries/Lin%20et%20al.%20-%202022%20-%20Robust%20High-Resolution%20Video%20Matting%20with%20Temporal%20Guidance.md)利用时序状态，两者虽都抠像，输入条件不能抹掉。单帧边缘好也不说明视频没有闪烁；时序稳定与精细边缘需要分别衡量。

[Vision Banana](../summaries/Gabeur%20et%20al.%20-%202026%20-%20Image%20Generators%20are%20Generalist%20Vision%20Learners.md)尝试把分割、深度与法线编码成生成图像。这是对任务输出形式的重写，并不让度量尺度、边界误差或实例对应失去意义。闭源底座与数据不完全透明还限制了归因：结果能支持这一方案在测量任务中可行，不能证明生成预训练是唯一有效原因。

| 层次 | 问题 | 合适的比较条件 |
| --- | --- | --- |
| 表征学习 | 如何从数据学可迁移特征 | 数据、预训练预算、下游微调 |
| 任务接口 | 框、mask、深度、文字如何输出 | 输出定义、容差和任务指标 |
| 合成/渲染 | 如何生成图像或新视点 | 身份、几何、视角与时序一致性 |
| 部署 | 多快、多省、是否稳定 | 设备、分辨率、精度与批量 |

[神经渲染综述](../summaries/Tewari%20et%20al.%20-%202020%20-%20State%20of%20the%20Art%20on%20Neural%20Rendering.md)和[神经渲染进展](../summaries/Tewari%20et%20al.%20-%202021%20-%20Advances%20in%20neural%20rendering.md)补充由几何、场景表示和学习结合的路线。本文把它作为邻接阅读路径，保留“从图像识别世界”与“从场景条件生成视图”的差别；它不能因为使用神经网络就被归入分类 backbone 的同一性能表。

## 关键争论与分歧

- **Transformer 是否已经“统一了视觉”**：现有证据只支持它已改写视觉基础架构与文档建模方式，不支持“所有视觉子任务都已被同一种训练范式稳定统一”。`ViT` 支撑的是基础表征层转向，`LayoutLMv3` 和 `TrOCR` 支撑的是部分任务接口统一，不能机械外推为全部视觉问题都已收敛。
- **文档 AI 应否继续留在传统 CV 中**：当前这样组织是合理的，因为 `LayoutLMv3` 与 `TrOCR` 仍然体现出强视觉结构约束与任务专用接口；但若后续知识库补入更多 `DocLLM`、通用文档 agent、版面生成与文档问答来源，文档 AI 可能更适合升级为独立 topic。也就是说，这个争论目前的成立条件是**证据面是否仍主要围绕版面与识别，而非围绕通用多模态推理**。
- **OCR 是否已经从识别任务变成纯生成任务**：`TrOCR` 证明生成式接口在 OCR 中可行且有效，但现有证据不足以说明 OCR 的评测逻辑、错误模式和数据依赖已经完全等同于通用文本生成。更稳妥的说法是：OCR 的**模型接口生成化了**，而问题本体并未因此消失。
- **生成式视觉预训练是否已经统一 CV**：`Vision Banana` 提供了强证据，说明强图像生成器经过轻量 instruction tuning 后可以在多类分割、深度与表面法线任务上达到接近或超过专门模型的结果。但由于其底座闭源、训练数据不完全透明、推理成本较高，且 instance segmentation 等任务仍存在短板，当前不能把它解释为“传统 CV 已经被图像生成完全替代”。
- **“传统 CV”这个总题是否过宽**：是的，而且这个宽度本身就是当前页面的风险。随着 `目标检测` 与 `OCR` 已经外溢成独立 topic，这一风险实际上已经开始被缓解；但文档 AI、表格理解、视觉基础模型等子线仍未完全拆稳，因此本页仍需要继续收缩为更强的总览页，而不是再次回到细节堆积。

### 统一接口不等于统一约束

越来越多方法共享 Transformer、对比学习或生成骨架，但具体任务仍有几何、拓扑、坐标和时序要求。本文较稳定的判断是骨架共享扩大了复用空间；不同任务评价并未消失。若声称新模型替代专门方法，应在相同输入、输出精度、资源预算和数据泄漏条件下对照，而不是只展示几张自然的结果图。

## 证据基础

- [Dosovitskiy et al. - 2020 - An Image is Worth 16x16 Words Transformers for Image Recognition at Scale](../../wiki/summaries/Dosovitskiy%20et%20al.%20-%202020%20-%20An%20Image%20is%20Worth%2016x16%20Words%20Transformers%20for%20Image%20Recognition%20at%20Scale.md)
- [He et al. - 2015 - Deep Residual Learning for Image Recognition](../../wiki/summaries/He%20et%20al.%20-%202015%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition.md)
- [Liu et al. - 2022 - A ConvNet for the 2020s](../../wiki/summaries/Liu%20et%20al.%20-%202022%20-%20A%20ConvNet%20for%20the%202020s.md)
- [Huang et al. - 2022 - LayoutLMv3 Pre-training for Document AI with Unified Text and Image Masking](../../wiki/summaries/Huang%20et%20al.%20-%202022%20-%20LayoutLMv3%20Pre-training%20for%20Document%20AI%20with%20Unified%20Text%20and%20Image%20Masking.md)
- [Li et al. - 2021 - TrOCR Transformer-based Optical Character Recognition with Pre-trained Models](../../wiki/summaries/Li%20et%20al.%20-%202021%20-%20TrOCR%20Transformer-based%20Optical%20Character%20Recognition%20with%20Pre-trained%20Models.md)
- [Gabeur et al. - 2026 - Image Generators are Generalist Vision Learners](../../wiki/summaries/Gabeur%20et%20al.%20-%202026%20-%20Image%20Generators%20are%20Generalist%20Vision%20Learners.md)
- [Chen, He - 2021 - Exploring Simple Siamese Representation Learning](../summaries/Chen%2C%20He%20-%202021%20-%20Exploring%20Simple%20Siamese%20Representation%20Learning.md)：无负样本自监督表示的机制与条件。
- [Khan et al. - 2021 - Transformers in Vision A Survey](../summaries/Khan%20et%20al.%20-%202021%20-%20Transformers%20in%20Vision%20A%20Survey.md)：视觉 Transformer 路线组织。
- [Li, Fan, Ai - Unknown - Scaling Language-Image Pre-training via Masking](../summaries/Li%2C%20Fan%2C%20Ai%20-%20Unknown%20-%20Scaling%20Language-Image%20Pre-training%20via%20Masking.md)：图文预训练遮蔽与效率折中。
- [Lin et al. - 2021 - Real-Time High-Resolution Background Matting](../summaries/Lin%20et%20al.%20-%202021%20-%20Real-Time%20High-Resolution%20Background%20Matting.md)：有额外背景输入的实时抠像。
- [Lin et al. - 2022 - Robust High-Resolution Video Matting with Temporal Guidance](../summaries/Lin%20et%20al.%20-%202022%20-%20Robust%20High-Resolution%20Video%20Matting%20with%20Temporal%20Guidance.md)：视频时序状态与高分辨率抠像。
- [Tewari et al. - 2020 - State of the Art on Neural Rendering](../summaries/Tewari%20et%20al.%20-%202020%20-%20State%20of%20the%20Art%20on%20Neural%20Rendering.md)：神经渲染与判别式视觉的任务边界。
- [Tewari et al. - 2021 - Advances in neural rendering](../summaries/Tewari%20et%20al.%20-%202021%20-%20Advances%20in%20neural%20rendering.md)：新视点与场景表示的进展。

## 代表页面

- [经典 CNN 架构](./经典%20CNN%20架构.md)
- [ResNet](../concepts/ResNet.md)
- [ResNeXt](../concepts/ResNeXt.md)
- [MobileNet](../concepts/MobileNet.md)
- [ConvNeXt](../concepts/ConvNeXt.md)
- [ViT](../concepts/ViT.md)
- [Transformer](../concepts/Transformer.md)
- [CLIP](../concepts/CLIP.md)
- [Vision Banana](../concepts/Vision%20Banana.md)
- [Faster R-CNN](../concepts/Faster%20R-CNN.md)
- [DETR](../concepts/DETR.md)
- [LayoutLMv3](../concepts/LayoutLMv3.md)
- [DocLayNet](../concepts/DocLayNet.md)
- [PubTables-1M](../concepts/PubTables-1M.md)
- [TrOCR](../concepts/TrOCR.md)
- [DocLLM](../concepts/DocLLM.md)
- [OCR](./OCR.md)
- [目标检测](目标检测.md)

## 未解决问题

- 共享生成骨架能否保持几何、尺度和拓扑约束？自然画面与准确深度、mask、字符是不同标准。
- 怎样分离自监督、数据规模和模型结构的贡献？视觉与生成式感知报告常共同改变多个条件。
- 动态场景怎样兼顾时序稳定、遮挡和细节？抠像与渲染有不同输入前提，单帧质量不足以判断跨帧可用性。

## 关联页面

- [Slide 理解与生成](./Slide%20理解与生成.md)
- [OCR](./OCR.md)
- [目标检测](目标检测.md)
- [经典 CNN 架构](./经典%20CNN%20架构.md)
- [ResNet](../concepts/ResNet.md)
- [ResNeXt](../concepts/ResNeXt.md)
- [MobileNet](../concepts/MobileNet.md)
- [ConvNeXt](../concepts/ConvNeXt.md)
- [ViT](../concepts/ViT.md)
- [Transformer](../concepts/Transformer.md)
- [CLIP](../concepts/CLIP.md)
- [Vision Banana](../concepts/Vision%20Banana.md)
- [LayoutLMv3](../concepts/LayoutLMv3.md)
- [DocLayNet](../concepts/DocLayNet.md)
- [PubTables-1M](../concepts/PubTables-1M.md)
- [Faster R-CNN](../concepts/Faster%20R-CNN.md)
- [DETR](../concepts/DETR.md)
- [TrOCR](../concepts/TrOCR.md)
- [Kosmos-2](../concepts/Kosmos-2.md)
- [Kosmos-2.5](../concepts/Kosmos-2.5.md)
- [MiniCPM-V](../concepts/MiniCPM-V.md)
- [OFA](../concepts/OFA.md)
- [data2vec](../concepts/data2vec.md)
- [HuBERT](../concepts/HuBERT.md)
- [Tip-Adapter](../concepts/Tip-Adapter.md)
- [文档与表格的输入输出接口](../comparisons/%E6%96%87%E6%A1%A3%E4%B8%8E%E8%A1%A8%E6%A0%BC%E7%9A%84%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA%E6%8E%A5%E5%8F%A3.md)：页面转写、表格结构恢复、单表问答和电子表格压缩需要不同表示。先决定保留哪些行列、坐标、样式与运算，再选模型；结构合法和答案正确要分别验收。
- [神经渲染](../concepts/%E7%A5%9E%E7%BB%8F%E6%B8%B2%E6%9F%93.md)：神经渲染把学习表示与相机、几何或渲染过程连接起来，目标常是生成条件视图。它与物体识别或普通文生图的约束不同，先看场景输入和能控制哪些变量。

## 这里的术语是什么意思

- **token**：词元：模型处理文本的基本单位，可能是一个字、一个词或其片段。
- **backbone**：模型骨干：主要负责提取或变换表示，其他任务模块在它之上工作。
- **OCR**：文字识别：从图像读取文字；整页任务还需处理布局和阅读顺序。

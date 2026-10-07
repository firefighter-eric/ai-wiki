# Corporation - 2022 - NVIDIA DGX A100 The Universal System for AI Infrastructure

- Source PDF: `raw/pdf/Corporation - 2022 - NVIDIA DGX A100 The Universal System for AI Infrastructure.pdf`
- Source SHA256: `2294b524beb7d696e22056d88e2444e4f43b17d3e72ae86befedcf5cce0e02f0`
- Generated from: `scripts/extract_pdf_text.py`

- Extraction: `pymupdf-pages-v2` (sorted text, page anchors; tables/formulas/figures require review)

## Extracted Text

<a id="source-section-0"></a>

<a id="page-1"></a>

### PDF 第 1 页

DATASHEET




NVIDIA DGX A100
The Universal System for AI Infrastructure



The Challenge of Scaling Enterprise AI                        SYSTEM SPECIFICATIONS
                                                                                          NVIDIA DGX A100 640GB
Every business needs to transform using artificial intelligence (AI), not only
                                                                        GPUs          8x NVIDIA A100 80GB Tensor Core GPUs
to survive, but to thrive in challenging times. However, the enterprise requires
                                                                   GPU Memory   640GB total
a platform for AI infrastructure that improves upon traditional approaches,
                                                                                  Performance   5 petaFLOPS AIwhich historically involved slow compute architectures that were siloed by
                                                                                                  10 petaOPS INT8
analytics, training, and inference workloads. The old approach created
                                                                                   NVIDIA        6complexity, drove up costs, constrained speed of scale, and was not ready
                                                                                   NVSwitches
for modern AI. Enterprises, developers, data scientists, and researchers
                                                                                 System Power   6.5 kW max
need a new platform that unifies all AI workloads, simplifying infrastructure        Usage
and accelerating ROI.                                                                    CPU           Dual AMD Rome 7742, 128 cores total,
                                                                                                           2.25 GHz (base), 3.4 GHz (max boost)
The Universal System for Every AI Workload                         System       2TB
                                                                        Memory
NVIDIA DGX™ A100 is the universal system for all AI workloads—from analytics
                                                                                    Networking    Up to 8x Single-     Up to 8x Single-
to training to inference. DGX A100 sets a new bar for compute density, packing                                                                                                       Port NVIDIA         Port NVIDIA
5 petaFLOPS of AI performance into a 6U form factor, replacing legacy                       ConnectX-7         ConnectX-6 VPI
compute infrastructure with a single, unified system. DGX A100 also offers                     200 Gb/s InfiniBand  200 Gb/s InfiniBand
the unprecedented ability to deliver a fine-grained allocation of computing                   Up to 2x Dual-Port   Up to 2x Dual-Port
power, using the Multi-Instance GPU (MIG) capability in the NVIDIA A100                      NVIDIA             NVIDIA
                                                                                                   ConnectX-7 VPI      ConnectX-6 VPITensor Core GPU. This enables administrators to assign resources that are
                                                                                                      10/25/50/100/200    10/25/50/100/200right-sized for specific workloads.                                                                                                    Gb/s Ethernet       Gb/s Ethernet
Available with up to 640 gigabytes (GB) of total GPU memory, which increases         Storage        OS: 2x 1.92TB M.2 NVME drives
performance in large-scale training jobs up to 3X and doubles the size of MIG                          Internal Storage: 30TB (8x 3.84 TB) U.2
instances, DGX A100 can tackle the largest and most complex jobs, along with                  NVMe drives
the simplest and smallest. This combination of dense compute power and         Software       Ubuntu / Red Hat Enterprise Linux /
complete workload flexibility makes DGX A100 an ideal choice for both single                     CentOS – Operating system
and scaled DGX deployments. Every DGX system is powered by NVIDIA                       NVIDIA Base Command – Orchestration,
                                                                                                            scheduling, and cluster management
Base Command™ for managing single node as well as large-scale Slurm or                                                                                                NVIDIA AI Enterprise – Optimized AI
Kubernetes clusters, and includes the NVIDIA AI Enterprise suite of optimized                        software
AI software containers.                                                              Support       Comes with 3-year business-standard
                                                                                             hardware and software support
Unmatched Level of Support and Expertise                           System Weight  271.5 lbs (123.16 kgs) max
NVIDIA DGX A100 is more than a server. It’s a complete hardware and          Packaged       359.7 lbs (163.16 kgs) max
                                                                                 System Weight
software platform built upon the knowledge gained from the world’s largest
                                                                                 System         Height: 10.4 in (264.0 mm)
DGX proving ground—NVIDIA DGX SATURNV—and backed by thousands of
                                                                                    Dimensions     Width: 19.0 in (482.3 mm) max
DGXperts at NVIDIA. DGXperts are AI-fluent practitioners who have built a
                                                                                                         Length: 35.3 in (897.1 mm) max
wealth of know-how and experience over the last decade to help maximize
                                                                                          Operating      5ºC to 30ºC (41ºF to 86ºF)
the value of a DGX investment. DGXperts help ensure that critical applications                                                                                    Temperature
get up and running quickly, and stay running smoothly, for dramatically-        Range
improved time to insights.
<a id="page-2"></a>

### PDF 第 2 页

Fastest Time to Solution
                                                          Up to 3X Higher Throughput for AI Training on Largest Models
NVIDIA DGX A100 features eight NVIDIA A100 Tensor Core GPUs,          DGX A100 640GB                                                                                                                                            FP16                                                             3X
which deliver unmatched acceleration, and is fully optimized for           DGX A100 320GB                                                                                                                                            FP16                    1X
NVIDIA CUDA-X™ software and the end-to-end NVIDIA data center                    DGX-2                                                                                                                                            FP16    .07X
solution stack. NVIDIA A100 GPUs bring Tensor Float 32 (TF32)                                 0                        1X                       2X                       3X
precision, the default precision format for both TensorFlow and                                        DLRM Training
                                                                                                                                                 Time Per 1,000 Iterations - Relative Performance
PyTorch AI frameworks. This works just like FP32 but provides 20X
                                                                                              DLRM on HugeCTR framework, precision = FP16 | 1x DGX A100 640GB batch size = 48 | 2x DGX A100 320GB batch size =
higher floating operations per second (FLOPS) for AI compared to              32 | 1x DGX-2 (16x V100 32GB) batch size = 32. Speedups normalized to number of GPUs.
the previous generation. Best of all, no code changes are required
to achieve this speedup.
                                                          Up to 1.25X Higher Throughput for AI Inference
The A100 80GB GPU increases GPU memory bandwidth 30 percent
over the A100 40GB GPU, making it the world’s first with 2 terabytes          DGX A100 640GB                                                       1.25X
per second (TB/s). It also has significantly more on-chip memory          DGX A100 320GB                                         1X
than the previous-generation NVIDIA GPU, including a 40 megabyte                                0                           0.5X                       1X                         1.5X
                                                                                                                     RNN-T Inference: Single Stream
(MB) level 2 cache that’s nearly 7X larger, maximizing compute                                             Sequences Per Second - Relative Performance
performance. DGX A100 also debuts the third generation of NVIDIA®             MLPerf 0.7 RNN-T measured with (1/7) MIG slices. Framework: TensorRT 7.2, dataset = LibriSpeech, precision = FP16.
NVLink®, which doubles the GPU-to-GPU direct bandwidth to 600
gigabytes per second (GB/s), almost 10X higher than PCIe Gen 4, and
a new NVIDIA NVSwitch™ that’s 2X faster than the last generation.      Up to 83X Higher Throughput than CPU, 2X Higher Throughput
                                                                        than DGX A100 320GB on Big Data Analytics Benchmark
This unprecedented power delivers the fastest time to solution,
allowing users to tackle challenges that weren't possible or            DGX A100 640GB                                                        83X
practical before.                                                                  DGX A100 320GB                             44X           Up to 2X
                                                                                                          DGX-1      11X
The World’s Most Secure AI System for Enterprise               CPU Only   1X                                           Up to 83X
                                                                                                                                             0       10       20       30      40       50       60       70       80       90
NVIDIA DGX A100 delivers robust security posture for the AI                                        Time to Solution - Relative Performance
enterprise, with a multi-layered approach that secures all major                 BigGolddata6252analytics2.10 GHz,benchmarkHadoop | 16x| 30DGX-1analytical(8x V100retail32GBqueries,each),ETL,RAPIDS/DaskML, NLP on| 12x10TBDGXdatasetA100 |320GBCPU: 19xandIntel6x DGXXeonA100
hardware and software components. Stretching across the                640GB, RAPIDS/Dask/BlazingSQL. Speedups normalized to number of GPUs.
baseboard management controller (BMC), CPU board, GPU board,
and self-encrypted drives, DGX A100 has security built in, allowing    OPEX on the data center infrastructure. The combination of
IT to focus on operationalizing AI rather than spending time on     massive GPU-accelerated compute with state-of-the-art
threat assessment and mitigation.                                networking hardware and software optimizations means DGX
                                                         A100 can scale to hundreds or thousands of nodes to meet the
Unparalleled Data Center Scalability with                 biggest challenges, such as conversational AI and large-scale
NVIDIA Networking                                      image classification.
With the fastest I/O architecture of any DGX system, NVIDIA DGX     Proven Infrastructure Solutions Built with Trusted
A100 is the foundational building block for large AI clusters like                                                Data Center Leaders
NVIDIA DGX SuperPOD™, the enterprise blueprint for scalable AI
infrastructure. DGX A100 features up to eight single-port NVIDIA®      In combination with leading storage and networking technology
ConnectX®-6 or ConnectX-7 adapters for clustering and up to two      providers, a portfolio of infrastructure solutions is available
dual-port ConnectX-6 or ConnectX-7 adapters for storage and        that incorporates the best of the NVIDIA DGX POD™ reference
networking, all capable of 200Gb/s. With ConnectX-7 connectivity      architecture. Delivered as fully integrated, ready-to-deploy
to the NVIDIA Quantum-2 InfiniBand switches, DGX SuperPOD      offerings through our NVIDIA Partner Network (NPN), these
can be built with fewer switches and cables, saving CAPEX and       solutions simplify and accelerate data center AI deployments.




Ready to Get Started?

To learn more about NVIDIA DGX A100, visit www.nvidia.com/dgxa100




© 2022 NVIDIA Corporation. All rights reserved. NVIDIA, the NVIDIA logo, DGX , NGC, CUDA-X, NVLink, NVSwitch, SuperPOD, ConnectX, and DGX POD,
are trademarks and/or registered trademarks of NVIDIA Corporation. All company and product names are trademarks or registered trademarks of the
respective owners with which they are associated. Features, pricing, availability, and specifications are all subject to change without notice. 2372157 Aug22

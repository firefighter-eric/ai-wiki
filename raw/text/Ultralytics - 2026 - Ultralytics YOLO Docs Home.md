# Ultralytics - 2026 - Ultralytics YOLO Docs Home

- Source HTML: `raw/html/Ultralytics - 2026 - Ultralytics YOLO Docs Home.html`
- Source SHA256: `2b39c951d01ae99546f4f88099e0b96427e18cd97b2a735f4ba6f922c1905a48`
- Source URL: https://docs.ultralytics.com/
- Generated from: `scripts/fetch_web_text.py`
- Extraction: `structured-html-v2` (headings, links, MathML/TeX and tables; figures require visual review)

## Extracted Text

<a id="source-section-0"></a>

[[图片：Ultralytics YOLO banner]](https://platform.ultralytics.com/ultralytics/yolo26?utm_source=docs&utm_medium=referral&utm_campaign=platform_launch&utm_content=banner&utm_term=ultralytics_docs)

[中文](https://docs.ultralytics.com/zh/) ·
[한국어](https://docs.ultralytics.com/ko/) ·
[日本語](https://docs.ultralytics.com/ja/) ·
[Русский](https://docs.ultralytics.com/ru/) ·
[Deutsch](https://docs.ultralytics.com/de/) ·
[Français](https://docs.ultralytics.com/fr/) ·
[Español](https://docs.ultralytics.com/es/) ·
[Português](https://docs.ultralytics.com/pt/) ·
[Türkçe](https://docs.ultralytics.com/tr/) ·
[Tiếng Việt](https://docs.ultralytics.com/vi/) ·
[العربية](https://docs.ultralytics.com/ar/)

[[图片：Ultralytics CI]](https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml)[[图片：Ultralytics Downloads]](https://clickpy.clickhouse.com/dashboard/ultralytics)[[图片：Ultralytics YOLO Citation]](https://zenodo.org/badge/latestdoi/264818686)[[图片：Ultralytics Discord]](https://discord.com/invite/ultralytics)[[图片：Ultralytics Forums]](https://community.ultralytics.com/)[[图片：Ultralytics Reddit]](https://www.reddit.com/r/ultralytics/)
[[图片：Run Ultralytics on Gradient]](https://console.paperspace.com/github/ultralytics/ultralytics)[[图片：Open Ultralytics In Colab]](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb)[[图片：Open Ultralytics In Kaggle]](https://www.kaggle.com/models/ultralytics/yolo26)[[图片：Open Ultralytics In Binder]](https://mybinder.org/v2/gh/ultralytics/ultralytics/HEAD?labpath=examples%2Ftutorial.ipynb)


<a id="source-section-1"></a>

# Home


Introducing Ultralytics [YOLO26](https://docs.ultralytics.com/models/yolo26/), the latest version of the acclaimed real-time object detection and image segmentation model. YOLO26 is built on [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) and [computer vision](https://www.ultralytics.com/blog/everything-you-need-to-know-about-computer-vision-in-2025) advancements, featuring end-to-end NMS-free inference and optimized edge deployment. Its streamlined design makes it suitable for various applications and easily adaptable to different hardware platforms, from edge devices to cloud APIs. For stable production workloads, both YOLO26 and [YOLO11](https://docs.ultralytics.com/models/yolo11/) are recommended.


Explore the Ultralytics Docs, a comprehensive resource designed to help you understand and utilize its features and capabilities. Whether you are a seasoned [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) practitioner or new to the field, this hub aims to maximize YOLO's potential in your projects.


[[图片：Ultralytics GitHub]](https://github.com/ultralytics) [图片：无替代文本] [[图片：Ultralytics LinkedIn]](https://www.linkedin.com/company/ultralytics/) [图片：无替代文本] [[图片：Ultralytics Twitter]](https://twitter.com/ultralytics) [图片：无替代文本] [[图片：Ultralytics YouTube]](https://www.youtube.com/ultralytics?sub_confirmation=1) [图片：无替代文本] [[图片：Ultralytics TikTok]](https://www.tiktok.com/%40ultralytics) [图片：无替代文本] [[图片：Ultralytics BiliBili]](https://ultralytics.com/bilibili) [图片：无替代文本] [[图片：Ultralytics Discord]](https://discord.com/invite/ultralytics)


<a id="source-section-2"></a>

## Where to Start


- **Getting Started**


Install `ultralytics` with pip and get up and running in minutes to train a YOLO model


[Quickstart](https://docs.ultralytics.com/quickstart/)

- **Predict**


Predict on new images, videos and streams with YOLO


[Learn more](https://docs.ultralytics.com/modes/predict/)

- **Train a Model**


Train a new YOLO model on your own custom dataset from scratch or load and train on a pretrained model


[Learn more](https://docs.ultralytics.com/modes/train/)

- **Explore Computer Vision Tasks**


Discover YOLO tasks like detect, segment, classify, pose, OBB and track


[Explore Tasks](https://docs.ultralytics.com/tasks/)

- [图片：🚀] **Explore YOLO26 🚀 NEW**


Discover Ultralytics' latest YOLO26 models with NMS-free inference and edge optimization


[YOLO26 Models 🚀](https://docs.ultralytics.com/models/yolo26/)

- **SAM 3: Segment Anything with Concepts 🚀 NEW**


Meta's latest SAM 3 with Promptable Concept Segmentation - segment all instances using text or image exemplars


[SAM 3 Models](https://docs.ultralytics.com/models/sam-3/)

- **Open Source, AGPL-3.0**


Ultralytics offers two YOLO licenses: AGPL-3.0 and Enterprise. Explore YOLO on [GitHub](https://github.com/ultralytics/ultralytics).


[YOLO License](https://www.ultralytics.com/license)


**Watch:** How to Train a YOLO26 model on Your Custom Dataset in [Google Colab](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb).


<a id="source-section-3"></a>

## YOLO: A Brief History


[YOLO](https://docs.ultralytics.com/models/) (You Only Look Once), a popular [object detection](https://www.ultralytics.com/glossary/object-detection) and [image segmentation](https://www.ultralytics.com/glossary/image-segmentation) model, was developed by Joseph Redmon and Ali Farhadi at the University of Washington. Launched in 2015, YOLO gained popularity for its high speed and accuracy.


- [YOLOv2](https://docs.ultralytics.com/models/), released in 2016, improved the original model by incorporating batch normalization, anchor boxes, and dimension clusters.

- [YOLOv3](https://docs.ultralytics.com/models/yolov3/), launched in 2018, further enhanced the model's performance using a more efficient backbone network, multiple anchors, and spatial pyramid pooling.

- [YOLOv4](https://docs.ultralytics.com/models/yolov4/) was released in 2020, introducing innovations like Mosaic [data augmentation](https://www.ultralytics.com/glossary/data-augmentation), a new anchor-free detection head, and a new [loss function](https://www.ultralytics.com/glossary/loss-function).

- [YOLOv5](https://docs.ultralytics.com/models/yolov5/) further improved the model's performance and added new features such as hyperparameter optimization, integrated experiment tracking, and automatic export to popular export formats.

- [YOLOv6](https://docs.ultralytics.com/models/yolov6/) was open-sourced by [Meituan](https://www.meituan.com/) in 2022 and is used in many of the company's autonomous delivery robots.

- [YOLOv7](https://docs.ultralytics.com/models/yolov7/) added additional tasks such as pose estimation on the COCO keypoints dataset.

- [YOLOv8](https://docs.ultralytics.com/models/yolov8/) released in 2023 by Ultralytics, introduced new features and improvements for enhanced performance, flexibility, and efficiency, supporting a full range of vision AI tasks.

- [YOLOv9](https://docs.ultralytics.com/models/yolov9/) introduces innovative methods like Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN).

- [YOLOv10](https://docs.ultralytics.com/models/yolov10/) created by researchers from [Tsinghua University](https://www.tsinghua.edu.cn/en/) using the [Ultralytics](https://www.ultralytics.com/)[Python package](https://pypi.org/project/ultralytics/), provides real-time [object detection](https://docs.ultralytics.com/tasks/detect/) advancements by introducing an End-to-End head that eliminates Non-Maximum Suppression (NMS) requirements.

- **[YOLO11](https://docs.ultralytics.com/models/yolo11/)**: Released in September 2024, YOLO11 delivers excellent performance across multiple tasks, including [object detection](https://docs.ultralytics.com/tasks/detect/), [segmentation](https://docs.ultralytics.com/tasks/segment/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), [tracking](https://docs.ultralytics.com/modes/track/), and [classification](https://docs.ultralytics.com/tasks/classify/), enabling deployment across diverse AI applications and domains.

- **[YOLO26](https://docs.ultralytics.com/models/yolo26/) 🚀**: Ultralytics' next-generation YOLO model optimized for edge deployment with end-to-end NMS-free inference.


<a id="source-section-4"></a>

## YOLO Licenses: How is Ultralytics YOLO licensed?


Ultralytics offers two licensing options to accommodate diverse use cases:


- **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/license/agpl-v3) open-source license is ideal for students and enthusiasts, promoting open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for more details.

- **Enterprise License**: Designed for commercial use, this license permits seamless integration of Ultralytics software and AI models into commercial goods and services, bypassing the open-source requirements of AGPL-3.0. If your scenario involves embedding our solutions into a commercial offering, reach out through [Ultralytics Licensing](https://www.ultralytics.com/license).


Our licensing strategy is designed to ensure that any improvements to our open-source projects are returned to the community. We believe in open source, and our mission is to ensure that our contributions can be used and expanded in ways that benefit everyone.


<a id="source-section-5"></a>

## The Evolution of Object Detection


Object detection has evolved significantly over the years, from traditional computer vision techniques to advanced deep learning models. The [YOLO family of models](https://www.ultralytics.com/blog/the-evolution-of-object-detection-and-ultralytics-yolo-models) has been at the forefront of this evolution, consistently pushing the boundaries of what's possible in real-time object detection.


YOLO's unique approach treats object detection as a single regression problem, predicting [bounding boxes](https://www.ultralytics.com/glossary/bounding-box) and class probabilities directly from full images in one evaluation. This revolutionary method has made YOLO models significantly faster than previous two-stage detectors while maintaining high accuracy.


With each new version, YOLO has introduced architectural improvements and innovative techniques that have enhanced performance across various metrics. YOLO26 continues this tradition by incorporating the latest advancements in computer vision research, featuring end-to-end NMS-free inference and optimized edge deployment for real-world applications.


<a id="source-section-6"></a>

## FAQ


<a id="source-section-7"></a>

### What is Ultralytics YOLO and how does it improve object detection?


Ultralytics YOLO is the acclaimed YOLO (You Only Look Once) series for real-time object detection and image segmentation. The latest model, [YOLO26](https://docs.ultralytics.com/models/yolo26/), builds on previous versions by introducing end-to-end NMS-free inference and optimized edge deployment. YOLO supports various [vision AI tasks](https://docs.ultralytics.com/tasks/) such as detection, segmentation, pose estimation, tracking, and classification. Its efficient architecture ensures excellent speed and accuracy, making it suitable for diverse applications, including edge devices and cloud APIs.


<a id="source-section-8"></a>

### How can I get started with YOLO installation and setup?


Getting started with YOLO is quick and straightforward. You can install the Ultralytics package using [pip](https://pypi.org/project/ultralytics/) and get up and running in minutes. Here's a basic installation command:


Installation using pip


CLI


```
pip install -U ultralytics
```


For a comprehensive step-by-step guide, visit our [Quickstart](https://docs.ultralytics.com/quickstart/) page. This resource will help you with installation instructions, initial setup, and running your first model.


<a id="source-section-9"></a>

### How can I train a custom YOLO model on my dataset?


Training a custom YOLO model on your dataset involves a few detailed steps:


- Prepare your annotated dataset.

- Configure the training parameters in a YAML file.

- Use the `yolo TASK train` command to start training. (Each `TASK` has its own argument)


Here's example code for the Object Detection Task:


Train Example for Object Detection Task


PythonCLI


```
from ultralytics import YOLO

# Load a pretrained YOLO model (you can choose n, s, m, l, or x versions)
model = YOLO("yolo26n.pt")

# Start training on your custom dataset
model.train(data="path/to/dataset.yaml", epochs=100, imgsz=640)
```


```
# Train a YOLO model from the command line
yolo detect train data=path/to/dataset.yaml epochs=100 imgsz=640
```


For a detailed walkthrough, check out our [Train a Model](https://docs.ultralytics.com/modes/train/) guide, which includes examples and tips for optimizing your training process.


<a id="source-section-10"></a>

### What are the licensing options available for Ultralytics YOLO?


Ultralytics offers two licensing options for YOLO:


- **AGPL-3.0 License**: This open-source license is ideal for educational and non-commercial use, promoting open collaboration.

- **Enterprise License**: This is designed for commercial applications, allowing seamless integration of Ultralytics software into commercial products without the restrictions of the AGPL-3.0 license.


For more details, visit our [Licensing](https://www.ultralytics.com/license) page.


<a id="source-section-11"></a>

### How can Ultralytics YOLO be used for real-time object tracking?


Ultralytics YOLO supports efficient and customizable multi-object tracking. To utilize tracking capabilities, you can use the `yolo track` command, as shown below:


Example for Object Tracking on a Video


PythonCLI


```
from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo26n.pt")

# Start tracking objects in a video
# You can also use live video streams or webcam input
model.track(source="path/to/video.mp4")
```


```
# Perform object tracking on a video from the command line
# You can specify different sources like webcam (0) or RTSP streams
yolo track source=path/to/video.mp4
```


For a detailed guide on setting up and running object tracking, check our [Track Mode](https://docs.ultralytics.com/modes/track/) documentation, which explains the configuration and practical applications in real-time scenarios.


📅 Created 2 years ago ✏️ Updated 25 days ago


[[图片：glenn-jocher]](https://github.com/glenn-jocher)[[图片：RizwanMunawar]](https://github.com/RizwanMunawar)[[图片：pderrenger]](https://github.com/pderrenger)[[图片：jk4e]](https://github.com/jk4e)[[图片：UltralyticsAssistant]](https://github.com/UltralyticsAssistant)[[图片：AyushExel]](https://github.com/AyushExel)[[图片：Laughing-q]](https://github.com/Laughing-q)[[图片：Y-T-G]](https://github.com/Y-T-G)[[图片：picsalex]](https://github.com/picsalex)[[图片：LexBarou]](https://github.com/LexBarou)[[图片：RizwanMunawar]](https://github.com/RizwanMunawar)


<a id="source-section-12"></a>

## Comments

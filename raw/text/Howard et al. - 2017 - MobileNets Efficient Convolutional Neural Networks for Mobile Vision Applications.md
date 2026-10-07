# Howard et al. - 2017 - MobileNets Efficient Convolutional Neural Networks for Mobile Vision Applications

- Source HTML: `raw/html/Howard et al. - 2017 - MobileNets Efficient Convolutional Neural Networks for Mobile Vision Applications.html`
- Source SHA256: `f0f10cb585e715743eed0b1f812bb4904e805b6aeb7a30f048aefcef5fafcf82`
- Source URL: https://ar5iv.labs.arxiv.org/html/1704.04861
- Generated from: `scripts/fetch_web_text.py`
- Extraction: `structured-html-v2` (headings, links, MathML/TeX and tables; figures require visual review)

## Extracted Text

<a id="source-section-0"></a>

<a id="source-section-1"></a>

# MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications


Andrew G. Howard
Menglong Zhu
Bo Chen
Dmitry Kalenichenko

Weijun Wang
Tobias Weyand
Marco Andreetto
Hartwig Adam


Google Inc.

{howarda,menglong,bochen,dkalenichenko,weijunw,weyand,anm,hadam}@google.com


<a id="source-section-2"></a>

###### Abstract


We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depthwise separable convolutions to build light weight deep neural networks. We introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. These hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints of the problem. We present extensive experiments on resource and accuracy tradeoffs and show strong performance compared to other popular models on ImageNet classification. We then demonstrate the effectiveness of MobileNets across a wide range of applications and use cases including object detection, finegrain classification, face attributes and large scale geo-localization.


<a id="source-section-3"></a>

## 1 Introduction


Convolutional neural networks have become ubiquitous in computer vision ever since AlexNet [[19](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib19)] popularized deep convolutional neural networks by winning the ImageNet Challenge: ILSVRC 2012 [[24](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib24)]. The general trend has been to make deeper and more complicated networks in order to achieve higher accuracy [[27](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib27), [31](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib31), [29](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib29), [8](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib8)]. However, these advances to improve accuracy are not necessarily making networks more efficient with respect to size and speed. In many real world applications such as robotics, self-driving car and augmented reality, the recognition tasks need to be carried out in a timely fashion on a computationally limited platform.


This paper describes an efficient network architecture and a set of two hyper-parameters in order to build very small, low latency models that can be easily matched to the design requirements for mobile and embedded vision applications. Section [2](https://ar5iv.labs.arxiv.org/html/1704.04861#S2) reviews prior work in building small models. Section [3](https://ar5iv.labs.arxiv.org/html/1704.04861#S3) describes the MobileNet architecture and two hyper-parameters width multiplier and resolution multiplier to define smaller and more efficient MobileNets. Section [4](https://ar5iv.labs.arxiv.org/html/1704.04861#S4) describes experiments on ImageNet as well a variety of different applications and use cases. Section [5](https://ar5iv.labs.arxiv.org/html/1704.04861#S5) closes with a summary and conclusion.


[图片：Refer to caption]


Figure 1: MobileNet models can be applied to various recognition tasks for efficient on device intelligence.


<a id="source-section-4"></a>

## 2 Prior Work


There has been rising interest in building small and efficient neural networks in the recent literature, e.g. [[16](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib16), [34](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib34), [12](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib12), [36](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib36), [22](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib22)]. Many different approaches can be generally categorized into either compressing pretrained networks or training small networks directly. This paper proposes a class of network architectures that allows a model developer to specifically choose a small network that matches the resource restrictions (latency, size) for their application. MobileNets primarily focus on optimizing for latency but also yield small networks. Many papers on small networks focus only on size but do not consider speed.


MobileNets are built primarily from depthwise separable convolutions initially introduced in [[26](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib26)] and subsequently used in Inception models [[13](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib13)] to reduce the computation in the first few layers. Flattened networks [[16](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib16)] build a network out of fully factorized convolutions and showed the potential of extremely factorized networks. Independent of this current paper, Factorized Networks[[34](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib34)] introduces a similar factorized convolution as well as the use of topological connections. Subsequently, the Xception network [[3](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib3)] demonstrated how to scale up depthwise separable filters to out perform Inception V3 networks. Another small network is Squeezenet [[12](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib12)] which uses a bottleneck approach to design a very small network. Other reduced computation networks include structured transform networks [[28](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib28)] and deep fried convnets [[37](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib37)].


A different approach for obtaining small networks is shrinking, factorizing or compressing pretrained networks. Compression based on product quantization [[36](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib36)], hashing [[2](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib2)], and pruning, vector quantization and Huffman coding [[5](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib5)] have been proposed in the literature. Additionally various factorizations have been proposed to speed up pretrained networks [[14](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib14), [20](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib20)]. Another method for training small networks is distillation [[9](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib9)] which uses a larger network to teach a smaller network. It is complementary to our approach and is covered in some of our use cases in section [4](https://ar5iv.labs.arxiv.org/html/1704.04861#S4). Another emerging approach is low bit networks [[4](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib4), [22](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib22), [11](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib11)].


<a id="source-section-5"></a>

## 3 MobileNet Architecture


In this section we first describe the core layers that MobileNet is built on which are depthwise separable filters. We then describe the MobileNet network structure and conclude with descriptions of the two model shrinking hyper-parameters width multiplier and resolution multiplier.


<a id="source-section-6"></a>

### 3.1 Depthwise Separable Convolution


The MobileNet model is based on depthwise separable convolutions which is a form of factorized convolutions which factorize a standard convolution into a depthwise convolution and a $1\times 1$ convolution called a pointwise convolution. For MobileNets the depthwise convolution applies a single filter to each input channel. The pointwise convolution then applies a $1\times 1$ convolution to combine the outputs the depthwise convolution. A standard convolution both filters and combines inputs into a new set of outputs in one step. The depthwise separable convolution splits this into two layers, a separate layer for filtering and a separate layer for combining. This factorization has the effect of drastically reducing computation and model size. Figure [2](https://ar5iv.labs.arxiv.org/html/1704.04861#S3.F2) shows how a standard convolution [2(a)](https://ar5iv.labs.arxiv.org/html/1704.04861#S3.F2.sf1) is factorized into a depthwise convolution [2(b)](https://ar5iv.labs.arxiv.org/html/1704.04861#S3.F2.sf2) and a $1\times 1$ pointwise convolution [2(c)](https://ar5iv.labs.arxiv.org/html/1704.04861#S3.F2.sf3).


A standard convolutional layer takes as input a $D_{F}\times D_{F}\times M$ feature map $\mathbf{F}$ and produces a $D_{F}\times D_{F}\times N$ feature map $\mathbf{G}$ where $D_{F}$ is the spatial width and height of a square input feature map111We assume that the output feature map has the same spatial dimensions as the input and both feature maps are square. Our model shrinking results generalize to feature maps with arbitrary sizes and aspect ratios., $M$ is the number of input channels (input depth), $D_{G}$ is the spatial width and height of a square output feature map and $N$ is the number of output channel (output depth).


The standard convolutional layer is parameterized by convolution kernel $\mathbf{K}$ of size $D_{K}\times D_{K}\times M\times N$ where $D_{K}$ is the spatial dimension of the kernel assumed to be square and $M$ is number of input channels and $N$ is the number of output channels as defined previously.


The output feature map for standard convolution assuming stride one and padding is computed as:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>\mathbf{G}_{k,l,n}=\sum_{i,j,m}\mathbf{K}_{i,j,m,n}\cdot\mathbf{F}_{k+i-1,l+j-1,m}<br>$$ | | (1) |


Standard convolutions have the computational cost of:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>D_{K}\cdot D_{K}\cdot M\cdot N\cdot D_{F}\cdot D_{F}<br>$$ | | (2) |


where the computational cost depends multiplicatively on the number of input channels $M$, the number of output channels $N$ the kernel size $D_{k}\times D_{k}$ and the feature map size $D_{F}\times D_{F}$. MobileNet models address each of these terms and their interactions. First it uses depthwise separable convolutions to break the interaction between the number of output channels and the size of the kernel.


The standard convolution operation has the effect of filtering features based on the
convolutional kernels and combining features in order to produce a new representation.
The filtering and combination steps can be split into two steps via the use of
factorized convolutions called depthwise separable convolutions for substantial reduction in computational cost.


Depthwise separable convolution are made up of two layers: depthwise convolutions and pointwise convolutions.
We use depthwise convolutions to apply a single filter per each input channel (input depth). Pointwise convolution, a
simple $1\times 1$ convolution, is then used to create a linear combination of the output of the depthwise layer. MobileNets use both
batchnorm and ReLU nonlinearities for both layers.


Depthwise convolution with one filter per input channel (input depth) can be written as:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>\hat{\mathbf{G}}_{k,l,m}=\sum_{i,j}\hat{\mathbf{K}}_{i,j,m}\cdot\mathbf{F}_{k+i-1,l+j-1,m}<br>$$ | | (3) |


where $\hat{\mathbf{K}}$ is the depthwise convolutional kernel of size $D_{K}\times D_{K}\times M$ where the $m_{th}$ filter in $\hat{\mathbf{K}}$ is applied to the $m_{th}$ channel in $\mathbf{F}$ to produce the $m_{th}$ channel of the filtered output feature map $\hat{\mathbf{G}}$.


Depthwise convolution has a computational cost of:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>D_{K}\cdot D_{K}\cdot M\cdot D_{F}\cdot D_{F}<br>$$ | | (4) |


Depthwise convolution is extremely efficient relative to standard convolution. However it only filters input channels, it does not combine them to create new features. So an additional layer that computes a linear combination of the output of depthwise convolution via $1\times 1$ convolution is needed in order to generate these new features.


The combination of depthwise convolution and $1\times 1$ (pointwise) convolution is called depthwise separable convolution which was originally introduced in [[26](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib26)].


Depthwise separable convolutions cost:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>D_{K}\cdot D_{K}\cdot M\cdot D_{F}\cdot D_{F}+M\cdot N\cdot D_{F}\cdot D_{F}<br>$$ | | (5) |


which is the sum of the depthwise and $1\times 1$ pointwise convolutions.


By expressing convolution as a two step process of filtering and combining we get a reduction in computation of:


| 列 1 | 列 2 | 列 3 | 列 4 | 列 5 |
| --- | --- | --- | --- | --- |
| | | | $\displaystyle\frac{D_{K}\cdot D_{K}\cdot M\cdot D_{F}\cdot D_{F}+M\cdot N\cdot D_{F}\cdot D_{F}}{D_{K}\cdot D_{K}\cdot M\cdot N\cdot D_{F}\cdot D_{F}}$ | |
| | | $\displaystyle=$ | $\displaystyle\frac{1}{N}+\frac{1}{D_{K}^{2}}$ | |


MobileNet uses $3\times 3$ depthwise separable convolutions which uses between 8 to 9 times less computation than standard convolutions at only a small reduction in accuracy as seen in Section [4](https://ar5iv.labs.arxiv.org/html/1704.04861#S4).


[图片：Refer to caption]


(a) Standard Convolution Filters


[图片：Refer to caption]


(b) Depthwise Convolutional Filters


[图片：Refer to caption]


(c) $1\times 1$ Convolutional Filters called Pointwise Convolution in the context of Depthwise Separable Convolution


Figure 2: The standard convolutional filters in (a) are replaced by two layers: depthwise convolution in (b) and pointwise convolution in (c) to build a depthwise separable filter.


Additional factorization in spatial dimension such as in [[16](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib16), [31](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib31)] does not save much additional computation as very little computation is spent in depthwise convolutions.


<a id="source-section-7"></a>

### 3.2 Network Structure and Training


The MobileNet structure is built on depthwise separable convolutions as mentioned in the previous section except for the first layer which is a full convolution. By defining the network in such simple terms we are able to easily explore network topologies to find a good network. The MobileNet architecture is defined in Table [1](https://ar5iv.labs.arxiv.org/html/1704.04861#S3.T1). All layers are followed by a batchnorm [[13](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib13)] and ReLU nonlinearity with the exception of the final fully connected layer which has no nonlinearity and feeds into a softmax layer for classification. Figure [3](https://ar5iv.labs.arxiv.org/html/1704.04861#S3.F3) contrasts a layer with regular convolutions, batchnorm and ReLU nonlinearity to the factorized layer with depthwise convolution, $1\times 1$ pointwise convolution as well as batchnorm and ReLU after each convolutional layer. Down sampling is handled with strided convolution in the depthwise convolutions as well as in the first layer. A final average pooling reduces the spatial resolution to 1 before the fully connected layer. Counting depthwise and pointwise convolutions as separate layers, MobileNet has 28 layers.


It is not enough to simply define networks in terms of a small number of Mult-Adds. It is also important to make sure these operations can be efficiently implementable. For instance unstructured sparse matrix operations are not typically faster than dense matrix operations until a very high level of sparsity. Our model structure puts nearly all of the computation into dense $1\times 1$ convolutions. This can be implemented with highly optimized general matrix multiply (GEMM) functions. Often convolutions are implemented by a GEMM but require an initial reordering in memory called im2col in order to map it to a GEMM. For instance, this approach is used in the popular Caffe package [[15](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib15)]. $1\times 1$ convolutions do not require this reordering in memory and can be implemented directly with GEMM which is one of the most optimized numerical linear algebra algorithms. MobileNet spends $95\%$ of it’s computation time in $1\times 1$ convolutions which also has $75\%$ of the parameters as can be seen in Table [2](https://ar5iv.labs.arxiv.org/html/1704.04861#S3.T2). Nearly all of the additional parameters are in the fully connected layer.


MobileNet models were trained in TensorFlow [[1](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib1)] using RMSprop [[33](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib33)] with asynchronous gradient descent similar to Inception V3 [[31](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib31)]. However, contrary to training large models we use less regularization and data augmentation techniques because small models have less trouble with overfitting. When training MobileNets we do not use side heads or label smoothing and additionally reduce the amount image of distortions by limiting the size of small crops that are used in large Inception training [[31](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib31)]. Additionally, we found that it was important to put very little or no weight decay (l2 regularization) on the depthwise filters since their are so few parameters in them. For the ImageNet benchmarks in the next section all models were trained with same training parameters regardless of the size of the model.


Table 1: MobileNet Body Architecture


| 列 1 | 列 2 | 列 3 |
| --- | --- | --- |
| Type / Stride | Filter Shape | Input Size |
| Conv / s2 | $3\times 3\times 3\times 32$ | $224\times 224\times 3$ |
| Conv dw / s1 | $3\times 3\times 32$ dw | $112\times 112\times 32$ |
| Conv / s1 | $1\times 1\times 32\times 64$ | $112\times 112\times 32$ |
| Conv dw / s2 | $3\times 3\times 64$ dw | $112\times 112\times 64$ |
| Conv / s1 | $1\times 1\times 64\times 128$ | $56\times 56\times 64$ |
| Conv dw / s1 | $3\times 3\times 128$ dw | $56\times 56\times 128$ |
| Conv / s1 | $1\times 1\times 128\times 128$ | $56\times 56\times 128$ |
| Conv dw / s2 | $3\times 3\times 128$ dw | $56\times 56\times 128$ |
| Conv / s1 | $1\times 1\times 128\times 256$ | $28\times 28\times 128$ |
| Conv dw / s1 | $3\times 3\times 256$ dw | $28\times 28\times 256$ |
| Conv / s1 | $1\times 1\times 256\times 256$ | $28\times 28\times 256$ |
| Conv dw / s2 | $3\times 3\times 256$ dw | $28\times 28\times 256$ |
| Conv / s1 | $1\times 1\times 256\times 512$ | $14\times 14\times 256$ |
| $5\times$ Conv dw / s1 | $3\times 3\times 512$ dw | $14\times 14\times 512$ |
| Conv / s1 | $1\times 1\times 512\times 512$ | $14\times 14\times 512$ |
| Conv dw / s2 | $3\times 3\times 512$ dw | $14\times 14\times 512$ |
| Conv / s1 | $1\times 1\times 512\times 1024$ | $7\times 7\times 512$ |
| Conv dw / s2 | $3\times 3\times 1024$ dw | $7\times 7\times 1024$ |
| Conv / s1 | $1\times 1\times 1024\times 1024$ | $7\times 7\times 1024$ |
| Avg Pool / s1 | Pool $7\times 7$ | $7\times 7\times 1024$ |
| FC / s1 | $1024\times 1000$ | $1\times 1\times 1024$ |
| Softmax / s1 | Classifier | $1\times 1\times 1000$ |


[图片：Refer to caption]


Figure 3: Left: Standard convolutional layer with batchnorm and ReLU. Right: Depthwise Separable convolutions with Depthwise and Pointwise layers followed by batchnorm and ReLU.


Table 2: Resource Per Layer Type


| Type | Mult-Adds | Parameters |
| --- | --- | --- |
| Conv $1\times 1$ | 94.86% | 74.59% |
| Conv DW $3\times 3$ | 3.06% | 1.06% |
| Conv $3\times 3$ | 1.19% | 0.02% |
| Fully Connected | 0.18% | 24.33% |


<a id="source-section-8"></a>

### 3.3 Width Multiplier: Thinner Models


Although the base MobileNet architecture is already small and low latency, many times a specific use case or application may require the model to be smaller and faster. In order to construct these smaller and less computationally expensive models we introduce a very simple parameter $\alpha$ called width multiplier. The role of the width multiplier $\alpha$ is to thin a network uniformly at each layer. For a given layer and width multiplier $\alpha$, the number of input channels $M$ becomes $\alpha M$ and the number of output channels $N$ becomes $\alpha N$.


The computational cost of a depthwise separable convolution with width multiplier $\alpha$ is:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>D_{K}\cdot D_{K}\cdot\alpha M\cdot D_{F}\cdot D_{F}+\alpha M\cdot\alpha N\cdot D_{F}\cdot D_{F}<br>$$ | | (6) |


where $\alpha\in(0,1]$ with typical settings of 1, 0.75, 0.5 and 0.25. $\alpha=1$ is the baseline MobileNet and $\alpha<1$
are reduced MobileNets. Width multiplier has the effect of reducing computational cost and the number of parameters quadratically by roughly $\alpha^{2}$. Width multiplier can be applied to any model structure to define a new smaller model with a reasonable accuracy, latency and size trade off. It is used to define a new reduced structure that needs to be trained from scratch.


<a id="source-section-9"></a>

### 3.4 Resolution Multiplier: Reduced Representation


The second hyper-parameter to reduce the computational cost of a neural network is a resolution multiplier $\rho$. We apply this to the input image and the internal representation of every layer is subsequently reduced by the same multiplier. In practice we implicitly set $\rho$ by setting the input resolution.


We can now express the computational cost for the core layers of our network as depthwise separable convolutions with width multiplier $\alpha$ and resolution multiplier $\rho$:


| 列 1 | 列 2 | 列 3 | 列 4 |
| --- | --- | --- | --- |
| | $$<br>D_{K}\cdot D_{K}\cdot\alpha M\cdot\rho D_{F}\cdot\rho D_{F}+\alpha M\cdot\alpha N\cdot\rho D_{F}\cdot\rho D_{F}<br>$$ | | (7) |


where $\rho\in(0,1]$ which is typically set implicitly so that the input resolution of the network is 224, 192, 160 or 128. $\rho=1$ is the baseline MobileNet and $\rho<1$
are reduced computation MobileNets. Resolution multiplier has the effect of reducing computational cost by $\rho^{2}$.


As an example we can look at a typical layer in MobileNet and see how depthwise separable convolutions, width multiplier and resolution multiplier reduce the cost and parameters. Table [3](https://ar5iv.labs.arxiv.org/html/1704.04861#S3.T3) shows the computation and number of parameters for a layer as architecture shrinking methods are sequentially applied to the layer. The first row shows the Mult-Adds and parameters for a full convolutional layer with an input feature map of size $14\times 14\times 512$ with a kernel $K$ of size $3\times 3\times 512\times 512$. We will look in detail in the next section at the trade offs between resources and accuracy.


Table 3: Resource usage for modifications to standard convolution. Note that each row is a cumulative effect adding on top of the previous row. This example is for an internal MobileNet layer with $D_{K}=3$, $M=512$, $N=512$, $D_{F}=14$.


| Layer/Modification | Million | Million |
| --- | --- | --- |
| | Mult-Adds | Parameters |
| Convolution | 462 | 2.36 |
| Depthwise Separable Conv | 52.3 | 0.27 |
| $\alpha=0.75$ | 29.6 | 0.15 |
| $\rho=0.714$ | 15.1 | 0.15 |


<a id="source-section-10"></a>

## 4 Experiments


In this section we first investigate the effects of depthwise convolutions as well as the choice of shrinking by reducing the width of the network rather than the number of layers. We then show the trade offs of reducing the network based on the two hyper-parameters: width multiplier and resolution multiplier and compare results to a number of popular models. We then investigate MobileNets applied to a number of different applications.


<a id="source-section-11"></a>

### 4.1 Model Choices


First we show results for MobileNet with depthwise separable convolutions compared to a model built with full convolutions. In Table [4](https://ar5iv.labs.arxiv.org/html/1704.04861#S4.T4) we see that using depthwise separable convolutions compared to full convolutions only reduces accuracy by $1\%$ on ImageNet was saving tremendously on mult-adds and parameters.


Table 4: Depthwise Separable vs Full Convolution MobileNet


| Model | ImageNet | Million | Million |
| --- | --- | --- | --- |
| | Accuracy | Mult-Adds | Parameters |
| Conv MobileNet | 71.7% | 4866 | 29.3 |
| MobileNet | 70.6% | 569 | 4.2 |


We next show results comparing thinner models with width multiplier to shallower models using less layers. To make MobileNet shallower, the $5$ layers of separable filters with feature size $14\times 14\times 512$ in Table [1](https://ar5iv.labs.arxiv.org/html/1704.04861#S3.T1) are removed. Table [5](https://ar5iv.labs.arxiv.org/html/1704.04861#S4.T5) shows that at similar computation and number of parameters, that making MobileNets thinner is $3\%$ better than making them shallower.


Table 5: Narrow vs Shallow MobileNet


| Model | ImageNet | Million | Million |
| --- | --- | --- | --- |
| | Accuracy | Mult-Adds | Parameters |
| 0.75 MobileNet | 68.4% | 325 | 2.6 |
| Shallow MobileNet | 65.3% | 307 | 2.9 |


<a id="source-section-12"></a>

### 4.2 Model Shrinking Hyperparameters


Table [6](https://ar5iv.labs.arxiv.org/html/1704.04861#S4.T6) shows the accuracy, computation and size trade offs of shrinking the MobileNet architecture with the width multiplier $\alpha$.
Accuracy drops off smoothly until the architecture is made too small at $\alpha=0.25$.


Table 6: MobileNet Width Multiplier


| Width Multiplier | ImageNet | Million | Million |
| --- | --- | --- | --- |
| | Accuracy | Mult-Adds | Parameters |
| 1.0 MobileNet-224 | 70.6% | 569 | 4.2 |
| 0.75 MobileNet-224 | 68.4% | 325 | 2.6 |
| 0.5 MobileNet-224 | 63.7% | 149 | 1.3 |
| 0.25 MobileNet-224 | 50.6% | 41 | 0.5 |


Table [7](https://ar5iv.labs.arxiv.org/html/1704.04861#S4.T7) shows the accuracy, computation and size trade offs for different resolution multipliers by
training MobileNets with reduced input resolutions. Accuracy drops off smoothly across resolution.


Table 7: MobileNet Resolution


| Resolution | ImageNet | Million | Million |
| --- | --- | --- | --- |
| | Accuracy | Mult-Adds | Parameters |
| 1.0 MobileNet-224 | 70.6% | 569 | 4.2 |
| 1.0 MobileNet-192 | 69.1% | 418 | 4.2 |
| 1.0 MobileNet-160 | 67.2% | 290 | 4.2 |
| 1.0 MobileNet-128 | 64.4% | 186 | 4.2 |


[图片：Refer to caption]


Figure 4: This figure shows the trade off between computation (Mult-Adds) and accuracy on the ImageNet benchmark. Note the log linear dependence between accuracy and computation.


[图片：Refer to caption]


Figure 5: This figure shows the trade off between the number of parameters and accuracy on the ImageNet benchmark. The colors encode input resolutions. The number of parameters do not vary based on the input resolution.


Figure [4](https://ar5iv.labs.arxiv.org/html/1704.04861#S4.F4) shows the trade off between ImageNet Accuracy and computation for the 16 models made from the cross product of width multiplier $\alpha\in\{1,0.75,0.5,0.25\}$ and resolutions $\{224,192,160,128\}$. Results are log linear with a jump when models get very small at $\alpha=0.25$.


Figure [5](https://ar5iv.labs.arxiv.org/html/1704.04861#S4.F5) shows the trade off between ImageNet Accuracy and number of parameters for the 16 models made from the cross product of width multiplier $\alpha\in\{1,0.75,0.5,0.25\}$ and resolutions $\{224,192,160,128\}$.


Table [8](https://ar5iv.labs.arxiv.org/html/1704.04861#S4.T8) compares full MobileNet to the original GoogleNet [[30](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib30)] and VGG16 [[27](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib27)]. MobileNet is nearly as accurate as VGG16 while being 32 times smaller and 27 times less compute intensive. It is more accurate than GoogleNet while being smaller and more than 2.5 times less computation.


Table [9](https://ar5iv.labs.arxiv.org/html/1704.04861#S4.T9) compares a reduced MobileNet with width multiplier $\alpha=0.5$ and reduced resolution $160\times 160$. Reduced MobileNet is $4\%$ better than AlexNet [[19](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib19)] while being $45\times$ smaller and $9.4\times$ less compute than AlexNet. It is also $4\%$ better than Squeezenet [[12](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib12)] at about the same size and $22\times$ less computation.


Table 8: MobileNet Comparison to Popular Models


| Model | ImageNet | Million | Million |
| --- | --- | --- | --- |
| | Accuracy | Mult-Adds | Parameters |
| 1.0 MobileNet-224 | 70.6% | 569 | 4.2 |
| GoogleNet | 69.8% | 1550 | 6.8 |
| VGG 16 | 71.5% | 15300 | 138 |


Table 9: Smaller MobileNet Comparison to Popular Models


| Model | ImageNet | Million | Million |
| --- | --- | --- | --- |
| | Accuracy | Mult-Adds | Parameters |
| 0.50 MobileNet-160 | 60.2% | 76 | 1.32 |
| Squeezenet | 57.5% | 1700 | 1.25 |
| AlexNet | 57.2% | 720 | 60 |


<a id="source-section-13"></a>

### 4.3 Fine Grained Recognition


We train MobileNet for fine grained recognition on the Stanford Dogs dataset [[17](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib17)].
We extend the approach of [[18](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib18)] and collect an even larger but noisy training set than [[18](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib18)] from the web.
We use the noisy web data to pretrain a fine grained dog recognition model and then fine tune the model on the Stanford Dogs training set.
Results on Stanford Dogs test set are in Table [10](https://ar5iv.labs.arxiv.org/html/1704.04861#S4.T10). MobileNet can almost achieve the state of the art results from
[[18](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib18)] at greatly reduced computation and size.


Table 10: MobileNet for Stanford Dogs


| Model | Top-1 | Million | Million |
| --- | --- | --- | --- |
| | Accuracy | Mult-Adds | Parameters |
| Inception V3 [[18](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib18)] | 84% | 5000 | 23.2 |
| 1.0 MobileNet-224 | 83.3% | 569 | 3.3 |
| 0.75 MobileNet-224 | 81.9% | 325 | 1.9 |
| 1.0 MobileNet-192 | 81.9% | 418 | 3.3 |
| 0.75 MobileNet-192 | 80.5% | 239 | 1.9 |


<a id="source-section-14"></a>

### 4.4 Large Scale Geolocalizaton


Table 11: Performance of PlaNet using the MobileNet architecture. Percentages are the fraction of the Im2GPS test dataset that were localized within a certain distance from the ground truth. The numbers for the original PlaNet model are based on an updated version that has an improved architecture and training dataset.


| Scale | Im2GPS [[7](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib7)] | PlaNet [[35](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib35)] | PlaNet |
| --- | --- | --- | --- |
| | | | MobileNet |
| Continent (2500 km) | 51.9% | 77.6% | 79.3% |
| Country (750 km) | 35.4% | 64.0% | 60.3% |
| Region (200 km) | 32.1% | 51.1% | 45.2% |
| City (25 km) | 21.9% | 31.7% | 31.7% |
| Street (1 km) | 2.5% | 11.0% | 11.4% |


PlaNet [[35](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib35)] casts the task of determining where on earth a photo was taken as a classification problem. The approach divides the earth into a grid of geographic cells that serve as the target classes and trains a convolutional neural network on millions of geo-tagged photos. PlaNet has been shown to successfully localize a large variety of photos and to outperform Im2GPS [[6](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib6), [7](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib7)] that addresses the same task.


We re-train PlaNet using the MobileNet architecture on the same data. While the full PlaNet model based on the Inception V3 architecture [[31](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib31)] has 52 million parameters and 5.74 billion mult-adds. The MobileNet model has only 13 million parameters with the usual 3 million for the body and 10 million for the final layer and 0.58 Million mult-adds. As shown in Tab. [11](https://ar5iv.labs.arxiv.org/html/1704.04861#S4.T11), the MobileNet version delivers only slightly decreased performance compared to PlaNet despite being much more compact. Moreover, it still outperforms Im2GPS by a large margin.


<a id="source-section-15"></a>

### 4.5 Face Attributes


Another use-case for MobileNet is compressing large systems with unknown or esoteric training procedures. In a face attribute classification task, we demonstrate a synergistic relationship between MobileNet and distillation [[9](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib9)], a knowledge transfer technique for deep networks. We seek to reduce a large face attribute classifier with $75$ million parameters and $1600$ million Mult-Adds. The classifier is trained on a multi-attribute dataset similar to YFCC100M [[32](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib32)].


We distill a face attribute classifier using the MobileNet architecture. Distillation [[9](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib9)] works by training the classifier to emulate the outputs of a larger model222The emulation quality is measured by averaging the per-attribute cross-entropy over all attributes. instead of the ground-truth labels, hence enabling training from large (and potentially infinite) unlabeled datasets. Marrying the scalability of distillation training and the parsimonious parameterization of MobileNet, the end system not only requires no regularization (e.g. weight-decay and early-stopping), but also demonstrates enhanced performances. It is evident from Tab. [12](https://ar5iv.labs.arxiv.org/html/1704.04861#S4.T12) that the MobileNet-based classifier is resilient to aggressive model shrinking: it achieves a similar mean average precision across attributes (mean AP) as the in-house while consuming only $1\%$ the Multi-Adds.


Table 12: Face attribute classification using the MobileNet architecture. Each row corresponds to a different hyper-parameter setting (width multiplier $\alpha$ and image resolution).


| Width Multiplier / | Mean | Million | Million |
| --- | --- | --- | --- |
| Resolution | AP | Mult-Adds | Parameters |
| 1.0 MobileNet-224 | 88.7% | 568 | 3.2 |
| 0.5 MobileNet-224 | 88.1% | 149 | 0.8 |
| 0.25 MobileNet-224 | 87.2% | 45 | 0.2 |
| 1.0 MobileNet-128 | 88.1% | 185 | 3.2 |
| 0.5 MobileNet-128 | 87.7% | 48 | 0.8 |
| 0.25 MobileNet-128 | 86.4% | 15 | 0.2 |
| Baseline | 86.9% | 1600 | 7.5 |


<a id="source-section-16"></a>

### 4.6 Object Detection


MobileNet can also be deployed as an effective base network in modern object detection systems.
We report results for MobileNet trained for object detection on COCO data based on the recent work that won the 2016 COCO challenge [[10](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib10)].
In table [13](https://ar5iv.labs.arxiv.org/html/1704.04861#S4.T13), MobileNet is compared to VGG and Inception V2 [[13](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib13)] under both Faster-RCNN [[23](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib23)] and SSD [[21](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib21)] framework.
In our experiments, SSD is evaluated with 300 input resolution (SSD 300) and Faster-RCNN is compared with both 300 and 600 input resolution (Faster-RCNN 300, Faster-RCNN 600).
The Faster-RCNN model evaluates 300 RPN proposal boxes per image. The models are trained on COCO train+val excluding 8k minival images and evaluated on minival.
For both frameworks, MobileNet achieves comparable results to other networks with only a fraction of computational complexity and model size.


Table 13: COCO object detection results comparison using different frameworks and network architectures. mAP is reported with COCO primary challenge metric (AP at IoU=0.50:0.05:0.95)


| Framework | Model | mAP | Billion | Million |
| --- | --- | --- | --- | --- |
| Resolution | | | Mult-Adds | Parameters |
| | deeplab-VGG | 21.1% | 34.9 | 33.1 |
| SSD 300 | Inception V2 | 22.0% | 3.8 | 13.7 |
| | MobileNet | 19.3% | 1.2 | 6.8 |
| Faster-RCNN | VGG | 22.9% | 64.3 | 138.5 |
| 300 | Inception V2 | 15.4% | 118.2 | 13.3 |
| | MobileNet | 16.4% | 25.2 | 6.1 |
| Faster-RCNN | VGG | 25.7% | 149.6 | 138.5 |
| 600 | Inception V2 | 21.9% | 129.6 | 13.3 |
| | Mobilenet | 19.8% | 30.5 | 6.1 |


[图片：Refer to caption]


Figure 6: Example objection detection results using MobileNet SSD.


<a id="source-section-17"></a>

### 4.7 Face Embeddings


The FaceNet model is a state of the art face recognition model [[25](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib25)]. It builds face embeddings based on the triplet loss. To build a mobile FaceNet model we use distillation
to train by minimizing the squared differences of the output of FaceNet and MobileNet on the training data. Results for very small MobileNet models can be found in table [14](https://ar5iv.labs.arxiv.org/html/1704.04861#S4.T14).


Table 14: MobileNet Distilled from FaceNet


| Model | 1e-4 | Million | Million |
| --- | --- | --- | --- |
| | Accuracy | Mult-Adds | Parameters |
| FaceNet [[25](https://ar5iv.labs.arxiv.org/html/1704.04861#bib.bib25)] | 83% | 1600 | 7.5 |
| 1.0 MobileNet-160 | 79.4% | 286 | 4.9 |
| 1.0 MobileNet-128 | 78.3% | 185 | 5.5 |
| 0.75 MobileNet-128 | 75.2% | 166 | 3.4 |
| 0.75 MobileNet-128 | 72.5% | 108 | 3.8 |


<a id="source-section-18"></a>

## 5 Conclusion


We proposed a new model architecture called MobileNets based on depthwise separable convolutions. We investigated some of the important design decisions leading to an efficient model. We then demonstrated how to build smaller and faster MobileNets using width multiplier and resolution multiplier by trading off a reasonable amount of accuracy to reduce size and latency. We then compared different MobileNets to popular models demonstrating superior size, speed and accuracy characteristics. We concluded by demonstrating MobileNet’s effectiveness when applied to a wide variety of tasks. As a next step to help adoption and exploration of MobileNets, we plan on releasing models in Tensor Flow.


<a id="source-section-19"></a>

## References


- [1]

M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S. Corrado,
A. Davis, J. Dean, M. Devin, et al.


Tensorflow: Large-scale machine learning on heterogeneous systems,
2015.


Software available from tensorflow. org, 1, 2015.


- [2]

W. Chen, J. T. Wilson, S. Tyree, K. Q. Weinberger, and Y. Chen.


Compressing neural networks with the hashing trick.


CoRR, abs/1504.04788, 2015.


- [3]

F. Chollet.


Xception: Deep learning with depthwise separable convolutions.


arXiv preprint arXiv:1610.02357v2, 2016.


- [4]

M. Courbariaux, J.-P. David, and Y. Bengio.


Training deep neural networks with low precision multiplications.


arXiv preprint arXiv:1412.7024, 2014.


- [5]

S. Han, H. Mao, and W. J. Dally.


Deep compression: Compressing deep neural network with pruning,
trained quantization and huffman coding.


CoRR, abs/1510.00149, 2, 2015.


- [6]

J. Hays and A. Efros.


IM2GPS: estimating geographic information from a single image.


In Proceedings of the IEEE International Conference on Computer
Vision and Pattern Recognition, 2008.


- [7]

J. Hays and A. Efros.


Large-Scale Image Geolocalization.


In J. Choi and G. Friedland, editors, Multimodal Location
Estimation of Videos and Images. Springer, 2014.


- [8]

K. He, X. Zhang, S. Ren, and J. Sun.


Deep residual learning for image recognition.


arXiv preprint arXiv:1512.03385, 2015.


- [9]

G. Hinton, O. Vinyals, and J. Dean.


Distilling the knowledge in a neural network.


arXiv preprint arXiv:1503.02531, 2015.


- [10]

J. Huang, V. Rathod, C. Sun, M. Zhu, A. Korattikara, A. Fathi, I. Fischer,
Z. Wojna, Y. Song, S. Guadarrama, et al.


Speed/accuracy trade-offs for modern convolutional object detectors.


arXiv preprint arXiv:1611.10012, 2016.


- [11]

I. Hubara, M. Courbariaux, D. Soudry, R. El-Yaniv, and Y. Bengio.


Quantized neural networks: Training neural networks with low
precision weights and activations.


arXiv preprint arXiv:1609.07061, 2016.


- [12]

F. N. Iandola, M. W. Moskewicz, K. Ashraf, S. Han, W. J. Dally, and K. Keutzer.


Squeezenet: Alexnet-level accuracy with 50x fewer parameters and¡ 1mb
model size.


arXiv preprint arXiv:1602.07360, 2016.


- [13]

S. Ioffe and C. Szegedy.


Batch normalization: Accelerating deep network training by reducing
internal covariate shift.


arXiv preprint arXiv:1502.03167, 2015.


- [14]

M. Jaderberg, A. Vedaldi, and A. Zisserman.


Speeding up convolutional neural networks with low rank expansions.


arXiv preprint arXiv:1405.3866, 2014.


- [15]

Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick,
S. Guadarrama, and T. Darrell.


Caffe: Convolutional architecture for fast feature embedding.


arXiv preprint arXiv:1408.5093, 2014.


- [16]

J. Jin, A. Dundar, and E. Culurciello.


Flattened convolutional neural networks for feedforward acceleration.


arXiv preprint arXiv:1412.5474, 2014.


- [17]

A. Khosla, N. Jayadevaprakash, B. Yao, and L. Fei-Fei.


Novel dataset for fine-grained image categorization.


In First Workshop on Fine-Grained Visual Categorization, IEEE
Conference on Computer Vision and Pattern Recognition, Colorado Springs, CO,
June 2011.


- [18]

J. Krause, B. Sapp, A. Howard, H. Zhou, A. Toshev, T. Duerig, J. Philbin, and
L. Fei-Fei.


The unreasonable effectiveness of noisy data for fine-grained
recognition.


arXiv preprint arXiv:1511.06789, 2015.


- [19]

A. Krizhevsky, I. Sutskever, and G. E. Hinton.


Imagenet classification with deep convolutional neural networks.


In Advances in neural information processing systems, pages
1097–1105, 2012.


- [20]

V. Lebedev, Y. Ganin, M. Rakhuba, I. Oseledets, and V. Lempitsky.


Speeding-up convolutional neural networks using fine-tuned
cp-decomposition.


arXiv preprint arXiv:1412.6553, 2014.


- [21]

W. Liu, D. Anguelov, D. Erhan, C. Szegedy, and S. Reed.


Ssd: Single shot multibox detector.


arXiv preprint arXiv:1512.02325, 2015.


- [22]

M. Rastegari, V. Ordonez, J. Redmon, and A. Farhadi.


Xnor-net: Imagenet classification using binary convolutional neural
networks.


arXiv preprint arXiv:1603.05279, 2016.


- [23]

S. Ren, K. He, R. Girshick, and J. Sun.


Faster r-cnn: Towards real-time object detection with region proposal
networks.


In Advances in neural information processing systems, pages
91–99, 2015.


- [24]

O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang,
A. Karpathy, A. Khosla, M. Bernstein, et al.


Imagenet large scale visual recognition challenge.


International Journal of Computer Vision, 115(3):211–252,
2015.


- [25]

F. Schroff, D. Kalenichenko, and J. Philbin.


Facenet: A unified embedding for face recognition and clustering.


In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition, pages 815–823, 2015.


- [26]

L. Sifre.


Rigid-motion scattering for image classification.


PhD thesis, Ph. D. thesis, 2014.


- [27]

K. Simonyan and A. Zisserman.


Very deep convolutional networks for large-scale image recognition.


arXiv preprint arXiv:1409.1556, 2014.


- [28]

V. Sindhwani, T. Sainath, and S. Kumar.


Structured transforms for small-footprint deep learning.


In Advances in Neural Information Processing Systems, pages
3088–3096, 2015.


- [29]

C. Szegedy, S. Ioffe, and V. Vanhoucke.


Inception-v4, inception-resnet and the impact of residual connections
on learning.


arXiv preprint arXiv:1602.07261, 2016.


- [30]

C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan,
V. Vanhoucke, and A. Rabinovich.


Going deeper with convolutions.


In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition, pages 1–9, 2015.


- [31]

C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna.


Rethinking the inception architecture for computer vision.


arXiv preprint arXiv:1512.00567, 2015.


- [32]

B. Thomee, D. A. Shamma, G. Friedland, B. Elizalde, K. Ni, D. Poland, D. Borth,
and L.-J. Li.


Yfcc100m: The new data in multimedia research.


Communications of the ACM, 59(2):64–73, 2016.


- [33]

T. Tieleman and G. Hinton.


Lecture 6.5-rmsprop: Divide the gradient by a running average of its
recent magnitude.


COURSERA: Neural Networks for Machine Learning, 4(2), 2012.


- [34]

M. Wang, B. Liu, and H. Foroosh.


Factorized convolutional neural networks.


arXiv preprint arXiv:1608.04337, 2016.


- [35]

T. Weyand, I. Kostrikov, and J. Philbin.


PlaNet - Photo Geolocation with Convolutional Neural Networks.


In European Conference on Computer Vision (ECCV), 2016.


- [36]

J. Wu, C. Leng, Y. Wang, Q. Hu, and J. Cheng.


Quantized convolutional neural networks for mobile devices.


arXiv preprint arXiv:1512.06473, 2015.


- [37]

Z. Yang, M. Moczulski, M. Denil, N. de Freitas, A. Smola, L. Song, and Z. Wang.


Deep fried convnets.


In Proceedings of the IEEE International Conference on Computer
Vision, pages 1476–1483, 2015.

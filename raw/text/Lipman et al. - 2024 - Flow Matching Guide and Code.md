# Lipman et al. - 2024 - Flow Matching Guide and Code

- Source PDF: `raw/pdf/Lipman et al. - 2024 - Flow Matching Guide and Code.pdf`
- Generated from: `scripts/extract_pdf_text.py`

## Extracted Text

Flow Matching Guide and Code
Yaron Lipman1, Marton Havasi1, Peter Holderrieth2, Neta Shaul3, Matt Le1, Brian Karrer1,
Ricky T. Q. Chen1, David Lopez-Paz1, Heli Ben-Hamu3, Itai Gat1
1FAIR at Meta, 2MIT CSAIL, 3Weizmann Institute of Science
Flow Matching (FM) is a recent framework for generative modeling that has achieved state-of-the-art
performance across various domains, including image, video, audio, speech, and biological structures.
This guide offers a comprehensive and self-contained review of FM, covering its mathematical foun-
dations, design choices, and extensions. By also providing a PyTorch package featuring relevant
examples (e.g., image and text generation), this work aims to serve as a resource for both novice and
experienced researchers interested in understanding, applying and further developing FM.
Date: December 10, 2024
Code: flow_matching library at https://github.com/facebookresearch/flow_matching
Contents
1
Introduction
3
2
Quick tour and key concepts
4
3
Flow models
8
3.1
Random vectors . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
8
3.2
Conditional densities and expectations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
8
3.3
Diffeomorphisms and push-forward maps . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
9
3.4
Flows as generative models
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
10
3.5
Probability paths and the Continuity Equation . . . . . . . . . . . . . . . . . . . . . . . . . .
13
3.6
Instantaneous Change of Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
14
3.7
Training flow models with simulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
15
4
Flow Matching
16
4.1
Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
16
4.2
Building probability paths . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
16
4.3
Deriving generating velocity fields
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
17
4.4
General conditioning and the Marginalization Trick . . . . . . . . . . . . . . . . . . . . . . . .
18
4.5
Flow Matching loss . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
19
4.6
Solving conditional generation with conditional flows . . . . . . . . . . . . . . . . . . . . . . .
21
4.7
Optimal Transport and linear conditional flow . . . . . . . . . . . . . . . . . . . . . . . . . . .
25
4.8
Affine conditional flows . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
26
4.9
Data couplings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
31
4.10 Conditional generation and guidance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
33
5
Non-Euclidean Flow Matching
35
5.1
Riemannian manifolds . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
35
5.2
Probabilities, flows and velocities on manifolds
. . . . . . . . . . . . . . . . . . . . . . . . . .
35
5.3
Probability paths on manifolds . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
36
5.4
The Marginalization Trick for manifolds . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
36
5.5
Riemannian Flow Matching loss . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
37
5.6
Conditional flows through premetrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
37
1
arXiv:2412.06264v1  [cs.LG]  9 Dec 2024

6
Continuous Time Markov Chain Models
40
6.1
Discrete state spaces and random variables
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
40
6.2
The CTMC generative model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
40
6.3
Probability paths and Kolmogorov Equation . . . . . . . . . . . . . . . . . . . . . . . . . . . .
41
7
Discrete Flow Matching
42
7.1
Data and coupling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
42
7.2
Discrete probability paths . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
42
7.3
The Marginalization Trick . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
42
7.4
Discrete Flow Matching loss . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
43
7.5
Factorized paths and velocities
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
44
8
Continuous Time Markov Process Models
52
8.1
General state spaces and random variables . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
52
8.2
The CTMP generative model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
52
8.3
Probability paths and Kolmogorov Equation . . . . . . . . . . . . . . . . . . . . . . . . . . . .
57
8.4
Universal representation theorem . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
61
9
Generator Matching
62
9.1
Data and coupling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
62
9.2
General probability paths . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
62
9.3
Parameterizing a generator via a neural network
. . . . . . . . . . . . . . . . . . . . . . . . .
62
9.4
Marginal and conditional generators
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
64
9.5
Generator Matching loss . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
65
9.6
Finding conditional generators as solutions to the KFE . . . . . . . . . . . . . . . . . . . . . .
66
9.7
Combining Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
68
9.8
Multimodal models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
70
10 Relation to Diffusion and other Denoising Models
71
10.1 Time convention
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
71
10.2 Forward process vs. probability paths
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
71
10.3 Training a diffusion model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
72
10.4 Sampling
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
73
10.5 The role of time-reversal and the backward process . . . . . . . . . . . . . . . . . . . . . . . .
74
10.6 Relation to Other Denoising Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
75
A Additional proofs
81
A.1 Discrete Mass Conservation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
81
A.2 Manifold Marginalization Trick . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
82
A.3 Regularity assumptions for KFE
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
82
2

(a) Flow
(b) Diffusion
(c) Jump
(d) CTMC
Figure 1 Four time-continuous processes (Xt)0≤t≤1 taking source sample X0 to a target sample X1. These are a flow in
a continuous state space, a diffusion in continuous state space, a jump process in continuous state space (densities
visualized with contours), and a jump process in discrete state space (states as disks, probabilities visualized with
colors).
1
Introduction
Flow matching (FM) (Lipman et al., 2022; Albergo and Vanden-Eijnden, 2022; Liu et al., 2022) is a simple
framework for generative modeling framework that has pushed the state-of-the-art in various fields and
large-scale applications including generation of images (Esser et al., 2024), videos (Polyak et al., 2024), speech
(Le et al., 2024), audio (Vyas et al., 2023), proteins (Huguet et al., 2024), and robotics (Black et al., 2024). This
manuscript and its accompanying codebase have two primary objectives. First, to serve as a comprehensive
and self-contained reference to Flow Matching, detailing its design choices and numerous extensions developed
by the research community. Second, to enable newcomers to quickly adopt and build upon Flow Matching for
their own applications.
The framework of Flow Matching is based on learning a velocity field (also called vector field). Each velocity
field defines a flow ψt by solving an ordinary differential equation (ODE) in a process called simulation. A
flow is a determinstic, time-continuous bijective transformation of the d-dimensional Euclidean space, Rd. The
goal of Flow Matching is to build a flow that transforms a sample X0 ∼p drawn from a source distribution
p into a target sample X1 := ψ1(X0) such that X1 ∼q has a desired distribution q, see figure 1a. Flow
models were introduced to the machine learning community by (Chen et al., 2018; Grathwohl et al., 2018) as
Continuous Normalizing Flows (CNFs). Originally, flows were trained by maximizing the likelihood p(X1)
of training examples X1, resulting in the need of simulation and its differentiation during training. Due to
the resulting computational burdens, later works attempted to learn CNFs without simulation (Rozen et al.,
2021; Ben-Hamu et al., 2022), evolving into modern-day Flow Matching algorithms (Lipman et al., 2022; Liu
et al., 2022; Albergo and Vanden-Eijnden, 2022; Neklyudov et al., 2023; Heitz et al., 2023; Tong et al., 2023).
The resulting framework is a recipe comprising two steps (Lipman et al., 2022), see figure 2: First, choose a
probability path pt interpolating between the source p and target q distributions. Second, train a velocity
field (neural network) that defines the flow transformation ψt implementing pt.
The principles of FM can be extended to state spaces S other than Rd and even evolution processes that are
not flows. Recently, Discrete Flow Matching (Campbell et al., 2024; Gat et al., 2024) develops a Flow Matching
algorithm for time-continuous Markov processes on discrete state spaces, also known as Continuous Time
Markov Chains (CTMC), see figure 1c. This advancement opens up the exciting possibility of using Flow
Matching in discrete generative tasks such as language modeling. RiemannianFlowMatching (Chen and Lipman,
2024) extends Flow Matching to flows on Riemannian manifolds S = M that now became the state-of-the-art
models for a wide variety of applications of machine learning in chemistry such as protein folding (Yim et al.,
2023; Bose et al., 2023). Even more generally, Generator Matching (Holderrieth et al., 2024) shows that the
Flow Matching framework works for any modality and for general Continuous Time Markov Processes (CTMPs)
including, as illustrated in figure 1, flows, diffusions, and jump processes in continuous spaces, in addition to
CTMC in discrete spaces. Remarkably, for any such CTMP, the Flow Matching recipe remains the same,
namely: First, choose a path pt interpolating source p and target q on the relevant state space S. Second,
train a generator, which plays a similar role to velocities for flows, and defines a CTMP process implementing
pt. This generalization of Flow Matching allows us to see many existing generative models in a unified light
and develop new generative models for any modality with a generative Markov process of choice.
3

Chronologically, Diffusion Models were the first to develop simulation-free training of a CTMP process, namely
a diffusion process, figure 1b. Diffusion Models were originally introduced as discrete time Gaussian processes
(Sohl-Dickstein et al., 2015; Ho et al., 2020) and later formulated in terms of continuous time Stochastic
Differential Equations (SDEs) (Song et al., 2021). In the lens of Flow Matching, Diffusion Models build
the probability path pt interpolating source and target distributions in a particular way via forward noising
processes modeled by particular SDEs. These SDEs are chosen to have closed form marginal probabilities
that are in turn used to parametrize the generator of the diffusion process (i.e., drift and diffusion coefficient)
via the score function (Song and Ermon, 2019). This parameterization is based on a reversal process to the
forward noising process (Anderson, 1982). Consequently, Diffusion Models learn the score function of the
marginal probabilities. Diffusion Models’ literature suggested also other parametrizations of the generator
besides the score, including noise prediction, denoisers (Kingma et al., 2021), or v-prediction (Salimans and
Ho, 2022)—where the latter coincides with velocity prediction for a particular choice of probability path pt.
Diffusion bridges (Peluchetti, 2023) offers another approach to design pt and generators for diffusion process
that extends diffusion models to arbitrary source-target couplings. In particular these constructions are build
again on SDEs with marginals known in closed form, and again use the score to formulate the generator
(using Doob’s h-transform). Shi et al. (2023); Liu et al. (2023) show that the linear version of Flow Matching
can be seen as a certain limiting case of bridge matching.
The rest of this manuscript is organized as follows. Section 2 offers a self-contained “cheat-sheet” to understand
and implement vanilla Flow Matching in PyTorch. Section 3 offers a rigorous treatment of flow models,
arguably the simplest of all CTMPs, for continuous state spaces. In section 4 we introduce the Flow Matching
framework in Rd and its various design choices and extensions. We show that flows can be constructed by
considering the significantly simpler conditional setting, offering great deal of flexibility in their design, e.g.,
by readily extending to Riemannian geometries, described in section 5. Section 6 provides an introduction
to Continuous Time Markov Chains (CTMCs) and the usage as generative models on discrete state spaces.
Then, section 7 discusses the extension of Flow Matching to CTMC processes. In section 8, we provide an
introduction to using Continuous Time Markov Process (CTMPs) as generative models for arbitrary state
spaces. In section 9, we describe Generator Matching (GM) - a generative modeling framework for arbitrary
modalities that describes a scalable way of training CTMPs. GM also unifies all models in previous sections
into a common framework. Finally, due to their wide-spread use, we discuss in section 10 denoising diffusion
models as a specific instance of the FM family of models.
2
Quick tour and key concepts
Given access to a training dataset of samples from some target distribution q over Rd, our goal is to build a
model capable of generating new samples from q. To address this task, Flow Matching (FM) builds a probability
path (pt)0≤t≤1, from a known source distribution p0 = p to the data target distribution p1 = q, where each pt
is a distribution over Rd. Specifically, FM is a simple regression objective to train the velocity field neural
network describing the instantaneous velocities of samples—later used to convert the source distribution p0
into the target distribution p1, along the probability path pt. After training, we generate a novel sample from
the target distribution X1 ∼q by (i) drawing a novel sample from the source distribution X0 ∼p, and (ii)
solving the Ordinary Differential Equation (ODE) determined by the velocity field.
More formally, an ODE is defined via a time-dependent vector field u : [0, 1] × Rd →Rd which, in our case, is
the velocity field modeled in terms of a neural network. This velocity field determines a time-dependent flow
ψ : [0, 1] × Rd →Rd, defined as
d
dtψt(x) = ut(ψt(x)),
where ψt := ψ(t, x) and ψ0(x) = x. The velocity field ut generates the probability path pt if its flow ψt
satisfies
Xt := ψt(X0) ∼pt for X0 ∼p0.
(2.1)
According to the equation above, the velocity field ut is the only tool necessary to sample from pt by solving the
ODE above. As illustrated in figure 2d, solving the ODE until t = 1 provides us with samples X1 = ψ1(X0),
resembling the target distribution q. Therefore, and in sum, the goal of Flow Matching is to learn a vector
field uθ
t such that its flow ψt generates a probability path pt with p0 = p and p1 = q.
4

(a) Data.
(b) Path design.
(c) Training.
(d) Sampling.
Figure 2 The Flow Matching blueprint. (a) The goal is to find a flow mapping samples X0 from a known source or noise
distribution q into samples X1 from an unknown target or data distribution q. (b) To do so, design a time-continuous
probability path (pt)0≤t≤1 interpolating between p := p0 and q := p1. (c) During training, use regression to estimate
the velocity field ut known to generate pt. (d) To draw a novel target sample X1 ∼q, integrate the estimated velocity
field uθ
t (Xt) from t = 0 to t = 1, where X0 ∼p is a novel source sample.
Using the notations above, the goal of Flow Matching is to learn the parameters θ of a velocity field uθ
t
implemented in terms of a neural network. As anticipated in the introduction, we do this in two steps: design
a probability path pt interpolating between p and q (see figure 2b), and train a velocity field uθ
t generating pt
by means of regression (see figure 2c).
Therefore, let us proceed with the first step of the recipe: designing the probability path pt. In this example,
let the source distribution p := p0 = N(x|0, I), and construct the probability path pt as the aggregation of the
conditional probability paths pt|1(x|x1), each conditioned on one of the data examples X1 = x1 comprising
the training dataset. (One of such conditional paths is illustrated in figure 3a.) The probability path pt
therefore follows the expression:
pt(x) =
Z
pt|1(x|x1)q(x1)dx1, where pt|1(x|x1) = N(x|tx1, (1 −t)2I).
(2.2)
This path, also known as the conditional optimal-transport or linear path, enjoys some desirable properties
that we will study later in this manuscript. Using this probability path, we may define the random variable
Xt ∼pt by drawing X0 ∼p, drawing X1 ∼q, and taking their linear combination:
Xt = tX1 + (1 −t)X0 ∼pt.
(2.3)
We now continue with the second step in the Flow Matching recipe: regressing our velocity field uθ
t (usually
implemented in terms of a neural network) to a target velocity field ut known to generate the desired probability
path pt. To this end, the Flow Matching loss reads:
LFM(θ) = Et,Xt
uθ
t(Xt) −ut(Xt)
2 , where t ∼U[0, 1] and Xt ∼pt.
(2.4)
In practice, one can rarely implement the objective above, because ut is a complicated object governing
the joint transformation between two high-dimensional distributions. Fortunately, the objective simplifies
drastically by conditioning the loss on a single target example X1 = x1 picked at random from the training
set. To see how, borrowing equation (2.3) to realize the conditional random variables
Xt|1 = tx1 + (1 −t)X0
∼
pt|1(·|x1) = N(· | tx1, (1 −t)2I).
(2.5)
Using these variables, solving
d
dtXt|1 = ut(Xt|1|x1) leads to the conditional velocity field
ut(x|x1) = x1 −x
1 −t ,
(2.6)
which generates the conditional probability path pt|1(·|x1). (For an illustration on these two conditional
objects, see figure 3c.) Equipped with the simple equation (2.6) for the conditional velocity fields generating
the designed conditional probability paths, we can formulate a tractable version of the Flow Matching loss
in (2.4). This is the conditional Flow Matching loss:
LCFM(θ) = Et,Xt,X1∥uθ
t (Xt) −ut(Xt|X1)∥2, where t ∼U[0, 1], X0 ∼p, X1 ∼q,
(2.7)
5

(a) Conditional probability
path pt(x|x1).
(b)
(Marginal) Probability
path pt(x).
(c) Conditional velocity field
ut(x|x1).
(d) (Marginal) Velocity field
ut(x).
Figure 3 Path design in Flow Matching. Given a fixed target sample X = x1, its conditional velocity field ut(x|x1)
generates the conditional probability path pt(x|x1). The (marginal) velocity field ut(x) results from the aggregation of
all conditional velocity fields—and similarly for the probability path pt(x).
and Xt = (1 −t)X0 + tX1. Remarkably, the objectives in (2.4) and (2.7) provide the same gradients to learn
uθ
t , i.e.,
∇θLFM(θ) = ∇θLCFM(θ).
(2.8)
Finally, by plugging ut(x|x1) from (2.6) into equation (2.7), we get the simplest implementation of Flow
Matching:
LOT,Gauss
CFM
(θ) = Et,X0,X1∥uθ
t(Xt) −(X1 −X0) ∥2, where t ∼U[0, 1], X0 ∼N(0, I), X1 ∼q.
(2.9)
A standalone implementation of this quick tour in pure PyTorch is provided in code 1. Later in the manuscript,
we will cover more sophisticated variants and design choices, all of them implemented in the accompanying
flow_matching library.
6

Code 1: Standalone Flow Matching code
flow_matching/examples/standalone_flow_matching.ipynb
1
import torch
2
from torch import nn, Tensor
3
import matplotlib.pyplot as plt
4
from sklearn.datasets import make_moons
5
6
class Flow(nn.Module):
7
def __init__(self, dim: int = 2, h: int = 64):
8
super().__init__()
9
self.net = nn.Sequential(
10
nn.Linear(dim + 1, h), nn.ELU(),
11
nn.Linear(h, h), nn.ELU(),
12
nn.Linear(h, h), nn.ELU(),
13
nn.Linear(h, dim))
14
15
def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
16
return self.net(torch.cat((t, x_t), -1))
17
18
def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
19
t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
20
# For simplicity, using midpoint ODE solver in this example
21
return x_t + (t_end - t_start) * self(x_t + self(x_t, t_start) * (t_end - t_start) / 2,
22
t_start + (t_end - t_start) / 2)
23
24
# training
25
flow = Flow()
26
optimizer = torch.optim.Adam(flow.parameters(), 1e-2)
27
loss_fn = nn.MSELoss()
28
29
for _ in range(10000):
30
x_1
= Tensor(make_moons(256, noise=0.05)[0])
31
x_0
= torch.randn_like(x_1)
32
t
= torch.rand(len(x_1), 1)
33
x_t
= (1 - t) * x_0 + t * x_1
34
dx_t = x_1 - x_0
35
optimizer.zero_grad()
36
loss_fn(flow(x_t, t), dx_t).backward()
37
optimizer.step()
38
39
# sampling
40
x = torch.randn(300, 2)
41
n_steps = 8
42
fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)
43
time_steps = torch.linspace(0, 1.0, n_steps + 1)
44
45
axes[0].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
46
axes[0].set_title(f't = {time_steps[0]:.2f}')
47
axes[0].set_xlim(-3.0, 3.0)
48
axes[0].set_ylim(-3.0, 3.0)
49
50
for i in range(n_steps):
51
x = flow.step(x, time_steps[i], time_steps[i + 1])
52
axes[i + 1].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
53
axes[i + 1].set_title(f't = {time_steps[i + 1]:.2f}')
54
55
plt.tight_layout()
56
plt.show()
7

3
Flow models
This section introduces flows, the mathematical object powering the simplest forms of Flow Matching. Later
parts in the manuscript will discuss Markov processes more general than flows, leading to more sophisticated
generative learning paradigms introducing many more design choices to the Flow Matching framework.
The reason we start with flows is three-fold: First, flows are arguably the simplest of all CTMPs—being
deterministic and having a compact parametrization via velocities—these models can transform any source
distribution p into any target distribution q, as long as these two have densities. Second, flows can be sampled
rather efficiently by approximating the solution of ODEs, compared, e.g., to the harder-to-simulate SDEs for
diffusion processes. Third, the deterministic nature of flows allows an unbiased model likelihood estimation,
while more general stochastic processes require working with lower bounds. To understand flows, we must
first review some background notions in probability and differential equations theory, which we do next.
3.1
Random vectors
Consider data in the d-dimensional Euclidean space x = (x1, . . . , xd) ∈Rd with the standard Euclidean inner
product ⟨x, y⟩= Pd
i=1 xiyi and norm ∥x∥=
p
⟨x, x⟩. We will consider random variables (RVs) X ∈Rd with
continuous probability density function (PDF), defined as a continuous function pX : Rd →R≥0 providing
event A with probability
P(X ∈A) =
Z
A
pX(x)dx,
(3.1)
where
R
pX(x)dx = 1. By convention, we omit the integration interval when integrating over the whole space
(
R
≡
R
Rd). To keep notation concise, we will refer to the PDF pXt of RV Xt as simply pt. We will use the
notation X ∼p or X ∼p(X) to indicate that X is distributed according to p. One common PDF in generative
modeling is the d-dimensional isotropic Gaussian:
N(x|µ, σ2I) = (2πσ2)−d
2 exp
 
−∥x −µ∥2
2
2σ2
!
,
(3.2)
where µ ∈Rd and σ ∈R>0 stand for the mean and the standard deviation of the distribution, respectively.
The expectation of a RV is the constant vector closest to X in the least-squares sense:
E [X] = arg min
z∈Rd
Z
∥x −z∥2 pX(x)dx =
Z
xpX(x)dx.
(3.3)
One useful tool to compute the expectation of functions of RVs is the Law of the Unconscious Statistician:
E [f(X)] =
Z
f(x)pX(x)dx.
(3.4)
When necessary, we will indicate the random variables under expectation as EXf(X).
3.2
Conditional densities and expectations
Figure 4 Joint PDF pX,Y (in
shades) and its marginals pX
and pY (in black lines).
Given two random variables X, Y ∈Rd, their joint PDF pX,Y (x, y) has
marginals
Z
pX,Y (x, y)dy = pX(x) and
Z
pX,Y (x, y)dx = pY (y).
(3.5)
See figure 4 for an illustration of the joint PDF of two RVs in R (d = 1). The
conditional PDF pX|Y describes the PDF of the random variable X when
conditioned on an event Y = y with density pY (y) > 0:
pX|Y (x|y) := pX,Y (x, y)
pY (y)
,
(3.6)
8

and similarly for the conditional PDF pY |X. Bayes’ rule expresses the condi-
tional PDF pY |X with pX|Y by
pY |X(y|x) = pX|Y (x|y)pY (y)
pX(x)
,
(3.7)
for pX(x) > 0.
The conditional expectation E [X|Y ] is the best approximating function g⋆(Y ) to X in the least-squares sense:
g⋆:= arg min
g:Rd→Rd E
h
∥X −g(Y )∥2i
= arg min
g:Rd→Rd
Z
∥x −g(y)∥2 pX,Y (x, y)dxdy
= arg min
g:Rd→Rd
Z Z
∥x −g(y)∥2 pX|Y (x|y)dx

pY (y)dy.
(3.8)
For y ∈Rd such that pY (y) > 0 the conditional expectation function is therefore
E [X|Y = y] := g⋆(y) =
Z
xpX|Y (x|y)dx,
(3.9)
where the second equality follows from taking the minimizer of the inner brackets in equation (3.8) for Y = y,
similarly to equation (3.3). Composing g⋆with the random variable Y , we get
E [X|Y ] := g⋆(Y ),
(3.10)
which is a random variable in Rd.
Rather confusingly, both E [X|Y = y] and E [X|Y ] are often called
conditional expectation, but these are different objects. In particular, E [X|Y = y] is a function Rd →Rd,
while E [X|Y ] is a random variable assuming values in Rd. To disambiguate these two terms, our discussions
will employ the notations introduced here.
The tower property is an useful property that helps simplify derivations involving conditional expectations of
two RVs X and Y :
E [E [X|Y ]] = E [X]
(3.11)
Because E [X|Y ] is a RV, itself a function of the RV Y , the outer expectation computes the expectation of
E [X|Y ]. The tower property can be verified by using some of the definitions above:
E [E [X|Y ]] =
Z Z
xpX|Y (x|y)dx

pY (y)dy
(3.6)
=
Z Z
xpX,Y (x, y)dxdy
(3.5)
=
Z
xpX(x)dx
= E [X] .
Finally, consider a helpful property involving two RVs f(X, Y ) and Y , where X and Y are two arbitrary RVs.
Then, by using the Law of the Unconscious Statistician with (3.9), we obtain the identity
E [f(X, Y )|Y = y] =
Z
f(x, y)pX|Y (x|y)dx.
(3.12)
3.3
Diffeomorphisms and push-forward maps
We denote by Cr(Rm, Rn) the collection of functions f : Rm →Rn with continuous partial derivatives of
order r:
∂rf k
∂xi1 · · · ∂xir ,
k ∈[n], ij ∈[m],
(3.13)
9

Figure 5 A flow model Xt = ψt(X0) is defined by a diffeomorphism ψt : Rd →Rd (visualized with a brown square grid)
pushing samples from a source RV X0 (left, black points) toward some target distribution q (right). We show three
different times t.
where [n] := {1, 2, . . . , n}. To keep notation concise, define also Cr(Rn) := Cr(Rm, R) so, for example,
C1(Rm) denotes the continuously differentiable scalar functions. An important class of functions are the Cr
diffeomorphism; these are invertible functions ψ ∈Cr(Rn, Rn) with ψ−1 ∈Cr(Rn, Rn).
Then, given a RV X ∼pX with density pX, let us consider a RV Y = ψ(X), where ψ : Rd →Rd is a C1
diffeomorphism. The PDF of Y , denoted pY , is also called the push-forward of pX. Then, the PDF pY can be
computed via a change of variables:
E [f(Y )] = E [f(ψ(X))] =
Z
f(ψ(x))pX(x)dx =
Z
f(y)pX(ψ−1(y))
det ∂yψ−1(y)
 dy,
where the third equality is due the change of variables x = ψ−1(y), ∂yϕ(y) denotes the Jacobian matrix (of
first order partial derivatives), i.e.,
[∂yϕ(y)]i,j = ∂ϕi
∂xj , i, j ∈[d],
and det A denotes the determinant of a square matrix A ∈Rd×d. Thus, we conclude that the PDF pY is
pY (y) = pX(ψ−1(y))
det ∂yψ−1(y)
 .
(3.14)
We will denote the push-forward operator with the symbol ♯, that is
[ψ♯pX] (y) := pX(ψ−1(y))
det ∂yψ−1(y)
 .
(3.15)
3.4
Flows as generative models
As mentioned in section 2, the goal of generative modeling is to transform samples X0 = x0 from a source
distribution p into samples X1 = x1 from a target distribution q. In this section, we start building the tools
necessary to address this problem by means of a flow mapping ψt. More formally, a Cr flow is a time-dependent
mapping ψ : [0, 1] × Rd →Rd implementing ψ : (t, x) 7→ψt(x). Such flow is also a Cr([0, 1] × Rd, Rd) function,
such that the function ψt(x) is a Cr diffeomorphism in x for all t ∈[0, 1]. A flow model is a continuous-time
Markov process (Xt)0≤t≤1 defined by applying a flow ψt to the RV X0:
Xt = ψt(X0),
t ∈[0, 1], where X0 ∼p.
(3.16)
See Figure 5 for an illustration of a flow model. To see why Xt is Markov, note that, for any choice of
0 ≤t < s ≤1, we have
Xs = ψs(X0) = ψs(ψ−1
t
(ψt(X0))) = ψs|t(Xt),
(3.17)
where the last equality follows from using equation (3.16) to set Xt = ψt(X0), and defining ψs|t := ψs ◦ψ−1
t
,
which is also a diffeomorphism. Xs = ψs|t(Xt) implies that states later than Xt depend only on Xt, so Xt is
Markov. In fact, for flow models, this dependence is deterministic.
10

Figure 6 A flow ψt : Rd →Rd (square grid) is defined by a velocity field ut : Rd →Rd (visualized with blue arrows)
that prescribes its instantaneous movements at all locations. We show three different times t.
In summary, the goal generative flow modeling is to find a flow ψt such that
X1 = ψ1(X0) ∼q.
(3.18)
3.4.1
Equivalence between flows and velocity fields
A Cr flow ψ can be defined in terms of a Cr([0, 1] × Rd, Rd) velocity field u : [0, 1] × Rd →Rd implementing
u : (t, x) 7→ut(x) via the following ODE:
d
dtψt(x) = ut(ψt(x))
(flow ODE)
(3.19a)
ψ0(x) = x
(flow initial conditions)
(3.19b)
See figure 6 for an illustration of a flow together with its velocity field.
A standard result regarding the existence and uniqueness of solutions ψt(x) to equation (3.19) is (see e.g.,
Perko (2013); Coddington et al. (1956)):
Theorem 1 (Flow local existence and uniqueness). If u is Cr([0, 1] × Rd, Rd), r ≥1 (in particular,
locally Lipschitz), then the ODE in (3.19) has a unique solution which is a Cr(Ω, Rd) diffeomorphism
ψt(x) defined over an open set Ωwhich is super-set of {0} × Rd.
This theorem guarantees only the local existence and uniqueness of a Cr flow moving each point x ∈Rd
by ψt(x) during a potentially limited amount of time t ∈[0, tx). To guarantee a solution until t = 1 for all
x ∈Rd, one must place additional assumptions beyond local Lipschitzness. For instance, one could consider
global Lipschitness, guaranteed by bounded first derivatives in the C1 case. However, we will later rely on
a different condition—namely, integrability—to guarantee the existence of the flow almost everywhere, and
until time t = 1.
So far, we have shown that a velocity field uniquely defines a flow. Conversely, given a C1 flow ψt, one can
extract its defining velocity field ut(x) for arbitrary x ∈Rd by considering the equation
d
dtψt(x′) = ut(ψt(x′)),
and using the fact that ψt is an invertible diffeomorphism for every t ∈[0, 1] to let x′ = ψ−1
t
(x). Therefore,
the unique velocity field ut determining the flow ψt is
ut(x) = ˙ψt(ψ−1
t
(x)),
(3.20)
where ˙ψt := d
dtψt. In conclusion, we have shown the equivalence between Cr flows ψt and Cr velocity fields ut.
3.4.2
Computing target samples from source samples
Computing a target sample X1—or, in general, any sample Xt—entails approximating the solution of the
ODE in equation (3.19) starting from some initial condition X0 = x0. Numerical methods for ODEs is a
classical and well researched topic in numerical analysis, and a myriad of powerful methods exist (Iserles,
2009). One of the simplest methods is the Euler method, implementing the update rule
Xt+h = Xt + hut(Xt)
(3.21)
11

Figure 7 A velocity field ut (in blue) generates a probability path pt (PDFs shown as contours) if the flow defined by
ut (square grid) reshapes p (left) to pt at all times t ∈[0, 1).
where h = n−1 > 0 is a step size hyper-parameter with n ∈N. To draw a sample X1 from the target
distribution, apply the Euler method starting at some X0 ∼p to produce the sequence Xh, X2h, . . . , X1. The
Euler method coincides with first-order Taylor expansion of Xt:
Xt+h = Xt + h ˙Xt + o(h) = Xt + hut(Xt) + o(h),
where o(h) stands for a function growing slower than h, that is, o(h)/h →0 as h →0. Therefore, the Euler
method accumulates o(h) error per step, and can be shown to accumulate o(1) error after n = 1/h steps.
Therefore, the error of the Euler method vanishes as we consider smaller step sizes h →0. The Euler method
is just one example among many ODE solvers. Code 2 exemplifies another alternative, the second-order
midpoint method, which often outperforms the Euler method in practice.
Code 2: Computing X1 with Midpoint solver
1
from flow_matching.solver import ODESolver
2
from flow_matching.utils import ModelWrapper
3
4
class Flow(ModelWrapper):
5
def __init__(self, dim=2, h=64):
6
super().__init__()
7
self.net = torch.nn.Sequential(
8
torch.nn.Linear(dim + 1, h), torch.nn.ELU(),
9
torch.nn.Linear(h, dim))
10
11
def forward(self, x, t):
12
t = t.view(-1, 1).expand(*x.shape[:-1], -1)
13
return self.net(torch.cat((t, x), -1))
14
15
velocity_model = Flow()
16
17
... # Optimize the model parameters s.t. model(x_t, t) = ut(Xt)
18
19
x_0 = torch.randn(batch_size, *data_dim)
# Specify the initial condition
20
21
solver = ODESolver(velocity_model=velocity_model)
22
num_steps = 100
23
x_1 = solver.sample(x_init=x_0, method='midpoint', step_size=1.0 / num_steps)
12

3.5
Probability paths and the Continuity Equation
We call a time-dependent probability (pt)0≤t≤1 a probability path. For our purposes, one important probability
path is the marginal PDF of a flow model Xt = ψt(X0) at time t:
Xt ∼pt.
(3.22)
For each time t ∈[0, 1], these marginal PDFs are obtained via the push-forward formula in equation (3.15),
that is,
pt(x) = [ψt♯p] (x).
(3.23)
Given some arbitrary probability path pt we define
ut generates pt if Xt = ψt(X0) ∼pt for all t ∈[0, 1).
(3.24)
In this way, we establish a close relationship between velocity fields, their flows, and the generated probability
paths, see Figure 7 for an illustration. Note that we use the time interval [0, 1), open from the right, to allow
dealing with target distributions q with compact support where the velocity is not defined precisely at t = 1.
To verify that a velocity field ut generates a probability path pt, one can verify if the pair (ut, pt) satisfies a
partial differential equation (PDE) known as the Continuity Equation:
d
dtpt(x) + div(ptut)(x) = 0,
(3.25)
where div(v)(x) = Pd
i=1 ∂xivi(x), and v(x) = (v1(x), . . . , vd(x)). The following theorem, a rephrased version
of the Mass Conservation Formula (Villani et al., 2009), states that a solution ut to the Continuity Equation
generates the probability path pt:
Theorem 2 (Mass Conservation). Let pt be a probability path and ut a locally Lipchitz integrable vector
field. Then, the following two statements are equivalent:
1. The Continuity Equation (3.25) holds for t ∈[0, 1).
2. ut generates pt, in the sense of (3.24).
In the previous theorem, local Lipschitzness assumes that there exists a local neighbourhood over which ut(x)
is Lipschitz, for all (t, x). Assuming that u is integrable means that:
Z 1
0
Z
∥ut(x)∥pt(x)dxdt < ∞.
(3.26)
Specifically, integrating a solution to the flow ODE (3.19a) across times [0, t] leads to the integral equation
ψt(x) = x +
Z t
0
us(ψs(x))ds.
(3.27)
Therefore, integrability implies
E ∥Xt∥
(3.16)
=
Z
∥ψt(x)∥p(x)dx
=
Z x +
Z t
0
us(ψs(x))ds
 p(x)dx
(i)
≤E ∥X0∥+
Z 1
0
Z
∥us(x)∥pt(x)dt
(ii)
< ∞,
where (i) follows from the triangle inequality, and (ii) assumes the integrability condition (3.26) and E ∥X0∥<
∞. In sum, integrability allows assuming that Xt has bounded expected norm, if X0 also does.
13

To gain further insights about the meaning of the Continuity Equation, we may write it in integral form
by means of the Divergence Theorem—see Matthews (2012) for an intuitive exposition, and Loomis and
Sternberg (1968) for a rigorous treatment. This result states that, considering some domain D and some
smooth vector field u : Rd →Rd, accumulating the divergences of u inside D equals the flux leaving D by
orthogonally crossing its boundary ∂D:
Z
D
div(u)(x)dx =
Z
∂D
⟨u(y), n(y)⟩dsy,
(3.28)
Figure8 The continuity equation
asserts that the local change in
probability equals minus the net
outgoing probability flux.
where n(y) is a unit-norm normal field pointing outward to the domain’s
boundary ∂D, and dsy is the boundary’s area element. To apply these
insights to the Continuity Equation, let us integrate (3.25) over a small
domain D ⊂Rd (for instance, a cube) and apply the Divergence Theorem
to obtain
d
dt
Z
D
pt(x)dx = −
Z
D
div(ptut)(x)dx = −
Z
∂D
⟨pt(y)ut(y), n(y)⟩dsy.
(3.29)
This equation expresses the rate of change of total probability mass in the
volume D (left-hand side) as the negative probability flux leaving the domain
(right-hand side). The probability flux, defined as jt(y) = pt(y)ut(y), is the
probability mass flowing through the hyperplane orthogonal to n(y) per unit
of time and per unit of (possibly high-dimensional) area. See figure 8 for
an illustration.
3.6
Instantaneous Change of Variables
One important benefit of using flows as generative models is that they allow
the tractable computation of exact likelihoods log p1(x), for all x ∈Rd. This feature is a consequence of the
Continuity Equation called the Instantaneous Change of Variables (Chen et al., 2018):
d
dt log pt(ψt(x)) = −div(ut)(ψt(x)).
(3.30)
This is the ODE governing the change in log-likelihood, log pt(ψt(x)), along a sampling trajectory ψt(x)
defined by the flow ODE (3.19a). To derive (3.30), differentiate log pt(ψt(x)) with respect to time, and apply
both the Continuity Equation (3.25) and the flow ODE (3.19a). Integrating (3.30) from t = 0 to t = 1 and
rearranging, we obtain
log p1(ψ1(x)) = log p0(ψ0(x)) −
Z 1
0
div(ut)(ψt(x))dt.
(3.31)
In practice, computing div(ut), which equals the trace of the Jacobian matrix ∂xut(x) ∈Rd×d, is increasingly
challenge as the dimensionality d grows. Because of this reason, previous works employ unbiased estimators
such as Hutchinson’s trace estimator (Grathwohl et al., 2018):
div(ut)(x) = tr [∂xut(x)] = EZ tr

ZT ∂xut(x)Z

,
(3.32)
where Z ∈Rd×d is any random variable with E [Z] = 0 and Cov (Z, Z) = I, (for example, Z ∼N(0, I)),
and tr[Z] = Pd
i=1 Zi,i. By plugging the equation above into (3.31) and switching the order of integral and
expectation, we obtain the following unbiased log-likelihood estimator:
log p1(ψ1(x)) = log p0(ψ0(x)) −EZ
Z 1
0
tr

ZT ∂xut(ψt(x))Z

dt.
(3.33)
In contrast to div(ut)(ψt(x)) in (3.30), computing tr

ZT ∂xut(ψt(x))Z

for a fixed sample Z in the equation
above can be done with a single backward pass via a vector-Jacobian product (JVP)1.
1E.g., see https://pytorch.org/docs/stable/generated/torch.autograd.functional.vjp.html.
14

In summary, computing an unbiased estimate of log p1(x) entails simulating the ODE
d
dt
f(t)
g(t)

=

ut(f(t))
−tr

ZT ∂xut(f(t))Z


,
(3.34a)
f(1)
g(1)

=
x
0

,
(3.34b)
backwards in time, from t = 1 to t = 0, and setting:
d
log p1(x) = log p0(f(0)) −g(0).
(3.35)
See code 3 for an example on how to obtain log-likelihood estimates from a flow model using the flow_matching
library.
Code 3: Computing the likelihood
1
from flow_matching.solver import ODESolver
2
from flow_matching.utils import ModelWrapper
3
from torch.distributions.normal import Normal
4
5
velocity_model: ModelWrapper = ...
# Train the model parameters s.t. model(x_t, t) = ut(xt)
6
7
x_1 = torch.randn(batch_size, *data_dim)
# Point X1 where we wish to compute log p1(x)
8
9
# Define log p0(x)
10
gaussian_log_density = Normal(torch.zeros(size=data_dim), torch.ones(size=data_dim)).log_prob
11
12
solver = ODESolver(velocity_model=velocity_model)
13
num_steps = 100
14
x_0, log_p1 = solver.compute_likelihood(
15
x_1=x_1,
16
method='midpoint',
17
step_size=1.0 / num_steps,
18
log_p0=gaussian_log_density
19
)
3.7
Training flow models with simulation
The Instantaneous Change of Variables, and the resulting ODE system (3.34), allows training a flow model by
maximizing the log-likelihood of training data (Chen et al., 2018; Grathwohl et al., 2018). Specifically, let uθ
t
be a velocity field with learnable parameters θ ∈Rp, and consider the problem of learning θ such that
pθ
1 ≈q.
(3.36)
We can pursue this goal, for instance, by minimizing the KL-divergence of pθ
1 and q:
L(θ) = DKL(q, pθ
1) = −EY ∼q log pθ
1(Y ) + constant,
(3.37)
where pθ
1 is the distribution of X1 = ψθ
1(X0), ψθ
t is defined by uθ
t , and we can obtain an unbiased estimate of
log pθ
1(Y ) via the solution to the ODE system (3.34). However, computing this loss—as well as its gradients—
requires precise ODE simulations during training, where only errorless solutions constitute unbiased gradients.
In contrast, Flow Matching, presented next, is a simulation-free framework to train flow generative models
without the need of solving ODEs during training.
15

4
Flow Matching
Given a source distribution p and a target distribution q, Flow Matching (FM) (Lipman et al., 2022; Liu
et al., 2022; Albergo and Vanden-Eijnden, 2022) is a scalable approach for training a flow model, defined by a
learnable velocity uθ
t, and solving the Flow Matching Problem:
Find uθ
t generating pt, with p0 = p and p1 = q.
(4.1)
In the equation above, “generating” is in the sense of equation (3.24). Revisiting the Flow Matching blueprint
from figure 2, the FM framework (a) identifies a known source distribution p and an unknown data target
distribution q, (b) prescribes a probability path pt interpolating from p0 = p to p1 = q, (c) learns a velocity
field uθ
t implemented in terms of a neural network and generating the path pt, and (d) samples from the
learned model by solving an ODE with uθ
t. To learn the velocity field uθ
t in step (c), FM minimizes the
regression loss:
LFM(θ) = EXt∼ptD
 ut(Xt), uθ
t(Xt)

,
(4.2)
where D is a dissimilarity measure between vectors, such as the squared ℓ2-norm D(u, v) = ∥u −v∥2.
Intuitively, the FM loss encourages our learnable velocity field uθ
t to match the ground truth velocity field ut
known to generate the desired probability path pt. Figure 9 depicts the main objects in the Flow Matching
framework and their dependencies. Let us start our exposition of Flow Matching by describing how to build
pt and ut, as well as a practical implementation of the loss (4.2).
4.1
Data
To reiterate, let source samples be a RV X0 ∼p and target samples a RV X1 ∼q. Commonly, source samples
follow a known distribution that is easy to sample, and target samples are given to us in terms of a dataset of
finite size. Depending on the application, target samples may constitute images, videos, audio segments, or
other types of high-dimensional, richly structured data. Source and target samples can be independent, or
originate from a general joint distribution known as the coupling
(X0, X1) ∼π0,1(X0, X1),
(4.3)
where, if no coupling is known, the source-target samples are following the independent coupling π0,1(X0, X1) =
p(X0)q(X1). One common example of independent source-target distributions is to consider the generation
of images X1 from random Gaussian noise vectors X0 ∼N(0, I). As an example of a dependent coupling,
consider the case of producing high-resolution images X1 from their low resolution versions X0, or producing
colorized videos X1 from their gray-scale counterparts X0.
4.2
Building probability paths
Flow Matching drastically simplifies the problem of designing a probability path pt—together with its
corresponding velocity field ut—by adopting a conditional strategy. As a first example, consider conditioning
the design of pt on a single target example X1 = x1, yielding the conditional probability path pt|1(x|x1)
illustrated in figure 3a. Then, we may construct the overall, marginal probability path pt by aggregating such
conditional probability paths pt|1:
pt(x) =
Z
pt|1(x|x1)q(x1)dx1,
(4.4)
as illustrated in figure 3b. To solve the Flow Matching Problem, we would like pt to satisfy the following
boundary conditions:
p0 = p,
p1 = q,
(4.5)
that is, the marginal probability path pt interpolates from the source distribution p at time t = 0 to the
target distribution q at time t = 1. These boundary conditions can be enforced by requiring the conditional
probability paths to satisfy
p0|1(x|x1) = π0|1(x|x1), and p1|1(x|x1) = δx1(x),
(4.6)
16

Flow
ψt(x)
ψt(x|x1)
tx1 + (1 −t)x
Velocity field
ut(x)
ut(x|x1)
(x1 −x)/(1 −t)
Probability path
pt(x)
pt(x|x1)
N(x|tx1, (1 −t)2I)
Boundary conds.
p0 = p
p1 = q
p0 = p
p1 = δx1
p0 = N(0, I)
p1 = δx1
Loss
Flow Matching (FM) (4.22)
D
 ut(Xt), uθ
t (Xt)

Conditional FM (CFM) (4.23)
D
 ut(Xt|X1), uθ
t(Xt)

OT, Gauss CFM (2.9)
∥uθ
t(Xt) −(X1 −X0)∥2
differentiation
solve ODE
differentiation
solve ODE
Continuity (3.25)
non-unique solution
Continuity (3.25)
non-unique solution
cond. expectation
???
conditioning
marginalization
push-forward X0
push-forward X0
Figure 9 Main objects of the Flow Matching framework and their relationships. A Flow is represented with a
Velocity field defining a random process generating a Probability path . The main idea of Flow Matching is to
break down the construction of a complex flow satisfying the desired Boundary conditions (top row) to conditional
flows (middle row) satisfying simpler Boundary conditions and consequently easier to solve. The arrows indicate
dependencies between different objects; Blue arrows signify relationships employed by the Flow Matching framework.
The Loss column lists the losses for learning the Velocity field , where the CFM loss (middle and bottom row) is
what used in practice. The bottom row lists the simplest FM algorithm instantiation as described in section 2.
where the conditional coupling π0|1(x0|x1) = π0,1(x0, x1)/q(x1) and δx1 is the delta measure centered at x1.
For the independent coupling π0,1(x0, x1) = p(x0)q(x1), the first constraint above reduces to p0|1(x|x1) = p(x).
Because the delta measure does not have a density, the second constraint should be read as
R
pt|1(x|y)f(y)dy →
f(x) as t →1 for continuous functions f. Note that the boundary conditions (4.5) can be verified plugging
(4.6) into (4.4).
A popular example of a conditional probability path satisfying the conditions in (4.6) was given in (2.2):
N(· | tx1, (1 −t)2I) →δx1(·) as t →1.
4.3
Deriving generating velocity fields
Equipped with a marginal probability path pt, we now build a velocity field ut generating pt. The generating
velocity field ut is an average of multiple conditional velocity fields ut(x|x1), illustrated in figure 3c, and
satisfying:
ut(·|x1) generates pt|1(·|x1).
(4.7)
Then, the marginal velocity field ut(x), generating the marginal path pt(x), illustrated in figure 3d, is given by
averaging the conditional velocity fields ut(x|x1) across target examples:
ut(x) =
Z
ut(x|x1)p1|t(x1|x)dx1.
(4.8)
To express the equation above using known terms, recall Bayes’ rule
p1|t(x1|x) = pt|1(x|x1)q(x1)
pt(x)
,
(4.9)
17

defined for all x with pt(x) > 0. Equation (4.8) can be interpreted as the weighted average of the conditional
velocities ut(x|x1), with weights p1|t(x1|x) representing the posterior probability of target samples x1 given the
current sample x. Another interpretation of (4.8) can be given with conditional expectations (see section 3.2).
Namely, if Xt is any RV such that Xt ∼pt|1(·|X1), or equivalently, the joint distribution of (Xt, X1) has
density pt,1(x, x1) = pt|1(x|x1)q(x1) then using (3.12) to write (4.8) as a conditional expectation, we obtain
ut(x) = E [ut(Xt|X1) | Xt = x] ,
(4.10)
which yields the useful interpretation of ut(x) as the least-squares approximation to ut(Xt|X1) given Xt = x,
see section 3.2. Note, that the Xt in (4.10) is in general a different RV that Xt defined by the final flow model
(3.16), although they share the same marginal probability pt(x).
4.4
General conditioning and the Marginalization Trick
To justify the constructions above, we need to show that the marginal velocity field ut from equations (4.8)
and (4.10) generates the marginal probability path pt from equation (4.4) under mild assumptions. The
mathematical tool to prove this is the Mass Conservation Theorem (theorem 2). To proceed, let us consider
a slightly more general setting that will be useful later in the manuscript. In particular, there is nothing
special about building conditional probability paths and velocity fields by conditioning on X1 = x1. As noted
in Tong et al. (2023), the analysis from the previous section carries through to conditioning on any arbitrary
RV Z ∈Rm with PDF pZ. This yields the marginal probability path
pt(x) =
Z
pt|Z(x|z)pZ(z)dz,
(4.11)
which in turn is generated by the marginal velocity field
ut(x) =
Z
ut(x|z)pZ|t(z|x)dz = E [ut(Xt|Z) | Xt = x] ,
(4.12)
where ut(·|z) generates pt|Z(·|z), pZ|t(z|x) =
pt|Z(x|z)pZ(z)
pt(x)
follows from Bayes’ rule given pt(x) > 0, and
Xt ∼pt|Z(·|Z). Naturally, we can recover the constructions in previous sections by setting Z = X1. Before
we prove the main result, we need some regularity assumptions, encapsulated as follows.
Assumption 1. pt|Z(x|z) is C1([0, 1) × Rd) and ut(x|z) is C1([0, 1) × Rd, Rd) as a function of (t, x).
Furthermore, pZ has bounded support, that is, pZ(x) = 0 outside some bounded set in Rm. Finally,
pt(x) > 0 for all x ∈Rd and t ∈[0, 1).
These are mild assumptions. For example, one can show that pt(x) > 0 by finding a condition z such that
pZ(z) > 0 and pt|Z(·|z) > 0. In practice, one can satisfy this by considering (1 −(1 −t)ϵ)pt|Z + (1 −t)ϵN(0, I)
for an arbitrarily small ϵ > 0. One example of pt|Z(·|z) satisfying this assumption is the path in (2.2), where
we let Z = X1. We are now ready to state the main result:
Theorem 3 (Marginalization Trick). Under assumption 1, if ut(x|z) is conditionally integrable and
generates the conditional probability path pt(·|z), then the marginal velocity field ut generates the marginal
probability path pt, for all t ∈[0, 1).
In the theorem above, conditionally integrable refers to a conditional version of the integrability condition
from the Mass Conservation Theorem (3.26), namely:
Z 1
0
Z Z
∥ut(x|z)∥pt|Z(x|z)pZ(x)dzdxdt < ∞.
(4.13)
Proof. The result follows from verifying the two conditions of the Mass Conservation in theorem 2. First, let
us check that the pair (ut, pt) satisfies the Continuity Equation (3.25). Because ut(·|x1) generates pt(·|x1), we
18

have that
d
dtpt(x)
(i)
=
Z
d
dtpt|Z(x|z)pZ(x)dz
(4.14)
(ii)
= −
Z
divx

ut(x|z)pt|Z(x|z)

pZ(z)dz
(4.15)
(i)
= −divx
Z
ut(x|z)pt|Z(x|z)pZ(z)dz
(4.16)
(iii)
= −divx [ut(x)pt(x)] .
(4.17)
Equalities (i) follows from switching differentiation ( d
dt and divx, respectively) and integration, as justified by
Leibniz’s rule, the fact that pt|Z(x|z) and ut(x|z) are C1 in t, x, and the fact that pZ has bounded support
(so all the integrands are integrable as continuous functions over bounded sets). Equality (ii) follows from the
fact that ut(·|z) generates pt|Z(·|z) and theorem 2. Equality (iii) follows from multiplying and dividing by
pt(x) (strictly positive by assumption) and using the formula (4.12) for ut.
To verify the second and last condition from theorem 2, we shall prove that ut is integrable and locally
Lipschitz. Because C1 functions are locally Lipschitz, it suffices to check that ut(x) is C1 for all (t, x). This
would follow from verifying that ut(x|z) and pt|Z(x|z) are C1 and pt(x) > 0, which hold by assumption.
Furthermore, ut(x) is integrable because ut(x|z) is conditionally integrable:
Z 1
0
Z
∥ut(x)∥pt(x)dxdt ≤
Z 1
0
Z Z
∥ut(x|z)∥pt|Z(x|z)pZ(z)dzdxdt < ∞,
(4.18)
where the first inequality follows from vector Jensen’s inequality.
4.5
Flow Matching loss
After having established that the target velocity field ut generates the prescribed probability path pt from p
to q, the missing ingredient is a tractable loss function to learn a velocity field model uθ
t as close as possible
to the target ut. One major roadblock towards stating this loss function directly is that computing the target
ut is infeasible, as it requires marginalizing over the entire training set (that is, integrating with respect to
x1 in equation (4.8) or with respect to z in equation (4.12)). Fortunately, a family of loss functions known
as Bregman divergences provides unbiased gradients to learn uθ
t(x) in terms of conditional velocities ut(x|z)
alone.
Figure 10 Bregman divergence.
Bregman divergences measure dissimilarity between two vectors u, v ∈Rd as
D(u, v) := Φ(u) −[Φ(v) + ⟨u −v, ∇Φ(v)⟩] ,
(4.19)
where Φ : Rd →R is a strictly convex function defined over some convex
set Ω⊂Rd. As illustrated in figure 10, the Bregman divergence measures
the difference between Φ(u) and the linear approximation to Φ developed
around v and evaluated at u. Because linear approximations are global
lower bounds for convex functions, it holds that D(u, v) ≥0. Further, as
Φ is strictly convex, it follows that D(u, v) = 0 if and only if u = v. The
most basic Bregman divergence is the squared Euclidean distance D(u, v) = ∥u −v∥2, esulting from choosing
Φ(u) = ∥u∥2. The key property that makes Bregman divergences useful for Flow Matching is that their
gradient with respect to the second argument is affine invariant (Holderrieth et al., 2024):
∇vD(au1 + bu2, v) = a∇vD(u1, v) + b∇vD(u2, v), for any a + b = 1,
(4.20)
as it can be verified from equation (4.19). Affine invariance allows us to swap expected values with gradients
as follows:
∇vD(E[Y ], v) = E[∇vD(Y, v)] for any RV Y ∈Rd.
(4.21)
19

The Flow Matching loss employs a Bregman divergence to regress our learnable velocity uθ
t (x) onto the target
velocity ut(x) along the probability path pt:
LFM(θ) = Et,Xt∼ptD(ut(Xt), uθ
t(Xt)),
(4.22)
where time t ∼U[0, 1]. As mentioned above, however, the target velocity ut is not tractable, so the loss above
cannot be computed as is. Instead, we consider the simpler and tractable Conditional Flow Matching (CFM) loss:
LCFM(θ) = Et,Z,Xt∼pt|Z(·|Z)D(ut(Xt|Z), uθ
t (Xt)).
(4.23)
The two losses are equivalent for learning purposes, since their gradients coincide (Holderrieth et al.,
2024):
Theorem 4. The gradients of the Flow Matching loss and the Conditional Flow Matching loss coincide:
∇θLFM(θ) = ∇θLCFM(θ).
(4.24)
In particular, the minimizer of the Conditional Flow Matching loss is the marginal velocity ut(x).
Proof. The proof follows a direct computation:
∇θLFM(θ) = ∇θEt,Xt∼ptD(ut(Xt), uθ
t(Xt))
= Et,Xt∼pt∇θD(ut(Xt), uθ
t(Xt))
(i)
= Et,Xt∼pt∇vD(ut(Xt), uθ
t (Xt))∇θuθ
t (Xt)
(4.12)
=
Et,Xt∼pt∇vD(EZ∼pZ|t(·|Xt) [ut(Xt|Z)], uθ
t (Xt))∇θuθ
t(Xt)
(ii)
= Et,Xt∼ptEZ∼pZ|t(·|Xt)

∇vD(ut(Xt|Z), uθ
t(Xt))∇θuθ
t(Xt)

(iii)
= Et,Xt∼ptEZ∼pZ|t(·|Xt)[∇θD(ut(Xt|Z), uθ
t(Xt))]
(iv)
= ∇θEt,Z∼q,Xt∼pt|Z(·|Z)[D(ut(Xt|Z), uθ
t(Xt))]
= ∇θLCFM(θ)
where in (i),(iii) we used the chain rule; (ii) follows from equation (4.21) applied conditionally on Xt; and in
(iv) we use Bayes’ rule.
Bregman divergences for learning conditional expectations.
Theorem 4 is a particular instance of a more
general result utilizing Bregman divergences for learning conditional expectations described next. It will be
used throughout this manuscript and provide the basis for all scalable losses behind Flow Matching:
Proposition 1 (Bregman divergence for learning conditional expectations). Let X ∈SX, Y ∈SY be
RVs over state spaces SX, SY and g : Rp × SX →Rn, (θ, x) 7→gθ(x), where θ ∈Rp denotes learnable
parameters. Let Dx(u, v), x ∈SX be a Bregman divergence over a convex set Ω⊂Rn that contains the
image of f. Then,
∇θEX,Y DX
 Y , gθ(X)

= ∇θEXDX
 E [Y | X], gθ(X)

.
(4.25)
In particular, for all x with pX(x) > 0, the global minimum of gθ(x) w.r.t. θ satisfies
gθ(x) = E [Y | X = x].
(4.26)
20

Proof. We assume gθ is differentiable w.r.t. θ and that the distributions of X and Y , as well as Dx, and g
allow switching differentiation and integration, develop:
∇θEX,Y DX
 Y, gθ(X)
 (i)
= EX

E

∇vDX
 Y, gθ(X)

∇θgθ(X) | X

(ii)
= EX

∇vDX
 E [Y | X], gθ(X)

∇θgθ(X)

(iii)
= EX

∇θDX
 E [Y | X], gθ(X)

= ∇θEXDX
 E [Y | X], gθ(X)

,
where (i) follows from the chain rule and the tower property of expectations (3.11). Equality (ii) follows
from (4.21). Equality (iii) uses the chain rule again. Lastly, for every x ∈SX with pX(x) > 0 we can choose
gθ(x) = E [Y |X = x], obtaining EXDX
 E [Y | X], gθ(X)

= 0, which must be the global minimum with
respect to θ.
Theorem 4 is readily shown from proposition 1 by making the choices X = Xt, Y = ut(Xt|Z), gθ(x) = uθ
t (x),
and taking the expectation with respect to t ∼U[0, 1].
General time distributions
One useful variation of the FM loss is to sample times t from a distribution
other than Uniform. Specifically, consider t ∼ω(t), where ω is a PDF over [0, 1]. This leads to the following
weighted objective:
LCFM(θ) = Et∼ω,Z,XtD(ut(Xt|Z), uθ
t(Xt)) = Et∼U,Z,Xtω(t)D(ut(Xt|Z), uθ
t(Xt)).
(4.27)
Although mathematically equivalent, sampling t ∼ω leads to better performance than using weights ω(t) in
large scale image generation tasks (Esser et al., 2024).
4.6
Solving conditional generation with conditional flows
So far, we have reduced the problem of training a flow model uθ
t to: (i) Find conditional probability paths
pt|Z(x|z) yielding a marginal probability path pt(x) satisfying the boundary conditions in (4.5). (ii) Find
conditional velocity fields ut(x|z) generating the conditional probability path. (iii) Train using the Conditional
Flow Matching loss (see equation (4.23)). We now discuss a concrete options on how to do step (i) and (ii),
i.e., design such conditional probability paths and velocity fields.
We will now propose a flexible method to design such conditional probability paths and velocity fields using
a specific construction via conditional flows. The idea is as follows: Define a flow model Xt|1 (similarly to
(3.16)) satisfying the boundary conditions (4.6), and extract the velocity field from Xt|1 by differentiation
(3.20). This process defines both pt|1(x|x1) and ut(x|x1). In more detail, define the conditional flow model
Xt|1 = ψt(X0|x1),
where X0 ∼π0|1(· | x1),
(4.28)
where ψ : [0, 1) × Rd × Rd →Rd is a conditional flow defined by
ψt(x|x1) =
(
x
t = 0
x1
t = 1 ,
(4.29)
smooth in (t, x), and a diffeomorphism in x. (Smooth here means that all derivatives of ψt(x|x1) with respect
to t and x exist and are continuous: C∞([0, 1) × Rd, Rd)). These conditions could be further relaxed to
C2([0, 1) × Rd, Rd) at the expense of simplicity.) The push-forward formula (3.15) defines the probability
density of Xt|1 as
pt|1(x|x1) :=

ψt(·|x1)♯π0|1(·|x1)

(x),
(4.30)
although we will not need this expression in practical optimization of the CFM loss it is used theoretically
to show that pt|1 satisfies the two boundary conditions (4.6). First, and according to (4.29), ψ0(·|x1) is
the identity map, keeping π0|1(·|x1) intact at time t = 0.
Second, ψ1(·|x1) = x1 is the constant map,
concentrating all probability mass at x1 as t →1. Furthermore, note that ψt(·|x1) is a smooth diffeomorphism
21

for t ∈[0, 1). Therefore, by the equivalence of flows and velocity fields (section 3.4.1), there exists a unique
smooth conditional velocity field (see equation (3.20)) taking form:
ut(x|x1) = ˙ψt(ψ−1
t
(x|x1)|x1).
(4.31)
To summarize: we have further reduced the task of finding the conditional path and a corresponding generating
velocity to simply building a conditional flow ψt(·|x1) satisfying (4.29). In section 4.7 we will pick a particularly
simple ψt(x|x1) with some desirable properties (conditional Optimal Transport flow) that leads to the standard
Flow Matching algorithm as seen in section 1, and in section 4.8 we will discuss a particular and well-known
family of conditional flows, namely affine flows that include some known examples from the diffusion models’
literature. In section 5 we will use conditional flows to define Flow Matching on manifold which showcase the
flexibility of this approach.
4.6.1
The Conditional Flow Matching loss, revisited
Let us revisit the CFM loss (4.23) by setting Z = X1 and using the conditional flows way of defining the
conditional probability path and velocity,
LCFM(θ) = Et,X1,Xt∼pt(·|X1)D
 ut(Xt|X1), uθ
t (Xt)

(3.4)
= Et,(X0,X1)∼π0,1D

˙ψt(X0|X1), uθ
t(Xt)

(4.32)
where in the second equality we used the Law of Unconscious Statistician with Xt = ψt(X0|X1) and
ut(Xt|X1)
(4.31)
=
˙ψt

ψ−1
t
 ψt (X0|X1)
X1
 X1

= ˙ψt(X0|X1).
(4.33)
The minimizer of the loss (4.32) according to proposition 1 takes the form as in (Liu et al., 2022),
ut(x) = E
h
˙ψt(X0|X1)
 Xt = x
i
.
(4.34)
In the flow_matching library the ProbPath object defines a probability path. This probability path can be
sampled at (t, X0, X1) to obtain Xt and ˙ψt(X0|X1). Then, one can compute a Monte Carlo estimate of the
CFM loss LCFM(θ). An example training loop with the CFM objective is shown in code 4.
Code 4: Training with the conditional flow matching (CFM) loss
1
import torch
2
from flow_matching.path import ProbPath
3
from flow_matching.path.path_sample import PathSample
4
5
path: ProbPath = ...
# The flow_matching library implements the most common probability paths
6
velocity_model: torch.nn.Module = ...
# Initialize the velocity model
7
optimizer = torch.optim.Adam(velocity_model.parameters())
8
9
for x_0, x_1 in dataloader:
# Samples from π0,1 of shape [batch_size, *data_dim]
10
t = torch.rand(batch_size)
# Randomize time t ∼U[0, 1]
11
sample: PathSample = path.sample(t=t, x_0=x_0, x_1=x_1)
12
x_t = sample.x_t
13
dx_t = sample.dx_t
# dX_t is
˙ψt(X0|X1).
14
# If D is the Euclidean distance, the CFM objective corresponds to the mean-squared error
15
cfm_loss = torch.pow(velocity_model(x_t, t) - dx_t, 2).mean()
# Monte Carlo estimate
16
optimizer.zero_grad()
17
cfm_loss.backward()
18
optimizer.step()
22

pt(x) =
ut(x) =
X1 conditioning
ψt(X0|x1) ∼pt|1(·|x1)
R
pt|1(x|x1)q(x1)dx1
E [ut(Xt|X1)|Xt = x]
=
=
X0 conditioning
ψt(X1|x0) ∼pt|0(·|x0)
R
pt|0(x|x0)p(x0)dx1
E [ut(Xt|X0)|Xt = x]
=
=
(X0,X1) conditioning
ψt(x0, x1) ∼pt|0,1(·|x0, x1)
R
pt|0,1(x|x0, x1)π0,1(x0, x1)dx0dx1
E [ut(Xt|X0, X1)|Xt = x]
Figure 11 Different forms of conditioning in Flow Matching and path design with corresponding conditional flows. When
the conditional flows are a diffeomorphism, all constructions are equivalent. When, they are not, extra conditions are
required to validate that the marginal velocity generates the marginal path, see text for more details.
4.6.2
The Marginalization Trick for probability paths built from conditional flows
Next, we introduce a version of the Marginalization trick for probability paths that are built from conditional
flows. To this end, note that if π0|1(·|x1) is C1, then pt(x|x1) is also C1 by construction; moreover, ut(x|x1)
is conditionally integrable if
Et,(X0,X1)∼π0,1
 ˙ψt(X0|X1)
 < ∞.
(4.35)
Therefore, by setting Z = X1, the following corollary to theorem 3 is obtained.
Corollary 1. Assume that q has bounded support, π0|1(·|x1) is C1(Rd) and strictly positive for some x1
with q(x1) > 0, and ψt(x|x1) is a conditional flow satisfying equations (4.29) and (4.35). Then pt|1(x|x1)
and ut(x|x1), defined in (4.30) and (4.31), respectively, define a marginal velocity field ut(x) generating
the marginal probability path pt(x) interpolating p and q.
Proof. If π0|1(·|x1) > 0 for some x1 ∈Rd such that q(x1) > 0, it follows that pt|1(x|x1) > 0 for all x ∈Rd and
is C1([0, 1) × Rd) (see (4.30) and (3.15) for definitions). Furthermore, ut(x|x1) (defined in (4.31)) is smooth
and satisfies
Z 1
0
Z
∥ut(x|x1)∥pt|1(x|x1)q(x1)dx1dxdt = Et,X1∼q,Xt∼pt|1(·|X1) ∥ut(Xt|X1)∥
(3.4)
= Et,X1∼q,X0∼π0|1(·|X1) ∥ut(ψt(X0|X1)|X1)∥
(4.33)
=
Et,(X0,X1)∼π0,1
 ˙ψt(X0|X1)

< ∞.
Therefore, ut(x|x1) is conditionally integrable (see (4.13)). By theorem 3, the marginal ut generates pt.
Because pt|1(x|x1) as defined by (4.30) satisfies (4.6), it follows that pt interpolates p and q.
This theorem will be used as a tool to show that particular choices of conditional flows lead to marginal
velocity ut(x) generating the marginal probability path pt(x).
23

4.6.3
Conditional flows with other conditions
Different conditioning choices Z exist but are essentially all equivalent. As illustrated in figure 11, main
options include fixing target samples Z = X1 (Lipman et al., 2022), source samples Z = X0 (Esser et al.,
2024), or two-sided Z = (X0, X1) (Albergo and Vanden-Eijnden, 2022; Liu et al., 2022; Pooladian et al., 2023;
Tong et al., 2023).
Let us focus on the two-sided condition Z = (X0, X1). Following the FM blueprint described above, we are
now looking to build a conditional probability path pt|0,1(x|x0, x1) and a corresponding generating velocity
ut(x|x0, x1) such that
p0|0,1(x|x0, x1) = δx0(x), and p1|0,1(x|x0, x1) = δx1(x).
(4.36)
We will keep this discussion formal as it requires usage of delta functions δ and our existing derivations
so far only deals with probability densities (and not general distributions). To build such a path we can
consider an interpolant (Albergo and Vanden-Eijnden, 2022) defined by Xt|0,1 = ψt(x0, x1) for a function
ψ : [0, 1] × Rd × Rd →Rd satisfying conditions similar to (4.29),
ψt(x0, x1) =
(
x0
t = 0
x1
t = 1.
(4.37)
Therefore, ψt(·, x1) pushes δx0(x) to δx1(x). We now, similarly to before, define the conditional probability
path to be
pt|0,1(·|x0, x1) := ψt(·, x1)♯δx0(·)
(4.38)
which satisfies the boundary constraints in (4.36). Albergo and Vanden-Eijnden (2022)’s stochastic interpolant
is defined by
Xt = ψt(X0, X1) ∼pt(·) =
Z
pt|0,1(·|x0, x1)π0,1(x0, x1)dx0dx1.
(4.39)
Next, the conditional velocity along this path can also be computed with (3.20) giving
ut(x|x0, x1) = ˙ψt(x0, x1)
(4.40)
which is defined only for x = ψt(x0, x1). Ignoring for a second the extra conditions, Theorem 3 now presumably
implies that the marginal velocity generating pt(x) is
ut(x) = E [ut(Xt|X0, X1) | Xt = x]
= E
h
˙ψt(X0, X1) | Xt = x
i
,
which leads to the same marginal formula as the X1-conditioned case (4.34), but with a seemingly more
permissive conditional flow ψt(x0, x1) which is only required to be an interpolant now, weakening the more
stringent diffeomorphism condition. However, A more careful look reveals that some extra conditions are still
required to make ut(x) a generating velocity for pt(x) and simple interpolation (as defined in (4.37)) is not
enough to guarantee this, not even with extra smoothness conditions, as required in Theorem 3. To see this,
consider
ψt(x0, x1) = (1 −2t)τ
+ x0 + (2t −1)τ
+ x1, where (s)+ = ReLU(s), τ > 2,
a C2([0, 1]) interpolant (in time) concentrating all probability mass at location 0 at time t = 0.5 for all x0, x1.
That is P(X 1
2 = 0) = 1. Therefore, assuming ut(x) indeed generates pt(x) its marginal at t = 1
2 is δ0 and since
a flow is both Markovian (as shown in (3.17)) and deterministic its marginal has to be a delta function for all
t > 0.5 leading to a contradiction since X1 = ψ1(X0, X1) ∼q, which is generally not a delta function. Albergo
and Vanden-Eijnden (2022) and Liu et al. (2022) provide some extra conditions that guarantee that ut(x)
indeed geenrates pt(x) but these are somewhat harder to verify compared to the conditions of Theorem 3.
Below we will show how to practically check the conditions of Theorem 3 to validate that particular paths of
interest are guaranteed to be generated by the respective marginal velocities.
Nevertheless, when ψt(x0, x1) is in addition a diffeomorphism in x0 for a fixed x1, and in x1 for a fixed x0,
the three constructions leads to the same marginal velocity, defined by (4.34), and same marginal probability
path pt, defined by Xt = ψt(X0, X1) = ψt(X0|X1) = ψt(X1|X0), see Figure 11.
24

4.7
Optimal Transport and linear conditional flow
We now ask: how to find a useful conditional flow ψt(x|x1)? One approach is to choose it as a minimizer of a
natural cost functional, ideally with some desirable properties. One popular example of such cost functional is
the dynamic Optimal Transport problem with quadratic cost (Villani et al., 2009; Villani, 2021; Peyré et al.,
2019), formalized as
(p⋆
t , u⋆
t ) = arg min
pt,ut
Z 1
0
Z
∥ut(x)∥2 pt(x)dxdt
(Kinetic Energy)
(4.41a)
s.t. p0 = p, p1 = q
(interpolation)
(4.41b)
d
dtpt + div(ptut) = 0.
(continuity equation)
(4.41c)
The (p⋆
t , u⋆
t ) above defines a flow (via equation (3.19)) with the form
ψ⋆
t (x) = tϕ(x) + (1 −t)x,
(4.42)
called the OT displacement interpolant (McCann, 1997), where ϕ : Rd →Rd is the Optimal Transport map.
The OT displacement interpolant also solves the Flow Matching Problem (4.1) by defining the random variable
Xt = ψ⋆
t (X0) ∼p⋆
t
when
X0 ∼p.
(4.43)
The Optimal Transport formulation promotes straight sample trajectories
Xt = ψ⋆
t (X0) = X0 + t(ϕ(X0) −X0),
with a constant velocity ϕ(X0) −X0, which are in general easier to sample with ODE solvers—in particular,
the target sample X1 is here perfectly solvable with a single step of the Euler Method (3.21).
We can now try to plug our marginal velocity formula (equation (4.34)) into the Optimal Transport problem
(4.41) and search for an optimal ψt(x|x1). While this seems like a challenge, we can instead find a bound for
the Kinetic Energy for which such a minimizer is readily found (Liu et al., 2022):
Z 1
0
EXt∼pt ∥ut(Xt)∥2 dt =
Z 1
0
EXt∼pt
E
h
˙ψt(X0|X1)
 Xt
i
2
dt
(4.44)
(i)
≤
Z 1
0
EXt∼ptE
 ˙ψt(X0|X1)

2  Xt

dt
(4.45)
(ii)
= E(X0,X1)∼π0,1
Z 1
0
 ˙ψt(X0|X1)

2
dt,
(4.46)
where in the (i) we used Jensen’s inequality, and in (ii) we used the tower property of conditional expectations
(see equation (3.11)) and switch integration of t and expectation. Now the integrand in (4.46) can be minimized
individually for each (X0, X1) — this leads to the following variational problem for γt = ψt(x|x1):
min
γ:[0,1]→Rd
Z 1
0
∥˙γt∥2 dt
(4.47a)
s.t.
γ0 = x, γ1 = x1.
(4.47b)
This problem can be solved using Euler-Lagrange equations (Gelfand et al., 2000), which in this case take the
form
d2
dt2 γt = 0. By incorporating the boundary conditions, we obtain the minimizer:
ψt(x|x1) = tx1 + (1 −t)x.
(4.48)
Note that although not constrained to be, this choice of ψt(x|x1) is a diffeomorphism in x for t ∈[0, 1) and
smooth in t, x, as required from conditional flows.
25

Several conclusions can be drawn:
1. The linear conditional flow minimizes a bound of the Kinetic Energy among all conditional flows.
2. In case the target q consists of a single data point q(x) = δx1(·) we have that the linear conditional flow
in (4.48) is the Optimal Transport (Lipman et al., 2022). Indeed, in this case Xt = ψt(X0|x1) ∼pt and
X0 = ψ−1(Xt|x1) is a function of Xt which makes E
h
˙ψt(X0|x1)
 Xt
i
= ˙ψt(X0|x1) and therefore (ii)
becomes an equality.
Theorem 5. If q = δx1, then the dynamic OT problem (4.41) has an analytic solution given by the
OT displacement interpolant in (4.48).
3. Plugging the linear conditional flow in (4.46) we get
Z 1
0
EXt∼pt ∥ut(Xt)∥2 dt ≤E(X0,X1)∼π0,1
Z 1
0
∥X1 −X0∥2 dt
(4.49)
showing that the Kinetic Energy of the marginal velocity ut(x) is not bigger than that of the original
coupling π0,1 (Liu et al., 2022).
The conditional flow in (4.48) is in particular affine and consequently motivates investigating the family of
affine conditional flows, discussed next.
4.8
Affine conditional flows
In the previous section we discovered the linear (Conditional-OT) flows as a minimizer to a bound of the
Kinetic Energy among all conditional flows. The linear conditional flow is a particular instance the wider
family of affine conditional flows, explored in this section.
ψt(x|x1) = αtx1 + σtx,
(4.50)
where αt, σt : [0, 1] →[0, 1] are smooth functions satisfying
α0 = 0 = σ1, α1 = 1 = σ0, and ˙αt, −˙σt > 0 for t ∈(0, 1).
(4.51)
We call the pair (αt, σt) a scheduler. The derivative condition above ensures that αt is strictly monotonically
increasing, while σt is strictly monotonically decreasing. The conditional flow (4.50) is a simple affine map in
x for each t ∈[0, 1), which satisfies the conditions (4.29). The associated marginal velocity field (4.34) is
ut(x) = E [ ˙αtX1 + ˙σtX0|Xt = x] .
(4.52)
By virtue of corollary 1, we can prove that, if using the independent coupling and a smooth and strictly positive
source density p with finite second moments—for instance, a Gaussian p = N(·|0, I)—then ut generates
a probability path pt interpolating p and q. We formally state this result, significant for Flow Matching
applications, as the following theorem.
Theorem 6. Assume that q has bounded support, p is C1(Rd) with strictly positive density with finite
second moments, and these two relate by the independent coupling π0,1(x0, x1) = p(x0)q(x1).
Let
pt(x) =
R
pt|1(x|x1)q(x1)dx1 be defined by equation (4.30), with ψt defined by equation (4.50). Then, the
marginal velocity (4.52) generates pt interpolating p and q.
Proof. We apply corollary 1. First, note that π0|1(·|x1) = p(·) is C1 and positive everywhere by assumption.
Second, ψt, defined in (4.50), satisfies (4.29). Third, we are left with checking (4.35):
Et,(X0,X1)
 ˙ψ(X0|X1)
 = Et,(X0,X1) ∥˙αtX1 + ˙σtX0∥
≤Et| ˙αt| EX1 ∥X1∥+ Et| ˙σt| EX0 ∥X0∥
= EX1 ∥X1∥+ EX0 ∥X0∥
< ∞,
26

where the last inequality follows from the fact that X1 ∼q has bounded support and X0 ∼p has bounded
second moments.
In this affine case, the CFM loss (4.32) takes the form
LCFM(θ) = Et,(X0,X1)∼π0,1D( ˙αtX1 + ˙σtX0, uθ
t(Xt)) .
(4.53)
Code 5: Examples of affine probability paths in the flow_matching library
1
from flow_matching.path import AffineProbPath, CondOTPath
2
from flow_matching.path.scheduler import (
3
CondOTScheduler, PolynomialConvexScheduler, LinearVPScheduler, CosineScheduler)
4
5
# Conditional Optimal Transport schedule with αt = t, σt = 1 −t
6
path = AffineProbPath(scheduler=CondOTScheduler())
7
path = CondOTPath()
# Shorthand for the affine path with the CondOTScheduler above
8
9
# Polynomial schedule with αt = tn, σt = 1 −tn
10
path = AffineProbPath(scheduler=PolynomialConvexScheduler(n=1.0))
11
12
# Linear variance preserving schedule with αt = t, σt =
√
1 −t2
13
path = AffineProbPath(scheduler=LinearVPScheduler())
14
15
# Cosine schedule with αt = sin(0.5tπ), σt = cos(0.5tπ)
16
path = AffineProbPath(scheduler=CosineScheduler())
4.8.1
Velocity parameterizations
In the affine case, the marginal velocity field ut admits multiple parametrizations, each of them learnable
using the Flow Matching losses introduced in section 4.5. To derive these parametrizations, use the equivalent
formulations of the affine paths
Xt = αtX1 + σtX0
⇔
X1 = Xt −σtX0
αt
⇔
X0 = Xt −αtX1
σt
,
(4.54)
in the marginal velocity formula (4.52), obtaining
ut(x) = ˙αtE [X1|Xt = x] + ˙σtE [X0|Xt = x]
(4.55)
= ˙σt
σt
x +

˙αt −αt
˙σt
σt

E [X1|Xt = x]
(4.56)
= ˙αt
αt
x +

˙σt −σt
˙αt
αt

E [X0|Xt = x],
(4.57)
where we have used the fact that E [Z|Z = z] = z. Then, denote the deterministic functions:
x1|t(x) = E [X1|Xt = x] as the x1-prediction (target),
(4.58)
x0|t(x) = E [X0|Xt = x] as the x0-prediction (source).
(4.59)
These provides two more opportunities to parameterize ut: via the x1-prediction x1|t (4.56) and via the
x0-prediction x0|t (4.57). Table 1 offers conversion formulas between the parameterizations. These param-
eterizations can also be learned using a Conditional Matching loss, similar to (4.23). In particular, any
function
gt(x) := E [ft(X0, X1)|Xt = x] ,
(4.60)
where ft(X0, X1) is a RV defined as a time-dependent function of X0 and X1, can be learned by minimizing a
Matching loss of the form
LM(θ) = Et,Xt∼ptD
 gt(Xt), gθ
t (Xt)

.
(4.61)
27

This loss has the same gradients as the Conditional Matching loss
LCM(θ) = Et,(X0,X1)∼π0,1D
 ft(X0, X1), gθ
t (Xt)

.
(4.62)
To learn x1|t, the Conditional Matching loss employs ft(x0, x1) = x1, and similarly for x0|t. This procedure is
justified by theorem 7, which is an immediate result from proposition 1 when letting X = Xt, Y = ft(X0, X1),
and integrating with respect to t ∼U[0, 1].
Theorem 7. The gradients of the Matching loss and the Conditional Matching loss coincide for arbitrary
functions ft(X0, X1) of X0, X1:
∇θLM(θ) = ∇θLCM(θ).
(4.63)
In particular, the minimizer of the Conditional Matching loss is the conditional expectation
gθ
t (x) = E [ft(X0, X1)|Xt = x] .
(4.64)
Code 6 shows how to train with x1-prediction using the flow_matching library.
Code 6: Training an X1-prediction model using the Conditional Matching (CM) objective
1
import torch
2
from flow_matching.path import AffineProbPath
3
from flow_matching.solver import ODESolver
4
from flow_matching.utils import ModelWrapper
5
6
path: AffineProbPath = ...
7
denoiser_model: torch.nn.Module = ...
# Initialize the denoiser
8
optimizer = torch.optim.Adam(velocity_model.parameters())
9
10
for x_0, x_1 in dataloader:
# Samples from π0,1 of shape [batch_size, *data_dim]
11
t = torch.rand(batch_size)
# Randomize time t ∼U[0, 1]
12
sample = path.sample(t=t, x_0=x_0, x_1=x_1)
# Sample the conditional path
13
cm_loss = torch.pow(model(sample.x_t, t) - sample.x_1, 2).mean()
# CM loss
14
optimizer.zero_grad()
15
cm_loss.backward()
16
optimizer.step()
17
18
# Convert from denoiser to velocity prediction
19
class VelocityModel(ModelWrapper):
20
def __init__(self, denoiser: nn.Module, path: AffineProbPath):
21
super().__init__(model=denoiser)
22
self.path=path
23
24
def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
25
x_1_prediction = super().forward(x, t, **extras)
26
return self.path.target_to_velocity(x_1=x_1_prediction, x_t=x, t=t)
27
28
# Sample X1
29
velocity_model = VelocityModel(denoiser=denoiser_model, path=path)
30
x_0 = torch.randn(batch_size, *data_dim)
# Specify the initial condition
31
solver = ODESolver(velocity_model=velocity_model)
32
num_steps = 100
33
x_1 = solver.sample(x_init=x_0, method='midpoint', step_size=1.0 / num_steps)
Singularities in the velocity parameterizations.
Seemingly, the coefficients of (4.56) would blow up as t →1,
and similarly for (4.57) as t →0. If E [X1|X0 = x] and E [X0|X1 = x] exist, which is the case for p(x) > 0 and
q(x) > 0, these are not essential singularities in theory, meaning that the singularities in x1|t and x0|t would
cancel with the singularities of the coefficients of the parameterization. However, these singularities could be
still problematic in practice when the learnable xθ
1|t and xθ
0|t are by construction continuous and therefore
do not perfectly regress their targets x1|t and x0|t. To understand how to fix these potential issues, recall
(4.55) and consider u0(x) = ˙α0E [X1|X0 = x] + ˙σ0x as t →0, and u1(x) = ˙α1x + ˙σ1E [X0|X1 = x] as t →1.
These can be computed in many cases of interest. Returning to our example π0,1(x0, x1) = N(x0|0, I)q(x1)
28

and assuming EX1X1 = 0, it follows that u0(x) = ˙σ0x and u1(x) = ˙α1x. These expressions can be used to fix
singularities when converting from x1|t and x0|t to ut(x) as t →1 or t →0, respectively.
4.8.2
Post-training velocity scheduler change
Affine conditional flows admit a closed-form transformation from a marginal velocity field ut(x), based on a
scheduler (αt, σt) and an arbitrary data coupling π0,1, to a marginal velocity field ¯ur(x), based on a different
scheduler (¯αr, ¯σr) and the same data coupling π0,1. Such a transformation is useful to adapt a trained velocity
field to a different scheduler, potentially improving sample efficiency and quality generation (Karras et al.,
2022; Shaul et al., 2023b; Pokle et al., 2023). To proceed, define the scale-time (ST) transformation (sr, tr)
between the two conditional flows:
¯ψr(x0|x1) = srψtr(x0|x1),
(4.65)
where ψt(x0|x1) = αtx1 + σtx0, ¯ψr(x0|x1) = ¯αrx1 + ¯σrx0, and s, t : [0, 1] →R≥0 are time-scale reparametriza-
tions. Solving (4.65) yields
tr = ρ−1(¯ρ(r))
sr = ¯σr/σtr,
(4.66)
where we define the signal-to-noise ratio by
ρ(t) = αt
σt
¯ρ(t) = ¯αt
¯σt
,
(4.67)
assumed to be an invertible function. The marginal velocity ¯ur(x) for the new scheduler (¯αr, ¯σr) follows the
expression
¯ur(x) = E
h ˙¯Xr
 ¯Xr = x
i
(4.65)
=
E
h
˙srXtr + sr ˙Xtr ˙tr
srXtr = x
i
= ˙srE

Xtr
Xtr = x
sr

+ sr ˙trE

˙Xtr
Xtr = x
sr

= ˙sr
sr
x + sr ˙trutr
 x
sr

,
where as before ¯Xr = ¯ψr(X0|X1) and Xt = ψt(X0|X1). This last term can be used to change a schedular
post-training. Code 7 shows how to change the scheduler of a velocity field trained with a variance preserving
schedule to the conditional Optimal Transport schedule using the flow_matching library.
Equivalence of schedulers.
One additional important consequence of the above formula is that all schedulers
theoretically lead to the same sampling at time t = 1 (Shaul et al., 2023a). That is,
¯ψ1(x0) = ψ1(x0), for all x0 ∈Rd.
(4.68)
To see that, denote ¯ψr(x) the flow defined by ¯ut(x), and differentiate ˜ψr(x) := srψtr(x) w.r.t. r and note that
it also satisfies
d
dt
˜ψr(x) = ¯ur( ˜ψr(x)).
(4.69)
Therefore, from uniqueness of ODE solutions we have that ¯ψr(x) = ˜ψr(x) = srψtr(x). Now, to avoid dealing
with infinite signal-to-noise ratio assume the schedulers satisfy σ1 = ϵ = ¯σ1 for arbitrary ϵ > 0 (in addition to
(4.51)), then for r = 1 we have t1 = 1 and s1 = 1 and therefore equation (4.68) holds.
29

Code 7: Post-training scheduler change
1
import torch
2
from flow_matching.path import AffineProbPath
3
from flow_matching.path.scheduler import ScheduleTransformedModel, CondOTScheduler, VPScheduler
4
from flow_matching.solver import ODESolver
5
from flow_matching.utils import ModelWrapper
6
7
training_scheduler = VPScheduler()
# Variance preserving schedule
8
path = AffineProbPath(scheduler=training_scheduler)
9
velocity_model: ModelWrapper = ...
# Train a velocity model with the variance preserving schedule
10
11
# Change the scheduler from variance preserving to conditional OT schedule
12
sampling_scheduler = CondOTScheduler()
13
transformed_model = ScheduleTransformedModel(
14
velocity_model=velocity_model,
15
original_scheduler=training_scheduler,
16
new_scheduler=sampling_scheduler
17
)
18
19
# Sample the transformed model with the conditional OT schedule
20
solver = ODESolver(velocity_model=transformed_model)
21
x_0 = torch.randn(batch_size, *data_dim)
# Specify the initial condition
22
solver = ODESolver(velocity_model=velocity_model)
23
num_steps = 100
24
x_1 = solver.sample(x_init=x_0, method='midpoint', step_size=1.0 / num_steps)
4.8.3
Gaussian paths
At the time of writing, the most popular class of affine probability paths is instantiated by the independent
coupling π0,1(x0, x1) = p(x0)q(x1) and a Gaussian source distribution p(x) = N(x|0, σ2I). Because Gaussians
are invariant to affine transformations, the resulting conditional probability paths take form
pt|1(x|x1) = N(x|αtx1, σ2
t I).
(4.70)
This case subsumes probability paths generated by standard diffusion models (although in diffusion the
generation is stochastic and follows an SDE, it has the same marginal probabilities). Two examples are
the Variance Preserving (VP) and Variance Exploding (VE) paths (Song et al., 2021), defined by choosing the
following schedulers:
αt ≡1, σ0 ≫1, σ1 = 0;
(VP)
αt = e−1
2 βt, σt =
p
1 −e−βt, β0 ≫1, β1 = 0.
(VE)
In the previous equations, “≫1” requires a sufficiently large scalar such that p0(x) =
R
p0|1(x|x1)q(x1)dx1
is close to a known Gaussian distribution for t = 0—that is, the Gaussian N(·|0, σ2
0I) for VE, and N(·|0, I)
for VP. Note that in both cases, pt(x) does not exactly reproduce p at t = 0, in contrast to the FM paths in
(4.51).
One useful quantity admitting a simple form in the Gaussian case is the score, defined as the gradient of the
log probability. Specifically, the score of the conditional path in (4.70) follows the expression
∇log pt|1(x|x1) = −1
σ2
t
(x −αtx1) .
(4.71)
30

B
A
velocity
x1-prediction
x0-prediction
score
velocity
0, 1
˙σt
σt , ˙αtσt−˙σtαt
σt
˙αt
αt , ˙σtαt−˙αtσt
αt
˙αt
αt , −˙σtσtαt−˙αtσ2
t
αt
x1-prediction
0, 1
1
αt , −σt
αt
1
αt , σ2
t
αt
x0-prediction
0, 1
0, −σt
score
0, 1
Table 1 Conversion between different model parameterizations: (at, bt) corresponds to f B
t (x) = atx + btf A
t (x). The
colors indicate the tranformation is relevant for all paths , Affine paths and Gaussian paths . The lower diagonal
is computed from the upper diagonal using the inverse transformation: f A
t (x) =
1
bt
 −atx + f B
t (x)
 which can be
expressed as the pair −at
bt , 1
bt . Note some of these conversions have singularities, as discussed at the end of Section
4.8.1.
The score of the corresponding marginal probability path (4.4) is
∇log pt(x) =
Z ∇pt|1(x|x1)q(x1)
pt(x)
dx1
(4.72)
=
Z
∇log pt|1(x|x1)pt|1(x|x1)q(x1)
pt(x)
dx1
(4.73)
= E

∇log pt|1(Xt|X1) | Xt = x

(4.74)
(4.71)
=
E

−1
σ2
t
(Xt −αtX1) | Xt = x

(4.75)
(4.54)
=
E

−1
σt
X0
 Xt = x

(4.76)
(4.59)
=
−1
σt
x0|t(x),
(4.77)
where we borrow the notation x0|t from (4.59). The literature on diffusion refers to x0-prediction (x0|t) as
noise-prediction, or ϵ-prediction. The formula above shows that the score is proportional to the x0-prediction,
and provides a conversion rule—for the Gaussian path case—from score to other parametrizations, as shown
in table 1.
Kinetic optimality of marginal velocity.
A consequence of the conversion formulas developed above (table 1)
is that the marginal velocity for Gaussian paths can be written in the form
ut(x) = ˙αt
αt
x −˙σtσtαt −˙αtσ2
t
αt
∇log pt(x)
(4.78)
= ∇
 ˙αt
2αt
∥x∥2 −˙σtσtαt −˙αtσ2
t
αt
log pt(x)

(4.79)
that shows ut(x) is a gradient and therefore Kinetic Optimal for the fixed marginalized Gaussian probability
path pt(x) defined by pt|1(x|x1) = N(x|αtx1, σ2
t I) (see e.g., Villani (2021) Section 8.1.2, or Neklyudov et al.
(2023) Theorem 2.1).
4.9
Data couplings
In developing the Flow Matching training algorithm, we have assumed we can draw samples (X0, X1) ∼
π0,1(X0, X1) from some coupling π0,1 (x0, x1) of the source p and target q distributions.
For example,
independent samples π0,1(x0, x1) = p(x0)q(x1), the simplest coupling preserving the marginal distributions p
and q, or paired samples (X0, X1) ∼π0,1 provided as part of the dataset. This section explores two examples
of concrete couplings that can be used to train Flow Matching models.
31

4.9.1
Paired data
Dependent couplings arise naturally in learning tasks on paired data. Consider, for instance, the task of image
in-painting, where q is a distribution of natural images, and p is the distribution of those same images with a
square region masked-out. Rather than transforming noise into data, our goal here is to learn a mapping from
masked-out images x0 to their filled counterparts x1. As this is an ill-defined problem—many filled images x1
are compatible with each masked-out image x0—solving this task can casted as learning to sample from the
unknown, data-dependent coupling π1|0(x1|x0).
Based on these insights, Liu et al. (2023); Albergo et al. (2024) propose learning a bridge or flow model with
data-dependent couplings, a simple modification enabling a new regime of applications. While the object
of interest π1|0(x1|x0) is unavailable to sample, it is often the case that one can sample from the reverse
dependency, π0|1(x0|x1). Returning to the example of image in-painting, it is easy to mask out a filled image
X1 ∼q (target sample) to produce a source sample X0 ∼p. To this end, specify
π0,1(x0, x1) = π0|1(x0|x1)q(x1).
(4.80)
Thus, we can obtain a pair (X0, X1) by (i) drawing X1 ∼q, and (ii) applying a predefined randomized
transformation to obtain X0 from X1. To satisfy the conditions of corollary 1 (making sure the source
is a density) and to encourage diversity, we add noise when sampling from π0|1(x0|x1). (Liu et al., 2023;
Albergo et al., 2024) demonstrated the capability of this approach on various applications, such as image
super-resolution, in-painting, and de-blurring, outperforming methods based on guided diffusion (Saharia
et al., 2022).
4.9.2
Multisample couplings
As discussed in section 4.7, straight probability paths yield ODEs simulations with smaller errors. Therefore,
it is natural ask: how could we change the training algorithm, so the learned velocity field induces straight(er)
trajectories?
As hinted above, straight trajectories are related to the Optimal Transport (OT) problem. Specifically,
consider a convex cost functional c : Rd →R≥0 and the conditional OT flow ψt(x0|x1) = tx1 + (1 −t)x0.
Then, the transport cost of the coupling admits an upper bound on the marginal transport cost (Liu et al.,
2022; Pooladian et al., 2023), that is:
E [c(ψ1(X0) −X0)] ≤E [c(X1 −X0)] ,
(4.81)
where the case of c(x) = ∥x∥2 can be understood from the bound in (4.49) after plugging the OT solution in
the l.h.s. that satisfies ut(Xt) = ut(ψt(x)) = ϕ(x) −x, where ϕ is the OT map. Therefore, one could construct
low-cost marginal transport maps by reducing the coupling cost. To this end, Pooladian et al. (2023) propose
multisample couplings, a process to implicitly construct non-trivial joints π0,1(x0, x1) introducing dependencies
between source and target distributions:
1. Sample X(i)
0
∼p and X(i)
1
∼q, i ∈[k] independently.
2. Construct πk ∈Bk by πk := arg minπ∈Bk Eπ
h
c(X(i)
0
−X(j)
1 )
i
.
3. Sample a pair (X(i)
0 , X(j)
0 ) uniformly at random from (X(i)
0 , X(j)
1 ) for which πk(i, j) = 1.
where Bk is the polytope of k × k doubly stochastic matrices.
The process above implicitly defines a joint distribution πk
0,1(x0, x1) by means of sampling. This implicit
joint preserves the marginals and obeys an optimality constraint (step 2) (Pooladian et al., 2023). For
k = 1, the method reduces to independent couplings. For k > 1, Pooladian et al. (2023) show that the
transport cost is reduced compared to independent couplings, that is, E(x0,x1)∼πk
0,1(x0,x1) [c(x1 −x0)] ≤
EX0∼p,X1∼q [c(X1 −X0)]. Furthermore, for the quadratic cost function, multisample couplings approach the
Optimal Transport cost and induces straight trajectories as k →∞(Pooladian et al., 2023; Tong et al., 2023).
32

4.10
Conditional generation and guidance
We now consider training a generative model under a guiding signal to further control the produced samples.
This technique has proved valuable in numerous practical applications, such as image-to-image translation
(Saharia et al., 2022) and text-to-image generation (Nichol et al., 2022; Esser et al., 2024). In this subsection,
we assume access to labeled target samples (x1, y), where y ∈Y ⊆Rk is a label or guidance variable.
4.10.1
Conditional models
One natural way to train a generative model under guidance is to learn to sample from the conditional
distribution q(x1|y), as demonstrated by both diffusion and FM models (Zheng et al., 2023). Following the
FM blueprint in figure 2, consider samples from the conditional target distribution q(x1|y) and prescribe a
simple—typically but not necessarily Gaussian—source distribution p. Next, design a guided probability path
as the aggregation of conditional probability paths:
pt|Y (x|y) =
Z
pt|1(x|x1)q(x1|y)dx1.
(4.82)
where we assume pt,1|Y (x, x1|y) = pt|1(x|x1)q(x1|y), meaning that the conditional path does not depend on
Y . The resulting guided probability path is conditioned on the guidance variable Y ∼pY , and satisfies the
marginal endpoints
p0|Y (·|y) = p(·),
p1|Y (·|y) = q(·|y).
(4.83)
The guided velocity field takes form
ut(x|y) =
Z
ut(x|x1)p1|t,Y (x1|x, y)dx1,
(4.84)
where, by Bayes’ Rule, it follows
p1|t,Y (x1|x, y) = pt|1(x|x1)q(x1|y)
pt|Y (x|y)
.
(4.85)
To show that ut(x|y) generates pt|Y (x|y), plug (4.82) and (4.84) into (4.14), and realize that the FM/CFM
losses remain unchanged for the guided case, and enable the same steps appearing in the proof of theorem 4.
In practice, we train a single neural network uθ
t : Rd × Rk →Rd to model the guided marginal velocity field
for all values of y. Then, the guided version of the CFM loss (4.32) follows the expression
LCFM(θ) = Et,(X0,X1,Y )∼π0,1,Y D

˙ψt(X0|X1), uθ
t (Xt|Y )

.
(4.86)
In practice, the literature in diffusion models shows that guidance is most effective in applications where
a large amount of target samples X1 share the same guiding signal Y , such as in class guidance (Nichol
and Dhariwal, 2021). However, guiding is more challenging in settings where the guidance variable Y is
non-repeating and complex, such as image captions.
4.10.2
Classifier guidance and classifier-free guidance
For flows trained with Gaussian paths, classifier guidance (Song et al., 2021; Dhariwal and Nichol, 2021) and
classifier-free guidance (Ho and Salimans, 2021) can be applied utilizing the transformations between velocity
fields and score functions for conditional distributions shown in table 1 (Zheng et al., 2023):
ut(x|y) = atx + bt∇log pt|Y (x|y).
(4.87)
Using Bayes’ rule over the guided probability path yields
pt|Y (x|y) = pY |t(y|x)pt(x)
pY (y)
.
(4.88)
33

Taking logarithms and gradients with respect to x, ∇= ∇x, we arrive at the fundamental relation between
the scores of the probability path pt(x) and its guided counterpart pt|Y (x|y):
conditional score
z
}|
{
∇log pt|Y (x|y) = ∇
classifier
z
}|
{
log pY |t(y|x) +
unconditional score
z
}|
{
∇log pt(x) .
(4.89)
Namely, the two are related by means of the score of a classifier model pY |t(y|x) attempting to predict the
guidance variable y given a sample x.
Based on this relation, Song et al. (2021) propose classifierguidance, that is sampling from the conditional model
q(x1|y) by guiding an unconditional model (parameterized with ∇log pt(x)) with a time-dependent classifier
(predicting the guiding variable y given x ∼pt(x)). The corresponding velocity field then translates to:
˜uθ,ϕ
t
(x|y) = atx + bt

∇log pϕ
Y |t(y|x) + ∇log pθ
t (x)

= uθ
t(x) + bt∇log pϕ
Y |t(y|x),
(4.90)
where uθ
t (x) is a velocity field trained on the unconditional target q(x), and log pϕ
Y |t(y|x) is a time-dependent
classifier with parameters ϕ ∈Rm. Dhariwal and Nichol (2021) show that this approach outperforms the
conditional model from section 4.10.1 for both class- and text-conditioning (Nichol et al., 2022). In practice,
because the classifier and the unconditional score are learned separately, it is often necessary to calibrate the
classifier guidance as
˜uθ,ϕ
t
(x|y) = uθ
t(x) + btw∇log pϕ
Y |t(y|x),
(4.91)
where w ∈R is the classifier scale, typically chosen to be w > 1 (Dhariwal and Nichol, 2021).
In a later work, (Ho and Salimans, 2021) propose a pure generative approach called classifier-free guidance.
By simply re-arranging (4.89), we obtain
∇
classifier
z
}|
{
log pY |t(y|x) =
conditional score
z
}|
{
∇log pt|Y (x|y)
−
unconditional score
z
}|
{
∇log pt(x),
(4.92)
revealing that the score of the classifier can be implicitly approximated by the difference between the scores of
the vanilla and guided probability paths. Then, the authors propose to learn the conditional and unconditional
scores simultaneously using the same model. In terms of velocities, Zheng et al. (2023) show one can also plug
4.92 into 4.91 and use the conversion from scores to velocities as in table 1 to get:
˜uθ
t (x|y) = (1 −w)uθ
t (x|∅) + wuθ
t (x|y),
(4.93)
where w is once again the guidance calibration scale. Now, only a single model is trained, uθ
t(x|y), where
y ∈{Y, ∅}, ∅is a place-holder value denoting the null-condition, and uθ
t(x|∅) is the velocity field generating
the unconditional probability path pt(x). The resulting loss reads:
LCFM(θ) = Et,ξ,(X0,X1,Y )∼π0,1,Y
h
D

˙ψt(X0|X1), uθ
t(Xt|(1 −ξ) · Y + ξ · ∅)
i
,
(4.94)
where ξ ∼Bernoulli(puncond), and puncond is the probability of drawing the null condition ∅during training.
The exact distribution which CFG samples from is unknown, with some works proposing different intuitive
or theoretical justifications for CFG sampling (Dieleman, 2022; Guo et al., 2024; Chidambaram et al., 2024;
Bradley and Nakkiran, 2024). Despite this, at the time of writing, CFG is the most popular approach to
training a conditional model. Esser et al. (2024); Polyak et al. (2024) show the application of classifier-free
guidance to train large-scale guided FM models.
34

5
Non-Euclidean Flow Matching
This section extends Flow Matching from Euclidean spaces Rd to general Riemannian manifolds M. Informally,
Riemannian manifolds are spaces behaving locally like Euclidean spaces, and are equipped with a generalized
notion of distances and angles. Riemannian manifolds are useful to model various types of data. For example,
probabilities of natural phenomena on Earth can be modeled on the sphere Mathieu and Nickel (2020),
and protein backbones are often parameterized inn terms of matrix Lie groups Jumper et al. (2021). The
extension of flows to Riemannian manifolds is due to Mathieu and Nickel (2020); Lou et al. (2020). However,
their original training algorithms required expensive ODE simulations. Following Chen and Lipman (2024),
the Flow Matching solutions in this section provide a scalable, simulation-free training algorithm to learn
generative models on Riemannian manifolds.
5.1
Riemannian manifolds
We consider complete connected, smooth Riemannian manifolds M with a metric g. The tangent space at
point x ∈M, a vector space containing all tangent vectors to M at x, is denoted with TxM. The Riemannian
metric g defines an inner product over TxM denoted by ⟨u, v⟩g, for u, v ∈TxM. Let TM = ∪x∈M {x}×TxM
be the tangent bundle that collects all the tangent planes of the manifold. In the following, vector fields
defined on tangent spaces are important objects to build flows on manifolds with velocity fields. We denote by
U = {ut} the space of time-dependent smooth vector fields (VFs) ut : [0, 1] × M →TM, where ut(x) ∈TxM
for all x ∈M. Also, divg(ut) is the Riemannian divergence with respect to the spatial (x) argument. Finally,
we denote by dvolx the volume element over M, and integration of a function f : M →R over M is denoted
R
f(x)dvolx.
5.2
Probabilities, flows and velocities on manifolds
Probability density functions over a manifold M are continuous non-negative functions p : M →R≥0
integrating to 1, namely
R
M p(x)dvolx = 1. We define a probability path in time pt as a time-dependent curve
in probability space P, namely pt : [0, 1] →P. A time-dependent flow, ψ : [0, 1] × M →M, similar to the
Euclidean space, defines a global diffeomorphism on M for every t.
Remarkably, constructing flow-based models via velocity fields naturally applies to general Riemannian
manifolds. Formally, and rephrasing Proposition 1 from Mathieu and Nickel (2020):
Theorem 8 (Flow local existence and uniqueness). Let M a smooth complete manifold and a velocity
field ut ∈U. If u is C∞([0, 1] × M, TM) (in particular, locally Lipschitz), then the ODE in (3.19) has a
unique solution which is a C∞(Ω, M) diffeomorphism ψt(x) defined over the open set Ω⊃{0} × M.
Similar to theorem 1, flow ODEs generally only define a local diffeomorphism on the manifold, meaning that
ψt(x) may be defined on a maximal interval in time [0, tx]) for different values of x ∈M. Similar to the
Euclidean case we will work with the semi-open time interval t ∈[0, 1) to allow q to have compact support
(for which ut is not everywhere defined). To ensure existence for the desired time interval, [0, 1), we add
the integrability constraint (see theorem 2) and rely on the Mass Conservation theorem once again. For a
Riemannian manifold with metric g, the Riemannian continuity equation reads
d
dtpt(x) + divg(ptut)(x) = 0,
(5.1)
and the corresponding Manifold Mass Conservation theorem (Villani et al., 2009) is stated as follows.
Theorem 9 (Manifold Mass Conservation). Let pt be a probability path and ut ∈U a locally Lipchitz
integrable vector field over a Riemannian manifold M with metric g. Then the following are equivalent
1. The Continuity Equation (5.1) holds for t ∈[0, 1).
2. ut generates pt in the sense of 3.24.
35

In the previous result, by integrable u we mean
Z 1
0
Z
M
∥ut(x)∥pt(x)dvolxdt < ∞.
(5.2)
Note that the assumptions of theorem 9 yield a global diffeomorphism on M, giving rise to the Riemannian
instantaneous change of variables formula:
d
dt log pt(ψt(x)) = −divg(ut)(ψt(x)).
(5.3)
Finally, we say that ut generates pt from p if
Xt = ψt(X0) ∼pt for X0 ∼p.
(5.4)
Having positioned flows as valid generative models on manifolds, it stands to reason that the FM principles
can be transferred to this domain as well. In the Riemannian version of FM we aim to find a velocity field
uθ
t ∈U generating a target probability path pt : [0, 1] →P with marginal constraints p0 = p and p1 = q, where
p, q denote the source and target distributions over the manifold M. As the velocity field lies on the tangent
spaces of the manifold, the Riemannian Flow Matching loss compares velocities using a Bregman divergence
defined over the individual tangent planes of the manifold,
LRFM(θ) = Et,Xt∼ptDXt
 ut(Xt), uθ
t (Xt)

.
(5.5)
In the equation above, the expectation is now an integral over the manifold, that is E[f(X)] =
R
M f(x)pX(x)dvolx,
for a smooth function f : M →M and a random variable X ∼pX. The Bregman divergences, Dx, x ∈M,
are potentially defined with the Riemannian inner product and a strictly convex function assigned to each
tangent space Φx : TxM →TxM, that is, Dx(u, v) := Φx(u) −[Φx(v) + ⟨u −v, ∇vΦx(v)⟩g]. For example,
choosing the Riemannian metric Φx = ∥·∥2
g then Dx(u, v) = ∥u −v∥2
g for u, v ∈TxM.
5.3
Probability paths on manifolds
Marginal probability paths are built as in the Euclidean case (4.4):
pt(x) =
Z
M
pt(x|x1)q(x1)dvolx1,
(5.6)
where pt|1(x|x1) is the conditional probability path defined on the manifold. We also require the boundary
constraints
p0 = p,
p1 = q.
(5.7)
For instance, these constraints can be implemented by requiring the conditional path pt|1(x|x1) to satisfy
p0|1(x|x1) = π0|1(x|x1), and p1|1(x|x1) = δx1(x),
(5.8)
where π0|1 is the conditional coupling, π0|1(x0|x1) = π0,1(x0, x1)/q(x1).
5.4
The Marginalization Trick for manifolds
The Marginalization Trick for the marginal velocity field (theorem 3) readily applies to the Riemannian case.
Consider the conditional velocity field ut(x|x1) ∈U such that
ut(·|x1) generates pt|1(·|x1).
(5.9)
Then, the marginal velocity field ut(x) is given by the following averaging of the conditional velocities,
ut(x) =
Z
M
ut(x|x1)p1|t(x1|x)dvolx1,
(5.10)
36

where, by Bayes’ Rule for PDFs, we obtain
p1|t(x1|x) = pt|1(x|x1)q(x1)
pt(x)
,
(5.11)
which is defined for all x ∈M for which pt(x) > 0.
The Marginalization Trick (theorem 3) for the Riemannian case requires adjusting assumption 1 as fol-
lows:
Assumption 2. pt|1(x|x1) is C∞([0, 1) × M) and ut(x|x1) is C∞([0, 1) × M, M) as function of (t, x).
Furthermore, we assume either q has bounded support, i.e., q(x1) = 0 outside some bounded set or M is
compact; and pt(x) > 0 for all x ∈M and t ∈[0, 1).
We are now ready to state the Manifold Marginalization Trick theorem:
Theorem 10 (Manifold Marginalization Trick). Under Assumption 2, if ut(x|x1) is conditionally
integrable and generates the conditional probability path pt(·|x1) then the marginal velocity field ut(·)
generates the marginal probability path pt(·).
By conditionally integrable, we mean a conditioned version of the integrability condition from the Mass
Conservation Theorem (5.2):
Z 1
0
Z
M
Z
M
∥ut(x|x1)∥g pt|1(x|x1)q(x1)dvolx1dvolxdt < ∞
(5.12)
The proof of theorem 10 is repeating the arguments of theorem 3 and is given in appendix A.2.
5.5
Riemannian Flow Matching loss
The Riemannian Conditional Flow Matching (RCFM) loss reads
LRCFM(θ) = Et,X1,Xt∼pt|1(·|X1)DXt
 ut(Xt|X1), uθ
t (Xt)

.
(5.13)
Once again, we have the equivalence:
Theorem 11. The gradients of the Riemannian Flow Matching loss and the Riemannian Conditional
Flow Matching loss coincide:
∇θLRFM(θ) = ∇θLRCFM(θ).
(5.14)
The above theorem can be proved using proposition 1 with X = Xt, Y = ut(Xt|X1), gθ(x) = uθ
t(x), and
integrating w.r.t. t ∈[0, 1].
5.6
Conditional flows through premetrics
Having established how to learn a flow model with the RCFM loss, we are left with specifying the conditional
probability path and its generating velocity field. Similar to section 4.6, we begin by stating the requirements
for the corresponding conditional flow ψ : [0, 1) × M × M →M, such that pt|1(·|x1) satisfies the boundary
conditions (5.8). The conditional flow model is
Xt|1 = ψt(X0|x1),
where X0 ∼π0|1(·|x1),
(5.15)
where the conditional flow is
ψt(x|x1) =
(
x
t = 0
x1
t = 1 ,
is smooth in t, x and diffeomorphism in x on M.
(5.16)
Our analysis in Euclidean space focused on affine conditional flows, as these served as a rich class of easily
computable (simulation-free) conditional flows. Unfortunately, combinations αtx1 + σtx0 for αt + σt ̸= 1 are
37

not naturally defined on manifolds. The manifold analog for the case αt + σt = 1 would be using geodesic
interpolation. Indeed, Chen and Lipman (2024) proposed building conditional flows by moving along geodesic
curves, in particular, generalizing the conditional OT paths moving along straight lines in Euclidean space
(see theorem 5). Geodesics represent the shortest paths between two points on a manifold, reducing to straight
lines in Euclidean spaces. For manifolds, we define the geodesic conditional flow as
ψt(x0|x1) = expx0(κ(t) logx0(x1)),
t ∈[0, 1],
(5.17)
where κ(t) : [0, 1] →[0, 1] is a monotonically increasing scheduler satisfying κ(0) = 0 and κ(1) = 1, making
sure all x0 are pushed to x1 at t = 1. The exponential map, evaluated at x ∈M, expx : TxM →M,
v 7→expx(v), returns the endpoint at time t = 1 of the unique geodesic starting at x with initial speed v. The
logarithmic map logx : M →TxM, y 7→logx(y), is the inverse of the exponential map. In Euclidean space,
the exponential map is simply vector addition, and the logarithmic map is vector subtraction. Now, if we plug
these in (5.17), we get ψt(x0|x1) = x0 + κ(t)(x1 −x0), and by choosing κ(t) = t we recover the conditional
OT flow.
For simple manifolds with closed-form exponential and logarithmic maps, this construction allows a simulation-
free recipe for training flows on manifolds, an arguably clear advantage compared to diffusion models approaches
built on manifolds (De Bortoli et al., 2022; Huang et al., 2022b; Lou et al., 2023). In particular, manifold
diffusion models require in-training simulation to sample from pt, and have to resort to approximations of the
score function on the manifold.
Nevertheless, while building geodesic conditional flows is a natural construction, geodesics may be hard
to compute for general manifolds that do not have closed-form exponential and logarithmic maps and/or
introduce undesired bias such as concentrating probability at boundary points. To overcome the difficulty
in computing geodesics and/or inject a desired implicit bias, one may seek an alternative notion of smooth
distance function, d(·, ·) : M × M →R≥0, and require that the conditional flow satisfies
d(ψt(x0|x1), x1) = ¯κ(t)d(x0, x1),
(5.18)
where ¯κ(t) = 1 −κ(t). This will assure that the conditional flow concentrates all the probability at x1 at time
t = 1 if the following conditions hold:
1. Non-negative: d(x, y) ≥0 for all x, y ∈M.
2. Positive: d(x, y) = 0 if and only if x = y.
3. Non-degenerate: ∇d(x, y) ̸= 0 if and only if x ̸= y.
Chen and Lipman (2024) showed that the minimal norm conditional velocity field corresponding to a flow
that satisfies (5.18) has the form:
ut(x|x1) = d log ¯κ(t)
dt
d(x, x1)
∇d(x, x1)
∥∇d(x, x1)∥2
g
,
(5.19)
Figure 12 Conditional flows on the manifold M.
where the non-degeneracy requirement of the premetric en-
sures that the velocity field has no discontinuities, since
ut(x|x1) ∝1/∥∇d(x, x1)∥g.
In particular, note that the
geodesic conditional flow in (5.17) satisfies (5.18) for the
choice d = dg, where dg is the geodesic distance. An ex-
ample of a choice of alternative premetrics is using spectral
distances on general geometries (Chen and Lipman, 2024),
where the conditional velocity offers a way to sample from
pt(x|x1) by simulation. Importantly, although conditional
flows with premetrics require in-training simulation—like dif-
fusion models on manifolds—the velocity field can still be
accurately recovered compared to approximations of the score
function.
38

Another issue, is that both conditional flows defined via geodesic interpolation and premetric can suffer from
singularities, e.g., for compact manifolds. For example on the 2-sphere the geodesic function d(x, x1) is not
differentiable at the antipodal point x = −x1. Furthermore, any smooth function such as x 7→d(x, x1) will
showcase at-least two critical points (maximum and minimum) where the velocity in (5.19) is not-defined.
However, the set of such problematic points is generally very small (in fact of zero volume usually). Therefore,
this issue does not cause problems in practice, at-least in use cases we are aware of.
In any case, to deal with this issue, we can include an augmented scheduler in the geodesic conditional flow.
That is, use ¯κ(t, x, x1), that depends also on x, x1 to make (5.17) globally smooth. To deal with the zero
gradient issue of the premetric conditional flow we can relax the non-degeneracy requirement as follows:
3. Non-degenerate (relaxed): The volume of the set Ay = {x ∈M | ∇d(x, y) = 0 and x ̸= y} is 0 for all
y ∈M.
Code 8: Training with geodesic flows on a Sphere using the CFM objective
1
import torch
2
from flow_matching.path import GeodesicProbPath, PathSample
3
from flow_matching.path.scheduler import CondOTScheduler
4
from flow_matching.utils.manifolds import Sphere
5
6
model = ...
# Define a trainable velocity model
7
optimizer = torch.optim.Adam(model.parameters())
8
loss_fn = torch.nn.MSELoss()
# Any Bregman divergence
9
10
manifold = Sphere()
11
scheduler = CondOTScheduler()
12
path = GeodesicProbPath(scheduler=scheduler, manifold=manifold)
13
14
for x_0, x_1 in dataloader:
# Samples from π0,1 of shape [batch_size, *data_dim]
15
t = torch.rand(batch_size)
# Randomize time t ∼U[0, 1]
16
sample: PathSample = path.sample(t=t, x_0=x_0, x_1=x_1)
# Sample the conditional path
17
18
model_output = model(sample.x_t, sample.t)
19
projected_model_output = manifold.proju(sample.x_t, model_output)
# Project to tangent space
20
21
loss = loss_fn(projected_model_output, sample.dx_t)
# CFM loss
22
23
optimizer.zero_grad()
24
loss.backward()
25
optimizer.step()
39

6
Continuous Time Markov Chain Models
This section presents the Continuous Time Markov Chains (CTMCs) as an alternative generative model to flow,
with the use-case of generating discrete data, i.e., data residing in a discrete (and finite) state space. CTMC
are Markov processes that form the building blocks behind the generative model paradigm of Discrete Flow
Matching (DFM) Campbell et al. (2024); Gat et al. (2024), later discussed in section 7. Therefore, this
section is analogous to section 3, where we presented flows as the building blocks behind the generative model
paradigm of Flow Matching (FM).
6.1
Discrete state spaces and random variables
Consider a finite version of Rd as our state space S = T d, where T = [K] = {1, 2, . . . , K}, sometimes called
vocabulary. Samples and states are denoted by x = (x1, . . . , xd) ∈S, where xi ∈T is single coordinate or
a token. We will similarly use states y, z ∈S. Next, X denotes a random variable taking values in the
state space S, with probabilities governed by the probability mass function (PMF) pX : S →R≥0, such that
P
x∈S pX(x) = 1, and the probability of an event A ⊂S being
P(X ∈A) =
X
x∈A
pX(x).
(6.1)
The notations X ∼pX or X ∼pX(X) indicate that X has the PMF pX. The δ PMF in the discrete case is
defined by
δ(x, z) =
(
1
x = z,
0
else.
(6.2)
where we sometimes also define δ PMFs on tokens, such as in δ(xi, yi), for some xi, yi ∈T .
6.2
The CTMC generative model
The CTMC model is an S-valued time-dependent family of random variables (Xt)0≤t≤1 that a form a Markov
chain characterized by the probability transition kernel pt+h|t defined via
pt+h|t(y|x) := P(Xt+h = y|Xt = x) = δ(y, x) + hut(y, x) + o(h), and P(X0 = x) = p(x),
(6.3)
Figure 13 The CTMC model is de-
fined by prescribing rates (velocities)
of probability between states.
where the PMF p indicates the initial distribution of the process at time
t = 0, and o(h) is an arbitrary function satisfying o(h)/h →0 as t →0.
The values ut(y, x), called rates or velocities, indicate the speed at which
the probability transitions between states as a function of time. By fully
characterized, we mean that all the joints P(Xt1 = x1, . . . , Xtn = xn),
for arbitrary 0 ≤t1 < · · · < tn ≤1 and xi ∈S, i ∈[n], are defined this
way.
To make sure the transition probabilities pt+h|t(y|x) are defined via (6.3),
velocities needs to satisfy the following rate conditions:
ut(y, x) ≥0 for all y ̸= x, and
X
y
ut(y, x) = 0.
(6.4)
If one of these conditions were to fail, then the transition probabilities
pt+h|t(·|x) would become negative or sum to c ̸= 1 for arbitrary small h > 0. Equation (6.3) plays he same
role as equation (3.16) and equation (3.19) when we were defining the flow generative modeling. The marginal
probability of the process Xt is denoted by the PMF pt(x) for time t ∈[0, 1]. Then, similarly to equation (3.24)
for the case of flows, we say that
ut generates pt if there exists pt+h|t satisfying (6.3) with marginals pt.
(6.5)
40

Simulating CTMC.
To sample Xt, sample X0 ∼p and take steps using the (naive) Euler method:
P(Xt+h = y | Xt) = δ(y, Xt) + hut(y, Xt).
(6.6)
According to (6.3), these steps introduce o(h) errors to the update probabilities. In practice, this means that
we would need a sufficiently small h > 0 to ensure that the right-hand side in (6.6) remains a valid PMF. One
possible remedy to assure that any choice of h > 0 results in a valid PMF, and maintains the o(h) local error
in probabilities is the following Euler method:
P(Xt+h = y | Xt) =
(
exp [hut(Xt, Xt)]
y = Xt
ut(y,Xt)
|ut(Xt,Xt)| (1 −exp [hut(Xt, Xt)])
y ̸= Xt
.
(6.7)
6.3
Probability paths and Kolmogorov Equation
Similarly to Continuity Equation in the continuous case, the marginal probabilities pt of the CTMC model
(Xt)0≤t≤1 are characterized by the Kolmogorov Equation
d
dtpt(y) =
X
x
ut(y, x)pt(x).
(6.8)
The following classical theorem (see also Theorems 5.1 and 5.2 in Coddington et al. (1956)) describes the
existence of unique solutions for this linear homogeneous system of ODEs.
Theorem 12 (Linear ODE existence and uniqueness). If ut(y, x) are in C([0, 1)) (continuous with respect
to time), then there exists a unique solution pt(x) to the Kolmogorov Equation (6.8), for t ∈[0, 1) and
satisfying p0(x) = p(x).
For the CTMC, the solution is guaranteed to exist for all times t ∈[0, 1) and no extra conditions are required
(unlike the non-linear case in theorem 1). The Kolmogorov Equation has an intimate connection with the
Continuity Equation (3.25). Rearranging the right-hand side of (6.8) by means of the rate conditions yields
X
x
ut(y, x)pt(x)
(6.4)
=
incoming flux
z
}|
{
X
x̸=y
ut(y, x)pt(x) −
outgoing flux
z
}|
{
X
x̸=y
ut(x, y)pt(y)
= −
X
x̸=y
[jt(x, y) −jt(y, x)] ,
where jt(y, x) := ut(y, x)pt(x) is the probability flux describing the probability of moving from state x to state
y per unit of time. The excess of outgoing flux is defined as the divergence, giving the Kolmogorov Equation
the same structure as the one described in section 3.5 for the Continuity Equation (Gat et al., 2024).
The following result is the main tool to build probability paths and velocities in the CTMC framework:
Theorem 13 (Discrete Mass Conservation). Let ut(y, x) be in C([0, 1)) and pt(x) a PMF in C1([0, 1))
in time t. Then, the following are equivalent:
1. pt, ut satisfy the Kolmogorov Equation (6.8) for t ∈[0, 1), and ut satisfies the rate conditions (6.4).
2. ut generates pt in the sense of 6.5 for t ∈[0, 1).
The proof of theorem 13 is given in appendix A.1.
41

6.3.1
Probability preserving velocities
As a consequence of the Discrete Mass Conservation (theorem 13), if velocity ut(y, x) generates the probability
path pt(x), then
˜ut(y, x) = ut(y, x) + vt(y, x) generates pt(x),
(6.9)
as long as vt(y, x) satisfies the rate conditions (6.4) and solves the divergence-free velocity equation
X
x
vt(y, x)pt(x) = 0.
(6.10)
In fact, ˜ut(y, x) solves the Kolmogorov Equation:
X
x
˜ut(y, x)pt(x) =
X
x
ut(y, x)pt(x) = ˙pt(y),
showing that one may add divergence-free velocities during sampling without changing the marginal probability.
This will be a useful fact when sampling from discrete Flow Matching models, described next.
7
Discrete Flow Matching
Remarkably, the Flow Matching blueprint in figure 2 carries out seamlessly from the continuous case to the
discrete case, yielding the Discrete Flow Matching (DFM) framework (Campbell et al., 2024; Gat et al., 2024).
In analogy to the continuous case, start by defining a probability path pt interpolating between a source
PMF p and a target PMF q. Second, we would like to find a CTMC model (Xt)0≤t≤1, defined by a learnable
velocity uθ
t , that generates the probability path pt. Finally, we train uθ
t by minimizing a Bregman divergence
that defines the Discrete Flow Matching loss. In sum, this is to solve the discrete version of the Flow Matching
problem (4.1).
7.1
Data and coupling
Our goal is to transfer samples X0 ∼p from a source PMF p to samples X1 ∈q from a target PMF q, where
X0, X1 ∈S are two RVs each taking values in the state space S. Source and target samples can be related
by means of the independent coupling (X0, X1) ∼p(X0)q(X1), or associate by means of a general PMF
coupling π0,1(x0, x1). For example, text translation data considers coupled data (x0, x1) representing the
same document written in two different languages. Another application, such as text generation, concerns
independent pairing where p(x0) is either the uniform probability over S giving all states equal probability, or
adding a special token m to the vocabulary T , i.e., T ∪{m}, and considering π0,1(x0, x1) = δ(x0, m)q(x1). Any
RV X0 ∼δ(X0, m) is the constant RV X0 = (m, . . . , m).
7.2
Discrete probability paths
The next step in the FM recipe is, as usual, to prescribe a probability path pt interpolating p and q.
Following section 4.4, we condition these objects on a general conditioning RV Z ∼pZ taking values in some
arbitrary space Z. The marginal probability path takes form
pt(x) =
X
z∈Z
pt|Z(x|z)pZ(z),
(7.1)
where pt|Z(·|z) is a conditional PMF, and the marginal probability path satisfies the boundary constraints
p0 = p and p1 = q.
7.3
The Marginalization Trick
The Marginalization Trick (see section 4.4) transfers to the discrete case as-is (Campbell et al., 2024; Gat
et al., 2024). Assuming that the conditional velocity field ut(·, ·|z) generates pt(·|z) in the sense of (6.5), we
obtain the marginal velocity field
ut(y, x) =
X
z
ut(y, x|z)pZ|t(z|x) = E [ut(y, Xt|Z) | Xt = x] ,
(7.2)
42

defined for all x, y ∈S where pt(x) > 0, and RV Xt ∼pt|Z(·|Z). By using Bayes’ rule, we get
pZ|t(z|x) = pt|Z(x|z)pZ(z)
pt(x)
.
(7.3)
To prove the discrete version of the Marginalization Trick Theorem (theorem 3), assume:
Assumption 3. pt|Z(x|z) ∈C1([0, 1)), ut(y, x|z) ∈C([0, 1)), and pt(x) > 0 for all x ∈S and t ∈[0, 1).
As it happened in the continuous case, the assumption pt > 0 is in practice mild, because we can always use
(1 −(1 −t)ϵ) · pZ|t + (1 −t)ϵ · puni , where puni is the uniform distribution over S, and ϵ > 0 is arbitrary small.
We are no ready to state and prove the result.
Theorem 14 (Discrete Marginalization Trick). Under Assumption 3, if ut(y, x|z) generates pt|Z(x|z)
then the marginal velocity ut(y, x) in (7.2) generates pt(x) in (7.1) for t ∈[0, 1).
Proof. The proof is conceptually similar to the continuous case. Start by computing:
d
dtpt(y) =
X
z
d
dtpt|Z(y|z)pZ(z)
(i)
=
X
z
"X
x
ut(y, x|z)pt|Z(x|z)
#
pZ(z)
(ii)
=
X
x
"X
z
ut(y, x|z)pt|Z(x|z)pZ(z)
pt(x)
#
pt(x)
(Bayes)
=
X
x
ut(y,x)
z
}|
{
X
z
ut(y, x|z)pZ|t(z|x) pt(x),
Equality (i) follows from theorem 13 and the fact that ut(y, x|z) generates pt|Z(y|z). Equality (ii) follows from
multiplying and dividing by pt(x) which is assumed positive. Therefore, ut(y, x) satisfies the Kolmogorov
Equation with pt. Also, ut(y, x) satisfies the rate conditions (6.4), because each ut(y, x|z) satisfies them. Lastly,
ut(y, x) ∈C([0, 1)) because both ut(y, x|z) and pZ|t(z|x) are in C([0, 1)). In particular, pZ|t(z|x) ∈C([0, 1))
follows from assuming pt(x) > 0 for t ∈[0, 1). By theorem 13, and because ut(x, y) satisfies the Kolmogorov
Equation with pt and the rate conditions, it generates pt in the sense of (6.5).
7.4
Discrete Flow Matching loss
To construct a CTMC generative model (Xt)0≤t≤1 we parameterize a velocity field uθ
t (y, x) with parameters
θ, e.g., using a neural network. One would construct the neural network to satisfy the rate conditions
equation (6.4). The Discrete Flow Matching loss to train the CTMC model is defined as:
LDFM(θ) = Et,Xt∼ptDXt(ut(·, Xt), uθ
t(·, Xt)),
(7.4)
for t ∼U[0, 1] and ut(·, x) ∈RS satisfying the rate conditions. This means that ut(·, x) ∈Ωx, where
Ωx =


v ∈RS
 v(y) ≥0 ∀y ̸= x, and v(x) = −
X
y̸=x
v(y)


⊂RS,
(7.5)
is a convex set, and Dx(u, v) is a Bregman divergence defined using a convex function Φx : Ωx →R. The
Conditional Discrete Flow Matching loss takes form
LCDFM(θ) = Et,Z,Xt∼pt|ZDXt(ut(·, Xt|Z), uθ
t (·, Xt)).
(7.6)
Once again, the two losses (7.4) and (7.6) both provide the same learning gradients.
43

Theorem 15. The gradients of the Discrete Flow Matching loss and the Conditional Discrete Flow
Matching loss coincide:
∇θLDFM(θ) = ∇θLCDFM(θ).
(7.7)
In particular, the minimizer of the Conditional Discrete Matching loss is the marginal velocity
uθ
t(y, x) = E [ut(y, Xt|Z) | Xt = x] .
(7.8)
The proof follows by applying proposition 1 when setting X = Xt, Y = (Xt, Z), defining f : S2 →RS as
(x, z) 7→ut(·, x|z) ∈RS, and integrating with respect to t ∈[0, 1].
7.5
Factorized paths and velocities
Figure 14 Factorized CTMC model
allows non-zero rates (velocities)
only between states that differ in
at-most one coordinate (token).
If implementing DFM as presented, we would require a learnable model
uθ
t(y, x)—for instance, a neural network—that outputs a rate for all
possible states y ∈S = T d. This would result in a huge output dimension
Kd, infeasible for common sequence lengths d and vocabulary sizes K.
One remedy to this issue is to consider factorized velocities (Campbell
et al., 2022),
ut(y, x) =
X
i
δ(y
¯i, x
¯i)ui
t(yi, x),
(7.9)
where ¯i = (1, . . . , i −1, i + 1, . . . , d) denotes all indices excluding i.
Therefore, the factorized velocity above connects state x to state y only if
these differ in at most one single token. When using factorized velocities,
we only require to model ui
t(yi, x), as these fully define ut(y, x). In turn,
each ui
t(yi, x) is a learnable model accepting x ∈S and returning a scalar
ui
t(yi, x) ∈R, for all i ∈[d] = {1, 2, . . . , d} and yi ∈T . Therefore, the
output of the model has a tractable dimension d · K. The rate conditions
for factorized velocities ui
t(y, x) are now required per dimension i ∈[d]:
ui
t(yi, x) ≥0 for all yi ̸= xi, and
X
yi∈T
ui
t(yi, x) = 0
for all x ∈S.
(7.10)
7.5.1
Simulating CTMC with factorized velocities
When using factorized velocities, we can sample CTMC models coordinate-wise (Campbell et al., 2024):
P(Xt+h = y | Xt = x) = δ(y, x) + h
X
i
δ(y
¯i, x
¯i)ui
t(yi, x) + o(h)
=
Y
i

δ(yi, xi) + hui
t(yi, x) + o(h)

,
where the second equality follows from δ(y, x) = Q
i δ(yi, xi) and the identity
Y
i

ai + hbi
=
Y
i
ai + h
X
i
(
Y
j̸=i
aj)bi + o(h).
Therefore, and up to an o(h) order, the transition kernel factorizes to coordinate-wise independent transitions
P(Xi
t+h = yi | Xt = x) = δ(yi, xi) + hui(yi, x) + o(h).
(7.11)
These can be sampled with the Euler method (6.7) per coordinate. Interestingly, continuous Flow Matching also
enjoys a similar factorization ut(x) = [u1
t(x), . . . , ud
t (x)] ∈Rd, where ˙Xi
t(x) = ui
t(Xt) determines the change
for coordinate i, and can be sampled independently (the “samples” in continuous FM are just deterministic).
44

7.5.2
Building probability paths with factorized velocities
If we construct probability paths in a certain way, it turns out to have factorized velocities (equation (7.9))
by construction. We explain this construction next. For this, we define a factorized probability path as a
probability path of the form:
qt(x) =
Y
i
qi
t(xi).
(7.12)
Then, the following result shows that these factorized probability paths have factorized velocities.
Proposition 2. Let qt(x) be a factorized probability path as in (7.12), where ui
t(yi, xi) ∈C([0, 1))
generates qi
t(xi). Then qt has a factorized generating velocity of the form
ut(y, x) =
X
i
δ(y
¯i, x
¯i)ui
t(yi, xi).
(7.13)
To proceed with the proof, let us denote the marginal distributions of a PMF q(x) by
qi(xi) :=
X
x¯i
q(x)
q
¯i(x
¯i) :=
X
xi
q(x)
(7.14)
Proof. Let qt be a factorized probability path (7.12). Let ui
t(yi, xi) be the generating velocity of qi
t(xi).
Differentiating with respect to t yields
d
dtqt(y) =
X
i
q
¯i
t(y
¯i) d
dtqi
t(yi)
(i)
=
X
i

X
x¯i
δ(y
¯i, x
¯i)q
¯i
t(x
¯i)


"X
xi
ui
t(yi, xi)qi
t(xi)
#
(ii)
=
X
x
"X
i
δ(y
¯i, x
¯i)ui
t(yi, xi)
#
qt(x),
Equality (i) follows from q¯i
t(y¯i) = P
x¯i δ(y¯i, x¯i)q¯i
t(x¯i) and the Kolmogorov Equation (6.8). Equality (ii) follows
from changing the summation order and noting that, by the definition of qt, we have q¯i
t(x¯i)qi
t(xi) = qt(x) and
P
x¯i
P
xi = P
x.
We are now ready to show the main tool for constructing paths pt with factorized velocities that interpolate
between arbitrary p and q (Campbell et al., 2024; Gat et al., 2024).
Theorem 16 (Discrete Factorized Marginalization Trick). Consider a marginal probability path con-
structed via
pt(x) =
X
z
pt|Z(x|z)pZ(z), with pt|Z(x|z) =
Y
i
pi
t|Z(xi|z),
(7.15)
i.e., where the conditional path factorizes in the sense of equation (7.12). Further, assume that ui
t(yi, xi|z)
is C([0, 1)) generates pi
t|Z(xi|z) in C1([0, 1)), and pt(x) > 0 for all x ∈S and t ∈[0, 1). Then, the
marginal velocity is
ut(y, x) =
X
i
δ(y
¯i, x
¯i)ui
t(yi, x)
(7.16)
with
ui
t(yi, x) =
X
z
ui
t(yi, xi|z)pZ|t(z|x) = E

ui
t(yi, Xi
t|Z)|Xt = x

(7.17)
generates pt(x).
45

Proof. According to proposition 2, the factorized conditional paths pt|Z(x|z) have factorized generating
velocities ut(y, x|z) = P
i δ(y¯i, x¯i)ui
t(yi, xi|z). Therefore,
ut(y, x)
(i)
=
X
z
ut(y, x|z)pZ|t(z|x)
(ii)
=
X
z
"X
i
δ(y
¯i, x
¯i)ui
t(yi, xi|z)
#
pZ|t(x|z)
(iii)
=
X
i
δ(y
¯i, x
¯i)
"X
z
ui
t(yi, xi|z)pZ|t(z|x)
#
.
Equality (i) follows from (7.2). Equality (ii) follows from assuming that pt|Z has factorized velocity. Equality
(iii) follows from changing summation order.
Because pi
t|Z(xi|z) ∈C1([0, 1)) and pt(x) > 0, it follows
that pt|Z(x|z) ∈C1([0, 1)). Similarly, because ui
t(yi, xi|z) ∈C([0, 1)), it follows that ut(y, x|z) ∈C([0, 1)).
Therefore, theorem 14 implies that ut(y, x) generates pt(x), as required.
By using theorem 16, we can design a probability path pt with factorized velocities interpolating between a
source PMF p and a target PMF q as follows.
1. Find factorized probability conditional paths pt|Z(x|z) = Q
i pi
t|Z(xi|z) such that the marginal pt(x)
satisfies p0 = p and p1 = q.
2. Find generating velocities ui
t(yi, xi|z) to pi
t|Z(xi|z). This can be done by finding solution ui
t(yi, xi|z) to
the Kolmogorov Equation:
X
xi
ui
t(yi, xi|z)pi
t|Z(xi|z) = d
dtpi
t|Z(yi|z),
(7.18)
for all yi ∈T , fixed values of i ∈[d], z ∈Z, and t ∈[0, 1). As a remark, (7.18) is an under-determined
linear system of equations with |T | unknowns (significantly less unknowns than the entire state space
|S|).
7.5.3
Conditional Discrete Flow Matching loss for factorized velocities
Representing the marginal velocity uθ
t in terms of factorized velocities uθ,i
t
enables the following Conditional
Flow Matching loss
LCDFM(θ) = Et,Z,Xt∼pt|Z
X
i
Di
Xt

ui
t(·, Xt|Z), uθ,i
t (·, Xt)

,
(7.19)
where t ∼U[0, 1], and ui
t(·, x|z), uθ,i
t (·, x) ∈RT satisfy the rate conditions. This means that ui
t(·, x|z), uθ,i
t (·, x) ∈
Ωxi where, for α ∈T , we define
Ωα =


v ∈RT
 v(β) ≥0 ∀β ∈T \ {α} , and v(α) = −
X
β̸=α
v(β)


⊂RT .
(7.20)
This is a convex set, and Di
x(u, v) is a Bregman divergence defined by a convex function Φi
x : Ωxi →R. As
before, we justify this loss using proposition 1 and setting X = Xt, Y = ui
t(·, Xt, Z) ∈RT , letting Di
x(u, v) be
a Bregman divergence over Ωxi ⊂RT , and integrating with respect to t ∈[0, 1].
7.5.4
Mixture paths
It is time to implement section 7.5.2 to build practical probability paths and their corresponding conditional
velocities. Following Gat et al. (2024), we condition on Z = (X0, X1) to accommodate arbitrary data couplings
(X0, X1) ∼π0,1(X0, X1). Then, we build the factorized conditional paths
pt|0,1(x|x0, x1) =
Y
i
pi
t|0,1(xi|x0, x1)
(7.21)
46

as mixtures
pi
t|0,1(xi|x0, x1) = κtδ(xi, xi
1) + (1 −κt)δ(xi, xi
0),
(7.22)
where κ : [0, 1] →[0, 1] is a C1([0, 1]) scheduler. Note that a RV Xi
t ∼pi
t|0,1(·|x0, x1) follows
Xi
t =
(
xi
1
with prob κt
xi
0
with prob (1 −κt) ,
(7.23)
i.e. it assumes either the source or the target states with a probability depending on the time t.
If κ0 = 0 and κ1 = 1, then the marginal pt(x) in (7.1) satisfies the boundary constraints. We also need
generating velocities ui
t(yi, xi|x0, x1) for pi
t|0,1(xi|x0, x1), which are solutions to (7.18). We derive these as
follows:
d
dtpi
t|Z(yi|z)
(7.22)
=
˙κt

δ(yi, xi
1) −δ(yi, xi
0)

(7.22)
=
˙κt
"
δ(yi, xi
1) −
pi
t|Z(yi|z) −κtδ(yi, xi
1)
1 −κt
#
=
˙κt
1 −κt
h
δ(yi, xi
1) −pi
t|Z(yi|z)
i
=
X
xi
˙κt
1 −κt

δ(yi, xi
1) −δ(yi, xi)

pi
t|Z(xi|z),
where we have used z = (x0, x1) and Z = (X0, X1) interchangeably to keep notation concise. In conclusion,
we have found a conditional velocity generating the path in (7.22), namely
ui
t(yi, xi|x0, x1) =
˙κt
1 −κt

δ(yi, xi
1) −δ(yi, xi)

.
(7.24)
Code 9 shows how mixture paths are defined in the library flow_matching library.
Code 9: Discrete probability path
1
import torch
2
from flow_matching.path import MixtureDiscreteProbPath
3
from flow_matching.path.path_sample import DiscretePathSample
4
5
# Create a discrete probability path object
6
path = MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=1.0))
7
8
# Sample the conditional path
9
# batch_size = 2 for t, X0 and X1
10
t = torch.tensor([0.25, 0.5])
11
x_0 = torch.tensor([0, 0])
12
x_1 = torch.tensor([1, 2])
13
sample: DiscretePathSample = path.sample(t=t, x_0=x_0, x_1=x_1)
14
sample.x_0
# X0 is [0, 0]
15
sample.x_1
# X1 is [1, 2]
16
# Xt is
17
#
[0 with probability 0.75 and 1 with probability 0.25,
18
#
0 with probability 0.5 and 2 with probability 0.5]
19
sample.x_t
20
sample.t
# t is [0.25, 0.5]
47

Velocity posterior parameterization.
Similar to the continuous case (e.g., section 4.8.1), we can choose to
parameterize our velocity ui
t(yi, x) in different ways. The first approach is to parameterize it directly, akin
to velocities in flows. Another way, which we take here, is motivated by the following computation of the
mixture marginal velocity following (7.17):
ui
t(yi, x) =
X
x0,x1
˙κt
1 −κt

δ(yi, xi
1) −δ(yi, xi)

p0,1|t(x0, x1|x)
=
X
xi
1
˙κt
1 −κt

δ(yi, xi
1) −δ(yi, xi)

pi
1|t(xi
1|x),
(7.25)
where for the second equality we denote the marginal of the posterior p0,1|t
pi
1|t(xi
1|x) =
X
x0,x¯i
1
p0,1|t(x0, x1|x)
(7.26)
= E

δ(xi
1, Xi
1) | Xt = x

.
(7.27)
This derivation represents the marginal ui
t(yi, x) using a learnable posterior pθ,i
1|t(xi
1|x), which can be understood
as a discrete version of x1-prediction (section 4.8.1). Next, we explore loss functions to learn this posterior.
CDFM losses for mixture paths
We present two options for learning pθ,i
1|t(xi
1|x), both justified by proposition 1.
First, the marginal posterior (7.26) and (7.27) can be learned by the conditional matching loss
LCM(θ) = Et,X0,X1,XtDXt

δ(·, Xi
1), pθ,i
1|t(·|Xt)

(7.28)
Since δ(·, Xi
1), pθ,i
1|t(·|Xt) are PMFs. Therefore, we can set the Bregman divergence to be the KL-divergence
D(p, q) = P
α∈T p(α) log p(α)
q(α) comparing PMFs, obtaining
LCM(θ) = −Et,X0,X1,Xt log pθ,i
1|t(Xi
1|Xt) + const.
(7.29)
Alternatively, we may follow section 7.5.3 and use the factorized loss in (7.19) with uθ,i
t
parametrized by pθ,i
1|t.
In this case, we can set the Bregman divergence to be the generalized KL comparing general (not necessarily
probability) vectors u, v ∈Rm
≥0:
D(u, v) =
X
j
uj log uj
vj −
X
j
uj +
X
j
vj.
(7.30)
For this choice of D, we get
D

ui
t(·, xi|x0, x1), uθ,i
t (·, x)

=
˙κt
1 −κt
h
(δ(xi
1, xi) −1) log pθ,i
1|t(xi
1|x) + δ(xi
1, xi) −pθ,i
1|t(xi|x)
i
(7.31)
which implements the loss (7.19) when conditioning on Z = (X0, X1). The generalized KL loss (7.31) also
provides an evidence lower bound (ELBO) on the likelihood of the target distribution (Shaul et al., 2024),
−log pθ
1(x1) ≤Et,X0,Xt∼pt|0,1
X
i
D

ui
t(·, Xi
t|X0, x1), uθ,i
t (·, Xt)

,
(7.32)
where pθ
1 is the marginal generated by the model at time t = 1. Hence, in addition to training, the generalized
KL loss is commonly used for evaluation.
48

Sampling mixture paths
The parametrization based on the posterior pθ,i
1|t leads to the following sampling
algorithm. As indicated in section 7.5.1 working with factorized velocities enables the coordinate-wise sampling
(7.11). According to (7.17) and (7.25),
P(Xi
t+h = yi | Xt = x) = δ(yi, xi) + hui(yi, x) + o(h)
(7.33)
=
X
xi
1

δ(yi, xi) + h
˙κt
1 −κt

δ(yi, xi
1) −δ(yi, xi)

+ o(h)

pi
1|t(xi
1|x).
(7.34)
Consequently, and given Xt = x, we may perform one step of sampling by performing the next two steps
for each i ∈[d]: (i) draw Xi
1 ∼pi
1|t(Xi
1|x); and (ii) update Xi
t+h according to the Euler step in (6.7) with
the velocity
˙κt
1−κt

δ(yi, Xi
1) −δ(yi, xi)

. Intuitively, (ii) decides whether to set Xi
t+h = Xi
1 or remain at
Xi
t+h = Xi
t.
One-sided mixture paths and probability preserving velocities
It is often useful to extend the design space
of the sampling algorithm by adding some divergence-free component, as described in section 6.3.1. For
factorized paths, the divergence-free velocity vi
t needs to satisfy (7.18), namely,
X
xi
vi
t(yi, xi|z)pi
t|Z(xi|z) = 0.
(7.35)
In general it could challenging to find such probability-preserving velocities without learning additional
quantities, e.g., pi
0|t. However, one useful case where a probability preserving velocity can be found in closed
form is when assuming iid source distribution i.e., p(x) = Q
i p(xi) and independent coupling π0,1(x0, x1) =
p(x0)q(x1). In this case the marginal mixture path takes the form
pt(x) =
X
x1
pt|1(x|x1)q(x1), where pt|1(x|x1) =
Y
i
pi
t|1(xi|x1),
where pi
t|1(xi|x1) =

κtδ(xi, xi
1) + (1 −κt)p(xi)

. The conditional velocity in (7.24), i.e.,
ui
t(yi, xi|x1) =
˙κt
1 −κt

δ(yi, xi
1) −δ(yi, xi)

(7.36)
also generates pi
t|1(xi|x1). To find a divergence-free velocity we can, for example, subtract from this velocity a
backward-time velocity ˜ui
t for pi
t|1(yi, xi|x1) (Gat et al., 2024), in the sense that it satisfies the Kolmogorov
equation with pi
t|1(yi, xi|x1) and −˜ui
t satisfies the rate conditions. Such a velocity can be found in a fashion
to equation (7.24),
˜ui
t(yi, xi|x1) = ˙κt
κt

δ(yi, xi) −p(xi)

.
(7.37)
Therefore, a divergence-free velocity for pi
t|1(xi|x1) conditional path can be defined via
vi
t(yi, xi|x1) = ui
t(yi, xi|x1) −˜ui
t(yi, xi|x1).
(7.38)
According to section 6.3.1, if we add a divergence-free field vi
t(yi, xi|x1) to the velocity ui
t(yi, xi|x1), the latter
still generates the same probability path pi
t|1(xi|x1). Consequently, theorem 16 implies that the marginal
velocity ui
t(yi, x) defined by
ui
t(yi, x) =
X
x1

ui
t(yi, xi|x1) + ctvi
t(yi, xi|x1)

p1|t(x1|x)
=
X
xi
1

ui
t(yi, xi|xi
1) + ctvi
t(yi, xi|xi
1)

pi
1|t(xi|x),
still generates the same marginal path pt(x), where the second equality follows from ui
t(yi, xi|x1) = ui
t(yi, xi|xi
1)
for mixture paths, and similarly for vi
t(yi, xi|xi
1). In conclusion, and given Xt = x, a single step of the
49

generalized sampling algorithm consists in (i) drawing Xi
1 ∼pi
1|t(Xi
1|x) and (ii) taking an Euler step (6.7)
with the velocity
ui
t(yi, xi|x1) =
˙κt
1 −κt

δ(yi, Xi
1) −δ(yi, xi)

+ ct

˙κt
1 −κt

δ(yi, xi
1) −δ(yi, xi)

−˙κt
κt

δ(yi, xi) −p(xi)

,
where ct > 0 is a time dependent constant.
Similar to the continuous flow matching example in code 1, we provide a standalone implementation of discrete
flow matching in pure PyTorch in code 11. Code 10 illustrates how to train a discrete flow with arbitrary
data coupling using the flow_matching library.
Code 10: Training and sampling DFM with mixture paths and arbitrary data coupling.
1
import torch
2
3
from flow_matching.path import MixtureDiscreteProbPath, DiscretePathSample
4
from flow_matching.path.scheduler import PolynomialConvexScheduler
5
from flow_matching.loss import MixturePathGeneralizedKL
6
from flow_matching.solver import MixtureDiscreteEulerSolver
7
from flow_matching.utils import ModelWrapper
8
9
10
model = ...
# Define a trainable velocity model
11
optimizer = torch.optim.Adam(model.parameters())
12
13
scheduler = PolynomialConvexScheduler(n=1.0)
14
path = MixtureDiscreteProbPath(scheduler=scheduler)
15
loss_fn = MixturePathGeneralizedKL(path=path)
# Generalized KL Bregman divergence
16
17
for x_0, x_1 in dataloader:
# Samples from π0,1 of shape [batch_size, *data_dim]
18
t = torch.rand(batch_size) * (1.0 - 1e-3)
# Randomize time t ∼U[0, 1 −10−3]
19
sample: DiscretePathSample = path.sample(t=t, x_0=x_0, x_1=x_1)
# Sample the conditional path
20
model_output = model(sample.x_t, sample.t)
21
22
loss = loss_fn(logits=model_output, x_1=sample.x_1, x_t=sample.x_t, t=sample.t) # CDFM loss
23
24
optimizer.zero_grad()
25
loss.backward()
26
optimizer.step()
27
28
class ProbabilityDenoiser(ModelWrapper):
29
def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
30
logits = self.model(x, t, **extras)
31
return torch.nn.functional.softmax(logits.float(), dim=-1)
32
33
# Sample X1
34
probability_denoiser = ProbabilityDenoiser(model=model)
35
x_0 = torch.randint(size=[batch_size, *data_dim])
# Specify the initial condition
36
solver = MixtureDiscreteEulerSolver(
37
model=probability_denoiser,
38
path=path,
39
vocabulary_size=vocabulary_size
40
)
41
42
step_size = 1 / 100
43
x_1 = solver.sample(x_init=x_0, step_size=step_size, time_grid=torch.tensor([0.0, 1.0-1e-3]))
50

Code 11: Standalone Discrete Flow Matching code
flow_matching/examples/standalone_discrete_flow_matching.ipynb
1
import torch
2
import matplotlib.pyplot as plt
3
from torch import nn, Tensor
4
from sklearn.datasets import make_moons
5
6
class DiscreteFlow(nn.Module):
7
def __init__(self, dim: int = 2, h: int = 128, v: int = 128):
8
super().__init__()
9
self.v = v
10
self.embed = nn.Embedding(v, h)
11
self.net = nn.Sequential(
12
nn.Linear(dim * h + 1, h), nn.ELU(),
13
nn.Linear(h, h), nn.ELU(),
14
nn.Linear(h, h), nn.ELU(),
15
nn.Linear(h, dim * v))
16
17
def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
18
return self.net(torch.cat((t[:, None], self.embed(x_t).flatten(1, 2)),
-1)).reshape(list(x_t.shape) + [self.v])
,→
19
20
batch_size = 256
21
vocab_size = 128
22
23
model = DiscreteFlow(v=vocab_size)
24
optim = torch.optim.Adam(model.parameters(), lr=0.001)
25
26
for _ in range(10000):
27
x_1 = Tensor(make_moons(batch_size, noise=0.05)[0])
28
x_1 = torch.round(torch.clip(x_1 * 35 + 50, min=0.0, max=vocab_size - 1)).long()
29
x_0 = torch.randint(low=0, high=vocab_size, size=(batch_size, 2))
30
31
t = torch.rand(batch_size)
32
x_t = torch.where(torch.rand(batch_size, 2) <
t[:, None], x_1, x_0)
33
34
logits = model(x_t, t)
35
loss = nn.functional.cross_entropy(logits.flatten(0, 1), x_1.flatten(0, 1)).mean()
36
optim.zero_grad()
37
loss.backward()
38
optim.step()
39
40
x_t = torch.randint(low=0, high=vocab_size, size=(200, 2))
41
t = 0.0
42
results = [(x_t, t)]
43
while t < 1.0 - 1e-3:
44
p1 = torch.softmax(model(x_t, torch.ones(200) * t), dim=-1)
45
h = min(0.1, 1.0 - t)
46
one_hot_x_t = nn.functional.one_hot(x_t, vocab_size).float()
47
u = (p1 - one_hot_x_t) / (1.0 - t)
48
x_t = torch.distributions.Categorical(probs=one_hot_x_t + h * u).sample()
49
t += h
50
results.append((x_t, t))
51
fig, axes = plt.subplots(1, len(results), figsize=(15, 2), sharex=True, sharey=True)
52
for (x_t, t), ax in zip(results, axes):
53
ax.scatter(x_t.detach()[:, 0], x_t.detach()[:, 1], s=10)
54
ax.set_title(f't={t:.1f}')
55
plt.tight_layout()
56
plt.show()
51

8
Continuous Time Markov Process Models
In the previous sections, we have developed a flow model on Rd and Riemannian manifolds (see section 3 and
section 5) and a CTMC model for discrete data (see section 6). In this section, we want to unify and extend
these models to a generative model that works for (1) general state spaces and (2) general Markov processes.
This generative model will allow us to extend the principles of Flow Matching to a wide variety of generative
models for a variety of modalities in section 9.
8.1
General state spaces and random variables
Working with general modalities.
Our explicit goal is not specify the modality we use. Hence, throughout
this section, let S be a general state space. Important examples are S = Rd (e.g., images, vectors), S discrete
(e.g., language), S a Riemannian manifold (e.g., geometric data) or their products for generation of multiple
data modalities jointly (multimodal models). For all modalities, we can define a metric (or distance function)
d : S × S →R≥0, (x, y) 7→d(x, y) on S. For example, for S discrete, the metric is simply d(x, y) = 1 if y ̸= x
and d(x, x) = 0 for all x ∈S. For S = Rd, we use d(x, y) = ∥x −y∥. We need to make a technical assumption
that (S, d) is a Polish metric space, i.e., it is complete (i.e., any Cauchy sequence converges) and separable
(i.e., it has a countable dense subset). Any modality of interest for machine learning has that property.
Densities over general state spaces.
So far, in this work we assumed that a probability distribution p over S
is represented by a density p : S →R≥0. For general state spaces, we use a general reference measure ν and
the density becomes the Radon-Nikodym derivative dp
dν . In other words, probabilities can be expressed as
integrals with respect to ν
P(A) =
Z
A
p(x)ν(dx) for all measurable A ⊂S
For S discrete, ν was the counting measure (so the integrals are just sums) and p(x) is just the probability
mass function (PMF). For S = Rd, ν was the Lebesgue measure (so the integrals are just the “usual” integral)
and p(x) is just the probability density function (PDF). The above generalizes that to arbitrary state spaces.
(Optional) Working with arbitrary distributions.
It is important to note that not every probability distribution
admits a density with respect to a reference measure. For the reader unfamiliar with general measure theory,
it is safe ignore this possibility as a technical remark as long one works with distributions of interest that
have a density p(x). However, note that these are not just pathological examples but there are real cases of
interest for machine learning applications where this matters: A simple example is a probability path of the
form pt = δ(1−t)x+ty on S = Rd connecting two points x, y ∈Rd in a straight line - this cannot be represented
by a density. Another example would be probability distributions over S = C([0, 1], R), e.g., for trajectory
modeling, that often do not have a density with respect to a common reference measure. To mathematically
handle such cases, we develop our framework for general probability measures p over S. For this, we use the
notation p(dx) where “dx” is a symbolic expression denoting integration with respect to p in a variable x. For
example, for a bounded measurable function f : S →R we write
EX∼p[f(X)] =
Z
f(x)p(dx)
for the Lebesgue integral or the expected value of f under p. As before, we use (with a slight abuse of
notation) the same notation p(x) to denote the density of the measure p(dx).
8.2
The CTMP generative model
Similarly, our explicit goal is build a model that works for arbitrary evolution process - regardless of whether
we use a flow, diffusion, a CTMC, a combination, or something else. Therefore, we define a evolution process
in this section that is general but satisfies the necessary regularity assumptions to build a generative model.
For t ∈[0, 1], let Xt ∈S be a random variable. We call (Xt)0≤t≤1 a Continuous-time Markov process (CTMP)
if it fulfills the following condition:
52

Name
Flow
Diffusion
Jump process
Continous-time
Markov chain
Space S
S = Rd
S = Rd
S arbitrary
|S| < ∞
Parameters
Velocity field:
ut(x) ∈Rd
Diffusion coefficient:
σt(x) ∈Rd×d (p.s.d.)
Jump measure Qt(dy, x)
ut ∈RS×S, 1T ut = 0
ut(x′; x) ≥0 (x′ ̸= x)
Sampling
Xt+h = Xt + hut(Xt)
Xt+h = Xt +
√
hσt(Xt)ϵt
ϵt ∼N(0, I)
Xt+h = Xt with prob. 1 −h
R
Qt(dy, x)
Xt+h ∼
Qt(dy,x)
R
Qt(dy,x) with prob. h
R
Qt(dy, x)
Xt+h ∼(I + hut)(·; Xt)
Generator Lt
∇f T ut
1
2∇2f · σ2
t
R
[f(y) −f(x)]Qt(dy, x)
f T uT
t
KFE
(Adjoint)
Continuity Equation:
∂tpt = −div(ptut)
Fokker-Planck Equation:
∂tpt = 1
2∇2 · (ptσ2
t )
Jump Continuity Equation: ∂tpt(x) =
R
Qt(x, y)pt(y) −Qt(y, x)pt(x)ν(dy)
Mass preservation:
∂tpt = utpt
Marginal
EZ∼pZ|t(·|x)[ut(x|Z)]
EZ∼pZ|t(·|x)[σ2
t (x|Z)]
EZ∼pZ|t(·|x)[Qt(y, x|Z)]
EZ∼pZ|t(·|x)[ut(y; x|Z)]
CGM Loss
(Example)
∥ut(X|Z) −uθ
t(X)∥2
∥σ2
t (X|Z) −(σθ
t )2(X)∥2
2
(
R
Qθ
t (y; X)ν(dy)
−Qt(y, X|Z) log Qθ
t (y; X)ν(dy))
( P
y̸=x
uθ
t(y; X)
−ut(y; X|Z) log uθ
t (y; X))
Table 2
Some examples of CTMP generative models and how they can be learnt with Generator Matching. This list
is not exhaustive. Derivations are in section 8. For diffusion, we assume zero drift as this is covered by the “Flow”
column. KFE is listed in its adjoint version, i.e., assumes jump kernel Qt(y, x) and density pt(x) exists with respect to
reference measure ν. “p.s.d.”: Positive semi-definite.
P[Xtn+1 ∈A|Xt1, Xt2, . . . , Xtn] = P[Xtn+1 ∈A|Xtn]
(0 ≤t1 < · · · < tn+1 ≤1, A ⊆S)
(8.1)
Informally, the above condition says that the process has no memory. If we know the present, knowing the
past will not influence our prediction of the future. In table 2, we give an overview over important classes of
Markov processes. For example, a flow on a manifold is a Markov process with deterministic transitions, a
diffusion is a Markov process with transitions driven by Brownian motion, and CTMCs are Markov processes
determined by rates (we will explain this in detail in section 8.2.2). Each Markov process has a transition
kernel (pt+h|t)0≤t<t+h≤1 that assigns every x ∈S a probability distribution pt+h|t(·|x) such that
P[Xt+h ∈A|Xt = x] = pt+h|t(A|x)
for all t, h ≥0, A ⊂S measurable
(8.2)
Due to the Markov assumption, a Markov process is uniquely determined by the transition kernel and
the distribution of X0. Conversely, any transition kernel and initial distribution defines a Markov process.
Therefore, there is a 1:1 correspondence.
Our next goal is to define a corresponding generalization of a velocity field for CTMPs. Informally, it would
be the 1st-order derivative of the transition kernel in t:
Lt := d
dh

h=0pt+h|t
(8.3)
We call the 1st-order derivative Lt the generator of pt+h|t (Ethier and Kurtz, 2009; Rüschendorf et al., 2016).
Similar to derivatives, generators are first-order linear approximations and easier to parameterize than pt+h|t.
As we will see, diffusion, flows, and other generative models can all be seen as algorithms to learn the generator
of a Markov process (see table 2). This leads to the general form of the CTMP generative model given by
CTMP model (informal):
pt+h|t(·|x) := δx + hLt(x) + o(h), and X0 ∼p.
(8.4)
However, as a transition kernel pt+h|t is not a real function, equation 8.3 and 8.4 are only heuristic and not
well-defined yet. Therefore the first goal of this section is to provide a formal definition of the generator and
the CTMP generative model.
53

8.2.1
Formal definition of generator
The first problem in equation (8.3) is that derivatives are usually defined with respect to functions mapping
to vector spaces but pt+h|t maps to a distribution. However, this can be alleviated by using test functions.
Test functions are a way to “probe” a probability distribution. They serve as a theoretical tool to handle
distributions as if they were real-valued functions. Specifically, a set of test functions is a family T of bounded,
measurable functions f : S →R that characterize probability distributions fully, i.e., for two probability
distributions µ1, µ2 on S it holds
µ1 = µ2
⇔
EX∼µ1[f(X)] = EX∼µ2[f(X)]
for all f ∈T
(8.5)
Generally speaking, one chooses T to be as “nice” (or regular) as possible. For example, if S = Rd, the space
T = C∞
c (Rd) of infinitely differentiable functions with compact support fulfills that property. For S discrete,
T = RS simply consists of all functions (which are just vectors in this case). Let Xt ∼pt. We define the
marginal action and transition action as
⟨pt, f⟩:=
Z
f(x)pt(dx) = EX∼pt [f(X)]
(8.6)

pt+h|t, f

(x) :=

pt+h|t(·|x), f

= E [f(Xt+h)|Xt = x]
(8.7)
where the marginal action maps each test function f to a scalar ⟨pt, f⟩∈R, while the transition action maps
a real-valued function x 7→f(x) to a another real-valued function x 7→

pt+h|t, f

(x). The tower property
implies that

pt,

pt+h|t, f

= ⟨pt+h, f⟩. We note that the above is only a “symbolic” dot product but becomes
a “proper” dot product if a density pt(x) exists, i.e., ⟨pt, f⟩=
R
f(x)pt(x)ν(dx).
The second step to formally define a derivative as in equation (8.3) is that we need to impose some notion
of “smoothness” on a Markov process that we define now. Let C0(S) be the space of continuous functions
f : S →R that vanish at infinity, i.e., for all ϵ > 0 there exists a compact set K ⊂S such that |f(x)| < ϵ for
all x ∈S \ K. We use the supremum norm ∥· ∥∞on C0(S). A CTMP Xt is called a Feller process if it fulfils
the following two conditions (Feller, 1955; Rüschendorf et al., 2016):
1. Strong continuity: The action of pt+h|t is continuous in time:
lim
h′→h,t′→t ∥

pt′+h′|t′, f

−

pt+h|t, f

∥∞= 0
for all h, t ≥0, f ∈C0(S)
2. No return from infinity: The action of pt+h|t preserves functions that vanish at infinity:

pt+h|t, f

∈C0(S)
for all h, t ≥0, f ∈C0(S)
Assumption 4. The CTMP (Xt)0≤t≤1 is a Feller process.
This is a reasonable assumption given that we want to use Xt in a machine learning model: We define
probability paths where the distribution of the generative process Xt vary smoothly, and all our data usually
lies in some bounded (compact) set.
Let us now revisit (8.3) and try define the derivative of pt+h|t. With the test function perspective in mind, we
can take derivatives of

pt+h|t, f

(x) per x ∈S and define
d
dh

h=0

pt+h|t, f

(x) = lim
h→0

pt+h|t, f

(x) −f(x)
h
:= [Ltf](x).
(8.8)
We call this action the generator Lt and define it for all f for which the above limit exists uniformly, i.e., in
the norm ∥·∥∞. Intuitively, a generator is defined as an operator of test functions. In table 2, there are several
examples of generators that we derive in section 8.2.2. There is a 1:1 correspondence between a generator and
a Feller process (Rogers and Williams, 2000; Ethier and Kurtz, 2009; Pazy, 2012) - in the same way as there is
a correspondence between a flow and a vector field (see theorem 1). This will later allows us to parameterize
a Feller process via a generator in a neural network.
54

With this definition, the CTMP model in (8.4) has the, now well-defined, form as

pt+h|t, f

= f + hLtf + o(h) (for all f ∈T )
and X0 ∼p
(8.9)
where o(h) describes an error terms E(h) ∈C0(S) such that lim
h→0
1
h∥E(h)∥∞= 0. Similarly to equation (3.24)
for the case of flows and to equation (6.5) for the case of CTMCs, we say that
Lt generates pt if there exists a pt+h|t satisfying (8.9) with CTMP Xt such that Xt ∼pt
(8.10)
In other words, a generator Lt generates the probability path pt if a Markov process that is (1) initialized
with p = p0 and (2) simulated with Lt has marginals pt.
8.2.2
Examples of CTMP models
We go through several examples to illustrate how to compute a generator of a Markov process. The results
from this section are summarized in table 2.
Flows.
Let S = Rd and u : [0, 1] × Rd →Rd, (t, x) 7→ut(x) be a time-dependent velocity field defining a
flow ψt (see section 3). Let T = C∞
c (Rd) be the space of infinitely differentiable and smooth functions with
compact support. Then we can compute the generator via
[Ltf](x) = lim
h→0
E [f(Xt+h)|Xt = x] −f(x)
h
(8.11)
(i)
= lim
h→0
E [f(Xt + hut(Xt) + o(h))|Xt = x] −f(x)
h
(8.12)
(ii)
= lim
h→0
E

f(Xt) + h∇f(Xt)T ut(Xt) + o(h)|Xt = x

−f(x)
h
(8.13)
= lim
h→0
f(x) + h∇f(x)T ut(x) + o(h) −f(x)
h
(8.14)
(8.15)
= ∇f(x)T ut(x),
(8.16)
where (i) follows from a Euler approximation of the flow and (ii) follows from a first-order Taylor approximation
of f around Xt. Therefore, the flow generator is given by
Ltf(x) = ∇f(x)T ut(x).
(8.17)
Diffusion.
Let S = Rd and σt : [0, 1] × Rd →Rd×d, (t, x) 7→σt(x) be a time-dependent function mapping
to symmetric positive semi-definite matrices σt in a continuous fashion. A diffusion process with diffusion
coefficient σt is defined via the SDE dXt = σt(Xt)dWt for a Wiener process Wt (Øksendal, 2003). This
process can be approximated via the infinitesimal sampling procedure:
Xt+h = Xt +
√
hσt(Xt)ϵt,
ϵt ∼N(0, I)
(8.18)
55

Let again be T = C∞
c (Rd). Then we can compute the generator via
[Ltf](x) = lim
h→0
E
h
f(Xt +
√
hσt(Xt)ϵt + o(h))|Xt = x
i
−f(x)
h
(8.19)
(i)
= lim
h→0
Eϵt[f(x) + ∇f(x)T √
hσt(x)ϵt + 1
2h[σt(x)ϵt]T ∇2f(x)[σt(x)ϵt] + o(h) −f(x)]
h
(8.20)
= lim
h→0
∇f(x)T √
hσt(x)Eϵt[ϵt] + Eϵt[ 1
2h[σt(x)ϵt]T ∇2f(x)[σt(x)ϵt]]
h
(8.21)
= 1
2Eϵt[ϵT
t [σt(x)]T ∇2f(x)[σt(x)]ϵt]]
(8.22)
(ii)
= 1
2tr
 σt(x)T ∇2f(x)σt(x)

(8.23)
(iii)
= 1
2tr(σt(x)σt(x)T ∇2f(x))
(8.24)
(iv)
= 1
2σ2
t (x) · ∇2f(x)
(8.25)
where in (i) we use a 2nd order Taylor approximation (2nd order because E[∥
√
hϵt∥2] ∝h), in (ii) the identity
tr(A) = Eϵt[ϵT
t Aϵt] for A ∈Rd×d, in (iii) the cyclic property of the trace, and in (iv) the symmetry of σt.
Further, we use A · B := tr(AT B) to denote the matrix inner product for matrices A, B ∈Rd×d. Therefore,
the diffusion generator is given by
Ltf(x) = 1
2σ2
t (x) · ∇2f(x).
(8.26)
Jumps.
Next, let S be arbitrary and let us consider a jump process. A jump process is defined by a
time-dependent kernel Qt(dy, x), i.e., for every 0 ≤t ≤1 and every x ∈S, Qt(dy, x) is a positive measure
over S \ {x}. The idea of a jump process is that the total volume assigned to S \ {x}
λt(x) =
Z
Qt(dy, x)
(8.27)
gives the jump intensity, i.e., the infinitesimal likelihood of jumping. Further, if λt(x) > 0, we can assign a
jump distribution by normalizing Qt to a probability kernel
Jt(dy, x) = Qt(dy, x)
λt(x)
.
(8.28)
A jump process can be approximated via the infinitesimal sampling procedure as follows:
Xt+h =
(
Xt
with probability 1 −hλt(Xt) + o(h)
∼Jt(dy, Xt)
with probability hλt(Xt) + o(h)
(8.29)
For a rigorous treatment of jump processes, see for example (Davis, 1984). The generator is then given by
Ltf(x) = lim
h→0
E[f(Xt+h) −f(Xt)|Xt = x]
h
(8.30)
= lim
h→0
E[f(Xt+h) −f(Xt)|Xt = x, Jump in [t, t + h)]P[Jump in [t, t + h)|Xt = x]
h
(8.31)
+ lim
h→0
E[f(Xt+h) −f(Xt)|Xt = x, No jump in [t, t + h)]P[No jump in [t, t + h)|Xt = x]
h
|
{z
}
=0
(8.32)
= lim
h→0
EY ∼Jt(dy,x) [f(Y ) −f(x)] hλt(x)
h
(8.33)
= EY ∼Jt(dy,x) [f(Y ) −f(x)] λt(x)
(8.34)
=
Z
(f(y) −f(x))Qt(dy, x)
(8.35)
56

where we have used that if Xt does not jump in [t, t + h], then Xt+h = Xt. Therefore, the jump generator is
given by
Ltf(x) =
Z
(f(y) −f(x))Qt(dy, x) = λt(x)EY ∼Jt(dy,x)[f(Y ) −f(x)].
(8.36)
Continuous-time Markov chain (CTMC).
Let us consider a continuous-time Markov chain Xt on a discrete
space S with |S| < ∞. In fact, this is simply a jump process on a discrete state space with a specific
parameterization. To see this, consider a vanilla jump kernel on a discrete state space S given by a matrix
Qt ∈RS×S
≥0
and using equation (8.36), the generator is given by
Ltf(x) =
X
y∈S
[f(y) −f(x)]Qt(y, x) =
X
y̸=x
[f(y) −f(x)]Qt(y, x)
for all x ∈S, f ∈RS
(8.37)
i.e., the value of Qt(x, x) does not matter and is underdetermined. Therefore, a natural convention is to
reparameterize the jump kernel on discrete state spaces by rates:
ut(y, x) =



Qt(y, x)
if y ̸= x
−P
z̸=x
Qt(z; x)
if y = x
With this, we recover the rates ut(y, x) from section 6 fulfilling the rate conditions in 6.4 by construction.
Therefore, this shows that a jump model on a discrete space coincides with the CTMC model (section 6).
Applying this on equation (8.37), we get that the CTMC generator is given by
Ltf(x) =
X
y∈S
f(y)ut(y, x) = f T ut
(8.38)
where we consider f = (f(x))x∈S as a column vector and ut ∈RS×S as a matrix. Therefore, the generator
function is simply vector multiplication from the left.
Flows on manifolds.
Next, we consider flows on Riemannian manifolds S = M as in section 5. A flow
ψ : [0, 1]×M →M is defined via a vector field u : [0, 1]×M →TM via the ODE in (3.19). Let us denote the
transition from time s to t via ψt|s(x) = ψt(ψ−1
s (x)) (as in (3.17)). Then, for a smooth function f : M →R
we have that the Riemannian flow generator is given via
Ltf(x) = lim
h→0
f(ψt+h|t(x)) −f(x)
h
=

∇f(x), d
dh

h=0ψt+h|t(x)

g
= ⟨∇f(x), ut(x)⟩g
(8.39)
where ⟨·, ·⟩g describes the dot product defining the Riemannian metric g and ∇f describes the gradient of
f with respect to g. In fact, the generator coincides with the Lie derivative of a function (Jost, 2008), a
fundamental concept in differential geometry.
8.3
Probability paths and Kolmogorov Equation
For Flow Matching on S = Rd, the Continuity Equation (see 3.25) is the central mathematical equation that
allows us to construct velocity fields that generate a desired probability path (see section 3.5). In this section,
we derive a corresponding - more general - equation for CTMPs. Let Xt be a CTMP with generator Lt and
let Xt ∼pt, then we know that:
d
dt ⟨pt, f⟩= d
dh

h=0 ⟨pt+h, f⟩= d
dh

h=0

pt,

pt+h|t, f

=

pt, d
dh

h=0

pt+h|t, f

= ⟨pt, Ltf⟩
where we used that the ⟨pt, ·⟩operation is linear to swap the derivative, and that by the tower property

pt,

pt+h|t, f

= ⟨pt+h, f⟩. This shows that given a generator Lt of a Markov process Xt we can recover its
marginal probabilities via their infinitesimal change, i.e., we arrive at the Kolmogorov Forward Equation (KFE)
57

d
dt ⟨pt, f⟩= ⟨pt, Ltf⟩for all f ∈T
(8.40)
The version of the KFE in equation (8.40) determines the evolution of expectations of test functions f. This
is necessary if we use probability distributions that do not have a density. If a density exists, a more familiar
version of the KFE can be used that directly prescribes the change of the probability densities. To present it,
we introduce the adjoint generator L∗
t , which acts on probability densities pt(x) with respect to a reference
measure ν, namely L∗
t pt(x) is (implicitly) defined by the identity
Z
pt(x)Ltf(x)ν(dx) =
Z
L∗
t pt(x)f(x)ν(dx)
∀f ∈T
(8.41)
Further, we need to assume that pt is differentiable in t. Now, (8.41) applied to the KFE (8.40) we get
Z
d
dtpt(x)f(x)ν(dx) = d
dt
Z
pt(x)f(x)ν(dx)
(8.42)
= d
dt ⟨pt, f⟩
(8.43)
= ⟨pt, Ltf⟩
(8.44)
=
Z
pt(x)Ltf(x)ν(dx)
(8.45)
=
Z
L∗
t pt(x)f(x)ν(dx)
(8.46)
As this holds for all test functions f, we can conclude using equation (8.5) that this is equivalent to the adjoint
KFE
d
dtpt(x) = L∗
t pt(x) for all x ∈S
(8.47)
As we will derive in the following examples, the adjoint KFE generalizes many famous equations used to
develop generative models such as the Continuity Equation or the Fokker-Planck Equation (Song et al.,
2021; Lipman et al., 2022) (see table 2). Whenever a probability density exists, we use the adjoint KFE - to
avoid using test functions and work with probability densities directly. We summarize our findings in the
following
Theorem 17 (General Mass Conservation). Let Lt be a generator of (Xt)0≤t≤1. Informally, the following
conditions are equivalent:
1. pt, Lt satisfies the KFE (8.40).
2.
dpt
dν (x), Lt satisfy the adjoint KFE (8.47).
3. Lt generates pt in the sense of equation (8.10).
Formally, (1) and (2) are equivalent whenever dpt
dν exists and is continuously differentiable in t. Further,
(3) implies (1) for arbitrary state spaces. There are weak regularity assumptions that ensure that (1)
implies (3) (see appendix A.3 for a list). In this work, we assume that these hold, i.e., we assume that (3)
implies (1).
To the best of our knowledge, there is no known result for abstract general state spaces that ensures that
in theorem 17 condition (3) implies (1). This is why we simply assume it here. For the machine learning
researcher, this assumption holds for any state space of interest and should therefore be of no concern (see
appendix A.3).
58

8.3.1
Examples of KFEs
Adjoint KFE for Flows.
Let us set S = Rd and assume that pt has a density pt(x) with respect to the
Lebesgue measure that is bounded and continuously differentiable. Then we can compute the adjoint generator
L∗
t via
⟨pt, Ltf⟩=Ex∼pt[Ltf(x)]
(8.48)
=
Z
Ltf(x)pt(x)dx
(8.49)
(i)
=
Z
∇f(x)T ut(x)pt(x)dx
(8.50)
(ii)
=
Z
f(x) [−div(ptut)(x)]
|
{z
}
=:L∗
t pt(x)
dx
(8.51)
=
Z
f(x)L∗
t pt(x)dx
(8.52)
where (i) follows by equation (8.15) and (ii) by integration by parts. The above derivation shows that the
adjoint generator is given by L∗
t pt = −div(ptut)(x) (because it fulfils the condition in equation (8.41)). Using
the adjoint KFE, we recover the Continuity Equation (see equation (3.25))
d
dtpt(x) = −div(ptut)(x),
(8.53)
an equation that we extensively studied in section 3.4.
Adjoint for diffusion.
Let’s set S = Rd and assume that pt has a density pt(x) with respect to the Lebesgue
measure that is bounded and continuously differentiable. We can compute the adjoint generator L∗
t via
⟨pt, Ltf⟩=Ex∼pt[Ltf(x)]
(8.54)
=
Z
Ltf(x)pt(x)dx
(8.55)
(i)
= 1
2
Z
σ2
t (x) · ∇2f(x)pt(x)dx
(8.56)
(ii)
=
Z
f(x) 1
2∇2 · (ptσ2
t )(x)
|
{z
}
=:L∗
t pt(x)
dx
(8.57)
=
Z
f(x)L∗
t pt(x)dx
(8.58)
by (i) follows by equation (8.26) and (ii) follows by applying integration by parts twice. The above derivation
shows that the adjoint generator is given by L∗
t pt =
1
2∇2 · (ptσ2
t )(x) (because it fulfils the condition in
equation (8.41)). The adjoint KFE then recovers the well-known Fokker-Planck equation
d
dtpt(x) = 1
2∇2 · (ptσ2
t )(x)
(8.59)
Adjoint KFE for jumps.
Let’s assume that pt has a density pt(x) with respect to the Lebesgue measure that
is bounded and continuously differentiable. Let’s assume that the jump measures Qt(dy, x) is given via a
kernel Qt : S × S →R≥0, (y, x) 7→Qt(y, x) such that
Z
f(y)Qt(dy, x) =
Z
f(y)Qt(y, x)ν(dy) for all integrable f : S →R.
(8.60)
59

Then we can derive the adjoint generator as follows:
⟨pt, Ltf⟩(x) =
Z Z
(f(y) −f(x))Qt(y, x)ν(dy)pt(x)ν(dx)
(8.61)
=
Z Z
f(y)Qt(y, x)pt(x)ν(dy)ν(dx) −
Z Z
f(x)Qt(y, x)pt(x)ν(dy)ν(dx)
(8.62)
(i)
=
Z Z
f(x)Qt(x, y)pt(y)ν(dy)ν(dx) −
Z Z
f(x)Qt(y, x)pt(x)ν(dy)ν(dx)
(8.63)
=
Z
f(x)
Z
Qt(x, y)pt(y) −Qt(y, x)pt(x)ν(dy)

|
{z
}
=:L∗
t pt
ν(dx)
(8.64)
=
Z
f(x)L∗
t pt(x)ν(dx)
(8.65)
where in (i) we simply swap the variables x and y. The above derivation shows that L∗
t as defined above
fulfils the condition in equation (8.41) and indeed describes the adjoint generator for jumps. With this, the
adjoint KFE becomes the Jump Continuity equation:
d
dtpt(x) =
Z
[Qt(x, y)pt(y) −Qt(y, x)pt(x)] ν(dy) =
Z
λt(y)Jt(x, y)pt(y)ν(dy) −λt(x)pt(x)
(8.66)
where we use the decomposition Qt(y, x) = λt(x)Jt(y, x) into a jump intensity λt and a jump distribution Jt
(see equation (8.28)).
Adjoint KFE for CTMCs.
For S discrete and generator given by f T ut as in equation (8.38), we get that
⟨pt, Ltf⟩=
Z
pt(x)Ltf(x)ν(dx) =
X
x∈S
pt(x)
X
y∈S
ut(y, x)f(y)
=
X
y∈S
[
X
x∈S
pt(x)ut(y, x)]
|
{z
}
=:L∗
t pt(x)
f(y)
=
Z
L∗
t pt(y)f(y)ν(dy)
where ν here just denotes the counting measure. Therefore, the adjoint is KFE simply given by
d
dtpt(x) =
X
y∈S
ut(x, y)pt(y)
(8.67)
This recovers the KFE for CTMCs as derived in equation (6.8) (with x and y switched to keep consistency
with the derivations in this section).
60

8.4
Universal representation theorem
Generators allow us to characterize the space of possible Markov processes. Specifically, the following result
allows us to not only characterize a wide class of CTMP generative models but to characterize the design
space exhaustively for S = Rd or S discrete.
Theorem 18 (Universal characterization of generators). Under weak regularity assumptions, the generators
of a Feller processes Xt (0 ≤t ≤1) take the form:
1. Discrete |S| < ∞: The generator is given by a rate transition matrix ut and the Markov process
corresponds to a continuous-time Markov chain (CTMC).
2. Euclidean space S = Rd: The generator has a representation as a sum of components described in table 2,
i.e.,
Ltf(x) = ∇f(x)T ut(x)
|
{z
}
flow
+ 1
2∇2f(x) · σ2
t (x)
|
{z
}
diffusion
+
Z
[f(y) −f(x)] Qt(dy, x)
|
{z
}
jump
(8.68)
where u : [0, 1] × Rd →Rd is a velocity field, σ : [0, 1] × Rd →S++
d
the diffusion coefficient (S++
d
denotes
the positive semi-definite matrices), and Qt(dy; x) a jump measure; ∇2f(x) describes the Hessian of f
and ∇2f(x) · σ2
t (x) describes the Frobenius inner product.
The proof adapts a known result in the mathematical literature (Courrege, 1965; von Waldenfels, 1965) and
can be found in (Holderrieth et al., 2024).
61

9
Generator Matching
In this section, we describe Generator Matching (GM) (Holderrieth et al., 2024), a generative modeling
framework for (1) arbitrary data modalities and (2) general Markov processes. GM unifies the vast majority
of generative models developed in recent years, including diffusion models, “discrete diffusion” models, and
the FM variants described in previous sections. To introduce GM, we defined the CTMP generative model in
section 8 that is constructed via a generator of a Markov process. GM describes a scalable algorithm to train
generators - giving the method its name. Beyond providing a unifying framework, GM gives rise to a variety
of new models, allows us to combine models of different classes, as well as allows to build models for arbitrary
modalities including models across multiple data modalities.
9.1
Data and coupling
As before, our goal is to transfer samples X0 ∼p from a distribution p to samples X1 ∼q from a target
distribution q, where X0, X1 ∈S are two RVs each taking values in the state space S. Source and target
samples can be related by means of the independent coupling (X0, X1) ∼p ⊗q (product distribution), or
associated by means of a general PMF coupling π0,1, i.e., distribution over S × S with marginal π0 = p
and π1 = q. The only difference to before is that S is a general state space and that p, q can be arbitrary
probability measures.
9.2
General probability paths
The next step in the GM recipe is, as before, to prescribe a probability path pt interpolating p and q.
Following section 4.4, we use a conditional probability path pt|Z(dx|z), i.e., a set of time-varying probability
measures dependent on a latent state z ∈Z. Given a distribution pZ over Z, we consider the corresponding
marginal probability path pt(dx) defined via the hierarchical sampling procedure:
Z ∼pZ, Xt ∼pt|Z(dx|z)
⇒
Xt ∼pt(dx)
i.e., we obtain a sample from pt by first sampling Z from pZ and then sampling Xt from pt|Z(dx|z). As before,
the marginal probability path is constructed to satisfy the boundary constraints p0 = p and p1 = q.
We have seen already two common constructions for Z = S and pZ = q: First, affine conditional flows for
S = Rd (as used in continuous FM; section 4) defined via
Z ∼q, X0 ∼p, Xt = σtX0 + αtZ
⇒
Xt ∼pt(dx)
(9.1)
where αt, σt ∈R≥0 are differentiable functions satisfying α0 = σ1 = 0 and α1 = σ0 = 1. Second, for arbitrary
S, we can use mixtures as used in discrete FM for discrete state spaces (equation (7.22)):
Z ∼q, X0 ∼p, Xt ∼
(
Z
with prob κt
X0
with prob (1 −κt)
⇒
Xt ∼pt(dx)
(9.2)
where κt ∈R≥0 is a differentiable functions satisfying κ0 = 0 and κ1 = 1 and 0 ≤κt ≤1. One can easily see
that the affine conditional and mixture probability paths interpolate p and q, i.e., p0 = p and p1 = q.
9.3
Parameterizing a generator via a neural network
Given a probability path pt, our goal is construct a CTMP model specified by a generator Lt that generates
this probability path (see equation (8.10)). To train a neural network for that, we first need to explain how to
parameterize a generator Lt with a neural network Lθ
t with parameters θ. We will do this in this section.
Let T again be a family of test functions (see section 8.2.1). A linear parameterization of Lt is defined as
follows: for every x ∈S there is (1) a convex closed set Ωx ⊂Vx that is a subset of a vector space Vx with an
inner product ⟨·, ·⟩x and (2) a linear operator K : T →C(S; Vx) such that every considered generator Lt can
be written as
62

Ltf(x) = ⟨Kf(x), Ft(x)⟩x
(9.3)
for a function Ft such that Ft(x) ∈Ωx for every x ∈S. Crucially, the operator K cannot depend on Lt, i.e.,
only Ft has to be learned. This leads to the
Parameterized generator: Lθ
tf(x) =

Kf(x), F θ
t (x)

x with neural network F θ
t and parameters θ, (9.4)
where again F θ
t maps an element x ∈S to F θ
t (x) ∈Ωx. We list several examples to make this definition more
concrete.
Linear parameterization of flows.
Let S = Rd and Ωx = Rd = Vx. Let’s consider all flows, i.e., the family of
generators is given by (see equation (8.15)):
Ltf = ∇f T ut,
ut : Rd →Rd.
(9.5)
Setting Kf = ∇f and Ft = ut we recover the shape of equation (9.3). This gives a natural linear parameteri-
zation of flow generators via their vector fields.
Linear parameterization of diffusion.
Let S = Rd and Ωx = S++
d
⊂Rd×d = Vx, where S++
d
denotes the set
of all positive semi-definite matrices. Then a diffusion generator is given by (see equation (8.26)):
Ltf = ∇2f · σ2
t ,
σt : Rd →S++
d
(9.6)
Setting Kf = ∇2f and Ft = σ2
t we recover the shape of equation (9.3).
This gives a natural linear
parameterization of diffusion generators.
Linear parameterization of jumps.
Let Ωx = {a : S \ {x} →R≥0 | a integrable} ⊂L2(S \ {x}) = Vx with dot
product ⟨a, b⟩x =
R
S\{x}
a(x)b(x)ν(dx). Then the jump generator is given by (see equation (8.36)):
Ltf(x) =
Z
[f(y) −f(x)]Qt(y, x)ν(dy) = ⟨Kf(x), Qt(·; x)⟩x
(9.7)
where we set Kf(x) as the function y 7→f(y) −f(x). Setting Ft = Qt we recover the shape of equation (9.3) -
giving a linear parameterization of jump generators. We note that the above only parameterizes jumps with a
jump kernel Qt(y, x), which does not necessarily include all jump measures.
Linear parameterization of CTMCs.
Let S be discrete and ut ∈RS×S be a rate matrix of a continuous-time
Markov chain. As for discrete FM (see equation (7.5)), we define
Ωx =


v ∈RS
 v(y) ≥0 ∀y ̸= x, and v(x) = −
X
y̸=x
v(y)


⊂Vx = RS.
(9.8)
Then by equation (8.38), the generator is given for f ∈RS by
Ltf(x) = f T ut(·, x) = ⟨f, ut(·, x)⟩x
(9.9)
where Vx = RS and Kf = f and ⟨·, ·⟩x is the standard Euclidean dot product. With this, we recover the
shape of equation (9.3). Therefore, this gives a natural linear parameterization of CTMCs via their rates ut.
Linear parameterization of flows on manifolds.
Let S = M be a Riemannian manifold and as in section 5,
let us consider flows on Riemannian manifolds. By equation (8.39), the generator is given by
Ltf(x) = ⟨∇f(x), ut(x)⟩g
(9.10)
with ut being a time-dependent smooth vector field ut : [0, 1] × M →TM, and ut(x) ∈TxM for all x ∈M.
Setting Ωx = Vx = TxM and K = ∇f the gradient operator, we recover the shape of equation (9.3). Therefore,
this gives a natural linear parameterization of Riemannian flow generators.
63

9.4
Marginal and conditional generators
In this section, we show how to find generators for marginal probability paths. The recipe is as follows: We
can find generators for conditional probability paths pt|Z(dx|z), often analytically, and use these to construct
generators for the marginal path. Specifically, let us assume that for every z ∈Z we found a (conditional)
generator Lz that generates pt|Z(dx|z), i.e., by theorem 17 this equivalent to the KFE (equation (8.40)):
d
dt

pt|Z(·|z), f

=

pt|Z(·|z), Lz
t f

for all f ∈T .
(9.11)
Further, let us assume that we found a linear parameterization (see equation (9.3)) as follows:
Lz
t f(x) = ⟨Kf(x), Ft(x|z)⟩x
z ∈Z
(9.12)
for functions Ft(x|z) ∈Ωx ⊂Vx. For example, Ft(x|z) could be conditional velocity field in continuous FM
(see section 4.3) or the conditional rates in discrete FM (see equation (7.2)). This allows us to find a formula
for a generator that generates the marginal path:
Theorem 19 (General Marginalization Trick). The marginal probability path (pt)0≤t≤1 is generated by a
Markov process Xt with generator
Ltf(x) = EZ∼pZ|t(·|x)[LZ
t f(x)]
(9.13)
where pZ|t(dz|x) is the posterior distribution (i.e., the conditional distribution of z given x). The generator
Lt has a linear parameterization given by
Ft(x) = EZ∼pZ|t(·|x)[Ft(x|Z)].
(9.14)
The above theorem gives us our training target: to approximate Lt in equation (9.13) with a neural network.
The marginalization tricks seen in previous chapters (theorem 3, theorem 10, theorem 14) are special cases of
this theorem. We give a proof here and then show a few examples of novel instantiations.
Proof. To prove that Lt generates pt, we need to show by theorem 17 that the KFE is fulfilled. Let pt+h|t(·|x, z)
be the transition kernel of Lz
t . Then:
d
dt ⟨pt, f⟩= lim
h→0
1
h (⟨pt+h, f⟩−⟨pt, f⟩)
= lim
h→0
1
h(EZ∼pZ,X′∼pt+h|Z(·|Z)(f(X′)) −EZ∼pZ,X∼pt|Z(·|z)(f(X)))
= lim
h→0
1
h(EZ∼pZ,X∼pt|Z(·|Z),X′∼pt+h|t(·|X,Z)(f(X′)) −EZ∼pZ,X∼pt|Z(·|z)(f(X)))
= lim
h→0
1
h(EZ∼pZ,X∼pt|Z(·|Z),X′∼pt+h|t(·|X,Z)(f(X′) −f(X)))
= lim
h→0
1
hEX∼pt

EZ∼pt|Z(·|X)

E(X′∼pt+h|t(·|X,Z)(f(X′)) −f(X)

= EX∼pt

EZ∼pZ|t(·|X)

lim
h→0
1
h

EX′∼pt+h|t(·|X,Z)(f(X′)) −f(X)

= EX∼pt(EZ∼pZ|t(·|X)(Lz
t f(X))
|
{z
}
=:Ltf(X)
)
= ⟨pt, Ltf⟩
The proof for the form of Ft follows then by
EZ∼pZ|t(·|X)(Lz
t f(x))
(9.12)
=
EZ∼pZ|t(·|X)(⟨Kf(x), Ft(x|z)⟩x) =
D
Kf(x), EZ∼pZ|t(·|X)(Ft(x|z))
E
x
= ⟨Kf(x), Ft(x)⟩x
where we use the linearity of the dot product to swap it with the expected value. This shows that Ft is a
linear parameterization (see equation (9.3)) of the marginal generator.
64

Example - Jumps.
Let S be arbitrary and Qt(y, x|z) be a conditional jump kernel on S for y, x ∈S, z ∈Z
generating the conditional probability path pt|Z(dx|z). Using the linear parameterization of the jump kernel
(see equation (9.7)), we get that the marginal jump kernel
Qt(y, x) = EZ∼pZ|t(·|x)[Qt(y, x|z)].
generates the marginal probability pt(dx).
Example - Marginal diffusion coefficient.
Let S = Rd and σ2
t (x|z) be a diffusion coefficient generating the
conditional probability path pt|Z(dx|z). Using the linear parameterization of the diffusion coefficient (see
equation (9.6)), we get that the marginal diffusion coefficient
σ2
t (x) = EZ∼pZ|t(·|x)[σ2
t (x|Z)]
generates the marginal probability path pt(dx).
9.5
Generator Matching loss
Our next goal is to develop a training objective for learning a CTMP model. Let us assume that we have
a neural network F θ
t that gives us a generator parameterization Lθ
t as in equation (9.4). As derived in
theorem 19, our goal is to approximate the true marginal linear parameterization Ft given by equation (9.14).
As before, let us assume that for every x ∈S we have a Bregman divergence Dx : Ωx × Ωx →R defined via
Dx(a, b) = Φx(a) −[Φx(b) + ⟨a −b, ∇Φx(b)⟩],
a, b ∈Ωx
(9.15)
for a strictly convex function Φx : Ωx →R (see figure 10). The Generator Matching loss to train the CTMP
model is defined as
LGM(θ) = Et,Xt∼ptDXt(Ft(Xt), F θ
t (Xt)),
(9.16)
for t ∼U[0, 1]. Unfortunately, the above training objective is intractable as we do not know the marginal
generator Lt and also not the parameterization Ft thereof (we only know the intractable formula in equa-
tion (9.14)). Hence, we introduce the Conditional Generator Matching loss as a tractable alternative that takes
the form
LCGM(θ) = Et,Z,Xt∼pt|ZDXt(Ft(Xt|Z), F θ
t (Xt)).
(9.17)
This objective is tractable as we can derive Ft(x|z) analytically in many cases (see section 9.6). As the next
theorem shows, the two losses (9.16) and (9.17) both provide the same learning gradients.
Theorem 20. The gradients of the Generator Matching loss and the Conditional Generator Matching
loss coincide:
∇θLGM(θ) = ∇θLCGM(θ).
(9.18)
In particular, the minimizer of the Conditional Generator Matching loss is the linear parameterization of
the marginal generator (equation (9.14)):
F θ
t (x) = EZ∼pZ|t(·|x)[Ft(x|Z)].
(9.19)
Furthermore, for these properties to hold, Dx must necessarily be a Bregman divergence.
The above theorem generalizes theorem 4, theorem 11, and theorem 15 derived in previous sections to
general CTMP models. It allows us to easily train any CTMP model parameterized by a neural network
F θ
t in a scalable fashion by minimizing the Conditional Generator Matching loss. In addition, it universally
characterizes the space of loss functions. The proof of theorem 20 is identical as the proof of theorem 4 with
ut replaced by Ft. For the proof of the necessity of D being a Bregman divergence, we refer to (Holderrieth
et al., 2024).
65

Example - Training a diffusion coefficient.
We illustrate how theorem 20 allows us to train a diffusion
coefficient of an SDE. Let S = Rd and σ2
t (x|z) be a diffusion coefficient generating the conditional probability
path pt|Z(dx|z). We can parameterize the diffusion coefficient in a neural network (σ2
t )θ(x) ∈Rd×d. The
Conditional Generator Matching loss then becomes
LCGM(θ) = Et,Z,Xt∼pt|Z∥σ2
t (Xt|Z) −(σ2
t )θ(Xt)∥2
where we used the mean squared error as a Bregman divergence (many other options are possible). In
(Holderrieth et al., 2024), examples are shown of models trained in this manner.
9.6
Finding conditional generators as solutions to the KFE
To enable scalable training with the Conditional Generator Matching loss (see theorem 20), we need to be
able to find a conditional generator Lz
t that solves the KFE
d
dt

pt|Z(·|z), f

=

pt|Z(·|z), Lz
t f

for all f ∈T , z ∈Z.
(9.20)
If pt|Z(dx|z) admits a density pt|Z(x|z) with respect to ν. In this case, we can equivalently solve the adjoint
KFE
d
dtpt|Z(x|z) = [(Lz
t )∗pt|Z(·|z)](x)
for all x ∈S, z ∈Z.
(9.21)
In general, equation (9.20) and equation (9.21) are challenging equations to solve analytically and there is no
general formula to solve it for arbitrary generators. Therefore, we give 2 examples on how this can be done as
illustration.
We illustrate it with jump models here, as these work for arbitrary state spaces. As explained in section 8.2.2,
they are specified by a jump measure Qt that can be decomposed into
Qt(dy, x) = λt(x)Jt(dy, x)
for all x ∈S
(9.22)
λt(x) ≥0
for all x ∈S
(9.23)
Z
Jt(dy, x) = 1
for all x ∈S
(9.24)
where λt(x) describes the jump intensity and Jt describes a probability kernel specifying the jump distribution.
Note that we drop the dependency on z ∈Z to simplify notation (we keep it for pt|Z(dx|z) to avoid confusion
with the marginal probability path).
Jump models for convex mixtures.
Let us consider the mixture probability path given by (see equation (7.22)):
pt|Z(dx|z) = κtδz(dx) + (1 −κt)p(dx),
z ∈S.
(9.25)
Using the form of the generator for jump processes (see equation (8.36)), the KFE becomes:
d
dt

pt|Z(dx|z), f

= EX∼pt|Z(·|z)λt(X)EY ∼Jt(dy,x)[f(Y ) −f(X)]
for all f ∈T , x ∈S
(9.26)
for λt, Jt satisfying the constraints in equation (9.23) and equation (9.24). We make the claim that this is
satisfied for a jump model with
Qt(dy, x) = λt(x)Jt(dy, x),
λt(x) =
˙κt
1 −κt
,
Jt(dy, x) = δz(dy)
66

i.e., the jump intensity is given by λt and once we decided to jump, we jump straight to z ∈S. To show this,
we show that the above jump process fulfils the KFE. We can derive:
EX∼pt|Z(·|z)

λt(X)EY ∼Jt(·,X)[f(Y ) −f(X)]

=
˙κt
1 −κt
EX∼pt|Z(·|z)[f(z) −f(X)]
=
˙κt
1 −κt
[f(z) −EX∼pt|Z(·|z)[f(X)]]
=
˙κt
1 −κt
[f(z) −[κtf(z) + (1 −κt)EX∼pf(X)]
= ˙κtf(z) −˙κtEX∼pf(X)
= d
dt [κtf(z) + (1 −κt)Ex∼p[f(x)]]
= d
dtEX∼pt|Z(·|z)[f(X)]
= d
dt

pt|Z(·|z), f

.
Therefore, we see that the process fulfills the jump KFE (equation (9.26)). Therefore, we have established a
jump model. We have seen a special example of that model for discrete state spaces in equation (7.24). Here,
we have shown that one could also build a similar jump model for Euclidean space Rd, for example.
Jump models for arbitrary paths with densities.
Let us assume that we have a probability pt|Z(dx|z) that
admits a density pt|Z(x|z) with respect to a reference measure ν on S and that is differentiable in t (note that
the mixture path in equation (9.25) would not fulfill that for S = Rd). Further, we restrict ourselves to jump
kernels Jt(y, x) that admit a density. With this, the adjoint KFE becomes the Jump Continuity Equation
(equation (8.66)):
d
dtpt|Z(x|z) =
Z
λt(y)Jt(x, y)pt|Z(y|z)dy −pt|Z(x|z)λt(x)
(9.27)
⇔
pt|Z(x|z)
 d
dt log pt|Z(x|z) + λt(x)

=
Z
λt(y)Jt(x, y)pt|Z(y|z)dy
(9.28)
Making Jt(x, y) = Jt(x) (“target-state-independent”) and using ∂t = d
dt, we get that this equivalent to:
pt|Z(x|z)

∂t log pt|Z(x|z) + λt(x)

= Jt(x)
Z
λt(y)pt|Z(y|z)ν(dy)
(9.29)
⇔
pt|Z(x|z)

∂t log pt|Z(x|z) + λt(x)

R
λt(y)pt|Z(y|z)ν(dy)
= Jt(x)
(9.30)
In order to define a valid jump process, we require λt, Jt to satisfy λt(x) ≥0, Jt(x) ≥0. Therefore, we get:
λt(x) ≥0, Jt(x) ≥0
⇔
λt(x) ≥[−∂t log pt(x|z)]+
(9.31)
where [x]+ = max(x, 0) describes the ReLU operation.
Further, we require Jt to define a valid jump
distribution, i.e., integrate to 1. This can be seen to hold:
1 =
Z
Jt(x)dx
⇔
Z
λt(x)pt|Z(x|z)ν(dx) =
Z
pt|Z(x|z)

∂t log pt|Z(x|z) + λt(x)

ν(dx)
⇔
0 =
Z
∂tpt|Z(x|z)ν(dx)
⇔
0 = ∂t
Z
pt|Z(x|z)ν(dx)
⇔
0 = 0
67

i.e., Jt indeed integrates to 1. Choosing the minimal λt(x), we get that a jump model defined via
λt(x) =

−∂t log pt|Z(x|z)

+ ,
Jt(x) =
pt|Z(x|z)[∂t log pt|Z(x|z)]+
R
pt|Z(y|z)[∂t log pt|Z(y|z)]+ν(dy) =
[∂tpt|Z(x|z)]+
R
[∂tpt|Z(y|z)]+ν(dy)
is a solution to the jump continuity equation and therefore generates the conditional probability path pt|Z(x|z).
At first, it seems dissatisfying that jump distribution is independent of the location. However, if we extend
this model to multiple dimensions, the jump distribution will depend on the location and leads to a powerful
generation model (Holderrieth et al., 2024).
9.7
Combining Models
In this section, we explain how GM allows us to combine generative models for the same state space S
in different ways. The underlying principle is simple: the generator is a linear operator and the KFE
∂t ⟨pt, f⟩= ⟨pt, Ltf⟩is a linear equation - so essentially, we can combine solutions for this equation like we do
for matrix equations in linear algebra. Specifically, let Lt, L′
t be two generators of two Markov processes that
solve the KFE for a probability path pt. Then for α1
t, α2
t ∈R with α1
t + α2
t = 1 it holds that:

pt, (α1
tLt + α2
tL′
t)f

(9.32)
= α1
t ⟨pt, Ltf⟩+ α2
t ⟨pt, L′
tf⟩
(9.33)
= α1
t∂t ⟨pt, f⟩+ α2
t∂t ⟨pt, f⟩
(9.34)
= (α1
t + α2
t)∂t ⟨pt, f⟩
(9.35)
= ∂t ⟨pt, f⟩,
(9.36)
i.e., α1
tLt + α2
tL′
t is again a solution of the KFE. A small but important detail is whether α1
t, α2
t are positive
or negative and whether Lt, L′
t correspond to Markov processes in forward or backward time. This leads to
the following
Proposition 3 (Combining models). Let pt be a marginal probability path, then the following generators
solve the KFE for pt and consequently define a generative model with pt as marginal:
1. Markov superposition: α1
tLt + α2
tL′
t, where Lt, L′
t are two generators of Markov processes solving
the KFE for pt, and α1
t, α2
t ≥0 satisfy α1
t + α2
t = 1.
2. Divergence-free components: Lt + βtLdiv
t
, where Ldiv
t
is a generator such that

pt, Ldiv
t
f

= 0 for
all f ∈T , and βt ≥0. We call such Ldiv
t
divergence-free.
3. Predictor-corrector: α1
tLt + α2
t ¯Lt, where Lt is a generator solving the KFE for pt in forward-time
and ¯Lt is a generator solving the KFE in backward time, and α1
t, α2
t ≥0 with α1
t −α2
t = 1.
We give examples for a Markov superposition and a divergenc-free component here to illustrate proposition 3.
An example of the power of a predictor-corrector scheme can be found in (Gat et al., 2024).
Markov superposition example - combining jump and flow.
Markov superpositions can be used to combine
generative models of different classes. These can be 2 networks trained separately or we can train two GM
models in one network simultaneously (Holderrieth et al., 2024). We illustrate this here by combining a jump
model and a flow model on S = Rd. Let us assume that we have two models where each of them generates the
probability path pt: (1) a flow model ut and (2) a jump model with jump intensity λt and jump distribution
Jt. By proposition 3, for α1
t, α2
t ≥0 with α1
t + α2
t = 1, it holds that the following generator defines a valid
GM model generating pt:
Ltf(x) = α1
tLjump
t
f(x) + α2
tLflow
t
f(x)
= (α1
tλt(x))EY ∼Jt(·,x)[f(Y ) −f(x)] + ∇f T (x)(α2
tut(x))
where we have used equation (8.17) and equation (8.36). In fact, the above generator describes a piecewise-
deterministic Markov process, a combined ODE and jump model (Davis, 1984). As the equation above shows,
68

we have to scale the jump intensity by α1
t and the vector field by α2
t . We can sample from the resulting GM
model with the following sampling procedure:
X0 ∼p0 = p
Xt+h =
(
∼Jt(dy, Xt)
with probability hα1
tλt(Xt)
Xt + hα2
tut(Xt)
with probability 1 −hα1
tλt(Xt)
In (Holderrieth et al., 2024), several examples of Markov superpositions of jump and flow are given and shown
to lead to performance improvements.
Divergence-free example - MCMC algorithms.
To find divergence-free components, one can use existing
Markov-Chain Monte-Carlo (MCMC) algorithms - all of these algorithms describe a general recipe to find a
divergence-free component. We illustrate this with 2 famous examples. Let us assume that we are given a
general probability path pt with density pt(x). Then for a generator Ldiv
t
to be divergence-free is equivalent
to that its adjoint maps pt to zero:

pt, Ldiv
t
f

= 0 for all f ∈T
⇔
[Ldiv
t
]∗pt(x) = 0 for all x ∈S
(9.37)
First, let us consider S = Rd. Langevin dynamics correspond to an SDE with velocity field 1
2β2
t ∇log pt(x) and
diffusion coefficient βt, i.e., the dynamics are given via
dXt = 1
2β2
t ∇log pt(x)dt + βtdWt
(9.38)
The adjoint generator of this SDE is given by
[Ldiv
t
]∗pt
(i)
= −div(pt
1
2β2
t ∇log pt)(x) + 1
2β2
t ∆pt(x)
(ii)
= −1
2div(β2
t ∇pt)(x) + 1
2β2
t ∆pt(x)
(iii)
= −1
2β2
t ∆pt(x) + 1
2β2
t ∆pt(x) = 0
where (i) holds by shape of flow and diffusion adjoint derived in section 8.3.1, (ii) holds because ∇log pt =
∇pt/pt, and (iii) holds by the identity div∇= ∆. The above shows that the generator of Langevin dynamics
fulfils equation (9.37) and is therefore divergence-free in sense of proposition 3. This fact is widely applied in
statistical physics and Markov chain Monte Carlo (Roberts and Tweedie, 1996). Proposition 3 shows that we
can add these dynamics for arbitrary βt ≥0 to any generative model. In section 10, we use this to derive
stochastic sampling for diffusion models.
Second, let S be a general state space again. The Metropolis-Hastings algorithm (Hastings, 1970) wdescribes
the construction of a jump process with jump kernel Qt(y, x) that satisfies the detailed balance condition:
Qt(y, x)pt(x) = Qt(x, y)pt(y)
for all x, y ∈S
⇒
[Ldiv
t
]∗pt(x)
(i)
=
Z
Qt(y, x)pt(x) −Qt(x, y)pt(y) = 0
where in (i) we used equation (8.66). This shows that equation (9.37) is fulfilled and Qt is divergence-free.
Proposition 3 shows that one can arbitrarily add such a Metropolis-scheme to any GM model following the
probability path pt.
69

9.8
Multimodal models
We finally comment on how GM enables the construction of generative models over multiple data modalities
jointly. For example, this could be a model that generates images and corresponding text descriptions at the
same time. Two modalities are represented as two state spaces S1, S2 (e.g., S1 are images and S2 is text)
and a multimodal model would be a generative model over the product space S = S1 × S2. As S is just
another state space and GM works for arbitrary state spaces, we could simply go about it like constructing
any other GM model. However, there is a specific construction of probability paths that allows us to reuse or
“recycle” GM models built for individual modalities. For example, we could build a joint text-image model by
combining a discrete and continuous FM model. This specific construction relies on factorized conditional
probability paths. We have seen a simple case of this already in section 7.5.2 for discrete FM where factorized
probability paths lead to factorized velocities. This holds more generally for arbitrary modalities. While
the construction is rather simple and intuitive, it is rather technical to express in full generality. We refer
to (Holderrieth et al., 2024) for a rigorous treatment. A specific instantiation of this was also realized in
(Campbell et al., 2024) for multimodal protein generation. This highlights that GM enables the construction
of multimodal models in a principled and rigorous manner.
70

10
Relation to Diffusion and other Denoising Models
In this section, we finally discuss the relation to denoising diffusion models (DDMs) and related models on
non-Euclidean spaces. We mainly focus here on the construction of diffusion models by (Song et al., 2021) via
SDEs and explain how it can be placed with the FM/GM family of models. At the end of the section, we also
discuss models in other modalities that took inspiration from DDMs (“denoising models”) and how they can
be framed as a GM model.
10.1
Time convention
The first simple difference between denoising diffusion models and flow matching is a difference how time
is parameterized. This is just a convention but is important to avoid confusion. Different to FM, time is
inverted in diffusion models and ranges from 0 to ∞. To differentiate the two time parameterizations, let us
use r for the time convention of diffusion models and t for the time convention of FM. Then we have:
FM time t: Noise ≡“t = 0”, Data ≡“t = 1”
(10.1)
Diffusion time r: Noise ≡“r = +∞”, Data ≡“r = 0”
(10.2)
Reparameterization: r = k(t),
t = k−1(r)
(10.3)
where k : (0, 1] →[0, +∞) is some strictly monotonically decreasing mapping with k(1) = 0 and limt→0 k(t) =
+∞.
10.2
Forward process vs. probability paths
The underlying idea of denoising diffusion models is to construct a forward process that corrupts the data
distribution. We will explain how this corresponds to a specific construction of a probability path as used in
FM. The forward process Xr is defined via the SDE
dXr = ar(Xr)dr + grdWr,
X0 ∼q
(10.4)
where q is the data distribution, Wr is a Brownian motion and a : R × Rd →Rd a velocity field, also called
drift in the context of SDEs, and g : R →R≥0 a diffusion coefficient (see section 8.2.2). Every such SDE
defines a conditional probability path and marginal probability path as
˜pr|0(x|z) = P[Xt = x|X0 = z],
˜pr(x) = P[Xt = x]
(10.5)
pt|1(x|z) = ˜pk(t)|0(x|z),
pt(x) = ˜pk(t)(x)
(10.6)
where in the second line we reparameterized time into the FM time parameterization. We see that pt|1(x|z)
gives a conditional probability path. Further, the forward process is constructed such that for R ≫0, the
distribution of XR is approximately a Gaussian. Therefore, we get that:
Every forward process in diffusion defines a “valid” conditional probability path as used in FM, i.e., the
corresponding marginal path interpolates between a data distribution q and (roughly) a Gaussian p.
Specifically:
(1) Deterministic initialization: The conditional probability path pt|1(x|z) corresponds to the distribution
of the forward process SDE when initialized with X0 = z.
(2) Data initialization: The marginal probability path pt(x) corresponds to the distribution of the forward
process when initialized with X0 ∼q where q is the data distribution.
An important requirement of diffusion models is that one can compute the conditional probability ˜pr|0(x|z)
in closed form. This enforces working with SDEs that have analytic solution to their respective KFE (i.e.,
Fokker-Planck equation). Throughout most of the literature, the forward process is therefore assumed to have
affine drift coefficients i.e., ar(x) = arx for some continuous function a : R →R (Song et al., 2021; Karras
71

et al., 2022). This assumption allows us to express the conditional distribution of Xr given X0 = z ∈Rd as a
Gaussian distribution (Särkkä and Solin, 2019; Song et al., 2021; Karras et al., 2022):
˜pr|0(x|z) = N
 ˜αrz, ˜σ2
rI

,
˜αr = exp


r
Z
0
awdw

,
˜σ2
r = ˜α2
r
r
Z
0
g2
w
˜α2w
dw
(10.7)
⇒
pt|1(x|z) = N
 αtz, σ2
t I

αt = ˜αk(t),
σt = ˜σk(t)
(10.8)
Note that we have discussed such probability paths extensively in section 4.8.3 as affine Gaussian probability
paths, i.e., they are constructed via the affine conditional flow (see section 4.8):
ψt(x|x1) = αtz + σtx,
z ∼q, x ∼N(α0, σ2
0I).
(10.9)
Therefore, we can see that:
Forward processes with affine drift coefficients correspond to using Gaussian probability paths (see section
4.8.3) defined by equation (10.8).
Note that for diffusion models in the above time parameterization, there is no finite times r < +∞at which
the marginal ˜pr(x) is an exact Gaussian.
10.3
Training a diffusion model
We now discuss how we can recover the training algorithm of diffusion models as a special case of FM training.
In section 4.8, we discussed several options of how to parameterize and train a FM model with Gaussian
probability paths (see theorem 7). One particular option was x0-prediction where the neural network xθ
0|t is
trained to approximate
xθ
0|t ≈E[X0|Xt = x]
via the following training algorithm
LCM(θ) = Et,Z∼q,X0∼p∥xθ
0|t(αtX0 + σtZ
|
{z
}
=Xt
) −X0∥2
(i)
= Et,Z∼q,Xt∼pt|1(·|Z)σ2
t ∥sθ
t (Xt) −[−1
σ2
t
(Xt −αtZ)]∥2
(ii)
= Et,Z∼q,Xt∼pt|1(·|Z)σ2
t ∥sθ
t(Xt) −∇log pt|1(Xt|Z)∥2,
where in (i) we reparameterized the neural network via sθ
t = −xθ
0|t/σt and in (ii) we used equation (4.71).
The above loss is also called the Denoising Score Matching (Vincent, 2011) loss and is the fundamental loss
function with which diffusion models are trained. By theorem 7, we get that the minimizer θ∗fulfills that
sθ∗
t (x) = −1
σt
E[X0|Xt = x]
(4.77)
=
∇log pt(x),
(10.10)
i.e., at minimal loss, sθ
t equals the score function ∇log pt(x) of the marginal probability path. Therefore, the
network sθ
t is also called the score network. We can therefore summarize:
The training algorithm for diffusion models is equivalent to training a specific FM model with x0-prediction.
Specifically, in addition to reparameterizing time, it is the same as training a FM model with:
(1) Probability path: Using a Gaussian probability path with independent coupling constructed via an
SDE with affine drift coefficients (αt, σt defined via (10.7)).
(2) Score parameterization: Reparameterizing the marginal velocity field via the score function.
In table 1, we list how one can easily convert a score network to other velocity parameterizations. Therefore,
different parameterizations are theoretically equivalent and one can even swap parameterizations post-training
(see section 4.8.1). Note however, that the score and x0 prediction parameterizations introduce a singularity
at time t = 0 (near noise).
72

10.4
Sampling
Next, we discuss sampling from a diffusion model and how it relates to sampling from FM or GM model.
Deterministic sampling with ODE.
If we consider the diffusion model a FM model, we would sample by
sampling from the marginal vector field. In (4.78) we expressed the marginal vector field via the score function
(for Gaussian paths):
ut(x) = ˙αt
αt
x −

˙σtσt −σ2
t
˙αt
αt

∇log pt(x).
(10.11)
Using the specific form of equation (10.7) for αt, σt, one derive the equivalent identity:
ut(x) = ˙k(t)

atx −g2
t
2 ∇log pt(x)

.
(10.12)
Alternatively, one insert the above ut(x) into the Continuity Equation to validate this directly. The corre-
sponding ODE to ut is also called the Probability Flow ODE (Song et al., 2021):
dXt = ˙k(t)

atXt −g2
t
2 sθ
t (Xt)

dt,
(10.13)
where we set sθ
t (x) = ∇log pt(x) as the learned score function. Note that we use here the notation for ODEs
that is common for SDEs for a reason that becomes clear next. We also note that in equation (10.11) we add
the term ˙k(t) compared to (Song et al., 2021) because of the time reparameterization.
Stochastic sampling with SDE.
In section 9.7, we have derived that we can add the Langevin dynamics
1
2β2
t ∇log pt(x)dt + βtdWt
(10.14)
to any CTMP generative model and we will obtain a generative model following the same probability path.
We can apply this to the Probability Flow ODE to get a whole family of SDEs that generate the probability
path pt:
dXt =
 
˙k(t)αtXt + β2
t −˙k(t)g2
t
2
∇log pt(Xt)
!
dt + βtdWt.
(10.15)
The above results in stochastic sampling of a diffusion model. In theory, all models above lead to the same
marginals for every βt ≥0. This is a mathematical fact about the ground truth underlying stochastic process.
In practice, we need to simulate the SDE
dXt =
 
˙k(t)αtXt + β2
t −˙k(t)g2
t
2
sθ
t(Xt)
!
dt + βtdWt
(10.16)
with a trained network sθ
t . We have both estimation errors (i.e., imperfect training of sθ
t ) as well as simulation
errors (i.e., imperfect sampling of the underlying SDE). Therefore, there is an optimal unknown noise level βt
(see e.g., (Albergo and Vanden-Eijnden, 2022, equation (2.45))) that can be determined empirically (Karras
et al., 2022) and theoretically (Ma et al., 2024).
Therefore, we get:
1. ODE sampling:
For a Gaussian source, independent coupling, fixing αt, σt according to (10.7), and
score parameterization, sampling from a diffusion model with the Probability Flow ODE is the same as
sampling from a FM model.
2. SDE sampling:
Under the same conditions, sampling from a diffusion model with stochastic SDE
sampling is equivalent to sampling from GM model defined via equation (10.15).
73

10.5
The role of time-reversal and the backward process
To finish our discussion of diffusion models, we discuss the role of time-reversal and the backward process in
the context of diffusion models. Given how central the idea of time-reversal is for diffusion models, it might
seem surprising to some readers that it is not needed for FM. We explain this, therefore, here in more detail.
Specifically, to generate data, diffusion models frame training as learning a backward process ¯Xr going from 0
to R > 0 such that
¯Xr
d=XR−r for all r ∈[0, R]
(10.17)
where
d= denotes equality in distribution. Once we found such a process, we can initialize ¯X0
d= XR with a
Gaussian and simulate it to obtain ¯XR
d= X0 ∼q, i.e., a sample the data distribution q. If Xr ∼˜pr (i.e., it is
the marginal probability path as in equation (10.5)), then equation (10.17) is equivalent to:
¯Xr ∼¯pr := ˜pR−r for all r ∈[0, R].
In other words, we want the stochastic process ¯Xr to generate the probability path ¯pr. But this is exactly
what we are trying to do in Generator Matching for general Markov processes (see equation (8.10)), and in
particular in Flow Matching with flows. Therefore, we get:
The following problems are equivalent:
(1) Time-reverse marginals: Finding an SDE (resp. ODE) for the backward process that has the same
marginals as the forward process - as done for diffusion models.
(2) Generate probability path: Finding an SDE (resp. ODE) that generates the probability path defined
by the forward process.
(3) Solve KFE: Finding an SDE (resp. ODE) that solves the Fokker-Planck Equation (resp. Continuity
Equation).
The original description of diffusion models included the “full” time-reversal of an SDE (Anderson, 1982).
This is a notion that is stronger than the one we use, i.e., it requires that the joint distribution across time
points are the same
P[ ¯Xr1 ∈A1, . . . , ¯Xrn ∈An] = P[XR−r1 ∈A1, . . . , XR−rn ∈An]
(10.18)
for all 0 ≤r1, . . . , rn, and A1, . . . , An ⊂S measurable.
(10.19)
As shown in Anderson (1982), one can obtain a time-reversal satisfying the above condition with a backward
process for a specific choice of βt in equation (10.15) . However, for the purposes of generative modeling,
we often only use the final point X1 of the Markov process (e.g., as a generated image) and discard earlier
time points. Therefore, whether a Markov process is a “true” time-reversal or only has the same marginals
(as in equation (10.17)) does not matter for many applications. A famous example is the probability flow
ODE. The probability flow ODE does not constitute a time-reversal of a diffusion process in the sense of
(Anderson, 1982) but it follows the same marginals. This illustrates that finding “true” time-reversals is a
harder mathematical problem to solve but (often) not necessary for the purposes of generative modeling.
The probability flow ODE is the current state-of-the-art for sampling with a low number of neural network
evaluations (Karras et al., 2022). This shows that the “true” time-reversal might even give suboptimal results
compared to other solutions of the Fokker-Planck equation (Ma et al., 2024).
74

10.6
Relation to Other Denoising Model
Inspired by the success of diffusion models, there were significant advances in translating the methods from
diffusion models to other state spaces (Campbell et al., 2022, 2024; De Bortoli et al., 2022; Huang et al., 2022a;
Benton et al., 2022). Similarly to diffusion models, they construct a forward Markov process that noises
data and then time-reverse it (i.e., “de-noises” it). We therefore informally refer to such models as “denoising
models”. Naturally, one can ask how these models relate to GM (see section 9). In brief, the relation of these
“denoising models” to GM models is the same as the relation between diffusion models and FM:
Generally speaking, denoising models are Generator Matching models with
(1) Time convention: They use the diffusion time convention where time 0 corresponds to data.
(2) Probability path construction: Probability paths are constructed via a “forward” or “noising” process.
(3) Solving the KFE: A particular solution to the KFE is found via a time-reversal of the forward process.
We acknowledge that this an informal rule and that there might be exceptions to that rule. Therefore, we
refer to an extended and detailed discussion of such related work (including a more complete list of references)
to (Holderrieth et al., 2024).
75

References
Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. arXiv preprint
arXiv:2209.15571, 2022.
Michael Samuel Albergo, Mark Goldstein, Nicholas Matthew Boffi, Rajesh Ranganath, and Eric Vanden-Eijnden.
Stochastic interpolants with data-dependent couplings. In Proceedings of the 41st International Conference on
Machine Learning, ICML’24, 2024.
Luigi Ambrosio. Transport equation and cauchy problem forbvvector fields. Inventiones mathematicae, 158(2), 2004.
Brian DO Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):
313–326, 1982.
Heli Ben-Hamu, Samuel Cohen, Joey Bose, Brandon Amos, Maximillian Nickel, Aditya Grover, Ricky T. Q. Chen, and
Yaron Lipman. Matching normalizing flows and probability paths on manifolds. Proceedings of the 39th International
Conference on Machine Learning, 162, 2022.
Joe Benton, Yuyang Shi, Valentin De Bortoli, George Deligiannidis, and Arnaud Doucet. From denoising diffusions to
denoising markov models. arXiv preprint arXiv:2211.03595, 2022.
Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom,
Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith
Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan
Wang, and Ury Zhilinsky. π0: Vision-language-action flow model for general robot control. arXiv, 2024. https:
//www.physicalintelligence.company/download/pi0.pdf.
Vladimir I Bogachev, Nicolai V Krylov, Michael Röckner, and Stanislav V Shaposhnikov. Fokker–Planck–Kolmogorov
Equations, volume 207. American Mathematical Society, 2022.
Avishek Joey Bose, Tara Akhound-Sadegh, Guillaume Huguet, Kilian Fatras, Jarrid Rector-Brooks, Cheng-Hao Liu,
Andrei Cristian Nica, Maksym Korablyov, Michael Bronstein, and Alexander Tong. Se(3)-stochastic flow matching
for protein backbone generation. arXiv preprint arXiv:2310.02391, 2023.
Arwen Bradley and Preetum Nakkiran. Classifier-free guidance is a predictor-corrector. arXiv preprint arXiv:2408.09000,
2024.
Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A
continuous time framework for discrete denoising models. Advances in Neural Information Processing Systems, 35:
28266–28279, 2022.
Andrew Campbell, Jason Yim, Regina Barzilay, Tom Rainforth, and Tommi Jaakkola. Generative flows on discrete
state-spaces: Enabling multimodal flows with applications to protein co-design. arXiv preprint arXiv:2402.04997,
2024.
Ricky T. Q. Chen and Yaron Lipman. Flow matching on general geometries. In The Twelfth International Conference
on Learning Representations, 2024.
Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations.
Advances in neural information processing systems, 31, 2018.
Muthu Chidambaram, Khashayar Gatmiry, Sitan Chen, Holden Lee, and Jianfeng Lu. What does guidance do? a
fine-grained analysis in a simple setting. arXiv preprint arXiv:2409.13074, 2024.
Earl A Coddington, Norman Levinson, and T Teichmann. Theory of ordinary differential equations, 1956.
Philippe Courrege. Sur la forme intégro-différentielle des opérateurs de c∞
k dans c satisfaisant au principe du maximum.
Séminaire Brelot-Choquet-Deny. Théorie du Potentiel, 10(1):1–38, 1965.
Mark HA Davis. Piecewise-deterministic markov processes: A general class of non-diffusion stochastic models. Journal
of the Royal Statistical Society: Series B (Methodological), 46(3):353–376, 1984.
Valentin De Bortoli, Emile Mathieu, Michael Hutchinson, James Thornton, Yee Whye Teh, and Arnaud Doucet.
Riemannian score-based generative modelling. Advances in Neural Information Processing Systems, 35:2406–2422,
2022.
Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In Advances in Neural
Information Processing Systems, volume 34, pages 8780–8794. Curran Associates, Inc., 2021.
76

Sander Dieleman. Guidance: a cheat code for diffusion models, 2022. https://benanne.github.io/2022/05/26/
guidance.html.
Ronald J DiPerna and Pierre-Louis Lions. Ordinary differential equations, transport theory and sobolev spaces.
Inventiones mathematicae, 98(3):511–547, 1989.
Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik
Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In
Forty-first International Conference on Machine Learning, 2024.
Stewart N Ethier and Thomas G Kurtz. Markov processes: characterization and convergence. John Wiley & Sons,
2009.
William Feller. On second order differential operators. Annals of Mathematics, 61(1):90–105, 1955.
Alessio Figalli. Existence and uniqueness of martingale solutions for sdes with rough or degenerate coefficients. Journal
of Functional Analysis, 254(1):109–153, 2008.
Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky T. Q. Chen, Gabriel Synnaeve, Yossi Adi, and Yaron Lipman.
Discrete flow matching. arXiv preprint arXiv:2407.15595, 2024.
Izrail Moiseevitch Gelfand, Richard A Silverman, et al. Calculus of variations. Courier Corporation, 2000.
Will Grathwohl, Ricky T. Q. Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. Ffjord: Free-form
continuous dynamics for scalable reversible generative models. arXiv preprint arXiv:1810.01367, 2018.
Yingqing Guo, Hui Yuan, Yukang Yang, Minshuo Chen, and Mengdi Wang. Gradient guidance for diffusion models:
An optimization perspective. arXiv preprint arXiv:2404.14743, 2024.
W Keith Hastings. Monte carlo sampling methods using markov chains and their applications. 1970.
Eric Heitz, Laurent Belcour, and Thomas Chambon. Iterative α-(de) blending: A minimalist deterministic diffusion
model. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–8, 2023.
Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative
Models and Downstream Applications, 2021.
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information
Processing Systems, 33:6840–6851, 2020.
Peter Holderrieth, Marton Havasi, Jason Yim, Neta Shaul, Itai Gat, Tommi Jaakkola, Brian Karrer, Ricky Chen,
and Yaron Lipman. Generator matching: Generative modeling with arbitrary markov processes. Preprint, 2024.
http://arxiv.org/abs/2410.20587.
Chin-Wei Huang, Milad Aghajohari, Joey Bose, Prakash Panangaden, and Aaron C Courville. Riemannian diffusion
models. Advances in Neural Information Processing Systems, 35:2750–2761, 2022a.
Chin-Wei Huang, Milad Aghajohari, Joey Bose, Prakash Panangaden, and Aaron C Courville. Riemannian diffusion
models. In Advances in Neural Information Processing Systems, 2022b.
Guillaume Huguet, James Vuckovic, Kilian Fatras, Eric Thibodeau-Laufer, Pablo Lemos, Riashat Islam, Cheng-Hao Liu,
Jarrid Rector-Brooks, Tara Akhound-Sadegh, Michael Bronstein, et al. Sequence-augmented se (3)-flow matching
for conditional protein backbone generation. arXiv preprint arXiv:2405.20313, 2024.
Arieh Iserles. A first course in the numerical analysis of differential equations. Cambridge university press, 2009.
Jürgen Jost. Riemannian geometry and geometric analysis, volume 42005. Springer, 2008.
John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasu-
vunakool, Russ Bates, Augustin Žídek, Anna Potapenko, et al. Highly accurate protein structure prediction with
alphafold. nature, 596(7873):583–589, 2021.
Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative
models. Advances in Neural Information Processing Systems, 35:26565–26577, 2022.
Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. Advances in neural
information processing systems, 34:21696–21707, 2021.
Thomas G Kurtz. Equivalence of stochastic equations and martingale problems. Stochastic analysis 2010, pages
113–130, 2011.
77

Matthew Le, Apoorv Vyas, Bowen Shi, Brian Karrer, Leda Sari, Rashel Moritz, Mary Williamson, Vimal Manohar,
Yossi Adi, Jay Mahadeokar, et al. Voicebox: Text-guided multilingual universal speech generation at scale. Advances
in neural information processing systems, 36, 2024.
Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative
modeling. arXiv preprint arXiv:2210.02747, 2022.
Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A. Theodorou, Weili Nie, and Anima Anandkumar. I2sb:
image-to-image schrödinger bridge. In Proceedings of the 40th International Conference on Machine Learning,
ICML’23. JMLR.org, 2023.
Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with
rectified flow. arXiv preprint arXiv:2209.03003, 2022.
Lynn Harold Loomis and Shlomo Sternberg. Advanced calculus. World Scientific, 1968.
Aaron Lou, Derek Lim, Isay Katsman, Leo Huang, Qingxuan Jiang, Ser-Nam Lim, and Christopher De Sa. Neural
manifold ordinary differential equations. In Proceedings of the 34th International Conference on Neural Information
Processing Systems, 2020.
Aaron Lou, Minkai Xu, Adam Farris, and Stefano Ermon. Scaling riemannian diffusion models. Advances in Neural
Information Processing Systems, 36:80291–80305, 2023.
Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring
flow and diffusion-based generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740,
2024.
Emile Mathieu and Maximilian Nickel. Riemannian continuous normalizing flows. In Advances in Neural Information
Processing Systems, 2020.
Paul C Matthews. Vector calculus. Springer Science & Business Media, 2012.
Robert J McCann. A convexity principle for interacting gases. Advances in mathematics, 128(1):153–179, 1997.
Kirill Neklyudov, Rob Brekelmans, Daniel Severo, and Alireza Makhzani. Action matching: Learning stochastic
dynamics from samples. In International conference on machine learning, pages 25858–25889. PMLR, 2023.
Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In Marina Meila
and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of
Proceedings of Machine Learning Research, pages 8162–8171. PMLR, 18–24 Jul 2021.
Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya
Sutskever, and Mark Chen. GLIDE: Towards photorealistic image generation and editing with text-guided diffusion
models. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of
Machine Learning Research, pages 16784–16804. PMLR, 17–23 Jul 2022.
Bernt Øksendal. Stochastic differential equations. Springer, 2003.
Amnon Pazy. Semigroups of linear operators and applications to partial differential equations, volume 44. Springer
Science & Business Media, 2012.
Stefano Peluchetti. Non-denoising forward-time diffusions. arXiv preprint arXiv:2312.14589, 2023.
Lawrence Perko. Differential equations and dynamical systems, volume 7. Springer Science & Business Media, 2013.
Gabriel Peyré, Marco Cuturi, et al. Computational optimal transport: With applications to data science. Foundations
and Trends® in Machine Learning, 11(5-6):355–607, 2019.
Ashwini Pokle, Matthew J Muckley, Ricky T. Q. Chen, and Brian Karrer. Training-free linear image inversion via
flows. arXiv preprint arXiv:2310.04432, 2023.
Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi,
Chih-Yao Ma, Ching-Yao Chuang, David Yan, Dhruv Choudhary, Dingkang Wang, Geet Sethi, Guan Pang, Haoyu
Ma, Ishan Misra, Ji Hou, Jialiang Wang, Kiran Jagadeesh, Kunpeng Li, Luxin Zhang, Mannat Singh, Mary
Williamson, Matt Le, Matthew Yu, Mitesh Kumar Singh, Peizhao Zhang, Peter Vajda, Quentin Duval, Rohit
Girdhar, Roshan Sumbaly, Sai Saketh Rambhatla, Sam Tsai, Samaneh Azadi, Samyak Datta, Sanyuan Chen, Sean
Bell, Sharadh Ramaswamy, Shelly Sheynin, Siddharth Bhattacharya, Simran Motwani, Tao Xu, Tianhe Li, Tingbo
Hou, Wei-Ning Hsu, Xi Yin, Xiaoliang Dai, Yaniv Taigman, Yaqiao Luo, Yen-Cheng Liu, Yi-Chiao Wu, Yue Zhao,
Yuval Kirstain, Zecheng He, Zijian He, Albert Pumarola, Ali Thabet, Artsiom Sanakoyeu, Arun Mallya, Baishan
78

Guo, Boris Araya, Breena Kerr, Carleigh Wood, Ce Liu, Cen Peng, Dimitry Vengertsev, Edgar Schonfeld, Elliot
Blanchard, Felix Juefei-Xu, Fraylie Nord, Jeff Liang, John Hoffman, Jonas Kohler, Kaolin Fire, Karthik Sivakumar,
Lawrence Chen, Licheng Yu, Luya Gao, Markos Georgopoulos, Rashel Moritz, Sara K. Sampson, Shikai Li, Simone
Parmeggiani, Steve Fine, Tara Fowler, Vladan Petrovic, and Yuming Du. Movie gen: A cast of media foundation
models, 2024. https://arxiv.org/abs/2410.13720.
Aram-Alexandre Pooladian, Heli Ben-Hamu, Carles Domingo-Enrich, Brandon Amos, Yaron Lipman, and Ricky T. Q.
Chen. Multisample flow matching: Straightening flows with minibatch couplings. In International Conference on
Machine Learning, 2023.
Jan Prüss, Mathias Wilke, and Mathias Wilke. Ordinary differential equations and dynamic systems. Springer, 2010.
Gareth O Roberts and Richard L Tweedie. Exponential convergence of langevin distributions and their discrete
approximations. Bernoulli 2(4): 341-363 (December 1996), 1996.
Leonard CG Rogers and David Williams. Diffusions, markov processes, and martingales: Volume 1, foundations.
Cambridge university press, 2000.
Noam Rozen, Aditya Grover, Maximilian Nickel, and Yaron Lipman. Moser flow: Divergence-based generative modeling
on manifolds. Advances in Neural Information Processing Systems, 34:17669–17680, 2021.
Ludger Rüschendorf, Alexander Schnurr, and Viktor Wolf. Comparison of time-inhomogeneous markov processes.
Advances in Applied Probability, 48(4):1015–1044, 2016.
Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad
Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 Conference Proceedings, SIGGRAPH
’22. Association for Computing Machinery, 2022.
Tim Salimans and Jonathan Ho.
Progressive distillation for fast sampling of diffusion models.
arXiv preprint
arXiv:2202.00512, 2022.
Simo Särkkä and Arno Solin. Applied stochastic differential equations, volume 10. Cambridge University Press, 2019.
Neta Shaul, Ricky T. Q. Chen, Maximilian Nickel, Matthew Le, and Yaron Lipman. On kinetic optimal probability
paths for generative models. In International Conference on Machine Learning, pages 30883–30907. PMLR, 2023a.
Neta Shaul, Juan Perez, Ricky T. Q. Chen, Ali Thabet, Albert Pumarola, and Yaron Lipman. Bespoke solvers for
generative flow models. arXiv preprint arXiv:2310.19075, 2023b.
Neta Shaul, Itai Gat, Marton Havasi, Daniel Severo, Anuroop Sriram, Peter Holderrieth, Brian Karrer, Yaron
Lipman, and Ricky T. Q. Chen. Flow matching with general discrete paths: A kinetic-optimal perspective, 2024.
https://arxiv.org/abs/2412.03487.
Yuyang Shi, Valentin De Bortoli, Andrew Campbell, and Arnaud Doucet. Diffusion schrödinger bridge matching. In
Thirty-seventh Conference on Neural Information Processing Systems, 2023.
Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using
nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265. PMLR, 2015.
Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In Advances in
Neural Information Processing Systems, pages 11895–11907, 2019.
Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-
based generative modeling through stochastic differential equations. In International Conference on Learning
Representations (ICLR), 2021.
Alexander Tong, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Kilian Fatras, Guy Wolf,
and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport.
arXiv preprint arXiv:2302.00482, 2023.
Cédric Villani. Topics in optimal transportation, volume 58. American Mathematical Soc., 2021.
Cédric Villani et al. Optimal transport: old and new, volume 338. Springer, 2009.
Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7):
1661–1674, 2011.
Wilhelm von Waldenfels. Fast positive operatoren. Zeitschrift für Wahrscheinlichkeitstheorie und Verwandte Gebiete,
4:159–174, 1965.
79

Apoorv Vyas, Bowen Shi, Matthew Le, Andros Tjandra, Yi-Chiao Wu, Baishan Guo, Jiemin Zhang, Xinyue Zhang,
Robert Adkins, William Ngan, et al. Audiobox: Unified audio generation with natural language prompts. arXiv
preprint arXiv:2312.15821, 2023.
Jason Yim, Andrew Campbell, Andrew YK Foong, Michael Gastegger, José Jiménez-Luna, Sarah Lewis, Victor Garcia
Satorras, Bastiaan S Veeling, Regina Barzilay, Tommi Jaakkola, et al. Fast protein backbone generation with se (3)
flow matching. arXiv preprint arXiv:2310.05297, 2023.
Qinqing Zheng, Matt Le, Neta Shaul, Yaron Lipman, Aditya Grover, and Ricky T. Q. Chen. Guided flows for generative
modeling and decision making, 2023. https://arxiv.org/abs/2311.13443.
80

Appendix
A
Additional proofs
A.1
Discrete Mass Conservation
Lemma 1 (PMF solutions to Kolmogoroc with rate conditions). Consider a solution ft(x) to Kolmogorov
Equation (6.8) with initial condition f0(x) = p(x), where p is a PMF, ut(y, x) is C([0, 1]) in time t and
satisfies the rate conditions (6.4). Then ft(x) is a probability mass function (PMF) for all t ∈[0, 1].
Proof. Let ft(x), t ∈[0, 1], be the solution to the Kolmogorov Equation, the existence and uniqueness of
which is guarateed by theorem 12. Now, ft(x) is a PMF if and only if it satisfies
ft(x) ≥0, and
X
x
ft(x) = 1.
(A.1)
The latter condition is shown to hold by summing both sides of the Kolmogorov Equation to get that the
solution satisfies
d
dt
X
x
ft(x) =
X
x
X
z
ut(x, z)pt(z) = 0
where the second equality is due to P
y ut(y, x) = 0 in the rate conditions. Since P
x f0(x) = P
x p(x) = 1 we
hae that P
x ft(x) ≡1 for all t ∈[0, 1].
To prove that ft(x) ≥0 for all x ∈S we will use a result on convex invariant sets of dynamical systems. In
particular, Theorem 7.3.4 in Prüss et al. (2010) asserts that as long as f0 = p satisfies this condition (which it
does) and whenever w(z) is on the boundary of this constraint, i.e., w is a PMF and w(z) = 0 for some z ∈S,
then a non-positive inner-product with the outer-normal to the constraint, i.e., P
x,y ut(y, x)w(x)δ(y, z) ≥0
implies that the solution ft(x) ≥0 for all t ∈[0, 1] and x ∈S. Let us check this condition:
X
x,y
ut(y, x)w(x)δ(y, z) =
X
x
ut(z, x)w(x) =
X
x̸=z
ut(z, x)w(x) ≥0,
where in the second equality we use the fact that w(z) = 0 and in the last inequality we used the rate condition
(6.4) that ut(z, x) ≥0 for z ̸= x and w(y) ≥0 for all y.
Theorem 13 (Discrete Mass Conservation). Let ut(y, x) be in C([0, 1)) and pt(x) a PMF in C1([0, 1)) in
time t. Then, the following are equivalent:
1. pt, ut satisfy the Kolmogorov Equation (6.8) for t ∈[0, 1), and ut satisfies the rate conditions (6.4).
2. ut generates pt in the sense of 6.5 for t ∈[0, 1).
Proof. Let us start by assuming 2. In this case, the probability transition kernel pt+h|t(y|x) satisfies (6.3),
pt+h|t(y|x) = δ(y, x) + hut(y, x) + o(h).
(A.2)
By expressing the marginal pt(y) using the Law of Total Probability, we obtain
pt+h(y) =
X
x
pt+h|t(y|x) pt(x).
(A.3)
By plugging (A.2) into (A.3) and rearranging, we get
pt+h(y) −pt(y)
h
=
X
x
ut(y, x)pt(x) + o(1),
81

where now o(1) = o(h)/h →0 as h →0, as per the definition of o(h). Taking the limit h →0, we get
that the pair (pt, ut) satisfies the Kolmogorov Equation (6.8). Next, let us prove that ut satisfies the rate
conditions (6.4). If ut(y, x) < 0 for some y ̸= x, it follows from (A.2) that pt+h|t(y|x) < 0 for small h > 0,
and this contradicts pt+h|t being a probability kernel. If P
y ut(y, x) = c ̸= 0, it follows from A.2 that
1 = P
x pt+h|t(y|x) = 1 + hc + o(h), leading to a contradiction for small h > 0.
Conversely, assume now condition 1. That is, the pair (ut, pt) satisfies the Kolmogorov Equation (6.8) with
initial condition p0 = p. By theorem 12, let ps|t(y|x) be the unique solution to the Kolomogorov Equation
d
dsps|t(y|x) =
X
z
ut(y, z)ps|t(z|x),
(A.4)
with initial condition pt|t(y, x) = δ(y, x), where 0 ≤t ≤s < 1 and t and y are held constant. By lemma 1,
ps|t(·|x) is a PMF. The term P
x ps|t(y|x)p(x) also satisfies the Kolmogorov Equation, since
d
dt
X
x
ps|t(y|x)p(x) =
X
x
"X
z
ut(y, z)ps|t(z|x)
#
p(x) =
X
z
ut(y, z)
"X
x
ps|t(z|x)p(x)
#
,
(A.5)
now with initial conditions P
x pt|t(y|x)p(x) = p(y). Due to the uniqueness of solutions to the Kolmogorov
Equation (theorem 12), it follows that P
x ps|t(y|x)p(x) = ps(y), as required. Lastly, the semi-group property
of the transition kernel, P
z ps|r(y|z)pr|t(z|x) = ps|t(y|x) for 0 ≤t ≤r ≤s < 1, can be shown by repeating
the argument in A.5 with pr|t as initial condition at time r. In conclusion, we found a transition kernel pt+h|t
that generates pt, as required.
A.2
Manifold Marginalization Trick
Theorem 10 (Manifold Marginalization Trick). Under Assumption 2, if ut(x|x1) is conditionally integrable
and generates the conditional probability path pt(·|x1) then the marginal velocity field ut(·) generates
the marginal probability path pt(·).
Proof. To prove that ut(·) generates pt(·), we will again show they satisfy the conditions in the Mass
Conservation Theorem. First, we check that ut(x) and pt(x) satisfy the Continuity Equation (5.1):
d
dtpt(x)
(i)
=
Z
M
d
dtpt|1(x|x1)q(x1)dvolx1
(A.6)
(ii)
= −
Z
M
divg

ut(x|x1)pt|1(x|x1)

q(x1)dvolx1
(A.7)
(i)
= −divg
Z
M
ut(x|x1)pt|1(x|x1)q(x1)dvolx1
(A.8)
(iii)
= −divg [ut(x)pt(x)] ,
(A.9)
where in (i) we switched differentiation ( d
dt and divg) and integration justified by Leibniz rule, and the fact
that pt|1(x|x1) and ut(x|x1) are C1 in t, x and q has bounded support or M is compact. In (ii), we used the
fact that ut(·|x1) generates pt|1(·|x1) and theorem 9. In (iii), we multiplied and divided by pt(x) (which is
strictly positive by assumption) and used the formula for ut in (5.10). Lastly, ut is integrable and locally
Lipschitz by employing the same arguments as in the proof of theorem 3.
A.3
Regularity assumptions for KFE
We note that assumption 5 is true under relatively weak assumptions and there is a diversity of mathematical
literature on showing uniqueness of the solution of the KFE in pt for different settings. However, to the best
of our knowledge, there is no known result that states regularity assumptions for general state spaces and
Markov processes, which is why simply state it here as an assumption. For the machine learning practitioner,
this assumption holds for any state space of interest. To illustrate this, we point the rich sources in the
mathematical literature that show uniqueness that list the regularity assumptions for specific spaces and
classes of Markov processes:
82

1. Flows in Rd and manifolds: (Villani et al., 2009, Mass conservation formula, page 15), (DiPerna and
Lions, 1989), (Ambrosio, 2004)
2. Diffusion in Rd and manifolds: (Villani et al., 2009, Diffusion theorem, page 16)
3. General Ito-SDEs in Rd: (Figalli, 2008, Theorem 1.3 and 1.4), (Kurtz, 2011, Corollary 1.3), (Bogachev
et al., 2022)
4. Discrete state spaces: Here, the KFE is a linear ODE, which has a unique solution under the assumption
that the coefficients are continuous (see theorem 13).
83

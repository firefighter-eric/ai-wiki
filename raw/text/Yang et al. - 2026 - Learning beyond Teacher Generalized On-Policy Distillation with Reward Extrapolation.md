# Yang et al. - 2026 - Learning beyond Teacher Generalized On-Policy Distillation with Reward Extrapolation

- Source PDF: `raw/pdf/Yang et al. - 2026 - Learning beyond Teacher Generalized On-Policy Distillation with Reward Extrapolation.pdf`
- Generated from: `scripts/extract_pdf_text.py`

## Extracted Text

February 27, 2026
Learning beyond Teacher: Generalized On-Policy Distillation
with Reward Extrapolation
Wenkai Yang1,∗, Weijie Liu2, Ruobing Xie2, Kai Yang2,
Saiyong Yang2, Yankai Lin1,†
1Gaoling School of Artificial Intelligence, Renmin University of China
2LLM Department, Tencent
B {wenkaiyang,yankailin}@ruc.edu.cn
Abstract
On-policy distillation (OPD), which aligns the student with the teacher’s logit distribution on
student-generated trajectories, has demonstrated strong empirical gains in improving student per-
formance and often outperforms off-policy distillation and reinforcement learning (RL) paradigms.
In this work, we first theoretically show that OPD is a special case of dense KL-constrained
RL where the reward function and the KL regularization are always weighted equally and the
reference model can by any model. Then, we propose the Generalized On-Policy Distillation (G-
OPD) framework, which extends the standard OPD objective by introducing a flexible reference
model and a reward scaling factor that controls the relative weight of the reward term against the
KL regularization. Through comprehensive experiments on math reasoning and code generation
tasks, we derive two novel insights: (1) Setting the reward scaling factor to be greater than 1 (i.e.,
reward extrapolation), which we term ExOPD, consistently improves over standard OPD across a
range of teacher-student size pairings. In particular, in the setting where we merge the knowledge
from different domain experts, obtained by applying domain-specific RL to the same student
model, back into the original student, ExOPD enables the student to even surpass the teacher’s
performance boundary and outperform the domain teachers. (2) Building on ExOPD, we
further find that in the strong-to-weak distillation setting (i.e., distilling a smaller student from
a larger teacher), performing reward correction by choosing the reference model as the teacher’s
base model before RL yields a more accurate reward signal and further improves distillation
performance. However, this choice assumes access to the teacher’s pre-RL variant and incurs
more computational overhead. We hope our work offers new insights for future research on OPD.1
43
44
45
46
47
48
Math Reasoning Accuracy (%)
59
60
61
62
63
64
Code Generation Accuracy (%)
SFT
ExPO
OPD
ExOPD
Domain Teachers
(a) Multi-teacher distillation results, student model is Qwen3-
4B-Non-Thinking, teachers are domain-specific RL variants
Qwen3-1.7B-Non-Thinking
Qwen3-4B-Non-Thinking
Student Model
0
10
20
30
40
50
Math Reasoning Accuracy (%)
13.5
35.1
23.1
42.6
25.4
45.3
SFT
OPD
ExOPD
(b) Strong-to-weak distillation results, teacher model is
Qwen3-30B-A3B-Instruct-2507
Figure 1: The empirical effectiveness of our method ExOPD compared with off-policy distillation (SFT), standard
OPD, and the weight-extrapolation method ExPO (Zheng et al., 2025) in multi-teacher and strong-to-weak distillation
settings (results averaged over 4 math reasoning and 3 code generation benchmarks). (a) When merging multiple
domain experts—obtained by applying domain-specific RL to the same base model—back into the original base
model, ExOPD is the only method that yields a unified student that consistently outperforms all domain teachers.
(b) ExOPD also yields significant improvements over standard OPD when distilling a smaller student from a larger
teacher. Moreover, applying reward correction in ExOPD can further boost distillation performance (Figure 6).
∗Work done during an internship at Tencent.
†Corresponding author.
1Code is available at https://github.com/RUCBM/G-OPD.
1
arXiv:2602.12125v2  [cs.LG]  26 Feb 2026

1
Introduction
Recently, on-policy distillation (OPD) (Agarwal et al., 2024; Yang et al., 2025a; Lu & Lab, 2025) has emerged as
an effective post-training paradigm for improving capabilities of Large Language Models (LLMs). Unlike prior
off-policy distillation methods (Taori et al., 2023; Guha et al., 2025) that train the student on teacher-generated
trajectories, OPD allows the student to learn from the teacher’s supervision (i.e., predicted logits) on student-
generated tokens. Previous studies have shown that OPD can not only serve as a promising multi-task post-training
paradigm to (near-)losslessly merge the capabilities acquired by different RL variants across domains back into the
original base model (Xiao et al., 2026), but also be effective and efficient in distilling the capabilities of a larger
teacher into a smaller student (Gu et al., 2024; Yang et al., 2025a).
Despite its empirical effectiveness, a mechanistic understanding of OPD remains limited in the field, leaving its full
potential under-explored. In this work, we bridge this gap by establishing a theoretical connection between OPD
and dense reinforcement learning (RL), and by extending standard OPD into a generalized formulation.
First, we make derivations to show that OPD is essentially a special case of the standard dense RL with Kull-
back–Leibler (KL) constraint, where the token-level reward function is always weighted equally with the KL
regularization and the reference model can be chosen arbitrarily. Building on this insight, we generalize the OPD
objective to a more universal formulation by further introducing a reward scaling factor that controls the relative
weight of the reward term against the KL regularization, in addition to the flexible reference model. We refer to this
generalized formulation as the Generalized On-Policy Distillation (G-OPD) framework.
Based on the G-OPD framework, we theoretically analyze how the reward scaling factor and the choice of reference
model affect distillation effectiveness across different settings, supported by comprehensive experiments in both
math reasoning and code generation domains. In the first setting, the teacher is obtained by applying domain-specific
RL to the student, and the reference model is naturally fixed to the student’s initial state. We show that (1) when
the reward scaling factor lies in (0, 1) (i.e., reward interpolation), the distilled student exhibits behaviors (e.g.,
performance and response length) that fall between the reference and teacher models; (2) when the reward scaling
factor is greater than 1 (i.e., reward extrapolation), the student can learn beyond the teacher’s capability boundary
and outperform teacher in domain tasks. We refer to the reward extrapolation variant as ExOPD. We further show
that ExOPD extends well to the multi-teacher distillation setting, enabling a unified student to surpass all domain
teachers. Second, we study the strong-to-weak distillation setting, where a smaller student is distilled from a
larger teacher. In this setting, we demonstrate that replacing the reference model from the student’s initial policy to
the teacher’s pre-RL variant (i.e., reward correction) in ExOPD yields a more accurate reward signal and further
improves distillation performance. However, the limitations of this practice are that it assumes access to an additional
model (the teacher’s pre-RL variant) and incurs more computational cost on computing the log-probabilities of
the larger reference model. Despite these limitations, ExOPD and ExOPD with reward correction significantly
outperform standard OPD in the strong-to-weak distillation setting.
2
Related Work
Off-Policy Distillation. Knowledge distillation (KD) (Hinton et al., 2015) is a widely used technique for transferring
knowledge from a domain expert (teacher) to a student model. Most prior studies focus on off-policy distillation,
where the student is trained on trajectories generated by the teacher, either by aligning the student’s logits distribution
with the teacher’s via a Kullback–Leibler (KL) divergence loss on token logits (Sanh et al., 2019; Kim & Rush,
2016; Guo et al., 2025b), or by directly performing supervised fine-tuning (SFT) with a cross-entropy loss on the
teacher-generated tokens (Taori et al., 2023; Zhou et al., 2023; Guha et al., 2025). This practice has been shown to
effectively improve the student model across a broad range of capabilities (Ding et al., 2023; Yang et al., 2025e; Ye
et al., 2025b) in the LLM era.
On-Policy Distillation. By sampling trajectories from the student and aligning the student with the teacher’s
logit distribution on each token of these student-generated trajectories, on-policy distillation (OPD) (Agarwal
et al., 2024; Gu et al., 2024) realizes dense on-policy learning. Empirically, OPD has been shown to achieve
faster and more effective distillation than off-policy distillation (Yang et al., 2025a; Lu & Lab, 2025). Recent
OPD studies have explored distillation across different model families (Pati˜no et al., 2025), developed black-box
on-policy distillation methods that do not require access to the teacher’s logits (Ye et al., 2025a), and investigated the
self-distillation paradigm that leverage the LLM’s in-context capabilities to distill textual context information into its
parameters (Yang et al., 2025c; H¨ubotter et al., 2026; Shenfeld et al., 2026; Zhao et al., 2026; Penaloza et al., 2026).
3
Methodology
3.1
Preliminaries
In this section, we start with a brief review of relevant preliminaries.
2

Off-Policy Distillation. Let D denote the input distribution, and let πθ and π∗denote the student and teacher
policies, respectively. The general form of Knowledge Distillation (KD) (Hinton et al., 2015) can be written as
JKD(θ) = min
θ
Ex∼D,y∼π∗(·|x)
h
DKL
 π∗(y|x)
 πθ(y|x)
i
,
(1)
where DKL denotes the Kullback–Leibler (KL) divergence loss. In the era of LLMs, obtaining the teacher’s full
output distribution (e.g., logits) is often expensive or even infeasible. As a result, KD is commonly implemented as
supervised fine-tuning (SFT) of the student on trajectories generated by the teacher. Though effective, the major
drawback of this paradigm is its off-policy nature: the student is trained to imitate the teacher’s behavior, rather
than to learn from reward signals induced by its own actions. As a result, it may fail to adapt and generalize from its
own experience at test time, when faced with similar problems.
On-Policy RL. We use πθ to denote the policy model to be optimized. The RL objective can be formulated as
JRL(θ) = max
θ
Ex∼D,y∼πθ(·|x)
h
r(x, y) −βDKL(πθ ∥πref)
i
.
(2)
In the above formulation, the trajectories y are sampled from the current policy model, making the training remain
on-policy. r(x, y) is the reward function that measures the quality of a response sequence y = (y1, · · · , yT) to a
query x. Depending on the setting, it can be either (i) a parameterized neural reward model trained on the specific
preference data for open-domain alignment (Cai et al., 2024; Dong et al., 2024; Liu et al., 2025a), or (ii) a rule-based,
deterministic outcome verifier commonly used in verifiable LLM reasoning tasks (Guo et al., 2025a; Hu et al., 2025;
Liu & Zhang, 2025; Yang et al., 2025b). DKL(πθ ∥πref) prevents the policy model πθ from drifting too far from
a reference model πref, and the coefficient β controls the strength of this constraint. To solve Eq. (2), a common
approach is to apply policy gradient (Sutton et al., 1998), updating the policy parameters using an estimated gradient
of the form
∇θJRL(θ) = Ex∼D, y∼πθ(·|x)
h T
∑
t=1
At∇θ log πθ(yt|x, y<t)
i
,
(3)
where At is the relative advantage of token yt over a baseline value. In practice, the reward signal in RL is often
sparse: the policy model only receives a reward at the final token after the response is completed, which may make
optimization inefficient and ineffective (Cui et al., 2025).
On-Policy Distillation. On-Policy Distillation (OPD) (Agarwal et al., 2024; Gu et al., 2024; Lu & Lab, 2025)
inherits the on-policy nature of policy training and the advantage of dense credit assignment, making it an efficient
post-training paradigm (Yang et al., 2025a; Xiao et al., 2026). The main idea of OPD is to let the student generate
its own trajectories, and then minimize the reverse KL divergence between the student and the teacher π∗on those
student-generated trajectories:
JOPD(θ) = min
θ
Ex∼D,y∼πθ(·|x)
h
DKL

πθ(y|x)
 π∗(y|x)
i
.
(4)
Notice that in Eq. (4), the trajectories y are generated by the policy model itself, resulting in the on-policy training.
Also, we can get the gradient of OPD as2
∇θJOPD(θ) = Ex∼D,y∼πθ(·|x)
h T
∑
t=1

T
∑
t′=t
 log πθ(yt′ |x, y<t′ ) −log π∗(yt′ |x, y<t′ )

∇θ log πθ(yt|x, y<t)
i
.
(5)
In practice, current studies (Lu & Lab, 2025; Xiao et al., 2026) use a discount factor of 0 (focus on next-token
optimization only) and approximate the gradient as
∇θJOPD(θ) = Ex∼D,y∼πθ(·|x)
h T
∑
t=1
 log πθ(yt|x, y<t) −log π∗(yt|x, y<t)
 ∇θ log πθ(yt|x, y<t)
i
.
(6)
Comparing Eq. (6) with Eq. (3), we can see that −
 log πθ(yt|x, y<t) −log π∗(yt|x, y<t)

can be regarded as the
token-level advantage in OPD, thereby providing dense credit assignment for each token-level action.
3.2
Generalized On-Policy Distillation
In this section, we first start from Eq. (4) and derive a generalized formulation of OPD.
2Detailed derivations are in Appendix A.
3

First, we re-formulate the OPD objective (Xiao et al., 2025) as
JOPD(θ) = min
θ
Ex∼D,y∼πθ(·|x)
h
DKL
 πθ(y|x)
 π∗(y|x)
i
= min
θ
Ex∼D,y∼πθ(·|x)
h
log πθ(y|x) −log π∗(y|x)
i
= max
θ
Ex∼D,y∼πθ(·|x)
h
log π∗(y|x) −log πθ(y|x)
i
= max
θ
Ex∼D,y∼πθ(·|x)
h log π∗(y|x) −log πref(y|x)
 −
 log πθ(y|x) −log πref(y|x)
i
= max
θ
Ex∼D,y∼πθ(·|x)
h
log π∗(y|x)
πref(y|x) −DKL
 πθ(y|x)
 πref(y|x)
i
.
(7)
Therefore, we have the following remark:
Remark
By introducing a third reference model πref, the OPD objective in Eq. (4) becomes equivalent to a specific
KL-constrained RL objective in Eq. (2), where the reward function r(x, y) = log π∗(y|x)
πref(y|x), the KL regular-
ization is applied between the policy model πθ and the reference model πref, and the reward and KL terms
are weighted equally (i.e., β = 1 in Eq. (2)).
From the above remark, we establish the connection between OPD and RL. However, we emphasize that OPD
differs from standard RL in the following key respects:
(1) Dense rewards. As discussed above, in standard RL the model typically receives an effective reward only at the
final token, while the rewards for all other tokens are zero:
rRL
t
=
0
t = 1, · · · , T −1,
Outcome Reward
t = T.
(8)
However, in OPD, each token-level action receives an effective reward
rOPD
t
= log π∗(yt|x, y<t)
πref(yt|x, y<t),
t = 1, · · · , T.
(9)
This token-level reward takes essentially the same form as the implicit reward defined in Rafailov et al. (2023).
Implicit reward is initially derived from the closed-form solution of Eq. (2), which can be written as
r(x, y) = β log πθ(y|x)
πref(y|x) + β log Z(x), where Z(x) = ∑
y
πref(y|x) exp( 1
βr(x, y)).
(10)
As we can see, since log Z(x) is a constant depending only on x, log πθ(y|x)
πref(y|x) can be regarded as a well-defined
proxy of the true reasoning reward, and this idea is adopted in previous studies (Yuan et al., 2024; Cui et al., 2025;
Yang et al., 2025d; Liu et al., 2025c) to provide dense supervision for RL. However, in OPD, the implicit reward
log π∗(y|x)
πref(y|x) does not require π∗to be obtained by applying RL starting from πref. In fact, π∗and πref can even be
models of different sizes. Nevertheless, this reward function still captures the log-probability shift from the
reference (πref) distribution to the expert (π∗) distribution, and thus provides a meaningful training signal.
(2) Fixed weighting between the reward function and the KL regularization. As revealed in the remark, in OPD,
the reward term and the KL regularization are always weighted equally. In what follows, we present and discuss our
generalized OPD formulation by introducing a reward scaling factor that allows us to adjust the relative weight of
the reward term against the KL regularization.
(3) Flexible choice of the reference model. In RL (i.e., Eq. (2)), the reference model is typically initialized as
the policy model’s starting checkpoint. However, we note that in OPD (i.e., Eq. (11)), the introduced reference
model can be any model, since this choice does not affect the final simplification of the objective back to its original
form in Eq. (4). In what follows, we discuss how different choices of πref affect our proposed generalized OPD
framework. By default, the reference model is selected as the student’s initial policy.
From the above discussion, we can see that OPD offers two key advantages over RL—dense reward signals and
a flexible choice of reference model—yet it fixes the relative weighting between the reward function and the KL
regularization to 1 : 1. This motivates us to follow Eq. (2) and generalize the original OPD objective in Eq. (4) into
a general dense RL objective with a flexible KL constraint, by introducing both a third reference model and an
additional reward scaling factor λ:
JG-OPD(θ) = max
θ
Ex∼D,y∼πθ(·|x)
h
λ log π∗(y|x)
πref(y|x) −DKL
 πθ(y|x)
 πref(y|x)
i
.
(11)
4

The above Eq. (11) presents our Generalized On-Policy Distillation (G-OPD) formulation, where λ controls the
relative weight of the reward term against the KL regularization in the objective, and is essential 1
β in Eq. (2). As we
can see, compared to RL, G-OPD enables dense credit assignment and a more flexible choice of reference model;
compared to OPD, it further allows more general control over the reward weight. In the following, we discuss in
detail about the two crucial components, λ and πref, in G-OPD.
Reward interpolation and extrapolation in G-OPD.
The optimal solution to G-OPD in Eq. (11) satisfies that
log πθ(y|x) = λ log π∗(y|x) + (1 −λ) log πref(y|x)
= log π∗(y|x) + (λ −1)(log π∗(y|x) −log πref(y|x)).
(12)
This reveals that, (1) when 0 < λ < 1, G-OPD encourages the student model’s log-probability distribution to match
a linear interpolation between that of the teacher and reference models. This can also be interpreted as replacing the
reward r in Eq. (7) with λ · r + (1 −λ) · 0. Therefore, we refer to this case as reward interpolation. We conjecture
that, under this setting, the student trained with G-OPD may exhibit behavior (e.g., performance, response length,
etc.) that lies between the reference model and the standard OPD with λ = 1. (2) When λ > 1, G-OPD encourages
the student’s log-probability distribution to go beyond matching the teacher’s log-probabilities by additionally fitting
an extra shift term (λ −1)(log π∗−log πref). From the perspective of rewards, G-OPD with λ > 1 performs an
extrapolation of the reward function’s weight in Eq. (7); thus, we refer to this regime as reward extrapolation. We
wonder whether reward extrapolation can outperform standard OPD, and in a special case, when the teachers are
domain experts obtained by applying RL to the same student (Xiao et al., 2026) in different domains, can reward
extrapolation in G-OPD distill a unified student that surpasses all the domain teachers?
Reward correction in strong-to-weak distillation.
When the reward scaling factor λ ̸= 1, different choices of
the reference model πref in Eq. (11) lead to different objectives. Based on distillation settings, in the following, we
discuss the choices of πref in two cases: (1) One application of G-OPD is to merge the capabilities of several experts,
each obtained by applying domain-specific RL starting from the same base model, back into the original base
model (Xiao et al., 2026). In this setting, πref is naturally chosen as the original base model, and the reward function
in G-OPD is exactly the implicit reward defined in Eq. (10). (2) Another distillation setting is strong-to-weak
distillation (Yang et al., 2025a), i.e., distilling a large teacher into a smaller student. In this case, πref admits two
choices: (i) the student’s base model, πstudent
base
, which corresponds to the default setting where we only have access
to π∗and πstudent
base
; or (ii) the teacher expert’s pre-RL base model, πteacher
base
(i.e., the teacher before post-training),
assuming it is available. To compare these two choices, we first rewrite the G-OPD objective into an equivalent
form:
JG-OPD(θ) = max
θ
Ex∼D,y∼πθ(·|x)
h
λ log π∗(y|x)
πref(y|x) −DKL
 πθ(y|x)
 πref(y|x)
i
= max
θ
Ex∼D,y∼πθ(·|x)
h
(λ −1) log π∗(y|x)
πref(y|x) −DKL
 πθ(y|x)
 π∗(y|x)
i
.
(13)
Now, under the same KL regularization strength, we can see that choosing πref = πteacher
base
is more reasonable. The
reason is that the reward log
π∗
πteacher
base
corresponds to the implicit reward induced by the teacher’s RL post-training,
and is thus well-defined according to Eq. (10). In contrast, log
π∗
πstudent
base
can be noisier, since there exists fundamental
gap between the internal knowledge and capacity of teacher and student base models. Therefore, in the strong-to-
weak distillation setting, we think that applying a reward correction to the default reward log
π∗
πstudent
base
—by adding
log
πstudent
base
πteacher
base
to obtain log
π∗
πteacher
base
—can lead to better distillation performance. The limitations, however, are that this
requires access to πteacher
base
and incurs additional computation, since computing log πteacher
base
requires more cost than
computing log πstudent
base
.
Remark
By introducing a reference model πref and a reward scaling factor λ, we formulate the Generalized On-Policy
Distillation framework as
JG-OPD(θ) = max
θ
Ex∼D,y∼πθ(·|x)
h
λ log π∗(y|x)
πref(y|x) −DKL
 πθ(y|x)
 πref(y|x)
i
.
We have two observations:
(1) For 0 < λ < 1, G-OPD yields a student whose behavior lies between that of the reference model and
that of the student trained by standard OPD (i.e., λ = 1). In contrast, λ > 1 can potentially deliver larger
gains than standard OPD, and may even produce a student that outperforms the teacher in certain cases.
Note that setting λ ̸= 1 incurs additional computational cost on computing log πref.
5

(2) In strong-to-weak distillation, choosing the teacher’s pre-RL base model πteacher
base
as the reference may
yield better distillation performance. However, this comes with two limitations: it requires access to an
additional model πteacher
base
, and it increases computational cost because computing log πteacher
base
requires
more computational cost than computing log πstudent
base
.
Finally, the approximated gradient of G-OPD can be written as
∇θJG-OPD(θ) = Ex∼D,y∼πθ(·|x)
h T
∑
t=1
AG-OPD
t
∇θ log πθ(yt|x, y<t)
i
,
(14)
where AG-OPD
t
=
 log πθ(yt|x, y<t) −log π∗(yt|x, y<t)
 + (λ −1)
 log πref(yt|x, y<t) −log π∗(yt|x, y<t)

.
4
Experiments and Analysis
In this section, we conduct a series of extensive experiments on math reasoning and code generation tasks to analyze
the properties of the proposed G-OPD framework and assess the effectiveness of ExOPD. We begin with preliminary
experiments on same-size teacher-student pairs in Section 4.1.2, where we investigate the impact of the reward
scaling factor within G-OPD. We then explore the effectiveness of ExOPD in the multi-teacher distillation setting in
Section 4.1.3. Finally, we present experimental results in the strong-to-weak distillation setting in Section 4.2.
4.1
Experiments with Same-Sized Student and Teacher
Here, we consider the scenario where the domain teachers are reinforced models derived from the student through
domain-specific RL.
4.1.1
Experimental Settings
Base Model. We primarily conduct experiments using the Qwen3-4B-Non-Thinking (Yang et al., 2025a) model.
The student model is initialized as Qwen3-4B-Non-Thinking, while the domain teachers are derived by applying RL
separately to Qwen3-4B-Non-Thinking on domain-specific data.
Training Datasets. We filter the DeepMath (He et al., 2025) dataset to select 57K samples with a difficulty level
greater than or equal to 6 to form the math RL data, and use Eurus-RL-Code (Cui et al., 2025) as the code RL data,
which consists of 25K samples. We then apply RL to the base model on two datasets separately to get domain
teachers, Qwen3-4B-Non-Thinking-RL-Math and Qwen3-4B-Non-Thinking-RL-Code. The distillation data is the
same as the RL data.
Training Settings. We apply Group Relative Policy Optimization (GRPO) (Shao et al., 2024) to obtain domain
teachers. In RL, a reward of 1.0 is given when the final answer is correct in math reasoning or when all unit tests pass
in code generation; otherwise, the reward is 0.0. Detailed training hyper-parameters in GRPO are in Appendix B.
After this, we implement G-OPD on the original student model (i.e., Qwen3-4B-Non-Thinking) with different
reward scaling factors λ ∈{0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}. Note that λ = 0.0 corresponds to the initial state
Qwen3-4B-Non-Thinking, and λ = 1.0 corresponds to standard OPD. The reference model here is fixed naturally
as Qwen3-4B-Non-Thinking. Detailed training hyper-parameters in G-OPD are in Appendix B. In both GRPO and
G-OPD, we implement token-level rollout correction (Liu et al., 2025b) to mitigate training-inference mismatch.
Our experiments are based on verl (Sheng et al., 2024) framework.
Evaluation. For the evaluation of math reasoning, we select four competition-level benchmarks: AIME24 (AI-
MO, 2024), AIME25 (OpenCompass, 2025), HMMT25 (February) (Balunovi´c et al., 2025), and HMMT25
(November) (Balunovi´c et al., 2025). For the evaluation of code generation, we select three test sets: HumanEval+,
MBPP+ (Liu et al., 2023), and LiveCodeBench (v6 only, February 2025∼May 2025) (Jain et al., 2024). In all
evaluations, we set the temperature to 1.0, top-p to 1.0, and the maximum generation length to 16,384. On each
math reasoning benchmark, we sample 32 solutions for each problem; whereas each code generation benchmark,
we sample 4 solutions per problem. We then report the average accuracy of each model on each benchmark. We
adopt Math-Verify3 as a rule-based verifier to validate answer correctness for math reasoning benchmarks.
4.1.2
Results of Single-Teacher Distillation
We first explore the impact of reward scaling factor λ in G-OPD in the same-sized single-teacher distillation setting
as the preliminary experiments (i.e., distilling Qwen3-4B-Non-Thinking-RL-Math or Qwen3-4B-Non-Thinking-RL-
Code back into Qwen3-4B-Non-Thinking). The evaluation results in math reasoning and code generation domains
3https://github.com/huggingface/Math-Verify
6

0.0
0.2
0.4
0.6
0.8
1.0
1.2
1.4
Lambda
0
10
20
30
40
50
60
70
Accuracy (%)
Reward Interpolation
Reward Extrapolation
AIME24
AIME25
HMMT25 (Feb.)
HMMT25 (Nov.)
Teacher Accuracy
Figure 2: On-policy distillation results on four math
reasoning benchmarks under different choices of reward
scaling factor λ.
0.0
0.2
0.4
0.6
0.8
1.0
1.2
1.4
Lambda
0
20
40
60
80
100
Accuracy (%)
Reward Interpolation
Reward Extrapolation
HumanEval+
MBPP+
LiveCodeBench
Teacher Accuracy
Figure 3: On-policy distillation results on three code gen-
eration benchmarks under different choices of reward
scaling factor λ.
4000
6000
8000
10000
12000
Average Number of Tokens
20
30
40
50
60
Accuracy (%)
Teacher
Student Base
Lambda=0.25
Lambda=0.5
Lambda=0.75
Lambda=1.0
Lambda=1.25
Lambda=1.5
(a) Results on AIME24
4000
6000
8000
10000
12000
Average Number of Tokens
25
30
35
40
45
50
55
Accuracy (%)
Teacher
Student Base
Lambda=0.25
Lambda=0.5
Lambda=0.75
Lambda=1.0
Lambda=1.25
Lambda=1.5
(b) Results on AIME25
4000
6000
8000
10000
12000
14000
Average Number of Tokens
10
15
20
25
30
35
Accuracy (%)
Teacher
Student Base
Lambda=0.25
Lambda=0.5
Lambda=0.75
Lambda=1.0
Lambda=1.25
Lambda=1.5
(c) Results on HMMT25 (Feb.)
500
1000
1500
2000
2500
Average Number of Tokens
76
78
80
82
84
86
Accuracy (%)
Teacher
Student Base
Lambda=0.25
Lambda=0.5
Lambda=0.75
Lambda=1.0
Lambda=1.25
Lambda=1.5
(d) Results on HumanEval+
200
400
600
800
1000
1200
1400
1600
1800
Average Number of Tokens
65
66
67
68
69
70
71
Accuracy (%)
Teacher
Student Base
Lambda=0.25
Lambda=0.5
Lambda=0.75
Lambda=1.0
Lambda=1.25
Lambda=1.5
(e) Results on MBPP+
2000
4000
6000
8000
Average Number of Tokens
18
20
22
24
26
28
Accuracy (%)
Teacher
Student Base
Lambda=0.25
Lambda=0.5
Lambda=0.75
Lambda=1.0
Lambda=1.25
Lambda=1.5
(f) Results on LiveCodeBench
Figure 4: Trends in the average number of tokens and the average accuracy of the on-policy distilled models across
different benchmarks under varying reward scaling factors. The teacher for math reasoning tasks is Qwen3-4B-Non-
Thinking-RL-Math, while the teacher for code generation tasks is Qwen3-4B-Non-Thinking-RL-Code.
are in Figure 2 and Figure 3 respectively. We also visualize the relationship between accuracy and response length
of each model in Figure 4 for deep analysis.
We can draw the following conclusions: (1) Standard OPD can fully recover the post-training behavior. As we
can see, the student produced by OPD closely matches the evaluation accuracy and response length of the domain
teacher. (2) Reward interpolation (0 < λ < 1) produces a student whose behavior (performance and response
length) lies between the base model and the teacher model. Also, both the performance and response length
increase monotonically as λ grows, approaching the behavior of the teacher. This property can be leveraged to
achieve budget-controlled reasoning (Yang et al., 2025e; Liang et al., 2026). (3) Reward extrapolation (λ > 1)
outperforms standard OPD and has the potential to produce a student that surpasses the domain teacher. As
observed, ExOPD with appropriate reward extrapolation (i.e., λ = 1.25) consistently outperforms OPD and the
domain teacher in all settings (also see Table 2), while excessive reward extrapolation (i.e., λ = 1.5) may lead to
instability and degrade performance. This can be explained by the fact that continuously increasing λ introduces the
risk of the student hacking the implicit reward in Eq. (9), by aggressively fitting the peak of the log ratio, even if
some tokens have excessively large log ratios due to bias. Furthermore, we can see that the response lengths of
7

Table 1: Comparison against the math domain teacher with continued RL. Each numerical subscript indicates the
absolute improvement or degradation compared to the teacher model.
Method
AIME24
AIME25
HMMT25 (Feb.)
HMMT25 (Nov.)
Avg.
Teacher
58.0
54.6
32.5
38.9
46.0
+ continued RL (100 steps)
60.9+2.9
55.6+0.5
32.8+0.3
38.4−0.5
46.9+0.9
ExOPD (50 steps)
62.7+4.7
56.1+1.5
33.9+1.4
39.3+0.4
48.0+2.0
Table 2: Comparison against off-policy distillation (SFT) and weight extrapolation (ExPO) methods in both single-
teacher and multi-teacher settings with same-sized teacher-student pairs. “Teacher” represents the performance of the
domain teacher model (Qwen3-4B-Non-Thinking-RL-Math for math reasoning and Qwen3-4B-Non-Thinking-RL-
Code for code generation), “Student” represents the initial performance of student model Qwen3-4B-Non-Thinking.
Each numerical subscript indicates the absolute improvement or degradation compared to the domain teacher
model.
Method
Math Reasoning
Code Generation
AIME24 AIME25 HMMT25 (Feb.) HMMT25 (Nov.)
Avg.
HumanEval+
MBPP+
LCB
Avg.
Teacher
58.0
54.6
32.5
38.9
46.0
86.0
70.2
27.3
61.2
Student
21.5
21.9
10.0
8.0
15.4
74.7
64.7
17.9
52.4
Single-Teacher Distillation
ExPO
58.7+0.7
55.2+0.6
32.4−0.1
37.0−1.9
45.8−0.2
84.8−1.2
70.2+0.0
28.0+0.7
61.0−0.2
OPD
60.7+2.7
55.0+0.4
32.4−0.1
37.9−1.0
46.5+0.5
85.2−0.8
69.9−0.3
27.3+0.0
60.8−0.3
ExOPD
62.7+4.7
56.1+1.5
33.9+1.4
39.3+0.4
48.0+2.0
86.9+0.9
70.7+0.5
28.6+1.3
62.1+0.9
Multi-Teacher Distillation
SFT
58.5+0.5
53.3−1.3
30.7−1.8
34.8−4.1
44.3−1.7
86.4+0.4
69.6−0.6
26.4−0.9
60.8−0.4
ExPO
57.5−0.5
54.5−0.1
31.7−0.8
36.3−2.6
45.0−1.0
86.7+0.7
72.0+1.8
29.0+1.7
62.6+1.4
OPD
60.6+2.6
54.1−0.5
32.5+0.0
38.3−0.6
46.4+0.4
84.6−1.4
69.5−0.7
27.6+0.3
60.6−0.6
ExOPD
61.0+3.0
56.0+1.4
34.4+1.9
39.2+0.3
47.7+1.7
86.3+0.3
70.6+0.4
29.0+1.7
62.0+0.8
the students produced by ExOPD continue to increase, which may be due to the length bias issue of the implicit
reward (Yang et al., 2025d).
To demonstrate that the improvement of ExOPD over the teacher is not due to less training of the teacher, we compare
the evaluation performance of ExOPD and the teacher after an additional 100 steps of RL training. The results in
Table 1 show that the teacher with more continued RL training show smaller improvement compared to ExOPD
with fewer steps. We also demonstrate the generalizability and effectiveness of ExOPD when the teacher models
are trained with sufficient RL steps (i.e., 1200 RL steps), with the corresponding results provided in Appendix C.
4.1.3
Results of Multi-Teacher Distillation
Based on above analysis, we conduct experiments in the multi-teacher distillation setting, where we aim to merge
the capabilities from different domain teachers, obtained by applying domain-specific RL to the same base model,
into the original base model through OPD (Xiao et al., 2026). This has been demonstrated to be an effective new
multi-task post-training paradigm. Specifically, the domain teachers are the above RL variants Qwen3-4B-Non-
Thinking-RL-Math/Code, and the student model is Qwen3-4B-Non-Thinking. From the preliminary results in
Section 4.1.2, we can see that λ = 1.25 in ExOPD consistently leads to better performance than OPD. Thus, in all
subsequent experiments, we fix λ = 1.25 for ExOPD without any further specific tuning. Besides OPD, we
also compare against two baselines: (1) Supervised fine-tuning (SFT), which trains the student on the teachers’
generated trajectories via Cross-Entropy Loss. We ensure that the number of trajectories used for SFT is consistent
with those in OPD and ExOPD. More details can be found in Appendix B. (2) ExPO (Zheng et al., 2025), a weight
extrapolation method. We implement ExPO by first averaging the weights of all domain teachers, then extrapolating
the weights against the student model using an extrapolation factor α, which is tuned from {0.25, 0.5} following the
recommendations. For a fair comparison, we downweight the sample size of the math RL data to match that of the
code RL data in both OPD and ExOPD here, ensuring that each domain has the same sample size.
The results of multi-teacher distillation are shown in Table 2. As we can see, SFT produces a sub-optimal student,
while the performance ceiling of OPD is typically bounded by the teachers. ExPO, though training-free, cannot
ensure that the weight-extrapolated student consistently surpasses all domain teachers, lacking good controllability.
However, our method ExOPD consistently outperforms OPD and is the only method that produces a unified
student capable of surpassing both domain teachers on all benchmarks.
Furthermore, we analyze the training dynamics of ExOPD compared to OPD to gain a deeper understanding of
8

OPD
ExOPD
0
10
20
30
40
50
Optimization Step
0.40
0.45
0.50
0.55
0.60
0.65
Training Rewards
(a) Training rewards
0
10
20
30
40
50
Optimization Step
2000
4000
6000
8000
10000
Response Length
(b) Response length
0
10
20
30
40
50
Optimization Step
0.30
0.35
0.40
0.45
0.50
0.55
Entropy
(c) Entropy
Figure 5: Training dynamics of OPD and ExOPD in multi-teacher distillation experiments. We visualize using
Exponential Moving Average (EMA) smoothing with a coefficient of 0.5.
Table 3: Evaluation accuracy on four math reasoning benchmarks in the strong-to-weak distillation setting. Teacher
model is Qwen3-30B-A3B-Instruct-2507. The numerical subscript indicates the absolute improvement or degra-
dation compared to the standard OPD.
Method
AIME24
AIME25
HMMT25 (Feb.)
HMMT25 (Nov.)
Avg.
Teacher
74.7
62.8
44.2
57.2
59.7
Student: Qwen3-1.7B-Non-Thinking
Base
12.3
11.4
6.8
4.5
8.8
SFT
18.1
20.5
9.2
6.3
13.5
OPD
33.0
28.7
15.7
14.9
23.1
ExOPD
37.3+4.3
31.5+2.8
16.2+0.5
16.5+1.6
25.4+2.3
Student: Qwen3-4B-Non-Thinking
Base
21.5
21.9
10.0
8.0
15.4
SFT
45.4
40.9
22.4
31.6
35.1
OPD
55.0
48.0
29.8
37.7
42.6
ExOPD
58.7+3.7
50.8+2.8
33.0+3.2
38.8+1.1
45.3+2.7
ExOPD. We put the comparison in Figure 5. ExOPD achieves higher training rewards but makes the student generate
longer response lengths, which is consistent with the evaluation results shown in Figure 4. We also observe that the
response entropy of the student trained by ExOPD is higher than that trained by OPD. We attribute this to the fact
that the former tends to generate longer responses, increasing the response diversity.
4.2
Experiments in the Strong-to-Weak Distillation Setting
Another practical usage of OPD is for strong-to-weak distillation (Yang et al., 2025a), i.e., distilling capabilities
from a larger teacher into a smaller student. Thus, in this section, we explore the effectiveness of ExOPD and the
additional reward correction practice in the strong-to-weak distillation setting.
4.2.1
Experimental Settings
We select Qwen3-30B-A3B-Instruct-2507 as the teacher model and perform distillation on Qwen3-1.7B-Non-
Thinking and Qwen3-4B-Non-Thinking, respectively. We primarily conduct experiments in the math reasoning
domain, where the training and evaluation datasets are the same as those used in Section 4.1. The training details are
in Appendix B. In ExOPD, we first conduct experiments in the default setting (Section 4.2.2), where we assume the
availability of only two models: the student base model and the stronger teacher model. Thus, in this default setting,
we set the reference model in ExOPD to the student base model. We also explore the effectiveness of the reward
correction technique in Section 4.2.3, where we assume extra access to the teacher’s pre-RL variant, which serves as
the reference model in ExOPD. We compare ExOPD against standard OPD and off-policy distillation (SFT).
4.2.2
Results of Strong-to-Weak Distillation
The results in default strong-to-weak distillation setting are put in Table 3. The main conclusion is that ExOPD
can bring significant improvements in strong-to-weak distillation, outperforming off-policy distillation and
9

Model
20
22
24
26
28
30
Average Accuracy (%)
22.7
27.5
28.1
28.7
SFT
OPD
ExOPD
ExOPD (w/ reward correction)
(a) Results averaged on four math reasoning benchmarks,
student is Qwen3-1.7B-Non-Thinking, teacher is Qwen3-4B-
Non-Thinking-RL-Math
Model
44
46
48
50
52
54
Average Accuracy (%)
47.0
50.5
51.3
52.3
SFT
OPD
ExOPD
ExOPD (w/ reward correction)
(b) Results averaged on three code generation benchmarks,
student is Qwen3-1.7B-Non-Thinking, teacher is Qwen3-4B-
Non-Thinking-RL-Code
Figure 6: Effect of reward correction in the strong-to-weak distillation setting.
standard OPD by a large margin. The results reveal that, although the implicit reward log
π∗
πstudent
base
may contain
noise due to the intrinsic knowledge gap and distribution bias between the small and large models, extrapolating the
rewards can still push the limits of OPD in strong-to-weak distillation.
4.2.3
Reward Correction in Strong-to-Weak Distillation
As shown above, the default ExOPD with the reference model fixed as the student base model can already bring
significant improvement over OPD. However, as discussed in Remark 3.2, setting the reference model to the teacher’s
pre-RL variant—if available—may further enhance the distillation performance. Here, we conduct experiments to
validate this analysis. Specifically, since we cannot get the pre-RL variant of Qwen3-30B-A3B-Instruct-2507, we
choose our trained Qwen3-4B-Non-Thinking-RL-Math/Code as the teachers and take Qwen3-4B-Non-Thinking as
the pre-RL variant. The student model is Qwen3-1.7B-Non-Thinking.
The comparison results are displayed in Figure 6. The results validate the effectivenss of the reward correction
practice, which consistently boosts the performance of ExOPD. However, we reiterate that reward correction
requires access to πteacher
base
and incurs higher computational cost, since it requires computing log-probabilities under
a larger reference model than in the default ExOPD.
5
Conclusion and Discussion
In this work, we conduct an in-depth analysis of the on-policy distillation paradigm. We first establish an interesting
connection between OPD and dense KL-constrained RL. Building on this insight, we propose a generalized OPD
framework (G-OPD) by introducing (i) a flexible reference model for the implicit reward function and (ii) a reward
scaling factor that controls the relative weight of the reward term versus KL regularization. Through comprehensive
experiments on math reasoning and code generation tasks, we provide several novel insights: (1) Appropriate reward
extrapolation (i.e., setting the reward scaling factor to be larger than 1) can improve OPD performance, and in
same-sized multi-teacher distillation it enables learning a unified student that surpasses all domain-specific teachers.
We refer to this variant as ExOPD. (2) Moreover, in strong-to-weak distillation, replacing the student’s initial policy
with the teacher’s pre-RL policy as the reference model can further boost the performance of ExOPD.
Regarding future work, we believe it is practical to explore: (1) validating the generalizability of ExOPD on
larger-scale models; (2) assessing the robustness of ExOPD in multi-teacher distillation with a broader and more
diverse set of domain teachers; and (3) evaluating the effectiveness of ExOPD for on-policy distillation across
different model families.
10

References
Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier
Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In The twelfth
international conference on learning representations, 2024.
AI-MO. Aime 2024. https://huggingface.co/datasets/AI-MO/aimo-validation-aime, 2024.
Mislav Balunovi´c, Jasper Dekoninck, Ivo Petrov, Nikola Jovanovi´c, and Martin Vechev. Matharena: Evaluating
llms on uncontaminated math competitions, February 2025. URL https://matharena.ai/.
Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei
Chu, et al. Internlm2 technical report. arXiv preprint arXiv:2403.17297, 2024.
Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu,
Weize Chen, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025.
Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou.
Enhancing chat language models by scaling high-quality instructional conversations. In Proceedings of the 2023
Conference on Empirical Methods in Natural Language Processing, pp. 3029–3051, 2023.
Hanze Dong, Wei Xiong, Bo Pang, Haoxiang Wang, Han Zhao, Yingbo Zhou, Nan Jiang, Doyen Sahoo, Caiming
Xiong, and Tong Zhang. Rlhf workflow: From reward modeling to online rlhf. arXiv preprint arXiv:2405.07863,
2024.
Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge distillation of large language models. In
The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/
forum?id=5h0qf7IBZZ.
Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean
Mercat, Trung Vu, Zayne Sprague, et al. Openthoughts: Data recipes for reasoning models. arXiv preprint
arXiv:2506.04178, 2025.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi
Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv
preprint arXiv:2501.12948, 2025a.
Yiju Guo, Wenkai Yang, Zexu Sun, Ning Ding, Zhiyuan Liu, and Yankai Lin. Learning to focus: Causal attention
distillation via gradient-guided token pruning. arXiv preprint arXiv:2506.07851, 2025b.
Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang,
Wenxuan Wang, et al. Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical
dataset for advancing reasoning. arXiv preprint arXiv:2504.11456, 2025.
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint
arXiv:1503.02531, 2015.
Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-
zero: An open source approach to scaling up reinforcement learning on the base model.
arXiv preprint
arXiv:2503.24290, 2025.
Jonas H¨ubotter, Frederike L¨ubeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan
Shenfeld, Thomas Kleine Buening, Carlos Guestrin, et al. Reinforcement learning via self-distillation. arXiv
preprint arXiv:2601.20802, 2026.
Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama,
Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language
models for code. arXiv preprint arXiv:2403.07974, 2024.
Yoon Kim and Alexander M Rush. Sequence-level knowledge distillation. In Proceedings of the 2016 conference
on empirical methods in natural language processing, pp. 1317–1327, 2016.
Kun Liang, Clive Bai, Xin Xu, Chenming Tang, Sanwoo Lee, Weijie Liu, Saiyong Yang, and Yunfang Wu. Orbit:
On-policy exploration-exploitation for controllable multi-budget reasoning. arXiv preprint arXiv:2601.08310,
2026.
Chris Yuhao Liu, Liang Zeng, Yuzhen Xiao, Jujie He, Jiacai Liu, Chaojie Wang, Rui Yan, Wei Shen, Fuxiang Zhang,
Jiacheng Xu, et al. Skywork-reward-v2: Scaling preference data curation via human-ai synergy. arXiv preprint
arXiv:2507.01352, 2025a.
11

Jiacai Liu, Yingru Li, Yuqian Fu, Jiawei Wang, Qian Liu, and Yu Shen. When speed kills stability: Demystify-
ing RL collapse from the training-inference mismatch, September 2025b. URL https://richardli.xyz/
rl-collapse.
Jiawei Liu and Lingming Zhang. Code-r1: Reproducing r1 for code with reliable rewards. 2025.
Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatGPT really
correct? rigorous evaluation of large language models for code generation. In Thirty-seventh Conference on
Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=1qvx610Cu7.
Xiaoqian Liu, Ke Wang, Yuchuan Wu, Fei Huang, Yongbin Li, Junge Zhang, and Jianbin Jiao. Agentic reinforcement
learning with implicit step rewards. arXiv preprint arXiv:2509.19199, 2025c.
Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi:
10.64434/tml.20251026. https://thinkingmachines.ai/blog/on-policy-distillation.
OpenCompass. Aime 2025. https://huggingface.co/datasets/opencompass/AIME2025, 2025.
Carlos Miguel Pati˜no, Kashif Rasul, Quentin Gallou´edec, Ben Burtenshaw, Sergio Paniego, Vaibhav Srivastav,
Thibaud Frere, Ed Beeching, Lewis Tunstall, Leandro von Werra, and Thomas Wolf. Unlocking on-policy
distillation for any model family, 2025.
Emiliano Penaloza, Dheeraj Vattikonda, Nicolas Gontier, Alexandre Lacoste, Laurent Charlin, and Massimo Caccia.
Privileged information distillation for language models. arXiv preprint arXiv:2602.04942, 2026.
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct
preference optimization: Your language model is secretly a reward model. Advances in neural information
processing systems, 36:53728–53741, 2023.
Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller,
faster, cheaper and lighter. arXiv preprint arXiv:1910.01108, 2019.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang,
YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models.
arXiv preprint arXiv:2402.03300, 2024.
Idan Shenfeld, Mehul Damani, Jonas H¨ubotter, and Pulkit Agrawal. Self-distillation enables continual learning.
arXiv preprint arXiv:2601.19897, 2026.
Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and
Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.
Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge,
1998.
Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and
Tatsunori B Hashimoto. Alpaca: A strong, replicable instruction-following model. Stanford Center for Research
on Foundation Models. https://crfm. stanford. edu/2023/03/13/alpaca. html, 3(6):7, 2023.
Bangjun Xiao, Bingquan Xia, Bo Yang, Bofei Gao, Bowen Shen, Chen Zhang, Chenhong He, Chiheng Lou, Fuli
Luo, Gang Wang, et al. Mimo-v2-flash technical report. arXiv preprint arXiv:2601.02780, 2026.
Teng Xiao, Yige Yuan, Mingxiao Li, Zhengyu Chen, and Vasant G Honavar. On a connection between imitation
learning and RLHF. In The Thirteenth International Conference on Learning Representations, 2025. URL
https://openreview.net/forum?id=2QdsjiNXgj.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen
Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.
Wenkai Yang, Jingwen Chen, Yankai Lin, and Ji-Rong Wen. Deepcritic: Deliberate critique with large language
models. arXiv preprint arXiv:2505.00662, 2025b.
Wenkai Yang, Yankai Lin, Jie Zhou, and Ji-Rong Wen. Distilling rule-based knowledge into large language models.
In Proceedings of the 31st International Conference on Computational Linguistics, pp. 913–932, 2025c.
Wenkai Yang, Weijie Liu, Ruobing Xie, Yiju Guo, Lulu Wu, Saiyong Yang, and Yankai Lin. Laser: Reinforcement
learning with last-token self-rewarding. arXiv preprint arXiv:2510.14943, 2025d.
Wenkai Yang, Shuming Ma, Yankai Lin, and Furu Wei. Towards thinking-optimal scaling of test-time compute for
llm reasoning. arXiv preprint arXiv:2502.18080, 2025e.
12

Tianzhu Ye, Li Dong, Zewen Chi, Xun Wu, Shaohan Huang, and Furu Wei. Black-box on-policy distillation of
large language models. arXiv preprint arXiv:2511.10643, 2025a.
Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning.
arXiv preprint arXiv:2502.03387, 2025b.
Lifan Yuan, Wendi Li, Huayu Chen, Ganqu Cui, Ning Ding, Kaiyan Zhang, Bowen Zhou, Zhiyuan Liu, and Hao
Peng. Free process rewards without process labels. arXiv preprint arXiv:2412.01981, 2024.
Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled
reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.
Chujie Zheng, Ziqi Wang, Heng Ji, Minlie Huang, and Nanyun Peng. Model extrapolation expedites alignment.
In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), pp. 1025–1041, 2025.
Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping
Yu, Lili Yu, et al. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36:
55006–55021, 2023.
13

A
Detailed Math Derivations
Here, we make mathematical derivations to calculate the expected gradients of OPD objective in Eq. (4).
Since
JOPD(θ) = min
θ
Ex∼D,y∼πθ(·|x)
h
DKL
 πθ(y|x)
 π∗(y|x)
i
= min
θ
Ex∼D,y∼πθ(·|x)
h
log πθ(y|x) −log π∗(y|x)
i
.
(15)
We can get
∇θJOPD(θ) = ∇θEx∼D,y∼πθ(·|x)
h
log πθ(y|x) −log π∗(y|x)
i
= ∇θEx
h
∑
y
πθ(y|x)
 log πθ(y|x) −log π∗(y|x)
i
= Ex
h
∇θ∑
y
πθ(y|x)
 log πθ(y|x) −log π∗(y|x)
i
= Ex
h
∑
y
 ∇θπθ(y|x)
 log πθ(y|x) −log π∗(y|x)
 + ∑
y
πθ(y|x)∇θ log πθ(y|x)
i
.
(16)
Notice that
Ex
h
∑
y
πθ(y|x)∇θ log πθ(y|x)
i
= Ex
h
∑
y
πθ(y|x)∇θπθ(y|x)
πθ(y|x)
i
= Ex
h
∑
y
∇θπθ(y|x)
i
= Ex
h
∇θ∑
y
πθ(y|x)
i
= Ex
h
∇θ1
i
= 0.
(17)
Therefore, Eq. (16) can be reduced to
∇θJOPD(θ) = Ex
h
∑
y
∇θπθ(y|x)
 log πθ(y|x) −log π∗(y|x)
i
= Ex
h
∑
y
πθ(y|x)∇θ log πθ(y|x)
 log πθ(y|x) −log π∗(y|x)
i
= Ex∼D,y∼πθ(·|x)
h log πθ(y|x) −log π∗(y|x)
∇θ log πθ(y|x)
i
= Ex∼D,y∼πθ(·|x)
h T
∑
t=1
T
∑
t′=1
 log πθ(yt′ |x, y<t′ ) −log π∗(yt′ |x, y<t′ )
 ∇θ log πθ(yt|x, y<t)
i
.
(18)
Now let’s denote
∆t′ =
 log πθ(yt′ |x, y<t′ ) −log π∗(yt′ |x, y<t′ )

,
and consider each term Ex∼D,y∼πθ(·|x)
h
∆t′ ∇θ log πθ(yt|x, y<t)
i
where t
′ < t:
Ex,y
h
∆t′ ∇θ log πθ(yt|x, y<t)
i
= Ex,y
h
Eyt
h
∆t′ ∇θ log πθ(yt|x, y<t)
x, y<t
ii
= Ex,y
h
∆t′ Eyt
h
∇θ log πθ(yt|x, y<t)
x, y<t
ii
= Ex,y
h
∆t′ Eyt∼πθ(·|x,y<t)
∇θ log πθ(yt|x, y<t)
i
= Ex,y
h
∆t′ ∑
yt
∇θπθ(yt|x, y<t)
i
= Ex,y
h
∆t′ ∇θ∑
yt
πθ(yt|x, y<t)
i
= Ex,y
h
∆t′ ∇θ1
i
= 0.
(19)
14

Table 4: Training hyper-parameters of GRPO in math
RL.
Hyper-parameter
Value
Train Batch Size
128
Micro Batch Size
128
Rollout n
8
Maximum Prompt Length
2048
Maximum Response Length
16,384
Temperature
1.0
Top-p
1.0
LR
1 × 10−6
Optimization Steps
500
KL Coefficient
0.0
Table 5: Training hyper-parameters of GRPO in code
RL.
Hyper-parameter
Value
Train Batch Size
128
Micro Batch Size
128
Rollout n
8
Maximum Prompt Length
2048
Maximum Response Length
8192
Temperature
1.0
Top-p
1.0
LR
1 × 10−6
Optimization Steps
300
KL Coefficient
0.0
Table 6: Training hyper-parameters of G-OPD in both
math and code domains.
Hyper-parameter
Value
Batch Size
1024
Rollout n
1
Maximum Prompt Length
2048
Maximum Response Length
16,384
Temperature
1.0
Top-p
1.0
LR
1 × 10−5
Table 7: Training hyper-parameters of SFT in both math
and code domains.
Hyper-parameter
Value
Batch Size
1024
Maximum Sequence Length
32,768
Warm-up Ratio
0.05
LR
1 × 10−5
Therefore, Eq. (18) can be reduced to
∇θJOPD(θ) = Ex∼D,y∼πθ(·|x)
h T
∑
t=1

T
∑
t′=t
 log πθ(yt′ |x, y<t′ ) −log π∗(yt′ |x, y<t′ )

∇θ log πθ(yt|x, y<t)
i
.
(20)
In practice, recent studies (Lu & Lab, 2025; Xiao et al., 2026) use a discount factor of 0 and approximate the
gradient as
∇θJOPD(θ) = Ex∼D,y∼πθ(·|x)
h T
∑
t=1
 log πθ(yt|x, y<t) −log π∗(yt|x, y<t)
 ∇θ log πθ(yt|x, y<t)
i
.
(21)
Similarly, the approximated gradient of G-OPD in Eq. (11) can be written as
∇θJG-OPD(θ) = Ex∼D,y∼πθ(·|x)
h T
∑
t=1
AG-OPD
t
∇θ log πθ(yt|x, y<t)
i
,
(22)
where AG-OPD
t
=
 log πθ(yt|x, y<t) −log π∗(yt|x, y<t)
 + (λ −1)
 log πref(yt|x, y<t) −log π∗(yt|x, y<t)

.
B
Detailed Training Settings
The training hyper-parameters in math and code RL training are put in Table 4 and Table 5 respectively.
The training hyper-parameters in G-OPD in both domains are in Table 6. In preliminary experiments, we find that
under the same prompt size ∗rollout n conditions, setting a larger prompt size leads to smoother convergence. The
number of optimization steps for G-OPD in all experiments with same-size teacher-student pairs (Section 4.1) is set
to 50, while it is set to 100 for experiments in the strong-to-weak distillation setting (Section 4.2). We find that
further increasing the number of distillation steps may degrade generalization performance due to overfitting.
The training hyper-parameters in SFT are in Table 7. We make sure the number of trajectories to each problem
generated by the teacher in SFT is consistent with that generated by the student in OPD and ExOPD. We keep the
number of optimization steps consistent with the corresponding G-OPD experiment for fair comparison.
15

Table 8: Distillation results under stronger teachers with sufficient RL trainings. Each numerical subscript indicates
the absolute improvement or degradation compared to the domain teacher model.
Method
Math Reasoning
Code Generation
AIME24 AIME25 HMMT25 (Feb.) HMMT25 (Nov.)
Avg.
HumanEval+
MBPP+
LCB
Avg.
Teacher
68.2
59.3
37.3
42.9
51.9
88.9
72.5
28.0
63.1
Student
21.5
21.9
10.0
8.0
15.4
74.7
64.7
17.9
52.4
Single-Teacher Distillation
OPD
68.3+0.1
58.7−0.6
38.7+1.4
41.2−1.7
51.7−0.2
89.3+0.4
71.3−1.2
28.0+0.0
62.9−0.2
ExOPD
68.4+0.2
59.2−0.1
38.2+0.9
42.8−0.1
52.2+0.3
89.9+1.0
73.7+1.2
29.3+1.3
64.3+1.2
Multi-Teacher Distillation
OPD
68.2+0.0
60.2+0.9
38.5+1.2
40.8−2.1
51.9+0.0
86.4−2.5
72.1−0.4
27.6−0.4
62.0−1.1
ExOPD
70.1+1.9
59.6+0.3
37.5+0.2
42.7−0.2
52.5+0.6
89.5+0.6
73.9+1.4
29.7+1.7
64.4+1.3
C
Results of Distillation from Domain Teachers with Sufficient RL Trainings
Here, we show the on-policy distillation results when the domain teachers are trained with sufficient RL steps (i.e.,
1200 steps). The experimental settings are the same as that in the Section 4.1. The results in Table 8 demonstrate the
generalizability and effectiveness of ExOPD in this case.
D
Prompt Templates
We show the prompt templates used in our experiments in the end.
16

Training and Evaluation Prompt Template for Math Reasoning
< |im start| >user
{question}
Please reason step by step, and put your final answer within \boxed{}.< |im end| >
< |im start| >assistant
Training and Evaluation Prompt Template for Code Generation
< |im start| >user
{question}
Write Python code to solve the problem. Present the code in
‘‘‘python
Your code
‘‘‘
at the end.
You need to think first then write the Python code.< |im end| >
< |im start| >assistant
17

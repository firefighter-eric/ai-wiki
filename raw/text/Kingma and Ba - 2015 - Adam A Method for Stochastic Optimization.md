# Kingma and Ba - 2015 - Adam: A Method for Stochastic Optimization

- Source PDF: `raw/pdf/Kingma and Ba - 2015 - Adam A Method for Stochastic Optimization.pdf`
- Source SHA256: `eab9c73ae2ceda884b94830bda99312254bac4806f6c9f045cbab90721ecda31`
- Source URL: https://arxiv.org/abs/1412.6980
- Generated from: `scripts/extract_pdf_text.py`

- Extraction: `pymupdf-pages-v2` (sorted text, page anchors; tables/formulas/figures require review)

## Extracted Text

<a id="source-section-0"></a>

<a id="page-1"></a>

### PDF 第 1 页

Published as a conference paper at ICLR 2015


         ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION

                                         Diederik P. Kingma*              Jimmy Lei Ba∗
                                        University of Amsterdam, OpenAI          University of Toronto
                                  dpkingma@openai.com         jimmy@psi.utoronto.ca


                                       ABSTRACT

                    We introduce Adam, an algorithm for ﬁrst-order gradient-based optimization of
                                  stochastic objective functions, based on adaptive estimates of lower-order mo-
                              ments. The method is straightforward to implement, is computationally efﬁcient,2017                        has little memory requirements, is invariant to diagonal rescaling of the gradients,
                            and is well suited for problems that are large in terms of data and/or parameters.
                          The method is also appropriate for non-stationary objectives and problems withJan                        very noisy and/or sparse gradients. The hyper-parameters have intuitive interpre-
                                   tations and typically require little tuning. Some connections to related algorithms,
30                     on which Adam was inspired, are discussed. We also analyze the theoretical con-
                              vergence properties of the algorithm and provide a regret bound on the conver-
                             gence rate that is comparable to the best known results under the online convex
                                optimization framework. Empirical results demonstrate that Adam works well in
                                  practice and compares favorably to other stochastic optimization methods. Finally,
                       we discuss AdaMax, a variant of Adam based on the inﬁnity norm.[cs.LG]
                1  INTRODUCTION
                        Stochastic gradient-based optimization is of core practical importance in many ﬁelds of science and
                       engineering. Many problems in these ﬁelds can be cast as the optimization of some scalar parameter-
                       ized objective function requiring maximization or minimization with respect to its parameters. If the
                       function is differentiable w.r.t. its parameters, gradient descent is a relatively efﬁcient optimization
                     method, since the computation of ﬁrst-order partial derivatives w.r.t. all the parameters is of the same
                      computational complexity as just evaluating the function. Often, objective functions are stochastic.
                     For example, many objective functions are composed of a sum of subfunctions evaluated at different
                     subsamples of data; in this case optimization can be made more efﬁcient by taking gradient steps
                              w.r.t. individual subfunctions, i.e. stochastic gradient descent (SGD) or ascent. SGD proved itself
                       as an efﬁcient and effective optimization method that was central in many machine learning success
                           stories, such as recent advances in deep learning (Deng et al., 2013; Krizhevsky et al., 2012; Hinton
            & Salakhutdinov, 2006; Hinton et al., 2012a; Graves et al., 2013). Objectives may also have other
                       sources of noise than data subsampling, such as dropout (Hinton et al., 2012b) regularization. ForarXiv:1412.6980v9                             all such noisy objectives, efﬁcient stochastic optimization techniques are required. The focus of this
                      paper is on the optimization of stochastic objectives with high-dimensional parameters spaces. In
                        these cases, higher-order optimization methods are ill-suited, and discussion in this paper will be
                          restricted to ﬁrst-order methods.
               We propose Adam, a method for efﬁcient stochastic optimization that only requires ﬁrst-order gra-
                        dients with little memory requirement. The method computes individual adaptive learning rates for
                          different parameters from estimates of ﬁrst and second moments of the gradients; the name Adam
                             is derived from adaptive moment estimation. Our method is designed to combine the advantages
                       of two recently popular methods: AdaGrad (Duchi et al., 2011), which works well with sparse gra-
                          dients, and RMSProp (Tieleman & Hinton, 2012), which works well in on-line and non-stationary
                          settings; important connections to these and other stochastic optimization methods are clariﬁed in
                        section 5. Some of Adam’s advantages are that the magnitudes of parameter updates are invariant to
                        rescaling of the gradient, its stepsizes are approximately bounded by the stepsize hyperparameter,
                                   it does not require a stationary objective, it works with sparse gradients, and it naturally performs a
                    form of step size annealing.

                          ∗Equal contribution. Author ordering determined by coin ﬂip over a Google Hangout.


                                                           1
<a id="page-2"></a>

### PDF 第 2 页

Published as a conference paper at ICLR 2015




Algorithm 1: Adam, our proposed algorithm for stochastic optimization. See section 2 for details,
and for a slightly more efﬁcient (but less clear) order of computation. g2t indicates the elementwise
square gt ⊙gt. Good default settings for the tested machine learning problems are α = 0.001,β1 = 0.9, β2 = 0.999 and ϵ = 10−8. All operations on vectors are element-wise. With βt1 and βt2
we denote β1 and β2 to the power t.
Require: α: Stepsize
Require: β1, β2 ∈[0, 1): Exponential decay rates for the moment estimatesRequire: f(θ): Stochastic objective function with parameters θ
Require: θ0: Initial parameter vector
  m0 ←0 (Initialize 1st moment vector)
  v0 ←0 (Initialize 2nd moment vector)
   t ←0 (Initialize timestep)  while θt not converged do
      t ←t + 1
     gt ←∇θft(θt−1) (Get gradients w.r.t. stochastic objective at timestep t)
   mt ←β1 · mt−1 + (1 −β1) · gt (Update biased ﬁrst moment estimate)                                              t (Update biased second raw moment estimate)     vt ←β2 · vt−1 + (1 −β2) · g2
     bmt ←mt/(1 −βt1) (Compute bias-corrected ﬁrst moment estimate)
        bvt ←vt/(1 −βt2) (Compute bias-corrected second raw moment estimate)     θt               bmt/(√bvt + ϵ) (Update parameters)     ←θt−1 −α ·  end while
  return θt (Resulting parameters)


In section 2 we describe the algorithm and the properties of its update rule.  Section 3 explains
our initialization bias correction technique, and section 4 provides a theoretical analysis of Adam’s
convergence in online convex programming. Empirically, our method consistently outperforms other
methods for a variety of models and datasets, as shown in section 6. Overall, we show that Adam is
a versatile algorithm that scales to large-scale high-dimensional machine learning problems.

2  ALGORITHM

See algorithm 1 for pseudo-code of our proposed algorithm Adam.  Let f(θ) be a noisy objec-
tive function: a stochastic scalar function that is differentiable w.r.t.  parameters θ. We are in-
terested in minimizing the expected value of this function, E[f(θ)] w.r.t.  its parameters θ. With
f1(θ), ..., , fT (θ) we denote the realisations of the stochastic function at subsequent timesteps
1, ..., T. The stochasticity might come from the evaluation at random subsamples (minibatches)
of datapoints, or arise from inherent function noise. With gt = ∇θft(θ) we denote the gradient, i.e.the vector of partial derivatives of ft, w.r.t θ evaluated at timestep t.
The algorithm updates exponential moving averages of the gradient (mt) and the squared gradient
(vt) where the hyper-parameters β1, β2 ∈[0, 1) control the exponential decay rates of these movingaverages. The moving averages themselves are estimates of the 1st moment (the mean) and the
2nd raw moment (the uncentered variance) of the gradient. However, these moving averages are
initialized as (vectors of) 0’s, leading to moment estimates that are biased towards zero, especially
during the initial timesteps, and especially when the decay rates are small (i.e. the βs are close to 1).
The good news is that this initialization bias can be easily counteracted, resulting in bias-corrected
estimates bmt and bvt. See section 3 for more details.
Note that the efﬁciency of algorithm 1 can, at the expense of clarity, be improved upon by changing
the order of computation, e.g. by replacing the last three lines in the loop with the following lines:    p
αt = α   1    2/(1    1) and θt            mt/(√vt + ˆϵ).           ·   −βt   −βt      ←θt−1 −αt ·

2.1  ADAM’S UPDATE RULE

An important property of Adam’s update rule is its careful choice of stepsizes. Assuming ϵ = 0, the
effective step taken in parameter space at timestep t is ∆t = α  bmt/√bvt. The effective stepsize has                                                                                          ·two upper bounds:           (1                  in the case (1    > √1      and                  |∆t| ≤α ·  −β1)/√1 −β2          −β1)     −β2,     |∆t| ≤α

                                       2
<a id="page-3"></a>

### PDF 第 3 页

Published as a conference paper at ICLR 2015




otherwise. The ﬁrst case only happens in the most severe case of sparsity: when a gradient has
been zero at all timesteps except at the current timestep. For less sparse cases, the effective stepsize
will be smaller. When (1    = √1    we have that      < 1 therefore   < α. In                             p                 −β1)     −β2                     |bmt/√bvt|              |∆t|more common scenarios, we will have that bmt/√bvt       since                 The effective                           ≈±1      |E[g]/  E[g2]| ≤1.magnitude of the steps taken in parameter space at each timestep are approximately bounded by
the stepsize setting α, i.e., |∆t| ⪅α. This can be understood as establishing a trust region aroundthe current parameter value, beyond which the current gradient estimate does not provide sufﬁcient
information. This typically makes it relatively easy to know the right scale of α in advance. For
many machine learning models, for instance, we often know in advance that good optima are with
high probability within some set region in parameter space; it is not uncommon, for example, to
have a prior distribution over the parameters. Since α sets (an upper bound of) the magnitude of
steps in parameter space, we can often deduce the right order of magnitude of α such that optima
can be reached from θ0 within some number of iterations. With a slight abuse of terminology,
we will call the ratio bmt/√bvt the signal-to-noise ratio (SNR). With a smaller SNR the effective
stepsize ∆t will be closer to zero. This is a desirable property, since a smaller SNR means that
there is greater uncertainty about whether the direction of bmt corresponds to the direction of the true
gradient. For example, the SNR value typically becomes closer to 0 towards an optimum, leading
to smaller effective steps in parameter space: a form of automatic annealing. The effective stepsize
∆t is also invariant to the scale of the gradients; rescaling the gradients g with factor c will scale bmt
with a factor c and bvt with a factor c2, which cancel out: (c  bmt)/(√ c2   bvt) = bmt/√bvt.                                                                                      ·                 ·

3  INITIALIZATION BIAS CORRECTION

As explained in section 2, Adam utilizes initialization bias correction terms. We will here derive
the term for the second moment estimate; the derivation for the ﬁrst moment estimate is completely
analogous. Let g be the gradient of the stochastic objective f, and we wish to estimate its second
raw moment (uncentered variance) using an exponential moving average of the squared gradient,
with decay rate β2. Let g1, ..., gT be the gradients at subsequent timesteps, each a draw from an
underlying gradient distribution gt ∼p(gt). Let us initialize the exponential moving average asv0 = 0 (a vector of zeros). First note that the update at timestep t of the exponential moving average
vt = β2 · vt−1 + (1 −β2) · g2                                      t (where g2t indicates the elementwise square gt ⊙gt) can be written asa function of the gradients at all previous timesteps:

              Xt
                                                                                        i                                  (1)                                                        2                                    vt = (1 −β2)    βt−i                                                                                      · g2
                                               i=1

We wish to know how E[vt], the expected value of the exponential moving average at timestep t,
relates to the true second moment E[g2t ], so we can correct for the discrepancy between the two.
Taking expectations of the left-hand and right-hand sides of eq. (1):
                            "               #
              Xt
                                                                                          i                                 (2)                                                         2                              E[vt] = E  (1 −β2)    βt−i                                                                                        · g2
                                                i=1
               Xt
                                                             2 + ζ                             (3)                   = E[g2t ] · (1 −β2)    βt−i
                                                    i=1
                   = E[g2t ] · (1 −βt2) + ζ                                      (4)
where ζ = 0 if the true second moment E[g2i ] is stationary; otherwise ζ can be kept small since
the exponential decay rate β1 can (and should) be chosen such that the exponential moving average
assigns small weights to gradients too far in the past. What is left is the term (1 −βt2) which iscaused by initializing the running average with zeros. In algorithm 1 we therefore divide by this
term to correct the initialization bias.
In case of sparse gradients, for a reliable estimate of the second moment one needs to average over
many gradients by chosing a small value of β2; however it is exactly this case of small β2 where a
lack of initialisation bias correction would lead to initial steps that are much larger.

                                       3
<a id="page-4"></a>

### PDF 第 4 页

Published as a conference paper at ICLR 2015



4  CONVERGENCE ANALYSIS

We analyze the convergence of Adam using the online learning framework proposed in (Zinkevich,
2003). Given an arbitrary, unknown sequence of convex cost functions f1(θ), f2(θ),..., fT (θ). At
each time t, our goal is to predict the parameter θt and evaluate it on a previously unknown cost
function ft. Since the nature of the sequence is unknown in advance, we evaluate our algorithm
using the regret, that is the sum of all the previous difference between the online prediction ft(θt)
and the best ﬁxed point parameter ft(θ∗) from a feasible set X for all the previous steps. Concretely,the regret is deﬁned as:

            XT
                         R(T) =    [ft(θt) −ft(θ∗)]                                 (5)
                                        t=1
             PT                             √ T) regret bound and a proof is givenwhere θ∗= arg                         t=1 ft(θ). We show Adam has O(            minθ∈Xin the appendix. Our result is comparable to the best known bound for this general convex online
learning problem. We also use some deﬁnitions simplify our notation, where gt ≜∇ft(θt) and gt,i
as the ith element. We deﬁne g1:t,i ∈Rt as a vector that contains the ith dimensionβ21  of the gradients
                                                                 √β2 . Our followingover all iterations till t, g1:t,i = [g1,i, g2,i, · · · , gt,i]. Also, we deﬁne γ ≜
                                                                                   2 and ﬁrst moment runningtheorem holds when the learning rate αt is decaying at a rate of t−1
average coefﬁcient β1,t decay exponentially with λ, that is typically close to 1, e.g. 1 −10−8.
Theorem 4.1. Assume that the function ft has bounded gradients, ∥∇ft(θ)∥2 ≤G, ∥∇ft(θ)∥∞≤
G∞for all θ ∈Rd and distance between any θt generated by Adam is bounded,β21 ∥θn −θm∥2 ≤D,α
                                                                                                                                  t∥θm −θn∥∞≤D∞for any m, n ∈{1, ..., T}, and β1, β2 ∈[0, 1) satisfy √β2 < 1. Let αt = √
and β1,t = β1λt−1, λ ∈(0, 1). Adam achieves the following guarantee, for all T ≥1.
         D2 Xd p           α(1 +    Xd    Xd  D2   √1R(T)                    TbvT,i+         β1)G∞               ∞G∞  −β2   ≤ 2α(1                    (1                          ∥g1:T,i∥2+   2α(1                                                           i=1           i=1     −β1)(1 −λ)2          −β1) i=1         −β1)√1 −β2(1 −γ)2

Our Theorem 4.1 implies when the data features are sparse and bounded gradients, the sum-
                                  Pd                                           √ T andmation       term can be much smaller than  its upper bound                                                            i=1   p                                                                    ∥g1:T,i∥2 << dG∞Pd            √ T, in particular if the class of function and data features are in the form of  i=1   TbvT,i <<           dG∞
section 1.2 in (Duchi et al., 2011). Their results for the expected value E[Pdi=1           also apply                                                                                ∥g1:T,i∥2]to Adam. In particular, the adaptive method, such as Adam and Adagrad, can achieve O(log d√ T),
an improvement over O(√ dT) for the non-adaptive method. Decaying β1,t towards zero is impor-
tant in our theoretical analysis and also matches previous empirical ﬁndings, e.g. (Sutskever et al.,
2013) suggests reducing the momentum coefﬁcient in the end of training can improve convergence.
Finally, we can show the average regret of Adam converges,
Corollary 4.2. Assume that the function ft has bounded gradients, ∥∇ft(θ)∥2 ≤G, ∥∇ft(θ)∥∞≤
G∞for all θ ∈Rd and distance between any θt generated by Adam is bounded, ∥θn −θm∥2 ≤D,
∥θm −θn∥∞≤D∞for any m, n ∈{1, ..., T}. Adam achieves the following guarantee, for all
T ≥1.                        R(T)       1
                             T                        = O( √ T )
                                 Pd
This result can be obtained by using Theorem 4.1 and                                          √ T.  Thus,                                                          i=1        R(T )                                                    ∥g1:T,i∥2 ≤dG∞        = 0.limT  →∞  T

5  RELATED WORK

Optimization methods bearing a direct relation to Adam are RMSProp (Tieleman & Hinton, 2012;
Graves, 2013) and AdaGrad (Duchi et al., 2011); these relationships are discussed below. Other
stochastic optimization methods include vSGD (Schaul et al., 2012), AdaDelta (Zeiler, 2012) and the
natural Newton method from Roux & Fitzgibbon (2010), all setting stepsizes by estimating curvature

                                       4
<a id="page-5"></a>

### PDF 第 5 页

Published as a conference paper at ICLR 2015




from ﬁrst-order information. The Sum-of-Functions Optimizer (SFO) (Sohl-Dickstein et al., 2014)
is a quasi-Newton method based on minibatches, but (unlike Adam) has memory requirements linear
in the number of minibatch partitions of a dataset, which is often infeasible on memory-constrained
systems such as a GPU. Like natural gradient descent (NGD) (Amari, 1998), Adam employs a
preconditioner that adapts to the geometry of the data, since bvt is an approximation to the diagonal
of the Fisher information matrix (Pascanu & Bengio, 2013); however, Adam’s preconditioner (like
AdaGrad’s) is more conservative in its adaption than vanilla NGD by preconditioning with the square
root of the inverse of the diagonal Fisher information matrix approximation.

RMSProp:  An optimization method closely related to Adam is RMSProp (Tieleman & Hinton,
2012). A version with momentum has sometimes been used (Graves, 2013). There are a few impor-
tant differences between RMSProp with momentum and Adam: RMSProp with momentum gener-
ates its parameter updates using a momentum on the rescaled gradient, whereas Adam updates are
directly estimated using a running average of ﬁrst and second moment of the gradient. RMSProp
also lacks a bias-correction term; this matters most in case of a value of β2 close to 1 (required in
case of sparse gradients), since in that case not correcting the bias leads to very large stepsizes and
often divergence, as we also empirically demonstrate in section 6.4.

AdaGrad:  An algorithm that works well for sparse gradients is AdaGrad (Duchi et al., 2011). Its
                            qPt
                                                         g2t . Note that if we choose β2 to bebasic version updates parameters as θt+1 = θt −α · gt/                                                        i=1Pt
                                                             i=1 g2t . AdaGrad corresponds to ainﬁnitesimally close to 1 from below, then limβ2→1 bvt = t−1 ·                                                                          annealed                                                                                        versionversion of Adam with β1 = 0, inﬁnitesimal (1                                q                  p−β2) and a replacement of α by an                                                    Pt
                                                                                    i=1 g2t =αt = α · t−1/2, namely θt −α · t−1/2 · bmt/  limβ2→1 bvt = θt −α · t−1/2 · gt/  t−1 ·      qPt
                 i=1 g2t . Note that this direct correspondence between Adam and Adagrad doesθt −α · gt/
not hold when removing the bias-correction terms; without bias correction, like in RMSProp, a β2
inﬁnitesimally close to 1 would lead to inﬁnitely large bias, and inﬁnitely large parameter updates.

6  EXPERIMENTS

To empirically evaluate the proposed method, we investigated different popular machine learning
models, including logistic regression, multilayer fully connected neural networks and deep convolu-
tional neural networks. Using large models and datasets, we demonstrate Adam can efﬁciently solve
practical deep learning problems.
We use the same parameter initialization when comparing different optimization algorithms. The
hyper-parameters, such as learning rate and momentum, are searched over a dense grid and the
results are reported using the best hyper-parameter setting.

6.1  EXPERIMENT: LOGISTIC REGRESSION

We evaluate our proposed method on L2-regularized multi-class logistic regression using the MNIST
dataset. Logistic regression has a well-studied convex objective, making it suitable for comparison
of different optimizers without worrying about local minimum issues. The stepsize α in our logistic
                                                                          that matches with our theorat-regression experiments is adjusted by 1/√ t decay, namely αt = √αt
ical prediction from section 4. The logistic regression classiﬁes the class label directly on the 784
dimension image vectors. We compare Adam to accelerated SGD with Nesterov momentum and
Adagrad using minibatch size of 128. According to Figure 1, we found that the Adam yields similar
convergence as SGD with momentum and both converge faster than Adagrad.
As discussed in (Duchi et al., 2011), Adagrad can efﬁciently deal with sparse features and gradi-
ents as one of its main theoretical results whereas SGD is low at learning rare features. Adam with
1/ √ t decay on its stepsize should theoratically match the performance of Adagrad. We examine the
sparse feature problem using IMDB movie review dataset from (Maas et al., 2011). We pre-process
the IMDB movie reviews into bag-of-words (BoW) feature vectors including the ﬁrst 10,000 most
frequent words. The 10,000 dimension BoW feature vector for each review is highly sparse. As sug-
gested in (Wang & Manning, 2013), 50% dropout noise can be applied to the BoW features during

                                       5
<a id="page-6"></a>

### PDF 第 6 页

Published as a conference paper at ICLR 2015





           0.7                     MNIST Logistic Regression                        0.50    IMDB BoW feature Logistic Regression
                                         AdaGrad                                   Adagrad+dropout
                                           SGDNesterov                               RMSProp+dropout
                                    Adam                    0.45                   SGDNesterov+dropout           0.6
                                                                         Adam+dropout
                                                                         0.40
              cost 0.5                                                                                  cost
                            training           0.4                                                                                                                                                                    training 0.35
                                                                         0.30

           0.3                                                              0.25



           0.2           0     5    10    15    20    25    30    35    40    45          0.200    20   40   60   80   100  120  140  160
                                  iterations over entire dataset                                                iterations over entire dataset

Figure 1: Logistic regression training negative log likelihood on MNIST images and IMDB movie
reviews with 10,000 bag-of-words (BoW) feature vectors.


training to prevent over-ﬁtting. In ﬁgure 1, Adagrad outperforms SGD with Nesterov momentum
by a large margin both with and without dropout noise. Adam converges as fast as Adagrad. The
empirical performance of Adam is consistent with our theoretical ﬁndings in sections 2 and 4. Sim-
ilar to Adagrad, Adam can take advantage of sparse features and obtain faster convergence rate than
normal SGD with momentum.

6.2  EXPERIMENT: MULTI-LAYER NEURAL NETWORKS

Multi-layer neural network are powerful models with non-convex objective functions. Although
our convergence analysis does not apply to non-convex problems, we empirically found that Adam
often outperforms other methods in such cases. In our experiments, we made model choices that are
consistent with previous publications in the area; a neural network model with two fully connected
hidden layers with 1000 hidden units each and ReLU activation are used for this experiment with
minibatch size of 128.
First, we study different optimizers using the standard deterministic cross-entropy objective func-
tion with L2 weight decay on the parameters to prevent over-ﬁtting. The sum-of-functions (SFO)
method (Sohl-Dickstein et al., 2014) is a recently proposed quasi-Newton method that works with
minibatches of data and has shown good performance on optimization of multi-layer neural net-
works. We used their implementation and compared with Adam to train such models. Figure 2
shows that Adam makes faster progress in terms of both the number of iterations and wall-clock
time. Due to the cost of updating curvature information, SFO is 5-10x slower per iteration com-
pared to Adam, and has a memory requirement that is linear in the number minibatches.
Stochastic regularization methods, such as dropout, are an effective way to prevent over-ﬁtting and
often used in practice due to their simplicity. SFO assumes deterministic subfunctions, and indeed
failed to converge on cost functions with stochastic regularization. We compare the effectiveness of
Adam to other stochastic ﬁrst order methods on multi-layer neural networks trained with dropout
noise. Figure 2 shows our results; Adam shows better convergence than other methods.

6.3  EXPERIMENT: CONVOLUTIONAL NEURAL NETWORKS

Convolutional neural networks (CNNs) with several layers of convolution, pooling and non-linear
units have shown considerable success in computer vision tasks. Unlike most fully connected neural
nets, weight sharing in CNNs results in vastly different gradients in different layers. A smaller
learning rate for the convolution layers is often used in practice when applying SGD. We show the
effectiveness of Adam in deep CNNs. Our CNN architecture has three alternating stages of 5x5
convolution ﬁlters and 3x3 max pooling with stride of 2 that are followed by a fully connected layer
of 1000 rectiﬁed linear hidden units (ReLU’s). The input image are pre-processed by whitening, and

                                       6
<a id="page-7"></a>

### PDF 第 7 页

Published as a conference paper at ICLR 2015



                   10 -1     MNIST Multilayer Neural Network + dropout
                                                   AdaGrad
                                                  RMSProp
                                                    SGDNesterov
                                                       AdaDelta
                                           Adam


                         cost
                                                  training

                   10 -2





                      0           50          100          150          200
                                                 iterations over entire dataset

                                     (a)                                        (b)

Figure 2: Training of multilayer neural networks on MNIST images.  (a) Neural networks using
dropout stochastic regularization. (b) Neural networks with deterministic cost function. We compare
with the sum-of-functions (SFO) optimizer (Sohl-Dickstein et al., 2014)





                   CIFAR10 ConvNet First 3 Epoches                                     CIFAR10 ConvNet
           3.0
                                  AdaGrad                                                  AdaGrad
                                                                                   102
                                   AdaGrad+dropout                                           AdaGrad+dropout
                                   SGDNesterov                                               SGDNesterov
           2.5                         SGDNesterov+dropout               101                          SGDNesterov+dropout
                             Adam                                             Adam
                                 Adam+dropout                                           Adam+dropout
           2.0                                                                       100
              cost                                                                                                                                           cost

                                                                                                10-1                            training                                                                                                                                                                                                                                                                                       training
           1.5

                                                                                                10-2


           1.0
                                                                                                10-3



           0.5             0.0       0.5       1.0       1.5       2.0       2.5       3.0             10-4 0     5    10    15    20    25    30    35    40    45
                                  iterations over entire dataset                                                           iterations over entire dataset


Figure 3: Convolutional neural networks training cost. (left) Training cost for the ﬁrst three epochs.
(right) Training cost over 45 epochs. CIFAR-10 with c64-c64-c128-1000 architecture.




dropout noise is applied to the input layer and fully connected layer. The minibatch size is also set
to 128 similar to previous experiments.
Interestingly, although both Adam and Adagrad make rapid progress lowering the cost in the initial
stage of the training, shown in Figure 3 (left), Adam and SGD eventually converge considerably
faster than Adagrad for CNNs shown in Figure 3 (right). We notice the second moment estimate bvt
vanishes to zeros after a few epochs and is dominated by the ϵ in algorithm 1. The second moment
estimate is therefore a poor approximation to the geometry of the cost function in CNNs comparing
to fully connected network from Section 6.2. Whereas, reducing the minibatch variance through
the ﬁrst moment is more important in CNNs and contributes to the speed-up. As a result, Adagrad
converges much slower than others in this particular experiment. Though Adam shows marginal
improvement over SGD with momentum, it adapts learning rate scale for different layers instead of
hand picking manually as in SGD.

                                       7
<a id="page-8"></a>

### PDF 第 8 页

Published as a conference paper at ICLR 2015




               β2=0.99      β2=0.999      β2=0.9999          β2=0.99      β2=0.999      β2=0.9999


  β1=0


 β1=0.9  Loss


                   log10(α)
                            (a) after 10 epochs                                 (b) after 100 epochs

Figure 4: Effect of bias-correction terms (red line) versus no bias correction terms (green line)
after 10 epochs (left) and 100 epochs (right) on the loss (y-axes) when learning a Variational Auto-
Encoder (VAE) (Kingma & Welling, 2013), for different settings of stepsize α (x-axes) and hyper-
parameters β1 and β2.



6.4  EXPERIMENT: BIAS-CORRECTION TERM

We also empirically evaluate the effect of the bias correction terms explained in sections 2 and 3.
Discussed in section 5, removal of the bias correction terms results in a version of RMSProp (Tiele-
man & Hinton, 2012) with momentum. We vary the β1 and β2 when training a variational auto-
encoder (VAE) with the same architecture as in (Kingma & Welling, 2013) with a single hidden
layer with 500 hidden units with softplus nonlinearities and a 50-dimensional spherical Gaussian
latent variable. We iterated over a broad range of hyper-parameter choices, i.e. β1 ∈[0, 0.9] and
β2 ∈[0.99, 0.999, 0.9999], and log10(α) ∈[−5, ..., −1]. Values of β2 close to 1, required for robust-ness to sparse gradients, results in larger initialization bias; therefore we expect the bias correction
term is important in such cases of slow decay, preventing an adverse effect on optimization.
In Figure 4, values β2 close to 1 indeed lead to instabilities in training when no bias correction term
was present, especially at ﬁrst few epochs of the training. The best results were achieved with small
values of (1−β2) and bias correction; this was more apparent towards the end of optimization whengradients tends to become sparser as hidden units specialize to speciﬁc patterns. In summary, Adam
performed equal or better than RMSProp, regardless of hyper-parameter setting.


7  EXTENSIONS


7.1  ADAMAX

In Adam, the update rule for individual weights is to scale their gradients inversely proportional to a
(scaled) L2 norm of their individual current and past gradients. We can generalize the L2 norm based
update rule to a Lp norm based update rule. Such variants become numerically unstable for large
p. However, in the special case where we let p →∞, a surprisingly simple and stable algorithmemerges; see algorithm 2. We’ll now derive the algorithm. Let, in case of the Lp norm, the stepsize
at time t be inversely proportional to v1/pt    , where:

                                  vt = βp2vt−1 + (1 −βp2)|gt|p                                   (6)
             Xt

                                                      2                   = (1 −βp2)    βp(t−i)                                                                                      · |gi|p                                (7)
                                             i=1


                                       8
<a id="page-9"></a>

### PDF 第 9 页

Published as a conference paper at ICLR 2015




Algorithm 2: AdaMax, a variant of Adam based on the inﬁnity norm. See section 7.1 for details.
Good default settings for the tested machine learning problems are α = 0.002, β1 = 0.9 and
β2 = 0.999. With βt1 we denote β1 to the power t. Here, (α/(1 −βt1)) is the learning rate with thebias-correction term for the ﬁrst moment. All operations on vectors are element-wise.
Require: α: Stepsize
Require: β1, β2 ∈[0, 1): Exponential decay ratesRequire: f(θ): Stochastic objective function with parameters θ
Require: θ0: Initial parameter vector
  m0 ←0 (Initialize 1st moment vector)
  u0 ←0 (Initialize the exponentially weighted inﬁnity norm)
   t ←0 (Initialize timestep)  while θt not converged do
      t ←t + 1
     gt ←∇θft(θt−1) (Get gradients w.r.t. stochastic objective at timestep t)
   mt ←β1 · mt−1 + (1 −β1) · gt (Update biased ﬁrst moment estimate)
     ut ←max(β2 · ut−1, |gt|) (Update the exponentially weighted inﬁnity norm)
     θt ←θt−1 −(α/(1 −βt1)) · mt/ut (Update parameters)  end while
  return θt (Resulting parameters)

Note that the decay term is here equivalently parameterised as βp2 instead of β2. Now let p →∞,
and deﬁne ut = limp→∞(vt)1/p, then:
                                                                          t            !1/p               X

                                                            2                                                                                                · |gi|p                         (8)               ut = p→∞(vt)1/plim    = p→∞lim   (1 −βp2) i=1 βp(t−i)
                                                                              t            !1/p                X

                                                               2                                                                                                    · |gi|p                      (9)                  = p→∞(1lim  −βp2)1/p   i=1 βp(t−i)
                                                              t              !1/p             X              p
                  =                                  lim                                                  β(t−i)2      · |gi|                               (10)                       p→∞                                          i=1

                                                        2                  = max βt−12   |g1|, βt−2                                                                   |g2|, . . . , β2|gt−1|, |gt|             (11)
Which corresponds to the remarkably simple recursive formula:
                                  ut = max(β2 · ut−1, |gt|)                               (12)
with initial value u0 = 0. Note that, conveniently enough, we don’t need to correct for initialization
bias in this case. Also note that the magnitude of parameter updates has a simpler bound with
AdaMax than Adam, namely: |∆t| ≤α.

7.2  TEMPORAL AVERAGING

Since the last iterate is noisy due to stochastic approximation, better generalization performance is
often achieved by averaging. Previously in Moulines & Bach (2011), Polyak-Ruppert averaging
(Polyak & Juditsky, 1992; Ruppert, 1988) has been shown to improve the convergence of standard                  1 PnSGD, where ¯θt = t   k=1 θk. Alternatively, an exponential moving average over the parameters can
be used, giving higher weight to more recent parameter values. This can be trivially implemented
by adding one line to the inner loop of algorithms 1 and 2: ¯θt ←β2 · ¯θt−1 +(1−β2)θt, with ¯θ0 = 0.
Initalization bias can again be corrected by the estimator bθt = ¯θt/(1 −βt2).

8  CONCLUSION

We have introduced a simple and computationally efﬁcient algorithm for gradient-based optimiza-
tion of stochastic objective functions. Our method is aimed towards machine learning problems with

                                       9
<a id="page-10"></a>

### PDF 第 10 页

Published as a conference paper at ICLR 2015




large datasets and/or high-dimensional parameter spaces. The method combines the advantages of
two recently popular optimization methods: the ability of AdaGrad to deal with sparse gradients,
and the ability of RMSProp to deal with non-stationary objectives. The method is straightforward
to implement and requires little memory. The experiments conﬁrm the analysis on the rate of con-
vergence in convex problems. Overall, we found Adam to be robust and well-suited to a wide range
of non-convex optimization problems in the ﬁeld machine learning.

9  ACKNOWLEDGMENTS

This paper would probably not have existed without the support of Google Deepmind. We would
like to give special thanks to Ivo Danihelka, and Tom Schaul for coining the name Adam. Thanks to
Kai Fan from Duke University for spotting an error in the original AdaMax derivation. Experiments
in this work were partly carried out on the Dutch national e-infrastructure with the support of SURF
Foundation. Diederik Kingma is supported by the Google European Doctorate Fellowship in Deep
Learning.

REFERENCES
Amari, Shun-Ichi. Natural gradient works efﬁciently in learning. Neural computation, 10(2):251–276, 1998.

Deng, Li, Li, Jinyu, Huang, Jui-Ting, Yao, Kaisheng, Yu, Dong, Seide, Frank, Seltzer, Michael, Zweig, Geoff,
  He, Xiaodong, Williams, Jason, et al. Recent advances in deep learning for speech research at microsoft.
  ICASSP 2013, 2013.

Duchi, John, Hazan, Elad, and Singer, Yoram. Adaptive subgradient methods for online learning and stochastic
   optimization. The Journal of Machine Learning Research, 12:2121–2159, 2011.

Graves, Alex. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.

Graves, Alex, Mohamed, Abdel-rahman, and Hinton, Geoffrey. Speech recognition with deep recurrent neural
   networks. In Acoustics, Speech and Signal Processing (ICASSP), 2013 IEEE International Conference on,
   pp. 6645–6649. IEEE, 2013.

Hinton, G.E. and Salakhutdinov, R.R. Reducing the dimensionality of data with neural networks. Science, 313
  (5786):504–507, 2006.

Hinton, Geoffrey, Deng, Li, Yu, Dong, Dahl, George E, Mohamed, Abdel-rahman, Jaitly, Navdeep, Senior,
  Andrew, Vanhoucke, Vincent, Nguyen, Patrick, Sainath, Tara N, et al. Deep neural networks for acoustic
  modeling in speech recognition: The shared views of four research groups. Signal Processing Magazine,
  IEEE, 29(6):82–97, 2012a.

Hinton, Geoffrey E, Srivastava, Nitish, Krizhevsky, Alex, Sutskever, Ilya, and Salakhutdinov, Ruslan R. Im-
   proving neural networks by preventing co-adaptation of feature detectors. arXiv preprint arXiv:1207.0580,
  2012b.

Kingma, Diederik P and Welling, Max. Auto-Encoding Variational Bayes. In The 2nd International Conference
  on Learning Representations (ICLR), 2013.

Krizhevsky, Alex, Sutskever, Ilya, and Hinton, Geoffrey E. Imagenet classiﬁcation with deep convolutional
   neural networks. In Advances in neural information processing systems, pp. 1097–1105, 2012.

Maas, Andrew L, Daly, Raymond E, Pham, Peter T, Huang, Dan, Ng, Andrew Y, and Potts, Christopher.
  Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association
   for Computational Linguistics: Human Language Technologies-Volume 1, pp. 142–150. Association for
  Computational Linguistics, 2011.

Moulines, Eric and Bach, Francis R.  Non-asymptotic analysis of stochastic approximation algorithms for
  machine learning. In Advances in Neural Information Processing Systems, pp. 451–459, 2011.

Pascanu, Razvan and Bengio, Yoshua.   Revisiting natural gradient for deep networks.   arXiv preprint
   arXiv:1301.3584, 2013.

Polyak, Boris T and Juditsky, Anatoli B. Acceleration of stochastic approximation by averaging. SIAM Journal
  on Control and Optimization, 30(4):838–855, 1992.


                                       10
<a id="page-11"></a>

### PDF 第 11 页

Published as a conference paper at ICLR 2015




Roux, Nicolas L and Fitzgibbon, Andrew W. A fast natural newton method.  In Proceedings of the 27th
   International Conference on Machine Learning (ICML-10), pp. 623–630, 2010.

Ruppert, David.  Efﬁcient estimations from a slowly convergent robbins-monro process.  Technical report,
   Cornell University Operations Research and Industrial Engineering, 1988.

Schaul, Tom, Zhang, Sixin, and LeCun, Yann. No more pesky learning rates. arXiv preprint arXiv:1206.1106,
  2012.

Sohl-Dickstein, Jascha, Poole, Ben, and Ganguli, Surya.  Fast large-scale optimization by unifying stochas-
   tic gradient and quasi-newton methods. In Proceedings of the 31st International Conference on Machine
  Learning (ICML-14), pp. 604–612, 2014.

Sutskever, Ilya, Martens, James, Dahl, George, and Hinton, Geoffrey. On the importance of initialization and
  momentum in deep learning.  In Proceedings of the 30th International Conference on Machine Learning
  (ICML-13), pp. 1139–1147, 2013.

Tieleman, T. and Hinton, G. Lecture 6.5 - RMSProp, COURSERA: Neural Networks for Machine Learning.
   Technical report, 2012.

Wang, Sida and Manning, Christopher. Fast dropout training. In Proceedings of the 30th International Confer-
  ence on Machine Learning (ICML-13), pp. 118–126, 2013.

Zeiler, Matthew D. Adadelta: An adaptive learning rate method. arXiv preprint arXiv:1212.5701, 2012.

Zinkevich, Martin. Online convex programming and generalized inﬁnitesimal gradient ascent. 2003.





                                       11
<a id="page-12"></a>

### PDF 第 12 页

Published as a conference paper at ICLR 2015



10  APPENDIX

10.1  CONVERGENCE PROOF
Deﬁnition 10.1. A function f : Rd →R is convex if for all x, y ∈Rd, for all λ ∈[0, 1],
                      λf(x) + (1 −λ)f(y) ≥f(λx + (1 −λ)y)
Also, notice that a convex function can be lower bounded by a hyperplane at its tangent.
Lemma 10.2.  If a function f : Rd →R is convex, then for all x, y ∈Rd,
                            f(y) ≥f(x) + ∇f(x)T (y −x)
The above lemma can be used to upper bound the regret and our proof for the main theorem is
constructed by substituting the hyperplane with the Adam update rules.
The following two lemmas are used to support our main theorem. We also use some deﬁnitions sim-
plify our notation, where gt ≜∇ft(θt) and gt,i as the ith element. We deﬁne g1:t,i ∈Rt as a vector
that contains the ith dimension of the gradients over all iterations till t, g1:t,i = [g1,i, g2,i, · · · , gt,i]
Lemma 10.3. Let gt = ∇ft(θt) and g1:t be deﬁned as above and bounded, ∥gt∥2 ≤G, ∥gt∥∞≤
G∞. Then,           s
          XT    g2t,i
                                                t  ≤2G∞∥g1:T,i∥2
                                 t=1

Proof. We will prove the inequality using induction over T.
              q
The base case for T = 1, we have   g21,i ≤2G∞∥g1,i∥2.
For the inductive step,
            s     s   s
       XT    g2t,i   TX−1   g2t,i      g2T,i
                   =      +
                                      t               t      T
                        t=1          t=1       s
                                                       g2T,i
                          ≤2G∞∥g1:T −1,i∥2 +   T
                          s
                  q                                                            g2T,i
                                                        2                                                    T +                                                T                   = 2G∞   ∥g1:T,i∥2                                    −g2


                               g4T,i
               2                                                2                       T,i +From, ∥g1:T,i∥2          −g2                                         2 ≥∥g1:T,i∥2                              −g2T,i, we can take square root of both side and                               4∥g1:T,i∥2
have,
           q                               g2T,i
                                      2     T,i                              ∥g1:T,i∥2 −g2  ≤∥g1:T,i∥2 −                                                            2∥g1:T,i∥2
                                                         g2T,i
                                        ≤∥g1:T,i∥2 − 2p                                             TG2∞

                 q

                                                      2                                                                   T,i term,Rearrange the inequality and substitute the   ∥g1:T,i∥2                                   −g2
                    s
            q                                              g2T,i
                                        2                                      T +             G∞   ∥g1:T,i∥2                          −g2                                     T  ≤2G∞∥g1:T,i∥2




                                       12
<a id="page-13"></a>

### PDF 第 13 页

Published as a conference paper at ICLR 2015


Lemma 10.4. Let γ ≜ √β2β21  . For β1, β2                     β21                                ∈[0, 1) that satisfy √β2 < 1 and bounded gt, ∥gt∥2 ≤G,
∥gt∥∞≤G∞, the following inequality holds
        XT   bm2t,i      2     1
              p                      ≤ 1                                                            ∥g1:T,i∥2                               √1                           t=1     tbvt,i                         −γ                                  −β2
               √                              1−βtProof. Under the assumption,                                            2        1     . We can expand the last term in the summation                              (1−βt1)2 ≤ (1−β1)2
using the update rules in Algorithm 1,
         T           T      p   X  bm2t,i X−1  bm2t,i      1    2 (PTk=1(1        1 −k gk,i)2     p   =  p   +   −βT q      −β1)βT
                                              1 )2  T PTj=1(1        2 −j g2j,i        t=1     tbvt,i    t=1     tbvt,i   (1 −βT                                            −β2)βT
                    T      p        T
       X−1  bm2t,i      1    2 X   T((1        1 −k gk,i)2           p   +   −βT  q    −β1)βT           ≤                                              1 )2 k=1  T PTj=1(1        2 −j g2j,i                      t=1     tbvt,i   (1 −βT                                               −β2)βT
                    T      p        T
       X−1  bm2t,i      1    2 X T((1        1 −k gk,i)2           p   +   −βT  q  −β1)βT           ≤                      t=1     tbvt,i   (1 −βT                                              1 )2 k=1   T(1 −β2)βT2 −k g2k,i
                p
                    TX−1  bm2t,i                  XT     β21   T −k                                 1    2   (1           p   +   −βT p −β1)2    T           ≤                                                √β2      ∥gk,i∥2                                                     k=1                      t=1     tbvt,i   (1 −βT                                              1 )2  T(1 −β2)
                    TX−1  bm2t,i       T  XT
           p                   + p           ≤                                            γT −k∥gk,i∥2                                             tbvt,i                      t=1                               T(1      k=1                               −β2)

Similarly, we can upper bound the rest of the terms in the summation.

        XT   bm2t,i XT             TX−t              p     p ∥gt,i∥2       tγj                      ≤                           t=1     tbvt,i                                         t=1   t(1 −β2) j=0
            XT    XT
                    p ∥gt,i∥2       tγj                      ≤                                         t=1   t(1 −β2) j=0

                          P
For γ < 1, using the upper bound on the arithmetic-geometric series,    t tγt <    1    :                                                                               (1−γ)2
      XT    XT        XT                                              1          p ∥gt,i∥2       tγj                           ∥gt,i∥2                                   √ t                                                              t=1                                 j=0                        ≤ (1 −γ)2√1 −β2                   t=1   t(1 −β2)
Apply Lemma 10.3,

        XT   bm2t,i             p        2G∞
                          t=1     tbvt,i                     ≤ (1 −γ)2√1 −β2 ∥g1:T,i∥2

To simplify the notation, we deﬁne γ ≜ √β2β21  .  Intuitively, our following theorem holds when the
learning rate αt is decaying at a rate of t−1                                                 2 and ﬁrst moment running average coefﬁcient β1,t decay
exponentially with λ, that is typically close to 1, e.g. 1 −10−8.
Theorem 10.5. Assume that the function ft has bounded gradients, ∥∇ft(θ)∥2 ≤G, ∥∇ft(θ)∥∞≤
G∞for all θ ∈Rd and distance between any θt generated by Adam is bounded, ∥θn −θm∥2 ≤D,

                                       13
<a id="page-14"></a>

### PDF 第 14 页

Published as a conference paper at ICLR 2015




                                                                          β21              α
                                                                                                                                  t∥θm −θn∥∞≤D∞for any m, n ∈{1, ..., T}, and β1, β2 ∈[0, 1) satisfy √β2 < 1. Let αt = √
and β1,t = β1λt−1, λ ∈(0, 1). Adam achieves the following guarantee, for all T ≥1.
         D2 Xd p           α(β1 +    Xd    Xd  D2   √1R(T)                    TbvT,i+         1)G∞               ∞G∞  −β2   ≤ 2α(1                    (1                          ∥g1:T,i∥2+   2α(1                                                           i=1           i=1     −β1)(1 −λ)2          −β1) i=1         −β1)√1 −β2(1 −γ)2


Proof. Using Lemma 10.2, we have,

                Xd
                       ft(θt) −ft(θ∗) ≤gT                                                         t (θt −θ∗) =      gt,i(θt,i −θ∗,i)
                                                      i=1

From the update rules presented in algorithm 1,

                  p
                      θt+1 = θt −αt bmt/    bvt
                                  αt     β1,t         (1               = θt            +   −β1,t) gt                  − 1    1   √bvt mt−1      √bvt                         −βt
                                                                                                                                     ,i and squareWe focus on the ith dimension of the parameter vector θt ∈Rd. Subtract the scalar θ∗both sides of the above update rule, we have,

                            2αt     β1,t                                                      bmt,i
                         p
                                                                                                    bvt,i                                                                                 gt,i)(θt,i −θ∗,i) + α2t( p bvt,i )2(θt+1,i −θ∗,i)2 =(θt,i −θ∗,i)2 − 1    1 ( pbvt,i mt−1,i + (1 −β1,t)                      −βt

We can rearrange the above equation and use Young’s inequality, ab    + b2/2. Also, it can be     p   qPt        p            ≤a2/2
                                                        2shown that     bvt,i =     j=1(1 −β2)βt−j                                        2  g2j,i/  1 −βt                                                    ≤∥g1:t,i∥2 and β1,t ≤β1. Then
          p
gt,i(θt,i     ,i) =(1 −βt1)    bvt,i   (θt,i     ,t)2                 ,i)2     −θ∗    2αt(1          −θ∗   −(θt+1,i −θ∗                    −β1,t)  1                  p
                                       4
                         β1,t      bv                                    t−1,i         +                                    (θ∗,i             mt−1,i1  + αt(1 −βt1)    bvt,i ( pbmt,i )2                                  −θt,i)√αt−1                                                                                 bv 4                  (1 −β1,t) √αt−1                                                             2(1 −β1,t)           bvt,i                                                               t−1,i
                  1                p              β1,t
                                                                                                                                             ,i                                                                        −θt,i)2p bvt−1,i        ≤ 2αt(1          (θt,i −θ∗,t)2 −(θt+1,i −θ∗,i)2       bvt,i +                −β1)                                   2αt−1(1 −β1,t)(θ∗
                                                                                   t,i         +  β1αt−1 pm2t−1,i +    αt  pbm2
                  2(1 −β1)    bvt−1,i   2(1 −β1)     bvt,i
We apply Lemma 10.4 to the above inequality and derive the regret bound by summing across all
the dimensions for i ∈1, ..., d in the upper bound of ft(θt) −ft(θ∗) and the sequence of convex
functions for t ∈1, ..., T:
          d                                    d  T          p   p  X          X X
R(T)               1            ,i)2pbv1,i +           1               ,i)2(    bvt,i        bvt−1,i )   ≤    2α1(1        −θ∗                 2(1       −θ∗     αt −        i=1      −β1)(θ1,i                i=1 t=2    −β1)(θt,i                 αt−1
           Xd           Xd
     +      β1αG∞            +      αG∞           (1                             ∥g1:T,i∥2   (1                             ∥g1:T,i∥2
                                    i=1                                      i=1         −β1)√1 −β2(1 −γ)2             −β1)√1 −β2(1 −γ)2
   Xd XT                           β1,t
     +                                       ,i                 bvt,i                 2αt(1         −θt,i)2p           i=1 t=1      −β1,t)(θ∗

                                       14
<a id="page-15"></a>

### PDF 第 15 页

Published as a conference paper at ICLR 2015


From the assumption, ∥θt −θ∗∥2 ≤D, ∥θm −θn∥∞≤D∞, we have:
        D2 Xd p            α(1 +    Xd         D2 Xd Xt      β1,t pR(T)                       TbvT,i +         β1)G∞           + ∞                                tbvt,i   ≤ 2α(1                     (1                             ∥g1:T,i∥2   2α         (1                                                           i=1                 i=1 t=1   −β1,t)         −β1) i=1          −β1)√1 −β2(1 −γ)2
        D2 Xd p            α(1 +    Xd                             TbvT,i +         β1)G∞   ≤ 2α(1                     (1                             ∥g1:T,i∥2
                                                           i=1         −β1) i=1          −β1)√1 −β2(1 −γ)2
        Xd Xt       D2   √1                                            β1,t     + ∞G∞  −β2         √ t
             2α               (1                            i=1 t=1   −β1,t)

We can use arithmetic geometric series upper bound for the last term:

        Xt     Xt
                                      β1,t  √ t         1            t                              (1      ≤     (1                          t=1   −β1,t)      t=1  −β1)λt−1√
              Xt                                              1
                        ≤     (1                                             t=1   −β1)λt−1t
                                              1
                        ≤ (1                                      −β1)(1 −λ)2
Therefore, we have the following regret bound:
        D2 Xd p            α(1 +    Xd    Xd D2   √1R(T)                       TbvT,i +         β1)G∞           +   ∞G∞  −β2   ≤ 2α(1                     (1                             ∥g1:T,i∥2       2αβ1(1                                                           i=1            i=1       −λ)2         −β1) i=1          −β1)√1 −β2(1 −γ)2





                                       15

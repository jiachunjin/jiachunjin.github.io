---
title: Representation Learning with Latent Variable Models
subtitle: In this post, I discuss what makes the posterior distribution of a directed latent variable model a useful representation. Also, I raise questions that deserve careful study in the case of undirected latent variable models.
layout: default
date: 2023-04-27
keywords: representation learning, latent-variable model
published: true
toc: true
---
As Ferenc Huszár discussed in his [blog post](https://www.inference.vc/maximum-likelihood-for-representation-learning-2/), training a latent variable model (LVM) via maximum likelihood estimation (MLE) does not necessarily lead to a "useful" representation. Actually, in the infinite model family limit, learning a "useful" representation of your data and achieving a high likelihood value (or generating high quality samples) are two **orthogonal** directions. For VAEs, what makes their latent representation useful is their auto-encoder based model structure as well as the ELBO maximization objective {%cite tschannen2018recent %}. However, what makes a useful representation in the case of undirected LVMs is still mysterious to me.

### Uninformative latent variables in LVMs

In the literature, it is reported that in the context of VAEs, a power decoder $p_\theta(x\mid z)$ like PixelCNN can generate high-quality samples, however the mutual information $\mathbb{I}(x;z)$ between the generated sample $x$ and the conditioned latent variable $z$ is very low, this is fatal to representation learning since the representation now contains almost no information of the original data. This is intuitive if you consider a LVM $p_\theta(x, z) = p_\theta(x\mid z)p_\theta(z)$, and when your decoder $p_\theta(x\mid z)$ remains to be a powerful density estimator (e.g. an EBM) on $p_{\text{data}}(x)$ after setting the weights connecting to the latent variable $z$ to 0, your LVM implicitly becomes to $p_\theta(x, z) = p_\theta(x)p_\theta(z)$ which leads to independence and zero mutual information.

I also come up with another way to approximately explain the above effect. Let's take the latent variable model as a mixture distribution and we would like to use its marginal $p_\theta(x) = \int p_\theta(x\mid z)p_\theta(z) \mathrm{d}z$ to model the $p_{\text{data}}(x)$. Consider the learning process as minimizing the forward KL-divergence (although MLE is minimizing the reverse KL-divergence, conceptually we can think they have similar effect on the model):
$$
\begin{aligned}
\operatorname{KL}\left(p_\theta(x) \parallel p_{\text{data}}(x)\right)
&= \int p_\theta(z) \int p_\theta(x\mid z) \log \frac{p_\theta(x)}{p_\text{data}(x)}\mathrm{d}x\mathrm{d}z \\
&= \int p_\theta(z) \int p_\theta(x\mid z) \log \frac{p_\theta(x\mid z)p_\theta(x)}{p_\text{data}(x)p_\theta(x\mid z)}\mathrm{d}x\mathrm{d}z \\
&= \int p_\theta(z) \int p_\theta(x\mid z) \log \frac{p_\theta(x\mid z)}{p_\text{data}(x)}\mathrm{d}x\mathrm{d}z - \int p_\theta(z) \int p_\theta(x\mid z) \log \frac{p_\theta(x\mid z)}{p_\theta(x)}\mathrm{d}x\mathrm{d}z \\
&= \mathbb{E}_{p_\theta(z)}\left[ \operatorname{KL}\left(p_\theta(x\mid z) \parallel p_{\text{data}}(x)\right) \right] - \mathbb{I}\left( x;z \right) \\
&\geq 0,
\end{aligned}
$$
this gives us an upper bound of the mutual information between $x$ and $z$:
$$
\mathbb{I}(x;z) \leq \mathbb{E}_{p_\theta(z)}\left[ \operatorname{KL}\left(p_\theta(x\mid z) \parallel p_{\text{data}}(x)\right) \right],
$$
thus when your decoder $p_\theta(x\mid z)$ itself can model $p_\text{data}(x)$ well in the sense of averaged forward KL divergence, then little information about $x$ is contained in $z$.

### Information theoretical effects of maximizing ELBO

When training a VAE, since the marginal distribution is intractable for direct MLE, we turn to maximizing the evidence lower bound (ELBO):
$$
\begin{aligned}
\mathbb{E}_{p_{\text{data}}(x)}\left[ \mathbb{E}_{q_\phi(z\mid x)}\left[ \log \frac{p_\theta(x,z)}{q_\phi(z\mid x)} \right] \right] &\approx
\underbrace{\frac{1}{N}\sum_{n=1}^N \mathbb{E}_{q_\phi(z_n \mid x_n)}\left[ \log {p_\theta(x_n\mid z_n)} \right]}_{\text{A}} - \underbrace{\frac{1}{N}\sum_{n=1}^N\operatorname{KL}\left( q_\phi(z_n \mid x_n) \parallel p_\theta(z_n)\right)}_{\text{B}},
\end{aligned}
$$
the term A is the reconstruction term, which can increase the mutual information between $x$ and $z$, and term B serves as a regularizer to enforce disentangled posterior.

Let's further examine term B. First, define the aggregated posterior as: $q_\phi(z) = \int q_\phi(z\mid x)p_{\text{data}}(x)\mathrm{d}x$, then we have:
$$
\begin{aligned}
\operatorname{KL}\left( q_\phi(z) \parallel p_\theta(z) \right) 
&= \int\left(\int q_\phi(z\mid x) p_{\text{data}}(x) \mathrm{d}x \right)\log \frac{q_\phi(z)}{p_\theta(z)}\mathrm{d}z \\
&= \iint q_\phi(z\mid x)p_{\text{data}}(x)\log \frac{q_\phi(z)q_\phi(z\mid x)}{p_\theta(z)q_\phi(z\mid x)}\mathrm{d}x\mathrm{d}z \\
&= \iint q_\phi(z\mid x)p_{\text{data}}(x)\log\frac{q_\phi(z\mid x)}{p_\theta(z)}\mathrm{d}x\mathrm{d}z - \iint  q_\phi(z\mid x)p_{\text{data}}(x)\log \frac{q_\phi(z\mid x)p_{\text{data}}(x)}{q_\phi(z)p_{\text{data}}(x)}\mathrm{d}x\mathrm{d}z \\
&= \underbrace{\mathbb{E}_{p_\text{data}(x)}\left[ \operatorname{KL}\left( q_\phi(z\mid x)\parallel p_\theta(z) \right) \right]}_{\text{B}} - \mathbb{I}(x;z).
\end{aligned}
$$
In this way, the regularizer over the posterior distribution becomes to:
$$
\text{B} =\operatorname{KL}\left( q_\phi(z) \parallel p_\theta(z) \right) + \mathbb{I}(x;z),
$$
and it will penalize the model for high mutual information between $x$ and $z$. Note that without this, the model will try its best to encode everything about $x$ into $z$ to achieve low reconstruction error, this will not lead to a useful representation.

In {% cite alemi2018fixing%}, the authors analyze the ELBO with the rate-distortion theory, where *distortion (D)* and *rate (R)* are defined as:
$$
\begin{aligned}
D &= -\mathbb{E}_{p_{\text{data}}(x)}\left[ \mathbb{E}_{q_\phi(z\mid x)}\left[ \log p_\theta(x\mid z) \right] \right] = -\text{A}, \\
R &= \mathbb{E}_{p_{\text{data}}(x)}\left[ q_\phi(z\mid x)\log \frac{q_\phi(z\mid x)}{p_\theta(z)} \right] = \text{B}, \\
\operatorname{ELBO} &= -(D + R) = \text{A} - \text{B}.
\end{aligned}
$$
Note that when we train a VAE by maximizing the ELBO, we are maximizing the sum of $D$ and $R$. And as shown in the paper, when your decoder $p_\theta(x\mid z)$ is pretty powerful, the $R$ term (which is equal to $\text{B}$) can be pushed to 0, this leads $\mathbb{I}(x;z) = 0$ because $\mathbb{I}(x;z) = \text{B} - \operatorname{KL}\left( q_\phi(z) \parallel p_\theta(z) \right)$ and both KL divergence and mutual information are non-negative terms. Adding a weighting parameter $\beta$ between $D$ and $R$ leads to a tradeoff:
$$
\begin{aligned}
-(D + \beta R) = {\frac{1}{N}\sum_{n=1}^N \mathbb{E}_{q_\phi(z_n \mid x_n)}\left[ \log {p_\theta(x_n\mid z_n)} \right]} - {\frac{1}{N}\sum_{n=1}^N\beta\operatorname{KL}\left( q_\phi(z_n \mid x_n) \parallel p_\theta(z_n)\right)},
\end{aligned}
$$
and this reveals the $\beta$-VAE objective {% cite higgins2017beta %}.

### Open questions: What about undirected LVMs?

The original hope of generative representation learning is that if we can create all the data that we have seen, then we implicitly may learn a representation that can be used to answer any question about the data. And in the sense of generative modeling, both directed and undirected latent variable models can perform well.

However, the undirected LVMs (i.e. energy-based latent variable models) are not trained by maximizing the ELBO, thus the above theory is not capable of analyzing its representational property. And it is interesting to study on the following questions:

1. What makes the learned representation in EBLVMs useful? The model or the learning algorithm? In the above discussion on the directed LVMs, we can say it's the auto-encoder based structure accompany with the ELBO  maximization objective makes the representation useful. Does a bipartite graphical (like RBMs) structure play a similar role as the auto-encoder struture in VAEs?
2. In {% cite wu2021conjugate%} {%cite liao2022gaussian %} {%cite lee2023guiding %}, the latent variables of the EBLVMs can be marginalized and the marginal energy functions are analytically available. This enables us to train these models as common EBMs with no latent variables. When the training procedure has no explicit connection with $z$, the representational usefulness of $p_\theta(z\mid x)$ highly depends on the extra constrain provided by the model structure. It is reported in {% cite wu2021conjugate%} that posterior collapse may happen in their conjugate energy-based model and the mutual information between $x$ and the encoded $z$ is low, while in {%cite liao2022gaussian %}, when the GRBM is trained with the marginal energy, it tends to map $x$ to an almost deterministic latent code $z$ which preserves high mutual information. So different design of joint energy function leads to different level of coupling between $x$ and $z$, then which part of an EBLVM is essential for that kind of coupling? Does the problem have any connection with latent variable identifiability {% cite wang2021posterior %}?
3. High mutual information between $x$ and $z$ does not necessarily lead to a good representation, can we establish similar rate-distortion theory in the context of undirected LVMs as in {% cite alemi2018fixing%} {%cite tschannen2018recent %}?

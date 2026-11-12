# Bridge_UQ

# BridgeUQ: Bayesian Aleatoric Uncertainty Quantification for Time-Series Image Registration via Brownian Bridge Prior

Official PyTorch implementation of **BridgeUQ**, a plug-and-play Bayesian module that quantifies **aleatoric uncertainty** of *any pretrained, deterministic* registration network over **spatiotemporal deformation trajectories** — without retraining the backbone.

<p align="center">
  <img src="assets/bridgeuq_overview.png" width="100%" alt="BridgeUQ overview">
</p>

**Fig. 1.** *Top:* BridgeUQ integrated with a frozen pretrained registration network — the backbone predicts a velocity sequence $\hat v$, and BridgeUQ turns it into a per-voxel scalar uncertainty map for every frame. *Bottom:* BridgeUQ in detail — a network $f_\theta$ predicts a spatially varying diffusion variance $\sigma$ that controls the perturbations of a GP-Brownian bridge around the registration trajectory, giving the posterior $P_\omega(v \mid \sigma, \hat v, I)$. Once trained, BridgeUQ is used as a plug-and-play module.

---

## 📌 Abstract

BridgeUQ is a Bayesian framework for aleatoric uncertainty quantification in pretrained deterministic registration models. Unlike prior approaches that focus on pairwise registration, BridgeUQ is the first to model posterior uncertainty over **spatiotemporal deformation trajectories**, capturing the intrinsic ambiguity of registration solutions across both space and time. We introduce a **hierarchical Gaussian-process prior** in the temporal deformation space, built on a **Brownian bridge** with an **inverse-Gamma hyperprior** on the diffusion variance, which governs stochastic sampling around the predicted registration solution. An **energy-based likelihood** derived from the pretrained registration objective keeps sampled deformations faithful to the observed data, and an **amortized MAP** scheme with conditional sampling makes posterior inference efficient. BridgeUQ is validated on cardiac cine MRI and longitudinal brain MRI with four registration backbones, where it produces anatomically meaningful uncertainty estimates and outperforms state-of-the-art UQ baselines without compromising registration accuracy.

## 🧠 Method in Brief

Given an image sequence $I^i = \{I^i_t\}_{t=0}^T$, a pretrained network predicts velocities $\hat v^i = \{\hat v^i_t\}$. BridgeUQ models the perturbed trajectory $v^i_t = \hat v^i_t + \eta^i_t$ with a hierarchical GP prior:

$$
\sigma^i \sim \mathrm{InverseGamma}(\alpha, \beta), \qquad
\eta^i_t \mid \sigma^i \sim \mathcal{GP}\big(0,\ \sigma^i\,\Sigma^i_{\mathrm{BB}}(t)\big), \qquad \eta^i_0 = \eta^i_T = 0,
$$

induced by the Brownian bridge SDE

$$
d\eta^i_t = \frac{\eta^i_T - \eta^i_t}{T - t}\,dt + \sqrt{\sigma^i}\,dW_t .
$$

Its Markov transitions are $v^i_t \mid v^i_{t-1} \sim \mathcal N\big(\mu^i_{t\mid t-1},\ \delta^i_{t\mid t-1}\big)$ with

$$
\mu^i_{t\mid t-1} = \hat v^i_t + \rho_t\big(v^i_{t-1} - \hat v^i_{t-1}\big), \qquad
\delta^i_{t\mid t-1} = \sigma^i \rho_t, \qquad
\rho_t = \frac{T-t}{T-(t-1)} .
$$

An energy-based likelihood ties the samples to the backbone's registration objective $E(\cdot)$:

$$
p(I^i \mid v^i) \propto \exp\!\Big(-\tfrac{1}{\gamma} E(v^i)\Big),
\qquad
p(v, \sigma \mid I, \hat v) \propto \prod_i p(I^i \mid v^i)\, p(v^i \mid \sigma^i; \hat v^i)\, p(\sigma^i).
$$

---

## ⚙️ Installation

```bash
conda create -n bridgeuq python=3.10 -y
conda activate bridgeuq
pip install torch torchvision          # pick the build matching your CUDA
pip install numpy matplotlib tqdm
```


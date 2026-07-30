# Reconstructing Parallel Decoding Distillation on CIFAR-10

Diffusion image generators normally refine noise through many sequential network calls, which makes high-quality sampling slow. Parallel Decoding Distillation (PDD) asks one call to predict several future updates at once, and this reproduction tests whether that shortcut improves a compact public image generator at very low evaluation counts. We reconstructed the training and fused inference path from the paper rather than using author code.

**Verdict — not reproduced in this bounded setup.** On a public CIFAR-10 teacher, the reconstructed PDD models were worse than a matched naive large-step solver on paired trajectory error and bounded feature fidelity. Midpoint targets were also worse than Euler targets, and changing the inference block size did not recover a useful quality–NFE trade-off.

**Scope.** This is an algorithmic small-image substitution: `google/ddpm-cifar10-32`, 1,024 fixed noise seeds, public UofT CIFAR-10 images, and a 128-NFE midpoint teacher reference. It does not test the paper's SiT-XL ImageNet model or 14B–22B media systems.

![KID across sampling budgets](images/headline_kid.png)

Lower curves are better. The naive solver improves steadily as it receives more network evaluations; the short-budget PDD curves remain high and nearly flat. KID is noisy with 1,024 images, so the paired trajectory and diversity results below are essential corroboration.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/parallel-decoding-distillation-for-fast-image-an/blob/main/notebooks/pdd_cifar10_reproduction.py)

## From the paper to executable code

The teacher is a pretrained epsilon-predicting U-Net, interpreted as a continuous variance-preserving probability flow. We copied the full network and repeated its final convolution into 16 output heads, one per fixed time interval. Each update sampled a noisy CIFAR-10 image and a starting grid point, rolled forward with detached student outputs, then regressed one selected head onto a stop-gradient teacher velocity.

The target was either one Euler evaluation or a midpoint update using two evaluations. To match teacher-call budgets, Euler accumulated two independently sampled targets per optimizer update. At inference, the weights and biases of every head inside a block were summed into one final convolution; blocks of 16, 8, 4, or 2 intervals yield 1, 2, 4, or 8 student evaluations. The fused layer agreed with the explicit head sum to within 0.0049 maximum absolute error in bfloat16.

## Claim-by-claim evidence

### 1. PDD versus naive large steps

At 8,000 updates across eight seeds, 1-NFE Euler-target PDD had KID × 1,000 of **421.86 ± 0.13**, versus **409.06** for naive Euler. Its paired trajectory MSE was **0.6664 ± 0.0006**, versus **0.6265**. At 8 NFE the gap widened: PDD MSE was **0.6611**, while the naive solver improved to **0.0474**.

![Paired trajectory error](images/trajectory_error.png)

Because every method used identical initial noise, trajectory MSE directly tests whether the fast sampler reached the fine teacher's endpoint. The result is not a marginal KID fluctuation: the reconstructed student did not preserve the teacher path.

### 2. Midpoint versus Euler targets

The paper reports 1-NFE ImageNet FID of 2.69 for midpoint targets versus 2.73 for Euler. Here midpoint instead raised 1-NFE KID from **421.86** to **422.76** and trajectory MSE from **0.6664** to **0.7101** at matched teacher-call budgets. The direction was stable across eight seeds.

![Target method direction](images/target_direction.png)

The figure compares relative direction, not metric scale: below zero favors midpoint. Our compact reconstruction therefore did not show the paper's midpoint advantage.

### 3. One model across multiple block sizes

![Feature diversity](images/diversity.png)

The fused decoder executed correctly at 1, 2, 4, and 8 NFE, so the mechanical multi-block claim is aligned. The quality claim is not: PDD diversity stayed near 0.056–0.060, far below the 128-NFE reference's 0.361, and did not improve with more evaluations. The naive solver rose from 0.050 to 0.290.

![Measured quality–latency frontier](images/latency_frontier.png)

Latency scaled almost linearly from roughly 13 ms at 1 NFE to 97 ms at 8 NFE for a batch of 64. PDD matched the naive sampler's latency at each NFE; its failure was fidelity, not hidden inference overhead.

## What the negative result means

The test is strong for this exact reconstruction: paired seeds, public real images, eight short-run seeds, matched target budgets, independent fusion validation, and convergence/optimizer/architecture diagnostics. It does **not** show that the paper's large-scale result is wrong. No author code or checkpoint exists, and consequential missing details may include velocity parameterization, loss weighting, optimizer schedule, or capacity needed to learn many interval heads. The paper also trains for 300,000 updates at batch size 2,048; this compact study uses a much smaller model and budget.

| Claim | Paper evidence | Observed here | Assessment |
|---|---|---|---|
| PDD beats naive few-step sampling | Strong large-model FID/quality | Higher KID and trajectory error | Not aligned in this setup |
| Midpoint beats Euler | 2.69 vs 2.73 1-NFE ImageNet FID | 422.76 vs 421.86 KID × 1,000 | Not aligned in this setup |
| One model spans block sizes | Improving quality over 1/2/4/8 NFE | Execution works; quality remains flat | Mechanism aligned, utility not aligned |

All formal runs used OpenResearch **Kubernetes** on **NVIDIA RTX PRO 6000 Blackwell** GPUs, with a peak concurrent allocation of **16 GPUs**. The final measured wall time and selected long-run diagnostics are recorded in `autoresearch.json` and the [self-contained notebook](../../notebooks/pdd_cifar10_reproduction.py).

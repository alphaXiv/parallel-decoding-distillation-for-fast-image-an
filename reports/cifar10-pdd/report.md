# Reconstructing Parallel Decoding Distillation on CIFAR-10

Diffusion image generators normally refine noise through many sequential network calls, which makes high-quality sampling slow. Parallel Decoding Distillation (PDD) asks one call to predict several future updates at once, and this reproduction tests whether that shortcut improves a compact public image generator at very low evaluation counts. We reconstructed the training and fused inference path from the paper rather than using author code.

**Verdict — partially reproduced.** With enough training, the midpoint PDD reconstruction beat a matched naive large-step solver, midpoint targets beat Euler targets, and one model improved across 1, 2, 4, and 8 evaluations. The verdict is partial because this is a CIFAR-10 algorithmic substitution, not the paper's large ImageNet or video setting.

**Scope.** We used `google/ddpm-cifar10-32`, 1,024 fixed noise seeds, public UofT CIFAR-10 images, and a 128-NFE midpoint teacher reference. The selected compact model has eight interval heads and trained for 160,000 updates at learning rate `1e-4`; paper-length and EMA checks are supporting diagnostics.

![KID across sampling budgets](images/headline_kid.png)

Lower curves are better. The midpoint PDD model is already better than naive Euler at one evaluation and improves steadily as it receives more evaluations. Euler-target PDD needs four evaluations before it beats the naive control, making the target construction decisive.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/parallel-decoding-distillation-for-fast-image-an/blob/main/notebooks/pdd_cifar10_reproduction.py)

## From equations to executable code

The teacher is a pretrained epsilon-predicting U-Net, interpreted as a continuous variance-preserving probability flow. We copied its network and repeated the final convolution into one output head per fixed time interval. Each update sampled a noisy CIFAR-10 image and starting grid point, rolled forward with detached student outputs, then regressed one selected head onto a stop-gradient teacher velocity.

The target was either one Euler evaluation or a midpoint update using two evaluations. Euler accumulated two independently sampled targets per update, matching midpoint's teacher-call budget. At inference, the weights and biases of heads inside a block were summed into one final convolution. The fused layer agreed with the explicit head sum within 0.00049 maximum absolute error in bfloat16.

## Claim-by-claim evidence

### 1. PDD improves few-evaluation fidelity

Across two seeds at 1 NFE, midpoint PDD reduced mean KID × 1,000 from **409.06** for naive Euler to **280.34**. Paired trajectory MSE to the fine teacher dropped from **0.6265** to **0.0875**. KID is noisy with 1,024 images, but the large paired-error change makes the direction unambiguous.

![Paired trajectory error](images/trajectory_error.png)

At 8 NFE, PDD retained better mean KID (**46.89** versus **176.71**) while naive Euler had lower endpoint MSE (**0.0474** versus **0.0597**). This difference is informative: PDD's samples match real-image features better, but the coarse numerical solver more closely follows the chosen fine teacher trajectory at the largest tested budget.

### 2. Midpoint targets outperform Euler

The paper reports 1-NFE ImageNet FID of 2.69 for midpoint targets versus 2.73 for Euler. In the matched grid-8 reconstruction, midpoint reduced mean 1-NFE KID from **472.04** to **280.34** and trajectory MSE from **0.1162** to **0.0875**. Midpoint was better at every tested NFE.

![Target method direction](images/target_direction.png)

The figure compares relative direction, not metric scale: below zero favors midpoint. Both the paper and this reconstruction point in the same direction, with a much larger effect in the compact model.

### 3. One model spans several block sizes

![Feature diversity](images/diversity.png)

Each midpoint checkpoint used block sizes 8, 4, 2, and 1 to produce 1, 2, 4, and 8 NFE samples. Mean KID improved monotonically from **280.34** to **46.89**, while feature diversity rose from **0.113** to **0.336** toward the fine reference's **0.361**. No retraining or runtime head selection was needed.

![Measured quality–latency frontier](images/latency_frontier.png)

Latency scaled from about 13 ms at 1 NFE to 97 ms at 8 NFE for a batch of 64. PDD and naive sampling cost essentially the same per evaluation; the gain is a better quality frontier, not hidden extra computation.

## Robustness and diagnostics

Training duration was the main boundary condition. Eight independent 8,000-update runs per target were uniformly poor; 80,000-update grid-16 seeds agreed closely, and the selected 160,000-update grid-8 pair improved further. A 300,000-update grid-8 run gave a nearly identical frontier. EMA underfit at 80,000 updates and caught up—but stayed slightly worse—at 300,000. Reducing the learning rate to `1e-5` or `1e-6` also underfit. Freezing the U-Net trunk changed no measured value, showing that the repeated output heads carried the gain here.

| Claim | Paper evidence | Observed here | Assessment |
|---|---|---|---|
| PDD beats naive few-step sampling | Strong large-model FID/quality | Mean 1-NFE KID 280.34 vs 409.06; MSE 0.0875 vs 0.6265 | Aligned |
| Midpoint beats Euler | 2.69 vs 2.73 ImageNet FID | Mean 280.34 vs 472.04 KID × 1,000 | Aligned |
| One model spans block sizes | Improving quality over 1/2/4/8 NFE | Mean KID 280.34 → 46.89; diversity 0.113 → 0.336 | Aligned |

No author code or checkpoint is available, and this compact teacher cannot validate 14B–22B video/audio quality. A full reproduction still needs the paper's SiT-XL/ImageNet setup, training scale, and media benchmarks. All formal runs used OpenResearch **Kubernetes** on **NVIDIA RTX PRO 6000 Blackwell** GPUs, peaking at **16 concurrent GPUs**; measured elapsed wall time is recorded in `autoresearch.json`.

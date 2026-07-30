# PDD reproduction: compact public teacher, negative bounded result

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/parallel-decoding-distillation-for-fast-image-an/blob/main/notebooks/pdd_cifar10_reproduction.py)

This repository reconstructs the central algorithmic claims of [Parallel Decoding Distillation for Fast Image and Video Generation (arXiv:2607.26004)](https://arxiv.org/abs/2607.26004): a stop-gradient on-policy parallel decoder should beat naive large steps, midpoint targets should beat Euler targets, and one trained decoder should expose a useful 1/2/4/8-NFE trade-off.

**Assessment: not reproduced in this bounded CIFAR-10 setup.** Across eight 8,000-update seeds, 1-NFE PDD produced KID × 1,000 of **421.86 ± 0.13** with Euler targets and **422.76 ± 0.19** with midpoint targets, versus **409.06** for naive Euler. The paper's non-comparable ImageNet number favors midpoint—FID **2.69** versus **2.73**—while our paired trajectory error also favored the naive control and Euler target. Longer convergence and architecture diagnostics are synthesized in the report.

The deliberate substitution is a public `google/ddpm-cifar10-32` teacher and public UofT CIFAR-10 images instead of the paper's SiT-XL ImageNet and 14B–22B media models. Formal evidence used 1,024 fixed seeds, a 128-NFE midpoint reference, trajectory MSE, paired Inception features, bounded KID, diversity, and measured latency. Runs used OpenResearch Kubernetes on NVIDIA RTX PRO 6000 Blackwell GPUs, peaking at 16 concurrent GPUs.

- [Illustrated report](reports/cifar10-pdd/report.md)
- [Self-contained marimo tutorial](notebooks/pdd_cifar10_reproduction.py)
- [Compact measured results](reports/cifar10-pdd/results.json)
- [Figure-generation code](reports/cifar10-pdd/make_figures.py)

## Experiment log

Every experiment used the immutable exact command `bash run.sh`; hyperparameters live in each linked branch.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Public report, notebook, and implementation | Not run as an experiment (publication surface) | Presentation only | — |
| [Naive coarse baseline](https://github.com/alphaXiv/parallel-decoding-distillation-for-fast-image-an/tree/orx/cifar-10-naive-coarse-baseline) | Public teacher, real CIFAR-10, 128-NFE reference, naive 1/2/4/8-NFE control | `bash run.sh` | Reference KID × 1,000 6.91; naive KID 409.06 → 176.71 | Kubernetes, 1× RTX PRO 6000 Blackwell |
| [Euler PDD, 8k seed 0](https://github.com/alphaXiv/parallel-decoding-distillation-for-fast-image-an/tree/orx/euler-pdd-seed-0) | Reconstructed on-policy PD loss with matched Euler targets | `bash run.sh` | Worse than naive; replicated over 8 seeds | Kubernetes, 1× RTX PRO 6000 Blackwell |
| [Midpoint PDD, 8k seed 0](https://github.com/alphaXiv/parallel-decoding-distillation-for-fast-image-an/tree/orx/midpoint-pdd-seed-0) | Same decoder with midpoint teacher targets | `bash run.sh` | Worse than Euler and naive; replicated over 8 seeds | Kubernetes, 1× RTX PRO 6000 Blackwell |
| [Euler PDD, 80k](https://github.com/alphaXiv/parallel-decoding-distillation-for-fast-image-an/tree/orx/euler-pdd-80000-steps-seed-0) | Tenfold convergence extension | `bash run.sh` | See report | Kubernetes, 1× RTX PRO 6000 Blackwell |
| [Midpoint PDD, 80k](https://github.com/alphaXiv/parallel-decoding-distillation-for-fast-image-an/tree/orx/midpoint-pdd-80000-steps-seed-0) | Matched tenfold convergence extension | `bash run.sh` | See report | Kubernetes, 1× RTX PRO 6000 Blackwell |
| [Grid-8 midpoint diagnostic](https://github.com/alphaXiv/parallel-decoding-distillation-for-fast-image-an/tree/orx/midpoint-pdd-grid-8-80000) | Halve the number of parallel interval heads | `bash run.sh` | See notebook / result file | Kubernetes, 1× RTX PRO 6000 Blackwell |
| [Frozen-trunk midpoint diagnostic](https://github.com/alphaXiv/parallel-decoding-distillation-for-fast-image-an/tree/orx/midpoint-pdd-80000-frozen-trunk) | Isolate output-head learning from trunk drift | `bash run.sh` | See notebook / result file | Kubernetes, 1× RTX PRO 6000 Blackwell |

## Implementation map

- `reproduce.py` contains the VP probability-flow conversion, parallel output heads, detached intra-block rollout, Euler/midpoint targets, fused block sampler, metrics, and latency benchmark.
- `config.json` is the baseline experiment configuration; each experiment branch changes code/config rather than the run command.
- `.orx/k8s.yaml` requests one GPU per independent seed or diagnostic. This avoided an NCCL failure specific to the cluster image while still filling all 16 GPUs with independent work.
- `run.sh` installs pinned public dependencies and prints machine-readable terminal evidence.

To inspect the evidence interactively without rerunning GPU training, open the Molab notebook above. For a local notebook:

```bash
marimo edit notebooks/pdd_cifar10_reproduction.py
```

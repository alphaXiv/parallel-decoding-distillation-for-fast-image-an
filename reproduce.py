#!/usr/bin/env python
"""Bounded CIFAR-10 reconstruction of Parallel Decoding Distillation.

The public google/ddpm-cifar10-32 epsilon model is interpreted as a continuous
VP probability-flow teacher.  The student reuses its full U-Net and repeats the
last convolution N times, one velocity head per fixed interval.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMPipeline, UNet2DModel
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.transforms import Compose, Normalize, ToTensor


MODEL_ID = "google/ddpm-cifar10-32"


def setup_dist() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # The public CIFAR archive can be slow from some clusters.  Establish the
    # local-rank device before NCCL and allow rank 0 to finish the one-time
    # download before the first collective.
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    return dist.get_rank(), dist.get_world_size(), device


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class VPSchedule:
    def __init__(self, alphas_cumprod: torch.Tensor, device: torch.device):
        self.log_ab = alphas_cumprod.to(device=device, dtype=torch.float32).log()
        self.T = len(self.log_ab)

    def values(self, generation_t: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return alpha, sigma, their d/d generation-time, and model timestep."""
        tau = (1.0 - generation_t.float()) * (self.T - 1)
        lo = tau.floor().long().clamp(0, self.T - 2)
        frac = tau - lo.float()
        left = self.log_ab[lo]
        right = self.log_ab[lo + 1]
        log_ab = left + frac * (right - left)
        dlog_dt = -(right - left) * (self.T - 1)
        ab = log_ab.exp()
        alpha = ab.sqrt()
        sigma = (1.0 - ab).clamp_min(1e-12).sqrt()
        dalpha = 0.5 * alpha * dlog_dt
        dsigma = -0.5 * ab / sigma * dlog_dt
        return alpha, sigma, dalpha, dsigma, tau


@torch.no_grad()
def teacher_velocity(
    teacher: nn.Module, schedule: VPSchedule, x: torch.Tensor, t: float
) -> torch.Tensor:
    tv = torch.full((x.shape[0],), t, device=x.device)
    alpha, sigma, dalpha, dsigma, tau = schedule.values(tv)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        eps = teacher(x, tau).sample.float()
    shape = (-1, 1, 1, 1)
    x0 = (x - sigma.view(shape) * eps) / alpha.view(shape).clamp_min(1e-4)
    return dalpha.view(shape) * x0 + dsigma.view(shape) * eps


@torch.no_grad()
def rk_target(
    teacher: nn.Module,
    schedule: VPSchedule,
    x: torch.Tensor,
    t: float,
    h: float,
    method: str,
) -> torch.Tensor:
    v0 = teacher_velocity(teacher, schedule, x, t)
    if method == "euler":
        return v0
    if method == "midpoint":
        return teacher_velocity(teacher, schedule, x + 0.5 * h * v0, t + 0.5 * h)
    raise ValueError(method)


def make_parallel_student(teacher: UNet2DModel, n_heads: int) -> UNet2DModel:
    student = copy.deepcopy(teacher)
    old = student.conv_out
    new = nn.Conv2d(
        old.in_channels,
        old.out_channels * n_heads,
        old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    ).to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(old.weight.repeat(n_heads, 1, 1, 1))
        if old.bias is not None:
            new.bias.copy_(old.bias.repeat(n_heads))
    student.conv_out = new
    return student


def student_heads(
    student: nn.Module, schedule: VPSchedule, x: torch.Tensor, t: float, n: int
) -> torch.Tensor:
    tv = torch.full((x.shape[0],), t, device=x.device)
    tau = schedule.values(tv)[-1]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = student(x, tau).sample
    b, nc, hh, ww = out.shape
    return out.reshape(b, n, nc // n, hh, ww).float()


class PublicCIFAR(torch.utils.data.Dataset):
    def __init__(self, split: str, transform: Compose):
        self.data = load_dataset("uoft-cs/cifar10", split=split)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.data[index]
        return self.transform(row["img"].convert("RGB")), int(row["label"])


def load_data(
    rank: int, world: int, batch: int
) -> tuple[DataLoader, PublicCIFAR]:
    transform = Compose([ToTensor(), Normalize((0.5,) * 3, (0.5,) * 3)])
    # Hugging Face hosts the canonical UofT CIFAR-10 data as fast public
    # parquet shards.  Rank 0 populates the pod-local cache once.
    if rank == 0:
        PublicCIFAR("train", transform)
        PublicCIFAR("test", transform)
    dist.barrier(device_ids=[torch.cuda.current_device()])
    train = PublicCIFAR("train", transform)
    test = PublicCIFAR("test", transform)
    sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(
        train,
        batch_size=batch,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    return loader, test


def train_student(
    student: DDP,
    teacher: nn.Module,
    schedule: VPSchedule,
    loader: DataLoader,
    cfg: dict,
    rank: int,
) -> list[dict]:
    target = cfg["target"]
    ngrid = cfg["grid_size"]
    h = 1.0 / ngrid
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=cfg["learning_rate"], weight_decay=0.0
    )
    iterator = iter(loader)
    history: list[dict] = []
    started = time.perf_counter()
    for step in range(1, cfg["train_steps"] + 1):
        try:
            clean, _ = next(iterator)
        except StopIteration:
            loader.sampler.set_epoch(step)
            iterator = iter(loader)
            clean, _ = next(iterator)
        clean = clean.cuda(non_blocking=True)
        n = cfg["min_block"] * random.randrange(ngrid // cfg["min_block"])
        t_n = n * h
        tv = torch.full((clean.shape[0],), t_n, device=clean.device)
        alpha, sigma, *_ = schedule.values(tv)
        noise = torch.randn_like(clean)
        x_n = alpha[:, None, None, None] * clean + sigma[:, None, None, None] * noise

        optimizer.zero_grad(set_to_none=True)
        u = student_heads(student, schedule, x_n, t_n, ngrid)
        # Midpoint costs two teacher calls. For matched teacher-call budgets,
        # Euler supervises two independently sampled intra-block intervals.
        repeats = 2 if target == "euler" else 1
        losses = []
        for _ in range(repeats):
            stop = min(n + cfg["max_block"], ngrid)
            k = random.randrange(n, stop)
            if k == n:
                x_k = x_n.detach()
            else:
                x_k = (x_n + h * u[:, n:k].detach().sum(dim=1)).detach()
            with torch.no_grad():
                target_u = rk_target(teacher, schedule, x_k, k * h, h, target)
            losses.append(F.mse_loss(u[:, k], target_u))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 250 == 0 or step == cfg["train_steps"]:
            reduced = loss.detach().clone()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            reduced /= dist.get_world_size()
            record = {
                "step": step,
                "loss": float(reduced),
                "elapsed_s": time.perf_counter() - started,
            }
            history.append(record)
            if rank == 0:
                print("TRAIN_JSON " + json.dumps(record), flush=True)
    return history


@torch.no_grad()
def teacher_sample(
    teacher: nn.Module,
    schedule: VPSchedule,
    noise: torch.Tensor,
    nfe: int,
    method: str,
) -> torch.Tensor:
    x = noise.clone()
    h = 1.0 / nfe
    for i in range(nfe):
        x = x + h * rk_target(teacher, schedule, x, i * h, h, method)
    return x.clamp(-1, 1)


def build_fused_convs(student: UNet2DModel, ngrid: int, block: int) -> dict[int, nn.Conv2d]:
    old = student.conv_out
    w = old.weight.reshape(ngrid, 3, *old.weight.shape[1:])
    b = old.bias.reshape(ngrid, 3) if old.bias is not None else None
    h = 1.0 / ngrid
    fused = {}
    for n in range(0, ngrid, block):
        layer = nn.Conv2d(
            old.in_channels,
            3,
            old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            groups=old.groups,
            bias=b is not None,
            padding_mode=old.padding_mode,
        ).to(device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            layer.weight.copy_(h * w[n : n + block].sum(dim=0))
            if b is not None:
                layer.bias.copy_(h * b[n : n + block].sum(dim=0))
        fused[n] = layer
    return fused


@torch.no_grad()
def pdd_sample(
    student: UNet2DModel,
    schedule: VPSchedule,
    noise: torch.Tensor,
    ngrid: int,
    nfe: int,
    fused: dict[int, nn.Conv2d] | None = None,
) -> torch.Tensor:
    block = ngrid // nfe
    if fused is None:
        fused = build_fused_convs(student, ngrid, block)
    original = student.conv_out
    x = noise.clone()
    try:
        for n in range(0, ngrid, block):
            student.conv_out = fused[n]
            tv = torch.full((x.shape[0],), n / ngrid, device=x.device)
            tau = schedule.values(tv)[-1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                displacement = student(x, tau).sample.float()
            x = x + displacement
    finally:
        student.conv_out = original
    return x.clamp(-1, 1)


@torch.no_grad()
def fusion_error(
    student: UNet2DModel, schedule: VPSchedule, x: torch.Tensor, ngrid: int, nfe: int
) -> float:
    block = ngrid // nfe
    n = 0
    full = student_heads(student, schedule, x, 0.0, ngrid)
    expected = full[:, n : n + block].sum(dim=1) / ngrid
    fused = build_fused_convs(student, ngrid, block)[n]
    original = student.conv_out
    student.conv_out = fused
    tv = torch.zeros(x.shape[0], device=x.device)
    tau = schedule.values(tv)[-1]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = student(x, tau).sample.float()
    student.conv_out = original
    return float((expected - actual).abs().max())


def inception_model(device: torch.device) -> nn.Module:
    model = inception_v3(
        weights=Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False,
    )
    model.fc = nn.Identity()
    model.eval().to(device)
    return model


@torch.no_grad()
def features(model: nn.Module, images: torch.Tensor, batch: int = 64) -> torch.Tensor:
    outputs = []
    for chunk in images.split(batch):
        inp = F.interpolate((chunk + 1) / 2, size=(299, 299), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs.append(model(inp).float())
    return torch.cat(outputs)


def gather_tensor(x: torch.Tensor) -> torch.Tensor:
    slots = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(slots, x)
    return torch.cat(slots).cpu()


def kid_unbiased(x: torch.Tensor, y: torch.Tensor, subsets: int = 20, size: int = 100) -> float:
    g = torch.Generator().manual_seed(12345)
    d = x.shape[1]
    vals = []
    for _ in range(subsets):
        a = x[torch.randperm(len(x), generator=g)[:size]]
        b = y[torch.randperm(len(y), generator=g)[:size]]
        kaa = (a @ a.T / d + 1).pow(3)
        kbb = (b @ b.T / d + 1).pow(3)
        kab = (a @ b.T / d + 1).pow(3)
        m = len(a)
        vals.append(
            ((kaa.sum() - kaa.diag().sum()) + (kbb.sum() - kbb.diag().sum()))
            / (m * (m - 1))
            - 2 * kab.mean()
        )
    return float(torch.stack(vals).mean())


def diversity(feat: torch.Tensor) -> float:
    g = torch.Generator().manual_seed(54321)
    i = torch.randint(len(feat), (4096,), generator=g)
    j = torch.randint(len(feat), (4096,), generator=g)
    a = F.normalize(feat[i], dim=1)
    b = F.normalize(feat[j], dim=1)
    return float((1 - (a * b).sum(dim=1)).mean())


def paired_feature_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((F.normalize(a, dim=1) - F.normalize(b, dim=1)).pow(2).sum(dim=1).mean())


@torch.no_grad()
def benchmark_ms(fn, noise: torch.Tensor, repeats: int = 3) -> float:
    fn(noise)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn(noise)
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    rank, world, device = setup_dist()
    seed_all(cfg["train_seed"] + rank)
    overall_start = time.perf_counter()
    if rank == 0:
        print(
            "CONFIG_JSON "
            + json.dumps(
                {
                    **cfg,
                    "backend": "kubernetes",
                    "gpu_model": torch.cuda.get_device_name(0),
                    "world_size": world,
                    "teacher": MODEL_ID,
                    "torch": torch.__version__,
                }
            ),
            flush=True,
        )

    loader, test = load_data(rank, world, cfg["per_gpu_batch"])
    pipe = DDPMPipeline.from_pretrained(MODEL_ID)
    teacher = pipe.unet.to(device).eval()
    teacher.requires_grad_(False)
    schedule = VPSchedule(pipe.scheduler.alphas_cumprod, device)
    del pipe

    student_raw = None
    history = []
    if cfg["train_steps"] > 0:
        student_raw = make_parallel_student(teacher, cfg["grid_size"]).to(device)
        student = DDP(student_raw, device_ids=[device.index], broadcast_buffers=False)
        history = train_student(student, teacher, schedule, loader, cfg, rank)
        student_raw = student.module.eval()
        dist.barrier(device_ids=[torch.cuda.current_device()])

    local_n = cfg["eval_samples"] // world
    g = torch.Generator(device=device).manual_seed(cfg["eval_seed"] + rank)
    noise = torch.randn((local_n, 3, 32, 32), generator=g, device=device)
    ref = teacher_sample(
        teacher, schedule, noise, cfg["reference_steps"], method="midpoint"
    )

    # Public evaluation images: fixed first 1,024 CIFAR-10 test examples.
    start = rank * local_n
    reals = torch.stack([test[i][0] for i in range(start, start + local_n)]).to(device)
    if rank == 0:
        feature_net = inception_model(device)
    dist.barrier(device_ids=[torch.cuda.current_device()])
    if rank != 0:
        feature_net = inception_model(device)
    real_feat = gather_tensor(features(feature_net, reals, cfg["eval_batch"]))
    ref_feat = gather_tensor(features(feature_net, ref, cfg["eval_batch"]))
    ref_all = gather_tensor(ref)

    result = {
        "variant": cfg["variant"],
        "target": cfg["target"],
        "train_seed": cfg["train_seed"],
        "train_steps": cfg["train_steps"],
        "eval_samples": cfg["eval_samples"],
        "reference": {
            "solver": "midpoint",
            "nfe": 2 * cfg["reference_steps"],
            "kid_x1000": 1000 * kid_unbiased(real_feat, ref_feat),
            "diversity": diversity(ref_feat),
        },
        "naive": {},
        "pdd": {},
        "training": history,
    }

    latency_noise = noise[: min(64, local_n)]
    for nfe in (1, 2, 4, 8):
        sample = teacher_sample(teacher, schedule, noise, nfe, method="euler")
        feat = gather_tensor(features(feature_net, sample, cfg["eval_batch"]))
        sample_all = gather_tensor(sample)
        result["naive"][str(nfe)] = {
            "trajectory_mse": float((sample_all - ref_all).pow(2).mean()),
            "paired_feature_distance": paired_feature_distance(feat, ref_feat),
            "kid_x1000": 1000 * kid_unbiased(real_feat, feat),
            "diversity": diversity(feat),
            "latency_ms_batch64_per_rank": benchmark_ms(
                lambda z, q=nfe: teacher_sample(teacher, schedule, z, q, "euler"),
                latency_noise,
            ),
        }

    if student_raw is not None:
        fused_sets = {
            nfe: build_fused_convs(
                student_raw, cfg["grid_size"], cfg["grid_size"] // nfe
            )
            for nfe in (1, 2, 4, 8)
        }
        result["fusion_max_abs_error"] = fusion_error(
            student_raw, schedule, noise[:2], cfg["grid_size"], 4
        )
        for nfe in (1, 2, 4, 8):
            sample = pdd_sample(
                student_raw,
                schedule,
                noise,
                cfg["grid_size"],
                nfe,
                fused_sets[nfe],
            )
            feat = gather_tensor(features(feature_net, sample, cfg["eval_batch"]))
            sample_all = gather_tensor(sample)
            result["pdd"][str(nfe)] = {
                "trajectory_mse": float((sample_all - ref_all).pow(2).mean()),
                "paired_feature_distance": paired_feature_distance(feat, ref_feat),
                "kid_x1000": 1000 * kid_unbiased(real_feat, feat),
                "diversity": diversity(feat),
                "latency_ms_batch64_per_rank": benchmark_ms(
                    lambda z, q=nfe: pdd_sample(
                        student_raw,
                        schedule,
                        z,
                        cfg["grid_size"],
                        q,
                        fused_sets[q],
                    ),
                    latency_noise,
                ),
            }

    result["elapsed_wall_s"] = time.perf_counter() - overall_start
    if rank == 0:
        print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
        print(
            "EVIDENCE_SUMMARY "
            f"variant={cfg['variant']} target={cfg['target']} "
            f"train_steps={cfg['train_steps']} eval_samples={cfg['eval_samples']} "
            f"elapsed_wall_s={result['elapsed_wall_s']:.1f}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

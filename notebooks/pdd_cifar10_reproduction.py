# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.23.0",
#   "matplotlib>=3.9",
#   "numpy>=2.0",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import urllib.request

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return json, mo, np, plt, urllib


@app.cell
def _(mo):
    mo.md(r"""
    # Parallel decoding distillation on a compact image teacher

    This executable walkthrough tests whether one neural-network call can
    predict several consecutive denoising updates at once. The paper reports
    that this *parallel decoder* preserves quality at very few model
    evaluations, benefits from midpoint teacher targets, and supports
    several speed–quality settings without retraining.

    **Reproduction verdict: partially reproduced.** With sufficient training,
    all three algorithmic claims align in this bounded CIFAR-10 reconstruction;
    the paper's billion-parameter image and video results remain outside scope.
    """)
    return


@app.cell
def _(json, urllib):
    DATA_URL = (
        "https://raw.githubusercontent.com/alphaXiv/"
        "parallel-decoding-distillation-for-fast-image-an/"
        "main/reports/cifar10-pdd/results.json"
    )
    with urllib.request.urlopen(DATA_URL) as response:
        data = json.load(response)
    return DATA_URL, data


@app.cell
def _(mo):
    metric = mo.ui.dropdown(
        options={
            "KID × 1,000": "kid_x1000",
            "Trajectory MSE": "trajectory_mse",
            "Feature diversity": "diversity",
            "Latency (ms)": "latency_ms",
        },
        value="KID × 1,000",
        label="Metric",
    )
    metric
    return (metric,)


@app.cell
def _(data, metric, np, plt):
    nfes = np.array([1, 2, 4, 8])
    primary = data.get("primary", data["short_8000"])
    groups = [
        ("Naive Euler", data["naive"], "#4b5563"),
        ("PDD—Euler target", primary["euler"], "#2563eb"),
        ("PDD—midpoint target", primary["midpoint"], "#dc2626"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, group, color in groups:
        values = [group[str(n)][metric.value] for n in nfes]
        ax.plot(nfes, values, "o-", lw=2.5, label=label, color=color)
    ax.set(xlabel="Sampling NFE", ylabel=metric.value.replace("_", " ").title())
    ax.set_xticks(nfes)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig
    return (primary,)


@app.cell
def _(mo):
    mo.md(r"""
    ## What was reconstructed

    A public `google/ddpm-cifar10-32` epsilon model was converted into a
    continuous probability-flow teacher. Its final convolution was repeated
    into 8 or 16 interval heads. During training, the student first rolled its
    own detached predictions to an interval inside a sampled block; the chosen
    head then regressed a stop-gradient Euler or midpoint teacher velocity.
    Euler used two sampled targets per update, matching midpoint's two teacher
    evaluations.

    At inference, weights and biases for every head in a block were summed
    into one fused convolution. For the selected eight-head model, block sizes
    8, 4, 2, and 1 therefore use 1, 2, 4, and 8 student evaluations. A direct
    numerical check compared the fused layer with the explicit sum of heads.
    """)
    return


@app.cell
def _(data, mo, primary):
    _nfe_one = "1"
    naive = data["naive"][_nfe_one]
    euler = primary["euler"][_nfe_one]
    midpoint = primary["midpoint"][_nfe_one]
    mo.md(
        f"""
        ## Reading the central comparison

        At 1 NFE, the naive control has **KID × 1,000 =
        {naive['kid_x1000']:.1f}**. The selected Euler- and midpoint-target PDD
        models obtain **{euler['kid_x1000']:.1f}** and
        **{midpoint['kid_x1000']:.1f}**, respectively. Their paired trajectory
        MSEs are **{euler['trajectory_mse']:.3f}** and
        **{midpoint['trajectory_mse']:.3f}**, versus
        **{naive['trajectory_mse']:.3f}** for the control.

        KID is noisy at 1,024 samples, so the paired trajectory metric and
        feature diversity are co-primary diagnostics. All methods use the same
        1,024 fixed noise seeds and the same public CIFAR-10 evaluation images.
        """
    )
    return


@app.cell
def _(data, mo, primary):
    rows = []
    for _nfe in ["1", "2", "4", "8"]:
        rows.append(
            {
                "NFE": int(_nfe),
                "naive KID": round(data["naive"][_nfe]["kid_x1000"], 2),
                "Euler PDD KID": round(primary["euler"][_nfe]["kid_x1000"], 2),
                "midpoint PDD KID": round(primary["midpoint"][_nfe]["kid_x1000"], 2),
                "naive MSE": round(data["naive"][_nfe]["trajectory_mse"], 4),
                "Euler PDD MSE": round(primary["euler"][_nfe]["trajectory_mse"], 4),
                "midpoint PDD MSE": round(primary["midpoint"][_nfe]["trajectory_mse"], 4),
            }
        )
    mo.ui.table(rows, pagination=False)
    return


@app.cell
def _(DATA_URL, mo):
    mo.md(
        f"""
        ## Scope and reproducibility

        Formal jobs ran only through OpenResearch Kubernetes on NVIDIA RTX PRO
        6000 Blackwell GPUs, with a peak allocation of 16 GPUs. The expensive
        runs do not need to be repeated to inspect the evidence: this notebook
        fetches the compact measured dataset from [the public result
        file]({DATA_URL}).

        The substitution from ImageNet/video to CIFAR-10 tests the algorithmic
        direction under a public compact teacher. It cannot validate the paper's
        headline large-model media quality, and KID here is a bounded surrogate
        rather than the paper's ImageNet FID.
        """
    )
    return


if __name__ == "__main__":
    app.run()

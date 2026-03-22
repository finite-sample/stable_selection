#!/usr/bin/env python3
"""
Reproducible pipeline for:
  "Selecting for Stability: Ensemble Proximity as a Criterion
   for Lower Out-of-Sample Prediction Variance"

Usage:
  python reproduce.py          # run everything
  python reproduce.py --quick  # reduced scale for testing
  python reproduce.py --help   # show all options

Outputs:
  ../figs/fig_decomposition.pdf
  ../figs/fig_factorial.pdf
  ../figs/fig_heatmap_transfer.pdf
  ../figs/fig_disagreement.pdf
  ../figs/fig_baselines.pdf
  ../figs/fig_robustness.pdf
  ../tabs/tab_transfer.tex
  ../tabs/tab_summary.tex
  ../tabs/tab_baseline_comparison.tex
  ../tabs/tab_robustness.tex

Requires Python 3.11+
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.special import rel_entr
from sklearn.calibration import calibration_curve
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_wine,
    make_classification,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, log_loss

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Config:
    """Experiment configuration."""

    quick: bool = False
    seed: int = 42
    n_bootstrap: int = 1000
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    # Pool sizes
    @property
    def K(self) -> int:
        return 10 if self.quick else 12

    @property
    def M(self) -> int:
        return 5 if self.quick else 6

    @property
    def epochs(self) -> int:
        return 30

    @property
    def figs_dir(self) -> Path:
        return self.output_dir / "figs"

    @property
    def tabs_dir(self) -> Path:
        return self.output_dir / "tabs"

    def __post_init__(self) -> None:
        self.figs_dir.mkdir(exist_ok=True)
        self.tabs_dir.mkdir(exist_ok=True)


@dataclass
class CellResult:
    """Results from evaluating a single DGP configuration."""

    rho: float
    pval: float
    var_red: float
    var_red_ci_lo: float
    var_red_ci_hi: float
    cohens_d: float
    bv_var: float
    ep_var: float
    bv_var_median: float
    ep_var_median: float
    bv_var_iqr: tuple[float, float]
    ep_var_iqr: tuple[float, float]
    disagree: float
    bv_acc: float
    ep_acc: float
    acc_gap: float
    baselines: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rho": self.rho,
            "pval": self.pval,
            "var_red": self.var_red,
            "var_red_ci": [self.var_red_ci_lo, self.var_red_ci_hi],
            "cohens_d": self.cohens_d,
            "bv_var": self.bv_var,
            "ep_var": self.ep_var,
            "disagree": self.disagree,
            "bv_acc": self.bv_acc,
            "ep_acc": self.ep_acc,
            "acc_gap": self.acc_gap,
            "baselines": self.baselines,
        }


# ═══════════════════════════════════════════════════════════════════════
# Plotting defaults
# ═══════════════════════════════════════════════════════════════════════

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


# ═══════════════════════════════════════════════════════════════════════
# MLP Implementation (NumPy)
# ═══════════════════════════════════════════════════════════════════════


def relu(x: NDArray) -> NDArray:
    return np.maximum(0, x)


def relu_grad(x: NDArray) -> NDArray:
    return (x > 0).astype(float)


def softmax(x: NDArray) -> NDArray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    """Simple multi-layer perceptron with ReLU activations."""

    def __init__(self, sizes: list[int], seed: int) -> None:
        rng = np.random.RandomState(seed)
        self.W: list[NDArray] = [
            rng.randn(sizes[i], sizes[i + 1]) * np.sqrt(2.0 / sizes[i])
            for i in range(len(sizes) - 1)
        ]
        self.b: list[NDArray] = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.a: list[NDArray] = []
        self.z: list[NDArray] = []

    def forward(self, X: NDArray) -> NDArray:
        self.a, self.z = [X], []
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            self.z.append(z)
            h = relu(z) if i < len(self.W) - 1 else softmax(z)
            self.a.append(h)
        return h

    def backward(self, yoh: NDArray) -> tuple[list[NDArray], list[NDArray]]:
        gw, gb = [], []
        n = yoh.shape[0]
        d = (self.a[-1] - yoh) / n
        for i in range(len(self.W) - 1, -1, -1):
            gw.insert(0, self.a[i].T @ d)
            gb.insert(0, d.sum(0))
            if i > 0:
                d = (d @ self.W[i].T) * relu_grad(self.z[i - 1])
        return gw, gb

    def predict_proba(self, X: NDArray) -> NDArray:
        return self.forward(X)

    def get_weights(self) -> list[NDArray]:
        return [w.copy() for w in self.W] + [b.copy() for b in self.b]

    def set_weights(self, weights: list[NDArray]) -> None:
        n_layers = len(self.W)
        self.W = [w.copy() for w in weights[:n_layers]]
        self.b = [b.copy() for b in weights[n_layers:]]


def train_sgd(
    m: MLP,
    X: NDArray,
    yoh: NDArray,
    epochs: int,
    lr: float,
    batch_size: int,
    rng: np.random.RandomState,
) -> None:
    n = X.shape[0]
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch_size):
            bi = idx[s : s + batch_size]
            m.forward(X[bi])
            gw, gb = m.backward(yoh[bi])
            for j in range(len(m.W)):
                m.W[j] -= lr * gw[j]
                m.b[j] -= lr * gb[j]


def train_adam(
    m: MLP,
    X: NDArray,
    yoh: NDArray,
    epochs: int,
    lr: float,
    batch_size: int,
    rng: np.random.RandomState,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    n = X.shape[0]
    mw = [np.zeros_like(w) for w in m.W]
    vw = [np.zeros_like(w) for w in m.W]
    mb = [np.zeros_like(b) for b in m.b]
    vb = [np.zeros_like(b) for b in m.b]
    t = 0

    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch_size):
            t += 1
            bi = idx[s : s + batch_size]
            m.forward(X[bi])
            gw, gb = m.backward(yoh[bi])

            for j in range(len(m.W)):
                mw[j] = beta1 * mw[j] + (1 - beta1) * gw[j]
                vw[j] = beta2 * vw[j] + (1 - beta2) * (gw[j] ** 2)
                mw_hat = mw[j] / (1 - beta1**t)
                vw_hat = vw[j] / (1 - beta2**t)
                m.W[j] -= lr * mw_hat / (np.sqrt(vw_hat) + eps)

                mb[j] = beta1 * mb[j] + (1 - beta1) * gb[j]
                vb[j] = beta2 * vb[j] + (1 - beta2) * (gb[j] ** 2)
                mb_hat = mb[j] / (1 - beta1**t)
                vb_hat = vb[j] / (1 - beta2**t)
                m.b[j] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


def onehot(y: NDArray, k: int) -> NDArray:
    oh = np.zeros((len(y), k))
    oh[np.arange(len(y)), y] = 1
    return oh


def split_scale(
    X: NDArray, y: NDArray, seed: int = 42
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    Xtv, Xte, ytv, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    Xtr, Xv, ytr, yv = train_test_split(
        Xtv, ytv, test_size=0.2, random_state=seed, stratify=ytv
    )
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xv), sc.transform(Xte), ytr, yv, yte


def arch_for(nf: int, nc: int) -> list[int]:
    h = max(32, min(96, nf * 2))
    return [nf, h, h // 2, nc]


def arch_deep(nf: int, nc: int) -> list[int]:
    return [nf, 64, 256, 128, 64, nc]


# ═══════════════════════════════════════════════════════════════════════
# Divergence Metrics
# ═══════════════════════════════════════════════════════════════════════


def l2_divergence(p: NDArray, q: NDArray) -> float:
    return float(np.mean(np.sum((p - q) ** 2, axis=1)))


def kl_divergence(p: NDArray, q: NDArray) -> float:
    p_clip = np.clip(p, 1e-10, 1)
    q_clip = np.clip(q, 1e-10, 1)
    return float(np.mean(np.sum(p_clip * np.log(p_clip / q_clip), axis=1)))


def js_divergence(p: NDArray, q: NDArray) -> float:
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))


def hellinger_divergence(p: NDArray, q: NDArray) -> float:
    return float(np.mean(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2, axis=1))))


def total_variation(p: NDArray, q: NDArray) -> float:
    return float(0.5 * np.mean(np.sum(np.abs(p - q), axis=1)))


DIVERGENCE_METRICS: dict[str, Callable[[NDArray, NDArray], float]] = {
    "l2": l2_divergence,
    "kl": kl_divergence,
    "js": js_divergence,
    "hellinger": hellinger_divergence,
    "tv": total_variation,
}


# ═══════════════════════════════════════════════════════════════════════
# Selection Criteria
# ═══════════════════════════════════════════════════════════════════════


def compute_ece(probs: NDArray, labels: NDArray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    accs = (preds == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = accs[mask].mean()
            bin_conf = confs[mask].mean()
            ece += mask.sum() * np.abs(bin_acc - bin_conf)
    return float(ece / len(labels))


def compute_entropy(probs: NDArray) -> float:
    """Mean entropy of predictions."""
    probs_clip = np.clip(probs, 1e-10, 1)
    return float(-np.mean(np.sum(probs_clip * np.log(probs_clip), axis=1)))


def compute_pairwise_disagreement(pool: list[NDArray]) -> NDArray:
    """Compute mean pairwise disagreement for each model."""
    K = len(pool)
    preds = [p.argmax(axis=1) for p in pool]
    disagreements = np.zeros(K)
    for i in range(K):
        dis = 0.0
        for j in range(K):
            if i != j:
                dis += np.mean(preds[i] != preds[j])
        disagreements[i] = dis / (K - 1)
    return disagreements


def average_weights(models: list[MLP]) -> MLP:
    """Create a new model with averaged weights."""
    avg_model = MLP([1, 1], seed=0)
    avg_model.W = []
    avg_model.b = []

    n_models = len(models)
    for i in range(len(models[0].W)):
        stacked = np.stack([m.W[i] for m in models], axis=0)
        avg_model.W.append(stacked.mean(axis=0))
    for i in range(len(models[0].b)):
        stacked = np.stack([m.b[i] for m in models], axis=0)
        avg_model.b.append(stacked.mean(axis=0))

    return avg_model


# ═══════════════════════════════════════════════════════════════════════
# Bootstrap utilities
# ═══════════════════════════════════════════════════════════════════════


def bootstrap_variance_reduction(
    bv_vars: NDArray, ep_vars: NDArray, n_bootstrap: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    """Bootstrap CI for variance reduction percentage."""
    rng = np.random.RandomState(seed)
    n = len(bv_vars)

    vr_boots = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        bv = bv_vars[idx].mean()
        ep = ep_vars[idx].mean()
        if bv > 1e-15:
            vr_boots.append((bv - ep) / bv * 100)
        else:
            vr_boots.append(0.0)

    vr_boots = np.array(vr_boots)
    ci_lo = float(np.percentile(vr_boots, 2.5))
    ci_hi = float(np.percentile(vr_boots, 97.5))
    return float(np.mean(vr_boots)), ci_lo, ci_hi


def cohens_d(x1: NDArray, x2: NDArray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(x1), len(x2)
    var1, var2 = x1.var(ddof=1), x2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-15:
        return 0.0
    return float((x1.mean() - x2.mean()) / pooled_std)


def benjamini_hochberg(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_pvals = np.array(pvals)[sorted_idx]

    thresholds = np.arange(1, n + 1) / n * alpha
    rejected = sorted_pvals <= thresholds

    max_reject = -1
    for i in range(n):
        if rejected[i]:
            max_reject = i

    result = [False] * n
    if max_reject >= 0:
        for i in range(max_reject + 1):
            result[sorted_idx[i]] = True
    return result


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Variance Decomposition
# ═══════════════════════════════════════════════════════════════════════


def experiment_decomposition(cfg: Config) -> dict[str, Any]:
    log.info("Variance Decomposition experiment")
    d = load_digits()
    X, y = d.data, d.target
    Xtr, Xv, Xte, ytr, yv, yte = split_scale(X, y, cfg.seed)
    yoh = onehot(ytr, 10)
    arch = [64, 128, 64, 10]
    K = 20 if cfg.quick else 30

    results: dict[str, dict[str, float]] = {}
    for label, seed_fn in [
        ("Init only", lambda k: (k, 0)),
        ("Shuffle only", lambda k: (0, k)),
        ("Both", lambda k: (k, k + 1000)),
    ]:
        preds = np.zeros((K, len(yte), 10))
        accs = []
        for k in range(K):
            iseed, sseed = seed_fn(k)
            m = MLP(arch, seed=iseed)
            train_sgd(m, Xtr, yoh, 60, 0.01, 32, np.random.RandomState(sseed))
            preds[k] = m.predict_proba(Xte)
            accs.append(accuracy_score(yte, preds[k].argmax(1)))

        pev = preds.var(axis=0).mean(axis=1)
        results[label] = {
            "mean_var": float(pev.mean()),
            "median_var": float(np.median(pev)),
            "var_iqr_lo": float(np.percentile(pev, 25)),
            "var_iqr_hi": float(np.percentile(pev, 75)),
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
        }
        log.info(f"  {label:15s}: var={pev.mean():.7f}, acc={np.mean(accs):.4f}")

    ratio = results["Init only"]["mean_var"] / max(
        results["Shuffle only"]["mean_var"], 1e-15
    )
    results["ratio"] = {"value": float(ratio)}
    log.info(f"  Ratio: {ratio:.0f}x")

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(5.5, 2.8), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    labs = ["Init\nonly", "Shuffle\nonly", "Both"]
    vals = [results[l]["mean_var"] for l in ["Init only", "Shuffle only", "Both"]]
    cols = ["#4C72B0", "#DD8452", "#55A868"]

    ax1.bar(
        labs, vals, color=cols, alpha=0.85, edgecolor="black", linewidth=0.4, width=0.55
    )
    ax1.set_ylabel("Mean prediction variance")
    ax1.set_title("(a) Linear scale")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.annotate(
        f"$\\approx${ratio:.0f}$\\times$",
        xy=(1, vals[1]),
        xytext=(1.6, vals[0] * 0.5),
        arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.2),
        fontsize=9,
        color="#C44E52",
        fontweight="bold",
    )

    ax2.bar(
        labs, vals, color=cols, alpha=0.85, edgecolor="black", linewidth=0.4, width=0.55
    )
    ax2.set_yscale("log")
    ax2.set_title("(b) Log scale")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(cfg.figs_dir / "fig_decomposition.pdf")
    plt.close()
    return results


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Single-cell evaluation with baselines
# ═══════════════════════════════════════════════════════════════════════


def evaluate_cell(
    dgp_params: dict[str, Any],
    cfg: Config,
    K: int | None = None,
    M: int | None = None,
    epochs: int | None = None,
    divergence: str = "l2",
    rashomon_eps: float = 0.03,
    optimizer: str = "sgd",
    lr: float = 0.01,
    arch_fn: Callable[[int, int], list[int]] = arch_for,
) -> CellResult:
    if K is None:
        K = cfg.K
    if M is None:
        M = cfg.M
    if epochs is None:
        epochs = cfg.epochs

    X, y = make_classification(**dgp_params)
    nc = len(np.unique(y))
    Xtr, Xv, Xte, ytr, yv, yte = split_scale(X, y, cfg.seed)
    yoh = onehot(ytr, nc)
    arch = arch_fn(Xtr.shape[1], nc)
    n_test = len(yte)

    train_fn = train_adam if optimizer == "adam" else train_sgd
    div_fn = DIVERGENCE_METRICS[divergence]

    # --- Val -> Test transfer ---
    pool_vp, pool_tp = [], []
    for k in range(K):
        m = MLP(arch, seed=k)
        train_fn(m, Xtr, yoh, epochs, lr, 32, np.random.RandomState(k + 500))
        pool_vp.append(m.predict_proba(Xv))
        pool_tp.append(m.predict_proba(Xte))

    ens_v = np.stack(pool_vp).mean(0)
    ens_t = np.stack(pool_tp).mean(0)
    val_divs = [div_fn(pool_vp[k], ens_v) for k in range(K)]
    test_divs = [div_fn(pool_tp[k], ens_t) for k in range(K)]

    rho, pval = stats.spearmanr(val_divs, test_divs)

    # --- Meta-stability with all selection criteria ---
    criteria = ["best_val", "ens_prox", "random", "ece", "max_entropy", "median_disagree", "weight_avg"]
    sel_preds: dict[str, NDArray] = {c: np.zeros((M, n_test, nc)) for c in criteria}
    sel_accs: dict[str, list[float]] = {c: [] for c in criteria}
    per_rep_vars: dict[str, list[float]] = {c: [] for c in criteria}

    for rep in range(M):
        base = (rep + 1) * 10000
        pool2: list[dict[str, Any]] = []
        models: list[MLP] = []

        for k in range(K):
            m = MLP(arch, seed=base + k)
            train_fn(
                m, Xtr, yoh, epochs, lr, 32, np.random.RandomState(base + k + 5000)
            )
            vp = m.predict_proba(Xv)
            tp = m.predict_proba(Xte)
            pool2.append(
                {
                    "vp": vp,
                    "tp": tp,
                    "va": accuracy_score(yv, vp.argmax(1)),
                    "div": None,
                    "ece": compute_ece(vp, yv),
                    "entropy": compute_entropy(vp),
                }
            )
            models.append(m)

        ev = np.stack([m["vp"] for m in pool2]).mean(0)
        for m_dict in pool2:
            m_dict["div"] = div_fn(m_dict["vp"], ev)

        # Compute pairwise disagreement
        disagreements = compute_pairwise_disagreement([m["vp"] for m in pool2])
        for i, m_dict in enumerate(pool2):
            m_dict["disagree"] = disagreements[i]

        # Rashomon set filtering
        bva = max(m["va"] for m in pool2)
        cands_idx = [i for i, m in enumerate(pool2) if m["va"] >= bva - rashomon_eps]
        if len(cands_idx) < 3:
            cands_idx = list(range(K))
        cands = [pool2[i] for i in cands_idx]
        cands_models = [models[i] for i in cands_idx]

        # Selection criteria
        rng = np.random.RandomState(base)
        selections = {
            "best_val": max(cands, key=lambda m: m["va"]),
            "ens_prox": min(cands, key=lambda m: m["div"]),
            "random": cands[rng.choice(len(cands))],
            "ece": min(cands, key=lambda m: m["ece"]),
            "max_entropy": max(cands, key=lambda m: m["entropy"]),
            "median_disagree": sorted(cands, key=lambda m: m["disagree"])[len(cands) // 2],
        }

        # Weight averaging
        avg_model = average_weights(cands_models)
        wa_tp = avg_model.predict_proba(Xte)
        wa_acc = accuracy_score(yte, wa_tp.argmax(1))

        for c, sel in selections.items():
            sel_preds[c][rep] = sel["tp"]
            sel_accs[c].append(accuracy_score(yte, sel["tp"].argmax(1)))

        sel_preds["weight_avg"][rep] = wa_tp
        sel_accs["weight_avg"].append(wa_acc)

    # Compute results for all criteria
    results_by_criterion: dict[str, dict[str, float]] = {}
    for c in criteria:
        preds = sel_preds[c]
        pv = preds.var(axis=0).mean(axis=1)
        hard = preds.argmax(axis=2)
        dis = np.mean([len(np.unique(hard[:, i])) > 1 for i in range(n_test)])
        results_by_criterion[c] = {
            "pred_var": float(pv.mean()),
            "pred_var_median": float(np.median(pv)),
            "pred_var_iqr_lo": float(np.percentile(pv, 25)),
            "pred_var_iqr_hi": float(np.percentile(pv, 75)),
            "disagree": float(dis),
            "acc_mean": float(np.mean(sel_accs[c])),
            "acc_std": float(np.std(sel_accs[c])),
        }

    # Compute bootstrap CI and effect size for ensemble proximity vs best_val
    bv_preds = sel_preds["best_val"]
    ep_preds = sel_preds["ens_prox"]
    bv_vars = bv_preds.var(axis=0).mean(axis=1)
    ep_vars = ep_preds.var(axis=0).mean(axis=1)

    vr_mean, ci_lo, ci_hi = bootstrap_variance_reduction(
        bv_vars, ep_vars, cfg.n_bootstrap, cfg.seed
    )
    d = cohens_d(bv_vars, ep_vars)

    bv_var = float(bv_vars.mean())
    ep_var = float(ep_vars.mean())
    vr = (bv_var - ep_var) / bv_var * 100 if bv_var > 1e-15 else 0.0

    return CellResult(
        rho=float(rho),
        pval=float(pval),
        var_red=float(vr),
        var_red_ci_lo=ci_lo,
        var_red_ci_hi=ci_hi,
        cohens_d=d,
        bv_var=bv_var,
        ep_var=ep_var,
        bv_var_median=results_by_criterion["best_val"]["pred_var_median"],
        ep_var_median=results_by_criterion["ens_prox"]["pred_var_median"],
        bv_var_iqr=(
            results_by_criterion["best_val"]["pred_var_iqr_lo"],
            results_by_criterion["best_val"]["pred_var_iqr_hi"],
        ),
        ep_var_iqr=(
            results_by_criterion["ens_prox"]["pred_var_iqr_lo"],
            results_by_criterion["ens_prox"]["pred_var_iqr_hi"],
        ),
        disagree=results_by_criterion["best_val"]["disagree"],
        bv_acc=results_by_criterion["best_val"]["acc_mean"],
        ep_acc=results_by_criterion["ens_prox"]["acc_mean"],
        acc_gap=float(
            results_by_criterion["best_val"]["acc_mean"]
            - results_by_criterion["ens_prox"]["acc_mean"]
        ),
        baselines=results_by_criterion,
    )


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Factorial simulation
# ═══════════════════════════════════════════════════════════════════════


def experiment_factorial(cfg: Config) -> dict[str, list[dict[str, Any]]]:
    log.info("Factorial Simulation experiment")
    base = dict(
        n_samples=1000,
        n_features=30,
        n_informative=10,
        n_redundant=10,
        n_classes=4,
        n_clusters_per_class=2,
        flip_y=0.05,
        class_sep=1.0,
        random_state=cfg.seed,
    )

    factors: dict[str, tuple[str, list[Any], dict[str, Any]]] = {
        "Class separation": (
            "class_sep",
            [0.3, 0.5, 0.8, 1.0, 1.5, 2.0] if not cfg.quick else [0.3, 0.8, 1.5],
            {},
        ),
        "Label noise": ("flip_y", [0.0, 0.05, 0.10, 0.20], {}),
        "Num. classes": (
            "n_classes",
            [2, 3, 4, 7] if not cfg.quick else [2, 4, 7],
            {"n_clusters_per_class": lambda nc: max(1, 3 - nc // 3)},
        ),
        "Sample size": (
            "n_samples",
            [300, 500, 1000, 2000] if not cfg.quick else [300, 1000, 3000],
            {},
        ),
        "Redundancy": (
            "n_redundant",
            [0, 5, 10, 20] if not cfg.quick else [0, 10, 20],
            {"n_features": lambda nr: 10 + nr + 5},
        ),
    }

    all_results: dict[str, list[dict[str, Any]]] = {}
    import time

    for fname, (pname, levels, transforms) in factors.items():
        log.info(f"  Factor: {fname}")
        all_results[fname] = []
        for val in levels:
            params = {**base, pname: val}
            for tname, tfn in transforms.items():
                params[tname] = tfn(val) if callable(tfn) else tfn
            t0 = time.time()
            r = evaluate_cell(params, cfg)
            result_dict = r.to_dict()
            result_dict["level"] = val
            all_results[fname].append(result_dict)
            log.info(
                f"    {pname}={val}: var_red={r.var_red:+.1f}% "
                f"[{r.var_red_ci_lo:+.1f}, {r.var_red_ci_hi:+.1f}], "
                f"d={r.cohens_d:.2f}, rho={r.rho:+.2f} [{time.time()-t0:.0f}s]"
            )

    return all_results


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: 2D Grid (noise x separation)
# ═══════════════════════════════════════════════════════════════════════


def experiment_grid(
    cfg: Config,
) -> tuple[dict[tuple[float, float], dict[str, Any]], list[float], list[float]]:
    log.info("2D Grid: noise x separation")
    base = dict(
        n_samples=1000,
        n_features=30,
        n_informative=10,
        n_redundant=10,
        n_classes=4,
        n_clusters_per_class=2,
        random_state=cfg.seed,
    )

    flips = [0.0, 0.05, 0.10, 0.20]
    seps = [0.3, 0.5, 0.8, 1.0, 1.5] if not cfg.quick else [0.5, 1.0, 1.5]
    grid: dict[tuple[float, float], dict[str, Any]] = {}
    import time

    for f in flips:
        for s in seps:
            params = {**base, "flip_y": f, "class_sep": s}
            t0 = time.time()
            K_g = 8 if cfg.quick else 10
            M_g = 4 if cfg.quick else 5
            r = evaluate_cell(params, cfg, K=K_g, M=M_g)
            grid[(f, s)] = r.to_dict()
            log.info(
                f"  flip={f:.2f}, sep={s}: var_red={r.var_red:+.1f}%, "
                f"dis={r.disagree:.2f} [{time.time()-t0:.0f}s]"
            )
    return grid, flips, seps


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Benchmark datasets (val->test transfer)
# ═══════════════════════════════════════════════════════════════════════


def load_openml_dataset(name: str, data_id: int) -> tuple[NDArray, NDArray]:
    """Load dataset from OpenML."""
    data = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    df = data.data
    y = data.target

    if hasattr(y, 'cat'):
        y = y.cat.codes.values
    elif y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y.values)
    else:
        y = y.values

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].values.astype(np.float64)

    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]

    return X, y.astype(np.int64)


def experiment_benchmarks(cfg: Config) -> dict[str, dict[str, Any]]:
    log.info("Benchmark Datasets: Val->Test Transfer")

    benchmarks: dict[str, tuple[NDArray, NDArray]] = {
        "Digits": (load_digits().data, load_digits().target),
        "Wine": (load_wine().data, load_wine().target),
        "Breast Cancer": (load_breast_cancer().data, load_breast_cancer().target),
        "Diabetes": (load_diabetes().data, (load_diabetes().target > 140).astype(int)),
    }

    if not cfg.quick:
        try:
            log.info("  Loading Adult dataset from OpenML...")
            X_adult, y_adult = load_openml_dataset("adult", 1590)
            if len(X_adult) > 10000:
                idx = np.random.RandomState(cfg.seed).choice(
                    len(X_adult), 10000, replace=False
                )
                X_adult, y_adult = X_adult[idx], y_adult[idx]
            benchmarks["Adult"] = (X_adult, y_adult)
        except Exception as e:
            log.warning(f"  Could not load Adult dataset: {e}")

        try:
            log.info("  Loading German Credit dataset from OpenML...")
            X_german, y_german = load_openml_dataset("german_credit", 31)
            benchmarks["German Credit"] = (X_german, y_german)
        except Exception as e:
            log.warning(f"  Could not load German Credit dataset: {e}")

        try:
            log.info("  Loading Vehicle dataset from OpenML...")
            X_vehicle, y_vehicle = load_openml_dataset("vehicle", 54)
            benchmarks["Vehicle"] = (X_vehicle, y_vehicle)
        except Exception as e:
            log.warning(f"  Could not load Vehicle dataset: {e}")

    K = 15 if cfg.quick else 20
    ep = 40 if cfg.quick else 50
    results: dict[str, dict[str, Any]] = {}

    for name, (X, y) in benchmarks.items():
        nc = len(np.unique(y))
        try:
            Xtr, Xv, Xte, ytr, yv, yte = split_scale(X, y, cfg.seed)
        except ValueError as e:
            log.warning(f"  Skipping {name}: {e}")
            continue

        yoh = onehot(ytr, nc)
        arch = arch_for(Xtr.shape[1], nc)

        pool_vp, pool_tp, pool_va = [], [], []
        for k in range(K):
            m = MLP(arch, seed=k)
            train_sgd(m, Xtr, yoh, ep, 0.01, 32, np.random.RandomState(k + 500))
            vp = m.predict_proba(Xv)
            tp = m.predict_proba(Xte)
            pool_vp.append(vp)
            pool_tp.append(tp)
            pool_va.append(accuracy_score(yv, vp.argmax(1)))

        ens_v = np.stack(pool_vp).mean(0)
        ens_t = np.stack(pool_tp).mean(0)
        vkl, tkl = [], []
        vl2, tl2 = [], []
        for k in range(K):
            vkl.append(kl_divergence(pool_vp[k], ens_v))
            tkl.append(kl_divergence(pool_tp[k], ens_t))
            vl2.append(l2_divergence(pool_vp[k], ens_v))
            tl2.append(l2_divergence(pool_tp[k], ens_t))

        rho_kl, pval_kl = stats.spearmanr(vkl, tkl)
        rho_l2, pval_l2 = stats.spearmanr(vl2, tl2)

        results[name] = {
            "rho_kl": float(rho_kl),
            "p_kl": float(pval_kl),
            "rho_l2": float(rho_l2),
            "p_l2": float(pval_l2),
            "n": len(y),
            "p": X.shape[1],
            "nc": nc,
        }
        log.info(
            f"  {name:15s}: rho_kl={rho_kl:+.2f} (p={pval_kl:.3f}), "
            f"rho_l2={rho_l2:+.2f} (p={pval_l2:.3f})"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 6: Robustness checks
# ═══════════════════════════════════════════════════════════════════════


def experiment_robustness(cfg: Config) -> dict[str, list[dict[str, Any]]]:
    log.info("Robustness checks")
    base = dict(
        n_samples=1000,
        n_features=30,
        n_informative=10,
        n_redundant=10,
        n_classes=4,
        n_clusters_per_class=2,
        flip_y=0.05,
        class_sep=0.8,
        random_state=cfg.seed,
    )

    results: dict[str, list[dict[str, Any]]] = {}
    import time

    # Rashomon threshold sensitivity
    log.info("  Rashomon threshold sensitivity")
    results["rashomon_eps"] = []
    eps_values = [0.01, 0.02, 0.03, 0.05, 0.10] if not cfg.quick else [0.02, 0.05]
    for eps in eps_values:
        t0 = time.time()
        r = evaluate_cell(base, cfg, rashomon_eps=eps)
        result_dict = r.to_dict()
        result_dict["eps"] = eps
        results["rashomon_eps"].append(result_dict)
        log.info(f"    eps={eps}: var_red={r.var_red:+.1f}% [{time.time()-t0:.0f}s]")

    # Divergence metrics
    log.info("  Divergence metrics")
    results["divergence"] = []
    div_metrics = ["l2", "kl", "js", "hellinger", "tv"] if not cfg.quick else ["l2", "kl"]
    for div in div_metrics:
        t0 = time.time()
        r = evaluate_cell(base, cfg, divergence=div)
        result_dict = r.to_dict()
        result_dict["divergence"] = div
        results["divergence"].append(result_dict)
        log.info(f"    {div}: var_red={r.var_red:+.1f}% [{time.time()-t0:.0f}s]")

    # Optimizer comparison
    log.info("  Optimizer comparison")
    results["optimizer"] = []
    optimizers = ["sgd", "adam"] if not cfg.quick else ["sgd"]
    for opt in optimizers:
        t0 = time.time()
        r = evaluate_cell(base, cfg, optimizer=opt)
        result_dict = r.to_dict()
        result_dict["optimizer"] = opt
        results["optimizer"].append(result_dict)
        log.info(f"    {opt}: var_red={r.var_red:+.1f}% [{time.time()-t0:.0f}s]")

    # Learning rate sensitivity
    log.info("  Learning rate sensitivity")
    results["learning_rate"] = []
    lrs = [0.001, 0.005, 0.01, 0.05] if not cfg.quick else [0.005, 0.01]
    for lr in lrs:
        t0 = time.time()
        r = evaluate_cell(base, cfg, lr=lr)
        result_dict = r.to_dict()
        result_dict["lr"] = lr
        results["learning_rate"].append(result_dict)
        log.info(f"    lr={lr}: var_red={r.var_red:+.1f}% [{time.time()-t0:.0f}s]")

    return results


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 7: Architecture breadth
# ═══════════════════════════════════════════════════════════════════════


def experiment_architectures(cfg: Config) -> dict[str, dict[str, Any]]:
    log.info("Architecture breadth experiments")
    base = dict(
        n_samples=1000,
        n_features=30,
        n_informative=10,
        n_redundant=10,
        n_classes=4,
        n_clusters_per_class=2,
        flip_y=0.05,
        class_sep=0.8,
        random_state=cfg.seed,
    )

    results: dict[str, dict[str, Any]] = {}
    import time

    # Standard MLP
    log.info("  Standard MLP [64, 32]")
    t0 = time.time()
    r = evaluate_cell(base, cfg, arch_fn=arch_for)
    results["mlp_standard"] = r.to_dict()
    log.info(f"    var_red={r.var_red:+.1f}% [{time.time()-t0:.0f}s]")

    # Deeper MLP
    log.info("  Deep MLP [64, 256, 128, 64]")
    t0 = time.time()
    r = evaluate_cell(base, cfg, arch_fn=arch_deep)
    results["mlp_deep"] = r.to_dict()
    log.info(f"    var_red={r.var_red:+.1f}% [{time.time()-t0:.0f}s]")

    return results


# ═══════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════


def make_fig_factorial(cfg: Config, factorial_results: dict[str, list[dict[str, Any]]]) -> tuple[float, float]:
    """Six-panel factor sweep figure with CIs."""
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.5))

    factor_order = [
        "Class separation",
        "Sample size",
        "Num. classes",
        "Redundancy",
        "Label noise",
    ]
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    for idx, (fname, plabel) in enumerate(zip(factor_order, panel_labels)):
        ax = axes.flat[idx]
        data = factorial_results[fname]
        xs = [d["level"] for d in data]
        vr = [d["var_red"] for d in data]
        ci_lo = [d["var_red_ci"][0] for d in data]
        ci_hi = [d["var_red_ci"][1] for d in data]
        dis = [d["disagree"] for d in data]

        c1, c2 = "#4C72B0", "#C44E52"
        x_pos = list(range(len(xs)))
        bars = ax.bar(
            x_pos,
            vr,
            color=[c1 if v > 0 else "#B0C4DE" for v in vr],
            alpha=0.75,
            edgecolor="black",
            linewidth=0.3,
            width=0.55,
        )

        errors = [[vr[i] - ci_lo[i] for i in range(len(vr))],
                  [ci_hi[i] - vr[i] for i in range(len(vr))]]
        ax.errorbar(
            x_pos,
            vr,
            yerr=errors,
            fmt="none",
            color="black",
            capsize=2,
            capthick=0.5,
            linewidth=0.5,
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(x) for x in xs])
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
        ax.set_title(f"{plabel} {fname}", fontsize=9)
        ax.spines["top"].set_visible(False)
        if idx % 3 == 0:
            ax.set_ylabel("Var. reduction (%)")

        ax2 = ax.twinx()
        ax2.plot(x_pos, dis, "o-", color=c2, lw=1, ms=3, alpha=0.6)
        if idx % 3 == 2:
            ax2.set_ylabel("Disagreement", color=c2, fontsize=8)
        ax2.tick_params(axis="y", colors=c2, labelsize=7)
        ax2.spines["top"].set_visible(False)

    # (f) Scatter: disagreement vs var reduction
    ax = axes.flat[5]
    all_dis, all_vr = [], []
    for fname, data in factorial_results.items():
        for d in data:
            all_dis.append(d["disagree"])
            all_vr.append(d["var_red"])

    ax.scatter(
        all_dis,
        all_vr,
        s=25,
        c="#4C72B0",
        alpha=0.7,
        edgecolors="black",
        linewidths=0.3,
    )
    coef = np.polyfit(all_dis, all_vr, 1)
    xl = np.linspace(min(all_dis) - 0.03, max(all_dis) + 0.03, 100)
    ax.plot(xl, np.polyval(coef, xl), "--", color="#C44E52", lw=1.2, alpha=0.6)
    r, p = stats.spearmanr(all_dis, all_vr)
    ax.text(
        0.05,
        0.92,
        f"$\\rho$={r:+.2f}\n$p$={p:.3f}",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="wheat", alpha=0.8),
    )
    ax.axhline(0, color="gray", lw=0.4, ls="--")
    ax.set_xlabel("Baseline disagreement")
    ax.set_ylabel("Var. reduction (%)")
    ax.set_title("(f) Disagreement predicts benefit", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(h_pad=1.0)
    plt.savefig(cfg.figs_dir / "fig_factorial.pdf")
    plt.close()
    return r, p


def make_fig_heatmap_transfer(
    cfg: Config,
    grid: dict[tuple[float, float], dict[str, Any]],
    flips: list[float],
    seps: list[float],
    benchmarks: dict[str, dict[str, Any]],
) -> None:
    """Two-panel: 2D heatmap + transfer correlation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # (a) Heatmap
    mat = np.array([[grid[(f, s)]["var_red"] for s in seps] for f in flips])
    im = ax1.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=25, origin="lower")
    ax1.set_xticks(range(len(seps)))
    ax1.set_xticklabels([str(s) for s in seps])
    ax1.set_yticks(range(len(flips)))
    ax1.set_yticklabels([str(f) for f in flips])
    ax1.set_xlabel("Class separation")
    ax1.set_ylabel("Label noise")
    ax1.set_title("(a) Variance reduction (\\%)")
    for i in range(len(flips)):
        for j in range(len(seps)):
            v = mat[i, j]
            c = "white" if abs(v) > 12 else "black"
            ax1.text(
                j, i, f"{v:+.0f}", ha="center", va="center", fontsize=7, fontweight="bold", color=c
            )
    plt.colorbar(im, ax=ax1, shrink=0.85, label="%")

    # (b) Transfer correlations
    names, rhos, sig = [], [], []
    for n, r in benchmarks.items():
        names.append(n)
        rhos.append(r["rho_kl"])
        sig.append(r["p_kl"] < 0.05)

    for fname, data_list in [
        ("Sep=0.3", [grid.get((0.05, 0.3))]),
        ("Sep=1.5", [grid.get((0.05, 1.5))]),
    ]:
        if data_list[0]:
            names.append(fname)
            rhos.append(data_list[0]["rho"])
            sig.append(data_list[0]["pval"] < 0.05)

    colors = ["#4C72B0" if s else "#AAAAAA" for s in sig]
    ax2.barh(
        range(len(names)), rhos, color=colors, alpha=0.75, edgecolor="black", linewidth=0.3
    )
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("Spearman $\\rho$ (val$\\to$test KL)")
    ax2.set_title("(b) Transfer correlation")
    ax2.axvline(0, color="black", lw=0.4)
    for i, (r, s) in enumerate(zip(rhos, sig)):
        star = "*" if s else ""
        ax2.text(max(r + 0.02, 0.02), i, f"{r:.2f}{star}", va="center", fontsize=7.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(cfg.figs_dir / "fig_heatmap_transfer.pdf")
    plt.close()


def make_fig_disagreement(cfg: Config, factorial_results: dict[str, list[dict[str, Any]]]) -> tuple[float, float]:
    """Standalone scatter: disagreement vs variance reduction."""
    fig, ax = plt.subplots(figsize=(3.5, 3))
    all_dis, all_vr = [], []
    for fname, data in factorial_results.items():
        for d in data:
            all_dis.append(d["disagree"])
            all_vr.append(d["var_red"])

    ax.scatter(
        all_dis, all_vr, s=35, c="#4C72B0", alpha=0.7, edgecolors="black", linewidths=0.4
    )
    coef = np.polyfit(all_dis, all_vr, 1)
    xl = np.linspace(0.1, 1.0, 100)
    ax.plot(xl, np.polyval(coef, xl), "--", color="#C44E52", lw=1.5, alpha=0.6)
    r, p = stats.spearmanr(all_dis, all_vr)
    ax.text(
        0.05,
        0.95,
        f"$\\rho = {r:+.2f}$, $p = {p:.3f}$",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8),
    )
    ax.axhline(0, color="gray", lw=0.4, ls="--")
    ax.set_xlabel("Baseline prediction disagreement rate")
    ax.set_ylabel("Variance reduction (\\%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(cfg.figs_dir / "fig_disagreement.pdf")
    plt.close()
    return r, p


def make_fig_baselines(cfg: Config, factorial_results: dict[str, list[dict[str, Any]]]) -> None:
    """Comparison of selection criteria across all DGP configurations."""
    criteria = ["best_val", "ens_prox", "random", "ece", "max_entropy", "median_disagree", "weight_avg"]
    criteria_labels = {
        "best_val": "Best Val Acc",
        "ens_prox": "Ens. Proximity",
        "random": "Random",
        "ece": "Min ECE",
        "max_entropy": "Max Entropy",
        "median_disagree": "Med. Disagree",
        "weight_avg": "Weight Avg",
    }

    all_vars: dict[str, list[float]] = {c: [] for c in criteria}
    all_accs: dict[str, list[float]] = {c: [] for c in criteria}

    for fname, data in factorial_results.items():
        for d in data:
            if "baselines" in d:
                for c in criteria:
                    if c in d["baselines"]:
                        all_vars[c].append(d["baselines"][c]["pred_var"])
                        all_accs[c].append(d["baselines"][c]["acc_mean"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # Variance comparison
    labels = [criteria_labels[c] for c in criteria if all_vars[c]]
    means = [np.mean(all_vars[c]) for c in criteria if all_vars[c]]
    stds = [np.std(all_vars[c]) for c in criteria if all_vars[c]]

    colors = ["#4C72B0" if c == "ens_prox" else "#AAAAAA" for c in criteria if all_vars[c]]
    x_pos = np.arange(len(labels))
    ax1.bar(x_pos, means, yerr=stds, color=colors, alpha=0.75, edgecolor="black", capsize=3)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Mean Prediction Variance")
    ax1.set_title("(a) Prediction Variance by Criterion")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Accuracy comparison
    means_acc = [np.mean(all_accs[c]) for c in criteria if all_accs[c]]
    stds_acc = [np.std(all_accs[c]) for c in criteria if all_accs[c]]

    ax2.bar(x_pos, means_acc, yerr=stds_acc, color=colors, alpha=0.75, edgecolor="black", capsize=3)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Mean Accuracy")
    ax2.set_title("(b) Accuracy by Criterion")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(cfg.figs_dir / "fig_baselines.pdf")
    plt.close()


def make_fig_robustness(cfg: Config, robustness_results: dict[str, list[dict[str, Any]]]) -> None:
    """Robustness checks visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))

    # Rashomon threshold
    ax = axes[0, 0]
    data = robustness_results.get("rashomon_eps", [])
    if data:
        eps_vals = [d["eps"] for d in data]
        vr_vals = [d["var_red"] for d in data]
        ci_lo = [d["var_red_ci"][0] for d in data]
        ci_hi = [d["var_red_ci"][1] for d in data]
        ax.errorbar(eps_vals, vr_vals, yerr=[[v - lo for v, lo in zip(vr_vals, ci_lo)],
                                              [hi - v for v, hi in zip(vr_vals, ci_hi)]],
                    fmt="o-", color="#4C72B0", capsize=3)
        ax.axhline(0, color="gray", lw=0.4, ls="--")
        ax.set_xlabel("Rashomon threshold ε")
        ax.set_ylabel("Var. reduction (%)")
        ax.set_title("(a) Rashomon Threshold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Divergence metrics
    ax = axes[0, 1]
    data = robustness_results.get("divergence", [])
    if data:
        div_names = [d["divergence"].upper() for d in data]
        vr_vals = [d["var_red"] for d in data]
        colors = ["#4C72B0" if d["divergence"] == "l2" else "#AAAAAA" for d in data]
        ax.bar(range(len(div_names)), vr_vals, color=colors, alpha=0.75, edgecolor="black")
        ax.set_xticks(range(len(div_names)))
        ax.set_xticklabels(div_names)
        ax.axhline(0, color="gray", lw=0.4, ls="--")
        ax.set_ylabel("Var. reduction (%)")
        ax.set_title("(b) Divergence Metrics")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Optimizer
    ax = axes[1, 0]
    data = robustness_results.get("optimizer", [])
    if data:
        opt_names = [d["optimizer"].upper() for d in data]
        vr_vals = [d["var_red"] for d in data]
        ax.bar(range(len(opt_names)), vr_vals, color="#4C72B0", alpha=0.75, edgecolor="black")
        ax.set_xticks(range(len(opt_names)))
        ax.set_xticklabels(opt_names)
        ax.axhline(0, color="gray", lw=0.4, ls="--")
        ax.set_ylabel("Var. reduction (%)")
        ax.set_title("(c) Optimizer")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Learning rate
    ax = axes[1, 1]
    data = robustness_results.get("learning_rate", [])
    if data:
        lr_vals = [d["lr"] for d in data]
        vr_vals = [d["var_red"] for d in data]
        ax.plot(lr_vals, vr_vals, "o-", color="#4C72B0")
        ax.axhline(0, color="gray", lw=0.4, ls="--")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Var. reduction (%)")
        ax.set_title("(d) Learning Rate")
        ax.set_xscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(cfg.figs_dir / "fig_robustness.pdf")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# TABLES (LaTeX fragments)
# ═══════════════════════════════════════════════════════════════════════


def make_tab_transfer(cfg: Config, benchmarks: dict[str, dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Dataset & $n$ & $p$ & Classes & $\rho_{\mathrm{KL}}$ & $\rho_{L_2}$ \\",
        r"\midrule",
    ]
    for name, r in benchmarks.items():
        rho_kl = (
            f"\\textbf{{{r['rho_kl']:+.2f}}}"
            if r["p_kl"] < 0.05
            else f"{r['rho_kl']:+.2f}"
        )
        rho_l2 = (
            f"\\textbf{{{r['rho_l2']:+.2f}}}"
            if r["p_l2"] < 0.05
            else f"{r['rho_l2']:+.2f}"
        )
        lines.append(
            f"{name} & {r['n']} & {r['p']} & {r['nc']} & {rho_kl} & {rho_l2} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (cfg.tabs_dir / "tab_transfer.tex").write_text("\n".join(lines))


def make_tab_summary(cfg: Config, factorial_results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Summary statistics with CIs across all factorial cells."""
    all_vr, all_rho, all_gap, all_dis = [], [], [], []
    all_ci_lo, all_ci_hi, all_d = [], [], []
    all_pvals = []

    for data in factorial_results.values():
        for d in data:
            all_vr.append(d["var_red"])
            all_rho.append(d["rho"])
            all_gap.append(d["acc_gap"])
            all_dis.append(d["disagree"])
            all_ci_lo.append(d["var_red_ci"][0])
            all_ci_hi.append(d["var_red_ci"][1])
            all_d.append(d["cohens_d"])
            all_pvals.append(d["pval"])

    n_pos = sum(1 for v in all_vr if v > 0)
    sig_raw = [p < 0.05 for p in all_pvals]
    sig_fdr = benjamini_hochberg(all_pvals, 0.05)
    n_sig_raw = sum(sig_raw)
    n_sig_fdr = sum(sig_fdr)
    N = len(all_vr)

    mean_ci = f"[{np.mean(all_ci_lo):+.1f}\\%, {np.mean(all_ci_hi):+.1f}\\%]"

    lines = [
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Statistic & Value \\",
        r"\midrule",
        f"DGP configurations & {N} \\\\",
        f"Mean variance reduction & {np.mean(all_vr):+.1f}\\% \\\\",
        f"Mean 95\\% CI & {mean_ci} \\\\",
        f"Median variance reduction & {np.median(all_vr):+.1f}\\% \\\\",
        f"IQR & [{np.percentile(all_vr, 25):+.1f}\\%, {np.percentile(all_vr, 75):+.1f}\\%] \\\\",
        f"Range & [{np.min(all_vr):+.1f}\\%, {np.max(all_vr):+.1f}\\%] \\\\",
        f"Mean Cohen's $d$ & {np.mean(all_d):.2f} \\\\",
        f"Positive (ens.\\ prox.\\ helps) & {n_pos}/{N} ({100*n_pos/N:.0f}\\%) \\\\",
        f"Mean val$\\to$test $\\rho$ & {np.mean(all_rho):.2f} \\\\",
        f"Significant transfer ($p<0.05$) & {n_sig_raw}/{N} ({100*n_sig_raw/N:.0f}\\%) \\\\",
        f"Significant (FDR corrected) & {n_sig_fdr}/{N} ({100*n_sig_fdr/N:.0f}\\%) \\\\",
        f"Mean accuracy cost & {np.mean(all_gap):+.3f} \\\\",
        f"Disagree vs.\\ var.\\ red.\\ ($\\rho$) & {stats.spearmanr(all_dis, all_vr)[0]:+.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (cfg.tabs_dir / "tab_summary.tex").write_text("\n".join(lines))

    return {
        "N": N,
        "mean_vr": np.mean(all_vr),
        "median_vr": np.median(all_vr),
        "min_vr": np.min(all_vr),
        "max_vr": np.max(all_vr),
        "mean_d": np.mean(all_d),
        "n_pos": n_pos,
        "mean_rho": np.mean(all_rho),
        "n_sig": n_sig_raw,
        "n_sig_fdr": n_sig_fdr,
        "mean_gap": np.mean(all_gap),
        "dis_vr_rho": stats.spearmanr(all_dis, all_vr)[0],
    }


def make_tab_baseline_comparison(cfg: Config, factorial_results: dict[str, list[dict[str, Any]]]) -> None:
    """Table comparing all selection criteria."""
    criteria = ["best_val", "ens_prox", "random", "ece", "max_entropy", "median_disagree", "weight_avg"]
    criteria_labels = {
        "best_val": "Best Val Acc",
        "ens_prox": "Ens. Proximity",
        "random": "Random",
        "ece": "Min ECE",
        "max_entropy": "Max Entropy",
        "median_disagree": "Med. Disagree",
        "weight_avg": "Weight Avg",
    }

    all_vars: dict[str, list[float]] = {c: [] for c in criteria}
    all_accs: dict[str, list[float]] = {c: [] for c in criteria}

    for fname, data in factorial_results.items():
        for d in data:
            if "baselines" in d:
                for c in criteria:
                    if c in d["baselines"]:
                        all_vars[c].append(d["baselines"][c]["pred_var"])
                        all_accs[c].append(d["baselines"][c]["acc_mean"])

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Criterion & Mean Var & Var Reduction & Mean Acc & Acc Gap \\",
        r"\midrule",
    ]

    bv_var_mean = np.mean(all_vars["best_val"]) if all_vars["best_val"] else 0
    bv_acc_mean = np.mean(all_accs["best_val"]) if all_accs["best_val"] else 0

    for c in criteria:
        if not all_vars[c]:
            continue
        var_mean = np.mean(all_vars[c])
        acc_mean = np.mean(all_accs[c])
        var_red = (bv_var_mean - var_mean) / bv_var_mean * 100 if bv_var_mean > 1e-15 else 0
        acc_gap = bv_acc_mean - acc_mean

        label = criteria_labels[c]
        if c == "ens_prox":
            label = f"\\textbf{{{label}}}"
            var_str = f"\\textbf{{{var_mean:.4f}}}"
            vr_str = f"\\textbf{{{var_red:+.1f}\\%}}"
        else:
            var_str = f"{var_mean:.4f}"
            vr_str = f"{var_red:+.1f}\\%"

        lines.append(f"{label} & {var_str} & {vr_str} & {acc_mean:.3f} & {acc_gap:+.3f} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    (cfg.tabs_dir / "tab_baseline_comparison.tex").write_text("\n".join(lines))


def make_tab_robustness(cfg: Config, robustness_results: dict[str, list[dict[str, Any]]]) -> None:
    """Table summarizing robustness checks."""
    lines = [
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Check & Setting & Var Reduction & 95\% CI \\",
        r"\midrule",
    ]

    for check_name, label in [
        ("rashomon_eps", "Rashomon $\\varepsilon$"),
        ("divergence", "Divergence"),
        ("optimizer", "Optimizer"),
        ("learning_rate", "Learning Rate"),
    ]:
        data = robustness_results.get(check_name, [])
        for i, d in enumerate(data):
            if check_name == "rashomon_eps":
                setting = f"$\\varepsilon = {d['eps']}$"
            elif check_name == "divergence":
                setting = d["divergence"].upper()
            elif check_name == "optimizer":
                setting = d["optimizer"].upper()
            elif check_name == "learning_rate":
                setting = f"lr = {d['lr']}"
            else:
                setting = str(i)

            vr = d["var_red"]
            ci = d.get("var_red_ci", [vr, vr])
            ci_str = f"[{ci[0]:+.1f}, {ci[1]:+.1f}]"

            if i == 0:
                lines.append(f"{label} & {setting} & {vr:+.1f}\\% & {ci_str} \\\\")
            else:
                lines.append(f" & {setting} & {vr:+.1f}\\% & {ci_str} \\\\")

        if data:
            lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (cfg.tabs_dir / "tab_robustness.tex").write_text("\n".join(lines))


def make_tab_architectures(cfg: Config, arch_results: dict[str, dict[str, Any]]) -> None:
    """Table comparing architecture results."""
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Architecture & Var Reduction & Best Val Var & Ens Prox Var \\",
        r"\midrule",
    ]

    arch_labels = {
        "mlp_standard": "MLP [64, 32]",
        "mlp_deep": "MLP [64, 256, 128, 64]",
    }

    for arch_key, label in arch_labels.items():
        if arch_key in arch_results:
            r = arch_results[arch_key]
            vr = r.get("var_red", 0)
            bv_var = r.get("bv_var", 0)
            ep_var = r.get("ep_var", 0)
            lines.append(f"{label} & {vr:+.1f}\\% & {bv_var:.4f} & {ep_var:.4f} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    (cfg.tabs_dir / "tab_architectures.tex").write_text("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible experiments for 'Selecting for Stability'",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run reduced-scale experiments for testing"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000, help="Number of bootstrap samples"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Output directory for figures and tables",
    )
    parser.add_argument(
        "--skip-robustness", action="store_true", help="Skip robustness experiments"
    )
    parser.add_argument(
        "--skip-architectures", action="store_true", help="Skip architecture experiments"
    )
    return parser.parse_args()


def main() -> None:
    import time

    args = parse_args()
    cfg = Config(
        quick=args.quick,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        output_dir=args.output_dir,
    )

    t0 = time.time()
    log.info(f"Mode: {'QUICK' if cfg.quick else 'FULL'}")

    # Core experiments
    decomp = experiment_decomposition(cfg)
    benchmarks = experiment_benchmarks(cfg)
    factorial = experiment_factorial(cfg)
    grid, flips, seps = experiment_grid(cfg)

    # Extended experiments
    robustness = {}
    if not args.skip_robustness:
        robustness = experiment_robustness(cfg)

    arch_results = {}
    if not args.skip_architectures:
        arch_results = experiment_architectures(cfg)

    # Figures
    log.info("Generating figures")
    make_fig_factorial(cfg, factorial)
    log.info("  fig_factorial.pdf")
    make_fig_disagreement(cfg, factorial)
    log.info("  fig_disagreement.pdf")
    make_fig_heatmap_transfer(cfg, grid, flips, seps, benchmarks)
    log.info("  fig_heatmap_transfer.pdf")
    make_fig_baselines(cfg, factorial)
    log.info("  fig_baselines.pdf")
    if robustness:
        make_fig_robustness(cfg, robustness)
        log.info("  fig_robustness.pdf")

    # Tables
    log.info("Generating tables")
    make_tab_transfer(cfg, benchmarks)
    log.info("  tab_transfer.tex")
    summary = make_tab_summary(cfg, factorial)
    log.info("  tab_summary.tex")
    make_tab_baseline_comparison(cfg, factorial)
    log.info("  tab_baseline_comparison.tex")
    if robustness:
        make_tab_robustness(cfg, robustness)
        log.info("  tab_robustness.tex")
    if arch_results:
        make_tab_architectures(cfg, arch_results)
        log.info("  tab_architectures.tex")

    # Save all results as JSON
    all_out = {
        "decomposition": decomp,
        "benchmarks": benchmarks,
        "factorial": factorial,
        "grid": {f"{k[0]},{k[1]}": v for k, v in grid.items()},
        "robustness": robustness,
        "architectures": arch_results,
        "summary": summary,
    }
    results_path = cfg.output_dir / "scripts" / "results.json"
    results_path.write_text(json.dumps(all_out, indent=2, default=str))

    elapsed = time.time() - t0
    log.info(f"Done in {elapsed:.0f}s. Outputs in {cfg.figs_dir} and {cfg.tabs_dir}")


if __name__ == "__main__":
    main()

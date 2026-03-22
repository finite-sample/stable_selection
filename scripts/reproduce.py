#!/usr/bin/env python3
"""
Reproducible pipeline for:
  "Selecting for Stability: Ensemble Proximity as a Criterion
   for Lower Out-of-Sample Prediction Variance"

Usage:
  python reproduce.py          # run everything
  python reproduce.py --quick  # reduced scale for testing

Outputs:
  ../figs/fig_decomposition.pdf
  ../figs/fig_factorial.pdf
  ../figs/fig_heatmap_transfer.pdf
  ../figs/fig_disagreement.pdf
  ../tabs/tab_transfer.tex
  ../tabs/tab_summary.tex
"""

import numpy as np
from sklearn.datasets import (load_digits, load_wine, load_breast_cancer,
                               make_classification)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, time, json, warnings
warnings.filterwarnings('ignore')

QUICK = '--quick' in sys.argv
SEED = 42
os.makedirs('../figs', exist_ok=True)
os.makedirs('../tabs', exist_ok=True)

# Matplotlib defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


# ═══════════════════════════════════════════════════════════════════════
# MLP
# ═══════════════════════════════════════════════════════════════════════

def relu(x): return np.maximum(0, x)
def relu_grad(x): return (x > 0).astype(float)
def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

class MLP:
    def __init__(self, sizes, seed):
        rng = np.random.RandomState(seed)
        self.W = [rng.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0/sizes[i])
                   for i in range(len(sizes)-1)]
        self.b = [np.zeros(sizes[i+1]) for i in range(len(sizes)-1)]

    def forward(self, X):
        self.a, self.z = [X], []
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b; self.z.append(z)
            h = relu(z) if i < len(self.W) - 1 else softmax(z)
            self.a.append(h)
        return h

    def backward(self, yoh):
        gw, gb = [], []; n = yoh.shape[0]
        d = (self.a[-1] - yoh) / n
        for i in range(len(self.W)-1, -1, -1):
            gw.insert(0, self.a[i].T @ d)
            gb.insert(0, d.sum(0))
            if i > 0:
                d = (d @ self.W[i].T) * relu_grad(self.z[i-1])
        return gw, gb

    def predict_proba(self, X): return self.forward(X)

def train_sgd(m, X, yoh, ep, lr, bs, rng):
    n = X.shape[0]
    for _ in range(ep):
        idx = rng.permutation(n)
        for s in range(0, n, bs):
            bi = idx[s:s+bs]; m.forward(X[bi])
            gw, gb = m.backward(yoh[bi])
            for j in range(len(m.W)):
                m.W[j] -= lr * gw[j]
                m.b[j] -= lr * gb[j]

def onehot(y, k):
    oh = np.zeros((len(y), k)); oh[np.arange(len(y)), y] = 1; return oh

def split_scale(X, y, seed=SEED):
    Xtv, Xte, ytv, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y)
    Xtr, Xv, ytr, yv = train_test_split(
        Xtv, ytv, test_size=0.2, random_state=seed, stratify=ytv)
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xv), sc.transform(Xte), ytr, yv, yte

def arch_for(nf, nc):
    h = max(32, min(96, nf * 2))
    return [nf, h, h // 2, nc]


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Variance Decomposition
# ═══════════════════════════════════════════════════════════════════════

def experiment_decomposition():
    print("\n[Exp 1] Variance Decomposition")
    d = load_digits(); X, y = d.data, d.target
    Xtr, Xv, Xte, ytr, yv, yte = split_scale(X, y)
    yoh = onehot(ytr, 10)
    arch = [64, 128, 64, 10]
    K = 20 if QUICK else 30

    results = {}
    for label, seed_fn in [
        ('Init only',    lambda k: (k, 0)),
        ('Shuffle only', lambda k: (0, k)),
        ('Both',         lambda k: (k, k+1000)),
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
            'mean_var': float(pev.mean()),
            'acc_mean': float(np.mean(accs)),
            'acc_std': float(np.std(accs)),
        }
        print(f"  {label:15s}: var={pev.mean():.7f}, acc={np.mean(accs):.4f}")

    ratio = results['Init only']['mean_var'] / max(results['Shuffle only']['mean_var'], 1e-15)
    results['ratio'] = float(ratio)
    print(f"  Ratio: {ratio:.0f}x")

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.8),
                                    gridspec_kw={'width_ratios': [1.2, 1]})
    labs = ['Init\nonly', 'Shuffle\nonly', 'Both']
    vals = [results[l]['mean_var'] for l in ['Init only', 'Shuffle only', 'Both']]
    cols = ['#4C72B0', '#DD8452', '#55A868']

    ax1.bar(labs, vals, color=cols, alpha=0.85, edgecolor='black', linewidth=0.4, width=0.55)
    ax1.set_ylabel('Mean prediction variance')
    ax1.set_title('(a) Linear scale')
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax1.annotate(f'$\\approx${ratio:.0f}$\\times$', xy=(1, vals[1]),
                 xytext=(1.6, vals[0]*0.5),
                 arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.2),
                 fontsize=9, color='#C44E52', fontweight='bold')

    ax2.bar(labs, vals, color=cols, alpha=0.85, edgecolor='black', linewidth=0.4, width=0.55)
    ax2.set_yscale('log')
    ax2.set_title('(b) Log scale')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('../figs/fig_decomposition.pdf')
    plt.close()
    return results


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Single-cell evaluation (used by factorial)
# ═══════════════════════════════════════════════════════════════════════

def evaluate_cell(dgp_params, K=None, M=None, epochs=None):
    if K is None: K = 10 if QUICK else 12
    if M is None: M = 5 if QUICK else 6
    if epochs is None: epochs = 30 if QUICK else 30

    X, y = make_classification(**dgp_params)
    nc = len(np.unique(y))
    Xtr, Xv, Xte, ytr, yv, yte = split_scale(X, y)
    yoh = onehot(ytr, nc)
    arch = arch_for(Xtr.shape[1], nc)
    n_test = len(yte)

    # --- Val -> Test transfer ---
    pool_vp, pool_tp = [], []
    for k in range(K):
        m = MLP(arch, seed=k)
        train_sgd(m, Xtr, yoh, epochs, 0.01, 32, np.random.RandomState(k+500))
        pool_vp.append(m.predict_proba(Xv))
        pool_tp.append(m.predict_proba(Xte))

    ens_v = np.stack(pool_vp).mean(0)
    ens_t = np.stack(pool_tp).mean(0)
    val_kls, test_kls = [], []
    for k in range(K):
        pv = np.clip(pool_vp[k], 1e-10, 1); qv = np.clip(ens_v, 1e-10, 1)
        val_kls.append((pv * np.log(pv/qv)).sum(1).mean())
        pt = np.clip(pool_tp[k], 1e-10, 1); qt = np.clip(ens_t, 1e-10, 1)
        test_kls.append((pt * np.log(pt/qt)).sum(1).mean())

    rho, pval = stats.spearmanr(val_kls, test_kls)

    # --- Meta-stability ---
    sel_preds = {c: np.zeros((M, n_test, nc))
                 for c in ['best_val', 'ens_prox']}
    sel_accs = {c: [] for c in ['best_val', 'ens_prox']}

    for rep in range(M):
        base = (rep+1) * 10000
        pool2 = []
        for k in range(K):
            m = MLP(arch, seed=base+k)
            train_sgd(m, Xtr, yoh, epochs, 0.01, 32,
                      np.random.RandomState(base+k+5000))
            vp = m.predict_proba(Xv); tp = m.predict_proba(Xte)
            pool2.append({
                'vp': vp, 'tp': tp,
                'va': accuracy_score(yv, vp.argmax(1)),
                'l2': None,
            })

        ev = np.stack([m['vp'] for m in pool2]).mean(0)
        for m in pool2:
            m['l2'] = np.mean(np.sum((m['vp'] - ev)**2, axis=1))

        bva = max(m['va'] for m in pool2)
        cands = [m for m in pool2 if m['va'] >= bva - 0.03]
        if len(cands) < 3: cands = pool2

        sel_bv = max(cands, key=lambda m: m['va'])
        sel_ep = min(cands, key=lambda m: m['l2'])

        for c, sel in [('best_val', sel_bv), ('ens_prox', sel_ep)]:
            sel_preds[c][rep] = sel['tp']
            sel_accs[c].append(accuracy_score(yte, sel['tp'].argmax(1)))

    res = {}
    for c in ['best_val', 'ens_prox']:
        preds = sel_preds[c]
        pv = preds.var(axis=0).mean(axis=1).mean()
        hard = preds.argmax(axis=2)
        dis = np.mean([len(np.unique(hard[:, i])) > 1 for i in range(n_test)])
        res[c] = {'pred_var': pv, 'disagree': dis,
                   'acc_mean': np.mean(sel_accs[c]),
                   'acc_std': np.std(sel_accs[c])}

    bv_var = res['best_val']['pred_var']
    ep_var = res['ens_prox']['pred_var']
    vr = (bv_var - ep_var) / bv_var * 100 if bv_var > 1e-15 else 0.0

    return {
        'rho': float(rho), 'pval': float(pval),
        'var_red': float(vr),
        'bv_var': float(bv_var), 'ep_var': float(ep_var),
        'disagree': float(res['best_val']['disagree']),
        'bv_acc': float(res['best_val']['acc_mean']),
        'ep_acc': float(res['ens_prox']['acc_mean']),
        'acc_gap': float(res['best_val']['acc_mean'] - res['ens_prox']['acc_mean']),
    }


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Factorial simulation
# ═══════════════════════════════════════════════════════════════════════

def experiment_factorial():
    print("\n[Exp 3] Factorial Simulation")
    base = dict(n_samples=1000, n_features=30, n_informative=10,
                n_redundant=10, n_classes=4, n_clusters_per_class=2,
                flip_y=0.05, class_sep=1.0, random_state=SEED)

    factors = {
        'Class separation': ('class_sep', [0.3, 0.5, 0.8, 1.0, 1.5, 2.0] if not QUICK
                              else [0.3, 0.8, 1.5], {}),
        'Label noise':      ('flip_y', [0.0, 0.05, 0.10, 0.20], {}),
        'Num. classes':     ('n_classes', [2, 3, 4, 7] if not QUICK
                              else [2, 4, 7],
                             {'n_clusters_per_class': lambda nc: max(1, 3-nc//3)}),
        'Sample size':      ('n_samples', [300, 500, 1000, 2000] if not QUICK
                              else [300, 1000, 3000], {}),
        'Redundancy':       ('n_redundant', [0, 5, 10, 20] if not QUICK
                              else [0, 10, 20],
                             {'n_features': lambda nr: 10 + nr + 5}),
    }

    all_results = {}
    for fname, (pname, levels, transforms) in factors.items():
        print(f"  Factor: {fname}")
        all_results[fname] = []
        for val in levels:
            params = {**base, pname: val}
            for tname, tfn in transforms.items():
                params[tname] = tfn(val) if callable(tfn) else tfn
            t0 = time.time()
            r = evaluate_cell(params)
            r['level'] = val
            all_results[fname].append(r)
            print(f"    {pname}={val}: var_red={r['var_red']:+.1f}%, "
                  f"rho={r['rho']:+.2f}, dis={r['disagree']:.2f} "
                  f"[{time.time()-t0:.0f}s]")

    return all_results


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: 2D Grid (noise x separation)
# ═══════════════════════════════════════════════════════════════════════

def experiment_grid():
    print("\n[Exp 4] 2D Grid: noise x separation")
    base = dict(n_samples=1000, n_features=30, n_informative=10,
                n_redundant=10, n_classes=4, n_clusters_per_class=2,
                random_state=SEED)

    flips = [0.0, 0.05, 0.10, 0.20]
    seps = [0.3, 0.5, 0.8, 1.0, 1.5] if not QUICK else [0.5, 1.0, 1.5]
    grid = {}
    for f in flips:
        for s in seps:
            params = {**base, 'flip_y': f, 'class_sep': s}
            t0 = time.time()
            K_g = 8 if QUICK else 10
            M_g = 4 if QUICK else 5
            r = evaluate_cell(params, K=K_g, M=M_g)
            grid[(f, s)] = r
            print(f"  flip={f:.2f}, sep={s}: var_red={r['var_red']:+.1f}%, "
                  f"dis={r['disagree']:.2f} [{time.time()-t0:.0f}s]")
    return grid, flips, seps


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Benchmark datasets (val->test transfer)
# ═══════════════════════════════════════════════════════════════════════

def experiment_benchmarks():
    print("\n[Exp 5] Benchmark Datasets: Val->Test Transfer")
    benchmarks = {
        'Digits': load_digits(),
        'Wine': load_wine(),
        'Breast Cancer': load_breast_cancer(),
    }
    K = 15 if QUICK else 20
    ep = 40 if QUICK else 50
    results = {}
    for name, d in benchmarks.items():
        X, y = d.data, d.target; nc = len(np.unique(y))
        Xtr, Xv, Xte, ytr, yv, yte = split_scale(X, y)
        yoh = onehot(ytr, nc); arch = arch_for(Xtr.shape[1], nc)

        pool_vp, pool_tp, pool_va = [], [], []
        for k in range(K):
            m = MLP(arch, seed=k)
            train_sgd(m, Xtr, yoh, ep, 0.01, 32, np.random.RandomState(k+500))
            vp = m.predict_proba(Xv); tp = m.predict_proba(Xte)
            pool_vp.append(vp); pool_tp.append(tp)
            pool_va.append(accuracy_score(yv, vp.argmax(1)))

        ens_v = np.stack(pool_vp).mean(0)
        ens_t = np.stack(pool_tp).mean(0)
        vkl, tkl = [], []
        for k in range(K):
            pv = np.clip(pool_vp[k], 1e-10, 1); qv = np.clip(ens_v, 1e-10, 1)
            vkl.append((pv * np.log(pv/qv)).sum(1).mean())
            pt = np.clip(pool_tp[k], 1e-10, 1); qt = np.clip(ens_t, 1e-10, 1)
            tkl.append((pt * np.log(pt/qt)).sum(1).mean())

        rho, pval = stats.spearmanr(vkl, tkl)
        rho_l2, pval_l2 = stats.spearmanr(
            [np.mean(np.sum((pool_vp[k]-ens_v)**2, 1)) for k in range(K)],
            [np.mean(np.sum((pool_tp[k]-ens_t)**2, 1)) for k in range(K)])

        results[name] = {'rho_kl': rho, 'p_kl': pval,
                          'rho_l2': rho_l2, 'p_l2': pval_l2,
                          'n': len(y), 'p': X.shape[1], 'nc': nc}
        print(f"  {name:15s}: rho_kl={rho:+.2f} (p={pval:.3f}), "
              f"rho_l2={rho_l2:+.2f} (p={pval_l2:.3f})")

    return results


# ═══════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════

def make_fig_factorial(factorial_results):
    """Six-panel factor sweep figure."""
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.5))

    factor_order = ['Class separation', 'Sample size', 'Num. classes',
                    'Redundancy', 'Label noise']
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']

    for idx, (fname, plabel) in enumerate(zip(factor_order, panel_labels)):
        ax = axes.flat[idx]
        data = factorial_results[fname]
        xs = [d['level'] for d in data]
        vr = [d['var_red'] for d in data]
        dis = [d['disagree'] for d in data]

        c1, c2 = '#4C72B0', '#C44E52'
        ax.bar(range(len(xs)), vr,
               color=[c1 if v > 0 else '#B0C4DE' for v in vr],
               alpha=0.75, edgecolor='black', linewidth=0.3, width=0.55)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(x) for x in xs])
        ax.axhline(0, color='gray', linewidth=0.4, linestyle='--')
        ax.set_title(f'{plabel} {fname}', fontsize=9)
        ax.spines['top'].set_visible(False)
        if idx % 3 == 0:
            ax.set_ylabel('Var. reduction (%)')

        ax2 = ax.twinx()
        ax2.plot(range(len(xs)), dis, 'o-', color=c2, lw=1, ms=3, alpha=0.6)
        if idx % 3 == 2:
            ax2.set_ylabel('Disagreement', color=c2, fontsize=8)
        ax2.tick_params(axis='y', colors=c2, labelsize=7)
        ax2.spines['top'].set_visible(False)

    # (f) Scatter: disagreement vs var reduction
    ax = axes.flat[5]
    all_dis, all_vr = [], []
    for fname, data in factorial_results.items():
        for d in data:
            all_dis.append(d['disagree'])
            all_vr.append(d['var_red'])

    ax.scatter(all_dis, all_vr, s=25, c='#4C72B0', alpha=0.7,
               edgecolors='black', linewidths=0.3)
    coef = np.polyfit(all_dis, all_vr, 1)
    xl = np.linspace(min(all_dis)-0.03, max(all_dis)+0.03, 100)
    ax.plot(xl, np.polyval(coef, xl), '--', color='#C44E52', lw=1.2, alpha=0.6)
    r, p = stats.spearmanr(all_dis, all_vr)
    ax.text(0.05, 0.92, f'$\\rho$={r:+.2f}\n$p$={p:.3f}',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='wheat', alpha=0.8))
    ax.axhline(0, color='gray', lw=0.4, ls='--')
    ax.set_xlabel('Baseline disagreement')
    ax.set_ylabel('Var. reduction (%)')
    ax.set_title('(f) Disagreement predicts benefit', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(h_pad=1.0)
    plt.savefig('../figs/fig_factorial.pdf')
    plt.close()
    return r, p


def make_fig_heatmap_transfer(grid, flips, seps, benchmarks):
    """Two-panel: 2D heatmap + transfer correlation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # (a) Heatmap
    mat = np.array([[grid[(f, s)]['var_red'] for s in seps] for f in flips])
    im = ax1.imshow(mat, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=25,
                     origin='lower')
    ax1.set_xticks(range(len(seps)))
    ax1.set_xticklabels([str(s) for s in seps])
    ax1.set_yticks(range(len(flips)))
    ax1.set_yticklabels([str(f) for f in flips])
    ax1.set_xlabel('Class separation')
    ax1.set_ylabel('Label noise')
    ax1.set_title('(a) Variance reduction (\\%)')
    for i in range(len(flips)):
        for j in range(len(seps)):
            v = mat[i, j]
            c = 'white' if abs(v) > 12 else 'black'
            ax1.text(j, i, f'{v:+.0f}', ha='center', va='center',
                     fontsize=7, fontweight='bold', color=c)
    plt.colorbar(im, ax=ax1, shrink=0.85, label='%')

    # (b) Transfer correlations from benchmarks + synthetics
    names, rhos, sig = [], [], []
    for n, r in benchmarks.items():
        names.append(n); rhos.append(r['rho_kl']); sig.append(r['p_kl'] < 0.05)
    # Add synthetic cells with extreme values
    for fname, data_list in [('Sep=0.3', [grid.get((0.05, 0.3))]),
                              ('Sep=1.5', [grid.get((0.05, 1.5))])]:
        if data_list[0]:
            names.append(fname)
            rhos.append(data_list[0]['rho'])
            sig.append(data_list[0]['pval'] < 0.05)

    colors = ['#4C72B0' if s else '#AAAAAA' for s in sig]
    ax2.barh(range(len(names)), rhos, color=colors, alpha=0.75,
             edgecolor='black', linewidth=0.3)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel('Spearman $\\rho$ (val$\\to$test KL)')
    ax2.set_title('(b) Transfer correlation')
    ax2.axvline(0, color='black', lw=0.4)
    for i, (r, s) in enumerate(zip(rhos, sig)):
        star = '*' if s else ''
        ax2.text(max(r+0.02, 0.02), i, f'{r:.2f}{star}',
                 va='center', fontsize=7.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('../figs/fig_heatmap_transfer.pdf')
    plt.close()


def make_fig_disagreement(factorial_results):
    """Standalone scatter: disagreement vs variance reduction."""
    fig, ax = plt.subplots(figsize=(3.5, 3))
    all_dis, all_vr = [], []
    for fname, data in factorial_results.items():
        for d in data:
            all_dis.append(d['disagree'])
            all_vr.append(d['var_red'])

    ax.scatter(all_dis, all_vr, s=35, c='#4C72B0', alpha=0.7,
               edgecolors='black', linewidths=0.4)
    coef = np.polyfit(all_dis, all_vr, 1)
    xl = np.linspace(0.1, 1.0, 100)
    ax.plot(xl, np.polyval(coef, xl), '--', color='#C44E52', lw=1.5, alpha=0.6)
    r, p = stats.spearmanr(all_dis, all_vr)
    ax.text(0.05, 0.95, f'$\\rho = {r:+.2f}$, $p = {p:.3f}$',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.8))
    ax.axhline(0, color='gray', lw=0.4, ls='--')
    ax.set_xlabel('Baseline prediction disagreement rate')
    ax.set_ylabel('Variance reduction (\\%)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('../figs/fig_disagreement.pdf')
    plt.close()
    return r, p


# ═══════════════════════════════════════════════════════════════════════
# TABLES (LaTeX fragments)
# ═══════════════════════════════════════════════════════════════════════

def make_tab_transfer(benchmarks):
    lines = [
        r'\begin{tabular}{lccccc}',
        r'\toprule',
        r'Dataset & $n$ & $p$ & Classes & $\rho_{\mathrm{KL}}$ & $\rho_{L_2}$ \\',
        r'\midrule',
    ]
    for name, r in benchmarks.items():
        rho_kl = f"\\textbf{{{r['rho_kl']:+.2f}}}" if r['p_kl'] < 0.05 else f"{r['rho_kl']:+.2f}"
        rho_l2 = f"\\textbf{{{r['rho_l2']:+.2f}}}" if r['p_l2'] < 0.05 else f"{r['rho_l2']:+.2f}"
        lines.append(f"{name} & {r['n']} & {r['p']} & {r['nc']} & {rho_kl} & {rho_l2} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    with open('../tabs/tab_transfer.tex', 'w') as f:
        f.write('\n'.join(lines))


def make_tab_summary(factorial_results):
    """Summary statistics across all factorial cells."""
    all_vr, all_rho, all_gap, all_dis = [], [], [], []
    for data in factorial_results.values():
        for d in data:
            all_vr.append(d['var_red'])
            all_rho.append(d['rho'])
            all_gap.append(d['acc_gap'])
            all_dis.append(d['disagree'])

    n_pos = sum(1 for v in all_vr if v > 0)
    n_sig = sum(1 for d in
                [d for data in factorial_results.values() for d in data]
                if d['pval'] < 0.05)
    N = len(all_vr)

    lines = [
        r'\begin{tabular}{lc}',
        r'\toprule',
        r'Statistic & Value \\',
        r'\midrule',
        f'DGP configurations & {N} \\\\',
        f'Mean variance reduction & {np.mean(all_vr):+.1f}\\% \\\\',
        f'Median variance reduction & {np.median(all_vr):+.1f}\\% \\\\',
        f'Range & [{np.min(all_vr):+.1f}\\%, {np.max(all_vr):+.1f}\\%] \\\\',
        f'Positive (ens.\\ prox.\\ helps) & {n_pos}/{N} ({100*n_pos/N:.0f}\\%) \\\\',
        f'Mean val$\\to$test $\\rho$ & {np.mean(all_rho):.2f} \\\\',
        f'Significant transfer ($p<0.05$) & {n_sig}/{N} ({100*n_sig/N:.0f}\\%) \\\\',
        f'Mean accuracy cost & {np.mean(all_gap):+.3f} \\\\',
        f'Disagree vs.\\ var.\\ red.\\ ($\\rho$) & {stats.spearmanr(all_dis, all_vr)[0]:+.2f} \\\\',
        r'\bottomrule',
        r'\end{tabular}',
    ]
    with open('../tabs/tab_summary.tex', 'w') as f:
        f.write('\n'.join(lines))

    return {
        'N': N, 'mean_vr': np.mean(all_vr), 'median_vr': np.median(all_vr),
        'min_vr': np.min(all_vr), 'max_vr': np.max(all_vr),
        'n_pos': n_pos, 'mean_rho': np.mean(all_rho), 'n_sig': n_sig,
        'mean_gap': np.mean(all_gap),
        'dis_vr_rho': stats.spearmanr(all_dis, all_vr)[0],
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()
    print(f"Mode: {'QUICK' if QUICK else 'FULL'}")

    decomp = experiment_decomposition()
    benchmarks = experiment_benchmarks()
    factorial = experiment_factorial()
    grid, flips, seps = experiment_grid()

    print("\n[Figures]")
    make_fig_factorial(factorial)
    print("  fig_factorial.pdf")
    rho_main, p_main = make_fig_disagreement(factorial)
    print("  fig_disagreement.pdf")
    make_fig_heatmap_transfer(grid, flips, seps, benchmarks)
    print("  fig_heatmap_transfer.pdf")
    # decomposition already saved

    print("\n[Tables]")
    make_tab_transfer(benchmarks)
    print("  tab_transfer.tex")
    summary = make_tab_summary(factorial)
    print("  tab_summary.tex")

    # Save all results as JSON for reference
    all_out = {
        'decomposition': decomp,
        'benchmarks': {k: v for k, v in benchmarks.items()},
        'factorial': {fname: data for fname, data in factorial.items()},
        'summary': summary,
    }
    with open('results.json', 'w') as f:
        json.dump(all_out, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. All outputs in figures/ and tables/.")

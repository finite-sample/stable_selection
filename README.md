## Selecting for Stability: Ensemble proximity as a criterion for lower out-of-sample prediction variance.

Neural networks trained with different random seeds produce models with similar accuracy but systematically different predictions on individual examples. This repository contains code to reproduce the experiments showing that selecting the model whose predictions most closely resemble the ensemble mean reduces out-of-sample prediction variance.

## Results

### Key Numbers

- Initialization variance is **561×** larger than shuffle-order variance
- Mean variance reduction: **+5.3%** (median +4.8%)
- Works in **88%** of cases (14/16 DGP configurations)
- Accuracy cost: **~0** (mean +0.01 pp)
- Val→test transfer: mean ρ = **0.52**

### When It Helps

| Condition | Variance Reduction |
|-----------|-------------------|
| Hard problems (class sep 0.3) | +17.4% |
| Small samples (n=300) | +15.3% |
| Large samples (n=3000) | +13.5% |
| Many classes (7) | +7.9% |

### When It Doesn't Help

- Easy binary problems: -6.8%
- High separation + high noise

### Diagnostic

Prediction disagreement predicts benefit (ρ=0.51). If models disagree on >40% of examples, use ensemble proximity.

## Requirements

- Python 3.7+
- numpy
- scikit-learn
- scipy
- matplotlib

Install with:
```bash
pip install numpy scikit-learn scipy matplotlib
```

## Usage

Run the full reproduction pipeline:
```bash
cd scripts
python reproduce.py
```

Run a quick version for testing (~1 minute):
```bash
cd scripts
python reproduce.py --quick
```

## Outputs

**Figures** (in `figs/`):
- `fig_decomposition.pdf` - Variance decomposition showing initialization dominates shuffle-order variance
- `fig_factorial.pdf` - Factor sweeps showing when ensemble proximity helps
- `fig_heatmap_transfer.pdf` - 2D grid of noise × separation and transfer correlations
- `fig_disagreement.pdf` - Relationship between disagreement and variance reduction

**Tables** (in `tabs/`):
- `tab_transfer.tex` - Val→test proximity transfer on benchmark datasets
- `tab_summary.tex` - Summary statistics across factorial DGP configurations

**Data**:
- `scripts/results.json` - All numerical results in JSON format

## Compiling the Paper

```bash
cd ms
pdflatex selecting_for_stability.tex
bibtex selecting_for_stability
pdflatex selecting_for_stability.tex
pdflatex selecting_for_stability.tex
```

## Citation

```bibtex
@article{sood2026selecting,
  title={Selecting for Stability: Ensemble Proximity as a Criterion for Lower Out-of-Sample Prediction Variance},
  author={Sood, Gaurav},
  year={2026}
}
```

# Adaptive Momentum Gradient Descent for Regularized Poisson Regression

**A Novel Optimization Algorithm with Adaptive Soft-Thresholding**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Research Paper](https://img.shields.io/badge/paper-published-brightgreen.svg)](https://github.com/elbakari01/amgd_poisson_regression)

## 📋 Overview

This repository contains the complete implementation and experimental validation of **Adaptive Momentum Gradient Descent (AMGD)**, a novel optimization algorithm for regularized Poisson regression developed by Ibrahim Bakari and M. Revan Özkale from Çukurova University.

AMGD addresses critical limitations of existing optimizers (AdaGrad's rapid learning rate decay, Adam's gradient instability in sparse data) by integrating adaptive learning rates, momentum updates, gradient clipping, and **adaptive soft-thresholding** into a unified framework.

## 🎯 Key Research Contributions

### Novel AMGD Algorithm Features:
- **Adaptive Soft-Thresholding**: `βⱼ = sign(βⱼ) × max(|βⱼ| - αₜλ₁/(|βⱼ| + ε), 0)`
- **Dual-Level Gradient Clipping**: Prevents numerical instability from exponential link function
- **Momentum with Bias Correction**: Enhanced convergence properties
- **Adaptive Learning Rate Decay**: `αₜ = α/(1 + ηt)` for guaranteed convergence

### Theoretical Guarantees:
- **Convergence Rate**: O(1/√T) after T iterations under convexity
- **Feature Selection Optimality**: Proven oracle properties for L1 regularization
- **Numerical Stability**: Robust to heavy-tailed gradients in sparse count data

## 📊 Performance Results

### Experimental Dataset:
- **Ecological Health Dataset**: 61,345 observations, 17 features
- **Target**: Biodiversity Index (count data, Poisson-distributed)
- **Features**: Environmental indicators (air quality, soil characteristics, water metrics)

### Performance Comparison:
| Optimizer | MAE ↓  | RMSE ↓ | MPD ↓  | Sparsity ↑ | Runtime (s) |
|-----------|--------|--------|--------|------------|-------------|
| **AMGD**  | **2.985** | **3.873** | **2.188** | **29.29%** | **0.002** |
| Adam      | 3.081  | 3.983  | 2.225  | 11.76%     | 0.004 |
| AdaGrad   | 6.862  | 7.579  | 10.965 | 5.00%      | 0.745 |
| GLMnet    | 9.007  | 9.551  | 28.848 | 52.93%     | 0.040 |

### Key Improvements:
- **56.6% better MAE** compared to AdaGrad
- **2.7% better MAE** compared to Adam  
- **Superior sparsity induction** (29.29% vs Adam's 11.76%)
- **Fastest computational efficiency** (0.002s runtime)

## 🔬 Statistical Significance

All improvements are statistically significant with **p < 0.0001**:

| Comparison | MAE | RMSE | Mean Deviance | Effect Size (Cohen's d) |
|------------|-----|------|---------------|------------------------|
| AMGD vs Adam | p<0.0001 | p<0.0001 | p<0.0001 | d=-1.33* |
| AMGD vs AdaGrad | p<0.0001 | p<0.0001 | p<0.0001 | d=-9.58* |
| AMGD vs GLMnet | p<0.0001 | p<0.0001 | p<0.0001 | d=-10.78* |

*Large effect sizes indicate substantial practical significance

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/amgd-poisson-regression-research.git
cd amgd-poisson-regression-research
```

### 2. Setup Environment
```bash
pip install -r requirements.txt
```

### 3. Run Complete Analysis
```bash
python amgd_implementation.py
```

### 4. Quick Test (5 minutes)
```bash
python amgd_implementation.py --quick --optimizers AMGD Adam
```

## 🔍 Algorithm Details

### Mathematical Formulation

**1. Poisson Log-Likelihood:**
```
L(β) = -∑[yᵢ log(μᵢ) - μᵢ - log(yᵢ!)]
where μᵢ = exp(xᵢᵀβ)
```

**2. AMGD Update Rules:**
```
# Momentum Updates
mₜ = ζ₁mₜ₋₁ + (1-ζ₁)∇L
vₜ = ζ₂vₜ₋₁ + (1-ζ₂)[∇L]²

# Bias Correction  
m̂ₜ = mₜ/(1-ζ₁ᵗ), v̂ₜ = vₜ/(1-ζ₂ᵗ)

# Parameter Update
β = β - αₜm̂ₜ/(√v̂ₜ + ε)

# Adaptive Soft-Thresholding
βⱼ = sign(βⱼ) × max(|βⱼ| - αₜλ₁/(|βⱼ| + ε), 0)
```

### Key Innovations:

**1. Adaptive Soft-Thresholding:**
- Traditional: Fixed threshold λ
- AMGD: Adaptive threshold `αₜλ₁/(|βⱼ| + ε)`
- **Benefit**: Large coefficients preserved, small coefficients aggressively shrunk

**2. Gradient Clipping:**
- **Element-wise clipping**: `clip(gⱼ) = max(-T, min(gⱼ, T))`
- **Linear predictor clipping**: Prevents exponential overflow
- **Critical for Poisson regression** due to exponential link function

**3. Convergence Guarantees:**
- **Rate**: O(1/√T) convergence to optimal solution
- **Conditions**: Diminishing step sizes `∑αₑ = ∞, ∑αₜ² < ∞`
- **Stability**: Robust to sparse, high-dimensional data

## 📁 Repository Structure

```
amgd-poisson-regression-research/
├── README.md                          # This file
├── amgd_implementation.py              # Complete research implementation
├── requirements.txt                    # Exact dependency versions
├── data/
│   ├── ecological_health_dataset.csv   # Biodiversity dataset (61,345 obs)
│   └── README.md                       # Data description
├── results/
│   ├── figures/                        # Generated plots and visualizations
│   ├── tables/                         # Numerical results (CSV format)
│   └── reports/                        # Experimental summaries
├── docs/
│   ├── algorithm_explanation.md        # Mathematical details
│   ├── experimental_setup.md           # Complete methodology
│   └── theoretical_analysis.md         # Convergence proofs
└── reproduction/
    ├── reproduction_guide.md           # Step-by-step reproduction
    └── troubleshooting.md              # Common issues and solutions
```

## 🔬 Experimental Design

### Cross-Validation Setup:
- **Strategy**: 5-fold cross-validation  
- **Data Split**: 70% training, 15% validation, 15% test
- **Hyperparameter Search**: 50 λ values (10⁻³ to 10¹)
- **Regularization**: L1 (Lasso) and ElasticNet

### Performance Metrics:
1. **Mean Absolute Error (MAE)** - Primary metric
2. **Root Mean Squared Error (RMSE)** - Secondary metric  
3. **Mean Poisson Deviance (MPD)** - Distributional fit quality
4. **Sparsity** - Feature selection capability
5. **Runtime** - Computational efficiency

### Statistical Validation:
- **Bootstrap Analysis**: 1000 resamples for confidence intervals
- **Significance Testing**: Paired t-tests with Bonferroni correction
- **Effect Size**: Cohen's d for practical significance
- **Feature Selection Stability**: Across bootstrap samples

## 📈 Real-World Application: Ecological Modeling

### Dataset Description:
- **Domain**: Environmental science and biodiversity conservation
- **Size**: 61,345 observations across diverse ecosystems
- **Features**: 
  - **Environmental**: Temperature, humidity, precipitation, vegetation coverage
  - **Pollution**: Air quality index, PM2.5, soil/water contamination
  - **Categorical**: Pollution levels, ecological health classifications
- **Target**: Biodiversity Index (species count per ecosystem)

### Scientific Impact:
- **Conservation Planning**: Identify key environmental drivers
- **Policy Development**: Quantify pollution impact on biodiversity  
- **Predictive Modeling**: Early warning systems for ecosystem degradation
- **Resource Allocation**: Prioritize conservation efforts

### Key Findings:
- **Ecological health indicators** most predictive (100% selection probability)
- **Pollution levels** consistently selected across bootstrap samples
- **Environmental variables** show moderate selection frequencies (53-70%)
- **Sparse models** (29% features) maintain predictive accuracy

## 🔄 Reproducibility

### Complete Reproducibility Package:
- **Fixed Random Seed**: 42 throughout all experiments
- **Exact Dependencies**: Pinned versions in requirements.txt
- **Statistical Validation**: Bootstrap confidence intervals
- **Cross-Platform**: Tested on Windows/macOS/Linux

### Expected Runtime:
- **Quick Test**: 2-5 minutes
- **Full Analysis**: 1-2 hours
- **Statistical Validation**: Additional 30 minutes

### Validation Metrics:
Your results should show:
- AMGD outperforming other methods on MAE/RMSE
- Statistical significance p < 0.05 for key comparisons
- Sparsity levels between 20-35% for AMGD
- Convergence within 10-20% of maximum iterations

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{bakari2024amgd,
  title={Adaptive Momentum Gradient Descent: A New Algorithm in Regularized Poisson Regression},
  author={Bakari, Ibrahim and Özkale, M. Revan},
  journal={[Journal Name]},
  year={2024},
  publisher={[Publisher]},
}
```

## 🔗 Related Work

- **Kingma & Ba (2014)**: Adam optimizer foundation
- **Duchi et al. (2011)**: AdaGrad adaptive learning rates  
- **Friedman et al. (2010)**: GLMnet coordinate descent
- **Zou (2006)**: Adaptive Lasso oracle properties
- **Tibshirani (1996)**: Original Lasso formulation

## 👥 Authors

**Ibrahim Bakari** (Corresponding Author) 
- Department of Statistics, Faculty of Science and Letters
- Çukurova University, Adana, 01330, Türkiye
- Email: 2020913072@ogr.cu.edu.tr/acbrhmbakari@gmail.com

**M. Revan Özkale** 
- Department of Statistics, Faculty of Science and Letters  
- Çukurova University, Adana, 01330, Türkiye
- Email: mrevan@cu.edu.tr

## 🤝 Contributing

We welcome contributions to improve the algorithm or extend the analysis:

1. **Algorithm Improvements**: Stochastic variants, distributed implementations
2. **Theoretical Extensions**: Non-convex convergence analysis
3. **Applications**: Additional datasets and domains
4. **Implementation**: GPU acceleration, online learning variants

## 📄 License

This research implementation is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Çukurova University for research support
- Ecological data providers
- Open source community for optimization libraries
- Reviewers and collaborators for valuable feedback

---

⭐ **Star this repository if you find this research useful!** ⭐

🔄 **Fork it to build upon this work!** 

📧 **Contact us for academic collaborations!**

📊 **Use our algorithm in your research and cite our work!**
# AMGD Algorithm: Mathematical Foundation and Implementation

## Overview

The Adaptive Momentum Gradient Descent (AMGD) algorithm addresses critical limitations in existing optimizers for regularized Poisson regression:

- **AdaGrad**: Suffers from  diminishing learning rates
- **Adam**: Exhibits instability in sparse high-dimensional settings
- **GLMnet**: Encounters numerical issues with extreme predictor values

AMGD integrates four key components into a unified framework: **adaptive learning rates**, **momentum updates**, **gradient clipping**, and **adaptive soft-thresholding**.

## Mathematical Formulation

### 1. Poisson Regression Framework

**Probability Mass Function:**
```
f(yᵢ) = e^(-μᵢ) × μᵢ^yᵢ / yᵢ!,  yᵢ = 0, 1, 2, ...
```

**Canonical Link Function:**
```
g(μᵢ) = log(μᵢ)  ⟹  μᵢ = exp(xᵢᵀβ)
```

**Log-Likelihood Function:**
```
ℓ(β) = ∑ᵢ₌₁ⁿ [yᵢxᵢᵀβ - exp(xᵢᵀβ) - log(yᵢ!)]
```

### 2. Regularized Objective Function

**General Form:**
```
f(β) = -ℓ(β) + λP(β)
```

**L1 Regularization (Lasso):**
```
P(β) = ∥β∥₁ = ∑ⱼ₌₁ᵖ |βⱼ|
```

**ElasticNet Regularization:**
```
P(β) = λ₁∥β∥₁ + (λ₂/2)∥β∥₂²
```

## AMGD Algorithm Components

### 1. Gradient Computation

**Negative Log-Likelihood Gradient:**
```
∇ℓ(β) = Xᵀ(μ - y)
where μᵢ = exp(xᵢᵀβ)
```

**Regularization Gradient:**
- **L1**: Handled in soft-thresholding step
- **ElasticNet**: Add λ₂β to gradient

### 2. Gradient Clipping (Critical Innovation)

**Element-wise Clipping:**
```
clip(gⱼ) = max(-T, min(gⱼ, T))
```

**Linear Predictor Clipping:**
```
linear_pred = clip(Xβ, -20, 20)
```

**Why Essential for Poisson Regression:**
- Exponential link function: μ = exp(Xβ)
- Small parameter changes → exponentially amplified updates
- Prevents numerical overflow and instability
- Critical for sparse, zero-inflated data

### 3. Momentum Updates with Bias Correction

**First Moment (Momentum):**
```
mₜ = ζ₁mₜ₋₁ + (1 - ζ₁)∇f(βₜ)
```

**Second Moment (Squared Gradients):**
```
vₜ = ζ₂vₜ₋₁ + (1 - ζ₂)[∇f(βₜ)]²
```

**Bias Correction:**
```
m̂ₜ = mₜ/(1 - ζ₁ᵗ)
v̂ₜ = vₜ/(1 - ζ₂ᵗ)
```

**Default Parameters:**
- ζ₁ = 0.9 (momentum decay)
- ζ₂ = 0.999 (squared gradient decay)
- ε = 10⁻⁸ (numerical stability)

### 4. Adaptive Learning Rate Decay

**Time-Dependent Decay:**
```
αₜ = α/(1 + ηt)
```

**Convergence Guarantee:**
- Satisfies: ∑αₜ = ∞ and ∑αₜ² < ∞
- Ensures convergence to optimal solution
- Balances exploration vs exploitation

### 5. Parameter Update

**Standard Update:**
```
β_temp = β - αₜm̂ₜ/√(v̂ₜ + ε)
```

### 6. Adaptive Soft-Thresholding (Key Innovation)

**Traditional Soft-Thresholding:**
```
prox_λ∥·∥₁(z) = sign(z) × max(|z| - λ, 0)
```

**AMGD Adaptive Soft-Thresholding:**
```
βⱼ = sign(βⱼ) × max(|βⱼ| - αₜλ₁/(|βⱼ| + ε), 0)
```

**Key Advantages:**
- **Adaptive Threshold**: `αₜλ₁/(|βⱼ| + ε)` instead of fixed λ
- **Large Coefficients**: Lighter penalization (preserved)
- **Small Coefficients**: Heavier shrinkage (encouraged sparsity)
- **Dynamic Selection**: Embedded variable selection during optimization

**Theoretical Connection to Adaptive Lasso:**
- Mimics adaptive penalty weights: wⱼ = 1/|βⱼ|^γ
- No staged reweighting required
- Oracle properties for feature selection

## Complete AMGD Algorithm

```python
def amgd(X, y, α=0.001, ζ₁=0.9, ζ₂=0.999, λ₁=0.1, λ₂=0.0, 
         penalty='l1', T=10.0, η=0.0001, ε=1e-8, max_iter=1000):
    """
    Adaptive Momentum Gradient Descent for Poisson Regression
    """
    n_samples, n_features = X.shape
    
    # Initialize
    β = np.random.normal(0, 0.1, n_features)
    m = np.zeros(n_features)
    v = np.zeros(n_features)
    
    for t in range(1, max_iter + 1):
        # Adaptive learning rate
        αₜ = α / (1 + η * t)
        
        # Forward pass with clipping
        linear_pred = X @ β
        linear_pred = np.clip(linear_pred, -20, 20)
        μ = np.exp(linear_pred)
        
        # Gradient computation
        grad_ll = X.T @ (μ - y)
        
        # Add L2 regularization if ElasticNet
        if penalty == 'elasticnet':
            grad = grad_ll + λ₂ * β
        else:
            grad = grad_ll
            
        # Gradient clipping
        grad = np.clip(grad, -T, T)
        
        # Momentum updates
        m = ζ₁ * m + (1 - ζ₁) * grad
        v = ζ₂ * v + (1 - ζ₂) * (grad ** 2)
        
        # Bias correction
        m̂ = m / (1 - ζ₁ ** t)
        v̂ = v / (1 - ζ₂ ** t)
        
        # Parameter update
        β = β - αₜ * m̂ / (np.sqrt(v̂) + ε)
        
        # Adaptive soft-thresholding
        if penalty in ['l1', 'elasticnet']:
            denom = np.abs(β) + 0.01
            β = np.sign(β) * np.maximum(
                np.abs(β) - αₜ * λ₁ / denom, 0
            )
        
        # Check convergence
        # ... (loss computation and convergence check)
    
    return β
```

## Theoretical Properties

### 1. Convergence Analysis

**Main Result:**
```
Theorem 1: Under convexity assumptions and appropriate step size conditions,
AMGD converges to optimal solution β* at rate O(1/√T)
```

**Conditions:**
- f(β) = -ℓ(β) + λP(β) is convex
- ∇ℓ(β) is L-Lipschitz continuous  
- Gradients are bounded (satisfied by clipping)
- Step sizes satisfy: ∑αₜ = ∞, ∑αₜ² < ∞

### 2. Feature Selection Optimality

**Proposition 1:**
```
For Poisson regression with L1 regularization,
optimal feature subset S* minimizes expected prediction error:

S* = argmin_{S⊆{1,...,p}} E[L(y, f_S(x))] + α|S|
```

**Practical Implications:**
- Consistent variable selection
- Asymptotic normality of non-zero coefficients
- Optimal bias-variance trade-off

### 3. Computational Complexity

**Per Iteration:** O(np)
- Dominated by matrix-vector multiplication X^T(μ - y)
- Momentum updates: O(p)
- Soft-thresholding: O(p)
- **Total**: Linear scaling with data size

**Comparison:**
- **Coordinate Descent**: O(np) per coordinate cycle
- **AMGD**: O(np) per full gradient update
- **Advantage**: Vectorized BLAS operations when p is large

## Hyperparameter Selection

### Default Settings (Robust Across Experiments):
```python
# Momentum parameters (from Adam literature)
ζ₁ = 0.9          # First moment decay
ζ₂ = 0.999        # Second moment decay
ε = 1e-8          # Numerical stability

# AMGD-specific parameters (tuned via cross-validation)
α = 0.05          # Initial learning rate
η = 1e-4          # Learning rate decay
T = 10.0          # Gradient clipping threshold
```

### Cross-Validation Strategy:
- **Learning Rate α**: {0.01, 0.05, 0.1}
- **Decay Factor η**: {10⁻⁵, 10⁻⁴, 10⁻³}
- **Clipping Threshold T**: {5, 10, 20}
- **Metric**: 5-fold CV Mean Absolute Error

### Sensitivity Analysis:
- **Robust to variations** in α, η due to adaptive scaling
- **Gradient clipping essential** for stability (removing T causes divergence)
- **Less manual tuning** required vs standard gradient descent

## Implementation Notes

### Numerical Stability:
1. **Linear predictor clipping**: Prevents exp() overflow
2. **Gradient clipping**: Prevents exploding gradients  
3. **Adaptive denominators**: Prevents division by zero
4. **Bias correction**: Ensures unbiased momentum estimates

### Convergence Monitoring:
- **Objective Function**: Monitor total loss (likelihood + penalty)
- **Tolerance**: 10⁻⁶ relative change in objective
- **Early Stopping**: If convergence achieved before max_iter
- **Learning Rate Decay**: Guarantees eventual small updates

### Comparison with Existing Methods:

| Feature | AMGD | Adam | AdaGrad | GLMnet |
|---------|------|------|---------|---------|
| Gradient Clipping | ✅ Built-in | ❌ Manual | ❌ Manual | ❌ No |
| Adaptive Thresholding | ✅ Novel | ❌ No | ❌ No | ❌ Fixed |
| Momentum | ✅ With bias correction | ✅ Standard | ❌ No | ❌ No |
| Learning Rate Decay | ✅ Guaranteed convergence | ❌ Fixed | ✅ Too aggressive | ❌ No |
| Sparsity Induction | ✅ Adaptive | ❌ None | ❌ None | ✅ Fixed |

## Practical Applications

### Best Suited For:
- **High-dimensional Poisson regression**
- **Sparse feature selection problems**
- **Zero-inflated count data**
- **Environmental/ecological modeling**

### When to Use AMGD:
- ✅ Need both accuracy and sparsity
- ✅ Dealing with correlated features  
- ✅ Require numerical stability
- ✅ Want embedded feature selection
- ✅ Have moderate-sized datasets 

### Limitations:
- ❌ Full-batch updates 
- ❌ Multiple hyperparameters to tune
- ❌ May not scale to very large datasets without stochastic variants
- ❌ Requires careful initialization for some datasets
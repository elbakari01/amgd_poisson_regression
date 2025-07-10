# Experimental Setup and Methodology

## Dataset Description

### Ecological Health Dataset
- **Size**: 61,345 observations with 17 features (after preprocessing)
- **Original Features**: 16 environmental and categorical variables
- **Target Variable**: Biodiversity Index (count data, Poisson-distributed)
- **Domain**: Environmental science and biodiversity conservation
- **Geographic Coverage**: Diverse ecosystem types
- **Time Period**: Multi-year environmental monitoring data

### Feature Categories

#### Environmental Measurements (Continuous, Features 1-12)
1. **Humidity** - Relative humidity percentage
2. **Air Quality Index** - Overall air quality score (0-500)
3. **PM2.5 Concentration** - Fine particulate matter (μg/m³)
4. **Soil Moisture** - Soil water content percentage
5. **Nutrient Level** - Soil nutrient availability index
6. **Water Quality** - Water quality assessment score
7. **Total Dissolved Solids** - Water TDS concentration (ppm)
8. **Soil pH** - Soil acidity/alkalinity level
9. **Biochemical Oxygen Demand** - BOD in water bodies (mg/L)
10. **Chemical Oxygen Demand** - COD in water bodies (mg/L)
11-12. **Additional Environmental Variables** - Temperature, precipitation metrics

#### Categorical Variables (Features 13-17, One-Hot Encoded)
13. **Pollution Level: Low** - Binary indicator
14. **Pollution Level: Moderate** - Binary indicator  
15. **Ecological Health: Ecologically Degraded** - Binary indicator
16. **Ecological Health: Ecologically Healthy** - Binary indicator
17. **Ecological Health: Ecologically Stable** - Binary indicator

### Target Variable: Biodiversity Index
- **Type**: Count data (non-negative integers)
- **Distribution**: Poisson with some zero-inflation
- **Range**: [0, 45] species count
- **Interpretation**: Number of species observed per ecosystem unit
- **Scientific Relevance**: Key indicator of ecosystem health and stability

## Data Preprocessing Pipeline

### 1. Missing Value Treatment
- **Numerical Features**: Median imputation (<2% missing values)
- **Categorical Features**: Mode imputation
- **Strategy**: Conservative approach to preserve data integrity

### 2. Feature Standardization
- **Method**: Z-score normalization: `z = (x - μ)/σ`
- **Applied to**: All continuous environmental variables
- **Purpose**: Ensure equal contribution to regularization penalties

### 3. Categorical Encoding
- **Method**: One-hot encoding with reference category dropped
- **Variables**: Pollution Level, Ecological Health Label
- **Result**: 16 → 17 features after encoding (prevents multicollinearity)

### 4. Outlier Handling
- **Detection**: Interquartile Range (IQR) method
- **Treatment**: Winsorization at 1st and 99th percentiles
- **Rationale**: Preserve extreme but plausible environmental conditions

## Experimental Design

### Data Partitioning Strategy
- **Training Set**: 70% (42,939 observations) - Model fitting
- **Validation Set**: 15% (9,204 observations) - Hyperparameter tuning
- **Test Set**: 15% (9,202 observations) - Final performance evaluation
- **Rationale**: Standard split ensuring sufficient data for robust validation

### Cross-Validation Framework
- **Method**: 5-fold cross-validation on validation set
- **Purpose**: Hyperparameter optimization and model selection
- **Evaluation Metric**: Mean Absolute Error (MAE) as primary criterion
- **Repetitions**: Multiple runs for statistical robustness

## Hyperparameter Optimization

### Search Strategy
- **Method**: Grid search across parameter combinations
- **Regularization Parameters**: 50 λ values logarithmically spaced
- **Range**: λ ∈ [10⁻³, 10¹] (0.001 to 10.0)
- **Regularization Types**: L1 (Lasso) and ElasticNet
- **Selection Criterion**: Minimum cross-validated MAE

### AMGD-Specific Parameters
Based on systematic grid search and stability analysis:

| Parameter | Search Range | Optimal Value | Rationale |
|-----------|--------------|---------------|-----------|
| Learning Rate (α) | {0.01, 0.05, 0.1} | 0.05 | Best convergence speed |
| Decay Factor (η) | {10⁻⁵, 10⁻⁴, 10⁻³} | 10⁻⁴ | Stable learning rate decay |
| Clipping Threshold (T) | {5, 10, 20} | 10.0 | Prevents instability |
| Momentum (ζ₁) | Fixed | 0.9 | Adam literature standard |
| Second Moment (ζ₂) | Fixed | 0.999 | Adam literature standard |

### Baseline Optimizer Configurations
- **Adam**: Default parameters (α=0.01, β₁=0.9, β₂=0.999)
- **AdaGrad**: Default parameters (α=0.01, ε=1e-8)
- **GLMnet**: Scikit-learn implementation with coordinate descent

## Performance Evaluation Framework

### Primary Metrics
1. **Mean Absolute Error (MAE)**
   - Formula: `MAE = (1/n)∑|yᵢ - ŷᵢ|`
   - **Primary optimization target**
   - Robust to outliers in count data

2. **Root Mean Squared Error (RMSE)**
   - Formula: `RMSE = √[(1/n)∑(yᵢ - ŷᵢ)²]`
   - Secondary performance measure
   - Sensitive to large prediction errors

3. **Mean Poisson Deviance (MPD)**
   - Formula: `MPD = (2/n)∑[yᵢlog(yᵢ/ŷᵢ) - (yᵢ - ŷᵢ)]`
   - Distribution-specific fit quality
   - Accounts for Poisson likelihood

### Secondary Metrics
4. **Sparsity Percentage**
   - Formula: `Sparsity = (1 - |{j: |βⱼ| > 10⁻⁶}|/p) × 100%`
   - Feature selection capability
   - Model interpretability measure

5. **Computational Runtime**
   - Wall-clock time to convergence
   - Efficiency comparison across optimizers
   - Measured on standardized hardware

### Convergence Criteria
- **Tolerance**: 10⁻⁶ relative change in objective function
- **Maximum Iterations**: 1000 (sufficient for all methods)
- **Early Stopping**: Applied when tolerance reached
- **Objective Function**: Negative log-likelihood + regularization penalty

## Statistical Validation

### Bootstrap Analysis
- **Method**: Non-parametric bootstrap resampling
- **Sample Size**: 1000 bootstrap resamples
- **Purpose**: 
  - Confidence interval estimation
  - Performance stability assessment
  - Variance characterization across optimizers

### Significance Testing
- **Primary Test**: Paired t-tests comparing AMGD vs each baseline
- **Null Hypothesis**: No difference in performance metrics
- **Alternative**: AMGD shows superior performance
- **Correction**: Bonferroni adjustment for multiple comparisons
- **Effect Size**: Cohen's d for practical significance assessment

### Feature Selection Stability
- **Analysis**: Selection probability across bootstrap samples
- **Threshold**: |βⱼ| > 10⁻⁶ for feature inclusion
- **Reporting**: Top features with selection probabilities
- **Interpretation**: Robust signal vs noise separation

## Implementation Details

### Software Environment
- **Language**: Python 3.8+
- **Core Libraries**: NumPy, SciPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Statistical Testing**: SciPy.stats, Statsmodels
- **Hardware**: Standardized computational environment

### Algorithm Implementation
- **Custom Optimizers**: AMGD, Adam, AdaGrad implemented from scratch
- **GLMnet Baseline**: Scikit-learn's LogisticRegression with Poisson family
- **Reproducibility**: Fixed random seeds (seed=42)
- **Numerical Precision**: Double precision (float64)

### Quality Assurance
- **Code Validation**: Unit tests for each optimizer component
- **Numerical Stability**: Gradient clipping and overflow protection
- **Convergence Monitoring**: Loss tracking and iteration counts
- **Error Handling**: Graceful failure modes for edge cases

## Experimental Results Summary

### Optimal Configurations (Cross-Validation Results)
| Algorithm | Best Regularization | Optimal λ | MAE | RMSE | MPD | Runtime (s) |
|-----------|-------------------|-----------|-----|------|-----|-------------|
| **AMGD** | ElasticNet | 0.01 | **2.985** | **3.873** | **2.188** | **0.002** |
| Adam | ElasticNet | 0.1 | 3.081 | 3.983 | 2.225 | 0.004 |
| AdaGrad | ElasticNet | 10.0 | 6.862 | 7.579 | 10.965 | 0.745 |
| GLMnet | Lasso | 0.01 | 9.007 | 9.551 | 28.848 | 0.040 |

### Key Performance Improvements
- **AMGD vs Adam**: 2.7% improvement in MAE
- **AMGD vs AdaGrad**: 56.6% improvement in MAE  
- **AMGD vs GLMnet**: 67.2% improvement in MAE
- **Computational Efficiency**: AMGD  at 0.002s

### Feature Selection Results
- **AMGD Sparsity**: 29.29% (11 out of 17 features selected)
- **Consistent Selection**: Ecological health indicators (100% probability)
- **Moderate Selection**: Environmental variables (53-70% probability)
- **Robust Signal Detection**: Clear separation of relevant predictors

### Statistical Significance
All AMGD improvements statistically significant:
- **p-values**: < 0.0001 for all comparisons
- **Effect sizes**: Large (Cohen's d > 0.8) for all metrics
- **Confidence intervals**: Non-overlapping across bootstrap samples

## Limitations and Considerations

### Dataset-Specific Factors
- **Hierarchical Structure**: Categorical indicators dominate predictions
- **Feature Correlation**: Some environmental variables moderately correlated
- **Sample Size**: Large dataset favors full-batch optimization methods
- **Domain**: Results specific to ecological/environmental applications

### Methodological Limitations
- **Single Dataset**: Limited generalizability across domains
- **Baseline Tuning**: Default configurations may not be optimal
- **Computational Scale**: Full-batch methods limit scalability
- **Hyperparameter Sensitivity**: Requires careful tuning in practice

### Future Validation Needs
- **Multi-domain Testing**: Validate across different application areas
- **Stochastic Variants**: Develop mini-batch versions for large-scale data
- **Theoretical Extensions**: Non-convex convergence guarantees
- **Automated Tuning**: Reduce hyperparameter sensitivity

## Reproducibility Guidelines

### Complete Reproduction Steps
1. **Environment Setup**: Install exact package versions (requirements.txt)
2. **Data Preparation**: Apply identical preprocessing pipeline
3. **Random Seeding**: Use seed=42 throughout all experiments
4. **Cross-Validation**: Replicate exact fold assignments
5. **Statistical Testing**: Apply identical bootstrap procedures

### Expected Variations
- **Platform Differences**: ±1-2% variation in numerical results
- **Library Versions**: Minor differences in optimization paths
- **Hardware**: Runtime variations, but relative performance preserved
- **Random Sampling**: Bootstrap results should be statistically equivalent

### Validation Checksums
Key result validation:
- AMGD should outperform other methods on MAE/RMSE
- Statistical significance p < 0.05 for key comparisons
- Sparsity levels: AMGD (25-35%), Adam (10-15%), AdaGrad (5-10%)
- Convergence: All methods should reach tolerance within 1000 iterations
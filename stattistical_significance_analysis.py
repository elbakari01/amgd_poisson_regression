## Statististical significance analysis 


def statistical_significance_analysis(X, y, best_params, n_bootstrap=1000, n_runs=30, random_state=42):
    """
    Perform statistical significance analysis for algorithm performance comparison
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target values
    best_params : dict
        Dictionary with best parameters for each optimizer
    n_bootstrap : int
        Number of bootstrap samples for confidence intervals
    n_runs : int
        Number of runs for statistical tests
    random_state : int
        Random seed
    
    Returns:
    --------
    significance_results : dict
        Dictionary with statistical test results
    """
    import numpy as np
    from scipy import stats
    from sklearn.model_selection import train_test_split
    
    np.random.seed(random_state)
    
    # Include GLMnet in the optimizers list
    optimizers = ['AMGD', 'Adam', 'AdaGrad', 'GLMnet']
    metrics = ['MAE', 'RMSE', 'Mean Deviance', 'Sparsity']
    optimizer_functions = {
        'AMGD': amgd, 
        'Adam': adam, 
        'AdaGrad': adagrad, 
        'GLMnet': glmnet
    }
    
    # 1. Split data for bootstrap validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # 2. Prepare optimizers with their best parameters (using MAE for this example)
    optimizer_configs = {}
    for optimizer_name in optimizers:
        params = best_params[f"{optimizer_name}_MAE"]
        reg_type = params['Regularization']
        lambda_val = params['Lambda']
        
        if optimizer_name == "AMGD":
            base_params = {"alpha": 0.01, "beta1": 0.9, "beta2": 0.999, "T": 100.0, 
                          "tol": 1e-6, "max_iter": 1000, "eta": 0.0001, "epsilon": 1e-8}
        elif optimizer_name == "Adam":
            base_params = {"alpha": 0.01, "beta1": 0.9, "beta2": 0.999, 
                          "tol": 1e-6, "max_iter": 1000, "epsilon": 1e-8}
        elif optimizer_name == 'GLMnet':
            base_params = {"alpha": 0.01, "tol": 1e-6, "max_iter": 1000, "epsilon": 1e-8}
        else:  # AdaGrad
            base_params = {"alpha": 0.01, "tol": 1e-6, "max_iter": 1000, "epsilon": 1e-8}
        
        if reg_type == "L1":
            base_params["lambda1"] = lambda_val
            base_params["lambda2"] = 0.0
            base_params["penalty"] = "l1"
        else:  # ElasticNet
            base_params["lambda1"] = lambda_val / 2
            base_params["lambda2"] = lambda_val / 2
            base_params["penalty"] = "elasticnet"
        
        optimizer_configs[optimizer_name] = base_params
    
    # 3. Multiple runs for performance metrics
    all_metrics = {opt: {metric: [] for metric in metrics} for opt in optimizers}
    feature_selection = {opt: [] for opt in optimizers}
    
    for run in range(n_runs):
        # Create a bootstrapped sample
        indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot, y_boot = X_train[indices], y_train[indices]
        
        # Train each optimizer on bootstrapped data
        for optimizer_name in optimizers:
            try:
                # Try with the standard 4-value return format
                beta, _, runtime, _ = optimizer_functions[optimizer_name](
                    X_boot, y_boot, 
                    **optimizer_configs[optimizer_name],
                    verbose=False,
                    return_iters=False
                )
            except Exception as e:
                try:
                    # Try with GLMnet's 5-value return format
                    beta, _, runtime, _, _ = optimizer_functions[optimizer_name](
                        X_boot, y_boot, 
                        **optimizer_configs[optimizer_name],
                        verbose=False,
                        return_iters=True
                    )
                except Exception as e2:
                    print(f"Error training {optimizer_name}: {str(e2)}")
                    # Create a default array of zeros as fallback
                    beta = np.zeros(X_boot.shape[1])
            
            # Evaluate on test data
            metrics_values = evaluate_model(beta, X_test, y_test)
            
            # Store metrics
            for metric in metrics:
                all_metrics[optimizer_name][metric].append(metrics_values[metric])
            
            # Store feature selection (which features have non-zero coefficients)
            feature_selection[optimizer_name].append(np.abs(beta) > 1e-6)
    
    # 4. Compute bootstrap statistics
    bootstrap_results = {}
    for optimizer_name in optimizers:
        bootstrap_results[optimizer_name] = {}
        
        for metric in metrics:
            values = all_metrics[optimizer_name][metric]
            
            # Bootstrap for confidence intervals
            bootstrap_means = []
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            # Compute statistics
            mean_value = np.mean(values)
            std_value = np.std(values)
            
            # 95% confidence interval
            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)
            
            bootstrap_results[optimizer_name][metric] = {
                'mean': mean_value,
                'std': std_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'values': values
            }
    
    # 5. Statistical tests
    statistical_tests = {}
    
    # Compare AMGD (reference) vs other optimizers
    reference = 'AMGD'
    for metric in metrics:
        statistical_tests[metric] = {}
        
        reference_values = all_metrics[reference][metric]
        
        for optimizer_name in [opt for opt in optimizers if opt != reference]:
            comparison_values = all_metrics[optimizer_name][metric]
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(reference_values, comparison_values)
            
            # Effect size (Cohen's d)
            mean_diff = np.mean(reference_values) - np.mean(comparison_values)
            pooled_std = np.sqrt((np.var(reference_values) + np.var(comparison_values)) / 2)
            effect_size = mean_diff / pooled_std
            
            statistical_tests[metric][optimizer_name] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'significant': p_value < 0.05
            }
    
    # 6. Feature selection analysis
    feature_selection_results = {}
    n_features = X.shape[1]
    
    for optimizer_name in optimizers:
        feature_selection_probs = np.mean(feature_selection[optimizer_name], axis=0)
        
        # For each feature, compute probability of selection
        features = []
        for idx in range(n_features):
            features.append({
                'feature_idx': idx,
                'selection_probability': feature_selection_probs[idx]
            })
        
        # Sort by selection probability
        features = sorted(features, key=lambda x: x['selection_probability'], reverse=True)
        
        feature_selection_results[optimizer_name] = features
    
    # Combine all results
    significance_results = {
        'bootstrap_results': bootstrap_results,
        'statistical_tests': statistical_tests,
        'feature_selection_results': feature_selection_results
    }
    
    return significance_results

def display_statistical_results(significance_results, feature_names=None):
    """
    Display the statistical significance results in a readable format
    
    Parameters:
    -----------
    significance_results : dict
        Results from statistical_significance_analysis
    feature_names : list
        Names of features for better readability
    """
    import numpy as np
    
    # Include GLMnet color
    colors = {
        'AMGD': '#3498db', 
        'Adam': '#e74c3c', 
        'AdaGrad': '#2ecc71', 
        'GLMnet': '#9b59b6'
    }
    
    # Print summary of bootstrap results
    print("\n===== BOOTSTRAP RESULTS (95% Confidence Intervals) =====")
    print("{:<10} {:<15} {:<15} {:<15} {:<15}".format(
        "Metric", "AMGD", "Adam", "AdaGrad", "GLMnet"))
    print("-" * 70)
    
    for metric in ['MAE', 'RMSE', 'Mean Deviance', 'Sparsity']:
        print("{:<10}".format(metric), end=" ")
        
        for optimizer in ['AMGD', 'Adam', 'AdaGrad', 'GLMnet']:
            result = significance_results['bootstrap_results'][optimizer][metric]
            print("{:.4f} [{:.4f}, {:.4f}]".format(
                result['mean'], result['ci_lower'], result['ci_upper']), end=" ")
        
        print()
    
    # Print summary of statistical tests (comparing AMGD vs others)
    print("\n===== STATISTICAL TESTS (AMGD vs Others) =====")
    print("{:<10} {:<15} {:<15} {:<15}".format(
        "Metric", "vs Adam", "vs AdaGrad", "vs GLMnet"))
    print("-" * 55)
    
    for metric in ['MAE', 'RMSE', 'Mean Deviance', 'Sparsity']:
        print("{:<10}".format(metric), end=" ")
        
        for optimizer in ['Adam', 'AdaGrad', 'GLMnet']:
            test = significance_results['statistical_tests'][metric][optimizer]
            result = "p={:.4f} (d={:.2f}){}".format(
                test['p_value'], 
                test['effect_size'], 
                "*" if test['significant'] else "")
            print("{:<15}".format(result), end=" ")
        
        print()
    
    # Print top selected features from AMGD
    print("\n===== TOP SELECTED FEATURES (AMGD) =====")
    amgd_features = significance_results['feature_selection_results']['AMGD']
    
    for i, feature in enumerate(amgd_features[:10]):  # Show top 10
        if feature_names and feature['feature_idx'] < len(feature_names):
            feature_name = feature_names[feature['feature_idx']]
        else:
            feature_name = f"Feature {feature['feature_idx']}"
        
        print("{:<4} {:<30} Selection Probability: {:.2f}".format(
            f"{i+1}.", feature_name, feature['selection_probability']))

def add_to_pipeline(X, y, best_params, feature_names, model_results):
    """
    Add statistical significance analysis to pipeline
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Include GLMnet in colors
    colors = {
        'AMGD': '#3498db', 
        'Adam': '#e74c3c', 
        'AdaGrad': '#2ecc71', 
        'GLMnet': '#9b59b6'
    }
    
    print("\nStep 8: Performing statistical significance analysis")
    significance_results = statistical_significance_analysis(X, y, best_params)
    display_statistical_results(significance_results, feature_names)
    
    # Create a figure to visualize confidence intervals for metrics
    plt.figure(figsize=(15, 10))
    
    metrics = ['MAE', 'RMSE', 'Mean Deviance', 'Sparsity']
    optimizers = ['AMGD', 'Adam', 'AdaGrad', 'GLMnet']
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        
        x_pos = np.arange(len(optimizers))
        means = []
        errors = []
        
        for j, optimizer in enumerate(optimizers):
            result = significance_results['bootstrap_results'][optimizer][metric]
            means.append(result['mean'])
            errors.append([result['mean'] - result['ci_lower'], result['ci_upper'] - result['mean']])
        
        errors = np.array(errors).T
        
        bars = plt.bar(x_pos, means, color=[colors[opt] for opt in optimizers], 
                      yerr=errors, capsize=10, alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + errors[1][bars.index(bar)],
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.title(f'{metric} with 95% Confidence Intervals')
        plt.xticks(x_pos, optimizers)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # If this is sparsity, highlight that higher is better
        if metric == 'Sparsity':
            plt.ylabel(f'{metric} (higher is better)')
        else:
            plt.ylabel(f'{metric} (lower is better)')
    
    plt.tight_layout()
    plt.savefig('metric_confidence_intervals.png', dpi=300)
    plt.show()
    
    # Create a figure to visualize feature selection consistency
    plt.figure(figsize=(15, 6))
    
    # Find top 10 features based on selection probability in AMGD
    amgd_features = significance_results['feature_selection_results']['AMGD']
    top_features = sorted(amgd_features, key=lambda x: x['selection_probability'], reverse=True)[:10]
    top_indices = [f['feature_idx'] for f in top_features]
    
    x_pos = np.arange(len(top_indices))
    width = 0.2  # Adjusted width to fit 4 optimizers
    
    for i, optimizer in enumerate(optimizers):
        features = significance_results['feature_selection_results'][optimizer]
        probs = [next(f for f in features if f['feature_idx'] == idx)['selection_probability'] for idx in top_indices]
        
        plt.bar(x_pos + (i-1.5)*width, probs, width, color=colors[optimizer], label=optimizer, alpha=0.7)
    
    plt.xlabel('Feature')
    plt.ylabel('Selection Probability')
    plt.title('Feature Selection Consistency Across Optimizers')
    
    x_labels = []
    for idx in top_indices:
        if feature_names and idx < len(feature_names):
            x_labels.append(feature_names[idx])
        else:
            x_labels.append(f'Feature {idx}')
    
    plt.xticks(x_pos, x_labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('feature_selection_consistency.png', dpi=300)
    plt.show()
    
    # Create a violin plot to show distribution of metrics across runs
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        
        data = []
        for optimizer in optimizers:
            data.append(significance_results['bootstrap_results'][optimizer][metric]['values'])
        
        violin_parts = plt.violinplot(data, showmeans=True, showmedians=True)
        
        # Color each violin
        for j, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors[optimizers[j]])
            pc.set_alpha(0.7)
        
        # Add data points
        for j, d in enumerate(data):
            plt.scatter([j+1] * len(d), d, color=colors[optimizers[j]], 
                      alpha=0.2, s=5)
        
        plt.xticks(np.arange(1, len(optimizers)+1), optimizers)
        plt.title(f'Distribution of {metric} Across Runs')
        
        # If this is sparsity, highlight that higher is better
        if metric == 'Sparsity':
            plt.ylabel(f'{metric} (higher is better)')
        else:
            plt.ylabel(f'{metric} (lower is better)')
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('metric_distributions.png', dpi=300)
    plt.show()
    
    # Create correlation plot for AMGD feature selection
    plt.figure(figsize=(10, 8))
    
    # Get feature selection matrix for AMGD
    amgd_selection = np.array(feature_selection['AMGD'])
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(amgd_selection.T)
    
    # Plot correlation matrix as heatmap
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    
    # Add grid lines
    plt.grid(False)
    
    # Set ticks and labels
    top_n = min(15, len(corr_matrix))  # Show at most 15 features
    feature_indices = [f['feature_idx'] for f in amgd_features[:top_n]]
    feature_labels = []
    for idx in feature_indices:
        if feature_names and idx < len(feature_names):
            feature_labels.append(feature_names[idx])
        else:
            feature_labels.append(f'F{idx}')
    
    plt.xticks(np.arange(top_n), feature_labels, rotation=90)
    plt.yticks(np.arange(top_n), feature_labels)
    
    plt.title('Feature Selection Correlation (AMGD)')
    plt.tight_layout()
    plt.savefig('feature_selection_correlation.png', dpi=300)
    plt.show()
    
    return significance_results
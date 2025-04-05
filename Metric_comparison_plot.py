
def create_algorithm_comparison_plots(cv_results_df):
    """
    Creating barplots to comparing the performance of AMGD, AdaGrad, Adam, and GLMnet algorithms
    across different metrics with L1 and ElasticNet regularization .
    
    Parameters:
    -----------
    cv_results_df : pandas.DataFrame
        DataFrame with cross-validation results
    
    Returns:
    --------
    figs : list
        List of matplotlib figures
    """
    #  List to store the figures
    figs = []
    
    # Metrics to compare
    metrics = ['MAE', 'RMSE', 'Mean Deviance', 'Runtime']
    
    best_results = []
    
    for optimizer in ['AMGD', 'AdaGrad', 'Adam', 'GLMnet']:
        optimizer_df = cv_results_df[cv_results_df['Optimizer'] == optimizer]
        
        for metric in metrics:
            if metric in ['MAE', 'RMSE', 'Mean Deviance']:
                # For these metrics, lower is better
                best_idx = optimizer_df[metric].idxmin()
            else:  # Runtime
                # For runtime, lower is better
                best_idx = optimizer_df['Runtime'].idxmin()
            
            best_results.append({
                'Optimizer': optimizer,
                'Metric': metric,
                'Value': optimizer_df.loc[best_idx, metric],
                'Regularization': optimizer_df.loc[best_idx, 'Regularization'],
                'Lambda': optimizer_df.loc[best_idx, 'Lambda']
            })
    
    # 
    best_results_df = pd.DataFrame(best_results)
    
    #  Barplot for each metric
    for metric in metrics:
        metric_df = best_results_df[best_results_df['Metric'] == metric]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        #  colors for each optimizer
        colors = {'AMGD': '#3498db', 'AdaGrad': '#2ecc71', 'Adam': '#e74c3c', 'GLMnet': '#9b59b6'}
        bar_colors = [colors[opt] for opt in metric_df['Optimizer']]
        
        #  barplot
        bars = ax.bar(metric_df['Optimizer'], metric_df['Value'], color=bar_colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02),
                   f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Add regularization and lambda information below bars
        for i, (_, row) in enumerate(metric_df.iterrows()):
            ax.text(i, 0, f"{row['Regularization']}\nλ={row['Lambda']:.4f}", 
                   ha='center', va='bottom', fontsize=8, color='black',
                   transform=ax.get_xaxis_transform())
        
        # title and labels
        ax.set_title(f'Best {metric} Comparison Across Optimizers (L1/ElasticNet)', fontsize=14)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xlabel('Optimizer', fontsize=12)
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.ylim(0, metric_df['Value'].max() * 1.15)
        
        plt.tight_layout()
        
        figs.append(fig)
    
    return figs
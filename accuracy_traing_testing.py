
#Accuracy


def plot_training_and_test_metrics(model_results, test_metrics, metric_to_plot='MAE'):
    """
    Plot training and test metrics across optimizers.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary containing all optimizer models and their results
    test_metrics : dict
        Dictionary containing test metrics for all optimizer models
    metric_to_plot : str
        Metric to plot ('MAE', 'RMSE', 'Mean Deviance')
    
    Returns:
    --------
    None (displays plots)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Set up the figure
    plt.figure(figsize=(15, 10))
    
    # 1. Plot loss histories (convergence) for all optimizers
    plt.subplot(2, 2, 1)
    # Updated colors dictionary to include GLMnet
    colors = {'AMGD': '#3498db', 'Adam': '#e74c3c', 'AdaGrad': '#2ecc71', 'GLMnet': '#9b59b6'}
    
    for optimizer_name, results in model_results.items():
        loss_history = results['loss_history']
        iterations = np.arange(1, len(loss_history) + 1)
        plt.semilogy(iterations, loss_history, label=f"{optimizer_name}", color=colors[optimizer_name], linewidth=2)
    
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title('Training Loss Convergence', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # 2. Plot training metrics for all optimizers
    plt.subplot(2, 2, 2)
    optimizer_names = list(model_results.keys())
    train_metrics = [model_results[opt]['train_metrics'][metric_to_plot] for opt in optimizer_names]
    
    # Create the bar chart for training metrics
    bars = plt.bar(optimizer_names, train_metrics, color=[colors[opt] for opt in optimizer_names])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02),
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.ylabel(f'Training {metric_to_plot}', fontsize=12)
    plt.title(f'Training {metric_to_plot} by Optimizer', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. Plot test metrics for all optimizers
    plt.subplot(2, 2, 3)
    test_metric_values = [test_metrics[opt][metric_to_plot] for opt in optimizer_names]
    
    # Create the bar chart for test metrics
    bars = plt.bar(optimizer_names, test_metric_values, color=[colors[opt] for opt in optimizer_names])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02),
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.ylabel(f'Test {metric_to_plot}', fontsize=12)
    plt.title(f'Test {metric_to_plot} by Optimizer', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 4. Plot train vs test comparison
    plt.subplot(2, 2, 4)
    x = np.arange(len(optimizer_names))
    width = 0.35
    
    # Create the grouped bar chart
    bars1 = plt.bar(x - width/2, train_metrics, width, label=f'Training {metric_to_plot}', alpha=0.7)
    bars2 = plt.bar(x + width/2, test_metric_values, width, label=f'Test {metric_to_plot}', alpha=0.7)
    
    plt.xlabel('Optimizer', fontsize=12)
    plt.ylabel(metric_to_plot, fontsize=12)
    plt.title(f'Training vs Test {metric_to_plot} Comparison', fontsize=14)
    plt.xticks(x, optimizer_names)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    # Create another figure for the feature importance comparison
    plt.figure(figsize=(16, 6))
    
    # Arrange subplots for each optimizer's feature importance (update for 4 optimizers)
    for i, optimizer_name in enumerate(optimizer_names, 1):
        plt.subplot(1, len(optimizer_names), i)
        
        # Get beta coefficients for this optimizer
        beta = model_results[optimizer_name]['beta']
        
        # Calculate feature importance based on absolute coefficient values
        importance = np.abs(beta)
        indices = np.argsort(importance)[::-1]
        
        # Get the top N features
        top_n = min(10, len(importance))
        
        # Get feature names if available, otherwise use indices
        feature_indices = indices[:top_n]
        feature_labels = [f"Feature {idx}" for idx in feature_indices]  # Replace with actual feature names if available
        
        # Plot importance
        plt.barh(range(top_n), importance[feature_indices], align='center', color=colors[optimizer_name])
        plt.yticks(range(top_n), feature_labels)
        plt.xlabel('Coefficient Magnitude')
        plt.title(f'{optimizer_name} Feature Importance')
        
    plt.tight_layout()
    plt.show()

    # Create non-zero features evolution plot
    plt.figure(figsize=(12, 6))
    
    for optimizer_name, results in model_results.items():
        nonzero_history = results['nonzero_history']
        iterations = np.arange(1, len(nonzero_history) + 1)
        plt.plot(iterations, nonzero_history, label=f"{optimizer_name}", color=colors[optimizer_name], linewidth=2)
    
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Number of Non-Zero Coefficients', fontsize=12)
    plt.title('Sparsity Evolution During Training', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_optimizer_comparison(model_results, test_metrics, metrics_to_compare=None):
    """
    Plot a comprehensive comparison of all optimizer performances.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary containing all optimizer models and their results
    test_metrics : dict
        Dictionary containing test metrics for all optimizer models
    metrics_to_compare : list, optional
        List of metrics to compare ('MAE', 'RMSE', 'Mean Deviance', 'Sparsity')
    
    Returns:
    --------
    None (displays plots)
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from matplotlib.ticker import PercentFormatter
    
    if metrics_to_compare is None:
        metrics_to_compare = ['MAE', 'RMSE', 'Mean Deviance', 'Sparsity']
    
    optimizer_names = list(model_results.keys())
    
    #  radar chart
    data = []
    for optimizer in optimizer_names:
        row = []
        for metric in metrics_to_compare:
            if metric in test_metrics[optimizer]:
                # For all metrics except Sparsity, lower is better
                if metric != 'Sparsity':
                    row.append(test_metrics[optimizer][metric])
                else:
                    row.append(1 - test_metrics[optimizer][metric])
        data.append(row)
    
    # Convert to numpy array
    data = np.array(data)
    
    # Normalize the data between 0 and 1 for radar chart
    data_normalized = np.zeros_like(data, dtype=float)
    for i in range(len(metrics_to_compare)):
        if metrics_to_compare[i] != 'Sparsity':
            # For error metrics, smaller is better, so normalize differently
            data_normalized[:, i] = (data[:, i] - np.min(data[:, i])) / (np.max(data[:, i]) - np.min(data[:, i]) + 1e-10)
        else:
            # For sparsity, we already inverted it, so smaller is better
            data_normalized[:, i] = (data[:, i] - np.min(data[:, i])) / (np.max(data[:, i]) - np.min(data[:, i]) + 1e-10)
    
    # Number of variables
    N = len(metrics_to_compare)
    
    # Create a figure for the radar chart
    plt.figure(figsize=(10, 10))
    
    # Plot the radar chart
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    ax = plt.subplot(111, polar=True)
    
    # Add variable labels
    plt.xticks(angles[:-1], metrics_to_compare, size=12)
    
    # Draw y-labels (percentages)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot data with updated colors to include GLMnet
    colors = {'AMGD': '#3498db', 'Adam': '#e74c3c', 'AdaGrad': '#2ecc71', 'GLMnet': '#9b59b6'}
    for i, optimizer in enumerate(optimizer_names):
        values = data_normalized[i].tolist()
        values += values[:1]  # Close the polygon
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=optimizer, color=colors[optimizer])
        ax.fill(angles, values, alpha=0.1, color=colors[optimizer])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Optimizer Performance Comparison\n(Closer to center is better)", size=15)
    
    plt.tight_layout()
    plt.show()
    
    # Create a summary table for runtime comparison
    runtimes = [model_results[opt]['runtime'] for opt in optimizer_names]
    iterations = [len(model_results[opt]['loss_history']) for opt in optimizer_names]
    
    plt.figure(figsize=(12, 6))
    
    # Runtime comparison
    plt.subplot(1, 2, 1)
    # Create a list of colors that match the optimizer order
    bar_colors = [colors[opt] for opt in optimizer_names]
    bars = plt.bar(optimizer_names, runtimes, color=bar_colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02),
                f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('Total Runtime Comparison', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Iterations comparison
    plt.subplot(1, 2, 2)
    bars = plt.bar(optimizer_names, iterations, color=bar_colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02),
                f'{height}', ha='center', va='bottom', fontsize=10)
    plt.ylabel('Number of Iterations', fontsize=12)
    plt.title('Convergence Iterations Comparison', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
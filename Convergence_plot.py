
# Function to compare convergence rates
def compare_convergence_rates(X_train, y_train, best_params):
    """
    Plot to compare convergence rates of optimization algorithms
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training feature matrix
    y_train : numpy.ndarray
        Training target values
    best_params : dict
        Dictionary with best parameters for each optimizer
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Plot comparing convergence rates of all optimizers
    """
    print("Comparing convergence rates of optimization algorithms...")
    
    # Include all optimizers
    optimizers = ['AMGD', 'Adam', 'AdaGrad', 'GLMnet']
    optimizer_functions = {
        'AMGD': amgd, 
        'Adam': adam, 
        'AdaGrad': adagrad, 
        'GLMnet': glmnet
    }
    colors = {
        'AMGD': '#3498db',  # Blue
        'Adam': '#e74c3c',  # Red
        'AdaGrad': '#2ecc71',  # Green
        'GLMnet': '#9b59b6'  # Purple
    }
    linestyles = {
        'AMGD': '-', 
        'Adam': '--', 
        'AdaGrad': '-.', 
        'GLMnet': ':'
    }
    
    # Store loss histories
    all_loss_histories = {}
    
    for optimizer_name in optimizers:
        # Get best parameters for MAE (or RMSE)
        params = best_params[f'{optimizer_name}_MAE']
        reg_type = params['Regularization']
        lambda_val = params['Lambda']
        
        # Setup base parameters
        if optimizer_name == "AMGD":
            base_params = {
                "alpha": 0.01, "beta1": 0.9, "beta2": 0.999, "T": 100.0, 
                "tol": 1e-6, "max_iter": 1000, "eta": 0.0001, "epsilon": 1e-8
            }
        elif optimizer_name == "Adam":
            base_params = {
                "alpha": 0.01, "beta1": 0.9, "beta2": 0.999, 
                "tol": 1e-6, "max_iter": 1000, "epsilon": 1e-8
            }
        elif optimizer_name == 'GLMnet':
            base_params = {
                "alpha": 0.01, "tol": 1e-6, "max_iter": 1000, 
                "epsilon": 1e-8, "is_pre_scaled": True
            }
        else:  # AdaGrad
            base_params = {
                "alpha": 0.01, "tol": 1e-6, "max_iter": 1000, "epsilon": 1e-8
            }
        
        # Configure regularization parameters
        if reg_type == "L1":
            base_params["lambda1"] = lambda_val
            base_params["lambda2"] = 0.0
            base_params["penalty"] = "l1"
        else:  # ElasticNet
            base_params["lambda1"] = lambda_val / 2
            base_params["lambda2"] = lambda_val / 2
            base_params["penalty"] = "elasticnet"
        
        # Run optimizer and track loss history
        _, loss_history, _, _ = optimizer_functions[optimizer_name](
            X_train, y_train, **base_params, verbose=False, return_iters=False
        )
        
        # Only store non-empty loss histories
        if len(loss_history) > 0:
            all_loss_histories[optimizer_name] = loss_history
        else:
            print(f"Warning: {optimizer_name} returned an empty loss history. Skipping in convergence plot.")
    
    # Check if we have any valid loss histories to plot
    if not all_loss_histories:
        print("Error: No valid loss histories to plot. Check that at least one optimizer returns non-empty loss history.")
        # Return an empty figure 
        return plt.figure(figsize=(15, 10))
    
    plt.figure(figsize=(15, 10))
    
    # Main convergence plot (log scale)
    plt.subplot(2, 1, 1)
    for optimizer_name, loss_history in all_loss_histories.items():
        # Calculate percentage of max iterations (normalization)
        iterations = np.linspace(0, 100, len(loss_history))
        
        # Plot with log scale for y-axis
        plt.semilogy(
            iterations, 
            loss_history, 
            label=optimizer_name, 
            color=colors[optimizer_name], 
            linestyle=linestyles[optimizer_name], 
            linewidth=2
        )
    
    plt.title('Convergence Rate Comparison (Log Scale)', fontsize=14)
    plt.xlabel('Percentage of Max Iterations (%)', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Optimizer', loc='best')
    
    # Normalized convergence plot 
    if len(all_loss_histories) > 1:
        plt.subplot(2, 1, 2)
        max_lengths = max(len(loss_history) for loss_history in all_loss_histories.values())
        
        for optimizer_name, loss_history in all_loss_histories.items():
            # Skip empty loss histories
            if len(loss_history) == 0:
                continue
                
            # Normalize loss history to same length
            if len(loss_history) < max_lengths:
                # Interpolate to match max length
                try:
                    x_new = np.linspace(0, 1, max_lengths)
                    x_old = np.linspace(0, 1, len(loss_history))
                    normalized_loss = np.interp(x_new, x_old, loss_history)
                except ValueError as e:
                    print(f"Error interpolating {optimizer_name} loss history: {e}")
                    continue
            else:
                normalized_loss = loss_history
            
            plt.plot(
                np.linspace(0, 100, len(normalized_loss)), 
                normalized_loss, 
                label=optimizer_name, 
                color=colors[optimizer_name], 
                linestyle=linestyles[optimizer_name], 
                linewidth=2
            )
        
        plt.title('Normalized Convergence Rate Comparison', fontsize=14)
        plt.xlabel('Percentage of Max Iterations (%)', fontsize=12)
        plt.ylabel('Normalized Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Optimizer', loc='best')
    
    plt.tight_layout()
    
    return plt.gcf()
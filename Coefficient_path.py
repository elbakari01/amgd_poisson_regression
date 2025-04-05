


def plot_coefficient_paths_for_ecological_data():
    """
    Plot coefficient paths for different optimizers using the ecological dataset
    """
    print("Analyzing coefficient paths for ecological dataset...")
    
    # 1. Load and preprocess the data
    X, y, feature_names = preprocess_ecological_dataset("ecological_health_dataset.csv")
    
    # 2. Split data into train, validation, and test sets (70/15/15)
    # First split: 85% train+val, 15% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    # Second split: 70% train, 15% validation (82.35% of train_val is train)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]:.1%})")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/X.shape[0]:.1%})")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]:.1%})")
    
    # 3. Configure lambda values for the regularization path
    lambda_values = np.logspace(-3, 1, 10)  # From 0.001 to 10
    
    # 4. Select only the top most important features for readability
    # First, run a basic model to identify important features
    params = {
        "alpha": 0.01, 
        "beta1": 0.9, 
        "beta2": 0.999, 
        "lambda1": 0.1,
        "lambda2": 0.0,
        "penalty": "l1",
        "T": 100.0, 
        "tol": 1e-6, 
        "max_iter": 200,  # Reduced for quicker execution
        "eta": 0.0001, 
        "epsilon": 1e-8,
        "verbose": False
    }
    
    initial_beta, _, _, _ = amgd(X_train, y_train, **params)
    
    # Find top 10 features by coefficient magnitude
    importance = np.abs(initial_beta)
    top_indices = np.argsort(importance)[-17:]  # Top 10 features
    top_feature_names = [feature_names[i] for i in top_indices]
    
    # 5. Create figure for the coefficient paths
    fig, axes = plt.subplots(4, 2, figsize=(18, 20), sharex=True)
    fig.suptitle('Coefficient Paths for Biodiversity Prediction: L1/ElasticNet Regularization', fontsize=16)
    
    # Configure plot settings
    optimizers = ['AMGD', 'Adam', 'AdaGrad', 'GLMnet'] # Added GLMnet
    penalty_types = ['l1', 'elasticnet']
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_indices)))
    
    # 6. Plot coefficient paths for each optimizer and regularization type
    for i, optimizer_name in enumerate(optimizers):
        for j, penalty in enumerate(penalty_types):
            ax = axes[i, j]
            
            # Storage for coefficient values at each lambda
            coef_paths = []
            
            # Run optimization for each lambda value
            for lambda_val in lambda_values:
                if optimizer_name == 'AMGD':
                    params = {
                        "alpha": 0.01, 
                        "beta1": 0.9, 
                        "beta2": 0.999, 
                        "T": 100.0, 
                        "tol": 1e-6, 
                        "max_iter": 200,  
                        "eta": 0.0001, 
                        "epsilon": 1e-8,
                        "lambda1": lambda_val if penalty == 'l1' else lambda_val/2,
                        "lambda2": 0.0 if penalty == 'l1' else lambda_val/2,
                        "penalty": penalty,
                        "verbose": False
                    }
                    beta, _, _, _ = amgd(X_train, y_train, **params)
                    
                elif optimizer_name == 'Adam':
                    params = {
                        "alpha": 0.01, 
                        "beta1": 0.9, 
                        "beta2": 0.999, 
                        "tol": 1e-6, 
                        "max_iter": 200,  
                        "epsilon": 1e-8,
                        "lambda1": lambda_val if penalty == 'l1' else lambda_val/2,
                        "lambda2": 0.0 if penalty == 'l1' else lambda_val/2,
                        "penalty": penalty,
                        "verbose": False
                    }
                    beta, _, _, _ = adam(X_train, y_train, **params)
                    
                elif optimizer_name == 'AdaGrad':
                    params = {
                        "alpha": 0.01, 
                        "tol": 1e-6, 
                        "max_iter": 200,  
                        "epsilon": 1e-8,
                        "lambda1": lambda_val if penalty == 'l1' else lambda_val/2,
                        "lambda2": 0.0 if penalty == 'l1' else lambda_val/2,
                        "penalty": penalty,
                        "verbose": False
                    }
                    beta, _, _, _ = adagrad(X_train, y_train, **params)
                
                else:  # GLMnet
                    params = {
                        "alpha": 0.01,
                        "lambda1": lambda_val if penalty == 'l1' else lambda_val/2,
                        "lambda2": 0.0 if penalty == 'l1' else lambda_val/2,
                        "penalty": penalty,
                        "tol": 1e-6,
                        "max_iter": 200,
                        "epsilon": 1e-8,
                        "is_pre_scaled": False,
                        "verbose": False
                    }
                    try:
                        # Try unpacking 5 values but use only the first (beta)
                        beta, _, _, _, _ = glmnet(X_train, y_train, **params, return_iters=True)
                    except ValueError:
                        # If only 4 values returned
                        beta, _, _, _ = glmnet(X_train, y_train, **params, return_iters=False)
                
                # Extract coefficients for the top features only
                selected_coeffs = [beta[idx] for idx in top_indices]
                coef_paths.append(selected_coeffs)
            
            # Convert to numpy array for easier manipulation
            coef_paths = np.array(coef_paths)
            
            # Plot coefficient paths for top features
            for idx, feature_idx in enumerate(range(len(top_indices))):
                ax.plot(lambda_values, coef_paths[:, idx], 
                        color=colors[idx], 
                        label=top_feature_names[idx],
                        linewidth=2)
            
            # Set labels and title
            ax.set_xscale('log')
            ax.set_xlabel('Regularization Strength (λ)' if i == 3 else '')  # Updated for 4 rows
            ax.set_ylabel('Coefficient Value' if j == 0 else '')
            ax.set_title(f'{optimizer_name} - {penalty.capitalize()} Regularization')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add vertical line for lambda=0.1 (a common default value)
            ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
            
            # Add horizontal line at y=0
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.08),
               title='Features', ncol=5, frameon=True)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.96])
    plt.savefig('ecological_coefficient_paths.png', dpi=300, bbox_inches='tight')
    plt.show()

    Plot coefficient evolution during training
    plot_coefficient_evolution_for_ecological_data(X_train, y_train, top_indices, top_feature_names)
##GLMnet implementation


def glmnet(
    X, y,
    alpha=1.0,
    lambda1=1.0,
    lambda2=1.0,
    penalty='elasticnet',
    tol=1e-4,
    max_iter=1000,
    fit_intercept=False,
    verbose=False,
    epsilon=1e-8, 
    return_iters=False,
    is_pre_scaled=False,
    lr_schedule='inverse_time',
    initial_lr=0.1,
    decay_rate=0.01,
    step_size=100,  # For step decay
    step_factor=0.5  # For step decay
):
   
    # Standardize features if needed
    if not is_pre_scaled:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Add intercept column if needed
    if fit_intercept:
        X = np.column_stack([np.ones(X.shape[0]), X])
    
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)  # Initialize coefficients
    
    # Compute regularization parameters
    total_penalty = lambda1 + lambda2
    if penalty == 'l1':
        l1_ratio = 1.0
    elif penalty == 'l2':
        l1_ratio = 0.0
    elif penalty == 'elasticnet':
        l1_ratio = lambda1 / total_penalty if total_penalty > 0 else 0.0
    else:  # 'none'
        l1_ratio = 0.0
        total_penalty = 0.0
    
    # Initialize tracking variables
    loss_history = []
    beta_history = []
    nonzero_history = []
    lr_history = []
    start_time = time.time()
    
    # Function to calculate learning rate based on the selected schedule
    def get_learning_rate(iteration):
        if lr_schedule == 'constant':
            return initial_lr
        elif lr_schedule == 'inverse_time':
            return initial_lr / (1.0 + decay_rate * iteration)
        elif lr_schedule == 'exponential':
            return initial_lr * np.exp(-decay_rate * iteration)
        elif lr_schedule == 'step':
            return initial_lr * (step_factor ** (iteration // step_size))
        else:
            return initial_lr  # Default to constant if invalid schedule
    
    # Main optimization loop
    for iteration in range(max_iter):
        # Store current coefficients for convergence check
        old_beta = beta.copy()
        
        # Compute linear predictor and expected values with clipping to prevent overflow
        eta = X @ beta
        # Add clipping to prevent exponential overflow
        eta = np.clip(eta, -20, 20)  # Safe range for exp()
        mu = np.exp(eta) + epsilon  # Add epsilon for numerical stability
        
        # Compute gradient with careful handling of potential NaN values
        residual = y - mu
        # Check for and handle NaN/Inf values
        residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
        gradient = -X.T @ residual / n_samples
        
        # Add regularization gradients
        if total_penalty > 0:
            # L2 penalty component with safeguards
            l2_grad = total_penalty * (1 - l1_ratio) * beta
            # L1 penalty component (subgradient)
            l1_grad = total_penalty * l1_ratio * np.sign(beta)
            # Don't regularize intercept if present
            if fit_intercept:
                l2_grad[0] = 0
                l1_grad[0] = 0
            gradient += l2_grad + l1_grad
        
        # Add gradient clipping to prevent extreme updates
        max_grad_norm = 10.0
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > max_grad_norm:
            gradient = gradient * (max_grad_norm / grad_norm)
        
        # Calculate current learning rate using the selected schedule
        learning_rate = get_learning_rate(iteration)
        lr_history.append(learning_rate)
        
        # Update coefficients with the current learning rate
        beta -= learning_rate * gradient
        
        # Compute loss with safety checks
        log_likelihood = np.sum(y * eta - mu)
        reg_penalty = 0
        if total_penalty > 0:
            # Clip beta to prevent overflow in beta**2
            beta_clipped = np.clip(beta, -100, 100)
            l2_penalty = 0.5 * total_penalty * (1 - l1_ratio) * np.sum(beta_clipped**2)
            l1_penalty = total_penalty * l1_ratio * np.sum(np.abs(beta))
            reg_penalty = l2_penalty + l1_penalty
        loss = -log_likelihood + reg_penalty
        
        # Check for invalid loss
        if not np.isfinite(loss):
            if verbose:
                print(f"Non-finite loss detected at iteration {iteration}, resetting to previous beta")
            beta = old_beta
            break
            
        loss_history.append(loss)
        
        # Store iteration information
        if return_iters:
            beta_history.append(beta.copy())
            nonzero_history.append(np.sum(np.abs(beta) > 1e-6))
        
        # Check convergence
        beta_change = np.linalg.norm(beta - old_beta)
        if beta_change < tol:
            if verbose:
                print(f"Converged after {iteration+1} iterations")
            break
        
        # Print progress if verbose
        if verbose and iteration % 100 == 0:
            print(f"Iter {iteration}: Loss = {loss:.4f}, LR = {learning_rate:.6f}, "
                  f"Non-zero = {np.sum(np.abs(beta) > 1e-6)}")
    
    runtime = time.time() - start_time
    
    if verbose:
        print(f"Training completed in {runtime:.4f} seconds")
        print(f"Final loss: {loss_history[-1]:.4f}")
        print(f"Final learning rate: {lr_history[-1]:.6f}")
        print(f"Non-zero coefficients: {np.sum(np.abs(beta) > 1e-6)}")
    
    # Always track non-zero coefficients for the final model
    if not nonzero_history:
        nonzero_history = [np.sum(np.abs(beta) > 1e-6)]
    
    # For consistency with other optimizers
    if return_iters:
        if not beta_history:
            beta_history = [beta.copy()]
        # Add learning rate history to the return tuple
        return beta, loss_history, runtime, nonzero_history, beta_history, lr_history
    else:
        return beta, loss_history, runtime, nonzero_history


def evaluate_lr_schedules(X, y, schedules=True, alphas=None, max_iter=500, verbose=True):
    """
    Compare different learning rate schedules for GLMnet optimization.
    
    """
    if schedules is None:
        schedules = ['constant', 'inverse_time', 'exponential', 'step']
    
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0]
    
    results = {}
    
    for schedule in schedules:
        schedule_results = {}
        
        for alpha in alphas:
            if verbose:
                print(f"\nEvaluating {schedule} schedule with alpha={alpha}")
            
            # Run GLMnet with the current schedule and alpha
            beta, losses, runtime, nonzeros, betas, lrs = glmnet(
                X, y,
                alpha=alpha,
                penalty='elasticnet',
                max_iter=max_iter,
                verbose=verbose,
                lr_schedule=schedule,
                return_iters=True
            )
            
            # Store results
            schedule_results[alpha] = {
                'beta': beta,
                'final_loss': losses[-1],
                'iterations': len(losses),
                'runtime': runtime,
                'nonzero_coefs': nonzeros[-1],
                'loss_history': losses,
                'lr_history': lrs
            }
            
            if verbose:
                print(f"  Final loss: {losses[-1]:.6f}")
                print(f"  Iterations: {len(losses)}")
                print(f"  Runtime: {runtime:.4f} seconds")
                print(f"  Non-zero coefficients: {nonzeros[-1]}")
        
        results[schedule] = schedule_results
    
    return results
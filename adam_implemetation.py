# Adam implementation with L1 or Elastic Net regularization

def adam(X, y, alpha=0.001, beta1=0.9, beta2=0.999, 
         lambda1=0.1, lambda2=0.0, penalty='l1',
         tol=1e-6, max_iter=1000, epsilon=1e-8, verbose=False, return_iters=False):
    """
    Adam optimizer for single-target Poisson regression
    with L1 or Elastic Net regularization
    """
    n_samples, n_features = X.shape
    
    # Initializing coefficient vector
    beta = np.random.normal(0, 0.1, n_features)
    
    # Initializing moment estimates
    m = np.zeros(n_features)  # First moment estimate
    v = np.zeros(n_features)  # Second moment estimate
    
    prev_loss = float('inf')
    loss_history = []
    start_time = time.time()
    
    # Tracking non-zero coefficients
    nonzero_history = []
    
    # Tracking values at each iteration 
    if return_iters:
        beta_history = []
    
    for t in range(1, max_iter + 1):
        # Computing predictions and gradient
        linear_pred = X @ beta
        linear_pred = np.clip(linear_pred, -20, 20)
        mu = np.exp(linear_pred)
        
        # Gradient of negative log-likelihood
        grad_ll = X.T @ (mu - y)
        
        # Adding regularization. 
        if penalty == 'l1':
            # Pure L1: adding subgradient for non-zero elements
            grad = grad_ll + lambda1 * np.sign(beta) * (np.abs(beta) > 0)
        elif penalty == 'elasticnet':
            # Elastic Net: add combined gradient
            grad = grad_ll + lambda1 * np.sign(beta) * (np.abs(beta) > 0) + lambda2 * beta
        else:
            raise ValueError(f"Unknown penalty: {penalty}")
        
        # Update the biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad
        # Update the biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** t)
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2 ** t)
        
        # Update parameters
        beta = beta - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Apply proximal operator for L1 regularization 
        if penalty == 'l1' or penalty == 'elasticnet':
            beta = np.sign(beta) * np.maximum(np.abs(beta) - lambda1 * alpha, 0)
        
        # Compute loss
        ll = poisson_log_likelihood(beta, X, y)
        
        # Adding regularization component to loss
        reg_pen = 0
        if penalty == 'l1':
            reg_pen = lambda1 * np.sum(np.abs(beta))
        elif penalty == 'elasticnet':
            reg_pen = lambda1 * np.sum(np.abs(beta)) + (lambda2 / 2) * np.sum(beta**2)
        
        total_loss = ll + reg_pen
        loss_history.append(total_loss)
        
        # Tracking non-zero coefficients
        non_zeros = np.sum(np.abs(beta) > 1e-6)
        nonzero_history.append(non_zeros)
        
        # Tracking beta values if requested
        if return_iters:
            beta_history.append(beta.copy())
        
        if verbose and t % 100 == 0:
            print(f"Iteration {t}, Loss: {total_loss:.4f}, Log-likelihood: {ll:.4f}, Penalty: {reg_pen:.4f}")
            print(f"Non-zero coefficients: {non_zeros}/{n_features}, Sparsity: {1-non_zeros/n_features:.4f}")
        
        # Check convergence
        if abs(prev_loss - total_loss) < tol:
            if verbose:
                print(f"Converged at iteration {t}")
            break
            
        prev_loss = total_loss
    
    runtime = time.time() - start_time
    
    if return_iters:
        return beta, loss_history, runtime, nonzero_history, beta_history
    else:
        return beta, loss_history, runtime, nonzero_history
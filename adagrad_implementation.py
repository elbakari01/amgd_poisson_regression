
# AdaGrad implementation with L1 or Elastic Net regularization
def adagrad(X, y, alpha=0.01, lambda1=0.1, lambda2=0.0, penalty='l1',
            tol=1e-6, max_iter=1000, epsilon=1e-8, verbose=False, return_iters=False):
    """
    AdaGrad optimizer for single-target Poisson regression
    with L1 or Elastic Net regularization
    """
    n_samples, n_features = X.shape
    
    # Initializing coefficient vector
    beta = np.random.normal(0, 0.1, n_features)
    
    # Initializing accumulator for squared gradients
    G = np.zeros(n_features)
    
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
        
        # Addind regularization to gradient
        if penalty == 'l1':
            # Pure L1:  subgradient of L1 penalty
            grad = grad_ll + lambda1 * np.sign(beta)
        elif penalty == 'elasticnet':
            # Elastic Net:  combined gradient
            grad = grad_ll + lambda1 * np.sign(beta) + lambda2 * beta
        else:
            raise ValueError(f"Unknown penalty: {penalty}")
        
        # Update accumulator
        G += grad ** 2
        
        # Parameter update with AdaGrad scaling
        beta = beta - alpha * grad / (np.sqrt(G) + epsilon)
        
        # Apply proximal operator for L1 regularization if using L1 or ElasticNet
        if penalty == 'l1' or penalty == 'elasticnet':
            beta = np.sign(beta) * np.maximum(np.abs(beta) - lambda1 * alpha / (np.sqrt(G) + epsilon), 0)
        
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
        
        # Tracking beta values 
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


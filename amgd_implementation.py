### Adaptive momentum gradient descent 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy import special
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from scipy.special import expit  
import time

# Ramdom seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

#  matplotlib style
plt.style.use('ggplot')

# clipping function
def clip(x, threshold=None):
    if threshold is None:
        return x
    return np.clip(x, -threshold, threshold)

# Poisson log-likelihood function 
def poisson_log_likelihood(beta, X, y):
    """
    Computing the negative Poisson log-likelihood
    """
    linear_pred = X @ beta
    linear_pred = np.clip(linear_pred, -20, 20)
    mu = np.exp(linear_pred)
    
    log_likelihood = np.sum(y * linear_pred - mu - special.gammaln(y + 1))
    
    return -log_likelihood  

#  metrics function 
def evaluate_model(beta, X, y, target_name='Target'):
    """
    Evaluate model performance for a single target
    """
    linear_pred = X @ beta
    linear_pred = np.clip(linear_pred, -20, 20)
    y_pred = np.exp(linear_pred)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y - y_pred))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    
    # Mean Poisson Deviance
    eps = 1e-10  # To avoid log(0)
    deviance = 2 * np.sum(y * np.log((y + eps) / (y_pred + eps)) - (y - y_pred))
    mean_deviance = deviance / len(y)
    
    results = {
        'MAE': mae,
        'RMSE': rmse,
        'Mean Deviance': mean_deviance,
        'Non-zero coeffs': np.sum(np.abs(beta) > 1e-6),
        'Sparsity': 1.0 - (np.sum(np.abs(beta) > 1e-6) / len(beta))
    }
    
    return results

# AMGD implementation with L1 or Elastic Net regularization
def amgd(X, y, alpha=0.001, beta1=0.9, beta2=0.999, 
         lambda1=0.1, lambda2=0.0, penalty='l1',
         T=100.0, tol=1e-6, max_iter=1000, eta=0.0001, epsilon=1e-8, 
         verbose=False, return_iters=False):
    """
    Adaptive Momentum Gradient Descent (AMGD) 
    """
    n_samples, n_features = X.shape
    
    # Initializing coefficient vector
    beta = np.random.normal(0, 0.1, n_features)
    
    # Initializing momentum variables
    m = np.zeros(n_features)
    v = np.zeros(n_features)
    
    prev_loss = float('inf')
    loss_history = []
    start_time = time.time()
    
    # Tracking non-zero coefficients for debugging
    nonzero_history = []
    
    # Tracking values at each iteration 
    if return_iters:
        beta_history = []
    
    for t in range(1, max_iter + 1):
        alpha_t = alpha / (1 + eta * t)
        
        # Predictions and gradient
        linear_pred = X @ beta
        linear_pred = np.clip(linear_pred, -20, 20)
        mu = np.exp(linear_pred)
        
        # Gradient of negative log-likelihood
        grad_ll = X.T @ (mu - y)
        
        # Adding regularization gradient
        if penalty == 'l1':
            # Pure L1: no gradient term (That is handled in soft thresholding step)
            grad = grad_ll
        elif penalty == 'elasticnet':
            # Elastic Net: add gradient of L2 component
            grad = grad_ll + lambda2 * beta
        else:
            raise ValueError(f"Unknown penalty: {penalty}")
        
        grad = clip(grad, T)
        
        # Momentum updates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Parameter update
        beta = beta - alpha_t * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Apply  regularization
        if penalty == 'l1' or penalty == 'elasticnet':
            # Adaptive soft-thresholding for L1 component
            denom = np.abs(beta) + 0.01
            beta = np.sign(beta) * np.maximum(np.abs(beta) - alpha_t * lambda1 / denom, 0)
        
        # Computing loss
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
        
        # Checking convergence
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
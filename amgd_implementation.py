
### Adaptive momentum gradient descent 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


# Seaborn style and context
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# Matplotlib global settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.title_fontsize': 15,
    'figure.titlesize': 20,
    'lines.linewidth': 3,
    'lines.markersize': 8,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'xtick.major.pad': 6,
    'ytick.major.pad': 6
})

import time
from scipy import special
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from scipy.special import expit  
from matplotlib.ticker import PercentFormatter
import time




RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# matplotlib style
plt.style.use('ggplot')

# Clipping function
def clip(x, threshold=None):
    if threshold is None:
        return x
    return np.clip(x, -threshold, threshold)

#Poisson log-likelihood function 
def poisson_log_likelihood(beta, X, y):
    """
     negative Poisson log-likelihood
    """
    linear_pred = X @ beta
    linear_pred = np.clip(linear_pred, -20, 20)
    mu = np.exp(linear_pred)
    
    log_likelihood = np.sum(y * linear_pred - mu - special.gammaln(y + 1))
    
    return -log_likelihood  # Negative because we want to minimize the function

# Evaluation metrics function 
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
    
#AMGD implementation 
def amgd(X, y, alpha=0.001, beta1=0.8, beta2=0.999, 
         lambda1=0.1, lambda2=0.0, penalty='l1',
         T=20.0, tol=1e-6, max_iter=1000, eta=0.0001, epsilon=1e-8, 
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
        
        # Computing predictions and gradient
        linear_pred = X @ beta
        linear_pred = np.clip(linear_pred, -20, 20)
        mu = np.exp(linear_pred)
        
        # Gradient of negative log-likelihood
        grad_ll = X.T @ (mu - y)
        
        # Adding regularization gradient
        if penalty == 'l1':
            # Pure L1: no gradient term (handled in soft thresholding step)
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
        
        # Apply appropriate regularization
        if penalty == 'l1' or penalty == 'elasticnet':
            # Adaptive soft-thresholding for L1 component
            denom = np.abs(beta) + 0.1
            beta = np.sign(beta) * np.maximum(np.abs(beta) - alpha_t * lambda1 / denom, 0)


        
        # Compute loss
        ll = poisson_log_likelihood(beta, X, y)
        
        # Add regularization component to loss
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

        
# AdaGrad implementation 

def adagrad(X, y, alpha=0.01, lambda1=0.1, lambda2=0.0, penalty='l1',
            tol=1e-6, max_iter=1000, epsilon=1e-8, verbose=False, return_iters=False):
    """
    AdaGrad optimizer 
    """
    n_samples, n_features = X.shape
    
    # Initialize coefficient vector
    beta = np.random.normal(0, 0.01, n_features)  # Smaller initialization
    
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
        
        # Gradient of negative log-likelihood (without regularization)
        grad_ll = X.T @ (mu - y) / n_samples  # Normalize by sample size
        
        # Add L2 regularization gradient (Ridge component for ElasticNet)
        if penalty == 'elasticnet':
            grad = grad_ll + lambda2 * beta
        else:
            grad = grad_ll
        
        # Update accumulator
        G += grad ** 2
        
        # Compute adaptive learning rates
        adaptive_lr = alpha / (np.sqrt(G) + epsilon)
        
        # Parameter update with AdaGrad scaling (without L1 penalty)
        beta_temp = beta - adaptive_lr * grad
        
        # Apply soft thresholding for L1 regularization 
        if penalty == 'l1' or penalty == 'elasticnet':
            # Soft thresholding with adaptive learning rate
            l1_threshold = lambda1 * adaptive_lr
            beta = np.sign(beta_temp) * np.maximum(np.abs(beta_temp) - l1_threshold, 0)
        else:
            beta = beta_temp
        
        # Compute loss
        ll = poisson_log_likelihood(beta, X, y)
        
        # Add regularization component to loss
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
            print(f"Mean adaptive LR: {np.mean(adaptive_lr):.6f}, Min: {np.min(adaptive_lr):.6f}, Max: {np.max(adaptive_lr):.6f}")
        
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


# Adam implementation 
def adam(X, y, alpha=0.001, beta1=0.9, beta2=0.999, 
         lambda1=0.1, lambda2=0.0, penalty='l1',
         tol=1e-6, max_iter=1000, epsilon=1e-8, verbose=False, return_iters=False):
    """
    Adam optimizer 
    """
    n_samples, n_features = X.shape
    
    # Initializing coefficient vector
    beta = np.random.normal(0, 0.1, n_features)
    
    # Initialize moment estimates
    m = np.zeros(n_features)  # First moment estimate
    v = np.zeros(n_features)  # Second moment estimate
    
    prev_loss = float('inf')
    loss_history = []
    start_time = time.time()
    
    # Tracking non-zero coefficients
    nonzero_history = []
    
    # Track values at each iteration 
    if return_iters:
        beta_history = []
    
    for t in range(1, max_iter + 1):
        # Compute predictions and gradient
        linear_pred = X @ beta
        #linear_pred = np.clip(linear_pred, -20, 20)
        mu = np.exp(linear_pred)
        
        # Gradient of negative log-likelihood
        grad_ll = X.T @ (mu - y)
        
        # Add regularization gradient
        if penalty == 'l1':
            # Pure L1:  Subgradient for non-zero elements
            grad = grad_ll + lambda1 * np.sign(beta) * (np.abs(beta) > 0)
        elif penalty == 'elasticnet':
            # Elastic Net:  Combined gradient
            grad = grad_ll + lambda1 * np.sign(beta) * (np.abs(beta) > 0) + lambda2 * beta
        else:
            raise ValueError(f"Unknown penalty: {penalty}")
        
        # Updating biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad
        # Updating biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # Computing bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** t)
        # Computing bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2 ** t)
        
        # Updating parameters
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

##GLM implementation

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
    initial_lr=0.01,  # Reduced from 0.1
    decay_rate=0.01,
    step_size=100,
    step_factor=0.5,
    family='poisson'  # Added family parameter
):
    """
    Glmnet implementation
    """
    # Standardizing features
    if not is_pre_scaled:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Adding intercept column
    if fit_intercept:
        X = np.column_stack([np.ones(X.shape[0]), X])
    
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)
    
    # Computing regularization parameters
    total_penalty = lambda1 + lambda2
    if penalty == 'l1':
        l1_ratio = 1.0
    elif penalty == 'l2':
        l1_ratio = 0.0
    elif penalty == 'elasticnet':
        l1_ratio = lambda1 / total_penalty if total_penalty > 0 else 0.0
    else:
        l1_ratio = 0.0
        total_penalty = 0.0
    
    # Tracking variables
    loss_history = []
    beta_history = []
    nonzero_history = []
    lr_history = []
    start_time = time.time()
    
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
            return initial_lr
    
    def soft_threshold(x, threshold):
        """Soft thresholding operator for L1 penalty"""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    # Main optimization loop
    for iteration in range(max_iter):
        old_beta = beta.copy()
        
        # Compute predictions based on family
        if family == 'gaussian':
            # Linear regression
            mu = X @ beta
            residual = y - mu
            gradient = -X.T @ residual / n_samples
        elif family == 'poisson':
            # Poisson regression
            eta = X @ beta
            eta = np.clip(eta, -20, 20)
            mu = np.exp(eta) + epsilon
            residual = y - mu
            gradient = -X.T @ residual / n_samples
        else:
            raise ValueError(f"Unknown family: {family}")
        
        # Get current learning rate
        learning_rate = get_learning_rate(iteration)
        lr_history.append(learning_rate)
        
        # Add L2 penalty (Ridge component)
        if total_penalty > 0:
            l2_penalty = total_penalty * (1 - l1_ratio)
            l2_grad = l2_penalty * beta
            if fit_intercept:
                l2_grad[0] = 0
            gradient += l2_grad
        
        # Gradient clipping for stability
        max_grad_norm = 5.0
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > max_grad_norm:
            gradient = gradient * (max_grad_norm / grad_norm)
        
        # Update coefficients (without L1 penalty first)
        beta_temp = beta - learning_rate * gradient
        
        # Apply L1 penalty via soft thresholding
        if total_penalty > 0 and l1_ratio > 0:
            l1_threshold = learning_rate * total_penalty * l1_ratio
            beta = soft_threshold(beta_temp, l1_threshold)
            # Don't apply L1 to intercept
            if fit_intercept:
                beta[0] = beta_temp[0]
        else:
            beta = beta_temp
        
        # Compute loss
        if family == 'gaussian':
            data_loss = 0.5 * np.sum((y - X @ beta)**2) / n_samples
        else:  # poisson
            eta = np.clip(X @ beta, -20, 20)
            mu = np.exp(eta) + epsilon
            data_loss = np.sum(mu - y * eta) / n_samples
        
        # Add regularization penalty
        reg_penalty = 0
        if total_penalty > 0:
            l2_penalty = 0.5 * total_penalty * (1 - l1_ratio) * np.sum(beta**2)
            l1_penalty = total_penalty * l1_ratio * np.sum(np.abs(beta))
            if fit_intercept:
                l2_penalty -= 0.5 * total_penalty * (1 - l1_ratio) * beta[0]**2
                l1_penalty -= total_penalty * l1_ratio * np.abs(beta[0])
            reg_penalty = l2_penalty + l1_penalty
        
        loss = data_loss + reg_penalty
        
        # Check for invalid loss
        if not np.isfinite(loss):
            if verbose:
                print(f"Non-finite loss at iteration {iteration}, stopping")
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
        
        # Print progress
        if verbose and iteration % 100 == 0:
            print(f"Iter {iteration}: Loss = {loss:.4f}, LR = {learning_rate:.6f}, "
                  f"Non-zero = {np.sum(np.abs(beta) > 1e-6)}")
    
    runtime = time.time() - start_time
    
    if verbose:
        print(f"Training completed in {runtime:.4f} seconds")
        print(f"Final loss: {loss_history[-1]:.4f}")
        print(f"Non-zero coefficients: {np.sum(np.abs(beta) > 1e-6)}")
    
    if not nonzero_history:
        nonzero_history = [np.sum(np.abs(beta) > 1e-6)]
    
    if return_iters:
        if not beta_history:
            beta_history = [beta.copy()]
        return beta, loss_history, runtime, nonzero_history, beta_history, lr_history
    else:
        return beta, loss_history, runtime, nonzero_history


def evaluate_lr_schedules(X, y, schedules=True, alphas=None, max_iter=500, verbose=True):
    """
    Compare different learning rate schedules for GLMnet optimization.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, shape (n_samples,)
        Target variable (count data).
    schedules : list or None
        List of learning rate schedules to evaluate.
    alphas : list or None
        List of regularization strengths to evaluate.
    max_iter : int
        Maximum number of iterations for each run.
    verbose : bool
        Whether to print results.
        
    Returns:
    --------
    dict
        Results for each schedule and alpha.
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



# Function to preprocess the ecological health dataset
    
def preprocess_ecological_dataset(filepath="ecological_health_dataset.csv"):
    """
    Load and preprocess the ecological health dataset
    
    Parameters:
    -----------
    filepath : str
        Path to the dataset CSV file
    
    Returns:
    --------
    X : numpy.ndarray
        Preprocessed feature matrix
    y : numpy.ndarray
        Biodiversity_Index target variable
    feature_names : list
        Names of the features after preprocessing
    """
    print("Loading and preprocessing the ecological health dataset...")
    
    # Load the CSV file
    df = pd.read_csv(filepath)
    
    # Displaying basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Target variable distribution:\n{df['Biodiversity_Index'].value_counts().sort_index().head()}")
    
    # Remove  timestamp column if it exists
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])
    
    # Checking for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("Missing values detected. Filling with appropriate values...")
        # Filling numeric columns with median
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Filling categorical columns with mode
        categorical_cols = ['Pollution_Level', 'Ecological_Health_Label']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    # Identifying categorical columns for encoding
    categorical_cols = [col for col in ['Pollution_Level', 'Ecological_Health_Label'] if col in df.columns]
    
    # Creating preprocessor with column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), 
             [col for col in df.columns if col not in categorical_cols + ['Biodiversity_Index']]),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ],
        remainder='drop'
    )
    
    # Extracting features and target
    X = df.drop(columns=['Biodiversity_Index'])
    y = df['Biodiversity_Index'].values
    
    # Fitting and transforming the features
    X_processed = preprocessor.fit_transform(X)
    
    # feature names after preprocessing
    numeric_cols = [col for col in df.columns if col not in categorical_cols + ['Biodiversity_Index']]
    
    #  one-hot encoded feature names
    ohe = preprocessor.named_transformers_['cat']
    cat_features = []
    for i, col in enumerate(categorical_cols):
        categories = ohe.categories_[i][1:]  
        cat_features.extend([f"{col}_{cat}" for cat in categories])
    
    feature_names = numeric_cols + cat_features
    
    print(f"Processed features shape: {X_processed.shape}")
    print(f"Target variable shape: {y.shape}")
    
    return X_processed, y, feature_names
    
## Cross validation

def k_fold_cross_validation(X_val, y_val, k=5, lambda_values=None, seed=42, optimizers_to_use=None):
    """
    Perform k-fold cross-validation to find optimal parameters for selected optimizers
    and regularization types
    
    Parameters:
    -----------
    X_val : numpy.ndarray
        Validation feature matrix
    y_val : numpy.ndarray
        Validation target values
    k : int
        Number of folds for cross-validation
    lambda_values : list
        List of lambda values to try
    seed : int
        Random seed for reproducibility
    optimizers_to_use : list or None
        List of optimizer names to evaluate. If None, use all available optimizers.
    
    Returns:
    --------
    best_params : dict
        Dictionary with best parameters for each optimizer and metric
    cv_results : pd.DataFrame
        DataFrame with all cross-validation results
    """
    if lambda_values is None:
        #lambda_values = list(np.logspace(-4, 1, 50))
        lambda_values = np.logspace(-4, np.log10(20), 50)

    # Defining all available optimizers with their base configurations
    all_optimizers = {
        "AMGD": {
            "func": amgd,
            "base_params": {"alpha": 0.01, "beta1": 0.9, "beta2": 0.999, "T": 20.0, 
                          "tol": 1e-6, "max_iter": 1000, "eta": 0.0001, "epsilon": 1e-8}
        },
        "Adam": {
            "func": adam,
            "base_params": {"alpha": 0.01, "beta1": 0.9, "beta2": 0.999, 
                          "tol": 1e-6, "max_iter": 1000, "epsilon": 1e-8}
        },
        "AdaGrad": {
            "func": adagrad,
            "base_params": {"alpha": 0.01, "tol": 1e-6, "max_iter": 1000, "epsilon": 1e-8}
        },
        "GLMnet": {
            "func": glmnet,
            "base_params": {"alpha": 0.01, "tol": 1e-6, "max_iter": 1000, "epsilon": 1e-8}
        }
    }
    
    # Filter optimizers
    if optimizers_to_use is not None:
        optimizers = {name: config for name, config in all_optimizers.items() 
                     if name in optimizers_to_use}
    else:
        optimizers = all_optimizers
    
    # Regularization types
    regularizations = ["L1", "ElasticNet"]
    
    # Set up k-fold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    
    # Store results
    cv_results = []
    
    # Loop through all combinations
    for optimizer_name, optimizer_info in optimizers.items():
        for reg_type in regularizations:
            for lambda_val in lambda_values:
                print(f"Evaluating {optimizer_name} with {reg_type} regularization, lambda={lambda_val}")
                
                # Prepare parameters based on regularization type
                params = optimizer_info["base_params"].copy()
                
                if reg_type == "L1":
                    params["lambda1"] = lambda_val
                    params["lambda2"] = 0.0
                    params["penalty"] = "l1"
                elif reg_type == "ElasticNet":
                    params["lambda1"] = lambda_val / 2  # Distribute lambda between L1 and L2
                    params["lambda2"] = lambda_val / 2
                    params["penalty"] = "elasticnet"
                
                # Metrics for this combination across all folds
                fold_maes = []
                fold_rmses = []
                fold_deviances = []
                fold_runtimes = []
                fold_sparsities = []
                
                # Run k-fold cross-validation
                for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_val)):
                    X_fold_train, X_fold_test = X_val[train_idx], X_val[test_idx]
                    y_fold_train, y_fold_test = y_val[train_idx], y_val[test_idx]
                    
                    # Train the model on this fold - handle different return value counts
                    try:
                        # gett all 5 return values
                        beta, loss_history, runtime, nonzero_history, _ = optimizer_info["func"](
                            X_fold_train, y_fold_train, **params, verbose=False, return_iters=False
                        )
                    except ValueError:
                        
                        beta, loss_history, runtime, nonzero_history = optimizer_info["func"](
                            X_fold_train, y_fold_train, **params, verbose=False, return_iters=False
                        )
                    
                    # Evaluating on the hold-out fold
                    metrics = evaluate_model(beta, X_fold_test, y_fold_test)
                    
                    # Store metrics
                    fold_maes.append(metrics['MAE'])
                    fold_rmses.append(metrics['RMSE'])
                    fold_deviances.append(metrics['Mean Deviance'])
                    fold_runtimes.append(runtime)
                    fold_sparsities.append(metrics['Sparsity'])
                
                # Calculating average metrics across folds
                avg_mae = np.mean(fold_maes)
                avg_rmse = np.mean(fold_rmses)
                avg_deviance = np.mean(fold_deviances)
                avg_runtime = np.mean(fold_runtimes)
                avg_sparsity = np.mean(fold_sparsities)
                
                # Calculating standard deviations
                std_mae = np.std(fold_maes)
                std_rmse = np.std(fold_rmses)
                std_deviance = np.std(fold_deviances)
                
                # Storing the results
                result = {
                    "Optimizer": optimizer_name,
                    "Regularization": reg_type,
                    "Lambda": lambda_val,
                    "MAE": avg_mae,
                    "MAE_std": std_mae,
                    "RMSE": avg_rmse,
                    "RMSE_std": std_rmse,
                    "Mean Deviance": avg_deviance,
                    "Mean Deviance_std": std_deviance,
                    "Runtime": avg_runtime,
                    "Sparsity": avg_sparsity
                }
                
                cv_results.append(result)
    
    # Converting results to DataFrame
    cv_results_df = pd.DataFrame(cv_results)
    
    # Finding best parameters for each metric
    best_params = {}
    
    for optimizer_name in optimizers.keys():
        optimizer_results = cv_results_df[cv_results_df['Optimizer'] == optimizer_name]
        
        # Finding best parameters for MAE
        best_mae_idx = optimizer_results['MAE'].idxmin()
        best_params[f"{optimizer_name}_MAE"] = {
            "Optimizer": optimizer_name,
            "Regularization": optimizer_results.loc[best_mae_idx, 'Regularization'],
            "Lambda": optimizer_results.loc[best_mae_idx, 'Lambda'],
            "Metric_Value": optimizer_results.loc[best_mae_idx, 'MAE']
        }
        
        # Finding best parameters for RMSE
        best_rmse_idx = optimizer_results['RMSE'].idxmin()
        best_params[f"{optimizer_name}_RMSE"] = {
            "Optimizer": optimizer_name,
            "Regularization": optimizer_results.loc[best_rmse_idx, 'Regularization'],
            "Lambda": optimizer_results.loc[best_rmse_idx, 'Lambda'],
            "Metric_Value": optimizer_results.loc[best_rmse_idx, 'RMSE']
        }
        
        # Finding best parameters for Mean Deviance
        best_dev_idx = optimizer_results['Mean Deviance'].idxmin()
        best_params[f"{optimizer_name}_Mean_Deviance"] = {
            "Optimizer": optimizer_name,
            "Regularization": optimizer_results.loc[best_dev_idx, 'Regularization'],
            "Lambda": optimizer_results.loc[best_dev_idx, 'Lambda'],
            "Metric_Value": optimizer_results.loc[best_dev_idx, 'Mean Deviance']
        }
        
        # Finding best parameters for Runtime (fastest)
        best_runtime_idx = optimizer_results['Runtime'].idxmin()
        best_params[f"{optimizer_name}_Runtime"] = {
            "Optimizer": optimizer_name,
            "Regularization": optimizer_results.loc[best_runtime_idx, 'Regularization'],
            "Lambda": optimizer_results.loc[best_runtime_idx, 'Lambda'],
            "Metric_Value": optimizer_results.loc[best_runtime_idx, 'Runtime']
        }
        
        # Finding best parameters for Sparsity
        best_sparsity_idx = optimizer_results['Sparsity'].idxmax()
        best_params[f"{optimizer_name}_Sparsity"] = {
            "Optimizer": optimizer_name,
            "Regularization": optimizer_results.loc[best_sparsity_idx, 'Regularization'],
            "Lambda": optimizer_results.loc[best_sparsity_idx, 'Lambda'],
            "Metric_Value": optimizer_results.loc[best_sparsity_idx, 'Sparsity']
        }
    
    # Finding the overall best parameters
    mae_results = [params for key, params in best_params.items() if key.endswith('_MAE')]
    best_mae_params = min(mae_results, key=lambda x: x['Metric_Value'])
    best_params['Overall_Best_MAE'] = best_mae_params
    
    rmse_results = [params for key, params in best_params.items() if key.endswith('_RMSE')]
    best_rmse_params = min(rmse_results, key=lambda x: x['Metric_Value'])
    best_params['Overall_Best_RMSE'] = best_rmse_params
    
    deviance_results = [params for key, params in best_params.items() if key.endswith('_Mean_Deviance')]
    best_deviance_params = min(deviance_results, key=lambda x: x['Metric_Value'])
    best_params['Overall_Best_Mean_Deviance'] = best_deviance_params
    
    runtime_results = [params for key, params in best_params.items() if key.endswith('_Runtime')]
    best_runtime_params = min(runtime_results, key=lambda x: x['Metric_Value'])
    best_params['Overall_Best_Runtime'] = best_runtime_params
    
    sparsity_results = [params for key, params in best_params.items() if key.endswith('_Sparsity')]
    best_sparsity_params = max(sparsity_results, key=lambda x: x['Metric_Value'])
    best_params['Overall_Best_Sparsity'] = best_sparsity_params
    
    # Printing a summary of best lambda values for each optimizer and metric
    print("\nSummary of best lambda values from cross-validation:")
    print("{:<15} {:<15} {:<15} {:<15} {:<15}".format("Optimizer", "MAE Lambda", "RMSE Lambda", "Deviance Lambda", "Sparsity Lambda"))
    print("-" * 75)
    
    for optimizer in optimizers.keys():
        mae_lambda = best_params[f"{optimizer}_MAE"]["Lambda"]
        rmse_lambda = best_params[f"{optimizer}_RMSE"]["Lambda"]
        dev_lambda = best_params[f"{optimizer}_Mean_Deviance"]["Lambda"]
        spar_lambda = best_params[f"{optimizer}_Sparsity"]["Lambda"]
        
        print("{:<15} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(
            optimizer, mae_lambda, rmse_lambda, dev_lambda, spar_lambda))
    
    print("\nOverall best parameters:")
    for metric in ["MAE", "RMSE", "Mean_Deviance", "Runtime", "Sparsity"]:
        params = best_params[f"Overall_Best_{metric}"]
        print(f"Best for {metric}: {params['Optimizer']} with {params['Regularization']} (λ={params['Lambda']:.6f})")
    
    return best_params, cv_results_df


#Function to train all optimizers

def train_all_optimizers(X_train, y_train, best_params, metric='MAE'):
    """
    Train all optimizer models using their best parameters from cross-validation
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training feature matrix
    y_train : numpy.ndarray
        Training target values
    best_params : dict
        Dictionary with best parameters for each optimizer and metric
    metric : str
        Metric to use for selecting the best parameters ('MAE', 'RMSE', 'Mean_Deviance', 'Runtime', 'Sparsity')
    
    Returns:
    --------
    model_results : dict
        Dictionary with trained models, their coefficients, and performance metrics
    """
    optimizers = ['AMGD', 'Adam', 'AdaGrad', 'GLMnet']
    model_results = {}
    
    for optimizer_name in optimizers:
        # Get best parameters for this optimizer and metric
        params = best_params[f"{optimizer_name}_{metric}"]
        
        reg_type = params['Regularization']
        lambda_val = params['Lambda']
        
        print(f"\nTraining {optimizer_name} with best parameters for {metric}:")
        print(f"  Regularization: {reg_type}")
        print(f"  Lambda: {lambda_val}")
        
        # Setting uptimizer function and parameters
        if optimizer_name == "AMGD":
            optimizer_func = amgd
            base_params = {"alpha": 0.01, "beta1": 0.9, "beta2": 0.999, "T": 20.0, 
                          "tol": 1e-6, "max_iter": 1000, "eta": 0.0001, "epsilon": 1e-8}
        elif optimizer_name == "Adam":
            optimizer_func = adam
            base_params = {"alpha": 0.01, "beta1": 0.9, "beta2": 0.999, 
                          "tol": 1e-6, "max_iter": 1000, "epsilon": 1e-8}
        elif optimizer_name == "AdaGrad":
            optimizer_func = adagrad
            base_params = {"alpha": 0.01, "tol": 1e-6, "max_iter": 1000, "epsilon": 1e-8}
        else:  # GLMnet
            optimizer_func = glmnet
            base_params = {"alpha": 0.01, "tol": 1e-6, "max_iter": 1000, "epsilon": 1e-8}
        
        # Configure regularization parameters
        if reg_type == "L1":
            base_params["lambda1"] = lambda_val
            base_params["lambda2"] = 0.0
            base_params["penalty"] = "l1"
        else:  # ElasticNet
            base_params["lambda1"] = lambda_val / 2
            base_params["lambda2"] = lambda_val / 2
            base_params["penalty"] = "elasticnet"
        
        # Train the model with tracking of per-iteration values
        try:
            # Try for GLMnet which returns 6 values when return_iters=True
            if optimizer_name == "GLMnet":
                beta, loss_history, runtime, nonzero_history, beta_history, lr_history = optimizer_func(
                    X_train, y_train, **base_params, verbose=True, return_iters=True
                )
            else:
                # For AMGD, Adam, and AdaGrad which return 5 values
                beta, loss_history, runtime, nonzero_history, beta_history = optimizer_func(
                    X_train, y_train, **base_params, verbose=True, return_iters=True
                )
                lr_history = None  # These optimizers don't return learning rate
        except ValueError as e:
            print(f"Error with {optimizer_name}: {e}")
            print("Trying alternative return value handling...")
            
            # Fallback approach
            result = optimizer_func(X_train, y_train, **base_params, verbose=True, return_iters=True)
            
            # Extracting values based on the length of the result tuple
            if len(result) == 6:  # GLMnet with lr_history
                beta, loss_history, runtime, nonzero_history, beta_history, lr_history = result
            elif len(result) == 5:  # AMGD, Adam, AdaGrad
                beta, loss_history, runtime, nonzero_history, beta_history = result
                lr_history = None
            else:
                # Handle any other unexpected return value count
                beta = result[0]
                loss_history = result[1] if len(result) > 1 else []
                runtime = result[2] if len(result) > 2 else 0
                nonzero_history = result[3] if len(result) > 3 else []
                beta_history = result[4] if len(result) > 4 else []
                lr_history = result[5] if len(result) > 5 else None
        
        # Evaluating model on training set
        train_metrics = evaluate_model(beta, X_train, y_train)
        
        # Storing results
        model_results[optimizer_name] = {
            'beta': beta,
            'loss_history': loss_history,
            'runtime': runtime,
            'nonzero_history': nonzero_history,
            'beta_history': beta_history,
            'lr_history': lr_history,
            'train_metrics': train_metrics,
            'params': base_params
        }
        
        print(f"  Training complete in {runtime:.2f} seconds")
        print(f"  Training MAE: {train_metrics['MAE']:.4f}")
        print(f"  Training RMSE: {train_metrics['RMSE']:.4f}")
        print(f"  Training Mean Deviance: {train_metrics['Mean Deviance']:.4f}")
        print(f"  Sparsity: {train_metrics['Sparsity']:.4f}")
    
    return model_results


def create_algorithm_comparison_plots(cv_results_df):
    """
    Create barplots comparing the performance of AMGD, AdaGrad, Adam, and GLMnet algorithms
    across different metrics with L1 and ElasticNet regularization only.
    
    Parameters:
    -----------
    cv_results_df : pandas.DataFrame
        DataFrame with cross-validation results
    
    Returns:
    --------
    figs : list
        List of matplotlib figures
    """
    # Create a list to store the figures
    figs = []
    
    # Metrics to compare
    metrics = ['MAE', 'RMSE', 'Mean Deviance', 'Runtime']
    
    #Best result for each optimizer and metric combination
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
    
    # Converting to DataFrame 
    best_results_df = pd.DataFrame(best_results)
    
    # Creating a barplot for each metric
    for metric in metrics:
        metric_df = best_results_df[best_results_df['Metric'] == metric]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Colors for each optimizer
        colors = {'AMGD': '#3498db', 'AdaGrad': '#2ecc71', 'Adam': '#e74c3c', 'GLMnet': '#9b59b6'}
        bar_colors = [colors[opt] for opt in metric_df['Optimizer']]
        
        # Creating barplot
        bars = ax.bar(metric_df['Optimizer'], metric_df['Value'], color=bar_colors)
        
        # Adding value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02),
                   f'{height:.4f}', ha='center', va='bottom', fontsize=12)
        
        # Adding regularization and lambda information below bars
        for i, (_, row) in enumerate(metric_df.iterrows()):
            ax.text(i, 0, f"{row['Regularization']}\nλ={row['Lambda']:.4f}", 
                   ha='center', va='bottom', fontsize=8, color='black',
                   transform=ax.get_xaxis_transform())
        
        # Set title and labels
        ax.set_title(f'Best {metric} Comparison across Optimizers (L1/ElasticNet)', fontsize=16)
        ax.set_ylabel(metric, fontsize=15)
        ax.set_xlabel('Optimizer', fontsize=15)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        
        plt.ylim(0, metric_df['Value'].max() * 1.15)
        plt.tight_layout()
        
        figs.append(fig)
    
    return figs

# Function to compare convergence rates
def compare_convergence_rates(X_train, y_train, best_params):
    """
    Compare convergence rates of optimization algorithms
    
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
                "alpha": 0.01, "beta1": 0.9, "beta2": 0.999, "T": 20.0, 
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
    
    # Create comprehensive convergence plot
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
    
    plt.title('Convergence Rate Comparison (Log Scale)', fontsize=16)
    plt.xlabel('Percentage of Max Iterations (%)', fontsize=15)
    plt.ylabel('Loss (log scale)', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Optimizer', loc='best')
    
    # Normalized convergence plot (if we have multiple valid histories)
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
        
        plt.title('Normalized Convergence Rate Comparison', fontsize=16)
        plt.xlabel('Percentage of Max Iterations (%)', fontsize=15)
        plt.ylabel('Normalized Loss', fontsize=15)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Optimizer', loc='best')
    
    plt.tight_layout()
    
    return plt.gcf()



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
    # Runing a basic model to identify important features
    params = {
        "alpha": 0.01, 
        "beta1": 0.9, 
        "beta2": 0.999, 
        "lambda1": 0.1,
        "lambda2": 0.0,
        "penalty": "l1",
        "T": 20.0, 
        "tol": 1e-6, 
        "max_iter": 200,  # Reduced for quicker execution
        "eta": 0.0001, 
        "epsilon": 1e-8,
        "verbose": False
    }
    
    initial_beta, _, _, _ = amgd(X_train, y_train, **params)
    
    # Finding top 10 features by coefficient magnitude
    importance = np.abs(initial_beta)
    top_indices = np.argsort(importance)[-17:]  # Top 10 features
    top_feature_names = [feature_names[i] for i in top_indices]
    
    # 5. Creating figure for the coefficient paths
    fig, axes = plt.subplots(4, 2, figsize=(18, 20), sharex=True)
    fig.suptitle('Coefficient Paths for Biodiversity Prediction: L1/ElasticNet Regularization', fontsize=16)
    
    # Configure plot settings
    optimizers = ['AMGD', 'Adam', 'AdaGrad', 'GLMnet'] 
    penalty_types = ['l1', 'elasticnet']
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_indices)))
    
    # 6. Plot coefficient paths for each optimizer and regularization type
    for i, optimizer_name in enumerate(optimizers):
        for j, penalty in enumerate(penalty_types):
            ax = axes[i, j]
            
            # Storage for coefficient values at each lambda
            coef_paths = []
            
            # Running optimization for each lambda value
            for lambda_val in lambda_values:
                if optimizer_name == 'AMGD':
                    params = {
                        "alpha": 0.01, 
                        "beta1": 0.9, 
                        "beta2": 0.999, 
                        "T": 20.0, 
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
                        # unpacking 5 values using only the first (beta)
                        beta, _, _, _, _ = glmnet(X_train, y_train, **params, return_iters=True)
                    except ValueError:
                        # If only 4 values returned
                        beta, _, _, _ = glmnet(X_train, y_train, **params, return_iters=False)
                
                # Extracting coefficients for the top features only
                selected_coeffs = [beta[idx] for idx in top_indices]
                coef_paths.append(selected_coeffs)
            
            # Converting to numpy array for easier manipulation
            coef_paths = np.array(coef_paths)
            
            # Plotting coefficient paths for top features
            for idx, feature_idx in enumerate(range(len(top_indices))):
                ax.plot(lambda_values, coef_paths[:, idx], 
                        color=colors[idx], 
                        label=top_feature_names[idx],
                        linewidth=2)
            
            # labels and title
            ax.set_xscale('log')
            ax.set_xlabel('Regularization Strength (λ)' if i == 3 else '')  
            ax.set_ylabel('Coefficient Value' if j == 0 else '')
            ax.set_title(f'{optimizer_name} - {penalty.capitalize()} Regularization')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # legend 
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.08),
               title='Features', ncol=5, frameon=True)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.96])
    plt.savefig('ecological_coefficient_paths.png', dpi=300, bbox_inches='tight')
    plt.show()

    #plot_coefficient_evolution_for_ecological_data(X_train, y_train, top_indices, top_feature_names)


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
    
    
    plt.figure(figsize=(15, 10))
    
    # 1. Plot loss histories (convergence) for all optimizers
    plt.subplot(2, 2, 1)
    
    colors = {'AMGD': '#3498db', 'Adam': '#e74c3c', 'AdaGrad': '#2ecc71', 'GLMnet': '#9b59b6'}
    
    for optimizer_name, results in model_results.items():
        loss_history = results['loss_history']
        iterations = np.arange(1, len(loss_history) + 1)
        plt.semilogy(iterations, loss_history, label=f"{optimizer_name}", color=colors[optimizer_name], linewidth=2)
    
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Loss (log scale)', fontsize=14)
    plt.title('Training Loss Convergence', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    # 2. Plot training metrics for all optimizers
    plt.subplot(2, 2, 2)
    optimizer_names = list(model_results.keys())
    train_metrics = [model_results[opt]['train_metrics'][metric_to_plot] for opt in optimizer_names]
    
    # Bar chart for training metrics
    bars = plt.bar(optimizer_names, train_metrics, color=[colors[opt] for opt in optimizer_names])

    #Value label
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02),
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.ylabel(f'Training {metric_to_plot}', fontsize=14)
    plt.title(f'Training {metric_to_plot} by Optimizer', fontsize=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. Plotting test metrics for all optimizers
    plt.subplot(2, 2, 3)
    test_metric_values = [test_metrics[opt][metric_to_plot] for opt in optimizer_names]
    
    # Bar chart for test metrics
    bars = plt.bar(optimizer_names, test_metric_values, color=[colors[opt] for opt in optimizer_names])
    
    # Value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02),
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.ylabel(f'Test {metric_to_plot}', fontsize=12)
    plt.title(f'Test {metric_to_plot} by Optimizer', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 4. Plotting train vs test comparison
    plt.subplot(2, 2, 4)
    x = np.arange(len(optimizer_names))
    width = 0.35
    
    #Grouped bar chart
    bars1 = plt.bar(x - width/2, train_metrics, width, label=f'Training {metric_to_plot}', alpha=0.7)
    bars2 = plt.bar(x + width/2, test_metric_values, width, label=f'Test {metric_to_plot}', alpha=0.7)
    
    plt.xlabel('Optimizer', fontsize=14)
    plt.ylabel(metric_to_plot, fontsize=14)
    plt.title(f'Training vs Test {metric_to_plot} Comparison', fontsize=16)
    plt.xticks(x, optimizer_names)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # figure for the feature importance comparison
    plt.figure(figsize=(16, 6))
    
    # Subplots for each optimizer's feature importance 
    for i, optimizer_name in enumerate(optimizer_names, 1):
        plt.subplot(1, len(optimizer_names), i)
        
        #Beta coefficients for this optimizer
        beta = model_results[optimizer_name]['beta']
        
        # Calculating feature importance based on absolute coefficient values
        importance = np.abs(beta)
        indices = np.argsort(importance)[::-1]
        
        #Top N features
        top_n = min(10, len(importance))
        
        # Get feature names 
        feature_indices = indices[:top_n]
        feature_labels = [f"Feature {idx}" for idx in feature_indices]  
        
        # Plot importance
        plt.barh(range(top_n), importance[feature_indices], align='center', color=colors[optimizer_name])
        plt.yticks(range(top_n), feature_labels)
        plt.xlabel('Coefficient Magnitude')
        plt.title(f'{optimizer_name} Feature Importance')
        
    plt.tight_layout()
    plt.show()

    #Non-zero features evolution plot
    plt.figure(figsize=(12, 6))
    
    for optimizer_name, results in model_results.items():
        nonzero_history = results['nonzero_history']
        iterations = np.arange(1, len(nonzero_history) + 1)
        plt.plot(iterations, nonzero_history, label=f"{optimizer_name}", color=colors[optimizer_name], linewidth=2)
    
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Number of Non-Zero Coefficients', fontsize=14)
    plt.title('Sparsity Evolution During Training', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
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
    
    
    if metrics_to_compare is None:
        metrics_to_compare = ['MAE', 'RMSE', 'Mean Deviance', 'Sparsity']
    
    optimizer_names = list(model_results.keys())
    
    data = []
    for optimizer in optimizer_names:
        row = []
        for metric in metrics_to_compare:
            if metric in test_metrics[optimizer]:
                # For all metrics except Sparsity, lower is better
                if metric != 'Sparsity':
                    row.append(test_metrics[optimizer][metric])
                else:
                    # For Sparsity, higher is better (inverted)
                    row.append(1 - test_metrics[optimizer][metric])
        data.append(row)
    
    # Converting to numpy array
    data = np.array(data)
    
    # Normalize the data between 0 and 1 for radar chart
    data_normalized = np.zeros_like(data, dtype=float)
    for i in range(len(metrics_to_compare)):
        if metrics_to_compare[i] != 'Sparsity':
            # For error metrics, smaller is better
            data_normalized[:, i] = (data[:, i] - np.min(data[:, i])) / (np.max(data[:, i]) - np.min(data[:, i]) + 1e-10)
        else:
            # For sparsity===> inverted, so smaller is better
            data_normalized[:, i] = (data[:, i] - np.min(data[:, i])) / (np.max(data[:, i]) - np.min(data[:, i]) + 1e-10)
    
    # Number of variables
    N = len(metrics_to_compare)
    
    # figure for the radar chart
    plt.figure(figsize=(10, 10))
    
    # Plotting the radar chart
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    ax = plt.subplot(111, polar=True)
    
    # Adding variable labels
    plt.xticks(angles[:-1], metrics_to_compare, size=12)
    
    # y-labels (percentages)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # colors
    colors = {'AMGD': '#3498db', 'Adam': '#e74c3c', 'AdaGrad': '#2ecc71', 'GLMnet': '#9b59b6'}
    for i, optimizer in enumerate(optimizer_names):
        values = data_normalized[i].tolist()
        values += values[:1]  # Close the polygon
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=optimizer, color=colors[optimizer])
        ax.fill(angles, values, alpha=0.1, color=colors[optimizer])
    
    #Legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Optimizer Performance Comparison\n(Closer to center is better)", size=15)
    
    plt.tight_layout()
    plt.show()
    
    # summary table for runtime comparison
    runtimes = [model_results[opt]['runtime'] for opt in optimizer_names]
    iterations = [len(model_results[opt]['loss_history']) for opt in optimizer_names]
    
    plt.figure(figsize=(12, 6))
    
    # Runtime comparison
    plt.subplot(1, 2, 1)
    
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

## Statististical significance analysis 


def statistical_significance_analysis(X, y, best_params, n_bootstrap=1000, n_runs=100, random_state=42):
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
            base_params = {"alpha": 0.01, "beta1": 0.9, "beta2": 0.999, "T": 20.0, 
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
    Add statistical significance analysis to pipeline with fixed feature_selection handling
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
    plt.title('Feature Selection Consistency across Optimizers')
    
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
        plt.title(f'Distribution of {metric} across Runs')
        
        # If this is sparsity, highlight that higher is better
        if metric == 'Sparsity':
            plt.ylabel(f'{metric} (higher is better)')
        else:
            plt.ylabel(f'{metric} (lower is better)')
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('metric_distributions.png', dpi=300)
    plt.show()
    
    
    return significance_results

def statistical_significance_analysis(X, y, best_params, n_bootstrap=500, n_runs=100, random_state=42):
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
            base_params = {"alpha": 0.01, "beta1": 0.9, "beta2": 0.999, "T": 20.0, 
                          "tol": 1e-6, "max_iter": 100, "eta": 0.0001, "epsilon": 1e-8}
        elif optimizer_name == "Adam":
            base_params = {"alpha": 0.01, "beta1": 0.9, "beta2": 0.999, 
                          "tol": 1e-6, "max_iter": 100, "epsilon": 1e-8}
        elif optimizer_name == 'GLMnet':
            base_params = {"alpha": 0.01, "tol": 1e-6, "max_iter": 100, "epsilon": 1e-8}
        else:  # AdaGrad
            base_params = {"alpha": 0.01, "tol": 1e-6, "max_iter": 100, "epsilon": 1e-8}
        
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
    feature_selection_data = {opt: [] for opt in optimizers}  # Store feature selection data
    
    for run in range(n_runs):
        print(f"Running statistical analysis iteration {run+1}/{n_runs}")
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
            feature_selection_data[optimizer_name].append(np.abs(beta) > 1e-6)
    
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
            effect_size = mean_diff / (pooled_std + 1e-8)  # Add small epsilon to avoid division by zero
            
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
        # Check if feature_selection_data[optimizer_name] is not empty
        if feature_selection_data[optimizer_name]:
            # Calculate the mean of the feature selection across all runs
            feature_selection_probs = np.mean(feature_selection_data[optimizer_name], axis=0)
        else:
            # If no data is available, initialize with zeros
            feature_selection_probs = np.zeros(n_features)
        
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


# Main function to run pipeline 
def run_poisson_regression_pipeline(filepath="ecological_health_dataset.csv", 
                                   random_state=42, k_folds=5,
                                   lambda_values=None, 
                                   metric_to_optimize='MAE',
                                   plot_coefficients=True,
                                   plot_metrics=True,
                                   optimizers=None):
    """
    Run the Poisson regression pipeline with L1 and ElasticNet regularization only:
    1. Load and preprocess data
    2. Split data into train, validation, and test sets
    3. Find optimal parameters using k-fold cross-validation on validation set
    4. Create comparison barplots for the algorithms
    5. Train model on training set using best parameters
    6. Compare convergence rates of the algorithms
    7. Evaluate model on test set
    8. Plot coefficient paths and evolution (if enabled)
    9. Plot comprehensive metric visualizations (if enabled)
    
    Parameters:
    -----------
    filepath : str
        Path to the dataset CSV file
    random_state : int
        Random seed for reproducibility
    k_folds : int
        Number of folds for cross-validation
    lambda_values : list, optional
        List of lambda values to try
    metric_to_optimize : str
        Metric to use for selecting best parameters ('MAE', 'RMSE', 'Mean_Deviance', 'Runtime')
    plot_coefficients : bool
        Whether to generate coefficient path and evolution plots
    plot_metrics : bool
        Whether to generate comprehensive metric visualizations
    optimizers : list, optional
        List of optimizers to use ('AMGD', 'Adam', 'AdaGrad', 'GLMnet'). If None, all optimizers will be used.
    
    Returns:
    --------
    best_params : dict
        Best parameters found through cross-validation
    test_metrics : dict
        Evaluation metrics on test set
    model_results : dict
        Results for all models including coefficients and performance
    """
    print("=" * 80)
    print("POISSON REGRESSION PIPELINE (L1 AND ELASTICNET)")
    print("=" * 80)
    
    # Set default optimizers if not provided
    if optimizers is None:
        optimizers = ['AMGD', 'Adam', 'AdaGrad', 'GLMnet']
    
    # Validating optimizers
    valid_optimizers = ['AMGD', 'Adam', 'AdaGrad', 'GLMnet']
    for opt in optimizers:
        if opt not in valid_optimizers:
            raise ValueError(f"Invalid optimizer: {opt}. Valid options are: {valid_optimizers}")
    
    # 1. Load and preprocess data
    print("\nStep 1: Loading and preprocessing data")
    X, y, feature_names = preprocess_ecological_dataset(filepath)
    
    # 2. Splitting data into train, validation, and test sets (70/15/15)
    print("\nStep 2: Splitting data into train, validation, and test sets (70/15/15)")
    # First split: 85% train+val, 15% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state
    )
    
    # Second split: 70% train, 15% validation (82.35% of train_val is train)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]:.1%})")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/X.shape[0]:.1%})")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]:.1%})")
    
    # 3. Find optimal parameters using k-fold cross-validation on validation set
    print(f"\nStep 3: Finding optimal parameters using {k_folds}-fold cross-validation on validation set")
    print(f"Regularization with L1 and ElasticNet using optimizers: {optimizers}")
    
    if lambda_values is None:
        lambda_values = np.logspace(-4, np.log10(20), 20)  

        #lambda_values = np.logspace(-4, 1, 6)  # [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
        print(f"Using lambda values: {lambda_values}")
    
    # optimizers_to_use
    best_params, cv_results_df = k_fold_cross_validation(
        X_val, y_val, k=k_folds, lambda_values=lambda_values, seed=random_state,
        optimizers_to_use=optimizers  # Fixed: optimizers -> optimizers_to_use
    )
    
    # Print best parameters
    print("\nBest parameters found through cross-validation:")
    for metric, params in best_params.items():
        if metric.startswith('Overall_Best_'):
            metric_name = metric.replace('Overall_Best_', '')
            print(f"Best for {metric_name}: {params['Optimizer']} with {params['Regularization']} (λ={params['Lambda']:.6f}), Value: {params['Metric_Value']:.6f}")
    
    # 4.  comparison barplots for the algorithms
    print("\nStep 4: Creating algorithm performance comparison plots")
    comparison_plots = create_algorithm_comparison_plots(cv_results_df)
    
    # Display comparison plots
    for fig in comparison_plots:
        plt.figure(fig.number)
        plt.show()
    
    # 5. Train model on training set using best parameters
    print(f"\nStep 5: Training model on training set using best parameters for {metric_to_optimize}")
    model_results = train_all_optimizers(
        X_train, y_train, best_params, metric=metric_to_optimize
    )
    
    # Selecting the best optimizer based on the chosen metric
    best_optimizer = best_params[f'Overall_Best_{metric_to_optimize}']['Optimizer']
    print(f"Using {best_optimizer} model for test evaluation (best for {metric_to_optimize})")
    
    # 6. Comparing convergence rates of the algorithms
    print("\nStep 6: Comparing convergence rates of optimization algorithms")
    # Check if compare_convergence_rates accepts an optimizers parameter
    try:
        convergence_plot = compare_convergence_rates(
            X_train, y_train, best_params
        )
    except TypeError:
        # If the function doesn't accept an optimizers parameter, catch the error
        print("Note: Using all optimizers for convergence rate comparison")
        convergence_plot = compare_convergence_rates(
            X_train, y_train, best_params
        )
    
    plt.figure(convergence_plot.number)
    plt.show()
    
    # 7. Evaluating all models on test set
    print("\nStep 7: Evaluating all models on test set")
    
    test_metrics = {}
    
    # Evaluating each optimizer with its best parameters
    for optimizer_name, results in model_results.items():
        # Skip optimizers that weren't requested
        if optimizer_name not in optimizers:
            continue
            
        beta = results['beta']
        metrics = evaluate_model(beta, X_test, y_test, target_name=optimizer_name)
        test_metrics[optimizer_name] = metrics
        
        print(f"\nTest metrics for {optimizer_name}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
                
    # Creating a comparison table
    comparison_metrics = ['MAE', 'RMSE', 'Mean Deviance', 'Sparsity']
    comparison_data = []
    
    for optimizer, metrics in test_metrics.items():
        row = [optimizer]
        for metric in comparison_metrics:
            row.append(metrics[metric])
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data, columns=['Optimizer'] + comparison_metrics)
    print("\nModel comparison on test set:")
    print(comparison_df)
    
    # Highlight the best model according to the chosen metric
    best_idx = comparison_df[metric_to_optimize if metric_to_optimize in comparison_metrics else 'MAE'].idxmin()
    best_test_optimizer = comparison_df.iloc[best_idx]['Optimizer']
    print(f"\nBest model on test set for {metric_to_optimize}: {best_test_optimizer}")
    
    # Visualizing feature importance for the best model
    best_beta = model_results[best_test_optimizer]['beta']
    
    top_n = 15
    importance = np.abs(best_beta)
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Feature Importances ({best_test_optimizer} model)')
    plt.bar(range(min(top_n, len(feature_names))), 
           importance[indices[:top_n]], 
           align='center')
    plt.xticks(range(min(top_n, len(feature_names))), 
               [feature_names[i] for i in indices[:top_n]], 
               rotation=90)
    plt.tight_layout()
    plt.show()
    
    # 8. Plot coefficient paths and evolution 
    if plot_coefficients:
        print("\nStep 8: Plotting coefficient paths and evolution")
        # Get top indices for plotting
        top_indices = indices[:17]  # Top 17 features
        top_feature_names = [feature_names[i] for i in top_indices]
        
        # Plot coefficient paths
        print("Plotting coefficient paths...")
        try:
            # try with the optimizers parameter
            plot_coefficient_paths_for_ecological_data()
        except Exception as e:
            print(f"Note: Could not use optimizers parameter for plotting. Using all optimizers. Error: {e}")
            plot_coefficient_paths_for_ecological_data()
    
    # 9. Plot comprehensive metric visualizations 
    if plot_metrics:
        print("\nStep 9: Creating comprehensive metric visualizations")
        
        # Add feature names to model_results for plotting
        for optimizer_name in model_results:
            model_results[optimizer_name]['feature_names'] = feature_names
        
        # Plot training and test metrics
        print("Plotting training and test metrics comparison...")
        plot_training_and_test_metrics(model_results, test_metrics, metric_to_plot=metric_to_optimize)
        
        # Plot radar chart and performance comparison
        print("Plotting optimizer performance comparison...")
        plot_optimizer_comparison(model_results, test_metrics)

      # 8. Statistical Significance Analysis
        print("\nStep 8: Performing Statistical Significance Analysis")
        significance_results = add_to_pipeline(
        X, y, 
        best_params, 
        feature_names, 
        model_results
        )
    
    print("\nPipeline completed successfully!")
    
    # Return best parameters, test metrics for all models, and all model results
    return best_params, test_metrics, model_results, significance_results



# execute the pipeline
if __name__ == "__main__":
    
    #lambda_values = np.logspace(-4, 1, 2)  # Creates 50 values between 10^-4 and 10^1
    lambda_values = np.logspace(-4, np.log10(20), 50)  
    
    try:
        best_params, test_metrics, model_results, significance_results = run_poisson_regression_pipeline(
        filepath="ecological_health_dataset.csv",
        random_state=42,
        k_folds=5,
        lambda_values=lambda_values,
        metric_to_optimize='MAE'
                                )
        
        
        print("\nAnalysis complete. Summary of test metrics:")
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
    except FileNotFoundError:
        print(f"Error: Dataset file not found.")
        print("Please ensure the dataset file is in the correct location.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


        
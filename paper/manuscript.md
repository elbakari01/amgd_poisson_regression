%% 
%% Copyright 2007-2025 Elsevier Ltd
%% 
%% This file is part of the 'Elsarticle Bundle'.
%% ---------------------------------------------
%% 
%% It may be distributed under the conditions of the LaTeX Project Public
%% License, either version 1.3 of this license or (at your option) any
%% later version.  The latest version of this license is in
%%    http://www.latex-project.org/lppl.txt
%% and version 1.3 or later is part of all distributions of LaTeX
%% version 1999/12/01 or later.
%% 
%% The list of all files belonging to the 'Elsarticle Bundle' is
%% given in the file `manifest.txt'.
%% 
%% Template article for Elsevier's document class `elsarticle'
%% with harvard style bibliographic references

%\documentclass[12pt]{elsarticle}
%\usepackage[numbers,sort&compress]{natbib}
%\documentclass[12pt,numbers]{elsarticle}
\documentclass[preprint,12pt]{elsarticle}



%% Use the option review to obtain double line spacing
%% \documentclass[authoryear,preprint,review,12pt]{elsarticle}

%% Use the options 1p,twocolumn; 3p; 3p,twocolumn; 5p; or 5p,twocolumn
%% for a journal layout:
%% \documentclass[final,1p,times,authoryear]{elsarticle}
%% \documentclass[final,1p,times,twocolumn,authoryear]{elsarticle}
%% \documentclass[final,3p,times,authoryear]{elsarticle}
%% \documentclass[final,3p,times,twocolumn,authoryear]{elsarticle}
%% \documentclass[final,5p,times,authoryear]{elsarticle}
%% \documentclass[final,5p,times,twocolumn,authoryear]{elsarticle}

%% For including figures, graphicx.sty has been loaded in
%% elsarticle.cls. If you prefer to use the old commands
%% please give \usepackage{epsfig}

%% The amssymb package provides various useful mathematical symbols
%\usepackage{amssymb}
%% The amsmath package provides various useful equation environments.
%\usepackage{amsmath}
%% The amsthm package provides extended theorem environments
%% \usepackage{amsthm}
\usepackage{amssymb}
\usepackage{amsmath, amssymb, amsthm} % Add amsthm for theorem environment
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}
\usepackage{caption}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage[ruled,vlined]{algorithm2e}

\usepackage[colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue]{hyperref}
\usepackage[hypcap=false]{caption}


%\usepackage{algorithm}
%\documentclass{article}
%\usepackage{algorithmicx}
\usepackage{algpseudocode}
%\usepackage{algpseudocode}  % Correct package for algorithmic environment
\usepackage{placeins}
\usepackage{float}
%% The lineno packages adds line numbers. Start line numbering with
%% \begin{linenumbers}, end it with \end{linenumbers}. Or switch it on
%% for the whole article with \linenumbers.
%% \usepackage{lineno}

\journal{Expert Systems with Applications}

\begin{document}

\begin{frontmatter}

%% Title, authors and addresses

%% use the tnoteref command within \title for footnotes;
%% use the tnotetext command for theassociated footnote;
%% use the fnref command within \author or \affiliation for footnotes;
%% use the fntext command for theassociated footnote;
%% use the corref command within \author for corresponding author footnotes;
%% use the cortext command for theassociated footnote;
%% use the ead command for the email address,
%% and the form \ead[url] for the home page:
%% \title{Title\tnoteref{label1}}
%% \tnotetext[label1]{}
%% \author{Name\corref{cor1}\fnref{label2}}
%% \ead{email address}
%% \ead[url]{home page}
%% \fntext[label2]{}
%% \cortext[cor1]{}
%% \affiliation{organization={},
%%            addressline={}, 
%%            city={},
%%            postcode={}, 
%%            state={},
%%            country={}}
%% \fntext[label3]{}

\title{Adaptive Momentum Gradient Descent: A New Algorithm in Regularized Poisson Regression} %% Article title

%% use optional labels to link authors explicitly to addresses:
%% \author[label1,label2]{}
%% \affiliation[label1]{organization={},
%%             addressline={},
%%             city={},
%%             postcode={},
%%             state={},
%%             country={}}
%%
%% \affiliation[label2]{organization={},
%%             addressline={},
%%             city={},
%%             postcode={},
%%             state={},
%%             country={}}


\author[label1]{Ibrahim Bakari} %% Author name
\ead{2020913072@ogr.cu.edu.tr}

\author[label2]{M. Revan Özkale}
\ead{mrevan@cu.edu.tr}

%% Author affiliation
\affiliation[label1]{organization={Çukurova University, Faculty of Science and Letters, Department of Statistics},%Department and Organization
            %addressline={}, 
            city={Adana},
            postcode={01330}, 
            country={Türkiye} }


%% Abstract
\begin{abstract}
  In this paper, we propose Adaptive Momentum Gradient Descent (AMGD), a novel optimization algorithm for regularized Poisson regression in high-dimensional settings. AMGD integrates adaptive learning rates, momentum updates, and adaptive soft-thresholding to address critical limitations of existing methods such as AdaGrad (rapid learning rate decay) and Adam (gradient instability in sparse data). Our analysis demonstrates that AMGD achieves significant performance improvements over competing methods  comprehensive numerical experiments: it reduces Mean Absolute Error (MAE) by approximately 2.7\% compared to Adam and 56.6\% compared to AdaGrad, while also achieving faster convergence and better sparsity induction. Specifically, AMGD achieves meaningful feature selection (23-29\% sparsity) without sacrificing predictive performance, making it particularly effective for high-dimensional data where identifying relevant predictors is crucial. Additionally, AMGD exhibits superior computational efficiency, reaching near-optimal solutions in fewer iterations.
\end{abstract}

%%Graphical abstract
%\begin{graphicalabstract}
%\includegraphics{grabs}hhhh
%\end{graphicalabstract}

%%Research highlights
%\begin{highlights}
  % \item We propose a novel Adaptive Momentum Gradient Descent (AMGD) algorithm that integrates adaptive learning rates, momentum updates, and adaptive soft-thresholding specifically designed for regularized Poisson regression.
  % \item AMGD demonstrates superior performance with 56.6\% reduction in Mean Absolute Error compared to AdaGrad and 2.7\% improvement over Adam, while achieving meaningful sparsity (35.29\%) through effective feature selection.
  % \item The adaptive soft-thresholding approach directly incorporates L1 regularization into the optimization process, providing both theoretical convergence guarantees and empirical advantages in high-dimensional Poisson regression settings.
 %\item Showed robust statistical significance: Statistical tests revealed extremely low p-values (p<0.0001) and large effect sizes (Cohen's d up to -713) when comparing AMGD to other optimization methods  multiple performance metrics.\end{highlights}

\begin{highlights}
\item Propose AMGD: a new optimizer for regularized Poisson regression.
\item AMGD improves MAE by 56.6\% over AdaGrad and 2.7\% over Adam.
\item AMGD employs adaptive soft-thresholding into $L_1$-regularized Poisson models
\item AMGD converges near the optimum at rate $O(1/\sqrt{T})$ after $T$ iterations. 
\item AMGD shows strong statistical significance with low p-values and large effect sizes.
\end{highlights}


%% Keywords
\begin{keyword}
%% keywords here, in the form: keyword \sep keyword
Adam \sep AdaGrad \sep Poisson regression \sep regularization \sep sparsity

\end{keyword}

\end{frontmatter}

%% Add \usepackage{lineno} before \begin{document} and uncomment 
%% following line to enable line numbers
%% \linenumbers

%% main text

\section{Introduction}
\label{sec:intro}

Generalized linear models (GLMs) are widely used for modeling non-Gaussian response variables across diverse fields such as epidemiology, insurance, and recommendation systems. Among them, Poisson regression is a foundational model for count data, arising naturally in applications such as ecological studies, disease incidence modeling, and web traffic analysis. However, two well-known challenges often compromise its reliability: multicollinearity among predictors and high-dimensional feature spaces where the number of variables exceeds the number of observations. In such scenarios, maximum likelihood estimation becomes unstable, leading to overfitting and poor model interpretability.

Regularization techniques address these issues by shrinking coefficients and enabling variable selection. The Lasso \citep{tibshirani1996lasso} promotes sparsity by driving some coefficients to zero, but struggles when predictors are highly correlated. Ridge regression \citep{hoerl1970ridge} handles multicollinearity well by shrinking coefficients uniformly, but does not induce sparsity. Elastic Net \citep{zou2005elastic} combines both penalties to balance shrinkage and variable selection, though it often requires expensive hyperparameter tuning and can suffer from numerical challenges in high-dimensional settings.

GLMNet \citep{friedman2010regularization} is one of the most popular implementations of regularized GLMs, employing coordinate descent to compute regularization paths. While widely used and generally stable, it may encounter numerical issues in Poisson regression settings with extreme values in the linear predictor, or in sparse and highly correlated datasets where coordinate updates become unstable \citep{simon2011regularization, zou2009adaptive}. Additionally, coordinate descent may converge slowly in large-scale settings with strongly coupled parameters.

Adaptive gradient methods such as AdaGrad \citep{duchi2011adaptive} and Adam \citep{kingma2014adam} have been applied to improve convergence and robustness. However, AdaGrad suffers from excessively diminishing learning rates, and Adam can exhibit instability in sparse high-dimensional settings, particularly with zero-inflated or overdispersed Poisson data \citep{cameron2013count, park2008bayesian}. These limitations highlight the need for optimization techniques that are both statistically stable and computationally efficient in complex Poisson regression environments.

To address these challenges, we propose Adaptive Momentum Gradient Descent (AMGD), a novel optimization framework that integrates adaptive learning rates, momentum terms, and adaptive soft-thresholding. This combination enables AMGD to overcome the limitations of both GLMNet and existing adaptive optimizers, particularly in high-dimensional, sparse, and multicollinear data environments.

Our key contributions include the development of AMGD, a novel optimizer that combines momentum, adaptive learning rates, and soft-thresholding; theoretical analysis of convergence under convexity assumptions; and empirical validation demonstrating superior accuracy, sparsity, and computational efficiency compared to existing methods.

The remainder of this paper is organized as follows: Section~\ref{sec:related} provides a comprehensive review of related work. Section~\ref{sec:glm} reviews the GLM and Poisson regression framework. Section~\ref{sec:amgd} introduces the AMGD algorithm and its theoretical properties. Section~\ref{sec:experiments} presents empirical evaluations. Section~\ref{sec:discussions} discusses the experimental findings and their implications. Section~\ref{sec:conclusion} concludes with a summary of contributions and future directions.


\section{Related Work}
\label{sec:related}  %Done 
The development of stable and efficient optimization algorithms for regularized GLMs has received significant attention in recent years. This section reviews advances in adaptive gradient methods, momentum-based optimization, and hybrid control-theoretic approaches.

Building on similar work in adaptive optimization \citep{duchi2011adaptive, kingma2014adam}, recent work has investigated stochastic gradient descent (SGD) variants that dynamically adjust step sizes based on historical gradients \citep{sun2020novel}. These methods often yield better empirical performance but require careful control of adaptivity to ensure convergence. Non-ergodic convergence results under smoothness assumptions have highlighted both their strengths and theoretical limitations.

In the context of regularized GLMs, coordinate descent methods \citep{friedman2010regularization} have been widely adopted for their computational efficiency, though they face challenges in high-dimensional Poisson regression settings with extreme predictor values or strong parameter coupling. Incremental and aggregated gradient methods have been proposed to improve scalability in large datasets. A proximal incremental aggregated gradient (PIAG) framework was introduced that unified multiple existing approaches and established the first non-ergodic $\mathcal{O}(1/k)$ convergence rate under mild conditions \citep{sun2019general} where $k$ is the iteration number. This framework allows for significantly larger step sizes compared to earlier bounds and uses Lyapunov-based techniques to extend results to nonconvex objectives.

Momentum-based optimization has also seen renewed interest with theoretical backing. Recent analyses have established convergence guarantees for Heavy-ball methods under general convexity, and even linear rates under restricted strong convexity \citep{sun2019non}. These results provide a deeper understanding of how momentum can improve convergence speed in first-order methods, especially when gradient landscapes are poorly conditioned.

Recent advances have also explored integrating control theory with stochastic optimization. Proportional-integral-derivative (PID) control mechanisms have been incorporated into SGD to dynamically adjust updates and reduce long-term error \citep{li2025learning}. Extensions using fuzzy logic to adapt PID parameters have demonstrated improved convergence in high-dimensional tasks \citep{yuan2024fuzzy}. In addition, parallelized momentum-based SGD has been applied effectively to recommendation problems \citep{luo2020efficient}, showing the value of combining adaptivity with distributed computation.

Despite these advances, existing methods either struggle with sparsity induction in adaptive settings or lack theoretical guarantees for non-smooth regularized objectives in count data models. Our proposed AMGD algorithm draws inspiration from these foundational works while addressing these limitations. It integrates adaptive learning rates \citep{sun2020novel}, incremental update principles \citep{sun2019general}, and momentum acceleration techniques \citep{sun2019non}. In contrast to prior approaches that use subgradient approximations for $\ell_1$ regularization, AMGD introduces an adaptive soft-thresholding mechanism directly into the optimization step. This combination enables more stable convergence, improved sparsity control, and robust performance in high-dimensional Poisson regression tasks.


\section{Generalized Linear Models}
\label{sec:glm}

GLMs are flexible frameworks for modeling univariate response variables that follow distributions from the exponential family, including many common distributions such as normal, binomial, gamma, or Poisson. GLMs extend traditional linear regression by providing a link function that connects the mean of the response variable to a linear predictor, allowing non-linear relationships to be modeled. This flexibility makes GLMs popular in fields such as medicine, finance, and engineering.

One application of GLMs is Poisson regression, which is used to model count data where the outcomes are non-negative integers, often representing the number of events occurring in a fixed period. It is widely used in fields such as epidemiology (to model disease incidence rates) and transportation (to analyze traffic flow data).

\subsection{Poisson Regression}

Let $y_i$, for $i = 1, \dots, n$, be independent observations drawn from a Poisson distribution. The probability mass function (PMF) of the Poisson distribution is given by:
\begin{equation}
f_{Y_i}(y_i) = \frac{e^{-\mu_i} \mu_i^{y_i}}{y_i!}, \quad y_i = 0, 1, 2, \dots, \quad i = 1,\dots, n.
\end{equation}

The expected value and variance of $Y_i$ are given by $E(Y_i) = \mu_i > 0$ and $\mathrm{Var}(Y_i) = \mu_i$.

In the GLM framework, the mean of $Y_i$ is linked to a set of explanatory variables $\mathbf{x}_i = (x_{i1}, \dots, x_{ip})^\top$ through a link function. For Poisson regression, the canonical link function is the logarithmic function, defined as:
\begin{equation}
g(\mu_i) = \log(\mu_i), \quad \text{or equivalently,} \quad \mu_i = \exp(\eta_i) = \exp(\mathbf{x}_i^\top \boldsymbol{\beta}),
\label{eq:poisson_mean}
\end{equation}
where $\eta_i$ is the linear predictor and $\boldsymbol{\beta} = (\beta_1, \dots, \beta_p)^\top$ is the vector of regression coefficients.

The log-likelihood function for the Poisson regression model, based on the independent sample of size $n$, is given in Equation(~\ref{eq:poisson_loglik}):
\begin{equation}
    \ell(\boldsymbol{\beta}) = \sum_{i=1}^n \left( y_i \mathbf{x}_i^\top \boldsymbol{\beta} - e^{\mathbf{x}_i^\top \boldsymbol{\beta}} - \log(y_i!) \right).
    \label{eq:poisson_loglik}
\end{equation}

Since the score equations derived from Equation(~\ref{eq:poisson_loglik}) are nonlinear in $\boldsymbol{\beta}$, an iterative Newton-Raphson method is commonly used to solve them. The iterative update formula is given by:
\begin{equation}
    \hat{\boldsymbol{\beta}}^{(t+1)} = \left(X^\top \hat{W}^{(t)} X\right)^{-1} X^\top \hat{W}^{(t)} z^{(t)},
\end{equation}
where $X$ is the $n \times p$ matrix of explanatory variables, $\hat{W}^{(t)} = \mathrm{diag}(1/w_{ii})$ with $w_{ii} = \mu_i$, and $z^{(t)}$ is a working response vector of size $n \times 1$ with elements:
$$
z^{(t)}_i = \sum_{j=1}^{p} x_{ij} \hat\beta^{(t)}_j + (y_i - \hat\mu_i^{(t)}) \frac{\partial \eta_i^{(t)}}{\partial \mu_i^{(t)}},
$$
all evaluated at the current iterate $\hat{\boldsymbol{\beta}}^{(t)}$.

Once convergence is achieved, the maximum likelihood estimate is computed as:
\begin{equation}
    \hat{\boldsymbol{\beta}}_{\text{ML}} = \left( X^\top \hat{W}_{\text{ML}} X \right)^{-1} X^\top \hat{W}_{\text{ML}} \hat{z}.
\end{equation}

Although $\hat{\boldsymbol{\beta}}_{\text{ML}}$ is asymptotically normal with distribution $N\left(\boldsymbol{\beta}, (X^\top \hat{W}_{\text{ML}} X)^{-1}\right)$, its performance degrades in high-dimensional settings and under multicollinearity. In such cases, the estimate becomes unstable and its variance inflates. To overcome these limitations, regularization is introduced to the likelihood framework in Equation(~\ref{eq:poisson_loglik}).

\subsection{Penalized Poisson Regression}

To address instability and overfitting in high-dimensional or multicollinear data, a penalty term is added to the log-likelihood function. The penalized objective function is defined as:
\begin{equation}
    f(\boldsymbol{\beta}) = -\ell(\boldsymbol{\beta}) + \lambda P(\boldsymbol{\beta}), \label{eq:penalized_objective}
\end{equation}
where $\ell(\boldsymbol{\beta})$ is the log-likelihood function in Equation(~\ref{eq:poisson_loglik}), $P(\boldsymbol{\beta})$ is a penalty function and $\lambda \geq 0$ is a regularization parameter.

The Lasso ($L_1$-norm) penalty is defined as:
\begin{equation}
    P(\boldsymbol{\beta}) = \|\boldsymbol{\beta}\|_1 = \sum_{j=1}^p |\beta_j|, \label{eq:L1penalty}
\end{equation}
which induces sparsity by shrinking some coefficients exactly to zero.

The Ridge ($L_2$-norm) penalty is defined as:
\begin{equation}
    P(\boldsymbol{\beta}) = \|\boldsymbol{\beta}\|_2^2 = \sum_{j=1}^p \beta_j^2, \label{eq:L2penalty}
\end{equation}
which shrinks all coefficients but retains all predictors.

The Elastic Net penalty combines both $L_1$ and $L_2$ norms:
\begin{equation}
    P(\boldsymbol{\beta}) = \lambda_1 \|\boldsymbol{\beta}\|_1 + \frac{\lambda_2}{2} \|\boldsymbol{\beta}\|_2^2.
\end{equation}

The optimization problem defined in Equation(~\ref{eq:penalized_objective}) is convex but generally lacks a closed-form solution. Numerical algorithms such as coordinate descent \citep{friedman2010regularization}, gradient descent \citep{beck2009fast}, and proximal gradient methods \citep{parikh2014proximal} are commonly used for $L_1$-based penalties. These methods are essential due to the non-differentiability of the $L_1$ term \citep{hastie2015statistical}, unlike Ridge regression, which under certain assumptions permits closed-form solutions \citep{hastie2009elements}.


\begin{table}[htbp]
\centering
\caption{Summary of Key Notation}
\label{tab:notation}
\begin{tabular}{cl}
\hline
\textbf{Symbol} & \textbf{Description} \\
\hline
$\beta$ & Coefficient vector in Poisson regression \\
$\beta_j$ & $j$-th element of coefficient vector $\beta$ \\
$\beta^{(t)}$ & Coefficient vector at iteration $t$ \\$\ell(\beta)$ & The log-likelihood function for Poisson regression \\
$\mu_i$ & Predicted mean for sample $i$, computed as $\exp(\mathbf{x}_i^\top \beta)$ \\
$\mathbf{x}_i$ & the $i$-th explanatory variable\\
$y_i$ & Observed count for sample $i$ \\
$X$ & Design matrix of size $n \times p$, $n$: number of samples, $p$: number of features \\
$\lambda_1$ & Regularization parameter for $L_1$ penalty \\
$\lambda_2$ & Regularization parameter for $L_2$ penalty \\
$\alpha$ & Initial learning rate \\
$\alpha_t$ & Learning rate at iteration $t$, defined as $\frac{\alpha}{1 + \eta t}$ \\
$\eta$ & Learning rate decay parameter \\
$\zeta_1$ & Decay rate for first moment estimate (momentum term) \\
$\zeta_2$ & Decay rate for second moment estimate  \\
$m_t$ & First moment estimate (moving average of gradients) at iteration $t$ \\
$v_t$ & Second moment estimate (moving average of squared gradients) at iteration $t$ \\
$\hat{m}_t$ & Bias-corrected first moment estimate \\
$\hat{v}_t$ & Bias-corrected second moment estimate \\
$\epsilon$ & Small constant to prevent division by zero in updates \\
$T$ & Gradient clipping threshold \\
$\text{grad}$ & Gradient of negative log-likelihood with respect to $\beta$ \\
$\text{MAE}$ & Mean Absolute Error \\
$\text{RMSE}$ & Root Mean Squared Error \\
$\text{MPD}$ & Mean Poisson Deviance \\
$\text{Sparsity}$ & Percentage of coefficients driven to exactly zero \\
$\text{prox}_{\lambda \|\cdot\|_1}$ & Proximal operator associated with the $\ell_1$ norm \\
$\max(a, b)$ & Element-wise maximum between $a$ and $b$ \\
$\min(a, b)$ & Element-wise minimum between $a$ and $b$ \\
$\text{clip}(g, T)$ & Function that limits gradient magnitude to threshold $T$ \\
$\text{sign}(\beta_j)$ & Sign of coefficient $\beta_j$: $+1$ if positive, $-1$ if negative, 0 if zero \\
$\odot $   & Element-wise (Hadamard) product between vectors \\
$\mathcal{O}(\cdot)$ & Big-O notation for computational complexity \\
\hline
\end{tabular}
\end{table}

\section{Adaptive Momentum Gradient Descent Algorithm}
\label{sec:amgd}

In this section, we introduce AMGD, a novel first-order optimization algorithm that integrates  gradient clipping, adaptive learning rates, momentum, and sparsity-promoting thresholding into a unified framework. The algorithm is motivated by known weaknesses of Adam \citep{kingma2014adam} and AdaGrad \citep{duchi2011adaptive}, particularly their instability or inefficiency in sparse high-dimensional settings such as Poisson regression.

A key design feature of AMGD is the integration of gradient clipping, which plays a critical role in maintaining numerical stability during optimization. This mechanism is particularly important in Poisson regression due to the exponential nature of its canonical link function, $\mu = \exp(X\beta)$. When the linear predictor $X\beta$ becomes large, even small parameter changes can cause exponentially amplified updates, resulting in large gradients that may lead to numerical overflow. To mitigate this, AMGD applies element-wise clipping to gradients exceeding a threshold $T$:
\begin{equation*}
\text{clip}(g_j) = \max(-T, \min(g_j, T)),
\end{equation*}
preventing any single gradient component from destabilizing the optimization process. Additionally, we clip the linear predictor prior to applying the exponential transformation, which regularizes both the predicted mean and the loss surface. These dual-level safeguards make AMGD more resilient to outliers and model mismatch, common issues in ecological and zero-inflated datasets.

Unlike adaptive optimizers such as Adam and AdaGrad, which lack built-in clipping mechanisms, AMGD explicitly incorporates this step to address the heavy-tailed gradients observed in sparse count data. While Adam adjusts learning rates based on gradient history, it may still permit harmful updates when gradients are extreme. In contrast, AMGD halts such updates at the source, ensuring stable convergence under challenging conditions. This approach aligns with established practices in deep learning, where gradient clipping is widely used to prevent training instability in noisy settings \citep{pascanu2013difficulty}. Empirically, removing gradient clipping leads to rapid divergence, with loss values exceeding $10^6$ within a few iterations. Based on cross-validation and stability testing, we set the clipping threshold at $T = 10$, which consistently stabilizes training across both Lasso and ElasticNet regularization settings.

AMGD incorporates adaptive momentum updates using exponential moving averages of both gradients and squared gradients, following the Adam framework. Let $m_t$ and $v_t$ denote the biased estimates of the first and second moments. The update rule scales gradients as $m_t / (\sqrt{v_t} + \epsilon)$, where $\epsilon$ prevents division by zero. Bias correction is applied to account for initialization, enabling efficient traversal of the loss surface and reducing oscillations in noisy regions. To ensure convergence, we introduce a decaying learning rate schedule $\alpha_t = \alpha / (1 + \eta t)$. Unlike Adam's fixed base learning rate, this schedule progressively reduces step sizes, balancing exploration and exploitation while improving stability in both convex and non-convex settings.

To promote sparsity, AMGD applies an adaptive soft-thresholding step after each parameter update:
\begin{equation*}
\beta_j \leftarrow \text{sign}(\beta_j) \max\left(|\beta_j| - \frac{\lambda}{|\beta_j| + \epsilon}, 0\right),
\end{equation*}
where $\epsilon = 0.01$ prevents division by zero. This operator dynamically adapts the shrinkage rate: smaller coefficients are penalized more heavily, encouraging sparsity, while larger coefficients remain relatively unshrunk. This mechanism mimics the Adaptive Lasso \citep{zou2006adaptive} and enables AMGD to perform embedded variable selection during the optimization process.

Together, these components form a unified optimization framework that enhances convergence stability, induces sparsity, and ensures reliable performance in high-dimensional Poisson regression. The integration of gradient clipping, adaptive momentum, learning rate decay, and soft-thresholding creates a robust algorithm that addresses the specific challenges of sparse count data modeling. The full procedure is outlined in Algorithm~\ref{alg:amgd}.


\begin{algorithm}[H]
\caption{AMGD: Adaptive Momentum Gradient Descent}
\label{alg:amgd}
\SetAlgoLined
\small  
\KwIn{Training set $(X, y)$; learning rate $\alpha$; momentum parameters $\zeta_1, \zeta_2$; regularization parameters $\lambda_1, \lambda_2$; penalty type; gradient clipping threshold $T$; tolerance $tol$; maximum iteration count $M$; decay rate $\eta$; small constant $\epsilon$}
\KwOut{Optimized parameter vector $\boldsymbol{\beta}$}
$\boldsymbol{\beta} \leftarrow \boldsymbol{\beta}^{(0)}$, $m \leftarrow 0$, $v \leftarrow 0$, $\text{prev\_loss} \leftarrow \infty$ \tcp*{Initialization}
\For{$t = 1, 2, \dots, M$}{
    $\alpha_t \leftarrow \frac{\alpha}{1 + \eta t}$ \tcp*{Adaptive learning rate}
    $\text{linear\_pred} \leftarrow X\boldsymbol{\beta}$\;
    $\text{linear\_pred} \leftarrow \text{clip}(\text{linear\_pred}, -20, 20)$\;
    $\mu \leftarrow \exp(\text{linear\_pred})$ \tcp*{Mean response}
    $\text{grad\_ll} \leftarrow X^T (\mu - y)$ \tcp*{Gradient of negative log-likelihood}
    
    \eIf{penalty $=$ `L1'}{
        $\text{grad} \leftarrow \text{grad\_ll}$ \tcp*{Pure $L_1$: handled in soft-thresholding}
    }{
        \If{penalty $=$ `elasticnet'}{
            $\text{grad} \leftarrow \text{grad\_ll} + \lambda_2 \boldsymbol{\beta}$ \tcp*{Add $L_2$ gradient}
        }
    }
    
    $\text{grad} \leftarrow \text{clip}(\text{grad}, T)$ \tcp*{Apply gradient clipping}
    $m \leftarrow \zeta_1 m + (1 - \zeta_1) \text{grad}$ \tcp*{Momentum update}
    $v \leftarrow \zeta_2 v + (1 - \zeta_2) (\text{grad})^2$ \tcp*{Squared gradient update}
    $\hat{m} \leftarrow \frac{m}{1 - \zeta_1^t}$, $\hat{v} \leftarrow \frac{v}{1 - \zeta_2^t}$ \tcp*{Bias correction}
    $\boldsymbol{\beta} \leftarrow \boldsymbol{\beta} - \frac{\alpha_t \hat{m}}{\sqrt{\hat{v}} + \epsilon}$ \tcp*{Parameter update}
    
    \If{penalty $=$ `L1' or penalty $=$ `elasticnet'}{
        $\text{denom} \leftarrow |\boldsymbol{\beta}| + 0.01$\;
        $\boldsymbol{\beta} \leftarrow \text{sign}(\boldsymbol{\beta}) \cdot \max(|\boldsymbol{\beta}| - \frac{\alpha_t \lambda_1}{\text{denom}}, 0)$ \tcp*{Adaptive soft-thresholding}
    }
    
    $\text{LL} \leftarrow \text{poisson\_log\_likelihood}(\boldsymbol{\beta}, X, y)$\;
    
    \eIf{penalty $=$ `L1'}{
        $\text{reg\_pen} \leftarrow \lambda_1 \sum |\beta_j|$\;
    }{
        \If{penalty $=$ `elasticnet'}{
            $\text{reg\_pen} \leftarrow \lambda_1 \sum |\beta_j| + \frac{\lambda_2}{2} \sum \beta_j^2$\;
        }
    }
    
    $\text{total\_loss} \leftarrow \text{LL} + \text{reg\_pen}$ \tcp*{Compute full objective}
    
    \If{$|\text{prev\_loss} - \text{total\_loss}| < tol$}{
        \tcp*{Stopping criterion}
        \textbf{break}\;
    }
    
    $\text{prev\_loss} \leftarrow \text{total\_loss}$ \tcp*{Update previous loss}
}
\Return{$\boldsymbol{\beta}$} \tcp*{Optimized parameters}
\end{algorithm}

The computational cost per iteration  of AMGD is \(O(np)\), driven by computing \(X^\top(\mu - y)\), matching the complexity of coordinate descent (CD) and proximal gradient methods. Momentum and soft-thresholding add \(O(p)\), yielding linear scaling with data size, optimal for full-batch methods. Unlike CD, which requires multiple \(O(np)\) cycles, AMGD uses the full gradient each iteration. Vectorized implementations can exploit BLAS routines \citep{lawson1979basic}, offering speed advantages when \(p\) is large.


\subsection{Theoretical Foundation of Adaptive Soft-Thresholding in AMGD}
\label{sec:soft_threshold_theory}

A key theoretical innovation of the proposed AMGD algorithm lies in its principled integration of adaptive soft-thresholding within the optimization process for $L_1$-regularized Poisson regression. Unlike Adam and AdaGrad, which apply adaptive gradient updates without explicitly accounting for nonsmooth regularization terms, AMGD directly incorporates a modified thresholding operator that generalizes the classical proximal mapping used in sparse regression. This enhances both algorithmic stability and sparsity enforcement.

Consider the regularized objective function Equation (~\ref{eq:penalized_objective}) with the $L_1$-norm penalty (~\ref{eq:L1penalty}) as the regularization term. In proximal gradient descent, the standard update rule is given by:
\begin{equation*}
\boldsymbol{\beta}^{(t+1)} = \text{prox}_{\lambda \|\cdot\|_1} \left( \boldsymbol{\beta}^{(t)} - \eta_t \nabla \ell(\boldsymbol{\beta}^{(t)}) \right),
%\label{eq:prox_update}
\end{equation*}
where $\eta_t$ is the step size and $\text{prox}_{\lambda \|\cdot\|_1}$ denotes the proximal operator of the $\ell_1$ norm. This operator has the closed-form solution:
\begin{equation*}
\text{prox}_{\lambda \|\cdot\|_1} (\beta_j) = \text{sign}(\beta_j) \cdot \max(|\beta_j| - \lambda, 0),
%\label{eq:soft_threshold}
\end{equation*}
commonly known as the soft-thresholding function. It shrinks each coefficient's magnitude by $\lambda$ and zeroes out those below the threshold, thereby encouraging sparsity in the learned model parameters.

AMGD extends this idea by introducing a coefficient-adaptive thresholding mechanism, defined by:
\begin{equation*}
\beta_j^{(t+1)} = \text{sign}(\beta_j^{(t)}) \cdot \max\left(|\beta_j^{(t)}| - \frac{\alpha_t \lambda}{|\beta_j^{(t)}| + \epsilon}, 0\right),
%\label{eq:adaptive_threshold}
\end{equation*}
where $\alpha_t$ is the learning rate at iteration $t$, $\lambda$ is the regularization strength, and $\epsilon > 0$ is a small constant used to prevent division by zero and to cap shrinkage for near-zero coefficients. This formulation replaces the fixed threshold $\lambda$ with a dynamically scaled one that depends inversely on the magnitude of the current coefficient. Consequently, large coefficients receive lighter penalization, while smaller ones are more aggressively shrunk.

This behavior mirrors the principle behind the Adaptive Lasso \citep{zou2006adaptive}, where penalty weights are inversely proportional to the size of coefficient estimates. However, unlike the Adaptive Lasso, which requires an initial estimate of $\boldsymbol{\beta}$ to compute weights, AMGD dynamically updates the threshold at every iteration. This eliminates the need for staged reweighting and improves efficiency in iterative and online settings.

From an optimization perspective, this coefficient-adaptive thresholding serves several important functions. First, it embeds sparsity enforcement directly into the update step, offering a stable alternative to the subgradient-based handling of nonsmooth terms in methods like AdaGrad and Adam \citep{duchi2011adaptive, kingma2014adam}. Second, the dynamic penalty scales naturally with feature importance, preserving dominant features while suppressing noisy or weak signals. Third, as demonstrated in our empirical results (Section~\ref{sec:experiments}), this mechanism leads to more effective variable selection by consistently zeroing out irrelevant predictors, resulting in sparse, interpretable models.

Overall, this adaptive proximal formulation situates AMGD within the theoretical framework of composite optimization \citep{parikh2014proximal} while introducing a novel layer of responsiveness to both gradient dynamics and parameter magnitudes. This innovation contributes to AMGD's practical effectiveness in high-dimensional Poisson regression and supports its theoretical stability.

\subsection{Convergence Analysis}
\label{sec:convergence}

We analyze convergence in the setting of a convex penalized objective function. While the Poisson log-likelihood $\ell(\boldsymbol{\beta})$ in Equation(~\ref{eq:poisson_loglik}) is not convex, the penalized objective function (~\ref{eq:penalized_objective}) is convex with penalty (~\ref{eq:L1penalty}) or (~\ref{eq:L2penalty}). AMGD can be viewed as a gradient-based proximal method with diminishing step size, and our analysis leverages established results from optimization theory for adaptive momentum methods and proximal gradient methods.

\begin{theorem}[Convergence in Convex Setting]
\label{thm:convergence}
Assume $f(\boldsymbol{\beta}) = -\ell(\boldsymbol{\beta}) + \lambda P(\boldsymbol{\beta})$ is convex, where $P(\boldsymbol{\beta})$ is either the $L_1$ norm $\|\boldsymbol{\beta}\|_1$ or the squared $L_2$ norm $\|\boldsymbol{\beta}\|_2^2$. Also assume that $\|\nabla \ell(\boldsymbol{\beta})\|_\infty$ is bounded (which holds if $X$ and $y$ are bounded or if gradient clipping with threshold $T$ is applied). Let $\{\boldsymbol{\beta}^{(t)}\}$ be the sequence produced by Algorithm~\ref{alg:amgd}, where the proximal operator is chosen appropriately for the regularization term $P(\boldsymbol{\beta})$. If the learning rates satisfy
\begin{equation*}
\sum_{t=1}^\infty \alpha_t = \infty \quad \text{and} \quad \sum_{t=1}^\infty \alpha_t^2 < \infty,
\end{equation*}
which holds for $\alpha_t = \frac{\alpha}{1 + \eta t}$, and momentum parameters $\zeta_1, \zeta_2 \in [0, 1)$, then $\{\boldsymbol{\beta}^{(t)}\}$ converges to an optimal solution $\boldsymbol{\beta}^*$ of the objective function $f(\boldsymbol{\beta})$.
\end{theorem}

\begin{proposition}[Feature Selection Optimality]
\label{prop:feature_selection}
For the Poisson regression model with $L_1$ regularization, under appropriate regularity conditions, there exists an optimal feature subset $S^*$ that minimizes the expected prediction error:
\begin{equation*}
S^* = \arg\min_{S \subseteq \{1,2,...,p\}} \mathbb{E}[L(y, f_S(x))] + \alpha |S|,
%\label{eq:optimal_subset}
\end{equation*}
where $L$ is the loss function, $f_S$ is the model using only features in subset $S$, and $\alpha > 0$ is a complexity penalty.
\end{proposition}

Proposition 1 builds upon the adaptive lasso framework established by \citep{zou2006adaptive}, extending the oracle properties to Poisson regression with $L_1$ regularization. While Zou's original work demonstrates that the adaptive lasso enjoys oracle properties for GLMs, including Poisson regression, our formulation specifically addresses the feature selection aspect in the context of AMGD's adaptive thresholding mechanism.

The significance of this result is that it provides theoretical justification for applying $L_1$ regularization in Poisson regression models. It formally establishes that feature selection performed by the regularized objective optimally balances model fit against complexity, leading to improved prediction performance and model interpretability in count data settings. This theoretical foundation is particularly valuable in high-dimensional applications where Poisson regression is appropriate, such as genomic studies, network modeling, and event frequency analysis.

%\textcolor{green}{teorem içinde referans verilmez. Ne kadarı önceki referansa ait bilemiyorum ... Proof: yazıp for details \citep{robbins1951stochastic} can be seen denilebilir} 

%\textcolor{green}{\citep{robbins1951stochastic}}
\subsection{Convergence Speed}
\label{sec:convergence_speed}

AMGD exhibits the following convergence property for $L_1$-regularized Poisson regression:

\begin{proposition}
Assume the negative log-likelihood has a Lipschitz-continuous gradient, the domain of $\boldsymbol{\beta}$ is bounded, and the regularization parameter $\lambda$ is appropriately tuned. Then, for the Poisson regression problem with $L_1$ regularization, AMGD converges to a neighborhood of the optimal solution at a rate of $O(1/\sqrt{T})$ after $T$ iterations.
\end{proposition}

This convergence rate holds under standard conditions in composite optimization: specifically, when the gradient of the smooth loss component is Lipschitz continuous (e.g., if $\|X^\top \operatorname{diag}(\exp(X\boldsymbol{\beta})) X\| \leq L$ for all $\boldsymbol{\beta}$), and the domain of $\boldsymbol{\beta}$ is bounded \citep{nesterov2013introductory}. 

Due to the non-differentiability of the $L_1$ penalty, convergence is generally to a neighborhood of the optimal point, commonly referred to as a "noise ball." Compared to existing methods, AMGD demonstrates improved convergence. While AdaGrad achieves a theoretical rate of $O(1/\sqrt{T})$ for convex objectives, it requires additional proximal modifications to handle nonsmooth penalties effectively. Adam, although widely used in practice, lacks strong convergence guarantees in the presence of $L_1$ regularization and may diverge in certain settings \citep{reddi2019convergence}.

In contrast, AMGD explicitly integrates both diminishing adaptive step sizes and a proximal thresholding mechanism into its update rule. This design choice leads to more stable updates and better empirical performance, as demonstrated by our experiments in Section~\ref{sec:experiments}.

\subsection{Hyperparameter Selection}
\label{sec:hyperparams}

The AMGD algorithm introduces several hyperparameters, including the initial learning rate $\alpha$, decay factor $\eta$, gradient clipping threshold $T$, and momentum parameters $\zeta_1, \zeta_2$. Following recommendations from the Adam optimizer literature \citep{kingma2014adam}, we adopted default values $\zeta_1 = 0.9$, $\zeta_2 = 0.999$, and $\epsilon = 10^{-8}$. These settings remained consistent across all experiments after confirming their stability and effectiveness in our Poisson regression setting.

For the AMGD-specific hyperparameters $\alpha$, $\eta$, and $T$, we performed a grid search using 5-fold cross-validation on the validation set, with Mean Absolute Error (MAE) as the primary performance metric. Specifically, we tested learning rates $\alpha$ in the range $\{0.01, 0.05, 0.1\}$, decay factors $\eta$ in $\{10^{-5}, 10^{-4}, 10^{-3}\}$, and gradient clipping thresholds $T$ in $\{5, 10, 20\}$. These values were chosen based on prior experimentation and theoretical guidance from adaptive optimization literature.

The optimal configuration was determined based on the lowest average MAE across folds. Across most experimental settings, a learning rate of $\alpha = 0.05$, decay factor $\eta = 10^{-4}$, and clipping threshold $T = 10$ provided the best trade-off between convergence speed, sparsity induction, and predictive accuracy.

We observed that AMGD exhibits relative stability to small variations in $\alpha$ or $\eta$, thanks to its coordinate-wise adaptive scaling via $\hat{v}_t$. In particular, features with consistently large gradients naturally receive smaller updates due to inflated second moment estimates, reducing sensitivity to global step size selection. Similarly, while gradient clipping proved essential for stabilizing training on heavy-tailed or zero-inflated datasets, performance was not overly sensitive to the exact value of $T$ as long as it remained within a reasonable range (5–20).

These findings suggest that AMGD requires less manual tuning than standard gradient descent methods, making it more user-friendly in practice, especially for sparse and high-dimensional count data applications.

\subsection{Convergence Criteria}
\label{sec:convergence_criteria}

We monitor the change in the penalized objective function (~\ref{eq:penalized_objective}). This objective function captures both the model's goodness of fit and the regularization-induced sparsity.

When the relative change $|f(\boldsymbol{\beta}^{(t)}) - f(\boldsymbol{\beta}^{(t-1)})|$ becomes smaller than a tolerance (e.g., $10^{-6}$), or after a maximum number of iterations is reached, the algorithm terminates. Because we employ learning rate decay, the algorithm is guaranteed to eventually make arbitrarily small updates, allowing stringent tolerances to be satisfied.

We emphasize that AMGD performs updates that directly incorporate the penalty term through adaptive soft-thresholding, rather than following the exact gradient trajectory of the unpenalized log-likelihood. This approach is analogous to how coordinate descent implicitly accounts for penalties in each coordinate update, or how the Iterative Shrinkage-Thresholding Algorithm (ISTA) \citep{daubechies2004iterative} incorporates proximal steps. This integration of sparsity enforcement within each iteration contributes to AMGD's effectiveness in achieving sparse solutions while maintaining convergence guarantees.

 \section{Numerical Experiments and Results}
\label{sec:experiments}

To evaluate the performance of the  AMGD algorithm, we conduct extensive numerical experiments using a large-scale ecological dataset. We compare AMGD with gradient-based methods (Adam and AdaGrad) and the widely-used GLMNet algorithm \citep{friedman2010regularization}, which implements coordinate descent for regularized GLMs. Our experiments focus on Lasso and Elastic Net regularization frameworks across multiple performance metrics.

We utilize a comprehensive ecological health dataset containing 61,345 observations with 16 features, targeting the Biodiversity Index as our response variable. This dataset encompasses environmental measurements including air quality indicators (PM2.5, Air Quality Index), soil characteristics (moisture, pH, nutrient levels), water quality metrics (Total Dissolved Solids, Biochemical and Chemical Oxygen Demand), and categorical variables representing pollution levels and ecological health classifications. Given the count nature of biodiversity measurements, we employ Poisson regression as our modeling framework.

Our preprocessing pipeline applies z-score normalization where each feature $x$ is transformed as $z = \frac{x - \mu}{\sigma}$, with $\mu$ and $\sigma$ representing the sample mean and standard deviation, respectively. Categorical variables undergo one-hot encoding with the reference category dropped to prevent multicollinearity issues. Missing values are imputed using median substitution for numerical features and mode substitution for categorical variables. Following preprocessing, our feature matrix expands to 17 dimensions including the encoded categorical variables.

We implement a 70/15/15 data partition, yielding 42,939 training samples, 9,204 validation samples, and 9,202 test samples. Hyperparameter optimization employs 5-fold cross-validation on the validation set. For each algorithm-regularization combination, we conduct grid search over regularization parameters $\lambda$ spanning a logarithmic range from $10^{-3}$ to $10^1$ with 50 equally-spaced values on the log scale. Our evaluation framework incorporates Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Poisson Deviance (MPD), and computational runtime efficiency. Optimal $\lambda$ values are selected based on cross-validated MAE performance.

All algorithms are implemented in Python utilizing scikit-learn libraries and custom optimization routines. To ensure fair comparison, all methods receive identical initialization parameters, convergence criteria, preprocessing procedures, and evaluation protocols. Our baseline comparison encompasses Adam \citep{kingma2014adam} and AdaGrad \citep{duchi2011adaptive} as representative gradient-based optimizers, alongside GLMNet \citep{friedman2010regularization} as the established coordinate descent benchmark.

Table \ref{tab:algorithm-performance} presents optimal configuration results from 5-fold cross-validation analysis. AMGD achieves MAE of 2.985, RMSE of 3.873, and MPD of 2.188 with Elastic Net regularization at $\lambda=0.01$, demonstrating the fastest computational runtime at 0.002 seconds. Adam achieves MAE of 3.081, RMSE of 3.983, and MPD of 2.225 using Elastic Net at $\lambda=0.1$. AdaGrad requires higher regularization ($\lambda=10.0$) and achieves MAE of 6.862, RMSE of 7.579, and MPD of 10.965. GLMNet exhibits MAE of 9.007, RMSE of 9.551, and MPD of 28.848 under Lasso regularization. 

\begin{table}[H]
\centering
\caption{Optimal configuration results based on 5-fold cross-validation for each algorithm on the ecological dataset}
\label{tab:algorithm-performance}
\begin{tabular}{|l|l|c|c|c|c|} \hline
\textbf{Algorithm} & \textbf{Optimal Configuration} & \textbf{MAE} & \textbf{RMSE} & \textbf{MPD} & \textbf{Runtime (s)} \\ \hline
AMGD & Elastic Net, $\lambda$=0.01 & \textbf{2.985} & \textbf{3.873} & \textbf{2.188} & \textbf{0.002} \\ \hline
Adam & Elastic Net, $\lambda$=0.1 & 3.081 & 3.983 & 2.225 & 0.004 \\ \hline
AdaGrad & Elastic Net, $\lambda$=10.0 & 6.862 & 7.579 & 10.965 & 0.745 \\ \hline
GLMNet & Lasso, $\lambda$=0.01 & 9.007 & 9.551 & 28.848 & 0.040 \\ \hline
\end{tabular}
\end{table}

Figure \ref{fig:convergence-rate} shows convergence behavior across optimization methods. AMGD reaches near-optimal loss values within approximately $10\%$ of the maximum iteration count. Adam exhibits comparable initial convergence rates but requires additional iterations. AdaGrad shows consistently slower convergence and converges to a suboptimal solution. GLMNet displays numerical instability with loss values occasionally falling below zero.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{conv.png}  
    \caption{Convergence rate comparison across optimization methods. Top panel: Log-scale loss values over training iterations. Bottom panel: Normalized loss values over the same iteration range.}
    \label{fig:convergence-rate}
\end{figure}

The dataset comprises 17 features spanning environmental measurements and categorical encodings as detailed in Table \ref{tab:model-coefficients}. Features 1-10 represent continuous environmental variables, Features 13-14 encode pollution level categories, and Features 15-17 represent ecological health classifications.

\begin{table}[H]
\centering
\caption{Feature descriptions in the ecological health dataset}
\label{tab:model-coefficients}
\begin{tabular}{|c|l|} \hline
\textbf{Feature Index} & \textbf{Description} \\ \hline
1 & Humidity \\ \hline
2 & Air Quality Index \\ \hline
3 & PM2.5 Concentration \\ \hline
4 & Soil Moisture \\ \hline
5 & Nutrient Level \\ \hline
6 & Water Quality \\ \hline
7 & Total Dissolved Solids \\ \hline
8 & Soil pH \\ \hline
9 & Biochemical Oxygen Demand \\ \hline
10 & Chemical Oxygen Demand \\ \hline
11-12 & Additional Environmental Variables$^{\dagger}$ \\ \hline
13 & Pollution Level: Low \\ \hline
14 & Pollution Level: Moderate \\ \hline
15 & Ecological Health: Ecologically Degraded \\ \hline
16 & Ecological Health: Ecologically Healthy \\ \hline
17 & Ecological Health: Ecologically Stable \\ \hline
\multicolumn{2}{|l|}{$^{\dagger}$Details not shown in coefficient plots} \\
\end{tabular}
\end{table}

Figure \ref{fig:Feat_Enet} presents feature importance analysis under Elastic Net regularization. Ecological health labels (Features 15-17) show coefficients of approximately 1.81 for AMGD, 1.55 for Adam, and 0.48 for AdaGrad. Pollution level indicators (Features 13-14) exhibit coefficients around 0.53 for AMGD, 0.79 for Adam, and 0.64 for AdaGrad. Environmental measurements (Features 1-12) display minimal coefficient values below 0.01 across all algorithms.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{coefvalue.png}
    \caption{Feature importance comparison for Elastic Net regularization across optimization methods.}
    \label{fig:Feat_Enet}
\end{figure}

Figure \ref{fig:coef_paths} demonstrates regularization paths of coefficient estimates. AMGD with $\lambda = 0.01$ exhibits dynamic feature selection, initiating with 11-12 non-zero coefficients ($29\%$-$35\%$ sparsity) during early iterations, fluctuating between 9 and 13 during intermediate phases ($24\%$-$47\%$ sparsity range), and stabilizing at 11 out of 17 features ($35\%$ sparsity). Adam with $\lambda = 0.1$ maintains 15-16 non-zero coefficients throughout training ($12\%$ sparsity). AdaGrad retains all 17 coefficients regardless of regularization strength ($0\%$ sparsity).

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{coefpath.png}
    \caption{Regularization paths of coefficient estimates across different optimization methods, illustrating sparsity evolution during training.}
    \label{fig:coef_paths}
\end{figure}

\newpage
Table \ref{tab:test-performance} summarizes test set performance.

\begin{table}[H]
\centering
\caption{Test set performance comparison across optimization algorithms}
\label{tab:test-performance}
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Algorithm} & \textbf{MAE} & \textbf{RMSE} & \textbf{MPD} & \textbf{Sparsity (\%)} \\ \hline
AMGD               & \textbf{3.03} & \textbf{3.91} & \textbf{2.23} & \textbf{29.29} \\ \hline
Adam               & 3.12          & 4.03          & 2.29          & 11.76           \\ \hline
AdaGrad            & 6.74          & 7.47          & 10.50         & 5.00            \\ \hline
GLMNet             & 8.98          & 9.54          & 29.39         & 52.93           \\ \hline
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{train_test.png}
\caption{Comparative visualization of algorithm performance on the test and training dataset.}
\label{fig:test-perf}
\end{figure}

Bootstrap analysis with 1000 resamples confirms performance stability. Table \ref{tab:bootstrap-results} shows $95\%$ confidence intervals for each algorithm. AMGD demonstrates narrow confidence intervals across all metrics. Adam exhibits similar stability with marginally wider intervals. AdaGrad shows considerably broader intervals, while GLMNet displays tight intervals but at elevated error levels.


\begin{table}[H]
\centering
\caption{Bootstrap analysis results with 95\% confidence intervals (1000 resamples)}
\label{tab:bootstrap-results}
\begin{tabular}{|c|c|c|c|c|}
\hline
\textbf{Metric} & \textbf{AMGD} & \textbf{Adam} & \textbf{AdaGrad} & \textbf{GLMNet} \\ \hline
MAE             & 5.2288        & 5.7765        & 8.5618           & 8.9795          \\ 
                & [5.1222, 5.3196] & [5.7176, 5.8376] & [8.5587, 8.5645] & [8.9789, 8.9801] \\ \hline
RMSE            & 6.1329        & 6.6117        & 9.1288           & 9.5199          \\ 
                & [6.0460, 6.2180] & [6.5598, 6.6644] & [9.1264, 9.1314] & [9.5193, 9.5205] \\ \hline
MPD             & 6.1012        & 7.3462        & 23.0005          & 28.9572         \\ 
                & [5.9055, 6.3016] & [7.1839, 7.5049] & [22.9681, 23.0334] & [28.9472, 28.9673] \\ \hline
Sparsity        & 0.1571& 0.0653        & 0.0271           & 0.5082          \\ 
                & [0.1229, 0.1829]& [0.0538, 0.0753] & [0.0197, 0.0353] & [0.4859, 0.5312] \\ \hline
\end{tabular}
\end{table}

Figure \ref{fig:bootstrap} presents bootstrap validation results through violin plots (1000 resamples). AMGD exhibits narrow distributions across MAE, RMSE, and Mean Deviance metrics. Sparsity distributions show AMGD achieving approximately 15\% coefficient sparsity, GLMNet reaching 52\% sparsity with high variance, while Adam and AdaGrad demonstrate 8\% and 5\% sparsity respectively.
%Figure \ref{fig:bootstrap} presents violin plots of bootstrap distributions across 1000 resamples. AMGD shows tight distributions centered at optimal values. Sparsity distributions demonstrate AMGD's consistent feature selection capability, maintaining approximately $35\%$ sparsity compared to minimal sparsity in competing methods.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{bootstrap.png}
    \caption{Bootstrap distribution analysis showing performance stability across 1000 resamples.}
    \label{fig:bootstrap}
\end{figure}

Statistical significance testing comparing AMGD against all baseline methods yields highly significant differences (p < 0.0001), with substantial effect sizes ranging from –1.32 to –30.64 for error metrics and from –3.71 to 1.84 for sparsity comparisons, as shown in Table \ref{tab:statistical-tests}.

\begin{table}[H]
\centering
\caption{Statistical significance tests comparing AMGD with baseline methods}
\label{tab:statistical-tests}
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Metric} & \textbf{vs Adam} & \textbf{vs AdaGrad} & \textbf{vs GLMNet} \\ \hline
MAE & p=0.0000 (d=$-$1.33)$^*$ & p=0.0000 (d=$-$9.58)$^*$ & p=0.0000 (d=$-$10.78)$^*$ \\ \hline
RMSE & p=0.0000 (d=$-$1.32)$^*$ & p=0.0000 (d=$-$9.77)$^*$ & p=0.0000 (d=$-$11.05)$^*$ \\ \hline
Mean Deviance & p=0.0000 (d=$-$1.32)$^*$ & p=0.0000 (d=$-$22.40)$^*$ & p=0.0000 (d=$-$30.64)$^*$ \\ \hline
Sparsity & p=0.0000 (d=1.10)$^*$ & p=0.0000 (d=1.84)$^*$ & p=0.0000 (d=$-$3.71)$^*$ \\ \hline
\multicolumn{4}{|l|}{$^*$Statistically significant at $\alpha=0.05$; d = Cohen's d effect size} \\ \hline
\end{tabular}
\end{table}



Feature selection stability analysis across bootstrap resamples shows that categorical features achieve $100\%$ selection probability across all 1000 bootstrap samples. Environmental measurements show moderate selection probabilities: Water Quality and Total Dissolved Solids at $70\%$, PM2.5 at $60\%$, Biochemical Oxygen Demand at $57\%$, and Humidity at $53\%$, as presented in Table \ref{tab:feature-selection}.

\begin{table}[H]
\centering
\caption{Feature selection stability analysis for AMGD across 1000 bootstrap samples}
\label{tab:feature-selection}
\small{
\begin{tabular}{|c|l|c|} \hline
\textbf{Rank} & \textbf{Feature Name} & \textbf{Selection Probability} \\ \hline
1 & Pollution Level: Low & 1.00 \\ \hline
2 & Pollution Level: Moderate & 1.00 \\ \hline
3 & Ecological Health: Ecologically Degraded & 1.00 \\ \hline
4 & Ecological Health: Ecologically Healthy & 1.00 \\ \hline
5 & Ecological Health: Ecologically Stable & 1.00 \\ \hline
6 & Water Quality & 0.70 \\ \hline
7 & Total Dissolved Solids & 0.70 \\ \hline
8 & PM2.5 Concentration & 0.60 \\ \hline
9 & Biochemical Oxygen Demand & 0.57 \\ \hline
10 & Humidity & 0.53 \\ \hline
\end{tabular}
}
\end{table}

The experimental evaluation demonstrates AMGD's optimal performance across error metrics while maintaining 15\% coefficient sparsity. Bootstrap validation with 1000 resamples confirms performance stability through narrow distribution widths, contrasting with the higher variance observed in alternative methods.
%The experimental evaluation demonstrates AMGD's superior performance across all evaluation metrics while achieving substantial model sparsity ($35\%$). The bootstrap validation confirms stable performance across different data configurations, with statistical significance established at p < 0.0001 for all comparisons.

\FloatBarrier

\section{Discussions}
\label{sec:discussions}

The experimental findings offer several insights into optimization approaches for regularized Poisson regression, particularly in high-dimensional, sparse, or correlated data contexts. While the observed results are specific to our ecological dataset, they reveal general challenges and opportunities that may extend to similar applications.

AMGD’s strong empirical performance suggests that unified frameworks integrating adaptive learning rates, momentum, gradient clipping, and soft-thresholding can outperform traditional optimizers that treat these components separately. Notably, embedding gradient clipping within the optimization process improves stability under the exponential link function of Poisson models, where small parameter changes can cause large shifts in predicted means, especially problematic in high-dimensional settings.
.
GLMNet, though well-established for regularized GLMs \citep{friedman2010regularization}, exhibited high sparsity (around 52\%) but also notable instability across runs. This variability may stem from coordinate descent’s sensitivity to Poisson-specific characteristics \citep{simon2011regularization, wu2008coordinate}, including exponential scaling and iterative reweighting \citep{tseng2001convergence}. These methods lack momentum and adaptivity, which are crucial for navigating sharp or ill-conditioned loss surfaces \citep{breheny2011coordinate}.

AMGD's full-gradient updates and coefficient-adaptive thresholding offer a more stable and selective approach. The soft-thresholding mechanism dynamically adjusts shrinkage based on coefficient magnitude, preserving relevant variables while suppressing noise. These features improve convergence speed and sparsity without sacrificing interpretability. Feature selection analysis further supports this: ecological health and pollution indicators were consistently selected across bootstrap samples, indicating robust signal-noise separation. In contrast, environmental variables showed moderate selection frequencies, which may reflect either residual correlations or weaker predictive power. 


Scalability remains a limitation for AMGD due to its full-batch nature. While well-suited to moderate-sized datasets requiring interpretability and stability, it may not scale efficiently for real-time or streaming applications without stochastic variants.

Hyperparameter tuning is another challenge. Despite AMGD’s relative robustness, achieving optimal performance still required careful tuning of learning rate, decay factor, and clipping threshold. Automation of this process would enhance its usability in production environments.

The dataset’s hierarchical structure, dominated by categorical indicators may have favored AMGD’s adaptive sparsity approach. Applications with different distributions or feature dynamics might yield different results. Additionally, sparsity may not be universally beneficial, depending on the modeling goals.

Bootstrap validation confirms the algorithm’s performance stability, but the use of a single dataset limits generalizability. Furthermore, baseline methods were tested under default configurations; more extensive tuning might narrow observed performance gaps.

Overall, AMGD appears best suited for applications prioritizing interpretability, stable convergence, and sparse modeling over computational speed. Domains such as environmental monitoring, public health, and risk assessment may benefit from its transparency and stable feature selection, though application-specific validation is necessary.

The integrated design of AMGD demonstrates the value of combining multiple optimization strategies. Extending this philosophy to other problem domains may prove fruitful, particularly where sparsity, adaptivity, and numerical stability intersect. Future work should include evaluations across multiple datasets and against finely tuned baselines to validate the broader applicability of these findings.

\section{Conclusion}
\label{sec:conclusion}
The Adaptive Momentum Gradient Descent (AMGD) algorithm introduces a unified framework that integrates momentum, adaptive learning rates, coefficient-dependent soft-thresholding, and dual-level gradient clipping. This combination promotes both numerical stability and sparsity, offering interpretable solutions without sacrificing predictive performance. The results demonstrate stability in handling high-dimensional regularized Poisson regression tasks, particularly where sparse representations are beneficial. Additionally, its simplicity and modularity make it amenable to extensions and integration into existing optimization framework.

Despite its advantages, AMGD introduces practical considerations that may affect usability. The presence of multiple interacting hyperparameters can pose challenges for tuning, especially in the absence of automated selection procedures. Moreover, the current implementation is based on full-batch updates, which may limit efficiency or responsiveness in settings involving streaming or large-scale data. The reliance on adaptive thresholds and clipping mechanisms, while effective, also adds complexity to the interpretability of the optimization dynamics.

This study highlights several areas for further exploration. The theoretical analysis currently assumes favorable properties of the composite objective function; formal convergence guarantees for broader non-convex settings remain an open problem. Furthermore, while the algorithm's design is conceptually general, its performance and stability across diverse problem domains require additional validation. Future research should explore stochastic and distributed variants of AMGD, assess its integration into deep learning pipelines, and conduct comparative evaluations against baseline methods with external gradient control. Such efforts would help establish its broader applicability and theoretical completeness.

\newpage
%\bibliographystyle{elsarticle-num}
%\bibliographystyle{elsarticle-harv}


%\bibliographystyle{plainnat}
%\bibliographystyle{plain}
%\bibliographystyle{unsrt}
%\bibliographystyle{elsarticle-num}

%\bibliography{references}
\bibliographystyle{elsarticle-num}
\bibliography{references}


\newpage 
% Acknowledgments
\section*{Acknowledgments}
All code and data sets are available at [ https://github.com/elbakari01/amgd-Poisson-regression ].




\newpage 
% Appendices
\appendix
\section*{Appendix A. }

%%%
\subsection*{A.1 Proof of Theorem 1}

We now prove the convergence of the Adaptive Momentum Gradient Descent (AMGD) algorithm under convexity assumptions for Poisson regression with $L_1$ regularization. The proof leverages results from optimization theory for adaptive momentum methods and proximal gradient descent, demonstrating that the sequence of iterates $\{\beta_t\}$ generated by AMGD converges to an optimal solution of the objective function $f(\beta)$, which combines the negative log-likelihood and the $L_1$-regularization term. This result establishes that for the Poisson regression model with $L_1$ regularization, the optimal feature subset $S^*$ that minimizes the expected prediction error satisfies the desired theoretical guarantees, ensuring both sparsity and predictive accuracy.


\section*{Proof of Convergence in Convex Setting}

We analyze the convergence of Adaptive Momentum Gradient Descent (AMGD) for the composite objective function:
\begin{equation}
    f(\beta) = g(\beta) + h(\beta) = -\ell(\beta) + \lambda \|\beta\|_1
\end{equation}
where $g(\beta) = -\ell(\beta)$ is smooth and $h(\beta) = \lambda\|\beta\|_1$ is non-smooth but convex.

\subsection*{Assumptions}
We make the following standard assumptions:

\begin{enumerate}
    \item[(A1)] The smooth component $g(\beta)$ has $L$-Lipschitz continuous gradient: 
    $$\|\nabla g(\beta) - \nabla g(\beta')\| \leq L\|\beta - \beta'\|, \quad \forall \beta, \beta'$$
    
    \item[(A2)] The composite function $f(\beta) = g(\beta) + h(\beta)$ is convex. This holds when the negative log-likelihood $-\ell(\beta)$ is convex, which occurs under certain conditions on the design matrix $X$.
    
    \item[(A3)] The gradient of $g(\beta)$ is bounded: $\|\nabla g(\beta)\|_\infty \leq G$ for some constant $G > 0$. This is satisfied when gradient clipping is applied or when the domain is bounded.
    
    \item[(A4)] Momentum parameters satisfy $0 \leq \zeta_1, \zeta_2 < 1$.
\end{enumerate}

\subsection*{Algorithm formulation}
AMGD can be viewed as a proximal gradient method with momentum. The update rule is:
\begin{align}
    m_t &= \zeta_1 m_{t-1} + (1 - \zeta_1) \nabla g(\beta_t) \\
    v_t &= \zeta_2 v_{t-1} + (1 - \zeta_2) (\nabla g(\beta_t))^2 \\
    \hat{m}_t &= \frac{m_t}{1 - \zeta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \zeta_2^t} \\
    \beta_{t+1} &= \text{prox}_{\alpha_t h} \left( \beta_t - \alpha_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \right)
\end{align}

where the proximal operator for the L1 penalty is:
\begin{equation}
    \text{prox}_{\alpha_t \lambda \|\cdot\|_1}(z) = \text{sign}(z) \odot \max(|z| - \alpha_t \lambda, 0)
\end{equation}

\subsection*{Convergence Analysis}
We establish convergence by analyzing the composite objective decrease.

\textbf{Lemma 1.} Under assumptions (A1)-(A3), for any $\beta_{t+1}$ obtained from the proximal step:
\begin{equation}
    f(\beta_{t+1}) \leq f(\beta_t) + \langle \nabla g(\beta_t), \beta_{t+1} - \beta_t \rangle + \frac{L}{2}\|\beta_{t+1} - \beta_t\|^2 + h(\beta_{t+1}) - h(\beta_t)
\end{equation}

\textbf{Proof of Lemma 1:} This follows from the $L$-smoothness of $g$ and the convexity of $h$.

\textbf{Lemma 2.} The proximal operator satisfies the optimality condition:
\begin{equation}
    \beta_{t+1} = \arg\min_{\beta} \left\{ \frac{1}{2\alpha_t}\left\|\beta - \left(\beta_t - \alpha_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}\right)\right\|^2 + h(\beta) \right\}
\end{equation}

This implies:
\begin{equation}
    \left\langle \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \frac{1}{\alpha_t}(\beta_{t+1} - \beta_t), \beta - \beta_{t+1} \right\rangle + h(\beta) - h(\beta_{t+1}) \geq 0, \quad \forall \beta
\end{equation}

\textbf{Main Result:} Lyapunov function:
\begin{equation}
    \Psi_t = f(\beta_t) + \frac{1}{2\alpha_t}\|\beta_t - \beta^*\|^2
\end{equation}

where $\beta^*$ is an optimal solution.

\textbf{Theorem.} Under assumptions (A1)-(A4) and the step size conditions:
\begin{equation}
    \sum_{t=1}^{\infty} \alpha_t = \infty, \quad \sum_{t=1}^{\infty} \alpha_t^2 < \infty
\end{equation}

The sequence $\{\beta_t\}$ generated by AMGD satisfies:
\begin{enumerate}
    \item[(i)] $f(\beta_t) - f(\beta^*)$ converges to zero
    \item[(ii)] $\lim_{t \to \infty} \text{dist}(\beta_t, \mathcal{X}^*) = 0$, where $\mathcal{X}^*$ is the set of optimal solutions
\end{enumerate}

\textbf{Proof Sketch:}
\begin{enumerate}
    \item From the proximal operator optimality and smoothness of $g$:
    \begin{equation}
        f(\beta_{t+1}) \leq f(\beta_t) - \frac{\alpha_t}{2(1 + \delta)}\left\|\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}\right\|^2 + \frac{L\alpha_t^2}{2}\left\|\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}\right\|^2
    \end{equation}
    
    where $\delta = \max_t \left\|\frac{\mathbb{E}[\hat{m}_t] - \nabla g(\beta_t)}{\nabla g(\beta_t)}\right\| \to 0$ as $t \to \infty$ due to bias correction.
    
    \item The adaptive scaling satisfies:
    \begin{equation}
        \left\|\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}\right\|^2 \geq \frac{1}{G + \epsilon}\|\hat{m}_t\|^2
    \end{equation}
    
    \item The bias-corrected momentum satisfies:
    \begin{equation}
        \mathbb{E}[\hat{m}_t | \mathcal{F}_{t-1}] = \nabla g(\beta_t) + \underbrace{\frac{\zeta_1^t}{1-\zeta_1^t}(\nabla g(\beta_t) - \nabla g(\beta_0))}_{\text{bias term} \to 0}
    \end{equation}
    Since $\zeta_1^t \to 0$ and gradients are bounded (A3), the bias vanishes asymptotically, yielding:
    \begin{equation}
        \sum_{t=1}^{\infty} \alpha_t \mathbb{E}[\|\nabla g(\beta_t)\|^2] < \infty
    \end{equation}
    
    \item By the diminishing step size conditions and bounded gradients:
    \begin{equation}
        \lim_{t \to \infty} \mathbb{E}[\|\nabla g(\beta_t)\|^2] = 0
    \end{equation}
    
    \item Using the convexity of $f$ and the variational characterization of the proximal operator, we conclude that $\beta_t$ converges to the set of stationary points, which coincides with the set of optimal solutions for convex functions.
\end{enumerate}

\textbf{Remark:} For the specific case of AMGD with adaptive soft-thresholding as described in the paper, the proximal operator is modified to:
\begin{equation}
    [\beta_{t+1}]_j = \text{sign}([\tilde{\beta}_t]_j) \cdot \max\left(|[\tilde{\beta}_t]_j| - \frac{\alpha_t \lambda}{|[\tilde{\beta}_t]_j| + \epsilon}, 0\right)
\end{equation}
where $\tilde{\beta}_t = \beta_t - \alpha_t \hat{m}_t/(\sqrt{\hat{v}_t} + \epsilon)$. 

This adaptive thresholding can be viewed as a generalized proximal operator with coefficient-dependent penalties:
\begin{equation}
    \text{prox}_{\alpha_t h_{\text{adaptive}}}(z) \text{ where } h_{\text{adaptive}}(\beta) = \lambda \sum_{j=1}^p \frac{|\beta_j|}{|\beta_j| + \epsilon}
\end{equation}

The convergence analysis extends to this case under the additional conditions:
\begin{enumerate}
    \item[(C1)] The adaptive penalty function $h_{\text{adaptive}}$ remains convex
    \item[(C2)] The effective penalty weights remain bounded: $\frac{1}{|\beta_j| + \epsilon} \leq \frac{1}{\epsilon}$
\end{enumerate}
These conditions are satisfied by construction in AMGD, ensuring the convergence guarantees.

\hfill$\square$


\subsection*{A.2 Proof of Proposition 1}
For the Poisson regression model with $L_1$ regularization, the optimal feature subset $S^*$ that minimizes the expected prediction error satisfies:
$S^* = \arg\min_{S \subseteq \{1,2,...,p\}} \mathbb{E}[L(y, f_S(x))] + \alpha |S|,$
where:
- $L$ is the objective function (e.g., negative log-likelihood for Poisson regression),
- $f_S$ is the model using only features in subset $S$,
- $\alpha > 0$ is the complexity penalty controlling the trade-off between model fit and sparsity.

\subsection*{Proof}

We begin by considering the Poisson regression model with $L_1$ regularization. The optimization problem can be expressed as:
$$
\min_{\beta} \mathbb{E}[L(y, f(x))] + \lambda \|\beta\|_1,
$$
where:
- $L(y, f(x)) = -[y \log(f(x)) - f(x)]$ is the negative log-likelihood for Poisson regression,
- $f(x) = \exp(x^\top \beta)$ is the prediction function,
- $\lambda > 0$ is the regularization parameter controlling the strength of the $L_1$ penalty.

For a given feature subset $S \subseteq \{1, 2, ..., p\}$, let $f_S(x)$ denote the prediction function restricted to features in $S$. That is:
$$
f_S(x) = \exp(x_S^\top \beta_S),
$$
where $x_S$ and $\beta_S$ are the restricted feature vector and coefficient vector, respectively. The expected prediction error for a subset $S$ is:
$$
\mathbb{E}[L(y, f_S(x))] = \mathbb{E}[-y \log(f_S(x)) + f_S(x)].
$$

The $L_1$ penalty enforces sparsity by shrinking some coefficients exactly to zero, effectively performing feature selection. This creates a direct relationship between the $L_1$ penalty and the size of the selected feature subset. Specifically, we can reformulate the problem as:
$$
\min_{\beta} \mathbb{E}[L(y, f(x))] + \lambda \|\beta\|_1 \quad \Rightarrow \quad \min_{S \subseteq \{1,2,...,p\}} \mathbb{E}[L(y, f_S(x))] + \alpha |S|,
$$
where $|S|$ is the cardinality of subset $S$ (i.e., the number of selected features), and $\alpha$ is a complexity penalty derived from $\lambda$. 

This equivalence is justified through \textbf{Lagrangian duality}, which ensures that for any value of $\lambda$ in the $L_1$-regularized problem, there exists a corresponding value of $\alpha$ in the subset selection problem such that both formulations yield equivalent solutions.

\paragraph{Step 2: Connection to Adaptive Lasso}
The adaptive lasso framework introduced by \citep{zou2006adaptive} establishes that an adaptively weighted $L_1$ penalty achieves \textbf{oracle properties} in generalized linear models (GLMs), including Poisson regression. Specifically:

    
- \textbf{Oracle Properties}: The adaptive lasso ensures consistent variable selection and asymptotic normality of the estimated coefficients for the true non-zero features.
    
- \textbf{Weighted $L_1$ Penalty}: The adaptive lasso uses weights $w_j$ for each coefficient $\beta_j$, defined as:
    $$
    w_j = \frac{1}{|\hat{\beta}_j|^\gamma}, \quad \gamma > 0,
    $$
    where $\hat{\beta}_j$ is an initial estimate of $\beta_j$. These weights adaptively shrink small coefficients to zero while preserving larger ones.


In the context of Poisson regression, the adaptive lasso modifies the penalized log-likelihood function as:
$$
\ell(\beta, \lambda) = -\ell(\beta) + \lambda \sum_{j=1}^p w_j |\beta_j|,
$$
where $\ell(\beta)$ is the log-likelihood function for Poisson regression:
$$
\ell(\beta) = \sum_{i=1}^n \left( y_i x_i^\top \beta - \exp(x_i^\top \beta) - \log(y_i!) \right).
$$

\paragraph{Step 3: Feature Selection via $L_1$ Regularization}
The $L_1$ penalty $\lambda \sum_{j=1}^p |\beta_j|$ encourages sparsity by shrinking some coefficients exactly to zero. For Poisson regression, this has the following implications:

    
- \textbf{Sparsity Induction}: Irrelevant or weakly relevant features are eliminated from the model, reducing overfitting and improving interpretability.
    
- \textbf{Optimal Subset Selection}: The subset $S^*$ corresponds to the indices of non-zero coefficients after applying $L_1$ regularization. Formally:
    $$
    S^* = \{j : \beta_j \neq 0\}.
    $$


The penalty term $\alpha |S|$ directly accounts for the number of selected features, balancing model fit ($\mathbb{E}[L(y, f_S(x))]$) against model complexity ($|S|$).

\paragraph{Step 4: Convergence to Optimal Solution}
The convergence of the optimization process to the optimal feature subset $S^*$ relies on the following properties:

    
- \textbf{Convexity of the Objective Function}: The penalized log-likelihood function is convex when $P(\beta)$ is convex (e.g., $L_1$ regularization). This ensures a unique solution under appropriate conditions.
    
- \textbf{Adaptive Thresholding}: The AMGD algorithm incorporates adaptive soft-thresholding, which directly handles the $L_1$ penalty:
    $$
    \beta_{t+1} = \text{sign}(\beta_t) \cdot \max\left(|\beta_t| - \alpha_t \lambda / (|\beta_t| + \epsilon), 0\right).
    $$
    This ensures that irrelevant coefficients are driven to zero efficiently.


Under these conditions, the iterative optimization process converges to a neighborhood of the optimal solution at a rate of $O(1/\sqrt{T})$ after $T$ iterations \citep{beck2009fast}.


$\square$

%bib starts here 





%%
%% End of file `elsarticle-template-harv.tex'.


 \end{document}
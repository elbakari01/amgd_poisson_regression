# References and Related Work

## Core References from AMGD Research

### Foundational Optimization Methods

**[1] Tibshirani, R. (1996)**  
*Regression shrinkage and selection via the lasso*  
Journal of the Royal Statistical Society: Series B, 58(1), 267-288.  
**Significance**: Original Lasso formulation, foundational for L1 regularization

**[2] Hoerl, A. E., & Kennard, R. W. (1970)**  
*Ridge regression: Biased estimation for nonorthogonal problems*  
Technometrics, 12(1), 55-67.  
**Significance**: Ridge regression foundation, L2 regularization

**[3] Zou, H., & Hastie, T. (2005)**  
*Regularization and variable selection via the elastic net*  
Journal of the Royal Statistical Society: Series B, 67(2), 301-320.  
**Significance**: Combines L1 and L2 penalties, theoretical foundation for ElasticNet

### Modern Adaptive Optimizers

**[4] Duchi, J., Hazan, E., & Singer, Y. (2011)**  
*Adaptive subgradient methods for online learning and stochastic optimization*  
Journal of Machine Learning Research, 12(7), 2121-2159.  
**Significance**: AdaGrad algorithm, adaptive learning rates based on gradient history

**[5] Kingma, D. P., & Ba, J. (2014)**  
*Adam: A method for stochastic optimization*  
arXiv preprint arXiv:1412.6980.  
**Significance**: Adam optimizer, combines momentum with adaptive learning rates

### Regularized GLM Implementations

**[6] Friedman, J., Hastie, T., & Tibshirani, R. (2010)**  
*Regularization paths for generalized linear models via coordinate descent*  
Journal of Statistical Software, 33(1), 1-22.  
**Significance**: GLMnet implementation, coordinate descent for regularized GLMs

**[7] Simon, N., Friedman, J., Hastie, T., & Tibshirani, R. (2011)**  
*Regularization paths for Cox's proportional hazards model via coordinate descent*  
Journal of Statistical Software, 39(5), 1-13.  
**Significance**: Extensions of coordinate descent to survival models

### Adaptive Lasso and Oracle Properties

**[8] Zou, H. (2006)**  
*The adaptive lasso and its oracle properties*  
Journal of the American Statistical Association, 101(476), 1418-1429.  
**Significance**: Theoretical foundation for AMGD's adaptive soft-thresholding

**[9] Zou, H., & Zhang, H. H. (2009)**  
*On the adaptive elastic-net with a diverging number of parameters*  
Annals of Statistics, 37(4), 1733-1751.  
**Significance**: High-dimensional extensions of adaptive penalties

### Proximal and Gradient Methods

**[10] Beck, A., & Teboulle, M. (2009)**  
*A fast iterative shrinkage-thresholding algorithm for linear inverse problems*  
SIAM Journal on Imaging Sciences, 2(1), 183-202.  
**Significance**: ISTA algorithm, proximal gradient methods

**[11] Parikh, N., & Boyd, S. (2014)**  
*Proximal algorithms*  
Foundations and Trends in Optimization, 1(3), 127-239.  
**Significance**: Comprehensive proximal algorithm theory

### Statistical Learning Theory

**[12] Hastie, T., Tibshirani, R., & Friedman, J. (2009)**  
*The Elements of Statistical Learning: Data Mining, Inference, and Prediction*  
2nd Edition, Springer.  
**Significance**: Comprehensive statistical learning reference

**[13] Hastie, T., Tibshirani, R., & Wainwright, M. (2015)**  
*Statistical Learning with Sparsity: The Lasso and Generalizations*  
CRC Press.  
**Significance**: Modern sparsity methods and theory

### Count Data Modeling

**[14] Cameron, A. C., & Trivedi, P. K. (2013)**  
*Regression Analysis of Count Data*  
2nd Edition, Cambridge University Press.  
**Significance**: Comprehensive treatment of Poisson and count data regression

**[15] McCullagh, P., & Nelder, J. A. (1989)**  
*Generalized Linear Models*  
2nd Edition, CRC Press.  
**Significance**: Foundation of GLM theory including Poisson regression

### Convergence Theory and Analysis

**[16] Nesterov, Y. (2013)**  
*Introductory Lectures on Convex Optimization: A Basic Course*  
Springer Science & Business Media, Volume 87.  
**Significance**: Convex optimization theory, convergence analysis

**[17] Reddi, S. J., Kale, S., & Kumar, S. (2019)**  
*On the convergence of Adam and beyond*  
arXiv preprint arXiv:1904.09237.  
**Significance**: Analysis of Adam convergence issues, motivation for improvements

### Gradient Clipping and Stability

**[18] Pascanu, R., Mikolov, T., & Bengio, Y. (2013)**  
*On the difficulty of training recurrent neural networks*  
International Conference on Machine Learning, PMLR, 1310-1318.  
**Significance**: Gradient clipping for training stability

### Advanced Optimization Methods

**[19] Sun, T., Qiao, L., Liao, Q., & Li, D. (2020)**  
*Novel convergence results of adaptive stochastic gradient descents*  
IEEE Transactions on Image Processing, 30, 1044-1056.  
**Significance**: Recent advances in adaptive gradient methods

**[20] Sun, T., Sun, Y., Li, D., & Liao, Q. (2019)**  
*General proximal incremental aggregated gradient algorithms: Better and novel results under general scheme*  
Advances in Neural Information Processing Systems, 32.  
**Significance**: Proximal incremental methods, theoretical contributions

**[21] Sun, T., Yin, P., Li, D., Huang, C., Guan, L., & Jiang, H. (2019)**  
*Non-ergodic convergence analysis of heavy-ball algorithms*  
Proceedings of the AAAI Conference on Artificial Intelligence, 33(01), 5033-5040.  
**Significance**: Momentum method analysis, theoretical foundations

### Control Theory in Optimization

**[22] Li, J., Yuan, Y., & Luo, X. (2025)**  
*Learning error refinement in stochastic gradient descent-based latent factor analysis via diversified PID controllers*  
IEEE Transactions on Emerging Topics in Computational Intelligence.  
**Significance**: PID control mechanisms in optimization

**[23] Yuan, Y., Li, J., & Luo, X. (2024)**  
*A fuzzy PID-incorporated stochastic gradient descent algorithm for fast and accurate latent factor analysis*  
IEEE Transactions on Fuzzy Systems.  
**Significance**: Fuzzy logic adaptation in gradient methods

**[24] Luo, X., Qin, W., Dong, A., Sedraoui, K., & Zhou, M. (2020)**  
*Efficient and high-quality recommendations via momentum-incorporated parallel stochastic gradient descent-based learning*  
IEEE/CAA Journal of Automatica Sinica, 8(2), 402-411.  
**Significance**: Momentum-based methods in recommendation systems

### Computational Linear Algebra

**[25] Lawson, C. L., Hanson, R. J., Kincaid, D. R., & Krogh, F. T. (1979)**  
*Basic linear algebra subprograms for Fortran usage*  
ACM Transactions on Mathematical Software, 5(3), 308-323.  
**Significance**: BLAS foundations for efficient linear algebra operations

### Iterative Thresholding Methods

**[26] Daubechies, I., Defrise, M., & De Mol, C. (2004)**  
*An iterative thresholding algorithm for linear inverse problems with a sparsity constraint*  
Communications on Pure and Applied Mathematics, 57(11), 1413-1457.  
**Significance**: Iterative thresholding algorithms, sparsity constraints

### Coordinate Descent Methods

**[27] Wu, T. T., & Lange, K. (2008)**  
*Coordinate descent algorithms for lasso penalized regression*  
The Annals of Applied Statistics, 2(1), 224-244.  
**Significance**: Coordinate descent for Lasso, algorithmic details

**[28] Tseng, P. (2001)**  
*Convergence of a block coordinate descent method for nondifferentiable minimization*  
Journal of Optimization Theory and Applications, 109(3), 475-494.  
**Significance**: Block coordinate descent convergence theory

**[29] Breheny, P., & Huang, J. (2011)**  
*Coordinate descent algorithms for nonconvex penalized regression, with applications to biological feature selection*  
The Annals of Applied Statistics, 5(1), 232-253.  
**Significance**: Nonconvex coordinate descent methods

### Bayesian Methods for Comparison

**[30] Park, T., & Casella, G. (2008)**  
*The Bayesian lasso*  
Journal of the American Statistical Association, 103(482), 681-686.  
**Significance**: Bayesian perspective on Lasso regularization

## Related Work in Ecological Modeling

### Biodiversity and Species Modeling

**[31] Hastie, T., & Tibshirani, R. (1990)**  
*Generalized Additive Models*  
CRC Press.  
**Significance**: Flexible modeling approaches for ecological data

**[32] Wood, S. N. (2017)**  
*Generalized Additive Models: An Introduction with R*  
2nd Edition, CRC Press.  
**Significance**: Modern GAM methods for ecological applications

### Environmental Statistics

**[33] Cressie, N., & Wikle, C. K. (2011)**  
*Statistics for Spatio-Temporal Data*  
John Wiley & Sons.  
**Significance**: Spatial-temporal modeling in environmental science

**[34] Zuur, A. F., Ieno, E. N., & Smith, G. M. (2007)**  
*Analysing Ecological Data*  
Springer.  
**Significance**: Statistical methods specifically for ecological data

## Contemporary Optimization Research

### Recent Advances in Adaptive Methods

**[35] Loshchilov, I., & Hutter, F. (2017)**  
*Decoupled weight decay regularization*  
arXiv preprint arXiv:1711.05101.  
**Significance**: AdamW, improved weight decay in adaptive optimizers

**[36] Zaheer, M., Reddi, S., Sachan, D., Kale, S., & Kumar, S. (2018)**  
*Adaptive methods for nonconvex optimization*  
Advances in Neural Information Processing Systems, 31.  
**Significance**: Adaptive methods for nonconvex problems

### Theoretical Optimization Advances

**[37] Allen-Zhu, Z. (2017)**  
*Katyusha: The first direct acceleration of stochastic gradient methods*  
Journal of Machine Learning Research, 18(1), 8194-8244.  
**Significance**: Advanced acceleration techniques

**[38] Ghadimi, S., & Lan, G. (2013)**  
*Stochastic first-and zeroth-order methods for nonconvex stochastic programming*  
SIAM Journal on Optimization, 23(4), 2341-2368.  
**Significance**: Stochastic optimization theory

### High-Dimensional Statistics

**[39] Wainwright, M. J. (2019)**  
*High-Dimensional Statistics: A Non-Asymptotic Viewpoint*  
Cambridge University Press.  
**Significance**: Modern high-dimensional statistical theory

**[40] Bühlmann, P., & Van De Geer, S. (2011)**  
*Statistics for High-Dimensional Data: Methods, Theory and Applications*  
Springer.  
**Significance**: High-dimensional data analysis methods

## Software and Implementation References

### Scientific Computing Libraries

**[41] Harris, C. R., et al. (2020)**  
*Array programming with NumPy*  
Nature, 585(7825), 357-362.  
**Significance**: NumPy foundation for scientific computing

**[42] Pedregosa, F., et al. (2011)**  
*Scikit-learn: Machine learning in Python*  
Journal of Machine Learning Research, 12, 2825-2830.  
**Significance**: Scikit-learn for machine learning implementations

**[43] McKinney, W. (2010)**  
*Data structures for statistical computing in Python*  
Proceedings of the 9th Python in Science Conference, 445, 51-56.  
**Significance**: Pandas for data manipulation

**[44] Hunter, J. D. (2007)**  
*Matplotlib: A 2D graphics environment*  
Computing in Science & Engineering, 9(3), 90-95.  
**Significance**: Matplotlib for scientific visualization

## Historical Context and Foundations

### Early Regularization Methods

**[45] Tikhonov, A. N. (1943)**  
*On the stability of inverse problems*  
Doklady Akademii Nauk SSSR, 39(5), 195-198.  
**Significance**: Tikhonov regularization, historical foundation

**[46] Phillips, D. L. (1962)**  
*A technique for the numerical solution of certain integral equations of the first kind*  
Journal of the ACM, 9(1), 84-97.  
**Significance**: Early regularization techniques

### Statistical Learning Evolution

**[47] Vapnik, V. N. (1999)**  
*The Nature of Statistical Learning Theory*  
2nd Edition, Springer.  
**Significance**: Statistical learning theory foundations

**[48] Breiman, L. (2001)**  
*Random forests*  
Machine Learning, 45(1), 5-32.  
**Significance**: Ensemble methods for comparison

## Future Directions and Extensions

### Emerging Optimization Paradigms

**[49] Li, X., & Orabona, F. (2019)**  
*On the convergence of stochastic gradient descent with adaptive stepsizes*  
The 22nd International Conference on Artificial Intelligence and Statistics, PMLR, 983-992.  
**Significance**: Future directions in adaptive optimization

**[50] Cutkosky, A., & Orabona, F. (2019)**  
*Momentum-based variance reduction in non-convex SGD*  
Advances in Neural Information Processing Systems, 32.  
**Significance**: Variance reduction in momentum methods

### Distributed and Parallel Optimization

**[51] Dean, J., et al. (2012)**  
*Large scale distributed deep networks*  
Advances in Neural Information Processing Systems, 25.  
**Significance**: Distributed optimization for large-scale problems

**[52] Li, M., Andersen, D. G., Park, J. W., Smola, A. J., Ahmed, A., Josifovski, V., ... & Su, B. Y. (2014)**  
*Scaling distributed machine learning with the parameter server*  
11th USENIX Symposium on Operating Systems Design and Implementation, 583-598.  
**Significance**: Parameter server architectures

## Key Theoretical Results Summary

### Convergence Rates
- **Gradient Descent**: O(1/k) for convex, O(1/k²) for strongly convex
- **Accelerated Methods**: O(1/k²) for convex (Nesterov acceleration)
- **Adaptive Methods**: O(1/√T) typical rate for stochastic settings
- **AMGD**: O(1/√T) with adaptive soft-thresholding benefits

### Sparsity and Feature Selection
- **Lasso**: Consistent feature selection under restricted eigenvalue condition
- **Adaptive Lasso**: Oracle properties under regularity conditions
- **Elastic Net**: Grouped variable selection for correlated features
- **AMGD**: Dynamic feature selection with oracle properties

### Regularization Theory
- **Bias-Variance Trade-off**: Fundamental principle in regularization
- **Degrees of Freedom**: Effective dimensionality in regularized models
- **Cross-Validation**: Model selection and hyperparameter tuning
- **Information Criteria**: AIC, BIC for model comparison

## Citation Guidelines

When citing this work and related references:

### For the AMGD Algorithm:
```bibtex
@article{bakari2024amgd,
  title={Adaptive Momentum Gradient Descent: A New Algorithm in Regularized Poisson Regression},
  author={Bakari, Ibrahim and Özkale, M. Revan},
  journal={[Expert System with Application]},
  year={2025},
}
```

### For Foundational Methods:
- **Lasso**: Cite Tibshirani (1996) [1]
- **Elastic Net**: Cite Zou & Hastie (2005) [3]
- **Adam**: Cite Kingma & Ba (2014) [5]
- **AdaGrad**: Cite Duchi et al. (2011) [4]
- **Adaptive Lasso**: Cite Zou (2006) [8]

### For Theoretical Foundations:
- **Convergence Theory**: Cite Nesterov (2013) [16]
- **Proximal Methods**: Cite Parikh & Boyd (2014) [11]
- **Statistical Learning**: Cite Hastie et al. (2009) [12]

This comprehensive reference list provides the theoretical foundation, historical context, and future directions for optimization methods in regularized regression, with particular emphasis on the contributions and positioning of the AMGD algorithm within the broader optimization literature.
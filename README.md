# AMGD-Poisson-Regression-

# Adaptive Momentum Gradient Descent (AMGD) for Regularized Poisson Regression

This repository contains the implementation of the Adaptive Momentum Gradient Descent (AMGD) algorithm for regularized Poisson regression in high-dimensional generalized linear models (GLMs).

## Overview

AMGD combines three key components:
- Adaptive learning rates to prevent premature convergence
- Momentum updates to accelerate convergence and reduce oscillation
- Adaptive soft-thresholding to enhance sparsity and feature selection

Our implementation demonstrates significant advantages over existing methods including Adam, AdaGrad, and GLMnet.

## Repository Structure

- `scripts/`: Algorithm implementations and analysis code
- `data/`: Dataset files or instructions to generate/obtain them
- `results/`: Output figures and tables from experiments

## Installation

```bash
git clone https://github.com/elbakari01/AMGD-Poisson-Regression.git
cd AMGD-Poisson-Regression
pip install -r requirements.txt
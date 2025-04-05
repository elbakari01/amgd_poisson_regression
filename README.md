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

# Ecological Health Dataset

## Dataset Description

This dataset is designed for the classification of ecological health in urban environments. It consists of hourly data generated from January 1, 2018, to December 31, 2024, simulating various environmental parameters that affect ecological health. The dataset includes data points to represent key indicators of ecological well-being, making it suitable for training machine learning models aimed at predicting ecological health classifications. The labels are unbalanced, reflecting real-world conditions where certain ecological states are more common than others.



In our AMGD Poisson Regression model, we use **Biodiversity_Index** as the target variable since it represents count data that follows a Poisson distribution. This allows us to demonstrate the efficacy of our algorithm on real-world ecological count data.

## Use Cases

This dataset can be utilized for various applications, including:
- Training machine learning models for ecological health classification
- Conducting research on the impacts of environmental parameters on urban ecology
- Developing predictive analytics tools for urban planners and environmental managers to make data-driven decisions regarding ecological health

## Downloading Instructions

1. Go to [https://www.kaggle.com/datasets/datasetengineer/ecological-health-dataset](https://www.kaggle.com/datasets/datasetengineer/ecological-health-dataset)
2. Sign in to Kaggle (create an account if necessary)
3. Click the "Download" button
4. Extract the downloaded file if needed
5. Place the `ecological_health_dataset.csv` file in this directory (`/data`)

## Data Preprocessing

Our code in `data_preprocessing.py` handles all necessary preprocessing steps:
- Standardization of numerical features
- One-hot encoding of categorical variables (Pollution_Level, Ecological_Health_Label)
- Missing value imputation if needed
- Train/validation/test splitting (70/15/15)

## Citation

If you use this dataset in your research, please cite both the Kaggle dataset and our paper:

```bibtex
@article{bakari2025adaptive,
  title={Adaptive Momentum Gradient Descent Algorithm: A New Machine Learning Algorithm in Regularized Poisson Regression},
  author={Bakari, Ibrahim and Ozkale, M. Revan},
  journal={Expert Systems with Applications},
  year={2025}
}

## Installation

```bash
git clone https://github.com/elbakari01/AMGD-Poisson-Regression.git
cd AMGD-Poisson-Regression
pip install -r requirements.txt

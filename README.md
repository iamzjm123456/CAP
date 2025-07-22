# Machine Learning-Based MSPME Optimization for CAP Extraction

## Overview

This repository contains the Python code and related resources for a study focused on globally optimizing the conditions of Molecularly Imprinted Solid-Phase Microextraction (MSPME) for Chloramphenicol (CAP) extraction using machine learning techniques. The study employs six machine learning models, including decision tree, random forest, support vector machine, and their optimized variants, to explore and optimize critical MSPME parameters.

## Research Objectives

The main goals of this research are:

- To globally optimize MSPME conditions for CAP extraction.
- To enhance model interpretability and maintain prediction accuracy through rigorous validation protocols and hardware-accelerated computation.
- To quantitatively resolve feature contributions to CAP extraction efficiency using SHAP value decomposition.

## Methodology

### Machine Learning Models

Six machine learning models are used in this study:

- Decision Tree  
- Random Forest  
- Support Vector Machine  
- Optimized variants of the above models  

### Libraries and Tools

The following Python libraries and tools are used:

- **Python Version**: 3.8.5  
- **Scikit-learn (v1.0.2)**: For algorithmic implementation of machine learning models  
- **XGBoost (v1.5.0)**: For gradient-boosted tree architectures  
- **SHAP (v0.41.0)**: For interpretability analysis  
- **pandas (v1.3.5)**: For data preprocessing, including median-based missing value imputation and one-hot encoding for categorical variables  
- **Scikit-optimize (v0.9.0)**: For Bayesian hyperparameter tuning  
- **Jupyter (v6.4.8)**: For version-controlled computational workflows  

### Data Preprocessing

The experimental datasets, which include aggregation dynamics and extraction parameters, are preprocessed as follows:

- **One-Hot Encoding**: Categorical variables such as buffer types and sorbent configurations are encoded using one-hot encoding  
- **Data Partitioning**: The data is stratified and partitioned into an 80% training set and a 20% independent test set to preserve population distribution characteristics  

### Feature Optimization

Multidimensional feature optimization is achieved through Bayesian hyperparameter tuning coupled with 5-fold cross-validation to prevent overfitting. Critical MSPME parameters, including:

- Aptamer dosage: 5–200 nM  
- Mg²⁺ concentration: 0.5–10 mM  
- Phase interaction time: 5–60 min  

are systematically explored.

### Model Evaluation

Model performance is evaluated using:

- Root Mean Squared Error (RMSE)  
- Coefficient of Determination (R²)  

## Computational Environment

All computational workflows are executed on an NVIDIA RTX 3090-accelerated high-performance workstation running **Ubuntu 20.04 LTS** with **CUDA 11.6** optimization.

## Code Structure

```
CAP/
├── Chloramphenicol(CAP)/
    ├── Aggregation-Condition-Optimization/
    │   └── ML-Models/  # AdaBoost, GBDT, CART, RF for aggregation optimization
    ├── Extraction-Condition-Optimization/
    │   ├── ML-models/  # Random Forest for extraction optimization
    │   └── Mechanism_Analysis/  # SHAP analysis for extraction
```

## How to Use

### Clone the Repository

```bash
git clone https://github.com/iamzjm123456/CAP.git
cd CAP
```

### Install Dependencies

Install the required Python libraries:

```bash
pip install scikit-learn==1.0.2 xgboost==1.5.0 shap==0.41.0 pandas==1.3.5 scikit-optimize==0.9.0 jupyter==6.4.8
```

### Run the Code

Navigate to the specific model directory and run the desired script. For example, to run the AdaBoost model for aggregation condition optimization:

```bash
cd CAP/Chloramphenicol(CAP)/Aggregation-Condition-Optimization/ML-Models
python AdaBoost-Agg.py
```

## Results

The optimal conditions for MSPME and the predicted CAP extraction values are printed in the console after running each model.  
The SHAP analysis results are saved as an image: `shap_summary.png` in the relevant directory.

## License

This project is licensed under [Specify the License].

## Contact

If you have any questions or suggestions, please contact [18305683229@163.com].

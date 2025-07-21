Machine Learning-Based MSPME Optimization for CAP Extraction
Overview
This repository contains the Python code and related resources for a study focused on globally optimizing the conditions of Molecularly Imprinted Solid - Phase Microextraction (MSPME) for Chloramphenicol (CAP) extraction using machine learning techniques. The study employs six machine learning models, including decision tree, random forest, support vector machine, and their optimized variants, to explore and optimize critical MSPME parameters.
Research Objectives
The main goals of this research are:
To globally optimize MSPME conditions for CAP extraction.
To enhance model interpretability and maintain prediction accuracy through rigorous validation protocols and hardware - accelerated computation.
To quantitatively resolve feature contributions to CAP extraction efficiency using SHAP value decomposition.
Methodology
Machine Learning Models
Six machine learning models are used in this study:
Decision Tree
Random Forest
Support Vector Machine
Optimized variants of the above models
Libraries and Tools
The following Python libraries and tools are used:
Python Version: 3.8.5
Scikit-learn (v1.0.2): For algorithmic implementation of machine learning models.
XGBoost (v1.5.0): For gradient - boosted tree architectures.
SHAP (v0.41.0): For interpretability analysis.
pandas (v1.3.5): For data preprocessing, including median - based missing value imputation and one - hot encoding for categorical variables.
Scikit-optimize v0.9.0: For Bayesian hyperparameter tuning.
Jupyter (v6.4.8): For version - controlled computational workflows.
Data Preprocessing
The experimental datasets, which include aggregation dynamics and extraction parameters, are preprocessed as follows:
Missing Value Imputation: Median - based imputation is used to handle missing values.
One-Hot Encoding: Categorical variables such as buffer types and sorbent configurations are encoded using one - hot encoding.
Data Partitioning: The data is stratified and partitioned into an 80% training set and a 20% independent test set to preserve population distribution characteristics.
Feature Optimization
Multidimensional feature optimization is achieved through Bayesian hyperparameter tuning coupled with 5-fold cross-validation to prevent overfitting. Critical MSPME parameters, including aptamer dosage (5-200 nM), Mg²⁺ concentration (0.5-10 mM), and phase interaction time (5-60 min), are systematically explored.
Model Evaluation
Model performance is evaluated using Root Mean Squared Error (RMSE) and the coefficient of determination (R²).
Computational Environment
All computational workflows are executed on an NVIDIA RTX 3090-accelerated high -performance workstation running Ubuntu 20.04 LTS with CUDA 11.6 optimization.
Code Structure
The code in this repository is organized as follows:
CAP/Chloramphenicol(CAP)/Aggregation-Condition-Optimization/ML-Models: Contains Python scripts for different machine learning models (AdaBoost, GBDT, CART, RF) used for aggregation condition optimization.
CAP/Chloramphenicol(CAP)/Extraction-Condition-Optimization/ML-models:Contains Python scripts for random forest model used for extraction condition optimization.
CAP/Chloramphenicol(CAP)/Extraction-Condition-Optimization/Mechanism_Analysis: Contains Python script for SHAP analysis in the extraction process.
How to Use
Clone the Repository
git clone https://github.com/iamzjm123456/CAP.git cd CAP
Install Dependencies
Install the required Python libraries using pip:
pip install scikit - learn==1.0.2 xgboost==1.5.0 shap==0.41.0 pandas==1.3.5 scikit - optimize==0.9.0 jupyter==6.4.8
Run the Code
Navigate to the specific directory of the model you want to run and execute the Python script. For example, to run the AdaBoost model for aggregation condition optimization:
cd CAP/Chloramphenicol (CAP)/Aggregation - Condition - Optimization/ML - Models python AdaBoost - Agg.py
Results
The optimal conditions for MSPME and the predicted CAP extraction values are printed in the console after running each model. The SHAP analysis results are saved as an image (shap_summary.png) in the relevant directory.
License
This project is licensed under [Specify the License].
Contact
If you have any questions or suggestions, please contact [Your Email Address].

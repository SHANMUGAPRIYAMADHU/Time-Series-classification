# Time-Series-classification
XGBoost for Imbalanced Time-Series Classification
1. Overview
This project focuses on implementing, optimizing, and evaluating an XGBoost model for an imbalanced time-series classification problem. The dataset mimics real-world financial or sensor time-series signals, where rare events need to be detected accurately. The key emphasis is on advanced hyperparameter tuning methods such as Grid Search and Bayesian Optimization, along with rigorous performance evaluation using stratified time-series splits.

Pipeline Steps
Step 1: Data Preparation
Load and inspect the time-series dataset.
Handle missing values, scaling, and feature extraction (lags, rolling windows).
Split the dataset using TimeSeriesSplit ensuring no leakage.

Step 2: Address Class Imbalance
Analyze class distribution.
Apply techniques like:
Class weights
SMOTE for time-series (cautious usage)
Data-driven threshold tuning

Step 3: Model Implementation
Configure XGBoost for classification (binary or multi-class).
Use time-based features and engineered features.

Step 4: Hyperparameter Tuning
Perform either:
Grid Search on key parameters
Bayesian Optimization (Hyperopt) for efficient searching
Parameters tuned include:
max_depth, learning_rate, n_estimators, gamma, scale_pos_weight, subsample, colsample_bytree

Step 5: Model Evaluation
Evaluate using:
F1-score
Precision–Recall AUC
Confusion matrix
Time-based cross-validation
Step 6: Model Interpretation
Feature importance (Gain, Weight, Cover)
SHAP values for deeper insights
Decision threshold adjustments for rare events

3. Deliverable
Fully functional Colab Notebook containing:
Data preprocessing
Feature engineering
XGBoost model training
Hyperparameter tuning
Performance evaluation
Visualizations:
Learning curves
Feature importance plots
Confusion matrix

4. Usage
Run the Project in Google Colab
Upload the dataset or let the notebook generate synthetic time-series data.
Run each cell sequentially to:
Prepare data
Train XGBoost
Perform tuning
Evaluate predictions
Adjust hyperparameters or threshold settings.
Export predictions or the trained model.

5. Dependencies
Make sure you have the following installed:
numpy
pandas
scikit-learn
xgboost
matplotlib
seaborn
hyperopt 
imbalanced-learn 
shap

6.Interpretations
Model Performance: The F1-score and PR-AUC are more reliable than accuracy due to class imbalance.
Feature Importance: Time-based lag features typically contribute heavily in temporal models.
SHAP Analysis: Helps explain individual predictions and identify which features influence rare-event detection.
Threshold Tuning: A critical part of improving recall on minority classes.
Bias/Variance Tradeoff: Overfitting can occur if time-series splits are ignored.

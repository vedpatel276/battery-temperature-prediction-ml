# battery-temperature-prediction-ml
PROJECT: Exploratory Data Analysis and Machine Learning for Battery Temperature Prediction 

PROJECT OVERVIEW
This project applies Exploratory Data Analysis (EDA) and Machine Learning to predict average battery temperature (Tavg) using real battery cycling data. The workflow covers statistical analysis, visualization, correlation study, feature selection, and model development in MATLAB.

Two models are implemented and compared:
1. Multiple Linear Regression (baseline)
2. Decision Tree Regression (nonlinear model)
The Decision Tree significantly outperforms linear regression, confirming that battery thermal behavior is nonlinear in nature.

DATASET USED: 'BC_35.csv'
As the dataset is confidential, it has not been shared here.

Input features selected after EDA and correlation analysis:
1. Discharge Current
2. Actual Voltage (Vact)
3. Ampere-hours (Ah)

Target variable:
Average Temperature (Tavg)

METHODOLOGY
1. Descriptive statistics (mean, std, skewness, kurtosis)
2. Histogram analysis for distribution & outliers
3. Pearson & Spearman correlation heatmaps
4. Feature selection based on statistics + physics relevance
5. Train/Test split (80/20)
6. Model evaluation using R², RMSE, MAE
7. Visualization: Predicted vs Actual (train/test)


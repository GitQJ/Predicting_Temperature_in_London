# Predicting_Temperature_in_London
Performed a machine learning experiment to find the best model that predicts the temperature in London

Built an automated ML workflow that predicts London's daily mean temperature and logs every model, hyperparameter and RMSE in MLflow. Best model: random forest with RMSE 2.76°C, beating the 3.0 target and a 3.66 linear regression baseline.

# CONTEXT
As the climate changes, predicting weather matters more. I tested which regression models best predict mean temperature from historical London weather data (1979-2020).

# MY ROLE
[Solo] I covered loading, cleaning, EDA, feature selection, preprocessing, training, evaluation and experiment tracking.

# WHAT I DID
• Cleaned and explored 15341 daily records; visualised a warming trend (roughly 10°C to 12.7°C annual mean) and a correlation heatmap of 12 variables
• Selected features from correlations: excluded max/min temp to avoid target leakage
• Preprocessed data for modelling: scaling, imputation and create date features
• Wrote reusable functions to train and evaluate Linear Regression, Decision Tree and Random Forest across max depths 1 to 9
• Logged each run's model, parameters and RMSE to MLflow, then queried runs with mlflow.search_runs to compare

# RESULTS
• Random forest (max depth 8): RMSE 2.76, best of 19 configurations
• About 25% lower error than the linear baseline (3.66); best decision tree reached 2.83 (depth 7)
• Error dropped sharply up to depth ~6, then plateaued and slightly worsened past depth 8, a clear overfitting signal

# TECH STACK
Python, pandas, seaborn, scikit-learn, MLflow

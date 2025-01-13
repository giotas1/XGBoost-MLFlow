import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb

# Set the MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
experiment_name = "mlflow grid search experiment"
mlflow.set_experiment(experiment_name)

# Load the dataset
path = r'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/winequality-white.csv'
data = pd.read_csv(path)

# Split the data into training and testing sets
train, test = train_test_split(data, test_size=0.2, random_state=42)
x_train = train.drop('quality', axis=1)
x_test = test.drop('quality', axis=1)
y_train = train['quality']
y_test = test['quality']

# Define model parameters for grid search
param_grid = {
    'max_depth': [3, 6, 9],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'n_estimators': [100, 200, 300]
}

# Start an MLflow run
with mlflow.start_run(run_name="xgboost_grid_search") as run:
    print("MLFlow Run Details:")
    print(" Run ID:", run.info.run_id)
    print(" Experiment ID:", run.info.experiment_id)

    # Initialize the model
    model = xgb.XGBRegressor(random_state=42)

    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1
    )
    grid_search.fit(x_train, y_train)

    # Log the best parameters
    best_params = grid_search.best_params_
    mlflow.log_params(best_params)
    print("Best Parameters:", best_params)

    # Train the model with the best parameters
    best_model = grid_search.best_estimator_

    # Make predictions
    predictions = best_model.predict(x_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Print and log metrics
    print("Metrics:")
    print(" RMSE:", rmse)
    print(" MSE:", mse)
    print(" R2:", r2)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # Save the model
    mlflow.xgboost.log_model(best_model, artifact_path="xgboost-model")

# Verify the experiment details
from mlflow.tracking import MlflowClient
client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)

if experiment:
    print(f"Experiment ID: {experiment.experiment_id}")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    print(f"Number of runs in experiment '{experiment.name}': {len(runs)}")
else:
    print(f"Experiment '{experiment_name}' not found.")

import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("mlflow experiment")  # Set the experiment by name

# Load the dataset
path = r'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/winequality-white.csv'
data = pd.read_csv(path)

# Split the data into training and testing sets
train, test = train_test_split(data, test_size=0.2, random_state=42)
x_train = train.drop('quality', axis=1)
x_test = test.drop('quality', axis=1)
y_train = train['quality']
y_test = test['quality']

# Define model parameters
params = {
    'max_depth': 6,
    'min_child_weight': 1,
    'random_state': 2022
}

# Start an MLflow run
with mlflow.start_run(run_name='xgboost') as run:
    print("MLFlow Run Details:")
    print(" Run ID:", run.info.run_id)
    print(" Experiment ID:", run.info.experiment_id)

    # Log parameters
    mlflow.log_param('max_depth', params['max_depth'])
    mlflow.log_param('min_child_weight', params['min_child_weight'])
    mlflow.log_param('random_state', params['random_state'])

    # Train the model
    model = xgb.XGBRegressor(**params)
    model.fit(x_train, y_train)

    # Make predictions
    prediction = model.predict(x_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    # Print metrics
    print("Metrics:")
    print(" RMSE:", rmse)
    print(" MSE:", mse)
    print(" R2:", r2)

    # Log metrics
    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('mse', mse)
    mlflow.log_metric('r2', r2)

    # Save the model
    mlflow.xgboost.log_model(model, artifact_path="xgboost-model")

# Verify the experiment details
from mlflow.tracking import MlflowClient
client = MlflowClient()
experiment = client.get_experiment_by_name("mlflow experiment")
print("Experiment ID:", experiment.experiment_id)
runs = client.search_runs(experiment_ids=[experiment.experiment_id])
print(f"Number of runs in experiment '{experiment.name}': {len(runs)}")

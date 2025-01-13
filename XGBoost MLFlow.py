import mlflow
experiment_id= mlflow.create_experiment('mlflow experimenemt')


mlflow.set_tracking_uri("http://127.0.0.1:5000/")
experiment= mlflow.get_experiment(experiment_id)
print('name:{experiment}')


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import mlflow
import mlflow.xgboost

path= r'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/winequality-white.csv'
data = pd.read_csv(path)
data.head()

train,test =  train_test_split(data, test_size=0.2 , random_state=42)

x_train= train.drop('quality',axis=1)
x_test = test.drop('quality',axis=1)

y_train= train['quality']
y_test= test['quality']

params= {
    'max_depth':6,
    'min_child_weight':1,
    'random_state':2022
}

mlflow.start_run(experiment_id= experiment.experiment.id,run_name='xgboost') as run:
    run_id= run.info.run_id
    experiment_id= run.info.experiment_id
    print("MLFlow:")
    print(" Run ID:", run_id)
    print(" Experiment ID:", experiment_id)
    print(" Experiment Name:", client.get_experiment(experiment_id).name)
    
    #mlflow parameters
    print('mlflow parameters')
    print('- max_depth:',params['max_depth'])
    print('min_child_weight',params['min_child_weight'])

    model= xgb.XGBRegressor(**params)
    model.fit(x_train,y_train)

    #mlflow artifacts
    prediction= model.predict(x_test)
    print('prediction':prediction)
    rmse=np.sqrt(mean_squared_error(y_test,prediction))
    mse= mean_squared_error(y_test,prediction)
    r2= r2_score(y_test,prediction)
    print('metrics:')
    print('rmse:',rmse)
    print('mse',mse)
    print('r2',r2)

    #mlflow metrics
    mlflow.log_metric('rmse',rmse)
    mlflow.log_metric('mse',mse)
    mlflow.log_metric('r2',r2)
    mlflow.log_metric('max depth',params['max_depth'])
    mlflow.log_metric("MinChildWeight", params['min_child_weight'])
    mlflow.log_metric("RandomState", params['random_state'])
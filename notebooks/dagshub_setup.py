import mlflow

import dagshub

mlflow.set_tracking_uri('https://dagshub.com/abhishekramgarh13/mlops_mini_project.mlflow')
dagshub.init(repo_owner='abhishekramgarh13', repo_name='mlops_mini_project', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
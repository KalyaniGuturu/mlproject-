import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object
from src.utils import evaluate_models
from src.exception import CustomException
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models={
                'LinearRegression':LinearRegression(),
                'Gradient Boosting':GradientBoostingRegressor(),
                'K-Neighbors Regressor':KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest regressor':RandomForestRegressor(),
                'XGBRegressor':XGBRegressor(),
                'CatBoostRegressor':CatBoostRegressor(verbose=False),
                'AdaBoostRegressor':AdaBoostRegressor(),
            }

            params={
                "Decision Tree":{
                    'criterion':['squard_error','friedman_mse','absolute_error','poisson'],

                },
                "Random Forest regressor":{
                    'n_estimators':[8,16,32,64,128,256]               
                },
    
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators':[8,16,32,64,128,256]         
                },
                "LinearRegression":{},
                "K-Neighbors Regressor":{
                    'n_neighbors':[5,7,9,11]
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "CatBoostRegressor":{
                    'depth':[6,8,10],
                    'learning_rate':[0.01,0.05,0.1],
                    'iterations':[30,50,100]

                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256]
                }
            }




            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ##to get the best model score from dict
            best_model_score=max(sorted(model_report.values()))
            ## to get best model name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]


            if best_model_score<0.6:
                raise CustomException("no best model found")
            logging.info(f"Best model found on both trainig and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model

            )
            predicted=best_model.predict(X_test)
            
            r2_square=r2_score(y_test,predicted)
            return r2_square
            
        except Exception as e:
            return CustomException(e,sys)
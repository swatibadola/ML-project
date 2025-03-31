# WHAT THIS FILE WILL INCLUDE
# confusion matrix can be called, if solving a regression prblm we can add adjusted r square value and so on....
# FOR EVERY COMPONENT WE CREATE A CONFIG FILE


import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass
class modelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class modelTrainer:
    def __init__(self):
        self.trained_model_config = modelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            # train_array[:, :-1]  ->	Selects all rows (:) and all columns except the last (:-1).
            # train_array[:, -1]  ->	Selects all rows (:) but only the last column (-1).
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                'Random Forest' : RandomForestRegressor(),
                'Decision Tree' : DecisionTreeRegressor(),
                'Gradient Boosting' : GradientBoostingRegressor(),
                'Linear Regression' : LinearRegression(),
                'KNeighbors Regressor': KNeighborsRegressor(),
                'XGBRegressor' : XGBRegressor(),
                'CatBoosting Regressor' : CatBoostRegressor(),
                'AdaBoost Regressor' : AdaBoostRegressor()
            }

            logging.info("Hyperparameter tuning started")
            # HYPERPARAMETER SEARCH SPACE FOR TUNING MODELS
            params={
                'Random Forest':{
                    # n estimators -> no. of boosting stages . More trees better learning. Too many can cause overfitting
                    'n_estimators': [8,16,32,64,128,256]
                },
                'Decision Tree':{
                    # squared error -> minimizes variance
                    # friedman_mse -> improved version of squared error. Used in gradient boosting because it balances variance reduction. IF SQUARED_ERROR STRUGGLES WITH CHOOSING THE BEST SPLIT IN SOME CASES, FRIEDMAN_MSE CAN DO IT BETTER
                    # absolute error -> Uses MAE. INSTEAD OF VARIANCE IT MINIMIZES THE ABSOLUTE DIFFERENCE BTW PREDICTED AND ACTUAL VALUES. This method is less sensitive to outliers.
                    # poisson -> Used for count data. POISSON REGRESSION GREAT FOR COUNTING THINGS
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                'Gradient Boosting': {
                    # LEARNING RATE: step size
                    # Lower values make learning slower but prevent overfitting.
                    # Higher values speed up learning but may lead to overfitting.
                    'learning_rate': [.1,.01,.05,.001],
                    # SUBSAMPLES -> samples used for each boosting step
                    'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                'Linear Regression': {},
                'KNeighbors Regressor':{
                    'n_neighbors': [5,7,9,11]
                },
                'XGBRegressor': {
                    'learning_rate': [.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                'CatBoosting Regressor': {
                    # depth -> Tree depth. Larger depth, more complex trees, more capacity to learn patterns.
                    'depth': [6,8,10],
                    'learning_rate': [0.01,0.05,0.1],
                    # iterations ->  no. of boosting rounds. More iterations, more trees, better performance
                    'iterations': [30,50,100]
                },
                'AdaBoost Regressor': {
                    'learning_rate': [.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }

            }

            model_report : dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            logging.info("Hyperparameter tuning excuted successfully!")

            # to get best model score from the dict
            best_model_score = max(sorted(model_report.values()))

            # to get best model name from the dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.65:
                raise CustomException("No best model found")
            
            logging.info(f'Best found model on both training and testing dataset')


            save_object(
                file_path=self.trained_model_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
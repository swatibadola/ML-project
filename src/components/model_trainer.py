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
                'K-Neighbors Classifier' : KNeighborsRegressor(),
                'XGBClassifier' : XGBRegressor(),
                'CatBoosting Regressor' : CatBoostRegressor(),
                'AdaBoost Classifier' : AdaBoostRegressor()
            }

            model_report : dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

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
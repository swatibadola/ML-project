# ex: if I want to create a mongodb for the database we can use this file. 
# If I want to save my model in the cloud I'll write the code for that here.
# USED FOR STORING SOME IMPORTANT FUNCTIONS

import numpy as np
import pandas as pd
# pickle is a Python module that saves objects to a file and loads them later.

# However, pickle cannot save some complex objects like:

# Lambda functions (lambda x: x + 1)

# Nested functions (functions inside functions)

# Custom-defined objects with certain decorators

# dill is an improved version of pickle that supports these complex objects.
import dill

import os
import sys

from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        # 'wb' -> write binary mode
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)    #used to serialize and save object

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        # Ensuring X_train and X_test are 2D
        X_train = X_train.reshape(-1,1) if X_train.ndim == 1 else X_train
        X_test = X_test.reshape(-1,1) if X_test.ndim == 1 else X_test


        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report


    except Exception as e:
        raise CustomException(e, sys)
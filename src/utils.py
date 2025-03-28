# ex: if I want to create a mongodb for the database we can use this file. 
# If I want to save my model in the cloud I'll write the code for that here.

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

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        # 'wb' -> write binary mode
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)    #used to serialize and save object

    except Exception as e:
        raise CustomException(e, sys)
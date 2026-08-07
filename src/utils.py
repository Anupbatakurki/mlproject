import os 
import sys

import pandas as pg
import numpy as np 
import pickle 

from src.exception import CustomException

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def save_object(file_path,obj):
    #this function save any python file as pickle file 
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok = True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def load_object(filepath):
    #this fucntion loads a pickle file

    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(model, X_test, y_test):
    """
    this fucntion is used to evaluate a trained classification model.
    """

    try:

        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
        )

        recall = recall_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

        return {

            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc

        }

    except Exception as e:

        raise CustomException(e, sys)
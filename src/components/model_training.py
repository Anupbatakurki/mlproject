import os 
import sys 
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.utils import save_object

@dataclass

class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacats",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,X_train,X_test,y_train,y_test):
        try:
            logging.info('applying SMOTE')

            smote = SMOTE(random_state=42)

            X_train_smote, y_train_smote = smote.fit_resample(X_train,y_train)

            logging.info("SMOTE Applied successfully")

            model = LogisticRegression(max_iter= 1000)

            param_grid = {
                "C": [0.001,0.01,0.1,1,10,100],
                "solver": ["liblinear","lbfgs"],
                "penalty": ["l2"],
                "class_weight": [None,"balanced"]
            }
            logging.info("Hyperparameter tunning has started")

            search = RandomizedSearchCV(
                estimator=lr,
                param_distributions=param_grid,
                n_iter=15,
                cv=5,
                scoring="f1",
                random_state=42,
                n_jobs=-1
            )

            search.fit(X_train_smote, y_train_smote)

            best_model = search.best_estimator_

            logging.info("best parameter ")
            logging.info(search.best_params_)
            logging.info("Training the final model")

            best_model.fit(X_train_smote,y_train_smote)
            y_pred = best_model.predict(X_test)

            y_prob = best_model.predict_proba(X_test)[:,1]

            accuracy = accuracy_score(y_test,y_pred)

            precision = precision_score(
                y_test,
                y_pred
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

            logging.info(f"Accuracy : {accuracy}")
            logging.info(f"Precision : {precision}")
            logging.info(f"Recall : {recall}")
            logging.info(f"F1 Score : {f1}")
            logging.info(f"ROC AUC : {roc_auc}")

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,

                obj=best_model

            )

            logging.info("Model Saved Successfully")

            return {

                "Accuracy": accuracy,

                "Precision": precision,

                "Recall": recall,

                "F1 Score": f1,

                "ROC AUC": roc_auc

            }

        except Exception as e:

            raise CustomException(e, sys)
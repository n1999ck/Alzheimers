import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from data_extractor import PatientData
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib
import math
import os
import dotenv

data = PatientData()
xgb = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,
    max_depth=1, random_state=0)
param_grid = {
            'n_estimators': [10, 50, 100, 500],
            'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'max_depth': [3, 5, 7, 9]
        }
gbr2 = GridSearchCV(xgb, param_grid, cv=3, n_jobs=1)
gbr2.fit(data.X_train_no_val, data.y_train_no_val)

print(gbr2.best_params_)
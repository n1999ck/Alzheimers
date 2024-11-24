import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from data_extractor import PatientData
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import joblib
import math
import os
import dotenv

data = PatientData()
rfc = RandomForestClassifier(n_estimators=64, max_depth=6, min_samples_leaf=2)
    
param_grid = {
            'n_estimators': [16, 64, 128, 256],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11],
            'min_samples_leaf': [2, 3, 5, 7, 9]
        }
gbr2 = GridSearchCV(rfc, param_grid, cv=3, n_jobs=1)
gbr2.fit(data.X_train_no_val, data.y_train_no_val)

print(gbr2.best_params_)
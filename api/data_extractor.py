import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

device = 'cuda' if torch.cuda.is_available() else 'cpu'
'''
Extracts the patient data from the CSV file
Provides a preset amount of data samples for
 training, validating, testing
Can use X or y to obtain all sample features and labels respectively
'''
class PatientData():
    def __init__(self):

        # dataset -> DataFrame Object    
        self.dataset = pd.read_csv('Dataset.csv', encoding="Latin-1") #ISO-8859-1 used in basic latin. UTF-8 for anything else
        self.dataset.dropna(inplace=True)
        self.dataset.pop("DoctorInCharge")
        self.dataset.pop("PatientID") 
        self.dataset = shuffle(self.dataset, random_state=42)
        self.labels = self.dataset.pop("Diagnosis")
        self.feature_names = self.dataset.columns

        #Max-abs Normalization 
        for column in self.dataset.columns:
            if self.dataset[column].abs().max() != 0:
                self.dataset[column] = self.dataset[column]/self.dataset[column].abs().max()
            else:
                self.dataset[column] = 0 

        self.X = np.array(self.dataset) #X=Features
        self.y = np.array(self.labels) #Y=Labels

        #Splitting the train, validation, and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, shuffle=False)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_test, self.y_test, test_size=0.5, shuffle=False)
        self.X_train_no_val, self.X_test_no_val, self.y_train_no_val, self.y_test_no_val = train_test_split(self.X, self.y, test_size=0.149777, shuffle=False)

        #print("Training set is: {} rows which is {} %".format(self.X_train.shape, round(self.X_train.shape[0]/self.dataset.shape[0], 4)*100))
        #print("Validation set is: {} rows which is {} %".format(self.X_val.shape, round(self.X_val.shape[0]/self.dataset.shape[0], 4)*100))
        #print("Testing set is: {} rows which is {} %".format(self.X_test.shape, round(self.X_test.shape[0]/self.dataset.shape[0], 4)*100))
        
        #print("Training set with no validation is: {} rows which is {} %".format(self.X_train_no_val.shape, round(self.X_train_no_val.shape[0]/self.dataset.shape[0], 4)*100))
        #print("Testing set with no validation is: {} rows which is {} %".format(self.X_test_no_val.shape, round(self.X_test_no_val.shape[0]/self.dataset.shape[0], 4)*100))
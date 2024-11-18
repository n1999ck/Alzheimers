import torch
import time
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading model 4...")
start_time = time.time()

'''
STEP 1: LOADING DATASET
'''
# Create test and train datasets    
dataset = pd.read_csv('Dataset.csv', encoding="Latin-1") #ISO-8859-1 used in basic latin. UTF-8 for anything else
dataset.dropna(inplace=True)
dataset.pop("DoctorInCharge")
dataset.pop("PatientID") 

#Max-abs Normalization
for column in dataset.columns:
    dataset[column] = dataset[column]/dataset[column].abs().max()

X = np.array(dataset.iloc[:,:-1]) #X=Features
Y = np.array(dataset.iloc[:, -1]) #Y=Labels

#Splitting the train, validation, and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

""" print("Training set is: {} rows which is {} %".format(X_train.shape, round(X_train.shape[0]/dataset.shape[0], 4)*100))
print("Validation set is: {} rows which is {} %".format(X_val.shape, round(X_val.shape[0]/dataset.shape[0], 4)*100))
print("Testing set is: {} rows which is {} %".format(X_test.shape, round(X_test.shape[0]/dataset.shape[0], 4)*100))
 """
'''
STEP 2: MAKING DATASET ITERABLE
'''
#Stores the samples and their corresponding labels
class CSVDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype= torch.float32).to(device)
        self.Y = torch.tensor(Y, dtype= torch.float32).to(device)

    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

training_data = CSVDataset(X_train, y_train)
validation_data = CSVDataset(X_val, y_val)
testing_data = CSVDataset(X_test, y_test)

class SVM():
    svm = SVC(kernel='rbf', gamma='scale', C=2)
    test_acc = -1

    def get_test_acc(self):
        return self.test_acc
    def __init__(self):
        inner_start_time = time.time()
        self.svm.fit(X_train, y_train)

        y_train_predict = self.svm.predict(X_train)
        y_test_predict = self.svm.predict(X_test)
        
        train_acc = accuracy_score(y_train,y_train_predict)
        self.test_acc = accuracy_score(y_test, y_test_predict)
        print(f"The SVC training {train_acc*100:.2f}%")
        print(f"The Support Vector model's accuracy on the testing dataset is: {self.test_acc*100:.2f}%")
        
        # Confusion matrix with true neg, false pos, false neg, true pos respectively
        cm = confusion_matrix(y_true=y_test, y_pred=y_test_predict)   
        tn, fp, fn, tp = cm.ravel()  

        specificity = -1
        precision = -1
        recall = -1
        f1 = -1
        mcc = -1
        # if statements to stop any division by 0 errors
        if((tn+fp)>0):
            specificity = tn / (tn+fp)  # Correctly predicted negatives over all actual negatives

        if((tp+fp) & (fn+tp)>0):
            precision = tp /(tp+fp)     # Correctly predicted positives over all predicted positives
            recall = tp / (fn+tp)       # Correctly predicted positives over all actual positives
            f1 = 2 * ((precision*recall)/(precision+recall))    # F1 Score 

        if((tp+fp) & (tp+fn) & (tn+fp) & (tn+fn) > 0):
            mcc = ((tp*tn) - (fp*fn))/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))  # Matthews Correlation Coefficient

        """print("Precision:\t{}".format(round(precision, 4)))        
        print("Specificity:\t{}".format(round(specificity, 4)))
        print("Recall:\t\t{}".format(round(recall, 4)))
        print("F1:\t\t{}".format(round(f1, 4)))
        print("MCC:\t\t{}".format(round(mcc, 4))) """
        print("Model 4 (within SVM()) loaded in {} seconds".format(round(time.time()-inner_start_time, 2)))

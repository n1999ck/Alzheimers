import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading model 2...")
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
print("Testing set is: {} rows which is {} %".format(X_test.shape, round(X_test.shape[0]/dataset.shape[0], 4)*100)) """

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

#HYPERPARAMETERS
BATCH_SIZE=4
EPOCHS=50
HIDDEN_NEURONS=8
LR=1e-2

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)
testing_dataloader = DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=False)

'''
STEP 3: CREATE MODEL CLASS
'''
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.input_layer = nn.Linear(X.shape[1], HIDDEN_NEURONS)
        self.hidden_layer1 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.hidden_layer2 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.output_layer = nn.Linear(HIDDEN_NEURONS, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() 

    # Forward pass
    def forward(self, x):
        x = self.relu(self.input_layer(x))  # Apply input layer and ReLU
        x = self.relu(self.hidden_layer1(x))  # Apply first hidden layer and ReLU
        x = self.relu(self.hidden_layer2(x))  # Apply second hidden layer and ReLU
        x = self.sigmoid(self.output_layer(x))  # Apply output layer and Sigmoid
        return x
    
    def get_test_acc(self):
        return total_acc_test/X_test.shape[0]
'''
STEP 4: INSTANTIATE MODEL CLASS
'''
model = MLP().to(device)
#summary(model, (X.shape[1],))  # TODO: Fix summary

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.BCELoss()

'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

'''
STEP 7: TRAIN THE MODEL
'''
total_loss_train_plot = []
total_loss_validation_plot = []
total_acc_train_plot = []
total_acc_validation_plot = []
for epoch in range(EPOCHS):
    total_acc_train = 0
    total_loss_train = 0
    total_acc_val = 0
    total_loss_val = 0
    for data in train_dataloader:
        inputs, labels = data
        prediction = model(inputs).squeeze(1)
        batch_loss = criterion(prediction, labels)
        total_loss_train += batch_loss.item()
        acc = ((prediction).round() == labels).sum().item()
        total_acc_train += acc

        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        acc=0
        for data in validation_dataloader:
            input, labels = data
            prediction = model(input).squeeze(1)
            batch_loss = criterion(prediction, labels)
            total_loss_val += batch_loss.item()
            acc = ((prediction).round() == labels).sum().item()
            total_acc_val += acc

    total_loss_train_plot.append(round(total_loss_train/(training_data.__len__()), 4))
    total_acc_train_plot.append(round(total_acc_train/(training_data.__len__())*100, 4))
    total_loss_validation_plot.append(round(total_loss_val/(validation_data.__len__()), 4))
    total_acc_validation_plot.append(round(total_acc_val/(validation_data.__len__())*100, 4))

"""   print(f"Epoch no. {epoch+1}, Train Loss: {total_loss_train/1000:.4f}, Train Accuracy {(total_acc_train/(X_train.shape[0])*100):.4f}")
    print("\n")
    print(f"Epoch no. {epoch+1}, Val Loss: {total_loss_val/1000:.4f}, Val Accuracy {(total_acc_val/(X_val.shape[0])*100):.4f}")
    print("="*60)
 """
'''
STEP 8: TEST THE MODEL
'''
with torch.no_grad():
    total_loss_test = 0
    total_acc_test = 0
    y_pred=[]
    y_label=[]
    acc=0
    for data in testing_dataloader:
        inputs, labels = data
        prediction = model(inputs).squeeze(1)
        batch_loss_test = criterion((prediction), labels)
        total_loss_test += batch_loss_test.item()
        acc = ((prediction).round() == labels).sum().item()
        total_acc_test += acc

        for item in prediction:
            y_pred.append(int(item.round()))
        for item in labels:
            y_label.append(int(item))  

''''
STEP 9: ASSESS TESTING OUTCOME
'''
# Confusion matrix with true neg, false pos, false neg, true pos respectively
cm = confusion_matrix(y_true=y_label, y_pred=y_pred)   
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

# Print all calculated metrics for test samples
print("Test Accuracy:\t\t{}%".format(round((total_acc_test/X_test.shape[0])*100, 4)))
print("Total correct:\t\t{}".format(total_acc_test))
print("Total predictions:\t{}".format(X_test.shape[0]))
"""
print("-"*60)
print("Precision:\t{}".format(round(precision, 4)))
print("Specificity:\t{}".format(round(specificity, 4)))
print("Recall:\t\t{}".format(round(recall, 4)))
print("F1:\t\t{}".format(round(f1, 4)))
print("MCC:\t\t{}".format(round(mcc, 4))) """

'''
STEP 10: PLOT METRICS
'''
""" # Plot confusion matrix for test samples
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i,j], va='center', ha='center')
ax.xaxis.set_ticks_position('bottom')
plt.xlabel('Predicted Label')
plt.ylabel('True label')
plt.show()

# Plot accuracy and loss for test samples
figs, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
axs[0].plot(total_loss_train_plot, label="Train Loss")
axs[0].plot(total_loss_validation_plot, label="Validation Loss")
axs[0].set_title("Train and Validation Loss Over Epochs")
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_ylim([0,.2])
axs[0].legend()

axs[1].plot(total_acc_train_plot, label="Train Accuracy")
axs[1].plot(total_acc_validation_plot, label="Validation Accuracy")
axs[1].set_title("Train and Validation Accuracy Over Epochs")
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].set_ylim([0,100])
axs[1].legend()

plt.tight_layout()
plt.show() """


#https://github.com/manujosephv/pytorch_tabular
#https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python

#https://www.isanasystems.com/machine-learning-handling-dataset-having-multiple-features/


# .78 + .60 + .82 + .75 = 2.95
.78/2.95 

print("Model 2 loaded in {} seconds".format(round(time.time()-start_time, 2)))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier 
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet

modelLR = LogisticRegression(penalty='12')
modelEN = ElasticNet(alpha=1.0, l1_ratio=0.5)
#import seaborn as sns

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) 
        # Linear function
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) 
        self.bn2 = nn.BatchNorm1d(hidden_dim) 
        self.fc3 = nn.Linear(hidden_dim, output_dim)  
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)
        out = self.bn1(out)
        # Non-linearity  # NON-LINEAR
        out = self.sigmoid(out)
        out = self.dropout(out)
        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.sigmoid(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

class CSVDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
# Create test and train datasets    
dataset = pd.read_csv('Dataset.csv', encoding="ISO-8859-1")
dataset_labels = dataset.pop("Diagnosis")
dataset.pop("DoctorInCharge")
#adding cross validation
x = dataset.values
y = dataset_labels

model = RandomForestClassifier()

modelLR = LogisticRegression(penalty='l2')
modelEN = ElasticNet(alpha=1.0, l1_ratio=0.5)
modelLR.fit(x, y)
modelEN.fit(x, y)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
#Cross-Validation for Logistic Regression
scores_rf = cross_val_score(model,x,y,cv=kf)
print("Random Forest Cross-validation scores:", scores_rf)
print("Random Forest Mean score:", np.mean(scores_rf))

#Cross-Validation for Elastic Net
scores_en = cross_val_score(modelEN, x,y,cv=kf)
print("Elastic Net Cross-validation scores:", scores_en)
print("Elastic Net Mean Score:", np.mean(scores_en))
 

test_portion_dataset = dataset[int((len(dataset) * (7/8))):]
test_portion_dataset_labels = dataset_labels[int((len(dataset) * (7/8))):]
train_portion_dataset = dataset[:int((len(dataset) * (7/8)))]
train_portion_dataset_labels = dataset_labels[:int((len(dataset) * (7/8)))]


print(len(test_portion_dataset))
print(len(test_portion_dataset_labels))
print(len(train_portion_dataset))
print(len(train_portion_dataset_labels))

dataset_features = np.vstack(train_portion_dataset.values).astype(np.float32)
test_dataset_features = np.vstack(test_portion_dataset.values).astype(np.float32)


mean = dataset_features.mean(axis=0)
std = dataset_features.std(axis=0)
dataset_features = (dataset_features - mean) / std
test_dataset_features = (test_dataset_features - mean) / std

batch_size = 128
n_iters = 2000
num_epochs = n_iters / (len(dataset_features) / batch_size)
num_epochs = int(num_epochs)
print(num_epochs)
input_dim = 33
output_dim = 2
hidden_dim = 1000

train_dataset = CSVDataset(dataset_features, train_portion_dataset_labels)
test_dataset = CSVDataset(test_dataset_features, test_portion_dataset_labels)
dataset_loader =torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)

test_loader =torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)


model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

val_split = 0.2  
val_size = int(len(train_portion_dataset) - val_split)
train_size = len(train_portion_dataset) - val_size

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,[train_size, val_size])
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

actual = np.random.binomial(1,.9, size=1000)
predicted = np.random.binomial(1,.9, size =1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

cm_display.plot()
plt.show()
Accuracy = metrics.accuracy_score(actual, predicted)
Precision = metrics.precision_score(actual, predicted)
Sensitivity_recall = metrics.recall_score(actual, predicted)
Specificity = metrics.recall_score(actual, predicted, pos_label=0)
F1_score = metrics.f1_score(actual, predicted)
print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score})

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        #Hidden Dimensions
        self.hidden_dim = hidden_dim
        #Number of Hidden Layers
        self.layer_dim = layer_dim
        #Building RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        #Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        #Initialize hidden state w/ zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, hn = self.rnn(x, h0.detach())
        #Index hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out
    
input_dim = 28
hidden_dim = 100
layer_dim = 1
output_dim = 10
    
model2 = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

iter = 0
for epoch in range(num_epochs):
    optimizer.step()
    scheduler.step()
    print('Epoch:', epoch, 'LR:', scheduler.get_last_lr())
    
    model.train()
    for i, (fields, labels) in enumerate(dataset_loader):
        optimizer.zero_grad()
        outputs = model(fields)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        
        iter += 1
        if iter % 500 == 0:
            print(fields, labels)
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            # t
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
            for fields, labels in val_loader:
                
                # Forward pass only to get logits/output
                outputs = model(fields)
                
                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                
                # Total number of labels
                total += labels.size(0)
                
                # Total correct predictions
                correct += (predicted == labels).sum().item()
            
            val_accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy:.2f}%')
        

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), val_accuracy))                      
total_accuracy = 100 * correct/total
print('Testing Accuracy:', total_accuracy)